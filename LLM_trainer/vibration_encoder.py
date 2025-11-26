import argparse
import functools
import types
from typing import Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    get_linear_schedule_with_warmup,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_outputs import BaseModelOutputWithPooling

from peft import LoraConfig, TaskType, get_peft_model

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProjector,
    MultimodalProjectorConfig,
)
from cornstarch.models.multimodal_language_model.processing_multimodal_language_model import (
    MultimodalProcessor,
)
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tokenizer_trainer.models.ViT_pytorch import VisionTransformerAE


MODALITY_KEYS: tuple[str, str] = ("x_stft", "ref_stft")
STFT_FEATURE_NUM: int = 1
MODALITY_TOKENS: list[str] = ["<x_stft>", "<ref_stft>"]

class VibrationEncoderConfig(PretrainedConfig):
    model_type = "vibration_encoder"

    def __init__(
        self,
        input_channels: int = 2,
        input_length: int = 1024,
        hidden_size: int = 256,
        add_pooling: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.input_length = input_length
        self.hidden_size = hidden_size
        self.add_pooling = add_pooling


class VibrationEncoderHF(PreTrainedModel):
    """사용자 정의 진동 인코더를 HF `PreTrainedModel` 인터페이스로 감싸는 래퍼."""

    config_class = VibrationEncoderConfig
    main_input_name = "stft"

    def __init__(self, config: VibrationEncoderConfig, vib_ae: Optional[nn.Module] = None, requires_grad_for_encode: bool = True):
        super().__init__(config)
        if vib_ae is None:
            raise ValueError(
                "VibrationEncoderHF requires a `vib_ae` (nn.Module with `.encode(...)`). "
                "Pass it via ctor or from_pretrained(..., vib_ae=...)."
            )
        self.vib_encoder = vib_ae
        self.requires_grad_for_encode = requires_grad_for_encode

    def forward(
        self,
        vibration: Optional[torch.FloatTensor] = None,
        stft: Optional[torch.FloatTensor] = None,
        x_stft: Optional[torch.FloatTensor] = None,
        ref_stft: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
        **kwargs,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.Tensor, torch.Tensor]]:
        candidates = [vibration, stft, x_stft, ref_stft]
        tensor_input = next((tensor for tensor in candidates if tensor is not None), None)
        if tensor_input is None:
            for key in (*MODALITY_KEYS, "stft"):
                value = kwargs.get(key)
                if torch.is_tensor(value):
                    tensor_input = value
                    break
        if tensor_input is None:
            raise ValueError("STFT encoder requires an input tensor.")

        if self.requires_grad_for_encode:
            z = self.vib_encoder.encode(tensor_input)
        else:
            with torch.no_grad():
                z = self.vib_encoder.encode(tensor_input)
        if isinstance(z, (tuple, list)):
            z = z[0]

        if z.dim() == 2:
            last_hidden_state = z.unsqueeze(1)
            pooler_output = z
        elif z.dim() == 3:
            last_hidden_state = z
            pooler_output = z.mean(dim=1)
        else:
            raise ValueError(f"Unexpected z shape {z.shape}. Expect (B, D) or (B, T, D).")

        if not return_dict:
            return last_hidden_state, pooler_output

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=None if not output_hidden_states else (last_hidden_state,),
            attentions=None,
        )


class DTypeAwareProjection(nn.Module):
    """LLM 임베딩 dtype에 맞춰 출력을 캐스팅하는 프로젝션 헤드."""

    def __init__(self, embedding_dim: int, token_embed_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=embedding_dim, out_features=int(embedding_dim * 2)),
            nn.Sigmoid(),
            nn.Linear(in_features=int(embedding_dim * 2), out_features=token_embed_dim),
        )
        self._target_dtype: torch.dtype | None = None

    def set_target_dtype(self, dtype: torch.dtype | None) -> None:
        self._target_dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.layers(x)
        if self._target_dtype is not None and outputs.dtype != self._target_dtype:
            outputs = outputs.to(dtype=self._target_dtype)
        return outputs


def build_stft_projector(embedding_dim: int, token_embed_dim: int) -> MultimodalProjector:
    """STFT 인코더 출력을 LLM 토큰 임베딩 차원으로 사상하는 프로젝터."""
    projection = DTypeAwareProjection(embedding_dim=embedding_dim, token_embed_dim=token_embed_dim)

    proj_cfg = MultimodalProjectorConfig(
        in_features=embedding_dim,
        out_features=token_embed_dim,
    )
    return MultimodalProjector(config=proj_cfg, projection=projection)

def build_stft_module(modality_key: str, llm_hidden_size: int, vib_enc_pth:str, *, trainable: bool = True) -> ModalEncoderModule:
    embedding_dim = 768
    ae = VisionTransformerAE(num_classes=5)
    """
    여기에 승하가 pretrain_ae로 가중치를 로드하는 내용을 넣어주어야 함
    load code 수정은 완료, pretrained_model_path => /checkpoints/best_model.pth
    """
    ckpt = torch.load(vib_enc_pth, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    ae.load_state_dict(state, strict=True)
    if not trainable:
        for param in ae.parameters():
            param.requires_grad_(False)
    
    vib_cfg = VibrationEncoderConfig(
        input_channels=2,
        input_length=224,
        hidden_size=embedding_dim,
    )
    vibration_encoder_hf = VibrationEncoderHF(vib_cfg, vib_ae=ae, requires_grad_for_encode=trainable)
    vibration_encoder_hf.main_input_name = modality_key
    projector = build_stft_projector(
        embedding_dim=embedding_dim,
        token_embed_dim=llm_hidden_size,
    )
    return ModalEncoderModule(
        model=vibration_encoder_hf,
        projector=projector,
    )
    
class STFTProcessor:
    """Cornstarch 프로세서에서 요구하는 배치 텐서 형태로 STFT 데이터를 정리한다."""

    def __init__(self, feature_key: str):
        self.feature_key = feature_key

    def __call__(
        self,
        stft: list[torch.Tensor] | torch.Tensor | None = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        if stft is None:
            if self.feature_key in kwargs:
                stft = kwargs[self.feature_key]
            elif "stft" in kwargs:
                stft = kwargs["stft"]
            elif kwargs:
                stft = next(iter(kwargs.values()))
        if stft is None:
            raise ValueError(f"{self.feature_key} tensor is required for STFTProcessor.")

        if isinstance(stft, list):
            stft = [torch.as_tensor(v) for v in stft]
            stft = torch.stack(stft, dim=0)
        else:
            stft = torch.as_tensor(stft)
            if stft.ndim == 2:
                stft = stft.unsqueeze(0)

        stft = stft.to(dtype=torch.float32)

        if return_tensors != "pt":
            raise ValueError(f"Unsupported tensor type: {return_tensors}")

        return BatchFeature(data={self.feature_key: stft})
    
def stft_num_features(inputs: dict, outputs: BatchFeature) -> list[int]:
    """모달 토큰 개수를 LLM에 알려주기 위한 헬퍼."""
    feature = next(iter(outputs.values()))
    batch_size = feature.shape[0]
    return [STFT_FEATURE_NUM] * batch_size
