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
    BitsAndBytesConfig,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_outputs import BaseModelOutputWithPooling

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProjector,
    MultimodalProjectorConfig,
)
from cornstarch.models.multimodal_language_model.processing_multimodal_language_model import (
    MultimodalProcessor,
)

from LLM_trainer.vibration_encoder import build_stft_module, STFTProcessor, stft_num_features


MODALITY_KEYS: tuple[str, str] = ("x_stft", "ref_stft")
STFT_FEATURE_NUM: int = 1
MODALITY_TOKENS: list[str] = ["<x_stft>", "<ref_stft>"]

def parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.lower()
    if lowered in ("none", "null"):
        return None
    if lowered in ("true", "1", "yes", "y"):
        return True
    if lowered in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value from '{value}'")


def patch_llm_forward(llm: PreTrainedModel) -> None:
    """Cornstarch가 넘겨주는 불필요한 인자를 제거해 Qwen3와의 충돌을 방지한다."""
    original_forward = llm.forward

    def forward_without_extra_kwargs(self, *args, **kwargs):
        kwargs.pop("position_embeddings", None)
        kwargs.pop("hidden_states", None)
        return original_forward(*args, **kwargs)

    llm.forward = types.MethodType(forward_without_extra_kwargs, llm)

def build_multimodal_system(
        apply_lora: bool = True,
        cache_dir: str = "/workspace/llm_cache",
        llm_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        vib_enc_pth:str = '',
        use_qlora: bool = False,
        freeze_vib_encoder: bool = True,
    ) -> tuple[MultimodalModel, AutoTokenizer, MultimodalProcessor]:
    gpu_available = torch.cuda.is_available()
    if gpu_available and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    elif gpu_available:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    model_kwargs: dict[str, object] = {
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
    }
    if use_qlora:
        if not gpu_available:
            raise RuntimeError("QLoRA는 GPU 환경에서만 지원됩니다.")
        quant_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=quant_dtype,
        )
        model_kwargs["quantization_config"] = quant_config
        current_device = torch.cuda.current_device()
        model_kwargs["device_map"] = {"": current_device}
    else:
        model_kwargs["dtype"] = compute_dtype

    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        **model_kwargs,
    )
    llm.is_quantized = bool(use_qlora)
    tokenizer = AutoTokenizer.from_pretrained(
        llm_name,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.add_special_tokens({"additional_special_tokens": MODALITY_TOKENS})

    if use_qlora:
        llm = prepare_model_for_kbit_training(llm)

    encoder_modules = {
        key: build_stft_module(
            modality_key=key,
            llm_hidden_size=llm.config.hidden_size,
            vib_enc_pth=vib_enc_pth,
            trainable=not freeze_vib_encoder,
        )
        for key in MODALITY_KEYS
    }

    if apply_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                    "q_proj", "k_proj", "v_proj",   # attention 쿼리/키/값 투영
                    "o_proj",                       # attention 출력 투영
                    "down_proj", "up_proj"          # feed-forward 내부 투영 (if present)
            ]
    )
        llm = get_peft_model(llm, peft_config)
        if hasattr(llm, "enable_input_require_grads"):
            llm.enable_input_require_grads()

    patch_llm_forward(llm)
    if hasattr(llm.config, "use_cache"):
        llm.config.use_cache = False

    mllm = MultimodalModel(
        encoders=encoder_modules,
        language_model=llm,
    )

    # LLM 임베딩 디바이스와 인코더 디바이스를 일치시켜 디바이스 mismatch 방지
    try:
        lm_device = llm.get_input_embeddings().weight.device
    except Exception:
        lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for module in mllm.encoders.values():
        module.module.to(lm_device)
        module.projector.to(lm_device)

    encoder_processors = {key: STFTProcessor(feature_key=key) for key in MODALITY_KEYS}
    mm_processor = MultimodalProcessor(
        encoder_processors=encoder_processors,
        llm_tokenizer=tokenizer,
        num_feature_calculation_funcs={key: stft_num_features for key in MODALITY_KEYS},
        model=mllm,
    )
    llm.resize_token_embeddings(len(tokenizer))
    return mllm, tokenizer, mm_processor

class BaseTrainer(pl.LightningModule):
    """Cornstarch 멀티모달 모델 학습을 위한 공통 Lightning 베이스 모듈."""

    def __init__(
        self,
        model, tokenizer, processor,
        *,
        apply_lora: bool = True,
        lr: float = 1e-3,
        encoder_mode: Optional[dict[str, tuple[bool, bool]]] = None,
        llm_mode: bool = True,
        generation_config: Optional[dict] = None,
        num_generations: int = 1,
        sample_log_interval: int | None = 10,
    ):
        super().__init__()
        self.model = model 
        self.tokenizer = tokenizer
        self.processor = processor
        self.lr = lr
        default_mode = {key: (False, True) for key in MODALITY_KEYS}
        self.encoder_mode = encoder_mode or default_mode
        self.llm_mode = llm_mode
        self.generation_config = generation_config.copy() if generation_config else {}
        self.num_generations = max(1, num_generations)
        self._align_modal_modules()
        self.sample_log_interval = sample_log_interval if (sample_log_interval and sample_log_interval > 0) else None
        self.wandb_run = None
        self._sample_table = None
        self._llm_quantized = bool(getattr(self.model.language_model, "is_quantized", False))

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def set_wandb_run(self, wandb_run) -> None:
        """Assign an active wandb run for manual logging."""
        self.wandb_run = wandb_run

    def _log_to_wandb(self, metrics: dict, *, step: Optional[int] = None) -> None:
        if self.wandb_run is None or not metrics:
            return
        log_step = step if step is not None else self.global_step
        self.wandb_run.log(metrics, step=log_step)

    def _should_log_samples(self, step: int) -> bool:
        return self.sample_log_interval is not None and step % self.sample_log_interval == 0

    def _align_modal_modules(self) -> None:
        """LLM과 STFT 인코더/프로젝터의 디바이스를 일치시킨다."""
        try:
            embed_layer = self.model.language_model.get_input_embeddings()
            lm_device = embed_layer.weight.device
            target_dtype = embed_layer.weight.dtype
        except Exception:
            lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            target_dtype = torch.float32
        if hasattr(self.model, "encoders"):
            for key, encoder in self.model.encoders.items():
                encoder.module.to(lm_device)
                encoder.projector.to(device=lm_device)
                projection_head = getattr(encoder.projector, "projection", None)
                if hasattr(projection_head, "set_target_dtype"):
                    projection_head.set_target_dtype(target_dtype)
        self._modal_device = lm_device

    def on_fit_start(self) -> None:
        self._align_modal_modules()
        llm_mode = self.llm_mode
        if llm_mode and self._llm_quantized:
            llm_mode = None
        self.model.train(
            encoders_mode=self.encoder_mode,
            llm_mode=llm_mode,
        )

    def _create_optimizer(self) -> Adam:
        return Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def configure_optimizers(self):
        return self._create_optimizer()

    def _configure_with_warmup(self, total_steps: int, warmup_ratio: float):
        optimizer = self._create_optimizer()
        num_training_steps = max(1, total_steps)
        num_warmup_steps = max(0, int(num_training_steps * warmup_ratio))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _prepare_generation_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        allowed_keys = {"input_ids", "attention_mask", *MODALITY_KEYS}
        device = self._modal_device
        gen_batch: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if key in allowed_keys and torch.is_tensor(value):
                gen_batch[key] = value.to(device)
        return gen_batch

    def generate_from_batch(
        self,
        batch: dict,
        *,
        num_return_sequences: Optional[int] = None,
        generation_overrides: Optional[dict] = None,
    ) -> torch.Tensor:
        if num_return_sequences is None:
            num_return_sequences = self.num_generations
        gen_kwargs = self.generation_config.copy()
        if generation_overrides:
            gen_kwargs.update(generation_overrides)
        gen_kwargs.setdefault("num_return_sequences", num_return_sequences)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        gen_batch = self._prepare_generation_batch(batch)
        return self.model.generate(**gen_batch, **gen_kwargs)

class SFTTrainer(BaseTrainer):
    """감독학습(SFT)을 위한 Lightning 모듈."""

    def __init__(
        self,
        model, tokenizer, processor,
        total_steps: int = 10,
        *,
        lr: float = 1e-3,
        warmup_ratio: float = 0.1,
        apply_lora: bool = True,
        generation_config: Optional[dict] = None,
        sample_log_interval: int | None = 10,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            apply_lora=apply_lora,
            lr=lr,
            llm_mode=False,
            generation_config=generation_config,
            num_generations=1,
            sample_log_interval=sample_log_interval,
        )
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio

    def training_step(self, batch, batch_idx: int):
        forward_inputs = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        outputs = self.model(**forward_inputs)
        loss = outputs.loss
        batch_size = batch["input_ids"].size(0) if torch.is_tensor(batch.get("input_ids")) else None
        self.log("train_loss", loss, on_step=True, prog_bar=True, batch_size=batch_size)
        current_step = self.global_step + 1
        self._log_to_wandb({"train/loss": float(loss.detach().item())}, step=current_step)
        return loss

    def configure_optimizers(self):
        return self._configure_with_warmup(self.total_steps, self.warmup_ratio)

class GRPOTrainer(BaseTrainer):
    """Cornstarch MultimodalModel용 GRPO 학습 Lightning 모듈."""

    def __init__(
        self,
        model, tokenizer, processor,
        reward_fns: list[Callable[[list[str], list[str], list[object]], list[float]]],
        *,
        reward_weights: Optional[list[float]] = None,
        num_generations: int = 2,
        total_steps: int = 10,
        warmup_ratio: float = 0.1,
        lr: float = 1e-3,
        apply_lora: bool = True,
        reward_fn_names: Optional[list[str]] = None,
        generation_config: Optional[dict] = None,
        sample_log_interval: int | None = 10,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            apply_lora=apply_lora,
            lr=lr,
            llm_mode=True,
            generation_config=generation_config,
            num_generations=num_generations,
            sample_log_interval=sample_log_interval,
        )
        self.reward_fns = reward_fns
        self.reward_weights = reward_weights
        if reward_fn_names is not None and len(reward_fn_names) != len(reward_fns):
            raise ValueError("reward_fn_names 길이는 reward_fns와 같아야 합니다.")
        self.reward_fn_names = reward_fn_names or [f"reward_{idx}" for idx in range(len(reward_fns))]
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        self.automatic_optimization = False
        self.generation_config.setdefault("do_sample", True)

    @torch.no_grad()
    def _generate(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """프롬프트 배치로부터 completions를 생성하고 반환."""
        device = self._modal_device
        attention_mask = batch["attention_mask"].to(device)
        prompt_lens = attention_mask.sum(dim=1)

        sequences = self.generate_from_batch(
            batch,
            num_return_sequences=self.num_generations,
        )
        return sequences, prompt_lens, attention_mask

    def _compute_logps_for_completions(
        self,
        sequences: torch.Tensor,
        prompt_lens: torch.Tensor,
        modal_tensors: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """completion 토큰에 대한 (per-token) logp와 마스크를 계산."""
        device = self._modal_device
        sequences = sequences.to(device)
        batch_size_gen = sequences.size(0)

        attn = (sequences != self.tokenizer.pad_token_id).long()
        inputs: dict[str, torch.Tensor] = {
            "input_ids": sequences,
            "attention_mask": attn,
            "labels": sequences.clone(),
        }

        for key, tensor in modal_tensors.items():
            tensor = tensor.to(device)
            if tensor.size(0) * self.num_generations == batch_size_gen:
                tensor = tensor.repeat_interleave(self.num_generations, dim=0)
            elif tensor.size(0) != batch_size_gen:
                raise ValueError(f"{key} batch size mismatch with generated sequences")
            inputs[key] = tensor

        outputs = self.model(**inputs)
        logits = outputs.logits
        logprobs = logits.log_softmax(dim=-1)

        next_token_ids = sequences[:, 1:]
        token_logps = torch.gather(
            logprobs[:, :-1, :],
            dim=-1,
            index=next_token_ids.unsqueeze(-1),
        ).squeeze(-1)

        prompt_lens_rep = prompt_lens.repeat_interleave(self.num_generations)
        L_total = sequences.size(1)
        idx = torch.arange(L_total - 1, device=device).unsqueeze(0).expand_as(token_logps)
        comp_mask = (idx >= (prompt_lens_rep.unsqueeze(1) - 1)) & (next_token_ids != self.tokenizer.pad_token_id)
        comp_mask = comp_mask.long()
        return token_logps, comp_mask

    def _decode_completions(self, sequences: torch.Tensor, prompt_lens: torch.Tensor) -> list[str]:
        seqs = sequences.cpu()
        plens = prompt_lens.cpu()
        completions: list[str] = []
        B = plens.size(0)
        for i in range(B):
            for g in range(self.num_generations):
                idx = i * self.num_generations + g
                comp_ids = seqs[idx, plens[i] :]
                text = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
                completions.append(text)
        return completions

    def training_step(self, batch, batch_idx: int):
        prompts = batch.get("prompts") or []
        gts = batch.get("gts") or []
        modal_tensors = {key: batch[key] for key in MODALITY_KEYS if key in batch}
        if len(modal_tensors) != len(MODALITY_KEYS):
            missing = [key for key in MODALITY_KEYS if key not in modal_tensors]
            raise KeyError(f"Missing modality tensors in batch: {missing}")
        opt = self.optimizers()

        with torch.no_grad():
            sequences, prompt_lens, _ = self._generate(batch)

        completions = self._decode_completions(sequences, prompt_lens)
        B = len(prompts)

        per_fn_rewards: list[torch.Tensor] = []
        prompts_rep = prompts * self.num_generations
        if isinstance(gts, list):
            gts_rep = gts * self.num_generations
        else:
            gts_rep = [gts] * (len(prompts) * self.num_generations)
        for fn in self.reward_fns:
            rewards = fn(prompts_rep, completions, gts_rep)
            per_fn_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=self._modal_device))

        if self.reward_weights is None:
            weights = [1.0 / len(per_fn_rewards)] * len(per_fn_rewards) if per_fn_rewards else []
        else:
            assert len(self.reward_weights) == len(per_fn_rewards), "reward_weights 길이가 reward_fns와 같아야 합니다."
            weights = self.reward_weights

        rewards = sum(w * r for w, r in zip(weights, per_fn_rewards)) if per_fn_rewards else torch.zeros(B * self.num_generations, device=self._modal_device)
        rewards = rewards.view(B, self.num_generations)
        group_mean = rewards.mean(dim=1, keepdim=True)
        advantages = (rewards - group_mean).view(-1)

        token_logps, comp_mask = self._compute_logps_for_completions(
            sequences,
            prompt_lens.to(self._modal_device),
            modal_tensors,
        )
        comp_lengths = comp_mask.sum(dim=1).clamp_min(1)
        seq_logp = (token_logps * comp_mask).sum(dim=1) / comp_lengths

        loss = -(seq_logp * advantages).mean()

        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        opt.step()
        schedulers = self.lr_schedulers()
        if isinstance(schedulers, list):
            for scheduler in schedulers:
                scheduler.step()
        elif schedulers is not None:
            schedulers.step()

        batch_size = B if B else len(seq_logp)
        avg_reward = rewards.mean()
        self.log("train_loss", loss, on_step=True, prog_bar=True, batch_size=batch_size)
        self.log("avg_reward", avg_reward, on_step=True, prog_bar=False, batch_size=batch_size)
        current_step = self.global_step + 1

        wandb_metrics = {
            "train/loss": float(loss.detach().item()),
            "train/avg_reward": float(avg_reward.detach().item()),
        }
        for name, tensor in zip(self.reward_fn_names, per_fn_rewards):
            wandb_metrics[f"reward/{name}"] = float(tensor.mean().detach().item())
        self._log_to_wandb(wandb_metrics, step=current_step)
        if self._should_log_samples(current_step):
            self._log_prompt_samples(prompts, completions, current_step)
        return loss

    def _log_prompt_samples(self, prompts: list[str], completions: list[str], step: int) -> None:
        if self.wandb_run is None or not prompts or not completions:
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if self._sample_table is None:
            self._sample_table = wandb.Table(
                columns=["step", "prompt_index", "generation_index", "prompt", "completion"]
            )
        for prompt_idx, prompt_text in enumerate(prompts):
            for gen_idx in range(self.num_generations):
                comp_idx = prompt_idx * self.num_generations + gen_idx
                if comp_idx >= len(completions):
                    break
                completion_text = completions[comp_idx]
                self._sample_table.add_data(step, prompt_idx, gen_idx, prompt_text, completion_text)
        preview_prompt = prompts[0]
        preview_completions = "\n\n".join(
            [
                f"[prompt {prompt_idx} gen {gen_idx}] {completions[prompt_idx * self.num_generations + gen_idx]}"
                for prompt_idx in range(min(len(prompts), 1))
                for gen_idx in range(min(self.num_generations, len(completions)))
                if (prompt_idx * self.num_generations + gen_idx) < len(completions)
            ]
        )
        self.wandb_run.log(
            {
                "samples_table": self._sample_table,
                "samples/latest_prompt": preview_prompt,
                "samples/latest_generations": preview_completions,
            },
            step=step,
        )

    def configure_optimizers(self):
        return self._configure_with_warmup(self.total_steps, self.warmup_ratio)

def collate_fn(batches: list[dict], processor: MultimodalProcessor) -> dict:
    """Cornstarch 프로세서를 활용해 모달리티별 입력을 LLM 형식으로 정리한다."""
    modal_inputs: dict[str, list[torch.Tensor]] = {key: [batch[key] for batch in batches] for key in MODALITY_KEYS}
    prompts = [batch.get("prompt", "") for batch in batches]
    gts = [batch.get("gt", "") for batch in batches]
    texts = [p + gt for p, gt in zip(prompts, gts)]

    processed = processor(
        encoder_inputs={key: {key: tensors} for key, tensors in modal_inputs.items()},
        llm_inputs={"text": texts, "padding": True},
        return_tensors="pt",
    )

    tokenizer = processor.llm_tokenizer
    max_length = processed["input_ids"].shape[1]
    offsets = None
    if getattr(tokenizer, "is_fast", False):
        offsets = tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )["offset_mapping"]

    labels = processed["input_ids"].clone()
    attention_mask = processed["attention_mask"]
    labels[attention_mask == 0] = -100

    if offsets is not None:
        prompt_char_lens = [len(p) for p in prompts]
        for i, prompt_char_len in enumerate(prompt_char_lens):
            if prompt_char_len <= 0:
                continue
            token_offsets = offsets[i]
            prompt_mask = token_offsets[:, 1] <= prompt_char_len
            labels[i][prompt_mask] = -100

    inputs: dict[str, torch.Tensor | list[str]] = {key: value for key, value in processed.items()}

    inputs["labels"] = labels
    inputs["prompts"] = prompts
    inputs["gts"] = gts

    return inputs

class MultimodalGRPOCollator:
    """GRPO 학습에 사용할 멀티모달 collator."""

    def __init__(self, processor: MultimodalProcessor):
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict:
        prompts = [item["prompt"] for item in features]
        gts = [item.get("gt") for item in features]

        encoder_inputs = {
            key: {key: [item[key] for item in features]}
            for key in MODALITY_KEYS
        }

        processed = self.processor(
            encoder_inputs=encoder_inputs,
            llm_inputs={"text": prompts, "padding": True, "truncation": True},
            return_tensors="pt",
        )

        batch: dict[str, torch.Tensor | list[str]] = {key: value for key, value in processed.items()}

        batch["prompts"] = prompts
        batch["gts"] = gts
        return batch

def get_trainer(
    args,
    mllm,
    tokenizer,
    mm_processor,
    reward_fns=None,
    reward_weights=None,
    sample_log_interval: int = 10,
):
    generation_config = {
        "max_new_tokens": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.top_k >= 0:
        generation_config["top_k"] = args.top_k
    if args.do_sample is not None:
        generation_config["do_sample"] = args.do_sample

    if args.trainer == "SFT":
        module = SFTTrainer(
            model=mllm,
            tokenizer=tokenizer,
            processor=mm_processor,
            total_steps=args.total_steps,
            lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            apply_lora=args.apply_lora,
            generation_config=generation_config,
            sample_log_interval=sample_log_interval,
        )
    elif args.trainer == "GRPO":
        reward_fn_names = [getattr(fn, "__name__", f"reward_{idx}") for idx, fn in enumerate(reward_fns)] if reward_fns else []
        module = GRPOTrainer(
            model=mllm,
            tokenizer=tokenizer,
            processor=mm_processor,
            reward_fns=reward_fns,
            reward_weights=reward_weights,
            num_generations=args.num_generations,
            total_steps=args.total_steps,
            warmup_ratio=args.warmup_ratio,
            lr=args.lr,
            apply_lora=args.apply_lora,
            generation_config=generation_config,
            reward_fn_names=reward_fn_names,
            sample_log_interval=sample_log_interval,
        )
    else:
        print(f"Wrong Trainer Type : {args.trainer}!")
        exit()

    # trainer_cfg = dict(
    #     max_steps=args.total_steps,
    #     enable_checkpointing=False,
    #     logger=False,
    #     enable_model_summary=False,
    #     strategy="auto",
    #     precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
    # )
    trainer_cfg = dict(
        max_steps=args.total_steps,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        accelerator="gpu", 
        devices=1,
        strategy="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
    )
    
    trainer = pl.Trainer(**trainer_cfg)
    
    return module, trainer

class FakeDataset(Dataset):
    """테스트용 데이터셋"""

    def __init__(self, length: int = 512):
        self.length = length
        self.x_stft = torch.tensor(np.random.rand(4, 224, 224), dtype=torch.float32)
        self.ref_stft = torch.tensor(np.random.rand(4, 224, 224), dtype=torch.float32)
        # 데모용 GT: 간단히 키워드(예: "이상")를 기대값으로 둔다.
        self.gt_token = "이상"

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> dict:
        prompt = "<x_stft> 측정 STFT와 <ref_stft> 기준 STFT를 비교해 이상 징후를 설명해줘."
        # 예시 GT: 답변에 "이상" 키워드가 포함되어야 한다고 가정
        gt = self.gt_token
        return {"prompt": prompt, "x_stft": self.x_stft, "ref_stft": self.ref_stft, "gt": gt}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cornstarch 멀티모달 + TRL GRPO 통합 예제")

    parser.add_argument("--trainer", type=str, choices=["SFT", "GRPO"], required=True, help="사용할 트레이너 유형")
    parser.add_argument("--apply_lora", type=bool, default=True, help="lora finetuning 적용 여부")
    parser.add_argument("--cache_dir", type=str, default='./llm_cache', help="llm_model 캐싱할 폴더")
    parser.add_argument("--llm_name", type=str, default='Qwen/Qwen3-4B-Instruct-2507', help="LLM 모델 이름")
    
    parser.add_argument("--total_steps",    type=int,   default=10,     help="학습 step 횟수")
    parser.add_argument("--lr",             type=float, default=1e-3,   help="학습 learning rate")
    parser.add_argument("--warmup_ratio",   type=float, default=0.1,    help="학습 warm-up ratio")
    
    parser.add_argument("--batch_size",    type=int,   default=4,     help="미니배치 크기")
    parser.add_argument("--num_generations",    type=int,   default=2,     help="GRPO 에서 response 생성 개수")
    parser.add_argument("--max_completion_length",    type=int,   default=128,     help="LLM 최대 응답 길이 제한 (max_new_tokens)")
    
    parser.add_argument("--temperature",             type=float, default=1.0,   help="LLM 생성 온도")
    parser.add_argument("--top_p",                  type=float, default=1.0,   help="LLM top-p 샘플링 값")
    parser.add_argument("--top_k",                  type=int,   default=-1,    help="LLM top-k 샘플링 값 (-1이면 비활성화)")
    parser.add_argument("--do_sample",              type=parse_optional_bool, default=None, help="샘플링 사용 여부 (true/false/none)")

    args = parser.parse_args()
    
    mllm, tokenizer, mm_processor = build_multimodal_system(
        apply_lora=args.apply_lora,
        cache_dir=args.cache_dir,
        llm_name=args.llm_name
    )
    
    reward_fns = []
    reward_weights = []
    
    if args.trainer == 'GRPO' and (len(reward_fns)==0 or len(reward_weights)==0):
        print('GRPO need reward functions & weights')
    
    module, trainer = get_trainer(
        args=args,
        mllm=mllm,
        tokenizer=tokenizer,
        mm_processor=mm_processor,
        reward_fns=reward_fns,
        reward_weights=reward_weights
    )
    
    train_dataset = FakeDataset()
    val_dataset = FakeDataset()
    
    if args.trainer == 'SFT':
        trainloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=functools.partial(collate_fn, processor=module.processor),
        )
        valloader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=functools.partial(collate_fn, processor=module.processor),
        )
    elif args.trainer == 'GRPO':
        data_collator = MultimodalGRPOCollator(processor=module.processor)
        trainloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=data_collator,
        )
        valloader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=data_collator,
        )
    else:
        print(f'Wrong Trainer Type : {args.trainer}!')
        exit()
    
    trainer.fit(module,
                train_dataloaders=trainloader,
                val_dataloaders=valloader)
