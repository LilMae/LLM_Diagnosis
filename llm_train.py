import argparse
import functools
import json
import os
import re
from typing import Optional, Tuple, Union, Callable

import torch

from LLM_trainer.trainer import (
    build_multimodal_system,
    get_trainer,
    collate_fn,
    MultimodalGRPOCollator,
    MODALITY_KEYS,
)
from data.dataset import get_dataset
from data.llm_dataset import get_llm_dataset
from torch.utils.data import DataLoader

from rewards import format_reward, accuracy_reward, fusion_reward, feature_usage_reward, no_hallucination_reward, structure_reward

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

def inference(
    checkpoint_path: str,
    prompts: Union[str, list[str]],
    modal_inputs: dict,
    *,
    args: argparse.Namespace,
    generation_overrides: Optional[dict] = None,
) -> list[str]:
    """
    저장된 MLLM 가중치를 로드해 주어진 모달 입력/프롬프트로 생성 결과를 반환한다.

    Args:
        checkpoint_path: PeriodicMLLMCheckpoint로 저장된 파일 경로.
        prompts: 단일 프롬프트 또는 프롬프트 리스트.
        modal_inputs: 각 모달리티 키(MODALITY_KEYS)에 대응하는 텐서/리스트.
        args: 학습 스크립트와 동일한 argparse.Namespace (모델 구성을 재현하는데 사용).
        generation_overrides: 추론시 사용할 generation kwargs 덮어쓰기.
    """
    prompts_list = [prompts] if isinstance(prompts, str) else list(prompts)
    if not prompts_list:
        raise ValueError("At least one prompt is required for inference.")

    mllm, tokenizer, mm_processor = build_multimodal_system(
        apply_lora=args.apply_lora,
        cache_dir=args.cache_dir,
        llm_name=args.llm_name,
        vib_enc_pth=args.vib_enc_pth,
        use_qlora=args.use_qlora,
        freeze_vib_encoder=args.freeze_vib_encoder if args.freeze_vib_encoder is not None else True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = mllm.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[Inference] Missing keys: {missing}, Unexpected keys: {unexpected}")

    generation_kwargs = {
        "max_new_tokens": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.top_k >= 0:
        generation_kwargs["top_k"] = args.top_k
    if args.do_sample is not None:
        generation_kwargs["do_sample"] = args.do_sample
    if generation_overrides:
        generation_kwargs.update(generation_overrides)

    def _expand_modal_value(value):
        if torch.is_tensor(value):
            return [v.detach().cpu() for v in value]
        if isinstance(value, list):
            return [torch.as_tensor(v) if not torch.is_tensor(v) else v for v in value]
        raise TypeError(f"Unsupported modal input type: {type(value)}")

    encoder_inputs = {}
    for key in MODALITY_KEYS:
        if key not in modal_inputs:
            raise KeyError(f"Missing modal input for '{key}'")
        samples = _expand_modal_value(modal_inputs[key])
        if len(samples) != len(prompts_list):
            raise ValueError(f"Modal input '{key}' batch({len(samples)}) != prompts({len(prompts_list)})")
        encoder_inputs[key] = {key: samples}

    processed = mm_processor(
        encoder_inputs=encoder_inputs,
        llm_inputs={"text": prompts_list, "padding": True, "truncation": True},
        return_tensors="pt",
    )

    model_inputs = {
        "input_ids": processed["input_ids"],
        "attention_mask": processed["attention_mask"],
    }
    for key in MODALITY_KEYS:
        model_inputs[key] = processed[key]

    device = mllm.language_model.get_input_embeddings().weight.device
    model_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in model_inputs.items()}
    mllm.eval()
    with torch.no_grad():
        sequences = mllm.generate(**model_inputs, **generation_kwargs)
    return tokenizer.batch_decode(sequences, skip_special_tokens=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cornstarch 멀티모달 + TRL GRPO 통합 예제")
    # 모델 옵션
    parser.add_argument("--trainer", type=str, choices=["SFT", "GRPO"], required=True, help="사용할 트레이너 유형")
    parser.add_argument("--apply_lora", type=bool, default=True, help="lora finetuning 적용 여부")
    parser.add_argument("--use_qlora", action="store_true", help="4bit QLoRA 양자화/미세튜닝 사용 여부")
    parser.add_argument("--cache_dir", type=str, default='./llm_cache', help="llm_model 캐싱할 폴더")
    parser.add_argument("--llm_name", type=str, default='Qwen/Qwen3-4B-Instruct-2507', help="LLM 모델 이름")
    parser.add_argument("--vib_enc_pth", type=str, default=None, help="학습된 vib_encoder 가중치 경로")
    parser.add_argument("--freeze_vib_encoder", type=parse_optional_bool, default=True, help="vibration encoder weight를 고정할지 여부 (기본: True)")
    parser.add_argument("--checkpoint_dir", type=str, default="/workspace/checkpoints/mllm", help="MLLM 체크포인트를 저장할 디렉토리")
    parser.add_argument("--save_every_n_steps", type=int, default=10, help="MLLM 모델 저장 주기 (iteration 단위)")
    
    # 학습 옵션
    parser.add_argument("--total_steps",    type=int,   default=50,     help="학습 step 횟수")
    parser.add_argument("--lr",             type=float, default=1e-3,   help="학습 learning rate")
    parser.add_argument("--warmup_ratio",   type=float, default=0.1,    help="학습 warm-up ratio")
    
    # 데이터셋 옵션
    parser.add_argument("--batch_size",    type=int,   default=1,     help="미니배치 크기")
    parser.add_argument("--num_workers",    type=int,   default=16,     help="미니배치 크기")
    parser.add_argument("--data_root",    type=str,   default='',     help="dataset 경로")
    parser.add_argument("--train_domain",    type=str, nargs='+',   default=['vat', 'mfd', 'dxai', 'vbl'],     help="dataset 경로")
    parser.add_argument("--valid_domain",    type=str, nargs='+',   default=['vat', 'mfd', 'dxai', 'vbl'],     help="dataset 경로")
    
    # LLM 데이터셋 옵션
    parser.add_argument("--embedding_model",    type=str,   default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',     help="RAG를 구축할 임베딩 모델")
    parser.add_argument("--docs_path",    type=str,   default='/workspace/docs_path',     help="RAG에 사용될 문서 폴더")
    parser.add_argument("--retriever_k",    type=int,   default=4,     help="RAG에서 Retreive 개수")
    
    # 생성 옵션
    parser.add_argument("--num_generations",    type=int,   default=2,     help="GRPO 에서 response 생성 개수")
    parser.add_argument("--max_completion_length",    type=int,   default=1024,     help="LLM 최대 응답 길이 제한 (max_new_tokens)")
    parser.add_argument("--temperature",             type=float, default=1.0,   help="LLM 생성 온도")
    parser.add_argument("--top_p",                  type=float, default=1.0,   help="LLM top-p 샘플링 값")
    parser.add_argument("--top_k",                  type=int,   default=-1,    help="LLM top-k 샘플링 값 (-1이면 비활성화)")
    parser.add_argument("--do_sample",              type=parse_optional_bool, default=None, help="샘플링 사용 여부 (true/false/none)")
    parser.add_argument("--sample_log_interval", type=int, default=10, help="W&B에 prompt/response 샘플을 기록할 step 간격 (0이면 비활성화)")
    parser.add_argument("--use_wandb", action="store_true", help="Weights & Biases 로깅 사용 여부")
    parser.add_argument("--wandb_project", type=str, default="LLM_Diagnosis", help="Weights & Biases 프로젝트 이름")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases 런 이름")

    args = parser.parse_args()
    args.save_every_n_steps = max(1, args.save_every_n_steps)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    wandb_run = None
    if args.use_wandb:
        import wandb  # type: ignore
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    
    freeze_vib = True if args.freeze_vib_encoder is None else args.freeze_vib_encoder
        
    mllm, tokenizer, mm_processor = build_multimodal_system(
        apply_lora=args.apply_lora,
        cache_dir=args.cache_dir,
        llm_name=args.llm_name,
        vib_enc_pth=args.vib_enc_pth,
        use_qlora=args.use_qlora,
        freeze_vib_encoder=freeze_vib,
    )
    
    reward_fns = [format_reward, 
                  accuracy_reward, 
                  fusion_reward, 
                  feature_usage_reward,
                  no_hallucination_reward
                  ]
    reward_weights = [1.0,
                      2.0,
                      0.5,
                      0.5,
                      0.5]
    
    if args.trainer == 'GRPO' and (len(reward_fns)==0 or len(reward_weights)==0):
        print('GRPO need reward functions & weights')
    
    module, trainer = get_trainer(
        args=args,
        mllm=mllm,
        tokenizer=tokenizer,
        mm_processor=mm_processor,
        reward_fns=reward_fns,
        reward_weights=reward_weights,
        sample_log_interval=args.sample_log_interval,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_steps=args.save_every_n_steps,
    )

    if wandb_run is not None:
        module.set_wandb_run(wandb_run)
    
    """
    Dataloader 생성
    변환 순서 : data.dataset.VibrationDataset -> data.llm_dataset.LLM_Dataset -> torch.utils.data.DataLoader
    """
    train_dataset, val_dataset = get_dataset(args, train_domain=args.train_domain,
                valid_domain=args.valid_domain)
    train_llm_dataset, val_llm_dataset = get_llm_dataset(args, train_dataset, val_dataset)

    if args.trainer == 'SFT':
        trainloader = DataLoader(
            dataset=train_llm_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=False,
            collate_fn=functools.partial(collate_fn, processor=module.processor),
        )
        valloader = DataLoader(
            dataset=val_llm_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=False,
            collate_fn=functools.partial(collate_fn, processor=module.processor),
        )
    elif args.trainer == 'GRPO':
        data_collator = MultimodalGRPOCollator(processor=module.processor)
        trainloader = DataLoader(
            dataset=train_llm_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=False,
            collate_fn=data_collator,
        )
        valloader = DataLoader(
            dataset=val_llm_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=False,
            collate_fn=data_collator,
        )
    else:
        print(f'Wrong Trainer Type : {args.trainer}!')
        exit()
    
    try:
        trainer.fit(module,
                    train_dataloaders=trainloader,
                    val_dataloaders=valloader)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
