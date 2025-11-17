import argparse
import functools
from typing import Optional, Tuple, Union, Callable
from LLM_trainer.trainer import build_multimodal_system, get_trainer, collate_fn, MultimodalGRPOCollator
from data.dataset import get_dataset
from data.llm_dataset import get_llm_dataset
from torch.utils.data import DataLoader


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cornstarch 멀티모달 + TRL GRPO 통합 예제")
    # 모델 옵션
    parser.add_argument("--trainer", type=str, choices=["SFT", "GRPO"], required=True, help="사용할 트레이너 유형")
    parser.add_argument("--apply_lora", type=bool, default=True, help="lora finetuning 적용 여부")
    parser.add_argument("--cache_dir", type=str, default='./llm_cache', help="llm_model 캐싱할 폴더")
    parser.add_argument("--llm_name", type=str, default='Qwen/Qwen3-4B-Instruct-2507', help="LLM 모델 이름")
    parser.add_argument("--vib_enc_pth", type=str, default='LLM_Diagnosis/checkpoints/best_model_0.pth', help="학습된 vib_encoder 가중치 경로")
    
    # 학습 옵션
    parser.add_argument("--total_steps",    type=int,   default=10,     help="학습 step 횟수")
    parser.add_argument("--lr",             type=float, default=1e-3,   help="학습 learning rate")
    parser.add_argument("--warmup_ratio",   type=float, default=0.1,    help="학습 warm-up ratio")
    
    # 데이터셋 옵션
    parser.add_argument("--batch_size",    type=int,   default=4,     help="미니배치 크기")
    parser.add_argument("--data_root",    type=str,   default='',     help="dataset 경로")
    
    # 생성 옵션
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
        llm_name=args.llm_name,
        vib_enc_pth=args.vib_enc_pth
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
    
    """
    Dataloader 생성
    변환 순서 : data.dataset.VibrationDataset -> data.llm_dataset.LLM_Dataset -> torch.utils.data.DataLoader
    """
    train_dataset, val_dataset = get_dataset(args, train_domain=['vat', 'vbl', 'mfd'],
                valid_domain=['dxai'])
    train_llm_dataset, val_llm_dataset = get_llm_dataset(train_dataset, val_dataset)
    if args.trainer == 'SFT':
        trainloader = DataLoader(
            dataset=train_llm_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=functools.partial(collate_fn, processor=module.processor),
        )
        valloader = DataLoader(
            dataset=val_llm_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=functools.partial(collate_fn, processor=module.processor),
        )
    elif args.trainer == 'GRPO':
        data_collator = MultimodalGRPOCollator(processor=module.processor)
        trainloader = DataLoader(
            dataset=train_llm_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=data_collator,
        )
        valloader = DataLoader(
            dataset=val_llm_dataset,
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
