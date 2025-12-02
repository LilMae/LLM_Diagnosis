import argparse
import json
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from LLM_trainer.trainer import (
    MODALITY_KEYS,
    MultimodalGRPOCollator,
    build_multimodal_system,
)
from data.dataset import get_dataset
from data.llm_dataset import get_llm_dataset


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


def build_generation_config(args: argparse.Namespace, tokenizer) -> dict:
    gen_cfg = {
        "max_new_tokens": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_return_sequences": max(1, args.num_generations),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.top_k >= 0:
        gen_cfg["top_k"] = args.top_k
    if args.do_sample is not None:
        gen_cfg["do_sample"] = args.do_sample
    return gen_cfg


def run_inference(args: argparse.Namespace) -> None:
    freeze_vib = True if args.freeze_vib_encoder is None else args.freeze_vib_encoder
    mllm, tokenizer, mm_processor = build_multimodal_system(
        apply_lora=args.apply_lora,
        cache_dir=args.cache_dir,
        llm_name=args.llm_name,
        vib_enc_pth=args.vib_enc_pth,
        use_qlora=args.use_qlora,
        freeze_vib_encoder=freeze_vib,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = mllm.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Checkpoint] Missing keys while loading: {missing}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys while loading: {unexpected}")

    mllm.eval()
    device = mllm.language_model.get_input_embeddings().weight.device

    train_dataset, val_dataset = get_dataset(
        args,
        train_domain=args.train_domain,
        valid_domain=args.valid_domain,
    )
    _, val_llm_dataset = get_llm_dataset(args, train_dataset, val_dataset)
    collator = MultimodalGRPOCollator(processor=mm_processor)
    valloader = DataLoader(
        dataset=val_llm_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
    )

    generation_cfg = build_generation_config(args, tokenizer)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    results: list[dict] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(valloader):
            model_inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            for key in MODALITY_KEYS:
                if key in batch:
                    model_inputs[key] = batch[key].to(device)

            sequences = mllm.generate(**model_inputs, **generation_cfg)
            generations = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            prompts = batch.get("prompts", [])
            gts = batch.get("gts", [])
            num_generations = generation_cfg["num_return_sequences"]

            for idx, prompt in enumerate(prompts):
                gt = gts[idx] if idx < len(gts) else None
                for gen_idx in range(num_generations):
                    list_idx = idx * num_generations + gen_idx
                    if list_idx >= len(generations):
                        break
                    results.append(
                        {
                            "batch_index": batch_idx,
                            "sample_index": idx,
                            "generation_index": gen_idx,
                            "prompt": prompt,
                            "gt": gt,
                            "generation": generations[list_idx],
                        }
                    )

    with open(args.output_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[Inference] Saved {len(results)} generations to {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description="MLLM validation inference script")
    # 모델 옵션
    parser.add_argument("--trainer", type=str, choices=["SFT", "GRPO"], required=True, help="사용할 트레이너 유형")
    parser.add_argument("--apply_lora", type=bool, default=True, help="lora finetuning 적용 여부")
    parser.add_argument("--use_qlora", action="store_true", help="4bit QLoRA 양자화/미세튜닝 사용 여부")
    parser.add_argument("--cache_dir", type=str, default="./llm_cache", help="llm_model 캐싱할 폴더")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="LLM 모델 이름")
    parser.add_argument("--vib_enc_pth", type=str, default="LLM_Diagnosis/checkpoints/best_model_0.pth", help="학습된 vib_encoder 가중치 경로")
    parser.add_argument("--freeze_vib_encoder", type=parse_optional_bool, default=True, help="vibration encoder weight를 고정할지 여부 (기본: True)")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="저장된 MLLM 체크포인트 경로")

    # 학습 옵션
    parser.add_argument("--total_steps", type=int, default=50, help="학습 step 횟수")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습 learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="학습 warm-up ratio")

    # 데이터셋 옵션
    parser.add_argument("--batch_size", type=int, default=1, help="미니배치 크기")
    parser.add_argument("--data_root", type=str, default="", help="dataset 경로")
    parser.add_argument("--train_domain", type=str, nargs="+", default=["vat", "vbl", "mfd", "dxai"], help="train dataset domains")
    parser.add_argument("--valid_domain", type=str, nargs="+", default=["vat", "vbl", "mfd", "dxai"], help="valid dataset domains")

    # LLM 데이터셋 옵션
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", help="RAG를 구축할 임베딩 모델")
    parser.add_argument("--docs_path", type=str, default="/workspace/docs_path", help="RAG에 사용될 문서 폴더")
    parser.add_argument("--retriever_k", type=int, default=4, help="RAG에서 Retreive 개수")

    # 생성 옵션
    parser.add_argument("--num_generations", type=int, default=1, help="프롬프트당 생성 개수")
    parser.add_argument("--max_completion_length", type=int, default=1024, help="LLM 최대 응답 길이 제한 (max_new_tokens)")
    parser.add_argument("--temperature", type=float, default=1.0, help="LLM 생성 온도")
    parser.add_argument("--top_p", type=float, default=1.0, help="LLM top-p 샘플링 값")
    parser.add_argument("--top_k", type=int, default=-1, help="LLM top-k 샘플링 값 (-1이면 비활성화)")
    parser.add_argument("--do_sample", type=parse_optional_bool, default=None, help="샘플링 사용 여부 (true/false/none)")

    # 출력 옵션
    parser.add_argument("--output_path", type=str, default="LLM_Diagnosis/inference_outputs.jsonl", help="생성 결과를 저장할 경로")

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
