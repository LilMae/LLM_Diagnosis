import argparse
import csv

from LLM_trainer.trainer import (
    MODALITY_KEYS,
    MultimodalGRPOCollator,
    build_multimodal_system,
)
import torch

from tqdm import tqdm
from data.dataset import CachedDataset
from torch.utils.data import DataLoader
from rewards import extract_blocks, parse_answer_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cornstarch 멀티모달 + TRL GRPO 통합 예제")
    # 모델 옵션
    parser.add_argument("--apply_lora", type=bool, default=True, help="lora finetuning 적용 여부")
    parser.add_argument("--use_qlora", action="store_true", help="4bit QLoRA 양자화/미세튜닝 사용 여부")
    parser.add_argument("--cache_dir", type=str, default='./llm_cache', help="llm_model 캐싱할 폴더")
    parser.add_argument("--llm_name", type=str, default='Qwen/Qwen3-4B-Instruct-2507', help="LLM 모델 이름")
    parser.add_argument("--vib_enc_pth", type=str, default=None, help="학습된 vib_encoder 가중치 경로")
    parser.add_argument("--freeze_vib_encoder", type=bool, default=True, help="vibration encoder weight를 고정할지 여부 (기본: True)")
    parser.add_argument("--checkpoint_path", type=str, default='/workspace/GRPO_Testing/mllm-step-step003400-20251204-051010.pt')
    
    #데이터 옵션
    parser.add_argument("--data_cache_path", type=str, default='/workspace/llm_cache/LLM_Train_vat+mfd+dxai+vbl_Val_vat+mfd+dxai+vbl/valid.pt')
    parser.add_argument("--batch_size",    type=int,   default=8,     help="미니배치 크기")
    parser.add_argument("--result_path", type=str, default='/workspace/test.csv')
    parser.add_argument("--err_path", type=str, default='/workspace/err.csv')

    args = parser.parse_args()
    
    mllm, tokenizer, mm_processor = build_multimodal_system(
        apply_lora=True,
        cache_dir=args.cache_dir,
        llm_name=args.llm_name,
        vib_enc_pth=args.vib_enc_pth,
        use_qlora=True,
        freeze_vib_encoder=True,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = mllm.load_state_dict(state_dict, strict=False)
    
    val_llm_dataset = CachedDataset(data_root=args.data_cache_path)
    collator = MultimodalGRPOCollator(processor=mm_processor)
    valloader = DataLoader(
        dataset=val_llm_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
    )
    
    generation_cfg = {
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    language_model = getattr(mllm, "language_model", None)
    if language_model is not None:
        language_model.eval()

    encoders = getattr(mllm, "encoders", {}) or {}
    for encoder in encoders.values():
        module = getattr(encoder, "module", None)
        projector = getattr(encoder, "projector", None)
        if module is not None:
            module.eval()
        if projector is not None:
            projector.eval()
            
    device = mllm.language_model.get_input_embeddings().weight.device

    result_fields = ["batch_index", "prompt", "response", "gt", "vib_only_label", "knowledge_only_label", "final_label"]
    err_fields = ["batch_index", "prompt", "gt", "raw_answer"]

    with open(args.result_path, "w", encoding="utf-8", newline="") as result_file, open(
        args.err_path, "w", encoding="utf-8", newline=""
    ) as err_file:
        result_writer = csv.DictWriter(result_file, fieldnames=result_fields)
        result_writer.writeheader()
        err_writer = csv.DictWriter(err_file, fieldnames=err_fields)
        err_writer.writeheader()

        for batch_idx, batch in tqdm(enumerate(valloader)):
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

            batch_records = []
            batch_errors = []
            for prompt, gt, generation in zip(prompts, gts, generations):
                response = generation[len(prompt) :]
                _, answer_json_str = extract_blocks(response)
                answer_json = parse_answer_json(answer_json_str)

                if isinstance(answer_json, dict):
                    batch_records.append(
                        {
                            "batch_index": batch_idx,
                            "prompt": prompt,
                            "response" : response,
                            "gt": gt,
                            "vib_only_label": answer_json.get("vib_only_label"),
                            "knowledge_only_label": answer_json.get("knowledge_only_label"),
                            "final_label": answer_json.get("final_label"),
                        }
                    )
                else:
                    batch_errors.append(
                        {
                            "batch_index": batch_idx,
                            "prompt": prompt,
                            "gt": gt,
                            "raw_answer": answer_json_str or response,
                        }
                    )

            for row in batch_records:
                result_writer.writerow(row)
            for row in batch_errors:
                err_writer.writerow(row)

            result_file.flush()
            err_file.flush()
