import argparse
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import get_dataset
from data.llm_dataset import get_llm_dataset


def _identity_collate(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    """DataLoader에서 dict를 그대로 전달하기 위한 collate 함수."""
    if len(batch) != 1:
        raise ValueError("이 캐싱 스크립트는 batch_size=1만 지원합니다.")
    return batch[0]


def cache_split(
    split_name: str,
    llm_dataset,
    save_path: str,
    *,
    num_workers: int,
    prefetch_factor: int,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    feature_dataset = llm_dataset.feature_dataset
    loader_kwargs: Dict[str, Any] = dict(
        dataset=feature_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=_identity_collate,
        num_workers=max(0, num_workers),
    )
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=max(1, prefetch_factor),
        )
    loader = DataLoader(**loader_kwargs)

    cached_samples: list[Dict[str, Any]] = []
    for feature_sample in tqdm(
        loader,
        total=len(feature_dataset),
        dynamic_ncols=True,
        desc=f"Caching {split_name}",
    ):
        cur_status = feature_sample.get("cur_status", {})
        plan = llm_dataset.planner(cur_status)
        prompt = llm_dataset._create_prompt(cur_status, plan)

        sample = dict(feature_sample)
        sample["prompt"] = prompt
        cached_samples.append(sample)

    torch.save(cached_samples, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM Dataset 미리 생성/캐싱 스크립트")

    parser.add_argument("--cache_dir", type=str, default='./llm_cache', help="LLM 모델/데이터 캐시 폴더")
    parser.add_argument("--llm_name", type=str, default='Qwen/Qwen3-4B-Instruct-2507', help="LLM 모델 이름")

    # 데이터셋 옵션
    parser.add_argument("--batch_size", type=int, default=1, help="(미사용) placeholder")
    parser.add_argument("--data_root", type=str, default='', help="dataset 경로")
    parser.add_argument("--train_domain", type=str, nargs='+', default=['vat', 'mfd', 'dxai', 'vbl'], help="train domain")
    parser.add_argument("--valid_domain", type=str, nargs='+', default=['vat', 'mfd', 'dxai', 'vbl'], help="valid domain")
    parser.add_argument("--num_workers", type=int, default=4, help="feature 추출용 DataLoader worker 수")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor(worker>0일 때)")

    # LLM 데이터셋 옵션
    parser.add_argument("--embedding_model", type=str, default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', help="RAG 임베딩 모델")
    parser.add_argument("--docs_path", type=str, default='/workspace/docs_path', help="RAG 문서 폴더")
    parser.add_argument("--retriever_k", type=int, default=4, help="RAG retrieve 개수")

    args = parser.parse_args()

    train_dataset, val_dataset = get_dataset(
        args,
        train_domain=args.train_domain,
        valid_domain=args.valid_domain,
    )
    train_llm_dataset, val_llm_dataset = get_llm_dataset(args, train_dataset, val_dataset)

    exp_name = 'LLM_Train_' + '+'.join(args.train_domain) + '_Val_' + '+'.join(args.valid_domain)
    data_cache_root = os.path.join(args.cache_dir, exp_name)
    train_pt = os.path.join(data_cache_root, 'train.pt')
    valid_pt = os.path.join(data_cache_root, 'valid.pt')

    cache_split(
        "valid",
        val_llm_dataset,
        valid_pt,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    cache_split(
        "train",
        train_llm_dataset,
        train_pt,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
