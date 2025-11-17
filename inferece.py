#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inference_dxai.py — VisionTransformerAE 추론 스크립트 (dxai 전용)
- VibrationDataset + OrderInvariantSignalImager 사용 (이미징/리컨 저장 없음)
- 단일 GPU, AMP(bfloat16/float16) 자동 사용
- 데이터셋 샘플 3개 정보 출력
- Accuracy 및 Error Rate(=1-Accuracy) 출력
"""

import os
import argparse
from functools import partial
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# === 프로젝트 의존 모듈 (학습 코드와 동일 경로 가정) ===
from data.dataset import VibrationDataset, OrderInvariantSignalImager
from tokenizer_trainer.models.ViT_pytorch import VisionTransformerAE
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# CuDNN 설정 — 재현성 보장
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except:
    pass

# ----------------------------- 유틸 -----------------------------

def vib_collate(batch):
    """
    학습 코드와 동일: 모델이 바로 쓰는 키만 stack, 나머지는 리스트로 유지
    """
    out = {}
    must_stack = {"x_stft", "ref_stft", "x_cls"}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        if k in must_stack:
            vals = [torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for v in vals]
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def load_checkpoint(model: nn.Module, ckpt_path: str, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    meta = {
        "epoch": ckpt.get("epoch", None),
        "val_acc": ckpt.get("val_acc", None),
    }
    return meta


def pretty_header(title: str):
    line = "─" * 60
    print(f"\n{line}\n{title}\n{line}")


def _shape(t):
    return tuple(t.shape) if isinstance(t, torch.Tensor) else None


def print_sample_info(sample, idx: int):
    # 안전하게 접근: 키가 없을 수 있으므로 get 사용
    x = sample.get("x_stft", None)
    ref = sample.get("ref_stft", None)
    y = sample.get("x_cls", None)
    if idx ==2:
        print(x[:,:10,:10])
    print(f"[Sample {idx}]")
    print(f"  x_stft    : {_shape(x)} dtype={getattr(x, 'dtype', None)}")
    print(f"  ref_stft  : {_shape(ref)} dtype={getattr(ref, 'dtype', None)}")
    if isinstance(y, torch.Tensor):
        try:
            yv = int(y.item()) if y.numel() == 1 else y.tolist()
        except Exception:
            yv = str(y)
        print(f"  label     : {yv}")
    else:
        print(f"  label     : None")


# ----------------------------- 추론 루프 -----------------------------

@torch.no_grad()
def run_inference(args):
    assert torch.cuda.is_available(), "CUDA is not available."
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # AMP 설정
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # -------- 이미지 변환기 (학습과 동일 파이프라인) --------
    output_size = 224 if args.pretrained else args.image_size
    signal_imger = OrderInvariantSignalImager(
        mode=args.stft_mode,
        log1p=True,
        normalize="per_channel",
        eps=1e-8,
        out_dtype=torch.float32,
        max_order=args.max_order,
        H_out=output_size,
        W_out=output_size,
        stft_nperseg=args.stft_nperseg,
        stft_hop=args.stft_hop,
        stft_window="hann",
        stft_center=True,
        stft_power=args.stft_power,
    )

    # -------- 데이터셋: dxai만 사용 --------
    data_root_abs = os.path.abspath(args.data_root)
    dataset = VibrationDataset(
        data_root=data_root_abs,
        using_dataset=['dxai'],              # 고정: dxai만 사용
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        transform=signal_imger
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(2, min(4, (os.cpu_count() or 8) // 2)),
        pin_memory=True,
        collate_fn=vib_collate
    )

    # -------- 데이터셋 샘플 3개 출력 --------
    pretty_header("Dataset Samples (Top 3)")
    sample_count = min(3, len(dataset))
    for i in range(sample_count):
        print_sample_info(dataset[i], i)
    # -------- 모델 구성 & 체크포인트 로드 --------
    pretty_header("Model & Checkpoint")
    model = VisionTransformerAE(
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        dropout=0.0,
        attention_dropout=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        image_size=224,
        image_channel=4,
        patch_size=16,
        masking_ratio=0.75,
        num_classes=args.num_classes,
    ).to(device)

    model.eval()

    meta = load_checkpoint(model, args.checkpoint, map_location=device)
    print(f"[OK] Loaded checkpoint: {args.checkpoint}")
    print(f"     epoch={meta.get('epoch')}, val_acc={meta.get('val_acc')}")

    # -------- 추론 --------
    pretty_header("Inference")
    use_bf16 = torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    total = 0
    labeled = 0
    correct = 0
    val_preds_local = []
    val_labels_local = []
    val_loss_sum_local = 0.0
    val_correct_local  = 0.0
    val_total_local    = 0.0

    start_tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[START] {start_tag} | batches={len(loader)} | batch_size={args.batch_size}")

    with torch.no_grad():
        for i, batch  in enumerate(loader):
            x = batch['x_stft'].to(device, non_blocking=True)
            y = batch['x_cls'].to(device, non_blocking=True)
            ref_x = batch['ref_stft'].to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                rec, _, masked_idx = model.forward_mae(img=x)
                # b_loss_mae = net.calculate_mae_loss(rec, x, masked_idx)
                b_loss_mae = nn.MSELoss()(rec, x)

                logits, diff = model.forward_classify(current_img=x, normal_img=ref_x)

            bs = y.size(0)
            pred = logits.argmax(dim=1)
            val_correct_local += (pred == y).sum().item()
            val_total_local += bs

            val_preds_local.append(pred.detach())
            val_labels_local.append(y.detach())

    t_val = torch.tensor([val_correct_local, val_total_local], dtype=torch.float64, device=device)
    val_acc  = (t_val[0] / t_val[1] * 100.0).item() if t_val[1] > 0 else 0.0

    # -------- 결과 요약 --------
    pretty_header("Results")
    acc = val_acc
    err = 100.0 - acc
    print(f"Total samples   : {val_total_local}")
    print(f"Accuracy        : {acc:.6f}")
    print(f"Error Rate      : {err:.6f}")
    
    end_tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[END] {end_tag}")


# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Inference for VisionTransformerAE on dxai dataset (no caching, no image saving)")

    # 데이터/전처리
    p.add_argument('--data_root', type=str, default='data/processed', help='Processed data root (dxai 데이터 포함)')
    p.add_argument('--window_sec', type=float, default=5.0)
    p.add_argument('--stride_sec', type=float, default=3.0)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--max_order', type=float, default=20.0)

    p.add_argument('--stft_mode', type=str, default='stft+cross', choices=['stft', 'stft+cross', 'stft_complex'])
    p.add_argument('--stft_nperseg', type=int, default=1024)
    p.add_argument('--stft_hop', type=int, default=256)
    p.add_argument('--stft_power', type=float, default=1.0)

    # 모델/체크포인트
    p.add_argument('--num_classes', type=int, default=5)
    p.add_argument('--pretrained', action='store_true', help='(참고) 학습 시 ImageNet 프리트레인 사용 여부. 추론 입력 크기 결정용 플래그.')
    p.add_argument('--checkpoint', type=str, required=True, help='학습된 모델 체크포인트(.pth)')

    # 실행
    p.add_argument('--batch_size', type=int, default=256)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
