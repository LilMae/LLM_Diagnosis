from datetime import datetime
import os
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from functools import partial

from data.dataset import VibrationDataset, OrderInvariantSignalImager, CachedDataset
from sklearn.metrics import precision_score, recall_score, f1_score

from tokenizer_trainer.models.ViT_pytorch import VisionTransformerAE
from tokenizer_trainer.visualize import create_reconstruction_figure

import wandb
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import types

# 성능 옵션
# torch.backends.cudnn.benchmark = True

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.fp32_precision = 'tf32'
# torch.backends.cudnn.conv.fp32_precision = 'tf32'

# torch.backends.cuda.matmul.fp32_precision = 'ieee'
# torch.backends.cudnn.conv.fp32_precision = 'ieee'

# try:
#     torch.set_float32_matmul_precision("high")
# except Exception:
#     pass

def unwrap_ddp(model):
    return model.module if isinstance(model, DDP) else model

def _ddp_sum_tensor(t):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def _downsample_chw(t: torch.Tensor, factor: int) -> torch.Tensor:
    """Stride 다운샘플(보간 없음) — 빠르고 메모리 절약"""
    if factor <= 1:
        return t
    return t[:, ::factor, ::factor]
def _channel_labels_for_mode(mode: str):
    m = (mode or "").lower()
    if m == "stft":
        return ["|X|^p", "|Y|^p"]
    if m == "stft+cross":
        return ["|X|^p", "|Y|^p", "|X·Y*|", "cos(Δφ)"]
    if m == "stft_complex":
        return ["|Z|", "cos(∠Z)", "sin(∠Z)", "∠Z−90°"]
    return [f"ch{i}" for i in range(16)]

def _to_uint8_gray2d(t2d: torch.Tensor) -> np.ndarray:
    """(H,W) float/half -> (H,W) uint8, per-image min-max 정규화"""
    x = t2d.detach().float().cpu()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x * 255.0).clamp(0, 255).byte().numpy()

def _side_by_side_gray(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(H,W) uint8 두 장을 가로로 붙여 (H,2W,3)"""
    if a.shape != b.shape:
        # 높이 기준 최근접 스케일
        H = min(a.shape[0], b.shape[0])
        def resize_w(h, img):
            scale = H / img.shape[0]
            new_w = max(1, int(round(img.shape[1] * scale)))
            x_old = np.linspace(0, img.shape[1]-1, img.shape[1])
            x_new = np.linspace(0, img.shape[1]-1, new_w)
            out = np.empty((H, new_w), dtype=img.dtype)
            for r in range(H):
                out[r, :] = np.interp(x_new, x_old, img[int(r/scale), :]).astype(np.uint8)
            return out
        a = resize_w(H, a); b = resize_w(H, b)
    # 그레이 → RGB 반복
    a3 = np.repeat(a[:, :, None], 3, axis=2)
    b3 = np.repeat(b[:, :, None], 3, axis=2)
    return np.concatenate([a3, b3], axis=1)

def log_batch_recon_single_channel_to_wandb(
    x: torch.Tensor,                # (B,C,H,W)
    rec: torch.Tensor,              # (B,C,H,W)
    tag: str,
    epoch: int,
    mode: str,
    ch_idx: int = 0,                # 사용할 채널 인덱스
    k: int = 4,                     # 배치에서 로그할 샘플 수
    downsample: int = 2
):
    """
    4채널(또는 C개) 중 ch_idx 채널만 선택해서 origin vs recon 2분할로 로깅.
    rank0에서만 호출할 것(외부 보장).
    """
    if wandb.run is None:
        return
    B, C, H, W = x.shape
    if C == 0:
        return
    ch_idx = int(ch_idx) % C  # 안전

    labels = _channel_labels_for_mode(mode)
    ch_name = labels[ch_idx] if ch_idx < len(labels) else f"ch{ch_idx}"

    k = min(k, B)
    idx_list = torch.linspace(0, B - 1, steps=k).long().tolist()

    for i in idx_list:
        xi   = _downsample_chw(x[i],   downsample)  # (C,h,w)
        reci = _downsample_chw(rec[i], downsample)
        # 선택 채널 슬라이스 → (h,w)
        xi_ch   = xi[ch_idx]
        reci_ch = reci[ch_idx]
        # uint8 그레이
        a = _to_uint8_gray2d(xi_ch)
        b = _to_uint8_gray2d(reci_ch)
        comp = _side_by_side_gray(a, b)  # (h,2w,3)
        wandb.log({tag: wandb.Image(comp, caption=f"channel={ch_name} ({ch_idx}) | left:orig, right:recon"),
                   "epoch": int(epoch)})

def sample_rows_for_table(arr_list, max_rows: int):
    """
    arr_list 길이가 max_rows보다 크면 균일 샘플. 작은 경우 그대로 반환.
    """
    n = len(arr_list)
    if n <= max_rows:
        return arr_list
    step = n / max_rows
    idx = [int(i*step) for i in range(max_rows)]
    return [arr_list[i] for i in idx]

def log_embeddings_table_wandb(emb: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor,
                               split: str, epoch: int, max_rows: int = 200, tag_prefix: str = "emb"):
    """
    emb: (B,D) on GPU
    y_true, y_pred: (B,)
    → CPU로 옮겨 소량만 테이블에 기록. 거대한 테이블/누적 금지.
    """
    if emb is None or emb.numel() == 0:
        return
    with torch.no_grad():
        E = emb.detach().float().cpu().numpy().tolist()
        Y = y_true.detach().cpu().tolist()
        P = y_pred.detach().cpu().tolist()

    # 행 샘플링
    rows = list(zip(E, P, Y))
    rows = sample_rows_for_table(rows, max_rows)

    table = wandb.Table(columns=["embedding", "pred", "label"], allow_mixed_types=True)
    for e, p, y in rows:
        table.add_data(e, int(p), int(y))
    wandb.log({f"{tag_prefix}/{split}": table, "epoch": int(epoch)})

def ddp_gather_small_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    각 rank의 (K, D) 텐서를 all_gather로 모아 rank0에서 cat.
    K가 작을 때만 사용(임베딩 일부 샘플). 크면 비용 ↑
    """
    if not (dist.is_available() and dist.is_initialized()):
        return t
    world_size = dist.get_world_size()
    tensors_gather = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(tensors_gather, t)
    return torch.cat(tensors_gather, dim=0)

def vib_collate(batch):
    out = {}
    must_stack = {"x_stft", "ref_stft", "x_cls"}  # 모델이 바로 쓰는 키만 stack
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        if k in must_stack:
            # tensor로 변환 보장
            vals = [torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for v in vals]
            out[k] = torch.stack(vals, dim=0)
        else:
            # 길이 제각각일 수 있으니 리스트로 유지
            out[k] = vals
    return out


def load_checkpoint_if_any(model, optimizer, path, device, strict=False, resume_optim=True):
    """
    반환: (start_epoch, best_val_acc)
    - ckpt의 'epoch'가 t라면 다음 에폭 t+1부터 시작
    - state_dict 키: {'model_state_dict', 'optimizer_state_dict', 'epoch', 'val_acc'}
    """
    if not path or not os.path.isfile(path):
        return 0, 0.0

    ckpt = torch.load(path, map_location=device)

    # 모델 로드 (DDP 안전하게)
    net = model.module if isinstance(model, DDP) else model
    state_dict = ckpt.get('model_state_dict', ckpt)  # 호환용
    missing, unexpected = net.load_state_dict(state_dict, strict=strict)
    if not strict and (missing or unexpected):
        print(f"[resume] non-strict load: missing={len(missing)}, unexpected={len(unexpected)}")

    # 옵티마이저 로드
    if resume_optim and ('optimizer_state_dict' in ckpt):
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception as e:
            print(f"[resume] optimizer state load skipped ({e})")

    start_epoch = int(ckpt.get('epoch', -1)) + 1
    best_val_acc = float(ckpt.get('val_acc', 0.0))
    print(f"[resume] loaded '{path}' (resume from epoch {start_epoch}, best_val_acc={best_val_acc:.4f})")
    return start_epoch, best_val_acc

def train_model(alpha, model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, rank, config, start_epoch=0, init_best_val_acc=0.0, dataset_select=None):
    net = unwrap_ddp(model)
    is_main = (rank == 0)
    best_val_acc = float(init_best_val_acc)
    warmup_epochs = int(getattr(config, "warmup_epochs", 0))
    
    # AMP
    use_bf16 = torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    # 로깅 주기/용량 제한
    IMG_LOG_EVERY   = getattr(config, "img_log_every", 100)
    IMG_LOG_K       = getattr(config, "img_log_k", 4)
    IMG_DOWNSAMPLE  = getattr(config, "img_downsample", 2)
    LOG_IMAGES      = bool(getattr(config, "log_images", False))
    LOG_CHANNEL    = int(getattr(config, "log_channel", 0))
    
    LOG_EMB         = bool(getattr(config, "log_embeddings", False))
    EMB_LOG_EVERY   = int(getattr(config, "emb_log_every", 100))
    EMB_PER_RANK    = int(getattr(config, "emb_per_rank", 32))
    TABLE_MAX_ROWS  = int(getattr(config, "table_max_rows", 200))

    for epoch in range(start_epoch, num_epochs):
        effective_alpha = 0.0 if epoch < warmup_epochs else float(alpha)

        # sampler epoch advance
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(val_loader.sampler, "set_epoch"):
            val_loader.sampler.set_epoch(epoch)

        # ---------------- Train ----------------
        # Training phase
        net.train()
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) if is_main else train_loader

        loss_sum_local = 0.0
        correct_local = 0.0
        total_local = 0.0
        train_preds_local = []
        train_labels_local = []

        # 마지막 배치의 diff/logits/y (임베딩 로깅용)만 잡아둠. 누적 금지.
        last_train_diff = None
        last_train_logits = None
        last_train_y = None
        reconstruction_image_to_log = None
        
        # dataset에서 getitem에 인자 true로 설정해놓으면 아래와 같이 info도 같이 줌
        for i, batch in enumerate(train_iter):
            x = batch['x_stft'].to(device, non_blocking=True)
            y = batch['x_cls'].to(device, non_blocking=True)
            ref_x = batch['ref_stft'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                rec, _, masked_idx = net.forward_mae(img=x)
                # loss_mae = net.calculate_mae_loss(rec, x, masked_idx)
                loss_mae = nn.MSELoss()(rec, x)

                logits, diff = net.forward_classify(current_img=x, normal_img=ref_x)
                loss_cls = criterion(logits, y)

                loss = effective_alpha * loss_cls + (1.0 - effective_alpha) * loss_mae

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            loss_sum_local += loss.item() * bs
            pred = logits.argmax(dim=1)
            correct_local += (pred == y).sum().item()
            total_local += bs

            train_preds_local.append(pred.detach())
            train_labels_local.append(y.detach())
            
            # 이미지 로깅: 첫 배치에서만, 큰 간격으로
            if LOG_IMAGES and is_main and (epoch % IMG_LOG_EVERY == 0):
                # 첫 번째 샘플에 대해 시각화 (inputs[0], rec_img[0] )
                fig = create_reconstruction_figure(
                    orig_tensor=x[0],
                    rec_tensor=rec[0],
                    mode=config.stft_mode,  # config에서 파라미터 가져오기
                    max_order=config.max_order,
                    window_sec=config.window_sec
                )
                # Figure를 wandb.Image 객체로 변환
                reconstruction_image_to_log = wandb.Image(fig)
                plt.close(fig)
            

            # 임베딩 테이블은 에폭 말에 한 번만 기록할 것이므로 마지막 배치 참조만 유지
            last_train_diff = diff.detach()
            last_train_logits = logits.detach()
            last_train_y = y.detach()

            if is_main and isinstance(train_iter, tqdm):
                train_iter.set_postfix({
                    "loss": f"{loss_sum_local/max(total_local,1):.4f}",
                    "acc":  f"{100.0*correct_local/max(total_local,1):.2f}%"
                })
        
        # reduce train metrics
        device0 = device

        t_train = torch.tensor([loss_sum_local, correct_local, total_local], dtype=torch.float64, device=device0)
        _ddp_sum_tensor(t_train)
        train_loss = (t_train[0] / t_train[2]).item() if t_train[2] > 0 else 0.0
        train_acc = (t_train[1] / t_train[2] * 100.0).item() if t_train[2] > 0 else 0.0

        # sklearn metrics (rank별 로컬; 보고용)
        train_preds_local = torch.cat(train_preds_local, dim=0).detach().cpu().numpy()
        train_labels_local = torch.cat(train_labels_local, dim=0).detach().cpu().numpy()
        train_metrics = {
            "precision_weighted": precision_score(train_labels_local, train_preds_local, average="weighted", zero_division=0),
            "recall_weighted":    recall_score(train_labels_local, train_preds_local, average="weighted", zero_division=0),
            "f1_weighted":        f1_score(train_labels_local, train_preds_local, average="weighted", zero_division=0),
            "precision_micro":  precision_score(train_labels_local, train_preds_local, average="micro", zero_division=0),
            "recall_micro":     recall_score(train_labels_local, train_preds_local, average="micro", zero_division=0),
            "f1_micro":         f1_score(train_labels_local, train_preds_local, average="micro", zero_division=0),
            "precision_macro":  precision_score(train_labels_local, train_preds_local, average="macro", zero_division=0),
            "recall_macro":     recall_score(train_labels_local, train_preds_local, average="macro", zero_division=0),
            "f1_macro":         f1_score(train_labels_local, train_preds_local, average="macro", zero_division=0),
        }
        
        # ---------------- Val ----------------
        # Validation phase
        net.eval()
        val_loss_sum_local = 0.0
        val_correct_local  = 0.0
        val_total_local    = 0.0
        val_preds_local = []
        val_labels_local = []

        last_val_diff = None
        last_val_logits = None
        last_val_y = None
        val_reconstruction_image_to_log = None

        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) if is_main else val_loader

        with torch.no_grad():
            for i, batch  in enumerate(val_iter):
                x = batch['x_stft'].to(device, non_blocking=True)
                y = batch['x_cls'].to(device, non_blocking=True)
                ref_x = batch['ref_stft'].to(device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                    rec, _, masked_idx = net.forward_mae(img=x)
                    # b_loss_mae = net.calculate_mae_loss(rec, x, masked_idx)
                    b_loss_mae = nn.MSELoss()(rec, x)

                    logits, diff = net.forward_classify(current_img=x, normal_img=ref_x)
                    b_loss_cls = criterion(logits, y)

                    b_loss = effective_alpha * b_loss_cls + (1.0 - effective_alpha) * b_loss_mae

                bs = y.size(0)
                val_loss_sum_local += b_loss.item() * bs
                pred = logits.argmax(dim=1)
                val_correct_local += (pred == y).sum().item()
                val_total_local += bs

                val_preds_local.append(pred.detach())
                val_labels_local.append(y.detach())

                # 이미지 로깅: 첫 배치에서만, 큰 간격으로
                if LOG_IMAGES and is_main and (epoch % IMG_LOG_EVERY == 0):
                    # 첫 번째 샘플에 대해 시각화 (inputs[0], rec_img[0] )
                    fig = create_reconstruction_figure(
                        orig_tensor=x[0],
                        rec_tensor=rec[0],
                        mode=config.stft_mode,  # config에서 파라미터 가져오기
                        max_order=config.max_order,
                        window_sec=config.window_sec
                    )
                    # Figure를 wandb.Image 객체로 변환
                    val_reconstruction_image_to_log = wandb.Image(fig)
                    plt.close(fig)

                # 임베딩 로깅용 참조 저장(마지막 배치)
                last_val_diff = diff.detach()
                last_val_logits = logits.detach()
                last_val_y = y.detach()
        
        t_val = torch.tensor([val_loss_sum_local, val_correct_local, val_total_local], dtype=torch.float64, device=device0)
        _ddp_sum_tensor(t_val)
        val_loss = (t_val[0] / t_val[2]).item() if t_val[2] > 0 else 0.0
        val_acc  = (t_val[1] / t_val[2] * 100.0).item() if t_val[2] > 0 else 0.0

        val_preds_local = torch.cat(val_preds_local, dim=0).detach().cpu().numpy()
        val_labels_local = torch.cat(val_labels_local, dim=0).detach().cpu().numpy()
        val_metrics = {
            "precision_weighted": precision_score(val_labels_local, val_preds_local, average="weighted", zero_division=0),
            "recall_weighted":    recall_score(val_labels_local, val_preds_local, average="weighted", zero_division=0),
            "f1_weighted":        f1_score(val_labels_local, val_preds_local, average="weighted", zero_division=0),
            "precision_micro":  precision_score(val_labels_local, val_preds_local, average="micro", zero_division=0),
            "recall_micro":     recall_score(val_labels_local, val_preds_local, average="micro", zero_division=0),
            "f1_micro":         f1_score(val_labels_local, val_preds_local, average="micro", zero_division=0),
            "precision_macro":  precision_score(val_labels_local, val_preds_local, average="macro", zero_division=0),
            "recall_macro":     recall_score(val_labels_local, val_preds_local, average="macro", zero_division=0),
            "f1_macro":         f1_score(val_labels_local, val_preds_local, average="macro", zero_division=0),
        }

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        # Log metrics to wandb
        # ---------------- Light Logging (scalars) ----------------
        if is_main:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "train/precision_micro": train_metrics["precision_micro"],
                "train/recall_micro":    train_metrics["recall_micro"],
                "train/f1_micro":        train_metrics["f1_micro"],
                "train/precision_macro": train_metrics["precision_macro"],
                "train/recall_macro":    train_metrics["recall_macro"],
                "train/f1_macro":        train_metrics["f1_macro"],
                "train/precision_weighted": train_metrics["precision_weighted"],
                "train/recall_weighted":    train_metrics["recall_weighted"],
                "train/f1_weighted":        train_metrics["f1_weighted"],
                "val/precision_micro":  val_metrics["precision_micro"],
                "val/recall_micro":     val_metrics["recall_micro"],
                "val/f1_micro":         val_metrics["f1_micro"],
                "val/precision_macro":  val_metrics["precision_macro"],
                "val/recall_macro":     val_metrics["recall_macro"],
                "val/f1_macro":         val_metrics["f1_macro"],
                "val/precision_weighted": val_metrics["precision_weighted"],
                "val/recall_weighted":    val_metrics["recall_weighted"],
                "val/f1_weighted":        val_metrics["f1_weighted"],
            }
            if reconstruction_image_to_log:
                log_dict['train/reconstruction_comparison'] = reconstruction_image_to_log
            if val_reconstruction_image_to_log:
                log_dict['val/reconstruction_comparison'] = val_reconstruction_image_to_log
            if wandb.run is not None:
                wandb.log(log_dict)

            # 베스트 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                if dataset_select is None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                    }, os.path.join('checkpoints', 'best_model.pth'))
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                    }, os.path.join(f'checkpoints', f'best_model_{dataset_select}.pth'))
            print(f"[{epoch+1}/{num_epochs}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")
        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        if dataset_select is None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join('checkpoints', 'last_model.pth'))
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(f'checkpoints', f'last_model_{dataset_select}.pth'))

        
        # ---------------- Light Logging (embeddings table) ----------------
        if LOG_EMB and wandb.run is not None and (epoch % EMB_LOG_EVERY == 0):
            # Train
            if (last_train_diff is not None and last_train_logits is not None and last_train_y is not None
                and last_train_diff.ndim == 2 and last_train_logits.ndim == 2 and last_train_y.ndim == 1
                and last_train_diff.size(0) > 0):
                K = min(EMB_PER_RANK, last_train_diff.size(0))
                idx = torch.randint(0, last_train_diff.size(0), (K,), device=last_train_diff.device)
                emb_small  = last_train_diff[idx]
                y_small    = last_train_y[idx]
                pred_small = last_train_logits.argmax(dim=1)[idx]

                emb_small  = ddp_gather_small_tensor(emb_small)
                y_small    = ddp_gather_small_tensor(y_small)
                pred_small = ddp_gather_small_tensor(pred_small)

                if is_main:
                    log_embeddings_table_wandb(
                        emb_small, y_small, pred_small,
                        split="train_diff", epoch=epoch,
                        max_rows=TABLE_MAX_ROWS, tag_prefix="emb"
                    )

            # Val
            if (last_val_diff is not None and last_val_logits is not None and last_val_y is not None
                and last_val_diff.ndim == 2 and last_val_logits.ndim == 2 and last_val_y.ndim == 1
                and last_val_diff.size(0) > 0):
                K = min(EMB_PER_RANK, last_val_diff.size(0))
                idx = torch.randint(0, last_val_diff.size(0), (K,), device=last_val_diff.device)
                emb_small  = last_val_diff[idx]
                y_small    = last_val_y[idx]
                pred_small = last_val_logits.argmax(dim=1)[idx]

                emb_small  = ddp_gather_small_tensor(emb_small)
                y_small    = ddp_gather_small_tensor(y_small)
                pred_small = ddp_gather_small_tensor(pred_small)

                if is_main:
                    log_embeddings_table_wandb(
                        emb_small, y_small, pred_small,
                        split="val_diff", epoch=epoch,
                        max_rows=TABLE_MAX_ROWS, tag_prefix="emb"
                    )



def setup(rank, world_size, args):
    # 환경 변수 설정
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    
    # 분산 처리 초기화
    torch.distributed.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    # CUDA 설정
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_with_config(rank, world_size, args):
    setup(rank, world_size, args)

    # wandb 초기화 (메인 프로세스에서만)
    if rank == 0:
        wandb.init(project=args.project_name, name=args.run_name, config=vars(args))
    
    # --- (B) rank0의 config(dict) -> 모든 rank로 브로드캐스트 ---
    if rank == 0 and wandb.run is not None:
        # sweep일 경우 wandb.config가 최종값을 담고 있으므로 그것을 기준으로
        cfg_dict = dict(wandb.config)
    else:
        # 비-rank0는 임시로 args를 dict로
        cfg_dict = vars(args).copy()
    
    obj_list = [cfg_dict]
    dist.broadcast_object_list(obj_list, src=0)   # 모든 프로세스에 동일한 설정 전달
    cfg_dict = obj_list[0]                        # 동기화된 최종 설정

    # 모든 rank에서 공통으로 사용할 네임스페이스 구성
    config = types.SimpleNamespace(**cfg_dict)
    
    # 1) stft_pair가 있으면 "NxM" 형식으로 파싱
    if hasattr(config, "stft_pair") and config.stft_pair is not None:
        n_str, h_str = str(config.stft_pair).lower().split("x")
        config.stft_nperseg = int(n_str)
        config.stft_hop = int(h_str)
        if rank == 0 and wandb.run is not None:
            wandb.config.update({
                "stft_nperseg": config.stft_nperseg,
                "stft_hop": config.stft_hop
            }, allow_val_change=True)
    else:
        config.stft_nperseg = int(config.stft_nperseg)
        config.stft_hop = int(config.stft_hop)

    setattr(config, "stft_nperseg", config.stft_nperseg)
    setattr(config, "stft_hop", config.stft_hop)

    # W&B 설정 업데이트는 rank0에서만
    if rank == 0 and wandb.run is not None:
        wandb.config.update({
            "stft_nperseg": config.stft_nperseg,
            "stft_hop": config.stft_hop
        }, allow_val_change=True)

    torch.cuda.set_device(rank)  # 각 프로세스의 GPU 설정
    device = torch.device(f"cuda:{rank}")
    
    # 데이터 준비
    data_root = os.path.join(os.getcwd(), config.data_root)
    if config.cached:
        # 0: all, 1: except vat, 2: except vbl, 3: except mfd, 4: except dxai
        selected = config.dataset_select
        if selected == 0:
            train_data_root = os.path.join(data_root, "llm_vib_trainset_4dataset.pt")
            train_dataset = CachedDataset(data_root=train_data_root)
            val_data_root = os.path.join(data_root, "llm_vib_validset_4dataset_only_dxai.pt")
            val_dataset = CachedDataset(data_root=val_data_root)
        elif selected == 1:
            train_data_root = os.path.join(data_root, "llm_vib_trainset_3dataset_except_vat.pt")
            train_dataset = CachedDataset(data_root=train_data_root)
            val_data_root = os.path.join(data_root, "llm_vib_validset_only_vat.pt")
            val_dataset = CachedDataset(data_root=val_data_root)
        elif selected == 2:
            train_data_root = os.path.join(data_root, "llm_vib_trainset_3dataset_except_vbl.pt")
            train_dataset = CachedDataset(data_root=train_data_root)
            val_data_root = os.path.join(data_root, "llm_vib_validset_only_vbl.pt")
            val_dataset = CachedDataset(data_root=val_data_root)
        elif selected == 3:
            train_data_root = os.path.join(data_root, "llm_vib_trainset_3dataset_except_mfd.pt")
            train_dataset = CachedDataset(data_root=train_data_root)
            val_data_root = os.path.join(data_root, "llm_vib_validset_only_mfd.pt")
            val_dataset = CachedDataset(data_root=val_data_root)
        elif selected == 4:
            train_data_root = os.path.join(data_root, "llm_vib_trainset_3dataset_except_dxai.pt")
            train_dataset = CachedDataset(data_root=train_data_root)
            val_data_root = os.path.join(data_root, "llm_vib_validset_only_dxai.pt")
            val_dataset = CachedDataset(data_root=val_data_root)
        else:
            raise ValueError("Unexpected selected caching dataset index.")
    else:
        # 이미지 변환기 설정 (pretrained 모델 사용 시 224x224로 강제)
        output_size = 224 if config.pretrained else config.image_size
        if rank == 0 and config.pretrained and config.image_size != 224:
            print(f"Warning: Pretrained model requires 224x224 input. "
                f"Automatically adjusting output size from {config.image_size} to 224.")
        
        signal_imger = OrderInvariantSignalImager(
            mode=config.stft_mode,
            log1p=True,
            normalize="per_channel", 
            eps=1e-8,
            out_dtype=torch.float32,
            max_order=config.max_order,
            H_out=output_size,
            W_out=output_size,
            stft_nperseg=config.stft_nperseg,
            stft_hop=config.stft_hop,
            stft_window="hann",
            stft_center=True,
            stft_power=config.stft_power,
        )
        
        # 학습용 데이터셋 생성
        train_dataset = VibrationDataset(
            data_root=data_root,
            using_dataset = ['vat', 'vbl', 'mfd', 'dxai'],
            window_sec=config.window_sec,
            stride_sec=config.stride_sec,
            transform=signal_imger
        )
        
        # 검증용 데이터셋 생성
        val_dataset = VibrationDataset(
            data_root=data_root,
            using_dataset = ['dxai'],
            window_sec=config.window_sec,
            stride_sec=config.stride_sec,
            transform=signal_imger
        )
    
    
    if len(val_dataset) == 0 and rank == 0:
        raise RuntimeError(
            "Validation dataset is empty. "
            "Check cached file: data/processed/llm_vib_validset_only_vbl.pt "
            "or adjust --cached/--dataset_select."
        )
    if rank == 0:
        print(f"#train_samples={len(train_dataset)}  #val_samples={len(val_dataset)}")
        # DataLoader가 만들 배치 수(대략치)
        try:
            tmp_train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
            tmp_val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
            print(f"#train_batches≈{len(tmp_train_loader)}  #val_batches≈{len(tmp_val_loader)}")
        except Exception as e:
            print("DataLoader dry-run failed:", e)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    # 워커 수는 보수적으로
    workers = max(2, min(4, (os.cpu_count() or 8) // max(1, 2*world_size)))
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, 
                            batch_size=config.batch_size, 
                            num_workers=2,
                            sampler=train_sampler,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=vib_collate)
    val_loader = DataLoader(val_dataset, 
                          batch_size=config.batch_size, 
                          num_workers=2,
                          sampler=val_sampler,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=vib_collate)
    
    # 모델 생성 및 DDP 설정
    model = VisionTransformerAE(
        num_layers = 12,
        num_heads = 12,
        hidden_dim = 768,
        mlp_dim = 3072,
        dropout = 0.0,
        attention_dropout  = 0.0,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        image_size = 224,
        image_channel = 4,
        patch_size = 16,
        masking_ratio=0.75,
        num_classes=config.num_classes,
    ).to(device)

    if rank == 0:
        print(f"Creating model with {'pretrained' if config.pretrained else 'random'} initialization")
    
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)
    
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    
    # --- 체크포인트 로드 ---
    start_epoch = 0
    init_best_val_acc = 0.0
    if getattr(config, "resume_path", ""):
        start_epoch, init_best_val_acc = load_checkpoint_if_any(
            model=model,
            optimizer=optimizer,
            path=config.resume_path,
            device=device,
            strict=bool(getattr(config, "resume_strict", False)),
            resume_optim=not bool(getattr(config, "no_resume_optim", False)),
        )
    # 명시적 override가 있으면 적용
    if int(getattr(config, "resume_epoch_override", -1)) >= 0:
        start_epoch = int(config.resume_epoch_override)
        print(f"[resume] epoch override -> start_epoch={start_epoch}")
    
    # 학습 실행
    train_model(
        alpha=config.alpha,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config.epochs,
        device=device,
        rank=rank,
        config=config,
        dataset_select=args.dataset_select,
        start_epoch=start_epoch,
        init_best_val_acc=init_best_val_acc
    )
    
    cleanup()  # process group destroy
    if (rank == 0) and (wandb.run is not None):
        wandb.finish()

def run_training(args):
    # 사용 가능한 GPU 수 확인
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs available")
    if world_size < 2:
        print("Warning: Less than 2 GPUs available. Using", world_size, "GPU(s)")
    
    # 메인 프로세스에서 필요한 디렉토리 생성
    os.makedirs('checkpoints', exist_ok=True)
    
    try:
        # mp.spawn(
        #     train_with_config,
        #     args=(world_size, args),
        #     nprocs=world_size,
        #     join=True
        # )
        train_with_config(rank=0, world_size=world_size, args=args)
    except Exception as e:
        print(f"Error during training: {e}")
        raise

def main():
    args = parse_args()
    
    # CUDA 사용 가능 여부 확인
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This code requires GPU support.")
    
    # sweep 설정이 있는 경우
    if args.sweep_config:
        import yaml
        with open(args.sweep_config, 'r') as f:
            sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep_configuration, project=args.project_name)
        wandb.agent(sweep_id, function=lambda: run_training(args), count=50)
    else:
        # 일반 학습
        run_training(args)

def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT for Vibration Diagnosis')
    parser.add_argument('--data_root', type=str, default='data/processed',
                        help='Path to the processed data directory')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--warmup_epochs', type=int, default=1500,
                        help='epochs for reconstruction-only warm-up (classification weight=0)')

    parser.add_argument('--alpha', type=float, default=0.5)

    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--window_sec', type=float, default=5.0)
    parser.add_argument('--stride_sec', type=float, default=3.0)
    parser.add_argument('--max_order', type=float, default=20.0)
    parser.add_argument('--stft_mode', type=str, default='stft+cross',
                        choices=['stft', 'stft+cross', 'stft_complex'])
    parser.add_argument('--stft_nperseg', type=int, default=1024,
                        help='Length of each STFT segment')
    parser.add_argument('--stft_hop', type=int, default=256,
                        help='Number of points between successive STFT segments')
    parser.add_argument('--stft_power', type=float, default=1.0,
                        help='Power of magnitude (1.0 for magnitude, 2.0 for power spectrum)')
    
    parser.add_argument('--project_name', type=str, default='vibration-diagnosis-final_1105night_2050')
    parser.add_argument('--run_name', type=str, default='All_dataset_non_masking')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Use ImageNet pretrained weights for ViT')
    parser.add_argument('--port', type=int, default=12355,
                        help='Port for distributed training')
    
    parser.add_argument('--log_channel', type=int, default=0,
                    help='로깅할 채널 인덱스 (예: stft+cross 기준 0:|X|^p, 1:|Y|^p, 2:|X·Y*|, 3:cosΔφ). '
                         '-1이면 에폭마다 채널을 순환.')
    parser.add_argument('--log_images', action='store_true',
                    help='이미지 로깅 활성화 (rank0 전용)')
    parser.add_argument('--img_log_every', type=int, default=100,
                        help='이미지 로깅 에폭 주기')
    parser.add_argument('--img_log_k', type=int, default=4,
                        help='배치에서 로깅할 최대 샘플 수')
    parser.add_argument('--img_downsample', type=int, default=2,
                        help='로깅 시 CHW stride 다운샘플 배수')

    parser.add_argument('--log_embeddings', action='store_true',
                        help='임베딩 테이블 로깅 활성화 (rank0 전용)')
    parser.add_argument('--emb_log_every', type=int, default=100,
                        help='임베딩 테이블 로깅 에폭 주기')
    parser.add_argument('--emb_per_rank', type=int, default=32,
                        help='각 rank에서 샘플링할 임베딩 개수 (작게 유지)')
    parser.add_argument('--table_max_rows', type=int, default=200,
                        help='W&B 테이블로 올릴 최대 행 수 (합쳐진 후)')


    parser.add_argument('--cached', action="store_true", help="Data loading True/False")
    parser.add_argument('--dataset_select', type=int, default=0, help="0: all, 1: except vat, 2: except vbl, 3: except mfd, 4: except dxai")

    parser.add_argument('--resume_path', type=str, default='',
                        help='재개할 체크포인트 경로(.pth). 비우면 새로 시작')
    parser.add_argument('--resume_strict', action='store_true',
                        help='모델 state_dict strict 로드 (레이어 변경 시 비권장)')
    parser.add_argument('--no_resume_optim', action='store_true',
                        help='설정 시 optimizer 상태는 복구하지 않음')
    parser.add_argument('--resume_epoch_override', type=int, default=-1,
                        help='>=0이면 ckpt의 epoch 대신 이 값을 시작 에폭으로 사용')


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # CUDA 사용 가능 여부 확인
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This code requires GPU support.")
    
    run_training(args)