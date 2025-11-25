import pandas as pd
from tqdm import tqdm
import ast
import numpy as np
from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window

import argparse

"""
Dataset Class (x_vib, x_stft, x_info, x_cls, ref_vib, ref_stft, ref_info, ref_cls 를 key로 가지는 dict를 반환)
"""
class VibrationDataset(Dataset):
    def __init__(
        self, data_root, window_sec, stride_sec,
        using_dataset = ['dxai', 'iis', 'vat', 'vbl', 'mfd'],
        drop_last=True, 
        dtype=torch.float32, 
        transform=None,
        channel_order=("x","y"),
        test_mode = False,
        include_ref= True
    ):
        self.data_root = data_root
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.using_dataset = using_dataset
        self.drop_last = drop_last
        self.dtype = dtype
        self.transform = transform
        self.channel_order = channel_order
        self.test_mode  = test_mode
        self.include_ref = include_ref  
        
        self.load_dataset()
    
    def load_dataset(self):
        meta_csv = os.path.join(self.data_root, 'meta.csv')
        meta_pd = pd.read_csv(meta_csv)
        meta_pd['sensor_position'] = meta_pd['sensor_position'].apply(ast.literal_eval)
        meta_pd = meta_pd[5 <= meta_pd['data_sec']]
        meta_pd = meta_pd[meta_pd['dataset'].isin(self.using_dataset)]
        meta_pd['class_name'].unique()
        self.test_mode = self.test_mode
        self.meta_df = meta_pd.reset_index(drop=True).copy()
        # 클래스 통합 매핑
        self.class_list = ['normal', 'unbalance', 'looseness', 'misalignment', 'bearing']
        merge_map = {
            # 정상
            'normal': 'normal',

            # 언밸런스
            'unbalance': 'unbalance',
            'unbalalnce': 'unbalance',  # 오타 수정
            'imbalance': 'unbalance',

            # 루즈니스
            'looseness': 'looseness',

            # 미스얼라인먼트
            'misalignment': 'misalignment',
            'horizontal-misalignment': 'misalignment',
            'vertical-misalignment': 'misalignment',

            # 베어링
            'bpfo': 'bearing',
            'bpfi': 'bearing',
            'bearing': 'bearing',
            'overhang_cage_fault': 'bearing',
            'overhang_ball_fault': 'bearing',
            'overhang_outer_race': 'bearing',
            'underhang_cage_fault': 'bearing',
            'underhang_ball_fault': 'bearing',
            'underhang_outer_race': 'bearing',
        }

        # 새로운 컬럼 생성 (통합 클래스명)
        self.meta_df['merged_class'] = self.meta_df['class_name'].map(merge_map)

        if isinstance(self.meta_df.loc[0, "sensor_position"], str):
            self.meta_df["sensor_position"] = self.meta_df["sensor_position"].apply(ast.literal_eval)

        self.index_map = []      # (row_idx, start)
        self._row_meta = {}      # per-row meta
        self._file_cache = {}    # cache_mode in ('file','windows'): row_idx -> np.ndarray or dict
        self._win_cache  = {}    # cache_mode == 'windows': row_idx -> (W, 2, win_n)


        # 인덱스/캐시 준비
        print('caching dataset ... ')
        for row_idx, row in tqdm(self.meta_df.iterrows()):
            file_path = os.path.join(self.data_root, row["file_name"])
            sr = float(row["sampling_rate"])
            sensor_pos = row["sensor_position"]
            dataset = row["dataset"]

            if dataset == "iis":
                x_idx = sensor_pos.index("disk_x"); y_idx = sensor_pos.index("disk_y")
            else:
                x_idx = sensor_pos.index("motor_x"); y_idx = sensor_pos.index("motor_y")

            arr = np.load(file_path, mmap_mode=(None))
            n_samples = arr.shape[1]
            win_n   = int(round(self.window_sec * sr))
            stride_n= int(round(self.stride_sec * sr))
            win_n = min(win_n, n_samples)
            starts  = starts_for(n_samples, win_n, stride_n, self.drop_last)

            # print(f"[vbl] row {row_idx}, n_samples={n_samples}, win_n={win_n}, len(starts)={len(starts)}")
            self._row_meta[row_idx] = {
                "file_path": file_path, "sr": sr,
                "x_idx": x_idx, "y_idx": y_idx, "win_n": win_n
            }
            for s in starts:
                self.index_map.append((row_idx, s))

            if row_idx not in self._file_cache:
                self._file_cache[row_idx] = np.load(file_path, mmap_mode='r')  # (S, N), ndarray in RAM
            

        # ---- Build normal-reference window pool: key = (dataset, load_condition) ----
        self._ref_pool = {}  # (dataset, load_condition) -> list of (row_idx, start)
        for n_row_idx, n_row in self.meta_df.iterrows():
            if self.meta_df.loc[n_row_idx, 'merged_class'] != 'normal':
                continue
                
            file_path_n = os.path.join(self.data_root, n_row["file_name"])
            arr_n = np.load(file_path_n)
            n_samples_n = arr_n.shape[1]
            sr_n = float(n_row["sampling_rate"])
            win_n_n = int(round(self.window_sec * sr_n))
            win_n_n = min(win_n_n, n_samples_n)
            stride_n_n = int(round(self.stride_sec * sr_n))
            starts_n = starts_for(np.load(os.path.join(self.data_root, n_row["file_name"])).shape[1],
                                win_n_n, stride_n_n, self.drop_last)
            key = (n_row["dataset"], n_row["load_condition"])
            if key not in self._ref_pool:
                self._ref_pool[key] = []
            for s_n in starts_n:
                self._ref_pool[key].append((n_row_idx, s_n))
                
        # fallback pool by dataset only (if no matching load_condition exists)
        self._ref_pool_by_ds = {}
        for n_row_idx, n_row in self.meta_df.iterrows():
            if self.meta_df.loc[n_row_idx, 'merged_class'] != 'normal':
                continue
            file_path_n = os.path.join(self.data_root, n_row["file_name"])
            arr_n = np.load(file_path_n, mmap_mode='r')
            n_samples_n = arr_n.shape[1]
            sr_n = float(n_row["sampling_rate"])
            win_n_n = int(round(self.window_sec * sr_n))
            win_n_n = min(win_n_n, n_samples_n)
            stride_n_n = int(round(self.stride_sec * sr_n))
            starts_n = starts_for(np.load(os.path.join(self.data_root, n_row["file_name"])).shape[1],
                                win_n_n, stride_n_n, self.drop_last)
            key_ds = (n_row["dataset"],)
            if key_ds not in self._ref_pool_by_ds:
                self._ref_pool_by_ds[key_ds] = []
            for s_n in starts_n:
                self._ref_pool_by_ds[key_ds].append((n_row_idx, s_n))
                
        if self.test_mode:
            self.index_map = self.index_map[:30]

    def _extract_segment(self, row_idx, start):
        """Return (seg ndarray shape (2, win_n)) for the given row & start, respecting cache_mode."""
        #row
        _ = self.meta_df.iloc[row_idx]
        meta = self._row_meta[row_idx]
        #sr
        _, x_idx, y_idx, win_n = meta["sr"], meta["x_idx"], meta["y_idx"], meta["win_n"]

        base = self._file_cache[row_idx]
        x_seg = base[x_idx, start:start+win_n]
        y_seg = base[y_idx, start:start+win_n]

        seg = np.stack([x_seg, y_seg], axis=0) if self.channel_order==("x","y") \
              else np.stack([y_seg, x_seg], axis=0)
        return seg

    def _pick_ref_reference(self, dataset, load_condition):
        """Return (row_idx, start) of a normal sample matching (dataset, load_condition) if available,
        otherwise same dataset only, otherwise None."""
        key = (dataset, load_condition)
        pool = self._ref_pool.get(key)
        if pool:
            ridx, s = pool[np.random.randint(len(pool))]
            return ridx, s
        # fallback by dataset only
        key_ds = (dataset,)
        pool2 = self._ref_pool_by_ds.get(key_ds)
        if pool2:
            ridx, s = pool2[np.random.randint(len(pool2))]
            return ridx, s
        return None

    def __len__(self):
        if self.test_mode:
            return 30
        else:
            return len(self.index_map)

    def __getitem__(self, idx):
        row_idx, start = self.index_map[idx]
        row = self.meta_df.iloc[row_idx]
        meta = self._row_meta[row_idx]
        sr = meta["sr"]

        # ---- current segment ----
        x_vib = self._extract_segment(row_idx, start)
        x_stft = self.transform(x_vib, sr=sr, rpm=float(row["rpm"]))
        class_idx = self.class_list.index(row['merged_class'])
        x_cls = torch.tensor(class_idx ,dtype=torch.long)

        x_info = {
            "sampling_rate": float(sr),
            "rpm": float(row["rpm"]),
            "label_class": str(row["class_name"]),
            "merged_class": str(row["merged_class"]),
            "severity": str(row["severity"]),
            "load_condition": str(row["load_condition"]),
            "dataset": str(row["dataset"]),
            "file_name": str(row["file_name"]),
        }
        
        data_dict = {
                # 'x_vib' : x_vib, 
                'x_stft' : x_stft,
                'x_cls' : x_cls,
                'x_info' : x_info
            }

        # ---- normal reference (same dataset & load_condition) ----
        
        if self.include_ref:
            pick = self._pick_ref_reference(row["dataset"], row["load_condition"])
            if pick is not None:
                ref_row_idx, ref_start = pick
                ref_row = self.meta_df.iloc[ref_row_idx]
                ref_meta = self._row_meta[ref_row_idx]
                ref_sr = ref_meta["sr"]
                
                ref_vib = self._extract_segment(ref_row_idx, ref_start)
                ref_stft = self.transform(ref_vib, sr=ref_sr, rpm=float(ref_row["rpm"]))
                tensor_cls_norm = torch.tensor(self.class_list.index('normal'), dtype=torch.long)
                ref_info = {
                    "sampling_rate": float(ref_sr),
                    "rpm": float(ref_row["rpm"]),
                    "label_class": str(ref_row["class_name"]),
                    "merged_class": str(ref_row["merged_class"]),
                    "severity": str(ref_row["severity"]),
                    "load_condition": str(ref_row["load_condition"]),
                    "dataset": str(ref_row["dataset"]),
                    "file_name": str(ref_row["file_name"]),
                }
                ref_dict = {
                            # 'ref_vib' : ref_vib,
                            'ref_stft' : ref_stft, 
                            'ref_cls' : tensor_cls_norm, 
                            'ref_info' : ref_info
                            }
                
            data_dict.update(ref_dict)
            
        return data_dict
  
class CachedDataset(Dataset):
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.data_list = torch.load(data_root)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def starts_for(n, win_n, stride_n, drop_last=True):
    if win_n <= 0 or stride_n <= 0: raise ValueError("win_n/stride_n must be > 0")
    if n < win_n: return []
    starts = list(range(0, n - win_n + 1, stride_n))
    if not drop_last and (n - win_n) % stride_n != 0:
        last_start = n - win_n
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
    return starts

def rolling_windows_1d(a, win_n, stride_n):
    # 반환: (num_windows, win_n) - zero-copy view (as_strided)
    n = a.shape[-1]
    num = (n - win_n) // stride_n + 1
    if num <= 0: return np.empty((0, win_n), dtype=a.dtype)
    shape = (num, win_n)
    strides = (a.strides[-1] * stride_n, a.strides[-1])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

"""
STFT + Cross Mag, Cross Phase 를 수행하는 Transform
"""

class OrderInvariantSignalImager:
    """
    seg: np.ndarray (2, T)  # [x, y]
    __call__(seg, sr, rpm) -> torch.Tensor (C, H_out, W_out)

    mode ∈ {
      "stft", "stft+cross", "stft_complex",
      "cwt",  "cwt+cross",  "cwt_complex",
    }

    파이프라인:
      1) STFT 또는 CWT 계수 계산 (+ 선택적 cross/complex 채널 구성)
      2) 주파수 배열(f) 또는 CWT 주파수(f_cwt)를 회전주파수 f_rot = rpm/60 으로 나눠 order 축으로 정규화
      3) order <= max_order 로 마스킹
      4) (order, time) 2D 리샘플로 (H_out, W_out) 고정
      5) log/정규화 후 텐서 반환
    """
    def __init__(
        self,
        mode="stft+cross",
        # 공통
        log1p=True,
        normalize="per_channel",  # None | "per_channel" | "global"
        eps=1e-8,
        out_dtype=torch.float32,
        max_order=20.0,           # order 축 상한
        H_out=128,                # order-bin 수
        W_out=256,                # time-bin 수
        # STFT
        stft_nperseg=1024,
        stft_hop=256,
        stft_window="hann",
        stft_center=True,
        stft_power=1.0,           # 1: magnitude, 2: power
    ):
        assert mode in {
            "stft", "stft+cross", "stft_complex",
        }
        self.mode = mode
        self.log1p = log1p
        self.normalize = normalize
        self.eps = eps
        self.out_dtype = out_dtype
        self.max_order = float(max_order)
        self.H_out = int(H_out)
        self.W_out = int(W_out)

        self.stft_nperseg = stft_nperseg
        self.stft_hop = stft_hop
        self.stft_window = stft_window
        self.stft_center = stft_center
        self.stft_power = stft_power

    # ---------- 유틸 ----------
    def _apply_log_norm(self, x: np.ndarray) -> np.ndarray:
        # x: (C, H, W)
        if self.log1p:
            x = np.sign(x) * np.log1p(np.abs(x))

        if self.normalize == "per_channel":
            for c in range(x.shape[0]):
                if c == x.shape[0] - 1:  # 마지막 채널(phase)은 정규화 건너뜀
                    continue
                m = x[c].mean()
                s = x[c].std() + self.eps
                x[c] = (x[c] - m) / s

        elif self.normalize == "global":
            # 마지막 채널 제외하고 전체 mean/std 계산
            chans = x[:-1]
            m = chans.mean()
            s = chans.std() + self.eps
            x[:-1] = (chans - m) / s

        return x.astype(np.float32)

    def _resize_CHW(self, arr: np.ndarray, H_new: int, W_new: int) -> np.ndarray:
        """
        arr: (C, H, W) → (C, H_new, W_new)  (2-pass 1D linear interp: 시간축→order축)
        """
        C, H, W = arr.shape
        # 1) 시간축 보간
        if W != W_new:
            x_old = np.linspace(0.0, 1.0, W, endpoint=True)
            x_new = np.linspace(0.0, 1.0, W_new, endpoint=True)
            out_t = np.empty((C, H, W_new), dtype=arr.dtype)
            for c in range(C):
                for h in range(H):
                    out_t[c, h] = np.interp(x_new, x_old, arr[c, h])
        else:
            out_t = arr

        # 2) order축 보간
        if H != H_new:
            y_old = np.linspace(0.0, 1.0, H, endpoint=True)
            y_new = np.linspace(0.0, 1.0, H_new, endpoint=True)
            out = np.empty((C, H_new, W_new), dtype=arr.dtype)
            for c in range(C):
                for w in range(W_new):
                    out[c, :, w] = np.interp(y_new, y_old, out_t[c, :, w])
        else:
            out = out_t
        return out

    # ---------- STFT ----------
    def _stft_xy(self, x: np.ndarray, y: np.ndarray, sr: float):
        pad = (self.stft_nperseg // 2) if self.stft_center else 0
        if pad > 0:
            x = np.pad(x, (pad, pad), mode="reflect")
            y = np.pad(y, (pad, pad), mode="reflect")
        win = get_window(self.stft_window, self.stft_nperseg, fftbins=True)
        noverlap = self.stft_nperseg - self.stft_hop
        f, t, X = stft(x, fs=sr, window=win, nperseg=self.stft_nperseg,
                       noverlap=noverlap, nfft=None, padded=False, boundary=None)
        _, _, Y = stft(y, fs=sr, window=win, nperseg=self.stft_nperseg,
                       noverlap=noverlap, nfft=None, padded=False, boundary=None)
        return f, t, X, Y

    def _build_stft_maps(self, seg: np.ndarray, sr: float):
        x, y = seg[0], seg[1]
        if self.mode == "stft":
            f, t, X, Y = self._stft_xy(x, y, sr)
            X_mag, Y_mag = np.abs(X), np.abs(Y)
            if self.stft_power == 2.0:
                X_map, Y_map = X_mag**2, Y_mag**2
            elif self.stft_power == 1.0:
                X_map, Y_map = X_mag, Y_mag
            else:
                X_map, Y_map = X_mag**self.stft_power, Y_mag**self.stft_power
            chans = [X_map, Y_map]
        elif self.mode == "stft+cross":
            f, t, X, Y = self._stft_xy(x, y, sr)
            X_mag, Y_mag = np.abs(X), np.abs(Y)
            if self.stft_power == 2.0:
                X_map, Y_map = X_mag**2, Y_mag**2
            elif self.stft_power == 1.0:
                X_map, Y_map = X_mag, Y_mag
            else:
                X_map, Y_map = X_mag**self.stft_power, Y_mag**self.stft_power
            XY = X * np.conj(Y)
            cross_mag = np.abs(XY)
            phase_cos = np.cos(np.angle(XY))
            chans = [X_map, Y_map, cross_mag, phase_cos]
        else:  # "stft_complex"
            z = x + 1j*y
            pad = (self.stft_nperseg // 2) if self.stft_center else 0
            if pad > 0:
                z = np.pad(z, (pad, pad), mode="reflect")
            win = get_window(self.stft_window, self.stft_nperseg, fftbins=True)
            noverlap = self.stft_nperseg - self.stft_hop
            f, t, Z = stft(z, fs=sr, window=win, nperseg=self.stft_nperseg,
                           noverlap=noverlap, nfft=None, padded=False, boundary=None)
            amp = np.abs(Z)
            phase = np.angle(Z)
            phase_cos = np.cos(phase)
            phase_sin = np.sin(phase)
            phase_dev90 = phase - (np.pi/2)
            chans = [amp, phase_cos, phase_sin, phase_dev90]
        arr = np.stack(chans, axis=0)  # (C, F, T)
        return f, t, arr


    # ---------- 호출 ----------
    def __call__(self, seg: np.ndarray, sr: float, rpm: float) -> torch.Tensor:
        f_rot = float(rpm) / 60.0  # 회전주파수 [Hz]
        if f_rot <= 0:
            raise ValueError("rpm must be > 0 for order normalization.")
        # 1) 계수/채널 맵 + 주파수축 얻기
        if self.mode.startswith("stft"):
            f, t, arr = self._build_stft_maps(seg, sr)  # arr: (C, F, T)
            order = f / f_rot                            # (F,)
        else:
            print('err')
            exit()
        # 2) order 마스킹 (0 < order ≤ max_order)
        mask = (order > 0) & (order <= self.max_order)
        if not np.any(mask):
            # 모든 bin이 마스크되면 최소한 한 줄은 유지
            idx = np.argmax(order > 0)
            mask = np.zeros_like(order, dtype=bool)
            mask[idx] = True
        arr = arr[:, mask, :]           # (C, Hm, T)
        order = order[mask]             # (Hm,)
        # 3) order축을 0..max_order 균일 그리드로 리샘플
        #    현재 arr는 order가 불균일일 수 있으므로, 채널/시간별로 1D 보간
        Hm, T = arr.shape[1], arr.shape[2]
        order_target = np.linspace(0.0, self.max_order, self.H_out, endpoint=True)
        out_ord = np.empty((arr.shape[0], self.H_out, T), dtype=arr.dtype)
        # order가 단조 증가임을 가정 (STFT/CWT에서 주파수 증가는 보장)
        for c in range(arr.shape[0]):
            for t_idx in range(T):
                out_ord[c, :, t_idx] = np.interp(order_target, order, arr[c, :, t_idx],
                                                 left=arr[c, 0, t_idx], right=arr[c, -1, t_idx])
        # 4) 시간축도 고정 bins로 리사이즈
        out = self._resize_CHW(out_ord, self.H_out, self.W_out)  # (C, H_out, W_out)
        # 5) log/정규화 → 텐서
        out = self._apply_log_norm(out)
        return torch.as_tensor(out, dtype=self.out_dtype)    

def _channel_labels_for_mode(mode: str):
    mode = mode.lower()
    if mode == "stft":
        return ["|X|^p", "|Y|^p"]
    if mode == "stft+cross":
        return ["|X|^p", "|Y|^p", "|X·Y*|", "cos(Δφ)"]
    if mode == "stft_complex":
        return ["|Z|", "cos(∠Z)", "sin(∠Z)", "∠Z − 90°"]
    # fallback
    return [f"ch{i}" for i in range(16)]

def visualize_imaging_tensor(
    tensor_chw,              # torch.Tensor or np.ndarray, shape (C,H,W)
    mode: str,
    max_order: float,
    window_sec: float,
    save_path: str | None = None,
    figsize=(18, 18),
    percent_clip=(2, 98),    # magnitude 계열 대비 향상용 퍼센타일 클리핑
):
    """
    단일 샘플 시각화: 채널들을 가로/세로 그리드로 출력
    - y축: Order (0 .. max_order)
    - x축: Time (0 .. window_sec)
    """
    # to numpy (C,H,W)
    arr = tensor_chw.detach().cpu().numpy() if hasattr(tensor_chw, "detach") else np.asarray(tensor_chw)
    assert arr.ndim == 3, "Input must be (C,H,W)"
    C, H, W = arr.shape

    ch_labels = _channel_labels_for_mode(mode)
    if len(ch_labels) < C:
        ch_labels += [f"ch{i}" for i in range(len(ch_labels), C)]

    # subplot grid 크기 잡기
    ncols = min(4, C)
    nrows = int(np.ceil(C / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    extent = [0.0, float(window_sec), 0.0, float(max_order)]  # x: time(sec), y: order
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if idx < C:
                img = arr[idx]

                # 채널 유형별 vmin/vmax 설정
                label = ch_labels[idx].lower()
                if any(k in label for k in ["cos(", "sin("]):
                    vmin, vmax = -1.0, 1.0
                elif "− 90°" in ch_labels[idx] or "- 90°" in ch_labels[idx] or "90" in ch_labels[idx]:
                    # 위상 편차 채널: 대략 -π..π
                    vmin, vmax = -np.pi, np.pi
                else:
                    # magnitude 계열: 퍼센타일 클리핑
                    lo, hi = np.percentile(img, percent_clip)
                    vmin, vmax = float(lo), float(hi) if hi > lo else (None, None)

                im = ax.imshow(
                    img,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    vmin=vmin, vmax=vmax,
                    cmap="magma",
                )
                ax.set_title(ch_labels[idx], fontsize=10)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Order")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis("off")
            idx += 1

    fig.suptitle(f"Mode: {mode} | (C,H,W)=({C},{H},{W})", fontsize=12)
    fig.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig) if save_path else plt.show()
    
def get_dataset(args,
                train_domain=['vat', 'vbl', 'mfd'],
                valid_domain=['dxai']):
    signal_imger = OrderInvariantSignalImager(
                                mode='stft+cross',
                                log1p=True,
                                normalize= "per_channel",  
                                eps=1e-8,
                                out_dtype=torch.float32,
                                max_order=20.0,           
                                H_out=224,                
                                W_out=224,               
                                stft_nperseg=1024,
                                stft_hop=256,
                                stft_window="hann",
                                stft_center=True,
                                stft_power=1.0,           
                            )
    vib_trainset = VibrationDataset(
                                data_root=args.data_root,
                                using_dataset = train_domain,
                                window_sec=5,
                                stride_sec=3,            
                                transform=signal_imger,
                            )
    vib_valset = VibrationDataset(
                                data_root=args.data_root,
                                using_dataset = valid_domain,
                                window_sec=5,
                                stride_sec=3,             
                                transform=signal_imger,
                            )
    
    return vib_trainset, vib_valset

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Vibration LLM training/evaluation script')
    parser.add_argument('--data_root',   type=str, default='/Volumes/dataset_onlyMac/processed', help='llm_dataset_caching.py를 통해 만들어진 데이터 pt파일경로')
    parser.add_argument('--batch_size',    type=int, default=32, help='학습 배치사이즈')
    args = parser.parse_args()
    
    #
    # ================= ALL DATASET TEST & SAVE ======================
    # signal_imger = OrderInvariantSignalImager(
    #                             mode='stft+cross',
    #                             log1p=True,
    #                             normalize= "per_channel",  
    #                             eps=1e-8,
    #                             out_dtype=torch.float32,
    #                             max_order=20.0,           
    #                             H_out=224,                
    #                             W_out=224,               
    #                             stft_nperseg=1024,
    #                             stft_hop=256,
    #                             stft_window="hann",
    #                             stft_center=True,
    #                             stft_power=1.0,           
    #                         )
    # vib_trainset = VibrationDataset(
    #                             data_root=args.data_root,
    #                             using_dataset = ['vat', 'vbl', 'mfd', 'dxai'],
    #                             window_sec=5,
    #                             stride_sec=3,            
    #                             transform=signal_imger,
    #                         )
    # vib_valset = VibrationDataset(
    #                             data_root=args.data_root,
    #                             using_dataset = ['dxai'],
    #                             window_sec=5,
    #                             stride_sec=3,             
    #                             transform=signal_imger,
    #                         )
    # print('Train Set Testing')
    # all_train_data = []
    # all_valid_data = []
    # for data_sample in tqdm(vib_trainset, dynamic_ncols=True):
    #     all_train_data.append(data_sample)
    # # =========== SAVE ===================
    # torch.save(all_train_data, os.path.join(args.data_root, 'llm_vib_trainset_4dataset.pt'))
    # del all_train_data
    
    # print('Validation Set Testing')
    # for data_sample in tqdm(vib_valset, dynamic_ncols=True):
    #     all_valid_data.append(data_sample)
    # # =========== SAVE ===================
    # torch.save(all_valid_data, os.path.join(args.data_root, 'llm_vib_validset_4dataset_only_dxai.pt'))
    # del all_valid_data

    # =========== LOAD & TEST CACHED DATASET ===================
    # train_data_root = os.path.join(args.data_root, 'llm_vib_trainset_4dataset.pt')
    # valid_data_root = os.path.join(args.data_root, 'llm_vib_validset_4dataset_only_dxai.pt')
    # vib_trainset_cached = CachedDataset(data_root=train_data_root)
    # vib_valset_cached = CachedDataset(data_root=valid_data_root)

    # data_dict = vib_trainset[0]
    # for k in data_dict.keys():
    #     print(f'\nkey : {k}')
    #     print(f'item : {data_dict[k]}')
    #     print(f'type : {type(data_dict[k])}\n')
    # data_dict =vib_trainset_cached[0]    
    # for k in data_dict.keys():
    #     print(f'\nkey : {k}')
    #     print(f'item : {data_dict[k]}')
    #     print(f'type : {type(data_dict[k])}\n')
    # print("Train len(before save):", len(vib_trainset))
    # print("Valid len(before save):", len(vib_valset))
    # print("Train len(after load):", len(vib_trainset_cached))
    # print("Valid len(after load):", len(vib_valset_cached))

    # Just one LOO dataset generation
    # dataset_list = ['vat', 'vbl', 'mfd', 'dxai']
    signal_imger = OrderInvariantSignalImager(
                                mode='stft+cross',
                                log1p=True,
                                normalize= "per_channel",  
                                eps=1e-8,
                                out_dtype=torch.float32,
                                max_order=20.0,           
                                H_out=224,                
                                W_out=224,               
                                stft_nperseg=1024,
                                stft_hop=256,
                                stft_window="hann",
                                stft_center=True,
                                stft_power=1.0,           
                            )
    # vib_trainset = VibrationDataset(
    #                             data_root=args.data_root,
    #                             using_dataset = ['vat', 'mfd', 'dxai'],
    #                             window_sec=5,
    #                             stride_sec=3,            
    #                             transform=signal_imger,
    #                         )     
    # all_train_data = []
    # for data_sample in tqdm(vib_trainset, dynamic_ncols=True, desc="Train dataset gathering"):
    #     all_train_data.append(data_sample)
    # torch.save(all_train_data, os.path.join(args.data_root, f'llm_vib_trainset_3dataset_except_vbl.pt'))
    vib_valset = VibrationDataset(
                                data_root=args.data_root,
                                using_dataset = ['vbl'],
                                window_sec=5,
                                stride_sec=3,             
                                transform=signal_imger,
                            )
    print(len(vib_valset))
    all_valid_data = []
    for data_sample in tqdm(vib_valset, dynamic_ncols=True, desc="Valid dataset gathering"):
        all_valid_data.append(data_sample)
    torch.save(all_valid_data, os.path.join(args.data_root, f'llm_vib_validset_only_vbl.pt'))



    # # Automatic LOO dataset generation
    # # # ================= LOO DATASET TEST & SAVE ======================
    # print("Processing LOO DATASET")
    # print("Processing LOO DATASET")
    # print("Processing LOO DATASET")
    # dataset_list = ['vat', 'vbl', 'mfd', 'dxai']
    # for except_dataset in dataset_list:
    #     signal_imger = OrderInvariantSignalImager(
    #                             mode='stft+cross',
    #                             log1p=True,
    #                             normalize= "per_channel",  
    #                             eps=1e-8,
    #                             out_dtype=torch.float32,
    #                             max_order=20.0,           
    #                             H_out=224,                
    #                             W_out=224,               
    #                             stft_nperseg=1024,
    #                             stft_hop=256,
    #                             stft_window="hann",
    #                             stft_center=True,
    #                             stft_power=1.0,           
    #                         )
    #     vib_trainset = VibrationDataset(
    #                                 data_root=args.data_root,
    #                                 using_dataset = [d for d in dataset_list if d != except_dataset],
    #                                 window_sec=5,
    #                                 stride_sec=3,            
    #                                 transform=signal_imger,
    #                             )
    #     vib_valset = VibrationDataset(
    #                                 data_root=args.data_root,
    #                                 using_dataset = [except_dataset],
    #                                 window_sec=5,
    #                                 stride_sec=3,             
    #                                 transform=signal_imger,
    #                             )
    #     all_train_data = []
    #     all_valid_data = []
    #     for data_sample in tqdm(vib_trainset, dynamic_ncols=True, desc="Train dataset gathering"):
    #         all_train_data.append(data_sample)
    #     for data_sample in tqdm(vib_valset, dynamic_ncols=True, desc="Valid dataset gathering"):
    #         all_valid_data.append(data_sample)
    #     # =========== SAVE ===================
    #     torch.save(all_train_data, os.path.join(args.data_root, f'llm_vib_trainset_3dataset_except_{except_dataset}.pt'))
    #     torch.save(all_valid_data, os.path.join(args.data_root, f'llm_vib_validset_only_{except_dataset}.pt'))

    #     # =========== LOAD & TEST CACHED DATASET ===================
    #     # train_data_root = os.path.join(args.data_root, f'llm_vib_trainset_3dataset_except_{except_dataset}.pt')
    #     # valid_data_root = os.path.join(args.data_root, f'llm_vib_validset_only_{except_dataset}.pt')

    #     # vib_trainset_cached = CachedDataset(data_root=train_data_root)
    #     # vib_valset_cached = CachedDataset(data_root=valid_data_root)

    #     # data_dict = vib_trainset[0]
    #     # for k in data_dict.keys():
    #     #     print(f'\nkey : {k}')
    #     #     print(f'item : {data_dict[k]}')
    #     #     print(f'type : {type(data_dict[k])}\n')
    #     # data_dict =vib_trainset_cached[0]    
    #     # for k in data_dict.keys():
    #     #     print(f'\nkey : {k}')
    #     #     print(f'item : {data_dict[k]}')
    #     #     print(f'type : {type(data_dict[k])}\n')
        
    #     # print(except_dataset, 'is valid dataset')
    #     # print("Train len(before save):", len(vib_trainset))
    #     # print("Valid len(before save):", len(vib_valset))
    #     # print("Train len(after load):", len(vib_trainset_cached))
    #     # print("Valid len(after load):", len(vib_valset_cached))
    