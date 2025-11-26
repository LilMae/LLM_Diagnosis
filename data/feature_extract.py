from torch.utils.data import Dataset
import numpy as np
import os

from data.dataset import VibrationDataset

def delete_dc(vib: np.ndarray):
    vib = vib - vib.mean()
    return vib

def feat_mean(vib: np.ndarray):
    """p1: Mean value"""
    vib = delete_dc(vib)
    mean = float(np.mean(vib))
    return mean

def feat_variance(vib: np.ndarray):
    vib = delete_dc(vib)
    mean_v = np.mean(vib)
    variance = float(np.mean((vib - mean_v) ** 2))
    return variance

def feat_std(vib: np.ndarray):
    v = feat_variance(vib)
    std = float(np.sqrt(v))
    return std

def feat_max(vib: np.ndarray):
    vib = delete_dc(vib)
    max_v = float(np.max(vib))
    return max_v

def feat_min(vib: np.ndarray):
    vib = delete_dc(vib)
    min_v = float(np.min(vib))
    return min_v

def feat_peak_abs(vib: np.ndarray):
    vib = delete_dc(vib)
    peak_abs = float(np.max(np.abs(vib)))
    return peak_abs

def feat_kurtosis(vib: np.ndarray):
    vib = delete_dc(vib)
    mean_v = np.mean(vib)
    m2 = np.mean((vib - mean_v) ** 2)
    m4 = np.mean((vib - mean_v) ** 4)
    kurtosis = float(m4 / (m2 ** 2))
    return kurtosis

def feat_skewness(vib: np.ndarray):
    vib = delete_dc(vib)
    mean_v = np.mean(vib)
    m2 = np.mean((vib - mean_v) ** 2)
    m3 = np.mean((vib - mean_v) ** 3)
    skewness = float(m3 / (m2 ** 1.5))
    return skewness

def feat_crest_factor(vib: np.ndarray):
    vib = delete_dc(vib)
    peak = float(np.max(np.abs(vib)))
    rms = float(np.sqrt(np.mean(vib ** 2)))
    crest = float(peak / (rms))
    return crest


def feat_peak_freq(vib: np.ndarray, sr):
    x = np.asarray(vib, dtype=np.float64)
    N = x.size
    x = (x - x.mean())
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0/sr)
    mag = np.abs(X)
    peak_freq = float(freqs[int(np.argmax(mag))])
    return peak_freq

def feat_rms_freq(vib: np.ndarray, sr):
    x = np.asarray(vib, dtype=np.float64)
    N = x.size
    x = (x - x.mean())
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0/sr)
    P = (np.abs(X) ** 2)
    denom = float(P.sum())
    rms_freq = float(np.sqrt(((freqs ** 2) * P).sum() / denom))
    return rms_freq

def feat_center_freq(vib: np.ndarray, sr):
    """p13: Center frequency = sum(f_k |X|^2) / sum(|X|^2)"""
    x = np.asarray(vib, dtype=np.float64)
    N = x.size # 수정된 부분: x.size() -> x.size
    x = (x - x.mean())
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)
    P = (np.abs(X) ** 2)
    denom = float(P.sum())
    center_freq = float((freqs * P).sum() / denom)
    return center_freq



def rms_ac(vib: np.ndarray):
    vib = vib - vib.mean()
    return float(np.sqrt(np.mean(np.power(vib, 2))))

def order_one_channel(sig: np.ndarray, sr: float, rpm, od):
    sig = np.asarray(sig, dtype=np.float64)
    N = sig.shape[-1] # 시간 영역 샘플 개수
    if N < 8 or sr <= 0 or rpm <= 0:
        return 0.0 
    sig_ac = sig - sig.mean()
    w = np.hanning(N)
    xw = sig_ac * w

    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)
    df = sr / N

    f_rot = rpm / 60.0
    f_target = od * f_rot # 분석하고자 하는 주파수 (order)

    bw_hz = max (2.0 * df, 0.1 * f_target)
    band_mask = np.abs(freqs - f_target) <= bw_hz 

    order_val = np.abs(X[band_mask])
    return float(order_val.max()) # order 근처를 포함한 대역의 |X|의 내의 최대값을 반환

def fft_spectrum(vib: np.ndarray, sr, use_hann):
    x = np.asarray(vib, dtype=np.float64)
    N = x.size

    x = x - x.mean()

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)
    mag = np.abs(X)
    return freqs, mag

def band_peak(freqs: np.ndarray, mag: np.ndarray, f0: float, bw_hz: float = 10.0):
    mask = np.abs(freqs - f0) <= bw_hz
    peak = float(mag[mask].max())
    return peak

def bpfo_bpfi_peaks(vib: np.ndarray, sr, bpfo_hz, bpfi_hz, bw_hz, use_hann):
    if bpfo_hz != None and bpfi_hz != None:
        fx, mx = fft_spectrum(vib, sr, use_hann)
        bpfo_peak = band_peak(fx, mx, bpfo_hz, bw_hz)
        bpfi_peak = band_peak(fx, mx, bpfi_hz, bw_hz)
        return bpfo_peak, bpfi_peak
    else:
        return None, None

def p2p(vib: np.ndarray):

    x = np.asarray(vib, dtype=np.float64)
    if x.size == 0:
        return 0.0
    peakTopeak = float(x.max() - x.min())
    return peakTopeak


class LLM_Dataset(Dataset):
    def __init__(self, vibration_dataset:VibrationDataset,
                include_ref= True):
        super().__init__()
        
        self.vibration_dataset = vibration_dataset
        
    def __len__(self):
        return len(self.vibration_dataset)

    def feature_extract(self, vibration:np.array, sr: float, rpm: float, bpfo_hz: float, bpfi_hz: float):
        """_summary_

        Args:
            vibration (np.array): 진동 데이터

        Returns:
            feature_dict (dict): vibration에서 추출한 특징 dict
        """
        rms_x = rms_ac(vibration[0])
        rms_y = rms_ac(vibration[1])
        
        odx_1x = order_one_channel(vibration[0], sr, rpm=rpm, od=1) # x축의 order
        odx_2x = order_one_channel(vibration[0], sr, rpm=rpm, od=2)
        odx_3x = order_one_channel(vibration[0], sr, rpm, od=3)

        ody_1x = order_one_channel(vibration[1], sr, rpm, od=1) # y축의 order
        ody_2x = order_one_channel(vibration[1], sr, rpm, od=2)
        ody_3x = order_one_channel(vibration[1], sr, rpm, od=3)

        p2p_x = p2p(vibration[0])
        p2p_y = p2p(vibration[1])

        bpfo_peak_x, bpfi_peak_x = bpfo_bpfi_peaks(vibration[0], sr, bpfo_hz, bpfi_hz, bw_hz=10.0, use_hann=False)
        bpfo_peak_y, bpfi_peak_y = bpfo_bpfi_peaks(vibration[1], sr, bpfo_hz, bpfi_hz, bw_hz=10.0, use_hann=False)

        fx = {
            "mean_x": feat_mean(vibration[0]),
            "var_x": feat_variance(vibration[0]),
            "std_x": feat_std(vibration[0]),
            "max_x": feat_max(vibration[0]),
            "min_x": feat_min(vibration[0]),
            "peak_abs_x": feat_peak_abs(vibration[0]),
            "kurtosis_x": feat_kurtosis(vibration[0]),
            "skewness_x": feat_skewness(vibration[0]),
            "crest_factor_x": feat_crest_factor(vibration[0]),
            "peak_freq_x": feat_peak_freq(vibration[0], sr),
            "rms_freq_x": feat_rms_freq(vibration[0], sr),
            "center_freq_x": feat_center_freq(vibration[0], sr),
        }
        fy = {
            "mean_y": feat_mean(vibration[1]),
            "var_y": feat_variance(vibration[1]),
            "std_y": feat_std(vibration[1]),
            "max_y": feat_max(vibration[1]),
            "min_y": feat_min(vibration[1]),
            "peak_abs_y": feat_peak_abs(vibration[1]),
            "kurtosis_y": feat_kurtosis(vibration[1]),
            "skewness_y": feat_skewness(vibration[1]),
            "crest_factor_y": feat_crest_factor(vibration[1]),
            "peak_freq_y": feat_peak_freq(vibration[1], sr),
            "rms_freq_y": feat_rms_freq(vibration[1], sr),
            "center_freq_y": feat_center_freq(vibration[1], sr),
        }

        feature_dict = {"rms_x": rms_x, "rms_y": rms_y, "order_x_1x": odx_1x, "order_x_2x": odx_2x, "order_x_3x": odx_3x,
                         "order_y_1x": ody_1x, "order_y_2x": ody_2x, "order_y_3x": ody_3x, "peak2peak_x": p2p_x, "peak2peak_y": p2p_y,
                         "bpfo_peak_x": bpfo_peak_x, "bpfi_peak_x": bpfi_peak_x, "bpfo_peak_y": bpfo_peak_y, "bpfi_peak_y": bpfi_peak_y,
                         "mean_x": feat_mean(vibration[0]), "var_x": feat_variance(vibration[0]), "std_x": feat_std(vibration[0]),
                        "max_x": feat_max(vibration[0]), "min_x": feat_min(vibration[0]), "peak_abs_x": feat_peak_abs(vibration[0]),
                        "kurtosis_x": feat_kurtosis(vibration[0]), "skewness_x": feat_skewness(vibration[0]), "crest_factor_x": feat_crest_factor(vibration[0]),
                        "peak_freq_x": feat_peak_freq(vibration[0], sr), "rms_freq_x": feat_rms_freq(vibration[0], sr), "center_freq_x": feat_center_freq(vibration[0], sr),
                        "mean_y": feat_mean(vibration[1]), "var_y": feat_variance(vibration[1]),"std_y": feat_std(vibration[1]),
                        "max_y": feat_max(vibration[1]), "min_y": feat_min(vibration[1]), "peak_abs_y": feat_peak_abs(vibration[1]),
                        "kurtosis_y": feat_kurtosis(vibration[1]), "skewness_y": feat_skewness(vibration[1]), "crest_factor_y": feat_crest_factor(vibration[1]),
                        "peak_freq_y": feat_peak_freq(vibration[1], sr), "rms_freq_y": feat_rms_freq(vibration[1], sr), "center_freq_y": feat_center_freq(vibration[1], sr),}
        return feature_dict
    
    def __getitem__(self, index):
        
        data_dict = self.vibration_dataset[index]

        sr = float(data_dict["x_info"]["sampling_rate"])
        rpm = float(data_dict["x_info"]["rpm"])
        dataset = data_dict["x_info"]["dataset"]
        
        bpfo_hz = None
        bpfi_hz = None

        if dataset == "dxai":
            bpfo_hz = 107.09
            bpfi_hz = 155.7
        elif dataset == "vat":
            bpfo_hz = 179.43
            bpfi_hz = 272.07
        elif dataset == "mfd":
            bpfo_hz = 61.85
            bpfi_hz = 103.2
        else:
            bpfo_hz = None
            bpfi_hz = None
        
        x_feat = self.feature_extract(data_dict['x_vib'], sr=sr, rpm=rpm, bpfo_hz=bpfo_hz, bpfi_hz=bpfi_hz)
        
        ref_feat = None
        if 'ref_vib' in data_dict.keys():
            ref_sr = float(data_dict["ref_info"]["sampling_rate"])
            ref_rpm = float(data_dict["ref_info"]["rpm"])
            ref_dataset = data_dict["ref_info"]["dataset"]

            ref_bpfo_hz = None
            ref_bpfi_hz = None

            if ref_dataset == "dxai":
                ref_bpfo_hz = 107.09
                ref_bpfi_hz = 155.7
            elif ref_dataset == "vat":
                ref_bpfo_hz = 179.43
                ref_bpfi_hz = 272.07
            elif ref_dataset == "mfd":
                ref_bpfo_hz = 61.85
                ref_bpfi_hz = 103.2
            else:
                ref_bpfo_hz = None
                ref_bpfi_hz = None

            ref_feat = self.feature_extract(data_dict['ref_vib'], 
                                            sr=ref_sr, 
                                            rpm=ref_rpm, 
                                            bpfo_hz=ref_bpfo_hz, 
                                            bpfi_hz=ref_bpfi_hz)

        cur_status = {}
        if ref_feat is not None and x_feat is not None:
            common_keys = set(x_feat.keys()) & set(ref_feat.keys())
            eps = 1e-6
            for k in common_keys:
                x_value = x_feat[k]
                ref_value = ref_feat[k]
                if isinstance(x_value, (int, float, np.floating)) and isinstance(ref_value, (int, float, np.floating)):
                    denom = ref_value if abs(ref_value) > eps else eps
                    ratio = (x_value - ref_value) / denom * 100.0
                    cur_status[k] = round(float(ratio), 1)  # 소수 1자리 정도만
        
        # cur_status를 딕셔너리 형식으로 반환 (0이 아닌 값만 필터링)
        cur_status_filtered = {k: v for k, v in cur_status.items() if abs(v) > 1e-6}
        
        # ground-truth 진단 라벨과 모달 입력을 함께 반환해 GRPO 보상 계산에 활용한다.
        gt_label = str(data_dict["x_info"].get("merged_class", ""))

        llm_data_dict = {
            'cur_status' : cur_status_filtered,  # 딕셔너리 형식: {'rms_x': 3622.7905, ...}
            'x_feat' : x_feat,
            'ref_feat' : ref_feat,
            'x_stft': data_dict.get('x_stft'),
            'ref_stft': data_dict.get('ref_stft'),
            'gt': gt_label
        }
        
        return llm_data_dict
