import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.integrate import cumtrapz
import os

DATASETS = ['dxai', 'vat', 'vbl', 'mfd']

def get_dataset_name(file_path):
    parts = os.path.normpath(file_path).split(os.sep)
    for p in parts:
        if p in DATASETS:
            return p
    raise ValueError(f"파일 경로에 dataset 이름이 없습니다: {file_path}")


def get_sampling_rate(dataset_name: str) -> float:
    """
    dataset_name 에 맞춰 샘플링레이트(Hz) 반환
    """
    if dataset_name == 'dxai':
        return 25000.0
    if dataset_name == 'vat':
        return 25600.0
    if dataset_name == 'vbl':
        return 20000.0
    if dataset_name == 'mfd':
        return 50000.0
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def bandpass_10_1000(signal: np.ndarray, fs: float, order: int = 4) -> np.ndarray:
    """
    10~1000 Hz 대역으로 대역통과 필터를 적용합니다.
    """
    nyq = 0.5 * fs
    low = 10.0 / nyq
    high = 1000.0 / nyq
    b, a = butter(order, [low, high], btype='band')
    # 필터링 및 드리프트 제거
    filtered = filtfilt(b, a, signal)
    return detrend(filtered)


def rms_velocity_mm(accel_g: np.ndarray, fs: float) -> float:
    """
    입력: 가속도 시계열(accel_g) in g 단위, 샘플링 레이트 fs in Hz
    처리:
      1) g->m/s^2
      2) 10~1000 Hz bandpass + detrend
      3) 적분하여 velocity (m/s)로 변환 후 mm/s
      4) RMS 반환
    """
    # 1) g -> m/s^2
    accel_ms2 = accel_g * 9.80665
    # 2) bandpass + detrend
    a_f = bandpass_10_1000(accel_ms2, fs)
    # 3) 적분 -> velocity (m/s)
    vel = cumtrapz(a_f, dx=1/fs, initial=0.0)
    # m/s -> mm/s
    vel_mm = vel * 1000.0
    # 4) RMS
    rms = float(np.sqrt(np.mean(vel_mm**2)))
    print(rms)
    return rms


def classify_severity(rms_mm_s: float) -> str:
    """
    ISO 10816-1 Class I 기준에 따른 RMS 속도(mm/s) 심각도 범주를 반환합니다.
    A: <=0.45, B: <=1.12, C: <=2.8, D: >2.8
    """
    if rms_mm_s <= 0.45:
        return 'A'
    elif rms_mm_s <= 1.12:
        return 'B'
    elif rms_mm_s <= 2.8:
        return 'C'
    else:
        return 'D'


# def calculate_severity(data_np: np.ndarray, dataset_name: str) -> str:
#     """
#     다축(raw accel in g) 데이터와 데이터셋 이름을 받아
#     각 축별 RMS 속도를 계산한 뒤 최대치를 기준으로
#     ISO 10816-1 Class I severity(A-D)를 반환합니다.
#     """
#     fs = get_sampling_rate(dataset_name)
#     # 다축 채널 중 최대 RMS velocity
#     rms_vals = [rms_velocity_mm(data_np[ch], fs) for ch in range(data_np.shape[0])]
#     return classify_severity(max(rms_vals))
