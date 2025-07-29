import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.integrate import cumtrapz
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

DATASETS = ['dxai', 'vat', 'vbl', 'mfd', 'kamp']

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
    if dataset_name == 'kamp':
        return 1000.0
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


def rms_velocity_mm(accel_g: np.ndarray, fs, dataset_name) -> float:
    """
    입력: 가속도 시계열(accel_g) in g 단위, 샘플링 레이트 fs in Hz 처리:
      1) g->m/s^2
      2) 10~1000 Hz bandpass + detrend
      3) 적분하여 velocity (m/s)로 변환 후 mm/s
      4) RMS 반환
    """
    
    # 1) g -> m/s^2
    if dataset_name in ['mfd', 'vbl']:
        accel_ms2 = accel_g  # 정규화된 값, 단위 변환 생략
    else:
        accel_ms2 = accel_g * 9.80665  # g -> m/s^2
        # accel_ms2 = accel_g 
    # 2) bandpass + detrend
    # a_f = bandpass_10_1000(accel_ms2, fs)
    # 3) 적분 -> velocity (m/s)
    vel = cumtrapz(accel_ms2, dx=1/fs, initial=0.0)
    # m/s -> mm/s
    vel_mm = vel * 1000.0
    # 4) RMS
    rms = float(np.sqrt(np.mean(vel_mm**2)))
    # print(f"RMS velocity: {rms}")
    return rms


def classify_severity(rms_mm_s: float) -> str:
    """
    ISO 10816-1 Class I 기준에 따른 RMS 속도(mm/s) 심각도 범주를 반환
    A: <=0.71, B: <=1.80, C: <=4.50, D: >4.50
    """
    if rms_mm_s <= 0.71:
        return 'A'
    elif rms_mm_s <= 1.80:
        return 'B'
    elif rms_mm_s <= 4.50:
        return 'C'
    else:
        return 'D'

def compute_max_rms(file_path):
    ds = get_dataset_name(file_path)
    fs = get_sampling_rate(ds)

    df = pd.read_csv(file_path)

    rms_vals = []
    for col in ['motor_x', 'motor_y']:
        acc = df[col].to_numpy()
        rms = rms_velocity_mm(acc, fs, ds)
        rms_vals.append(rms)

    return max(rms_vals)

def vis_distribution(data_root):
    rms_values = []
    classes = []
    
    for ds in os.listdir(data_root):
        ds_path = os.path.join(data_root, ds)
        if not os.path.isdir(ds_path): continue
        for cls in os.listdir(ds_path):
            cls_path = os.path.join(ds_path, cls)
            if not os.path.isdir(cls_path): continue
            for fname in tqdm(os.listdir(cls_path), desc=f"Processing {ds}/{cls}", unit="file"):
                if not fname.endswith('.csv'): continue
                fp = os.path.join(cls_path, fname)
                try:
                    rms = compute_max_rms(fp)
                    rms_values.append(rms)
                    classes.append(cls)
                except Exception as e:
                    print(f"Error processing {fp}: {e}")
    
    df = pd.DataFrame({
        'class': classes,
        'rms': rms_values
    })

    plt.figure(figsize=(10, 6))
    # 클래스별 히스토그램
    for cls, grp in df.groupby('class'):
        plt.hist(
            grp['rms'],
            bins=50,
            density=True,
            alpha=0.5,
            label=cls
        )
    
    # ISO 10816-1 심각도 기준선
    # plt.axvline(x=0.71, color='green', linestyle='--', label='A/B (0.71 mm/s)')
    # plt.axvline(x=1.80, color='orange', linestyle='--', label='B/C (1.80 mm/s)')
    # plt.axvline(x=4.50, color='red', linestyle='--', label='C/D (4.50 mm/s)')

    # 클래스별 평균선 및 텍스트
    colors = ['purple', 'cyan', 'magenta', 'brown', 'lime']  # 클래스별 색상 (최대 5개 클래스 가정)
    for idx, (cls, grp) in enumerate(df.groupby('class')):
        mean_rms = grp['rms'].mean()
        color = colors[idx % len(colors)]  # 클래스 수 초과 시 색상 순환
        plt.axvline(x=mean_rms, color=color, linestyle='-', alpha=0.7, label=f'{cls} mean ({mean_rms:.2f} mm/s)')
        # 평균값 텍스트 표시 (y축 상단 80% 위치)
        plt.text(mean_rms, plt.ylim()[1] * 0.8, f'{mean_rms:.2f}', color=color, ha='center', va='bottom')

    plt.xlabel('RMS Velocity [mm/s]')
    plt.ylabel('Density')
    plt.title('RMS Velocity Distribution (Each Classes)')
    plt.legend()
    plt.tight_layout()
    
    # plot 디렉토리 생성
    os.makedirs('plot', exist_ok=True)
    plt.savefig('plot/rms_distribution.png')
    plt.show()


# def vis_distribution(data_root):
#     rms_values = []
    
#     for ds in os.listdir(data_root):
#         ds_path = os.path.join(data_root, ds)
#         if not os.path.isdir(ds_path): continue
#         for cls in os.listdir(ds_path):
#             cls_path = os.path.join(ds_path, cls)
#             if not os.path.isdir(cls_path): continue
#             for fname in tqdm(os.listdir(cls_path), desc=f"Processing {ds}/{cls}", unit="file"):
#                 if not fname.endswith('.csv'): continue
#                 fp = os.path.join(cls_path, fname)
#                 try:
#                     rms = compute_max_rms(fp)
#                     rms_values.append(rms)
#                 except Exception as e:
#                     print(f"Error processing {fp}: {e}")
    
#     plt.figure(figsize=(10, 6))
#     plt.hist(
#         rms_values,
#         bins=50,
#         density=True,
#         alpha=0.7,
#         color='blue',
#         label='All Classes'
#     )
#     plt.xlabel('RMS Velocity [mm/s]')
#     plt.ylabel('Density')
#     plt.title('RMS Velocity Distribution (Each Classes)')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('rms_distribution.png')  # PNG 파일로 저장
#     plt.show()


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
