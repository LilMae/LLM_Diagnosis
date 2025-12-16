# Blackwell + CUDA 12.8 + cuDNN 포함된 NVIDIA 공식 이미지
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 기본 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# CUDA 관련 환경변수 (flash-attn 빌드용)
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0"

# 필수 패키지 + Python3 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python-is-python3 \
        python3-dev \
        build-essential \
        cmake \
        ninja-build \
        git \
        libopenblas-dev \
        libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# pip3를 기본 pip로 사용하도록 심볼릭 링크 생성
RUN ln -sf "$(which pip3)" /usr/local/bin/pip && \
    python --version && pip --version

# 최신 pip로 업그레이드
RUN pip install --upgrade pip

# flash-attn setup.py 에서 사용하는 packaging 미리 설치
RUN pip install packaging

# PyTorch nightly (CUDA 12.8 빌드) 설치
# → Blackwell(sm_120) 지원을 위해 nightly 채널 사용
RUN pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# 작업 디렉토리 설정
WORKDIR /workspace