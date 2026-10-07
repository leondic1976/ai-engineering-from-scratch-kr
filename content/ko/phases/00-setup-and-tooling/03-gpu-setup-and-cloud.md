---
title_en: "GPU Setup & Cloud"
source: "phases/00-setup-and-tooling/03-gpu-setup-and-cloud"
source_sha: "d7420f66a80b1018"
model: "gemini-3.8-flash"
translated_at: "2026-10-07"
---
# GPU 설정 및 클라우드

> 학습 목적으로는 CPU로도 충분하지만, 실전 훈련에는 GPU가 필요합니다.

**유형:** Build (실습)
**사용 언어:** Python
**선수 레슨:** Phase 0, Lesson 01
**소요 시간:** ~45분

## 학습 목표

- `nvidia-smi` 및 PyTorch의 CUDA API를 사용하여 로컬 GPU 사용 가능 여부 확인하기
- 무료 클라우드 기반 실험을 위해 T4 GPU가 장착된 Google Colab 구성하기
- CPU와 GPU에서 행렬 곱셈(matrix multiplication) 성능을 벤치마크하고 속도 향상 측정하기
- fp16 경험 법칙(rule of thumb)을 사용하여 VRAM에 올릴 수 있는 최대 모델 크기 추정하기

## 문제 상황

Phase 1~3의 대부분의 강의는 CPU에서도 원활하게 실행됩니다. 하지만 CNN, Transformer 또는 LLM 훈련을 시작하면(Phase 4 이상) GPU 가속이 필수적입니다. CPU에서 8시간 걸리는 훈련 작업이 GPU에서는 10분 만에 끝납니다.

선택할 수 있는 옵션은 로컬 GPU, 클라우드 GPU, 또는 무료 Google Colab 등 세 가지가 있습니다.

## 개념

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## 직접 구현하기

### 옵션 1: 로컬 NVIDIA GPU

GPU가 있는지 확인합니다:

```bash
nvidia-smi
```

CUDA 지원 PyTorch 설치:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 옵션 2: Google Colab

1. [colab.research.google.com](https://colab.research.google.com)으로 이동합니다.
2. 런타임 > 런타임 유형 변경 > T4 GPU 선택
3. `!nvidia-smi`를 실행하여 확인

이 과정의 노트북을 Colab에 직접 업로드하여 사용할 수 있습니다.

### 옵션 3: 클라우드 GPU

Lambda Labs, RunPod, 또는 Vast.ai를 사용하는 경우:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### GPU가 없어도 괜찮습니다

대부분의 강의는 CPU에서 동작합니다. GPU가 필요한 강의에는 별도 안내와 함께 Colab 링크가 포함되어 있습니다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 직접 구현하기: GPU vs CPU 벤치마크

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 실습 과제

1. 위의 벤치마크를 실행하고 CPU와 GPU의 소요 시간을 비교해 보세요.
2. GPU가 없다면 Google Colab에서 실행하여 비교해 보세요.
3. 사용 중인 GPU 메모리 용량을 확인하고, 적재 가능한 최대 모델 크기를 추정해 보세요 (경험 법칙: fp16 기준 파라미터당 2바이트).

## 핵심 용어

| 용어 | 일반적인 표현 | 실제 의미 |
|------|----------------|----------------------|
| CUDA | "GPU 프로그래밍" | GPU에서 코드를 실행할 수 있게 해주는 NVIDIA의 병렬 컴퓨팅 플랫폼 |
| VRAM | "GPU 메모리" | 시스템 RAM과 분리된 GPU 상의 비디오 RAM으로, 모델 크기를 제한함 |
| fp16 | "반정밀도(Half precision)" | 16비트 부동소수점으로, 정확도 손실을 최소화하면서 fp32의 절반 수준의 메모리를 사용함 |
| Tensor Core | "고속 행렬 연산 하드웨어" | 행렬 곱셈에 특화된 GPU 코어로, 일반 코어보다 4~8배 빠름 |
