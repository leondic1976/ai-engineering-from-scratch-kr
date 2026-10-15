---
title_en: "GPU Setup & Cloud"
source: "phases/00-setup-and-tooling/03-gpu-setup-and-cloud"
source_sha: "1e6fdb7c16aeab51"
model: "gemini-3.8-flash"
translated_at: "2026-10-15"
---
# GPU 설정 및 클라우드

> 학습 목적으로는 CPU로도 충분합니다. 하지만 본격적인 학습에는 GPU가 필요합니다.

**유형:** Build (실습)
**사용 언어:** Python
**선수 레슨:** Phase 0, Lesson 01
**소요 시간:** ~45분

## 학습 목표

- `nvidia-smi` 및 PyTorch의 CUDA API를 사용하여 로컬 GPU 사용 가능 여부 확인하기
- 무료 클라우드 기반 실험을 위해 Google Colab에 T4 GPU 구성하기
- CPU와 GPU에서 행렬 곱셈 벤치마크를 실행하고 속도 향상 측정하기
- fp16 어림수(rule of thumb) 규칙을 사용하여 VRAM에 들어갈 수 있는 최대 모델 크기 추정하기

## 문제 상황

Phase 1~3의 대부분의 레슨은 CPU에서도 원활하게 실행됩니다. 하지만 CNN, Transformer, LLM을 학습하기 시작하면(Phase 4 이상) GPU 가속이 필수적입니다. CPU에서 8시간이 걸리는 학습 작업이 GPU에서는 10분 만에 끝납니다.

선택할 수 있는 방법은 세 가지입니다: 로컬 GPU, 클라우드 GPU, 또는 Google Colab(무료).

## 핵심 개념

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

사용 가능한 GPU가 있는지 확인합니다:

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

1. [colab.research.google.com](https://colab.research.google.com) 접속
2. 런타임 > 런타임 유형 변경 > T4 GPU 선택
3. `!nvidia-smi`를 실행하여 확인

이 코스의 노트북을 Colab에 직접 업로드하여 사용할 수 있습니다.

### 옵션 3: 클라우드 GPU

Lambda Labs, RunPod 또는 Vast.ai를 사용하는 경우:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### GPU가 없으신가요? 문제없습니다.

대부분의 레슨은 CPU에서도 동작합니다. GPU가 필요한 레슨에는 별도로 안내가 되어 있으며 Colab 링크가 포함되어 있습니다.

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

## 연습 문제

1. 위의 벤치마크를 실행하고 CPU와 GPU의 소요 시간을 비교해 보세요.
2. GPU가 없다면 Google Colab에서 실행하고 비교해 보세요.
3. GPU 메모리 용량을 확인하고 적재할 수 있는 가장 큰 모델 크기를 추정해 보세요(어림수 규칙: fp16 기준 파라미터당 2바이트).

### 힌트

**연습 문제 1.** 직접 구현하기 섹션의 코드를 그대로 실행한 다음 CPU 시간, GPU 시간, 속도 향상(speedup) 수치를 기록하세요. 한 번 더 실행해 보세요. 첫 번째 GPU 실행에는 초기 구동 비용이 포함되므로, 두 번째 실행 수치를 비교해야 정확합니다.

**연습 문제 2.** Colab에서 런타임 > 런타임 유형 변경 > GPU를 선택한 다음 셀에 벤치마크 코드를 붙여넣으세요. 먼저 다른 셀에서 `!nvidia-smi`를 실행해 보세요. GPU 이름이 정상적으로 출력되어야 벤치마크 결과가 의미를 갖습니다.

**연습 문제 3.** `code/gpu_check.py`처럼 `torch.cuda.get_device_properties(0).total_memory / 1e9`를 사용하여 전체 메모리를 확인하세요. 바이트 단위의 메모리를 2로 나누면 fp16 기준 파라미터 수를 얻을 수 있습니다. 단, 활성화 값(activation)과 CUDA 컨텍스트도 메모리를 사용하므로 여유 마진을 남겨두어야 합니다.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| CUDA | "GPU 프로그래밍" | GPU에서 코드를 실행할 수 있게 해주는 NVIDIA의 병렬 컴퓨팅 플랫폼 |
| VRAM | "GPU 메모리" | 시스템 RAM과 분리된 GPU 상의 비디오 RAM. 적재 가능한 모델 크기를 제한함. |
| fp16 | "반정밀도(Half precision)" | 16비트 부동소수점으로, 정확도 손실을 최소화하면서 fp32 대비 메모리를 절반만 사용함 |
| Tensor Core | "고속 행렬 하드웨어" | 행렬 곱셈을 위해 특화된 GPU 코어로, 일반 코어보다 4~8배 빠름 |
