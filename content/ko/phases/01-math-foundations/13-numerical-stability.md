---
title_en: "Numerical Stability"
source: "phases/01-math-foundations/13-numerical-stability"
source_sha: "12856656d74e92c2"
model: "gemini-3.8-flash"
translated_at: "2026-10-18"
---
# 수치적 안정성 (Numerical Stability)

> 부동소수점은 불완전한 추상화(leaky abstraction)입니다. 학습 도중 문제를 일으킬 것이며, 그 순간을 미리 알아채기는 어렵습니다.

**유형:** Build (실습)
**Language:** Python
**선수 레슨:** Phase 1, Lessons 01-04
**소요 시간:** ~120분

## 학습 목표

- 최댓값 빼기 트릭(max-subtraction trick)을 사용하여 수치적으로 안정적인 softmax 및 log-sum-exp 구현하기
- 부동소수점 연산에서 오버플로, 언더플로, 유효숫자 소실(catastrophic cancellation) 식별하기
- 중심 유한 차분(centered finite differences)을 사용하여 해석적 그래디언트를 수치적 그래디언트와 대조 검증하기
- 학습 시 float16보다 bfloat16이 선호되는 이유와 손실 스케일링(loss scaling)이 그래디언트 언더플로를 방지하는 방식 설명하기

## 문제 상황

모델을 세 시간 동안 학습시켰는데 갑자기 손실(loss)이 NaN이 됩니다. print 문을 추가해 봅니다. 9,000번째 스텝에서 로짓(logits)은 정상입니다. 9,001번째 스텝에서 로짓은 `inf`가 됩니다. 9,002번째 스텝에 이르면 모든 그래디언트가 `nan`가 되고 학습은 완전히 중단됩니다.

또 다른 상황입니다. 모델 학습은 끝까지 완료되었지만 정확도가 논문에서 보고된 수치보다 2% 낮습니다. 모든 것을 확인해 봅니다. 아키텍처도 일치합니다. 하이퍼파라미터도 일치합니다. 데이터도 일치합니다. 문제는 논문에서는 float32를 사용한 반면, 여러분은 적절한 스케일링 없이 float16을 사용했다는 점입니다. 32비트 연산과의 누적된 반올림 오차가 눈에 띄지 않게 정확도를 갉아먹은 것입니다.

또 다른 상황입니다. 밑바닥부터 교차 엔트로피 손실(cross-entropy loss)을 직접 구현합니다. 작은 로짓 값에서는 잘 작동합니다. 하지만 로짓이 100을 넘어가면 `inf`를 반환합니다. `exp(100)`가 float32로 표현할 수 있는 범위를 넘어서면서 softmax가 오버플로를 일으켰기 때문입니다. 모든 ML 프레임워크는 단 두 줄짜리 트릭으로 이 문제를 처리합니다. 하지만 여러분은 이 트릭의 존재를 몰랐습니다.

수치적 안정성은 단순한 이론적 관심사가 아닙니다. 성공하는 학습 실행과 조용히 실패하는 학습 실행을 가르는 결정적 차이입니다. 여러분이 디버깅하게 될 심각한 ML 버그는 결국 모두 부동소수점 문제로 귀결됩니다.

## 개념 설명

### IEEE 754: 컴퓨터가 실수를 저장하는 방식

컴퓨터는 IEEE 754 표준에 따라 실수를 부동소수점 값으로 저장합니다. 부동소수점 수는 부호 비트, 지수부(exponent), 가수부(mantissa 또는 significand)의 세 부분으로 구성됩니다.

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

가수부는 정밀도(유효숫자가 몇 자리인지)를 결정합니다. 지수부는 범위(숫자가 얼마나 크거나 작을 수 있는지)를 결정합니다.

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32는 약 7자리의 십진수 정밀도를 제공합니다. 이는 1.0000001과 1.0000002는 구별할 수 있지만, 1.00000001과 1.00000002는 구별할 수 없음을 의미합니다. 7자리를 넘어가면 모든 것이 반올림 노이즈(rounding noise)가 됩니다.

float16은 약 3자리의 정밀도를 제공합니다. 표현할 수 있는 가장 큰 숫자는 65,504입니다. 로짓, 그래디언트, 활성화(activation) 값이 이 범위를 일상적으로 초과하는 ML 분야에서는 불안할 정도로 작은 크기입니다.

bfloat16은 float16의 범위 문제에 대한 Google의 해법입니다. float32와 동일한 8비트 지수부를 가져 최대 3.4e38까지 동일한 범위를 지원하지만, 가수부 비트는 7비트에 불과합니다(float16보다 정밀도가 낮음). 신경망 학습에서는 정밀도보다 범위가 훨씬 중요하므로 보통 bfloat16이 더 유리합니다.

### 0.1 + 0.2 != 0.3인 이유

0.1이라는 숫자는 이진 부동소수점으로 정확하게 표현할 수 없습니다. 2진법 체계에서는 순환소수가 되기 때문입니다.

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32는 이를 가수부 23비트까지만 남기고 잘라냅니다. 실제로 저장되는 값은 대략 0.100000001490116입니다. 마찬가지로 0.2는 대략 0.200000002980232로 저장됩니다. 둘의 합은 0.3이 아니라 0.300000004470348이 됩니다.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

이 점이 ML에서 중요한 이유는 다음과 같습니다.

1. `if loss < threshold`와 같은 손실 비교가 잘못된 결과를 낼 수 있습니다.
2. 수많은 작은 값을 누적할 때(수천 스텝에 걸친 그래디언트 업데이트) 실제 참값의 합에서 벗어나게 됩니다.
3. 부동소수점을 `==`로 비교하면 체크섬(checksum) 및 재현성 테스트가 실패합니다.

해결 방법: 부동소수점을 결코 `==`로 비교하지 마십시오. 대신 `abs(a - b) < epsilon`나 `math.isclose()`를 사용하십시오.

### 유효숫자 소실 (Catastrophic Cancellation)

거의 동일한 두 부동소수점 수를 뺄 때, 유효숫자들이 상쇄되면서 반올림 노이즈만 남아 상위 자릿수로 올라오는 현상이 발생합니다.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

단 한 번의 뺄셈으로 19%의 상대 오차가 발생했습니다. ML에서는 다음과 같은 경우에 이러한 현상이 발생합니다.

- 평균이 큰 데이터의 분산을 계산할 때: $\mathbb{E}[x]$가 클 때의 $\mathbb{E}[x^2] - \mathbb{E}[x]^2$
- 거의 동일한 로그 확률(log-probability)을 뺄 때
- 너무 작은 엡실론(epsilon)으로 유한 차분 그래디언트를 계산할 때

해결 방법: 크기가 거의 같으면서 큰 두 수를 빼지 않도록 공식을 재배열하십시오. 분산 계산에는 Welford 알고리즘을 사용하거나 데이터의 중심을 먼저 맞추십시오(centering). 로그 확률의 경우, 전체 과정에서 로그 공간(log-space)을 유지하며 연산하십시오.

### 오버플로와 언더플로

오버플로는 계산 결과가 너무 커서 표현할 수 없을 때 발생합니다. 언더플로는 결과가 너무 작을 때(표현 가능한 가장 작은 양수보다 0에 더 가까울 때) 발생합니다.

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 함수는 ML에서 오버플로를 일으키는 주된 원인입니다.

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 함수는 반대 방향의 문제를 일으킵니다.

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

ML에서 `exp()`은 softmax, sigmoid, 확률 계산에 등장합니다. `log()`는 교차 엔트로피, 로그 우도(log-likelihood), KL 발산(KL divergence)에 등장합니다. 적절한 트릭이 없다면 `log(exp(x))` 조합은 언제 터질지 모르는 지뢰밭과 같습니다.

### Log-Sum-Exp 트릭

`log(sum(exp(x_i)))`을 직접 계산하는 것은 수치적으로 위험합니다. 어떤 `x_i`가 크면 `exp(x_i)`이 오버플로를 일으킵니다. 모든 `x_i`가 매우 큰 음수라면, 모든 `exp(x_i)`이 0으로 언더플로되어 `log(0)`의 결과가 `-inf`이 됩니다.

해결 트릭: 지수 연산을 취하기 전에 최댓값을 뺍니다.

```math
\log\left(\sum_i \exp(x_i)\right) = \max(x) + \log\left(\sum_i \exp(x_i - \max(x))\right)
```

작동 원리: `max(x)`를 빼고 나면 가장 큰 지수 값은 `exp(0) = 1`이 됩니다. 따라서 오버플로가 발생할 수 없습니다. 합의 항 중 적어도 하나는 1이 되므로 전체 합은 최소 1 이상이며, `log(1) = 0`가 성립합니다. 따라서 `-inf`으로 언더플로될 위험도 없습니다.

증명:

```math
\begin{aligned}
& \log\left(\sum_i \exp(x_i)\right) \\
&= \log\left(\sum_i \exp(x_i - c + c)\right) && \text{(add and subtract } c\text{)} \\
&= \log\left(\sum_i \exp(x_i - c) \cdot \exp(c)\right) && (\exp(a+b) = \exp(a) \cdot \exp(b)) \\
&= \log\left(\exp(c) \cdot \sum_i \exp(x_i - c)\right) && \text{(factor out } \exp(c)\text{)} \\
&= c + \log\left(\sum_i \exp(x_i - c)\right) && (\log(a \cdot b) = \log(a) + \log(b))
\end{aligned}
```

`c = max(x)`로 설정하면 오버플로가 제거됩니다.

이 트릭은 ML 전반에 걸쳐 어디서나 사용됩니다.
- Softmax 정규화
- 교차 엔트로피 손실 계산
- 시퀀스 모델에서의 로그 확률 합산
- 가우시안 혼합 모델(Mixture of Gaussians)
- 변분 추론(Variational inference)

### Softmax에 최댓값 빼기 트릭이 필요한 이유

Softmax는 로짓을 확률로 변환합니다.

```math
\mathop{\mathrm{softmax}}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
```

트릭을 쓰지 않으면 [100, 101, 102]와 같은 로짓은 오버플로를 발생시킵니다.

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

트릭을 적용하여 max(x) = 102를 뺍니다.

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

계산된 확률은 완전히 동일합니다. 하지만 연산은 안전합니다. 이것은 단순한 최적화가 아닙니다. 정확성을 위한 필수 요구 사항입니다.

### NaN과 Inf: 감지 및 방지

`nan`(Not a Number)과 `inf`(무한대)는 연산 과정 전체에 바이러스처럼 번집니다. 그래디언트 업데이트에서 단 하나의 `nan`만 발생해도 가중치가 `nan`이 되며, 이는 이후의 모든 출력을 `nan`로 만듭니다. 단 한 스텝 만에 학습이 완전히 중단됩니다.

`inf`가 발생하는 경로:
- 매우 큰 양수에 대한 `exp()` 연산
- 0으로 나누기: `1.0 / 0.0`
- 누적 연산 중의 `float32` 오버플로

`nan`이 발생하는 경로:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 음수에 대한 `sqrt()` 연산
- 음수에 대한 `log()` 연산
- 이미 존재하는 `nan`이 포함된 모든 산술 연산

감지 방법:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

방지 전략:

1. 입력을 `exp()` 범위로 클램핑(clamping): `exp(clamp(x, -80, 80))`
2. 분모에 엡실론 추가: `x / (y + 1e-8)`
3. `log()` 내부에 엡실론 추가: `log(x + 1e-8)`
4. 안정적인 구현체 사용 (log-sum-exp, stable softmax)
5. 가중치 폭발을 방지하기 위한 그래디언트 클리핑(gradient clipping)
6. 디버깅 중에는 매 순전파(forward pass) 후 `nan`/`inf` 존재 여부 확인
### 수치 그래디언트 검사 (Numerical Gradient Checking)

해석적 그래디언트(analytical gradient, 역전파로부터 계산됨)에는 버그가 있을 수 있습니다. 수치 그래디언트 검사는 유한 차분(finite differences)으로 그래디언트를 계산하여 이를 검증합니다.

중심 차분(centered difference) 공식:

```math
\frac{df}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}
```

이는 $O(h^2)$의 정확도를 가지며, $O(h)$에 불과한 전진 차분(forward difference) $(f(x+h) - f(x)) / h$보다 훨씬 우수합니다.

$h$의 선택: 너무 크면 근사치가 부정확해집니다. 너무 작으면 치명적 소거(catastrophic cancellation)로 인해 결과가 망가집니다. 일반적으로 `h = 1e-5`에서 `1e-7` 사이의 값을 사용합니다.

검증 방법: 해석적 그래디언트와 수치 그래디언트 사이의 상대 오차(relative difference)를 계산합니다.

```math
\text{relative\_error} = \frac{\lvert \text{grad}_{\text{analytical}} - \text{grad}_{\text{numerical}} \rvert}{\max(\lvert \text{grad}_{\text{analytical}} \rvert, \lvert \text{grad}_{\text{numerical}} \rvert, 10^{-8})}
```

경험 법칙:
- relative_error < 1e-7: 완벽함, 그래디언트가 올바름
- relative_error < 1e-5: 수용 가능, 아마도 올바름
- relative_error > 1e-3: 어딘가 문제가 있음
- relative_error > 1: 그래디언트가 완전히 틀림

새로운 레이어나 손실 함수를 구현할 때는 항상 그래디언트를 검사해야 합니다. PyTorch는 이를 위해 `torch.autograd.gradcheck()`을 제공합니다.

### 혼합 정밀도 훈련 (Mixed Precision Training)

최신 GPU에는 float16 행렬 곱셈을 float32보다 2~8배 빠르게 계산하는 특화된 하드웨어(Tensor Cores)가 있습니다. 혼합 정밀도(mixed precision) 훈련은 이를 활용합니다:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

순수 float16 훈련의 문제점: 그래디언트가 매우 작은 경우가 많습니다(1e-8 이하). float16은 약 6e-8 미만의 모든 값을 언더플로(underflow)시켜 0으로 만듭니다. 모든 그래디언트 업데이트가 0이 되므로 모델의 학습이 중단됩니다.

해결책은 손실 스케일링(loss scaling)입니다:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

동적 손실 스케일링(dynamic loss scaling)은 스케일 팩터를 자동으로 조정합니다. 큰 값(65536)으로 시작합니다. 그래디언트가 오버플로하여 `inf`이 되면 값을 절반으로 줄입니다. 오버플로 없이 $N$ 스텝이 지나면 값을 두 배로 늘립니다.

### bfloat16 대 float16: 훈련에서 bfloat16이 우세한 이유

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16은 더 높은 정밀도를 갖지만(가수부 비트가 7개가 아닌 10개) 표현 범위가 제한적입니다(최댓값 약 65,504). bfloat16은 정밀도는 낮지만 float32와 동일한 범위를 가집니다(최댓값 약 3.4e38).

신경망 훈련 시:

- 훈련 스파이크가 발생할 때 활성화값(activation)과 로짓(logit)이 65,504를 초과하는 경우가 빈번합니다. float16은 오버플로되지만, bfloat16은 이를 처리할 수 있습니다.
- float16에서는 손실 스케일링이 필수적이지만, bfloat16은 표현 범위가 그래디언트 크기 스펙트럼을 포괄하므로 대개 필요하지 않습니다.
- bfloat16은 단순히 float32를 잘라낸 형태입니다. 가수부의 하위 16비트를 버리기만 하면 됩니다. 변환이 단순하며 지수부의 손실이 없습니다.

값이 제한되어 있고 정밀도가 더 중요한 추론(inference)에는 float16이 선호됩니다. 표현 범위가 더 중요한 훈련에는 bfloat16이 선호됩니다. TPU와 최신 NVIDIA GPU(A100, H100)가 bfloat16을 기본적으로 지원하는 이유가 바로 이것입니다.

### 그래디언트 클리핑 (Gradient Clipping)

그래디언트 폭주(exploding gradients)는 그래디언트가 여러 레이어를 거치며 지수적으로 증가할 때 발생합니다(RNN, 딥 네트워크, Transformer에서 흔함). 단 한 번의 커다란 그래디언트만으로도 한 번의 스텝 만에 모든 가중치를 손상시킬 수 있습니다.

두 가지 유형의 클리핑:

**값 기준 클리핑(Clip by value):** 각 그래디언트 요소를 독립적으로 고정합니다.

```
grad = clamp(grad, -max_val, max_val)
```

간단하지만 그래디언트 벡터의 방향이 바뀔 수 있습니다.

**노름 기준 클리핑(Clip by norm):** 전체 그래디언트 벡터의 노름이 임계값을 초과하지 않도록 스케일을 조정합니다.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

그래디언트의 방향을 유지합니다. 이는 `torch.nn.utils.clip_grad_norm_()`가 수행하는 방식이며 표준적인 선택지입니다.

일반적인 값: Transformer의 경우 `max_norm=1.0`, 강화학습(RL)의 경우 `max_norm=0.5`, 단순한 네트워크의 경우 `max_norm=5.0`.

그래디언트 클리핑은 임시방편(hack)이 아닙니다. 안전 장치입니다. 이것이 없다면 단 하나의 이상치(outlier) 배치만으로도 수주 간의 훈련을 망칠 만큼 커다란 그래디언트가 발생할 수 있습니다.

### 수치 안정화 장치로서의 정규화 레이어

배치 정규화(batch normalization), 레이어 정규화(layer normalization), RMS 정규화(RMS normalization)는 대개 훈련 수렴을 돕는 정규화 기법(regularizer)으로 소개됩니다. 하지만 이들은 수치 안정화 장치이기도 합니다.

정규화가 없으면 활성화값이 레이어를 거치면서 지수적으로 증가하거나 감소할 수 있습니다:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

정규화는 모든 레이어에서 활성화값의 중심을 다시 맞추고 스케일을 재조정합니다:

```math
\mathop{\mathrm{LayerNorm}}(x) = \frac{x - \mathop{\mathrm{mean}}(x)}{\mathop{\mathrm{std}}(x) + \epsilon} \cdot \gamma + \beta
```

`epsilon`(일반적으로 1e-5)는 모든 활성화값이 동일할 때 0으로 나누는 것을 방지합니다. 학습된 파라미터인 `gamma`와 `beta`는 네트워크가 필요로 하는 모든 스케일을 복원할 수 있도록 합니다.

이를 통해 네트워크 전반에 걸쳐 값을 수치적으로 안전한 범위 내로 유지하며, 순전파(forward pass)에서의 오버플로와 역전파(backward pass)에서의 그래디언트 폭주를 모두 방지합니다.

### 흔히 발생하는 ML 수치 버그

**버그: 몇 에포크 후에 손실이 NaN이 됨.**
원인: 로짓이 너무 커져서 softmax가 오버플로됨. 또는 학습률(learning rate)이 너무 높아 가중치가 발산함.
해결책: 안정적인 softmax(최댓값 빼기)를 사용하고, 학습률을 줄이며, 그래디언트 클리핑을 추가합니다.

**버그: 손실이 log(num_classes)에서 멈춤.**
원인: 모델 출력이 균등에 가까운 확률을 나타냄. 흔히 그래디언트가 소실되고 있거나 모델이 전혀 학습하지 못하고 있음을 의미함.
해결책: 데이터 레이블이 올바른지 확인하고, 손실 함수를 검증하며, Dead ReLU가 있는지 확인합니다.

**버그: 검증 정확도가 예상보다 1~3% 낮음.**
원인: 적절한 손실 스케일링 없는 혼합 정밀도 사용. 그래디언트 언더플로가 작은 업데이트를 조용히 0으로 만들어 버림.
해결책: 동적 손실 스케일링을 활성화하거나 bfloat16으로 전환합니다.

**버그: 일부 레이어의 그래디언트 노름이 0.0임.**
원인: Dead ReLU 뉴런(모든 입력이 음수임) 또는 float16 언더플로.
해결책: LeakyReLU나 GELU를 사용하고, 그래디언트 스케일링을 적용하며, 가중치 초기화를 확인합니다.

**버그: 모델이 한 GPU에서는 작동하지만 다른 GPU에서는 다른 결과를 냄.**
원인: 비결정적인 부동소수점 누적 순서. GPU 병렬 축소(reduction)는 하드웨어마다 다른 순서로 합산하며, 부동소수점 덧셈은 결합 법칙이 성립하지 않음.
해결책: 작은 차이(1e-6)는 감수하거나, `torch.use_deterministic_algorithms(True)`을 설정하고 속도 저하를 감수합니다.

**버그: 손실 계산 시 `exp()`가 `inf`을 반환함.**
원인: 최댓값 빼기 트릭 없이 raw 로짓을 `exp()`에 전달함.
해결책: 내부적으로 log-sum-exp를 구현하는 `torch.nn.functional.log_softmax()`를 사용합니다.

**버그: float32에서 float16으로 전환한 후 훈련이 발산함.**
원인: float16은 6e-8 미만의 그래디언트 크기나 65,504를 초과하는 활성화값을 표현할 수 없음.
해결책: 손실 스케일링을 포함한 혼합 정밀도(AMP)를 사용하거나 대신 bfloat16을 사용합니다.

```figure
logsumexp-stability
```

## 직접 만들어 보기

### 1단계: 부동소수점 정밀도 한계 시연

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 2단계: 단순한 softmax 대 안정적인 softmax 구현

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 3단계: 안정적인 log-sum-exp 구현

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 4단계: 안정적인 cross-entropy 구현

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 5단계: 그래디언트 검사

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 직접 사용해 보기

### 혼합 정밀도 시뮬레이션

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 그래디언트 클리핑

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 감지

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

모든 엣지 케이스가 시연된 완전한 구현체는 `code/numerical.py`을 참고하십시오.

## 완성된 작업물

이 레슨을 통해 다음이 완성됩니다:
- 안정적인 softmax, log-sum-exp, cross-entropy, 그래디언트 검사 및 혼합 정밀도 시뮬레이션이 포함된 `code/numerical.py`
- 훈련 중 NaN/Inf 및 수치적 문제를 진단하기 위한 `outputs/prompt-numerical-debugger.md`

이러한 안정적인 구현체들은 Phase 3의 훈련 루프 구축 시 및 Phase 4의 어텐션(attention) 메커니즘 구현 시 다시 등장합니다.
## 연습 문제

1. **치명적 소거(Catastrophic cancellation).** float32 환경에서 순진한(naive) 공식 $\mathbb{E}[x^2] - \mathbb{E}[x]^2$을 사용하여 [1000000.0, 1000001.0, 1000002.0]의 분산을 계산합니다. 그런 다음 웰포드 온라인 알고리즘(Welford's online algorithm)을 사용하여 계산합니다. 참 분산(0.6667)과 비교하여 오차를 대조해 봅니다.

2. **정밀도 탐색.** Python에서 `1.0 + x == 1.0`을 만족하는 가장 작은 양수 float32 값 `x`을 찾습니다. 이것이 머신 엡실론(machine epsilon)입니다. 이 값이 `numpy.finfo(numpy.float32).eps`와 일치하는지 확인합니다.

3. **Log-sum-exp 엣지 케이스.** 다음 조건에서 작성한 `logsumexp_stable` 함수를 테스트합니다: (a) 모든 값이 동일한 경우, (b) 하나의 값이 다른 값들보다 훨씬 큰 경우, (c) 모든 값이 매우 작은 음수(-1000)인 경우. 순진한 버전이 실패하는 지점에서 올바른 결과를 생성하는지 확인합니다.

4. **신경망 레이어의 그래디언트 검사.** 단일 선형 레이어 `y = Wx + b` 및 해석적 역전파(analytical backward pass)를 구현합니다. `numerical_gradient`을(를) 사용하여 3x2 가중치 행렬에 대한 정확성을 검증합니다.

5. **손실 스케일링(Loss scaling) 실험.** float16을 사용한 훈련을 시뮬레이션합니다: [1e-9, 1e-3] 범위의 무작위 그래디언트를 생성하고, float16으로 변환한 뒤, 0이 되는 비율을 측정합니다. 그런 다음 손실 스케일링을 적용(1024를 곱함)하고, float16으로 변환한 후 다시 스케일을 원래대로 되돌려 0이 되는 비율을 재측정합니다.

### 힌트

**연습 문제 1.** 세 값을 제곱하기 전에 `np.float32`을(를) 사용하여 변환한 다음 `variance_naive` 및 `welford_variance`을(를) 호출합니다. float32에서 $10^{12}$ 근처의 값은 간격이 1보다 훨씬 크므로, 순진한 방식의 결과는 0.6667에서 크게 벗어나고 웰포드 방식의 결과는 근접할 것으로 예상할 수 있습니다.

**연습 문제 2.** `x = np.float32(1.0)`부터 시작하여 `np.float32(1.0) + x != np.float32(1.0)` 조건이 만족되는 동안 값을 절반으로 줄여 나갑니다. 모든 값을 float32로 유지한 다음, 여전히 합을 변경했던 마지막 값을 `numpy.finfo(numpy.float32).eps`(즉, $2^{-23}$)와 비교합니다.

**연습 문제 3.** 먼저 각 정답을 예측해 봅니다. $n$개의 동일한 값 $c$에 대해 결과는 $c + \ln n$입니다. 하나의 큰 값에 대해서는 결과가 그 값에 가까워집니다. -1000 근처의 값에 대해서는 결과가 $-1000 + \ln n$에 가까워지는 반면, `logsumexp_naive`은(는) 각 지수 함수가 0으로 언더플로우(underflow)되기 때문에 `-inf`을 반환합니다.

**연습 문제 4.** 손실 $L$을 갖는 $y = Wx + b$의 경우, $W$에 대한 그래디언트는 상류(upstream) 그래디언트와 $x$의 외적(outer product)입니다. 손실을 평탄화된(flattened) $W$의 함수로 감싸고, `numerical_gradient`을(를) 호출한 뒤 두 결과를 모두 `check_gradient`에 전달합니다.

**연습 문제 5.** `10 ** np.random.uniform(-9, -3, n)`을(를) 사용하여 로그 스케일로 크기를 샘플링하고 `np.float16`을(를) 사용하여 변환합니다. float16은 대략 $6 \times 10^{-8}$ 미만의 값을 유지할 수 없으므로, 스케일링 전에는 약 4분의 1의 값이 0이 되고 1024를 곱한 후에는 거의 0이 되지 않을 것으로 예상할 수 있습니다.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| IEEE 754 | "부동소수점 표준" | 이진 부동소수점 형식, 반올림 규칙, 특수 값(inf, nan)을 정의하는 국제 표준입니다. 모든 현대 CPU와 GPU가 이를 구현합니다. |
| 머신 엡실론 (Machine epsilon) | "정밀도 한계" | 주어진 부동소수점 형식에서 $1.0 + e \ne 1.0$을 만족하는 가장 작은 값 $e$입니다. float32의 경우 약 1.19e-7입니다. |
| 치명적 소거 (Catastrophic cancellation) | "뺄셈으로 인한 정밀도 손실" | 거의 동일한 두 부동소수점 숫자를 뺄 때 유효 숫자가 소거되고 반올림 노이즈가 결과에 지배적으로 남는 현상입니다. |
| 오버플로우 (Overflow) | "숫자가 너무 큼" | 결과가 표현 가능한 최댓값을 초과하여 inf가 되는 현상입니다. exp(89)는 float32에서 오버플로우를 발생시킵니다. |
| 언더플로우 (Underflow) | "숫자가 너무 작음" | 결과가 표현 가능한 가장 작은 양수보다 0에 더 가까워 0.0이 되는 현상입니다. exp(-104)는 float32에서 언더플로우를 발생시킵니다. |
| Log-sum-exp 트릭 | "최댓값을 먼저 빼기" | 오버플로우와 언더플로우를 방지하기 위해 $\exp(\max(x))$을(를) 밖으로 묶어 내어 $\log(\sum \exp(x))$을(를) 계산하는 기법입니다. softmax, 교차 엔트로피(cross-entropy), 로그 확률 연산에 사용됩니다. |
| 안정한 소프트맥스 (Stable softmax) | "폭발하지 않는 소프트맥스" | 지수 연산을 취하기 전에 $\max(\text{logits})$을(를) 빼는 기법입니다. 수치적으로 동일한 결과를 내며 오버플로우가 발생하지 않습니다. |
| 그래디언트 검사 (Gradient checking) | "역전파 검증하기" | 구현 버그를 잡기 위해 역전파(backpropagation)에서 계산된 해석적 그래디언트를 유한 차분(finite differences)에서 얻은 수치 그래디언트와 비교하는 기법입니다. |
| 혼합 정밀도 (Mixed precision) | "순전파는 float16, 역전파는 float32" | 속도가 중요한 연산에는 정밀도가 낮은 부동소수점을 사용하고, 수치적으로 민감한 연산에는 정밀도가 높은 부동소수점을 사용하는 기법입니다. 일반적으로 2~3배의 속도 향상을 얻습니다. |
| 손실 스케일링 (Loss scaling) | "그래디언트 언더플로우 방지" | 역전파 전에 손실에 큰 상수를 곱하여 그래디언트가 float16의 표현 범위 내에 머물도록 한 뒤, 가중치를 업데이트하기 전에 동일한 상수로 나누어 주는 기법입니다. |
| bfloat16 | "Brain 부동소수점" | 8비트 지수부(float32와 동일한 범위)와 7비트 가수부(float16보다 낮은 정밀도)를 가진 Google의 16비트 형식입니다. 훈련에 주로 선호됩니다. |
| 그래디언트 클리핑 (Gradient clipping) | "그래디언트 노름 제한" | 그래디언트 벡터의 노름(norm)이 임계값을 초과하지 않도록 크기를 조절하는 기법입니다. 그래디언트 폭주(exploding gradients)가 가중치를 망치는 것을 방지합니다. |
| NaN | "숫자가 아님 (Not a Number)" | 정의되지 않은 연산(0/0, inf-inf, sqrt(-1))에서 발생하는 특수 부동소수점 값입니다. 이후의 모든 산술 연산으로 전파됩니다. |
| Inf | "무한대 (Infinity)" | 오버플로우나 0으로 나누기 연산에서 발생하는 특수 부동소수점 값입니다. 다른 연산과 결합하여 NaN을 생성할 수 있습니다(inf - inf, inf * 0). |
| 수치 그래디언트 (Numerical gradient) | "무차별 대입 미분" | $f(x+h)$ 및 $f(x-h)$을(를) 계산하고 $2h$로 나누어 도함수를 근사하는 기법입니다. 느리지만 검증에 신뢰할 수 있습니다. |

## 더 읽을거리

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 가장 신뢰할 수 있는 참고 자료이며, 난이도가 있지만 완벽한 내용을 다룹니다.
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- float16 훈련을 위한 손실 스케일링을 도입한 NVIDIA의 논문입니다.
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch의 혼합 정밀도에 관한 실무 가이드입니다.
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Google이 TPU에 이 형식을 선택한 이유를 설명합니다.
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 부동소수점 합산에서 반올림 오차를 줄이기 위한 알고리즘입니다.
