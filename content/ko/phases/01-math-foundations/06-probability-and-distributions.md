---
title_en: "Probability and Distributions"
source: "phases/01-math-foundations/06-probability-and-distributions"
source_sha: "a07e78b85e86e435"
model: "gemini-3.8-flash"
translated_at: "2026-10-23"
---
# 확률과 분포

> 확률은 AI가 불확실성을 표현하기 위해 사용하는 언어입니다.

**유형:** Learn (이론)
**Language:** Python
**선수 레슨:** Phase 1, Lessons 01-04
**소요 시간:** ~75분

## 학습 목표

- 베르누이(Bernoulli), 범주형(categorical), 포아송(Poisson), 균등(uniform), 정규(normal) 분포의 PMF와 PDF를 밑바닥부터 구현합니다.
- 기댓값(expected value)과 분산(variance)을 계산하고, 중심 극한 정리(Central Limit Theorem)를 사용하여 가우스 분포가 널리 쓰이는 이유를 설명합니다.
- 수치적 안정성을 위한 트릭(최대 로짓 빼기)을 적용하여 softmax 및 log-softmax 함수를 구현합니다.
- 로짓으로부터 교차 엔트로피 손실(cross-entropy loss)을 계산하고, 이를 음의 로그 우도(negative log-likelihood)와 연결합니다.

## 문제 정의

분류기(classifier)가 `[0.03, 0.91, 0.06]`을 출력합니다. 언어 모델은 50,000개의 후보 단어 중에서 다음 단어를 선택합니다. 확산 모델(diffusion model)은 학습된 분포로부터 샘플링하여 이미지를 생성합니다. 이 모든 것이 실제로 동작하는 확률의 예입니다.

모델이 수행하는 모든 예측은 확률 분포입니다. 모든 손실 함수는 예측된 분포가 실제 분포와 얼마나 다른지 측정합니다. 모든 훈련 단계는 한 분포가 다른 분포와 더 비슷해지도록 매개변수를 조정합니다. 확률에 대한 이해 없이는 머신러닝 논문을 한 편도 제대로 읽을 수 없고, 단 하나의 모델도 디버깅할 수 없으며, 훈련 손실이 왜 NaN이 되는지 이해할 수 없습니다.

## 핵심 개념

### 사건, 표본 공간, 그리고 확률

표본 공간(sample space) $S$은 일어날 수 있는 모든 결과의 집합입니다. 사건(event)은 표본 공간의 부분집합입니다. 확률은 사건을 0과 1 사이의 숫자에 대응시킵니다.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

세 가지 공리가 확률의 모든 것을 정의합니다:
1. 모든 사건 $A$에 대해 $P(A) \ge 0$
2. $P(S) = 1$ (어떤 사건은 반드시 일어남)
3. $A$와 $B$가 동시에 일어날 수 없을 때 $P(A \text{ or } B) = P(A) + P(B)$

그 밖의 모든 것(베이즈 정리, 기댓값, 분포)은 이 세 가지 규칙으로부터 유도됩니다.

### 조건부 확률과 독립

$P(A \mid B)$는 $B$가 일어났을 때 $A$가 일어날 확률입니다.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

한 사건을 아는 것이 다른 사건에 대해 아무런 정보도 주지 않을 때, 두 사건은 서로 독립(independent)입니다:

```math
\begin{aligned}
\text{Independent:} \quad & P(A \mid B) = P(A) \\
\text{Equivalent to:} \quad & P(A \text{ and } B) = P(A) \cdot P(B)
\end{aligned}
```

동전 던지기는 독립입니다. 비복원 추출로 카드를 뽑는 것은 독립이 아닙니다.

### 확률 질량 함수 vs 확률 밀도 함수

이산 확률 변수는 확률 질량 함수(probability mass function, PMF)를 갖습니다. 각 결과는 직접 읽을 수 있는 특정 확률 값을 가집니다.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

연속 확률 변수는 확률 밀도 함수(probability density function, PDF)를 갖습니다. 단일 지점에서의 밀도는 확률이 아닙니다. 확률은 특정 구간에 걸쳐 밀도를 적분함으로써 얻어집니다.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

이러한 구분은 머신러닝에서 중요합니다. 분류(classification) 모델의 출력은 PMF(이산적인 선택)입니다. VAE 잠재 공간(latent space)은 PDF(연속적)를 사용합니다.

### 주요 분포

**베르누이 분포(Bernoulli):** 단 1회의 시행, 2가지 결과. 이진 분류(binary classification)를 모델링합니다.

```math
\begin{aligned}
P(X = 1) &= p \\
P(X = 0) &= 1 - p \\
\text{Mean} = p, &\quad \text{Variance} = p(1-p)
\end{aligned}
```

**범주형 분포(Categorical):** 단 1회의 시행, $k$가지 결과. 다중 클래스 분류(multi-class classification, softmax 출력)를 모델링합니다.

```math
\begin{aligned}
& P(X = i) = p_i, \quad \text{where } \textstyle\sum_i p_i = 1 \\
& \text{Example: } P(\text{cat}) = 0.7, \quad P(\text{dog}) = 0.2, \quad P(\text{bird}) = 0.1
\end{aligned}
```

**균등 분포(Uniform):** 모든 결과의 발생 확률이 동일합니다. 무작위 초기화(random initialization)에 사용됩니다.

```math
\begin{aligned}
\text{Discrete:} \quad & P(X = k) = 1/n \text{ for } k \in \lbrace 1, \ldots, n \rbrace \\
\text{Continuous:} \quad & f(x) = 1/(b-a) \text{ for } x \in [a, b]
\end{aligned}
```

**정규 분포(가우스 분포, Normal/Gaussian):** 종 모양 곡선입니다. 평균($\mu$)과 분산($\sigma^2$)을 모수(parameter)로 가집니다.

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**포아송 분포(Poisson):** 고정된 구간에서 발생하는 희귀 사건의 횟수입니다. 사건 발생률을 모델링합니다.

```math
\begin{aligned}
P(X = k) &= \frac{\lambda^k e^{-\lambda}}{k!} \\
\text{Mean} = \lambda, &\quad \text{Variance} = \lambda
\end{aligned}
```

### 기댓값과 분산

기댓값은 결과들의 가중 평균입니다.

```math
\begin{aligned}
\text{Discrete:} \quad & \mathbb{E}[X] = \sum_i x_i \, P(X = x_i) \\
\text{Continuous:} \quad & \mathbb{E}[X] = \int x \, f(x) \, dx
\end{aligned}
```

분산은 평균을 중심으로 흩어진 정도를 측정합니다.

```math
\begin{aligned}
\mathrm{Var}(X) &= \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \\
\text{Standard deviation} &= \sqrt{\mathrm{Var}(X)}
\end{aligned}
```

머신러닝에서 기댓값은 손실 함수(데이터 분포에 대한 평균 손실)의 형태로 나타납니다. 분산은 모델의 안정성을 알려줍니다. 그래디언트의 분산이 크다는 것은 훈련 과정에 노이즈가 많다는 것을 의미합니다.

### 결합 분포와 주변 분포

결합 분포(joint distribution) $P(X, Y)$는 두 확률 변수를 함께 설명합니다.

결합 PMF 예시 ($X$ = 날씨, $Y$ = 우산):

| | $Y=0$ (우산 없음) | $Y=1$ (우산 있음) | 주변 $P(X)$ |
|---|---|---|---|
| $X=0$ (맑음) | 0.40 | 0.10 | $P(X=0) = 0.50$ |
| $X=1$ (비) | 0.05 | 0.45 | $P(X=1) = 0.50$ |
| **주변 $P(Y)$** | $P(Y=0) = 0.45$ | $P(Y=1) = 0.55$ | 1.00 |

주변 분포(marginal distribution)는 다른 변수에 대해 모두 합산하여 구합니다:

```math
P(X = x) = \sum_{y} P(X = x, Y = y)
```

위 표의 행과 열 합계가 바로 주변 확률입니다.

### 정규 분포가 도처에 나타나는 이유

중심 극한 정리(Central Limit Theorem): 원래 분포의 형태와 상관없이 분산이 유한하기만 하다면, 동일한 분포에서 추출한 독립적인 확률 변수 다수의 평균을 중심화하고 $\sqrt{n}$로 스케일링했을 때 정규 분포에 수렴합니다. 중심화되고 스케일링된 합에 대해서도 동일하게 적용됩니다.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution with finite variance.
```

이것이 다음과 같은 이유입니다:
- 측정 오차는 대략 정규 분포를 따릅니다(작고 독립적인 여러 요인의 영향).
- 신경망의 가중치 초기화에 정규 분포를 사용합니다.
- SGD에서의 그래디언트 노이즈는 대략 정규 분포를 따릅니다(여러 표본 그래디언트의 합).
- 정규 분포는 주어진 평균과 분산에 대해 최대 엔트로피(maximum entropy)를 갖는 분포입니다.

### 로그 확률

확률 값을 그대로 다루면 수치적 문제가 발생합니다. 작은 확률 여러 개를 함께 곱하면 빠르게 언더플로(underflow)가 발생하여 0이 됩니다.

```math
\begin{aligned}
P(\text{sentence}) &= P(\text{word}_1) \cdot P(\text{word}_2) \cdot \ldots \cdot P(\text{word}_n) \\
&= 0.01 \cdot 0.003 \cdot 0.02 \cdot \ldots \\
&\to 0.0 \quad \text{(underflow after } \sim 30 \text{ terms)}
\end{aligned}
```

로그 확률(log probability)은 이 문제를 해결합니다. 곱셈이 덧셈으로 바뀝니다.

```math
\begin{aligned}
\log P(\text{sentence}) &= \log P(\text{word}_1) + \log P(\text{word}_2) + \ldots + \log P(\text{word}_n) \\
&= -4.6 + -5.8 + -3.9 + \ldots \\
&\to \text{finite number (no underflow)}
\end{aligned}
```

규칙:
- $\log(a \cdot b) = \log(a) + \log(b)$
- 로그 확률은 항상 $\le 0$입니다 ($0 < P \le 1$이므로).
- 음수 값이 더 작을수록(절댓값이 클수록) = 일어날 가능성이 낮음
- 교차 엔트로피 손실은 정답 클래스의 음의 로그 확률입니다.

### 확률 분포로서의 Softmax

신경망은 가공되지 않은 점수(로짓, logits)를 출력합니다. Softmax는 이를 유효한 확률 분포로 변환합니다.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Softmax 트릭: 지수 계산 전 최댓값 로짓을 빼서 오버플로(overflow)를 방지합니다.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax는 수치적 안정성을 위해 softmax와 로그를 결합합니다. PyTorch는 교차 엔트로피 손실 계산 시 내부적으로 이를 사용합니다.

### 샘플링

샘플링(sampling)은 분포로부터 무작위 값을 추출하는 것을 의미합니다. 머신러닝에서는 다음과 같이 쓰입니다:
- 드롭아웃(Dropout)은 0으로 만들 뉴런을 무작위로 샘플링합니다.
- 데이터 증강(Data augmentation)은 무작위 변환을 샘플링합니다.
- 언어 모델은 예측된 분포로부터 다음 토큰을 샘플링합니다.
- 확산 모델은 노이즈를 샘플링한 후 점진적으로 노이즈를 제거합니다.

임의의 분포에서 샘플링하려면 역변환 샘플링(inverse transform sampling), 기각 샘플링(rejection sampling), 또는 재매개변수화 트릭(reparameterization trick, VAE에서 사용)과 같은 기법이 필요합니다.

```figure
gaussian-pdf
```

## 직접 구현하기

### 1단계: 확률 기초

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 2단계: PMF와 PDF 밑바닥부터 구현하기

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 3단계: 기댓값과 분산

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 4단계: 분포로부터 샘플링하기

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 5단계: Softmax와 로그 확률

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 6단계: 중심 극한 정리 시연

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 7단계: 시각화

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

모든 시각화가 포함된 전체 구현은 `code/probability.py`에 있습니다.

## 라이브러리 활용하기

NumPy와 SciPy를 사용하면 위의 모든 내용을 한 줄로 작성할 수 있습니다:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

지금까지 이를 밑바닥부터 직접 구현해 보았습니다. 이제 라이브러리 함수가 내부적으로 어떤 동작을 수행하는지 이해할 수 있을 것입니다.
## 연습 문제

1. 지수 분포(exponential distribution)에 대한 역변환 샘플링(inverse transform sampling)을 구현합니다. 10,000개의 값을 샘플링하고 히스토그램을 실제 PDF와 비교하여 검증합니다.

2. 조작된 주사위(loaded dice) 두 개에 대한 결합 분포표를 만듭니다. 주변 분포(marginal distribution)를 계산하고 두 주사위가 독립인지 확인합니다.

3. 정답 클래스가 인덱스 3일 때 로짓(logit) `[2.0, 0.5, -1.0, 3.0, 0.1]`를 출력하는 5개 클래스 분류기의 교차 엔트로피 손실(cross-entropy loss)을 계산합니다. 그런 다음 PyTorch의 `nn.CrossEntropyLoss`를 사용하여 정답을 확인합니다.

4. 로그 확률(log probability) 리스트를 받아 가장 가능성이 높은 시퀀스, 총 로그 확률, 이에 상응하는 원래 확률(raw probability)을 반환하는 함수를 작성합니다. 각 단어의 확률이 0.01인 50개 단어로 이루어진 문장으로 테스트합니다.

### 힌트

**연습 문제 1.** 균등 분포(uniform distribution)에서 `u`을 추출하고 $-\ln(1 - u) / \lambda$을 반환합니다. 표본 평균이 $1/\lambda$에 가까운지, 히스토그램 빈이 $\lambda e^{-\lambda x}$을 따르는지 확인합니다.

**연습 문제 2.** 모든 셀이 확률이고 모든 셀의 합이 1인 6x6 표를 채운 다음 `joint_to_marginals`를 호출합니다. `check_independence`을 사용합니다. 모든 셀이 두 주변 확률의 곱과 같을 때만 주사위가 독립입니다.

**연습 문제 3.** 로짓에 `softmax`를 적용하고 인덱스 3의 확률에 $-\ln$를 취합니다. 결과는 동일한 로짓과 타깃에 대해 `nn.CrossEntropyLoss`가 제공하는 값인 0.421에 가까워야 합니다.

**연습 문제 4.** 원래 확률을 곱하지 말고 로그 확률을 더합니다. 확률이 0.01인 50개 단어의 경우 총합은 $50 \ln(0.01) \approx -230.26$이어야 하며, 원래 확률은 $10^{-100}$입니다.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| 표본 공간(Sample space) | "모든 가능성" | 실험에서 발생 가능한 모든 결과의 집합 $S$ |
| PMF(확률 질량 함수) | "확률 함수" | 각 이산적인 결과의 정확한 확률을 제공하며 합이 1이 되는 함수 |
| PDF(확률 밀도 함수) | "확률 곡선" | 연속 변수에 대한 밀도 함수. 특정 구간에 대해 적분하여 확률을 계산함 |
| 조건부 확률(Conditional probability) | "무언가가 주어졌을 때의 확률" | $P(A \mid B) = P(A \text{ and } B) / P(B)$. 베이즈적 사고와 베이즈 정리(Bayes' theorem)의 기초 |
| 독립(Independence) | "서로 영향을 주지 않음" | $P(A \text{ and } B) = P(A) \cdot P(B)$. 한 사건을 안다고 해서 다른 사건에 대해 아무런 정보도 얻을 수 없음 |
| 기댓값(Expected value) | "평균" | 모든 결과의 확률 가중합. 손실 함수도 일종의 기댓값임 |
| 분산(Variance) | "얼마나 퍼져 있는지" | 평균으로부터의 편차 제곱의 기댓값. 높은 분산 = 노이즈가 많고 불안정한 추정치 |
| 정규 분포(Normal distribution) | "종형 곡선" | $f(x) = (1/\sqrt{2\pi\sigma^2}) \cdot \exp(-(x-\mu)^2/(2\sigma^2))$. 중심 극한 정리(CLT)로 인해 도처에서 나타남 |
| 중심 극한 정리(Central Limit Theorem) | "평균은 정규 분포가 됨" | 분산이 유한한 임의의 분포에서 추출한 다수의 독립 표본에 대해, 중심화 및 척도화된 평균은 정규 분포로 수렴함 |
| 결합 분포(Joint distribution) | "두 변수가 함께 있는 것" | $P(X, Y)$은 $X$과 $Y$ 결과의 모든 조합에 대한 확률을 기술함 |
| 주변 분포(Marginal distribution) | "다른 변수를 합산하여 제거함" | $P(X) = \sum_y P(X, Y)$. 결합 분포로부터 한 변수의 분포를 복원함 |
| 로그 확률(Log probability) | "확률의 로그" | $\log P(x)$. 곱셈을 덧셈으로 변환하여 긴 시퀀스에서 수치적 언더플로(underflow)를 방지함 |
| Softmax | "점수를 확률로 변환함" | $\text{softmax}(z_i) = \exp(z_i) / \sum_j \exp(z_j)$. 실수 값 로짓을 유효한 확률 분포로 매핑함 |
| 교차 엔트로피(Cross-entropy) | "손실 함수" | $-\sum p_{\text{true}} \cdot \log(p_{\text{predicted}})$. 두 분포가 얼마나 다른지 측정함. 낮을수록 좋음 |
| 로짓(Logits) | "모델의 미가공 출력" | Softmax 이전의 정규화되지 않은 점수. 로지스틱 함수(logistic function)에서 유래함 |
| 샘플링(Sampling) | "무작위 값 추출" | 확률 분포에 따라 값을 생성함. 모델이 출력을 생성하는 방식 |

## 더 읽을거리

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 평균이 정규 분포를 이루는 이유에 대한 시각적 증명
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 여기서 다룬 내용과 그 이상을 다루는 간결한 참고 자료
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 수치적 안정성이 중요한 이유와 이를 달성하는 방법
