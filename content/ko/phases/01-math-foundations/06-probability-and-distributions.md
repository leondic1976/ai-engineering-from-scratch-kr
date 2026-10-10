---
title_en: "Probability and Distributions"
source: "phases/01-math-foundations/06-probability-and-distributions"
source_sha: "6af905cb9c5318be"
model: "gemini-3.8-flash"
translated_at: "2026-10-10"
---
# 확률과 분포

> 확률은 AI가 불확실성을 표현하기 위해 사용하는 언어입니다.

**유형:** Learn (이론)
**Language:** Python
**선수 레슨:** Phase 1, Lessons 01-04
**소요 시간:** ~75분

## 학습 목표

- 베르누이, 카테고리컬, 포아송, 균등, 정규 분포의 PMF와 PDF를 밑바닥부터 직접 구현합니다.
- 기댓값과 분산을 계산하고, 중심 극한 정리를 사용해 가우스 분포가 지배적인 이유를 설명합니다.
- 수치적 안정성 트릭(최댓값 logit 빼기)을 적용한 softmax 및 log-softmax 함수를 작성합니다.
- logit으로부터 교차 엔트로피 손실(cross-entropy loss)을 계산하고 이를 음의 로그 우도(negative log-likelihood)와 연결 짓습니다.

## 문제 의식

분류기는 `[0.03, 0.91, 0.06]`를 출력합니다. 언어 모델은 50,000개의 후보 중에서 다음 단어를 선택합니다. 확산 모델(diffusion model)은 학습된 분포로부터 샘플링하여 이미지를 생성합니다. 이 모든 것이 실제로 동작하는 확률입니다.

모델이 만드는 모든 예측은 확률 분포입니다. 모든 손실 함수는 예측된 분포가 실제 분포와 얼마나 멀리 떨어져 있는지를 측정합니다. 모든 학습 단계는 하나의 분포를 다른 분포와 더 유사하게 만들기 위해 파라미터를 조정합니다. 확률 없이는 단 하나의 머신러닝 논문도 읽을 수 없고, 단 하나의 모델도 디버깅할 수 없으며, 학습 손실이 NaN이 되는 이유도 이해할 수 없습니다.

## 핵심 개념

### 사건, 표본 공간, 그리고 확률

표본 공간 S는 가능한 모든 결과의 집합입니다. 사건은 표본 공간의 부분집합입니다. 확률은 사건을 0과 1 사이의 숫자로 매핑합니다.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

세 가지 공리가 확률의 모든 것을 정의합니다:
1. 임의의 사건 A에 대해 P(A) >= 0
2. P(S) = 1 (항상 어떤 사건은 일어남)
3. A와 B가 동시에 일어날 수 없을 때 P(A or B) = P(A) + P(B)

그 밖의 모든 것(베이즈 정리, 기댓값, 분포)은 이 세 가지 규칙으로부터 도출됩니다.

### 조건부 확률과 독립

P(A|B)는 B가 일어났을 때 A가 일어날 확률입니다.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

두 사건 중 하나를 안다고 해서 다른 하나에 대해 아무것도 알 수 없을 때, 두 사건은 독립입니다:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

동전 던지기는 독립입니다. 비복원 추출로 카드를 뽑는 것은 독립이 아닙니다.

### 확률 질량 함수 vs 확률 밀도 함수

이산 확률 변수는 확률 질량 함수(PMF, probability mass function)를 가집니다. 각 결과는 직접 읽어낼 수 있는 구체적인 확률을 가집니다.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

연속 확률 변수는 확률 밀도 함수(PDF, probability density function)를 가집니다. 단일 지점에서의 밀도는 확률이 아닙니다. 확률은 특정 구간에 걸쳐 밀도를 적분함으로써 얻어집니다.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

이 차이는 ML에서 매우 중요합니다. 분류 출력은 PMF(이산적 선택)입니다. VAE 잠재 공간(latent space)은 PDF(연속적)를 사용합니다.

### 자주 쓰이는 분포

**베르누이(Bernoulli):** 1회 시행, 2가지 결과. 이진 분류를 모델링합니다.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**카테고리컬(Categorical):** 1회 시행, k가지 결과. 다중 클래스 분류(softmax 출력)를 모델링합니다.

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**균등(Uniform):** 모든 결과가 나타날 확률이 동일함. 무작위 초기화에 사용됩니다.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**정규(Normal, Gaussian):** 종형 곡선. 평균(mu)과 분산(sigma^2)으로 모수화됩니다.

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**포아송(Poisson):** 고정된 구간에서 발생하는 희귀 사건의 횟수. 사건 발생률을 모델링합니다.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 기댓값과 분산

기댓값은 결과의 가중 평균입니다.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

분산은 평균 주변으로 퍼진 정도를 측정합니다.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

ML에서 기댓값은 손실 함수(데이터 분포에 대한 평균 손실)로 나타납니다. 분산은 모델의 안정성에 대해 알려줍니다. 그래디언트의 분산이 크다는 것은 학습 과정에 노이즈가 많다는 것을 의미합니다.

### 결합 분포와 주변 분포

결합 분포 P(X, Y)는 두 확률 변수를 함께 설명합니다.

결합 PMF 예시 (X = 날씨, Y = 우산):

| | Y=0 (우산 없음) | Y=1 (우산 있음) | 주변 P(X) |
|---|---|---|---|
| X=0 (맑음) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (비) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **주변 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

주변 분포는 다른 변수를 합산하여 소거합니다:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

위 표의 행 합계와 열 합계가 바로 주변 확률입니다.

### 정규 분포가 도처에 나타나는 이유

중심 극한 정리(Central Limit Theorem): 독립인 다수의 확률 변수의 합(또는 평균)은 원래의 분포와 관계없이 정규 분포로 수렴합니다.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

이것이 다음과 같은 현상이 발생하는 이유입니다:
- 측정 오차는 대략 정규 분포를 따릅니다(독립적인 수많은 미세 요인 때문).
- 신경망의 가중치 초기화에 정규 분포가 사용됩니다.
- SGD에서 그래디언트 노이즈는 대략 정규 분포를 따릅니다(다수의 샘플 그래디언트의 합이기 때문).
- 정규 분포는 주어진 평균과 분산에 대해 최대 엔트로피 분포입니다.

### 로그 확률

원시 확률(raw probability)은 수치적 문제를 야기합니다. 작은 확률을 여러 번 곱하면 빠르게 0으로 언더플로됩니다.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

로그 확률은 이 문제를 해결합니다. 곱셈이 덧셈으로 바뀝니다.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

규칙:
- log(a * b) = log(a) + log(b)
- 로그 확률은 항상 <= 0 (0 < P <= 1이므로)
- 더 음수일수록 = 일어날 가능성이 더 낮음
- 교차 엔트로피 손실은 정답 클래스의 음의 로그 확률입니다.

### 확률 분포로서의 Softmax

신경망은 가공되지 않은 점수(logit)를 출력합니다. Softmax는 이를 유효한 확률 분포로 변환합니다.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Softmax 트릭: 오버플로를 방지하기 위해 지수화하기 전에 최대 logit 값을 뺍니다.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax는 수치적 안정성을 위해 softmax와 log를 결합합니다. PyTorch는 교차 엔트로피 손실을 계산할 때 내부적으로 이를 사용합니다.

### 샘플링

샘플링은 분포에서 무작위 값을 추출하는 것을 의미합니다. ML에서의 예시는 다음과 같습니다:
- 드롭아웃(Dropout)은 0으로 만들 뉴런을 무작위로 샘플링합니다.
- 데이터 증강(Data augmentation)은 무작위 변환을 샘플링합니다.
- 언어 모델은 예측된 분포로부터 다음 토큰을 샘플링합니다.
- 확산 모델은 노이즈를 샘플링하고 점진적으로 노이즈를 제거합니다.

임의의 분포로부터 샘플링하려면 역변환 샘플링(inverse transform sampling), 기각 샘플링(rejection sampling), 또는 재매개변수화 트릭(reparameterization trick, VAE에서 사용)과 같은 기법이 필요합니다.

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

### 2단계: PMF와 PDF를 밑바닥부터 구현하기

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

## 사용해 보기

NumPy와 SciPy를 사용하면 위의 모든 작업이 한 줄로 끝납니다:

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

여러분은 이것들을 밑바닥부터 직접 구현했습니다. 이제 라이브러리 호출이 내부에서 무엇을 하는지 알 수 있습니다.

## 연습 문제

1. 지수 분포에 대한 역변환 샘플링을 구현하세요. 10,000개의 값을 샘플링하고 히스토그램을 실제 PDF와 비교하여 검증하세요.

2. 조작된 두 개의 주사위에 대한 결합 분포 표를 작성하세요. 주변 분포를 계산하고 두 주사위가 독립인지 확인하세요.

3. 정답 클래스가 인덱스 3일 때, logit `[2.0, 0.5, -1.0, 3.0, 0.1]`를 출력하는 5개 클래스 분류기의 교차 엔트로피 손실을 계산하세요. 그런 다음 PyTorch의 `nn.CrossEntropyLoss`로 정답을 검증하세요.

4. 로그 확률 리스트를 받아 가장 가능성이 높은 시퀀스, 전체 로그 확률, 그리고 이에 상응하는 원시 확률을 반환하는 함수를 작성하세요. 각 단어의 확률이 0.01인 50개 단어로 이루어진 문장으로 테스트하세요.
## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| 표본 공간(Sample space) | "모든 가능성" | 어떤 실험에서 발생 가능한 모든 결과의 집합 S |
| 확률 질량 함수(PMF) | "확률 함수" | 각 이산 결과의 정확한 확률을 제공하며, 합이 1이 되는 함수 |
| 확률 밀도 함수(PDF) | "확률 곡선" | 연속 변수에 대한 밀도 함수. 특정 구간에 걸쳐 적분하여 확률을 구함 |
| 조건부 확률(Conditional probability) | "무언가가 주어졌을 때의 확률" | P(A\|B) = P(A and B) / P(B). 베이즈적 사고와 베이즈 정리(Bayes' theorem)의 기초 |
| 독립(Independence) | "서로 영향을 주지 않음" | P(A and B) = P(A) * P(B). 한 사건을 알아도 다른 사건에 대해 아무런 정보를 얻을 수 없음 |
| 기댓값(Expected value) | "평균" | 모든 결과의 확률 가중합. 손실 함수는 기댓값에 해당함 |
| 분산(Variance) | "얼마나 퍼져 있는지" | 평균으로부터의 편차 제곱에 대한 기댓값. 높은 분산 = 노이즈가 많고 불안정한 추정치 |
| 정규 분포(Normal distribution) | "종형 곡선" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). 중심 극한 정리(CLT)로 인해 도처에 등장함 |
| 중심 극한 정리(Central Limit Theorem) | "평균은 정규 분포가 됨" | 표본의 원래 분포와 상관없이 독립적인 다수의 표본 평균은 정규 분포로 수렴함 |
| 결합 분포(Joint distribution) | "두 변수를 함께 고려한 것" | P(X, Y)는 X와 Y 결과의 모든 조합에 대한 확률을 나타냄 |
| 주변 분포(Marginal distribution) | "다른 변수를 합산하여 제거함" | P(X) = sum_y P(X, Y). 결합 분포로부터 한 변수의 분포를 복원함 |
| 로그 확률(Log probability) | "확률의 로그값" | log P(x). 곱셈을 덧셈으로 변환하여 긴 시퀀스에서 수치적 언더플로우(numerical underflow)를 방지함 |
| 소프트맥스(Softmax) | "점수를 확률로 변환함" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). 실수형 로짓을 유효한 확률 분포로 매핑함 |
| 크로스 엔트로피(Cross-entropy) | "손실 함수" | -sum(p_true * log(p_predicted)). 두 분포가 얼마나 다른지 측정함. 낮을수록 우수함 |
| 로짓(Logits) | "가공되지 않은 모델 출력값" | softmax 적용 전의 정규화되지 않은 점수. 로지스틱 함수(logistic function)에서 유래함 |
| 샘플링(Sampling) | "무작위 값을 추출함" | 확률 분포에 따라 값을 생성함. 모델이 출력을 생성하는 방식 |

## 추가 참고 자료

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 평균이 왜 정규 분포를 따르게 되는지에 대한 시각적 증명
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 여기에 설명된 모든 내용과 그 이상을 다루는 간결한 참고 자료
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 수치적 안정성이 왜 중요하며 이를 어떻게 달성하는지에 대한 설명
