---
title_en: "Sampling Methods"
source: "phases/01-math-foundations/16-sampling-methods"
source_sha: "5d04a2aa6d6b06c1"
model: "gemini-3.8-flash"
translated_at: "2026-10-24"
---
# 샘플링 기법

> 샘플링은 AI가 가능성의 공간을 탐색하는 방식입니다.

**유형:** Build (실습)
**Language:** Python
**선수 레슨:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**소요 시간:** ~120분

## 학습 목표

- 균등 난수만을 사용하여 역누적분포함수(inverse CDF), 기각(rejection), 중요도(importance) 샘플링을 밑바닥부터 구현합니다.
- 언어 모델 토큰 생성을 위한 온도(temperature), top-k, top-p(nucleus) 샘플링을 구축합니다.
- 재매개변수화 트릭(reparameterization trick)과 이것이 VAE에서 샘플링을 통한 역전파(backpropagation)를 가능하게 하는 이유를 설명합니다.
- 정규화되지 않은 목표 분포에서 샘플링하기 위해 메트로폴리스-헤이스팅스(Metropolis-Hastings) MCMC를 실행합니다.

## 문제 정의

언어 모델이 프롬프트 처리를 마치고 50,000개의 로짓(logit)으로 이루어진 벡터를 생성합니다. 어휘집(vocabulary)의 모든 토큰마다 하나씩 할당된 값입니다. 이제 모델은 토큰 하나를 선택해야 합니다. 어떻게 선택해야 할까요?

항상 확률이 가장 높은 토큰만 선택한다면, 모든 응답이 동일해집니다. 결정론적이고, 지루합니다. 반대로 균등하게 무작위로 선택한다면 횡설수설하는 출력이 나옵니다. 해답은 이 양극단 사이 어딘가에 존재하며, 그 지점을 제어하는 것이 바로 샘플링입니다.

샘플링은 텍스트 생성에만 국한되지 않습니다. 강화학습(reinforcement learning)은 궤적(trajectory)을 샘플링하여 정책 그래디언트(policy gradient)를 추정합니다. VAE는 학습된 분포에서 샘플링하고 무작위성을 거쳐 역전파함으로써 잠재 표현(latent representation)을 학습합니다. 확산 모델(diffusion model)은 노이즈를 샘플링하고 반복적으로 노이즈를 제거하여 이미지를 생성합니다. 몬테카를로(Monte Carlo) 기법은 닫힌 형식의 해(closed-form solution)가 없는 적분을 추정합니다. MCMC 알고리즘은 일일이 나열하기 불가능한 고차원 사후 분포(posterior distribution)를 탐색합니다.

모든 생성형 AI 시스템은 샘플링 시스템입니다. 샘플링 전략이 출력의 품질, 다양성, 제어 가능성을 결정합니다. 이번 강의에서는 균등 난수에서 시작하여 최신 LLM과 생성 모델을 구동하는 기법에 이르기까지, 모든 주요 샘플링 방법을 밑바닥부터 직접 구축합니다.

## 핵심 개념

### 샘플링이 중요한 이유

샘플링은 AI와 머신러닝 전반에 걸쳐 네 가지 근본적인 역할을 수행합니다.

**생성(Generation).** 언어 모델, 확산 모델, GAN은 모두 샘플링을 통해 출력을 생성합니다. 샘플링 알고리즘은 창의성, 일관성, 다양성을 직접 제어합니다. 온도(temperature), top-k, nucleus 샘플링은 엔지니어가 매일 조절하는 조작 장치입니다.

**학습(Training).** 확률적 경사 하강법(SGD)은 미니배치를 샘플링합니다. 드롭아웃(Dropout)은 비활성화할 뉴런을 샘플링합니다. 데이터 증강(Data augmentation)은 무작위 변환을 샘플링합니다. 중요도 샘플링(Importance sampling)은 강화학습(PPO, TRPO)에서 그래디언트 분산을 줄이기 위해 샘플 가중치를 재조정합니다.

**추정(Estimation).** 머신러닝의 수많은 값에는 닫힌 형식의 해가 없습니다. 데이터 분포에 대한 기대 손실, 에너지 기반 모델의 분배 함수(partition function), 베이즈 추론의 증거(evidence) 등이 이에 해당합니다. 몬테카를로 추정은 샘플들의 평균을 구하여 이 모든 값을 근사합니다.

**탐색(Exploration).** MCMC 알고리즘은 베이즈 추론에서 사후 분포를 탐색합니다. 진화 전략(Evolutionary strategies)은 파라미터 섭동(perturbation)을 샘플링합니다. 톰슨 샘플링(Thompson sampling)은 밴딧(bandit) 문제에서 탐색과 활용(exploration and exploitation)의 균형을 맞춥니다.

핵심 과제: 우리는 단순한 분포(균등 분포, 정규 분포)에서만 직접 샘플링할 수 있습니다. 그 외의 모든 분포에 대해서는 단순한 샘플을 목표 분포의 샘플로 변환하는 방법이 필요합니다.

### 균등 무작위 샘플링

모든 샘플링 기법은 여기서 시작합니다. 균등 난수 생성기는 길이가 같은 모든 하위 구간이 동일한 확률을 갖는 $[0, 1)$ 범위의 값을 생성합니다.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

$n$개의 항목으로 구성된 이산 집합에서 균등하게 샘플링하려면, $U$를 생성하고 $\lfloor n U \rfloor$를 반환합니다. 연속 범위 $[a, b]$에서 샘플링하려면 $a + (b - a) U$를 계산합니다.

핵심 통찰: 단 하나의 균등 난수만으로도 임의의 분포에서 샘플 하나를 생성하기에 정확히 충분한 양의 무작위성을 얻을 수 있습니다. 관건은 올바른 변환을 찾는 것입니다.

### 역누적분포함수법 (역변환 샘플링)

누적분포함수(CDF)는 값을 확률로 매핑합니다.

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

역누적분포함수(inverse CDF)는 확률을 다시 값으로 매핑합니다. 만약 $U \sim \text{Uniform}(0, 1)$라면, $X = F^{-1}(U)$는 목표 분포를 따릅니다.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**지수 분포 예시:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

이 방법은 $F^{-1}$를 닫힌 형식으로 표현할 수 있을 때 완벽하게 작동합니다. 정규 분포의 경우 역 CDF의 닫힌 형식이 존재하지 않으므로, 다른 방법(Box-Muller 또는 수치적 근사)을 사용합니다.

**이산 버전:** 이산 분포의 경우 누적합(cumulative sum)으로 CDF를 구성하고, $U$를 생성한 뒤 누적합이 $U$를 초과하는 첫 번째 인덱스를 찾습니다. 이것이 Lesson 06의 `sample_categorical`가 작동하는 방식입니다.

### 기각 샘플링

CDF를 역전시킬 수 없지만 상수 배수까지 목표 PDF를 계산할 수 있는 경우, 기각 샘플링(rejection sampling)이 효과적입니다.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = Z/M, where Z is the integral of p(x)
                (Z = 1, so the rate is 1/M, when p is normalized)
```

상한선 $M$가 더 정밀할수록 수락률(acceptance rate)이 높아집니다. 저차원(1~3차원)에서는 기각 샘플링이 잘 작동합니다. 고차원에서는 제안 영역(proposal volume)의 대부분이 기각되므로 수락률이 지수적으로 떨어집니다. 이것이 기각 샘플링에서의 차원의 저주(curse of dimensionality)입니다.

**예시: 절단된 정규 분포(truncated normal) 샘플링.** 절단된 범위에 걸친 균등 제안 분포를 사용합니다. 포락선(envelope) $M$는 해당 범위에서 정규 PDF의 최댓값입니다.

**예시: 반원 샘플링.** 경계 직사각형 내부에서 균등하게 제안합니다. 점이 반원 내부에 위치하면 수락합니다. 이것이 몬테카를로 기법이 $\pi$를 계산하는 방식입니다. 수락률은 면적 비 $\pi/4$와 같습니다.

### 중요도 샘플링

때로는 목표 분포 $p(x)$로부터 샘플이 필요하지 않을 수도 있습니다. $p(x)$ 하에서의 기댓값을 추정해야 하는데, 다른 분포 $q(x)$의 샘플만 가지고 있는 경우입니다.

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

이는 강화학습에서 매우 중요합니다. PPO(Proximal Policy Optimization)에서는 이전 정책 $\pi_{\text{old}}$ 하에서 궤적을 수집하지만 새로운 정책 $\pi_{\text{new}}$를 최적화하고자 합니다. 이때 중요도 가중치(importance weight)는 $\pi_{\text{new}}(a \mid s) / \pi_{\text{old}}(a \mid s)$입니다. PPO는 새로운 정책이 이전 정책에서 너무 멀리 벗어나지 않도록 이 가중치를 클리핑(clipping)합니다.

중요도 샘플링 추정량의 분산은 $q$가 $p$와 얼마나 유사한지에 따라 달라집니다. $q$가 $p$와 크게 다르면, 소수의 샘플이 막대한 가중치를 부여받아 추정값을 지배하게 됩니다. 자가 정규화 중요도 샘플링(Self-normalized importance sampling)은 가중치의 합으로 나누어 이 문제를 완화합니다.

```math
\mathbb{E}_p[f(x)] \approx \frac{\sum_i w_i \, f(x_i)}{\sum_i w_i}
```

### 몬테카를로 추정

몬테카를로 추정(Monte Carlo estimation)은 무작위 샘플들의 평균을 내어 적분을 근사합니다. 대수의 법칙(law of large numbers)에 의해 수렴이 보장됩니다.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

오차율은 차원과 무관합니다. 격자 기반 적분(grid-based integration)이 불가능한 고차원에서 몬테카를로 방법이 지배적으로 사용되는 이유가 바로 여기에 있습니다.

**원주율 추정:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**기댓값 추정:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```
### 마르코프 연쇄 몬테카를로(Markov Chain Monte Carlo, MCMC): 메트로폴리스-헤이스팅스

MCMC는 정적 분포(stationary distribution)가 목표 분포 $p(x)$가 되는 마르코프 연쇄를 구성합니다. 충분한 단계를 거치면, 이 연쇄에서 추출한 샘플은 (근사적으로) $p(x)$로부터 얻은 샘플이 됩니다.

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

대칭 제안(symmetric proposal) 분포 $(q(x' \mid x) = q(x \mid x'))$의 경우, 비율은 $p(x')/p(x)$로 단순화됩니다. 이것이 원래의 메트로폴리스 알고리즘입니다.

**동작 원리.** 채택 규칙은 상세 균형(detailed balance)을 보장합니다. 즉, $x$에 머물며 $x'$로 이동할 확률은 $x'$에 머물며 $x$로 이동할 확률과 같습니다. 상세 균형은 $p(x)$가 연쇄의 정적 분포임을 의미합니다.

**실무적 고려사항:**
- 번인(Burn-in): 연쇄가 평형 상태에 도달하기 전의 초기 샘플들을 폐기합니다.
- 시닝(Thinning): 자기상관(autocorrelation)을 줄이기 위해 매 $k$번째 샘플만 유지합니다.
- 제안 스케일: 너무 작으면 연쇄가 느리게 이동하고(높은 채택률, 느린 탐색), 너무 크면 대부분의 제안이 거부됩니다(낮은 채택률, 제자리 정체).
- 고차원에서 가우시안 제안의 최적 채택률은 약 0.234입니다.

### 깁스 샘플링(Gibbs Sampling)

깁스 샘플링은 다변량 분포를 위한 MCMC의 특수한 경우입니다. 모든 차원에서 한 번에 이동을 제안하는 대신, 조건부 분포로부터 변수를 한 번에 하나씩 업데이트합니다.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

깁스 샘플링은 각 조건부 분포 $p(x_i \mid x_{-i})$에서 샘플링할 수 있어야 합니다. 이는 다음과 같은 많은 모델에서 간단히 적용됩니다.
- 베이지안 네트워크: 조건부 분포가 그래프 구조로부터 도출됩니다.
- 가우시안 혼합 모델: 조건부 분포가 가우시안입니다.
- 이징(Ising) 모델: 각 스핀의 조건부 분포가 이웃에만 의존합니다.

정확한 조건부 분포에서 샘플링하면 상세 균형이 자동으로 충족되므로 채택률은 항상 1입니다(모든 제안이 채택됨).

**한계점.** 변수 간의 상관관계가 높을 때, 변수를 한 번에 하나씩 업데이트하면 분포를 가로지르는 큰 대각선 방향 이동을 할 수 없으므로 깁스 샘플링의 혼합(mixing) 속도가 느려집니다.

### 온도 샘플링(Temperature Sampling, LLM에서 사용)

언어 모델은 어휘집(vocabulary)의 각 토큰에 대해 로짓(logit) $z_1, \ldots, z_V$를 출력합니다. 소프트맥스(softmax)는 이를 확률로 변환합니다. 온도는 softmax 전에 로짓의 스케일을 재조정합니다.

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**동작 원리.** 로짓을 $T < 1$로 나누면 로짓 간의 차이가 증폭됩니다. 예를 들어 $z_1 = 2$ 및 $z_2 = 1$일 때, $T = 0.5$로 나누면 $z_1/T = 4$와 $z_2/T = 2$가 되어 격차가 더 커집니다. 소프트맥스 후에는 가장 높은 로짓을 가진 토큰이 훨씬 더 큰 비중을 차지하게 됩니다.

**실제 적용:**
- $T = 0.0$: 탐욕적 디코딩(greedy decoding), 사실 기반 Q&A에 최적
- $T = 0.3\text{-}0.7$: 약간의 창의성, 코드 생성에 적합
- $T = 0.7\text{-}1.0$: 균형 잡힘, 일반적인 대화에 적합
- $T = 1.0\text{-}1.5$: 창의적 글쓰기, 브레인스토밍
- $T > 1.5$: 점점 더 무작위적이며, 거의 유용하지 않음

온도는 어떤 토큰이 생성 가능한지를 바꾸지 않으며, 각 토큰에 할당되는 확률 질량(probability mass)을 변경합니다.

### Top-k 샘플링(Top-k Sampling)

Top-k 샘플링은 후보군을 확률이 가장 높은 $k$개의 토큰으로 제한한 다음, 재정규화(renormalization)를 거쳐 해당 제한된 집합에서 샘플링합니다.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k는 모델이 어휘 분포의 롱테일(long tail)에 존재하는 극히 희박한 토큰(오타, 무의미한 단어)을 선택하지 않도록 방지합니다. 문제는 문맥과 무관하게 $k$가 고정되어 있다는 점입니다. 모델이 확신할 때(한 토큰의 확률이 95%인 경우)에도 $k = 40$는 여전히 39개의 대안을 허용합니다. 반대로 모델이 불확실할 때(확률이 1000개 토큰에 분산된 경우)에는 $k = 40$가 그럴듯한 선택지들을 잘라냅니다.

### Top-p (뉴클리어스) 샘플링(Top-p (Nucleus) Sampling)

Top-p 샘플링은 후보군의 크기를 동적으로 조정합니다. 고정된 수의 토큰을 유지하는 대신, 누적 확률이 $p$를 초과하는 가장 작은 토큰 집합을 유지합니다.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

모델이 확신할 때 뉴클리어스 샘플링은 소수의 토큰(약 2~3개)만 유지합니다. 모델이 불확실할 때는 많은 토큰(약 200개)을 유지합니다. 이러한 적응적 특성 덕분에 뉴클리어스 샘플링은 일반적으로 Top-k보다 더 나은 텍스트를 생성합니다.

**자주 사용되는 조합:**
- Temperature 0.7 + top-p 0.9: 범용으로 적합한 설정
- Temperature 0.0 (탐욕적): 결정론적(deterministic) 작업에 최적
- Temperature 1.0 + top-k 50: Fan et al. (2018) 원본 논문 설정

Top-k와 Top-p는 결합하여 사용할 수 있습니다. 먼저 Top-k를 적용한 다음, 남은 집합에 Top-p를 적용합니다.

### 재매개변수화 트릭(Reparameterization Trick, VAE에서 사용)

변이형 오토인코더(Variational Autoencoder, VAE)는 입력을 잠재 공간(latent space)의 분포로 인코딩하고, 해당 분포에서 샘플링한 후, 그 샘플을 다시 디코딩하는 방식으로 학습합니다. 문제는 샘플링 연산을 통해서는 역전파(backpropagation)를 수행할 수 없다는 점입니다.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

재매개변수화 트릭은 매개변수로부터 무작위성(randomness)을 분리합니다.

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

이는 $\mathcal{N}(\mu, \sigma^2)$가 $\mu + \sigma \cdot \mathcal{N}(0, 1)$와 동일한 분포를 가지기 때문에 동작합니다. 핵심 통찰은 무작위성을 매개변수가 없는 소스($\epsilon$)로 옮긴 뒤, 샘플을 매개변수에 대한 미분 가능한 변환으로 표현하는 것입니다.

**VAE 학습 루프에서의 동작:**
1. 인코더가 각 입력에 대해 $\mu$ 및 $\log(\sigma^2)$를 출력합니다.
2. $\epsilon \sim \mathcal{N}(0, 1)$를 샘플링합니다.
3. $z = \mu + \sigma \epsilon$를 계산합니다.
4. 입력을 재구성하기 위해 $z$를 디코딩합니다.
5. 단계 4, 3, 2, 1을 거쳐 역전파합니다(3단계가 미분 가능하므로 가능함).

재매개변수화 트릭이 없다면 VAE는 표준 역전파로 학습할 수 없습니다. 이 하나의 통찰이 VAE를 실용적으로 만들었습니다.

### 검벨-소프트맥스(Gumbel-Softmax, 미분 가능한 범주형 샘플링)

재매개변수화 트릭은 연속 분포(가우시안)에서 동작합니다. 이산 범주형 분포의 경우에는 다른 접근 방식이 필요합니다. Gumbel-Softmax는 범주형 샘플링에 대한 미분 가능한 근사를 제공합니다.

**검벨-맥스 트릭(Gumbel-Max trick, 미분 불가능):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**검벨-소프트맥스(Gumbel-Softmax, 미분 가능한 근사):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax는 이산 샘플의 연속적 완화(continuous relaxation)를 생성합니다. 출력은 하드 원-핫(hard one-hot) 대신 확률 벡터(소프트 원-핫)가 됩니다. 그래디언트는 소프트맥스를 통해 흐릅니다. 학습 중 순전파(forward pass) 시에는 "스트레이트-쓰루(straight-through)" 추정기를 사용할 수 있습니다. 즉, 순전파에는 하드 argmax를 사용하고 역전파에는 소프트 Gumbel-Softmax 그래디언트를 사용합니다.

**응용 분야:**
- VAE에서의 이산 잠재 변수
- 신경망 구조 탐색(Neural Architecture Search, 이산 연산 선택)
- 하드 어텐션(hard attention) 메커니즘
- 이산적 행동을 다루는 강화학습(reinforcement learning)

### 층화 샘플링(Stratified Sampling)

표준 몬테카를로 샘플링은 우연에 의해 샘플 공간에 빈틈을 남길 수 있습니다. 층화 샘플링은 공간을 층(strata)으로 분할하고 각 층에서 샘플링함으로써 균일한 커버리지를 강제합니다.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

층화 샘플링은 표준 몬테카를로에 비해 분산이 항상 낮거나 같습니다.

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**응용 분야:**
- 수치 적분(준 몬테카를로, quasi-Monte Carlo)
- 학습 데이터 분할(각 폴드에서 클래스 균형 보장)
- 층화를 적용한 중요도 샘플링(두 기법의 결합)
- NeRF(Neural Radiance Fields)에서의 카메라 광선을 따른 층화 샘플링
### 디퓨전 모델과의 연결점

디퓨전 모델(diffusion model)은 샘플링 과정을 통해 이미지를 생성합니다. 순방향 과정(forward process)에서는 이미지가 완전한 노이즈가 될 때까지 $T$단계에 걸쳐 가우시안 노이즈(Gaussian noise)를 추가합니다. 역방향 과정(reverse process)에서는 노이즈를 제거하는 법을 학습하여 원래 이미지를 단계별로 복원합니다.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

이번 단원의 방법들과의 연결점은 다음과 같습니다:
- 각 노이즈 제거(denoising) 단계는 재매개변수화 트릭(reparameterization trick)을 사용합니다(노이즈를 샘플링하고 결정론적 변환을 적용).
- 노이즈 스케줄 $\lbrace \alpha_t \rbrace$은 일종의 온도 어닐링(temperature annealing)을 제어합니다.
- 학습 시에는 몬테카를로 추정(Monte Carlo estimation)을 사용하여 ELBO(증거 하한, evidence lower bound)를 근사합니다.
- 디퓨전 모델의 조상 샘플링(ancestral sampling)은 마르코프 체인(각 단계는 현재 상태에만 의존)입니다.

전체 이미지 생성 과정은 반복적 샘플링(iterative sampling)입니다. 즉, 노이즈에서 시작하여 각 단계마다 학습된 노이즈 제거 모델을 조건으로 하여 노이즈가 약간 덜한 버전을 샘플링합니다.

```figure
monte-carlo-pi
```

## 직접 구현해보기

### 1단계: 균등 샘플링 및 역 누적 분포 함수(역 CDF) 샘플링

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

10,000개의 지수 분포(exponential distribution) 샘플을 생성하고 평균이 $1/\lambda$인지 확인합니다.

### 2단계: 기각 샘플링

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

기각 샘플링(rejection sampling)을 사용하여 절단된 정규 분포(truncated normal distribution)에서 샘플을 추출합니다. 샘플의 히스토그램을 그려 분포의 형태를 확인합니다.

### 3단계: 중요도 샘플링

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

균등 제안 분포를 사용하여 정규 분포 하에서의 $\mathbb{E}[X^2]$를 추정합니다. 알려진 정답($\mu^2 + \sigma^2$)과 비교합니다.

### 4단계: 파이(pi)의 몬테카를로 추정

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 5단계: 메트로폴리스-헤이스팅스 MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

이봉 분포(bimodal distribution, 두 가우시안의 혼합)에서 샘플링합니다. 체인의 궤적을 시각화합니다.

### 6단계: 깁스 샘플링

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 7단계: 온도 샘플링

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

토큰 로짓 집합에 대해 온도가 출력 분포를 어떻게 변화시키는지 살펴봅니다.

### 8단계: Top-k 및 Top-p 샘플링

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 9단계: 재매개변수화 트릭

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

재매개변수화된 샘플을 통해서는 그래디언트가 흐르지만, 직접 샘플링을 통해서는 흐르지 않음을 입증합니다.

### 10단계: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

온도를 낮출수록 출력이 원-핫(one-hot) 벡터에 어떻게 가까워지는지 살펴봅니다.

모든 시각화가 포함된 전체 구현은 `code/sampling.py`에 있습니다.

## 라이브러리 활용하기

NumPy와 SciPy를 사용한 프로덕션용 버전은 다음과 같습니다:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

대규모 MCMC의 경우 전용 라이브러리를 사용합니다:
- PyMC: NUTS(적응형 HMC)를 지원하는 완전한 베이지안 모델링
- emcee: 앙상블 MCMC 샘플러
- NumPyro/JAX: GPU 가속 MCMC

지금까지 이를 바닥부터 직접 구현해 보았습니다. 이제 라이브러리 호출 시 내부에서 어떤 작업이 수행되는지 이해할 수 있습니다.

## 연습 문제

1. 코시 분포(Cauchy distribution)에 대한 역 CDF 샘플링을 구현합니다. 누적 분포 함수(CDF)는 $F(x) = 0.5 + \arctan(x)/\pi$입니다. 10,000개의 샘플을 생성하고 실제 확률 밀도 함수(PDF)와 히스토그램을 함께 플롯합니다. 두터운 꼬리(heavy tails, 중심에서 멀리 떨어진 극단값)에 주목합니다.

2. Uniform(0, 1) 제안 분포를 사용하는 기각 샘플링으로 Beta(2, 5) 분포에서 샘플을 생성합니다. 수락된 샘플을 실제 베타 PDF와 비교하여 플롯합니다. 이론적인 수락률은 얼마입니까?

3. 1,000개, 10,000개, 100,000개의 샘플을 사용한 몬테카를로 방법으로 0부터 $\pi$까지 $\sin(x)$의 적분을 추정합니다. 각 단계에서의 오차를 비교합니다. 오차가 $O(1/\sqrt{N})$에 비례하여 줄어드는지 확인합니다.

4. $\exp(-(x^2 y^2 + x^2 + y^2 - 8x - 8y) / 2)$에 비례하는 2차원 분포 $p(x, y)$에서 샘플링하는 메트로폴리스-헤이스팅스 알고리즘을 구현합니다. 샘플과 체인 궤적을 플롯합니다. 제안 분포의 표준편차를 다양하게 변경하며 실험해 봅니다.

5. 완전한 텍스트 생성 데모를 구축합니다. 로짓이 주어진 10개 단어의 어휘 사전에서 (a) 탐욕적 디코딩(greedy), (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9를 사용하여 20개 토큰으로 구성된 시퀀스를 생성합니다. 5회 실행에 걸쳐 출력의 다양성을 비교합니다.

### 힌트

**연습 문제 1.** $u = F(x)$를 $x$에 대해 풀면 $x = \tan(\pi(u - 0.5))$이 얻어집니다. 표본 중앙값(sample median)이 0에 가깝고 샘플의 약 6%가 $\lvert x \rvert > 10$을 만족하는지($1 - (2/\pi)\arctan(10)$) 확인합니다.

**연습 문제 2.** Beta(2, 5)의 밀도 함수는 $30x(1-x)^4$입니다. 최빈값(mode) $x = 1/5$에서의 최댓값을 구하고 그 값을 `rejection_sample`의 $M$로 사용합니다. 이론적 수락률은 $1/M$이므로, 실제 수락한 제안의 비율과 비교합니다.

**연습 문제 3.** 정확한 값은 2입니다. 각 $n$에 대해 `monte_carlo_integral(math.sin, 0, math.pi, n)`을 호출하고 여러 시드로 각 실행을 반복합니다. $n$이 10배 증가할 때 평균 오차는 약 $\sqrt{10} \approx 3.2$배 감소해야 합니다.

**연습 문제 4.** 대상 분포의 로그($-(x^2 y^2 + x^2 + y^2 - 8x - 8y)/2$)를 작성하여 `metropolis_hastings_2d`에 전달합니다. 각 제안 표준편차별로 수락률을 기록합니다. 값이 너무 작으면 거의 모든 스텝을 수락하지만 이동이 느리고, 값이 너무 크면 거의 모든 스텝을 기각합니다.

**연습 문제 5.** 탐욕적 디코딩은 각 단계에서 가장 큰 로짓의 인덱스를 선택합니다. 나머지 세 가지 방법에는 `temperature_sample`, `top_k_sample`, `top_p_sample`을 사용합니다. 탐욕적 디코딩은 5번의 실행 모두에서 동일한 시퀀스를 생성해야 하므로, 다양성을 측정하기 위해 각 방법별로 서로 다른 고유한 시퀀스의 수를 세어 봅니다.
## 주요 용어

| 용어 | 통칭 | 실제 의미 |
|------|----------------|----------------------|
| 샘플링(Sampling) | "무작위 값 뽑기" | 확률 분포에 따라 값을 생성하는 것. 모든 생성형 AI의 기반이 되는 메커니즘 |
| 균등 분포(Uniform distribution) | "모두 동일한 확률" | $[a, b]$의 모든 값이 동일한 확률 밀도 $1/(b-a)$를 가짐. 모든 샘플링 기법의 출발점 |
| 역누적분포함수(Inverse CDF) | "확률 변환" | $F^{-1}(U)$는 균등 샘플을 CDF가 알려진 임의의 분포에서 나온 샘플로 변환함. 정확하고 효율적임 |
| 기각 샘플링(Rejection sampling) | "제안 후 수락/기각" | 단순한 제안 분포에서 생성하고, 목표 분포와 제안 분포의 비율에 비례하는 확률로 수락함. 정확하지만 샘플 낭비가 발생함 |
| 중요도 샘플링(Importance sampling) | "샘플 가중치 재조정" | 각 샘플에 $p(x)/q(x)$ 가중치를 부여하여, $q(x)$의 샘플을 사용해 $p(x)$ 하에서의 기댓값을 추정함. 강화학습(RL) PPO의 핵심 |
| 몬테카를로(Monte Carlo) | "무작위 샘플 평균 내기" | 적분을 샘플 평균으로 근사함. 차원과 무관하게 오차는 $O(1/\sqrt{N})$임 |
| MCMC | "수렴하는 무작위 보행" | 정상 분포(stationary distribution)가 목표 분포가 되도록 마르코프 체인(Markov chain)을 구성함. Metropolis-Hastings가 대표적인 기본 알고리즘임 |
| Metropolis-Hastings | "오르막은 수락, 내리막은 때때로 수락" | 이동을 제안하고, 밀도 비율에 기반하여 수락함. 상세 균형 조건(detailed balance)이 목표 분포로의 수렴을 보장함 |
| 깁스 샘플링(Gibbs sampling) | "한 번에 하나의 변수씩" | 다른 변수들을 고정한 채 각 변수를 조건부 분포로부터 갱신함. 수락률이 100%임 |
| 온도(Temperature) | "확신도 조절 손잡이" | softmax를 적용하기 전에 로짓(logit)을 $T$로 나눔. $T<1$는 분포를 뾰족하게(더 확신을 가짐), $T>1$는 분포를 평평하게(더 다양해짐) 만듦 |
| Top-k 샘플링 | "상위 k개만 유지" | 확률이 가장 높은 $k$개 토큰을 제외한 나머지를 0으로 만들고, 재정규화한 후 샘플링함. 고정된 후보군 크기를 가짐 |
| 핵 샘플링(Nucleus sampling, top-p) | "확률이 높은 것들만 유지" | 누적 확률이 $p$를 넘는 최소한의 토큰 집합을 유지함. 동적인 후보군 크기를 가짐 |
| 재매개변수화 트릭(Reparameterization trick) | "무작위성을 외부로 이동" | $\epsilon \sim \mathcal{N}(0,1)$인 $z = \mu + \sigma \epsilon$ 형태로 작성함. 샘플링을 미분 가능하게 만들어 VAE 훈련에 필수적임 |
| Gumbel-Softmax | "부드러운 범주형 샘플링" | Gumbel 노이즈와 온도가 적용된 softmax를 사용하여 범주형 샘플링을 미분 가능하게 근사함 |
| 층화 샘플링(Stratified sampling) | "강제적인 표본 분할 추출" | 표본 공간을 여러 층(strata)으로 나누고 각 층에서 샘플을 추출함. 단순 몬테카를로보다 항상 분산이 낮음 |
| 번인(Burn-in) | "예열 기간" | 체인이 정상 분포에 도달하기 전에 폐기되는 초기 MCMC 샘플 구간 |
| 상세 균형 조건(Detailed balance) | "가역성 조건" | $p(x) T(x \to y) = p(y) T(y \to x)$. $p$가 마르코프 체인의 정상 분포가 되기 위한 충분조건 |
| 디퓨전 샘플링(Diffusion sampling) | "반복적 노이즈 제거" | 노이즈에서 시작하여 학습된 노이즈 제거(denoising) 단계를 거쳐 데이터를 생성함. 각 단계는 조건부 샘플링 연산임 |

## 더 읽을거리

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 기초에 대한 상세 튜토리얼
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 원작 논문
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 핵(top-p) 샘플링 논문
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 재매개변수화 트릭을 소개한 VAE 논문
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - 샘플링을 이미지 생성과 연결한 DDPM 논문
