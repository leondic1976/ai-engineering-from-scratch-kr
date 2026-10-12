---
title_en: "Norms and Distances"
source: "phases/01-math-foundations/14-norms-and-distances"
source_sha: "f0949b87124ae76e"
model: "gemini-3.8-flash"
translated_at: "2026-10-12"
---
# 노름과 거리 (Norms and Distances)

> 거리 함수는 "유사하다"는 것이 무엇을 의미하는지 정의합니다. 잘못 선택하면 다운스트림의 모든 것이 망가집니다.

**유형:** Build (실습)
**Language:** Python
**선수 레슨:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**소요 시간:** ~90분

## 학습 목표

- L1, L2, 코사인, 마할라노비스(Mahalanobis), 자카드(Jaccard), 편집 거리(edit distance) 함수를 밑바닥부터 구현합니다.
- 주어진 머신러닝 과제에 적합한 거리 측도를 선택하고, 다른 대안이 적합하지 않은 이유를 설명합니다.
- L1 및 L2 노름을 LASSO 및 Ridge 정규화(regularization)와 이들의 기하학적 제약 영역에 연결합니다.
- 동일한 데이터셋이라도 측도에 따라 서로 다른 최근접 이웃(nearest neighbors)이 도출됨을 확인합니다.

## 문제

두 개의 벡터가 있습니다. 단어 임베딩(word embedding)일 수도 있고, 사용자 프로필이나 픽셀 배열일 수도 있습니다. 여러분은 이 둘이 얼마나 가까운지 알아내야 합니다.

그 답은 어떤 거리 함수를 선택하느냐에 전적으로 달려 있습니다. 어떤 측도에서는 두 데이터 포인트가 가장 가까운 이웃일 수 있지만, 다른 측도에서는 서로 멀리 떨어져 있을 수 있습니다. KNN 분류기, 추천 엔진, 벡터 데이터베이스, 클러스터링 알고리즘, 손실 함수 등 모든 것이 이 선택에 의존합니다. 잘못 선택하면 모델이 엉뚱한 방향으로 최적화됩니다.

모든 경우에 보편적으로 가장 좋은 단 하나의 거리란 존재하지 않습니다. L2는 공간 데이터에 적합합니다. 코사인 유사도(cosine similarity)는 NLP를 지배하고 있습니다. 자카드는 집합을 다룹니다. 편집 거리는 문자열을 다룹니다. 마할라노비스는 상관관계를 고려합니다. 바서슈타인(Wasserstein)은 확률 질량을 이동시킵니다. 각각의 거리는 "유사하다"는 의미에 대해 서로 다른 가정을 담고 있습니다.

이번 레슨에서는 주요 거리 함수를 모두 밑바닥부터 구현하고, 각 함수가 언제 올바른 도구가 되는지 살펴보며, 동일한 데이터라도 어떤 측도를 사용하는지에 따라 완전히 다른 최근접 이웃이 생성되는 과정을 확인합니다.

## 개념

### 노름: 벡터 크기 측정

노름(norm)은 벡터의 "크기"를 측정합니다. 두 벡터 사이의 모든 거리 함수는 두 벡터 차이의 노름으로 표현할 수 있습니다: d(a, b) = ||a - b||. 따라서 노름을 이해하는 것은 거리를 이해하는 것입니다.

### L1 노름 (맨해튼 거리)

L1 노름은 모든 성분의 절댓값의 합입니다.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

축을 따라서만 이동할 수 있고 대각선으로는 이동할 수 없는 도시 격자 위에서 걷는 거리를 측정하기 때문에 맨해튼 거리(Manhattan distance)라고 부릅니다.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

L1을 사용해야 하는 경우:
- 고차원 희소(sparse) 데이터 (텍스트 피처, 원-핫 인코딩)
- 이상치(outlier)에 대한 강건함(robustness)이 필요할 때 (단 하나의 거대한 차이가 전체를 지배하지 않음)
- 피처 선택(feature selection) 문제 (L1 정규화는 희소성을 유도함)

L1 정규화(Lasso)와의 연결: 손실 함수에 ||w||_1을 추가하면 가중치 절댓값의 합에 페널티를 부여합니다. 이는 작은 가중치를 정확히 0으로 밀어내어 자동 피처 선택을 수행합니다. L1 페널티는 가중치 공간에서 다이아몬드 형태의 제약 영역을 생성하며, 다이아몬드의 모서리는 일부 가중치가 0이 되는 축 위에 놓입니다.

손실 함수와의 연결: 평균 절대 오차(MAE, Mean Absolute Error)는 예측값과 타깃 사이의 평균 L1 거리입니다. 모든 오차에 선형적으로 페널티를 부여하므로 MSE에 비해 이상치에 강건합니다.

### L2 노름 (유클리드 거리)

L2 노름은 직선거리입니다. 성분들의 제곱합의 제곱근입니다.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

기하학 수업에서 배웠던 바로 그 거리입니다. n차원에서의 피타고라스 정리입니다.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

L2를 사용해야 하는 경우:
- 저차원 내지 중차원의 연속형 데이터
- 피처들의 스케일이 유사할 때
- 물리적 거리 (공간 데이터, 센서 판독값)
- 픽셀 수준에서의 이미지 유사도

L2 정규화(Ridge)와의 연결: 손실 함수에 ||w||_2^2를 추가하면 큰 가중치에 페널티를 부여합니다. L1과 달리 가중치를 0으로 밀어내지 않습니다. 모든 가중치를 0을 향해 비례적으로 축소시킵니다. L2 페널티는 원형의 제약 영역을 생성하므로 축 위에 모서리가 없습니다. 가중치가 작아지기는 하지만 정확히 0이 되는 경우는 드뭅니다.

손실 함수와의 연결: 평균 제곱 오차(MSE, Mean Squared Error)는 L2 거리 제곱의 평균입니다. 제곱을 하면 작은 오차보다 큰 오차에 훨씬 더 무거운 페널티가 부여됩니다.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 노름: 일반화된 제품군

L1과 L2는 Lp 노름의 특수한 경우입니다:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

p의 값에 따라 서로 다른 형태의 "단위 구(unit ball)"(원점으로부터 거리가 1인 모든 점의 집합)가 생성됩니다:

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-무한대 노름 (체비쇼프 거리)

p가 무한대로 접근함에 따라, Lp 노름은 가장 큰 절댓값을 갖는 성분으로 수렴합니다.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

두 점 사이의 거리는 가장 큰 차이를 보이는 단 하나의 차원에 의해 결정됩니다. 다른 모든 차원은 무시됩니다.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

L-무한대를 사용해야 하는 경우:
- 단일 차원에서의 최악의 편차가 중요할 때
- 게임 보드 (체스에서 킹은 L-무한대로 이동합니다. 어느 방향으로든 한 칸 이동하는 비용은 1입니다)
- 제조 공차 (모든 차원이 규격 내에 있어야 함)

### 코사인 유사도와 코사인 거리

코사인 유사도(cosine similarity)는 두 벡터의 크기를 무시하고 두 벡터 사이의 각도를 측정합니다.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

범위는 -1(반대 방향)부터 +1(동일한 방향)까지입니다. 수직인 벡터의 코사인 유사도는 0입니다.

코사인 거리(cosine distance)는 이를 거리로 변환합니다: cosine_distance = 1 - cosine_similarity. 범위는 0(동일한 방향)부터 2(반대 방향)까지입니다.

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

코사인이 NLP와 임베딩을 지배하는 이유: 텍스트에서 문서의 길이가 유사도에 영향을 미쳐서는 안 됩니다. 고양이에 관한 어떤 문서가 고양이에 관한 다른 문서보다 길이가 두 배 길더라도 여전히 "유사"해야 합니다. 코사인 유사도는 크기(길이)를 무시하고 오직 방향에만 관심을 둡니다. 단어 분포는 동일하지만 길이가 다른 두 문서는 같은 방향을 가리키므로 코사인 유사도 1.0을 얻습니다.

코사인 유사도를 사용해야 하는 경우:
- 텍스트 유사도 (TF-IDF 벡터, 단어 임베딩, 문장 임베딩)
- 크기가 노이즈이고 방향이 신호인 모든 도메인
- 추천 시스템 (사용자 선호도 벡터)
- 임베딩 검색 (벡터 데이터베이스는 거의 항상 코사인이나 내적을 사용함)

### 점곱 유사도 vs 코사인 유사도

두 벡터의 점곱(내적, dot product)은 다음과 같습니다:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

코사인 유사도는 두 벡터의 크기로 정규화한 점곱입니다. 두 벡터가 이미 단위 정규화(unit-normalized, 크기 = 1)되어 있다면 점곱과 코사인 유사도는 동일합니다.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

둘이 달라지는 경우: 점곱에는 크기 정보가 포함됩니다. 크기가 더 큰 벡터가 더 높은 점곱 점수를 받습니다. 이는 "인기 있는" 아이템의 순위를 더 높이고자 하는 일부 검색 시스템에서 중요합니다. 크기가 암묵적인 품질 또는 중요도 신호 역할을 합니다.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

실무에서의 지침:
- 순수한 방향성 유사도를 원할 때는 코사인 유사도를 사용합니다.
- 크기에 의미 있는 정보가 담겨 있을 때는 점곱을 사용합니다.
- 많은 벡터 데이터베이스(Pinecone, Weaviate, Qdrant)에서 둘 중 하나를 선택할 수 있습니다.
- 임베딩이 L2 정규화되어 있다면 어느 것을 선택하든 상관없습니다.
### 마할라노비스 거리(Mahalanobis Distance)

유클리드 거리는 모든 차원을 동등하게 취급합니다. 하지만 특성 간에 상관관계가 있거나 스케일이 서로 다르다면, L2 거리는 왜곡된 결과를 제공합니다.

마할라노비스 거리는 데이터의 공분산 구조를 고려합니다.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

여기서 S는 데이터의 공분산 행렬입니다.

직관적 설명: 마할라노비스 거리는 먼저 데이터의 상관관계를 제거하고 정규화(백색화, whitening)한 다음, 그 변환된 공간에서 L2 거리를 계산합니다. 만약 S가 단위 행렬이라면(상관관계가 없고 분산이 1인 특성들), 마할라노비스 거리는 유클리드 거리와 같아집니다.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

마할라노비스 거리를 사용해야 하는 경우:
- 이상치 탐지(평균으로부터 마할라노비스 거리가 큰 점이 이상치임)
- 특성들의 스케일과 상관관계가 서로 다른 경우의 분류
- 신뢰할 수 있는 공분산 행렬을 추정하기에 충분한 데이터가 있을 때
- 제조업의 품질 관리(다변량 공정 모니터링)

### 자카드 유사도(Jaccard Similarity, 집합용)

자카드 유사도는 두 집합 간의 겹침(overlap) 정도를 측정합니다.

```
J(A, B) = |A intersect B| / |A union B|
```

값의 범위는 0(겹침 없음)부터 1(동일한 집합)까지입니다. 자카드 거리 = 1 - 자카드 유사도입니다.

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

자카드 유사도를 사용해야 하는 경우:
- 태그, 카테고리 또는 특성 집합 비교
- 단어 빈도가 아닌 존재 여부에 기반한 문서 유사도
- 유사 중복 탐지(MinHash를 통한 자카드 근사)
- 이진 특성 벡터 비교(존재/부재 데이터)
- 세그멘테이션 모델 평가(IoU(Intersection over Union) = 자카드)

### 편집 거리(Edit Distance / Levenshtein Distance)

편집 거리는 한 문자열을 다른 문자열로 변환하는 데 필요한 최소 단일 문자 연산 횟수를 측정합니다. 연산의 종류는 삽입, 삭제, 대체입니다.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

동적 프로그래밍(dynamic programming)을 사용하여 계산합니다. 행렬의 (i, j) 원소가 문자열 A의 처음 i개 문자와 문자열 B의 처음 j개 문자 사이의 편집 거리가 되도록 행렬을 채웁니다.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

편집 거리를 사용해야 하는 경우:
- 맞춤법 검사 및 교정
- DNA 서열 정렬(가중치가 적용된 연산 사용)
- 퍼지 문자열 매칭(fuzzy string matching)
- 정제되지 않은 텍스트 데이터의 중복 제거

### KL 발산(KL Divergence, 거리는 아니지만 거리처럼 사용됨)

KL 발산은 한 확률 분포가 다른 확률 분포와 어떻게 다른지 측정합니다. 09강에서 다루었지만, 실제로는 "거리"가 아님에도 거리처럼 사용되기 때문에 여기서도 함께 논의합니다.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

결정적인 특성: KL 발산은 대칭적이지 않습니다(NOT symmetric).

```
D_KL(P || Q) != D_KL(Q || P)
```

이는 거리 메트릭의 기본 요건을 충족하지 못함을 의미합니다. 또한 삼각 부등식도 만족하지 않습니다. 따라서 거리가 아니라 발산(divergence)입니다.

순방향 KL(D_KL(P || Q))은 "평균 탐색(mean-seeking)" 특성을 갖습니다: Q가 P의 모든 최빈값(mode)을 포괄하려고 합니다.
역방향 KL(D_KL(Q || P))은 "최빈값 탐색(mode-seeking)" 특성을 갖습니다: Q가 P의 단일 최빈값에 집중합니다.

KL 발산이 활용되는 분야:
- VAE(ELBO의 KL 항이 잠재 분포를 사전 분포에 가깝게 유도함)
- 지식 증류(knowledge distillation, 학생 모델이 교사 모델의 분포를 모방하려고 시도함)
- RLHF(KL 페널티를 통해 미세 조정(fine-tuning)된 모델이 베이스 모델과 너무 멀어지지 않도록 유지함)
- 정책 경사(policy gradient) 방법(정책 업데이트 폭 제한)

### 바서슈타인 거리(Wasserstein Distance / Earth Mover's Distance)

바서슈타인 거리는 한 확률 분포를 다른 확률 분포로 변환하는 데 필요한 최소 "일(work)"을 측정합니다. 한 분포는 흙더미이고 다른 분포는 구덩이라고 할 때, 얼마나 많은 흙을 얼마나 멀리 옮겨야 하는지로 생각하면 됩니다.

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

1차원 분포의 경우, 누적분포함수의 차이의 절댓값에 대한 적분으로 단순화됩니다:

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

바서슈타인 거리가 중요한 이유:
- 진정한 의미의 메트릭입니다(대칭적이며 삼각 부등식을 만족함).
- 분포들이 겹치지 않더라도 그래디언트를 제공합니다(KL 발산은 무한대로 발산함).
- 이러한 특성 덕분에 오리지널 GAN의 학습 불안정성을 해결한 WGAN(Wasserstein GAN)의 핵심이 되었습니다.

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

바서슈타인 거리를 사용해야 하는 경우:
- GAN 학습(WGAN, WGAN-GP)
- 겹치지 않을 수 있는 분포 간의 비교
- 최적 수송(optimal transport) 문제
- 이미지 검색(색상 히스토그램 비교)

### 태스크마다 서로 다른 거리가 필요한 이유

| 태스크 | 최적의 거리 | 이유 |
|------|--------------|-----|
| 텍스트 유사도 | 코사인 | 크기는 노이즈이고 방향이 의미를 나타냄 |
| 이미지 픽셀 비교 | L2 | 공간적 관계가 중요하며 특성들의 스케일이 비슷함 |
| 희소 고차원 특성 | L1 | 견고하며, 드물게 나타나는 큰 차이를 증폭시키지 않음 |
| 집합 겹침(태그, 카테고리) | 자카드 | 데이터가 본질적으로 벡터가 아닌 집합 형태임 |
| 문자열 매칭 | 편집 거리 | 연산이 인간의 직관적인 편집 방식과 일치함 |
| 이상치 탐지 | 마할라노비스 | 특성 간의 상관관계와 스케일을 고려함 |
| 분포 비교 | KL 발산 | P 대신 Q를 사용함으로써 손실되는 정보량을 측정함 |
| GAN 학습 | 바서슈타인 | 분포들이 겹치지 않더라도 그래디언트를 제공함 |
| 임베딩(벡터 DB) | 코사인 또는 점곱 | 임베딩은 방향에 의미를 인코딩하도록 학습됨 |
| 추천 시스템 | 점곱 | 크기가 인기나 신뢰도를 인코딩할 수 있음 |
| DNA 서열 | 가중치 편집 거리 | 뉴클레오타이드 쌍에 따라 대체 비용이 다름 |
| 제조 QC | L-무한대 | 어느 한 차원에서의 최악의 편차가 중요함 |

### 손실 함수와의 연결점

손실 함수는 예측값과 타깃값 사이에 적용되는 거리 함수입니다.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 규제와의 연결점

규제(regularization)는 손실 함수에 가중치에 대한 노름 페널티를 추가합니다.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

L1은 희소성(sparsity)을 만들어내지만 L2는 그렇지 않은 이유: 2차원 가중치 공간에서의 제약 영역을 머릿속에 그려보십시오. L1은 다이아몬드 형태이고, L2는 원 형태입니다. 손실 함수의 등고선(타원)은 다이아몬드의 모서리, 즉 하나의 가중치가 0인 지점에서 접할 가능성이 가장 높습니다. 반면 원과는 두 가중치가 모두 0이 아닌 매끄러운 지점에서 접하게 됩니다.

### 최근접 이웃 탐색(Nearest Neighbor Search)

모든 거리 함수는 최근접 이웃 탐색 문제를 수반합니다: 쿼리 포인트가 주어졌을 때 데이터셋에서 가장 가까운 포인트들을 찾는 것입니다.

정확한 최근접 이웃 탐색(exact nearest neighbor search)은 d차원을 갖는 n개 포인트의 데이터셋에서 쿼리당 O(n * d)의 시간 복잡도를 갖습니다. 대규모 데이터셋의 경우 이 방식은 너무 느립니다.

근사 최근접 이웃(Approximate Nearest Neighbor, ANN) 알고리즘은 약간의 정확도를 희생하는 대신 속도를 대폭 향상합니다:

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW(Hierarchical Navigable Small World)는 현대 벡터 데이터베이스에서 가장 지배적으로 사용되는 알고리즘입니다. 각 노드가 근사 최근접 이웃들과 연결되는 다층 그래프를 구성합니다. 탐색은 최상위 계층(희소하며 도약 거리가 긺)에서 시작하여 최하위 계층(조밀하며 도약 거리가 짧음)으로 내려갑니다.

```figure
norm-unit-balls
```

## 직접 구현해 보기

### 1단계: 모든 노름 및 거리 함수

전체 구현은 `code/distances.py`를 참고하십시오. 모든 함수는 순수 Python 수학 연산만을 사용하여 바닥부터 구현되었습니다.

### 2단계: 동일한 데이터, 서로 다른 거리, 서로 다른 이웃

`distances.py`의 데모는 데이터셋을 생성하고, 쿼리 포인트를 선택한 다음, 거리 메트릭에 따라 최근접 이웃이 어떻게 달라지는지 보여줍니다. L1 거리 기준에서 "가장 가까운" 포인트가 L2나 코사인 거리 기준에서는 가장 가깝지 않을 수 있습니다.

### 3단계: 임베딩 유사도 탐색

코드에는 코사인 유사도와 L2 거리를 사용하여 쿼리와 가장 유사한 "문서"를 찾는 모의 임베딩 유사도 탐색이 포함되어 있으며, 두 메트릭 간에 순위가 어떻게 달라질 수 있는지를 보여줍니다.
## 실전 활용

가장 흔한 실전 활용 사례는 벡터 데이터베이스(vector database)에서 유사한 항목을 찾는 것입니다.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

`model.encode(text)`를 호출한 뒤 벡터 데이터베이스를 검색할 때 내부적으로 일어나는 과정은 다음과 같습니다. 임베딩 모델이 텍스트를 벡터로 매핑합니다. 벡터 데이터베이스는 쿼리 벡터와 저장된 모든 벡터 사이의 코사인 유사도(또는 내적)를 계산하며, 모든 벡터를 일일이 대조하지 않기 위해 ANN 알고리즘을 사용합니다.

## 연습 문제

1. (1, 2, 3)과 (4, 0, 6) 사이의 L1, L2, L-무한대(L-infinity) 거리를 계산하세요. 임의의 두 점에 대해 L-inf <= L2 <= L1이 항상 성립함을 확인하세요. 이러한 대소 관계가 보장되는 이유를 증명하세요.

2. 코사인 유사도는 높지만(> 0.9) L2 거리는 큰(> 10) 두 벡터를 만드세요. 기하학적으로 어떤 상황인지 설명하세요. 그런 다음 코사인 유사도는 낮지만(< 0.3) L2 거리는 작은(< 0.5) 두 벡터를 만드세요.

3. 데이터셋과 쿼리 포인트를 입력받아 L1, L2, 코사인, 마할라노비스 거리(Mahalanobis distance) 기준으로 각각 가장 가까운 이웃을 반환하는 함수를 구현하세요. 네 가지 기준에서 가장 가까운 포인트가 모두 서로 다른 데이터셋을 찾아보세요.

4. 누적 분포 함수(CDF) 방식을 사용하여 [0.5, 0.5, 0, 0]과 [0, 0, 0.5, 0.5] 사이의 바서슈타인 거리(Wasserstein distance)를 직접 계산하세요. 그런 다음 [0.25, 0.25, 0.25, 0.25]와 [0, 0, 0.5, 0.5] 사이의 거리도 계산하세요. 어느 쪽이 더 크며, 그 이유는 무엇인가요?

5. 자카드 유사도(Jaccard similarity)를 근사하기 위한 MinHash를 구현하세요. 무작위 집합 100개를 생성하고, 모든 쌍에 대한 정확한 자카드 유사도를 계산한 뒤, 50개, 100개, 200개의 해시 함수를 사용한 MinHash 근사 결과와 비교하세요. 근사 오차를 시각화하세요.

## 핵심 용어

| 용어 | 일반적인 표현 | 실제 의미 |
|------|----------------|----------------------|
| 노름(Norm) | "벡터의 크기" | 벡터를 음이 아닌 스칼라로 매핑하며, 삼각 부등식, 절대 동차성(absolute homogeneity)을 만족하고 영벡터에 대해서만 0이 되는 함수 |
| L1 노름 | "맨해튼 거리" | 성분들의 절댓값 합. 최적화에서 희소성(sparsity)을 유도함. 이상치(outlier)에 강건함 |
| L2 노름 | "유클리드 거리" | 성분 제곱합의 제곱근. 유클리드 공간에서의 직선거리 |
| Lp 노름 | "일반화된 노름" | 성분 절댓값의 p제곱 합에 대한 p제곱근. L1과 L2는 특수한 경우에 해당함 |
| L-무한대 노름 | "최대 노름" 또는 "체비쇼프 거리" | 성분의 절댓값 중 최댓값. p가 무한대로 갈 때 Lp의 극한 |
| 코사인 유사도 | "벡터 사이의 각도" | 두 벡터의 크기로 정규화한 내적. -1에서 +1 사이의 범위를 가짐. 벡터의 길이는 무시함 |
| 코사인 거리 | "1에서 코사인 유사도를 뺀 값" | 코사인 유사도를 거리로 변환한 것. 0에서 2 사이의 범위를 가짐 |
| 내적(Dot product) | "정규화되지 않은 코사인" | 대응하는 성분별 곱의 합. 코사인 유사도에 두 벡터의 크기를 곱한 값과 같음 |
| 마할라노비스 거리 | "상관관계를 고려한 거리" | 데이터 공분산 행렬을 사용하여 백색화(무상관화 및 정규화)된 공간에서의 L2 거리 |
| 자카드 유사도 | "집합 간 중복도" | 교집합의 크기를 합집합의 크기로 나눈 값. 벡터가 아닌 집합에 적용됨 |
| 편집 거리(Edit distance) | "레벤슈타인 거리" | 한 문자열을 다른 문자열로 변환하는 데 필요한 최소 삽입, 삭제, 치환 횟수 |
| KL 발산(KL divergence) | "분포 간의 거리" | 진정한 의미의 거리는 아님(대칭적이지 않음). P를 인코딩하기 위해 Q를 사용할 때 발생하는 추가 비트 수를 측정함 |
| 바서슈타인 거리 | "Earth mover's distance" | 한 분포의 질량을 다른 분포로 이동시키는 데 필요한 최소 일의 양. 진정한 메트릭(metric)임 |
| 근사 최근접 이웃(ANN) | "ANN 검색" | 정확한 탐색보다 훨씬 빠르게 근사적으로 가장 가까운 점을 찾는 알고리즘(HNSW, LSH, IVF) |
| HNSW | "벡터 DB 알고리즘" | Hierarchical Navigable Small World 그래프. 빠른 근사 최근접 이웃 탐색을 위한 다층 그래프 구조 |
| L1 정규화 | "라쏘(Lasso)" | 손실 함수에 가중치의 L1 노름을 추가함. 가중치를 0으로 만들어 희소성을 유도함 |
| L2 정규화 | "릿지(Ridge)" 또는 "가중치 감쇠(weight decay)" | 손실 함수에 가중치의 L2 노름 제곱을 추가함. 희소성을 만들지 않으면서 가중치를 0에 가깝게 축소함 |
| 엘라스틱 넷(Elastic Net) | "L1 + L2" | L1과 L2 정규화를 결합한 방식. 상관관계가 있는 특성 그룹을 어느 하나만 사용할 때보다 더 잘 다룸 |

## 추가 참고 자료

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - 10억 단위 규모의 ANN 검색을 지원하는 Meta의 라이브러리
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - GAN에 Earth Mover 거리를 도입한 논문
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 기초적인 ANN 알고리즘 논문
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - 코사인 유사도가 임베딩의 표준이 된 Word2Vec 논문
- [sklearn.neighbors 문서](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn의 거리 메트릭 및 최근접 이웃 알고리즘에 대한 실용적 가이드
