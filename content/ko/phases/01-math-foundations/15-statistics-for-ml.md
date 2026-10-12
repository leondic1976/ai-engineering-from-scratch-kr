---
title_en: "Statistics for Machine Learning"
source: "phases/01-math-foundations/15-statistics-for-ml"
source_sha: "9770d8e715f37b25"
model: "gemini-3.8-flash"
translated_at: "2026-10-12"
---
# 머신러닝을 위한 통계학

> 통계학은 모델이 실제로 효과가 있는지, 아니면 단지 운이 좋았을 뿐인지를 판별하는 방법입니다.

**유형:** Build (실습)
**Language:** Python
**선수 레슨:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**소요 시간:** ~120분

## 학습 목표

- 기술 통계(descriptive statistics), 피어슨/스피어만 상관계수, 공분산 행렬을 밑바닥부터 계산하기
- 가설 검정(t-test, 카이제곱 검정)을 수행하고 p-값 및 신뢰구간을 올바르게 해석하기
- 분포에 대한 가정 없이 부트스트랩 리샘플링(bootstrap resampling)을 활용하여 모든 지표에 대한 신뢰구간 구축하기
- 효과 크기(effect size) 측정을 사용하여 통계적 유의성과 실질적 유의성을 구분하기

## 문제 상황

두 개의 모델을 훈련했습니다. 모델 A는 테스트 세트에서 0.87점을 기록하고, 모델 B는 0.89점을 기록했습니다. 이에 모델 B를 배포합니다. 3주 후, 프로덕션 지표가 이전보다 더 나빠졌습니다. 무슨 일이 일어난 걸까요?

모델 B는 실제로 모델 A보다 뛰어난 성능을 보인 것이 아니었습니다. 0.02의 차이는 노이즈에 불과했습니다. 테스트 세트가 너무 작았거나, 분산이 너무 컸거나, 둘 다였던 것입니다. 개선인 것처럼 포장된 무작위성을 배포한 셈입니다.

이런 일은 끊임없이 발생합니다. 캐글(Kaggle) 리더보드의 순위 요동. 재현에 실패하는 논문들. 불과 수백 개의 샘플을 기반으로 승자를 선언하는 A/B 테스트. 근본적인 원인은 언제나 같습니다. 바로 누군가가 통계학을 건너뛰었다는 점입니다.

통계학은 신호와 노이즈를 구별할 수 있는 도구를 제공합니다. 차이가 실제 존재하는지, 결과를 얼마나 확신할 수 있는지, 결과를 신뢰하기까지 데이터가 얼마나 더 필요한지 알려줍니다. 모든 ML 파이프라인, 모든 모델 비교, 모든 실험에는 통계학이 필요합니다. 통계학이 없다면 그저 추측에 불과합니다.

## 핵심 개념

### 기술 통계: 데이터 요약하기

모델을 만들기 전에 먼저 데이터가 어떻게 생겼는지 알아야 합니다. 기술 통계는 데이터셋의 형태를 포착하는 몇 가지 숫자로 데이터를 압축합니다.

**중심 경향 척도(Measures of central tendency)**는 "중심이 어디인가?"라는 질문에 답합니다.

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

평균은 균형점입니다. 중앙값은 중간 지점입니다. 두 값이 갈라진다면 분포가 치우쳐(skewed) 있는 것입니다. 소득 분포는 평균 >> 중앙값 형태를 보입니다(억만장자들로 인한 오른쪽 꼬리 치우침). 학습 중 손실(loss) 분포는 종종 평균 << 중앙값 형태를 띱니다(다루기 쉬운 샘플들로 인한 왼쪽 꼬리 치우침).

**산포도 척도(Measures of spread)**는 "데이터가 얼마나 흩어져 있는가?"라는 질문에 답합니다.

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**백분위수(Percentile)**는 정렬된 데이터를 100개의 동등한 부분으로 나눕니다. 제25백분위수(Q1)는 값의 25%가 이 지점 아래에 속함을 의미합니다. 제50백분위수는 중앙값입니다. 제75백분위수는 Q3입니다.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

ML에서는 추론(inference) 지연 시간, 예측 신뢰도 분포, 오차 분포를 파악할 때 백분위수를 중요하게 다룹니다. 평균 오차는 낮지만 P99 오차가 형편없는 모델은 안전이 중요한(safety-critical) 애플리케이션에서는 쓸모가 없을 수 있습니다.

**표본 통계량 vs 모집단 모수.** 표본에서 분산을 계산할 때는 {{P3_0}} 대신 {{P3_1}}으로 나눕니다. 이것이 베셀 보정(Bessel's correction)입니다. 표본 평균이 실제 모집단 평균과 일치하지 않는다는 점을 보정해 줍니다. 분모에 {{P3_2}}을 넣으면 실제 분산을 체계적으로 과소평가하게 됩니다. {{P3_3}}을 사용하면 추정치가 편향되지 않습니다(unbiased).

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

실무적인 관점에서, {{P3_0}}이 크다면(수천 개의 샘플) 그 차이는 미미합니다. {{P3_0}}이 작다면(수십 개의 샘플) 큰 차이가 발생합니다.

### 상관관계: 변수들이 함께 움직이는 방식

상관관계는 두 변수 간 선형 관계의 강도와 방향을 측정합니다.

**피어슨 상관계수(Pearson correlation coefficient)**는 선형적 연관성을 측정합니다:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

피어슨은 관계가 선형적이고 두 변수가 대략 정규분포를 따른다고 가정합니다. 이상치(outlier)에 민감하여 단 하나의 극단값만으로도 {{P4_0}}를 0.1에서 0.9로 끌어올릴 수 있습니다.

**스피어만 순위 상관계수(Spearman rank correlation)**는 단조적(monotonic) 연관성을 측정합니다:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**상황별 사용 기준:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**황금률:** 상관관계가 인과관계를 의미하지는 않습니다. 아이스크림 판매량과 익사 사고 사망자는 둘 다 여름에 증가하기 때문에 상관관계가 있습니다. 모델의 정확도와 파라미터 수는 상관관계가 있지만, 파라미터를 추가한다고 해서 자동으로 정확도가 향상되는 것은 아닙니다(과적합(overfitting) 참고).

### 공분산 행렬

두 변수 사이의 공분산은 변수들이 함께 변하는 방식을 측정합니다:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

{{P7_0}}개의 특성에 대해, 공분산 행렬 {{P7_1}}는 {{P7_3}}인 {{P7_2}} 행렬입니다. 대각 성분 {{P7_4}}는 각 특성의 분산입니다.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**PCA와의 연결점.** PCA는 공분산 행렬을 고유값 분해(eigendecomposition)합니다. 고유벡터는 주성분(최대 분산의 방향)이 됩니다. 고유값은 각 성분이 분산을 얼마나 포착하는지 알려줍니다. 이것이 바로 레슨 10에서 다룬 내용이지만, 이제 왜 공분산 행렬이 분해하기에 적절한 대상인지 이해할 수 있습니다. 데이터의 모든 쌍별 선형 관계를 인코딩하고 있기 때문입니다.

**상관관계와의 연결점.** 상관 행렬(correlation matrix)은 표준화된 변수들(각 변수를 표준편차로 나눈 값)의 공분산 행렬입니다. 상관관계는 공분산을 정규화하여 모든 값이 [-1, 1] 사이에 오도록 합니다.

### 가설 검정

가설 검정은 불확실성 속에서 의사결정을 내리기 위한 프레임워크입니다. 하나의 주장에서 출발하여 데이터를 수집하고, 데이터가 그 주장과 일치하는지 여부를 판단합니다.

**기본 설정:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p-값(p-value)**은 {{P10_0}}가 참이라고 가정할 때 관측된 것만큼 극단적인 데이터를 볼 확률입니다. {{P10_0}}가 참일 확률이 **아닙니다**. 이것이 통계학에서 가장 흔히 발생하는 단 하나의 오해입니다.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**신뢰구간(Confidence interval)**은 모수가 가질 수 있는 그럴듯한 값의 범위를 제공합니다:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

신뢰구간의 너비는 정밀도(precision)를 나타냅니다. 구간이 넓다는 것은 불확실성이 높음을 의미합니다. 구간이 좁다는 것은 추정치가 정밀하다는 뜻입니다(단, 데이터에 편향이 있다면 반드시 정확(accurate)하다는 뜻은 아닙니다).

### t-검정(t-test)

t-검정은 평균을 비교합니다. 몇 가지 유형이 있습니다.

**단일표본 t-검정(One-sample t-test):** 모집단 평균이 가설의 값과 다른가?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**독립표본 t-검정(Two-sample t-test):** 두 집단의 평균이 다른가?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**대응표본 t-검정(Paired t-test):** 측정값이 쌍을 이룰 때(동일한 데이터 분할에서 평가된 동일한 모델):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

ML에서는 대응표본 t-검정이 흔히 사용됩니다. 동일한 10개의 교차 검증(cross-validation) 폴드에서 두 모델을 모두 실행하고 점수를 쌍별로 비교합니다.

### 카이제곱 검정(Chi-squared Test)

카이제곱 검정은 관측된 빈도가 기대 빈도와 일치하는지 확인합니다. 범주형 데이터에 유용합니다.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### ML 모델을 위한 A/B 테스트

ML에서의 A/B 테스트는 웹 A/B 테스트와 다릅니다. 모델 비교에는 특유의 어려움이 따릅니다:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**절차:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 통계적 유의성 vs 실질적 유의성

결과가 통계적으로 유의하더라도 실질적으로는 무의미할 수 있습니다. 데이터가 충분히 많으면 사소한 차이조차 통계적으로 유의해집니다.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**효과 크기(Effect size)**는 표본 크기와 무관하게 차이가 얼마나 큰지를 정량화합니다:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

항상 p-값과 효과 크기를 함께 보고하십시오. p-값은 차이가 실제 존재하는지를 알려주고, 효과 크기는 그 차이가 유의미한지(가치가 있는지)를 알려줍니다.
### 다중 비교 문제(Multiple Comparison Problem)

많은 가설을 검정하면 그중 일부는 우연히 "유의미"해집니다. alpha = 0.05에서 20개의 검정을 수행하면, 실제로는 아무런 효과가 없더라도 1개의 위양성(false positive)이 발생할 것으로 예상됩니다.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**본페로니 보정(Bonferroni correction):** 검정 횟수로 alpha를 나눕니다.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

ML에서는 여러 지표에 걸쳐 모델을 비교하거나, 수많은 하이퍼파라미터 설정을 테스트하거나, 여러 데이터셋에서 평가할 때 이 문제가 중요해집니다.

### 부트스트랩 방법(Bootstrap Methods)

부트스트래핑(Bootstrapping)은 복원 추출(resampling with replacement)을 통해 통계량의 표본 분포를 추정합니다. 기저 분포에 대한 어떤 가정도 필요하지 않습니다.

**알고리즘:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**부트스트랩 신뢰구간(백분위수 방법):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**부트스트랩이 ML에서 중요한 이유:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**모델 비교를 위한 부트스트랩:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

이는 분포에 대한 가정을 하지 않기 때문에 대응표본 t-검정(paired t-test)보다 더 강건(robust)합니다.

### 모수 검정 vs 비모수 검정

**모수 검정(Parametric tests)**은 특정 분포(주로 정규분포)를 가정합니다:

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**비모수 검정(Non-parametric tests)**은 분포에 대한 가정을 하지 않습니다:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**비모수 검정을 사용해야 하는 경우:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**모수 검정을 사용해야 하는 경우:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

ML 실험에서는 일반적으로 n이 작으므로(5개 또는 10개의 교차 검증 폴드), 윌콕슨 부호순위 검정(Wilcoxon signed-rank test) 같은 비모수 검정이 t-검정보다 더 적절한 경우가 많습니다.

### 중심극한정리: 실무적 의미

중심극한정리(Central Limit Theorem, CLT)는 모집단의 기저 분포와 관계없이 n이 커질수록 표본 평균의 분포가 정규분포에 가까워진다고 말합니다.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**ML에서 이것이 중요한 이유:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**중심극한정리가 하지 않는 것:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### ML 논문에서 흔히 발생하는 통계적 실수

1. **학습 세트에서의 검정.** 과적합을 보장하는 행위입니다. 학습 중에 모델이 전혀 보지 않은 데이터는 반드시 따로 남겨두어야 합니다.

2. **신뢰구간 누락.** 불확실성 없이 단일 정확도 수치만 보고하면 결과를 재현하거나 검증할 수 없습니다.

3. **다중 비교 무시.** 보정 없이 50개의 설정을 테스트하고 가장 좋은 것만 보고하면 위양성률이 증가합니다.

4. **통계적 유의성과 실질적 유의성의 혼동.** 0.01%의 정확도 향상에 대한 p-값이 0.001이라 하더라도 이는 의미가 없습니다.

5. **불균형 데이터에 정확도 사용.** 음성 클래스가 99%인 데이터셋에서 99%의 정확도는 모델이 아무것도 학습하지 못했음을 의미합니다. 정밀도(precision), 재현율(recall), F1, 또는 AUC를 사용해야 합니다.

6. **지표 체리피킹(Cherry-picking).** 자신의 모델이 이긴 지표만 보고하는 행위입니다. 정직한 평가를 위해서는 모든 관련 지표를 보고해야 합니다.

7. **학습/테스트 분할 간 정보 누출(Data leakage).** 분할 전에 정규화를 수행하거나, 과거를 예측하기 위해 미래 데이터를 사용하는 경우입니다.

8. **분산 추정치 없는 소규모 테스트 세트.** 100개의 샘플로 평가하고 2% 향상되었다고 주장하는 것은 신호(signal)가 아니라 노이즈입니다.

9. **데이터가 독립적이지 않은데 독립성을 가정.** 동일 환자의 의료 영상, 동일 문서의 여러 문장 등이 해당합니다. 한 그룹 내의 관측치들은 서로 상관되어 있습니다.

10. **P-해킹(P-hacking).** p < 0.05가 나올 때까지 다양한 검정, 하위 집합, 또는 제외 기준을 시도하는 행위입니다. 그 결과는 탐색 과정이 낳은 인위적인 결과물일 뿐입니다.

## 직접 구현해 보기

다음을 구현합니다:

1. **밑바닥부터 구현하는 기술통계량** (평균, 중앙값, 최빈값, 표준편차, 백분위수, IQR)
2. **상관관계 함수** (피어슨 및 스피어만, 공분산 행렬 포함)
3. **가설 검정** (단일표본 t-검정, 독립표본 t-검정, 카이제곱 검정)
4. **부트스트랩 신뢰구간** (가정 없이 모든 통계량에 적용 가능)
5. **A/B 테스트 시뮬레이터** (데이터 생성, 검정, 제1종 및 제2종 오류 확인)
6. **통계적 유의성 vs 실질적 유의성 데모** (n이 크면 모든 것이 "유의미"해짐을 시연)

numpy나 scipy 없이, 오직 `math`와 `random`만을 사용하여 밑바닥부터 전부 구현합니다.

```figure
f3-bootstrap-resample
```

## 핵심 용어

| 용어 | 정의 |
|---|---|
| 평균(Mean) | 값의 합을 개수로 나눈 값. 이상치에 민감함. |
| 중앙값(Median) | 정렬된 데이터의 중앙에 위치한 값. 이상치에 강건함. |
| 표준편차(Standard deviation) | 분산의 제곱근. 원래 단위로 데이터의 퍼짐 정도를 측정함. |
| 백분위수(Percentile) | 데이터의 특정 백분율이 그 아래에 위치하게 되는 값. |
| IQR | 사분위범위(Interquartile range). Q3에서 Q1을 뺀 값. 중앙 50%의 퍼짐 정도. |
| 피어슨 상관계수(Pearson correlation) | 두 변수 간의 선형적 연관성을 측정함. 범위는 [-1, 1]. |
| 스피어만 상관계수(Spearman correlation) | 순위를 사용하여 단조적 연관성을 측정함. |
| 공분산 행렬(Covariance matrix) | 모든 특성 간의 쌍별 공분산으로 이루어진 행렬. |
| 귀무가설(Null hypothesis) | 효과나 차이가 없다는 기본 가정. |
| p-값(p-value) | 귀무가설이 참일 때 이 정도로 극단적인 데이터가 관측될 확률. |
| 신뢰구간(Confidence interval) | 주어진 신뢰수준에서 모수가 존재할 것으로 타당하게 여겨지는 값의 범위. |
| t-검정(t-test) | 평균이 유의미하게 다른지 검정함. t-분포를 사용함. |
| 카이제곱 검정(Chi-squared test) | 관측 빈도가 기대 빈도와 다른지 검정함. |
| 효과 크기(Effect size) | 표본 크기와 무관한 차이의 크기. 코헨의 d(Cohen's d)가 흔히 사용됨. |
| 본페로니 보정(Bonferroni correction) | 위양성을 제어하기 위해 유의수준 임계값을 검정 횟수로 나눔. |
| 부트스트랩(Bootstrap) | 표본 분포를 추정하기 위해 복원 추출을 수행하는 방법. |
| 제1종 오류(Type I error) | 위양성. 귀무가설(H0)이 참인데 기각하는 오류. |
| 제2종 오류(Type II error) | 위음성. 귀무가설(H0)이 거짓인데 기각하지 못하는 오류. |
| 검정력(Statistical power) | 거짓인 귀무가설(H0)을 올바르게 기각할 확률. 검정력 = 1 - 제2종 오류율. |
| 중심극한정리(Central limit theorem) | 표본 크기가 커질수록 표본 평균은 정규분포로 수렴함. |
| 모수 검정(Parametric test) | 데이터에 대해 특정 분포(주로 정규분포)를 가정함. |
| 비모수 검정(Non-parametric test) | 분포에 대한 가정을 하지 않음. 순위나 부호에 기반하여 작동함. |
