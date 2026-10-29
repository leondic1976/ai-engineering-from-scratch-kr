---
title_en: "Monitoring Models in Production — Data Drift and Concept Drift"
source: "phases/02-ml-fundamentals/20-model-monitoring-and-drift"
source_sha: "67ed1a1ed2ff0330"
model: "gemini-3.8-flash"
translated_at: "2026-10-29"
---
# 프로덕션 모델 모니터링 — 데이터 드리프트와 컨셉 드리프트

> 배포된 모델은 에러 메시지 없이 실패합니다. 세상이 학습 데이터로부터 멀어지는 동안에도 모델은 계속해서 점수를 반환하며, 오직 모니터링만이 그 간극을 보여줍니다.

**유형:** Build (실습)
**사용 언어:** Python
**선수 레슨:** Phase 2, Lessons 09 (Model Evaluation), 16 (Anomaly Detection), and 19 (ML System Design)
**소요 시간:** ~110분

## 학습 목표

- 시스템 헬스, 입력값, 예측값, 결과의 4개 계층에서 모니터링 신호를 선택합니다.
- 결합 분포에서 변화하는 부분을 기준으로 공변량 드리프트(covariate drift), 사전 확률 드리프트(prior drift), 컨셉 드리프트(concept drift)를 구별합니다.
- Population Stability Index, 2표본 Kolmogorov-Smirnov 통계량, 카이제곱 검정을 밑바닥부터 구현합니다.
- 고정된 참조 윈도우와 슬라이딩 현재 윈도우를 비교하고, 윈도우 크기가 노이즈와 지연 시간에 미치는 영향을 설명합니다.
- 효과 크기(effect size), 다중 검정 보정, 지속성 규칙을 활용해 오경보를 제한하는 알림 정책을 설계합니다.
- 어떤 종류의 드리프트에 재학습이 필요한지, 새로운 레이블이 필요한지, 혹은 아무런 조치가 필요 없는지 결정합니다.

## 문제 상황

결제 위험 평가 모델이 95%의 정확도로 운영 환경에 배포됩니다. 2개월 후, 한 마케팅 캠페인으로 인해 장바구니 규모가 더 큰 앱 사용자들이 유입됩니다. 4개월 후에는 사기범들이 수법을 바꾸면서 이전에는 안전했던 결제 건들이 지급 거절(chargeback)로 이어지기 시작합니다.

이 모든 과정 동안 모델 서비스는 모든 요청에 대해 HTTP 200을 반환합니다. 지연 시간(latency)은 정상이며, 로그에는 아무런 에러도 기록되지 않습니다. 문제의 첫 징후는 피해가 발생하고 수주가 지난 뒤에야 나오는 재무 보고서의 지급 거절 내역입니다.

소프트웨어 모니터링은 서비스가 응답하는지 확인합니다. 모델 모니터링은 그 응답이 여전히 올바른지, 그리고 데이터가 모델을 학습시켰던 데이터와 여전히 유사한지 확인합니다. 이번 레슨에서는 이 두 번째 작업을 위한 통계량, 윈도우, 알림 정책을 구축합니다. 두 가지 변화를 인위적으로 주입한 합성 스트림에서 이를 실행하고, 어떤 모니터가 어떤 변화에 반응하는지 관찰합니다.

## 핵심 개념

### 모니터링 대상

4개 계층을 모니터링합니다. 처음 3개 계층은 조기 경보를 제공합니다. 오직 4번째 계층만이 정확성을 측정하며, 이는 뒤늦게 도착합니다.

| 계층 | 신호 | 확인 시점 | 답을 주는 질문 |
|---|---|---|---|
| 시스템 헬스 | p50 및 p99 지연 시간, 에러율, 처리량(throughput), 메모리 | 즉시 | 서비스가 응답하는가? |
| 입력값 | 각 특성의 분포, 결측값 비율, 스키마, 새로운 범주 | 즉시 | 실시간 데이터가 학습 데이터와 유사한가? |
| 예측값 | 점수 분포, 양성 비율, 기본 응답의 점유율 | 즉시 | 모델이 이전과 동일하게 동작하는가? |
| 결과 | 정확도, 정밀도, 캘리브레이션(calibration), 비즈니스 지표 | 레이블 지연 이후 | 모델이 여전히 올바른가? |

입력값 모니터링은 레이블이 도착하기 전에 작동하므로 이번 레슨의 주요 주제입니다. 다만 데모에서 볼 수 있듯이 입력값 모니터링에는 한계가 있습니다. 어떤 변화는 입력값을 전혀 건드리지 않고 지나갑니다.

### 세 가지 드리프트

입력값 `x`와 레이블 `y`의 결합 분포를 두 가지 방식으로 표현할 수 있습니다.

```math
P(x, y) = P(x) \cdot P(y \mid x) = P(y) \cdot P(x \mid y)
```

각 드리프트 유형은 서로 다른 요소를 변화시킵니다.

| 드리프트 | 변화하는 것 | 유지되는 것 | 결제 예시 | 감지하는 모니터 |
|---|---|---|---|---|
| 공변량 드리프트 | `P(x)` | `P(y \| x)` | 장바구니 규모가 더 큰 앱 사용자의 증가 | 입력값 모니터 |
| 사전 확률 드리프트 | `P(y)` | `P(x \| y)` | 공격 기간 동안 사기 비율이 2배로 증가하지만, 사기 형태는 동일함 | 예측값 및 결과 모니터 |
| 컨셉 드리프트 | `P(y \| x)` | 유지될 수 있음 | 동일한 장바구니의 사기 위험도가 달라짐 | 결과 모니터 전용 |

Gama 등은 `P(y | x)`의 변화를 실제 컨셉 드리프트(real concept drift)라고 부릅니다. `P(y | x)`는 그대로 둔 채 `P(x)`만 변하는 것은 가상 드리프트(virtual drift)라고 부릅니다. 해당 서베이 논문은 드리프트를 시간에 따른 발생 형태별로도 분류합니다. 즉, 급격한(sudden), 점진적인(incremental), 단계적인(gradual), 또는 주기적으로 되돌아오는(recurrent) 형태로 나뉩니다.

공변량 드리프트가 항상 부정적인 영향을 미치는 것은 아닙니다. 모델이 `P(y | x)`를 잘 학습했다면 각 입력에 대해 여전히 올바른 예측을 수행합니다. 그러나 트래픽이 학습 데이터가 거의 없는 영역이나 결정 경계(decision boundary) 근처로 이동하면 정확도가 떨어질 수 있습니다.

### 지연된 레이블

결과는 예측보다 늦게 도착합니다. 지연 시간은 문제에 따라 다릅니다.

| 문제 | 레이블 도착 시점 |
|---|---|
| 광고 클릭 | 수 초 |
| 향후 30일 이내 이탈 | 30일 |
| 카드 사기 | 지급 거절이 접수되는 시점 (종종 수주 후) |
| 대출 연체 | 수개월 또는 수년 |

지연 기간에는 입력값과 예측값만 존재하고 결과는 없습니다. 입력값과 예측값을 모니터링하고, 레이블이 도착하면 조인(join)하여 레이블이 확보된 부분에 대해서만 결과 지표를 계산합니다. 데모에서는 500개 이벤트의 레이블 지연을 적용하므로, 결과 모니터는 항상 과거 500개 이벤트 이전의 시점을 확인합니다.

### 참조 윈도우와 현재 윈도우

모든 드리프트 검정은 두 표본을 비교합니다.

- **참조 윈도우(Reference window).** 학습 데이터, 또는 수동으로 검증한 프로덕션 운영 기간의 데이터입니다. 모델을 재학습할 때까지 고정됩니다.
- **현재 윈도우(Current window).** 가장 최근의 이벤트들로, 최근 `N`개 이벤트이거나 최근 `T`시간 동안의 데이터입니다.

슬라이딩 윈도우(sliding window)는 윈도우 크기보다 작은 보폭으로 전진하므로 연속된 윈도우들이 서로 겹칩니다. 텀블링 윈도우(tumbling window)는 윈도우 크기만큼 전진합니다. 윈도우 크기는 트레이드오프 관계입니다. 작은 윈도우는 빠르게 반응하지만 단순 노이즈를 드리프트로 판단할 수 있습니다. 큰 윈도우는 안정적이지만 변화를 늦게 감지합니다.

계절성(seasonality)은 단순한 윈도우 방식을 무너뜨립니다. 많은 제품에서 월요일의 트래픽은 토요일의 트래픽과 다릅니다. 참조 윈도우가 전체 주기를 포괄하도록 설정하거나, 각 윈도우를 이전 주기의 동일한 기간과 비교해야 합니다.

### Population Stability Index

PSI는 신용 평가(credit scoring) 분야에서 유래했습니다. 참조 데이터를 여러 빈(bin)으로 나눕니다. 보통 참조 데이터의 십분위수를 기준으로 10개 빈으로 나누어 각 빈이 참조 데이터의 10%씩을 차지하도록 합니다. 그런 다음 각 빈에 속하는 현재 데이터의 비율을 계산합니다. 마지막으로 빈마다 다음 항을 계산하여 모두 합산합니다.

```math
\text{PSI} = \sum_{\text{bins}} (\text{current\_share} - \text{reference\_share}) \cdot \ln\left(\frac{\text{current\_share}}{\text{reference\_share}}\right)
```

어느 빈의 비율이 0이 되면 로그 값이 무한대가 되므로, 모든 비율을 0.0001과 같은 작은 값으로 클리핑합니다. PSI는 대칭적입니다. 현재 데이터에서 참조 데이터로의 KL 발산(KL divergence)과 참조 데이터에서 현재 데이터로의 KL 발산을 더한 값과 같습니다.

현업에서는 고정된 기준 구간을 두고 PSI를 해석합니다. 0.1 미만은 안정적, 0.1에서 0.25 사이는 중간 수준의 변화, 0.25 초과는 유의미한 변화로 봅니다. 이 기준 구간은 통계적 검정이라기보다는 관례로 취급해야 합니다. PSI는 빈의 수가 많아질수록 커지고, 표본 크기가 커질수록 작아집니다. 데모에서는 드리프트가 없는 500개 이벤트 윈도우의 PSI 점수가 0.01에서 0.04 사이로 측정됩니다.

### Kolmogorov-Smirnov 통계량

2표본 KS 통계량 `D`는 두 경험적 누적 분포 함수(ECDF) 사이의 최대 수직 거리입니다.

```math
D = \max_x \left\lvert F_{\text{reference}}(x) - F_{\text{current}}(x) \right\rvert
```

`D`는 두 표본이 동일할 때 0이고, 전혀 겹치지 않을 때 1입니다. 빈을 나눌 필요가 없습니다. p-값은 유효 표본 크기 `n * m / (n + m)`을 사용하는 콜모고로프 분포로부터 도출됩니다.

표본 크기가 크면 아무런 의미가 없는 사소한 격차에도 p-값이 0으로 수렴합니다. 윈도우당 100만 개의 이벤트가 있는 경우 0.02 표준편차만큼의 이동만으로도 "유의미"하다고 판정됩니다. 따라서 `D` 자체를 효과 크기(effect size)로 사용하고, `D`와 p-값이 모두 임계값을 통과할 때만 알림을 보내야 합니다.
### 범주형 특성을 위한 카이제곱 검정

결제 채널과 같은 범주형 특성(categorical feature)의 경우, 두 개의 행(기준, 현재)과 범주당 하나의 열로 구성된 교차표를 작성합니다. 각 셀의 기댓값은 행 합계와 열 합계로부터 계산됩니다:

```math
\begin{aligned}
\text{expected}[r][c] &= \text{row\_total}[r] \cdot \text{column\_total}[c] / \text{grand\_total} \\
\text{chi2} &= \sum_{\text{cells}} (\text{observed} - \text{expected})^2 / \text{expected}
\end{aligned}
```

이 검정은 `k`개의 범주에 대해 `k - 1`의 자유도를 갖습니다. 현재 윈도우에만 나타나는 범주는 큰 항을 더하므로 검정이 새로운 값에 반응하게 됩니다. 크래머 V(Cramér's V)는 효과 크기(effect size)를 나타냅니다. 행이 두 개인 표의 경우, 이는 `chi2 / grand_total`의 제곱근과 같습니다.

### 경보 임계값과 오경보

아무것도 변경되지 않았을 때에도 각 검정은 `alpha`의 오경보율(false-alarm rate)을 갖습니다. 모니터링에서는 수많은 검정이 실행됩니다. `alpha = 0.05`에서 50개의 특성을 매시간 확인하면, 아무것도 변하지 않았더라도 하루에 약 60건의 오경보가 발생할 것으로 예상됩니다. 사람들은 그보다 훨씬 전에 경보 확인을 중단하게 됩니다.

다음 네 가지 규칙을 통해 경보 발생률을 유용한 수준으로 유지할 수 있습니다:

1. **다중 검정 보정(Multiple testing correction)을 적용합니다.** 본페로니 보정(Bonferroni correction)은 `alpha`를 검정 횟수로 나눕니다. 이 데모에서는 두 개의 검정에 대해 패밀리 단위 `alpha`로 0.01을 사용하므로, 각 검정에는 0.005를 적용합니다.
2. **효과 크기 조건을 요구합니다.** PSI, `D`, 또는 Cramér's V 또한 임계값을 통과할 때만 경보를 발생시킵니다.
3. **지속성 조건을 요구합니다.** 연속된 `k`개의 드리프트 윈도우가 발생한 후에만 경보를 발생시킵니다. 데모에서는 `k = 2`를 사용합니다.
4. **경고(warning)와 경보(alert)를 분리합니다.** 경고는 대시보드로 전달되고, 경보는 담당자에게 전달됩니다.

데모에서는 규칙 2가 작동하는 모습을 보여줍니다. 정상적인 한 윈도우의 카이제곱 p-값은 0.0009로, 검정당 `alpha` 미만입니다. 하지만 Cramér's V가 0.1 미만이므로 정책은 경보가 아닌 경고를 기록합니다.

### 재학습 트리거

재학습을 위한 세 가지 일반적인 트리거는 다음과 같습니다:

- **일정(Schedule).** 매주 또는 매달 재학습합니다. 단순하지만 아무것도 바뀌지 않았을 때도 재학습을 수행하며, 변화가 생겼을 때는 대기하게 됩니다.
- **드리프트(Drift).** 중요 특성의 입력 드리프트가 임계값 이상으로 지속될 때 재학습합니다.
- **성능(Performance).** 결과 지표가 기준값에서 허용 오차를 뺀 값 아래로 떨어질 때 재학습합니다.

드리프트의 종류에 따라 재학습이 수행할 수 있는 역할이 결정됩니다:

| 드리프트 | 유용한 조치 |
|---|---|
| 공변량 드리프트(Covariate drift), 정확도 안정적 | 대개의 경우 조치가 불필요합니다. 결과 지표를 계속 모니터링합니다. |
| 데이터가 적은 영역으로의 공변량 드리프트 | 해당 영역을 포함하는 최근 데이터로 재학습합니다. |
| 사전 드리프트(Prior drift) | 점수를 재보정(recalibrate)하거나 결정 임계값을 조정합니다. |
| 개념 드리프트(Concept drift) | 먼저 새로운 레이블을 수집합니다. 이전 레이블로 재학습하면 기존 개념을 다시 학습할 뿐입니다. |

재학습 후에는 가장 최근의 레이블이 지정된 데이터에서 새 모델을 테스트합니다. 시스템 설계 단원에서 다룬 섀도(shadow) 및 카나리(canary) 단계를 통해 배포합니다. 그런 다음 기준 윈도우를 새 훈련 데이터로 재설정합니다.

```figure
mlprod-drift-psi
```

## 직접 구현해 보기

`code/main.py`의 코드는 표준 라이브러리만을 사용합니다.

### 1단계: 분위수 빈과 PSI

빈의 경곗값(edge)은 기준 데이터의 십분위수입니다. `bisect_right`는 각 값이 속하는 빈을 찾습니다.

```python
def quantile_edges(reference: list[float], bins: int = 10) -> list[float]:
    ordered = sorted(reference)
    n = len(ordered)
    edges = [ordered[min(n - 1, (n * i) // bins)] for i in range(1, bins)]
    return sorted(set(edges))


def psi_from_fractions(expected: list[float], actual: list[float], eps: float = 1e-4) -> float:
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(e, eps), max(a, eps)
        total += (a - e) * math.log(a / e)
    return total
```

### 2단계: KS 통계량과 p-값

정렬된 두 표본을 함께 탐색합니다. 각 고윳값(distinct value)마다 두 표본에서 해당 값의 모든 복사본을 지나친 다음, 두 누적 비율을 비교합니다. 이렇게 하면 동점(tie)을 올바르게 처리할 수 있습니다.

```python
def ks_statistic(sample_a: list[float], sample_b: list[float]) -> float:
    a, b = sorted(sample_a), sorted(sample_b)
    n, m = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < n and j < m:
        x = min(a[i], b[j])
        while i < n and a[i] <= x:
            i += 1
        while j < m and b[j] <= x:
            j += 1
        d = max(d, abs(i / n - j / m))
    return d
```

p-값은 `lambda = sqrt(n * m / (n + m)) * D`인 콜모고로프 분포(Kolmogorov distribution)의 급수 `2 * sum((-1)^(k-1) * exp(-2 k^2 lambda^2))`를 사용합니다.

### 3단계: 카이제곱 검정

`chi2_sf`는 정규화된 상부 불완전 감마 함수(regularized upper incomplete gamma function)를 통해 카이제곱 생존 함수(survival function)를 계산합니다. 인자(argument)가 작을 때는 급수를 사용하고, 클 때는 연분수(continued fraction)를 사용합니다. `chi_square_test`는 두 행으로 된 표를 구성하고 통계량을 계산한 뒤 p-값과 Cramér's V를 반환합니다.

```python
def chi_square_test(reference_counts: dict[str, int], current_counts: dict[str, int]) -> ChiSquareResult:
    categories = [c for c in sorted(set(reference_counts) | set(current_counts))
                  if reference_counts.get(c, 0) + current_counts.get(c, 0) > 0]
    rows = [[reference_counts.get(c, 0) for c in categories], [current_counts.get(c, 0) for c in categories]]
    row_totals = [sum(r) for r in rows]
    grand = sum(row_totals)
    if len(categories) < 2 or 0 in row_totals:
        return ChiSquareResult(0.0, 0, 1.0, 0.0)
    statistic = 0.0
    for col, _ in enumerate(categories):
        col_total = rows[0][col] + rows[1][col]
        for r in range(2):
            expected = row_totals[r] * col_total / grand
            statistic += (rows[r][col] - expected) ** 2 / expected
    dof = len(categories) - 1
    return ChiSquareResult(statistic, dof, chi2_sf(statistic, dof), math.sqrt(statistic / grand))
```

### 4단계: 윈도우와 드리프트 모니터

`SlidingWindow`는 마지막 `size`개의 이벤트를 `deque`에 유지합니다. `DriftMonitor`는 기준 빈과 카운트를 한 번 계산합니다. 이 클래스의 `check` 메서드는 각 수치형 특성에 대해 PSI, `D`, KS p-값을 반환하고, 각 범주형 특성에 대해 카이제곱 검정 결과를 반환합니다.

### 5단계: 경보 정책

이 정책은 각 지표 세트를 등급(level)으로 변환하고 신호별 연속 드리프트 윈도우 수를 계산합니다:

```python
def level(self, metrics: dict[str, float]) -> str:
    if "psi" in metrics:
        if metrics["psi"] >= self.psi_drift or (metrics["ks"] >= self.ks_drift and metrics["ks_p"] < self.alpha):
            return "drift"
        return "warn" if metrics["psi"] >= self.psi_warn else "ok"
    if metrics["p"] < self.alpha and metrics["v"] >= self.min_cramers_v:
        return "drift"
    return "warn" if metrics["p"] < self.alpha else "ok"
```

`update`는 연속 조건(streak)을 충족하는 윈도우에서 `ALERT`를 반환하고, 경보 이후 처음으로 나타난 정상 윈도우에서 `resolved`를 반환합니다.

### 6단계: 드리프트가 주입된 스트림

이 스트림은 `amount`, `channel`, 사기 여부인 `label`로 구성된 9,000건의 결제 이벤트를 포함합니다. 스트림은 네 개의 구간으로 정의됩니다:

| 이벤트 | 금액(Amount) | 채널 구성 비율(app, store, web) | 레이블 경계 | 변경 사항 |
|---|---|---|---|---|
| 0 ~ 2999 | 평균 100, 표준편차 20 | 35, 15, 50퍼센트 | 120 | 기준 기간 |
| 3000 ~ 4999 | 평균 118, 표준편차 24 | 60, 10, 30퍼센트 | 120 | 공변량 드리프트 |
| 5000 ~ 6499 | 평균 100, 표준편차 20 | 35, 15, 50퍼센트 | 120 | 정상 복귀 |
| 6500 ~ 8999 | 평균 100, 표준편차 20 | 35, 15, 50퍼센트 | 105 | 개념 드리프트 |

모델은 처음 2,000개의 이벤트로 학습된 `amount` 기반의 임계값 모델입니다. 윈도우는 500개의 이벤트를 유지하며 250개씩 이동합니다. 레이블은 500개 이벤트만큼 지연되어 도착합니다.

```bash
python3 code/main.py
```

```console
  end    PSI   KS D   chi2 p    acc  levels (amount/channel/accuracy)  events
 3000  0.010  0.041  3.0e-01  0.932  ok/ok/ok
 3250  0.152  0.163  7.6e-06  0.924  drift/warn/ok
 3500  0.463  0.298  2.8e-26  0.936  drift/drift/ok                     ALERT amount
 3750  0.609  0.343  3.5e-35  0.942  drift/drift/ok                     ALERT channel
 ...
 5500  0.026  0.044  2.6e-01  0.898  ok/ok/ok                           resolved amount, resolved channel
 ...
 7250  0.040  0.048  7.1e-01  0.848  ok/ok/drift
 7500  0.022  0.039  6.1e-01  0.778  ok/ok/drift                        ALERT accuracy
```

결과는 세 부분으로 나누어 해석할 수 있습니다:

- **공변량 드리프트.** 드리프트는 3000번 이벤트에서 시작되며, 3500번에서 `amount` 경보가 뒤따릅니다. 레이블 규칙은 변경되지 않았으므로 정확도는 0.89에서 0.95 사이를 유지합니다.
- **정상 복귀.** 5000번에서 입력이 정상으로 돌아오고, 5500번에서 두 경보 모두 해제됩니다.
- **개념 드리프트.** 드리프트는 6500번에서 시작되며, 입력값은 변경되지 않았으므로 입력 모니터는 경보를 발생시키지 않습니다. 정확도는 0.946에서 0.76~0.79 수준으로 떨어집니다. 정확도 경보는 변화 발생 1,000개 이벤트 뒤인 7500번에서 발생하며, 이 중 500개는 레이블 지연에 기인합니다.
## 활용하기

**scipy.** `scipy.stats`는 두 검정을 모두 구현하고 있습니다. 통계량은 바닥부터 구현한(from-scratch) 코드와 일치합니다. 이 레슨을 실행하는 데 scipy가 필수로 요구되지는 않습니다.

```python
from scipy import stats

result = stats.ks_2samp(reference_amounts, current_amounts)
print(result.statistic, result.pvalue)

table = [[1000, 700, 300], [140, 290, 70]]
chi2 = stats.chi2_contingency(table)
print(chi2.statistic, chi2.pvalue, chi2.dof)
```

무작위 샘플을 대상으로 `ks_statistic`와 `chi_square_test`를 scipy 1.18.1과 비교 검증했습니다. KS 통계량, 카이제곱 통계량, 카이제곱 p-값은 부동소수점 정밀도 수준까지 일치했습니다. 200개의 무작위 샘플 쌍에 걸쳐 KS p-값은 scipy 값 대비 최대 18%까지 차이가 났습니다. 가장 큰 차이는 p-값이 매우 작을 때 발생했습니다.

이 차이는 계산 방식에서 기인합니다. 우리가 작성한 코드는 극한 콜모고로프 분포(limiting Kolmogorov distribution)를 사용합니다. 기본적으로 `ks_2samp`는 두 샘플 모두 10,000개 이하의 값을 가질 때 정확한(exact) 방식을 사용합니다. 점근적(asymptotic) 모드에서는 반올림된 유효 표본 크기를 갖는 유한 표본 분포를 사용합니다. 두 개의 범주가 있는 경우, `chi2_contingency`는 기본적으로 예이츠의 연속성 보정(Yates' continuity correction)을 적용하므로 통계량에 차이가 발생합니다.

**모니터링 라이브러리.** 여러 오픈소스 라이브러리가 이러한 검정을 리포트 및 대시보드와 함께 제공합니다:

- **Evidently**는 컬럼별로 드리프트 검정을 선택합니다. 1,000개 이하의 행을 가진 참조 데이터에 대해서는 수치형 컬럼에 KS 검정을, 범주형 컬럼에 카이제곱 검정을 사용합니다. 더 큰 데이터의 경우 임계값 0.1의 바서슈타인 거리(Wasserstein distance)와 젠센-섀넌 발산(Jensen-Shannon divergence)을 사용합니다. PSI 또한 선택 가능한 방법 중 하나입니다.
- **NannyML**은 레이블이 도착하기 전에 모델 성능을 추정합니다. 신뢰도 기반 성능 추정(Confidence-based Performance Estimation, CBPE)은 잘 보정(calibrated)되어 있어야 하는 모델의 예측 확률을 사용합니다. NannyML 문서에 따르면 CBPE는 공변량 변화(covariate shift)를 처리하지만 개념 드리프트(concept drift)에는 대처하지 못합니다. 데모 스트림이 그 이유를 보여줍니다. 레이블만이 개념 드리프트를 드러냈기 때문입니다.
- **TensorFlow Data Validation**은 데이터의 연속된 구간을 비교합니다. 범주형 특성에는 L-무한대 거리를, 수치형 특성에는 근사 젠센-섀넌 발산을 사용합니다. 또한 피처 스토어 레슨에서 다루는 학습 데이터와 서빙 데이터의 비교도 수행합니다.

바닥부터 구현한 코드는 이러한 도구들이 각각 무엇을 계산하는지 보여줍니다. 도구들은 스토리지, 스케줄링, 대시보드, 그리고 훨씬 더 많은 검정을 추가로 제공합니다.

LLM 애플리케이션의 경우 인프라 단계의 [LLM 관측 가능성(observability) 레슨](../../../17-infrastructure-and-production/13-llm-observability/docs/en.md)에서 트레이싱 및 평가 도구를 다룹니다. 이 레슨의 개념들은 해당 시스템의 입력과 출력에도 여전히 적용됩니다.

## 실무 적용

이 레슨은 `outputs/skill-drift-monitoring.md`을 생성합니다. 이 스킬은 새 모델을 위한 모니터를 계획하고 드리프트 알림을 분류(triage)합니다. 어떤 계층에서 알림이 발생했는지, 증거에 부합하는 드리프트 유형은 무엇인지, 대기·재보정·레이블 수집·재학습 중 올바른 조치가 무엇인지 질문합니다.

## 실습 과제

1. 200개 및 2000개 이벤트 윈도우로 스트림을 실행하세요. 이벤트 3000 이전의 거짓 경고(false warning)와 알림 지연 시간을 기록하세요.
2. `n_tests=1`, `patience=1`, `min_cramers_v=0`를 설정하세요. 5000부터 6500까지의 정상 기간 동안 발생한 알림 수를 세어보세요.
3. 100개 이벤트당 평균이 0.01씩 증가하는 특성을 추가하세요. PSI와 KS 중 어떤 신호가 먼저 반응하는지 확인하세요.
4. 동일한 10분위수 빈(bin)에 대해 젠센-섀넌 거리를 구현하세요. 모든 윈도우에서 이를 PSI와 비교하세요.
5. 모델의 양성 예측 비율(positive-prediction rate)에 대한 모니터를 추가하세요. 개념 드리프트가 발생하는 동안 이 모니터가 왜 조용히 유지되는지 설명하세요.

## 핵심 용어

| 용어 | 일상적인 표현 | 의미 |
|------|----------------|----------------------|
| 데이터 드리프트(Data drift) | "데이터가 변경됨" | 참조 윈도우와 비교했을 때 입력, 예측 또는 레이블 분포의 변화 |
| 공변량 드리프트(Covariate drift) | "새로운 유형의 사용자" | `P(y \| x)`는 그대로 유지되면서 `P(x)`가 변경됨 |
| 사전 확률 드리프트(Prior drift) | "이전보다 양성이 더 많아짐" | `P(x \| y)`는 그대로 유지되면서 `P(y)`가 변경됨, 레이블 시프트(label shift)라고도 함 |
| 개념 드리프트(Concept drift) | "게임의 규칙이 바뀜" | `P(y \| x)`가 변경되어 동일한 입력에 대해 이제 다른 정답이 나옴 |
| 참조 윈도우(Reference window) | "기준 데이터" | 모든 현재 윈도우와 비교 대상이 되는 고정된 샘플(주로 학습 데이터) |
| PSI | "안정성 점수" | 빈 전반에 걸쳐 (현재 점유율 - 참조 점유율)에 두 점유율 비율의 로그를 곱한 대칭 합 |
| KS 통계량(KS statistic) | "KS 검정" | 두 경험적 누적 분포 함수(ECDF) 간의 가장 큰 차이 |
| 크래머의 V(Cramér's V) | "범주형 데이터의 효과 크기" | 0에서 1 사이의 범위로 정규화된 카이제곱 통계량 |
| 레이블 지연(Label delay) | "정답이 나중에 도착함" | 예측 시점과 실제 결과 도착 시점 사이의 시간 |
| 알림 피로(Alert fatigue) | "호출이 너무 많음" | 대부분이 오경보(false alarm)였기 때문에 알림을 무시하게 되는 현상 |

## 더 읽을거리

- [Gama et al., A Survey on Concept Drift Adaptation (ACM Computing Surveys, 2014)](https://research.tue.nl/en/publications/a-survey-on-concept-drift-adaptation): 실제 드리프트와 가상 드리프트, 드리프트 패턴 및 적응 방법
- [Rabanser, Günnemann, and Lipton, Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift (NeurIPS 2019)](https://arxiv.org/abs/1810.11953): 시프트 감지를 위한 이표본 검정(two-sample tests)
- [Lu et al., Learning under Concept Drift: A Review (2020)](https://arxiv.org/abs/2004.05785): 드리프트 감지, 이해 및 적응 방법
- [Yurdakul, Statistical Properties of Population Stability Index (Western Michigan University, 2018)](https://scholarworks.wmich.edu/dissertations/3208/): 샘플 크기 및 빈 개수에 따른 PSI의 통계적 특성
- [scipy.stats.ks_2samp 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html): 이표본 KS 검정을 위한 정확한(exact) 및 점근적(asymptotic) p-값 계산 방식
- [Evidently, 데이터 드리프트 감지 작동 방식](https://docs.evidentlyai.com/metrics/explainer_drift): 컬럼 유형 및 데이터 크기에 따른 기본 검정 선택 방식
- [NannyML, 성능 추정](https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html): CBPE와 그 가정사항
