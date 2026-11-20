---
title_en: "Monitoring Models in Production — Data Drift and Concept Drift"
source: "phases/02-ml-fundamentals/20-model-monitoring-and-drift"
source_sha: "c7d059e55efd1e93"
model: "gemini-3.8-flash"
translated_at: "2026-11-20"
---
# 프로덕션 모델 모니터링 — 데이터 드리프트와 컨셉 드리프트

> 배포된 모델은 오류 메시지 없이 실패합니다. 세상이 훈련 데이터로부터 멀어지는 동안에도 모델은 계속해서 점수를 반환하며, 오직 모니터링만이 그 격차를 드러냅니다.

**유형:** Build (실습)
**사용 언어:** Python, TypeScript
**선수 레슨:** Phase 2, Lessons 09 (Model Evaluation), 16 (Anomaly Detection), and 19 (ML System Design)
**소요 시간:** ~110분

## 학습 목표

- 네 가지 계층(시스템 상태, 입력, 예측, 결과)에서 모니터링 신호 선택하기
- 결합 분포 중 변화하는 부분을 기준으로 공변량 드리프트, 사전 드리프트, 컨셉 드리프트 구분하기
- 모집단 안정성 지수(PSI), 2표본 콜모고로프-스미르노프(KS) 통계량, 카이제곱 검정을 밑바닥부터 구현하기
- 고정된 참조 윈도우와 슬라이딩 현재 윈도우를 비교하고, 윈도우 크기가 노이즈와 지연에 미치는 영향 설명하기
- 효과 크기, 다중 검정 보정, 지속성 규칙을 활용하여 오경보를 제한하는 알림 정책 설계하기
- 재훈련이 필요한 드리프트, 새 레이블이 필요한 드리프트, 조치가 필요 없는 드리프트를 판별하기

## 문제 상황

결제 위험 평가 모델이 95%의 정확도로 배포됩니다. 2개월 후, 마케팅 캠페인으로 인해 장바구니 규모가 더 큰 앱 사용자들이 유입됩니다. 4개월 후, 사기범들이 수법을 바꾸면서 이전에는 안전했던 결제들이 지급 거절(chargeback)로 이어지기 시작합니다.

이 모든 과정 동안 모델 서비스는 모든 요청에 대해 HTTP 200을 반환합니다. 지연 시간(latency)은 정상이며, 로그에는 아무런 오류도 기록되지 않습니다. 문제가 발생했다는 첫 번째 징후는 피해가 발생한 지 수 주일이 지난 후에야 나오는 지급 거절 관련 재무 보고서뿐입니다.

소프트웨어 모니터링은 서비스가 응답하는지 확인합니다. 모델 모니터링은 그 응답이 여전히 올바른지, 그리고 데이터가 모델을 훈련했던 데이터와 여전히 유사한지 확인합니다. 이번 레슨에서는 이 두 번째 작업을 위한 통계량, 윈도우, 알림 정책을 구축합니다. 두 가지 변화가 주입된 합성 스트림에서 이를 실행하고, 어떤 모니터가 어떤 변화에 반응하는지 관찰합니다.

## 핵심 개념

### 모니터링 대상

네 가지 계층을 모니터링합니다. 처음 세 계층은 조기 경보를 제공합니다. 네 번째 계층만이 정확성을 측정하며, 늦게 도착합니다.

| 계층 | 신호 | 사용 가능 시점 | 답을 제공하는 질문 |
|---|---|---|---|
| 시스템 상태 | p50 및 p99 지연 시간, 오류율, 처리량, 메모리 | 즉시 | 서비스가 응답하는가? |
| 입력 | 각 특성의 분포, 결측값 비율, 스키마, 새로운 범주 | 즉시 | 라이브 데이터가 훈련 데이터와 유사한가? |
| 예측 | 점수 분포, 양성 비율, 기본값 응답 비율 | 즉시 | 모델이 이전과 동일하게 동작하는가? |
| 결과 | 정확도, 정밀도, 캘리브레이션, 비즈니스 지표 | 레이블 지연 이후 | 모델이 여전히 올바른가? |

입력 모니터링은 레이블이 도착하기 전에 작동하므로 이번 레슨의 주요 주제입니다. 데모에서 보여주듯이 입력 모니터링에는 한계가 있습니다. 어떤 변화는 입력에 전혀 영향을 주지 않습니다.

### 세 가지 종류의 드리프트

입력 `x`과 레이블 `y`의 결합 분포를 두 가지 방식으로 표현합니다.

```math
P(x, y) = P(x) \cdot P(y \mid x) = P(y) \cdot P(x \mid y)
```

각 드리프트 유형은 서로 다른 요소를 변화시킵니다.

| 드리프트 | 변화하는 것 | 유지되는 것 | 결제 예시 | 감지하는 모니터 |
|---|---|---|---|---|
| 공변량 드리프트(Covariate drift) | `P(x)` | `P(y \| x)` | 장바구니 규모가 더 큰 앱 사용자 증가 | 입력 모니터 |
| 사전 드리프트(Prior drift) | `P(y)` | `P(x \| y)` | 공격 기간 동안 사기 비율이 2배로 증가하지만, 사기 형태는 동일함 | 예측 및 결과 모니터 |
| 컨셉 드리프트(Concept drift) | `P(y \| x)` | 유지될 수 있음 | 동일한 장바구니가 이제는 다른 사기 위험도를 가짐 | 결과 모니터 전용 |

Gama 등은 `P(y | x)`의 변화를 실제 컨셉 드리프트(real concept drift)라고 부릅니다. 이들은 `P(y | x)`을 그대로 둔 채 `P(x)`만 변하는 것을 가상 드리프트(virtual drift)라고 부릅니다. 해당 서베이에서는 드리프트가 시간에 따라 발생하는 방식(돌발적, 점진적, 점차적, 또는 되돌아오는 컨셉)에 따라 분류하기도 합니다.

공변량 드리프트가 항상 악영향을 미치는 것은 아닙니다. 모델이 `P(y | x)`를 잘 학습했다면, 각 입력에 대해 여전히 올바른 결과를 냅니다. 하지만 트래픽이 훈련 데이터가 적은 영역이나 결정 경계에 가까운 영역으로 이동하면 여전히 정확도가 떨어질 수 있습니다.

### 지연된 레이블

결과는 예측 이후에 도착합니다. 지연 시간은 문제에 따라 다릅니다.

| 문제 | 레이블 도착 시점 |
|---|---|
| 광고 클릭 | 수 초 |
| 향후 30일 이내 이탈 | 30일 |
| 카드 사기 | 지급 거절이 접수될 때 (보통 수 주일 후) |
| 대출 연체 | 수개월 또는 수년 |

지연 기간 동안에는 입력과 예측값은 있지만 결과는 없습니다. 입력과 예측을 모니터링하고, 레이블이 도착하면 이를 결합(join)하여 레이블이 지정된 부분에 대해서만 결과 지표를 계산합니다. 데모에서는 500개 이벤트의 레이블 지연을 사용하므로 결과 모니터는 항상 과거 500개 이벤트를 되돌아봅니다.

### 참조 윈도우와 현재 윈도우

모든 드리프트 검정은 두 표본을 비교합니다.

- **참조 윈도우(Reference window).** 훈련 데이터 또는 수동으로 검증한 프로덕션 기간입니다. 모델을 재훈련하기 전까지 고정된 상태로 유지됩니다.
- **현재 윈도우(Current window).** 가장 최근 이벤트들이며, 최근 `N`개 이벤트이거나 최근 `T`시간입니다.

슬라이딩 윈도우(sliding window)는 크기보다 작은 스텝으로 앞으로 이동하므로 연속된 윈도우들이 서로 겹칩니다. 텀블링 윈도우(tumbling window)는 전체 크기만큼 이동합니다. 윈도우 크기는 트레이드오프 관계입니다. 작은 윈도우는 빠르게 반응하지만 노이즈를 드리프트로 판정할 수 있습니다. 큰 윈도우는 안정적이지만 변화를 늦게 보고합니다.

계절성(seasonality)은 단순한 윈도우를 무력화합니다. 많은 제품에서 월요일 트래픽은 토요일 트래픽과 다릅니다. 참조 윈도우가 전체 주기를 포괄하도록 만들거나, 각 윈도우를 이전 주기의 동일한 기간과 비교하십시오.

### 모집단 안정성 지수(PSI)

PSI(Population Stability Index)는 신용 평가(credit scoring)에서 유래했습니다. 참조 값을 보통 참조 십분위수 기준 10개 빈(bin)으로 나누어, 각 빈이 참조 데이터의 10%를 포함하도록 합니다. 각 빈에서 현재 값의 비율을 계산합니다. 그런 다음 빈마다 하나의 항을 더합니다.

```math
\text{PSI} = \sum_{\text{bins}} (\text{current\_share} - \text{reference\_share}) \cdot \ln\left(\frac{\text{current\_share}}{\text{reference\_share}}\right)
```

비율이 0인 빈은 로그를 무한대로 만드므로, 모든 비율을 0.0001과 같은 작은 값으로 클리핑(clip)합니다. PSI는 대칭적입니다. 이는 현재에서 참조로의 KL 발산(Kullback-Leibler divergence)에 참조에서 현재로의 KL 발산을 더한 것과 같습니다.

실무자들은 고정된 기준 구간으로 PSI를 해석합니다. 0.1 미만은 안정적이고, 0.1에서 0.25는 보통 수준의 변화이며, 0.25 초과는 상당한 변화입니다. 이러한 구간은 검정이 아닌 관례로 취급하십시오. PSI는 빈의 수가 많을수록 커지고 표본 크기가 클수록 작아집니다. 데모에서 드리프트가 없는 500개 이벤트 윈도우는 0.01에서 0.04 사이의 점수를 기록합니다.

### 콜모고로프-스미르노프 통계량

2표본 KS 통계량 `D`은 두 경험적 누적 분포 함수(empirical cumulative distribution function) 간의 가장 큰 수직 격차입니다.

```math
D = \max_x \left\lvert F_{\text{reference}}(x) - F_{\text{current}}(x) \right\rvert
```

`D`은 동일한 표본에 대해서는 0이고 서로 겹치지 않는 표본에 대해서는 1입니다. 빈을 나눌 필요가 없습니다. p-값은 유효 표본 크기 `n * m / (n + m)`을 사용하는 콜모고로프 분포로부터 도출됩니다.

표본 크기가 크면 아무도 신경 쓰지 않을 만큼 미미한 격차에 대해서도 p-값이 0으로 수렴합니다. 윈도우당 100만 개의 이벤트가 있다면 0.02 표준편차의 변화도 "유의미"해집니다. `D` 자체를 효과 크기(effect size)로 사용하고, `D`과 p-값이 모두 각자의 임계값을 통과할 때만 알림을 트리거하십시오.
### 범주형 피처에 대한 카이제곱 검정

결제 수단과 같은 범주형(categorical) 피처의 경우 두 개의 행(기준, 현재)과 각 범주에 해당하는 열로 구성된 표를 작성합니다. 각 셀의 기대 빈도는 행과 열의 합계로부터 구합니다.

```math
\begin{aligned}
\text{expected}[r][c] &= \text{row\_total}[r] \cdot \text{column\_total}[c] / \text{grand\_total} \\
\text{chi2} &= \sum_{\text{cells}} (\text{observed} - \text{expected})^2 / \text{expected}
\end{aligned}
```

이 검정은 `k`개의 범주에 대해 `k - 1`의 자유도(degrees of freedom)를 가집니다. 현재 윈도우에만 나타나는 범주는 큰 항을 추가하므로, 이 검정은 새로운 값의 등장에 민감하게 반응합니다. 크래머 V(Cramér's V)는 효과 크기(effect size)를 나타냅니다. 두 개의 행을 가진 표의 경우, 이는 `chi2 / grand_total`의 제곱근과 같습니다.

### 알림 임계값과 오경보

각 검정은 아무것도 변하지 않았을 때도 오경보율(false-alarm rate) `alpha`을 가집니다. 모니터링은 수많은 검정을 실행합니다. 50개의 피처를 매시간 `alpha = 0.05`에서 점검한다면, 아무것도 변하지 않았더라도 하루에 약 60건의 오경보가 발생할 것으로 예상할 수 있습니다. 사람들은 그보다 훨씬 전에 알림을 확인하지 않게 됩니다.

네 가지 규칙을 적용하면 알림 비율을 유용한 수준으로 유지할 수 있습니다.

1. **다중 검정 보정(Multiple testing correction)을 적용합니다.** 본페로니 보정(Bonferroni correction)은 `alpha`를 검정 횟수로 나눕니다. 이 데모에서는 두 개의 검정에 대해 0.01의 패밀리 단위 `alpha`를 사용하므로 각 검정에는 0.005가 적용됩니다.
2. **효과 크기를 요구합니다.** PSI, `D` 또는 크래머 V 또한 임계값을 넘을 때만 알림을 트리거합니다.
3. **지속성을 요구합니다.** `k`회 연속으로 드리프트 윈도우가 발생한 후에만 알림을 보냅니다. 데모에서는 `k = 2`를 사용합니다.
4. **경고(warning)와 알림(alert)을 분리합니다.** 경고는 대시보드로 전달되고, 알림은 담당자에게 전달됩니다.

데모에서는 규칙 2가 작동하는 모습을 보여줍니다. 정상적인 한 윈도우의 카이제곱 p-값이 0.0009로, 개별 검정 기준인 `alpha`보다 낮습니다. 하지만 크래머 V가 0.1 미만이므로 정책상 알림이 아닌 경고만 기록됩니다.

### 재학습 트리거

재학습(retraining)을 트리거하는 일반적인 세 가지 방식은 다음과 같습니다.

- **일정 기반(Schedule).** 매주 또는 매월 재학습을 수행합니다. 간단하지만 아무것도 변하지 않았을 때도 재학습을 수행하고, 변화가 생겼을 때도 정해진 시간까지 기다려야 합니다.
- **드리프트 기반(Drift).** 주요 피처의 입력 드리프트가 임계값을 초과한 상태로 유지될 때 재학습을 수행합니다.
- **성능 기반(Performance).** 결과 지표가 기준값에서 허용 오차를 뺀 값 아래로 떨어질 때 재학습을 수행합니다.

드리프트의 종류에 따라 재학습이 취할 수 있는 유용한 조치가 달라집니다.

| 드리프트 | 유용한 조치 |
|---|---|
| 공변량 드리프트(covariate drift), 정확도 안정적 | 대개 조치가 필요하지 않습니다. 결과 지표를 계속 관찰합니다. |
| 데이터가 적은 영역으로의 공변량 드리프트 | 해당 영역을 포함하는 최신 데이터로 재학습합니다. |
| 사전 드리프트(prior drift) | 점수를 재보정하거나 결정 임계값(decision threshold)을 조정합니다. |
| 개념 드리프트(concept drift) | 먼저 새로운 레이블을 수집합니다. 과거 레이블로 재학습하면 이전 개념을 다시 학습하게 됩니다. |

재학습 후에는 가장 최근의 레이블이 지정된 데이터에서 새 모델을 테스트합니다. 시스템 설계 단원에서 다룬 섀도(shadow) 및 카나리(canary) 단계를 통해 배포합니다. 그런 다음 새 훈련 데이터를 기준으로 기준 윈도우를 재설정합니다.

```figure
mlprod-drift-psi
```

## 직접 구현해 보기

`code/main.py`의 코드는 표준 라이브러리만을 사용합니다.

### 1단계: 분위수 빈과 PSI

빈의 경계는 기준 데이터의 십분위수(decile)입니다. `bisect_right`는 각 값이 속하는 빈을 찾습니다.

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

정렬된 두 표본을 함께 탐색합니다. 고유한 값이 나타날 때마다 두 표본 모두에서 해당 값의 중복 항목을 지나친 후, 두 누적 비율을 비교합니다. 이렇게 하면 동점(tie)을 올바르게 처리할 수 있습니다.

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

p-값은 `lambda = sqrt(n * m / (n + m)) * D`을 사용하는 콜모고로프 분포의 급수 `2 * sum((-1)^(k-1) * exp(-2 k^2 lambda^2))`를 통해 구합니다.

### 3단계: 카이제곱 검정

`chi2_sf`는 정규화된 상부 불완전 감마 함수(regularized upper incomplete gamma function)를 통해 카이제곱 생존 함수(survival function)를 계산합니다. 작은 인자에는 급수를, 큰 인자에는 연분수(continued fraction)를 사용합니다. `chi_square_test`는 2행 표를 만들고 통계량을 계산하여 p-값과 크래머 V를 반환합니다.

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

`SlidingWindow`는 `deque` 안에 마지막 `size`개의 이벤트를 유지합니다. `DriftMonitor`는 기준 빈과 빈도를 한 번 계산합니다. 이 클래스의 `check` 메서드는 각 수치형 피처에 대해 PSI, `D`, KS p-값을 반환하고, 각 범주형 피처에 대해 카이제곱 검정 결과를 반환합니다.

### 5단계: 알림 정책

정책은 각 지표 세트를 등급으로 변환하고 시그널별로 연속 드리프트 윈도우 수를 계산합니다.

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

`update`는 연속 발생 조건을 채운 윈도우에서는 `ALERT`를 반환하고, 알림 발생 후 처음으로 나타난 정상 윈도우에서는 `resolved`를 반환합니다.

### 6단계: 드리프트가 주입된 스트림

스트림에는 `amount`, `channel`, 사기 여부 `label`를 포함한 9,000건의 결제 이벤트가 있습니다. 스트림은 네 개의 구간으로 정의됩니다.

| 이벤트 | 금액 | 채널 비율 (app, store, web) | 레이블 경계 | 변화 내용 |
|---|---|---|---|---|
| 0 ~ 2999 | 평균 100, 표준편차 20 | 35, 15, 50% | 120 | 기준 기간 |
| 3000 ~ 4999 | 평균 118, 표준편차 24 | 60, 10, 30% | 120 | 공변량 드리프트 |
| 5000 ~ 6499 | 평균 100, 표준편차 20 | 35, 15, 50% | 120 | 정상 복구 |
| 6500 ~ 8999 | 평균 100, 표준편차 20 | 35, 15, 50% | 105 | 개념 드리프트 |

모델은 처음 2,000개 이벤트에 맞춰 피팅된 `amount`에 대한 임계값 모델입니다. 윈도우는 500개의 이벤트를 포함하며 250개씩 이동합니다. 레이블은 500개 이벤트만큼 지연되어 도착합니다.

```bash
python3 code/main.py
npx tsx code/main.ts
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

결과는 세 부분으로 나누어 해석할 수 있습니다.

- **공변량 드리프트.** 이벤트 3000에서 드리프트가 시작되고, 3500에서 `amount` 알림이 뒤따릅니다. 레이블 규칙은 변경되지 않았으므로 정확도는 0.89에서 0.95 사이를 유지합니다.
- **복구.** 5000에서 입력값이 정상으로 돌아오고, 5500에서 두 알림이 모두 해제됩니다.
- **개념 드리프트.** 이벤트 6500에서 드리프트가 시작되며, 입력값 자체는 변하지 않았기 때문에 입력 모니터는 아무 반응도 보이지 않습니다. 정확도는 0.946에서 0.76~0.79 사이로 떨어집니다. 정확도 알림은 변화가 발생하고 1,000개 이벤트가 지난 7500에 발생하며, 이 중 500개는 레이블 지연 때문입니다.
## 활용하기

**scipy.** `scipy.stats`는 두 검정을 모두 구현하고 있습니다. 통계량은 밑바닥부터 구현한 코드와 일치합니다. 이번 레슨을 실행하는 데 scipy가 필수로 요구되지는 않습니다.

```python
from scipy import stats

result = stats.ks_2samp(reference_amounts, current_amounts)
print(result.statistic, result.pvalue)

table = [[1000, 700, 300], [140, 290, 70]]
chi2 = stats.chi2_contingency(table)
print(chi2.statistic, chi2.pvalue, chi2.dof)
```

무작위 샘플을 대상으로 `ks_statistic` 및 `chi_square_test`를 scipy 1.18.1과 대조하여 검증했습니다. KS 통계량, 카이제곱 통계량, 카이제곱 p-값은 부동소수점 정밀도 수준까지 일치했습니다. 200개의 무작위 샘플 쌍에 대해 KS p-값은 scipy 값 대비 최대 18% 차이가 났습니다. 가장 큰 차이는 p-값이 매우 작을 때 나타났습니다.

이러한 차이는 방법론에서 비롯됩니다. 본 레슨의 코드는 극한 콜모고로프 분포(limiting Kolmogorov distribution)를 사용합니다. 기본적으로 `ks_2samp`는 두 샘플의 크기가 모두 10,000개 이하일 때 정확한 방법(exact method)을 사용합니다. 점근 모드(asymptotic mode)에서는 반올림된 유효 표본 크기를 갖는 유한 표본 분포(finite-sample distribution)를 사용합니다. 두 개의 범주가 있을 때 `chi2_contingency`는 기본적으로 예이츠 연속성 보정(Yates' continuity correction)을 적용하므로 통계량에서 차이가 발생합니다.

**모니터링 라이브러리.** 여러 오픈소스 라이브러리가 이러한 검정을 리포트 및 대시보드와 함께 패키지로 제공합니다:

- **Evidently**는 컬럼별로 드리프트 검정을 선택합니다. 1,000개 이하의 행을 가진 참조 데이터의 경우 수치형 컬럼에는 KS 검정을, 범주형 컬럼에는 카이제곱 검정을 사용합니다. 더 큰 데이터의 경우 임계값 0.1의 바서슈타인 거리(Wasserstein distance)와 옌센-섀넌 발산(Jensen-Shannon divergence)을 사용합니다. PSI는 선택 가능한 다른 방법 중 하나입니다.
- **NannyML**은 레이블이 도착하기 전에 모델 성능을 추정합니다. 신뢰도 기반 성능 추정(Confidence-based Performance Estimation, CBPE)은 잘 보정(calibrated)되어 있어야 하는 모델의 예측 확률을 사용합니다. NannyML 문서에 따르면 CBPE는 공변량 변화(covariate shift)를 처리하지만 개념 드리프트(concept drift)에는 대처하지 못합니다. 데모 스트림이 그 이유를 보여줍니다. 개념 드리프트는 레이블을 통해서만 드러났기 때문입니다.
- **TensorFlow Data Validation**은 연속된 데이터 구간을 비교합니다. 범주형 특성에는 L-무한대 거리(L-infinity distance)를, 수치형 특성에는 근사 옌센-섀넌 발산을 사용합니다. 또한 특성 저장소(feature store) 레슨에서 다루는 학습 데이터와 서빙 데이터의 비교도 수행합니다.

밑바닥부터 작성된 코드는 이러한 각 도구가 무엇을 계산하는지 보여줍니다. 도구들은 여기에 스토리지, 스케줄링, 대시보드, 그리고 훨씬 더 많은 검정을 추가합니다.

LLM 애플리케이션의 경우 인프라 단계의 [LLM 관찰 가능성(observability) 레슨](../../../17-infrastructure-and-production/13-llm-observability/docs/en.md)에서 트레이싱 및 평가 도구를 다룹니다. 본 레슨의 개념들은 해당 시스템의 입력과 출력에도 여전히 적용됩니다.

## 실무 적용

이번 레슨을 통해 `outputs/skill-drift-monitoring.md`을 제작합니다. 이 스킬은 새로운 모델을 위한 모니터를 계획하고 드리프트 알림을 분류(triage)합니다. 어떤 계층에서 경고가 발생했는지, 증거에 부합하는 드리프트 유형은 무엇인지, 그리고 대기, 재보정, 레이블 수집, 재학습 중 올바른 조치가 무엇인지 질문합니다.

## 실습 과제

1. 200개 및 2,000개 이벤트 윈도우로 스트림을 실행하세요. 3,000번째 이벤트 이전의 오경보와 알림 지연 시간을 기록하세요.
2. `n_tests=1`, `patience=1`, `min_cramers_v=0`로 설정하세요. 5,000번째부터 6,500번째까지의 정상 구간에서 알림 발생 횟수를 세어보세요.
3. 100개 이벤트당 평균이 0.01씩 증가하는 특성을 추가하세요. PSI와 KS 중 어떤 신호가 먼저 반응하는지 확인하세요.
4. 동일한 십분위수 빈(decile bins)에 대해 옌센-섀넌 거리를 구현하세요. 모든 윈도우에서 이를 PSI와 비교하세요.
5. 모델의 양성 예측률(positive-prediction rate)에 대한 모니터를 추가하세요. 개념 드리프트가 발생하는 동안 이 모니터가 왜 조용히 유지되는지 설명하세요.

## 핵심 용어

| 용어 | 일상적인 표현 | 의미 |
|------|----------------|----------------------|
| 데이터 드리프트(Data drift) | "데이터가 바뀌었다" | 참조 윈도우와 비교하여 입력, 예측 또는 레이블의 분포가 변화하는 것 |
| 공변량 드리프트(Covariate drift) | "새로운 유형의 사용자들" | `P(y \| x)`는 동일하게 유지되면서 `P(x)`가 변화하는 것 |
| 사전 확률 드리프트(Prior drift) | "이전보다 양성이 많아졌다" | `P(x \| y)`는 동일하게 유지되면서 `P(y)`가 변화하는 것이며, 레이블 이동(label shift)이라고도 함 |
| 개념 드리프트(Concept drift) | "게임의 규칙이 바뀌었다" | `P(y \| x)`가 변화하여 동일한 입력에 대해 이제 다른 정답을 갖는 것 |
| 참조 윈도우(Reference window) | "기준 데이터" | 현재의 모든 윈도우와 비교되는 고정된 샘플(주로 학습 데이터) |
| PSI | "안정성 점수" | 빈(bin)별 (현재 점유율 - 참조 점유율)에 두 점유율 비율의 로그 값을 곱한 대칭 합 |
| KS 통계량(KS statistic) | "KS 검정" | 두 경험적 누적 분포 함수(ECDF) 간의 가장 큰 차이 |
| 크래머의 V(Cramér's V) | "범주형에 대한 효과 크기" | 0에서 1 사이의 범위로 스케일링된 카이제곱 통계량 |
| 레이블 지연(Label delay) | "정답은 나중에 나온다" | 예측 시점과 실제 결과가 도착하는 시점 사이의 시간 |
| 알림 피로(Alert fatigue) | "호출이 너무 많이 온다" | 대부분이 거짓 경보(false alarm)여서 사람들이 알림을 무시하게 되는 현상 |

## 더 읽을거리

- [Gama et al., A Survey on Concept Drift Adaptation (ACM Computing Surveys, 2014)](https://research.tue.nl/en/publications/a-survey-on-concept-drift-adaptation): 실제 드리프트 및 가상 드리프트, 드리프트 패턴, 적응 기법
- [Rabanser, Günnemann, and Lipton, Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift (NeurIPS 2019)](https://arxiv.org/abs/1810.11953): 데이터셋 이동 감지를 위한 이표본 검정
- [Lu et al., Learning under Concept Drift: A Review (2020)](https://arxiv.org/abs/2004.05785): 개념 드리프트 감지, 이해 및 적응 기법
- [Yurdakul, Statistical Properties of Population Stability Index (Western Michigan University, 2018)](https://scholarworks.wmich.edu/dissertations/3208/): 샘플 크기 및 빈 개수에 따른 PSI의 동작 특성
- [scipy.stats.ks_2samp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html): 이표본 KS 검정을 위한 정확한 p-값 및 점근적 p-값 계산 방법
- [Evidently, How data drift detection works](https://docs.evidentlyai.com/metrics/explainer_drift): 컬럼 유형 및 데이터 크기에 따른 기본 검정 선택 방식
- [NannyML, Performance estimation](https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html): CBPE와 그 가정 사항들
