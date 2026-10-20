---
title_en: "Decision Trees and Random Forests"
source: "phases/02-ml-fundamentals/04-decision-trees"
source_sha: "4b0f504a762eee19"
model: "gemini-3.8-flash"
translated_at: "2026-10-20"
---
# 의사결정 트리와 랜덤 포레스트

> 의사결정 트리는 순서도에 불과합니다. 하지만 이들이 모여 숲을 이루면 머신러닝에서 가장 강력한 도구 중 하나가 됩니다.

**유형:** Build (실습)
**Language:** Python
**선수 레슨:** Phase 1 (Lessons 09 Information Theory, 06 Probability)
**소요 시간:** ~90분

## 학습 목표

- 지니 불순도(Gini impurity), 엔트로피(entropy), 정보 획득량(information gain) 계산을 구현하여 최적의 의사결정 트리 분할을 찾습니다.
- 사전 가지치기(pre-pruning) 제어 조건(최대 깊이, 최소 샘플 수)을 포함한 의사결정 트리 분류기를 밑바닥부터 구현합니다.
- 부트스트랩 샘플링(bootstrap sampling)과 특성 무작위 선택(feature randomization)을 사용하여 랜덤 포레스트(random forest)를 구축하고, 이것이 분산(variance)을 줄이는 이유를 설명합니다.
- MDI 특성 중요도와 순열 중요도(permutation importance)를 비교하고, MDI에 편향이 발생하는 경우를 파악합니다.

## 해결하려는 문제

정형 데이터(tabular data)가 주어졌다고 가정해 보겠습니다. 행은 샘플이고, 열은 특성(feature)이며, 예측하려는 타깃(target) 열이 존재합니다. 신경망을 적용할 수도 있겠지만, 정형 데이터에서는 트리 기반 모델(의사결정 트리, 랜덤 포레스트, 그래디언트 부스팅 트리)이 딥러닝보다 일관되게 우수한 성능을 보입니다. 정형 데이터를 다루는 Kaggle 경진대회는 Transformer가 아니라 XGBoost와 LightGBM이 지배하고 있습니다.

왜 그럴까요? 트리는 전처리 없이도 혼합된 특성 유형(수치형 및 범주형)을 잘 다룹니다. 특성 공학(feature engineering) 없이도 비선형 관계를 처리합니다. 또한 해석 가능성도 뛰어납니다. 트리를 살펴보면 예측이 이루어진 이유를 정확히 파악할 수 있습니다. 그리고 여러 트리의 평균을 취하는 랜덤 포레스트는 중간 규모의 데이터셋에서 과적합(overfitting)에 매우 강합니다.

이번 강의에서는 재귀적 분할을 사용하여 의사결정 트리를 밑바닥부터 구현한 다음, 그 위에 랜덤 포레스트를 구축합니다. 분할 기준(지니 불순도, 엔트로피, 정보 획득량) 이면의 수학을 구현하고, 약한 학습기(weak learner)의 앙상블(ensemble)이 어떻게 강력한 학습기가 되는지 이해하게 됩니다.

## 핵심 개념

### 의사결정 트리의 동작 원리

의사결정 트리는 일련의 예/아니오 질문을 던져 특성 공간을 직사각형 영역으로 분할합니다.

```mermaid
graph TD
    A["Age < 30?"] -->|Yes| B["Income > 50k?"]
    A -->|No| C["Credit Score > 700?"]
    B -->|Yes| D["Approve"]
    B -->|No| E["Deny"]
    C -->|Yes| F["Approve"]
    C -->|No| G["Deny"]
```

각 내부 노드는 특성을 임곗값(threshold)과 비교하여 검사합니다. 각 리프 노드(leaf node)는 예측을 수행합니다. 새로운 데이터 포인트를 분류하려면 루트 노드에서 시작하여 리프에 도달할 때까지 가지(branch)를 따라갑니다.

트리는 각 노드에서 데이터를 가장 잘 분리하는 특성과 임곗값을 선택하여 하향식(top-down)으로 구축됩니다. "가장 잘" 분리한다는 것은 분할 기준에 의해 정의됩니다.

### 분할 기준: 불순도 측정

각 노드에는 샘플 집합이 있습니다. 우리는 결과로 나오는 자식 노드가 가능한 한 "순수(pure)"해지도록, 즉 각 자식 노드가 대부분 하나의 클래스만 포함하도록 분할하고자 합니다.

**지니 불순도(Gini impurity)**는 해당 노드의 클래스 분포에 따라 레이블이 무작위로 지정될 때, 무작위로 선택된 샘플이 잘못 분류될 확률을 측정합니다.

```math
\text{Gini}(S) = 1 - \sum_k p_k^2 \quad \text{where } p_k \text{ is the proportion of class } k \text{ in set } S.
```

순수한 노드(모두 단일 클래스)의 경우 Gini = 0입니다. 클래스가 50/50으로 나뉜 이진 분할의 경우 Gini = 0.5입니다. 값이 낮을수록 좋습니다.

```
Example: 6 cats, 4 dogs

Gini = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48
```

**엔트로피(Entropy)**는 노드의 정보량(무질서도)을 측정합니다. Phase 1 Lesson 09에서 다루었습니다.

```math
\text{Entropy}(S) = -\sum_k p_k \log_2(p_k)
```

순수한 노드의 경우 엔트로피 = 0입니다. 50/50 이진 분할의 경우 엔트로피 = 1.0입니다. 값이 낮을수록 좋습니다.

```
Example: 6 cats, 4 dogs

Entropy = -(0.6 * log2(0.6) + 0.4 * log2(0.4))
        = -(0.6 * -0.737 + 0.4 * -1.322)
        = 0.442 + 0.529
        = 0.971 bits
```

**정보 획득량(Information gain)**은 분할 후의 불순도(엔트로피 또는 지니 불순도) 감소량입니다.

```math
\begin{aligned}
& IG(S, \text{feature}, \text{threshold}) = \text{Impurity}(S) - \text{weighted\_avg}(\text{Impurity}(S_{\text{left}}), \text{Impurity}(S_{\text{right}})) \\[4pt]
& \text{where the weights are the proportions of samples in each child.}
\end{aligned}
```

각 노드에서의 탐욕 알고리즘(greedy algorithm)은 모든 특성과 가능한 모든 임곗값을 시도합니다. 그런 다음 정보 획득량을 최대화하는 (특성, 임곗값) 쌍을 선택합니다.

### 분할 동작 방식

현재 노드에 $n$개의 특성과 $m$개의 샘플이 있는 데이터셋의 경우:

1. 각 특성 $j$($j = 1$부터 $n$까지)에 대해:
   - 특성 $j$을 기준으로 샘플을 정렬합니다.
   - 연속된 서로 다른 값 사이의 모든 중간값을 임곗값으로 시도합니다.
   - 각 임곗값에 대해 정보 획득량을 계산합니다.
2. 정보 획득량이 가장 높은 특성과 임곗값을 선택합니다.
3. 데이터를 왼쪽(특성 <= 임곗값)과 오른쪽(특성 > 임곗값)으로 분할합니다.
4. 각 자식 노드에 대해 재귀적으로 반복합니다.

이 탐욕적 접근법이 전역적으로 최적인 트리를 보장하지는 않습니다. 최적의 트리를 찾는 것은 NP-하드 문제입니다. 하지만 탐욕적 분할은 실제 환경에서 잘 작동합니다.

### 정지 조건

정지 조건이 없으면 트리는 모든 리프가 순수해질 때까지(리프당 샘플 1개) 계속 자라납니다. 이는 훈련 데이터를 완벽하게 암기하게 되어 일반화 성능이 매우 떨어집니다.

**사전 가지치기(Pre-pruning)**는 트리가 완전히 자라기 전에 성장을 멈춥니다.
- 최대 깊이(Maximum depth): 트리가 설정된 깊이에 도달하면 분할을 중단합니다.
- 리프당 최소 샘플 수(Minimum samples per leaf): 노드의 샘플 수가 $k$개 미만이면 분할을 중단합니다.
- 최소 정보 획득량(Minimum information gain): 최적의 분할로 인한 불순도 개선이 임곗값보다 작으면 분할을 중단합니다.
- 최대 리프 노드 수(Maximum leaf nodes): 전체 리프 노드 수를 제한합니다.

**사후 가지치기(Post-pruning)**는 트리를 완전히 성장시킨 다음 거꾸로 잘라냅니다.
- 비용-복잡도 가지치기(Cost-complexity pruning, scikit-learn에서 사용): 리프 노드 수에 비례하는 페널티를 추가합니다. 페널티를 높일수록 트리가 작아집니다.
- 오차 감소 가지치기(Reduced error pruning): 검증 오차가 증가하지 않는다면 서브트리를 제거합니다.

사전 가지치기가 더 간단하고 빠릅니다. 사후 가지치기는 유용한 추가 분할로 이어질 수 있는 분할을 조기에 중단하지 않으므로 종종 더 나은 트리를 생성합니다.

### 회귀용 의사결정 트리

회귀의 경우, 리프 노드의 예측값은 해당 리프에 속한 타깃값들의 평균이 됩니다. 분할 기준 또한 변경됩니다.

정보 획득량 대신 **분산 감소량(variance reduction)**을 사용합니다.

```math
VR(S, \text{feature}, \text{threshold}) = \text{Var}(S) - \text{weighted\_avg}(\text{Var}(S_{\text{left}}), \text{Var}(S_{\text{right}}))
```

분산을 가장 많이 감소시키는 분할을 선택합니다. 트리는 입력 공간을 여러 영역으로 분할하고, 각 영역에서 상수(평균값)를 예측합니다.

### 랜덤 포레스트: 앙상블의 힘

단일 의사결정 트리는 분산이 높습니다. 데이터의 작은 변화만으로도 완전히 다른 트리가 생성될 수 있습니다. 랜덤 포레스트는 여러 트리의 평균을 취하여 이 문제를 해결합니다.

```mermaid
graph TD
    D["Training Data"] --> B1["Bootstrap Sample 1"]
    D --> B2["Bootstrap Sample 2"]
    D --> B3["Bootstrap Sample 3"]
    D --> BN["Bootstrap Sample N"]
    B1 --> T1["Tree 1<br>(random feature subset)"]
    B2 --> T2["Tree 2<br>(random feature subset)"]
    B3 --> T3["Tree 3<br>(random feature subset)"]
    BN --> TN["Tree N<br>(random feature subset)"]
    T1 --> V["Aggregate Predictions<br>(majority vote or average)"]
    T2 --> V
    T3 --> V
    TN --> V
```

두 가지 무작위성 요소가 트리들을 다양하게 만듭니다.

**배깅(Bagging, bootstrap aggregating):** 각 트리는 훈련 데이터로부터 복원 추출한 무작위 표본인 부트스트랩 샘플(bootstrap sample)로 훈련됩니다. 원래 샘플의 약 63%가 각 부트스트랩에 포함됩니다(나머지는 검증에 사용할 수 있는 OOB(out-of-bag) 샘플이 됩니다).

**특성 무작위 선택(Feature randomization):** 각 분할 시 특성의 무작위 부분집합만 고려합니다. 분류의 경우 기본값은 $\sqrt{n_{\text{features}}}$입니다. 회귀의 경우 $n_{\text{features}}/3$입니다. 이를 통해 모든 트리가 동일한 지배적 특성으로 분할되는 것을 방지합니다.

핵심 원리는 다음과 같습니다. 탈상관된(decorrelated) 많은 트리의 평균을 구하면 편향(bias)을 증가시키지 않으면서 분산을 줄일 수 있습니다. 개별 트리는 평범할 수 있지만, 앙상블은 강력합니다.

### 특성 중요도

랜덤 포레스트는 자연스럽게 특성 중요도 점수를 제공합니다. 가장 일반적인 방법은 다음과 같습니다.

**불순도 감소 평균(Mean Decrease in Impurity, MDI):** 각 특성에 대해, 해당 특성이 사용된 모든 트리와 모든 노드에서의 불순도 총 감소량을 합산합니다. 더 이른 분할에서 불순도를 더 크게 감소시킨 특성일수록 더 중요합니다.

```math
\text{importance}(\text{feature}_j) = \sum_{\text{nodes where feature}_j \text{ is used}} \frac{n_{\text{samples at node}}}{n_{\text{total samples}}} \cdot \text{impurity\_decrease}
```

이 방법은 빠르지만(훈련 중에 계산됨), 카디널리티가 높은(high-cardinality) 특성이나 분할 가능한 지점이 많은 특성에 편향되는 경향이 있습니다.

대안으로 **순열 중요도(Permutation importance)**가 있습니다. 한 특성의 값을 무작위로 섞은 후 모델의 정확도가 얼마나 떨어지는지 측정합니다. 더 신뢰할 수 있지만 속도가 느립니다.
### 트리 모델이 신경망을 이기는 경우

정형 데이터(tabular data)에서는 트리와 포레스트 모델이 신경망보다 우수한 성능을 보입니다. 여기에는 몇 가지 이유가 있습니다.

| 요소 | 트리 | 신경망 |
|--------|-------|----------------|
| 혼합 데이터 타입 (수치형 + 범주형) | 기본 지원 | 인코딩 필요 |
| 소규모 데이터셋 (1만 행 미만) | 잘 동작함 | 과적합(overfitting) 발생 |
| 특성 상호작용 (Feature interactions) | 분할을 통해 발견 | 아키텍처 설계 필요 |
| 해석 가능성 (Interpretability) | 완전한 투명성 | 블랙박스 |
| 학습 시간 | 수 분 | 수 시간 |
| 하이퍼파라미터 민감도 | 낮음 | 높음 |

데이터에 공간적 또는 순차적 구조가 있는 경우(이미지, 텍스트, 오디오)에는 신경망이 유리합니다. 반면 평면적인 테이블 형태의 특성 데이터에는 트리 모델이 기본 선택지입니다.

```figure
decision-tree-depth
```

## 직접 만들어보기

### 1단계: 지니 불순도와 엔트로피

두 가지 분할 기준을 밑바닥부터 구현하고, 어떤 분할이 좋은 분할인지에 대해 두 기준이 일치하는지 확인합니다.

```python
import math

def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum(
        (c / n) * math.log2(c / n) for c in counts.values() if c > 0
    )
```

### 2단계: 최적의 분할 찾기

모든 특성과 모든 임곗값을 시도합니다. 그런 다음 정보 이득(information gain)이 가장 높은 분할을 반환합니다.

```python
def information_gain(parent_labels, left_labels, right_labels, criterion="gini"):
    measure = gini_impurity if criterion == "gini" else entropy
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = measure(parent_labels)
    child_impurity = (
        (n_left / n) * measure(left_labels) +
        (n_right / n) * measure(right_labels)
    )
    return parent_impurity - child_impurity
```

### 3단계: DecisionTree 클래스 구현

재귀적 분할, 예측, 특성 중요도(feature importance) 추적 기능을 구현합니다. `_build` 함수는 트리의 핵심입니다. 노드가 순수(pure)하거나 사전 가지치기(pre-pruning) 제한에 도달하면 분할을 멈추고, 그렇지 않으면 최적의 분할을 찾아 양쪽 자식 노드로 재귀 호출을 진행합니다.

```python
import random

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini",
                 max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.tree = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.n_features = len(X[0])
        self.feature_importances_ = [0.0] * self.n_features
        self.n_samples = len(X)
        self.tree = self._build(X, y, depth=0)
        total = sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = [
                fi / total for fi in self.feature_importances_
            ]

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

    def _build(self, X, y, depth):
        if len(set(y)) == 1:
            return {"leaf": True, "value": y[0]}

        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(y)

        if len(y) < self.min_samples_split:
            return self._make_leaf(y)

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_feature is None or best_gain <= 0:
            return self._make_leaf(y)

        left_X, left_y, right_X, right_y = self._split_data(
            X, y, best_feature, best_threshold
        )

        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            return self._make_leaf(y)

        weight = len(y) / self.n_samples
        self.feature_importances_[best_feature] += weight * best_gain

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build(left_X, left_y, depth + 1),
            "right": self._build(right_X, right_y, depth + 1),
        }

    def _make_leaf(self, y):
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        return {"leaf": True, "value": max(counts, key=counts.get)}

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = -1.0

        if self.max_features == "sqrt":
            k = max(1, int(math.sqrt(self.n_features)))
            feature_indices = random.sample(range(self.n_features), k)
        elif isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError("max_features must be at least 1 when given as an integer")
            k = min(self.max_features, self.n_features)
            feature_indices = random.sample(range(self.n_features), k)
        else:
            feature_indices = list(range(self.n_features))

        for feature_idx in feature_indices:
            values = sorted(set(X[i][feature_idx] for i in range(len(X))))
            if len(values) <= 1:
                continue

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2.0
                left_y = [y[j] for j in range(len(X)) if X[j][feature_idx] <= threshold]
                right_y = [y[j] for j in range(len(X)) if X[j][feature_idx] > threshold]

                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue

                gain = information_gain(y, left_y, right_y, self.criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _split_data(self, X, y, feature, threshold):
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        return left_X, left_y, right_X, right_y

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])
```

### 4단계: RandomForest 클래스 구현

부트스트랩 샘플링(bootstrap sampling), 특성 무작위 선택, 다수결 투표(majority voting)를 구현합니다.

```python
class RandomForest:
    def __init__(self, n_trees=100, max_depth=None,
                 min_samples_split=2, max_features="sqrt",
                 criterion="gini"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.n_trees):
            indices = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                criterion=self.criterion,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        predictions = []
        for i in range(len(X)):
            votes = {}
            for preds in all_preds:
                v = preds[i]
                votes[v] = votes.get(v, 0) + 1
            predictions.append(max(votes, key=votes.get))
        return predictions
```

모든 헬퍼 메서드가 포함된 전체 구현은 `code/trees.py`에서 확인할 수 있습니다.

## 활용하기

scikit-learn을 사용하면 단 세 줄로 랜덤 포레스트(Random Forest)를 학습시킬 수 있습니다.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Accuracy: {rf.score(X_test, y_test):.4f}")
print(f"Feature importances: {rf.feature_importances_}")
```

실무에서는 그래디언트 부스팅 트리(XGBoost, LightGBM, CatBoost)가 랜덤 포레스트보다 더 강력한 경우가 많습니다. 트리를 순차적으로 구축하면서 각 트리가 이전 트리의 오차를 보정하기 때문입니다. 하지만 랜덤 포레스트는 잘못 설정하기가 어렵고 하이퍼파라미터 튜닝이 거의 필요하지 않다는 장점이 있습니다.

## 배포하기

이번 레슨에서는 비즈니스 이해관계자를 위해 의사결정나무(decision tree)의 분할을 해석해 주는 프롬프트인 `outputs/prompt-tree-interpreter.md`를 만듭니다. 학습된 트리의 구조(깊이, 특성, 분할 임곗값, 정확도)를 입력하면 모델을 일상 언어 규칙으로 변환하고, 특성 중요도의 순위를 매기며, 과적합이나 데이터 누출(leakage)을 감지하고, 다음 단계를 추천해 줍니다. 코드를 읽지 못하는 사람에게 트리 기반 모델을 설명해야 할 때 언제든 활용할 수 있습니다.

## 실습 과제

1. 3개 클래스를 가진 2차원 데이터셋에서 단일 의사결정나무를 학습시키세요. 분할 과정을 수동으로 추적하고 직사각형 형태의 결정 경계(decision boundary)를 그려보세요. max_depth=2와 max_depth=10일 때의 결정 경계를 비교해 보세요.

2. 회귀 트리(regression tree)를 위한 분산 감소(variance reduction) 분할을 구현하세요. 200개 포인트에 대한 $y = \sin(x) + \text{noise}$를 생성하고 회귀 트리를 적합시키세요. 트리의 구간별 상수(piecewise-constant) 예측값과 실제 곡선을 함께 플롯해 보세요.

3. 트리 개수를 1, 5, 10, 50, 200개로 늘려가며 랜덤 포레스트를 구축하세요. 트리 개수에 따른 학습 정확도와 테스트 정확도를 플롯해 보세요. 테스트 정확도가 정체(plateau)되기는 하지만 감소하지는 않는 현상(포레스트는 과적합에 강함)을 관찰해 보세요.

4. 5개의 서로 다른 데이터셋에서 분할 기준으로 지니 불순도와 엔트로피를 비교해 보세요. 정확도와 트리 깊이를 측정해 보세요. 대부분의 경우 두 기준은 거의 동일한 결과를 냅니다. 그 이유를 설명해 보세요.

5. 순열 중요도(permutation importance)를 구현하세요. 한 특성이 무작위 노이즈이지만 고유값 개수(cardinality)가 큰 데이터셋에서 MDI(불순도 기반 특성 중요도)와 비교해 보세요. MDI는 노이즈 특성을 높게 평가하지만, 순열 중요도는 그렇지 않습니다.

## 핵심 용어

| 용어 | 일상적인 표현 | 실제 의미 |
|------|----------------|----------------------|
| 의사결정나무 (Decision tree) | "예측을 위한 순서도" | 일련의 if/else 분할을 학습하여 특성 공간을 직사각형 영역으로 분할하는 모델 |
| 지니 불순도 (Gini impurity) | "노드가 얼마나 섞여 있는가" | 한 노드에서 무작위 샘플을 잘못 분류할 확률. 0 = 순수함, 0.5 = 이진 분류에서 최대 불순도 |
| 엔트로피 (Entropy) | "노드 내의 무질서도" | 노드의 정보량. 0 = 순수함, 1.0 = 이진 분류에서 최대 불확실성. 정보 이론에서 유래 |
| 정보 이득 (Information gain) | "분할이 얼마나 좋은가" | 분할 후 불순도의 감소량. 분할을 선택하기 위한 탐욕적(greedy) 기준 |
| 사전 가지치기 (Pre-pruning) | "트리 성장을 일찍 멈추기" | 최대 깊이, 최소 샘플 수, 최소 이득 임곗값을 설정하여 트리의 성장을 조기에 중단하는 것 |
| 사후 가지치기 (Post-pruning) | "나중에 트리를 다듬기" | 트리를 완전히 성장시킨 후, 검증 성능을 향상하지 않는 서브트리를 제거하는 것 |
| 배깅 (Bagging) | "무작위 부분집합으로 학습하기" | 부트스트랩 집계(Bootstrap aggregating). 각 모델을 복원 추출된 서로 다른 무작위 샘플로 학습시키는 방법 |
| 랜덤 포레스트 (Random forest) | "나무들의 묶음" | 의사결정나무의 앙상블로, 각 트리는 부트스트랩 샘플로 학습되며 각 분할마다 무작위 특성 부분집합을 사용함 |
| 특성 중요도 (Feature importance, MDI) | "어떤 특성이 중요한가" | 모든 트리와 노드에 걸쳐 각 특성이 기여한 불순도 감소량의 총합 |
| 순열 중요도 (Permutation importance) | "섞어서 확인하기" | 특정 특성의 값을 무작위로 섞었을 때 발생하는 정확도 하락 폭. 노이즈 특성에 대해 MDI보다 신뢰할 수 있음 |
| 분산 감소 (Variance reduction) | "정보 이득의 회귀 버전" | 정보 이득에 대응하는 회귀 트리의 개념. 타깃 분산을 가장 크게 줄이는 분할을 선택함 |
| 부트스트랩 샘플 (Bootstrap sample) | "중복을 허용한 무작위 샘플" | 원본 데이터셋에서 복원 추출한 무작위 샘플. 크기는 같지만 중복 데이터가 포함됨 |

## 더 읽을거리

- [Breiman: Random Forests (2001)](https://link.springer.com/article/10.1023/A:1010933404324) - 오리지널 랜덤 포레스트 논문
- [Grinsztajn et al.: Why do tree-based models still outperform deep learning on tabular data? (2022)](https://arxiv.org/abs/2207.08815) - 정형 데이터 작업에서 트리 모델과 신경망을 엄밀하게 비교한 논문
- [scikit-learn Decision Trees documentation](https://scikit-learn.org/stable/modules/tree.html) - 시각화 도구가 포함된 실무 가이드
- [XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)](https://arxiv.org/abs/1603.02754) - Kaggle을 장악한 그래디언트 부스팅 논문
