---
title_en: "Logistic Regression"
source: "phases/02-ml-fundamentals/03-logistic-regression"
source_sha: "890e592523f62b14"
model: "gemini-3.8-flash"
translated_at: "2026-10-25"
---
# Logistic Regression

> 로지스틱 회귀(logistic regression)는 직선을 S자 곡선으로 구부려 예/아니오 질문에 확률로 답합니다.

**유형:** Build (실습)
**사용 언어:** Python
**선수 레슨:** Phase 2 Lesson 1-2 (What Is ML, Linear Regression)
**소요 시간:** ~90분

## Learning Objectives

- 시그모이드(sigmoid) 함수와 이진 크로스 엔트로피(binary cross-entropy) 손실을 사용하여 밑바닥부터 로지스틱 회귀를 구현합니다.
- 이진 분류를 위한 정밀도(precision), 재현율(recall), F1 점수(F1 score), 혼동 행렬(confusion matrix)을 계산하고 해석합니다.
- 분류 작업에서 MSE가 실패하는 이유와 이진 크로스 엔트로피가 볼록한(convex) 비용 곡면을 만들어내는 이유를 설명합니다.
- 다중 클래스 분류를 위한 소프트맥스(softmax) 회귀 모델을 구축하고 임계값(threshold) 튜닝의 트레이드오프를 평가합니다.

## The Problem

종양의 크기를 바탕으로 해당 종양이 악성인지 양성인지를 예측하고자 합니다. 선형 회귀(linear regression)를 시도해 보면, 0.3이나 1.7, 혹은 -0.5와 같은 숫자가 출력됩니다. 이 숫자들은 무엇을 의미할까요? 1.7은 "매우 악성"이라는 뜻일까요? -0.5는 "매우 양성"이라는 뜻일까요? 선형 회귀는 범위 제한이 없는 값을 출력합니다. 하지만 분류에는 0과 1 사이로 제한된 확률과 예/아니오라는 명확한 결정이 필요합니다.

로지스틱 회귀가 이 문제를 해결합니다. 선형 회귀와 동일한 선형 결합($wx + b$)을 구한 뒤, 어떤 숫자든 (0, 1) 범위로 압축하는 시그모이드 함수를 통과시킵니다. 그 결과 출력값은 확률이 됩니다. 여기에 임계값(보통 0.5)을 설정하여 결정을 내립니다.

로지스틱 회귀는 실무에서 가장 널리 사용되는 알고리즘 중 하나입니다. 이름과 달리 로지스틱 회귀는 회귀 알고리즘이 아니라 분류(classification) 알고리즘입니다. 이 이름은 모델 내부에서 사용하는 로지스틱(시그모이드) 함수에서 유래했습니다.

## The Concept

### Why Linear Regression Fails for Classification

공부 시간에 따른 시험 합격/불합격(1/0)을 예측한다고 상상해 봅시다. 선형 회귀는 데이터 사이에 직선을 적합(fit)합니다.

```
hours:  1   2   3   4   5   6   7   8   9   10
actual: 0   0   0   0   1   1   1   1   1   1
```

선형 적합은 공부 시간 1시간에 대해 -0.2, 10시간에 대해 1.3과 같은 예측을 출력할 수 있습니다. 이러한 값은 확률이 아닙니다. 0보다 작거나 1보다 커지기 때문입니다. 더 심각한 문제는, 단 하나의 이상치(outlier, 예: 50시간 동안 공부한 학생)가 직선 전체를 끌어당겨 다른 모든 데이터에 대한 예측을 바꿔놓을 수 있다는 점입니다.

분류 작업에는 다음과 같은 특성을 지닌 함수가 필요합니다.
- 0과 1 사이의 값(확률)을 출력해야 합니다.
- 뚜렷한 전환 구간(결정 경계, decision boundary)을 형성해야 합니다.
- 결정 경계에서 멀리 떨어진 이상치에 의해 왜곡되지 않아야 합니다.

### The Sigmoid Function

시그모이드 함수가 정확히 이 역할을 수행합니다.

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

특성:
- $z$가 양의 방향으로 매우 크면, $\sigma(z)$은 1에 수렴합니다.
- $z$가 음의 방향으로 매우 크면, $\sigma(z)$는 0에 수렴합니다.
- $z = 0$일 때, $\sigma(z) = 0.5$입니다.
- 출력값은 항상 0과 1 사이에 있습니다.
- 모든 지점에서 매끄럽고 미분 가능합니다.

도함수 형태도 매우 편리합니다: $\sigma'(z) = \sigma(z) (1 - \sigma(z))$. 덕분에 그래디언트 계산이 효율적입니다.

### Logistic Regression = Linear Model + Sigmoid

모델은 선형 회귀와 동일하게 $z = wx + b$를 계산한 다음, 시그모이드 함수를 적용합니다.

```mermaid
flowchart LR
    X[Input features x] --> L["Linear: z = wx + b"]
    L --> S["Sigmoid: p = 1/(1+e^-z)"]
    S --> D{"p >= 0.5?"}
    D -->|Yes| P[Predict 1]
    D -->|No| N[Predict 0]
```

출력값 $p$은 입력이 클래스 1에 속할 확률인 $P(y=1 \mid x)$로 해석됩니다. 결정 경계는 $wx + b = 0$인 지점이며, 이때 시그모이드 출력값은 정확히 0.5가 됩니다.

### Binary Cross-Entropy Loss

로지스틱 회귀에는 MSE를 사용할 수 없습니다. 시그모이드와 MSE를 함께 사용하면 여러 국소 최솟값(local minima)을 가진 비볼록(non-convex) 비용 곡면이 형성됩니다. 대신 이진 크로스 엔트로피(로그 손실, log loss)를 사용합니다.

```math
\text{Loss} = -\frac{1}{n} \sum \left( y \log(p) + (1-y) \log(1-p) \right)
```

이 손실 함수가 작동하는 원리는 다음과 같습니다.
- $y=1$이고 $p$이 1에 가까울 때: $\log(1) = 0$이므로 손실은 0에 가깝습니다(올바른 예측, 낮은 비용).
- $y=1$이고 $p$이 0에 가까울 때: $\log(0)$은 음의 무한대에 가까워지므로 손실이 매우 큽니다(잘못된 예측, 높은 비용).
- $y=0$이고 $p$이 0에 가까울 때: $\log(1) = 0$이므로 손실은 0에 가깝습니다(올바른 예측, 낮은 비용).
- $y=0$이고 $p$이 1에 가까울 때: $\log(0)$은 음의 무한대에 가까워지므로 손실이 매우 큽니다(잘못된 예측, 높은 비용).

이 손실 함수는 로지스틱 회귀에 대해 볼록하므로, 모든 국소 최솟값이 전역 최솟값(global minimum)이 되며 경사 하강법(gradient descent)이 잘못된 최솟값에 갇히지 않습니다. 완전히 선형 분리 가능한(separable) 데이터의 경우 가중치가 커질수록 손실이 계속 감소하여 유한한 최솟값이 존재하지 않는데, L2 규제(regularization)를 추가하면 이 문제가 해결되어 최솟값이 유일해집니다.

### Gradient Descent for Logistic Regression

시그모이드와 결합된 이진 크로스 엔트로피의 그래디언트는 깔끔한 형태를 띱니다.

```math
\begin{aligned}
\frac{\partial L}{\partial w} &= \frac{1}{n} \sum (p - y) \, x \\
\frac{\partial L}{\partial b} &= \frac{1}{n} \sum (p - y)
\end{aligned}
```

이 식은 선형 회귀의 그래디언트와 동일해 보입니다. 차이점은 $p = wx + b$ 대신 $p = \sigma(wx + b)$를 사용한다는 것입니다. 시그모이드 함수가 비선형성을 도입하지만, 그래디언트 업데이트 규칙 자체는 동일하게 유지됩니다.

```mermaid
flowchart TD
    A[Initialize w=0, b=0] --> B[Forward pass: z = wx+b, p = sigmoid z]
    B --> C[Compute loss: binary cross-entropy]
    C --> D["Compute gradients: dw = (1/n) * sum((p-y)*x)"]
    D --> E[Update: w = w - lr*dw, b = b - lr*db]
    E --> F{Converged?}
    F -->|No| B
    F -->|Yes| G[Model trained]
```

### The Decision Boundary

2차원 입력(두 개의 특성)에서 결정 경계는 다음을 만족하는 직선입니다.

```math
w_1 x_1 + w_2 x_2 + b = 0
```

경계의 한쪽에 있는 점들은 1로 분류되고, 다른 쪽에 있는 점들은 0으로 분류됩니다. 로지스틱 회귀는 항상 선형 결정 경계를 생성합니다. 곡선 형태의 경계가 필요하다면 다항 특성(polynomial feature)을 추가하거나 비선형 모델을 사용해야 합니다.

### Multi-Class Classification with Softmax

이진 로지스틱 회귀는 두 개의 클래스를 처리합니다. $k$개의 클래스를 다룰 때는 소프트맥스(softmax) 함수를 사용합니다.

```math
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
```

각 클래스는 고유한 가중치 벡터를 가집니다. 모델은 각 클래스에 대한 점수 $z_i$를 계산한 다음, 소프트맥스를 통해 점수들을 합이 1이 되는 확률로 변환합니다. 예측 클래스는 확률이 가장 높은 클래스가 됩니다.

손실 함수는 카테고리컬 크로스 엔트로피(categorical cross-entropy)가 됩니다.

```math
\text{Loss} = -\frac{1}{n} \sum \sum_k y_k \log(p_k)
```

여기서 $y_k$는 실제 클래스에 대해서는 1이고 나머지 모든 클래스에 대해서는 0입니다(원-핫 인코딩, one-hot encoding).

### Evaluation Metrics

정확도(accuracy) 하나만으로는 충분하지 않습니다. 음성이 95%, 양성이 5%인 데이터셋에서 항상 음성만 예측하는 모델은 95%의 정확도를 얻지만 아무 쓸모가 없습니다.

**Confusion Matrix**:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actually Positive | True Positive (TP) | False Negative (FN) |
| Actually Negative | False Positive (FP) | True Negative (TN) |

**Precision**: 양성으로 예측한 것 중 실제 양성은 몇 개인가?
```math
\text{Precision} = \frac{TP}{TP + FP}
```

**Recall** (Sensitivity, 재현율): 실제 양성 중 모델이 잡아낸 양성은 몇 개인가?
```math
\text{Recall} = \frac{TP}{TP + FN}
```

**F1 Score**: 정밀도와 재현율의 조화 평균. 두 지표의 균형을 맞춥니다.
```math
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
```

지표 우선순위 결정 기준:
- **Precision**: 거짓 양성(false positive)의 비용이 큰 경우(스팸 필터, 정상 이메일이 차단되어서는 안 됨)
- **Recall**: 거짓 음성(false negative)의 비용이 큰 경우(암 검진, 종양을 놓쳐서는 안 됨)
- **F1**: 단일화된 균형 잡힌 지표가 필요한 경우

```figure
logistic-sigmoid
```

## Build It

### Step 1: Sigmoid function and data generation

```python
import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")
```

### Step 2: Logistic regression from scratch

```python
class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def compute_loss(self, X, y):
        n = len(y)
        total = 0.0
        for i in range(n):
            p = self.predict_proba(X[i])
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n):
                p = self.predict_proba(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | w: [{self.weights[0]:.3f}, {self.weights[1]:.3f}] | b: {self.bias:.3f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("\n=== Training Logistic Regression ===")
model = LogisticRegression(n_features=2, learning_rate=0.1)
model.fit(X_train, y_train, epochs=1000, print_every=200)

print(f"\nTrain accuracy: {model.accuracy(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.accuracy(X_test, y_test):.4f}")
print(f"Weights: [{model.weights[0]:.4f}, {model.weights[1]:.4f}]")
print(f"Bias: {model.bias:.4f}")
```

### Step 3: Confusion matrix and metrics from scratch

```python
class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        self.tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        self.fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        self.fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0

    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def print_confusion_matrix(self):
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Pos   Neg")
        print(f"  Actual Pos     {self.tp:4d}  {self.fn:4d}")
        print(f"  Actual Neg     {self.fp:4d}  {self.tn:4d}")

    def print_report(self):
        self.print_confusion_matrix()
        print(f"\n  Accuracy:  {self.accuracy():.4f}")
        print(f"  Precision: {self.precision():.4f}")
        print(f"  Recall:    {self.recall():.4f}")
        print(f"  F1 Score:  {self.f1():.4f}")


y_pred_test = [model.predict(x) for x in X_test]
print("\n=== Classification Report (Test Set) ===")
metrics = ClassificationMetrics(y_test, y_pred_test)
metrics.print_report()
```

### Step 4: Decision boundary analysis

```python
print("\n=== Decision Boundary ===")
w1, w2 = model.weights
b = model.bias
print(f"Decision boundary: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0")
if abs(w2) > 1e-10:
    print(f"Solved for x2:     x2 = {-w1/w2:.4f}*x1 + {-b/w2:.4f}")

print("\nSample predictions near the boundary:")
test_points = [
    [3.0, 3.0],
    [3.5, 3.5],
    [4.0, 4.0],
    [2.5, 2.5],
    [5.0, 5.0],
]
for point in test_points:
    prob = model.predict_proba(point)
    pred = model.predict(point)
    print(f"  [{point[0]}, {point[1]}] -> prob={prob:.4f}, class={pred}")
```

### Step 5: Multi-class with softmax

```python
class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.biases = [0.0] * n_classes

    def softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    def predict_proba(self, x):
        scores = [
            sum(self.weights[k][j] * x[j] for j in range(self.n_features)) + self.biases[k]
            for k in range(self.n_classes)
        ]
        return self.softmax(scores)

    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.index(max(probs))

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        for epoch in range(epochs):
            grad_w = [[0.0] * self.n_features for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            total_loss = 0.0
            for i in range(n):
                probs = self.predict_proba(X[i])
                for k in range(self.n_classes):
                    target = 1.0 if y[i] == k else 0.0
                    error = probs[k] - target
                    for j in range(self.n_features):
                        grad_w[k][j] += error * X[i][j]
                    grad_b[k] += error
                true_prob = max(probs[y[i]], 1e-15)
                total_loss -= math.log(true_prob)
            for k in range(self.n_classes):
                for j in range(self.n_features):
                    self.weights[k][j] -= self.lr * (grad_w[k][j] / n)
                self.biases[k] -= self.lr * (grad_b[k] / n)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {total_loss / n:.4f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


random.seed(42)
X_3class = []
y_3class = []

centers = [(1, 1), (5, 1), (3, 5)]
for label, (cx, cy) in enumerate(centers):
    for _ in range(50):
        X_3class.append([random.gauss(cx, 0.8), random.gauss(cy, 0.8)])
        y_3class.append(label)

combined = list(zip(X_3class, y_3class))
random.shuffle(combined)
X_3class, y_3class = zip(*combined)
X_3class = list(X_3class)
y_3class = list(y_3class)

split_3 = int(0.8 * len(X_3class))
X_train_3 = X_3class[:split_3]
y_train_3 = y_3class[:split_3]
X_test_3 = X_3class[split_3:]
y_test_3 = y_3class[split_3:]

print("\n=== Multi-class Softmax Regression (3 classes) ===")
softmax_model = SoftmaxRegression(n_features=2, n_classes=3, learning_rate=0.1)
softmax_model.fit(X_train_3, y_train_3, epochs=1000, print_every=200)
print(f"\nTrain accuracy: {softmax_model.accuracy(X_train_3, y_train_3):.4f}")
print(f"Test accuracy:  {softmax_model.accuracy(X_test_3, y_test_3):.4f}")

print("\nSample predictions:")
for i in range(5):
    probs = softmax_model.predict_proba(X_test_3[i])
    pred = softmax_model.predict(X_test_3[i])
    print(f"  True: {y_test_3[i]}, Predicted: {pred}, Probs: [{', '.join(f'{p:.3f}' for p in probs)}]")
```

### Step 6: Threshold tuning

```python
print("\n=== Threshold Tuning ===")
print("Default threshold: 0.5. Adjusting the threshold trades precision for recall.\n")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 52)

for t in thresholds:
    y_pred_t = [1 if model.predict_proba(x) >= t else 0 for x in X_test]
    m = ClassificationMetrics(y_test, y_pred_t)
    print(f"{t:>10.1f} {m.accuracy():>10.4f} {m.precision():>10.4f} {m.recall():>10.4f} {m.f1():>10.4f}")
```

## Use It

이제 scikit-learn을 사용하여 동일한 작업을 수행해 봅니다.

```python
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X_0 = np.random.randn(100, 2) + [2, 2]
X_1 = np.random.randn(100, 2) + [5, 5]
X_sk = np.vstack([X_0, X_1])
y_sk = np.array([0] * 100 + [1] * 100)

X_tr, X_te, y_tr, y_te = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

lr = SklearnLR()
lr.fit(X_tr_sc, y_tr)
y_pred = lr.predict(X_te_sc)

print("=== Scikit-learn Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_te, y_pred):.4f}")
print(f"Precision: {precision_score(y_te, y_pred):.4f}")
print(f"Recall:    {recall_score(y_te, y_pred):.4f}")
print(f"F1:        {f1_score(y_te, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_te, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_te, y_pred)}")
```

밑바닥부터 직접 구현한 모델도 동일한 결정 경계와 지표를 산출합니다. Scikit-learn은 다양한 솔버 옵션(liblinear, lbfgs, saga), 자동 규제, 다중 클래스 전략(one-vs-rest, multinomial), 수치 안정성 최적화를 추가로 제공합니다.

## Ship It

이 레슨에서 생성한 결과물:
- `code/logistic_regression.py` - 지표 계산을 포함하여 밑바닥부터 구현한 로지스틱 회귀
## 연습 문제

1. 선형적으로 분리되지 않는(linearly separable) 데이터셋을 생성합니다(예: 두 개의 동심원). 로지스틱 회귀(logistic regression)를 학습시키고 제대로 분류하지 못하는 현상을 관찰합니다. 그런 다음 다항 특성($x_1^2, x_2^2, x_1 x_2$)을 추가하고 다시 학습시킵니다. 정확도가 향상됨을 확인합니다.
2. 3개 클래스를 가진 softmax 모델을 위한 다중 클래스 혼동 행렬(confusion matrix)을 구현합니다. 클래스별 정밀도(precision)와 재현율(recall)을 계산합니다. 어떤 클래스가 가장 분류하기 어렵습니까?
3. ROC 곡선(ROC curve)을 밑바닥부터 직접 구현합니다. 0부터 1까지 100개의 임계값(threshold)에 대해 참 긍정률(true positive rate)과 거짓 긍정률(false positive rate)을 계산합니다. 사다리꼴 공식(trapezoidal rule)을 사용하여 AUC(area under the curve)를 계산합니다.

## 핵심 용어

| 용어 | 흔히 부르는 표현 | 실제 의미 |
|------|----------------|----------------------|
| 로지스틱 회귀(Logistic regression) | "분류를 위한 회귀" | 선형 모델 뒤에 시그모이드(sigmoid) 함수가 이어져 클래스 확률을 출력하는 모델 |
| 시그모이드 함수(Sigmoid function) | "S자형 곡선" | 임의의 실수를 (0, 1) 범위로 매핑하는 함수 $1/(1+e^{-z})$ |
| 이진 교차 엔트로피(Binary cross-entropy) | "로그 손실(Log loss)" | 확신에 찬 잘못된 예측에 큰 페널티를 부여하는 손실 함수 $-[y \log(p) + (1-y) \log(1-p)]$ |
| 결정 경계(Decision boundary) | "경계선" | 예측 클래스를 구분하며 모델의 출력 확률이 0.5가 되는 경계면 |
| 소프트맥스(Softmax) | "다중 클래스 시그모이드" | 점수 벡터를 합이 1이 되는 확률 분포로 변환하는 함수 |
| 정밀도(Precision) | "선택된 것 중 관련된 것의 비율" | 양성(positive)으로 예측한 것 중 실제 양성의 비율인 $TP / (TP + FP)$ |
| 재현율(Recall) | "관련된 것 중 선택된 것의 비율" | 모델이 올바르게 찾아낸 실제 양성의 비율인 $TP / (TP + FN)$ |
| F1 점수(F1 score) | "균형 잡힌 정확도" | 정밀도와 재현율의 조화 평균: $2PR / (P+R)$ |
| 혼동 행렬(Confusion matrix) | "오류 분석표" | 각 클래스 쌍에 대한 TP, TN, FP, FN 수를 나타내는 표 |
| 임계값(Threshold) | "기준값" | 모델이 클래스 1로 예측하는 기준 확률값(기본값 0.5, 조정 가능) |
| 원-핫 인코딩(One-hot encoding) | "범주형 데이터를 위한 이진 열" | 클래스 $k$을 $k$ 위치만 1이고 나머지는 0인 벡터로 표현하는 방식 |
| 범주형 교차 엔트로피(Categorical cross-entropy) | "다중 클래스 로그 손실" | 원-핫 인코딩된 레이블을 사용하여 이진 교차 엔트로피를 $k$개 클래스로 확장한 손실 함수 |
