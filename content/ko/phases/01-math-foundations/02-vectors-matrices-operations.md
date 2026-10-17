---
title_en: "Vectors, Matrices & Operations"
source: "phases/01-math-foundations/02-vectors-matrices-operations"
source_sha: "a5849d742948a87e"
model: "gemini-3.8-flash"
translated_at: "2026-10-17"
---
# 벡터, 행렬 및 연산

> 모든 신경망은 몇 가지 단계가 추가된 행렬 곱셈에 불과합니다.

**유형:** Build (실습)
**사용 언어:** Python, Julia
**선수 레슨:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**소요 시간:** ~60분

## 학습 목표

- 원소별 연산, 행렬 곱셈, 전치, 행렬식, 역행렬을 지원하는 Matrix 클래스 구현하기
- 원소별 곱셈과 행렬 곱셈을 구분하고 각각이 언제 적용되는지 설명하기
- 밑바닥부터 구현한 Matrix 클래스만을 사용하여 단일 밀집 신경망 레이어(dense neural network layer)(`relu(W @ x + b)`) 구현하기
- 브로드캐스팅(broadcasting) 규칙 및 신경망 프레임워크에서 편향(bias) 덧셈이 동작하는 방식 설명하기

## 문제 상황

신경망을 구축하려고 합니다. 코드를 읽다 보면 다음과 같은 구문을 마주하게 됩니다.

```
output = activation(weights @ input + bias)
```

여기서 `@`는 행렬 곱셈입니다. `weights`는 행렬입니다. `input`는 벡터입니다. 이러한 연산이 무엇을 하는지 모른다면 이 한 줄은 마법처럼 보일 것입니다. 하지만 알고 있다면, 이는 단 세 가지 연산으로 레이어의 전체 순전파(forward pass)를 수행하는 것임을 이해할 수 있습니다.

모델이 처리하는 모든 이미지는 픽셀 값으로 이루어진 행렬입니다. 모든 단어 임베딩(embedding)은 벡터입니다. 모든 신경망의 모든 레이어는 행렬 변환입니다. 변수를 이해하지 않고 코드를 작성할 수 없듯이, 행렬 연산에 능숙하지 않고서는 AI 시스템을 구축할 수 없습니다.

이번 레슨에서는 이러한 숙련도를 밑바닥부터 기릅니다.

## 핵심 개념

### 벡터: 순서가 있는 숫자 목록

벡터는 방향과 크기를 가진 숫자 목록입니다. AI에서 벡터는 데이터 포인트, 특성(feature), 또는 파라미터를 나타냅니다.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

2D 벡터 `[3, 4]`는 평면 위의 좌표 (3, 4)를 가리킵니다. 이 벡터의 길이(크기)는 5입니다(3-4-5 직각삼각형).

### 행렬: 숫자들의 격자

행렬은 2차원 격자입니다. 행(row)과 열(column)로 이루어져 있습니다. $m \times n$ 행렬은 $m$개의 행과 $n$개의 열을 가집니다.

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \text{2x3 matrix (2 rows, 3 columns)}
```

신경망에서 가중치 행렬은 입력 벡터를 출력 벡터로 변환합니다. 784개의 입력과 128개의 출력을 가진 레이어는 128x784 크기의 가중치 행렬을 사용합니다.

### 차원(shape)이 중요한 이유

행렬 곱셈에는 엄격한 규칙이 있습니다: `(m x n) @ (n x p) = (m x p)`. 내부 차원이 반드시 일치해야 합니다.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

PyTorch에서 차원 불일치(shape mismatch) 오류가 발생한다면 바로 이 때문입니다.

### 연산 맵

| 연산 | 역할 | 신경망에서의 활용 |
|-----------|-------------|-------------------|
| 덧셈 | 원소별 결합 | 출력에 편향 더하기 |
| 스칼라 곱 | 모든 원소의 스케일 조절 | 학습률(learning rate) * 그래디언트 |
| 행렬 곱 | 벡터 변환 | 레이어 순전파 |
| 전치(Transpose) | 행과 열 뒤집기 | 역전파(backpropagation) |
| 행렬식(Determinant) | 단일 숫자로의 요약 | 가역성(invertibility) 확인 |
| 역행렬(Inverse) | 변환 되돌리기 | 선형 시스템 풀이 |
| 단위행렬(Identity) | 아무것도 하지 않는 행렬 | 초기화, 잔차 연결(residual connections) |

### 원소별 곱셈 vs 행렬 곱셈

이 차이는 초심자들이 끊임없이 헷갈리는 부분입니다.

원소별 곱셈: 대응하는 위치끼리 곱합니다. 두 행렬의 크기가 동일해야 합니다.

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} * \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}
```

행렬 곱셈: 행과 열의 내적(dot product)입니다. 내부 차원이 반드시 일치해야 합니다.

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} @ \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
```

연산이 다르고, 결과가 다르며, 규칙이 다릅니다.

### 브로드캐스팅

편향 벡터를 출력 행렬에 더할 때 차원이 일치하지 않습니다. 브로드캐스팅은 더 작은 배열을 늘려 크기를 맞춥니다.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

모든 현대적 프레임워크는 이를 자동으로 수행합니다. 이를 이해하면 차원이 맞지 않아 보이는데도 코드가 정상 실행될 때 혼란을 방지할 수 있습니다.

```figure
vector-projection
```

## 직접 만들기

### 1단계: Vector 클래스

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 2단계: 핵심 연산을 갖춘 Matrix 클래스

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 3단계: 동작 확인

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 4단계: 신경망과 연결하기

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

이것이 바로 단일 밀집 레이어입니다: `output = relu(W @ x + b)`. 모든 신경망의 모든 밀집 레이어는 정확히 이 작업을 수행합니다.

## 활용하기

NumPy는 위의 모든 작업을 훨씬 적은 줄의 코드로, 몇 배는 더 빠르게 수행합니다.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python의 `@` 연산자는 `__matmul__`를 호출합니다. NumPy는 C와 Fortran으로 작성된 최적화된 BLAS 루틴으로 이를 구현합니다. 수학적으로는 동일하지만 100배 더 빠릅니다.

NumPy에서의 브로드캐스팅:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy는 1D 편향을 두 행 전체로 자동 브로드캐스팅합니다. 이것이 모든 신경망 프레임워크에서 편향 덧셈이 동작하는 방식입니다.

## 적용하기

이번 레슨에서는 기하학적 직관을 통해 행렬 연산을 가르치기 위한 프롬프트를 생성합니다. `outputs/prompt-matrix-operations.md`를 참고하십시오.

여기서 구현한 Matrix 클래스는 Phase 3, Lesson 10에서 만들 미니 신경망 프레임워크의 기반이 됩니다.

## 연습 문제

1. **역행렬 검증하기.** `A @ A.inverse_2x2()`를 곱하여 단위행렬이 나오는지 확인하십시오. 세 가지 서로 다른 2x2 행렬로 시도해 보십시오. 행렬식이 0일 때는 어떻게 됩니까?

2. **3x3 역행렬 구현하기.** 수반행렬(adjugate) 방식을 사용하여 3x3 행렬의 역행렬을 계산하도록 Matrix 클래스를 확장하십시오. NumPy의 `np.linalg.inv`와 비교하여 테스트하십시오.

3. **2층 신경망 구축하기.** NumPy 없이 오직 직접 만든 Matrix 클래스만을 사용하여 2층 신경망을 만드십시오: 입력 (3) -> 은닉층 (4) -> 출력 (2). 무작위 가중치를 초기화하고, 순전파를 실행한 뒤, 모든 차원이 올바른지 확인하십시오.

### 힌트

**연습 문제 1.** 세 행렬에 대해 `A @ A.inverse_2x2()`를 곱하고, 각 셀이 `1e-9` 이내에서 단위행렬과 일치하는지 비교하십시오. [[1, 2], [2, 4]]와 같이 행렬식이 0인 행렬의 경우, `inverse_2x2`는 `ValueError`를 발생시키며 이는 예상된 결과입니다.

**연습 문제 2.** 3x3 여인수 행렬(cofactor matrix)을 계산하고, 이를 전치하여 수반행렬을 구한 다음, 각 셀을 `determinant()`로 나눕니다. `A @ A_inv`가 단위행렬을 반환하는지, 그리고 각 셀이 `1e-9` 이내에서 `np.linalg.inv(A)`와 일치하는지 확인하십시오.

**연습 문제 3.** 곱하기 전에 각 행렬의 차원을 적어 보십시오. `demo_neural_network_layer`에서처럼 `W1`는 4x3이고 `W2`는 2x4입니다. 3x1 입력은 4x1 은닉 벡터와 2x1 출력을 생성해야 합니다.

## 핵심 용어

| 용어 | 흔히 하는 표현 | 실제 의미 |
|------|----------------|----------------------|
| 벡터(Vector) | "화살표" | 순서가 있는 숫자 목록. AI에서는 고차원 공간의 한 점입니다. |
| 행렬(Matrix) | "숫자 표" | 선형 변환. 한 공간의 벡터를 다른 공간으로 매핑합니다. |
| 행렬 곱(Matrix multiply) | "그냥 숫자끼리 곱하기" | 첫 번째 행렬의 각 행과 두 번째 행렬의 각 열 사이의 내적. 순서가 중요합니다. |
| 전치(Transpose) | "뒤집기" | 행과 열을 맞바꿉니다. $m \times n$ 행렬을 $n \times m$ 행렬로 변환합니다. 역전파에서 매우 중요합니다. |
| 행렬식(Determinant) | "행렬에서 나오는 어떤 숫자" | 행렬이 면적(2D)이나 부피(3D)를 얼마나 스케일링하는지 측정합니다. 0은 변환이 차원을 축소시킴(crush)을 의미합니다. |
| 역행렬(Inverse) | "행렬 되돌리기" | 변환을 역전시키는 행렬입니다. 행렬식이 0이 아닐 때만 존재합니다. |
| 단위행렬(Identity matrix) | "심심한 행렬" | 행렬에서 1을 곱하는 것과 같은 개념입니다. 잔차 연결(ResNet)에 사용됩니다. |
| 브로드캐스팅(Broadcasting) | "마법 같은 차원 맞춤" | 누락된 차원을 따라 반복함으로써 더 큰 배열에 맞게 더 작은 배열을 늘립니다. |
| 원소별 연산(Element-wise) | "일반 곱셈" | 대응하는 위치끼리 곱합니다. 두 배열의 크기가 같아야 합니다(또는 브로드캐스팅이 가능해야 함). |

## 더 읽을거리

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 이번 레슨에서 다룬 모든 연산에 대한 시각적 직관 제공
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy가 따르는 정확한 브로드캐스팅 규칙
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 머신러닝 맞춤형 선형대수 핵심 요약 참고 자료
