---
title_en: "Tensor Operations"
source: "phases/01-math-foundations/12-tensor-operations"
source_sha: "8f37a3ac0484d015"
model: "gemini-3.8-flash"
translated_at: "2026-10-18"
---
# 텐서 연산

> 텐서는 데이터와 딥러닝 사이의 공통 언어입니다. 모든 이미지, 모든 문장, 모든 그래디언트가 텐서를 통해 흐릅니다.

**유형:** Build (실습)
**Language:** Python
**선수 레슨:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**소요 시간:** ~90분

## 학습 목표

- 형상(shape), 스트라이드(strides), reshape, 전치(transpose), 요소별 연산(element-wise operations)을 갖춘 텐서 클래스를 밑바닥부터 구현합니다.
- 브로드캐스팅(broadcasting) 규칙을 적용하여 데이터 복사 없이 서로 다른 형상의 텐서를 연산합니다.
- 내적, 행렬 곱셈, 외적, 배치 연산을 위한 einsum 표현식을 작성합니다.
- 멀티헤드 어텐션(multi-head attention)의 모든 단계에서 정확한 텐서 형상을 추적합니다.

## 문제 상황

트랜스포머(Transformer)를 구축합니다. 순전파(forward pass) 코드는 깔끔해 보입니다. 코드를 실행하자 다음 오류가 발생합니다: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)`. 형상을 응시합니다. 전치를 시도해 봅니다. 이제 `Expected 4D input (got 3D input)` 오류가 나타납니다. unsqueeze를 추가해 봅니다. 다른 곳이 깨집니다.

형상 오류는 딥러닝 코드에서 가장 흔한 버그입니다. 개념적으로 어렵지는 않습니다. 각 연산마다 형상 규약(shape contract)이 정해져 있기 때문입니다. 하지만 오류는 빠르게 증폭됩니다. 트랜스포머에는 수십 개의 reshape, transpose, broadcast가 서로 연결되어 있습니다. 축 하나만 잘못 지정해도 오류가 연쇄적으로 번집니다. 더 나쁜 점은, 일부 형상 실수는 오류를 전혀 발생시키지 않는다는 것입니다. 잘못된 차원을 따라 브로드캐스팅하거나 엉뚱한 축에 대해 합을 구함으로써 조용히 엉터리 결과를 만들어냅니다.

행렬은 두 대상 집합 간의 쌍별(pairwise) 관계를 다룹니다. 하지만 실제 데이터는 2차원에 들어맞지 않습니다. 224x224 크기의 RGB 이미지 32장으로 구성된 배치는 4D 텐서입니다: `(32, 3, 224, 224)`. 12개 헤드를 가진 셀프 어텐션(self-attention) 역시 4D입니다: `(batch, heads, seq_len, head_dim)`. 어떤 차원 수로도 일반화할 수 있고, 모든 차원에 걸쳐 깔끔하게 조합되는 연산을 갖춘 자료구조가 필요합니다. 그 자료구조가 바로 텐서(tensor)입니다. 텐서 연산을 완벽히 익히면 형상 오류를 손쉽게 디버깅할 수 있게 됩니다.

## 핵심 개념

### 텐서란 무엇인가

텐서는 균일한 데이터 타입을 갖는 다차원 숫자 배열입니다. 차원의 수를 **랭크**(rank, 또는 **차수**(order))라고 합니다. 각 차원은 **축**(axis)입니다. **형상**(shape)은 각 축의 크기를 나열한 튜플입니다.

```mermaid
graph LR
    S["Scalar<br/>rank 0<br/>shape: ()"] --> V["Vector<br/>rank 1<br/>shape: (3,)"]
    V --> M["Matrix<br/>rank 2<br/>shape: (2,3)"]
    M --> T3["3D Tensor<br/>rank 3<br/>shape: (2,2,2)"]
    T3 --> T4["4D Tensor<br/>rank 4<br/>shape: (B,C,H,W)"]
```

총 요소 수 = 모든 크기의 곱. 형상 `(2, 3, 4)`는 $2 \cdot 3 \cdot 4 = 24$개의 요소를 담습니다.

### 딥러닝에서의 텐서 형상

다양한 데이터 타입은 관례적으로 특정 텐서 형상에 매핑됩니다.

```mermaid
graph TD
    subgraph Vision
        V1["(B, C, H, W)<br/>32, 3, 224, 224"]
    end
    subgraph NLP
        N1["(B, T, D)<br/>16, 128, 768"]
    end
    subgraph Attention
        A1["(B, H, T, D)<br/>16, 12, 128, 64"]
    end
    subgraph Weights
        W1["Linear: (out, in)<br/>Conv2D: (out_c, in_c, kH, kW)<br/>Embedding: (vocab, dim)"]
    end
```

PyTorch는 NCHW(채널 우선, channels-first) 방식을 사용합니다. TensorFlow는 기본적으로 NHWC(채널 마지막, channels-last) 방식을 사용합니다. 레이아웃이 일치하지 않으면 소리 없는 속도 저하나 오류가 발생합니다.

### 메모리 레이아웃의 동작 원리

메모리 상에서 2D 배열은 1D 바이트 시퀀스입니다. **스트라이드**(strides)는 각 축을 따라 한 단계 이동하기 위해 건너뛰어야 하는 요소의 수를 나타냅니다.

```mermaid
graph LR
    subgraph "Row-major (C order)"
        R["a b c d e f<br/>strides: (3, 1)"]
    end
    subgraph "Column-major (F order)"
        C["a d b e c f<br/>strides: (1, 2)"]
    end
```

전치는 데이터를 이동시키지 않습니다. 스트라이드만 맞바꾸어 텐서를 **비연속적**(non-contiguous) 상태로 만듭니다. 즉, 한 행의 요소들이 메모리 상에서 더 이상 인접하지 않게 됩니다.

### 브로드캐스팅 규칙

브로드캐스팅을 사용하면 데이터를 복사하지 않고도 서로 다른 형상의 텐서를 연산할 수 있습니다. 형상을 오른쪽부터 맞춥니다. 두 차원의 크기가 같거나 둘 중 하나가 1일 때 두 차원은 호환됩니다. 차원 수가 적은 쪽은 왼쪽에 1이 채워집니다.

```
Tensor A:     (8, 1, 6, 1)
Tensor B:        (7, 1, 5)
Padded B:     (1, 7, 1, 5)
Result:       (8, 7, 6, 5)
```

### Einsum: 범용 텐서 연산

아인슈타인 표기법(Einstein summation)은 각 축에 문자를 부여합니다. 입력에는 있지만 출력에는 없는 축은 합산(sum)됩니다. 양쪽 모두에 있는 축은 유지됩니다.

```mermaid
graph LR
    subgraph "matmul: ik,kj -> ij"
        A["A(I,K)"] --> |"sum over k"| C["C(I,J)"]
        B["B(K,J)"] --> |"sum over k"| C
    end
```

주요 패턴: `i,i->` (내적), `i,j->ij` (외적), `ii->` (대각합, trace), `ij->ji` (전치), `bij,bjk->bik` (배치 행렬 곱), `bhtd,bhsd->bhts` (어텐션 점수).

```figure
tensor-broadcast
```

## 직접 만들기

코드는 `code/tensors.py`에 있습니다. 각 단계는 해당 파일의 구현을 참조합니다.

### 1단계: 텐서 스토리지와 스트라이드

텐서는 1차원 플랫(flat) 숫자 리스트와 형상 메타데이터를 저장합니다. 스트라이드는 인덱싱 로직에 다차원 인덱스를 1차원 플랫 위치로 매핑하는 방법을 알려줍니다.

```python
class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)
            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape = tuple(shape)

        self._strides = self._compute_strides(self._shape)

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)
```

형상 `(3, 4)`의 경우 스트라이드는 `(4, 1)`입니다. 한 행을 이동하려면 4개 요소를 건너뛰고, 한 열을 이동하려면 1개 요소를 건너뜁니다.

### 2단계: Reshape, squeeze, unsqueeze

Reshape는 요소의 순서를 바꾸지 않고 형상을 변경합니다. 총 요소 수는 동일하게 유지되어야 합니다. 한 차원의 크기를 추론하려면 `-1`을 사용합니다.

```python
t = Tensor(list(range(12)), shape=(2, 6))
r = t.reshape((3, 4))
r = t.reshape((-1, 3))
```

Squeeze는 크기가 1인 축을 제거합니다. Unsqueeze는 크기가 1인 축을 삽입합니다. Unsqueeze는 브로드캐스팅에 필수적입니다. 바이어스(bias) 벡터 `(D,)`를 배치 `(B, T, D)`에 더하려면 `(1, 1, D)`로 unsqueeze해야 합니다.

```python
t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
s = t.squeeze()
v = Tensor([1, 2, 3])
u = v.unsqueeze(0)
```

### 3단계: Transpose와 permute

Transpose는 두 축을 맞바꿉니다. Permute는 모든 축의 순서를 재배치합니다. NCHW와 NHWC 사이를 변환할 때 이 방식을 사용합니다.

```python
mat = Tensor(list(range(6)), shape=(2, 3))
tr = mat.transpose(0, 1)

t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))
perm = t4d.permute((0, 2, 3, 1))
```

전치 또는 permute를 수행한 후, 텐서는 메모리 상에서 비연속적인 상태가 됩니다. PyTorch에서는 비연속 텐서에 대해 `view` 호출이 실패하므로, `reshape`을 사용하거나 먼저 `.contiguous()`를 호출해야 합니다.

### 4단계: 요소별 연산과 축소 연산

요소별 연산(덧셈, 곱셈, 뺄셈)은 각 요소에 독립적으로 적용되며 형상을 유지합니다. 축소(reduction) 연산(합, 평균, 최댓값)은 하나 이상의 축을 압축합니다.

```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[10, 20], [30, 40]])
c = a + b
d = a * 2
s = a.sum(axis=0)
```

CNN에서의 글로벌 평균 풀링(global average pooling): `(B, C, H, W).mean(axis=[2, 3])`은 `(B, C)`을 생성합니다. NLP에서의 시퀀스 평균 풀링(sequence mean pooling): `(B, T, D).mean(axis=1)`은 `(B, D)`을 생성합니다.

### 5단계: NumPy를 활용한 브로드캐스팅

`tensors.py`의 `demo_broadcasting_numpy()` 함수는 핵심 패턴들을 보여줍니다.

```python
activations = np.random.randn(4, 3)
bias = np.array([0.1, 0.2, 0.3])
result = activations + bias

images = np.random.randn(2, 3, 4, 4)
scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)
result = images * scale

a = np.array([1, 2, 3]).reshape(-1, 1)
b = np.array([10, 20, 30, 40]).reshape(1, -1)
outer = a * b
```

브로드캐스팅을 통한 쌍별 거리(pairwise distance) 계산: `(M, 2)`를 `(M, 1, 2)`로, `(N, 2)`를 `(1, N, 2)`로 reshape하고, 뺀 다음, 제곱하고, 마지막 축을 따라 합산한 뒤, 제곱근을 구합니다. 결과: `(M, N)`.

### 6단계: Einsum 연산

`demo_einsum()` 및 `demo_einsum_gallery()` 함수는 흔히 사용되는 모든 패턴을 차례대로 살펴봅니다.

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
dot = np.einsum("i,i->", a, b)

A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
matmul = np.einsum("ik,kj->ij", A, B)

batch_A = np.random.randn(4, 3, 5)
batch_B = np.random.randn(4, 5, 2)
batch_mm = np.einsum("bij,bjk->bik", batch_A, batch_B)
```

축약(contraction)의 연산 비용은 유지되거나 합산되는 모든 인덱스 크기의 곱입니다. $B=32$, $I=128$, $J=64$, $K=128$ 크기를 갖는 `bij,bjk->bik`의 경우, $32 \cdot 128 \cdot 64 \cdot 128 = 33{,}554{,}432$회의 곱셈-덧셈 연산이 필요합니다.

### 7단계: Einsum을 이용한 어텐션 메커니즘

`demo_attention_einsum()` 함수는 멀티헤드 어텐션을 엔드투엔드로 구현합니다.

```python
B, H, T, D = 2, 4, 8, 16
E = H * D

X = np.random.randn(B, T, E)
W_q = np.random.randn(E, E) * 0.02

Q = np.einsum("bte,ek->btk", X, W_q)
Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)

scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
weights = softmax(scores, axis=-1)
attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)

concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
output = np.einsum("bte,ek->btk", concat, W_o)
```

모든 단계가 텐서 연산입니다: 프로젝션(einsum을 통한 행렬 곱), 헤드 분할(reshape + transpose), 어텐션 점수(einsum을 통한 배치 행렬 곱), 가중합(einsum을 통한 배치 행렬 곱), 헤드 결합(transpose + reshape), 출력 프로젝션(einsum을 통한 행렬 곱).

## 사용해 보기

### 직접 구현 vs NumPy

| 연산 | 직접 구현 (Tensor 클래스) | NumPy |
|---|---|---|
| 생성 | `Tensor([[1,2],[3,4]])` | `np.array([[1,2],[3,4]])` |
| Reshape | `t.reshape((3,4))` | `a.reshape(3,4)` |
| Transpose | `t.transpose(0,1)` | `a.T` 또는 `a.transpose(0,1)` |
| Squeeze | `t.squeeze(0)` | `np.squeeze(a, 0)` |
| 합 (Sum) | `t.sum(axis=0)` | `a.sum(axis=0)` |
| Einsum | N/A | `np.einsum("ij,jk->ik", a, b)` |

### 직접 구현 vs PyTorch

```python
import torch

t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
t.shape
t.stride()
t.is_contiguous()

t.reshape(3, 2)
t.unsqueeze(0)
t.transpose(0, 1)
t.transpose(0, 1).contiguous()

torch.einsum("ik,kj->ij", A, B)
```

PyTorch는 autograd, GPU 지원, 최적화된 BLAS 커널을 추가로 제공합니다. 형상 시맨틱은 완전히 동일합니다. 직접 구현한 버전을 이해하고 있다면 PyTorch의 형상 오류도 명확히 파악할 수 있습니다.

### 텐서 연산으로서의 신경망 레이어

| 연산 | 텐서 형태 | Einsum |
|---|---|---|
| 선형 레이어 (Linear layer) | `Y = X @ W.T + b` | `"bd,od->bo"` + bias |
| 어텐션 QKV | `Q = X @ W_q` | `"btd,dh->bth"` |
| 어텐션 점수 (Attention scores) | `Q @ K.T / sqrt(d)` | `"bhtd,bhsd->bhts"` |
| 어텐션 출력 (Attention output) | `softmax(scores) @ V` | `"bhts,bhsd->bhtd"` |
| 배치 정규화 (Batch norm) | `(X - mu) / sigma * gamma` | 요소별 연산 + 브로드캐스트 |
| Softmax | `exp(x) / sum(exp(x))` | 요소별 연산 + 축소 연산 |
## Ship It

이번 강의에서는 재사용 가능한 두 가지 프롬프트를 만듭니다.

1. **`outputs/prompt-tensor-shapes.md`** -- 텐서 형태 불일치(shape mismatch) 디버깅을 위한 체계적인 프롬프트입니다. 자주 쓰이는 모든 연산(matmul, broadcast, cat, Linear, Conv2d, BatchNorm, softmax)에 대한 결정 테이블과 해결책 조회 테이블을 포함합니다.

2. **`outputs/prompt-tensor-debugger.md`** -- 형태 에러로 진행이 막혔을 때 AI 어시스턴트에 붙여넣을 수 있는 단계별 디버깅 프롬프트입니다. 에러 메시지와 텐서 형태를 입력하면 정확한 해결책을 돌려받을 수 있습니다.

## Exercises

1. **초급 -- Reshape 왕복.** `(2, 3, 4)` 형태의 텐서를 준비합니다. 이를 `(6, 4)`로 변환한 다음, `(24,)`로 변환하고, 다시 `(2, 3, 4)`로 변환합니다. 평탄화(flatten)된 데이터를 출력하여 각 단계마다 원소 순서가 유지되는지 확인하세요.

2. **중급 -- 브로드캐스팅(broadcasting) 구현.** 크기가 1인 차원을 대상 형태에 맞게 확장하는 `broadcast_to(shape)` 메서드를 `Tensor` 클래스에 추가합니다. 그런 다음 연산 전에 자동으로 브로드캐스팅을 수행하도록 `_elementwise_op`를 수정하세요. 형태가 `(3, 1)` 및 `(1, 4)`인 텐서로 테스트하여 `(3, 4)`가 생성되는지 확인하세요.

3. **고급 -- 밑바닥부터 einsum 구현하기.** 최소한 내적(`i,i->`), 행렬 곱(`ij,jk->ik`), 외적(`i,j->ij`), 전치(`ij->ji`)를 처리하는 기본적인 `einsum(subscripts, *tensors)` 함수를 구현합니다. 서브스크립트 문자열을 파싱하고, 축약(contracted) 인덱스를 식별한 뒤, 모든 인덱스 조합에 대해 루프를 수행하세요. 결과를 `np.einsum`와 비교해 보세요.

4. **고급 -- 어텐션(attention) 형태 추적기.** `batch_size`, `seq_len`, `embed_dim`, `num_heads`를 입력으로 받아 멀티 헤드 어텐션(multi-head attention)의 모든 단계(입력, Q/K/V 프로젝션, 헤드 분할, 어텐션 스코어, softmax 가중치, 가중합, 헤드 병합, 출력 프로젝션)에서 정확한 형태를 출력하는 함수를 작성하세요. `demo_attention_einsum()` 출력과 일치하는지 확인하세요.

### Hints

**연습문제 1.** 각 원소가 자신의 위치를 나타내도록 `list(range(24))`로 텐서를 만드세요. 각 `reshape` 호출 후 `flatten().to_list()`를 출력하여 0부터 23까지 순서대로 유지되는지 확인하세요.

**연습문제 2.** 차원은 두 크기가 같거나 둘 중 하나가 1일 때만 브로드캐스팅될 수 있으므로, 이 규칙을 먼저 확인하고 위반 시 에러를 발생시키세요. 형태가 `(3, 1)`와 `(1, 4)`인 경우, 결과의 `[i][j]` 항목이 `a[i][0] + b[0][j]`과 일치하는지 확인하고 전체 결과를 NumPy와 비교하세요.

**연습문제 3.** 서브스크립트를 `->` 기준으로 분리하세요. 입력에는 있지만 출력에는 없는 문자가 합산되는 인덱스이므로, 모든 문자의 각 값에 대해 루프를 돌며 출력 위치에 더하세요. 작은 랜덤 배열에 `np.allclose(mine, np.einsum(...))`를 사용하여 네 가지 경우를 각각 테스트하세요.

**연습문제 4.** 헤드 크기 `embed_dim // num_heads`로 시작하고 `embed_dim % num_heads == 0`임을 단언(assert)하세요. 어텐션 스코어의 형태는 `(batch_size, num_heads, seq_len, seq_len)`이므로, 여러분이 출력한 형태를 이 형태 및 `demo_attention_einsum()`가 출력하는 형태와 비교해 확인하세요.

## Key Terms

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| 텐서(Tensor) | "그냥 차원이 더 많은 행렬" | 균일한 타입과 정의된 형태, 스트라이드, 연산을 갖춘 다차원 배열 |
| 랭크(Rank) | "차원의 수" | 축(axes)의 수. 행렬의 랭크는 2이며, 선형대수의 행렬 계수(matrix rank)와 다름 |
| 형태(Shape) | "텐서의 크기" | 각 축의 크기를 나열한 튜플. `(2, 3)`는 2개 행, 3개 열을 의미 |
| 스트라이드(Stride) | "메모리가 배치된 방식" | 각 축을 따라 한 위치 전진하기 위해 건너뛰어야 하는 원소의 수 |
| 브로드캐스팅(Broadcasting) | "형태가 달라도 알아서 맞춰줌" | 엄격한 규칙 집합: 오른쪽부터 정렬하며, 차원 크기가 같거나 둘 중 하나가 1이어야 함 |
| 인접(Contiguous) | "텐서가 일반적인 상태임" | 논리적 배치에서 빈틈이나 재정렬 없이 메모리에 순차적으로 저장된 상태 |
| Einsum | "행렬 곱을 멋있게 쓰는 방법" | 모든 텐서 축약, 외적, 대각합(trace), 전치를 한 줄로 표현하는 일반 표기법 |
| 뷰(View) | "reshape과 같음" | 동일한 메모리 버퍼를 공유하지만 형태/스트라이드 메타데이터가 다른 텐서. 비인접(non-contiguous) 데이터에서는 실패 |
| 축약(Contraction) | "특정 인덱스에 대해 합산하기" | 텐서 간에 공유된 인덱스를 곱하고 합산하여 더 낮은 랭크의 결과를 산출하는 일반적인 연산 |
| NCHW / NHWC | "PyTorch 대 TensorFlow 포맷" | 이미지 텐서의 메모리 배치 규칙. NCHW는 채널을 공간 차원 앞에 두고, NHWC는 뒤에 둠 |

## Further Reading

- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) -- 시각적 예시와 함께 정리된 표준 규칙
- [PyTorch Tensor Views](https://pytorch.org/docs/stable/tensor_view.html) -- 뷰가 동작하는 경우와 복사가 일어나는 경우
- [einops](https://github.com/arogozhnikov/einops) -- 텐서 reshape을 가독성 높고 안전하게 만들어 주는 라이브러리
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) -- 어텐션을 통과하는 텐서 형태 흐름 시각화
- [Einstein Summation in NumPy](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) -- 예제가 포함된 einsum 전체 공식 문서
