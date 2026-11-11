---
title_en: "Tensor Operations"
source: "phases/01-math-foundations/12-tensor-operations"
source_sha: "07b6d7983ec360dd"
model: "gemini-3.8-flash"
translated_at: "2026-11-11"
---
# 텐서 연산

> 텐서는 데이터와 딥러닝 사이의 공통 언어입니다. 모든 이미지, 모든 문장, 모든 그래디언트가 텐서를 통해 흐릅니다.

**유형:** Build (실습)
**사용 언어:** Python, TypeScript
**선수 레슨:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**소요 시간:** ~90분

## 학습 목표

- shape, stride, reshape, transpose, 원소별(element-wise) 연산을 지원하는 텐서 클래스를 밑바닥부터 구현합니다.
- 브로드캐스팅(broadcasting) 규칙을 적용하여 데이터를 복사하지 않고 서로 다른 shape의 텐서를 연산합니다.
- 내적, 행렬 곱셈, 외적, 배치 연산을 위한 einsum 표현식을 작성합니다.
- 멀티 헤드 어텐션(multi-head attention)의 모든 단계에서 정확한 텐서 shape의 변화를 추적합니다.

## 문제 상황

트랜스포머(Transformer)를 구축합니다. 순전파(forward pass) 코드는 깔끔해 보입니다. 실행해 보니 다음과 같은 에러가 발생합니다: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)`. shape을 유심히 살펴봅니다. transpose를 시도합니다. 이제는 `Expected 4D input (got 3D input)` 에러가 뜹니다. unsqueeze를 추가합니다. 그러자 다른 곳이 깨집니다.

shape 에러는 딥러닝 코드에서 가장 흔하게 발생하는 버그입니다. 개념상 어렵지는 않습니다. 각 연산마다 shape에 대한 계약(contract)이 정해져 있기 때문입니다. 하지만 에러는 순식간에 불어납니다. 트랜스포머에는 수십 개의 reshape, transpose, 브로드캐스팅이 연쇄적으로 얽혀 있습니다. 축 하나만 잘못 지정해도 에러가 연쇄적으로 번집니다. 더 최악인 점은, 일부 shape 실수는 에러를 전혀 발생시키지 않는다는 것입니다. 엉뚱한 차원을 따라 브로드캐스팅하거나 잘못된 축에 대해 합을 구함으로써 조용히 엉터리 결과를 만들어냅니다.

행렬은 두 대상 집합 간의 쌍별(pairwise) 관계를 다룹니다. 하지만 실제 데이터는 2차원에 들어맞지 않습니다. 224x224 크기의 RGB 이미지 32장으로 이루어진 배치는 4D 텐서입니다: `(32, 3, 224, 224)`. 12개 헤드를 가진 셀프 어텐션(self-attention) 역시 4D입니다: `(batch, heads, seq_len, head_dim)`. 따라서 임의의 차원 수로 일반화되고, 모든 차원에 걸쳐 깔끔하게 조합되는 연산을 갖춘 자료구조가 필요합니다. 그 자료구조가 바로 텐서입니다. 텐서 연산을 마스터하면 shape 에러는 아주 쉽게 디버깅할 수 있게 됩니다.

## 개념

### 텐서란 무엇인가

텐서는 단일한 데이터 타입을 갖는 다차원 숫자 배열입니다. 차원의 수를 **랭크(rank)**(또는 **차수(order)**)라고 합니다. 각 차원은 **축(axis)**입니다. **shape**은 각 축을 따른 크기를 나열한 튜플입니다.

```mermaid
graph LR
    S["Scalar<br/>rank 0<br/>shape: ()"] --> V["Vector<br/>rank 1<br/>shape: (3,)"]
    V --> M["Matrix<br/>rank 2<br/>shape: (2,3)"]
    M --> T3["3D Tensor<br/>rank 3<br/>shape: (2,2,2)"]
    T3 --> T4["4D Tensor<br/>rank 4<br/>shape: (B,C,H,W)"]
```

전체 원소 수 = 모든 크기의 곱. shape `(2, 3, 4)`은 $2 \cdot 3 \cdot 4 = 24$개의 원소를 담습니다.

### 딥러닝에서의 텐서 shape

관례에 따라 서로 다른 데이터 타입은 특정 텐서 shape으로 매핑됩니다.

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

PyTorch는 NCHW(채널 우선, channels-first)를 사용합니다. TensorFlow는 기본적으로 NHWC(채널 마지막, channels-last)를 사용합니다. 레이아웃이 일치하지 않으면 조용히 성능이 저하되거나 에러가 발생합니다.

### 메모리 레이아웃의 동작 원리

메모리 상에서 2D 배열은 1D 바이트 시퀀스로 존재합니다. **Stride**는 각 축을 따라 한 걸음 이동할 때 몇 개의 원소를 건너뛰어야 하는지 알려줍니다.

```mermaid
graph LR
    subgraph "Row-major (C order)"
        R["a b c d e f<br/>strides: (3, 1)"]
    end
    subgraph "Column-major (F order)"
        C["a d b e c f<br/>strides: (1, 2)"]
    end
```

Transpose는 데이터를 이동시키지 않습니다. stride를 맞바꿀 뿐이므로, 텐서는 **비연속적(non-contiguous)**이 됩니다. 즉, 한 행에 속한 원소들이 더 이상 메모리에서 인접하지 않게 됩니다.

### 브로드캐스팅 규칙

브로드캐스팅을 사용하면 데이터를 복사하지 않고도 서로 다른 shape의 텐서에 연산을 수행할 수 있습니다. shape을 오른쪽부터 맞춥니다. 두 차원의 크기가 같거나 둘 중 하나가 1이면 호환됩니다. 차원 수가 적은 쪽은 왼쪽에 1이 채워집니다.

```
Tensor A:     (8, 1, 6, 1)
Tensor B:        (7, 1, 5)
Padded B:     (1, 7, 1, 5)
Result:       (8, 7, 6, 5)
```

### Einsum: 범용 텐서 연산

아인슈타인 표기법(Einstein summation)은 각 축에 문자를 부여합니다. 입력에는 있지만 출력에는 없는 축은 합산(소거)됩니다. 양쪽 모두에 있는 축은 유지됩니다.

```mermaid
graph LR
    subgraph "matmul: ik,kj -> ij"
        A["A(I,K)"] --> |"sum over k"| C["C(I,J)"]
        B["B(K,J)"] --> |"sum over k"| C
    end
```

주요 패턴: `i,i->` (내적), `i,j->ij` (외적), `ii->` (대각합/trace), `ij->ji` (transpose), `bij,bjk->bik` (배치 행렬 곱), `bhtd,bhsd->bhts` (어텐션 점수).

```figure
tensor-broadcast
```

## 직접 만들기

코드는 `code/tensors.py`에 있습니다. 각 단계는 그곳의 구현을 참조합니다. `code/tensors.ts`에 있는 TypeScript 포팅 버전도 동일한 출력을 생성합니다.

```bash
python3 code/tensors.py
npx tsx code/tensors.ts
```

### 1단계: 텐서 스토리지와 stride

텐서는 1차원으로 펼쳐진(flat) 숫자 리스트와 shape 메타데이터를 저장합니다. Stride는 다차원 인덱스를 1차원 위치로 매핑하는 방법을 인덱싱 로직에 알려줍니다.

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

shape `(3, 4)`의 경우 stride는 `(4, 1)`입니다. 즉, 다음 행으로 이동하려면 4개 원소를 건너뛰고, 다음 열로 이동하려면 1개 원소를 건너뜁니다.

### 2단계: Reshape, squeeze, unsqueeze

Reshape은 원소의 순서를 바꾸지 않고 shape만 변경합니다. 전체 원소 수는 반드시 동일하게 유지되어야 합니다. 한 차원의 크기를 추론하려면 `-1`을 사용합니다.

```python
t = Tensor(list(range(12)), shape=(2, 6))
r = t.reshape((3, 4))
r = t.reshape((-1, 3))
```

Squeeze는 크기가 1인 축을 제거합니다. Unsqueeze는 크기가 1인 축을 삽입합니다. Unsqueeze는 브로드캐스팅에 매우 중요합니다. 배치 `(B, T, D)`에 편향(bias) 벡터 `(D,)`를 더하려면 `(1, 1, D)`로 unsqueeze해야 합니다.

```python
t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
s = t.squeeze()
v = Tensor([1, 2, 3])
u = v.unsqueeze(0)
```

### 3단계: Transpose와 permute

Transpose는 두 축을 맞바꿉니다. Permute는 모든 축의 순서를 재배치합니다. NCHW와 NHWC 사이의 변환은 이 방식으로 수행합니다.

```python
mat = Tensor(list(range(6)), shape=(2, 3))
tr = mat.transpose(0, 1)

t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))
perm = t4d.permute((0, 2, 3, 1))
```

Transpose나 permute를 수행한 후, 텐서는 메모리 상에서 비연속적이 됩니다. PyTorch에서는 비연속적인 텐서에 대해 `view` 연산이 실패하므로, `reshape`을 사용하거나 먼저 `.contiguous()`를 호출해야 합니다.

### 4단계: 원소별 연산과 축약

원소별(element-wise) 연산(덧셈, 곱셈, 뺄셈)은 각 원소에 독립적으로 적용되며 shape을 유지합니다. 축약(reduction) 연산(합, 평균, 최댓값)은 하나 이상의 축을 압축합니다.

```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[10, 20], [30, 40]])
c = a + b
d = a * 2
s = a.sum(axis=0)
```

CNN에서의 전역 평균 풀링(global average pooling): `(B, C, H, W).mean(axis=[2, 3])`은 `(B, C)`를 생성합니다. NLP에서의 시퀀스 평균 풀링(sequence mean pooling): `(B, T, D).mean(axis=1)`은 `(B, D)`을 생성합니다.

### 5단계: NumPy를 이용한 브로드캐스팅

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

브로드캐스팅을 통한 쌍별 거리(pairwise distance) 계산: `(M, 2)`를 `(M, 1, 2)`로, `(N, 2)`을 `(1, N, 2)`로 reshape한 다음 뺄셈, 제곱, 마지막 축을 따른 합산, 제곱근을 차례로 수행합니다. 결과: `(M, N)`.

### 6단계: Einsum 연산

`demo_einsum()` 및 `demo_einsum_gallery()` 함수는 흔히 쓰이는 모든 패턴을 단계별로 살펴봅니다.

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

텐서 축약(contraction)의 계산 비용은 모든 인덱스 크기(유지되는 축과 합산되는 축 모두)의 곱입니다. 크기가 $B=32$, $I=128$, $J=64$, $K=128$인 `bij,bjk->bik`의 경우: $32 \cdot 128 \cdot 64 \cdot 128 = 33{,}554{,}432$번의 곱셈-덧셈(multiply-add) 연산이 수행됩니다.

### 7단계: Einsum을 이용한 어텐션 메커니즘

`demo_attention_einsum()` 함수는 멀티 헤드 어텐션을 처음부터 끝까지 구현합니다.

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

모든 단계가 텐서 연산입니다: 프로젝션(einsum을 통한 행렬 곱), 헤드 분할(reshape + transpose), 어텐션 점수(einsum을 통한 배치 행렬 곱), 가중합(einsum을 통한 배치 행렬 곱), 헤드 병합(transpose + reshape), 출력 프로젝션(einsum을 통한 행렬 곱).

## 사용하기

### 직접 구현한 클래스 vs NumPy

| 연산 | 직접 구현 (Tensor 클래스) | NumPy |
|---|---|---|
| 생성 | `Tensor([[1,2],[3,4]])` | `np.array([[1,2],[3,4]])` |
| Reshape | `t.reshape((3,4))` | `a.reshape(3,4)` |
| Transpose | `t.transpose(0,1)` | `a.T` 또는 `a.transpose(0,1)` |
| Squeeze | `t.squeeze(0)` | `np.squeeze(a, 0)` |
| Sum | `t.sum(axis=0)` | `a.sum(axis=0)` |
| Einsum | N/A | `np.einsum("ij,jk->ik", a, b)` |

### 직접 구현한 클래스 vs PyTorch

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

PyTorch는 autograd, GPU 지원, 최적화된 BLAS 커널을 추가로 제공합니다. shape의 시맨틱은 동일합니다. 직접 구현한 버전을 이해하고 나면 PyTorch의 shape 에러도 쉽게 읽을 수 있게 됩니다.

### 텐서 연산으로서의 모든 신경망 레이어

| 연산 | 텐서 형태 | Einsum |
|---|---|---|
| 선형 레이어(Linear layer) | `Y = X @ W.T + b` | `"bd,od->bo"` + bias |
| 어텐션 QKV | `Q = X @ W_q` | `"btd,dh->bth"` |
| 어텐션 점수 | `Q @ K.T / sqrt(d)` | `"bhtd,bhsd->bhts"` |
| 어텐션 출력 | `softmax(scores) @ V` | `"bhts,bhsd->bhtd"` |
| 배치 정규화(Batch norm) | `(X - mu) / sigma * gamma` | 원소별 연산 + 브로드캐스팅 |
| 소프트맥스(Softmax) | `exp(x) / sum(exp(x))` | 원소별 연산 + 축약 |
## Ship It

이번 레슨에서는 재사용 가능한 두 가지 프롬프트를 제공합니다:

1. **`outputs/prompt-tensor-shapes.md`** -- 텐서 shape 불일치를 디버깅하기 위한 체계적인 프롬프트입니다. 일반적인 모든 연산(matmul, broadcast, cat, Linear, Conv2d, BatchNorm, softmax)에 대한 결정 테이블과 해결책 조회 테이블이 포함되어 있습니다.

2. **`outputs/prompt-tensor-debugger.md`** -- shape 오류로 작업이 막혔을 때 모든 AI 어시스턴트에 붙여넣어 사용할 수 있는 단계별 디버깅 프롬프트입니다. 에러 메시지와 텐서 shape을 입력하면 정확한 해결책을 제시합니다.

## Exercises

1. **Easy -- Reshape 왕복(round-trip).** shape이 `(2, 3, 4)`인 텐서를 준비합니다. 이를 `(6, 4)`로 reshape한 다음, `(24,)`로, 다시 `(2, 3, 4)`으로 변환합니다. 평탄화(flat)된 데이터를 출력하여 각 단계에서 원소 순서가 유지되는지 확인합니다.

2. **Medium -- 브로드캐스팅(broadcasting) 구현.** 크기가 1인 차원을 타깃 shape에 맞게 확장하는 `broadcast_to(shape)` 메서드를 추가하여 `Tensor` 클래스를 확장합니다. 그런 다음 연산 전에 자동으로 브로드캐스팅을 수행하도록 `_elementwise_op`을 수정합니다. shape `(3, 1)` 및 `(1, 4)`로 테스트하여 `(3, 4)`이 생성되는지 확인합니다.

3. **Hard -- einsum 밑바닥부터 구현하기.** 최소한 내적(`i,i->`), 행렬 곱(`ij,jk->ik`), 외적(`i,j->ij`), 전치(`ij->ji`)를 처리할 수 있는 기본 `einsum(subscripts, *tensors)` 함수를 구현합니다. 첨자(subscript) 문자열을 파싱하고, 축약(contracted) 인덱스를 식별한 다음, 모든 인덱스 조합에 대해 루프를 실행합니다. 구현 결과를 `np.einsum`와 비교합니다.

4. **Hard -- 어텐션(attention) shape 추적기.** `batch_size`, `seq_len`, `embed_dim`, `num_heads`을 입력으로 받아 멀티 헤드 어텐션의 각 단계(입력, Q/K/V 프로젝션, 헤드 분할, 어텐션 스코어, softmax 가중치, 가중합, 헤드 병합, 출력 프로젝션)별 정확한 shape을 출력하는 함수를 작성합니다. `demo_attention_einsum()` 출력과 일치하는지 확인합니다.

### Hints

**Exercise 1.** 각 원소가 자신의 위치를 나타낼 수 있도록 `list(range(24))`로부터 텐서를 생성합니다. 매 `reshape` 호출 후 `flatten().to_list()`을 출력하여 0부터 23까지의 순서가 그대로 유지되는지 확인합니다.

**Exercise 2.** 차원은 두 크기가 동일하거나 둘 중 하나가 1일 때만 브로드캐스팅될 수 있으므로, 먼저 이 규칙을 확인하고 조건을 만족하지 못하면 에러를 발생시킵니다. shape `(3, 1)` 및 `(1, 4)`의 경우, 결과의 `[i][j]` 위치 값이 `a[i][0] + b[0][j]`와 일치하는지 확인하고 전체 결과를 NumPy와 비교합니다.

**Exercise 3.** 첨자를 `->` 기준으로 분할합니다. 입력에는 있지만 출력에는 없는 문자는 합산 인덱스이므로, 모든 문자의 각 값에 대해 루프를 돌며 출력 위치에 더합니다. `np.allclose(mine, np.einsum(...))`을 사용해 작은 크기의 무작위 배열로 네 가지 케이스를 각각 테스트합니다.

**Exercise 4.** 헤드 크기 `embed_dim // num_heads`부터 시작하여 `embed_dim % num_heads == 0`인지 확인(assert)합니다. 어텐션 스코어의 shape은 `(batch_size, num_heads, seq_len, seq_len)`이므로, 출력 결과가 이 shape 및 `demo_attention_einsum()`이 출력하는 shape과 일치하는지 확인합니다.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Tensor | "차원이 더 많은 행렬" | 균일한 타입과 정의된 shape, stride, 연산을 갖는 다차원 배열 |
| Rank | "차원의 수" | 축(axis)의 개수. 행렬의 rank는 행렬 계수(matrix rank)가 아니라 2입니다 |
| Shape | "텐서의 크기" | 각 축을 따른 크기를 나열한 튜플. `(2, 3)`은 2개 행, 3개 열을 의미합니다 |
| Stride | "메모리가 배치된 방식" | 각 축을 따라 한 위치 전진하기 위해 건너뛰어야 하는 원소의 개수 |
| Broadcasting | "shape이 달라도 알아서 잘 동작함" | 엄격한 규칙 집합: 오른쪽부터 정렬하며, 차원 크기가 같거나 둘 중 하나가 1이어야 합니다 |
| Contiguous | "텐서가 정상적인 상태임" | 논리적 배치와 비교했을 때 빈틈이나 순서 변경 없이 원소가 메모리에 순차적으로 저장된 상태 |
| Einsum | "행렬 곱을 쓰는 멋진 방법" | 모든 텐서 축약, 외적, 대각합(trace), 전치를 한 줄로 표현하는 일반화된 표기법 |
| View | "reshape과 동일함" | 동일한 메모리 버퍼를 공유하지만 다른 shape/stride 메타데이터를 갖는 텐서. 비연속적(non-contiguous) 데이터에서는 실패합니다 |
| Contraction | "인덱스에 대해 합산하는 것" | 텐서 간에 공유된 인덱스를 곱하고 합산하여 더 낮은 rank의 결과를 생성하는 일반적인 연산 |
| NCHW / NHWC | "PyTorch 대 TensorFlow 포맷" | 이미지 텐서를 위한 메모리 레이아웃 규칙. NCHW는 채널을 공간 차원 앞에 두고, NHWC는 뒤에 둡니다 |

## Further Reading

- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) -- 시각적 예시와 함께 살펴보는 표준 규칙
- [PyTorch Tensor Views](https://pytorch.org/docs/stable/tensor_view.html) -- 뷰(view)가 동작하는 경우와 복사가 발생하는 경우
- [einops](https://github.com/arogozhnikov/einops) -- 텐서 reshape을 읽기 쉽고 안전하게 만들어 주는 라이브러리
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) -- 어텐션을 통해 흐르는 텐서 shape 시각화
- [Einstein Summation in NumPy](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) -- 예제가 포함된 einsum 전체 공식 문서
