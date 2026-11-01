---
title_en: "Introduction to JAX"
source: "phases/03-deep-learning-core/12-intro-to-jax"
source_sha: "9fe3133043ee5c58"
model: "gemini-3.8-flash"
translated_at: "2026-11-01"
---
# JAX 소개

> PyTorch는 텐서를 직접 변경합니다. TensorFlow는 그래프를 생성합니다. JAX는 순수 함수(pure function)를 컴파일합니다. 이 마지막 특성은 딥러닝을 바라보는 방식을 완전히 바꿉니다.

**유형:** Build (실습)
**사용 언어:** Python
**선수 레슨:** Phase 03 Lessons 01-10, basic NumPy
**소요 시간:** ~90분

## 학습 목표

- JAX의 함수형 API(jax.numpy, jax.grad, jax.jit, jax.vmap)를 사용하여 순수 함수 기반의 신경망 코드를 작성합니다.
- PyTorch의 즉시 실행 및 상태 변경(eager mutation) 방식과 JAX의 함수형 컴파일 모델 간의 핵심적인 설계 차이를 설명합니다.
- jit 컴파일과 vmap 벡터화를 적용하여 순수 Python 코드 대비 훈련 루프를 가속합니다.
- JAX에서 간단한 신경망을 훈련하고, 명시적 상태 관리 방식을 PyTorch의 객체 지향적 접근 방식과 비교 대조합니다.

## 문제 상황

PyTorch로 신경망을 구축하는 방법은 이미 잘 알려져 있습니다. `nn.Module`을 정의하고, `.backward()`을 호출하며, 옵티마이저를 스텝합니다. 잘 동작하며, 수백만 명이 사용하고 있습니다.

하지만 PyTorch의 DNA에는 근본적인 제약이 내재되어 있습니다. Python에서 연산을 하나씩 즉시(eager) 추적한다는 점입니다. 모든 `tensor + tensor`은 별도의 커널 실행(kernel launch)으로 이어집니다. 매 훈련 스텝마다 동일한 Python 코드를 재해석합니다. 이는 2,048개의 TPU에 걸쳐 5,400억 개 매개변수 규모의 모델을 훈련해야 하는 상황 전까지는 문제없이 동작합니다. 하지만 대규모 환경에서는 이러한 오버헤드가 치명적으로 작용합니다.

Google DeepMind는 Gemini를 JAX로 훈련합니다. Anthropic도 Claude를 JAX로 훈련했습니다. 이는 단순한 작업이 아니며, 지구상에서 가장 거대한 신경망 훈련 작업들입니다. 이들이 JAX를 선택한 이유는 훈련 루프를 일련의 Python 호출이 아닌 컴파일 가능한 프로그램으로 취급하기 때문입니다.

JAX는 세 가지 강력한 기능(자동 미분(automatic differentiation), XLA로의 JIT 컴파일, 자동 벡터화(automatic vectorization))을 갖춘 NumPy입니다. 단일 예제를 처리하는 함수를 작성하면, JAX는 원래 함수를 전혀 변경하지 않고도 배치를 처리하고, 기울기(gradient)를 계산하며, 기계어로 컴파일하고, 여러 디바이스에서 실행되는 함수로 변환해 줍니다.

## 핵심 개념

### JAX의 철학

JAX는 함수형 프레임워크입니다. 클래스도, 변경 가능한 상태(mutable state)도, `.backward()` 메서드도 없습니다. 그 대신 다음과 같은 방식을 사용합니다:

| PyTorch | JAX |
|---------|-----|
| 상태를 가진 `nn.Module` 클래스 | 순수 함수: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 즉시 실행(Eager execution) | XLA 기반 JIT 컴파일 |
| `for x in batch:` 수동 루프 | `jax.vmap(f)` 자동 벡터화 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 자동 병렬화 |
| 변경 가능한 `model.parameters()` | 배열로 이루어진 불변(Immutable) pytree |

이는 단순한 코딩 스타일의 선호 문제가 아닙니다. 컴파일러의 제약 조건입니다. JIT 컴파일을 수행하려면 순수 함수(동일한 입력에 대해 항상 동일한 출력을 생성하고 부작용(side effect)이 없는 함수)여야 합니다. 이러한 제약 덕분에 100배의 속도 향상이 가능해집니다.

### jax.numpy: 익숙한 인터페이스

JAX는 가속기 상에서 NumPy API를 재구현합니다:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

함수 이름도 동일합니다. 브로드캐스팅 규칙도 동일합니다. 슬라이싱 방식도 동일합니다. 하지만 배열은 GPU/TPU에 상주하며, 모든 연산은 컴파일러에 의해 추적(trace)될 수 있습니다.

한 가지 중요한 차이점은 JAX 배열이 불변(immutable)이라는 것입니다. `a[0] = 5`과 같은 방식은 불가능합니다. 대신 `a = a.at[0].set(5)`를 사용해야 합니다. 처음 일주일은 어색하게 느껴질 수 있지만, 이내 이해하게 됩니다. 바로 이 불변성이 `grad`, `jit`, `vmap`와 같은 변환을 조합 가능(composable)하게 만드는 핵심이라는 점을 말입니다.

### jax.grad: 함수형 자동 미분

PyTorch는 텐서에 기울기를 연결합니다(`.grad`). 반면 JAX는 함수에 기울기를 연결합니다.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`는 함수를 받아 기울기를 계산하는 새로운 함수를 반환합니다. `.backward()` 호출도 없고, 텐서에 계산 그래프가 저장되지도 않습니다. 기울기는 그저 호출하거나 조합하거나 JIT 컴파일할 수 있는 또 하나의 함수일 뿐입니다.

이는 얼마든지 임의로 조합할 수 있습니다:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

2계 도함수, 3계 도함수, 야코비안(Jacobian), 헤시안(Hessian) 모두 `grad`를 조합하는 것만으로 계산할 수 있습니다. PyTorch도 이를 수행할 수 있지만(`torch.autograd.functional.hessian`), 사후에 덧붙여진 기능에 가깝습니다. 반면 JAX에서는 이것이 기본 토대입니다.

제약 사항: `grad`는 오직 순수 함수에서만 동작합니다. 함수 내부에 print 문을 넣어서는 안 됩니다(실행 시점이 아닌 추적(tracing) 시점에 실행됩니다). 외부 상태를 변경해서도 안 됩니다. 명시적인 키 관리 없는 난수 생성도 허용되지 않습니다.

### jit: XLA로 컴파일

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

첫 번째 호출 시 JAX는 함수를 추적(trace)합니다. 즉, 실제로 연산을 실행하지 않고 어떤 연산이 일어나는지 기록합니다. 그런 다음 해당 추적 결과를 TPU 및 GPU용 Google 컴파일러인 XLA(Accelerated Linear Algebra)에 전달합니다. XLA는 연산을 융합(fuse)하고, 불필요한 메모리 복사를 제거하며, 최적화된 기계어를 생성합니다.

이후의 호출에서는 Python을 완전히 건너뜁니다. 컴파일된 코드가 가속기 상에서 C++ 수준의 속도로 실행됩니다.

JIT가 유용한 경우:
- 훈련 스텝 (동일한 계산이 수천 번 반복됨)
- 추론(inference) (동일한 모델에 다른 입력이 주어짐)
- 유사한 형태의 입력을 받아 두 번 이상 호출되는 모든 함수

JIT가 불리한 경우:
- 값에 의존하는 Python 제어 흐름을 가진 함수 (x가 추적 대상 배열인 `if x > 0`)
- 일회성 계산 (컴파일 오버헤드가 실행 시간보다 큰 경우)
- 디버깅 (추적 과정에서 실제 실행 내용이 가려짐)

제어 흐름에 대한 제약은 매우 현실적입니다. `jax.lax.cond`는 `if/else`를 대체합니다. `jax.lax.scan`는 `for` 루프를 대체합니다. 이는 선택 사항이 아니며, 컴파일을 위해 치러야 할 대가입니다.

### vmap: 자동 벡터화

단일 예제를 처리하는 함수를 작성합니다:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`는 이 함수를 배치를 처리할 수 있도록 변환(lift)합니다:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`의 의미는 다음과 같습니다. `params`에 대해서는 배치 처리를 하지 않고(공유됨), `x`의 0번 축을 기준으로 배치 처리를 수행합니다. 수동으로 `for` 루프를 돌릴 필요가 없습니다. reshape할 필요도 없고, 배치 차원을 일일이 전달할 필요도 없습니다. JAX가 배치 차원을 파악하여 전체 계산을 벡터화합니다.

이는 단순한 문법적 설탕(syntactic sugar)이 아닙니다. `vmap`는 Python 루프보다 10~100배 빠르게 실행되는 융합된(fused) 벡터화 코드를 생성합니다. 또한 `jit` 및 `grad`와도 결합할 수 있습니다:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

예제별 기울기(per-example gradient)를 단 한 줄로 계산합니다. PyTorch에서는 편법(hack) 없이는 거의 불가능한 작업입니다.

### pmap: 다중 디바이스 간 데이터 병렬 처리

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`는 사용 가능한 모든 디바이스(GPU/TPU)에 함수를 복제하고 배치를 분할합니다. 함수 내부에서는 `jax.lax.pmean`과 `jax.lax.psum`가 디바이스 간 기울기를 동기화합니다.

Google은 `pmap`(및 그 후속인 `shard_map`)을 사용하여 수천 개의 TPU v5e 칩에 걸쳐 Gemini를 훈련합니다. 프로그래밍 모델은 간단합니다. 단일 디바이스용 버전을 작성하고, `pmap`로 감싸면 끝납니다.

### Pytree: 범용 데이터 구조

JAX는 리스트, 튜플, 딕셔너리, 배열 등이 중첩된 구조인 'pytree'를 기반으로 동작합니다. 모델 매개변수 역시 pytree입니다:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

모든 JAX 변환(`grad`, `jit`, `vmap`)은 pytree를 순회하는 방법을 알고 있습니다. `jax.tree.map(f, tree)`는 모든 리프(leaf)에 `f`를 적용합니다. 옵티마이저가 모든 매개변수를 한 번에 업데이트하는 방식이 바로 이렇습니다:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

`.parameters()` 메서드도, 매개변수 등록 과정도 없습니다. 트리 구조 자체가 곧 모델입니다.

### 함수형 vs 객체 지향

PyTorch는 객체 내부에 상태를 저장합니다:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX는 명시적 상태를 가진 순수 함수를 사용합니다:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

매개변수(params)는 인자로 전달됩니다. 아무것도 내부에 저장되지 않으며, 아무것도 변경되지 않습니다. 덕분에 모든 함수는 테스트 가능하고, 조합 가능하며, 컴파일할 수 있게 됩니다. 이는 매개변수를 직접 관리해야 한다는 의미이기도 합니다. 또는 Flax나 Equinox 같은 라이브러리를 활용할 수도 있습니다.
### JAX 생태계

JAX는 원시 연산(primitives)을 제공합니다. 편의성(ergonomics)은 라이브러리가 제공합니다.

| 라이브러리 | 역할 | 스타일 |
|---------|------|-------|
| **Flax** (Google) | 신경망 레이어 | 명시적 상태를 가진 `nn.Module` |
| **Equinox** (Patrick Kidger) | 신경망 레이어 | Pytree 기반, 파이썬다운(Pythonic) 방식 |
| **Optax** (DeepMind) | 옵티마이저 + LR 스케줄 | 조합 가능한 그래디언트 변환 |
| **Orbax** (Google) | 체크포인팅 | pytree 저장/복원 |
| **CLU** (Google) | 메트릭 + 로깅 | 훈련 루프 유틸리티 |

Optax는 표준 옵티마이저 라이브러리입니다. 그래디언트 변환(Adam, SGD, 클리핑)을 파라미터 업데이트와 분리하여 손쉽게 조합할 수 있도록 만듭니다.

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### JAX vs PyTorch: 언제 무엇을 사용할 것인가

| 요소 | JAX | PyTorch |
|--------|-----|---------|
| TPU 지원 | 최고 수준(First-class, Google이 둘 다 개발) | 커뮤니티 유지 관리(torch_xla) |
| GPU 지원 | 우수함(XLA를 통한 CUDA) | 업계 최고(네이티브 CUDA) |
| 디버깅 | 어려움(추적 + 컴파일) | 쉬움(즉시 실행(eager), 한 줄씩 실행) |
| 생태계 | 연구 중심(Flax, Equinox) | 방대함(HuggingFace, torchvision 등) |
| 채용 | 틈새 시장(Google/DeepMind/Anthropic) | 주류(모든 곳) |
| 대규모 훈련 | 뛰어남(XLA, pmap, mesh) | 우수함(FSDP, DeepSpeed) |
| 프로토타이핑 속도 | 느림(함수형 오버헤드) | 빠름(상태 변경 후 바로 실행) |
| 프로덕션 추론 | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| 주요 사용자 | DeepMind(Gemini), Anthropic(Claude) | Meta(Llama), OpenAI(GPT), Stability AI |

솔직한 답변을 드리자면, JAX를 사용할 구체적인 이유가 없다면 PyTorch를 사용하십시오. 그 구체적인 이유란 TPU 접근 권한, 샘플별(per-example) 그래디언트의 필요성, 대규모 멀티 디바이스 훈련, 또는 Google/DeepMind/Anthropic 근무 등입니다.

### JAX의 난수

JAX에는 전역 난수 상태가 없습니다. 모든 난수 연산에는 명시적인 PRNG 키가 필요합니다.

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

처음에는 이것이 번거롭게 느껴집니다. 하지만 이는 여러 디바이스와 컴파일 전반에 걸쳐 재현성을 보장합니다. 이는 멀티 GPU 환경에서 PyTorch의 `torch.manual_seed`이 보장할 수 없는 특성입니다.

```figure
batchnorm-effect
```

## 구현하기

### 1단계: 환경 설정 및 데이터

JAX와 Optax를 사용하여 MNIST 데이터셋에 3계층 MLP를 훈련합니다. 784개의 입력, 각각 256개와 128개의 뉴런을 가진 두 개의 은닉층, 10개의 출력 클래스로 구성됩니다.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 2단계: 파라미터 초기화

클래스는 없습니다. pytree를 반환하는 함수뿐입니다.

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

수동으로 구현한 He 초기화입니다. 하나의 시드에서 세 개의 PRNG 키를 분할(split)합니다. 모든 가중치는 중첩된 dict 안의 불변(immutable) 배열입니다.

### 3단계: 순전파

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

순수 함수(pure functions)입니다. 파라미터가 들어가고, 예측값이 나옵니다. `self`도 없고, 저장된 상태도 없습니다. `loss_fn`은 softmax, log, 음의 평균을 사용하여 교차 엔트로피를 바닥부터 직접 계산합니다.

### 4단계: JIT 컴파일된 훈련 스텝

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad`는 한 번의 패스로 손실값과 그래디언트를 모두 반환합니다. `@jax.jit` 데코레이터는 두 함수를 모두 XLA로 컴파일합니다. 첫 번째 호출 이후, 각 훈련 스텝은 Python을 거치지 않고 실행됩니다.

### 5단계: 훈련 루프

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 에포크. 약 97%의 테스트 정확도. 첫 번째 에포크는 느립니다(JIT 컴파일). 2~10 에포크는 빠릅니다.

어떤 것이 없는지 주목하십시오. `.zero_grad()`도 없고, `.backward()`도 없으며, `.step()`도 없습니다. 전체 업데이트는 하나의 조합된 함수 호출입니다. 그래디언트가 계산되고, Adam에 의해 변환되며, 파라미터에 적용되는 과정이 모두 `train_step` 내부에서 일어납니다.

## 활용하기

### Flax: Google의 표준

Flax는 가장 널리 쓰이는 JAX 신경망 라이브러리입니다. `nn.Module`를 다시 도입하되, 명시적인 상태 관리를 결합했습니다.

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

PyTorch와 구조는 같지만, `params`가 모델과 분리되어 있습니다. `model.init()`는 파라미터를 생성합니다. `model.apply(params, x)`는 순전파를 실행합니다. 모델 객체는 상태를 갖지 않습니다.

### Equinox: 파이썬다운 대안

Equinox(Patrick Kidger 개발)는 모델을 pytree로 표현합니다.

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

모델 자체가 pytree입니다. `.apply()`가 필요하지 않습니다. 파라미터는 단지 모델의 리프(leaves)일 뿐입니다. 이는 JAX의 사고방식에 더 가깝습니다.

### Optax: 조합 가능한 옵티마이저

Optax는 그래디언트 변환을 업데이트로부터 분리합니다.

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

그래디언트 클리핑, 학습률 웜업, 가중치 감쇠(weight decay) 등이 모두 하나의 변환 체인으로 조합됩니다. 각 변환은 그래디언트를 확인하고, 이를 수정한 후 다음 변환으로 전달합니다. 거대한 단일 옵티마이저 클래스는 존재하지 않습니다.

## 배포하기

**설치:**

```bash
pip install jax jaxlib optax flax
```

GPU 지원:

```bash
pip install jax[cuda12]
```

TPU(Google Cloud) 지원:

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**성능 주의 사항:**

- 첫 번째 JIT 호출은 느립니다(컴파일). 벤치마킹 전에 웜업을 수행하십시오.
- JIT 내부에서 JAX 배열에 대해 Python 루프를 사용하는 것을 피하십시오. `jax.lax.scan` 또는 `jax.lax.fori_loop`를 사용하십시오.
- `jax.debug.print()`는 JIT 내부에서 작동합니다. 일반 `print()`는 작동하지 않습니다.
- `jax.profiler` 또는 TensorBoard로 프로파일링하십시오. XLA 컴파일로 인해 병목 현상이 가려질 수 있습니다.
- JAX는 기본적으로 GPU 메모리의 75%를 미리 할당합니다. 비활성화하려면 `XLA_PYTHON_CLIENT_PREALLOCATE=false`를 설정하십시오.

**체크포인팅:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**이번 레슨의 결과물:**
- `outputs/prompt-jax-optimizer.md` -- 올바른 JAX 옵티마이저 구성을 선택하기 위한 프롬프트
- `outputs/skill-jax-patterns.md` -- JAX의 함수형 패턴을 다루는 스킬

## 실습 과제

1. MLP에 드롭아웃(dropout)을 추가하십시오. JAX에서 드롭아웃은 PRNG 키를 필요로 합니다. 순전파를 통해 키를 전달하고 각 드롭아웃 레이어에 맞게 분할하십시오. 드롭아웃 적용 전후의 테스트 정확도를 비교하십시오.

2. `jax.vmap`를 사용하여 32개의 MNIST 이미지 배치에 대해 샘플별 그래디언트를 계산하십시오. 각 샘플에 대한 그래디언트 노름(gradient norm)을 계산하십시오. 어떤 샘플이 가장 큰 그래디언트를 가지며, 그 이유는 무엇입니까?

3. 수동으로 작성된 forward 함수를 레이어 수에 관계없이 작동하는 일반화된 `mlp_forward(params, x)` 함수로 대체하십시오. `jax.tree.leaves`를 사용하여 깊이를 자동으로 결정하십시오.

4. `@jax.jit`가 있을 때와 없을 때의 훈련 스텝을 벤치마킹하십시오. 각각 100 스텝의 시간을 측정하십시오. 사용 중인 하드웨어에서 속도 향상이 얼마나 됩니까? 첫 번째 호출 시 컴파일 오버헤드는 얼마입니까?

5. `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`를 조합하여 그래디언트 클리핑을 구현하십시오. 클리핑을 적용했을 때와 적용하지 않았을 때를 각각 훈련하십시오. 훈련 과정에서 그래디언트 노름을 시각화하여 그 효과를 확인하십시오.
## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| XLA | "JAX를 빠르게 만들어 주는 것" | Accelerated Linear Algebra의 약칭으로, 연산을 융합(fuse)하고 계산 그래프로부터 최적화된 GPU/TPU 커널을 생성하는 컴파일러 |
| JIT | "적시 컴파일(Just-in-time compilation)" | JAX가 함수를 처음 호출할 때 추적(trace)하여 XLA로 컴파일한 다음, 이후 호출부터는 컴파일된 버전을 실행함 |
| 순수 함수(Pure function) | "부수 효과(side effect)가 없는 함수" | 출력이 오직 입력에만 의존하는 함수. 전역 상태, 내부 변경(mutation), 명시적 키 없는 무작위성(randomness)이 모두 없음 |
| vmap | "자동 배치화(Auto-batching)" | 코드를 다시 작성하지 않고도 단일 예제를 처리하는 함수를 배치(batch) 단위로 처리하는 함수로 변환함 |
| pmap | "자동 병렬화(Auto-parallelism)" | 여러 디바이스에 걸쳐 함수를 복제하고 입력 배치를 분할함 |
| Pytree | "배열이 중첩된 딕셔너리" | JAX가 순회하고 변환할 수 있는 리스트, 튜플, 딕셔너리, 배열 등의 모든 중첩 구조 |
| 트레이싱(Tracing) | "계산 기록하기" | JAX가 실제 결과를 계산하지 않고 추상적인 값을 사용하여 함수를 실행함으로써 계산 그래프를 구축함 |
| 함수형 자동 미분(Functional autodiff) | "함수의 grad" | 텐서에 그래디언트 저장소를 연결하는 대신, 함수를 변환하여 도함수를 계산함 |
| Optax | "JAX의 옵티마이저 라이브러리" | Adam, SGD, 클리핑(clipping), 스케줄링(scheduling) 등 서로 연결하여 조합할 수 있는 그래디언트 변환 라이브러리 |
| Flax | "JAX 버전의 nn.Module" | 상태를 명시적으로 유지하면서 레이어 추상화를 제공하는 Google의 JAX용 신경망 라이브러리 |

## 참고 자료

- JAX 공식 문서: https://jax.readthedocs.io/ -- 공식 문서로, grad, jit, vmap에 관한 훌륭한 튜토리얼을 제공합니다.
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- 설계 철학을 설명하는 원저 논문
- Flax 공식 문서: https://flax.readthedocs.io/ -- Google의 JAX용 신경망 라이브러리
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Flax의 파이썬다운(Pythonic) 대안
- DeepMind, "Optax: composable gradient transformation and optimisation" -- 표준 옵티마이저 라이브러리
- "You Don't Know JAX" (Colin Raffel, 2020) -- T5의 공동 저자가 작성한 JAX의 주의점과 주요 패턴에 대한 실용적인 가이드
