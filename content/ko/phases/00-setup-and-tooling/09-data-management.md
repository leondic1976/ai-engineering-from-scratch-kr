---
title_en: "Data Management"
source: "phases/00-setup-and-tooling/09-data-management"
source_sha: "772d99967d3ccde7"
model: "gemini-3.8-flash"
translated_at: "2026-10-15"
---
# 데이터 관리

> 데이터는 연료입니다. 데이터를 어떻게 관리하느냐가 얼마나 빠르게 나아갈 수 있는지를 결정합니다.

**유형:** Build (실습)
**Language:** Python
**선수 레슨:** Phase 0, Lesson 01
**소요 시간:** ~45분

## 학습 목표

- Hugging Face `datasets` 라이브러리를 사용하여 데이터셋을 로드, 스트리밍 및 캐시하기
- CSV, JSON, Parquet, Arrow 형식 간에 변환하고 각 형식의 장단점 설명하기
- 고정된 난수 시드(random seed)를 사용하여 재현 가능한 훈련/검증/테스트(train/validation/test) 분할 만들기
- `.gitignore`, Git LFS 또는 DVC를 사용하여 대용량 모델 및 데이터셋 파일 관리하기

## 문제

모든 AI 프로젝트는 데이터에서 시작합니다. 데이터셋을 찾고, 다운로드하고, 형식을 변환하고, 훈련 및 평가용으로 분할하며, 실험을 재현할 수 있도록 버전을 관리해야 합니다. 매번 이 작업을 수동으로 수행하는 것은 느리고 오류가 발생하기 쉽습니다. 반복 가능한 워크플로가 필요합니다.

## 개념

```mermaid
graph TD
    A["Hugging Face Hub"] --> B["datasets library"]
    B --> C["Load / Stream"]
    C --> D["Local Cache<br/>~/.cache/huggingface/"]
    B --> E["Format Conversion<br/>CSV, JSON, Parquet, Arrow"]
    E --> F["Data Splits<br/>train / val / test"]
    F --> G["Your Training Pipeline"]
```

Hugging Face `datasets` 라이브러리는 AI 작업에서 데이터를 로드하는 표준적인 방법입니다. 다운로드, 캐싱, 형식 변환, 스트리밍을 별도 설정 없이 바로 처리해 줍니다.

```figure
s0-data-pipeline
```

## 직접 만들기

### 1단계: datasets 라이브러리 설치

```bash
pip install datasets huggingface_hub
```

### 2단계: 데이터셋 로드

```python
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
print(dataset)
print(dataset["train"][0])
```

이 코드는 IMDB 영화 리뷰 데이터셋을 다운로드합니다. 최초 다운로드 이후에는 `~/.cache/huggingface/datasets/`의 캐시에서 로드됩니다.

### 3단계: 대용량 데이터셋 스트리밍

어떤 데이터셋은 디스크에 담기에는 너무 큽니다. 스트리밍을 사용하면 전체를 다운로드하지 않고 행 단위로 로드할 수 있습니다.

```python
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

for i, example in enumerate(dataset):
    print(example["title"])
    if i >= 4:
        break
```

스트리밍을 사용하면 `IterableDataset`을 얻게 됩니다. 데이터가 도착하는 대로 행을 처리하므로 데이터셋 크기와 관계없이 메모리 사용량이 일정하게 유지됩니다.

### 4단계: 데이터셋 형식

`datasets` 라이브러리는 내부적으로 Apache Arrow를 사용합니다. 파이프라인의 필요에 따라 다른 형식으로 변환할 수 있습니다.

```python
dataset = load_dataset("stanfordnlp/imdb", split="train")

dataset.to_csv("imdb_train.csv")
dataset.to_json("imdb_train.json")
dataset.to_parquet("imdb_train.parquet")
```

형식 비교:

| Format | Size | Read Speed | Best For |
|--------|------|-----------|----------|
| CSV | 큼 | 느림 | 사람이 직접 읽기, 스프레드시트 |
| JSON | 큼 | 느림 | API, 중첩 데이터 |
| Parquet | 작음 | 빠름 | 분석, 컬럼형(columnar) 쿼리 |
| Arrow | 작음 | 가장 빠름 | 인메모리(in-memory) 처리 (`datasets`가 내부적으로 사용하는 형식) |

AI 작업에서는 Parquet이 최적의 저장 형식입니다. Arrow는 메모리에서 작업할 때 사용되는 형식입니다. CSV와 JSON은 데이터 교환용입니다.

### 5단계: 데이터 분할

모든 ML 프로젝트에는 세 가지 분할이 필요합니다.

- **Train**: 모델이 학습하는 데 사용됩니다 (일반적으로 80%)
- **Validation**: 훈련 중에 진행 상황을 점검합니다 (일반적으로 10%)
- **Test**: 훈련 완료 후 최종 평가에 사용됩니다 (일반적으로 10%)

미리 분할되어 제공되는 데이터셋도 있습니다. 그렇지 않은 경우에는 직접 분할해야 합니다.

```python
dataset = load_dataset("stanfordnlp/imdb", split="train")

split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
```

재현성을 위해 항상 시드를 설정하십시오. 동일한 시드를 사용하면 매번 동일한 분할이 생성됩니다.

### 6단계: 모델 다운로드 및 캐싱

모델은 대용량 파일입니다. `huggingface_hub` 라이브러리가 다운로드와 캐싱을 처리합니다.

```python
from huggingface_hub import hf_hub_download, snapshot_download

model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json"
)
print(f"Cached at: {model_path}")

model_dir = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print(f"Full model at: {model_dir}")
```

모델은 `~/.cache/huggingface/hub/`에 캐시됩니다. 한 번 다운로드되면 이후 실행 시 즉시 로드됩니다.

### 7단계: 대용량 파일 다루기

모델 가중치와 대용량 데이터셋은 Git에 커밋하지 않아야 합니다. 세 가지 옵션이 있습니다.

**옵션 A: .gitignore (가장 단순함)**

```
*.bin
*.safetensors
*.pt
*.onnx
data/*.parquet
data/*.csv
models/
```

**옵션 B: Git LFS (Git에서 대용량 파일 추적)**

```bash
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
```

Git LFS는 저장소에 포인터를 저장하고 실제 파일은 별도의 서버에 보관합니다. GitHub는 1GB를 무료로 제공합니다.

**옵션 C: DVC (데이터 버전 관리)**

```bash
pip install dvc
dvc init
dvc add data/training_set.parquet
git add data/training_set.parquet.dvc data/.gitignore
git commit -m "Track training data with DVC"
```

DVC는 데이터를 가리키는 작은 `.dvc` 파일을 생성합니다. 데이터 자체는 S3, GCS 또는 다른 원격 스토리지 백엔드에 저장됩니다.

| Approach | Complexity | Best For |
|----------|-----------|----------|
| .gitignore | 낮음 | 개인 프로젝트, 다시 다운로드할 수 있는 데이터 |
| Git LFS | 중간 | Git을 통해 모델 가중치를 공유하는 팀 |
| DVC | 높음 | 재현 가능한 실험, 대용량 데이터셋, 팀 협업 |

이 과정에서는 `.gitignore`로 충분합니다. 여러 머신에서 정확한 실험을 재현해야 할 때는 DVC를 사용하십시오.

### 8단계: 스토리지 패턴

**로컬 스토리지**는 약 10GB 미만의 데이터셋에 적합합니다. HF 캐시가 이를 자동으로 처리합니다.

**클라우드 스토리지**는 이보다 더 크거나 여러 머신 간에 공유해야 하는 데이터에 사용됩니다.

```python
import os

local_path = os.path.expanduser("~/.cache/huggingface/datasets/")

# s3_path = "s3://my-bucket/datasets/"
# gcs_path = "gs://my-bucket/datasets/"
```

DVC는 S3 및 GCS와 직접 연동됩니다.

```bash
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

이 과정에서는 로컬 스토리지로 충분합니다. 클라우드 스토리지는 원격 GPU 인스턴스에서 미세 조정(fine-tuning)을 수행할 때 유용해집니다.

## 이 과정에서 사용되는 데이터셋

| Dataset | Lessons | Size | What It Teaches |
|---------|---------|------|----------------|
| IMDB | 토큰화(tokenization), 분류 | 84 MB | 텍스트 분류 기초 |
| WikiText | 언어 모델링(language modeling) | 181 MB | 다음 토큰 예측 |
| SQuAD | QA 시스템 | 35 MB | 질의응답, 스팬(span) |
| Common Crawl (일부) | 임베딩(embedding) | 가변적 | 대규모 텍스트 처리 |
| MNIST | 비전 기초 | 21 MB | 이미지 분류 기초 |
| COCO (일부) | 멀티모달(multimodal) | 가변적 | 이미지-텍스트 쌍 |

이 데이터셋들을 지금 모두 다운로드할 필요는 없습니다. 각 레슨에서 필요한 데이터셋을 안내합니다.

## 실습하기

유틸리티 스크립트를 실행하여 모든 것이 정상적으로 작동하는지 확인합니다.

```bash
python code/data_utils.py
```

이 스크립트는 작은 데이터셋을 다운로드하고, 변환하고, 분할한 뒤 요약 정보를 출력합니다.

## 결과물

이 레슨의 결과물은 다음과 같습니다.
- `code/data_utils.py` - 재사용 가능한 데이터 로드 및 캐싱 유틸리티
- `outputs/prompt-data-helper.md` - 작업에 적합한 데이터셋을 찾기 위한 프롬프트

## 연습 문제

1. `glue` 데이터셋을 `mrpc` 설정으로 로드하고 처음 5개 예제를 확인해 보세요.
2. `c4` 데이터셋을 스트리밍하고 10초 동안 몇 개의 예제를 처리할 수 있는지 세어 보세요.
3. 데이터셋을 Parquet 형식으로 변환하고 CSV와 파일 크기를 비교해 보세요.
4. 고정된 시드를 사용하여 70/15/15 비율의 train/val/test 분할을 생성하고 크기를 확인해 보세요.

### 힌트

**연습 문제 1.** `code/data_utils.py`의 `load_and_inspect("glue", "mrpc")`을 호출하거나 `load_dataset("glue", "mrpc")`를 직접 호출한 뒤 `ds["train"][:5]`을 출력합니다. 각 예제는 `sentence1`, `sentence2`, `label` 필드를 포함해야 합니다.

**연습 문제 2.** `stream_dataset`에서처럼 `streaming=True`로 데이터셋을 로드하고, `time.time()`가 시작 시간 더하기 10초를 초과하면 중단되는 루프에서 예제를 읽어 들입니다. 카운트를 출력합니다. 첫 번째 실행에는 연결 설정 시간이 포함되므로 두 번 실행해 보세요.

**연습 문제 3.** `convert_format`에서처럼 동일한 데이터셋을 `to_csv` 및 `to_parquet`로 저장한 다음, `ls -lh` 또는 `os.path.getsize`로 크기를 비교합니다. `load_from_parquet`로 Parquet 파일을 다시 로드하여 행 수가 변경되지 않았는지 확인합니다.

**연습 문제 4.** 고정된 `seed`를 사용하여 `train_test_split`를 두 번 사용합니다. 먼저 30%를 분할해 낸 다음, 해당 30%를 다시 반으로 분할합니다. 세 분할의 길이 합은 원래 길이와 같아야 하며, 동일한 시드로 다시 실행했을 때 각 분할의 첫 번째 행이 동일해야 합니다.
## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| 데이터셋 분할(Dataset split) | "학습 데이터" | ML 수명 주기의 서로 다른 단계에서 사용되는 이름이 지정된 하위 집합(train/val/test) |
| 스트리밍(Streaming) | "지연 로딩(lazy load)" | 전체 데이터셋을 다운로드하지 않고 원격 소스에서 행 단위로 데이터를 처리하는 방식 |
| Parquet | "압축된 CSV" | 분석 쿼리와 저장 효율성에 최적화된 열 지향(columnar) 파일 형식 |
| Arrow | "빠른 데이터프레임" | 제로 카피(zero-copy) 읽기를 위해 datasets 라이브러리 내부에서 사용하는 인메모리 열 지향 형식 |
| Git LFS | "대용량 파일용 Git" | 포인터만 버전 관리에 유지하면서 대용량 파일은 Git 저장소 외부에 저장하는 확장 도구 |
| DVC | "데이터용 Git" | 클라우드 스토리지와 연동되는 데이터셋 및 모델 전용 버전 관리 시스템 |
| 캐시(Cache) | "이미 다운로드됨" | 이전에 가져온 데이터의 로컬 복사본으로, 기본적으로 ~/.cache/huggingface/에 저장됨 |
