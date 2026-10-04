# AI Engineering from Scratch — 한국어 번역

[aiengineeringfromscratch.com](https://aiengineeringfromscratch.com/catalog.html)의 전체 커리큘럼
(20개 Phase, 523개 레슨)을 **무료 LLM API**로 한국어 번역해 **GitHub Pages**로 호스팅하는 파이프라인입니다.

- 원본: [rohitg00/ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (MIT)
- 사이트: https://leondic1976.github.io/ai-engineering-from-scratch-kr/
- 번역 대상: 각 레슨의 `docs/en.md` (설명문만 번역, 코드·수식·링크·이미지는 원문 그대로 보존)

## 동작 방식

```
scripts/fetch_source.py   원본 레포를 sparse clone  →  source/phases/**/docs/en.md   (gitignore)
scripts/translate.py      무료 LLM으로 번역         →  content/ko/phases/<phase>/<lesson>.md  (커밋)
scripts/build_site.py     MkDocs 입력 생성          →  docs/ + mkdocs.yml               (gitignore)
mkdocs build              정적 사이트 생성           →  site/                            (Pages 배포)
```

- 코드 블록, 인라인 코드, 수식, URL, 이미지, HTML은 `{{P12}}` 자리표시자로 치환한 뒤 번역하고 원래대로 복원합니다.
  자리표시자가 하나라도 사라지면 재시도하고, 끝내 실패하면 해당 레슨은 건너뛰어 다음 실행에서 다시 시도합니다.
- 레슨별 해시 캐시(`content/ko/.translate-cache.json`)로 **중단 후 재개**가 가능하고, 원문이 바뀐 레슨만 다시 번역합니다.
- 무료 티어의 **일일 한도**에 걸리면 정상 종료(exit 0)하고, 다음 스케줄 실행에서 이어서 진행합니다.
- 아직 번역되지 않은 레슨은 사이트에 영어 원문으로 표시되므로 사이트는 첫날부터 완전한 카탈로그를 제공합니다.

## 무료 모델 선택

| provider | 환경 변수 | 기본 모델 | 메모 |
|---|---|---|---|
| `gemini` (기본) | `GEMINI_API_KEY` | `gemini-2.5-flash` | [AI Studio](https://aistudio.google.com/apikey)에서 무료 키 발급. 한국어 품질 좋음. 일일 한도 때문에 전체 번역에 며칠 소요 (`gemini-2.5-flash-lite`는 한도가 더 큼) |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | 매우 빠름, 일일 토큰 한도 작음 |
| `openrouter` | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct:free` | `:free` 모델만 무료, 목록이 자주 바뀜 → `--model`로 지정 |
| `cerebras` | `CEREBRAS_API_KEY` | `llama-3.3-70b` | 빠름 |
| `ollama` | (없음) | `gemma3:12b` | 로컬 실행, 한도 없음. `qwen3:8b`, `gemma3:27b` 등 추천 |
| `custom` | `LLM_API_KEY`, `LLM_BASE_URL` | `TRANSLATE_MODEL` | LM Studio, vLLM 등 OpenAI 호환 서버 |
| `echo` | (없음) | — | 번역 없이 파이프라인 점검용 |

모델은 `--model ...` 또는 환경 변수 `TRANSLATE_MODEL`로 바꿀 수 있습니다.

## 로컬 실행

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...            # Windows PowerShell: $env:GEMINI_API_KEY="..."

python scripts/pipeline.py                        # 가져오기 → 번역(한도까지) → 사이트 빌드
python scripts/pipeline.py --provider ollama      # 로컬 Ollama로 무제한 번역
python scripts/translate.py --limit 5             # 5개만 번역해 보기
python scripts/translate.py --phase 10-llms-from-scratch
python scripts/translate.py --only phases/00-setup-and-tooling/01-dev-environment --force
python scripts/translate.py --dry-run             # 남은 레슨 목록만 출력
python -m mkdocs serve                            # http://127.0.0.1:8000 미리보기
```

## GitHub에서 자동 번역 + 호스팅

1. **Secrets** (Settings → Secrets and variables → Actions): `GEMINI_API_KEY` 추가
   (다른 provider를 쓰려면 해당 키를 추가하고 Variables에 `TRANSLATE_PROVIDER`, `TRANSLATE_MODEL` 설정)
2. **Pages** (Settings → Pages): Source를 **GitHub Actions**로 설정
   ```bash
   gh api -X POST repos/leondic1976/ai-engineering-from-scratch-kr/pages -f build_type=workflow
   ```
3. Actions 탭에서 **Translate lessons (ko)** 워크플로를 `Run workflow`로 한 번 실행하거나, 6시간마다 도는 스케줄을 기다립니다.
   번역이 커밋될 때마다 **Deploy site** 워크플로가 사이트를 다시 배포합니다.

| 워크플로 | 트리거 | 하는 일 |
|---|---|---|
| `translate.yml` | 6시간마다 / 수동 | 원본 가져오기 → 번역(최대 320분 또는 일일 한도) → `content/` 커밋 → 배포 호출 |
| `deploy.yml` | `main` push / 수동 / 호출 | `build_site.py` + `mkdocs build` → GitHub Pages |

## 디렉터리

```
content/ko/phases/<phase>/<lesson>.md   번역 결과 (frontmatter: 영어 제목, 원문 해시, 모델, 날짜)
content/ko/.translate-cache.json        레슨별 원문 해시 캐시
scripts/common.py                       경로·레슨 탐색·Phase 한국어 이름
scripts/fetch_source.py                 원본 sparse clone / 갱신
scripts/translate.py                    번역기 (보호 토큰, 청크 분할, 레이트리밋, 재개)
scripts/build_site.py                   MkDocs 문서 트리 + mkdocs.yml 생성
scripts/pipeline.py                     위 단계를 한 번에 실행
.github/workflows/                      translate.yml, deploy.yml
```

## 한계와 메모

- 원본 사이트의 인터랙티브 그림(```` ```figure ````)은 사이트 전용 JS로 그려지므로 원문 링크로 대체합니다.
- 퀴즈(`quiz.json`)와 코드 파일은 번역하지 않고 원본 GitHub 디렉터리로 링크합니다.
- 기계 번역이므로 오역이 있을 수 있습니다. 수정은 `content/ko/...md`를 직접 고치고, 원문이 바뀌어도 덮어쓰이지 않게 하려면
  캐시의 `sha`를 유지하면 됩니다(원문 변경 시에만 재번역).

## 라이선스

원본 콘텐츠는 Rohit Ghumare 및 기여자들의 저작물로 MIT License를 따르며, 이 저장소의 번역본과 스크립트 역시 MIT License로 배포합니다.
자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
