"""Shared paths, lesson discovery and small helpers for the KR translation pipeline.

Layout (relative to the repo root):

    source/            upstream shallow clone (gitignored; refreshed by fetch_source.py)
    content/ko/        translated lesson markdown (committed; written by translate.py)
    docs/              generated MkDocs input (gitignored; written by build_site.py)
    site/              MkDocs output (gitignored)
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "source"
PHASES_DIR = SOURCE_DIR / "phases"
CONTENT_DIR = ROOT / "content" / "ko"
CACHE_PATH = CONTENT_DIR / ".translate-cache.json"
DOCS_DIR = ROOT / "docs"

UPSTREAM_REPO = "https://github.com/rohitg00/ai-engineering-from-scratch"
UPSTREAM_BRANCH = "main"
UPSTREAM_SITE = "https://aiengineeringfromscratch.com"

# Same definition of "a lesson" as the upstream catalog builder, so the
# translated tree mirrors catalog.html one-to-one.
PHASE_DIR_RE = re.compile(r"^([0-9]{2})-([a-z0-9][a-z0-9-]*)$")
LESSON_DIR_RE = re.compile(r"^([0-9]{2})-([a-z0-9][a-z0-9-]*)$")

PHASE_NAMES_KO = {
    "00-setup-and-tooling": "환경 설정과 도구",
    "01-math-foundations": "수학 기초",
    "02-ml-fundamentals": "머신러닝 기초",
    "03-deep-learning-core": "딥러닝 핵심",
    "04-computer-vision": "컴퓨터 비전",
    "05-nlp-foundations-to-advanced": "NLP: 기초부터 심화까지",
    "06-speech-and-audio": "음성과 오디오",
    "07-transformers-deep-dive": "트랜스포머 심층 탐구",
    "08-generative-ai": "생성형 AI",
    "09-reinforcement-learning": "강화학습",
    "10-llms-from-scratch": "밑바닥부터 만드는 LLM",
    "11-llm-engineering": "LLM 엔지니어링",
    "12-multimodal-ai": "멀티모달 AI",
    "13-tools-and-protocols": "도구와 프로토콜",
    "14-agent-engineering": "에이전트 엔지니어링",
    "15-autonomous-systems": "자율 시스템",
    "16-multi-agent-and-swarms": "멀티 에이전트와 스웜",
    "17-infrastructure-and-production": "인프라와 프로덕션",
    "18-ethics-safety-alignment": "윤리·안전·정렬",
    "19-capstone-projects": "캡스톤 프로젝트",
}


@dataclass(frozen=True)
class Lesson:
    phase: str          # e.g. "00-setup-and-tooling"
    slug: str           # e.g. "01-dev-environment"
    src: Path           # source/phases/<phase>/<slug>/docs/en.md

    @property
    def rel(self) -> str:
        """Stable key, identical to the upstream lesson path: phases/<phase>/<slug>."""
        return f"phases/{self.phase}/{self.slug}"

    @property
    def out(self) -> Path:
        return CONTENT_DIR / "phases" / self.phase / f"{self.slug}.md"

    @property
    def upstream_page(self) -> str:
        return f"{UPSTREAM_SITE}/lesson?path=phases%2F{self.phase}%2F{self.slug}"

    @property
    def upstream_dir(self) -> str:
        return f"{UPSTREAM_REPO}/tree/{UPSTREAM_BRANCH}/{self.rel}"


def iter_lessons() -> Iterator[Lesson]:
    if not PHASES_DIR.is_dir():
        raise SystemExit(
            f"source tree not found at {PHASES_DIR}. Run: python scripts/fetch_source.py"
        )
    for phase in sorted(PHASES_DIR.iterdir()):
        if not (phase.is_dir() and PHASE_DIR_RE.match(phase.name)):
            continue
        for lesson in sorted(phase.iterdir()):
            doc = lesson / "docs" / "en.md"
            if lesson.is_dir() and LESSON_DIR_RE.match(lesson.name) and doc.is_file():
                yield Lesson(phase.name, lesson.name, doc)


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache() -> dict:
    if CACHE_PATH.is_file():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


_FM_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.S)


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a leading `--- key: value ---` block (flat, string values only)."""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"')
    return meta, text[m.end():]


def with_frontmatter(meta: dict, body: str) -> str:
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f'{k}: "{str(v).replace(chr(34), chr(39))}"')
    lines.append("---")
    return "\n".join(lines) + "\n" + body


def title_of(markdown: str, fallback: str = "") -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def phase_title_en(dirname: str) -> str:
    m = PHASE_DIR_RE.match(dirname)
    words = (m.group(2) if m else dirname).split("-")
    return " ".join(w.upper() if w in {"nlp", "ai", "ml", "llm", "llms"} else w.capitalize() for w in words)


def phase_title_ko(dirname: str) -> str:
    return PHASE_NAMES_KO.get(dirname, phase_title_en(dirname))
