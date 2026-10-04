#!/usr/bin/env python3
"""Translate the English lessons in ./source into Korean using a free LLM API.

Only prose reaches the model. Fenced code, inline code, math, image refs, link
targets, raw HTML and bare URLs are swapped for placeholders ({{P12}}) before the
request and restored byte-for-byte afterwards. A response that loses or invents a
placeholder is retried, and if it still fails the lesson is left untranslated so a
later run retries it instead of freezing a broken translation.

Runs are hash-cached per lesson and resumable: kill it any time, run it again,
and it continues where it stopped. Free-tier daily quotas are detected and the
run ends cleanly (exit 0) so a scheduled job can simply pick up tomorrow.

Providers (all free tiers, all speak the OpenAI chat-completions dialect):

    gemini      GEMINI_API_KEY      https://aistudio.google.com/apikey   (default)
    groq        GROQ_API_KEY        https://console.groq.com/keys
    openrouter  OPENROUTER_API_KEY  https://openrouter.ai/keys  (use a ":free" model)
    cerebras    CEREBRAS_API_KEY    https://cloud.cerebras.ai
    ollama      (no key)            local, http://localhost:11434
    custom      LLM_API_KEY + LLM_BASE_URL   any OpenAI-compatible server (LM Studio, vLLM)
    echo        (no key)            copies English through; for testing the pipeline

Usage:
    python scripts/translate.py                                 # provider=gemini
    python scripts/translate.py --provider groq --limit 20
    python scripts/translate.py --provider ollama --model qwen3:8b
    python scripts/translate.py --phase 10-llms-from-scratch
    python scripts/translate.py --only phases/00-setup-and-tooling/01-dev-environment
    python scripts/translate.py --dry-run
    TRANSLATE_MODEL=<model-id> python scripts/translate.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CONTENT_DIR, Lesson, iter_lessons, load_cache, save_cache, source_hash,
    title_of, with_frontmatter,
)

# ── providers ────────────────────────────────────────────────────────────────

PROVIDERS = {
    "gemini": dict(
        base="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env="GEMINI_API_KEY", model="gemini-3.8-flash", rpm=8,
    ),
    "groq": dict(
        base="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY", model="llama-3.3-70b-versatile", rpm=25,
    ),
    "openrouter": dict(
        base="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY", model="meta-llama/llama-3.3-70b-instruct:free", rpm=15,
    ),
    "cerebras": dict(
        base="https://api.cerebras.ai/v1",
        key_env="CEREBRAS_API_KEY", model="llama-3.3-70b", rpm=25,
    ),
    "ollama": dict(
        base="http://localhost:11434/v1",
        key_env=None, model="gemma3:12b", rpm=0,
    ),
    "custom": dict(
        base=os.environ.get("LLM_BASE_URL", ""),
        key_env="LLM_API_KEY", model=os.environ.get("TRANSLATE_MODEL", ""), rpm=0,
    ),
    "echo": dict(base=None, key_env=None, model="echo", rpm=0),
}

# ── protection of non-prose spans ────────────────────────────────────────────

SENT = "{{P%d}}"
SENT_RE = re.compile(r"\{\{P(\d+)\}\}")

PROTECT = [
    # fenced blocks (code, mermaid, figure, ...) incl. indented fences inside lists
    re.compile(r"^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$", re.S | re.M),
    re.compile(r"<!--.*?-->", re.S),                               # html comments
    re.compile(r"\$\$.*?\$\$", re.S),                              # display math
    re.compile(r"`[^`\n]+`"),                                      # inline code
    re.compile(r"(?<![\\$\w])\$(?!\s)[^$\n]{1,200}?(?<!\s)\$(?!\w)"),  # inline math
    re.compile(r"!\[[^\]]*\]\([^)]+\)"),                           # images (whole)
    re.compile(r"\]\([^)\s]+(?:\s+\"[^\"]*\")?\)"),                # link target: ](url)
    re.compile(r"<https?://[^>]+>"),                               # autolinks
    re.compile(r"https?://[^\s)>\]]+"),                            # bare urls
    re.compile(r"</?[a-zA-Z][^>\n]*>"),                            # inline html tags
]


def protect(text: str) -> tuple[str, list[str]]:
    store: list[str] = []

    def stash(m: re.Match) -> str:
        store.append(m.group(0))
        return SENT % (len(store) - 1)

    for pat in PROTECT:
        text = pat.sub(stash, text)
    return text, store


def restore(text: str, store: list[str]) -> str:
    # Reverse order: a span may itself contain a lower-indexed placeholder.
    for i in range(len(store) - 1, -1, -1):
        text = text.replace(SENT % i, store[i])
    return text


def placeholders_ok(src: str, out: str) -> bool:
    return sorted(SENT_RE.findall(src)) == sorted(SENT_RE.findall(out))


# ── chunking (protected text only, so fences are already opaque) ─────────────

def _split_big(block: str, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    out, cur = [], ""
    for para in re.split(r"(?<=\n\n)", block):
        if cur and len(cur) + len(para) > max_chars:
            out.append(cur)
            cur = ""
        cur += para
    if cur:
        out.append(cur)
    return out


def chunk(text: str, max_chars: int) -> list[str]:
    atoms: list[str] = []
    for section in re.split(r"(?m)^(?=#{1,6} )", text):
        if section:
            atoms.extend(_split_big(section, max_chars))
    chunks, cur = [], ""
    for a in atoms:
        if cur and len(cur) + len(a) > max_chars:
            chunks.append(cur)
            cur = ""
        cur += a
    if cur:
        chunks.append(cur)
    return chunks


# ── prompt ───────────────────────────────────────────────────────────────────

SYSTEM = """You are a professional technical translator localizing an AI/ML engineering curriculum from English into Korean.

Rules:
1. Placeholders of the form {{P12}} stand for code, math, URLs and HTML. Keep every placeholder EXACTLY as written, in the same relative position. Never translate, drop, duplicate, merge or reorder them.
2. Preserve the Markdown structure exactly: heading levels (#), list markers and numbering, tables (same number of rows and columns), blockquotes (>), bold/italic markers, and blank lines between paragraphs. Do not add or remove headings.
3. Style: natural, concise Korean as used in professional technical books. Use the polite declarative register (~합니다 / ~입니다). Keep sentences faithful; do not summarize, expand, add notes, or add marketing language.
4. Terminology: keep proper nouns, product/library/framework names, model names, acronyms, file names, CLI flags and identifiers in English (PyTorch, Transformer, softmax, ReLU, Adam, GPT, BERT, CUDA, uv). For common ML concepts use the standard Korean term and add the English in parentheses on first use when it helps, e.g. 역전파(backpropagation), 임베딩(embedding), 어텐션(attention), 미세 조정(fine-tuning), 추론(inference).
5. Leave the metadata lines that start with **Type:**, **Languages:**, **Prerequisites:**, **Time:** unchanged (label and value); they are post-processed.
6. Output ONLY the translated Markdown. No preamble, no closing remarks, no code fence around the whole response."""


# ── HTTP ─────────────────────────────────────────────────────────────────────

class QuotaExhausted(Exception):
    """Daily / hard quota hit: stop the run cleanly, resume next time."""


class FatalAPIError(Exception):
    pass


_DAILY_RE = re.compile(r"per[ _-]?day|daily|PerDay|free-models-per-day", re.I)
_RETRY_DELAY_RE = re.compile(r'retryDelay"?\s*:\s*"?(\d+(?:\.\d+)?)s')


class Client:
    def __init__(self, provider: str, model: str | None, rpm: int | None, max_tokens: int, temperature: float):
        cfg = PROVIDERS[provider]
        self.provider = provider
        self.base = (cfg["base"] or "").rstrip("/")
        self.model = model or os.environ.get("TRANSLATE_MODEL") or cfg["model"]
        self.rpm = cfg["rpm"] if rpm is None else rpm
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.key = os.environ.get(cfg["key_env"]) if cfg["key_env"] else None
        if provider == "echo":
            return
        if not self.base:
            raise SystemExit("custom provider needs LLM_BASE_URL (e.g. http://localhost:1234/v1)")
        if not self.model:
            raise SystemExit("no model: pass --model or set TRANSLATE_MODEL")
        if cfg["key_env"] and not self.key and provider != "custom":
            raise SystemExit(
                f"missing API key: set {cfg['key_env']} (see docstring for where to get a free one)"
            )
        self._stamps: list[float] = []
        self.requests = 0

    def _throttle(self) -> None:
        if not self.rpm:
            return
        now = time.time()
        self._stamps = [t for t in self._stamps if now - t < 60]
        if len(self._stamps) >= self.rpm:
            wait = 60 - (now - self._stamps[0]) + 0.5
            time.sleep(max(wait, 0))
        self._stamps.append(time.time())

    def chat(self, user: str) -> str:
        if self.provider == "echo":
            return user
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.provider == "ollama":
            body["options"] = {"num_ctx": 16384}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/leondic1976/ai-engineering-from-scratch-kr"
            headers["X-Title"] = "ai-engineering-from-scratch-kr"
        data = json.dumps(body).encode("utf-8")

        delay = 5.0
        for attempt in range(1, 7):
            self._throttle()
            req = urllib.request.Request(f"{self.base}/chat/completions", data=data, headers=headers, method="POST")
            try:
                self.requests += 1
                with urllib.request.urlopen(req, timeout=300) as resp:
                    payload = json.load(resp)
                return payload["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")
                if e.code == 429:
                    if _DAILY_RE.search(text):
                        raise QuotaExhausted(text[:300])
                    retry_after = e.headers.get("Retry-After")
                    m = _RETRY_DELAY_RE.search(text)
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else (
                        float(m.group(1)) if m else delay
                    )
                    wait = min(max(wait, 2.0), 180.0)
                    print(f"    429 rate limited, waiting {wait:.0f}s (attempt {attempt})", file=sys.stderr)
                    time.sleep(wait)
                elif e.code >= 500:
                    print(f"    HTTP {e.code}, retrying in {delay:.0f}s", file=sys.stderr)
                    time.sleep(delay)
                else:
                    raise FatalAPIError(f"HTTP {e.code}: {text[:400]}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                print(f"    network error {e!r}, retrying in {delay:.0f}s", file=sys.stderr)
                time.sleep(delay)
            delay = min(delay * 2, 120.0)
        raise FatalAPIError("gave up after 6 attempts")


# ── output clean-up ──────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
_WRAP_RE = re.compile(r"\A\s*```(?:markdown|md)?\s*\n(.*)\n```\s*\Z", re.S)


def clean_response(text: str) -> str:
    text = _THINK_RE.sub("", text)
    m = _WRAP_RE.match(text)
    if m:
        text = m.group(1)
    return text.strip("\n") + "\n"


META_LABELS = {
    "Type": "유형",
    "Languages": "사용 언어",
    "Prerequisites": "선수 레슨",
    "Time": "소요 시간",
}
_META_RE = re.compile(r"^(\s*)\*\*(Type|Languages|Prerequisites|Time):\*\*(.*)$", re.M)
_TYPE_VALUES = {"Build": "Build (실습)", "Learn": "Learn (이론)"}


def localize_metadata(md: str) -> str:
    def repl(m: re.Match) -> str:
        label, value = m.group(2), m.group(3)
        v = value.strip()
        if label == "Type":
            v = _TYPE_VALUES.get(v, v)
        elif label == "Time":
            v = re.sub(r"(\d+)\s*(?:minutes?|mins?)\b", r"\1분", v)
            v = re.sub(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b", r"\1시간", v)
        elif label == "Prerequisites" and v.lower() == "none":
            v = "없음"
        return f"{m.group(1)}**{META_LABELS[label]}:** {v}"

    return _META_RE.sub(repl, md)


# ── per-lesson translation ───────────────────────────────────────────────────

def translate_markdown(src: str, client: Client, max_chars: int, retries: int) -> str | None:
    protected, store = protect(src)
    pieces = chunk(protected, max_chars)
    out_parts: list[str] = []
    for i, piece in enumerate(pieces, 1):
        if not piece.strip():
            out_parts.append(piece)
            continue
        ok = False
        for attempt in range(1, retries + 1):
            raw = clean_response(client.chat(piece))
            if placeholders_ok(piece, raw):
                out_parts.append(raw if piece.endswith("\n") else raw.rstrip("\n"))
                ok = True
                break
            print(f"    chunk {i}/{len(pieces)}: placeholder mismatch (attempt {attempt})", file=sys.stderr)
        if not ok:
            return None
    joined = "".join(p if p.endswith("\n") else p + "\n" for p in out_parts)
    return localize_metadata(restore(joined, store))


def write_translation(lesson: Lesson, src: str, body: str, model: str) -> None:
    meta = {
        "title_en": title_of(src, lesson.slug),
        "source": lesson.rel,
        "source_sha": source_hash(src)[:16],
        "model": model,
        "translated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    lesson.out.parent.mkdir(parents=True, exist_ok=True)
    lesson.out.write_text(with_frontmatter(meta, body), encoding="utf-8", newline="\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default=os.environ.get("TRANSLATE_PROVIDER", "gemini"), choices=sorted(PROVIDERS))
    ap.add_argument("--model", help="override the provider's default model (or TRANSLATE_MODEL)")
    ap.add_argument("--rpm", type=int, help="requests per minute cap (default per provider)")
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("TRANSLATE_MAX_TOKENS", 8192)))
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-chars", type=int, default=7000, help="max English chars per request")
    ap.add_argument("--retries", type=int, default=3, help="retries per chunk on placeholder mismatch")
    ap.add_argument("--limit", type=int, default=0, help="stop after N lessons translated this run")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop cleanly after this many minutes")
    ap.add_argument("--phase", help="only this phase dir, e.g. 10-llms-from-scratch")
    ap.add_argument("--only", help="only this lesson path, e.g. phases/00-setup-and-tooling/01-dev-environment")
    ap.add_argument("--force", action="store_true", help="ignore the cache and retranslate")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = Client(args.provider, args.model, args.rpm, args.max_tokens, args.temperature)
    cache = load_cache()
    started = time.time()
    done = skipped = failed = 0
    pending = 0
    stop_reason = None

    lessons = list(iter_lessons())
    for lesson in lessons:
        if args.phase and lesson.phase != args.phase:
            continue
        if args.only and lesson.rel != args.only.strip("/"):
            continue

        src = lesson.src.read_text(encoding="utf-8")
        h = source_hash(src)
        entry = cache.get(lesson.rel)
        if not args.force and entry and entry.get("sha") == h and lesson.out.is_file():
            skipped += 1
            continue

        pending += 1
        if args.dry_run:
            print(f"would translate {lesson.rel}")
            continue
        if args.limit and done >= args.limit:
            stop_reason = f"--limit {args.limit} reached"
            break
        if args.max_minutes and (time.time() - started) / 60 >= args.max_minutes:
            stop_reason = f"--max-minutes {args.max_minutes} reached"
            break

        print(f"[{done + 1}] {lesson.rel} ({len(src)} chars)")
        try:
            body = translate_markdown(src, client, args.max_chars, args.retries)
        except QuotaExhausted as e:
            stop_reason = f"daily quota exhausted for {args.provider}/{client.model}: {e}"
            break
        except FatalAPIError as e:
            print(f"FATAL: {e}", file=sys.stderr)
            save_cache(cache)
            return 1

        if body is None:
            failed += 1
            print(f"    FAILED (placeholders); left untranslated, will retry next run", file=sys.stderr)
            continue
        write_translation(lesson, src, body, client.model)
        cache[lesson.rel] = {
            "sha": h, "model": client.model,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        save_cache(cache)  # persist after every lesson: a killed run resumes here
        done += 1

    if not args.dry_run:
        save_cache(cache)
    total = len(lessons)
    translated_total = sum(1 for l in lessons if l.out.is_file())
    print(
        f"\n{args.provider}/{client.model}: {done} translated, {failed} failed, {skipped} cached"
        f" | pending before run: {pending} | site total: {translated_total}/{total}"
    )
    if stop_reason:
        print(f"stopped: {stop_reason} (run again to continue)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
