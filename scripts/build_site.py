#!/usr/bin/env python3
"""Assemble the MkDocs input tree (./docs) and mkdocs.yml from the translated
lessons in ./content/ko, falling back to the English source for lessons that are
not translated yet (clearly flagged), so the site is complete from day one.

    python scripts/build_site.py
    mkdocs build          # -> ./site
    mkdocs serve          # local preview
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import urllib.parse
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DOCS_DIR, ROOT, SOURCE_DIR, UPSTREAM_REPO, UPSTREAM_SITE, Lesson, iter_lessons,
    phase_title_en, phase_title_ko, split_frontmatter, title_of,
)

SITE_URL = "https://leondic1976.github.io/ai-engineering-from-scratch-kr/"
REPO_URL = "https://github.com/leondic1976/ai-engineering-from-scratch-kr"

_FIGURE_RE = re.compile(r"^[ \t]*```figure[ \t]*\n(.*?)\n[ \t]*```[ \t]*$", re.S | re.M)
_META_RE = re.compile(r"^\*\*(?:Type|유형):\*\*\s*(.+?)\s*$", re.M)
_LANG_RE = re.compile(r"^\*\*(?:Languages|사용 언어):\*\*\s*(.+?)\s*$", re.M)


_XREF_RE = re.compile(
    r"\]\((?:\.\./)+(?:([0-9]{2}-[a-z0-9-]+)/)?([0-9]{2}-[a-z0-9-]+)/docs/en\.md(#[^)]*)?\)"
)
_ASSET_RE = re.compile(r"\]\(\.\./assets/([^)\s]+)")


def rewrite_cross_links(md: str, lesson: Lesson) -> str:
    """Upstream repo-relative paths -> our docs layout.

    ../../../<phase>/<lesson>/docs/en.md  -> ../<phase>/<lesson>.md
    ../../<lesson>/docs/en.md             -> <lesson>.md            (same phase)
    ../assets/<file>                      -> <lesson>/assets/<file> (copied beside the page)
    """
    def xref(m: re.Match) -> str:
        phase, slug, frag = m.group(1), m.group(2), m.group(3) or ""
        return f"](../{phase}/{slug}.md{frag})" if phase else f"]({slug}.md{frag})"

    md = _XREF_RE.sub(xref, md)
    return _ASSET_RE.sub(lambda m: f"]({lesson.slug}/assets/{m.group(1)}", md)


def copy_assets(lesson: Lesson) -> None:
    src = lesson.src.parent.parent / "assets"
    if src.is_dir():
        dst = DOCS_DIR / "phases" / lesson.phase / lesson.slug / "assets"
        shutil.copytree(src, dst, dirs_exist_ok=True)


def replace_figures(md: str, lesson: Lesson) -> str:
    """Upstream renders ```figure blocks with site-side JS; link to the original instead."""
    def repl(m: re.Match) -> str:
        fid = m.group(1).strip()
        return (
            f'!!! info "그림 `{fid}`"\n'
            f"    이 그림은 원문 사이트에서 인터랙티브하게 렌더링됩니다. "
            f"[원문 레슨에서 보기]({lesson.upstream_page}){{ target=_blank }}\n"
        )
    return _FIGURE_RE.sub(repl, md)


def lesson_page(lesson: Lesson, body: str, meta: dict, translated: bool) -> str:
    title_en = meta.get("title_en") or title_of(lesson.src.read_text(encoding="utf-8"), lesson.slug)
    lines = body.split("\n")
    # insert a provenance line right after the H1
    for i, line in enumerate(lines):
        if line.startswith("# "):
            info = [
                "",
                f"<small>원문: <a href=\"{lesson.upstream_page}\" target=\"_blank\">{title_en}</a>"
                f" · <a href=\"{lesson.upstream_dir}\" target=\"_blank\">코드 · 퀴즈 (GitHub)</a>"
                + (f" · 번역: {meta.get('model', '?')} ({meta.get('translated_at', '')})" if translated else "")
                + "</small>",
            ]
            if not translated:
                info += [
                    "",
                    '!!! warning "아직 번역되지 않은 레슨입니다"',
                    "    아래는 영어 원문입니다. 번역 파이프라인이 다음 실행에서 이 레슨을 처리합니다.",
                ]
            lines[i + 1:i + 1] = info
            break
    body = "\n".join(lines)
    body = rewrite_cross_links(body, lesson)
    body = replace_figures(body, lesson)
    footer = (
        "\n\n---\n\n"
        f"<small>이 문서는 <a href=\"{UPSTREAM_REPO}\" target=\"_blank\">AI Engineering from Scratch</a>"
        f" (MIT License)의 한국어 기계 번역본입니다. 오역을 발견하면 "
        f"<a href=\"{REPO_URL}/issues\" target=\"_blank\">이슈</a>로 알려 주세요.</small>\n"
    )
    return body + footer


def _rmtree_retry(path: Path, attempts: int = 5) -> None:
    """rmtree that survives Windows' transient 'access denied' from indexers/AV."""
    import os
    import stat
    import time

    def _onexc(func, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            raise exc

    for i in range(attempts):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onexc=_onexc)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.5 * (i + 1))


def rel_link(lesson: Lesson) -> str:
    return f"phases/{lesson.phase}/{lesson.slug}.md"


def main() -> None:
    lessons = list(iter_lessons())
    _rmtree_retry(DOCS_DIR)
    (DOCS_DIR / "phases").mkdir(parents=True)

    phases: "OrderedDict[str, list[dict]]" = OrderedDict()
    n_translated = 0
    for lesson in lessons:
        src_md = lesson.src.read_text(encoding="utf-8")
        translated = lesson.out.is_file()
        if translated:
            meta, body = split_frontmatter(lesson.out.read_text(encoding="utf-8"))
            n_translated += 1
        else:
            meta, body = {}, src_md
        title_en = title_of(src_md, lesson.slug)
        title = title_of(body, title_en)
        page = lesson_page(lesson, body, meta, translated)
        dst = DOCS_DIR / rel_link(lesson)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(page, encoding="utf-8", newline="\n")
        copy_assets(lesson)

        m_type = _META_RE.search(body)
        m_lang = _LANG_RE.search(body)
        phases.setdefault(lesson.phase, []).append(dict(
            lesson=lesson, title=title, title_en=title_en, translated=translated,
            type=m_type.group(1) if m_type else "—", lang=m_lang.group(1) if m_lang else "—",
        ))

    total = len(lessons)
    pct = (100 * n_translated // total) if total else 0

    # phase index pages
    for phase, rows in phases.items():
        ko, en = phase_title_ko(phase), phase_title_en(phase)
        num = phase.split("-")[0]
        out = [f"# Phase {num}. {ko}", "", f"<small>{en}</small>", "",
               "| # | 레슨 | 유형 | 언어 | 번역 |", "|---|---|---|---|---|"]
        for r in rows:
            L = r["lesson"]
            label = r["title"] if r["translated"] else f"{r['title_en']}"
            mark = "✅" if r["translated"] else "⏳"
            out.append(f"| {L.slug.split('-')[0]} | [{label}]({L.slug}.md) | {r['type']} | {r['lang']} | {mark} |")
        (DOCS_DIR / "phases" / phase / "index.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")

    # home / catalog
    home = [
        "# AI Engineering from Scratch — 한국어",
        "",
        f"[AI Engineering from Scratch]({UPSTREAM_SITE})의 전체 커리큘럼({total}개 레슨, 20개 Phase)을 "
        "무료 LLM으로 한국어 번역해 제공합니다. 코드, 수식, 링크는 원문 그대로 유지하고 설명문만 번역합니다.",
        "",
        f"**번역 진행률: {n_translated} / {total} ({pct}%)**",
        "",
        f'<progress value="{n_translated}" max="{total}" style="width:100%"></progress>',
        "",
        "## 커리큘럼",
        "",
        "| Phase | 주제 | 레슨 수 | 번역됨 |",
        "|---|---|---|---|",
    ]
    for phase, rows in phases.items():
        num = phase.split("-")[0]
        t = sum(1 for r in rows if r["translated"])
        home.append(f"| {num} | [{phase_title_ko(phase)}](phases/{phase}/index.md) | {len(rows)} | {t} |")
    home += [
        "",
        "## 전체 카탈로그",
        "",
        '<input type="search" id="catalog-filter" placeholder="레슨 검색 (한국어/영어)…" '
        'style="width:100%;padding:.6em;margin-bottom:1em;font-size:1em" oninput="filterCatalog(this.value)">',
        "",
        "| Phase | 레슨 | 유형 | 언어 | 번역 |",
        "|---|---|---|---|---|",
    ]
    for phase, rows in phases.items():
        num = phase.split("-")[0]
        for r in rows:
            L = r["lesson"]
            label = r["title"] if r["translated"] else r["title_en"]
            sub = f" <small>({r['title_en']})</small>" if r["translated"] and r["title"] != r["title_en"] else ""
            mark = "✅" if r["translated"] else "⏳"
            home.append(f"| {num} | [{label}]({rel_link(L)}){sub} | {r['type']} | {r['lang']} | {mark} |")
    home += [
        "",
        "<script>",
        "function filterCatalog(q){q=q.toLowerCase();const t=document.getElementById('catalog-filter')"
        ".nextElementSibling;if(!t)return;for(const tr of t.querySelectorAll('tbody tr'))"
        "tr.style.display=tr.textContent.toLowerCase().includes(q)?'':'none';}",
        "</script>",
        "",
        "## 라이선스",
        "",
        f"원문 콘텐츠와 이 번역본은 모두 [MIT License]({UPSTREAM_REPO}/blob/main/LICENSE)를 따릅니다. "
        "원저작자: [Rohit Ghumare 외 기여자]("
        f"{UPSTREAM_REPO}/graphs/contributors).",
    ]
    (DOCS_DIR / "index.md").write_text("\n".join(home) + "\n", encoding="utf-8", newline="\n")

    # mathjax bootstrap
    (DOCS_DIR / "javascripts").mkdir()
    (DOCS_DIR / "javascripts" / "mathjax.js").write_text(
        "window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],"
        "['\\\\[','\\\\]']],processEscapes:true,processEnvironments:true},"
        "options:{ignoreHtmlClass:'.*|',processHtmlClass:'arithmatex'}};\n"
        "document$.subscribe(()=>{MathJax.startup.output.clearCache();MathJax.typesetClear();"
        "MathJax.texReset();MathJax.typesetPromise()})\n",
        encoding="utf-8",
    )
    lic = SOURCE_DIR / "LICENSE"
    if lic.is_file():
        shutil.copy(lic, DOCS_DIR / "LICENSE-upstream.txt")

    # mkdocs.yml (nav generated; JSON strings are valid YAML scalars)
    nav = ['  - "홈 · 카탈로그": index.md']
    for phase, rows in phases.items():
        num = phase.split("-")[0]
        nav.append(f'  - {json.dumps(f"{num}. {phase_title_ko(phase)}", ensure_ascii=False)}:')
        nav.append(f'      - {json.dumps("개요", ensure_ascii=False)}: phases/{phase}/index.md')
        for r in rows:
            L = r["lesson"]
            label = r["title"] if r["translated"] else r["title_en"]
            nav.append(f'      - {json.dumps(label, ensure_ascii=False)}: {rel_link(L)}')
    cfg = MKDOCS_TEMPLATE.replace("__NAV__", "\n".join(nav))
    (ROOT / "mkdocs.yml").write_text(cfg, encoding="utf-8", newline="\n")
    print(f"docs ready: {total} lessons ({n_translated} translated, {pct}%) -> {DOCS_DIR}")


MKDOCS_TEMPLATE = f"""# GENERATED by scripts/build_site.py — edit the template there, not this file.
site_name: AI Engineering from Scratch (한국어)
site_description: AI Engineering from Scratch 커리큘럼의 한국어 번역
site_url: {SITE_URL}
repo_url: {REPO_URL}
repo_name: ai-engineering-from-scratch-kr
edit_uri: ""
docs_dir: docs
site_dir: site
copyright: 원문 © AI Engineering from Scratch contributors (MIT) · 한국어 번역 MIT

theme:
  name: material
  language: ko
  features:
    - navigation.sections
    - navigation.indexes
    - navigation.prune
    - navigation.top
    - navigation.footer
    - search.suggest
    - search.highlight
    - content.code.copy
    - toc.follow
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/weather-night
        name: 다크 모드
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/weather-sunny
        name: 라이트 모드

plugins:
  - search:
      lang:
        - ko
        - en

markdown_extensions:
  - admonition
  - attr_list
  - md_in_html
  - tables
  - footnotes
  - toc:
      permalink: true
  - pymdownx.details
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.tasklist:
      custom_checkbox: true
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format

extra_javascript:
  - javascripts/mathjax.js
  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js

nav:
__NAV__
"""


if __name__ == "__main__":
    main()
