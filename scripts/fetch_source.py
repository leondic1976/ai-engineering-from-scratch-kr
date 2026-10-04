#!/usr/bin/env python3
"""Fetch (or refresh) the upstream English lessons into ./source using a sparse,
blob-filtered shallow clone, so only the lesson markdown is downloaded.

    python scripts/fetch_source.py            # clone or fast-forward
    python scripts/fetch_source.py --force    # wipe and re-clone
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SOURCE_DIR, UPSTREAM_BRANCH, UPSTREAM_REPO, iter_lessons  # noqa: E402

# gitignore-style patterns (non-cone sparse checkout): lesson prose + licence only.
SPARSE_PATTERNS = [
    "/phases/*/*/docs/en.md",
    "/phases/*/*/assets/*",
    "/LICENSE",
    "/languages.json",
]


def run(*cmd: str, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def clone() -> None:
    SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
    run(
        "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
        "--branch", UPSTREAM_BRANCH, UPSTREAM_REPO, str(SOURCE_DIR),
    )
    run("git", "sparse-checkout", "set", "--no-cone", *SPARSE_PATTERNS, cwd=SOURCE_DIR)


def update() -> None:
    run("git", "sparse-checkout", "set", "--no-cone", *SPARSE_PATTERNS, cwd=SOURCE_DIR)
    run("git", "fetch", "--depth", "1", "origin", UPSTREAM_BRANCH, cwd=SOURCE_DIR)
    run("git", "reset", "--hard", f"origin/{UPSTREAM_BRANCH}", cwd=SOURCE_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="delete ./source and clone again")
    args = ap.parse_args()

    if args.force and SOURCE_DIR.exists():
        shutil.rmtree(SOURCE_DIR, ignore_errors=True)

    if (SOURCE_DIR / ".git").exists():
        update()
    else:
        if SOURCE_DIR.exists():
            shutil.rmtree(SOURCE_DIR, ignore_errors=True)
        clone()

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=SOURCE_DIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    n = sum(1 for _ in iter_lessons())
    print(f"source ready: {SOURCE_DIR} @ {head}, {n} lessons")


if __name__ == "__main__":
    main()
