#!/usr/bin/env python3
"""One-shot runner: fetch upstream -> translate (resumable) -> build docs -> mkdocs build.

    python scripts/pipeline.py                       # gemini, until quota/finished
    python scripts/pipeline.py --provider ollama --limit 10
    python scripts/pipeline.py --skip-translate      # just rebuild the site
    python scripts/pipeline.py --serve               # ... then `mkdocs serve`

Unknown arguments are forwarded to translate.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(*cmd: str) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=HERE.parent).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-translate", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--serve", action="store_true", help="run `mkdocs serve` at the end")
    args, passthrough = ap.parse_known_args()

    if not args.skip_fetch and run(PY, str(HERE / "fetch_source.py")):
        return 1
    if not args.skip_translate and run(PY, str(HERE / "translate.py"), *passthrough):
        return 1
    if not args.skip_build:
        if run(PY, str(HERE / "build_site.py")):
            return 1
        if run(PY, "-m", "mkdocs", "build"):
            return 1
    if args.serve:
        return run(PY, "-m", "mkdocs", "serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
