"""Stage the Hugging Face Space: space/ plus the stemscribe package, ready to push.

    .venv/bin/python scripts/stage_space.py [out-dir]      # default: space-dist/

A Space repo only sees its own folder, so the pipeline's code is copied in beside
app.py. The local web UI (stemscribe/web) stays out: the Space has its own. No model
weights, audio or caches are copied; the Space downloads its models at start-up.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"
PACKAGE = ROOT / "src" / "stemscribe"
SPACE_FILES = ("app.py", "split.py", "requirements.txt", "packages.txt", "README.md")
LEAVE_OUT = shutil.ignore_patterns("__pycache__", "*.pyc", "web")


def stage(out_dir) -> list[str]:
    out = pathlib.Path(out_dir)
    if out.exists():
        # keep a git checkout of the Space repo intact: replace only what we write
        for name in (*SPACE_FILES, "stemscribe"):
            p = out / name
            shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for name in SPACE_FILES:
        shutil.copy2(SPACE / name, out / name)
    shutil.copytree(PACKAGE, out / "stemscribe", ignore=LEAVE_OUT)
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE")
    return sorted(str(p.relative_to(out)) for p in out.rglob("*")
                  if p.is_file() and ".git" not in p.relative_to(out).parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stage_space", description=__doc__.split("\n\n")[0])
    p.add_argument("out_dir", nargs="?", default=str(ROOT / "space-dist"))
    a = p.parse_args(argv)
    files = stage(a.out_dir)
    print(f"staged {len(files)} files in {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
