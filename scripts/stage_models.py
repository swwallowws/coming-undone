"""Stage the public model repo for the "In your browser" engine, ready to upload.

    .venv/bin/python scripts/stage_models.py [models-dir]   # default: browser/models-out/

Copies the ADT_STR int8 drum graphs (browser/models/finish.mjs makes them) into
models-dist/ and writes models.json, the manifest the page reads. models-dist/README.md
(the model card, tracked in git) stays as it is. htdemucs is not copied: the manifest
points at timcsy/demucs-web-onnx on Hugging Face, pinned. basic-pitch ships with the
page itself (static/vendor/browser-engine/basic-pitch/). MuScriptor small is left out
on purpose: its weights are gated and CC-BY-NC.

out/upload_models.py uploads the result to swwallowws/coming-undone-browser-models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "models-dist"
REPO_ID = "swwallowws/coming-undone-browser-models"
HTDEMUCS_REV = "92e33df61cfc9eb820272aaa62d2ef6dcf4d950d"   # same as browser/src/models.js
HTDEMUCS = {
    "url": f"https://huggingface.co/timcsy/demucs-web-onnx/resolve/{HTDEMUCS_REV}/htdemucs_embedded.onnx",
    "bytes": 180534758,
    "licence": "no licence stated on timcsy/demucs-web-onnx; Demucs (Meta) is MIT",
}
ADT_FILES = {"encoder": "adt_encoder.int8.onnx", "decoder": "adt_decoder.int8.onnx"}
#: every file the upload publishes, and nothing else
PUBLISHED = ("README.md", "models.json", *ADT_FILES.values())


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage(src: pathlib.Path, out: pathlib.Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    adt = {}
    for part, name in ADT_FILES.items():
        shutil.copy2(src / name, out / name)
        adt[part] = {"file": name, "bytes": (out / name).stat().st_size, "sha256": sha256(out / name)}
    adt["licence"] = "CC-BY-SA-4.0 (ADT_STR, Melucci, Merialdo, Akama 2026; int8 adaptation)"
    manifest = {"version": 1, "htdemucs": HTDEMUCS, "adt": adt}
    (out / "models.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stage_models", description=__doc__.split("\n\n")[0])
    p.add_argument("models_dir", nargs="?", default=str(ROOT / "browser" / "models-out"))
    a = p.parse_args(argv)
    src = pathlib.Path(a.models_dir)
    missing = [n for n in ADT_FILES.values() if not (src / n).is_file()]
    if missing:
        print(f"missing in {src}: {', '.join(missing)} (run browser/models/finish.mjs first)", file=sys.stderr)
        return 1
    stage(src)
    for name in PUBLISHED:
        f = OUT / name
        print(f"{name:26} {f.stat().st_size:>11,} bytes" if f.exists() else f"{name:26} MISSING")
    print(f"staged in {OUT}; upload with .venv/bin/python out/upload_models.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
