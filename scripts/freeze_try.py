"""Freeze one finished stemscribe run into the static guided demo page.

    .venv/bin/python scripts/freeze_try.py <job-output-dir> <out-dir> [--title T]

<job-output-dir> is what `stemscribe -o` wrote: stems/*.wav, <song>.mid and
manifest.json. <out-dir> gets a self-contained static site:

    <out-dir>/try/index.html, try.js, try.css   the page (from web/static/try)
    <out-dir>/try/data.json                      parts, notes, bar lines
    <out-dir>/try/stems/<id>.mp3, mix.mp3         128 kbps (ffmpeg)
    <out-dir>/vendor/design/                     the design system

Serve <out-dir> and open /try/. Everything is put on the stems' timeline: a run
that trimmed leading silence writes its MIDI on the original file's timeline,
`offset` seconds later than the stems, so notes and bar lines move back by it.

Built from a copyrighted song, the output stays local (try-dist/ is gitignored).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pretty_midi
import soundfile as sf

from stemscribe.grid import Meter
from stemscribe.merge import track_for_stem

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "stemscribe" / "web" / "static"

#: stem -> the label on its button, in the order the buttons appear.
# Named as the MIDI names its tracks, so a part keeps one name whether you hear its
# audio stem or the notes written from it.
PART_NAMES = {
    "drums": "drums",
    "bass": "bass",
    "vocals": "melody",
    "guitar": "guitar",
    "piano": "piano",
    "other": "comping",
}


def _encode(src: pathlib.Path, dst: pathlib.Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-b:a", "128k", str(dst)],
        check=True,
    )


def _bars(manifest: dict, pm: pretty_midi.PrettyMIDI, offset: float, duration: float):
    """(bpm, meter, bar-line times on the stems' timeline)."""
    grid = manifest.get("grid") or {}
    if grid.get("fitted"):
        meter = Meter.parse(grid.get("meter", "4/4"))
        bpm = float(grid["bpm"])
        bar = meter.pulses * 60.0 / bpm
        first = float(grid["first_bar"]) - offset
        k = math.ceil(-first / bar - 1e-9)
        bars = []
        while first + k * bar <= duration:
            bars.append(first + k * bar)
            k += 1
        return bpm, str(meter), bars
    tempos = pm.get_tempo_changes()[1]
    bpm = float((manifest.get("tempo") or {}).get("bpm") or (tempos[0] if len(tempos) else 120.0))
    bars = [float(t) - offset for t in pm.get_downbeats()]
    return bpm, "4/4", [t for t in bars if 0 <= t <= duration]


def freeze(job_dir, out_dir, *, encode: bool = True, title: str | None = None,
           min_db: float | None = None) -> dict:
    """min_db leaves out any stem quieter than that (RMS, dBFS): a song with no singer still
    gets a vocals stem, and the notes written from its bleed aren't worth showing."""
    job, out = pathlib.Path(job_dir), pathlib.Path(out_dir)
    mids = sorted(job.glob("*.mid"))
    if not mids:
        raise FileNotFoundError(f"no .mid in {job}")
    stems = {p.stem: p for p in sorted((job / "stems").glob("*.wav"))}
    if not stems:
        raise FileNotFoundError(f"no stems/*.wav in {job} (run stemscribe without --no-stems)")
    mpath = job / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.is_file() else {}
    offset = float((manifest.get("prepared_audio") or {}).get("offset") or 0.0)

    page = out / "try"
    shutil.rmtree(page / "stems", ignore_errors=True)
    (page / "stems").mkdir(parents=True, exist_ok=True)
    for f in (STATIC / "try").iterdir():
        if f.is_file():
            shutil.copy2(f, page / f.name)
    shutil.copytree(STATIC / "vendor" / "design", out / "vendor" / "design", dirs_exist_ok=True)
    shutil.copytree(STATIC / "favicons", out / "favicons", dirs_exist_ok=True)

    pm = pretty_midi.PrettyMIDI(str(mids[0]))
    tracks = {i.name: i for i in pm.instruments}
    ext = "mp3" if encode else "wav"

    order = [s for s in PART_NAMES if s in stems] + sorted(set(stems) - set(PART_NAMES))
    duration = max(sf.info(str(p)).duration for p in stems.values())
    mix = None
    parts = []
    for stem in order:
        y, sr = sf.read(str(stems[stem]), dtype="float32", always_2d=True)
        if min_db is not None:
            rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
            if rms <= 0.0 or 20 * np.log10(rms) < min_db:
                continue
        if mix is None:
            mix, mix_sr = np.zeros((round(duration * sr), y.shape[1]), dtype="float32"), sr
        if sr == mix_sr and y.shape[1] == mix.shape[1]:
            mix[: len(y)] += y[: len(mix)]
        audio = f"stems/{stem}.{ext}"
        (_encode if encode else shutil.copy2)(stems[stem], page / audio)

        inst = tracks.get(track_for_stem(stem)[0])
        notes = []
        for n in sorted(inst.notes if inst else [], key=lambda n: (n.start, n.pitch)):
            s, e = n.start - offset, n.end - offset
            if e <= 0 or s >= duration:
                continue
            notes.append([round(max(s, 0.0), 4), round(min(e, duration), 4), int(n.pitch), int(n.velocity)])
        parts.append({"id": stem, "name": PART_NAMES.get(stem, stem), "audio": audio,
                      "drums": bool(inst.is_drum) if inst else stem == "drums", "notes": notes})

    peak = float(np.abs(mix).max()) if mix.size else 0.0
    if peak > 0.99:
        mix *= 0.99 / peak
    mix_wav = page / "mix.wav"
    sf.write(str(mix_wav), mix, mix_sr)
    if encode:
        _encode(mix_wav, page / "mix.mp3")
        mix_wav.unlink()

    bpm, meter, bars = _bars(manifest, pm, offset, duration)
    data = {
        "title": title or mids[0].stem,
        "bpm": round(bpm, 3),
        "meter": meter,
        "duration": round(duration, 4),
        "bars": [round(b, 4) for b in bars],
        "mix": f"mix.{ext}",
        "parts": parts,
    }
    (page / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="freeze_try", description=__doc__.split("\n\n")[0])
    p.add_argument("job_dir", help="a stemscribe -o output folder")
    p.add_argument("out_dir", help="where the static site goes (try-dist/)")
    p.add_argument("--title", default=None, help="song title on the page (default: the MIDI's name)")
    p.add_argument("--no-encode", dest="encode", action="store_false",
                   help="copy the WAVs instead of encoding mp3")
    p.add_argument("--min-db", type=float, default=-60.0,
                   help="leave out stems quieter than this RMS level in dBFS (default -60; "
                        "use --keep-silent to keep every stem)")
    p.add_argument("--keep-silent", action="store_true", help="keep near-silent stems too")
    a = p.parse_args(argv)
    data = freeze(a.job_dir, a.out_dir, encode=a.encode, title=a.title,
                  min_db=None if a.keep_silent else a.min_db)
    counts = ", ".join(f"{q['name']}({len(q['notes'])})" for q in data["parts"])
    print(f"froze {data['title']}: {data['duration']:.1f}s, {len(data['bars'])} bars, {counts}")
    print(f"serve {a.out_dir} and open /try/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
