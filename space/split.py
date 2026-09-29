"""The Space's one job, without Gradio: audio in, MIDI and parts out.

app.py wraps `run` in a ZeroGPU function and a Gradio API endpoint. Keeping the work
here, free of gradio and spaces, lets the repo's tests cover it on any machine.

The options are one JSON object, so the endpoint's signature stays two inputs while
the page grows. Every key is optional; see DEFAULTS. Unknown keys are refused, so a
typo on the page fails loudly instead of being ignored.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
from typing import Callable

import pretty_midi

from stemscribe import grid as _grid
from stemscribe.cleanup import CleanupParams
from stemscribe.core import _jsonable, process
from stemscribe.prepare import PrepareParams

#: The longest stretch of audio one online run takes, in seconds. Visitors get 2
#: minutes of GPU a day signed out, 5 with a free Hugging Face account, so a run has
#: to fit well inside that. A longer file is cut to this from its `start`.
MAX_SECONDS = 30.0

BACKENDS = ("muscriptor", "basic-pitch")
DEMUCS_MODELS = ("htdemucs", "htdemucs_6s")
MONO_STEMS = ("bass", "vocals")

DEFAULTS: dict = {
    "backend": "muscriptor",
    "demucs_model": "htdemucs",
    "meter": "4/4",
    "downbeat": None,
    "snap": False,
    "tempo": None,
    "start": None,
    "duration": None,
    "include_vocals_melody": True,
    "mono_stems": [],
    "cleanup": True,
    "de_overlap": True,
    "max_duration_beats": 2.0,
    "velocity_floor": 15,
    "legato": False,
    "legato_max_gap_beats": 0.25,
    "trim_silence": True,
    "strip_metadata": True,
    "stems_audio": True,
}


class OptionError(ValueError):
    """A request the Space will not run; the message is shown to the visitor."""


def _num(opts: dict, key: str, lo: float, hi: float) -> None:
    v = opts[key]
    if v is None:
        return
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise OptionError(f"{key} must be a number")
    if not lo <= v <= hi:
        raise OptionError(f"{key} must be between {lo:g} and {hi:g}")
    opts[key] = float(v)


def parse_options(raw: str | dict | None) -> dict:
    """The request's options merged over DEFAULTS, checked, with the section clamped
    to MAX_SECONDS."""
    if raw in (None, ""):
        given: dict = {}
    elif isinstance(raw, dict):
        given = raw
    else:
        try:
            given = json.loads(raw)
        except json.JSONDecodeError as e:
            raise OptionError(f"options are not valid JSON: {e.msg}") from None
    if not isinstance(given, dict):
        raise OptionError("options must be a JSON object")
    unknown = sorted(set(given) - set(DEFAULTS))
    if unknown:
        raise OptionError(f"unknown option(s): {', '.join(unknown)}")
    opts = {**DEFAULTS, **given}

    if opts["backend"] not in BACKENDS:
        raise OptionError(f"backend must be one of {', '.join(BACKENDS)}")
    if opts["demucs_model"] not in DEMUCS_MODELS:
        raise OptionError(f"demucs_model must be one of {', '.join(DEMUCS_MODELS)}")
    try:
        meter = _grid.Meter.parse(str(opts["meter"]))
    except ValueError as e:
        raise OptionError(str(e)) from None
    opts["meter"] = str(meter)
    if opts["downbeat"] is not None:
        d = opts["downbeat"]
        if isinstance(d, bool) or not isinstance(d, int) or not 1 <= d <= meter.pulses:
            raise OptionError(f"downbeat must be a whole number from 1 to {meter.pulses}")
    mono = opts["mono_stems"]
    if not isinstance(mono, list) or any(s not in MONO_STEMS for s in mono):
        raise OptionError(f"mono_stems may list {', '.join(MONO_STEMS)}")
    for key in ("snap", "include_vocals_melody", "cleanup", "de_overlap", "legato",
                "trim_silence", "strip_metadata", "stems_audio"):
        if not isinstance(opts[key], bool):
            raise OptionError(f"{key} must be true or false")
    _num(opts, "tempo", 20, 400)
    _num(opts, "start", 0, 24 * 3600)
    _num(opts, "duration", 0, 24 * 3600)
    _num(opts, "max_duration_beats", 0, 64)
    _num(opts, "velocity_floor", 0, 127)
    _num(opts, "legato_max_gap_beats", 0, 16)
    opts["velocity_floor"] = int(opts["velocity_floor"])
    # the section: never more than MAX_SECONDS, wherever it starts
    opts["duration"] = min(opts["duration"] or MAX_SECONDS, MAX_SECONDS)
    return opts


def gpu_seconds(opts: dict) -> int:
    """How long to ask ZeroGPU for, from the section length and the work it needs.

    A guess to tune on the real hardware: every muscriptor stem is a model load plus
    its transcription, demucs and the drums are a few seconds each. Asking for less
    queues sooner and is refused less often when a visitor's quota runs low; asking
    for too little stops the run, so it keeps a margin."""
    seconds = opts["duration"] or MAX_SECONDS
    pitched = 5 if opts["demucs_model"] == "htdemucs_6s" else 3
    if opts["backend"] == "muscriptor":
        work = 15 + pitched * (8 + 0.6 * seconds)
    else:
        work = 15 + pitched * (2 + 0.1 * seconds)
    return int(min(120, max(30, round(work))))


def _parts(midi: pathlib.Path, out: pathlib.Path) -> tuple[dict, list[pathlib.Path]]:
    """Every track as its own part (the local server's /parts shape) and its own .mid,
    on the same tempo map and bar lines, so it drops into a DAW lined up."""
    pm = pretty_midi.PrettyMIDI(str(midi))
    out.mkdir(parents=True, exist_ok=True)
    parts, files = [], []
    for inst in pm.instruments:
        name = pathlib.Path(inst.name or "part").name
        parts.append({"name": inst.name, "file": f"{name}.mid", "drum": bool(inst.is_drum),
                      "notes": [[int(n.pitch), round(float(n.start), 4), round(float(n.end), 4),
                                 int(n.velocity)] for n in inst.notes]})
        one = pretty_midi.PrettyMIDI(str(midi))
        one.instruments = [i for i in one.instruments if i.name == inst.name]
        target = out / f"{name}.mid"
        one.write(str(target))
        files.append(target)
    return {"end": float(pm.get_end_time()), "parts": parts}, files


def _mp3(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-b:a", "128k", str(dst)],
                   check=True)
    return dst


def run(audio: str | pathlib.Path, opts: dict, work: str | pathlib.Path,
        progress: Callable[[str, str], None] | None = None) -> dict:
    """Run the pipeline on one upload. Returns the page's result JSON and the files to
    hand back: the combined MIDI, one MIDI per part, and the audio (instrumental,
    stems as mp3) plus manifest.json. File references in the JSON are base names, which
    is how the page finds each file among the endpoint's outputs."""
    work = pathlib.Path(work)
    out = work / "out"
    res = process(
        audio,
        out_dir=out,
        backend=opts["backend"],
        demucs_model=opts["demucs_model"],
        include_vocals_melody=opts["include_vocals_melody"],
        mono_stems=tuple(opts["mono_stems"]),
        cleanup=opts["cleanup"],
        cleanup_params=CleanupParams(
            de_overlap=opts["de_overlap"],
            max_duration_beats=opts["max_duration_beats"],
            velocity_floor=opts["velocity_floor"],
            legato=opts["legato"],
            legato_max_gap_beats=opts["legato_max_gap_beats"],
        ),
        prepare=PrepareParams(
            trim_silence=opts["trim_silence"],
            strip_metadata=opts["strip_metadata"],
            start=opts["start"],
            duration=opts["duration"],
        ),
        tempo=opts["tempo"],
        meter=opts["meter"],
        downbeat=opts["downbeat"],
        snap=opts["snap"],
        keep_stems=True,
        instrumental=opts["stems_audio"],
        cache=False,           # one visitor's audio never feeds another's run
        progress=progress,
    )
    if not res.midi_path:
        raise RuntimeError("the run wrote no MIDI")

    parts, part_files = _parts(res.midi_path, work / "parts")
    files: list[pathlib.Path] = []
    stems: dict[str, str] = {}
    if opts["stems_audio"]:
        (work / "stems").mkdir(parents=True, exist_ok=True)
        for name, wav in sorted(res.stem_paths.items()):
            stems[name] = _mp3(pathlib.Path(wav), work / "stems" / f"{name}.mp3").name
            files.append(work / "stems" / stems[name])
    if res.instrumental_path:
        files.append(pathlib.Path(res.instrumental_path))
    if res.manifest_path:
        files.append(pathlib.Path(res.manifest_path))

    m = res.manifest
    result = _jsonable({
        "tempo": res.tempo.as_dict() if res.tempo else None,
        "grid": m.get("grid"),
        "prepared": res.prepared.as_dict() if res.prepared else None,
        "source": None,
        "track_map": res.track_map,
        "tracks": m.get("tracks", {}),
        "timings": m.get("timings_sec", {}),
        "warnings": res.warnings,
        "midi": pathlib.Path(res.midi_path).name,
        "instrumental": pathlib.Path(res.instrumental_path).name if res.instrumental_path else None,
        "stems": stems,
        "manifest": pathlib.Path(res.manifest_path).name if res.manifest_path else None,
        "parts": parts,
        "section": {"start": opts["start"] or 0.0, "duration": opts["duration"],
                    "max_seconds": MAX_SECONDS},
    })
    return {"result": result, "midi": pathlib.Path(res.midi_path), "parts": part_files,
            "files": files}


def sweep(root: str | pathlib.Path, max_age: float, now: float | None = None) -> int:
    """Delete run folders under `root` older than `max_age` seconds; returns how many."""
    import time

    root = pathlib.Path(root)
    if not root.is_dir():
        return 0
    now = time.time() if now is None else now
    gone = 0
    for d in root.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > max_age:
            shutil.rmtree(d, ignore_errors=True)
            gone += 1
    return gone
