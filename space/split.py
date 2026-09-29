"""The Space's one job, without Gradio: audio in, MIDI and parts out.

app.py wraps `run` in a ZeroGPU function and a Gradio API endpoint. Keeping the work
here, free of gradio and spaces, lets the repo's tests cover it on any machine.

The options are one JSON object, so the endpoint's signature stays two inputs while
the page grows. Every key is optional; see DEFAULTS. Unknown keys are refused, so a
typo on the page fails loudly instead of being ignored.
"""
from __future__ import annotations

import json
import math
import pathlib
import shutil
import subprocess
from typing import Callable

import pretty_midi
import soundfile as sf

from stemscribe import backends as _backends
from stemscribe import grid as _grid
from stemscribe.cleanup import CleanupParams
from stemscribe.core import _jsonable, process
from stemscribe.core import gpu_stage as _gpu_stage
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


# ---- the GPU budget ------------------------------------------------------------------
# Only core.gpu_stage runs on the GPU: demucs, then MuScriptor on each pitched stem and
# ADT_STR on the drums. The request is sized to the prepared audio (after the section
# cut and the silence trim), measured before asking.
#
# Measured on the Space (2026-09-29, a 20 s clip, htdemucs + muscriptor: 3 pitched
# stems and the drums): separate 1.6 s, transcribe 28.2 s. Each MuScriptor stem is a
# fresh `muscriptor` process that loads its model, so it has a fixed cost as well as a
# per-second one; call it 4 s + 0.2 s per audio second, and the drums 4 s. Every term
# below is that plus about half again, and BASE covers the worker start (the models'
# move onto the GPU). A request that is too short stops the run and still costs the
# visitor, so the margin stays; the numbers are to tighten once more runs are logged
# (the result's timings carry gpu_call, the whole call, and the stage timings).
GPU_BASE = 8.0
GPU_SEPARATE_PER_S = {"htdemucs": 0.2, "htdemucs_6s": 0.35}
GPU_MUSCRIPTOR_STEM = (6.0, 0.3)       # seconds per stem, + seconds per audio second
GPU_DRUMS = (4.0, 0.15)
GPU_MAX = 120

#: What ZeroGPU charged per requested second: the first run asked for 93 s and a
#: signed-out visitor's 120 s were gone after it (the quota counted 140 s).
QUOTA_COST = 1.5
#: Daily ZeroGPU quota in seconds (huggingface.co/docs/hub/en/spaces-zerogpu).
QUOTA_SIGNED_OUT = 120
QUOTA_FREE_ACCOUNT = 300

PITCHED_STEMS_OF = {"htdemucs": ("vocals", "bass", "other"),
                    "htdemucs_6s": ("vocals", "bass", "guitar", "piano", "other")}


def gpu_seconds(seconds: float, demucs_model: str = "htdemucs", backend: str = "muscriptor",
                include_vocals_melody: bool = True, drums: bool = True) -> int:
    """How long to ask ZeroGPU for, for `seconds` of prepared audio.
    web/static/js/engines.js (onlineGpuSeconds) mirrors this for the page."""
    s = max(0.0, min(float(seconds), MAX_SECONDS))
    work = GPU_BASE + GPU_SEPARATE_PER_S[demucs_model] * s
    if backend in _backends.GPU_BACKENDS:
        pitched = [x for x in PITCHED_STEMS_OF[demucs_model]
                   if include_vocals_melody or x != "vocals"]
        work += len(pitched) * (GPU_MUSCRIPTOR_STEM[0] + GPU_MUSCRIPTOR_STEM[1] * s)
    if drums:
        work += GPU_DRUMS[0] + GPU_DRUMS[1] * s
    return int(min(GPU_MAX, math.ceil(work)))


def runs_per_day(request: int, quota: int) -> int:
    """How many runs of `request` seconds fit in a day's `quota`: a run starts while
    the quota left covers the request, and costs QUOTA_COST times it."""
    if request > quota:
        return 0
    return int((quota - request) // (request * QUOTA_COST)) + 1


def gpu_estimate(seconds: float, opts: dict) -> dict:
    """The request for this run, and what a day's quota holds of runs like it."""
    req = gpu_seconds(seconds, opts["demucs_model"], opts["backend"],
                      opts["include_vocals_melody"], _backends.drums_available())
    return {"seconds": round(float(seconds), 2), "request": req,
            "runs_signed_out": runs_per_day(req, QUOTA_SIGNED_OUT),
            "runs_free_account": runs_per_day(req, QUOTA_FREE_ACCOUNT)}


def job_seconds(job: dict) -> float:
    """The length of a gpu_stage job's prepared audio, in seconds."""
    return float(sf.info(job["audio"]).duration)


def job_gpu_seconds(job: dict) -> int:
    """The ZeroGPU duration for one core.gpu_stage job (app.py's duration callable)."""
    return gpu_seconds(job_seconds(job), job["demucs_model"], job["backend"],
                       job["include_vocals_melody"],
                       bool(job["drums"]) and _backends.drums_available())


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
        progress: Callable[[str, str], None] | None = None,
        gpu: Callable[[dict], dict] | None = None) -> dict:
    """Run the pipeline on one upload. Returns the page's result JSON and the files to
    hand back: the combined MIDI, one MIDI per part, and the audio (instrumental,
    stems as mp3) plus manifest.json. File references in the JSON are base names, which
    is how the page finds each file among the endpoint's outputs.

    gpu: runs core.gpu_stage's job (app.py: inside the ZeroGPU function); None runs it
    in this process. Everything else here runs on the CPU."""
    work = pathlib.Path(work)
    out = work / "out"
    asked: dict = {}

    def on_gpu(job: dict) -> dict:
        asked.update(gpu_estimate(job_seconds(job), opts))
        return gpu(job) if gpu else _gpu_stage(job)

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
        gpu=on_gpu,
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
        # what this run asked ZeroGPU for, and a day's quota in runs like it
        "gpu": asked or None,
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
