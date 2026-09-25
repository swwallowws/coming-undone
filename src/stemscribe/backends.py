"""Pluggable audio->MIDI backends.

Registry contract: fn(stem_wav_path, out_mid_path) -> pathlib.Path | None
Each fn writes ONE single-track-ish .mid for ONE stem, or returns None if it
declines the stem. Never hand a backend the full mix -- that is what the stem
separation is for.

Shape follows rearranged/test-harness/transcribe.py. Klangio and YourMT3+
slot in here later; they are intentionally not built yet.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
from typing import Callable, Protocol

# Pitched backends cannot transcribe an unpitched drum kit. Skipped with a
# warning recorded in the manifest; see transcribe_drums() for the hook.
UNPITCHED_STEMS = frozenset({"drums"})


class BackendError(RuntimeError):
    pass


class Backend(Protocol):
    def __call__(
        self, stem_wav: pathlib.Path, out_mid: pathlib.Path, **kwargs
    ) -> pathlib.Path | None: ...


# --- backend: basic-pitch (Apache-2.0, commercial-safe default) --------------
def transcribe_basic_pitch(
    stem_wav: pathlib.Path,
    out_mid: pathlib.Path,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 127.7,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
    midi_tempo: float = 120.0,
    **_,
) -> pathlib.Path | None:
    """Spotify basic-pitch via the Python API.

    Uses predict() rather than predict_and_save() or the CLI: the CLI pays the
    import cost per call, and predict() hands back a PrettyMIDI we can write
    where we want without the `<stem>_basic_pitch.mid` name dance.
    """
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, midi, _ = predict(
        str(stem_wav),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        midi_tempo=midi_tempo,
    )
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out_mid))
    return out_mid if out_mid.exists() else None


# --- backend: muscriptor (code MIT, weights CC-BY-NC) -----------------------
def transcribe_muscriptor(
    stem_wav: pathlib.Path,
    out_mid: pathlib.Path,
    timeout: int = 1800,
    **_,
) -> pathlib.Path | None:
    """MuScriptor CLI. NON-COMMERCIAL WEIGHTS -- prototype/eval only.

    Needs `huggingface-cli login` and accepted terms at
    huggingface.co/MuScriptor/muscriptor-medium. First run downloads ~2GB into
    the HF cache, so it is slow before it is fast.
    """
    if not shutil.which("muscriptor"):
        raise BackendError(
            "`muscriptor` not on PATH. pip install 'stemscribe[muscriptor]' "
            "(CC-BY-NC weights: non-commercial use only)"
        )
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["muscriptor", "transcribe", str(stem_wav), "-o", str(out_mid)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        if "401" in tail or "gated" in tail.lower() or "authenticate" in tail.lower():
            raise BackendError(
                "muscriptor could not fetch its gated weights. Run "
                "`huggingface-cli login` and accept the terms at "
                f"huggingface.co/MuScriptor/muscriptor-medium.\n{tail}"
            )
        raise BackendError(f"muscriptor failed on {stem_wav.name}:\n{tail}")
    return out_mid if out_mid.exists() else None


# --- drums: ADT_STR (code and weights CC BY-SA 4.0) --------------------------
#: Melucci, Merialdo, Akama 2026, https://github.com/pier-maker92/ADT_STR. The model
#: repo ships its own code, which is imported, so it is pinned to a revision.
ADT_STR_REPO = "Pierfrancesco/adt-str"
ADT_STR_REVISION = "a33c5c6b191a4ca1e0f6dc22140947485eb36ce8"   # 2026-09-17
ADT_STR_VARIANT = "setting-tau-0.8"

#: ADT_STR writes its own "GM custom" class numbers without converting them back.
#: Each class -> the first standard GM drum in it (its MappingUtils table).
ADT_STR_TO_GM = {35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42, 43: 44,
                 44: 46, 45: 47, 46: 49, 47: 50, 48: 51, 49: 52, 50: 54, 51: 55, 52: 56,
                 53: 58, 54: 60, 55: 69, 56: 71, 57: 73, 58: 75, 59: 78, 60: 80}


def to_standard_gm(pm):
    """ADT_STR output as one standard-GM drum track named "drums"."""
    import pretty_midi

    out = pretty_midi.PrettyMIDI()
    kit = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    for inst in pm.instruments:
        for n in inst.notes:
            kit.notes.append(pretty_midi.Note(n.velocity, ADT_STR_TO_GM.get(n.pitch, n.pitch),
                                              n.start, n.end))
    kit.notes.sort(key=lambda n: (n.start, n.pitch))
    out.instruments.append(kit)
    return out


def transcribe_adt_str(stem_wav: pathlib.Path, out_mid: pathlib.Path,
                       variant: str = ADT_STR_VARIANT, **_) -> pathlib.Path | None:
    """ADT_STR on the drums stem. Needs `pip install 'stemscribe[drums]'`; the first
    run downloads the model repo from Hugging Face."""
    import sys
    import tempfile

    import pretty_midi
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise BackendError("drum transcription needs pip install 'stemscribe[drums]'") from e
    repo = snapshot_download(ADT_STR_REPO, revision=ADT_STR_REVISION)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from adt_transcriber import ADTTranscriber
    except ImportError as e:
        raise BackendError(f"ADT_STR needs its model's requirements ({e.name} is missing): "
                           f"pip install 'stemscribe[drums]'") from e

    tr = ADTTranscriber.from_pretrained(repo, variant=variant)
    with tempfile.TemporaryDirectory() as tmp:
        raw = pathlib.Path(tr.transcribe(str(stem_wav), output_dir=tmp))
        pm = pretty_midi.PrettyMIDI(str(raw))
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    to_standard_gm(pm).write(str(out_mid))
    return out_mid if out_mid.exists() else None


DRUM_BACKENDS: dict[str, Callable[..., pathlib.Path | None]] = {
    "adt-str": transcribe_adt_str,   # CC BY-SA 4.0: commercial use allowed, with credit
}


#: What ADT_STR's own code imports at inference (its repo's pyproject also pins
#: torch 2.8 / torchaudio 2.8 / torchcodec, which stemscribe does not force).
ADT_STR_NEEDS = ("huggingface_hub", "transformers", "omegaconf", "safetensors")


def drums_available() -> bool:
    """Whether the drum backend's extra is installed (it is optional)."""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ADT_STR_NEEDS)


# --- sparse stems -------------------------------------------------------------
SILENT_RMS = 1e-3


def is_sparse(n_notes: int, seconds: float, bpm: float, rms: float) -> bool:
    """A stem with sound in it but under one note per 4 bars came out nearly empty
    (MuScriptor gave 3 bass notes for a 4-minute song): transcribe it again with the
    fallback backend. Short sections (under 8 bars) are never judged."""
    bars = seconds * bpm / 240.0
    return rms > SILENT_RMS and bars >= 8 and n_notes < bars / 4


FALLBACK_BACKEND = "basic-pitch"


BACKENDS: dict[str, Callable[..., pathlib.Path | None]] = {
    "basic-pitch": transcribe_basic_pitch,  # Apache-2.0, commercial-safe
    "muscriptor": transcribe_muscriptor,    # DEFAULT: best quality, CC-BY-NC (non-commercial)
    # "klangio": ...   # commercial API -- scaffold in rearranged, not built
    # "yourmt3": ...   # open self-host, GPLv3 -- not built
}

NONCOMMERCIAL_BACKENDS = frozenset({"muscriptor"})


def get_backend(name: str) -> Callable[..., pathlib.Path | None]:
    if name not in BACKENDS:
        raise BackendError(f"unknown backend {name!r} (have: {', '.join(BACKENDS)})")
    return BACKENDS[name]
