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


# --- extension point: drums -------------------------------------------------
def transcribe_drums(stem_wav: pathlib.Path, out_mid: pathlib.Path, **_):
    """Not built (v1 non-goal). A drum backend would map onsets to GM channel 10
    percussion keys rather than pitches, so it does not fit the pitched contract
    above. Wire a real one in here and drop "drums" from UNPITCHED_STEMS."""
    raise NotImplementedError("drum transcription is a v1 non-goal")


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
