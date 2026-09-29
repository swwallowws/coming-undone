"""Stem separation via the demucs *API*.

Deliberately does not use the `demucs.separate` CLI: its save path imports
torchcodec on torchaudio >= 2.9 and dies with ModuleNotFoundError. We read with
demucs.audio.AudioFile, run apply_model ourselves, and save with soundfile.
"""
from __future__ import annotations

import pathlib
from typing import Sequence

import soundfile as sf
import torch

SOURCES = ("drums", "bass", "other", "vocals")

#: model name -> a demucs model already loaded and placed on its device. The Hugging
#: Face Space fills this at start-up (ZeroGPU wants models on cuda at module level);
#: a preloaded model is used as it is, never reloaded or moved back to the CPU.
PRELOADED: dict = {}


def separate(
    audio_path: str | pathlib.Path,
    stems_dir: str | pathlib.Path,
    model_name: str = "htdemucs",
    device: str | None = None,
    overlap: float = 0.25,
    shifts: int = 0,
    progress: bool = False,
) -> dict[str, pathlib.Path]:
    """Split `audio_path` into stems, written as wavs under `stems_dir`.

    Returns {stem_name: path}. Stem names come from the model itself, so a
    non-htdemucs model with a different source list still works.
    """
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import get_model

    audio_path = pathlib.Path(audio_path)
    stems_dir = pathlib.Path(stems_dir)
    stems_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PRELOADED.get(model_name)
    if model is None:
        model = get_model(model_name)
        model.cpu()
    model.eval()

    wav = AudioFile(audio_path).read(
        streams=0,
        samplerate=model.samplerate,
        channels=model.audio_channels,
    )

    # demucs was trained on per-track normalized audio; normalize going in and
    # invert it coming out, or the stems come back at the wrong level.
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    wav = (wav - mean) / (std + 1e-8)

    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None],
            device=device,
            split=True,
            overlap=overlap,
            shifts=shifts,
            progress=progress,
        )[0]

    sources = sources * std + mean

    paths: dict[str, pathlib.Path] = {}
    for name, source in zip(model.sources, sources):
        out = stems_dir / f"{name}.wav"
        sf.write(str(out), source.cpu().numpy().T, model.samplerate)
        paths[name] = out
    return paths


def stem_names(model_name: str = "htdemucs") -> Sequence[str]:
    from demucs.pretrained import get_model

    return tuple(get_model(model_name).sources)
