"""Input conditioning, run before anything expensive touches the audio.

Four jobs, all optional:
  1. decode to a standard wav once, instead of per-stage
  2. drop metadata and embedded album art
  3. cut to a section / duration cap
  4. trim leading and trailing silence

Timing contract: every offset we introduce here is recorded and added back to
the MIDI at the end, so note times always refer to the ORIGINAL input file. If
you trim 4.2s of dead air off the front, the MIDI still says the first note
happens at 4.2s -- because in the file you handed us, it does.
"""
from __future__ import annotations

import json
import logging
import pathlib
import shutil
import subprocess
from dataclasses import dataclass

log = logging.getLogger("stemscribe.prepare")

TARGET_SR = 44100
TARGET_CHANNELS = 2


class PrepareError(RuntimeError):
    pass


@dataclass
class PrepareParams:
    #: Decode once to 44.1k stereo wav so downstream stages skip mp3 decoding.
    normalize_wav: bool = True
    #: Drop ID3 tags and embedded cover art.
    strip_metadata: bool = True
    #: Trim leading/trailing silence.
    trim_silence: bool = True
    #: Anything this far below peak counts as silence.
    silence_top_db: float = 50.0
    #: Section start in seconds (None = from the beginning).
    start: float | None = None
    #: Section length in seconds (None = to the end).
    duration: float | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    @property
    def any_enabled(self) -> bool:
        return (
            self.normalize_wav
            or self.strip_metadata
            or self.trim_silence
            or self.start is not None
            or self.duration is not None
        )


@dataclass
class PreparedAudio:
    #: What downstream stages should actually read.
    path: pathlib.Path
    #: Seconds between the original file's t=0 and this audio's t=0. Added back
    #: to every MIDI note at the end.
    offset: float = 0.0
    original_duration: float | None = None
    duration: float | None = None
    trimmed_head: float = 0.0
    trimmed_tail: float = 0.0
    applied: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.applied is None:
            self.applied = []

    def as_dict(self) -> dict:
        return {
            "offset": round(self.offset, 4),
            "original_duration": round(self.original_duration, 3) if self.original_duration else None,
            "duration": round(self.duration, 3) if self.duration else None,
            "trimmed_head": round(self.trimmed_head, 4),
            "trimmed_tail": round(self.trimmed_tail, 4),
            "applied": self.applied,
        }


def probe_duration(path: str | pathlib.Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _decode(src: pathlib.Path, dst: pathlib.Path, params: PrepareParams) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PrepareError("ffmpeg not found on PATH; needed to prepare input audio")

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    # -ss before -i seeks fast; accurate enough on a decoded stream.
    if params.start:
        cmd += ["-ss", str(params.start)]
    cmd += ["-i", str(src)]
    if params.duration:
        cmd += ["-t", str(params.duration)]
    if params.strip_metadata:
        # -vn drops the cover-art "video" stream; -map_metadata -1 drops tags.
        cmd += ["-map_metadata", "-1", "-vn"]
    cmd += ["-ar", str(TARGET_SR), "-ac", str(TARGET_CHANNELS), str(dst)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PrepareError(f"ffmpeg failed preparing {src.name}:\n{proc.stderr[-800:]}")


def _trim_silence(path: pathlib.Path, top_db: float) -> tuple[float, float]:
    """Trim silence in place. Returns (head_seconds, tail_seconds) removed."""
    import librosa
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1)
    if not np.any(np.abs(mono) > 0):
        log.warning("%s is entirely silent; not trimming", path.name)
        return 0.0, 0.0

    _, (i0, i1) = librosa.effects.trim(np.asfortranarray(mono), top_db=top_db)
    if i1 <= i0:
        return 0.0, 0.0

    head = i0 / sr
    tail = (len(mono) - i1) / sr
    if head <= 0 and tail <= 0:
        return 0.0, 0.0

    sf.write(str(path), data[i0:i1], sr)
    return head, tail


def prepare_audio(
    input_path: str | pathlib.Path,
    work_dir: str | pathlib.Path,
    params: PrepareParams | None = None,
) -> PreparedAudio:
    """Condition `input_path` for the pipeline. Returns what to read + the offset.

    Never mutates the input file: everything happens on a copy in `work_dir`.
    """
    p = params or PrepareParams()
    input_path = pathlib.Path(input_path)
    work_dir = pathlib.Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    original_duration = probe_duration(input_path)

    if not p.any_enabled:
        return PreparedAudio(
            path=input_path,
            original_duration=original_duration,
            duration=original_duration,
            applied=[],
        )

    dst = work_dir / f"{input_path.stem}_prepared.wav"
    _decode(input_path, dst, p)

    applied: list[str] = []
    if p.normalize_wav:
        applied.append(f"decoded to {TARGET_SR}Hz/{TARGET_CHANNELS}ch wav")
    if p.strip_metadata:
        applied.append("stripped metadata + cover art")
    if p.start or p.duration:
        applied.append(
            f"section start={p.start or 0}s duration={p.duration or 'end'}"
        )

    head = tail = 0.0
    if p.trim_silence:
        head, tail = _trim_silence(dst, p.silence_top_db)
        if head or tail:
            applied.append(f"trimmed silence head={head:.2f}s tail={tail:.2f}s")

    # Offset back to the original timeline: where the section started, plus the
    # dead air we removed from the front of it.
    offset = (p.start or 0.0) + head

    return PreparedAudio(
        path=dst,
        offset=offset,
        original_duration=original_duration,
        duration=probe_duration(dst),
        trimmed_head=head,
        trimmed_tail=tail,
        applied=applied,
    )


def shift_midi(midi_path: str | pathlib.Path, offset: float) -> int:
    """Shift every note in `midi_path` later by `offset` seconds, in place.

    This is what puts the MIDI back on the original file's timeline after we
    trimmed or sectioned the audio. Returns the number of notes shifted.
    """
    import pretty_midi

    if not offset:
        return 0

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    n = 0
    for inst in pm.instruments:
        for note in inst.notes:
            note.start += offset
            note.end += offset
            n += 1
    pm.write(str(midi_path))
    return n
