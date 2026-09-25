"""Tempo estimation.

Scope note: this is only the starting guess. The grid stage (grid.py) refines it
from the transcribed notes themselves and adds the bar lines; beat_this was 1.2% off
on the reference song, so it is not used for the grid. This module sits behind a
registry so a better estimator can replace it without touching the pipeline.

Why the drums stem: demucs has already isolated it by the time we need a tempo,
and beat tracking on isolated drums beats beat tracking on a dense mix. Falls
back to the full mix if there is no drums stem.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger("stemscribe.tempo")

#: Musically plausible range. Estimates outside this are octave errors.
MIN_BPM = 50.0
MAX_BPM = 200.0

#: The ratios beat trackers actually confuse. Half/double time dominate;
#: 3/2 and 2/3 show up in swung and compound-meter material.
OCTAVE_RATIOS = (0.5, 1.0, 2.0, 1.5, 2 / 3)

DEFAULT_TEMPO = 120.0


@dataclass
class TempoCandidate:
    bpm: float
    #: Tempogram strength, normalized so the strongest candidate is 1.0.
    score: float
    #: Ratio to the primary estimate (1.0 = primary, 0.5 = half-time, ...).
    ratio: float

    def as_dict(self) -> dict:
        return {"bpm": round(self.bpm, 2), "score": round(self.score, 4), "ratio": self.ratio}


@dataclass
class TempoEstimate:
    bpm: float
    source: str  # "detected" | "user" | "default"
    method: str
    candidates: list[TempoCandidate] = field(default_factory=list)
    analyzed_stem: str | None = None

    def as_dict(self) -> dict:
        return {
            "bpm": round(self.bpm, 2),
            "source": self.source,
            "method": self.method,
            "analyzed_stem": self.analyzed_stem,
            "candidates": [c.as_dict() for c in self.candidates],
        }


def _in_range(bpm: float) -> bool:
    return MIN_BPM <= bpm <= MAX_BPM


def _refine_from_beat_times(beat_times) -> float | None:
    """BPM from a least-squares fit through the beat times.

    librosa's own tempo is quantized to tempogram bins, and its beat times are
    snapped to onset-envelope frames (~23ms at hop 512), so both land ~2.5% off
    a true 140 BPM. That error is not cosmetic: tempo sets the bar grid, so 2.5%
    drifts the grid several seconds away from the notes over a full song.
    Fitting a line through every beat averages the frame-snapping out and gets
    within ~0.01%.
    """
    import numpy as np

    if len(beat_times) < 3:
        return None
    slope = np.polyfit(np.arange(len(beat_times)), np.asarray(beat_times), 1)[0]
    if slope <= 0:
        return None
    bpm = 60.0 / float(slope)
    return bpm if _in_range(bpm) else None


def estimate_librosa(audio_path: str | pathlib.Path, sr: int = 22050) -> TempoEstimate:
    """librosa beat tracking, refined by regression, with scored alternates."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    if not np.any(onset_env):
        log.warning("no onsets found in %s; falling back to %s BPM", audio_path, DEFAULT_TEMPO)
        return TempoEstimate(DEFAULT_TEMPO, "default", "librosa:no-onsets")

    coarse, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop
    )
    coarse = float(np.atleast_1d(coarse)[0])

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    refined = _refine_from_beat_times(beat_times)
    primary = refined if refined is not None else coarse
    method = (
        "librosa.beat_track+regression" if refined is not None else "librosa.beat_track"
    )

    # Score octave alternates against the global tempogram so the UI can offer
    # real options instead of arbitrary halves and doubles.
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop)
    bpm_axis = librosa.tempo_frequencies(tg.shape[0], sr=sr, hop_length=hop)
    strength = tg.mean(axis=1)

    def score_of(bpm: float) -> float:
        idx = int(np.argmin(np.abs(bpm_axis - bpm)))
        return float(strength[idx])

    seen: dict[int, TempoCandidate] = {}
    for ratio in OCTAVE_RATIOS:
        bpm = primary * ratio
        if not _in_range(bpm):
            continue
        key = round(bpm)
        if key in seen:
            continue
        seen[key] = TempoCandidate(bpm=bpm, score=score_of(bpm), ratio=ratio)

    if not seen:  # primary itself out of range: pull it into range by octaves
        bpm = primary
        while bpm > MAX_BPM:
            bpm /= 2
        while bpm < MIN_BPM and bpm > 0:
            bpm *= 2
        seen[round(bpm)] = TempoCandidate(bpm=bpm, score=1.0, ratio=bpm / primary)

    candidates = sorted(seen.values(), key=lambda c: c.score, reverse=True)
    top = max((c.score for c in candidates), default=0.0) or 1.0
    for c in candidates:
        c.score /= top

    # Trust librosa's own pick when it survived the range filter; the tempogram
    # score orders the alternates but is not reliable enough to overrule it.
    chosen = next((c for c in candidates if c.ratio == 1.0), candidates[0])
    return TempoEstimate(
        bpm=chosen.bpm,
        source="detected",
        method=method,
        candidates=candidates,
    )


def estimate_beat_this(audio_path: str | pathlib.Path) -> TempoEstimate:
    """Hook for beat_this (ISMIR 2024), the current SOTA beat tracker.

    Deliberately not implemented: the spec assigns beat/downbeat tracking to
    rearranged's glue research. When that lands, implement here and register
    below -- or just pass tempo= into process() from the caller.
    """
    raise NotImplementedError(
        "beat_this is rearranged's glue research (v1 non-goal). "
        "Pass tempo=<bpm> into process(), or register an estimator here."
    )


ESTIMATORS: dict[str, Callable[..., TempoEstimate]] = {
    "librosa": estimate_librosa,
    # "beat-this": estimate_beat_this,   # not built -- see docstring
    # "essentia": ...                    # RhythmExtractor2013; heavy dep
}


def get_estimator(name: str) -> Callable[..., TempoEstimate]:
    if name not in ESTIMATORS:
        raise ValueError(f"unknown tempo estimator {name!r} (have: {', '.join(ESTIMATORS)})")
    return ESTIMATORS[name]


def resolve_tempo(
    tempo: float | None,
    stem_paths: dict[str, pathlib.Path] | None = None,
    fallback_audio: str | pathlib.Path | None = None,
    estimator: str = "librosa",
) -> TempoEstimate:
    """Decide the tempo to write into the MIDI.

    An explicit `tempo` always wins -- you know the song, the tracker does not.
    """
    if tempo is not None:
        return TempoEstimate(bpm=float(tempo), source="user", method="explicit")

    target, stem_name = None, None
    if stem_paths and "drums" in stem_paths:
        target, stem_name = stem_paths["drums"], "drums"
    elif fallback_audio is not None:
        target, stem_name = fallback_audio, "mix"

    if target is None:
        return TempoEstimate(DEFAULT_TEMPO, "default", "no-audio-to-analyze")

    try:
        est = get_estimator(estimator)(target)
        est.analyzed_stem = stem_name
        return est
    except Exception as e:  # never let tempo detection sink a whole run
        log.warning("tempo detection failed (%s); using %s BPM", e, DEFAULT_TEMPO)
        return TempoEstimate(DEFAULT_TEMPO, "default", f"failed:{type(e).__name__}")
