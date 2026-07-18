"""stemscribe -- song in, stems + multi-track MIDI out."""
from __future__ import annotations

__version__ = "0.1.0"

from .backends import BACKENDS, BackendError
from .cleanup import CleanupParams, CleanupStats, clean_instrument
from .core import Result, process
from .fetch import FetchedAudio, FetchError, fetch_audio, is_url
from .merge import TRACK_ORDER, TRACK_SPEC
from .prepare import PrepareError, PrepareParams, PreparedAudio
from .tempo import ESTIMATORS, TempoCandidate, TempoEstimate

__all__ = [
    "process",
    "Result",
    "fetch_audio",
    "FetchedAudio",
    "FetchError",
    "is_url",
    "CleanupParams",
    "CleanupStats",
    "clean_instrument",
    "PrepareParams",
    "PreparedAudio",
    "PrepareError",
    "TempoEstimate",
    "TempoCandidate",
    "ESTIMATORS",
    "BACKENDS",
    "BackendError",
    "TRACK_SPEC",
    "TRACK_ORDER",
    "__version__",
]
