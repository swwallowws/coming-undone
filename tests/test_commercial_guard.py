"""The commercial safety guard.

The default backend is now muscriptor (CC-BY-NC), so "safe by default" is gone.
STEMSCRIBE_COMMERCIAL=1 restores it structurally: a non-commercial backend then
hard-fails before doing any work, so rearranged's shipping build cannot land
CC-BY-NC weights in a product no matter what the default is.
"""
import numpy as np
import pytest
import soundfile as sf

from stemscribe.backends import BackendError
from stemscribe.core import process


def tiny_wav(tmp_path):
    p = tmp_path / "in.wav"
    sf.write(str(p), np.zeros(22050, dtype=np.float32), 22050)
    return p


def test_commercial_env_blocks_noncommercial_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("STEMSCRIBE_COMMERCIAL", "1")
    with pytest.raises(BackendError, match="CC-BY-NC"):
        process(tiny_wav(tmp_path), out_dir=tmp_path / "out", backend="muscriptor")


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes"])
def test_commercial_env_truthy_values(tmp_path, monkeypatch, val):
    monkeypatch.setenv("STEMSCRIBE_COMMERCIAL", val)
    with pytest.raises(BackendError):
        process(tiny_wav(tmp_path), out_dir=tmp_path / "out", backend="muscriptor")


def test_guard_fires_before_any_separation(tmp_path, monkeypatch):
    """It must raise cheaply, before demucs runs, or the guard is pointless."""
    monkeypatch.setenv("STEMSCRIBE_COMMERCIAL", "1")
    monkeypatch.setattr(
        "stemscribe.separate.separate",
        lambda *a, **k: pytest.fail("separation ran despite the commercial guard"),
    )
    with pytest.raises(BackendError):
        process(tiny_wav(tmp_path), out_dir=tmp_path / "out", backend="muscriptor")


def test_commercial_env_unset_allows_muscriptor_default(tmp_path, monkeypatch):
    """Personal use (env unset): muscriptor default must not be blocked. We stop
    at separation to avoid the real 15-min transcription."""
    monkeypatch.delenv("STEMSCRIBE_COMMERCIAL", raising=False)
    monkeypatch.setattr(
        "stemscribe.separate.separate",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop-after-guard")),
    )
    with pytest.raises(RuntimeError, match="stop-after-guard"):
        process(tiny_wav(tmp_path), out_dir=tmp_path / "out")  # default backend


def test_commercial_env_does_not_block_basic_pitch(tmp_path, monkeypatch):
    """basic-pitch is Apache-2.0; the guard must let it through."""
    monkeypatch.setenv("STEMSCRIBE_COMMERCIAL", "1")
    monkeypatch.setattr(
        "stemscribe.separate.separate",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop-after-guard")),
    )
    with pytest.raises(RuntimeError, match="stop-after-guard"):
        process(tiny_wav(tmp_path), out_dir=tmp_path / "out", backend="basic-pitch")


def test_default_backend_is_muscriptor():
    import inspect

    assert inspect.signature(process).parameters["backend"].default == "muscriptor"
