"""Mixdown tests.

Regression: the mix was hard-clipping at 1.0. The first fix *looked* right --
alimiter was in the chain and ffmpeg exited 0 -- but alimiter's `level` option
defaults to true and auto-levels the output back to 0dBFS, undoing `limit`. A
test that only checked "ffmpeg succeeded" would have passed the whole time.
So these assert the actual sample values.
"""
import numpy as np
import pytest
import soundfile as sf

from stemscribe.mixdown import MixdownError, mix_instrumental

SR = 44100


def loud_stems(tmp_path, n=3, amp=0.6, seconds=1.0):
    """Stems that sum well past full scale, like a real near-0dBFS mix does."""
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    paths = {}
    for i, name in enumerate(["drums", "bass", "other"][:n]):
        y = (amp * np.sin(2 * np.pi * (110 * (i + 1)) * t)).astype(np.float32)
        p = tmp_path / f"{name}.wav"
        sf.write(str(p), np.stack([y, y], axis=1), SR)
        paths[name] = p
    # vocals must be excluded from the mix, so make it obvious if it leaks in
    v = (0.9 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    vp = tmp_path / "vocals.wav"
    sf.write(str(vp), np.stack([v, v], axis=1), SR)
    paths["vocals"] = vp
    return paths


def peak(path):
    d, _ = sf.read(str(path))
    return float(np.abs(d).max())


def test_limiter_actually_limits(tmp_path):
    """The bug: peak came back exactly 1.0 because alimiter auto-leveled."""
    stems = loud_stems(tmp_path)
    out = mix_instrumental(stems, tmp_path / "inst.wav", limit=0.97)
    p = peak(out)
    assert p <= 0.9701, f"peak {p} exceeds the limit"
    assert p != pytest.approx(1.0, abs=0.001), "auto-level defeated the limiter again"


def test_limit_value_is_respected(tmp_path):
    stems = loud_stems(tmp_path)
    out = mix_instrumental(stems, tmp_path / "inst.wav", limit=0.5)
    assert peak(out) <= 0.5001


def test_limit_zero_opts_out(tmp_path):
    stems = loud_stems(tmp_path)
    out = mix_instrumental(stems, tmp_path / "inst.wav", limit=0)
    # Raw sum clips against the container's full scale.
    assert peak(out) == pytest.approx(1.0, abs=0.001)


def test_vocals_are_excluded(tmp_path):
    """The acceptance criterion: vocals absent from the instrumental."""
    stems = loud_stems(tmp_path)
    out = mix_instrumental(stems, tmp_path / "inst.wav", limit=0.97)
    d, sr = sf.read(str(out))
    mono = d.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    # the vocals stem is a lone 3kHz tone; it must not appear in the mix
    band = spec[(freqs > 2900) & (freqs < 3100)].max()
    assert band < spec.max() * 0.01, "vocals leaked into the instrumental"


def test_excluding_everything_raises(tmp_path):
    stems = loud_stems(tmp_path, n=0)
    with pytest.raises(MixdownError, match="no stems left"):
        mix_instrumental(stems, tmp_path / "inst.wav", exclude=("vocals",))


def test_mixes_arbitrary_stem_counts(tmp_path):
    stems = loud_stems(tmp_path, n=2)
    out = mix_instrumental(stems, tmp_path / "inst.wav", limit=0.97)
    assert peak(out) <= 0.9701
