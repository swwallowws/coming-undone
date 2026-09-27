"""Latency past half a 16th: the grid alone folds it (a track 195 ms late reads as
+63 ms at 114 BPM), so each track is also lined up against its own stem's onsets."""
import numpy as np
import pretty_midi
import pytest

from stemscribe import grid as G

BPM = 114.0
SIX = 60.0 / BPM / 4
BAR = 16 * SIX
ANCHOR = 0.8
SR = 22050
LATE = 0.195                     # MuScriptor vs the audio in rearranged's donor build


def _irregular(n_bars=24, seed=0):
    """Onsets on a random half of the 16ths: no period a lag search could slip onto."""
    rng = np.random.default_rng(seed)
    k = np.flatnonzero(rng.random(n_bars * 16) < 0.5)
    return ANCHOR + k * SIX


def _clicks(times, seconds, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(int(SR * seconds)).astype(np.float32) * 1e-4
    burst = (rng.standard_normal(400) * np.exp(-np.arange(400) / 60)).astype(np.float32) * 0.5
    for t in times:
        i = int(round(t * SR))
        y[i:i + 400] += burst[: len(y[i:i + 400])]
    return y


def _env(times, seconds):
    return G.onset_envelope(_clicks(times, seconds), SR)


def test_audio_lag_finds_a_constant_offset_past_half_a_sixteenth():
    audio = _irregular()
    lag, score = G.audio_lag(audio + LATE, *_env(audio, 24 * BAR + 2))
    assert lag == pytest.approx(LATE, abs=0.012) and score >= G.LAG_MIN_SCORE


def test_audio_lag_on_unrelated_audio_has_a_low_score():
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(int(SR * (24 * BAR + 2))).astype(np.float32) * 0.1
    _, score = G.audio_lag(_irregular() + LATE, *G.onset_envelope(noise, SR))
    assert score < G.LAG_MIN_SCORE


def test_latency_unfolds_toward_a_hint():
    g = G.Grid(BPM, ANCHOR)
    late = _irregular() + LATE
    assert G.latency(late, g) == pytest.approx(LATE - SIX, abs=2e-3)       # folded
    assert G.latency(late, g, hint=0.18) == pytest.approx(LATE, abs=2e-3)


def _song():
    """comping on time (it sets the grid), bass 195 ms late against its own audio."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    comp = pretty_midi.Instrument(0, name="comping")
    prog = [(60, 64, 67), (57, 60, 64), (53, 57, 60), (55, 59, 62)]
    comp_on = [ANCHOR + b * BAR + beat * 4 * SIX for b in range(24) for beat in range(4)]
    for i, t in enumerate(comp_on):
        for p in prog[(i // 4) % 4]:
            comp.notes.append(pretty_midi.Note(70, p, t, t + 4 * SIX * 0.9))
    bass_true = _irregular(seed=2)
    jitter = np.random.default_rng(3).uniform(-0.006, 0.006, len(bass_true))   # looser than the comp
    bass = pretty_midi.Instrument(33, name="bass")
    bass.notes = [pretty_midi.Note(90, 40, t + LATE + j, t + LATE + j + 0.1)
                  for t, j in zip(bass_true, jitter)]
    pm.instruments += [comp, bass]
    seconds = 24 * BAR + 2
    return pm, {"comping": _env(comp_on, seconds), "bass": _env(bass_true, seconds)}, bass_true


def test_apply_with_audio_removes_the_whole_offset_when_snapping():
    pm, audio, bass_true = _song()
    _, folded, _ = G.apply(pm, BPM, snap_notes=True)
    assert folded["tracks"]["bass"]["latency_ms"] == pytest.approx((LATE - SIX) * 1000, abs=5)

    out, info, _ = G.apply(pm, BPM, snap_notes=True, audio=audio)
    assert info["tracks"]["bass"]["latency_ms"] == pytest.approx(LATE * 1000, abs=5)
    assert info["tracks"]["bass"]["audio_lag_ms"] == pytest.approx(LATE * 1000, abs=15)
    snapped = sorted(n.start for n in out.instruments[1].notes)
    assert np.max(np.abs(np.array(snapped) - bass_true)) < 1e-3       # back on its own 16ths


def test_apply_ignores_audio_that_does_not_match_the_notes():
    pm, audio, _ = _song()
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(int(SR * (24 * BAR + 2))).astype(np.float32) * 0.1
    audio["bass"] = G.onset_envelope(noise, SR)
    _, info, _ = G.apply(pm, BPM, audio=audio)
    assert info["tracks"]["bass"]["latency_ms"] == pytest.approx((LATE - SIX) * 1000, abs=5)
    assert "audio_lag_ms" not in info["tracks"]["bass"]
