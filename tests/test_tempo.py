import numpy as np
import pytest
import soundfile as sf

from stemscribe.tempo import (
    DEFAULT_TEMPO,
    ESTIMATORS,
    TempoCandidate,
    TempoEstimate,
    estimate_librosa,
    get_estimator,
    resolve_tempo,
)


def click_track(tmp_path, bpm, seconds=12.0, sr=22050):
    """A metronome at a known BPM -- ground truth a tracker should find."""
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    period = 60.0 / bpm
    click = np.exp(-np.linspace(0, 12, int(0.02 * sr))).astype(np.float32)
    for i in range(int(seconds / period)):
        at = int(i * period * sr)
        y[at : at + len(click)] += click
    path = tmp_path / f"click_{bpm}.wav"
    sf.write(str(path), y, sr)
    return path


def test_explicit_tempo_always_wins():
    est = resolve_tempo(92.0, stem_paths={"drums": "nonexistent.wav"})
    assert est.bpm == 92.0
    assert est.source == "user"
    assert est.method == "explicit"


def test_no_audio_falls_back_to_default():
    est = resolve_tempo(None, stem_paths={}, fallback_audio=None)
    assert est.bpm == DEFAULT_TEMPO
    assert est.source == "default"


def test_detection_failure_does_not_sink_the_run(tmp_path):
    """A whole demucs run must not be lost because beat tracking threw."""
    bad = tmp_path / "not-audio.wav"
    bad.write_text("this is not a wav file")
    est = resolve_tempo(None, stem_paths={"drums": bad})
    assert est.bpm == DEFAULT_TEMPO
    assert est.source == "default"
    assert est.method.startswith("failed:")


def test_prefers_drums_stem_over_full_mix(tmp_path):
    drums = click_track(tmp_path, 120)
    est = resolve_tempo(None, stem_paths={"drums": drums}, fallback_audio=drums)
    assert est.analyzed_stem == "drums"


def test_falls_back_to_mix_when_no_drums(tmp_path):
    mix = click_track(tmp_path, 120)
    est = resolve_tempo(None, stem_paths={"bass": mix}, fallback_audio=mix)
    assert est.analyzed_stem == "mix"


@pytest.mark.parametrize("bpm", [90, 92, 120, 140, 171])
def test_detects_click_track_accurately(tmp_path, bpm):
    """Regression refinement should land within 0.5% of a known click track.

    Tight on purpose: tempo sets the bar grid, and a few percent of error drifts
    the grid seconds away from the notes across a full song.
    """
    est = estimate_librosa(click_track(tmp_path, bpm, seconds=20.0))
    assert est.bpm == pytest.approx(bpm, rel=0.005), f"got {est.bpm} for {bpm}"
    assert est.method.endswith("regression")


def test_octave_alternates_are_offered(tmp_path):
    """Trackers make octave errors; half/double must be reachable from the UI."""
    est = estimate_librosa(click_track(tmp_path, 90, seconds=20.0))
    ratios = {c.ratio for c in est.candidates}
    assert 1.0 in ratios
    assert 2.0 in ratios or 0.5 in ratios


def test_candidates_are_scored_and_sorted(tmp_path):
    est = estimate_librosa(click_track(tmp_path, 120))
    scores = [c.score for c in est.candidates]
    assert scores == sorted(scores, reverse=True)
    assert max(scores) == pytest.approx(1.0)


def test_candidates_stay_in_musical_range(tmp_path):
    est = estimate_librosa(click_track(tmp_path, 120))
    assert all(50.0 <= c.bpm <= 200.0 for c in est.candidates)


def test_silence_does_not_crash(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros(22050 * 3, dtype=np.float32), 22050)
    est = estimate_librosa(path)
    assert est.bpm == DEFAULT_TEMPO
    assert est.source == "default"


def test_estimate_serializes_for_manifest(tmp_path):
    import json

    from stemscribe.core import _json_default

    est = estimate_librosa(click_track(tmp_path, 120))
    payload = json.loads(json.dumps(est.as_dict(), default=_json_default))
    assert payload["source"] == "detected"
    assert isinstance(payload["candidates"], list)


def test_registry_exposes_librosa_and_rejects_unknown():
    assert "librosa" in ESTIMATORS
    assert callable(get_estimator("librosa"))
    with pytest.raises(ValueError, match="unknown tempo estimator"):
        get_estimator("beat-this")
