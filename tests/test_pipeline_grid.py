"""process() with the grid, the drum backend and the sparse-stem fallback, all faked
except prepare/cleanup/merge/grid."""
import json

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from stemscribe import backends, core
from stemscribe import grid as G

BPM = 114.0
BEAT = 60.0 / BPM
SR = 8000
SECONDS = 40.0


def _wav(path, loud=True):
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(int(SR * SECONDS)) * (0.1 if loud else 0.0)).astype(np.float32)
    sf.write(str(path), y, SR)
    return path


def _write(notes, out_mid, drum=False):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, is_drum=drum)
    inst.notes = [pretty_midi.Note(90, p, s, s + d) for p, s, d in notes]
    pm.instruments.append(inst)
    pm.write(str(out_mid))
    return out_mid


def fake_main(stem_wav, out_mid, **_):
    if stem_wav.stem == "bass":                      # nearly empty: the fallback must kick in
        return _write([(40, 1.0, 0.5), (40, 3.0, 0.5)], out_mid)
    prog = [(60, 64, 67), (57, 60, 64), (53, 57, 60), (55, 59, 62)]
    notes = [(p, 0.5 + b * 4 * BEAT + k * BEAT + 0.012, BEAT * 0.9)      # 12 ms late
             for b in range(int(SECONDS / (4 * BEAT)) - 1) for k in range(4) for p in prog[b % 4]]
    return _write(notes, out_mid)


def fake_fallback(stem_wav, out_mid, **_):
    return _write([(40, 0.5 + i * BEAT + 0.02, 0.3) for i in range(int(SECONDS / BEAT) - 2)], out_mid)


def fake_drums(stem_wav, out_mid, **_):
    # like ADT_STR on the reference song: 40 ms early and loose
    jitter = np.random.default_rng(1).uniform(-0.015, 0.015, int(SECONDS / BEAT))
    return _write([(36, 0.5 + i * BEAT - 0.04 + jitter[i], 0.05) for i in range(int(SECONDS / BEAT) - 2)],
                  out_mid, drum=True)


@pytest.fixture
def run(tmp_path, monkeypatch):
    def separate(audio, out, model_name=None, device=None):
        out.mkdir(parents=True, exist_ok=True)
        return {s: _wav(out / f"{s}.wav") for s in ("drums", "bass", "other", "vocals")}
    monkeypatch.setattr("stemscribe.separate.separate", separate)
    monkeypatch.setitem(backends.BACKENDS, "fake", fake_main)
    monkeypatch.setitem(backends.BACKENDS, "basic-pitch", fake_fallback)
    monkeypatch.setitem(backends.DRUM_BACKENDS, "fake-drums", fake_drums)
    monkeypatch.setattr(backends, "drums_available", lambda: True)

    def go(**kw):
        return core.process(_wav(tmp_path / "in.wav"), out_dir=tmp_path / "out", backend="fake",
                            include_vocals_melody=False, instrumental=False, cache=False,
                            tempo=BPM * 1.01, drums="fake-drums",
                            prepare=core.PrepareParams(trim_silence=False), **kw)
    return go


def test_grid_drums_and_fallback_land_in_the_midi_and_manifest(run):
    res = run(snap=True)
    m = json.loads(res.manifest_path.read_text())
    assert m["grid"]["fitted"] and m["grid"]["bpm"] == pytest.approx(BPM, rel=1e-3)
    assert m["grid"]["source_track"] == "comping" and m["grid"]["snapped"] is True
    assert m["fallbacks"] == {"bass": "basic-pitch"}
    assert res.tempo.bpm == pytest.approx(BPM, rel=1e-3)
    pm = pretty_midi.PrettyMIDI(str(res.midi_path))
    by = {i.name: i for i in pm.instruments}
    assert by["drums"].is_drum and len(by["bass"].notes) > 50
    g = G.Grid(m["grid"]["bpm"], m["grid"]["anchor"])
    for name in ("comping", "bass", "drums"):
        assert np.max(np.abs(g.signed([n.start for n in by[name].notes]))) < 2e-3, name
    # latency is relative to the grid, which the comping (12 ms late) set: -40 - 12
    assert m["grid"]["tracks"]["drums"]["latency_ms"] == pytest.approx(-52, abs=5)


def test_a_failing_drum_model_warns_and_the_run_goes_on(run, monkeypatch):
    def broken(stem_wav, out_mid, **_):
        raise backends.BackendError("ADT_STR needs its model's requirements")
    monkeypatch.setitem(backends.DRUM_BACKENDS, "fake-drums", broken)
    res = run()
    assert any("drums" in w and "requirements" in w for w in res.warnings)
    assert "drums" not in {i.name for i in pretty_midi.PrettyMIDI(str(res.midi_path)).instruments}


def test_drums_are_available_only_with_the_model_requirements(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "transformers" else real(name, *a))
    assert backends.drums_available() is False


def _main_with_bass(bass):
    def main(stem_wav, out_mid, **kw):
        return bass(out_mid) if stem_wav.stem == "bass" else fake_main(stem_wav, out_mid, **kw)
    return main


@pytest.mark.parametrize("bass", [
    lambda out_mid: _write([], out_mid),     # a MIDI file with an empty track
    lambda out_mid: None,                    # no MIDI file at all
], ids=["zero-notes", "no-midi"])
def test_a_loud_stem_with_no_notes_falls_back(run, monkeypatch, bass):
    monkeypatch.setitem(backends.BACKENDS, "fake", _main_with_bass(bass))
    res = run()
    m = json.loads(res.manifest_path.read_text())
    assert m["fallbacks"] == {"bass": "basic-pitch"}
    by = {i.name: i for i in pretty_midi.PrettyMIDI(str(res.midi_path)).instruments}
    assert len(by["bass"].notes) > 50


def test_a_silent_stem_with_no_notes_says_why_it_was_not_redone(run, monkeypatch):
    def separate(audio, out, model_name=None, device=None):
        out.mkdir(parents=True, exist_ok=True)
        return {s: _wav(out / f"{s}.wav", loud=s != "bass") for s in ("drums", "bass", "other", "vocals")}
    monkeypatch.setattr("stemscribe.separate.separate", separate)
    monkeypatch.setitem(backends.BACKENDS, "fake", _main_with_bass(lambda out_mid: _write([], out_mid)))
    res = run()
    assert json.loads(res.manifest_path.read_text())["fallbacks"] == {}
    assert any("bass" in w and "silent" in w for w in res.warnings)


def test_no_snap_keeps_note_times_and_grid_can_be_turned_off(run):
    res = run()
    pm = pretty_midi.PrettyMIDI(str(res.midi_path))
    comp = next(i for i in pm.instruments if i.name == "comping")
    assert comp.notes[0].start == pytest.approx(0.512, abs=2e-3)       # 12 ms late, kept
    res = run(grid=False)
    assert json.loads(res.manifest_path.read_text())["grid"] == {"fitted": False, "disabled": True}
