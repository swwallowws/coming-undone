"""The tempo map: beat times that follow a band that drifts, bar lines of the meter
across it, and the constant grid kept (byte for byte) when one tempo fits."""
import hashlib

import numpy as np
import pretty_midi
import pytest

from stemscribe import cli
from stemscribe import grid as G
from stemscribe.grid import Meter

PROG = [(60, 64, 67), (57, 60, 64), (53, 57, 60), (55, 59, 62)]


def drifting_song(spec="4/4", bpm0=96.0, bpm1=102.0, bars=64, first_bar=0.9,
                  bass_lat=0.0, seed=0):
    """A band whose pulse tempo moves from bpm0 to bpm1, pulse by pulse, over `bars`.

    drums: hats on every half pulse, kick on group starts (on beats 1 and 3 in 4/4), snare
    elsewhere, 3 ms jitter (the tightest track). comping: a chord per bar, one stab per
    pulse, 6 ms jitter. bass: the root on group starts, 8 ms jitter plus bass_lat.
    Returns (pm, the pulse times, the true bar lines)."""
    meter = Meter.parse(spec)
    rng = np.random.default_rng(seed)
    P = meter.pulses
    n = bars * P
    bpm = np.linspace(bpm0, bpm1, n)
    beats = first_bar + np.r_[0.0, np.cumsum(60.0 / bpm)]
    simple = all(x == 1 for x in meter.groups)
    kicks = {0, 2} if simple and P == 4 else set(meter.accents)

    def j(ms):
        return rng.uniform(-ms, ms) / 1000

    drums = pretty_midi.Instrument(0, name="drums", is_drum=True)
    comp = pretty_midi.Instrument(0, name="comping")
    bass = pretty_midi.Instrument(33, name="bass")
    for k in range(n):
        b, per = beats[k], beats[k + 1] - beats[k]
        pos = k % P
        chord = PROG[(k // P) % 4]
        for half in (0.0, 0.5):
            t = b + half * per + j(3)
            drums.notes.append(pretty_midi.Note(70, 42, t, t + 0.05))
        t = b + j(3)
        if pos in kicks:
            drums.notes.append(pretty_midi.Note(120 if pos == 0 else 95, 36, t, t + 0.05))
        else:
            drums.notes.append(pretty_midi.Note(90, 38, t, t + 0.05))
        t = b + j(6)
        for p in chord:
            comp.notes.append(pretty_midi.Note(70, p, t, t + per * 0.9))
        if pos in meter.accents:
            t = b + bass_lat + j(8)
            bass.notes.append(pretty_midi.Note(90, chord[0] - 24, t, t + per * 0.8))
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.instruments = [drums, comp, bass]
    return pm, beats, beats[::P]


def _bar_lines(info):
    m = G.TempoMap.from_info(info)
    return m.bar_lines()


def _close_to(found, truth, tol):
    """Every true bar line has a found bar line within tol, and vice versa inside the song."""
    found = np.asarray(found)
    lo, hi = truth[0] - 1e-6, truth[-1] + 1e-6
    inner = found[(found >= lo - tol) & (found <= hi + tol)]
    d_truth = np.min(np.abs(truth[:, None] - inner[None, :]), axis=1)
    d_found = np.min(np.abs(inner[:, None] - truth[None, :]), axis=1)
    return float(max(d_truth.max(), d_found.max()))


# --- the map follows the band ---------------------------------------------------
@pytest.mark.parametrize("spec, bpm0, bpm1, guess", [
    ("4/4", 96.0, 102.0, 99.0),
    ("9/8:2+2+2+3", 192.0, 204.0, 198.0),      # eighth pulse: quarter 96 to 102
])
def test_map_recovers_drifting_bar_lines(spec, bpm0, bpm1, guess):
    pm, beats, bars = drifting_song(spec, bpm0, bpm1)
    out, info, warnings = G.apply(pm, guess, meter=Meter.parse(spec))
    assert info["fitted"] and info["tempo"] == "map"
    assert info["source_track"] == "drums"
    # every bar line the notes can tell (the closing one, after the last note, cannot)
    assert _close_to(_bar_lines(info), bars[:-1], 0.5) < 0.012
    lo, hi = info["bpm_range"]
    assert lo == pytest.approx(bpm0, rel=0.02) and hi == pytest.approx(bpm1, rel=0.02)
    if spec != "4/4":
        assert info["meter"] == "9/8"


def test_map_drums_sit_tight_and_constant_does_not():
    pm, _, _ = drifting_song()
    _, m, _ = G.apply(pm, 99.0, tempo="map")
    _, c, _ = G.apply(pm, 99.0, tempo="constant")
    assert m["tracks"]["drums"]["offset_16th"] < 0.05
    # one tempo for a drifting band: no grid at all, or a loose one
    assert not c["fitted"] or c["tracks"]["drums"]["offset_16th"] > 0.15


def test_map_midi_has_tempo_changes_bar_lines_and_unmoved_notes(tmp_path):
    spec = "9/8:2+2+2+3"
    pm, _, bars = drifting_song(spec, 192.0, 204.0, bars=32)
    out, info, _ = G.apply(pm, 198.0, meter=Meter.parse(spec))
    p = tmp_path / "m.mid"
    out.write(str(p))
    back = pretty_midi.PrettyMIDI(str(p))
    times, qpm = back.get_tempo_changes()
    assert len(times) > 2 * 32         # a tempo per pulse (a repeat of the same tempo reads back as one)
    assert qpm[1:].min() == pytest.approx(96.0, rel=0.02) and qpm.max() <= 102.0 * 1.02
    db = back.get_downbeats()                          # db[0] is the pickup bar's start, 0 s
    # a bar line under half a bar in is the pickup's end; the file's bar lines stop at its
    # last note, before the closing bar line
    real = bars[(bars >= db[1] - 0.5) & (bars <= db[-1] + 0.5)]
    assert len(real) >= len(bars) - 2
    assert _close_to(db[1:], real, 0.5) < 0.012        # bar lines in the file follow the band
    ts = back.time_signature_changes[0]
    assert (ts.numerator, ts.denominator) == (9, 8)
    for a, b in zip(pm.instruments, back.instruments):
        x = np.array(sorted(n.start for n in a.notes))
        y = np.array(sorted(n.start for n in b.notes))
        assert np.max(np.abs(x - y)) < 1e-3


def test_snap_and_latency_work_against_the_map():
    pm, _, _ = drifting_song(bass_lat=0.025)
    out, info, _ = G.apply(pm, 99.0, tempo="map", snap_notes=True)
    assert info["tracks"]["bass"]["latency_ms"] == pytest.approx(25, abs=4)
    m = G.TempoMap.from_info(info)
    for inst in out.instruments:
        assert np.max(np.abs(m.signed([n.start for n in inst.notes]))) < 1e-3, inst.name


def test_downbeat_override_moves_the_map_bar_by_pulses():
    pm, _, _ = drifting_song(bars=24)
    _, a, _ = G.apply(pm, 99.0, tempo="map")
    _, b, _ = G.apply(pm, 99.0, tempo="map", downbeat=2)
    assert (b["first"] - a["first"]) % 4 == 1


# --- auto -------------------------------------------------------------------------
def test_auto_threshold_is_on_the_constant_fits_mean_offset():
    t = G.MAP_THRESHOLD
    assert 0.05 <= t <= 0.15
    assert G.use_map(None) and G.use_map(t + 0.01)
    assert not G.use_map(t - 0.01)


def test_auto_picks_constant_for_a_steady_song_and_map_for_a_drifting_one():
    steady, _, _ = drifting_song(bpm0=99.0, bpm1=99.0)
    drifting, _, _ = drifting_song()
    _, s, _ = G.apply(steady, 99.0)
    _, d, _ = G.apply(drifting, 99.0)
    assert s["tempo"] == "constant" and s["constant_offset_16th"] < G.MAP_THRESHOLD
    assert d["tempo"] == "map"


def test_steady_four_four_output_is_byte_identical_to_before_the_map(tmp_path):
    """test_grid's _song, fitted, stamped and snapped in auto: the same bytes the
    constant-only grid wrote (hash taken before the tempo map existed)."""
    from test_grid import _song
    for snap, want in ((False, BEFORE_PLAIN), (True, BEFORE_SNAPPED)):
        out, info, _ = G.apply(_song(), 114.0 * 1.01, snap_notes=snap)
        assert info["tempo"] == "constant"
        p = tmp_path / f"{snap}.mid"
        out.write(str(p))
        assert hashlib.sha256(p.read_bytes()).hexdigest() == want


# research/tempo-map/baseline_hash.py at f4e12a8 (the grid before the map)
BEFORE_PLAIN = "80543a1429e9e91c8ca49ad30b4b2dc935ecb26856fd227ec288d89457dc9c8b"
BEFORE_SNAPPED = "3460d251e331f1801b89e2d4ca3afde71e2b60021ef81616077d9ab4d1998915"


# --- the flag -------------------------------------------------------------------------
def test_tempo_mode_flag():
    a = cli.build_parser().parse_args(["x.mp3", "-o", "out"])
    assert a.tempo_mode == "auto" and a.tempo is None
    a = cli.build_parser().parse_args(["x.mp3", "-o", "out", "--tempo-mode", "map", "--tempo", "98"])
    assert a.tempo_mode == "map" and a.tempo == 98.0
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["x.mp3", "-o", "out", "--tempo-mode", "wobbly"])


def test_web_bar_shift_keeps_the_map(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from stemscribe.web import server as webapp
    monkeypatch.setattr(webapp, "JOBS_ROOT", tmp_path / "jobs")
    webapp.JOBS.clear()
    job = webapp.Job(id="j1", dir=tmp_path / "j1", name="song.mp3", status="done")
    (job.dir / "out").mkdir(parents=True)
    webapp.JOBS["j1"] = job
    pm, _, _ = drifting_song(bars=16)
    out, info, _ = G.apply(pm, 99.0, tempo="map")
    out.write(str(job.dir / "out" / "song.mid"))
    job.result = {"midi": "out/song.mid", "tempo": {"bpm": 99.0, "source": "detected"}, "grid": dict(info)}
    before = G.TempoMap.from_info(info).bar_lines()
    assert TestClient(webapp.app).post("/api/jobs/j1/bar", data={"beats": 1}).status_code == 200
    after = G.TempoMap.from_info(job.result["grid"])
    beats = np.asarray(info["beats"])
    assert np.allclose(after.bar_lines(), beats[(info["first"] + 1) % 4::4])
    assert not np.allclose(after.bar_lines()[:3], before[:3])
    back = pretty_midi.PrettyMIDI(str(job.dir / "out" / "song.mid"))
    assert len(back.get_tempo_changes()[0]) > 2 * 16


def test_grid_command_writes_a_map(tmp_path):
    pm, _, _ = drifting_song(bars=24)
    src = tmp_path / "in.mid"
    pm.write(str(src))
    out = tmp_path / "out.mid"
    assert cli.grid_main([str(src), "-o", str(out), "--tempo", "99", "--tempo-mode", "map"]) == 0
    back = pretty_midi.PrettyMIDI(str(out))
    assert len(back.get_tempo_changes()[0]) > 2 * 24
