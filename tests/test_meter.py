"""Meters other than 4/4: parsing, the pulse, bar "one", and stamping. 4/4 stays as it was."""
import numpy as np
import pretty_midi
import pytest

from stemscribe import grid as G
from stemscribe.grid import Meter


def test_parse_defaults():
    assert Meter.parse("4/4") == Meter(4, 4, (1, 1, 1, 1))
    assert Meter.parse("4/4") == G.DEFAULT
    assert Meter.parse("3/4").groups == (1, 1, 1)
    assert Meter.parse("6/8").groups == (3, 3)
    assert Meter.parse("9/8").groups == (2, 2, 2, 3)
    assert Meter.parse("12/8").groups == (3, 3, 3, 3)
    assert Meter.parse("9/8:3+3+3").groups == (3, 3, 3)


def test_default_grouping_rule_matches_its_docstring():
    """The reference rule rearranged adopts (grid._default_groups)."""
    want = {"6/4": (1,) * 6, "2/2": (1, 1), "1/8": (1,), "3/8": (3,), "2/8": (2,),
            "4/8": (2, 2), "8/8": (2, 2, 2, 2), "5/8": (2, 3), "7/8": (2, 2, 3),
            "11/8": (2, 2, 2, 2, 3), "9/16": (2, 2, 2, 3), "15/8": (3,) * 5}
    for spec, groups in want.items():
        assert Meter.parse(spec).groups == groups, spec


def test_parse_rejects_bad_groups():
    with pytest.raises(ValueError):
        Meter.parse("9/8:2+2+2")          # groups must sum to 9
    for bad in ("4", "x/4", "4/5", "0/4", "9/8:2+x"):
        with pytest.raises(ValueError):
            Meter.parse(bad)


def test_accents_pulses_and_text():
    m = Meter.parse("9/8")
    assert m.accents == (0, 2, 4, 6) and m.pulses == 9
    assert str(m) == "9/8" and str(Meter.parse("9/8:3+3+3")) == "9/8:3+3+3"
    assert Meter.parse(str(Meter.parse("7/8:3+2+2"))) == Meter.parse("7/8:3+2+2")


def test_grid_bar_is_pulses_times_the_pulse():
    g = G.Grid(180.0, 0.0, Meter.parse("6/8"))
    assert g.beat == pytest.approx(60 / 180) and g.bar == pytest.approx(6 * 60 / 180)
    assert g.as_dict()["meter"] == "6/8"
    assert "meter" not in G.Grid(120.0, 0.0).as_dict()        # 4/4 manifests unchanged


def _song(meter: Meter, bpm_pulse: float, bars: int, pickup_pulses: int):
    """A note on every pulse: loudest (and lowest) on the downbeat, louder on group
    starts, starting after a pickup. Every note is a C, so harmony says nothing and
    the accents have to find bar "one"."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name="comping")
    pulse = 60.0 / bpm_pulse
    t0 = pickup_pulses * pulse
    for b in range(bars):
        for k in range(meter.pulses):
            start = t0 + (b * meter.pulses + k) * pulse
            vel, pitch = (110, 48) if k == 0 else (70, 60) if k in meter.accents else (50, 60)
            inst.notes.append(pretty_midi.Note(vel, pitch, start, start + pulse * 0.9))
    pm.instruments.append(inst)
    return pm, t0


@pytest.mark.parametrize("spec, tracked", [
    ("3/4", 180.0),       # /4: the pulse is the tracked beat
    ("6/8", 60.0),        # tracked the dotted quarter: pulse = beat / 3
    ("6/8", 90.0),        # tracked the quarter: pulse = beat / 2
    ("9/8", 90.0),
    ("9/8", 180.0),       # tracked the eighth itself: pulse = beat
    ("12/8", 60.0),
])
def test_bar_one_found_in_meter(spec, tracked):
    meter = Meter.parse(spec)
    pm, t0 = _song(meter, bpm_pulse=180.0, bars=24, pickup_pulses=1)
    out, info, _ = G.apply(pm, tracked, meter=meter)
    assert info["fitted"] and info["bpm"] == pytest.approx(180.0, rel=1e-3)
    g = G.Grid(info["bpm"], info["anchor"], meter)
    off = (info["first_bar"] - t0) % g.bar
    assert min(off, g.bar - off) < 0.02
    assert info["meter"] == spec
    ts = out.time_signature_changes[0]
    assert (ts.numerator, ts.denominator) == (meter.num, meter.den)


def test_eighth_tracker_with_some_sixteenths_keeps_the_eighth_pulse():
    """A slow 6/8 ballad: a note on every eighth (180) plus 16ths on 30% of the off
    positions, and the tracker reported the eighth. The 16th grid fits tighter, but it
    is not a 16th-pulse song: the pulse stays the eighth and the bar six of them."""
    meter = Meter.parse("6/8")
    rng = np.random.default_rng(3)
    pulse = 60.0 / 180
    inst = pretty_midi.Instrument(0, name="comping")
    for k in range(24 * 6):
        t = k * pulse
        inst.notes.append(pretty_midi.Note(110 if k % 6 == 0 else 70, 48 if k % 6 == 0 else 60,
                                           t, t + pulse * 0.4))
        if rng.random() < 0.3:           # a C too: this test is about the pulse, not harmony
            inst.notes.append(pretty_midi.Note(60, 72, t + pulse / 2, t + pulse * 0.9))
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(inst)
    _, info, _ = G.apply(pm, 180.0, meter=meter)
    assert info["bpm"] == pytest.approx(180.0, rel=1e-3)
    g = G.Grid(info["bpm"], info["anchor"], meter)
    assert g.bar == pytest.approx(2.0, rel=1e-3)
    off = info["first_bar"] % g.bar
    assert min(off, g.bar - off) < 0.02


def test_a_real_sixteenth_pulse_still_wins():
    """The tracker reported a quarter (90) and a note sits on every eighth: the finer
    pulse's off positions are all filled, so beat/2 still wins."""
    meter = Meter.parse("6/8")
    pm, t0 = _song(meter, bpm_pulse=180.0, bars=24, pickup_pulses=0)
    _, info, _ = G.apply(pm, 90.0, meter=meter)
    assert info["bpm"] == pytest.approx(180.0, rel=1e-3)


AUG = [(60, 64, 68), (61, 65, 69), (62, 66, 70), (63, 67, 71)]   # four disjoint triads


def _harmony_song(meter: Meter, pickup_pulses: int, per: str, bars: int = 24):
    """Every pulse plays the current chord at one velocity, so loudness says nothing.
    per="bar": the chord changes on each bar line. per="group": it changes on every
    group start (the triads share no pitch class, so every change is equally big)."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name="comping")
    pulse = 60.0 / 180.0
    t0 = pickup_pulses * pulse
    n = 0
    for b in range(bars):
        for k in range(meter.pulses):
            if k == 0 if per == "bar" else k in meter.accents:
                n += 1
            start = t0 + (b * meter.pulses + k) * pulse
            for p in AUG[n % 4]:
                inst.notes.append(pretty_midi.Note(80, p, start, start + pulse * 0.9))
    pm.instruments.append(inst)
    return pm, t0


@pytest.mark.parametrize("spec, tracked, pickup, per", [
    ("6/8", 60.0, 0, "bar"), ("6/8", 60.0, 2, "bar"), ("6/8", 90.0, 2, "bar"),
    ("9/8", 90.0, 0, "bar"), ("9/8", 90.0, 2, "bar"),
    ("9/8", 90.0, 0, "group"), ("9/8", 90.0, 2, "group"),     # only the 2+2+2+3 shape tells
])
def test_bar_one_from_harmony_and_groups_at_even_velocity(spec, tracked, pickup, per):
    meter = Meter.parse(spec)
    pm, t0 = _harmony_song(meter, pickup, per)
    _, info, _ = G.apply(pm, tracked, meter=meter)
    assert info["bpm"] == pytest.approx(180.0, rel=1e-3)
    g = G.Grid(info["bpm"], info["anchor"], meter)
    off = (info["first_bar"] - t0) % g.bar
    assert min(off, g.bar - off) < 0.02
    assert info["bar_one_confidence"] > 0.05


@pytest.mark.parametrize("spec", ["3/4", "6/8", "9/8", "7/8"])
def test_stamp_puts_bar_lines_of_the_meter_and_never_moves_a_note(tmp_path, spec):
    meter = Meter.parse(spec)
    g = G.Grid(180.0, 0.7, meter)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name="x")
    inst.notes = [pretty_midi.Note(80, 60, 0.05 + 0.37 * i, 0.3 + 0.37 * i) for i in range(60)]
    pm.instruments.append(inst)
    p = tmp_path / "o.mid"
    G.stamp(pm, g).write(str(p))
    back = pretty_midi.PrettyMIDI(str(p))
    before = np.array([n.start for n in inst.notes])
    after = np.array([n.start for n in back.instruments[0].notes])
    assert np.max(np.abs(before - after)) < 1e-3
    first = back.get_downbeats()[1]
    assert first == pytest.approx(0.7 if 0.7 >= g.bar / 2 else 0.7 + g.bar, abs=1e-3)
    ts = back.time_signature_changes[0]
    assert (ts.numerator, ts.denominator) == (meter.num, meter.den)
    # the MIDI tempo counts quarter notes: an eighth pulse at 180 is 90 BPM
    assert back.get_tempo_changes()[1][-1] == pytest.approx(180.0 * 4 / meter.den, rel=1e-4)


def test_downbeat_and_shift_count_pulses_of_the_meter():
    meter = Meter.parse("6/8")
    pm, _ = _song(meter, 180.0, 24, 1)
    _, a, _ = G.apply(pm, 60.0, meter=meter)
    _, b, _ = G.apply(pm, 60.0, meter=meter, downbeat=6)
    assert b["anchor"] - a["anchor"] == pytest.approx(5 * 60 / a["bpm"], abs=1e-6)
    with pytest.raises(ValueError, match="1 to 6"):
        G.apply(pm, 60.0, meter=meter, downbeat=7)


# --- 4/4 stays exactly as it was -----------------------------------------------
BPM = 114.0
SIX = 60.0 / BPM / 4
BAR = 16 * SIX
ANCHOR = 0.8


def _onsets(n, every=2, lat_ms=0.0, jitter_ms=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = ANCHOR + np.arange(n) * every * SIX + lat_ms / 1000 + rng.uniform(-jitter_ms, jitter_ms, n) / 1000
    return [float(x) for x in t if x >= 0]


def _four_four_song():
    """test_grid's _song: a chord per bar with a stab on every beat, plus a late bass."""
    prog = [(60, 64, 67), (57, 60, 64), (53, 57, 60), (55, 59, 62)]
    comp = pretty_midi.Instrument(0, name="comping")
    for b in range(24):
        for beat in range(4):
            t = ANCHOR + b * BAR + beat * 4 * SIX
            for p in prog[b % 4]:
                comp.notes.append(pretty_midi.Note(70, p, t, t + 4 * SIX * 0.9))
    bass = pretty_midi.Instrument(33, name="bass")
    bass.notes = [pretty_midi.Note(90, 40, t, t + 0.2) for t in _onsets(180, lat_ms=25, jitter_ms=8)]
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.instruments = [comp, bass]
    return pm


@pytest.mark.parametrize("guess", [BPM * 1.01, None])
def test_four_four_explicit_meter_matches_the_old_result(guess, tmp_path):
    """test_grid.test_apply_fits_phases_and_snaps with meter=4/4 spelled out: same
    values, and the same info and MIDI bytes as the call without a meter."""
    out, info, warnings = G.apply(_four_four_song(), guess, snap_notes=True, meter=Meter.parse("4/4"))
    assert info["fitted"] and info["bpm"] == pytest.approx(BPM, rel=1e-3)
    assert info["source_track"] == "comping"
    k = (info["anchor"] - ANCHOR) / BAR
    assert abs(k - round(k)) < 1e-3
    assert info["tracks"]["bass"]["latency_ms"] == pytest.approx(25, abs=4)
    assert "meter" not in info
    old_out, old_info, old_warnings = G.apply(_four_four_song(), guess, snap_notes=True)
    assert info == old_info and warnings == old_warnings
    out.write(str(tmp_path / "a.mid"))
    old_out.write(str(tmp_path / "b.mid"))
    assert (tmp_path / "a.mid").read_bytes() == (tmp_path / "b.mid").read_bytes()
    ts = out.time_signature_changes[0]
    assert (ts.numerator, ts.denominator) == (4, 4)


def test_four_four_downbeat_bounds_and_message_unchanged():
    with pytest.raises(ValueError, match="downbeat must be 1 to 4, got 5"):
        G.apply(_four_four_song(), BPM, downbeat=5)


def test_sparse_bars_use_the_meter():
    from stemscribe.backends import is_sparse
    # 60 s at 120 BPM: 30 bars of 4/4 (under 7.5 notes is sparse), 40 bars of 3/4 (under 10)
    assert is_sparse(7, 60.0, 120.0, 0.1) and not is_sparse(8, 60.0, 120.0, 0.1)
    assert is_sparse(9, 60.0, 120.0, 0.1, beats_per_bar=3.0)


def test_quantization_grid_is_per_pulse():
    from stemscribe.merge import quantization_error
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(80, 60, 0.0625 * i, 0.0625 * i + 0.05) for i in range(8)]
    assert quantization_error(inst, tempo=120.0)["grid"] == "1/16"
    q8 = quantization_error(inst, tempo=120.0, den=8)
    assert q8["grid"] == "1/32" and q8["mean"] == 0.0     # 32nds at 120: every 0.0625 s


def test_cli_meter_flag():
    from stemscribe import cli
    a = cli.build_parser().parse_args(["x.mp3", "-o", "out"])
    assert a.meter == G.DEFAULT
    a = cli.build_parser().parse_args(["x.mp3", "-o", "out", "--meter", "9/8:3+3+3", "--downbeat", "7"])
    assert a.meter.groups == (3, 3, 3) and a.downbeat == 7
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["x.mp3", "-o", "out", "--meter", "9/8:2+2"])
    with pytest.raises(SystemExit):                          # 4/4 has no beat 5
        cli.main(["x.mp3", "-o", "out", "--downbeat", "5"])
    with pytest.raises(SystemExit):
        cli.grid_main(["x.mid", "-o", "out.mid", "--meter", "6/8", "--downbeat", "7"])


def test_web_bar_shift_bounds_follow_the_meter(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from stemscribe.web import server as webapp
    monkeypatch.setattr(webapp, "JOBS_ROOT", tmp_path / "jobs")
    webapp.JOBS.clear()
    client = TestClient(webapp.app)
    job = webapp.Job(id="j1", dir=tmp_path / "j1", name="song.mp3", status="done")
    (job.dir / "out").mkdir(parents=True)
    webapp.JOBS["j1"] = job
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name="comping")
    inst.notes.append(pretty_midi.Note(90, 60, 1.0, 20.0))
    pm.instruments.append(inst)
    meter = Meter.parse("6/8")
    G.stamp(pm, G.Grid(180.0, 1.5, meter)).write(str(job.dir / "out" / "song.mid"))
    job.result = {"midi": "out/song.mid", "tempo": {"bpm": 180.0, "source": "detected"},
                  "grid": {"fitted": True, "bpm": 180.0, "anchor": 1.5, "first_bar": 1.5, "meter": "6/8"}}
    assert client.post("/api/jobs/j1/bar", data={"beats": 6}).status_code == 400
    assert client.post("/api/jobs/j1/bar", data={"beats": 5}).status_code == 200
    assert job.result["grid"]["anchor"] == pytest.approx(1.5 + 5 * 60 / 180)
    after = pretty_midi.PrettyMIDI(str(job.dir / "out" / "song.mid"))
    ts = after.time_signature_changes[0]
    assert (ts.numerator, ts.denominator) == (6, 8)
