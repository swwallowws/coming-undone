"""The beat grid: fit, bar phase, latency, snapping, and stamping it into the MIDI."""
import numpy as np
import pretty_midi
import pytest

from stemscribe import grid as G

BPM = 114.0
SIX = 60.0 / BPM / 4
BAR = 16 * SIX
ANCHOR = 0.8             # a real bar line, in seconds


def onsets(n=400, jitter_ms=0.0, lat_ms=0.0, seed=0, every=2, start_six=0):
    """Onsets on every `every`-th 16th from the anchor, with jitter and a fixed latency."""
    rng = np.random.default_rng(seed)
    t = ANCHOR + (start_six + np.arange(n) * every) * SIX
    t = t + lat_ms / 1000 + rng.uniform(-jitter_ms, jitter_ms, n) / 1000
    return [float(x) for x in t if x >= 0]


def test_fit_finds_tempo_and_a_sixteenth_point_from_a_close_guess():
    g, r = G.fit(onsets(), BPM * 1.02)
    assert g.bpm == pytest.approx(BPM, rel=1e-4)
    k = (ANCHOR - g.anchor) / SIX
    assert abs(k - round(k)) < 0.02 and r > 0.99


def test_search_finds_tempo_without_a_guess_and_prefers_the_faster_octave():
    g, r = G.search(onsets(every=2))                   # only 8ths: 57 and 114 both fit
    assert g.bpm == pytest.approx(BPM, rel=1e-3)


def test_the_tightest_track_sets_the_grid():
    tracks = {"comping": onsets(jitter_ms=1.0), "bass": onsets(jitter_ms=30.0, seed=1)}
    g, source, fits = G.fit_tracks(tracks, BPM)
    assert source == "comping" and fits["comping"] > fits["bass"]


def test_too_few_onsets_give_no_grid():
    assert G.fit_tracks({"bass": onsets(n=20)}, BPM) is None


def test_beat_alignment_puts_the_anchor_on_a_beat():
    # most notes on beats (every 4th 16th), a few on the "e"; start the fit off-beat
    notes = onsets(every=4) + onsets(n=40, every=16, start_six=1)
    g = G.Grid(BPM, ANCHOR + SIX)                       # a 16th-grid point that is not a beat
    aligned = G.align_to_beats(g, notes)
    k = (aligned.anchor - ANCHOR) / (4 * SIX)
    assert abs(k - round(k)) < 1e-6


def _chords(changes_on_bar=True, bars=24):
    """A comp track: one chord per bar starting on the bar line, held through the bar,
    with a repeated stab on every beat (so onsets alone cannot tell the bar)."""
    prog = [(60, 64, 67), (57, 60, 64), (53, 57, 60), (55, 59, 62)]
    inst = pretty_midi.Instrument(0, name="comping")
    for b in range(bars):
        for beat in range(4):
            t = ANCHOR + b * BAR + beat * 4 * SIX
            for p in prog[b % 4]:
                inst.notes.append(pretty_midi.Note(70, p, t, t + 4 * SIX * 0.9))
    return [inst]


def test_bar_phase_from_harmony_finds_where_chords_change():
    g = G.Grid(BPM, ANCHOR + 4 * SIX)                  # beat-aligned, but one beat late
    phased, conf = G.phase_from_notes(g, _chords())
    k = (phased.anchor - ANCHOR) / BAR
    assert abs(k - round(k)) < 1e-6 and conf > 0.5


def test_bar_phase_with_no_pitched_notes_has_no_confidence():
    g = G.Grid(BPM, ANCHOR)
    assert G.phase_from_notes(g, []) == (g, 0.0)


def test_shift_moves_the_bar_by_whole_beats():
    g = G.Grid(BPM, ANCHOR)
    assert G.shift(g, 1).anchor == pytest.approx(ANCHOR + 4 * SIX)
    assert G.shift(g, -1).anchor == pytest.approx(ANCHOR - 4 * SIX)


def test_latency_is_the_median_signed_offset():
    g = G.Grid(BPM, ANCHOR)
    assert G.latency(onsets(lat_ms=-41.0, jitter_ms=5.0), g) == pytest.approx(-0.041, abs=0.002)


def test_snap_removes_latency_then_snaps_starts_and_ends():
    g = G.Grid(BPM, ANCHOR)
    inst = pretty_midi.Instrument(0, name="drums", is_drum=False)
    for t in onsets(n=16, lat_ms=-41.0, jitter_ms=5.0):
        inst.notes.append(pretty_midi.Note(80, 60, t, t + SIX * 1.9))
    out, lat = G.snap(inst, g)
    assert lat == pytest.approx(-0.041, abs=0.003)
    for n, want in zip(sorted(out.notes, key=lambda n: n.start), onsets(n=16)):
        assert n.start == pytest.approx(want, abs=1e-6)
        k = (n.end - ANCHOR) / (SIX / 2)
        assert abs(k - round(k)) < 1e-6 and n.end > n.start


def test_snap_keeps_one_drum_hit_per_drum_per_sixteenth():
    g = G.Grid(BPM, ANCHOR)
    kit = pretty_midi.Instrument(0, name="drums", is_drum=True)
    kit.notes += [pretty_midi.Note(60, 36, ANCHOR - 0.01, ANCHOR + 0.05),
                  pretty_midi.Note(100, 36, ANCHOR + 0.01, ANCHOR + 0.05),
                  pretty_midi.Note(90, 42, ANCHOR, ANCHOR + 0.05)]
    out, _ = G.snap(kit, g, latency_s=0.0)
    assert sorted((n.pitch, n.velocity) for n in out.notes) == [(36, 100), (42, 90)]


def test_stamp_writes_real_bar_lines_and_never_moves_a_note(tmp_path):
    g = G.Grid(BPM, 1.5)                                # past half a bar: a short pickup
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(33, name="bass")
    inst.notes = [pretty_midi.Note(90, 40, t, t + 0.2) for t in onsets(n=50)]
    pm.instruments.append(inst)
    out = G.stamp(pm, g)
    p = tmp_path / "o.mid"
    out.write(str(p))
    back = pretty_midi.PrettyMIDI(str(p))
    before = [n.start for n in inst.notes]
    after = [n.start for n in back.instruments[0].notes]
    assert np.max(np.abs(np.array(before) - np.array(after))) < 1e-3
    downbeats = back.get_downbeats()
    assert downbeats[1] == pytest.approx(1.5, abs=1e-3)           # the pickup ends on a real bar line
    assert downbeats[2] == pytest.approx(1.5 + BAR, abs=1e-3)
    times, bpms = back.get_tempo_changes()
    assert bpms[-1] == pytest.approx(BPM, rel=1e-4)


def _song():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)   # a placeholder tempo map
    pm.instruments = _chords()
    bass = pretty_midi.Instrument(33, name="bass")
    bass.notes = [pretty_midi.Note(90, 40, t, t + 0.2) for t in onsets(n=180, lat_ms=25, jitter_ms=8)]
    pm.instruments.append(bass)
    return pm


@pytest.mark.parametrize("guess", [BPM * 1.01, None])
def test_apply_fits_phases_and_snaps(guess):
    out, info, warnings = G.apply(_song(), guess, snap_notes=True)
    assert info["fitted"] and info["bpm"] == pytest.approx(BPM, rel=1e-3)
    assert info["source_track"] == "comping"
    assert abs(((info["anchor"] - ANCHOR) / BAR) - round((info["anchor"] - ANCHOR) / BAR)) < 1e-3
    assert info["tracks"]["bass"]["latency_ms"] == pytest.approx(25, abs=4)
    g = G.Grid(info["bpm"], info["anchor"])
    assert np.max(np.abs(g.signed([n.start for n in out.instruments[1].notes]))) < 1e-3


def test_apply_downbeat_override_and_shift():
    _, a, _ = G.apply(_song(), BPM)
    _, b, _ = G.apply(_song(), BPM, downbeat=3)
    _, c, _ = G.apply(_song(), BPM, shift_beats=-1)
    beat = 60.0 / a["bpm"]
    assert b["anchor"] - a["anchor"] == pytest.approx(2 * beat, abs=1e-6)
    assert c["anchor"] - a["anchor"] == pytest.approx(-beat, abs=1e-6)
    with pytest.raises(ValueError, match="downbeat"):
        G.apply(_song(), BPM, downbeat=5)


def test_apply_without_a_grid_leaves_the_midi_alone():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name="x")
    inst.notes = [pretty_midi.Note(80, 60, 0.3 * i, 0.3 * i + 0.1) for i in range(10)]
    pm.instruments.append(inst)
    out, info, warnings = G.apply(pm, 120.0)
    assert out is pm and info == {"fitted": False} and warnings


def test_stamp_with_a_bar_line_near_zero_uses_a_long_pickup():
    g = G.Grid(BPM, 0.01)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(80, 60, 0.0, 10.0))
    pm.instruments.append(inst)
    out = G.stamp(pm, g)
    db = out.get_downbeats()
    assert db[1] == pytest.approx(0.01 + BAR, abs=1e-3)
