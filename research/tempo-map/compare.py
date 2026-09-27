"""How well a MIDI's tracks sit on a constant grid against the tempo map, and which
subdivision fits. Everything in 16ths-of-a-step (0 exact, 0.25 random).

compare.py IN.mid OUT_DIR BPM_GUESS METER [START END] [--fixed]

--fixed: BPM_GUESS is the pulse (what stemscribe does with a --tempo you give it).

- constant: grid.apply(tempo="constant"); when no grid fits, the relaxed pulse grid
  (fit_pulse with min_alignment 0), which is what one tempo would give.
- map at several beat-tracker tightnesses (MAP_TIGHTNESS is the default), and the
  default map with its pulse lengths median-smoothed over 5 and 9 pulses (the "light
  smoothing" question). report-variants-v1.json kept the dropped methods' numbers.
- subdivisions against the default map: 4 per pulse (16ths in 4/4), 3 per pulse (8th
  triplets, the 12/8 reading of a 4/4 beat), 6 per pulse, 2 per pulse.
- bar lines: per pulse of the bar, kick hits (35/36) and bass onsets, and how often the
  harmony changes there.
Writes OUT_DIR/constant.mid and OUT_DIR/map.mid (grid.apply, no snapping).
"""
import json
import pathlib
import sys

import numpy as np
import pretty_midi
from scipy.ndimage import median_filter

from stemscribe import grid as G

argv = [a for a in sys.argv[1:] if a != "--fixed"]
FIXED = "--fixed" in sys.argv
src, out_dir = argv[0], pathlib.Path(argv[1])
guess, meter = float(argv[2]), G.Meter.parse(argv[3])
window = (float(argv[4]), float(argv[5])) if len(argv) > 5 else None
out_dir.mkdir(parents=True, exist_ok=True)

pm = pretty_midi.PrettyMIDI(src)
if window:
    a, b = window
    for inst in pm.instruments:
        inst.notes = [pretty_midi.Note(n.velocity, n.pitch, n.start - a, n.end - a)
                      for n in inst.notes if a <= n.start < b]
    pm.instruments = [i for i in pm.instruments if i.notes]


def starts(inst):
    return np.array([n.start for n in inst.notes])


def off(t, g, per=4):
    """mean distance to the nearest step, with `per` steps per pulse, in steps."""
    if isinstance(g, G.TempoMap):
        q = g.position(t) * per
    else:
        q = (np.asarray(t) - g.anchor) / g.beat * per
    return float(np.mean(np.abs(q - np.round(q))))


def ms(t, g):
    return float(np.median(np.abs(g.signed(t))) * 1000)


report = {"src": src, "window": window, "meter": str(meter)}

# --- constant --------------------------------------------------------------------
cpm, cinfo, cwarn = G.apply(pm, guess, meter=meter, tempo="constant", fixed_pulse=FIXED)
if cinfo["fitted"]:
    cg = G.Grid(cinfo["bpm"], cinfo["anchor"], meter)
    cpm.write(str(out_dir / "constant.mid"))
else:
    tracks = {i.name: [n.start for n in i.notes] for i in pm.instruments}
    cg = G.fit_pulse(tracks, guess, meter, min_alignment=0.0, fixed=FIXED)[0]
report["constant"] = {"fitted": cinfo["fitted"], "bpm": round(cg.bpm, 3), "warnings": cwarn,
                      "tracks": {i.name: round(off(starts(i), cg), 3) for i in pm.instruments}}

# --- map, widths -------------------------------------------------------------------
mpm, minfo, mwarn = G.apply(pm, guess, meter=meter, tempo="map", fixed_pulse=FIXED)
mpm.write(str(out_dir / "map.mid"))
m = G.TempoMap.from_info(minfo)
report["map"] = {k: minfo[k] for k in ("bpm", "bpm_range", "first_bar", "source_track",
                                       "bar_one_confidence")}
report["map"]["warnings"] = mwarn
report["map"]["tracks"] = {i.name: {"off": round(off(starts(i), m), 3), "ms": round(ms(starts(i), m), 1)}
                           for i in pm.instruments}
# the same with each track's own latency removed (median signed offset, as grid.latency):
# a transcription running 30 ms late reads as loose otherwise
lat_c = {i.name: G.latency(starts(i), cg) for i in pm.instruments}
lat_m = {i.name: minfo["tracks"][i.name]["latency_ms"] / 1000 for i in pm.instruments}
report["latency_removed"] = {
    "constant": {i.name: round(off(starts(i) - lat_c[i.name], cg), 3) for i in pm.instruments},
    "map": {i.name: round(off(starts(i) - lat_m[i.name], m), 3) for i in pm.instruments},
    "map_latency_ms": {k: round(v * 1000, 1) for k, v in lat_m.items()}}
# the tempo over time: pulse BPM per 8 bars, from a straight line through their pulses
bl = m.bar_lines()
last = max(n.end for i in pm.instruments for n in i.notes)
report["tempo_per_8_bars"] = []
for k in range(0, len(bl) - 8, 8):
    seg = m.beats[(m.beats >= bl[k]) & (m.beats < bl[k + 8])]
    if 0 <= bl[k] < last and len(seg) > 2:
        report["tempo_per_8_bars"].append(
            [round(float(bl[k]), 1), round(float(60 / np.polyfit(np.arange(len(seg)), seg, 1)[0]), 2)])
src_track = minfo["source_track"]
src_on = [n.start for i in pm.instruments if i.name == src_track for n in i.notes]
all_on = [n.start for i in pm.instruments for n in i.notes]
end = max(n.end for i in pm.instruments for n in i.notes)
seed = G.fit_pulse({i.name: [n.start for n in i.notes] for i in pm.instruments}, guess, meter,
                   min_alignment=0.0, fixed=FIXED)[0]
src_inst = next(i for i in pm.instruments if i.name == src_track)
vel = [n.velocity / 127 for n in src_inst.notes]
variants = {f"tightness {t}": dict(tightness=t) for t in (100, 400, 1600)}
res = {}
for name, kw in variants.items():
    beats = G.fit_map(src_on, seed, end, all_times=all_on, weights=vel, **kw)
    mw, _ = G.phase_map(G.TempoMap(beats, 0, meter), pm.instruments)
    res[name] = {i.name: round(off(starts(i), mw), 3) for i in pm.instruments}
    lo, hi = mw.bpm_range()
    res[name]["bpm"] = f"{lo:.1f}-{hi:.1f}"
report["variants"] = res
smooth = {}
for k in (5, 9):         # pulse lengths median-smoothed, re-summed (the prior session's way)
    per = median_filter(np.diff(m.beats), size=k, mode="nearest")
    sm = np.r_[m.beats[0], m.beats[0] + np.cumsum(per)]
    sm += np.median(m.beats - sm)
    ms_ = G.TempoMap(sm, m.first, meter)
    smooth[f"lengths median {k}"] = {i.name: round(off(starts(i) - lat_m[i.name], ms_), 3)
                                     for i in pm.instruments}
for s in (0, 1, 2, 4, 8, 16):   # pulse times from a line through s pulses either side
    beats = G.fit_map(src_on, seed, end, all_times=all_on, weights=vel, smooth=s)
    ms_, _ = G.phase_map(G.TempoMap(beats, 0, meter), pm.instruments)
    smooth[f"times line, smooth {s}"] = {i.name: round(off(starts(i) - lat_m[i.name], ms_), 3)
                                         for i in pm.instruments}
report["smoothed_latency_removed"] = smooth

# held out: the map fitted on a random half of the source track's notes, measured on the
# other half (the source track measured against its own map flatters it)
rng = np.random.default_rng(0)
notes = list(src_inst.notes)
mask = rng.random(len(notes)) < 0.5
fit_half = [n for n, k in zip(notes, mask) if k]
test_half = np.array([n.start for n, k in zip(notes, mask) if not k])
held = {"constant": round(off(test_half, cg), 3)}
for s in (0, 1, 2, 4, 8, 16):          # 2 (MAP_SMOOTH) is the default
    beats = G.fit_map([n.start for n in fit_half], seed, end,
                      all_times=[t for t in all_on if t not in set(test_half)],
                      weights=[n.velocity / 127 for n in fit_half], smooth=s)
    held[f"map, smooth {s}"] = round(off(test_half, G.TempoMap(beats, 0, meter)), 3)
report["held_out_" + src_track] = held

# --- subdivisions -------------------------------------------------------------------
report["subdivisions"] = {str(per): {i.name: round(off(starts(i), m, per), 3) for i in pm.instruments}
                          for per in (2, 3, 4, 6)}
report["subdivisions_constant"] = {str(per): {i.name: round(off(starts(i), cg, per), 3)
                                              for i in pm.instruments} for per in (3, 4)}

# where in the beat notes start, in twelfths: 0 the beat, 3 and 9 the 16ths "e" and "a",
# 6 the "and", 4 and 8 the triplets. A straight 16th feel peaks at 0/3/6/9, a 12/8 or
# triplet feel at 0/4/8.
report["within_beat_twelfths"] = {
    i.name: np.bincount(np.round((m.position(starts(i)) % 1) * 12).astype(int) % 12, minlength=12).tolist()
    for i in pm.instruments}

# --- bar lines -----------------------------------------------------------------------
P = meter.pulses
slots = {}
for inst in pm.instruments:
    notes = [n for n in inst.notes if (not inst.is_drum) or n.pitch in (35, 36)]
    if not notes:
        continue
    pos = np.round(m.position([n.start for n in notes])).astype(int)
    k = (pos - m.first) % P
    near = np.abs(m.position([n.start for n in notes]) - np.round(m.position([n.start for n in notes]))) < 0.2
    slots[inst.name + (" kick" if inst.is_drum else "")] = np.bincount(k[near], minlength=P).tolist()
report["per_pulse_of_bar_from_one"] = slots

(out_dir / "report.json").write_text(json.dumps(report, indent=1))
for key in ("constant", "map", "latency_removed", "tempo_per_8_bars", "smoothed_latency_removed",
            "held_out_" + src_track, "subdivisions", "within_beat_twelfths",
            "per_pulse_of_bar_from_one"):
    print(key, json.dumps(report[key]))
