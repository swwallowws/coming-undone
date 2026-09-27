"""The auto threshold: how far each song's constant grid sits from its source track (in
16ths), and what auto picks. Steady songs should sit under grid.MAP_THRESHOLD, drifting
ones over it (or have no constant grid at all).

calibrate.py MIDI BPM METER [MIDI BPM METER ...]
"""
import sys

import pretty_midi

from stemscribe import grid as G

args = sys.argv[1:]
print(f"MAP_THRESHOLD {G.MAP_THRESHOLD}")
for k in range(0, len(args), 3):
    path, bpm, meter = args[k], float(args[k + 1]), G.Meter.parse(args[k + 2])
    pm = pretty_midi.PrettyMIDI(path)
    _, c, _ = G.apply(pm, bpm, meter=meter, tempo="constant")
    _, a, _ = G.apply(pm, bpm, meter=meter)
    off = c.get("constant_offset_16th") if c["fitted"] else None
    src = c.get("source_track") if c["fitted"] else "-"
    print(f"{path}: constant {'no fit' if off is None else f'{off:.3f}'} (from {src}), auto picks {a.get('tempo')}")
