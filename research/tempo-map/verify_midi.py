"""Check a gridded MIDI against its source: no note moved, the time signature, how many
tempo changes, and the first bar lines as a DAW will read them.

verify_midi.py GRIDDED.mid SOURCE.mid [OFFSET]

OFFSET: seconds the source was sliced at (the gridded file's 0 is the source's OFFSET).
"""
import sys

import numpy as np
import pretty_midi

out, src = pretty_midi.PrettyMIDI(sys.argv[1]), pretty_midi.PrettyMIDI(sys.argv[2])
offset = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
end = offset + out.get_end_time()
worst = 0.0
for a in out.instruments:
    b = next(i for i in src.instruments if i.name == a.name)
    x = np.array(sorted(n.start for n in a.notes))
    y = np.array(sorted(n.start - offset for n in b.notes if offset <= n.start < end))[:len(x)]
    worst = max(worst, float(np.max(np.abs(x - y))) if len(x) else 0.0)
ts = out.time_signature_changes[0]
times, qpm = out.get_tempo_changes()
print(f"{sys.argv[1]}")
print(f"  time signature {ts.numerator}/{ts.denominator}, {len(times)} tempo changes, "
      f"quarter tempo {qpm[1:].min():.1f} to {qpm[1:].max():.1f}" if len(qpm) > 1 else "  one tempo")
print(f"  first bar lines (s): {', '.join(f'{t:.2f}' for t in out.get_downbeats()[:6])}")
print(f"  worst note shift: {worst * 1000:.2f} ms")
