"""Loudness of each stem over time (RMS in dBFS per window), to find a window where every part
plays. Usage: stem_levels.py <stems dir> [window seconds]"""
import pathlib
import sys

import numpy as np
import soundfile as sf

d = pathlib.Path(sys.argv[1])
win = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
levels = {}
for p in sorted(d.glob("*.wav")):
    y, sr = sf.read(str(p), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    n = int(win * sr)
    levels[p.stem] = [20 * np.log10(max(float(np.sqrt(np.mean(y[i:i + n] ** 2))), 1e-9))
                      for i in range(0, len(y) - n + 1, n)]
names = list(levels)
print("start  " + "  ".join(f"{s:>7}" for s in names))
for k in range(min(len(v) for v in levels.values())):
    print(f"{k * win:5.0f}s " + "  ".join(f"{levels[s][k]:7.1f}" for s in names))
