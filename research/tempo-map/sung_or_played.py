"""Is the vocals stem singing or an instrument, per window? Sung words leave consonant
hiss (s, t, ch): noisy 5-10 kHz bursts. An instrumental lead has little of that. Prints the
vocals and bass levels, the vocals' 5-10 kHz share, and its spectral flatness there.
Usage: sung_or_played.py <stems dir> [window seconds]"""
import pathlib
import sys

import numpy as np
import soundfile as sf

d = pathlib.Path(sys.argv[1])
win = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0


def mono(name):
    y, sr = sf.read(str(d / f"{name}.wav"), dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def db(x):
    return 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-9))


voc, sr = mono("vocals")
bass, _ = mono("bass")
n = int(win * sr)
frame = 2048
freqs = np.fft.rfftfreq(frame, 1 / sr)
hi = (freqs >= 5000) & (freqs <= 10000)
print("start  vocals   bass  hiss%  flat")
for i in range(0, len(voc) - n + 1, n):
    seg = voc[i:i + n]
    frames = seg[: len(seg) // frame * frame].reshape(-1, frame) * np.hanning(frame)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-12
    share = spec[:, hi].sum() / spec.sum()
    band = spec[:, hi]
    flat = float(np.mean(np.exp(np.mean(np.log(band), axis=1)) / np.mean(band, axis=1)))
    print(f"{i / sr:5.0f}s {db(seg):6.1f} {db(bass[i:i + n]):6.1f} {100 * share:6.2f} {flat:5.2f}")
