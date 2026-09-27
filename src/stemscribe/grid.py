"""The beat grid: one constant tempo plus a bar line, fitted from the notes themselves.

Why from the notes: a good transcription backend (MuScriptor) writes its onsets on
the song's real 16th grid to within ~1 ms, while beat trackers land a percent or so
off (beat_this was 1.2% fast on the reference song) and drift a bar away over a
whole song. So the tightest transcribed track sets the grid; the drums-stem tempo
is only the starting guess.

Why a pickup bar: note times are the contract (they refer to the original file),
so the grid never moves a note. Bar lines are made real instead by a first bar of
its own tempo that ends exactly on the first real bar line.

Snapping is optional and separate: each track's latency (its signed offset from the
grid) is removed first, or plain snapping pushes a model that runs early onto the
previous 16th. The grid alone folds an offset into half a 16th, so with the stems'
audio each track is also lined up against its own stem (see latency, audio_lag).

Meters: the grid counts pulses, one per note of the meter's denominator (a quarter in
4/4 and 3/4, an eighth in 6/8 and 9/8), and a bar is `meter.pulses` of them. The fine
grid is always four steps per pulse. 4/4 is the default and behaves exactly as before.

Tempo map: a live band drifts, and no one tempo fits a whole song (Đurđevdan moves a
few BPM, Harman Dalı 103 to 108). Then each pulse gets its own time, fitted from the
tightest track's onsets around it (TempoMap, fit_map), bar "one" is found across the
map, and the MIDI gets a tempo change per pulse so a DAW's bar lines follow the band.
In "auto" the map is used only when the constant grid fits poorly (use_map); a steady
song keeps the constant grid, byte for byte.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import pretty_midi

CHORD_TOL = 0.03         # onsets closer than this count once
MIN_ONSETS = 64          # a track needs this many onsets to set the grid
MIN_ALIGNMENT = 0.5      # below this no track sits on any steady grid
TEMPO_PRIOR = 120.0      # among equally good tempi, the one nearest this wins
RESOLUTION = 960
PULSE_TOL = 0.002        # s: a finer pulse must sit tighter than a coarser one by this
FINER_GAIN = 0.5         # ...and its mean offset must be under this share of the coarser one's
OFF_FILL = 0.6           # ...and this share of its off positions must be played
ACCENT_WEIGHT = 0.5      # in odd meters, how much loud onsets count next to harmony
GROUP_WEIGHT = 0.5       # how much the other group starts count next to the downbeat

_DENOMINATORS = (1, 2, 4, 8, 16, 32)


def _default_groups(num: int, den: int) -> tuple[int, ...]:
    """How a bar is grouped when the meter does not say. This rule is the reference
    (rearranged adopts it), applied in this order:

    - denominator 4 or less (x/4, x/2, x/1), or a numerator of 1: one group per pulse
      (3/4 is 1+1+1, 6/4 is 1+1+1+1+1+1);
    - otherwise (x/8, x/16, x/32):
      - 9: 2+2+2+3 (the aksak 9/8; write 9/8:3+3+3 for compound 9/8);
      - divisible by 3: threes (3/8 is 3, 6/8 is 3+3, 12/8 is 3+3+3+3);
      - even: twos (2/8 is 2, 4/8 is 2+2, 8/8 is 2+2+2+2);
      - odd: twos, then one three (5/8 is 2+3, 7/8 is 2+2+3, 11/8 is 2+2+2+2+3).

    An explicit grouping ("7/8:3+2+2") always overrides this."""
    if den < 8 or num < 2:
        return (1,) * num
    if num == 9:
        return (2, 2, 2, 3)
    if num % 3 == 0:
        return (3,) * (num // 3)
    if num % 2 == 0:
        return (2,) * (num // 2)
    return (2,) * ((num - 3) // 2) + (3,)          # 5/8 is 2+3, 7/8 is 2+2+3


@dataclass(frozen=True)
class Meter:
    num: int
    den: int
    groups: tuple[int, ...]

    def __post_init__(self):
        if self.num < 1 or self.den not in _DENOMINATORS:
            raise ValueError(f"meter {self.num}/{self.den}: need a numerator of 1 or more "
                             f"and a denominator of {', '.join(map(str, _DENOMINATORS))}")
        if not self.groups or min(self.groups) < 1 or sum(self.groups) != self.num:
            raise ValueError(f"meter {self.num}/{self.den}: groups {'+'.join(map(str, self.groups))} "
                             f"must be positive and add up to {self.num}")

    @classmethod
    def parse(cls, spec: str) -> "Meter":
        """"4/4", "6/8", or with explicit groups "9/8:2+2+2+3"."""
        sig, _, grp = str(spec).strip().partition(":")
        num, sep, den = sig.partition("/")
        try:
            n, d = int(num), int(den)
            groups = tuple(int(x) for x in grp.split("+")) if grp else None
        except ValueError:
            raise ValueError(f"meter must look like 4/4, 6/8 or 9/8:2+2+2+3, got {spec!r}") from None
        if not sep or n < 1:
            raise ValueError(f"meter must look like 4/4, 6/8 or 9/8:2+2+2+3, got {spec!r}")
        return cls(n, d, groups if groups is not None else _default_groups(n, d))

    @property
    def pulses(self) -> int:
        return self.num

    @property
    def accents(self) -> tuple[int, ...]:
        """Where each group starts, in pulses from the bar line."""
        return tuple(int(x) for x in np.cumsum((0,) + self.groups[:-1]))

    @property
    def quarters(self) -> float:
        """The bar's length in quarter notes (the MIDI tempo's unit)."""
        return self.num * 4 / self.den

    def __str__(self) -> str:
        base = f"{self.num}/{self.den}"
        return base if self.groups == _default_groups(self.num, self.den) else \
            base + ":" + "+".join(map(str, self.groups))


DEFAULT = Meter(4, 4, (1, 1, 1, 1))


@dataclass(frozen=True)
class Grid:
    bpm: float           # pulses per minute (quarters in 4/4, eighths in 6/8)
    anchor: float        # the time of one bar line (16ths: four steps per pulse)
    meter: Meter = DEFAULT

    @property
    def sixteenth(self) -> float:
        return 60.0 / self.bpm / 4

    @property
    def beat(self) -> float:
        """One pulse, in seconds."""
        return 60.0 / self.bpm

    @property
    def bar(self) -> float:
        return self.meter.pulses * self.beat

    def snap_time(self, t: float, unit: float | None = None) -> float:
        u = unit or self.sixteenth
        return self.anchor + round((t - self.anchor) / u) * u

    def signed(self, t) -> np.ndarray:
        """Signed distance of each time to its nearest 16th, in seconds."""
        s = self.sixteenth
        return (np.asarray(t, dtype=float) - self.anchor + s / 2) % s - s / 2

    def signed16(self, t) -> np.ndarray:
        """Signed distance of each time to its nearest 16th, in 16ths (-0.5 to 0.5)."""
        return self.signed(t) / self.sixteenth

    def sixteenth_at(self, t: float) -> float:
        return self.sixteenth

    def snap_to(self, t: float, per: int = 1) -> float:
        """The nearest point of the grid with `per` steps per 16th."""
        return self.snap_time(t, None if per == 1 else self.sixteenth / per)

    def step(self, t: float) -> int:
        """The index of the 16th nearest t."""
        return round((t - self.anchor) / self.sixteenth)

    def as_dict(self) -> dict:
        d = {"bpm": round(self.bpm, 4), "first_bar": round(self.anchor % self.bar, 4)}
        if self.meter != DEFAULT:
            d["meter"] = str(self.meter)
        return d


@dataclass(frozen=True, eq=False)
class TempoMap:
    """A pulse time per pulse, for a band whose tempo drifts. Between two pulses time runs
    evenly (four 16ths per pulse); before the first and after the last it runs on at the
    edge pulse's length. beats[first] is a bar line, and so is every meter.pulses-th
    pulse from it. Has Grid's interface, so latency, fit_ms and snap work on either."""
    beats: np.ndarray
    first: int
    meter: Meter = DEFAULT

    @classmethod
    def from_info(cls, info: dict) -> "TempoMap":
        """Rebuild the map an apply() info (or a manifest's grid) describes."""
        return cls(np.asarray(info["beats"], dtype=float), int(info["first"]),
                   Meter.parse(info.get("meter", "4/4")))

    def position(self, t) -> np.ndarray:
        """Each time's place in pulses from beats[0] (fractional)."""
        b = self.beats
        t = np.asarray(t, dtype=float)
        pos = np.interp(t, b, np.arange(len(b), dtype=float))
        pos = np.where(t < b[0], (t - b[0]) / (b[1] - b[0]), pos)
        return np.where(t > b[-1], len(b) - 1 + (t - b[-1]) / (b[-1] - b[-2]), pos)

    def time(self, pos) -> np.ndarray:
        """The time at each place in pulses (the inverse of position)."""
        b = self.beats
        pos = np.asarray(pos, dtype=float)
        t = np.interp(pos, np.arange(len(b), dtype=float), b)
        t = np.where(pos < 0, b[0] + pos * (b[1] - b[0]), t)
        return np.where(pos > len(b) - 1, b[-1] + (pos - len(b) + 1) * (b[-1] - b[-2]), t)

    def period_at(self, t) -> np.ndarray:
        """The length of the pulse each time falls in, in seconds."""
        k = np.clip(np.floor(self.position(t)).astype(int), 0, len(self.beats) - 2)
        return np.diff(self.beats)[k]

    @property
    def beat(self) -> float:
        """The typical pulse, in seconds (the median)."""
        return float(np.median(np.diff(self.beats)))

    @property
    def bpm(self) -> float:
        return 60.0 / self.beat

    @property
    def sixteenth(self) -> float:
        return self.beat / 4

    @property
    def bar(self) -> float:
        return self.meter.pulses * self.beat

    @property
    def anchor(self) -> float:
        """The time of one bar line."""
        return float(self.beats[self.first])

    def bar_lines(self) -> np.ndarray:
        return self.beats[self.first::self.meter.pulses]

    def signed16(self, t) -> np.ndarray:
        q = self.position(t) * 4
        return q - np.round(q)

    def signed(self, t) -> np.ndarray:
        """Signed distance of each time to its nearest 16th, in seconds (at the local tempo)."""
        return self.signed16(t) * self.period_at(t) / 4

    def sixteenth_at(self, t: float) -> float:
        return float(self.period_at(t)) / 4

    def snap_to(self, t: float, per: int = 1) -> float:
        n = 4 * per
        return float(self.time(np.round(self.position(t) * n) / n))

    def step(self, t: float) -> int:
        return int(np.round(self.position(t) * 4))

    def shift(self, pulses: int) -> "TempoMap":
        return replace(self, first=(self.first + pulses) % self.meter.pulses)

    def first_bar_index(self) -> int:
        """The index in beats of the first real bar line of the stamped MIDI: the first
        bar line at least half a bar after time 0 (before it sits a pickup bar)."""
        P = self.meter.pulses
        i = self.first
        while self.beats[i] < 0.5 * P * float(self.period_at(max(self.beats[i], 0.0))):
            i += P
        return i

    def bpm_range(self) -> tuple[float, float]:
        """The slowest and fastest bar, as pulse tempi (single pulses carry the beat
        tracker's frame steps, a bar averages them out)."""
        bars = self.bar_lines()
        per = 60.0 * self.meter.pulses / np.diff(bars) if len(bars) > 1 else \
            60.0 / np.diff(self.beats)
        return float(per.min()), float(per.max())

    def as_dict(self) -> dict:
        lo, hi = self.bpm_range()
        d = {"bpm": round(self.bpm, 4), "first_bar": round(float(self.beats[self.first_bar_index()]), 4),
             "tempo": "map", "bpm_range": [round(lo, 2), round(hi, 2)], "first": self.first,
             "beats": [round(float(x), 5) for x in self.beats]}
        if self.meter != DEFAULT:
            d["meter"] = str(self.meter)
        return d


MAP_THRESHOLD = 0.08     # 16ths: auto uses the map when the constant grid's source track
#                          sits further than this from it on average (0 exact, 0.25 random)
TEMPO_MODES = ("auto", "constant", "map")


MAP_GAIN = 0.02          # 16ths: ...and a map must beat it by this much on held-out notes


def use_map(constant_offset: float | None, gain: float | None = None) -> bool:
    """auto: follow the band when no constant grid fits (None), or it fits poorly and a
    map does clearly better on notes it was not fitted on (gain, see map_gain; None when
    not measured). The second test keeps a steady but loose band on the constant grid:
    Đurđevdan's late excerpt sat 0.13 off one tempo, yet a map gained 0.008 on held-out
    drums and put bass and comping further off; its early excerpt, which drifts from 91
    to 100 BPM, gained 0.055."""
    if constant_offset is None:
        return True
    return constant_offset > MAP_THRESHOLD and (gain is None or gain >= MAP_GAIN)


def offset_16th(times, g) -> float | None:
    """How far the onsets sit from the grid (or map) on average, in 16ths: 0 is exact,
    0.25 is what random onsets give."""
    t = np.asarray(list(times), dtype=float)
    return float(np.mean(np.abs(g.signed16(t)))) if len(t) else None


def _onsets(times) -> np.ndarray:
    out: list[float] = []
    for x in sorted(float(t) for t in times):
        if not out or x - out[-1] > CHORD_TOL:
            out.append(x)
    return np.array(out)


def _alignment(on: np.ndarray, sixes: np.ndarray, chunk: int = 1000):
    """For each candidate 16th length: how tightly the onsets share one phase (0 to 1),
    and that phase."""
    r = np.empty(len(sixes))
    ph = np.empty(len(sixes))
    for i in range(0, len(sixes), chunk):
        z = np.exp(2j * np.pi * on[None, :] / sixes[i:i + chunk, None]).mean(axis=1)
        r[i:i + chunk], ph[i:i + chunk] = np.abs(z), np.angle(z)
    return r, ph


def _grid_at(six: float, phase: float, meter: Meter = DEFAULT) -> Grid:
    return Grid(60.0 / (4 * six), (phase / (2 * np.pi)) * six % six, meter)


def fit(times, bpm_guess: float, span: float = 0.03, steps: int = 6001,
        meter: Meter = DEFAULT) -> tuple[Grid, float]:
    """The 16th grid near `bpm_guess` (within +-span) that the onsets fit best, and how
    well they fit (1 = every onset exactly on it). Its anchor is a 16th, not yet a bar."""
    on = _onsets(times)
    sixes = (60.0 / bpm_guess / 4) * np.linspace(1 - span, 1 + span, steps)
    r, ph = _alignment(on, sixes)
    i = int(np.argmax(r))
    return _grid_at(sixes[i], ph[i], meter), float(r[i])


def search(times, lo: float = 60.0, hi: float = 200.0) -> tuple[Grid, float]:
    """A grid with no tempo guess (a MIDI whose tempo map is a placeholder). Regular
    onsets fit every subdivision of their spacing equally, so among near-best tempi the
    one nearest TEMPO_PRIOR wins."""
    on = _onsets(times)
    bpms = np.exp(np.arange(np.log(lo), np.log(hi), 1e-4))
    r, _ = _alignment(on, 60.0 / bpms / 4)
    good = r >= 0.95 * r.max()
    peaks, i = [], 0
    while i < len(bpms):                       # one best tempo per run of near-best ones
        if good[i]:
            j = i
            while j + 1 < len(bpms) and good[j + 1]:
                j += 1
            peaks.append(i + int(np.argmax(r[i:j + 1])))
            i = j + 1
        else:
            i += 1
    best = min(peaks, key=lambda k: abs(np.log(bpms[k] / TEMPO_PRIOR)))
    return fit(times, float(bpms[best]), span=0.002, steps=2001)


def fit_tracks(tracks: dict[str, list[float]], bpm_guess: float,
               min_onsets: int = MIN_ONSETS, meter: Meter = DEFAULT):
    """Fit every track with enough onsets; the best-aligned one sets the grid.
    Returns (grid, source track, {track: alignment}) or None."""
    fits, grids = {}, {}
    for name, times in tracks.items():
        if len(_onsets(times)) < min_onsets:
            continue
        grids[name], fits[name] = fit(times, bpm_guess, meter=meter)
    if not fits:
        return None
    source = max(fits, key=fits.get)
    return grids[source], source, fits


def align_to_beats(g: Grid, times) -> Grid:
    """Move the anchor onto a pulse: of the four 16ths in a pulse, the one most notes start on."""
    on = _onsets(times)
    pos = np.round((on - g.anchor) / g.sixteenth).astype(int) % 4
    k = int(np.argmax(np.bincount(pos, minlength=4)))
    return replace(g, anchor=g.anchor + k * g.sixteenth)


def _pulse_offset(g: Grid, times) -> float:
    """Mean distance of the onsets to the nearest pulse, in seconds (anchor on a pulse)."""
    on, p = _onsets(times), g.beat
    return float(np.mean(np.abs((on - g.anchor + p / 2) % p - p / 2)))


def _off_fill(fine: Grid, coarse: Grid, times) -> float:
    """Of the fine grid's pulses that fall between the coarse grid's pulses (its "off"
    positions), the share with an onset on them. A song really in the fine pulse plays
    most of them; a song in the coarse pulse with the odd fill-in plays few."""
    on = np.sort(_onsets(times))
    if len(on) < 2:
        return 0.0
    p, tol = fine.beat, fine.beat / 4
    t = fine.anchor + p * np.arange(np.ceil((on[0] - fine.anchor) / p),
                                    np.floor((on[-1] - fine.anchor) / p) + 1)
    c = coarse.beat
    off = t[np.abs((t - coarse.anchor + c / 2) % c - c / 2) > tol]
    if not len(off):
        return 0.0
    i = np.clip(np.searchsorted(on, off), 1, len(on) - 1)
    near = np.minimum(np.abs(on[i] - off), np.abs(on[i - 1] - off))
    return float(np.mean(near <= tol))


def fit_pulse(tracks: dict[str, list[float]], beat_bpm: float, meter: Meter = DEFAULT,
              min_alignment: float = MIN_ALIGNMENT, fixed: bool = False):
    """The grid in the meter's pulse, from a tracked beat. In x/4 (and x/2) the pulse is
    the tracked beat. In x/8 a tracker may have followed the eighth, the quarter or the
    dotted quarter, so try pulse = beat, beat/2 and beat/3, coarsest first.

    A finer grid always fits at least as tightly as a coarser one it contains, so a finer
    pulse replaces the one kept so far only when it is clearly needed: its mean onset
    offset is under FINER_GAIN of the kept one's (and PULSE_TOL smaller), and its own off
    positions are mostly played (OFF_FILL). An eighth-pulse 6/8 ballad with some 16th
    fill-ins keeps the eighth; a song with a note on every eighth under a quarter-note
    tracker moves to beat/2. Returns (grid on a pulse, source track, {track: alignment})
    or None. min_alignment: 0 still picks the pulse for a drifting song (the map's seed).
    fixed: beat_bpm is the pulse (a tempo the user gave), so no finer one is tried."""
    best = None
    for d in ((1, 2, 3) if meter.den >= 8 and not fixed else (1,)):
        res = fit_tracks(tracks, beat_bpm * d, meter=meter)
        if res is None or res[2][res[1]] < min_alignment:
            continue
        g = align_to_beats(res[0], [t for v in tracks.values() for t in v])
        src = tracks[res[1]]
        off = _pulse_offset(g, src)
        if best is None:
            best = (off, g, (g, res[1], res[2]))
        elif (off < FINER_GAIN * best[0] and off < best[0] - PULSE_TOL
              and _off_fill(g, best[1], src) >= OFF_FILL):
            best = (off, g, (g, res[1], res[2]))
    return None if best is None else best[2]


def _phase(g: Grid, weight_by_beat: np.ndarray) -> tuple[Grid, float]:
    if len(weight_by_beat) < 2:          # a one-pulse bar: every pulse is "one"
        return g, 1.0
    order = np.argsort(weight_by_beat)[::-1]
    best, second = weight_by_beat[order[0]], weight_by_beat[order[1]]
    conf = float((best - second) / best) if best > 0 else 0.0
    return replace(g, anchor=g.anchor + int(order[0]) * g.beat), conf


def _with_groups(per_pulse: np.ndarray, meter: Meter) -> np.ndarray:
    """Score each candidate bar "one" by its own value plus, at GROUP_WEIGHT, the mean
    value at the other group starts it implies. Simple meters (a group per pulse) score
    by the downbeat alone, as 4/4 always has."""
    if all(x == 1 for x in meter.groups):
        return per_pulse
    P, rest = meter.pulses, meter.accents[1:]
    return np.array([per_pulse[k] + GROUP_WEIGHT * np.mean([per_pulse[(k + a) % P] for a in rest])
                     for k in range(P)])


def _accents(g: Grid, instruments) -> np.ndarray:
    """Per pulse of the bar: the summed loudness of the onsets nearest it (drums too)."""
    acc = np.zeros(g.meter.pulses)
    for i in instruments:
        for n in i.notes:
            acc[int(round((n.start - g.anchor) / g.beat)) % g.meter.pulses] += n.velocity / 127
    return acc


def phase_from_notes(g: Grid, instruments) -> tuple[Grid, float]:
    """Bar "one" = the pulse of the bar where the harmony changes most: per pulse, the pitch
    classes sounding (weighted by how long they sound in it) against the pulse before.
    Returns the grid anchored on it and a confidence from 0 to 1 (the margin over the
    runner-up). A guess: right on a sequenced song by a wide margin, a near tie on a
    song whose chords are pushed ahead of the bar. Low confidence means confirm it.
    (Low-end energy in the drums stem, the first method tried, was two beats off.)
    In meters other than 4/4 the other group starts count too, and so do accents (loud
    onsets), which carry bar "one" when the harmony holds still."""
    notes = [n for i in instruments if not i.is_drum for n in i.notes]
    if not notes:
        return g, 0.0
    end = max(n.end for n in notes)
    beats = np.arange(g.anchor - np.ceil(g.anchor / g.beat) * g.beat, end, g.beat)
    chroma = np.zeros((len(beats), 12))
    for n in notes:
        a = max(int(np.searchsorted(beats, n.start, "right")) - 1, 0)
        b = min(int(np.searchsorted(beats, n.end, "right")) - 1, len(beats) - 1)
        for k in range(a, b + 1):
            overlap = min(n.end, beats[k] + g.beat) - max(n.start, beats[k])
            if overlap > 0:
                chroma[k, n.pitch % 12] += overlap
    norm = np.linalg.norm(chroma, axis=1)
    c = chroma / np.where(norm > 0, norm, 1.0)[:, None]
    change = np.r_[0.0, 1 - np.sum(c[1:] * c[:-1], axis=1)]
    change[(norm == 0) | (np.r_[0.0, norm[:-1]] == 0)] = 0.0     # silence is not a change
    P = g.meter.pulses
    idx = np.round((beats - g.anchor) / g.beat).astype(int) % P
    harmony = np.array([change[idx == k].mean() if np.any(idx == k) else 0.0 for k in range(P)])
    if g.meter == DEFAULT:
        return _phase(g, harmony)

    def unit(x: np.ndarray) -> np.ndarray:
        return x / x.max() if x.max() > 0 else x
    return _phase(g, unit(_with_groups(harmony, g.meter))
                  + ACCENT_WEIGHT * unit(_with_groups(_accents(g, instruments), g.meter)))


def shift(g: Grid, beats: int) -> Grid:
    """Move bar "one" by whole pulses (the override for a wrong guess)."""
    if isinstance(g, TempoMap):
        return g.shift(beats)
    return replace(g, anchor=g.anchor + beats * g.beat)


# --- the tempo map ------------------------------------------------------------------
MAP_SR = 200             # frames per second of the beat tracker's onset envelope
MAP_TIGHTNESS = 400      # librosa's tightness: how hard the tracker holds the tempo
MAP_SMOOTH = 0           # pulses: each pulse time from a line through it and this many either
#                          side (0: none; see fit_map for why that is the default)


def _smooth(beats: np.ndarray, half: int = MAP_SMOOTH) -> np.ndarray:
    """Light smoothing of the pulse times themselves (never their lengths re-summed, which
    lets a phase error pile up): each from a straight line through its neighbours."""
    if half < 1 or len(beats) < 3:
        return beats
    k = np.arange(len(beats), dtype=float)
    out = np.empty(len(beats))
    for i in range(len(beats)):
        a, b = max(0, i - half), min(len(beats), i + half + 1)
        out[i] = np.polyval(np.polyfit(k[a:b], beats[a:b], 1), k[i])
    return out


def _extend(beats: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fill gaps the tracker left (a break) at the local pulse, and run on past both ends."""
    out = [float(beats[0])]
    for b in beats[1:]:
        p = float(np.median(np.diff(out[-8:]))) if len(out) > 2 else float(b - out[-1])
        while b - out[-1] > 1.5 * p:
            out.append(out[-1] + p)
        out.append(float(b))
    p0, p1 = out[1] - out[0], out[-1] - out[-2]
    while out[0] > lo:
        out.insert(0, out[0] - p0)
    while out[-1] < hi:
        out.append(out[-1] + p1)
    return np.array(out)


def _track_beats(times, weights, g: Grid, end: float, tightness: float = MAP_TIGHTNESS) -> np.ndarray:
    """Pulse times from a dynamic-programming beat tracker (librosa's) run on the onsets
    as an envelope, each weighted by its velocity, the tempo held near g's. Loud hits
    (kick, snare) pull the pulse onto themselves, so it cannot slip onto a 16th."""
    import librosa

    n = int((end + 2.0) * MAP_SR) + 2
    env = np.zeros(n)
    idx = np.clip(np.round(np.asarray(times, dtype=float) * MAP_SR).astype(int), 0, n - 1)
    np.add.at(env, idx, np.asarray(weights, dtype=float))
    env = np.convolve(env, np.hanning(9), mode="same")
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=MAP_SR, hop_length=1, bpm=g.bpm,
                                       tightness=tightness, trim=False, units="time")
    return np.asarray(beats, dtype=float)


def _on_the_pulse(beats: np.ndarray, all_times) -> np.ndarray:
    """Move the pulse onto the 16th most notes start on. Every note counts (unlike
    align_to_beats), so a beat where kick, chord and bass land together outweighs an
    off-beat hat."""
    m = TempoMap(beats, 0)
    q = np.round(m.position(np.asarray(list(all_times), dtype=float)) * 4).astype(int) % 4
    k = int(np.argmax(np.bincount(q, minlength=4)))
    return beats if k == 0 else m.time(np.arange(len(beats)) + k / 4)


def fit_map(times, g: Grid, end: float, all_times=None, weights=None,
            tightness: float = MAP_TIGHTNESS, smooth: int = MAP_SMOOTH) -> np.ndarray:
    """Pulse times that follow the onsets `times` (weighted by `weights`, say velocity)
    through a song whose tempo drifts, from before 0 to past `end`: a beat tracker's
    pulses (see _track_beats), gaps filled, run on past both ends, and moved onto the
    16th most notes (all_times) start on. g: a grid in the right pulse (its tempo is the
    guess). smooth: pulses either side for _smooth (default none). Returns an empty
    array when there is too little to track.

    The evidence (research/tempo-map/compare.py, the source track measured on held-out
    notes): smoothing the pulse times over 2 either side gained about 0.01 of a 16th for
    Đurđevdan's bass and comping, but never helped the held-out drums, and on Harman
    Dalı's slow 9/8 (1.1 s pulses) it lost the drift (0.11 unsmoothed, 0.15 smoothed).
    So none by default. Dropped: walking from the tightest stretch a pulse at a time,
    each pulse a local line through the onsets rounded to 16ths (it slipped by whole
    16ths); refining the tracker's pulses that way; median-smoothing the pulse lengths
    and re-summing them (every track further off)."""
    t = np.asarray(list(times), dtype=float)
    if len(t) < 4:
        return np.array([])
    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)
    beats = _track_beats(t, w, g, end, tightness)
    if len(beats) < 3:
        return np.array([])
    beats = _extend(beats, min(0.0, float(t.min())) - 1e-9, max(end, float(t.max())) + 1e-9)
    return _on_the_pulse(_smooth(beats, smooth), all_times if all_times is not None else t)


def _warp(m: TempoMap, instruments) -> list[pretty_midi.Instrument]:
    """The instruments with every time replaced by its place in pulses."""
    out = []
    for inst in instruments:
        w = pretty_midi.Instrument(inst.program, is_drum=inst.is_drum, name=inst.name)
        if inst.notes:
            s = m.position([n.start for n in inst.notes])
            e = m.position([n.end for n in inst.notes])
            w.notes = [pretty_midi.Note(n.velocity, n.pitch, float(a), float(b))
                       for n, a, b in zip(inst.notes, s, e)]
        out.append(w)
    return out


def phase_map(m: TempoMap, instruments) -> tuple[TempoMap, float]:
    """Bar "one" across a tempo map: phase_from_notes on the notes placed in pulses, so
    a bar is always meter.pulses of them whatever the tempo did."""
    unit = Grid(60.0, 0.0, m.meter)                  # one pulse = one "second"
    g, conf = phase_from_notes(unit, _warp(m, instruments))
    return replace(m, first=int(round(g.anchor)) % m.meter.pulses), conf


def latency(times, g: Grid, hint: float | None = None) -> float:
    """A track's typical signed offset from the grid, in seconds (negative = early).

    The grid alone only sees it folded into half a 16th either way: a track 195 ms late
    at 114 BPM reads as +63 ms. hint: the offset measured some other way (audio_lag
    against the source track's), which picks the whole number of 16ths; the grid's
    median still gives the fine value."""
    t = np.asarray(list(times), dtype=float)
    m = float(np.median(g.signed(t))) if len(t) else 0.0
    if hint is None or not len(t):
        return m
    return m + round((hint - m) / g.sixteenth) * g.sixteenth


LAG_WINDOW = 0.3          # s: audio_lag searches this far either way
LAG_MIN_SCORE = 1.0       # below this the notes do not line up with the audio at any lag


def onset_envelope(y, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """(times, envelope) of an audio's onset strength, about 3 ms a frame, z-scored."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    hop = max(1, int(round(sr * 0.003)))
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env = (env - env.mean()) / (env.std() + 1e-9)
    return np.arange(len(env)) * hop / sr, env


def audio_lag(times, env_times, env, window: float = LAG_WINDOW) -> tuple[float, float]:
    """A track's constant offset from its own audio (positive = the notes come later),
    searched to +-window, and the score there: the mean z-scored onset strength under
    the shifted onsets. The score is low (< LAG_MIN_SCORE) when nothing lines up."""
    on = _onsets(times)
    if not len(on) or not len(env):
        return 0.0, 0.0
    lags = np.arange(-window, window + 1e-9, 0.001)
    score = np.array([np.interp(on - d, env_times, env, left=0.0, right=0.0).mean() for d in lags])
    k = int(np.argmax(score))
    return float(lags[k]), float(score[k])


def fit_ms(times, g: Grid, latency_s: float = 0.0) -> float | None:
    """Median distance of the onsets to the grid once the latency is removed, in ms."""
    t = np.asarray(list(times), dtype=float)
    return float(np.median(np.abs(g.signed(t - latency_s))) * 1000) if len(t) else None


def snap(inst: pretty_midi.Instrument, g: Grid, latency_s: float | None = None):
    """The track with its latency removed, starts on 16ths and ends on 32nds (never
    shorter than a 32nd). A drum kit keeps one hit per drum per 16th, the loudest.
    Returns (new instrument, the latency removed)."""
    lat = latency([n.start for n in inst.notes], g) if latency_s is None else latency_s
    out = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
    seen: dict[tuple[int, int], pretty_midi.Note] = {}
    for n in inst.notes:
        on = max(0.0, g.snap_to(n.start - lat))
        off = max(on + g.sixteenth_at(on) / 2, g.snap_to(n.end - lat, 2))
        note = pretty_midi.Note(n.velocity, n.pitch, on, off)
        if inst.is_drum:
            key = (g.step(on), n.pitch)
            if key in seen and seen[key].velocity >= n.velocity:
                continue
            seen[key] = note
        else:
            out.notes.append(note)
    if inst.is_drum:
        out.notes = list(seen.values())
    out.notes.sort(key=lambda n: (n.start, n.pitch))
    return out, lat


PHASE_WARN = 0.2          # bar "one" guesses below this confidence get a warning


SOURCE_LAG_WARN = 0.1     # s: warn when the grid's own track sits this far off its audio


def apply(pm: pretty_midi.PrettyMIDI, bpm_guess: float | None, snap_notes: bool = False,
          downbeat: int | None = None, shift_beats: int = 0, meter: Meter = DEFAULT,
          audio: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
          tempo: str = "auto", fixed_pulse: bool = False):
    """Fit the grid to `pm`'s tracks, stamp it, and optionally snap every track.

    bpm_guess: the tracked beat's tempo to search near (None searches 60-200 BPM, for a
    MIDI whose tempo map is a placeholder). meter: the bar (default 4/4); the grid's pulse
    is its denominator note, see fit_pulse. downbeat: 1 to meter.pulses, which pulse of
    the guessed bar is really "one" (the override). shift_beats: move bar "one" by whole
    pulses after that. audio: per track name, the onset_envelope of its own stem on the
    MIDI's timeline; a track and the source track that both line up with their audio
    get their latency unfolded past half a 16th (see latency). tempo: "constant" (one
    tempo), "map" (a tempo per pulse that follows the band) or "auto" (the map only when
    the constant grid fits poorly, see use_map). fixed_pulse: bpm_guess is the meter's
    pulse as the user gave it (in 9/8 the eighth), so the grid never moves to a finer
    one. Returns (new pm, info for the manifest, warnings). With no usable grid, `pm`
    comes back untouched, info["fitted"] is False, and a meter other than 4/4 gets a
    warning of its own: it was asked for and is not in the MIDI."""
    if tempo not in TEMPO_MODES:
        raise ValueError(f"tempo must be one of {', '.join(TEMPO_MODES)}, got {tempo!r}")
    P = meter.pulses
    if downbeat is not None and not 1 <= downbeat <= P:
        raise ValueError(f"downbeat must be 1 to {P}, got {downbeat}")
    warnings: list[str] = []
    tracks = {i.name or f"track {k}": [n.start for n in i.notes] for k, i in enumerate(pm.instruments)}
    if bpm_guess is None:
        best = max(tracks, key=lambda k: len(_onsets(tracks[k])), default=None)
        if best and len(_onsets(tracks[best])) >= MIN_ONSETS:
            bpm_guess = search(tracks[best])[0].bpm
    res = None if bpm_guess is None else fit_pulse(tracks, bpm_guess, meter, fixed=fixed_pulse)
    constant_offset = None if res is None else offset_16th(tracks[res[1]], res[0])
    g = None
    gain = None
    if tempo == "auto" and res is not None and use_map(constant_offset):
        gain = map_gain(pm, tracks, bpm_guess, meter, res[0], fixed_pulse)
    tried_map = tempo == "map" or (tempo == "auto" and use_map(constant_offset, gain))
    if tried_map:
        fitted = None if bpm_guess is None else \
            _fit_map_tracks(pm, tracks, bpm_guess, meter, fixed_pulse)
        if fitted is not None:
            g, source, fits, conf = fitted
    if g is None:
        if res is None:
            warnings.append("no track sits on a steady grid (too few notes, or too loose); "
                            "the MIDI keeps its old tempo and nothing was snapped")
            if meter != DEFAULT:
                few = max((len(_onsets(v)) for v in tracks.values()), default=0) < MIN_ONSETS
                why = (f"too few notes (no track has {MIN_ONSETS} onsets)" if few else
                       "no constant tempo fits and --tempo-mode constant rules out a tempo map, "
                       "try --tempo-mode map" if not tried_map else
                       "neither a constant tempo nor a tempo map fits")
                warnings.append(f"the meter {meter} was not applied: {why}; the MIDI has no "
                                f"{meter} bar lines")
            return pm, {"fitted": False}, warnings
        g, source, fits = res
        g, conf = phase_from_notes(g, pm.instruments)
    if downbeat is not None:
        g = shift(g, downbeat - 1)
    g = shift(g, shift_beats)
    if downbeat is None and conf < PHASE_WARN and P > 1:
        later = "2" if P == 2 else ", ".join(map(str, range(2, P))) + f" or {P}"
        warnings.append(f"bar 'one' is a low-confidence guess ({conf:.2f}); check the bar lines "
                        f"and pass --downbeat {later} if 'one' is a later beat of the bar")
    lags: dict[str, float] = {}           # only the tracks that line up with their audio
    for k, inst in enumerate(pm.instruments):
        name = inst.name or f"track {k}"
        if audio and name in audio and inst.notes:
            lag, score = audio_lag([n.start for n in inst.notes], *audio[name])
            if score >= LAG_MIN_SCORE:
                lags[name] = lag
    if source in lags and abs(lags[source]) >= SOURCE_LAG_WARN:
        warnings.append(f"the {source} notes run {lags[source] * 1000:+.0f} ms from their stem's "
                        "audio; the grid follows the notes, so check the MIDI against the audio")
    per_track = {}
    out = stamp(pm, g)
    for k, inst in enumerate(pm.instruments):
        name = inst.name or f"track {k}"
        starts = [n.start for n in inst.notes]
        hint = lags[name] - lags[source] if name in lags and source in lags else None
        lat = latency(starts, g, hint)
        per_track[name] = {"alignment": round(fits[name], 4) if name in fits else None,
                           "latency_ms": round(lat * 1000, 1),
                           "grid_fit_ms": None if not starts else round(fit_ms(starts, g, lat), 1),
                           "offset_16th": None if not starts else round(offset_16th(starts, g), 4)}
        if name in lags:
            per_track[name]["audio_lag_ms"] = round(lags[name] * 1000, 1)
        if snap_notes and inst.notes:
            out.instruments[k], _ = snap(inst, g, lat)
    info = {"fitted": True, "tempo": "constant", **g.as_dict(), "anchor": g.anchor,
            "source_track": source, "constant_offset_16th":
                None if constant_offset is None else round(constant_offset, 4),
            "map_gain_16th": None if gain is None else round(gain, 4),
            "bar_one_confidence": round(conf, 3), "downbeat_override": downbeat,
            "shift_beats": shift_beats, "snapped": snap_notes, "tracks": per_track}
    return out, info, warnings


def _fit_map_tracks(pm: pretty_midi.PrettyMIDI, tracks: dict[str, list[float]],
                    bpm_guess: float, meter: Meter, fixed_pulse: bool = False):
    """The tempo map from the drums (with no drum track, the best-aligned track, as for
    the constant grid), bar "one" found across it. The drums go first: their kick and
    snare mark the pulse, while a dense comping track can align well overall and still
    smear it (on Đurđevdan, following the comping lost the tempo). Returns
    (map, source track, {track: alignment}, bar-one confidence), or None when no track
    has enough onsets."""
    res = fit_pulse(tracks, bpm_guess, meter, min_alignment=0.0, fixed=fixed_pulse)
    if res is None:
        return None
    g, best, fits = res
    source, inst = _map_source(pm, fits, best)
    end = max((n.end for i in pm.instruments for n in i.notes), default=0.0)
    beats = fit_map(tracks[source], g, end, all_times=[t for v in tracks.values() for t in v],
                    weights=[n.velocity / 127 for n in inst.notes])
    if len(beats) < 3:
        return None
    m, conf = phase_map(TempoMap(beats, 0, meter), pm.instruments)
    return m, source, fits, conf


def _map_source(pm: pretty_midi.PrettyMIDI, fits: dict[str, float], best: str):
    """The track the map follows: the best-aligned drum track, else `best`. Returns
    (name, instrument)."""
    def name(k, i):
        return i.name or f"track {k}"
    drums = [n for n in fits if any(i.is_drum and name(k, i) == n for k, i in enumerate(pm.instruments))]
    source = max(drums, key=fits.get) if drums else best
    return source, next(i for k, i in enumerate(pm.instruments) if name(k, i) == source)


def map_gain(pm: pretty_midi.PrettyMIDI, tracks: dict[str, list[float]], bpm_guess: float,
             meter: Meter, constant: Grid, fixed_pulse: bool = False) -> float | None:
    """How much closer the map's source track sits to a map than to the constant grid,
    in 16ths, measured fairly: the map is fitted on every other note of the track and
    both are measured on the notes in between. A steady but loose band gains nothing
    (only a drifting one does). None when there is no map to fit."""
    res = fit_pulse(tracks, bpm_guess, meter, min_alignment=0.0, fixed=fixed_pulse)
    if res is None:
        return None
    g, best, fits = res
    source, inst = _map_source(pm, fits, best)
    notes = sorted(inst.notes, key=lambda n: n.start)
    fit_n, test = notes[0::2], [n.start for n in notes[1::2]]
    end = max((n.end for i in pm.instruments for n in i.notes), default=0.0)
    beats = fit_map([n.start for n in fit_n], g, end, all_times=[t for v in tracks.values() for t in v],
                    weights=[n.velocity / 127 for n in fit_n])
    if len(beats) < 3 or not test:
        return None
    return offset_16th(test, constant) - offset_16th(test, TempoMap(beats, 0, meter))


def stamp(pm: pretty_midi.PrettyMIDI, g: Grid) -> pretty_midi.PrettyMIDI:
    """A copy of `pm` whose tempo map puts real bar lines of the grid's meter on the grid.
    The first bar is a pickup of its own tempo that ends on the first real bar line; notes
    keep their times exactly (to the tick). The MIDI tempo counts quarter notes, so an
    eighth pulse at 180 is written as 90 BPM. A TempoMap gets a tempo change per pulse."""
    if isinstance(g, TempoMap):
        return _stamp_map(pm, g)
    first = g.anchor % g.bar
    pickup = first if first >= g.bar / 2 else first + g.bar
    q = g.meter.quarters                       # the bar in quarter notes (4.0 in 4/4)
    out = pretty_midi.PrettyMIDI(resolution=RESOLUTION, initial_tempo=60.0 * q / pickup)
    bar_tick = int(round(q * RESOLUTION))
    qpm = g.bpm * 4 / g.meter.den              # exact: den is a power of two
    out._tick_scales = [(0, pickup / q / RESOLUTION), (bar_tick, 60.0 / (qpm * RESOLUTION))]
    out._update_tick_to_time(bar_tick + RESOLUTION)
    out.time_signature_changes = [pretty_midi.TimeSignature(g.meter.num, g.meter.den, 0.0)]
    out.key_signature_changes = copy.deepcopy(pm.key_signature_changes)
    out.instruments = copy.deepcopy(pm.instruments)
    return out


def _stamp_map(pm: pretty_midi.PrettyMIDI, m: TempoMap) -> pretty_midi.PrettyMIDI:
    """stamp for a tempo map: the pickup bar, then one tempo per pulse from the first real
    bar line on, so every pulse (and every bar line) lands on its fitted time."""
    i0 = m.first_bar_index()
    pickup = float(m.beats[i0])
    q = m.meter.quarters
    out = pretty_midi.PrettyMIDI(resolution=RESOLUTION, initial_tempo=60.0 * q / pickup)
    bar_tick = int(round(q * RESOLUTION))
    tpp = int(round(RESOLUTION * 4 / m.meter.den))       # ticks per pulse
    per = np.diff(m.beats[i0:])
    out._tick_scales = [(0, pickup / q / RESOLUTION)] + \
        [(bar_tick + k * tpp, float(p) / tpp) for k, p in enumerate(per)]
    out._update_tick_to_time(bar_tick + len(per) * tpp + tpp)
    out.time_signature_changes = [pretty_midi.TimeSignature(m.meter.num, m.meter.den, 0.0)]
    out.key_signature_changes = copy.deepcopy(pm.key_signature_changes)
    out.instruments = copy.deepcopy(pm.instruments)
    return out
