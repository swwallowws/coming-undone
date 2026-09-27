"""The beat grid: one constant tempo plus a bar line, fitted from the notes themselves.

Why from the notes: a good transcription backend (MuScriptor) writes its onsets on
the song's real 16th grid to within ~1 ms, while beat trackers land a percent or so
off (beat_this was 1.2% fast on the reference song) and drift a bar away over a
whole song. So the tightest transcribed track sets the grid; the drums-stem tempo
is only the starting guess.

Why a pickup bar: note times are the contract (they refer to the original file),
so the grid never moves a note. Bar lines are made real instead by a first bar of
its own tempo that ends exactly on the first real bar line.

Snapping is optional and separate: each track's latency (its median signed offset
from the grid, ADT_STR drums ran 41 ms early) is removed first, or plain snapping
pushes a model that runs early onto the previous 16th.

Meters: the grid counts pulses, one per note of the meter's denominator (a quarter in
4/4 and 3/4, an eighth in 6/8 and 9/8), and a bar is `meter.pulses` of them. The fine
grid is always four steps per pulse. 4/4 is the default and behaves exactly as before.
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

    def as_dict(self) -> dict:
        d = {"bpm": round(self.bpm, 4), "first_bar": round(self.anchor % self.bar, 4)}
        if self.meter != DEFAULT:
            d["meter"] = str(self.meter)
        return d


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


def fit_pulse(tracks: dict[str, list[float]], beat_bpm: float, meter: Meter = DEFAULT):
    """The grid in the meter's pulse, from a tracked beat. In x/4 (and x/2) the pulse is
    the tracked beat. In x/8 a tracker may have followed the eighth, the quarter or the
    dotted quarter, so try pulse = beat, beat/2 and beat/3, coarsest first.

    A finer grid always fits at least as tightly as a coarser one it contains, so a finer
    pulse replaces the one kept so far only when it is clearly needed: its mean onset
    offset is under FINER_GAIN of the kept one's (and PULSE_TOL smaller), and its own off
    positions are mostly played (OFF_FILL). An eighth-pulse 6/8 ballad with some 16th
    fill-ins keeps the eighth; a song with a note on every eighth under a quarter-note
    tracker moves to beat/2. Returns (grid on a pulse, source track, {track: alignment})
    or None."""
    best = None
    for d in ((1, 2, 3) if meter.den >= 8 else (1,)):
        res = fit_tracks(tracks, beat_bpm * d, meter=meter)
        if res is None or res[2][res[1]] < MIN_ALIGNMENT:
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
    return replace(g, anchor=g.anchor + beats * g.beat)


def latency(times, g: Grid) -> float:
    """A track's typical signed offset from the grid, in seconds (negative = early)."""
    t = np.asarray(list(times), dtype=float)
    return float(np.median(g.signed(t))) if len(t) else 0.0


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
    half = g.sixteenth / 2
    seen: dict[tuple[int, int], pretty_midi.Note] = {}
    for n in inst.notes:
        on = max(0.0, g.snap_time(n.start - lat))
        off = max(on + half, g.snap_time(n.end - lat, half))
        note = pretty_midi.Note(n.velocity, n.pitch, on, off)
        if inst.is_drum:
            key = (round((on - g.anchor) / g.sixteenth), n.pitch)
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


def apply(pm: pretty_midi.PrettyMIDI, bpm_guess: float | None, snap_notes: bool = False,
          downbeat: int | None = None, shift_beats: int = 0, meter: Meter = DEFAULT):
    """Fit the grid to `pm`'s tracks, stamp it, and optionally snap every track.

    bpm_guess: the tracked beat's tempo to search near (None searches 60-200 BPM, for a
    MIDI whose tempo map is a placeholder). meter: the bar (default 4/4); the grid's pulse
    is its denominator note, see fit_pulse. downbeat: 1 to meter.pulses, which pulse of
    the guessed bar is really "one" (the override). shift_beats: move bar "one" by whole
    pulses after that. Returns (new pm, info for the manifest, warnings). With no usable
    grid, `pm` comes back untouched and info["fitted"] is False."""
    warnings: list[str] = []
    tracks = {i.name or f"track {k}": [n.start for n in i.notes] for k, i in enumerate(pm.instruments)}
    if bpm_guess is None:
        best = max(tracks, key=lambda k: len(_onsets(tracks[k])), default=None)
        if best and len(_onsets(tracks[best])) >= MIN_ONSETS:
            bpm_guess = search(tracks[best])[0].bpm
    res = None if bpm_guess is None else fit_pulse(tracks, bpm_guess, meter)
    if res is None:
        warnings.append("no track sits on a steady grid (too few notes, or too loose); "
                        "the MIDI keeps its old tempo and nothing was snapped")
        return pm, {"fitted": False}, warnings
    g, source, fits = res
    g, conf = phase_from_notes(g, pm.instruments)
    P = meter.pulses
    if downbeat is not None:
        if not 1 <= downbeat <= P:
            raise ValueError(f"downbeat must be 1 to {P}, got {downbeat}")
        g = shift(g, downbeat - 1)
    g = shift(g, shift_beats)
    if downbeat is None and conf < PHASE_WARN and P > 1:
        later = "2" if P == 2 else ", ".join(map(str, range(2, P))) + f" or {P}"
        warnings.append(f"bar 'one' is a low-confidence guess ({conf:.2f}); check the bar lines "
                        f"and pass --downbeat {later} if 'one' is a later beat of the bar")
    per_track = {}
    out = stamp(pm, g)
    for k, inst in enumerate(pm.instruments):
        name = inst.name or f"track {k}"
        starts = [n.start for n in inst.notes]
        lat = latency(starts, g)
        per_track[name] = {"alignment": round(fits[name], 4) if name in fits else None,
                           "latency_ms": round(lat * 1000, 1),
                           "grid_fit_ms": None if not starts else round(fit_ms(starts, g, lat), 1)}
        if snap_notes and inst.notes:
            out.instruments[k], _ = snap(inst, g, lat)
    info = {"fitted": True, **g.as_dict(), "anchor": g.anchor, "source_track": source,
            "bar_one_confidence": round(conf, 3), "downbeat_override": downbeat,
            "shift_beats": shift_beats, "snapped": snap_notes, "tracks": per_track}
    return out, info, warnings


def stamp(pm: pretty_midi.PrettyMIDI, g: Grid) -> pretty_midi.PrettyMIDI:
    """A copy of `pm` whose tempo map puts real bar lines of the grid's meter on the grid.
    The first bar is a pickup of its own tempo that ends on the first real bar line; notes
    keep their times exactly (to the tick). The MIDI tempo counts quarter notes, so an
    eighth pulse at 180 is written as 90 BPM."""
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
