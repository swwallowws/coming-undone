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
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pretty_midi

CHORD_TOL = 0.03         # onsets closer than this count once
MIN_ONSETS = 64          # a track needs this many onsets to set the grid
MIN_ALIGNMENT = 0.5      # below this no track sits on any steady grid
TEMPO_PRIOR = 120.0      # among equally good tempi, the one nearest this wins
RESOLUTION = 960


@dataclass(frozen=True)
class Grid:
    bpm: float
    anchor: float        # the time of one bar line (the grid is 4/4, 16ths)

    @property
    def sixteenth(self) -> float:
        return 60.0 / self.bpm / 4

    @property
    def beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar(self) -> float:
        return 4 * self.beat

    def snap_time(self, t: float, unit: float | None = None) -> float:
        u = unit or self.sixteenth
        return self.anchor + round((t - self.anchor) / u) * u

    def signed(self, t) -> np.ndarray:
        """Signed distance of each time to its nearest 16th, in seconds."""
        s = self.sixteenth
        return (np.asarray(t, dtype=float) - self.anchor + s / 2) % s - s / 2

    def as_dict(self) -> dict:
        return {"bpm": round(self.bpm, 4), "first_bar": round(self.anchor % self.bar, 4)}


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


def _grid_at(six: float, phase: float) -> Grid:
    return Grid(60.0 / (4 * six), (phase / (2 * np.pi)) * six % six)


def fit(times, bpm_guess: float, span: float = 0.03, steps: int = 6001) -> tuple[Grid, float]:
    """The 16th grid near `bpm_guess` (within +-span) that the onsets fit best, and how
    well they fit (1 = every onset exactly on it). Its anchor is a 16th, not yet a bar."""
    on = _onsets(times)
    sixes = (60.0 / bpm_guess / 4) * np.linspace(1 - span, 1 + span, steps)
    r, ph = _alignment(on, sixes)
    i = int(np.argmax(r))
    return _grid_at(sixes[i], ph[i]), float(r[i])


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
               min_onsets: int = MIN_ONSETS):
    """Fit every track with enough onsets; the best-aligned one sets the grid.
    Returns (grid, source track, {track: alignment}) or None."""
    fits, grids = {}, {}
    for name, times in tracks.items():
        if len(_onsets(times)) < min_onsets:
            continue
        grids[name], fits[name] = fit(times, bpm_guess)
    if not fits:
        return None
    source = max(fits, key=fits.get)
    return grids[source], source, fits


def align_to_beats(g: Grid, times) -> Grid:
    """Move the anchor onto a beat: of the four 16ths in a beat, the one most notes start on."""
    on = _onsets(times)
    pos = np.round((on - g.anchor) / g.sixteenth).astype(int) % 4
    k = int(np.argmax(np.bincount(pos, minlength=4)))
    return Grid(g.bpm, g.anchor + k * g.sixteenth)


def _phase(g: Grid, weight_by_beat: np.ndarray) -> tuple[Grid, float]:
    order = np.argsort(weight_by_beat)[::-1]
    best, second = weight_by_beat[order[0]], weight_by_beat[order[1]]
    conf = float((best - second) / best) if best > 0 else 0.0
    return Grid(g.bpm, g.anchor + int(order[0]) * g.beat), conf


def phase_from_notes(g: Grid, instruments) -> tuple[Grid, float]:
    """Bar "one" = the beat of four where the harmony changes most: per beat, the pitch
    classes sounding (weighted by how long they sound in it) against the beat before.
    Returns the grid anchored on it and a confidence from 0 to 1 (the margin over the
    runner-up). A guess: right on a sequenced song by a wide margin, a near tie on a
    song whose chords are pushed ahead of the bar. Low confidence means confirm it.
    (Low-end energy in the drums stem, the first method tried, was two beats off.)"""
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
    idx = np.round((beats - g.anchor) / g.beat).astype(int) % 4
    return _phase(g, np.array([change[idx == k].mean() if np.any(idx == k) else 0.0
                               for k in range(4)]))


def shift(g: Grid, beats: int) -> Grid:
    """Move bar "one" by whole beats (the override for a wrong guess)."""
    return Grid(g.bpm, g.anchor + beats * g.beat)


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
          downbeat: int | None = None, shift_beats: int = 0):
    """Fit the grid to `pm`'s tracks, stamp it, and optionally snap every track.

    bpm_guess: the tempo to search near (None searches 60-200 BPM, for a MIDI whose
    tempo map is a placeholder). downbeat: 1-4, which beat of the guessed bar is really
    "one" (the override). shift_beats: move bar "one" by whole beats after that.
    Returns (new pm, info for the manifest, warnings). With no usable grid, `pm` comes
    back untouched and info["fitted"] is False."""
    warnings: list[str] = []
    tracks = {i.name or f"track {k}": [n.start for n in i.notes] for k, i in enumerate(pm.instruments)}
    if bpm_guess is None:
        best = max(tracks, key=lambda k: len(_onsets(tracks[k])), default=None)
        res = None
        if best and len(_onsets(tracks[best])) >= MIN_ONSETS:
            bpm_guess = search(tracks[best])[0].bpm
            res = fit_tracks(tracks, bpm_guess)
    else:
        res = fit_tracks(tracks, bpm_guess)
    if res is None or res[2][res[1]] < MIN_ALIGNMENT:
        warnings.append("no track sits on a steady grid (too few notes, or too loose); "
                        "the MIDI keeps its old tempo and nothing was snapped")
        return pm, {"fitted": False}, warnings
    g, source, fits = res
    g = align_to_beats(g, [t for v in tracks.values() for t in v])
    g, conf = phase_from_notes(g, pm.instruments)
    if downbeat is not None:
        if not 1 <= downbeat <= 4:
            raise ValueError(f"downbeat must be 1 to 4, got {downbeat}")
        g = shift(g, downbeat - 1)
    g = shift(g, shift_beats)
    if downbeat is None and conf < PHASE_WARN:
        warnings.append(f"bar 'one' is a low-confidence guess ({conf:.2f}); check the bar lines "
                        f"and pass --downbeat 2, 3 or 4 if 'one' is a later beat of the bar")
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
    """A copy of `pm` whose tempo map puts real 4/4 bar lines on the grid. The first bar
    is a pickup of its own tempo that ends on the first real bar line; notes keep their
    times exactly (to the tick)."""
    first = g.anchor % g.bar
    pickup = first if first >= g.bar / 2 else first + g.bar
    out = pretty_midi.PrettyMIDI(resolution=RESOLUTION, initial_tempo=240.0 / pickup)
    bar_tick = 4 * RESOLUTION
    out._tick_scales = [(0, pickup / 4 / RESOLUTION), (bar_tick, 60.0 / (g.bpm * RESOLUTION))]
    out._update_tick_to_time(bar_tick + RESOLUTION)
    out.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]
    out.key_signature_changes = copy.deepcopy(pm.key_signature_changes)
    out.instruments = copy.deepcopy(pm.instruments)
    return out
