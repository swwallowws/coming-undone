"""Post-transcription cleanup.

Raw transcriptions of reverby comping arrive as walls of sustained overlapping
notes. A downstream style engine reads that texture as "pad" when the source was
actually "stabs", so this pass exists to make the MIDI say what the audio meant.

Every step is togglable and parameterized; the defaults are tuned on real
material (Turkish pop w/ reverby comping), not derived from first principles.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import pretty_midi


@dataclass
class CleanupParams:
    de_overlap: bool = True
    #: Notes longer than this many beats are candidates for smear-trimming.
    max_duration_beats: float = 2.0
    duration_cap: bool = True
    #: Transcription noise lives below here.
    velocity_floor: int = 15
    #: Notes shorter than this (seconds) after trimming are dropped as artifacts.
    min_duration: float = 0.02
    #: No beat tracking in v1 (non-goal), so beats->seconds uses a fixed tempo.
    tempo: float = 120.0

    #: Collapse the track to one note at a time. For a bass this is usually
    #: right: a bass plays one note, and everything above its fundamental is
    #: either an overtone basic-pitch mistook for a note, or another instrument
    #: bleeding into the stem. Measured on real material, this removed 11% of
    #: bass notes and two thirds of its simultaneity, and what went was almost
    #: entirely octaves, fifths and octave-fifths -- the overtone series.
    #: It WILL delete real double-stops. Off by default; opt in per stem.
    monophonic: bool = False
    #: Which note survives when several sound at once. "lowest" keeps the
    #: fundamental, which is what makes this work for bass -- overtones and
    #: bleed sit above it, never below.
    mono_keep: str = "lowest"

    #: Close gaps between consecutive notes -- the inverse of the trims above.
    #: Off by default: it pushes a track toward "pad", which is exactly the
    #: reading the rest of this module exists to prevent. Turn it on when the
    #: source really is legato and basic-pitch has chopped it up.
    legato: bool = False
    #: Only close gaps up to this many beats. THIS is what separates legato from
    #: a drone: an unconditional stretch would swallow rests and phrase endings,
    #: turning silence the player intended into sustain nobody played. Roughly a
    #: 16th at the detected tempo.
    legato_max_gap_beats: float = 0.25
    #: Leave this much silence at the joint (seconds). Notes that touch exactly
    #: retrigger badly on some synths; a hair of gap keeps the articulation.
    legato_gap: float = 0.005

    #: Snap note onsets toward a fixed grid so the MIDI lines up with the bars
    #: in a DAW. Off by default: a live performance has no true grid, and hard
    #: snapping destroys the human feel. This is the fix for "not aligned".
    quantize: bool = False
    #: Grid resolution as a division of the beat: 4 = 16th notes, 2 = 8ths.
    quantize_division: int = 4
    #: How far to pull each onset toward the grid, 0..1. 1.0 is rigid machine
    #: timing; 0.5 halves the human error while keeping the feel. This is the
    #: dial that separates "tightened" from "robotic".
    quantize_strength: float = 0.5
    #: Move note ends with their starts, preserving duration. Off snaps ends to
    #: the grid independently, which is tighter but can swallow short notes.
    quantize_keep_duration: bool = True

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CleanupStats:
    """Before/after report for one track. Lands in manifest.json."""

    notes_before: int = 0
    notes_after: int = 0
    median_duration_before: float = 0.0
    median_duration_after: float = 0.0
    dropped_velocity_floor: int = 0
    trimmed_de_overlap: int = 0
    trimmed_duration_cap: int = 0
    dropped_too_short: int = 0
    dropped_monophonic: int = 0
    extended_legato: int = 0
    quantized: int = 0
    #: Mean absolute onset error against the grid, in fractions of a grid step.
    #: 0 = dead on the grid, 0.5 = worst case. Falls when quantize is on.
    grid_error_before: float = 0.0
    grid_error_after: float = 0.0
    #: Mean simultaneous notes. 1.0 is a monophonic line; a bass reading well
    #: above that is a sign of overtones or bleed, not of virtuosity.
    polyphony_before: float = 0.0
    polyphony_after: float = 0.0
    #: Fraction of the track that is silence between notes. The number that
    #: actually answers "does this sound staccato?" -- median duration does not.
    silence_ratio_before: float = 0.0
    silence_ratio_after: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _median_duration(notes) -> float:
    if not notes:
        return 0.0
    return round(statistics.median(n.end - n.start for n in notes), 4)


def _polyphony(notes) -> float:
    """Mean simultaneous notes over the track's sounding time."""
    if not notes:
        return 0.0
    span = max(n.end for n in notes) - min(n.start for n in notes)
    if span <= 0:
        return 0.0
    return round(sum(n.end - n.start for n in notes) / span, 3)


def _keep_lowest(notes, keep: str = "lowest"):
    """Collapse to one voice, keeping the fundamental.

    "Lowest wins" is the whole trick. An overtone is by definition above the
    note that produced it, and a chord bleeding into a bass stem sits above the
    bass line. So a note with something lower sounding underneath it is almost
    never the fundamental.
    """
    # The sort must follow the mode: the winning voice has to be considered
    # first, or it is never in `out` to clash against and nothing gets dropped.
    lowest = keep == "lowest"
    ordered = sorted(notes, key=lambda n: (n.start, n.pitch if lowest else -n.pitch))
    out = []
    for n in ordered:
        beaten = any(
            m.start < n.end and n.start < m.end  # overlapping
            and ((m.pitch < n.pitch) if lowest else (m.pitch > n.pitch))
            for m in out
        )
        if not beaten:
            out.append(n)
    return out


def _silence_ratio(notes) -> float:
    """Fraction of the track's span with no note sounding.

    Median duration cannot see this: shortening every note leaves the median
    shorter, but so does a track of the same notes spread further apart. This is
    the number that tracks how staccato something actually sounds.
    """
    if len(notes) < 2:
        return 0.0
    ns = sorted(notes, key=lambda n: n.start)
    span = max(n.end for n in ns) - ns[0].start
    if span <= 0:
        return 0.0
    # Union of sounding intervals, so overlapping notes are not double-counted.
    sounding, cur_s, cur_e = 0.0, ns[0].start, ns[0].end
    for n in ns[1:]:
        if n.start > cur_e:
            sounding += cur_e - cur_s
            cur_s, cur_e = n.start, n.end
        else:
            cur_e = max(cur_e, n.end)
    sounding += cur_e - cur_s
    return round(max(0.0, 1.0 - sounding / span), 4)


def clean_instrument(
    inst: pretty_midi.Instrument,
    params: CleanupParams | None = None,
) -> tuple[pretty_midi.Instrument, CleanupStats]:
    """Return a cleaned copy of `inst` plus the before/after stats."""
    p = params or CleanupParams()
    stats = CleanupStats(
        notes_before=len(inst.notes),
        median_duration_before=_median_duration(inst.notes),
        silence_ratio_before=_silence_ratio(inst.notes),
        polyphony_before=_polyphony(inst.notes),
    )

    notes = [
        pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end)
        for n in inst.notes
    ]

    # 1. Velocity floor. First, so noise notes cannot influence the trims below.
    if p.velocity_floor > 0:
        kept = [n for n in notes if n.velocity >= p.velocity_floor]
        stats.dropped_velocity_floor = len(notes) - len(kept)
        notes = kept

    notes.sort(key=lambda n: (n.start, n.pitch))

    stats.grid_error_before = _grid_error(notes, p.tempo, p.quantize_division)

    # 1b. Monophonic: before any trimming, drop the voices that are not the
    #     fundamental. Doing it early means the trims below never spend their
    #     effort reconciling notes that were overtones all along.
    if p.monophonic:
        kept = _keep_lowest(notes, p.mono_keep)
        stats.dropped_monophonic = len(notes) - len(kept)
        notes = kept

    # 1c. Quantize: pull onsets toward the grid. Early, so every step below
    #     (de-overlap, duration cap, legato) reconciles notes against gridded
    #     positions rather than raw human timing. Partial strength keeps feel.
    if p.quantize and notes and 0 < p.quantize_strength <= 1.0:
        step = (60.0 / p.tempo) / p.quantize_division
        for n in notes:
            target = round(n.start / step) * step
            delta = (target - n.start) * p.quantize_strength
            if delta == 0:
                continue
            n.start += delta
            if p.quantize_keep_duration:
                n.end += delta  # move rigidly, preserving the note's length
            else:
                end_target = round(n.end / step) * step
                n.end += (end_target - n.end) * p.quantize_strength
            if n.end <= n.start:  # guard: a snap must not invert a note
                n.end = n.start + p.min_duration
            stats.quantized += 1
        notes.sort(key=lambda n: (n.start, n.pitch))

    # 2. De-overlap: a pitch cannot sound twice at once. Trim the earlier note
    #    back to where the same pitch re-onsets.
    if p.de_overlap:
        by_pitch: dict[int, list] = {}
        for n in notes:
            by_pitch.setdefault(n.pitch, []).append(n)
        for pitch_notes in by_pitch.values():
            pitch_notes.sort(key=lambda n: n.start)
            for prev, nxt in zip(pitch_notes, pitch_notes[1:]):
                if prev.end > nxt.start:
                    prev.end = nxt.start
                    stats.trimmed_de_overlap += 1

    # 3. Duration cap: a long note with other notes re-onsetting underneath it is
    #    smear, not a sustain. Trim it to the next onset. Long notes over silence
    #    are left alone -- they are probably real.
    if p.duration_cap and p.max_duration_beats > 0:
        max_seconds = p.max_duration_beats * (60.0 / p.tempo)
        onsets = sorted({n.start for n in notes})
        for n in notes:
            if (n.end - n.start) <= max_seconds:
                continue
            nxt = _next_onset_after(onsets, n.start, exclude_before=n.start + 1e-6)
            if nxt is not None and nxt < n.end:
                n.end = nxt
                stats.trimmed_duration_cap += 1

    # 4. Sweep up anything the trims collapsed.
    kept = [n for n in notes if (n.end - n.start) >= p.min_duration]
    stats.dropped_too_short = len(notes) - len(kept)
    notes = kept
    notes.sort(key=lambda n: (n.start, n.pitch))

    # 5. Legato: close the small gaps basic-pitch leaves behind. Runs last, so
    #    it works on notes the trims have already settled -- and only ever
    #    lengthens, so it cannot resurrect the smear steps 2-3 removed.
    if p.legato and notes:
        max_gap = p.legato_max_gap_beats * (60.0 / p.tempo)
        onsets = sorted({n.start for n in notes})
        for n in notes:
            nxt = _next_onset_after(onsets, n.start, exclude_before=n.end - 1e-9)
            if nxt is None:
                continue  # last note: nothing to reach for, leave its tail alone
            gap = nxt - n.end
            # Only close a gap that is small AND real. A big gap is a rest
            # someone played on purpose; stretching across it invents sustain.
            if 0 < gap <= max_gap:
                n.end = nxt - p.legato_gap
                stats.extended_legato += 1

    out = pretty_midi.Instrument(
        program=inst.program, is_drum=inst.is_drum, name=inst.name
    )
    out.notes = notes
    stats.notes_after = len(notes)
    stats.median_duration_after = _median_duration(notes)
    stats.silence_ratio_after = _silence_ratio(notes)
    stats.polyphony_after = _polyphony(notes)
    stats.grid_error_after = _grid_error(notes, p.tempo, p.quantize_division)
    return out, stats


def _grid_error(notes, tempo: float, division: int) -> float:
    """Mean absolute onset distance from the grid, in fractions of a grid step.

    0 means every onset sits on a gridline; 0.5 is the worst possible. This is
    what "not aligned" is: a high number means the notes float between the bars.
    """
    if not notes:
        return 0.0
    step = (60.0 / tempo) / division
    errs = []
    for n in notes:
        off = n.start % step
        errs.append(min(off, step - off) / step)
    return round(statistics.mean(errs), 4)


def _next_onset_after(onsets: list[float], start: float, exclude_before: float):
    """First onset strictly later than `exclude_before`. onsets is sorted."""
    import bisect

    i = bisect.bisect_right(onsets, exclude_before)
    return onsets[i] if i < len(onsets) else None
