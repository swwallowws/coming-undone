"""Merge per-stem MIDI into one named, GM-programmed multi-track file."""
from __future__ import annotations

import pathlib
import statistics

import pretty_midi

#: stem -> (track name, General MIDI program). The names are the contract
#: rearranged reads ("melody" is the vocal line it overlays), so do not
#: rename them casually.
TRACK_SPEC: dict[str, tuple[str, int]] = {
    "vocals": ("melody", 53),  # GM 54 Voice Oohs
    "bass": ("bass", 33),      # GM 34 Electric Bass (finger)
    "other": ("comping", 0),   # GM 1  Acoustic Grand Piano
    # htdemucs_6s only. Without these the 6-stem model separates guitar and
    # piano and then stemscribe silently drops them.
    "guitar": ("guitar", 25),  # GM 26 Acoustic Guitar (steel)
    "piano": ("piano", 0),     # GM 1  Acoustic Grand Piano
    "drums": ("drums", 0),     # a GM drum kit (channel 10)
}

#: Stems whose track is a drum kit.
DRUM_STEMS = frozenset({"drums"})

#: Track order in the merged file. res.track_map indexes into this.
TRACK_ORDER = ("melody", "bass", "guitar", "piano", "comping", "drums")


def track_for_stem(stem: str) -> tuple[str, int]:
    return TRACK_SPEC.get(stem, (stem, 0))


def merge_midis(
    stem_midis: dict[str, pathlib.Path],
    out_path: str | pathlib.Path,
    tempo: float = 120.0,
) -> tuple[pathlib.Path, dict[str, int]]:
    """Merge {stem: mid_path} into one multi-track MIDI at `out_path`.

    Returns (path, track_map) where track_map is {track_name: track_index}.
    """
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    named: dict[str, pretty_midi.Instrument] = {}

    for stem, mid_path in stem_midis.items():
        name, program = track_for_stem(stem)
        src = pretty_midi.PrettyMIDI(str(mid_path))
        inst = pretty_midi.Instrument(program=program, is_drum=stem in DRUM_STEMS, name=name)
        # A backend may return several instruments per stem; they all belong to
        # this stem's one track.
        for si in src.instruments:
            inst.notes.extend(si.notes)
        inst.notes.sort(key=lambda n: (n.start, n.pitch))
        named[name] = inst

    track_map: dict[str, int] = {}
    for name in TRACK_ORDER:
        if name in named:
            track_map[name] = len(merged.instruments)
            merged.instruments.append(named.pop(name))
    for name, inst in named.items():  # anything outside the known order
        track_map[name] = len(merged.instruments)
        merged.instruments.append(inst)

    merged.write(str(out_path))
    return out_path, track_map


def quantization_error(
    inst: pretty_midi.Instrument,
    tempo: float = 120.0,
    grid_division: int = 4,
    den: int = 4,
) -> dict:
    """How far off a fixed grid the onsets sit, in fractions of a grid step.

    This assumes a constant `tempo` (in quarters) starting at t=0 and runs before
    the grid stage, so it is only meaningful as a relative signal -- a rough "is
    this track rhythmically legible" number. The grid stage's per-track
    grid_fit_ms in the manifest is the measured one. The grid divides the meter's
    pulse (its denominator note, `den`): 16ths in 4/4, 32nds in 6/8.
    """
    if not inst.notes:
        return {"grid": f"1/{grid_division * den}", "n": 0, "mean": 0.0, "median": 0.0}

    pulse = 60.0 / tempo if den == 4 else (60.0 / tempo) * 4 / den
    step = pulse / grid_division
    errs = []
    for n in inst.notes:
        offset = n.start % step
        errs.append(min(offset, step - offset) / step)  # 0 = on grid, 0.5 = worst

    return {
        "grid": f"1/{grid_division * den}",
        "n": len(errs),
        "mean": round(statistics.mean(errs), 4),
        "median": round(statistics.median(errs), 4),
    }
