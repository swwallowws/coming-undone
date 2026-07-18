import pretty_midi
import pytest

from stemscribe.cleanup import CleanupParams, clean_instrument


def inst(notes, name="comping"):
    i = pretty_midi.Instrument(program=0, name=name)
    for pitch, start, end, vel in notes:
        i.notes.append(
            pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end)
        )
    return i


def test_de_overlap_trims_prev_same_pitch_to_next_onset():
    # Same pitch sounding twice at once is impossible; the first must yield.
    i = inst([(60, 0.0, 2.0, 90), (60, 1.0, 3.0, 90)])
    out, stats = clean_instrument(i, CleanupParams(duration_cap=False))
    assert [(n.start, n.end) for n in out.notes] == [(0.0, 1.0), (1.0, 3.0)]
    assert stats.trimmed_de_overlap == 1


def test_de_overlap_leaves_different_pitches_alone():
    i = inst([(60, 0.0, 2.0, 90), (64, 1.0, 3.0, 90)])
    out, _ = clean_instrument(i, CleanupParams(duration_cap=False))
    assert [(n.pitch, n.start, n.end) for n in out.notes] == [
        (60, 0.0, 2.0),
        (64, 1.0, 3.0),
    ]


def test_duration_cap_trims_smear_to_next_onset():
    # 4s note at 120bpm = 8 beats, way over the 2-beat cap, with another note
    # re-onsetting underneath at 1.0 -> it is smear, trim it.
    i = inst([(60, 0.0, 4.0, 90), (67, 1.0, 1.5, 90)])
    out, stats = clean_instrument(i, CleanupParams(tempo=120.0, max_duration_beats=2.0))
    assert out.notes[0].end == 1.0
    assert stats.trimmed_duration_cap == 1


def test_duration_cap_spares_long_note_over_silence():
    # Long, but nothing re-onsets under it -> probably a real sustain.
    i = inst([(60, 0.0, 4.0, 90)])
    out, stats = clean_instrument(i, CleanupParams(tempo=120.0, max_duration_beats=2.0))
    assert out.notes[0].end == 4.0
    assert stats.trimmed_duration_cap == 0


def test_duration_cap_spares_short_notes():
    i = inst([(60, 0.0, 0.5, 90), (67, 0.1, 0.5, 90)])
    out, stats = clean_instrument(i, CleanupParams(tempo=120.0, max_duration_beats=2.0))
    assert stats.trimmed_duration_cap == 0
    assert out.notes[0].end == 0.5


def test_velocity_floor_drops_noise():
    i = inst([(60, 0.0, 1.0, 90), (61, 0.0, 1.0, 5)])
    out, stats = clean_instrument(i, CleanupParams(velocity_floor=15))
    assert [n.pitch for n in out.notes] == [60]
    assert stats.dropped_velocity_floor == 1


def test_notes_collapsed_to_nothing_are_dropped():
    # Second note onsets exactly where the first starts -> first trims to zero.
    i = inst([(60, 1.0, 2.0, 90), (60, 1.0, 2.0, 90)])
    out, stats = clean_instrument(i, CleanupParams())
    assert len(out.notes) == 1
    assert stats.dropped_too_short == 1


def test_stats_report_before_and_after():
    i = inst([(60, 0.0, 4.0, 90), (67, 1.0, 5.0, 90), (72, 0.0, 1.0, 3)])
    _, stats = clean_instrument(i, CleanupParams(tempo=120.0))
    assert stats.notes_before == 3
    assert stats.notes_after == 2
    # The whole point: cleanup shortens the median note.
    assert stats.median_duration_after < stats.median_duration_before


def test_cleanup_is_non_destructive_to_input():
    i = inst([(60, 0.0, 4.0, 90), (67, 1.0, 1.5, 90)])
    clean_instrument(i, CleanupParams(tempo=120.0))
    assert i.notes[0].end == 4.0  # original untouched


def test_toggles_off_is_a_passthrough():
    i = inst([(60, 0.0, 4.0, 90), (60, 1.0, 3.0, 2)])
    out, _ = clean_instrument(
        i, CleanupParams(de_overlap=False, duration_cap=False, velocity_floor=0)
    )
    assert len(out.notes) == 2


def test_empty_instrument_survives():
    out, stats = clean_instrument(inst([]), CleanupParams())
    assert out.notes == []
    assert stats.notes_before == 0 and stats.notes_after == 0


# --- monophonic (overtone / bleed removal) ----------------------------------
def test_monophonic_drops_the_octave_above():
    """The 2nd harmonic: basic-pitch hears it as a note, a bassist did not
    play it."""
    i = inst([(40, 0.0, 1.0, 90), (52, 0.0, 1.0, 60)])
    out, stats = clean_instrument(i, CleanupParams(monophonic=True, duration_cap=False))
    assert [n.pitch for n in out.notes] == [40]
    assert stats.dropped_monophonic == 1


def test_monophonic_drops_the_octave_fifth():
    """+19, the 3rd harmonic, and the most common phantom on a real bass."""
    i = inst([(28, 0.0, 1.0, 90), (47, 0.2, 0.8, 50)])
    out, _ = clean_instrument(i, CleanupParams(monophonic=True, duration_cap=False))
    assert [n.pitch for n in out.notes] == [28]


def test_monophonic_drops_bled_chord_tones():
    """A chord bleeding into the bass stem sits above the bass line."""
    i = inst([(36, 0.0, 1.0, 90), (39, 0.0, 1.0, 40), (46, 0.0, 1.0, 35)])
    out, stats = clean_instrument(i, CleanupParams(monophonic=True, duration_cap=False))
    assert [n.pitch for n in out.notes] == [36]
    assert stats.dropped_monophonic == 2


def test_monophonic_keeps_sequential_notes():
    """It removes simultaneity, not the line itself."""
    i = inst([(40, 0.0, 0.5, 90), (52, 0.6, 1.0, 90), (45, 1.1, 1.5, 90)])
    out, stats = clean_instrument(i, CleanupParams(monophonic=True, duration_cap=False))
    assert len(out.notes) == 3
    assert stats.dropped_monophonic == 0


def test_monophonic_keeps_a_higher_note_once_the_lower_stops():
    i = inst([(40, 0.0, 1.0, 90), (52, 1.0, 2.0, 90)])
    out, _ = clean_instrument(i, CleanupParams(monophonic=True, duration_cap=False))
    assert [n.pitch for n in out.notes] == [40, 52]


def test_monophonic_keep_highest_inverts_it():
    i = inst([(40, 0.0, 1.0, 90), (52, 0.0, 1.0, 90)])
    out, _ = clean_instrument(
        i, CleanupParams(monophonic=True, mono_keep="highest", duration_cap=False)
    )
    assert [n.pitch for n in out.notes] == [52]


def test_monophonic_is_off_by_default():
    """It deletes real double-stops, so it must never be a silent default."""
    i = inst([(40, 0.0, 1.0, 90), (52, 0.0, 1.0, 90)])
    out, stats = clean_instrument(i, CleanupParams(duration_cap=False))
    assert len(out.notes) == 2
    assert stats.dropped_monophonic == 0


def test_monophonic_reduces_polyphony_stat():
    i = inst([(40, 0.0, 1.0, 90), (52, 0.0, 1.0, 90), (59, 0.0, 1.0, 90)])
    _, stats = clean_instrument(i, CleanupParams(monophonic=True, duration_cap=False))
    assert stats.polyphony_before == pytest.approx(3.0, abs=0.01)
    assert stats.polyphony_after == pytest.approx(1.0, abs=0.01)


def test_polyphony_of_a_monophonic_line_is_one():
    i = inst([(40, 0.0, 1.0, 90), (42, 1.0, 2.0, 90)])
    _, stats = clean_instrument(i, CleanupParams(velocity_floor=0, duration_cap=False))
    assert stats.polyphony_after == pytest.approx(1.0, abs=0.01)


# --- legato -----------------------------------------------------------------
def test_legato_closes_small_gaps():
    # 0.1s gap at 120bpm = 0.2 beats, under the 0.25-beat default
    i = inst([(60, 0.0, 0.4, 90), (62, 0.5, 0.9, 90)])
    out, stats = clean_instrument(i, CleanupParams(legato=True, tempo=120.0))
    assert out.notes[0].end == pytest.approx(0.495, abs=0.001)  # next onset - gap
    assert stats.extended_legato == 1


def test_legato_leaves_real_rests_alone():
    """The whole point: a big gap is a rest, not a transcription artifact."""
    i = inst([(60, 0.0, 0.4, 90), (62, 3.0, 3.4, 90)])
    out, stats = clean_instrument(i, CleanupParams(legato=True, tempo=120.0))
    assert out.notes[0].end == 0.4  # untouched
    assert stats.extended_legato == 0


def test_legato_never_shortens():
    """It only ever lengthens -- it must not undo the smear trims above it."""
    i = inst([(60, 0.0, 0.49, 90), (62, 0.5, 0.9, 90)])
    out, _ = clean_instrument(i, CleanupParams(legato=True, tempo=120.0))
    assert out.notes[0].end >= 0.49


def test_legato_leaves_the_last_note_alone():
    i = inst([(60, 0.0, 0.4, 90)])
    out, stats = clean_instrument(i, CleanupParams(legato=True, tempo=120.0))
    assert out.notes[0].end == 0.4
    assert stats.extended_legato == 0


def test_legato_max_gap_scales_with_tempo():
    """0.25 beats is a different number of seconds at 60bpm vs 180bpm."""
    notes = [(60, 0.0, 0.4, 90), (62, 0.6, 1.0, 90)]  # 0.2s gap
    slow, _ = clean_instrument(inst(notes), CleanupParams(legato=True, tempo=60.0))
    fast, _ = clean_instrument(inst(notes), CleanupParams(legato=True, tempo=180.0))
    assert slow.notes[0].end > 0.4   # 0.25 beat @60bpm = 0.25s -> closes
    assert fast.notes[0].end == 0.4  # 0.25 beat @180bpm = 0.083s -> too big, left


def test_legato_off_by_default():
    i = inst([(60, 0.0, 0.4, 90), (62, 0.5, 0.9, 90)])
    out, stats = clean_instrument(i, CleanupParams())
    assert out.notes[0].end == 0.4
    assert stats.extended_legato == 0


def test_legato_reduces_silence_ratio():
    i = inst([(60, 0.0, 0.4, 90), (62, 0.5, 0.9, 90), (64, 1.0, 1.4, 90)])
    _, off = clean_instrument(i, CleanupParams(legato=False, tempo=120.0))
    _, on = clean_instrument(i, CleanupParams(legato=True, tempo=120.0))
    assert on.silence_ratio_after < off.silence_ratio_after


# --- quantize (the "not aligned" fix) ---------------------------------------
def q(**kw):
    # quantize tests want the grid steps isolated from the other passes
    base = dict(velocity_floor=0, duration_cap=False, de_overlap=False, tempo=120.0)
    base.update(kw)
    return CleanupParams(**base)


def test_quantize_snaps_onset_to_grid():
    # 120bpm, div 4 -> 16th grid at 0.125s. A note at 0.14 should pull toward 0.125.
    i = inst([(60, 0.14, 0.5, 90)])
    out, stats = clean_instrument(i, q(quantize=True, quantize_strength=1.0))
    assert out.notes[0].start == pytest.approx(0.125, abs=0.001)
    assert stats.quantized == 1


def test_quantize_partial_strength_keeps_feel():
    """0.5 strength moves halfway, not all the way. This is the whole point."""
    i = inst([(60, 0.145, 0.5, 90)])  # 0.02 past the 0.125 gridline
    out, _ = clean_instrument(i, q(quantize=True, quantize_strength=0.5))
    assert out.notes[0].start == pytest.approx(0.135, abs=0.001)  # moved halfway


def test_quantize_preserves_duration_by_default():
    i = inst([(60, 0.14, 0.64, 90)])  # duration 0.5
    out, _ = clean_instrument(i, q(quantize=True, quantize_strength=1.0))
    n = out.notes[0]
    assert (n.end - n.start) == pytest.approx(0.5, abs=0.002)


def test_quantize_reduces_grid_error():
    i = inst([(60, 0.14, 0.5, 90), (62, 0.39, 0.7, 90), (64, 0.63, 0.9, 90)])
    _, off = clean_instrument(i, q(quantize=False))
    _, on = clean_instrument(i, q(quantize=True, quantize_strength=1.0))
    assert on.grid_error_after < off.grid_error_after
    assert on.grid_error_after == pytest.approx(0.0, abs=0.01)


def test_quantize_off_leaves_onsets_raw():
    i = inst([(60, 0.14, 0.5, 90)])
    out, stats = clean_instrument(i, q(quantize=False))
    assert out.notes[0].start == 0.14
    assert stats.quantized == 0


def test_quantize_division_changes_grid():
    # a note at 0.20: nearest 16th (0.125 step) is 0.25; nearest 8th (0.25 step) is 0.25
    i = inst([(60, 0.20, 0.6, 90)])
    out16, _ = clean_instrument(i, q(quantize=True, quantize_strength=1.0, quantize_division=4))
    out8, _ = clean_instrument(i, q(quantize=True, quantize_strength=1.0, quantize_division=2))
    assert out16.notes[0].start == pytest.approx(0.25, abs=0.001)
    assert out8.notes[0].start == pytest.approx(0.25, abs=0.001)


def test_quantize_never_inverts_a_note():
    """Independent end-snapping can pull an end onto its start; guard against it."""
    # start 0.24 and end 0.26 both snap toward the 0.25 gridline -> would collide.
    i = inst([(60, 0.24, 0.26, 90)])
    out, _ = clean_instrument(
        i, q(quantize=True, quantize_strength=1.0, quantize_keep_duration=False)
    )
    n = out.notes[0]
    assert n.end > n.start


def test_grid_error_zero_when_on_grid():
    i = inst([(60, 0.0, 0.1, 90), (62, 0.125, 0.2, 90), (64, 0.25, 0.3, 90)])
    _, stats = clean_instrument(i, q())
    assert stats.grid_error_before == pytest.approx(0.0, abs=0.001)


# --- silence ratio ----------------------------------------------------------
def test_silence_ratio_zero_when_notes_touch():
    i = inst([(60, 0.0, 1.0, 90), (62, 1.0, 2.0, 90)])
    _, stats = clean_instrument(i, CleanupParams(velocity_floor=0, duration_cap=False))
    assert stats.silence_ratio_after == pytest.approx(0.0, abs=0.001)


def test_silence_ratio_half_when_half_silent():
    i = inst([(60, 0.0, 1.0, 90), (62, 2.0, 3.0, 90)])
    _, stats = clean_instrument(i, CleanupParams(velocity_floor=0, duration_cap=False))
    assert stats.silence_ratio_after == pytest.approx(1 / 3, abs=0.01)


def test_silence_ratio_ignores_overlap_double_counting():
    """Overlapping notes must not make sounding time exceed the span."""
    i = inst([(60, 0.0, 2.0, 90), (64, 0.5, 2.0, 90), (67, 1.0, 2.0, 90)])
    _, stats = clean_instrument(i, CleanupParams(velocity_floor=0, duration_cap=False))
    assert stats.silence_ratio_after == pytest.approx(0.0, abs=0.001)
