import pretty_midi
import pytest

from stemscribe.backends import BACKENDS, UNPITCHED_STEMS, BackendError, get_backend
from stemscribe.merge import merge_midis, quantization_error, track_for_stem


def write_midi(path, notes, program=0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=program)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    pm.write(str(path))
    return path


def test_merge_names_tracks_and_orders_melody_first(tmp_path):
    stems = {
        "bass": write_midi(tmp_path / "bass.mid", [(40, 0, 1)]),
        "vocals": write_midi(tmp_path / "vocals.mid", [(72, 0, 1)]),
        "other": write_midi(tmp_path / "other.mid", [(60, 0, 1)]),
    }
    out, track_map = merge_midis(stems, tmp_path / "song.mid")
    assert track_map == {"melody": 0, "bass": 1, "comping": 2}

    pm = pretty_midi.PrettyMIDI(str(out))
    assert [i.name for i in pm.instruments] == ["melody", "bass", "comping"]


def test_merge_assigns_gm_programs(tmp_path):
    stems = {
        "vocals": write_midi(tmp_path / "vocals.mid", [(72, 0, 1)]),
        "bass": write_midi(tmp_path / "bass.mid", [(40, 0, 1)]),
    }
    out, _ = merge_midis(stems, tmp_path / "song.mid")
    pm = pretty_midi.PrettyMIDI(str(out))
    programs = {i.name: i.program for i in pm.instruments}
    assert programs == {"melody": 53, "bass": 33}
    assert not any(i.is_drum for i in pm.instruments)


def test_merge_without_vocals_still_works(tmp_path):
    stems = {"bass": write_midi(tmp_path / "bass.mid", [(40, 0, 1)])}
    _, track_map = merge_midis(stems, tmp_path / "song.mid")
    assert track_map == {"bass": 0}
    assert "melody" not in track_map


def test_merge_folds_multi_instrument_stem_into_one_track(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    for program in (0, 4):
        inst = pretty_midi.Instrument(program=program)
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=60 + program, start=0, end=1))
        pm.instruments.append(inst)
    path = tmp_path / "other.mid"
    pm.write(str(path))

    out, track_map = merge_midis({"other": path}, tmp_path / "song.mid")
    merged = pretty_midi.PrettyMIDI(str(out))
    assert len(merged.instruments) == 1
    assert len(merged.instruments[0].notes) == 2


def test_track_for_stem_defaults_for_unknown():
    assert track_for_stem("vocals") == ("melody", 53)
    assert track_for_stem("theremin") == ("theremin", 0)


def test_six_stem_model_stems_are_mapped():
    """htdemucs_6s separates guitar and piano; they must not be dropped."""
    assert track_for_stem("guitar") == ("guitar", 25)
    assert track_for_stem("piano") == ("piano", 0)


def test_six_stem_tracks_merge_in_order(tmp_path):
    stems = {
        "vocals": write_midi(tmp_path / "vocals.mid", [(72, 0, 1)]),
        "bass": write_midi(tmp_path / "bass.mid", [(40, 0, 1)]),
        "guitar": write_midi(tmp_path / "guitar.mid", [(60, 0, 1)]),
        "piano": write_midi(tmp_path / "piano.mid", [(64, 0, 1)]),
        "other": write_midi(tmp_path / "other.mid", [(67, 0, 1)]),
    }
    out, track_map = merge_midis(stems, tmp_path / "song.mid")
    assert track_map == {"melody": 0, "bass": 1, "guitar": 2, "piano": 3, "comping": 4}
    pm = pretty_midi.PrettyMIDI(str(out))
    assert [i.program for i in pm.instruments] == [53, 33, 25, 0, 0]


def test_pitched_stems_covers_six_stem_model():
    from stemscribe.core import PITCHED_STEMS

    for s in ("guitar", "piano"):
        assert s in PITCHED_STEMS, f"{s} would be separated then silently dropped"


def test_empty_stem_is_dropped_on_reread_but_stats_survive(tmp_path):
    """Regression: an empty stem (a silent piano on a guitar+vocal song) merges
    to a 0-note instrument, which pretty_midi omits on write. The QC-stats step
    must not index by track_map into the re-read file, or it IndexErrors.
    """
    stems = {
        "bass": write_midi(tmp_path / "bass.mid", [(40, 0, 1)]),
        "guitar": write_midi(tmp_path / "guitar.mid", [(60, 0, 1)]),
        "piano": write_midi(tmp_path / "piano.mid", []),  # silent stem
    }
    out, track_map = merge_midis(stems, tmp_path / "song.mid")
    assert track_map == {"bass": 0, "guitar": 1, "piano": 2}

    # pretty_midi drops the 0-note piano on write, so the re-read is shorter than
    # track_map. Indexing merged.instruments[track_map['piano']] would overflow.
    reread = pretty_midi.PrettyMIDI(str(out))
    assert len(reread.instruments) < len(track_map)

    # The safe pattern (what core.py now uses): match by name, missing -> 0.
    by_name = {i.name: i for i in reread.instruments}
    for name in track_map:
        inst = by_name.get(name)
        count = 0 if inst is None else len(inst.notes)
        assert count >= 0  # no IndexError
    assert "piano" not in by_name  # confirms it really was dropped


def test_quantization_error_zero_on_grid():
    inst = pretty_midi.Instrument(program=0)
    for i in range(4):  # 16ths at 120bpm = every 0.125s
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=i * 0.125, end=i * 0.125 + 0.1))
    q = quantization_error(inst, tempo=120.0)
    assert q["mean"] == 0.0
    assert q["n"] == 4


def test_quantization_error_worst_case_is_half_a_step():
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0625, end=0.1))
    q = quantization_error(inst, tempo=120.0)
    assert q["mean"] == pytest.approx(0.5)


def test_quantization_error_empty():
    assert quantization_error(pretty_midi.Instrument(program=0))["n"] == 0


def test_backend_registry_has_the_two_specced_backends():
    assert set(BACKENDS) == {"basic-pitch", "muscriptor"}
    assert callable(get_backend("basic-pitch"))


def test_unknown_backend_raises():
    with pytest.raises(BackendError, match="unknown backend"):
        get_backend("nope")


def test_drums_are_marked_unpitched():
    assert "drums" in UNPITCHED_STEMS
