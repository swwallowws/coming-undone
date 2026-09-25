"""ADT_STR drums (class numbers mapped to standard GM) and the sparse-stem fallback."""
import pretty_midi
import pytest

from stemscribe import backends, merge


def test_adt_str_classes_map_to_the_first_standard_gm_drum():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, is_drum=True)
    inst.notes = [pretty_midi.Note(100, p, 0.0, 0.1) for p in (36, 43, 44, 55, 99)]
    pm.instruments.append(inst)
    out = backends.to_standard_gm(pm)
    assert [n.pitch for n in out.instruments[0].notes] == [36, 44, 46, 69, 99]
    assert out.instruments[0].is_drum and out.instruments[0].name == "drums"


def test_adt_str_is_a_drum_backend_pinned_to_a_revision():
    assert backends.DRUM_BACKENDS["adt-str"] is backends.transcribe_adt_str
    assert len(backends.ADT_STR_REVISION) == 40


def test_drums_merge_as_a_drum_track(tmp_path):
    for stem, drum in (("bass", False), ("drums", True)):
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0, is_drum=drum)
        inst.notes = [pretty_midi.Note(90, 36 if drum else 40, 0.0, 0.2)]
        pm.instruments.append(inst)
        pm.write(str(tmp_path / f"{stem}.mid"))
    path, track_map = merge.merge_midis({"bass": tmp_path / "bass.mid", "drums": tmp_path / "drums.mid"},
                                        tmp_path / "out.mid")
    back = pretty_midi.PrettyMIDI(str(path))
    drums = back.instruments[track_map["drums"]]
    assert drums.is_drum and drums.name == "drums"


@pytest.mark.parametrize("notes,seconds,rms,want", [
    (3, 240.0, 0.05, True),        # MuScriptor's 3-note bass on a 4-minute song
    (200, 240.0, 0.05, False),
    (3, 240.0, 0.0005, False),     # the stem really is silent
    (3, 8.0, 0.05, False),         # 8 s at 114 BPM is 4 bars: 3 notes is plenty
])
def test_sparse_stem_fallback_rule(notes, seconds, rms, want):
    assert backends.is_sparse(notes, seconds, bpm=114.0, rms=rms) is want
