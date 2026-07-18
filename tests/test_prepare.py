import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from stemscribe.prepare import (
    PrepareParams,
    PreparedAudio,
    prepare_audio,
    probe_duration,
    shift_midi,
)

SR = 22050


def tone_with_silence(tmp_path, head=2.0, body=3.0, tail=1.5, name="in.wav"):
    """Silence, then a tone, then silence -- so we know exactly what to trim."""
    t = np.linspace(0, body, int(body * SR), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y = np.concatenate(
        [np.zeros(int(head * SR), np.float32), tone, np.zeros(int(tail * SR), np.float32)]
    )
    path = tmp_path / name
    sf.write(str(path), np.stack([y, y], axis=1), SR)
    return path


def test_trim_silence_reports_head_and_tail(tmp_path):
    src = tone_with_silence(tmp_path, head=2.0, body=3.0, tail=1.5)
    out = prepare_audio(src, tmp_path / "work", PrepareParams(trim_silence=True))
    assert out.trimmed_head == pytest.approx(2.0, abs=0.15)
    assert out.trimmed_tail == pytest.approx(1.5, abs=0.15)
    assert out.duration == pytest.approx(3.0, abs=0.3)


def test_offset_equals_trimmed_head(tmp_path):
    """The offset is the contract: it is what puts MIDI back on the original."""
    src = tone_with_silence(tmp_path, head=2.0)
    out = prepare_audio(src, tmp_path / "work", PrepareParams(trim_silence=True))
    assert out.offset == pytest.approx(out.trimmed_head)


def test_section_start_becomes_offset(tmp_path):
    src = tone_with_silence(tmp_path, head=0.0, body=10.0, tail=0.0)
    out = prepare_audio(
        src, tmp_path / "work", PrepareParams(trim_silence=False, start=4.0)
    )
    assert out.offset == pytest.approx(4.0)


def test_section_and_trim_offsets_add(tmp_path):
    """start=1.0 into a file whose audio begins at 2.0 -> offset 2.0, not 3.0."""
    src = tone_with_silence(tmp_path, head=2.0, body=4.0, tail=0.0)
    out = prepare_audio(
        src, tmp_path / "work", PrepareParams(trim_silence=True, start=1.0)
    )
    # 1.0s section skip + ~1.0s of remaining head silence == ~2.0s into original
    assert out.offset == pytest.approx(2.0, abs=0.2)


def test_duration_cap_shortens_output(tmp_path):
    src = tone_with_silence(tmp_path, head=0.0, body=10.0, tail=0.0)
    out = prepare_audio(
        src, tmp_path / "work", PrepareParams(trim_silence=False, duration=2.0)
    )
    assert out.duration == pytest.approx(2.0, abs=0.2)


def test_input_file_is_never_mutated(tmp_path):
    src = tone_with_silence(tmp_path, head=2.0)
    before = src.read_bytes()
    out = prepare_audio(src, tmp_path / "work", PrepareParams(trim_silence=True))
    assert src.read_bytes() == before
    assert out.path != src


def test_disabling_everything_passes_input_through(tmp_path):
    src = tone_with_silence(tmp_path)
    out = prepare_audio(
        src,
        tmp_path / "work",
        PrepareParams(
            normalize_wav=False, strip_metadata=False, trim_silence=False,
            start=None, duration=None,
        ),
    )
    assert out.path == src
    assert out.offset == 0.0
    assert out.applied == []


def test_all_silent_input_does_not_trim_to_nothing(tmp_path):
    path = tmp_path / "silent.wav"
    sf.write(str(path), np.zeros((SR * 2, 2), np.float32), SR)
    out = prepare_audio(path, tmp_path / "work", PrepareParams(trim_silence=True))
    assert out.offset == 0.0
    assert out.duration == pytest.approx(2.0, abs=0.2)


def test_normalizes_to_target_format(tmp_path):
    """Mono 22050 in -> 44100 stereo out, so downstream stages see one shape."""
    path = tmp_path / "mono.wav"
    sf.write(str(path), np.zeros(SR, np.float32), SR)
    out = prepare_audio(path, tmp_path / "work", PrepareParams(trim_silence=False))
    info = sf.info(str(out.path))
    assert info.samplerate == 44100
    assert info.channels == 2


# --- the timing contract ----------------------------------------------------
def test_shift_midi_moves_every_note(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=1.0))
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=64, start=2.0, end=2.5))
    pm.instruments.append(inst)
    path = tmp_path / "x.mid"
    pm.write(str(path))

    assert shift_midi(path, 4.2) == 2

    notes = pretty_midi.PrettyMIDI(str(path)).instruments[0].notes
    assert notes[0].start == pytest.approx(4.2, abs=0.01)
    assert notes[0].end == pytest.approx(5.2, abs=0.01)
    assert notes[1].start == pytest.approx(6.2, abs=0.01)


def test_shift_midi_preserves_durations(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=1.0, end=1.75))
    pm.instruments.append(inst)
    path = tmp_path / "x.mid"
    pm.write(str(path))
    shift_midi(path, 3.0)
    n = pretty_midi.PrettyMIDI(str(path)).instruments[0].notes[0]
    assert (n.end - n.start) == pytest.approx(0.75, abs=0.01)


def test_zero_offset_is_a_noop(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=1.0))
    pm.instruments.append(inst)
    path = tmp_path / "x.mid"
    pm.write(str(path))
    assert shift_midi(path, 0.0) == 0


def test_probe_duration(tmp_path):
    path = tmp_path / "d.wav"
    sf.write(str(path), np.zeros(SR * 3, np.float32), SR)
    assert probe_duration(path) == pytest.approx(3.0, abs=0.05)


def test_prepared_audio_serializes():
    import json

    from stemscribe.core import _json_default

    p = PreparedAudio(path="/x.wav", offset=1.5, original_duration=10.0, duration=8.0)
    assert json.loads(json.dumps(p.as_dict(), default=_json_default))["offset"] == 1.5
