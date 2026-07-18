"""The manifest is the integration surface; it must always serialize.

Regression: the first real run separated stems, transcribed 2789 notes, mixed
the instrumental -- then died on json.dumps because pretty_midi returns
instrument.program as numpy int64. Five minutes of work lost at the last step.
"""
import json
import pathlib

import numpy as np
import pretty_midi
import pytest

from stemscribe.core import _json_default
from stemscribe.merge import quantization_error


def dumps(obj):
    return json.dumps(obj, default=_json_default)


def test_numpy_int_serializes():
    assert dumps({"program": np.int64(53)}) == '{"program": 53}'


def test_numpy_float_serializes():
    assert json.loads(dumps({"mean": np.float64(0.25)}))["mean"] == 0.25


def test_paths_serialize():
    assert json.loads(dumps({"p": pathlib.Path("/a/b.mid")}))["p"] == "/a/b.mid"


def test_genuinely_unserializable_still_raises():
    with pytest.raises(TypeError, match="not JSON serializable"):
        dumps({"x": object()})


def test_program_read_back_from_midi_serializes(tmp_path):
    """The exact shape that broke: write a MIDI, read it back, serialize."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=53, name="melody")
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=72, start=0.0, end=0.5))
    pm.instruments.append(inst)
    path = tmp_path / "song.mid"
    pm.write(str(path))

    read_back = pretty_midi.PrettyMIDI(str(path)).instruments[0]
    payload = {
        "program": read_back.program,
        "note_count": len(read_back.notes),
        "quantization_error": quantization_error(read_back),
    }
    assert json.loads(dumps(payload))["program"] == 53


def test_quantization_error_stats_serialize():
    inst = pretty_midi.Instrument(program=0)
    for i in range(8):
        start = np.float64(i * 0.13)
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=60, start=start, end=start + 0.1)
        )
    assert json.loads(dumps(quantization_error(inst)))["n"] == 8
