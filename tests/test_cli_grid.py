"""The grid flags on `stemscribe`, and `stemscribe-grid` for any existing MIDI."""
import json

import numpy as np
import pretty_midi

from stemscribe import cli
from stemscribe import grid as G

BPM = 114.0
BEAT = 60.0 / BPM


def _loose_midi(path):
    """Chords on a real 114 BPM grid, written under a 120 BPM placeholder tempo map, 12 ms late
    with jitter (a transcription nobody gridded)."""
    rng = np.random.default_rng(0)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(0, name="comping")
    prog = [(60, 64, 67), (57, 60, 64), (53, 57, 60), (55, 59, 62)]
    for b in range(24):
        for k in range(4):
            t = 0.7 + (b * 4 + k) * BEAT + 0.012 + rng.uniform(-0.004, 0.004)
            for p in prog[b % 4]:
                inst.notes.append(pretty_midi.Note(80, p, t, t + BEAT * 0.9))
    pm.instruments.append(inst)
    pm.write(str(path))
    return path


def test_grid_command_fits_snaps_and_writes_a_report(tmp_path, capsys):
    src = _loose_midi(tmp_path / "in.mid")
    out = tmp_path / "out.mid"
    assert cli.grid_main([str(src), "-o", str(out), "--snap"]) == 0
    report = json.loads(out.with_suffix(".grid.json").read_text())
    assert report["bpm"] == BPM or abs(report["bpm"] - BPM) < 0.1
    pm = pretty_midi.PrettyMIDI(str(out))
    g = G.Grid(report["bpm"], report["anchor"])
    assert np.max(np.abs(g.signed([n.start for n in pm.instruments[0].notes]))) < 2e-3
    assert "BPM" in capsys.readouterr().out


def test_grid_command_with_a_tempo_hint_and_downbeat(tmp_path):
    src = _loose_midi(tmp_path / "in.mid")
    out = tmp_path / "out.mid"
    assert cli.grid_main([str(src), "-o", str(out), "--tempo", "115", "--downbeat", "2"]) == 0
    report = json.loads(out.with_suffix(".grid.json").read_text())
    assert report["downbeat_override"] == 2 and abs(report["bpm"] - BPM) < 0.1


def test_grid_command_errors_cleanly(tmp_path, capsys):
    assert cli.grid_main([str(tmp_path / "missing.mid"), "-o", str(tmp_path / "o.mid")]) == 1
    assert "stemscribe-grid:" in capsys.readouterr().err


def test_pipeline_parser_has_the_grid_flags():
    a = cli.build_parser().parse_args(["x.mp3", "-o", "out", "--snap", "--downbeat", "3",
                                       "--drums", "none", "--no-fallback", "--no-grid"])
    assert a.snap and a.downbeat == 3 and a.drums == "none" and not a.fallback and not a.grid
