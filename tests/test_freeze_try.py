"""scripts/freeze_try.py: one finished stemscribe run -> the static /try/ page's data."""
import importlib.util
import json
import pathlib
import wave

import pretty_midi
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("freeze_try", ROOT / "scripts" / "freeze_try.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _silent_wav(path: pathlib.Path, seconds: float = 1.0, sr: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))


def _job(tmp_path: pathlib.Path, manifest: dict | None = None, seconds: float = 1.0) -> pathlib.Path:
    job = tmp_path / "job"
    _silent_wav(job / "stems" / "bass.wav", seconds)
    _silent_wav(job / "stems" / "drums.wav", seconds)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    bass = pretty_midi.Instrument(program=33, name="bass")
    bass.notes.append(pretty_midi.Note(velocity=90, pitch=40, start=0.5, end=0.75))
    bass.notes.append(pretty_midi.Note(velocity=80, pitch=43, start=0.75, end=0.95))
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.5, end=0.6))
    pm.instruments += [bass, drums]
    pm.write(str(job / "song.mid"))
    if manifest is not None:
        (job / "manifest.json").write_text(json.dumps(manifest))
    return job


def test_freeze_writes_parts_notes_and_audio(tmp_path):
    out = tmp_path / "dist"
    _load().freeze(_job(tmp_path), out, encode=False, title="Test song")

    data = json.loads((out / "try" / "data.json").read_text())
    assert data["title"] == "Test song"
    assert data["meter"] == "4/4"
    assert data["bpm"] == pytest.approx(120, abs=0.5)
    assert data["bars"] and all(isinstance(b, (int, float)) for b in data["bars"])
    assert data["duration"] == pytest.approx(1.0, abs=0.01)

    parts = {p["id"]: p for p in data["parts"]}
    assert sorted(parts) == ["bass", "drums"]
    assert parts["bass"]["name"] == "bass" and parts["drums"]["name"] == "drums"
    assert parts["drums"]["drums"] is True and parts["bass"]["drums"] is False
    assert len(parts["bass"]["notes"]) == 2
    for p in data["parts"]:
        assert all(len(n) == 4 and all(isinstance(x, (int, float)) for x in n) for n in p["notes"])
        assert (out / "try" / p["audio"]).is_file()
    assert parts["bass"]["notes"][0] == [0.5, 0.75, 40, 90]
    assert (out / "try" / data["mix"]).is_file()


def test_freeze_puts_notes_and_bars_on_the_stems_timeline(tmp_path):
    # a trimmed run: the MIDI is on the original file's timeline, the stems start `offset` in
    manifest = {"prepared_audio": {"offset": 0.25},
                "grid": {"fitted": True, "bpm": 120.0, "first_bar": 0.5, "meter": "3/4"}}
    out = tmp_path / "dist"
    _load().freeze(_job(tmp_path, manifest, seconds=4.0), out, encode=False)

    data = json.loads((out / "try" / "data.json").read_text())
    assert data["title"] == "song"
    assert len(data["bars"]) == 3                               # 0.25, 1.75, 3.25 within 4 s
    assert data["meter"] == "3/4"
    assert data["bars"][0] == pytest.approx(0.25)          # 0.5 - offset
    assert data["bars"][1] - data["bars"][0] == pytest.approx(1.5)  # three beats at 120
    bass = next(p for p in data["parts"] if p["id"] == "bass")
    assert bass["notes"][0][:2] == pytest.approx([0.25, 0.5])


def test_freeze_copies_the_page_and_the_design_system(tmp_path):
    out = tmp_path / "dist"
    _load().freeze(_job(tmp_path), out, encode=False)
    for f in ("try/index.html", "try/try.js", "try/try.css",
              "vendor/design/tokens.css", "vendor/design/steprail.js"):
        assert (out / f).is_file(), f
