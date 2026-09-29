"""The Hugging Face Space: options, the run's outputs, clean-up and staging. The
pipeline is stubbed, as in test_web.py; what matters is the endpoint's contract with
the page (file names in the JSON match the returned files) and what it refuses."""
import json
import os
import pathlib
import sys
import time

import pretty_midi
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "space"))
sys.path.insert(0, str(ROOT / "scripts"))

import split  # noqa: E402
from stemscribe import grid as G  # noqa: E402


def test_empty_options_are_the_defaults_with_the_section_capped():
    opts = split.parse_options("{}")
    assert opts["backend"] == "muscriptor" and opts["meter"] == "4/4"
    assert opts["duration"] == split.MAX_SECONDS
    assert split.parse_options(None) == opts == split.parse_options("")


def test_a_long_section_is_cut_to_the_online_limit():
    opts = split.parse_options(json.dumps({"start": 12.5, "duration": 300}))
    assert opts["start"] == 12.5 and opts["duration"] == split.MAX_SECONDS
    assert split.parse_options({"duration": 10})["duration"] == 10


@pytest.mark.parametrize("bad, word", [
    ('{"backend": "klangio"}', "backend"),
    ('{"meter": "7/5"}', "meter"),
    ('{"downbeat": 5}', "downbeat"),
    ('{"meter": "3/4", "downbeat": 4}', "downbeat"),
    ('{"snap": "yes"}', "snap"),
    ('{"tempo": 5}', "tempo"),
    ('{"mono_stems": ["drums"]}', "mono_stems"),
    ('{"url": "https://example.com/a.mp3"}', "unknown"),
    ("[1, 2]", "object"),
    ("{nope", "JSON"),
])
def test_bad_options_are_refused_with_a_reason(bad, word):
    with pytest.raises(split.OptionError, match=word):
        split.parse_options(bad)


def test_the_pages_options_are_accepted_as_sent():
    """What the page's onlineOptions(collectOptions()) sent in a headless browser run."""
    sent = {"backend": "muscriptor", "demucs_model": "htdemucs", "include_vocals_melody": True,
            "mono_stems": [], "cleanup": True, "de_overlap": True, "trim_silence": True,
            "strip_metadata": True, "max_duration_beats": 2, "velocity_floor": 15, "legato": False,
            "legato_max_gap_beats": 0.25, "tempo": None, "meter": "4/4", "downbeat": 2,
            "snap": False, "start": 30, "duration": 15, "stems_audio": True}
    opts = split.parse_options(json.dumps(sent))
    assert opts["start"] == 30 and opts["duration"] == 15 and opts["downbeat"] == 2


def test_downbeat_and_meter_pass_through():
    opts = split.parse_options({"meter": "9/8:2+2+2+3", "downbeat": 3, "snap": True})
    assert opts["meter"] == str(G.Meter.parse("9/8:2+2+2+3"))
    assert opts["downbeat"] == 3 and opts["snap"] is True


def test_gpu_seconds_stays_inside_zerogpu_bounds_and_grows_with_work():
    fast = split.gpu_seconds(split.parse_options({"backend": "basic-pitch", "duration": 5}))
    slow = split.gpu_seconds(split.parse_options({"demucs_model": "htdemucs_6s"}))
    assert 30 <= fast <= slow <= 120


def _fake_process(calls):
    def fake(audio, out_dir, **kw):
        calls.append(kw)
        out = pathlib.Path(out_dir)
        (out / "stems").mkdir(parents=True, exist_ok=True)
        pm = pretty_midi.PrettyMIDI()
        bass = pretty_midi.Instrument(program=33, name="bass")
        bass.notes.append(pretty_midi.Note(velocity=80, pitch=40, start=1.0, end=1.5))
        kit = pretty_midi.Instrument(program=0, name="drums", is_drum=True)
        kit.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
        pm.instruments += [bass, kit]
        midi = out / "song.mid"
        G.stamp(pm, G.Grid(114.0, 0.5)).write(str(midi))
        stems = {}
        for s in ("bass", "drums"):
            stems[s] = out / "stems" / f"{s}.wav"
            stems[s].write_bytes(b"RIFF")
        inst = out / "song_instrumental.mp3"
        inst.write_bytes(b"ID3")
        man = out / "manifest.json"
        man.write_text("{}")

        class R:
            midi_path, stem_paths, instrumental_path, manifest_path = midi, stems, inst, man
            track_map = {"bass": 0}
            manifest = {"tracks": {"bass": {"note_count": 1}}, "timings_sec": {}, "grid": {"fitted": True}}
            warnings = ["w"]
            tempo = prepared = None
        return R()
    return fake


def test_run_returns_files_the_json_names(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(split, "process", _fake_process(calls))
    monkeypatch.setattr(split, "_mp3", lambda src, dst: (dst.write_bytes(b"ID3"), dst)[1])
    opts = split.parse_options({"backend": "basic-pitch", "downbeat": 2, "snap": True})
    out = split.run(tmp_path / "in.mp3", opts, tmp_path / "work")

    kw = calls[0]
    assert kw["backend"] == "basic-pitch" and kw["downbeat"] == 2 and kw["snap"] is True
    assert kw["cache"] is False                       # no visitor's audio reaches another's run
    assert kw["prepare"].duration == split.MAX_SECONDS

    res = out["result"]
    json.dumps(res)                                   # plain JSON, all the way down
    names = {p.name for p in out["files"]}
    assert res["midi"] == out["midi"].name == "song.mid"
    assert res["instrumental"] in names and res["manifest"] in names
    assert set(res["stems"].values()) <= names
    assert [p["file"] for p in res["parts"]["parts"]] == [p.name for p in out["parts"]]
    assert [(p["name"], p["drum"]) for p in res["parts"]["parts"]] == [("bass", False), ("drums", True)]
    one = pretty_midi.PrettyMIDI(str(out["parts"][0]))
    assert [i.name for i in one.instruments] == ["bass"]
    assert one.get_downbeats()[:2] == pytest.approx(
        pretty_midi.PrettyMIDI(str(out["midi"])).get_downbeats()[:2], abs=2e-3)


def test_run_without_stem_audio_sends_no_stems(tmp_path, monkeypatch):
    monkeypatch.setattr(split, "process", _fake_process([]))
    out = split.run(tmp_path / "in.mp3", split.parse_options({"stems_audio": False}), tmp_path / "w")
    assert out["result"]["stems"] == {}
    assert not any(p.suffix == ".mp3" and p.parent.name == "stems" for p in out["files"])


def test_sweep_removes_only_old_runs(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(), new.mkdir()
    past = time.time() - 7200
    os.utime(old, (past, past))
    assert split.sweep(tmp_path, 1800) == 1
    assert not old.exists() and new.exists()
    assert split.sweep(tmp_path / "missing", 1800) == 0


def test_stage_space_copies_the_package_without_the_web_ui(tmp_path):
    import stage_space
    files = stage_space.stage(tmp_path / "dist")
    for need in ("app.py", "split.py", "requirements.txt", "packages.txt", "README.md",
                 "LICENSE", "stemscribe/__init__.py", "stemscribe/core.py"):
        assert need in files
    assert not any(f.startswith("stemscribe/web") or "__pycache__" in f for f in files)
    readme = (tmp_path / "dist" / "README.md").read_text()
    assert "sdk: gradio" in readme and "non-commercial" in readme
    # staging again over a Space checkout keeps its .git
    (tmp_path / "dist" / ".git").mkdir()
    stage_space.stage(tmp_path / "dist")
    assert (tmp_path / "dist" / ".git").is_dir()


def test_space_requirements_pin_a_zerogpu_torch():
    req = (ROOT / "space" / "requirements.txt").read_text()
    assert "torch==2.8.0" in req and "torchcodec" not in req.replace("torchcodec stays", "")


def test_separate_uses_a_preloaded_model_as_is(tmp_path, monkeypatch):
    """The Space puts demucs on cuda at start-up; separate() must not reload it or
    move it back to the CPU."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("demucs")
    import demucs.apply
    import demucs.audio
    import demucs.pretrained
    from stemscribe import separate as S

    class Model:
        samplerate, audio_channels, sources = 44100, 2, ["drums", "bass"]
        moved = False

        def cpu(self):
            Model.moved = True
            return self

        def eval(self):
            return self

    monkeypatch.setitem(S.PRELOADED, "htdemucs", Model())
    monkeypatch.setattr(demucs.pretrained, "get_model", lambda name: pytest.fail("reloaded"))
    monkeypatch.setattr(demucs.audio, "AudioFile",
                        lambda p: type("A", (), {"read": lambda self, **k: torch.ones(2, 100)})())
    monkeypatch.setattr(demucs.apply, "apply_model", lambda m, wav, **k: torch.zeros(1, 2, 2, 100))
    paths = S.separate(tmp_path / "a.wav", tmp_path / "stems", device="cpu")
    assert sorted(paths) == ["bass", "drums"] and not Model.moved
