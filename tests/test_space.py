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


def test_gpu_seconds_follow_the_clip_and_the_work():
    # the first real run: 20 s, 4 stems, muscriptor on 3 stems + drums took 29.8 s of
    # GPU work (separate 1.6 + transcribe 28.2); the request keeps a margin over it
    assert split.gpu_seconds(20) == 55 and split.gpu_seconds(20) > 29.8 * 1.5
    assert split.gpu_seconds(30) == 68 < 93                   # the old flat request
    assert split.gpu_seconds(5) < split.gpu_seconds(20) < split.gpu_seconds(30)
    assert split.gpu_seconds(600) == split.gpu_seconds(30)    # never past the section cap
    assert split.gpu_seconds(30, backend="basic-pitch") < 30  # basic-pitch stays on the CPU
    assert split.gpu_seconds(30, include_vocals_melody=False) < split.gpu_seconds(30)
    assert split.gpu_seconds(30, "htdemucs_6s") <= split.GPU_MAX


def test_runs_per_day_count_what_zerogpu_charges():
    # a run starts while the quota left covers the request, and costs 1.5 times it:
    # the first run asked 93 s and used up a signed-out 120 s
    assert split.runs_per_day(93, 120) == 1
    assert split.runs_per_day(68, split.QUOTA_SIGNED_OUT) == 1
    assert split.runs_per_day(68, split.QUOTA_FREE_ACCOUNT) == 3
    assert split.runs_per_day(43, split.QUOTA_SIGNED_OUT) == 2   # a 10 s section
    assert split.runs_per_day(130, 120) == 0


def test_the_page_computes_the_same_budget_as_the_space():
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    cases = [(s, m, b, v) for s in (0, 3.3, 10, 17.5, 20, 30, 45)
             for m in ("htdemucs", "htdemucs_6s") for b in ("muscriptor", "basic-pitch")
             for v in (True, False)]
    engines = ROOT / "src" / "stemscribe" / "web" / "static" / "js" / "engines.js"
    script = (f"const E = await import({json.dumps(engines.as_uri())});\n"
              f"const cases = {json.dumps(cases)};\n"
              "console.log(JSON.stringify(cases.map(([s, m, b, v]) => {\n"
              "  const r = E.onlineGpuSeconds(s, { demucs_model: m, backend: b, include_vocals_melody: v });\n"
              "  return [r, E.onlineRunsPerDay(r, 120), E.onlineRunsPerDay(r, 300)];\n"
              "})));\n")
    p = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    want = []
    for s, m, b, v in cases:
        r = split.gpu_seconds(s, m, b, v, True)
        want.append([r, split.runs_per_day(r, split.QUOTA_SIGNED_OUT),
                     split.runs_per_day(r, split.QUOTA_FREE_ACCOUNT)])
    assert json.loads(p.stdout) == want


@pytest.mark.parametrize("backend", ["muscriptor", "basic-pitch"])
def test_only_separation_and_transcription_run_on_the_gpu(tmp_path, monkeypatch, backend):
    """The real pipeline with the models faked: the GPU hook gets one plain-data job
    (it crosses into a ZeroGPU worker process by pickle), and prepare, tempo, the grid
    and the part files run outside it."""
    import pickle

    import numpy as np
    import soundfile as sf

    from stemscribe import backends, core
    from stemscribe import grid as grid_mod
    from stemscribe import prepare as prep_mod
    from stemscribe import tempo as tempo_mod

    sr, secs = 8000, 12.0

    def wav(p):
        y = (np.random.default_rng(0).standard_normal(int(sr * secs)) * 0.1).astype(np.float32)
        sf.write(str(p), y, sr)
        return p

    where: list[tuple[str, bool]] = []
    inside = {"on": False}

    def note(name):
        where.append((name, inside["on"]))

    def separate(audio, out, model_name=None, device=None):
        note("separate")
        out.mkdir(parents=True, exist_ok=True)
        return {s: wav(out / f"{s}.wav") for s in ("drums", "bass", "other", "vocals")}

    def transcriber(name, drum=False):
        def fn(stem_wav, out_mid, **_):
            note(name)
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(0, is_drum=drum)
            inst.notes = [pretty_midi.Note(90, 36 if drum else 60, 0.5 * i, 0.5 * i + 0.2)
                          for i in range(20)]
            pm.instruments.append(inst)
            pm.write(str(out_mid))
            return out_mid
        return fn

    def spy(mod, attr):
        real = getattr(mod, attr)

        def wrapped(*a, **k):
            note(attr)
            return real(*a, **k)
        monkeypatch.setattr(mod, attr, wrapped)

    monkeypatch.setattr("stemscribe.separate.separate", separate)
    monkeypatch.setitem(backends.BACKENDS, "muscriptor", transcriber("muscriptor"))
    monkeypatch.setitem(backends.BACKENDS, "basic-pitch", transcriber("basic-pitch"))
    monkeypatch.setitem(backends.DRUM_BACKENDS, "adt-str", transcriber("adt-str", drum=True))
    monkeypatch.setattr(backends, "drums_available", lambda: True)
    for mod, attr in ((prep_mod, "prepare_audio"), (tempo_mod, "resolve_tempo"), (grid_mod, "apply")):
        spy(mod, attr)
    spy(split, "_parts")

    jobs = []

    def gpu(job):
        jobs.append(job)
        job = pickle.loads(pickle.dumps(job))          # what ZeroGPU does with it
        inside["on"] = True
        try:
            out = core.gpu_stage(job)
        finally:
            inside["on"] = False
        return pickle.loads(pickle.dumps(out))

    opts = split.parse_options({"stems_audio": False, "trim_silence": False, "backend": backend})
    out = split.run(wav(tmp_path / "in.wav"), opts, tmp_path / "work", gpu=gpu)

    assert len(jobs) == 1
    on_gpu = {n for n, i in where if i}
    off_gpu = {n for n, i in where if not i}
    if backend == "muscriptor":
        assert on_gpu == {"separate", "muscriptor", "adt-str"}
    else:                                   # basic-pitch runs on the CPU anyway
        assert on_gpu == {"separate", "adt-str"} and "basic-pitch" in off_gpu
    assert {"prepare_audio", "resolve_tempo", "apply", "_parts"} <= off_gpu
    assert not off_gpu & on_gpu
    res = out["result"]
    assert {p["name"] for p in res["parts"]["parts"]} == {"melody", "bass", "comping", "drums"}
    assert res["gpu"] == split.gpu_estimate(secs, opts)
    assert res["gpu"]["request"] == split.gpu_seconds(secs, backend=backend)
    assert res["gpu"]["seconds"] == secs
    assert "gpu_call" in res["timings"]


def test_the_space_decorates_only_the_gpu_stage(tmp_path, monkeypatch):
    """app.py with spaces and gradio stubbed: the one @spaces.GPU function is the GPU
    stage, its duration comes from the job, and the endpoint runs the rest outside."""
    import types

    gpu_calls = []

    def GPU(duration=None, **_):
        def deco(fn):
            def wrapped(job):
                gpu_calls.append(duration(job))
                return fn(job)
            wrapped.duration = duration
            return wrapped
        return deco

    class Anything:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __getattr__(self, name): return lambda *a, **k: None

    class GrError(Exception):
        pass

    progress_seen = []
    gr = types.ModuleType("gradio")
    gr.Error = GrError
    gr.Progress = lambda: (lambda value=None, desc=None: progress_seen.append(desc))
    for name in ("Blocks", "Markdown", "Row", "Column", "File", "Textbox", "Button"):
        setattr(gr, name, Anything)
    monkeypatch.setitem(sys.modules, "gradio", gr)
    monkeypatch.setitem(sys.modules, "spaces", types.SimpleNamespace(GPU=GPU))
    monkeypatch.setenv("COMING_UNDONE_PRELOAD", "0")
    monkeypatch.delitem(sys.modules, "app", raising=False)
    import app

    assert app._gpu_stage.duration is app._gpu_seconds
    seen = {}

    def fake_run(audio, opts, work, progress=None, gpu=None):
        seen["gpu"] = gpu
        return {"result": {}, "midi": pathlib.Path("a.mid"), "parts": [], "files": []}
    monkeypatch.setattr(app.split, "run", fake_run)
    monkeypatch.setattr(app, "WORK_ROOT", tmp_path)
    app.split_api(str(tmp_path / "in.wav"), "{}", progress=gr.Progress())
    assert seen["gpu"] is not None and seen["gpu"] is not app._gpu_stage   # wrapped with a progress note

    job = {"audio": str(tmp_path / "x.wav"), "demucs_model": "htdemucs", "backend": "muscriptor",
           "include_vocals_melody": True, "drums": "adt-str"}
    import numpy as np
    import soundfile as sf
    sf.write(job["audio"], np.zeros(8000 * 20, dtype=np.float32), 8000)
    monkeypatch.setattr(app._core, "gpu_stage", lambda j: {"timings": {}})
    seen["gpu"](job)
    assert gpu_calls == [split.gpu_seconds(20, drums=split._backends.drums_available())]
    assert "waiting for a GPU" in progress_seen


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
