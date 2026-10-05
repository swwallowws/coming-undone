"""Web API tests. The pipeline itself is stubbed -- demucs is far too slow for
a unit test, and the pipeline is covered elsewhere. What matters here is the
API contract: serialization, event delivery, and not serving arbitrary files.
"""
import io
import json
import pathlib

import numpy as np
import pretty_midi
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from stemscribe.web import server as webapp  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp, "JOBS_ROOT", tmp_path / "jobs")
    webapp.JOBS.clear()
    return TestClient(webapp.app)


def make_job(tmp_path, **kw):
    job = webapp.Job(id="j1", dir=tmp_path / "j1", name="song.mp3", **kw)
    job.dir.mkdir(parents=True, exist_ok=True)
    webapp.JOBS["j1"] = job
    return job


def test_index_and_config(client):
    assert client.get("/").status_code == 200
    cfg = client.get("/api/config").json()
    names = [b["name"] for b in cfg["backends"]]
    assert "basic-pitch" in names
    assert next(b for b in cfg["backends"] if b["name"] == "muscriptor")["noncommercial"] is True


def test_page_uses_the_shared_design_system(client):
    page = client.get("/").text
    assert 'data-category="transcribe"' in page and "/vendor/design/tokens.css" in page
    css = client.get("/vendor/design/tokens.css")
    assert css.status_code == 200 and "--acc" in css.text
    assert client.get("/vendor/design/fonts/InterTight.woff2").status_code == 200


def test_page_links_a_favicon_that_resolves(client):
    page = client.get("/").text
    for href in ("/favicons/favicon.svg", "/favicons/favicon-32.png", "/favicons/apple-touch-icon.png"):
        assert href in page
        assert client.get(href).status_code == 200, href


def test_basic_pitch_is_not_flagged_noncommercial(client):
    cfg = client.get("/api/config").json()
    bp = next(b for b in cfg["backends"] if b["name"] == "basic-pitch")
    assert bp["noncommercial"] is False


def test_unknown_job_404s(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_numpy_in_result_serializes_over_http(client, tmp_path):
    """Regression: pretty_midi hands back int64 for program, which broke
    GET /api/jobs/{id} even after the manifest *file* write was fixed."""
    job = make_job(tmp_path, status="done")
    job.result = {"tracks": {"melody": {"program": np.int64(53), "note_count": 7}}}
    r = client.get("/api/jobs/j1")
    assert r.status_code == 200
    assert r.json()["result"]["tracks"]["melody"]["program"] == 53


def test_events_replay_each_event_exactly_once(client, tmp_path):
    """Regression: replaying the log AND draining a queue double-delivered."""
    job = make_job(tmp_path)
    job.emit("prepare", "preparing ...")
    job.emit("separate", "separating ...")
    job.emit("done", "finished")
    job.status = "done"

    with client.stream("GET", "/api/jobs/j1/events") as r:
        body = "".join(r.iter_text())
    msgs = [json.loads(l[6:]) for l in body.splitlines() if l.startswith("data: ")]
    assert [m["message"] for m in msgs] == [
        "preparing ...", "separating ...", "finished",
    ]


def test_events_terminate_on_error(client, tmp_path):
    job = make_job(tmp_path)
    job.emit("error", "BackendError: boom")
    job.status = "error"
    with client.stream("GET", "/api/jobs/j1/events") as r:
        body = "".join(r.iter_text())
    assert "BackendError: boom" in body


def test_file_serving_stays_inside_job_dir(client, tmp_path):
    job = make_job(tmp_path, status="done")
    (job.dir / "out").mkdir(parents=True, exist_ok=True)
    (job.dir / "out" / "song.mid").write_bytes(b"MThd")
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve me")

    assert client.get("/api/jobs/j1/files/out/song.mid").status_code == 200
    for bad in ("../secret.txt", "../../secret.txt", "/etc/passwd"):
        assert client.get(f"/api/jobs/j1/files/{bad}").status_code == 404


def test_tempo_restamp_moves_grid_not_notes(client, tmp_path):
    """The claim the UI makes: re-stamping is lossless."""
    job = make_job(tmp_path, status="done")
    out = job.dir / "out"
    out.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, name="comping")
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=1.0, end=2.0))
    pm.instruments.append(inst)
    pm.write(str(out / "song.mid"))
    job.result = {"midi": "out/song.mid", "tempo": {"bpm": 120.0, "source": "detected"}}

    r = client.post("/api/jobs/j1/tempo", data={"bpm": 92.5})
    assert r.status_code == 200

    after = pretty_midi.PrettyMIDI(str(out / "song.mid"))
    assert float(after.get_tempo_changes()[1][0]) == pytest.approx(92.5, abs=0.1)
    note = after.instruments[0].notes[0]
    assert note.start == pytest.approx(1.0, abs=0.01)  # unmoved
    assert note.end == pytest.approx(2.0, abs=0.01)
    assert job.result["tempo"]["source"] == "user"


def _gridded_job(tmp_path):
    from stemscribe import grid as G
    job = make_job(tmp_path, status="done")
    out = job.dir / "out"
    out.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="comping")
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=1.0, end=20.0))
    pm.instruments.append(inst)
    g = G.Grid(114.0, 1.5)
    G.stamp(pm, g).write(str(out / "song.mid"))
    job.result = {"midi": "out/song.mid", "tempo": {"bpm": 114.0, "source": "detected"},
                  "grid": {"fitted": True, "bpm": 114.0, "anchor": 1.5, "first_bar": 1.5}}
    return job, out


def test_bar_shift_moves_bar_one_by_a_beat_and_not_the_notes(client, tmp_path):
    job, out = _gridded_job(tmp_path)
    r = client.post("/api/jobs/j1/bar", data={"beats": 1})
    assert r.status_code == 200
    after = pretty_midi.PrettyMIDI(str(out / "song.mid"))
    assert after.get_downbeats()[1] == pytest.approx(1.5 + 60 / 114, abs=2e-3)
    assert after.instruments[0].notes[0].start == pytest.approx(1.0, abs=2e-3)
    assert job.result["grid"]["anchor"] == pytest.approx(1.5 + 60 / 114)


def test_bar_shift_without_a_grid_409s(client, tmp_path):
    job = make_job(tmp_path, status="done")
    job.result = {"midi": "out/song.mid", "tempo": {"bpm": 120.0}, "grid": {"fitted": False}}
    assert client.post("/api/jobs/j1/bar", data={"beats": 1}).status_code == 409


def test_tempo_restamp_keeps_the_bar_lines_of_a_grid(client, tmp_path):
    job, out = _gridded_job(tmp_path)
    assert client.post("/api/jobs/j1/tempo", data={"bpm": 57.0}).status_code == 200
    after = pretty_midi.PrettyMIDI(str(out / "song.mid"))
    assert float(after.get_tempo_changes()[1][-1]) == pytest.approx(57.0, abs=0.01)
    bar = 4 * 60 / 57.0                                    # 1.5 s sits inside the long pickup
    first = after.get_downbeats()[1]
    assert abs(((first - 1.5) / bar) - round((first - 1.5) / bar)) < 1e-3


def test_tempo_restamp_rejects_nonsense(client, tmp_path):
    job = make_job(tmp_path, status="done")
    job.result = {"midi": "out/song.mid", "tempo": {"bpm": 120.0}}
    assert client.post("/api/jobs/j1/tempo", data={"bpm": 5}).status_code == 400
    assert client.post("/api/jobs/j1/tempo", data={"bpm": 9000}).status_code == 400


def test_tempo_restamp_on_unfinished_job_409s(client, tmp_path):
    make_job(tmp_path, status="running")
    assert client.post("/api/jobs/j1/tempo", data={"bpm": 100}).status_code == 409


def test_unknown_backend_rejected(client, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    with open(audio, "rb") as fh:
        r = client.post("/api/jobs", files={"file": ("a.wav", fh, "audio/wav")},
                        data={"backend": "nope"})
    assert r.status_code == 400


def _post_job(client, tmp_path, monkeypatch, **data):
    """Create a job with process() stubbed; returns the response and process()'s kwargs."""
    seen = {}

    def fake_process(audio, **kw):
        seen.update(kw)
        raise RuntimeError("stub")
    monkeypatch.setattr(webapp, "process", fake_process)
    monkeypatch.setattr(webapp.threading, "Thread",
                        lambda target, args, daemon: type("T", (), {"start": lambda s: target(*args)})())
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    with open(audio, "rb") as fh:
        r = client.post("/api/jobs", files={"file": ("a.wav", fh, "audio/wav")},
                        data={"backend": "basic-pitch", **data})
    return r, seen


@pytest.mark.parametrize("spec", ["4/4", "3/4", "6/8", "9/8:2+2+2+3", "12/8"])
def test_job_meter_reaches_process_like_the_cli_flag(client, tmp_path, monkeypatch, spec):
    from stemscribe import grid as G
    r, seen = _post_job(client, tmp_path, monkeypatch, meter=spec)
    assert r.status_code == 200
    assert seen["meter"] == G.Meter.parse(spec)


def test_job_meter_defaults_to_four_four(client, tmp_path, monkeypatch):
    from stemscribe import grid as G
    r, seen = _post_job(client, tmp_path, monkeypatch)
    assert r.status_code == 200 and seen["meter"] == G.DEFAULT


def test_job_rejects_a_bad_meter(client, tmp_path, monkeypatch):
    r, seen = _post_job(client, tmp_path, monkeypatch, meter="7/5")
    assert r.status_code == 400 and "meter" in r.json()["detail"] and not seen


def test_page_offers_the_meters(client):
    page = client.get("/").text
    for v in ("4/4", "3/4", "6/8", "9/8:2+2+2+3", "12/8"):
        assert f'data-v="{v}"' in page


def test_roll_lists_each_tracks_pitched_notes(client, tmp_path):
    job = make_job(tmp_path, status="done")
    out = job.dir / "out"
    out.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI()
    mel = pretty_midi.Instrument(program=53, name="melody")
    mel.notes.append(pretty_midi.Note(velocity=100, pitch=72, start=0.5, end=1.0))
    kit = pretty_midi.Instrument(program=0, name="drums", is_drum=True)
    kit.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments += [mel, kit]
    pm.write(str(out / "song.mid"))
    job.result = {"midi": "out/song.mid"}
    roll = client.get("/api/jobs/j1/roll").json()
    assert roll["tracks"] == [{"name": "melody", "notes": [[72, 0.5, 1.0, 100]]}]
    assert roll["end"] == pytest.approx(1.0)


def _two_part_job(tmp_path):
    from stemscribe import grid as G
    job = make_job(tmp_path, status="done")
    out = job.dir / "out"
    out.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI()
    bass = pretty_midi.Instrument(program=33, name="bass")
    bass.notes.append(pretty_midi.Note(velocity=80, pitch=40, start=1.0, end=1.5))
    kit = pretty_midi.Instrument(program=0, name="drums", is_drum=True)
    kit.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    empty = pretty_midi.Instrument(program=53, name="melody")
    pm.instruments += [empty, bass, kit]
    G.stamp(pm, G.Grid(114.0, 0.5)).write(str(out / "song.mid"))
    job.result = {"midi": "out/song.mid"}
    return job, out


def test_parts_list_every_track_with_its_file_name(client, tmp_path):
    _two_part_job(tmp_path)
    r = client.get("/api/jobs/j1/parts").json()
    # the empty melody track never reaches the file, so it is no part
    assert [(p["name"], p["file"], p["drum"]) for p in r["parts"]] == [
        ("bass", "bass.mid", False), ("drums", "drums.mid", True)]
    assert r["parts"][0]["notes"] == [pytest.approx([40, 1.0, 1.5, 80], abs=2e-3)]
    assert r["parts"][1]["notes"] == [pytest.approx([36, 0.0, 0.1, 100], abs=2e-3)]


def test_part_midi_is_that_track_alone_on_the_same_grid(client, tmp_path):
    _two_part_job(tmp_path)
    r = client.get("/api/jobs/j1/parts/bass.mid")
    assert r.status_code == 200
    assert 'filename="bass.mid"' in r.headers["content-disposition"]
    one = pretty_midi.PrettyMIDI(io.BytesIO(r.content))
    whole = pretty_midi.PrettyMIDI(str(tmp_path / "j1" / "out" / "song.mid"))
    assert [i.name for i in one.instruments] == ["bass"]
    assert one.instruments[0].notes[0].start == pytest.approx(1.0, abs=2e-3)
    assert one.get_downbeats()[:4] == pytest.approx(whole.get_downbeats()[:4], abs=2e-3)


def test_part_midi_of_an_unknown_part_404s(client, tmp_path):
    _two_part_job(tmp_path)
    assert client.get("/api/jobs/j1/parts/vocals.mid").status_code == 404
    assert client.get("/api/jobs/j1/parts/..%2Fsong.mid").status_code == 404


def test_parts_of_an_unfinished_job_409s(client, tmp_path):
    make_job(tmp_path, status="running")
    assert client.get("/api/jobs/j1/parts").status_code == 409


def test_roll_of_an_unfinished_job_409s(client, tmp_path):
    make_job(tmp_path, status="running")
    assert client.get("/api/jobs/j1/roll").status_code == 409


# ---- the public page's "This computer" engine: CORS, private network access, options

PUBLIC = webapp.PUBLIC_ORIGINS[0]


@pytest.mark.parametrize("origin", [PUBLIC, "http://localhost:7860", "http://127.0.0.1:8002"])
def test_allowed_origins_get_cors_headers(client, origin):
    r = client.get("/api/config", headers={"Origin": origin})
    assert r.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost.evil.example"])
def test_other_origins_get_no_cors(client, origin):
    r = client.get("/api/config", headers={"Origin": origin})
    assert "access-control-allow-origin" not in r.headers
    pre = client.options("/api/jobs", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
    assert pre.status_code == 403


def test_preflight_answers_private_network_access(client):
    """A public https page calling http://localhost: Chrome sends this preflight first."""
    r = client.options("/api/jobs", headers={
        "Origin": PUBLIC, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Private-Network": "true"})
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == PUBLIC
    assert r.headers["access-control-allow-private-network"] == "true"
    assert "POST" in r.headers["access-control-allow-methods"]


def test_events_stream_carries_cors(client, tmp_path):
    job = make_job(tmp_path)
    job.emit("done", "finished")
    job.status = "done"
    with client.stream("GET", "/api/jobs/j1/events", headers={"Origin": PUBLIC}) as r:
        assert r.headers["access-control-allow-origin"] == PUBLIC
        assert "finished" in "".join(r.iter_text())


def test_allow_origin_flag_adds_a_site(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "PUBLIC_ORIGINS", list(webapp.PUBLIC_ORIGINS))
    monkeypatch.setattr(webapp, "JOBS_ROOT", webapp.JOBS_ROOT)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    webapp.main(["--allow-origin", "https://coming-undone.example/", "--jobs-dir", str(tmp_path)])
    assert webapp.origin_allowed("https://coming-undone.example")


def test_default_port_is_where_the_page_looks():
    page = (pathlib.Path(webapp.STATIC) / "index.html").read_text()
    assert webapp.DEFAULT_PORT == 8002 and "http://localhost:8002" in page


def test_downbeat_and_snap_reach_process(client, tmp_path, monkeypatch):
    r, seen = _post_job(client, tmp_path, monkeypatch, meter="3/4", downbeat="3", snap="true")
    assert r.status_code == 200 and seen["downbeat"] == 3 and seen["snap"] is True
    r, seen = _post_job(client, tmp_path, monkeypatch)
    assert seen["downbeat"] is None and seen["snap"] is False


def test_downbeat_past_the_meter_is_refused(client, tmp_path, monkeypatch):
    r, seen = _post_job(client, tmp_path, monkeypatch, meter="3/4", downbeat="4")
    assert r.status_code == 400 and not seen


def test_page_has_the_runs_switch_and_one_config(client):
    page = client.get("/").text
    assert "Runs:" in page and 'data-engine="online"' in page and 'data-engine="local"' in page
    assert 'space: params.get("space") || "swwallowws/coming-undone"' in page
    assert page.count("formspree.io") == 1                  # the endpoint lives in CONFIG only
    for field in ('name="email"', 'type="email"', "required", 'name="_gotcha"', 'name="_subject"'):
        assert field in page
    assert "—" not in page


def test_page_online_limit_matches_the_space():
    import re
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "space"))
    import split
    page = (pathlib.Path(webapp.STATIC) / "index.html").read_text()
    assert float(re.search(r"onlineMaxSeconds: ([\d.]+)", page).group(1)) == split.MAX_SECONDS


def test_page_is_cross_origin_isolated_for_wasm_threads(client):
    r = client.get("/")
    assert r.headers["cross-origin-opener-policy"] == "same-origin"
    assert r.headers["cross-origin-embedder-policy"] == "credentialless"


def test_browser_engine_bundle_and_basic_pitch_model_are_served(client):
    js = client.get("/vendor/browser-engine/engine.js")
    assert js.status_code == 200 and "createBrowserEngine" in js.text
    assert client.get("/vendor/browser-engine/basic-pitch/model.json").status_code == 200
    assert client.get("/vendor/browser-engine/LICENSES.txt").status_code == 200


def test_browser_models_mount_only_with_a_manifest(client, tmp_path):
    assert webapp.mount_browser_models(tmp_path / "missing") is False
    (tmp_path / "models.json").write_text('{"version": 1}')
    assert webapp.mount_browser_models(tmp_path) is True
    assert webapp.mount_browser_models(tmp_path) is True        # mounting twice is fine
    assert client.get("/browser-models/models.json").json() == {"version": 1}


def test_page_offers_the_browser_engine_and_credits_its_models(client):
    page = client.get("/").text
    assert 'data-engine="browser"' in page and 'browserEngine: here("./vendor/browser-engine/engine.js")' in page
    assert "ADT_STR" in page and "CC BY-SA 4.0" in page
    assert "CC BY-NC 4.0" in page and "non-commercial" in page
    assert "muscriptor-small" in page


def test_engines_and_gradio_client_are_served(client):
    js = client.get("/js/engines.js")
    assert js.status_code == 200 and "export function onlineEngine" in js.text
    g = client.get("/vendor/gradio-client/browser.js")
    assert g.status_code == 200 and "handle_file" in g.text
    assert client.get("/vendor/gradio-client/LICENSE").status_code == 200
