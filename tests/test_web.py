"""Web API tests. The pipeline itself is stubbed -- demucs is far too slow for
a unit test, and the pipeline is covered elsewhere. What matters here is the
API contract: serialization, event delivery, and not serving arbitrary files.
"""
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
    assert client.get("/vendor/design/fonts/Archivo.woff2").status_code == 200


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


def test_roll_of_an_unfinished_job_409s(client, tmp_path):
    make_job(tmp_path, status="running")
    assert client.get("/api/jobs/j1/roll").status_code == 409
