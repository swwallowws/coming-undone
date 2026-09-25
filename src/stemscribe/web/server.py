"""Local web UI for stemscribe.

Deliberately local-only: it runs demucs on your machine, reads and writes your
files, and has no auth. Bind it to localhost and keep it there.

Design note -- the tempo question: the whole point is that you should not have to
think about tempo. A run detects it and proceeds. The alternates only surface
*after* the fact, and applying one is instant (see /tempo below), so the fast
path never asks you anything.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from .. import __version__
from .. import grid as _grid
from ..backends import BACKENDS, NONCOMMERCIAL_BACKENDS
from ..cleanup import CleanupParams
from ..core import _jsonable, process
from ..fetch import is_url
from ..prepare import PrepareParams

STATIC = pathlib.Path(__file__).parent / "static"

app = FastAPI(title="stemscribe", version=__version__)


@dataclass
class Job:
    id: str
    dir: pathlib.Path
    name: str
    status: str = "queued"  # queued | running | done | error
    error: str | None = None
    result: dict[str, Any] | None = None
    log: list[dict] = field(default_factory=list)
    #: Streams read `log` by index under this, so a late subscriber replays from
    #: the start and a live one blocks -- with no event delivered twice.
    cv: threading.Condition = field(default_factory=threading.Condition)

    def emit(self, stage: str, message: str) -> None:
        with self.cv:
            self.log.append({"stage": stage, "message": message})
            self.cv.notify_all()

    def finished(self) -> bool:
        return self.status in ("done", "error")

    def close(self) -> None:
        with self.cv:
            self.cv.notify_all()


JOBS: dict[str, Job] = {}
JOBS_ROOT = pathlib.Path(tempfile.gettempdir()) / "stemscribe-jobs"


def _run(job: Job, audio: pathlib.Path, opts: dict) -> None:
    job.status = "running"
    try:
        res = process(
            audio,
            out_dir=job.dir / "out",
            audio_format=opts["audio_format"],
            backend=opts["backend"],
            include_vocals_melody=opts["include_vocals_melody"],
            cleanup=opts["cleanup"],
            demucs_model=opts["demucs_model"],
            mono_stems=opts["mono_stems"],
            cleanup_params=CleanupParams(
                de_overlap=opts["de_overlap"],
                max_duration_beats=opts["max_duration_beats"],
                velocity_floor=opts["velocity_floor"],
                legato=opts["legato"],
                legato_max_gap_beats=opts["legato_max_gap_beats"],
            ),
            prepare=PrepareParams(
                trim_silence=opts["trim_silence"],
                strip_metadata=opts["strip_metadata"],
                start=opts["start"],
                duration=opts["duration"],
            ),
            tempo=opts["tempo"],
            keep_stems=True,
            instrumental=True,
            progress=job.emit,
        )
        job.result = _payload(job, res)
        # Emit before flipping status: a stream exits once status is terminal,
        # so setting it first can drop the final event.
        job.emit("done", "finished")
        job.status = "done"
    except Exception as e:  # surface it in the UI rather than only the console
        job.error = f"{type(e).__name__}: {e}"
        job.emit("error", job.error)
        job.status = "error"
        traceback.print_exc()
    finally:
        job.close()  # release any SSE stream still waiting


def _rel(job: Job, p) -> str | None:
    if not p:
        return None
    try:
        return str(pathlib.Path(p).relative_to(job.dir))
    except ValueError:
        return None


def _payload(job: Job, res) -> dict:
    m = res.manifest
    return {
        "tempo": res.tempo.as_dict() if res.tempo else None,
        "grid": m.get("grid"),
        "prepared": res.prepared.as_dict() if res.prepared else None,
        "source": res.source.as_dict() if res.source else None,
        "track_map": res.track_map,
        "tracks": m["tracks"],
        "timings": m["timings_sec"],
        "warnings": res.warnings,
        "midi": _rel(job, res.midi_path),
        "instrumental": _rel(job, res.instrumental_path),
        "stems": {k: _rel(job, v) for k, v in res.stem_paths.items()},
        "manifest": _rel(job, res.manifest_path),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/api/config")
def config() -> dict:
    return {
        "version": __version__,
        "backends": [
            {"name": n, "noncommercial": n in NONCOMMERCIAL_BACKENDS}
            for n in sorted(BACKENDS)
        ],
    }


@app.post("/api/jobs")
async def create_job(
    file: UploadFile | None = None,
    url: str = Form(""),
    audio_format: str = Form("native"),
    backend: str = Form("muscriptor"),
    demucs_model: str = Form("htdemucs"),
    include_vocals_melody: bool = Form(True),
    cleanup: bool = Form(True),
    de_overlap: bool = Form(True),
    max_duration_beats: float = Form(2.0),
    velocity_floor: int = Form(15),
    legato: bool = Form(False),
    legato_max_gap_beats: float = Form(0.25),
    quantize: bool = Form(False),
    quantize_strength: float = Form(0.5),
    quantize_division: int = Form(4),
    mono_stems: str = Form(""),
    trim_silence: bool = Form(True),
    strip_metadata: bool = Form(True),
    start: float | None = Form(None),
    duration: float | None = Form(None),
    tempo: float | None = Form(None),
) -> dict:
    if backend not in BACKENDS:
        raise HTTPException(400, f"unknown backend {backend!r}")

    url = (url or "").strip()
    has_file = file is not None and file.filename
    if not has_file and not url:
        raise HTTPException(400, "provide an audio file or a URL")
    if has_file and url:
        raise HTTPException(400, "provide either a file or a URL, not both")
    if url and not is_url(url):
        raise HTTPException(400, "that does not look like a URL")

    jid = uuid.uuid4().hex[:12]
    jdir = JOBS_ROOT / jid
    (jdir / "in").mkdir(parents=True, exist_ok=True)

    if has_file:
        safe = pathlib.Path(file.filename or "input").name
        audio: str | pathlib.Path = jdir / "in" / safe
        with open(audio, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
    else:
        # process() fetches it -- so download progress streams like every other
        # stage instead of blocking this request.
        audio, safe = url, url

    job = Job(id=jid, dir=jdir, name=safe)
    JOBS[jid] = job

    opts = dict(
        audio_format=audio_format,
        backend=backend,
        demucs_model=demucs_model,
        include_vocals_melody=include_vocals_melody,
        cleanup=cleanup,
        de_overlap=de_overlap,
        max_duration_beats=max_duration_beats,
        velocity_floor=velocity_floor,
        legato=legato,
        legato_max_gap_beats=legato_max_gap_beats,
        mono_stems=tuple(s.strip() for s in mono_stems.split(",") if s.strip()),
        trim_silence=trim_silence,
        strip_metadata=strip_metadata,
        start=start,
        duration=duration,
        tempo=tempo,  # None -> detected; the UI's default
    )
    threading.Thread(target=_run, args=(job, audio, opts), daemon=True).start()
    return {"id": jid, "name": safe}


def _job(jid: str) -> Job:
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "no such job")
    return job


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict:
    job = _job(jid)
    # _jsonable here too, not only in core: pydantic dies on a stray numpy
    # scalar, and this endpoint should not depend on an upstream promise.
    return _jsonable(
        {
            "id": job.id,
            "name": job.name,
            "status": job.status,
            "error": job.error,
            "result": job.result,
            "log": job.log,
        }
    )


@app.get("/api/jobs/{jid}/events")
def events(jid: str):
    job = _job(jid)

    def stream():
        i = 0
        while True:
            with job.cv:
                # Wait only when we have caught up and there is more to come.
                while i >= len(job.log) and not job.finished():
                    job.cv.wait(timeout=1.0)
                pending = job.log[i:]
                i = len(job.log)
                done = job.finished()
            for evt in pending:
                yield f"data: {json.dumps(evt)}\n\n"
            if done and not pending:
                return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/jobs/{jid}/tempo")
def restamp_tempo(jid: str, bpm: float = Form(...)) -> dict:
    """Apply a different tempo to a finished MIDI. Instant, and lossless.

    Note times are stored in seconds, so the tempo only decides where the bar
    lines fall -- restamping moves the grid, never the notes. That is why an
    octave error is cheap to fix here and does not need a re-run.
    """
    import pretty_midi

    job = _job(jid)
    if job.status != "done" or not job.result:
        raise HTTPException(409, "job is not finished")
    if not 20 <= bpm <= 400:
        raise HTTPException(400, "bpm out of range")

    midi_path = job.dir / job.result["midi"]
    old = pretty_midi.PrettyMIDI(str(midi_path))
    grid = job.result.get("grid") or {}
    if grid.get("fitted"):          # keep bar "one" where it is; only the tempo changes
        g = _grid.Grid(bpm, grid["anchor"])
        _grid.stamp(old, g).write(str(midi_path))
        grid.update(g.as_dict())
    else:
        new = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        new.instruments = old.instruments
        new.write(str(midi_path))

    job.result["tempo"]["bpm"] = round(bpm, 2)
    job.result["tempo"]["source"] = "user"
    return {"bpm": round(bpm, 2)}


@app.post("/api/jobs/{jid}/bar")
def shift_bar(jid: str, beats: int = Form(...)) -> dict:
    """Move bar "one" by whole beats (the guess can be off). Re-stamps the tempo map
    only: no note moves, like the tempo re-stamp."""
    import pretty_midi

    job = _job(jid)
    if job.status != "done" or not job.result:
        raise HTTPException(409, "job is not finished")
    grid = job.result.get("grid") or {}
    if not grid.get("fitted"):
        raise HTTPException(409, "this job has no beat grid")
    if not -3 <= beats <= 3:
        raise HTTPException(400, "shift by -3 to 3 beats")
    midi_path = job.dir / job.result["midi"]
    g = _grid.shift(_grid.Grid(grid["bpm"], grid["anchor"]), beats)
    _grid.stamp(pretty_midi.PrettyMIDI(str(midi_path)), g).write(str(midi_path))
    grid.update({**g.as_dict(), "anchor": g.anchor,
                 "shift_beats": grid.get("shift_beats", 0) + beats})
    return {"first_bar": grid["first_bar"]}


@app.get("/api/jobs/{jid}/files/{path:path}")
def get_file(jid: str, path: str):
    job = _job(jid)
    target = (job.dir / path).resolve()
    # Never serve outside the job's own directory.
    if not str(target).startswith(str(job.dir.resolve())) or not target.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(target, filename=target.name)


def main(argv: list[str] | None = None) -> int:
    global JOBS_ROOT

    import argparse

    import uvicorn

    p = argparse.ArgumentParser(prog="stemscribe-web", description="stemscribe local web UI")
    p.add_argument("--host", default="127.0.0.1", help="default: localhost only")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--jobs-dir", default=None, help=f"default: {JOBS_ROOT}")
    args = p.parse_args(argv)

    if args.jobs_dir:
        JOBS_ROOT = pathlib.Path(args.jobs_dir).expanduser()
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.host not in ("127.0.0.1", "localhost"):
        print(f"! serving on {args.host}: this UI has no auth and runs local jobs")
    print(f"stemscribe {__version__}  ->  http://{args.host}:{args.port}")
    print(f"jobs in {JOBS_ROOT}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
