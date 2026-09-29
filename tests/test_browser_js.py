"""The "In your browser" engine's pure JS parts (browser/src), tested in node when it is
here and the engine's packages are installed (npm --prefix browser install)."""
import json
import pathlib
import random
import shutil
import subprocess
import zlib

import pretty_midi
import pytest

from stemscribe.cleanup import CleanupParams, clean_instrument

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST = ROOT / "tests" / "js" / "browser.test.mjs"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(
    not NODE or not (ROOT / "browser" / "node_modules" / "protobufjs").is_dir(),
    reason="needs node and npm --prefix browser install",
)


def test_browser_engine_js():
    p = subprocess.run([NODE, str(TEST)], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr or p.stdout
    assert "ok" in p.stdout


def _notes(seed, n=80):
    rng = random.Random(seed)
    out, t = [], 0.0
    for _ in range(n):
        t += rng.choice([0.0, 0.05, 0.12, 0.25, 0.5])
        out.append({"pitch": rng.randint(40, 72), "start": round(t, 4),
                    "end": round(t + rng.choice([0.01, 0.1, 0.3, 1.2, 2.5]), 4),
                    "velocity": rng.randint(5, 120)})
    return out


@pytest.mark.parametrize("params", [
    {},
    {"legato": True},
    {"monophonic": True},
    {"velocity_floor": 40, "max_duration_beats": 1.0, "tempo": 150.0},
    {"de_overlap": False, "duration_cap": False},
])
def test_cleanup_port_matches_python(tmp_path, params):
    notes = _notes(zlib.crc32(json.dumps(params, sort_keys=True).encode()))
    inst = pretty_midi.Instrument(program=0, name="comping")
    inst.notes = [pretty_midi.Note(velocity=x["velocity"], pitch=x["pitch"], start=x["start"], end=x["end"]) for x in notes]
    py, stats = clean_instrument(inst, CleanupParams(**params))
    src = tmp_path / "in.json"
    src.write_text(json.dumps({"notes": notes, "params": params}))
    p = subprocess.run([NODE, str(TEST), str(src)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    js = json.loads(p.stdout)
    key = lambda x: (round(x[1], 4), x[0], round(x[2], 4))  # noqa: E731
    want = sorted(((n.pitch, n.start, n.end) for n in py.notes), key=key)
    got = sorted(((n["pitch"], n["start"], n["end"]) for n in js["notes"]), key=key)
    assert [key(x) for x in got] == [key(x) for x in want]
    for k, v in stats.as_dict().items():
        assert js["stats"][k] == pytest.approx(v, abs=1e-4), k
