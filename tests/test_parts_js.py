"""The page's part names (web/static/js/parts.js): tested in node, and checked against
merge.TRACK_SPEC so the names on the page are the tracks' own."""
import json
import pathlib
import re
import shutil
import subprocess

import pytest

from stemscribe import merge

ROOT = pathlib.Path(__file__).parent
TEST = ROOT / "js" / "parts.test.mjs"
PARTS = ROOT.parent / "src" / "stemscribe" / "web" / "static" / "js" / "parts.js"


def _js_object(name: str) -> str:
    m = re.search(rf"export const {name} = (\{{.*?\}}|\[.*?\]);", PARTS.read_text(), re.S)
    assert m, name
    return m.group(1)


def test_parts_js():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    p = subprocess.run([node, str(TEST)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr or p.stdout
    assert "ok" in p.stdout


def test_stem_part_mirrors_track_spec():
    body = _js_object("STEM_PART")
    pairs = dict(re.findall(r"(\w+):\s*\"(\w+)\"", body))
    assert pairs == {stem: name for stem, (name, _) in merge.TRACK_SPEC.items()}


def test_programs_mirror_track_spec():
    body = _js_object("PROGRAM")
    progs = {k: int(v) for k, v in re.findall(r"(\w+):\s*(\d+)", body)}
    assert progs == {name: program for name, program in merge.TRACK_SPEC.values()}


def test_part_order_names_every_track():
    order = json.loads(_js_object("PART_ORDER"))
    assert sorted(order) == sorted(name for name, _ in merge.TRACK_SPEC.values())
    assert order[0] == "melody" and order[-1] == "drums"
