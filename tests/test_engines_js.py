"""The page's engine layer (web/static/js/engines.js), tested in node when it is here."""
import pathlib
import shutil
import subprocess

import pytest

TEST = pathlib.Path(__file__).parent / "js" / "engines.test.mjs"


def test_engines_js():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    p = subprocess.run([node, str(TEST)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr or p.stdout
    assert "ok" in p.stdout
