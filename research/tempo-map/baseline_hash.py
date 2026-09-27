"""Hash the steady 4/4 grid output (tests/test_grid.py's _song) with whatever grid.py is
installed, so the tempo-map work can prove the constant path writes the same bytes.

baseline_hash.py OUT_DIR
"""
import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests"))

from stemscribe import grid as G  # noqa: E402
from test_grid import _song  # noqa: E402

out_dir = pathlib.Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
for snap in (False, True):
    out, info, _ = G.apply(_song(), 114.0 * 1.01, snap_notes=snap)
    p = out_dir / f"steady-{'snapped' if snap else 'plain'}.mid"
    out.write(str(p))
    print(f"snap={snap} sha256={hashlib.sha256(p.read_bytes()).hexdigest()}")
