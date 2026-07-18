"""Instrumental mix: sum the pitched+percussive stems, leave vocals out."""
from __future__ import annotations

import pathlib
import shutil
import subprocess


class MixdownError(RuntimeError):
    pass


def mix_instrumental(
    stem_paths: dict[str, pathlib.Path],
    out_path: str | pathlib.Path,
    exclude: tuple[str, ...] = ("vocals",),
    limit: float = 0.97,
) -> pathlib.Path:
    """Mix every stem except `exclude` into `out_path` (.wav or .mp3).

    Uses ffmpeg's amix. Note `normalize=0` is NOT used: it isn't supported by
    every ffmpeg build in the wild (including the one on this machine), so we
    let amix divide by N and multiply the gain back with volume=N, which is the
    portable way to say "sum these, don't average them".

    That sum can exceed 0dBFS -- summing demucs stems back together overshoots
    where the original mix was already near full scale (measured 1.19 peak on
    real material). alimiter catches the overshoot instead of letting the file
    clip; set limit=0 to opt out and get the raw sum.

    alimiter's `level` MUST be disabled. It defaults to true, which auto-levels
    the limited signal back up to 0dBFS -- silently undoing `limit` and landing
    the output at exactly 1.0 again. With it off, the wav really does peak at
    `limit`.

    Caveat: an mp3 of a dense limited mix still decodes ~1.1 peak no matter what
    we do here -- lossy reconstruction overshoots the source. That is normal and
    near-universal in released music; use instrumental_format="wav" if you need
    a file that is provably under full scale.
    """
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise MixdownError("ffmpeg not found on PATH; needed for the instrumental mix")

    inputs = [p for name, p in sorted(stem_paths.items()) if name not in exclude]
    if not inputs:
        raise MixdownError(f"no stems left to mix after excluding {exclude}")

    n = len(inputs)
    chain = f"amix=inputs={n},volume={n}"
    if limit:
        chain += f",alimiter=limit={limit}:level=disabled"

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    for p in inputs:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", chain, str(out_path)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MixdownError(f"ffmpeg amix failed:\n{proc.stderr[-800:]}")
    return out_path
