"""Content-addressed cache for the expensive, deterministic stages.

Why this exists: on a 5-minute song, demucs is ~87% of the runtime and
transcription another ~8%. Both are pure functions of their inputs --
separation of (audio bytes, model), transcription of (stem bytes, backend,
params). Re-running the same song to tune a cleanup knob recomputes 150 seconds
of identical tensors to change a note length. So key those stages by content and
skip them.

Cleanup, merge and tempo are NOT cached: together they are under 3 seconds, and
they are exactly the knobs anyone iterates on.

Keys are content hashes, never filenames or URLs (except the fetch stage, where
the URL is the only identity available before download). Edit a wav and the key
changes; the cache cannot serve you a stale stem.

SHARED BETWEEN PROCESSES. The CLI and the web UI default to the same root, so a
stem separated by one is free for the other. That makes concurrency real rather
than theoretical, so writes are published atomically: a writer builds its entry
in a private .tmp- dir and rename()s it into place at the end. Consequences:

  - An entry at its final path is always complete. Readers never see a half
    written stem set, and nothing deletes a directory a reader may be inside.
  - Two processes computing the same key both succeed. The loser's rename finds
    the path taken and discards its own copy -- the content is identical by
    construction, so whoever got there first is just as good.
  - A crashed run leaves a .tmp- dir, never a corrupt entry. Those get swept.

There is no lock: two simultaneous runs of the same song will both do the work.
That wastes CPU but cannot produce a wrong answer, which is the right trade for
a local tool.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import shutil
import time
import uuid

log = logging.getLogger("stemscribe.cache")

DEFAULT_ROOT = pathlib.Path.home() / ".stemscribe" / "cache"


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\x00")
    return h.hexdigest()[:32]


def file_hash(path: str | pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()[:32]


#: A .tmp- dir older than this belonged to a run that died. Safe to sweep.
STALE_TMP_SECONDS = 24 * 3600


class CacheWrite:
    """One entry under construction.

    Write into `.path`, then commit(). Until commit() lands nothing is visible
    to any other process, so a crash leaves litter rather than a corrupt entry.
    """

    def __init__(self, cache: "Cache", kind: str, key: str, path: pathlib.Path, live: bool):
        self.cache = cache
        self.kind = kind
        self.key = key
        self.path = path
        #: False when caching is off: commit() is then a no-op passthrough.
        self.live = live

    def commit(self, meta: dict | None = None) -> pathlib.Path:
        """Publish atomically. Returns the directory to read from afterwards.

        The return value matters: rename() moves the entry, so any path built
        under `.path` before this call is stale. Use what comes back.
        """
        if not self.live:
            return self.path

        if meta is not None:
            (self.path / "meta.json").write_text(json.dumps(meta, indent=2))
        (self.path / ".done").write_text(str(time.time()))

        final = self.cache._dir(self.kind, self.key)
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(self.path, final)
        except OSError:
            # Someone published this key while we worked. Content-addressed, so
            # their entry is ours; drop the duplicate and use theirs.
            log.debug("cache race on %s/%s; keeping the existing entry", self.kind, self.key)
            shutil.rmtree(self.path, ignore_errors=True)
        return final


class Cache:
    """A directory of content-addressed entries, safe to share across processes."""

    def __init__(self, root: str | pathlib.Path | None = None, enabled: bool = True):
        self.root = pathlib.Path(root or DEFAULT_ROOT).expanduser()
        self.enabled = enabled
        self.hits: list[str] = []
        self.misses: list[str] = []

    # --- generic ------------------------------------------------------------
    def _dir(self, kind: str, key: str) -> pathlib.Path:
        return self.root / kind / key

    def get_dir(self, kind: str, key: str) -> pathlib.Path | None:
        """A finished entry, or None.

        Entries only appear at their final path via rename, so existence is
        near enough proof of completeness. The .done check also rejects any
        half-written directory left by an older version of this code.
        """
        if not self.enabled:
            return None
        d = self._dir(kind, key)
        if (d / ".done").exists():
            self.hits.append(kind)
            return d
        return None

    def begin(self, kind: str, key: str, fallback: pathlib.Path | None = None) -> CacheWrite:
        """Start writing an entry. Write into the returned .path, then commit().

        `fallback` is where to write when caching is off, so callers need no
        branch of their own.
        """
        if not self.enabled:
            d = pathlib.Path(fallback) if fallback else pathlib.Path(self.root / "_nocache")
            d.mkdir(parents=True, exist_ok=True)
            return CacheWrite(self, kind, key, d, live=False)

        self.misses.append(kind)
        self.sweep_stale()
        # PID + uuid: unique across processes and across threads within one.
        tmp = self.root / kind / f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True, exist_ok=True)
        return CacheWrite(self, kind, key, tmp, live=True)

    def sweep_stale(self, older_than: float = STALE_TMP_SECONDS) -> int:
        """Remove .tmp- dirs abandoned by dead runs. Never touches live entries."""
        if not self.root.exists():
            return 0
        n, now = 0, time.time()
        for d in self.root.glob("*/.tmp-*"):
            try:
                if now - d.stat().st_mtime > older_than:
                    shutil.rmtree(d, ignore_errors=True)
                    n += 1
            except OSError:
                pass
        return n

    # --- stage keys ---------------------------------------------------------
    @staticmethod
    def fetch_key(url: str, audio_format: str) -> str:
        return _hash("fetch", url.strip(), audio_format)

    @staticmethod
    def stems_key(audio_hash: str, model: str) -> str:
        return _hash("stems", audio_hash, model)

    @staticmethod
    def midi_key(audio_hash: str, model: str, stem: str, backend: str, kwargs: dict) -> str:
        # Sorted so dict ordering cannot produce two keys for one config.
        return _hash(
            "midi", audio_hash, model, stem, backend,
            json.dumps(kwargs or {}, sort_keys=True),
        )

    # --- housekeeping -------------------------------------------------------
    def size_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())

    def clear(self) -> int:
        n = self.size_bytes()
        shutil.rmtree(self.root, ignore_errors=True)
        return n

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "root": str(self.root),
            "hits": sorted(set(self.hits)),
            "misses": sorted(set(self.misses)),
            "bytes": self.size_bytes(),
        }


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"
