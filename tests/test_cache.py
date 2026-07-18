"""Cache tests.

The cache is a correctness hazard before it is a speed feature: serving a stale
stem silently produces wrong output that looks right. So these lean on the keys
being content-derived and on partial writes never being served.
"""
import pytest

from stemscribe.cache import Cache, file_hash, human_bytes


def test_stems_key_changes_with_audio():
    a = Cache.stems_key("hash_a", "htdemucs")
    b = Cache.stems_key("hash_b", "htdemucs")
    assert a != b


def test_stems_key_changes_with_model():
    a = Cache.stems_key("hash_a", "htdemucs")
    b = Cache.stems_key("hash_a", "htdemucs_ft")
    assert a != b


def test_stems_key_is_stable():
    assert Cache.stems_key("h", "htdemucs") == Cache.stems_key("h", "htdemucs")


def test_midi_key_changes_with_backend_params():
    """Different thresholds must not share a cached transcription."""
    a = Cache.midi_key("h", "htdemucs", "bass", "basic-pitch", {"frame_threshold": 0.3})
    b = Cache.midi_key("h", "htdemucs", "bass", "basic-pitch", {"frame_threshold": 0.2})
    assert a != b


def test_midi_key_ignores_dict_ordering():
    a = Cache.midi_key("h", "m", "bass", "bp", {"x": 1, "y": 2})
    b = Cache.midi_key("h", "m", "bass", "bp", {"y": 2, "x": 1})
    assert a == b


def test_midi_key_separates_stems():
    a = Cache.midi_key("h", "m", "bass", "bp", {})
    b = Cache.midi_key("h", "m", "vocals", "bp", {})
    assert a != b


def test_fetch_key_changes_with_format():
    assert Cache.fetch_key("https://x/a", "native") != Cache.fetch_key("https://x/a", "mp3")


def test_fetch_key_ignores_surrounding_whitespace():
    assert Cache.fetch_key(" https://x/a ", "native") == Cache.fetch_key("https://x/a", "native")


def test_file_hash_tracks_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert file_hash(a) == file_hash(b)
    b.write_bytes(b"different")
    assert file_hash(a) != file_hash(b)


# --- the correctness hazards ------------------------------------------------
def test_entry_is_invisible_until_committed(tmp_path):
    """A run killed mid-write must not leave something another process serves."""
    c = Cache(tmp_path)
    w = c.begin("stems", "k1")
    (w.path / "bass.wav").write_bytes(b"partial")
    assert c.get_dir("stems", "k1") is None  # still a private .tmp- dir
    final = w.commit()
    assert c.get_dir("stems", "k1") == final
    assert (final / "bass.wav").read_bytes() == b"partial"


def test_commit_returns_the_moved_path(tmp_path):
    """commit() renames, so paths built before it are stale. Callers must use
    the return value -- getting this wrong points at a vanished temp dir."""
    c = Cache(tmp_path)
    w = c.begin("stems", "k1")
    before = w.path
    final = w.commit()
    assert final != before
    assert not before.exists()
    assert final.exists()


def test_concurrent_writers_do_not_corrupt(tmp_path):
    """Two processes computing the same key: both finish, one entry survives,
    and it is whole. Previously the second writer rmtree'd the first."""
    c = Cache(tmp_path)
    a = c.begin("stems", "same")
    b = c.begin("stems", "same")
    assert a.path != b.path  # private scratch each

    (a.path / "bass.wav").write_bytes(b"A" * 100)
    (b.path / "bass.wav").write_bytes(b"B" * 100)

    fa = a.commit({"who": "a"})
    fb = b.commit({"who": "b"})

    assert fa == fb  # same key, same final home
    assert len(list(fa.glob("*.wav"))) == 1  # not a mix of both writers
    assert fa.stat().st_size >= 0
    # First writer wins; the loser discards rather than clobbering a live entry.
    assert (fa / "bass.wav").read_bytes() == b"A" * 100


def test_a_reader_survives_a_concurrent_writer(tmp_path):
    """The actual bug: web UI reading stems while a CLI run publishes the key."""
    c = Cache(tmp_path)
    w = c.begin("stems", "k")
    (w.path / "bass.wav").write_bytes(b"first")
    entry = w.commit()

    # Another process recomputes the same key while we hold `entry`.
    w2 = c.begin("stems", "k")
    (w2.path / "bass.wav").write_bytes(b"second")
    w2.commit()

    # The reader's files are still there and still readable.
    assert (entry / "bass.wav").read_bytes() == b"first"


def test_disabled_cache_writes_to_fallback(tmp_path):
    c = Cache(tmp_path, enabled=False)
    fb = tmp_path / "out" / "stems"
    w = c.begin("stems", "k", fallback=fb)
    assert w.path == fb
    (w.path / "x.wav").write_bytes(b"x")
    assert w.commit() == fb  # no move, no cache entry
    assert not (tmp_path / "stems" / "k").exists()


def test_disabled_cache_never_hits(tmp_path):
    c = Cache(tmp_path, enabled=True)
    c.begin("stems", "k1").commit()
    off = Cache(tmp_path, enabled=False)
    assert off.get_dir("stems", "k1") is None


def test_hits_and_misses_are_tracked(tmp_path):
    c = Cache(tmp_path)
    assert c.get_dir("stems", "k") is None
    c.begin("stems", "k").commit()
    assert c.get_dir("stems", "k") is not None
    assert "stems" in c.summary()["hits"]
    assert "stems" in c.summary()["misses"]


def test_stale_tmp_dirs_are_swept(tmp_path):
    import os
    import time

    c = Cache(tmp_path)
    dead = tmp_path / "stems" / ".tmp-999-deadbeef"
    dead.mkdir(parents=True)
    old = time.time() - 48 * 3600
    os.utime(dead, (old, old))

    assert c.sweep_stale() == 1
    assert not dead.exists()


def test_sweep_leaves_live_tmp_dirs_alone(tmp_path):
    c = Cache(tmp_path)
    w = c.begin("stems", "k")
    assert c.sweep_stale() == 0
    assert w.path.exists()


def test_size_and_clear(tmp_path):
    c = Cache(tmp_path)
    w = c.begin("stems", "k")
    (w.path / "a.wav").write_bytes(b"x" * 5000)
    w.commit()
    assert c.size_bytes() >= 5000
    freed = c.clear()
    assert freed >= 5000
    assert c.size_bytes() == 0


def test_size_of_missing_root_is_zero(tmp_path):
    assert Cache(tmp_path / "nope").size_bytes() == 0


@pytest.mark.parametrize("n,want", [(0, "0B"), (2048, "2.0KB"), (5 * 1024**2, "5.0MB")])
def test_human_bytes(n, want):
    assert human_bytes(n) == want


def test_summary_serializes():
    import json

    from stemscribe.core import _json_default

    assert json.loads(json.dumps(Cache(enabled=False).summary(), default=_json_default))
