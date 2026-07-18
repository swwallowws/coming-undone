"""Fetch tests. No network: yt-dlp is stubbed. What is worth pinning is our
own logic -- URL detection, where the file landed, provenance capture, and
turning yt-dlp's wall of text into something a human can act on.
"""
import pathlib

import pytest

from stemscribe.fetch import (
    JS_RUNTIMES,
    FetchError,
    FetchedAudio,
    _downloaded_path,
    _explain,
    available_js_runtimes,
    fetch_audio,
    is_url,
)


@pytest.mark.parametrize("s", [
    "https://www.youtube.com/watch?v=abc123",
    "http://youtu.be/abc",
    "HTTPS://EXAMPLE.COM/a.mp3",
    "  https://example.com/a.mp3  ",
])
def test_is_url_accepts_urls(s):
    assert is_url(s)


@pytest.mark.parametrize("s", [
    "song.mp3", "/abs/path/song.wav", "~/Music/x.m4a", "", "ftp://host/a.mp3",
    "C:\\music\\song.mp3",
])
def test_is_url_rejects_paths(s):
    assert not is_url(s)


def test_is_url_accepts_pathlib(tmp_path):
    assert not is_url(tmp_path / "song.mp3")


# --- where did the file land ------------------------------------------------
def test_downloaded_path_prefers_requested_downloads():
    info = {"requested_downloads": [{"filepath": "/tmp/x.opus"}], "_filename": "/tmp/wrong.webm"}
    assert _downloaded_path(info) == pathlib.Path("/tmp/x.opus")


def test_downloaded_path_falls_back_to_filename():
    assert _downloaded_path({"_filename": "/tmp/y.m4a"}) == pathlib.Path("/tmp/y.m4a")


def test_downloaded_path_none_when_absent():
    assert _downloaded_path({"id": "abc"}) is None


# --- error messages a human can act on --------------------------------------
@pytest.mark.parametrize("raw,want", [
    ("ERROR: Private video. Sign in if you've been granted access", "private"),
    ("ERROR: Video unavailable. This video has been removed", "unavailable"),
    ("ERROR: Unsupported URL: https://example.com/x", "yt-dlp can handle"),
    ("ERROR: HTTP Error 403: Forbidden", "refused"),
])
def test_explain_translates_common_failures(raw, want):
    assert want in _explain(Exception(raw))


def test_explain_keeps_unknown_errors_visible():
    assert "kaboom" in _explain(Exception("kaboom"))


# --- the 403 that actually happened -----------------------------------------
def test_403_points_at_stale_ytdlp(monkeypatch):
    """The real cause was a 220-day-old yt-dlp. Say so, don't make them bisect."""
    msg = _explain(Exception("ERROR: unable to download video data: HTTP Error 403: Forbidden"))
    assert "403" in msg
    assert "pip install -U yt-dlp" in msg
    assert "unable to download video data" in msg  # original preserved


def test_403_mentions_js_runtime_only_when_missing(monkeypatch):
    monkeypatch.setattr("stemscribe.fetch.available_js_runtimes", lambda: [])
    assert "JS runtime" in _explain(Exception("HTTP Error 403: Forbidden"))

    monkeypatch.setattr("stemscribe.fetch.available_js_runtimes", lambda: ["node"])
    assert "JS runtime" not in _explain(Exception("HTTP Error 403: Forbidden"))


def test_available_js_runtimes_finds_what_is_installed(monkeypatch):
    monkeypatch.setattr(
        "stemscribe.fetch.shutil.which", lambda r: "/usr/bin/" + r if r == "node" else None
    )
    assert available_js_runtimes() == ["node"]


def test_available_js_runtimes_empty_when_none(monkeypatch):
    monkeypatch.setattr("stemscribe.fetch.shutil.which", lambda r: None)
    assert available_js_runtimes() == []


def test_js_runtime_preference_order():
    assert JS_RUNTIMES[0] == "deno"  # yt-dlp's own default
    assert "node" in JS_RUNTIMES


# --- fetch_audio, with yt-dlp stubbed ---------------------------------------
class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL."""

    info: dict = {}
    written: str | None = None
    raises: Exception | None = None
    last_opts: dict = {}

    def __init__(self, opts):
        FakeYDL.last_opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=True):
        if FakeYDL.raises:
            raise FakeYDL.raises
        if FakeYDL.written:
            pathlib.Path(FakeYDL.written).write_bytes(b"\x00audio")
        return FakeYDL.info


@pytest.fixture
def fake_ytdlp(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("yt_dlp")
    mod.YoutubeDL = FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", mod)
    FakeYDL.raises = None
    FakeYDL.written = None
    FakeYDL.info = {}
    return FakeYDL


def test_fetch_captures_provenance(tmp_path, fake_ytdlp):
    dest = tmp_path / "src" / "vid123.opus"
    dest.parent.mkdir(parents=True)
    fake_ytdlp.written = str(dest)
    fake_ytdlp.info = {
        "id": "vid123",
        "title": "Some Song",
        "uploader": "Some Channel",
        "duration": 214.0,
        "extractor_key": "Youtube",
        "requested_downloads": [{"filepath": str(dest)}],
    }
    got = fetch_audio("https://youtube.com/watch?v=vid123", tmp_path / "src")
    assert got.path == dest
    assert got.source_id == "vid123"
    assert got.title == "Some Song"
    assert got.uploader == "Some Channel"
    assert got.duration == 214.0
    assert got.ext == "opus"
    assert got.url.endswith("vid123")


def test_js_runtime_is_enabled_when_present(tmp_path, fake_ytdlp, monkeypatch):
    """yt-dlp only enables deno by default; a node-only machine 403s silently."""
    monkeypatch.setattr("stemscribe.fetch.available_js_runtimes", lambda: ["node"])
    dest = tmp_path / "src" / "a.opus"
    dest.parent.mkdir(parents=True)
    fake_ytdlp.written = str(dest)
    fake_ytdlp.info = {"id": "a", "requested_downloads": [{"filepath": str(dest)}]}
    fetch_audio("https://x/a", tmp_path / "src")
    # The API wants {runtime: config|None}; a list raises ValueError. Only the
    # CLI takes a list, which is exactly how this got shipped wrong once.
    assert FakeYDL.last_opts["js_runtimes"] == {"node": {}}


def test_no_js_runtimes_key_when_none_available(tmp_path, fake_ytdlp, monkeypatch):
    monkeypatch.setattr("stemscribe.fetch.available_js_runtimes", lambda: [])
    dest = tmp_path / "src" / "a.opus"
    dest.parent.mkdir(parents=True)
    fake_ytdlp.written = str(dest)
    fake_ytdlp.info = {"id": "a", "requested_downloads": [{"filepath": str(dest)}]}
    seen = []
    fetch_audio("https://x/a", tmp_path / "src", progress=lambda s, m: seen.append(m))
    assert "js_runtimes" not in FakeYDL.last_opts  # let yt-dlp use its own default
    assert any("JavaScript runtime" in m for m in seen)


def test_native_format_adds_no_postprocessor(tmp_path, fake_ytdlp):
    dest = tmp_path / "src" / "a.opus"
    dest.parent.mkdir(parents=True)
    fake_ytdlp.written = str(dest)
    fake_ytdlp.info = {"id": "a", "requested_downloads": [{"filepath": str(dest)}]}
    fetch_audio("https://x/a", tmp_path / "src", audio_format="native")
    assert "postprocessors" not in FakeYDL.last_opts


def test_mp3_format_requests_transcode(tmp_path, fake_ytdlp):
    dest = tmp_path / "src" / "a.mp3"
    dest.parent.mkdir(parents=True)
    fake_ytdlp.written = str(dest)
    fake_ytdlp.info = {"id": "a", "requested_downloads": [{"filepath": str(dest)}]}
    fetch_audio("https://x/a", tmp_path / "src", audio_format="mp3")
    pp = FakeYDL.last_opts["postprocessors"][0]
    assert pp["preferredcodec"] == "mp3"


def test_playlists_are_not_expanded(tmp_path, fake_ytdlp):
    """A link with &list= must not pull the whole playlist."""
    dest = tmp_path / "src" / "a.opus"
    dest.parent.mkdir(parents=True)
    fake_ytdlp.written = str(dest)
    fake_ytdlp.info = {"id": "a", "requested_downloads": [{"filepath": str(dest)}]}
    fetch_audio("https://youtube.com/watch?v=a&list=PL123", tmp_path / "src")
    assert FakeYDL.last_opts["noplaylist"] is True


def test_missing_file_after_success_is_an_error(tmp_path, fake_ytdlp):
    fake_ytdlp.info = {"id": "a", "requested_downloads": [{"filepath": str(tmp_path / "gone.opus")}]}
    with pytest.raises(FetchError, match="no audio file was written"):
        fetch_audio("https://x/a", tmp_path / "src")


def test_ytdlp_errors_are_translated(tmp_path, fake_ytdlp):
    fake_ytdlp.raises = Exception("ERROR: Private video. Sign in to view")
    with pytest.raises(FetchError, match="private"):
        fetch_audio("https://x/a", tmp_path / "src")


def test_progress_is_reported(tmp_path, fake_ytdlp):
    dest = tmp_path / "src" / "a.opus"
    dest.parent.mkdir(parents=True)
    fake_ytdlp.written = str(dest)
    fake_ytdlp.info = {"id": "a", "requested_downloads": [{"filepath": str(dest)}]}
    seen = []
    fetch_audio("https://x/a", tmp_path / "src", progress=lambda s, m: seen.append((s, m)))
    hook = FakeYDL.last_opts["progress_hooks"][0]
    hook({"status": "downloading", "_percent_str": " 42.0%", "_eta_str": "00:10"})
    hook({"status": "finished"})
    assert any("42.0%" in m for _, m in seen)


def test_fetched_audio_serializes():
    import json

    from stemscribe.core import _json_default

    f = FetchedAudio(path=pathlib.Path("/a.opus"), url="https://x/a", title="T", duration=1.5)
    d = json.loads(json.dumps(f.as_dict(), default=_json_default))
    assert d["url"] == "https://x/a"
    assert d["title"] == "T"
