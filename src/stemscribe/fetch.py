"""Fetch audio from a URL (YouTube et al) via yt-dlp.

Quality note: by default we keep whatever codec the source serves (Opus/AAC)
rather than transcoding to mp3. prepare.py decodes to wav before demucs sees
anything, so an intermediate mp3 is a second lossy generation that costs quality
and buys nothing. Ask for mp3 only when you want a file to keep.

Provenance: everything we learn about the source (URL, id, title, uploader,
duration) is recorded and lands in manifest.json. stemscribe already hashes its
input; for fetched audio the hash alone cannot tell you where it came from, and
knowing that later is the difference between an auditable pipeline and a folder
of anonymous wavs.
"""
from __future__ import annotations

import logging
import pathlib
import re
import shutil
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("stemscribe.fetch")

URL_RE = re.compile(r"^https?://", re.I)

#: YouTube requires executing JS to work out stream signatures. yt-dlp only
#: enables deno by default, so a machine with node and no deno silently loses
#: formats and 403s on the data download while metadata still resolves fine.
#: Detect whatever is actually here rather than making the user pass a flag.
JS_RUNTIMES = ("deno", "node", "bun", "quickjs")

#: Codecs we accept as-is. Anything else gets normalized to mp3 so downstream
#: ffmpeg/soundfile never meets something exotic.
NATIVE_OK = frozenset({"m4a", "mp3", "opus", "webm", "ogg", "wav", "flac", "aac"})


class FetchError(RuntimeError):
    pass


def is_url(s: str) -> bool:
    return bool(URL_RE.match(str(s).strip()))


def available_js_runtimes() -> list[str]:
    """JS runtimes present on PATH, in yt-dlp's preference order."""
    return [r for r in JS_RUNTIMES if shutil.which(r)]


@dataclass
class FetchedAudio:
    path: pathlib.Path
    url: str
    source_id: str | None = None
    title: str | None = None
    uploader: str | None = None
    duration: float | None = None
    extractor: str | None = None
    ext: str | None = None

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "source_id": self.source_id,
            "title": self.title,
            "uploader": self.uploader,
            "duration": self.duration,
            "extractor": self.extractor,
            "ext": self.ext,
        }


def fetch_audio(
    url: str,
    dest_dir: str | pathlib.Path,
    audio_format: str = "native",
    progress: Callable[[str, str], None] | None = None,
) -> FetchedAudio:
    """Download the audio track of `url` into `dest_dir`.

    audio_format: "native" keeps the source codec (best quality, the default);
    "mp3" transcodes to a 192k mp3 you can keep.
    """
    try:
        import yt_dlp
    except ImportError as e:  # optional extra
        raise FetchError(
            "yt-dlp is not installed. pip install 'stemscribe[fetch]'"
        ) from e

    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _hook(d: dict) -> None:
        if not progress:
            return
        if d.get("status") == "downloading":
            pct = d.get("_percent_str", "").strip()
            eta = d.get("_eta_str", "").strip()
            if pct:
                progress("fetch", f"  downloading {pct}" + (f" (eta {eta})" if eta else ""))
        elif d.get("status") == "finished":
            progress("fetch", "  download complete, extracting audio ...")

    opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,  # a link with &list= should not pull 200 tracks
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [_hook],
        "retries": 3,
    }

    runtimes = available_js_runtimes()
    if runtimes:
        # The Python API wants {runtime: {config}}; only the CLI takes a list.
        # Use {} not None: validation accepts None, then .get() is called on it.
        opts["js_runtimes"] = {r: {} for r in runtimes}
        log.debug("js runtimes enabled: %s", ", ".join(runtimes))
    elif progress:
        progress("fetch", "  no JavaScript runtime found; YouTube may refuse (see README)")
    if audio_format != "native":
        opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ]

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        raise FetchError(_explain(e)) from e

    if info.get("_type") == "playlist":  # noplaylist should prevent this
        entries = [e for e in info.get("entries") or [] if e]
        if not entries:
            raise FetchError("that URL resolved to an empty playlist")
        info = entries[0]

    path = _downloaded_path(info)
    if path is None or not path.exists():
        raise FetchError("yt-dlp reported success but no audio file was written")

    # Normalize anything unusual so downstream never meets a surprise container.
    ext = path.suffix.lstrip(".").lower()
    if ext not in NATIVE_OK:
        raise FetchError(f"unexpected audio format {ext!r} from {url}")

    return FetchedAudio(
        path=path,
        url=url,
        source_id=info.get("id"),
        title=info.get("title"),
        uploader=info.get("uploader") or info.get("channel"),
        duration=info.get("duration"),
        extractor=info.get("extractor_key") or info.get("extractor"),
        ext=ext,
    )


def _downloaded_path(info: dict) -> pathlib.Path | None:
    """Where yt-dlp actually put it, post-processing included."""
    reqs = info.get("requested_downloads") or []
    for r in reqs:
        for key in ("filepath", "_filename"):
            if r.get(key):
                return pathlib.Path(r[key])
    for key in ("filepath", "_filename"):
        if info.get(key):
            return pathlib.Path(info[key])
    return None


def _stale_ytdlp_hint() -> str:
    """yt-dlp goes stale fast -- YouTube changes, yt-dlp patches, weekly-ish.

    A months-old yt-dlp is the single most common cause of a 403 here, so say so
    rather than making the next person bisect it.
    """
    try:
        import yt_dlp

        ver = yt_dlp.version.__version__
    except Exception:
        return "  - update yt-dlp:  pip install -U yt-dlp"
    return f"  - update yt-dlp (you have {ver}):  pip install -U yt-dlp"


def _explain(e: Exception) -> str:
    """yt-dlp's errors are long; surface the part a human can act on."""
    msg = str(e)
    low = msg.lower()
    if "private video" in low or "sign in" in low:
        return "that video is private or requires sign-in"
    if "video unavailable" in low or "removed" in low:
        return "that video is unavailable or has been removed"
    if "is not a valid url" in low or "unsupported url" in low:
        return f"not a URL yt-dlp can handle: {msg[:150]}"
    if "age" in low and "confirm" in low:
        return "that video is age-restricted and needs sign-in"
    if "403" in low or "forbidden" in low:
        # Almost always a stale extractor or a missing JS runtime, not the URL.
        lines = [
            "the site refused the request (HTTP 403). Usually one of:",
            _stale_ytdlp_hint(),
        ]
        if not available_js_runtimes():
            lines.append(
                "  - install a JS runtime (YouTube needs one to decipher streams): "
                "brew install deno   (node/bun/quickjs also work)"
            )
        lines.append(f"  original: {msg[:200]}")
        return "\n".join(lines)
    if "http error 4" in low or "http error 5" in low:
        return f"the site refused the request: {msg[:150]}"
    return f"download failed: {msg[:300]}"
