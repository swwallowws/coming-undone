"""The pipeline: separate -> instrumental -> transcribe -> cleanup -> merge."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import pretty_midi
import soundfile as _sf

from . import backends as _backends
from . import grid as _grid
from . import fetch as _fetch
from .cache import Cache, file_hash, human_bytes
from . import merge as _merge
from . import mixdown as _mixdown
from . import prepare as _prepare
from . import separate as _separate
from . import tempo as _tempo
from .cleanup import CleanupParams, clean_instrument
from .prepare import PrepareParams

log = logging.getLogger("stemscribe")

#: Stems handed to a pitched backend, in track order. guitar/piano only exist
#: under htdemucs_6s; listing them here is harmless for the 4-stem model, which
#: simply never produces them.
PITCHED_STEMS = ("vocals", "bass", "guitar", "piano", "other")


@dataclass
class Result:
    midi_path: pathlib.Path | None = None
    stem_paths: dict[str, pathlib.Path] = field(default_factory=dict)
    instrumental_path: pathlib.Path | None = None
    track_map: dict[str, int] = field(default_factory=dict)
    manifest_path: pathlib.Path | None = None
    manifest: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: The tempo written into the MIDI, plus how we arrived at it and what the
    #: runner-up candidates were. Never None -- detection always resolves.
    tempo: "_tempo.TempoEstimate | None" = None
    #: What conditioning was applied to the input, and the offset that was added
    #: back so MIDI times match the original file.
    prepared: "_prepare.PreparedAudio | None" = None
    #: Where the audio came from, when the input was a URL. None for local files.
    source: "_fetch.FetchedAudio | None" = None


def _json_default(o):
    """Coerce the non-JSON types that reach the manifest.

    numpy scalars leak in from two directions: pretty_midi hands back int64 for
    instrument.program when reading a file, and statistics.mean/median over
    numpy-backed note times return float64. Catch them at the boundary rather
    than casting at every call site and missing one.
    """
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    if isinstance(o, pathlib.PurePath):
        return str(o)
    raise TypeError(f"manifest: object of type {o.__class__.__name__} is not JSON serializable")


def _jsonable(obj):
    """Recursively coerce a structure into plain JSON types.

    Result.manifest is a public surface -- rearranged reads it, and the web
    UI hands it to a JSON serializer that does not know about _json_default. So
    the manifest has to be clean as a *data structure*, not merely writable by
    our own writer.
    """
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


def _safe_stem(title: str, limit: int = 60) -> str:
    """A filesystem-safe output name from a source title."""
    s = re.sub(r"[^\w\s.-]", "", title, flags=re.UNICODE).strip()
    s = re.sub(r"\s+", "_", s)
    return (s[:limit].rstrip("._-") or "audio")


def _file_sha256(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def process(
    input_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    backend: str = "muscriptor",
    include_vocals_melody: bool = True,
    cleanup: bool = True,
    cleanup_params: CleanupParams | None = None,
    keep_stems: bool = True,
    instrumental: bool = True,
    instrumental_format: str = "mp3",
    demucs_model: str = "htdemucs",
    device: str | None = None,
    tempo: float | None = None,
    tempo_estimator: str = "librosa",
    prepare: PrepareParams | None = None,
    audio_format: str = "native",
    keep_source: bool = True,
    cache: bool | Cache = True,
    mono_stems: tuple[str, ...] = (),
    backend_kwargs: dict | None = None,
    progress: Callable[[str, str], None] | None = None,
    drums: str | None = "adt-str",
    grid: bool = True,
    snap: bool = False,
    downbeat: int | None = None,
    fallback: bool = True,
) -> Result:
    """Audio in, stems + labeled multi-track MIDI out.

    Processes whole songs; section cutting stays with the caller.

    input_path: a local file, or a URL yt-dlp can handle (YouTube et al) --
    fetched audio is recorded in the manifest so its provenance survives.
    tempo: None (default) detects it from the audio -- you never have to think
    about it. Pass a number to override when you know better than the tracker.
    cache: reuse the download, stems and raw transcription when the inputs are
    byte-identical. Re-running the same song to tune a cleanup knob then costs
    seconds instead of minutes. False disables it; pass a Cache to relocate it.
    mono_stems: stems to collapse to one note at a time, keeping the lowest,
    e.g. ("bass",). Removes overtones and bleed; also removes real double-stops.
    progress: optional callback(stage, message) for UIs.
    drums: the drum backend for the drums stem ("adt-str", needs the [drums] extra),
    or None to skip drums. Skipped with a warning when the extra is not installed.
    grid: fit the beat grid from the tightest track and write real bar lines.
    snap: also remove each track's latency and snap it to the grid (off by default:
    snapping deletes real feel). downbeat: 1-4, which beat of the guessed bar is "one".
    fallback: re-transcribe a nearly empty stem with basic-pitch.
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    warnings: list[str] = []

    out_dir = pathlib.Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cch = cache if isinstance(cache, Cache) else Cache(enabled=bool(cache))

    def _emit0(stage: str, msg: str) -> None:
        log.info(msg)
        if progress:
            progress(stage, msg)

    # --- fetch (only when handed a URL) -------------------------------------
    source: _fetch.FetchedAudio | None = None
    if _fetch.is_url(input_path):
        url = str(input_path).strip()
        t0 = time.perf_counter()
        # The URL is the key here: before downloading there is no content to
        # hash. Every later stage keys on bytes instead.
        fkey = Cache.fetch_key(url, audio_format)
        hit = cch.get_dir("fetch", fkey)
        if hit:
            meta = json.loads((hit / "meta.json").read_text())
            source = _fetch.FetchedAudio(path=hit / meta["filename"], **meta["source"])
            _emit0("fetch", f"cached: {source.title or url}")
        else:
            _emit0("fetch", f"fetching {url} ...")
            w = cch.begin("fetch", fkey, fallback=pathlib.Path(tempfile.mkdtemp()))
            source = _fetch.fetch_audio(
                url, w.path, audio_format=audio_format, progress=_emit0
            )
            final = w.commit({"filename": source.path.name, "source": source.as_dict()})
            source.path = final / source.path.name  # commit() moved it
            _emit0(
                "fetch",
                f"got {source.title or source.source_id!r}"
                + (f" by {source.uploader}" if source.uploader else "")
                + f" [{source.ext}]",
            )
        timings["fetch"] = round(time.perf_counter() - t0, 2)
        if keep_source:
            keep = out_dir / "source"
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.path, keep / source.path.name)
        input_path = source.path

    input_path = pathlib.Path(input_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    # Prefer the source's real title over a bare video id for output names.
    song = _safe_stem(source.title) if (source and source.title) else input_path.stem

    backend_fn = _backends.get_backend(backend)
    if backend in _backends.NONCOMMERCIAL_BACKENDS:
        # Structural guard for the commercial path. The default is now muscriptor
        # (better quality, fine for personal/non-commercial use), so the old
        # "safe by default" property is gone. rearranged's shipping build sets
        # STEMSCRIBE_COMMERCIAL=1 and is then protected regardless of the default:
        # a CC-BY-NC backend hard-fails instead of silently landing in a product.
        if os.environ.get("STEMSCRIBE_COMMERCIAL", "").lower() in ("1", "true", "yes"):
            raise _backends.BackendError(
                f"backend {backend!r} has CC-BY-NC weights and STEMSCRIBE_COMMERCIAL "
                "is set. Use backend='basic-pitch' (Apache-2.0) for a commercial path."
            )
        w = (
            f"backend {backend!r} uses CC-BY-NC weights: non-commercial use only "
            "(fine for personal/eval, not for anything you ship)"
        )
        warnings.append(w)
        log.warning(w)

    def _emit(stage: str, msg: str) -> None:
        log.info(msg)
        if progress:
            progress(stage, msg)

    # --- 1. separate --------------------------------------------------------
    # Stems always land somewhere real (demucs + soundfile need a path); when
    # keep_stems is False that somewhere is a temp dir we delete on the way out.
    tmp_dir = tempfile.mkdtemp(prefix="stemscribe-")
    stems_dir = (out_dir / "stems") if keep_stems else (pathlib.Path(tmp_dir) / "stems")

    try:
        # --- 0. prepare -----------------------------------------------------
        # Condition the input before anything expensive reads it. Any offset
        # introduced here is added back to the MIDI at the end (step 6), so note
        # times always refer to the original file the caller handed us.
        pp = prepare if prepare is not None else PrepareParams()
        t0 = time.perf_counter()
        _emit("prepare", f"preparing {input_path.name} ...")
        prepared = _prepare.prepare_audio(input_path, pathlib.Path(tmp_dir), pp)
        timings["prepare"] = round(time.perf_counter() - t0, 2)
        for step in prepared.applied:
            _emit("prepare", f"  {step}")
        if prepared.offset:
            _emit(
                "prepare",
                f"  audio starts {prepared.offset:.2f}s into the original; "
                "MIDI will be shifted back to match",
            )

        # --- 1. separate ----------------------------------------------------
        # 87% of the runtime, and a pure function of (audio bytes, model).
        t0 = time.perf_counter()
        audio_hash = file_hash(prepared.path)
        skey = Cache.stems_key(audio_hash, demucs_model)
        hit = cch.get_dir("stems", skey)
        if hit:
            stem_paths = {p.stem: p for p in sorted(hit.glob("*.wav"))}
            _emit(
                "separate",
                f"cached stems ({', '.join(sorted(stem_paths))}) — skipping demucs",
            )
        else:
            _emit("separate", f"separating with {demucs_model} ...")
            w = cch.begin("stems", skey, fallback=stems_dir)
            stem_paths = _separate.separate(
                prepared.path, w.path, model_name=demucs_model, device=device
            )
            final = w.commit({"model": demucs_model, "audio_sha": audio_hash})
            stem_paths = {k: final / v.name for k, v in stem_paths.items()}
        timings["separate"] = round(time.perf_counter() - t0, 2)

        # Stems live in the cache; the out_dir gets its own copies so deleting
        # the cache can never gut someone's finished output folder.
        if keep_stems and stem_paths and next(iter(stem_paths.values())).parent != stems_dir:
            stems_dir.mkdir(parents=True, exist_ok=True)
            copied = {}
            for name, p in stem_paths.items():
                dst = stems_dir / p.name
                if not dst.exists() or dst.stat().st_size != p.stat().st_size:
                    shutil.copy2(p, dst)
                copied[name] = dst
            stem_paths = copied

        # --- 1b. tempo ------------------------------------------------------
        # After separation on purpose: beat tracking on the isolated drums beats
        # beat tracking on a dense mix, and demucs has already given us drums.
        _emit("tempo", "estimating tempo ..." if tempo is None else f"using tempo {tempo}")
        t0 = time.perf_counter()
        tempo_est = _tempo.resolve_tempo(
            tempo,
            stem_paths=stem_paths,
            fallback_audio=prepared.path,
            estimator=tempo_estimator,
        )
        timings["tempo"] = round(time.perf_counter() - t0, 2)
        bpm = tempo_est.bpm
        if tempo_est.source == "detected":
            alts = ", ".join(f"{c.bpm:.0f}" for c in tempo_est.candidates if c.ratio != 1.0)
            _emit("tempo", f"tempo {bpm:.2f} BPM (from {tempo_est.analyzed_stem}); alternates: {alts or 'none'}")
        elif tempo_est.source == "default":
            w = f"tempo detection unavailable ({tempo_est.method}); MIDI written at {bpm} BPM"
            warnings.append(w)
            log.warning(w)

        # cleanup's duration cap is measured in beats, so it must use the same
        # tempo we write into the MIDI or "2 beats" means two different things.
        cp = cleanup_params or CleanupParams()
        cp.tempo = bpm

        # --- 2. instrumental ------------------------------------------------
        instrumental_path = None
        if instrumental:
            t0 = time.perf_counter()
            _emit("instrumental", "mixing instrumental ...")
            instrumental_path = _mixdown.mix_instrumental(
                stem_paths, out_dir / f"{song}_instrumental.{instrumental_format}"
            )
            timings["instrumental"] = round(time.perf_counter() - t0, 2)

        # --- 3. transcribe --------------------------------------------------
        wanted = [s for s in PITCHED_STEMS if s in stem_paths]
        if not include_vocals_melody:
            wanted = [s for s in wanted if s != "vocals"]

        drum_jobs: list[str] = []
        for skipped in sorted(set(stem_paths) & _backends.UNPITCHED_STEMS):
            if drums and _backends.drums_available():
                drum_jobs.append(skipped)
                continue
            why = ("drums are off" if not drums else
                   "the drum backend is not installed (pip install 'stemscribe[drums]')")
            w = f"stem {skipped!r} skipped: {why}"
            warnings.append(w)
            log.warning(w)

        raw_dir = out_dir / "_raw_midi"
        stem_midis: dict[str, pathlib.Path] = {}
        fallbacks: dict[str, str] = {}
        t0 = time.perf_counter()
        raw_dir.mkdir(parents=True, exist_ok=True)

        def _transcribe(stem: str, name: str, fn, kwargs: dict) -> pathlib.Path | None:
            # Also deterministic, so cache it: this is what makes tuning a
            # cleanup knob cost seconds rather than another basic-pitch pass.
            mkey = Cache.midi_key(audio_hash, demucs_model, stem, name, kwargs)
            hit = cch.get_dir("midi", mkey)
            dst = raw_dir / f"{stem}.mid"
            if hit and (hit / "out.mid").exists():
                shutil.copy2(hit / "out.mid", dst)
                _emit("transcribe", f"cached {stem} transcription ({name})")
                return dst
            _emit("transcribe", f"transcribing {stem} stem with {name} ...")
            mid = fn(stem_paths[stem], dst, **kwargs)
            if mid is None:
                w = f"backend {name!r} produced no MIDI for stem {stem!r}"
                warnings.append(w)
                log.warning(w)
                return None
            if cch.enabled:
                w = cch.begin("midi", mkey)
                shutil.copy2(mid, w.path / "out.mid")
                w.commit({"stem": stem, "backend": name})
            return mid  # stays in raw_dir; the cache holds a copy

        for stem in wanted:
            mid = _transcribe(stem, backend, backend_fn, backend_kwargs or {})
            if mid and fallback and backend != _backends.FALLBACK_BACKEND:
                n_notes = sum(len(i.notes) for i in pretty_midi.PrettyMIDI(str(mid)).instruments)
                y, sr = _sf.read(str(stem_paths[stem]), dtype="float32", always_2d=True)
                rms = float((y ** 2).mean() ** 0.5) if y.size else 0.0
                if _backends.is_sparse(n_notes, len(y) / sr, bpm, rms):
                    w = (f"stem {stem!r}: {backend} gave only {n_notes} notes; "
                         f"transcribed it again with {_backends.FALLBACK_BACKEND}")
                    warnings.append(w)
                    log.warning(w)
                    again = _transcribe(stem, _backends.FALLBACK_BACKEND,
                                        _backends.get_backend(_backends.FALLBACK_BACKEND), {})
                    if again:
                        mid = again
                        fallbacks[stem] = _backends.FALLBACK_BACKEND
            if mid:
                stem_midis[stem] = mid
        drum_midis: dict[str, pathlib.Path] = {}
        for stem in drum_jobs:
            try:        # drums are optional: a broken drum model never sinks the run
                mid = _transcribe(stem, drums, _backends.DRUM_BACKENDS[drums], {})
            except _backends.BackendError as e:
                w = f"stem {stem!r} skipped: drum backend {drums!r} failed: {e}"
                warnings.append(w)
                log.warning(w)
                continue
            if mid:
                drum_midis[stem] = mid
        timings["transcribe"] = round(time.perf_counter() - t0, 2)

        if not stem_midis:
            raise RuntimeError(f"no stem produced MIDI with backend {backend!r}")

        # --- 4. cleanup -----------------------------------------------------
        track_stats: dict[str, dict] = {}
        t0 = time.perf_counter()
        for stem, mid_path in list(stem_midis.items()):
            name, _program = _merge.track_for_stem(stem)
            pm = pretty_midi.PrettyMIDI(str(mid_path))
            flat = pretty_midi.Instrument(program=0, name=name)
            for inst in pm.instruments:
                flat.notes.extend(inst.notes)

            if cleanup:
                # Per stem: bass wants one voice, comping emphatically does not.
                scp = replace(cp, monophonic=True) if stem in mono_stems else cp
                cleaned, stats = clean_instrument(flat, scp)
                track_stats[name] = stats.as_dict()
                out_pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
                out_pm.instruments.append(cleaned)
                out_pm.write(str(mid_path))
            else:
                track_stats[name] = {
                    "notes_before": len(flat.notes),
                    "notes_after": len(flat.notes),
                    "cleanup": "disabled",
                }
        timings["cleanup"] = round(time.perf_counter() - t0, 2)

        # --- 5. merge -------------------------------------------------------
        t0 = time.perf_counter()
        midi_path, track_map = _merge.merge_midis(
            {**stem_midis, **drum_midis}, out_dir / f"{song}.mid", tempo=bpm
        )
        timings["merge"] = round(time.perf_counter() - t0, 2)

        # Look up by name, not by track_map index: pretty_midi drops empty
        # instruments on write, so a track that transcribed to zero notes (a
        # silent piano stem, say) is simply absent on re-read and its index no
        # longer lines up. Match on name and treat a missing track as 0 notes.
        merged = pretty_midi.PrettyMIDI(str(midi_path))
        by_name = {i.name: i for i in merged.instruments}
        for name in track_map:
            inst = by_name.get(name)
            if inst is None:
                track_stats.setdefault(name, {}).update(
                    {"note_count": 0, "quantization_error": None}
                )
                continue
            track_stats.setdefault(name, {})["note_count"] = len(inst.notes)
            track_stats[name]["program"] = inst.program
            track_stats[name]["quantization_error"] = _merge.quantization_error(
                inst, tempo=bpm
            )

        # --- 6. realign to the original timeline ----------------------------
        # Deliberately after the QC stats: the quantization grid should start
        # where the music starts, not where the original file starts.
        if prepared.offset:
            n = _prepare.shift_midi(midi_path, prepared.offset)
            _emit(
                "merge",
                f"shifted {n} notes +{prepared.offset:.2f}s back onto the original timeline",
            )

        # --- 7. grid --------------------------------------------------------
        # After realign on purpose: the grid is fitted on the original file's
        # timeline, which is the one the MIDI's notes refer to.
        grid_info: dict = {"fitted": False, "disabled": True}
        if grid:
            t0 = time.perf_counter()
            _emit("grid", "fitting the beat grid ...")
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            gridded, grid_info, gw = _grid.apply(pm, bpm, snap_notes=snap, downbeat=downbeat)
            for w in gw:
                warnings.append(w)
                log.warning(w)
            if grid_info["fitted"]:
                gridded.write(str(midi_path))
                tempo_est.bpm = grid_info["bpm"]
                tempo_est.method += "+grid-fit"
                _emit("grid", f"grid {grid_info['bpm']:.3f} BPM from {grid_info['source_track']}, "
                              f"first bar line at {grid_info['first_bar']:.2f}s"
                              + (", snapped" if snap else ""))
            timings["grid"] = round(time.perf_counter() - t0, 2)

        shutil.rmtree(raw_dir, ignore_errors=True)
        timings["total"] = round(time.perf_counter() - t_start, 2)

        # --- manifest -------------------------------------------------------
        manifest = {
            "stemscribe_version": "0.1.0",
            "input": {
                "path": str(input_path.resolve()),
                "sha256": _file_sha256(input_path),
                "bytes": input_path.stat().st_size,
            },
            # Where the audio came from. A hash identifies a file; it cannot say
            # what that file is or who made it. For anything that might end up
            # in a commercial path, that answer needs to survive the pipeline.
            "source": source.as_dict() if source else None,
            "params": {
                "backend": backend,
                "demucs_model": demucs_model,
                "include_vocals_melody": include_vocals_melody,
                "cleanup": cleanup,
                "cleanup_params": cp.as_dict() if cleanup else None,
                "keep_stems": keep_stems,
                "instrumental": instrumental,
                "prepare": pp.as_dict(),
                "backend_kwargs": backend_kwargs or {},
            },
            "tempo": tempo_est.as_dict(),
            "grid": grid_info,
            "drums": {"backend": drums, "stems": sorted(drum_midis)} if drum_midis else None,
            "fallbacks": fallbacks,
            "prepared_audio": prepared.as_dict(),
            "cache": cch.summary(),
            "outputs": {
                "midi": str(midi_path),
                "instrumental": str(instrumental_path) if instrumental_path else None,
                "stems": {k: str(v) for k, v in stem_paths.items()} if keep_stems else None,
            },
            "track_map": track_map,
            "tracks": track_stats,
            "timings_sec": timings,
            "warnings": warnings,
        }
        manifest = _jsonable(manifest)
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default))

        return Result(
            midi_path=midi_path,
            stem_paths=stem_paths if keep_stems else {},
            instrumental_path=instrumental_path,
            track_map=track_map,
            manifest_path=manifest_path,
            manifest=manifest,
            warnings=warnings,
            tempo=tempo_est,
            prepared=prepared,
            source=source,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
