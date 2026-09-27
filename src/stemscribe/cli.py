"""stemscribe CLI -- mirrors stemscribe.process()."""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from . import __version__
from .backends import BACKENDS, BackendError
from .cache import DEFAULT_ROOT, Cache, human_bytes
from .cleanup import CleanupParams
from .core import process
from .fetch import FetchError
from .grid import DEFAULT as DEFAULT_METER, Meter
from .prepare import PrepareError, PrepareParams


def _meter(spec: str) -> Meter:
    try:
        return Meter.parse(spec)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _add_meter_args(g) -> None:
    g.add_argument("--meter", type=_meter, default=DEFAULT_METER,
                   help="the bar: 4/4 (default), 3/4, 6/8, 12/8, or grouped like 9/8:2+2+2+3 "
                        "(9/8 alone is 2+2+2+3). The grid counts the denominator note")
    g.add_argument("--downbeat", type=int, default=None,
                   help="which beat of the guessed bar is really 'one' (1 to the meter's "
                        "numerator; 1-4 in 4/4)")


def _add_tempo_mode_arg(g) -> None:
    g.add_argument("--tempo-mode", default="auto", choices=("auto", "constant", "map"),
                   help="auto (default): one tempo when it fits, else a tempo map that follows "
                        "a live band's drift. constant: always one tempo. map: always follow")


def _check_downbeat(p: argparse.ArgumentParser, a: argparse.Namespace) -> None:
    if a.downbeat is not None and not 1 <= a.downbeat <= a.meter.pulses:
        p.error(f"argument --downbeat: must be 1 to {a.meter.pulses} in {a.meter}, got {a.downbeat}")


def _later_beats(m: Meter) -> str:
    return "/".join(map(str, range(2, m.pulses + 1)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stemscribe",
        description="Song in, stems + labeled multi-track MIDI out.",
    )
    p.add_argument(
        "input", help="audio file (mp3/wav/m4a), or a URL yt-dlp can fetch"
    )
    p.add_argument("-o", "--out-dir", required=True, help="output directory")
    p.add_argument("--version", action="version", version=f"stemscribe {__version__}")

    p.add_argument(
        "--backend",
        default="muscriptor",
        choices=sorted(BACKENDS),
        help="transcription backend (default: muscriptor, best quality, CC-BY-NC "
        "non-commercial). Use basic-pitch (Apache-2.0) for anything commercial",
    )
    p.add_argument(
        "--demucs-model",
        default="htdemucs",
        help="htdemucs (default, 4 stems) or htdemucs_6s, which adds separate "
        "guitar and piano stems (slower; those two stems are experimental)",
    )
    p.add_argument("--device", default=None, help="torch device (default: auto)")
    p.add_argument(
        "--tempo",
        type=float,
        default=None,
        help="song BPM, written into the MIDI. Omit and it is detected from the "
        "drums stem -- you only need this when you disagree with the detection. "
        "Counted in the meter's pulse: in 9/8 it is the eighth, and the grid keeps it",
    )

    p.add_argument(
        "--no-vocals-melody",
        dest="include_vocals_melody",
        action="store_false",
        help="skip the vocal stem (no melody track)",
    )
    p.add_argument(
        "--midi-only",
        action="store_true",
        help="write only the MIDI: no stems, no instrumental",
    )
    p.add_argument("--no-stems", dest="keep_stems", action="store_false")
    p.add_argument("--no-instrumental", dest="instrumental", action="store_false")
    p.add_argument(
        "--instrumental-format", default="mp3", choices=("mp3", "wav")
    )

    f = p.add_argument_group("fetching (when input is a URL)")
    f.add_argument(
        "--audio-format",
        default="native",
        choices=("native", "mp3"),
        help="native (default) keeps the source codec -- best quality, since the "
        "pipeline decodes to wav anyway. mp3 transcodes a file you can keep",
    )
    f.add_argument(
        "--no-keep-source",
        dest="keep_source",
        action="store_false",
        help="discard the downloaded audio after processing",
    )

    s = p.add_argument_group("input conditioning")
    s.add_argument(
        "--no-trim-silence",
        dest="trim_silence",
        action="store_false",
        help="keep leading/trailing silence (default: trim it)",
    )
    s.add_argument(
        "--keep-metadata",
        dest="strip_metadata",
        action="store_false",
        help="keep ID3 tags and cover art (default: strip them)",
    )
    s.add_argument(
        "--no-normalize",
        dest="normalize_wav",
        action="store_false",
        help="read the input directly instead of decoding to a standard wav first",
    )
    s.add_argument(
        "--start", type=float, default=None, help="process from this many seconds in"
    )
    s.add_argument(
        "--duration", type=float, default=None, help="process only this many seconds"
    )
    s.add_argument(
        "--silence-top-db",
        type=float,
        default=50.0,
        help="dB below peak that counts as silence (default: 50)",
    )

    c = p.add_argument_group("cleanup")
    c.add_argument("--no-cleanup", dest="cleanup", action="store_false")
    c.add_argument(
        "--max-duration-beats",
        type=float,
        default=2.0,
        help="trim notes longer than this that have re-onsets underneath (default: 2)",
    )
    c.add_argument(
        "--velocity-floor",
        type=int,
        default=15,
        help="drop notes below this velocity (default: 15)",
    )
    c.add_argument("--no-de-overlap", dest="de_overlap", action="store_false")
    c.add_argument(
        "--mono",
        default="",
        metavar="STEMS",
        help="comma-separated stems to collapse to one note at a time, keeping "
        "the lowest (e.g. --mono bass). Strips overtones and chord bleed; also "
        "strips real double-stops",
    )
    c.add_argument(
        "--legato",
        action="store_true",
        help="close the small gaps basic-pitch leaves, so notes almost touch. "
        "Rests are preserved -- only gaps under --legato-max-gap-beats close",
    )
    c.add_argument(
        "--legato-max-gap-beats",
        type=float,
        default=0.25,
        help="largest gap legato will close, in beats (default: 0.25, a 16th)",
    )
    c.add_argument(
        "--quantize",
        action="store_true",
        help="snap note onsets toward the grid so the MIDI lines up with the "
        "bars in a DAW. This is the fix for a MIDI that looks 'not aligned'",
    )
    c.add_argument(
        "--quantize-division",
        type=int,
        default=4,
        help="grid resolution as a division of the beat: 4=16ths, 2=8ths (default: 4)",
    )
    c.add_argument(
        "--quantize-strength",
        type=float,
        default=0.5,
        help="how hard to snap, 0..1. 1.0 is rigid, 0.5 keeps human feel (default: 0.5)",
    )

    g = p.add_argument_group("beat grid and drums")
    g.add_argument("--no-grid", dest="grid", action="store_false",
                   help="do not fit a beat grid (the MIDI gets the detected tempo only)")
    g.add_argument("--snap", action="store_true",
                   help="remove each track's latency and snap it to the grid (off: keeps feel)")
    _add_meter_args(g)
    _add_tempo_mode_arg(g)
    g.add_argument("--drums", default="adt-str", choices=("adt-str", "none"),
                   help="drum transcription (adt-str needs the [drums] extra; default: adt-str)")
    g.add_argument("--no-fallback", dest="fallback", action="store_false",
                   help="do not re-transcribe a nearly empty stem with basic-pitch")

    k = p.add_argument_group("cache")
    k.add_argument(
        "--no-cache",
        dest="cache",
        action="store_false",
        help="recompute stems and transcription even if cached",
    )
    k.add_argument("--cache-dir", default=None, help=f"default: {DEFAULT_ROOT}")
    k.add_argument(
        "--clear-cache",
        action="store_true",
        help="delete the cache and exit (stems are ~230MB per song)",
    )

    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Handled before the parser: --clear-cache takes no input file.
    if "--clear-cache" in argv:
        i = argv.index("--clear-cache")
        root = None
        if "--cache-dir" in argv:
            root = argv[argv.index("--cache-dir") + 1]
        c = Cache(root)
        freed = c.clear()
        print(f"cleared {human_bytes(freed)} from {c.root}")
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    _check_downbeat(parser, args)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    keep_stems = args.keep_stems and not args.midi_only
    instrumental = args.instrumental and not args.midi_only

    try:
        res = process(
            args.input,
            out_dir=args.out_dir,
            backend=args.backend,
            include_vocals_melody=args.include_vocals_melody,
            cleanup=args.cleanup,
            cleanup_params=CleanupParams(
                de_overlap=args.de_overlap,
                max_duration_beats=args.max_duration_beats,
                velocity_floor=args.velocity_floor,
                legato=args.legato,
                legato_max_gap_beats=args.legato_max_gap_beats,
                quantize=args.quantize,
                quantize_division=args.quantize_division,
                quantize_strength=args.quantize_strength,
            ),
            prepare=PrepareParams(
                normalize_wav=args.normalize_wav,
                strip_metadata=args.strip_metadata,
                trim_silence=args.trim_silence,
                silence_top_db=args.silence_top_db,
                start=args.start,
                duration=args.duration,
            ),
            audio_format=args.audio_format,
            keep_source=args.keep_source,
            cache=Cache(args.cache_dir, enabled=args.cache),
            mono_stems=tuple(s.strip() for s in args.mono.split(",") if s.strip()),
            keep_stems=keep_stems,
            instrumental=instrumental,
            instrumental_format=args.instrumental_format,
            demucs_model=args.demucs_model,
            device=args.device,
            tempo=args.tempo,
            drums=None if args.drums == "none" else args.drums,
            grid=args.grid,
            snap=args.snap,
            downbeat=args.downbeat,
            fallback=args.fallback,
            meter=args.meter,
            tempo_mode=args.tempo_mode,
        )
    except (BackendError, FetchError, PrepareError, FileNotFoundError, RuntimeError) as e:
        print(f"stemscribe: {e}", file=sys.stderr)
        return 1

    if res.source:
        who = f" — {res.source.uploader}" if res.source.uploader else ""
        print(f"\nsource     {res.source.title or res.source.url}{who}")

    t = res.tempo
    detail = f"{t.bpm:.2f} BPM ({t.source})"
    if t.source == "detected":
        alts = ", ".join(f"{c.bpm:.0f}" for c in t.candidates if c.ratio != 1.0)
        if alts:
            detail += f" -- if wrong, try --tempo {alts}"
    print(f"\ntempo      {detail}")
    gi = res.manifest.get("grid") or {}
    if gi.get("tempo") == "map":
        lo, hi = gi["bpm_range"]
        print(f"tempo map  {lo:.1f} to {hi:.1f} BPM (the band drifts; one tempo sat "
              + ("nowhere" if gi.get("constant_offset_16th") is None
                 else f"{gi['constant_offset_16th']:.2f} of a 16th off")
              + "), a tempo change per beat in the MIDI")
    if gi.get("fitted"):
        print(f"grid       first bar line at {gi['first_bar']:.2f}s (from {gi['source_track']}, "
              f"'one' confidence {gi['bar_one_confidence']:.2f})"
              + (", snapped" if gi["snapped"] else "")
              + f" -- if 'one' is wrong, try --downbeat {_later_beats(args.meter)}")
    for w in res.warnings:
        print(f"warning    {w}")

    tracks = ", ".join(f"{n}({res.manifest['tracks'][n]['note_count']})" for n in res.track_map)
    print(f"MIDI       {res.midi_path}  [{tracks}]")
    if res.instrumental_path:
        print(f"instrumental {res.instrumental_path}")
    if res.stem_paths:
        print(f"stems      {next(iter(res.stem_paths.values())).parent}")
    print(f"manifest   {res.manifest_path}")
    print(f"took       {res.manifest['timings_sec']['total']}s")
    return 0


def grid_main(argv: list[str] | None = None) -> int:
    """stemscribe-grid: fit and stamp a beat grid on any MIDI (a donor transcribed
    elsewhere, say), optionally snapping it. Writes OUT and OUT's .grid.json report."""
    import json

    import pretty_midi

    from . import grid as _grid

    p = argparse.ArgumentParser(prog="stemscribe-grid", description=grid_main.__doc__)
    p.add_argument("midi")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--tempo", type=float, default=None,
                   help="a tempo to search near, counted in the meter's pulse (in 9/8 the "
                        "eighth), which it then keeps (default: search 60-200 BPM; the "
                        "file's own tempo map is often a placeholder)")
    p.add_argument("--snap", action="store_true")
    _add_meter_args(p)
    _add_tempo_mode_arg(p)
    a = p.parse_args(argv)
    _check_downbeat(p, a)
    try:
        pm = pretty_midi.PrettyMIDI(a.midi)
        out, info, warnings = _grid.apply(pm, a.tempo, snap_notes=a.snap, downbeat=a.downbeat,
                                          meter=a.meter, tempo=a.tempo_mode,
                                          fixed_pulse=a.tempo is not None)
    except (OSError, ValueError) as e:
        print(f"stemscribe-grid: {e}", file=sys.stderr)
        return 1
    for w in warnings:
        print(f"warning    {w}", file=sys.stderr)
    if not info["fitted"]:
        return 1
    dst = pathlib.Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(dst))
    dst.with_suffix(".grid.json").write_text(json.dumps(info, indent=2))
    if info.get("tempo") == "map":
        print(f"tempo map  {info['bpm_range'][0]:.1f} to {info['bpm_range'][1]:.1f} BPM, "
              "a tempo change per beat")
    print(f"grid       {info['bpm']:.3f} BPM from {info['source_track']}, first bar line at "
          f"{info['first_bar']:.2f}s ('one' confidence {info['bar_one_confidence']:.2f})"
          + (", snapped" if a.snap else ""))
    print(f"MIDI       {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
