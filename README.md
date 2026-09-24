# stemscribe

Song in, stems + labeled multi-track MIDI out.

Takes an audio file (with or without vocals), separates it into stems, and
transcribes it into a multi-track MIDI with named tracks and GM programs.
Built to be rearranged's transcription front-end; useful standalone for any
producer workflow.

```
audio in (mp3/wav/m4a) — a file, or a URL
  └─ 0. fetch         yt-dlp, when handed a URL (provenance → manifest)
  └─ 0. prepare       decode → strip metadata → section → trim silence
  └─ 1. separate      demucs htdemucs → drums / bass / other / vocals
  └─ 1b. tempo        detect BPM from the isolated drums stem
  └─ 2. instrumental  mix drums+bass+other
  └─ 3. transcribe    per-stem audio→MIDI (pluggable backend; drums skipped)
  └─ 4. cleanup       de-overlap + duration-trim pass
  └─ 5. merge         one multi-track MIDI, GM programs + named tracks
  └─ 6. realign       shift MIDI back onto the original file's timeline
```

## Install

```bash
pip install -e .            # library + CLI
pip install -e '.[web]'     # + local web UI
pip install -e '.[fetch]'   # + fetch audio from a URL (yt-dlp)
```

Needs Python 3.11 and `ffmpeg` on PATH. Developed on macOS (Apple Silicon);
Linux works.

## Use

```python
from stemscribe import process

res = process("song.mp3", out_dir="out/")  # default backend: muscriptor
# res = process("song.mp3", out_dir="out/", backend="basic-pitch")  # commercial-safe

res.midi_path          # out/song.mid  (multi-track)
res.stem_paths         # {"drums": ..., "bass": ..., "other": ..., "vocals": ...}
res.instrumental_path  # out/song_instrumental.mp3
res.track_map          # {"melody": 0, "bass": 1, "comping": 2}
res.tempo              # TempoEstimate: .bpm, .source, .candidates
res.prepared           # PreparedAudio: .offset, what was trimmed
res.manifest_path      # out/manifest.json
```

```bash
stemscribe song.mp3 -o out/ --no-vocals-melody --midi-only  # default: muscriptor
stemscribe song.mp3 -o out/ --backend basic-pitch           # commercial-safe, fast
stemscribe song.mp3 -o out/ --start 30 --duration 20      # fast iteration
stemscribe 'https://www.youtube.com/watch?v=...' -o out/  # URL works anywhere a path does
stemscribe --help
```

### Fetching from a URL

Anywhere `process()` or the CLI takes a path, it also takes a URL that yt-dlp
can handle; the web UI has a link field next to the drop zone.

```python
res = process("https://www.youtube.com/watch?v=...", out_dir="out/")
res.source.title      # what it was
res.source.url        # where it came from
```

**Default keeps the native codec, not mp3.** YouTube serves Opus/AAC, and
`prepare` decodes to wav before demucs sees anything — so transcoding to mp3 in
between is a second lossy generation that costs quality and buys nothing. Use
`--audio-format mp3` when you want a file to keep.

**If a URL 403s, your yt-dlp is stale.** YouTube changes its player constantly
and yt-dlp patches roughly weekly, so an extractor a few months old fails with
`HTTP Error 403: Forbidden` on the data download while metadata still resolves
fine. That is not a stemscribe bug and no flag will fix it:

```bash
pip install -U yt-dlp     # first thing to try, every time
```

Treat yt-dlp as a rolling dependency, not a pinned one. `fetch.py` names the
version in its 403 message so the next person doesn't have to bisect it.

**YouTube also needs a JavaScript runtime** to decipher stream signatures.
yt-dlp only enables `deno` by default, so a machine with `node` and no `deno`
loses formats and 403s with only a warning to explain why. `fetch.py` detects
whatever is on PATH (`deno`, `node`, `bun`, `quickjs`) and enables it. If you
have none: `brew install deno`.

Fetched audio lands in `out/source/` (`--no-keep-source` discards it), and the
URL, title, uploader and duration go into `manifest.json` under `source`. That
matters more than it looks: stemscribe hashes its input, but a hash tells you
*which* file, never *whose*. Since rearranged is meant to ship commercially,
being able to answer "where did this come from?" months later — for material
that may be someone else's recording — is worth the two lines it costs. The same
diligence that keeps MuScriptor's CC-BY-NC weights out of the commercial path
applies to the audio going in.

### Web UI

```bash
stemscribe-web            # → http://127.0.0.1:8000
```

Drop a file, watch progress stream, audition stems, download MIDI. Tempo is
detected automatically and never asked about up front; if the grid looks wrong,
the alternates are one click away and re-stamp instantly.

It is **localhost-only by design**: no auth, runs jobs on your machine, reads
and writes your files. Don't expose it.

### Outputs

| File | What |
|---|---|
| `song.mid` | multi-track MIDI: melody (from vocals), bass, comping |
| `song_instrumental.mp3` | stems-minus-vocals mix |
| `stems/{drums,bass,other,vocals}.wav` | the separated stems |
| `manifest.json` | input hash, params, backend, per-track note counts, quantization-error stats, timings, warnings |

Track names (`melody`, `bass`, `comping`) are the contract downstream consumers
read — `melody` is always the vocal line. Renaming them breaks rearranged.

## Backends

| Backend | License | Notes |
|---|---|---|
| `muscriptor` (default) | code MIT, **weights CC-BY-NC** | Best quality, esp. on dense polyphony. Slower (~3 min/stem). **Non-commercial only.** No velocity data. |
| `basic-pitch` | Apache-2.0 | Commercial-safe fallback. Fast (~1 min). Pitched stems only; run per stem, never on the full mix. |

The default is **muscriptor** because it produces the best transcriptions and is
free for personal / non-commercial use. Its weights are CC-BY-NC, so it must
**not** be used in anything you ship.

**Commercial safety.** Because the default is now the non-commercial backend,
the "safe by default" property is gone. Set `STEMSCRIBE_COMMERCIAL=1` in the
environment and stemscribe will *hard-fail* on any CC-BY-NC backend before doing
any work, so a commercial build (rearranged shipping, say) cannot land
non-commercial weights by accident regardless of the default. Or just pass
`backend="basic-pitch"` explicitly. `process()` also records the license in
`manifest.json` and logs a warning whenever a non-commercial backend runs.

`muscriptor` is an optional extra (`pip install 'stemscribe[muscriptor]'`) and
needs `hf auth login` plus accepted terms at
[huggingface.co/MuScriptor/muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium).
First run downloads ~2GB into the HF cache.

Add a backend in `backends.py`: a callable `(stem_wav, out_mid) -> Path | None`
registered in `BACKENDS`. Klangio and YourMT3+ can slot in the same way.

**Drums are skipped.** Pitched backends can't transcribe an unpitched kit; the
skip is recorded in `manifest.json`. `backends.transcribe_drums()` is the hook.

## Cleanup

Raw transcriptions of reverby comping come out as walls of sustained
overlapping notes, and downstream style engines then read the texture as "pad"
instead of "stabs". Per track, cleanup:

1. **Velocity floor** — drop notes below velocity 15 (transcription noise).
2. **De-overlap** — same-pitch overlapping notes: trim the earlier note's end to
   the next note's start.
3. **Duration cap** — notes longer than N beats (default 2) that have *other*
   notes re-onsetting during their sustain get trimmed to the next onset. Long
   notes over silence are left alone; they're probably real.
4. **Report** — before/after note counts and median duration, into the manifest.

Toggle with `cleanup=False` / `--no-cleanup`, or tune via `CleanupParams`
(`--max-duration-beats`, `--velocity-floor`, `--no-de-overlap`). The duration
cap is measured in beats against the detected tempo, so it means the same
musical thing at any BPM.

**How much does it actually do?** On `koprualti` via per-stem basic-pitch:
almost nothing — 1731 → 1727 notes, median duration unchanged. That is the
correct result, not a bug. The "wall of sustained notes" this pass was written
for comes from transcribing a *dense* source; per-stem transcription on
separated audio (which this pipeline mandates) mostly prevents it upstream.
Measured on that material: median 0.257s, polyphony 1.68, only 8 of 1731 notes
over the 1s cap, and zero notes under velocity 15.

So the pass earns its keep defensively — for muscriptor, for reverby material,
for backends that smear — and the defaults are deliberately *not* tuned harder
just to move the number. Trimming aggressively enough to drop the median would
delete real notes.

The smear is real, though — it's just on the other side of a threshold. Running
basic-pitch with `frame_threshold=0.05` on the same bass stem yields a median
note duration of **4.3 seconds** and zero silence: exactly the wall of sustained
overlapping notes the spec describes. Don't go there.

## Legato (the opposite problem)

basic-pitch ends a note when frame activation drops below `frame_threshold`
(default 0.3), which is well before the note stops sounding. The result is
output that's too *staccato*: measured on koprualti, bass came out **68%
silence**, melody 50%.

`legato=True` / `--legato` closes those gaps by extending each note to the next
onset, like Ableton's Span (legato) or its Legato command. Off by default — it
pushes a track toward "pad", the exact reading the rest of cleanup exists to
prevent.

**It only closes gaps under `legato_max_gap_beats` (default 0.25, a 16th).**
That guard is the whole design: an unconditional stretch swallows rests and
phrase endings, turning silence someone played on purpose into sustain nobody
did. A gap it won't close is information — it's telling you the note is *really*
that short, and the fix is upstream, not here.

Don't reach for `frame_threshold` to get legato. It changes which notes are
detected, not just their length — at 0.1 the bass stem explodes from 411 notes
to 10,594. Extending note ends is deterministic: it invents nothing and loses
nothing.

`silence_ratio_before/after` in the manifest is the number that tracks this.
Median duration can't see it — shortening every note and spreading the same
notes further apart move the median identically.

## Tempo

You should not have to think about tempo, so by default you don't: it's
detected and written into the MIDI. Pass `tempo=` / `--tempo` only when you
disagree.

Detection runs on the **isolated drums stem** (demucs has already produced it,
and beat tracking on drums beats beat tracking on a dense mix), then refines the
BPM by least-squares fit through the detected beat times. That refinement
matters: librosa's own tempo is quantized to tempogram bins and lands ~2.5% off
a true 140 BPM, which drifts the bar grid several seconds away from the notes
over a full song. The fit gets within ~0.01%.

Embedding the tempo **never moves a note**. `pretty_midi` stores note times in
seconds; tempo only decides where the bar lines fall. That's also why fixing an
octave error is instant and lossless — see the web UI's alternates, or
`POST /api/jobs/{id}/tempo`.

Beat trackers confuse half and double time, so `res.tempo.candidates` carries
the scored alternates. A full beat/downbeat grid is a non-goal (rearranged's
glue research owns it, via beat_this): `tempo.ESTIMATORS` is the registry where
a better estimator drops in, exactly like `BACKENDS`.

## Input conditioning

Before anything expensive touches the audio, `prepare` can decode to a standard
44.1k stereo wav, strip metadata and cover art, cut to a section, and trim
leading/trailing silence. All optional; all recorded in the manifest.

**The timing contract:** any offset introduced here is added back at the end, so
MIDI note times always refer to the *original* file you passed in. Trim 4.2s of
dead air and the first note still reports at 4.2s — because in your file, it is.
`res.prepared.offset` is that number.

`--start` / `--duration` are for fast iteration (a 20s section runs in ~14s vs
~150s for a full song). Note the spec assigns real section cutting to
rearranged; this is a convenience, not a takeover.

## Known gotchas (encoded here so you don't rediscover them)

- **demucs CLI save crashes** on torchaudio ≥2.9 (`ModuleNotFoundError:
  torchcodec`). `separate.py` uses the demucs *API* + `soundfile.write` instead.
  Don't "simplify" it back to `demucs.separate`.
- **ffmpeg `amix`**: `normalize=0` isn't in every build. `mixdown.py` uses
  `amix=inputs=N,volume=N`, which is the portable way to sum rather than average.
- **basic-pitch**: the Python API (`predict`), not the CLI — the CLI pays the
  import cost per call.
- **Summing stems clips.** drums+bass+other overshoots 0dBFS where the original
  mix was already near full scale (measured 1.19 peak on real material).
  `mixdown.py` ends the chain with `alimiter`; `limit=0` opts out.
- **yt-dlp's `js_runtimes` takes a dict, not a list.** The CLI's `--js-runtimes`
  accepts a list, but the Python API wants `{runtime: {config}}` and raises
  `ValueError` otherwise. Pass `{}` as the config, not `None` — validation
  accepts `None`, then something downstream calls `.get()` on it.
- **`alimiter=level` defaults to true**, which auto-levels the limited signal
  back up to 0dBFS and silently undoes `limit`. It must be `level=disabled` or
  the limiter does nothing you can measure. An mp3 of a dense limited mix still
  decodes ~1.1 peak regardless — lossy reconstruction overshoots its source, and
  that's normal; use `--instrumental-format wav` if you need provable headroom.
- **numpy leaks into the manifest.** `pretty_midi` returns `instrument.program`
  as `int64`, and `statistics.median` over note times returns `float64`; both
  are unserializable. `core._jsonable` cleans the manifest as a data structure,
  not just on write — `res.manifest` is a public surface that other code
  serializes with its own serializer.
- mp3 input is fine; wav is not required.

## Scope

v1 processes whole songs. Section cutting stays with the caller. Beat/downbeat
grids, chord labels, section detection, melody extraction from instrumental
tracks, and drum transcription are explicit non-goals — hooks only.

Because there's no beat tracker, anything beat-based (the cleanup duration cap,
the manifest's quantization-error stats) assumes a constant `tempo` (default
120). The quantization numbers are a *relative* legibility signal, not ground
truth.

## Licenses

demucs MIT, basic-pitch Apache-2.0, soundfile BSD, pretty_midi MIT, yt-dlp
Unlicense — all fine for a commercial path. MuScriptor's weights are CC-BY-NC
and are kept as an optional, clearly-marked extra.

Tooling licenses are only half of it: what you feed the pipeline carries its own
rights, and that's on the operator, not the code. `manifest.json` records the
source URL for fetched audio so the question stays answerable later.

## Tests

```bash
pip install -e '.[dev]' && pytest
```
