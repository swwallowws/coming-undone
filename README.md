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
  └─ 3. transcribe    per-stem audio→MIDI (pluggable backend; drums via ADT_STR;
  │                   a nearly empty stem is re-run with basic-pitch)
  └─ 4. cleanup       de-overlap + duration-trim pass
  └─ 5. merge         one multi-track MIDI, GM programs + named tracks
  └─ 6. realign       shift MIDI back onto the original file's timeline
  └─ 7. grid          tempo + real bar lines fitted from the tightest track;
                      optional latency removal + snapping (--snap)
```

## Install

Needs Python 3.11 and `ffmpeg` on PATH. Developed on macOS (Apple Silicon);
Linux works.

The tested setup is one virtualenv at `.venv` with torch and torchaudio pinned
to 2.8.0 (no torchcodec, see Known gotchas):

```bash
python3.11 -m venv .venv
.venv/bin/pip install 'torch==2.8.0' 'torchaudio==2.8.0'
.venv/bin/pip install -e '.[web,fetch,drums,dev]'
```

Or pick extras one at a time:

```bash
pip install -e .                 # library + CLI
pip install -e '.[web]'          # + local web UI
pip install -e '.[fetch]'        # + fetch audio from a URL (yt-dlp)
pip install -e '.[drums]'        # + drum transcription (ADT_STR, CC BY-SA 4.0)
pip install -e '.[muscriptor]'   # + the default backend (non-commercial, see Backends)
```

The default backend, muscriptor, is not in the base install. Without that extra,
pass `--backend basic-pitch` (or `backend="basic-pitch"`).

No audio, stems, MIDI or reference songs are included in this repo, and no model
weights: the models download on first use and keep their own licences (see
Licenses).

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
| `song.mid` | multi-track MIDI: melody (from vocals), bass, comping, drums; real tempo map and bar lines |
| `song_instrumental.mp3` | stems-minus-vocals mix |
| `stems/{drums,bass,other,vocals}.wav` | the separated stems |
| `manifest.json` | input hash, params, backend, per-track note counts, quantization-error stats, the beat grid (tempo, bar lines, per-track latency and fit), drum backend, fallbacks, timings, warnings |

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

### Drums

Pitched backends can't transcribe an unpitched kit, so the drums stem goes to its
own registry, `DRUM_BACKENDS`. The one there is **ADT_STR** (Melucci, Merialdo,
Akama 2026, [github.com/pier-maker92/ADT_STR](https://github.com/pier-maker92/ADT_STR)),
code and weights CC BY-SA 4.0: commercial use is allowed with credit. Install
`pip install 'stemscribe[drums]'`; the first run downloads the model repo from
Hugging Face at a pinned revision (`backends.ADT_STR_REVISION`), because that repo
ships code stemscribe imports. Without the extra the drums stem is skipped with a
warning, as before, and a drum model that fails to load warns instead of sinking
the run. `--drums none` turns it off.

**Environment catch:** ADT_STR's own pyproject pins `torch==2.8.0`,
`torchaudio==2.8.0` and `torchcodec`, and torchcodec is exactly what breaks
demucs's save path here (see Known gotchas; `separate.py` avoids it). The
`[drums]` extra lists only what the model imports and does not force those pins.
The tested environment (see Install) pins torch and torchaudio 2.8.0 and leaves
torchcodec out.

ADT_STR writes its own "GM custom" class numbers without converting them back;
stemscribe maps each class to the first standard GM drum in it
(`backends.ADT_STR_TO_GM`). It runs early: 41 ms on the reference song, which is
why snapping removes each track's latency first (see Beat grid).

CC BY-SA's share-alike clause covers adaptations of the model itself. My reading is
that transcribed MIDI is output, not an adaptation, but that is a reading, not
legal advice: check it before shipping drums commercially.

### Sparse stems

A backend can come back nearly empty on a stem that clearly has sound in it
(MuScriptor gave 3 bass notes for a whole song). Under one note per 4 bars, on a
stem that is not silent and at least 8 bars long, stemscribe transcribes that stem
again with basic-pitch and records it under `fallbacks` in the manifest.
`--no-fallback` turns it off.

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
the scored alternates. `tempo.ESTIMATORS` is the registry where a better
estimator drops in, exactly like `BACKENDS`. This tempo is only the starting
guess: the grid stage refines it (below).

## Beat grid

A tempo alone does not make bars. stemscribe fits a real grid, one constant tempo
plus bar lines, and writes it into the MIDI. This matters downstream: rearranged
is MIDI-only and copies its donor's timing, and a transcription without a real
grid (a 120 BPM placeholder, half its notes 20 ms or more off) had to be repaired
by hand.

- **The tempo comes from the notes.** A good backend writes onsets on the song's
  real 16th grid to within about 1 ms, while beat trackers land a percent or so off
  (beat_this was 1.2% fast on the reference song), which drifts a bar away over a
  song. Each track is fitted within 3% of the detected tempo, and the best-aligned
  one with at least 64 onsets sets the grid. On the reference song, per-stem
  MuScriptor's comping gave 113.998 BPM against a truth of 114.0; even basic-pitch's
  comping found it.
- **Bar "one" is a guess you can correct.** It is the beat of four where the
  harmony changes most. That was exact on a sequenced song but a near tie on a song
  whose chords are pushed ahead of the bar; low-end drum energy, tried first, was
  two beats off. The confidence goes into the manifest, a low one warns, and
  `--downbeat 2|3|4` says which beat of the guessed bar is really "one". The web UI
  moves it a beat at a time and re-stamps instantly.
- **No note moves for the grid.** The first bar is a pickup of its own tempo that
  ends exactly on the first real bar line; from there the tempo is the fitted one.
- **Snapping is opt-in (`--snap`).** It removes each track's latency (its median
  signed offset from the grid) and then puts starts on 16ths, ends on 32nds and
  keeps one hit per drum per 16th. Removing latency first matters: ADT_STR's drums
  ran 41 ms early, and plain snapping pushed 4 hits in 10 onto the previous 16th.
  Off by default because snapping deletes real feel.
- **The manifest's `grid`** records the tempo, first bar line, source track, the
  "one" confidence and any override, and per track its alignment, latency and
  median distance to the grid (`grid_fit_ms`).
- If no track sits on a steady grid, the MIDI keeps the detected tempo, nothing is
  snapped, and a warning says so. `--no-grid` skips the stage.

**Any MIDI, not only stemscribe's:** `stemscribe-grid in.mid -o out.mid [--snap]
[--downbeat N] [--tempo BPM]` fits and stamps a grid on a file transcribed
elsewhere and writes `out.grid.json` beside it. Without `--tempo` it searches 60
to 200 BPM, since such files often carry a placeholder tempo map.

One constant tempo per song: a song that speeds up or has tempo changes needs
more than this.

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

v1 processes whole songs. Section cutting stays with the caller. Chord labels,
section detection, melody extraction from instrumental tracks and tempo changes
are non-goals, hooks only. Vocals need a dedicated singing-transcription model
later: MuScriptor does not transcribe singing (0 notes on the reference song's
vocal stem).

The cleanup duration cap and the manifest's older quantization-error stats run
before the grid stage, against a constant tempo from t=0; they are a relative
legibility signal. The grid's per-track `grid_fit_ms` is the measured one.

## Licenses

stemscribe's own code is MIT (see `LICENSE`). That licence covers this
repository's code only. The models stemscribe downloads and the libraries it
depends on keep their own licences, and some of them do not allow commercial
use. The vendored fonts in `src/stemscribe/web/static/vendor/design/fonts/` are
SIL OFL 1.1 (licence files beside them).

Every model dependency, stated explicitly:

| Component | Code | Weights | Commercial path |
|---|---|---|---|
| demucs (htdemucs) | MIT | **research only** | **no** |
| basic-pitch | Apache-2.0 | Apache-2.0 | yes |
| MuScriptor (optional extra) | MIT | CC-BY-NC | no |
| ADT_STR drums (optional extra) | CC BY-SA 4.0 | CC BY-SA 4.0 | yes, with credit (see Drums) |
| soundfile, pretty_midi, librosa, yt-dlp | BSD, MIT, ISC, Unlicense | none | yes |

**demucs's weights are not MIT.** Its author, on facebookresearch/demucs#327
(2022-05-23): "The model weights are not covered by the MIT license, and are
provided only for scientific purposes" (they are trained on MUSDB). An earlier
version of this README called demucs fine for a commercial path; that was wrong.
`STEMSCRIBE_COMMERCIAL=1` guards the transcription backends only, not separation,
so a commercial build needs a different separator first.

**Separation decision (2026-09-25): keep htdemucs for v1.** v1 is a portfolio
tool, where research-only weights are fine, and htdemucs is well tested here.
Compared: BS-Roformer and SCNet separate better (higher SDR) but their public
weights mostly carry no licence, so they are no better for a commercial path. The
first commercial candidate is the Mel-Band-Roformer vocal checkpoint by Kimberley
Jensen, MIT since April 2026 (vocals only; the other stems still need a licensed
model). Revisit before any commercial release.

**basic-pitch is the commercial-safe transcription path** (confirmed 2026-09-25):
with `STEMSCRIBE_COMMERCIAL=1`, MuScriptor hard-fails before separation starts and
basic-pitch runs; `tests/test_commercial_guard.py` holds both.

Tooling licenses are only half of it: what you feed the pipeline carries its own
rights, and that's on the operator, not the code. Anything you download with the
URL fetch (yt-dlp) is your responsibility: make sure you have the right to
download and process it. `manifest.json` records the source URL for fetched audio
so the question stays answerable later.

## Tests

```bash
pip install -e '.[dev]' && pytest
```
