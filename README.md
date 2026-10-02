# Coming Undone

simply split.

Formerly stemscribe; the code, package and commands keep that name.

A song in, its parts out, each one written down as a named MIDI track.

Takes an audio file (with or without vocals), separates it into its parts
(stems), and writes them down as a multi-track MIDI file with named tracks and
General MIDI instruments. Built to be Rearranged's transcription front-end;
useful on its own in any producer workflow.

```
audio in (mp3/wav/m4a): a file, or a URL
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
  └─ 7. grid          tempo + real bar lines fitted from the tightest track,
                      a tempo map when a live band drifts (--tempo-mode);
                      optional latency removal + snapping (--snap)
```

## Run it on this computer (for the website)

The website (https://swwallowws.github.io/coming-undone/) runs a song Online or In
your browser. "This computer" runs it here instead: full quality, whole songs, and the
audio never leaves the machine. It needs Python 3.11, `ffmpeg` and a few GB of disk:
torch, and the models, which download on the first run (MuScriptor alone is about 2 GB).

1. Install Python 3.11 and ffmpeg (on a Mac: `brew install python@3.11 ffmpeg`).
2. Get the code and install it:
   ```bash
   git clone https://github.com/swwallowws/coming-undone.git
   cd coming-undone
   python3.11 -m venv .venv
   .venv/bin/pip install 'torch==2.8.0' 'torchaudio==2.8.0'
   .venv/bin/pip install -e '.[web,drums,muscriptor]'
   ```
3. For MuScriptor, the best notes: accept its terms at
   https://huggingface.co/MuScriptor/muscriptor-medium, then `.venv/bin/hf auth login`.
   Without it, choose basic-pitch on the page.
4. Start it: `.venv/bin/stemscribe-web`. Leave it running.
5. Open the website and pick "This computer" under Runs. Chrome asks once whether the
   page may reach this computer; allow it.

It also runs on its own at http://127.0.0.1:8002, no website needed.

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
`prepare` decodes to wav before demucs sees anything, so transcoding to mp3 in
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
being able to answer "where did this come from?" months later (for material
that may be someone else's recording) is worth the two lines it costs. The same
diligence that keeps MuScriptor's CC-BY-NC weights out of the commercial path
applies to the audio going in.

### Web UI

```bash
stemscribe-web            # → http://127.0.0.1:8002
```

Drop a file, watch progress stream, audition stems, download MIDI. Tempo is
detected automatically and never asked about up front; if the grid looks wrong,
the alternates are one click away and re-stamp instantly.

It is **localhost-only by design**: no auth, runs jobs on your machine, reads
and writes your files. Don't expose it.

### Where a run happens (the Runs: switch)

The page has a **Runs:** switch with the engines it can use:

- **Online** (default): the Hugging Face Space in `space/` (ZeroGPU), called through
  Gradio's JavaScript client (vendored in `web/static/vendor/gradio-client/`). Up to
  30 s of audio per run; every visitor has their own daily GPU quota (2 minutes
  signed out, which is what calls from this page count as, 5 with a free Hugging Face
  account on the Space's own page). Only separation and transcription use the GPU,
  sized to the clip, so a 30 s run with muscriptor fits about once a day signed out
  (10 s: twice; basic-pitch: 3 times); the page shows the count for the chosen length
  (engines.js mirrors `space/split.py`'s budget), and running out offers the other
  engines. The Space id is `CONFIG.space` at the top of the page's script; `?space=`
  in the page's address overrides it (a local Gradio URL, for testing).
- **This computer**: this server, on port 8002. Shown only when it answers. Full
  quality, whole songs, nothing leaves the machine. It sends CORS headers for the
  public page's origin (`PUBLIC_ORIGINS` in `web/server.py`, more with
  `--allow-origin`) and for localhost. From a public https page, Chrome asks the
  visitor before reaching localhost, so the page only looks when the permission is
  already granted or the visitor clicks "look for it".
- **In your browser**: the whole run in the page, nothing uploaded. Shown as "later"
  until its models answer (a `models.json` at `CONFIG.browser.models`, by default
  `/browser-models/` on the page's own server). See the next section.

### In your browser

A lighter engine that runs entirely in the visitor's browser, desktop first:
htdemucs separation (ONNX), basic-pitch on vocals / bass / other (TF.js), ADT_STR
drums (ONNX, int8), and MuScriptor small as the opt-in pitched transcriber
(`muscriptor-small` in the Transcriber choice). WebGPU when the browser has it,
wasm otherwise. The same Run comes back as from the other engines (roll, parts,
stems, MIDI, manifest), made in the page as blob URLs.

What it leaves out, for now: 6 stems, the beat grid (bar lines, meter, bar one,
snap, tempo maps: tempo is one number from the drum hits, the MIDI's bars start at
0 s), re-stamping the tempo after the run, links (it takes files), and `manifest.json`
has no input hash. Cleanup, the silence trim and sections work as on the server
(the cleanup is a port of `cleanup.py`, checked against it by
`tests/test_browser_js.py`).

Sizes (first run; kept in Cache Storage after that):

| Part | Download |
|---|---|
| htdemucs (ONNX, from Hugging Face `timcsy/demucs-web-onnx`, pinned) | 180.5 MB |
| ADT_STR drums, int8 encoder + decoder | 72.9 MB |
| ONNX Runtime wasm (jsDelivr) + engine bundle + basic-pitch | 30.5 MB |
| **total** | **about 284 MB** |
| MuScriptor small, int8 (only when chosen) | +104.6 MB |

Times on an M-series Mac (10 cores), headless Chrome, a 20 s clip, models cached:
about 9 to 11 s on WebGPU (separation 6 s, drums 2.2 s, basic-pitch 2.1 s for three
parts), 22 s on wasm (8 threads). With MuScriptor small: about 18 s on WebGPU (it
takes about 5 s per pitched part). Peak Chrome memory 4 to 5 GB; scales roughly with
length, so it is for sections and short songs more than whole albums.

Checked end to end by `browser/verify/run.mjs` (the real page in headless Chrome)
against the server's own outputs for the same clips: drums onset F1 0.99 (house
excerpt, 95/97 hits) and 0.995 (/try/ clip, 100/99 hits) against PyTorch ADT_STR on
the server's stem, raw basic-pitch counts within a few notes of the server's
(50/21/173 vs 51/15/171 on house; the /try/ clip equal to the spike's 49/49/260),
and MuScriptor small's comping onset F1 0.72 against the server's MuScriptor medium
(native small scores 0.71). Tempo 150.00 on both clips (the server says 150 and 75,
the half-time reading of the same pulse).

Build and serve it:

```bash
npm --prefix browser install
npm --prefix browser run build           # -> web/static/vendor/browser-engine/ (committed)
python browser/models/export_adt.py --out browser/models-out
python browser/models/export_muscriptor.py --out browser/models-out/ms-small   # optional; gated, CC-BY-NC
node browser/models/finish.mjs browser/models-out    # int8 files + models.json
stemscribe-web --browser-models browser/models-out   # serves them at /browser-models/
node browser/verify/run.mjs clip.mp3 --port 8002     # the end-to-end check
```

How the models were made to run in a page (worked out in a separate feasibility
spike, not part of this repo):

- ADT_STR's attention is written out by hand for export (the fused PyTorch path does
  not export), and its decoder re-runs the whole token prefix each step: its causal
  mask is additive -1e4 and the model leans on what leaks through it, so a KV cache
  changes the output. With that the ONNX graphs reproduce `ADTModel.sample` token for
  token (`export_adt.py` checks it). Its mel front end and torchaudio's resampler are
  ported to JS exactly; the browser's own resampler flips cymbal classes.
- Weights ship as int8 (`browser/models/quantize.mjs`, per channel) and are turned
  back into float weights in the page before the session is made
  (`browser/src/onnxwire.js`): onnxruntime-web 1.30's WebGPU backend computes
  DequantizeLinear feeding MatMul wrong. Both work at the protobuf wire level on
  protobufjs 7.
- MuScriptor small: the log-mel runs in JS with the STFT window stored in the
  weights (it is not an exact Hann, and the quiet top bands depend on it), generation
  is a JS loop over a KV-cached step graph, and muscriptor's final note pass is
  ported. The beat_this grid it uses for its onset delay is not; the page passes its
  own constant beat grid.
- The page is cross-origin isolated (COOP `same-origin`, COEP `credentialless`,
  set by `web/server.py`) so the wasm fallback gets threads. A host that cannot set
  headers (GitHub Pages) still runs, single-threaded on wasm.

Models for the public page: the page looks for `/browser-models/models.json` on its
own server first, then at `CONFIG.publicModels`, the public Hugging Face repo
`swwallowws/coming-undone-browser-models`. That repo holds the ADT_STR int8 files
(CC BY-SA 4.0, credited in its model card, the int8 copy under the same licence) and
a `models.json` pointing at htdemucs on `timcsy/demucs-web-onnx`, pinned to a
revision. basic-pitch ships with the page. MuScriptor small stays out (gated,
CC-BY-NC): on the public page the choice shows switched off and points to Online and
This computer; served by `stemscribe-web --browser-models` it stays available. Until
the repo exists the public page shows the engine as "later".

The public page lives at https://swwallowws.github.io/coming-undone/, this repo's
GitHub Pages: CI stages it on every push to `main` and publishes it.
`scripts/deploy-web.sh --stage DIR` copies the studio page and the /try/ demo (the
frozen song's data, tracked in `try-dist/try/`, with the current try page code) into
DIR. Every path in the page is relative, so it runs under any subpath.
`node browser/verify/public_models.mjs --site DIR` runs the end-to-end check against a
staged copy.

```bash
.venv/bin/python scripts/stage_models.py          # -> models-dist/ (README.md tracked, .onnx not)
.venv/bin/python out/upload_models.py --dry-run   # the file list, no upload
node browser/verify/public_models.mjs --staged models-dist   # before the upload
node browser/verify/public_models.mjs                        # after it, from the public URLs
```

To pin the page to an upload, put its commit sha in place of `main` in
`CONFIG.publicModels`.

`space/README.md` has the Space's API, licences and set-up;
`scripts/stage_space.py` copies `space/` and the package into `space-dist/`, ready to
push to the Space.

### Outputs

| File | What |
|---|---|
| `song.mid` | multi-track MIDI: melody (from vocals), bass, comping, drums; real tempo map and bar lines |
| `song_instrumental.mp3` | stems-minus-vocals mix |
| `stems/{drums,bass,other,vocals}.wav` | the separated stems |
| `manifest.json` | input hash, params, backend, per-track note counts, quantization-error stats, the beat grid (tempo, bar lines, per-track latency and fit), drum backend, fallbacks, timings, warnings |

Track names (`melody`, `bass`, `comping`) are the contract downstream consumers
read: `melody` is always the vocal line. Renaming them breaks rearranged.

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
(`backends.ADT_STR_TO_GM`). Its hits sit on the drums stem's audio (+5 ms on
rearranged's donor song; on Elleri Ellerime about 15 ms ahead of MuScriptor's
pitched tracks). An earlier "41 ms early" figure here was wrong: it was a 195 ms gap
to MuScriptor's notes in an old build, folded into half a 16th (see Beat grid).

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

1. **Velocity floor**: drop notes below velocity 15 (transcription noise).
2. **De-overlap**: same-pitch overlapping notes: trim the earlier note's end to
   the next note's start.
3. **Duration cap**: notes longer than N beats (default 2) that have *other*
   notes re-onsetting during their sustain get trimmed to the next onset. Long
   notes over silence are left alone; they're probably real.
4. **Report**: before/after note counts and median duration, into the manifest.

Toggle with `cleanup=False` / `--no-cleanup`, or tune via `CleanupParams`
(`--max-duration-beats`, `--velocity-floor`, `--no-de-overlap`). The duration
cap is measured in beats against the detected tempo, so it means the same
musical thing at any BPM.

**How much does it actually do?** On `koprualti` via per-stem basic-pitch:
almost nothing: 1731 → 1727 notes, median duration unchanged. That is the
correct result, not a bug. The "wall of sustained notes" this pass was written
for comes from transcribing a *dense* source; per-stem transcription on
separated audio (which this pipeline mandates) mostly prevents it upstream.
Measured on that material: median 0.257s, polyphony 1.68, only 8 of 1731 notes
over the 1s cap, and zero notes under velocity 15.

So the pass earns its keep defensively (for muscriptor, for reverby material,
for backends that smear), and the defaults are deliberately *not* tuned harder
just to move the number. Trimming aggressively enough to drop the median would
delete real notes.

The smear is real, though: it's just on the other side of a threshold. Running
basic-pitch with `frame_threshold=0.05` on the same bass stem yields a median
note duration of **4.3 seconds** and zero silence: exactly the wall of sustained
overlapping notes the spec describes. Don't go there.

## Legato (the opposite problem)

basic-pitch ends a note when frame activation drops below `frame_threshold`
(default 0.3), which is well before the note stops sounding. The result is
output that's too *staccato*: measured on koprualti, bass came out **68%
silence**, melody 50%.

`legato=True` / `--legato` closes those gaps by extending each note to the next
onset, like Ableton's Span (legato) or its Legato command. Off by default: it
pushes a track toward "pad", the exact reading the rest of cleanup exists to
prevent.

**It only closes gaps under `legato_max_gap_beats` (default 0.25, a 16th).**
That guard is the whole design: an unconditional stretch swallows rests and
phrase endings, turning silence someone played on purpose into sustain nobody
did. A gap it won't close is information: it's telling you the note is *really*
that short, and the fix is upstream, not here.

Don't reach for `frame_threshold` to get legato. It changes which notes are
detected, not just their length: at 0.1 the bass stem explodes from 411 notes
to 10,594. Extending note ends is deterministic: it invents nothing and loses
nothing.

`silence_ratio_before/after` in the manifest is the number that tracks this.
Median duration can't see it. Shortening every note and spreading the same
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
octave error is instant and lossless: see the web UI's alternates, or
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
- **Meters other than 4/4 (`--meter`).** `--meter 3/4`, `6/8`, `12/8` or a grouped
  `9/8:2+2+2+3` (plain `9/8` means 2+2+2+3). The grid then counts the meter's
  denominator note, so in 6/8 the manifest's `bpm` counts eighths while the MIDI's
  tempo counts quarters as usual. In x/8 the tracked beat may be an eighth, a quarter
  or a dotted quarter; the grid tries all three and keeps the one the notes sit
  tightest on. Bar "one" also weighs the other group starts and loud onsets, and
  `--downbeat` runs from 1 to the numerator. 4/4 stays the default and is unchanged.
  A `--tempo` you give counts the meter's pulse and is kept: `--meter 9/8 --tempo
  52.8` grids a slow 9/8 whose eighth is 52.8 BPM, even when its drums play the half
  eighths (Harman Dalı's dum . tek-tek moved the grid to 105.6 and halved every bar).
- **A requested meter is never dropped silently.** When no grid fits, a meter other
  than 4/4 gets its own warning, naming it and why: too few notes, no constant tempo
  under `--tempo-mode constant` (try `map`), or `--no-grid`. In `auto` a song no
  constant tempo fits gets a tempo map in its meter instead.
- **No note moves for the grid.** The first bar is a pickup of its own tempo that
  ends exactly on the first real bar line; from there the tempo is the fitted one.
- **Snapping is opt-in (`--snap`).** It removes each track's latency (its signed
  offset from the grid) and then puts starts on 16ths, ends on 32nds and keeps one
  hit per drum per 16th. Off by default because snapping deletes real feel.
- **Latency past half a 16th.** Measured against the grid alone, an offset folds
  into half a 16th either way: a track 195 ms late at 114 BPM reads as +63 ms, and
  snapping then moves every hit a 16th off. So each track is also lined up with its
  own stem's onsets (cross-correlation within 300 ms), and the difference from the
  grid's source track picks the whole number of 16ths; the grid still gives the fine
  value. A track whose notes do not line up with its audio keeps the folded value. If
  the source track itself runs 100 ms or more off its audio, a warning says so.
  `stemscribe-grid` on a bare MIDI file has no audio and keeps the folded value.
- **The manifest's `grid`** records the tempo, first bar line, source track, the
  "one" confidence and any override, and per track its alignment, latency, median
  distance to the grid (`grid_fit_ms`), mean distance in 16ths (`offset_16th`: 0 is
  exact, 0.25 is random) and, when it lines up with its stem, its offset from that
  audio (`audio_lag_ms`).
- If no track sits on a steady grid, the MIDI keeps the detected tempo, nothing is
  snapped, and a warning says so. `--no-grid` skips the stage.

**Any MIDI, not only stemscribe's:** `stemscribe-grid in.mid -o out.mid [--snap]
[--downbeat N] [--tempo BPM] [--tempo-mode auto|constant|map]` fits and stamps a
grid on a file transcribed elsewhere and writes `out.grid.json` beside it. Without
`--tempo` it searches 60 to 200 BPM, since such files often carry a placeholder
tempo map.

### Tempo map (live bands that drift)

A live band speeds up and slows down, and one constant tempo cannot follow it: bar
lines slide off the music. On Đurđevdan (Bijelo Dugme) a 40 s excerpt sat 0.13 of a
16th off its best constant grid, and the whole song fitted no constant grid at all;
Harman Dalı (a live 9/8) moved between 103 and 108 BPM and fitted none either.

- **`--tempo-mode auto` (the default)** keeps the constant grid when it fits: when
  its source track sits within 0.08 of a 16th of it on average (steady songs here sat
  at 0.04 to 0.06). Then the output is byte for byte what it was before the map
  existed. Past 0.08 the map must also earn its place: fitted on every other note of
  its track, it has to sit at least 0.02 of a 16th closer to the notes in between than
  the constant grid does. A loose but steady band fails that test and keeps the
  constant grid (Đurđevdan's steady late excerpt: the map gained nothing there and put
  bass and comping further off); a drifting one passes (its early excerpt, 91 to 100
  BPM, gained 0.045). With no constant grid at all, it writes the map.
  `--tempo-mode constant` and `--tempo-mode map` force one or the other.
- **How the map is fitted.** The drums (or, with no drum track, the best-aligned
  track) go through a beat tracker held near the detected tempo, each hit weighted
  by its velocity, so the loud kick and snare pin the beat. The meter's pulse and bar
  "one" are found as for the constant grid, but counted in beats of the map, so every
  bar has the meter's number of pulses however the tempo moved. Fitting half the drum
  hits and measuring the other half, the map sat 0.17 of a 16th off against the
  constant grid's 0.18 on the whole of Đurđevdan, 0.16 against 0.22 on an early
  excerpt, and 0.11 against 0.20 on Harman Dalı at its slow eighth. The beats are not smoothed:
  smoothing them never helped the held-out drums and lost Harman Dalı's drift.
- **In the MIDI** the map is a tempo change on every beat after a pickup bar, so a
  DAW's bar lines follow the band. Notes never move. `--snap` and each track's latency
  work against the map, and so do `--downbeat` and the web UI's bar shift.
- **The manifest's `grid`** says `"tempo": "map"` and adds the beat times (`beats`),
  which of them is a bar line (`first`), the slowest and fastest bar (`bpm_range`),
  how far the constant grid sat (`constant_offset_16th`) and what the map gained on
  held-out notes (`map_gain_16th`). `bpm` is then the median tempo.

## Input conditioning

Before anything expensive touches the audio, `prepare` can decode to a standard
44.1k stereo wav, strip metadata and cover art, cut to a section, and trim
leading/trailing silence. All optional; all recorded in the manifest.

**The timing contract:** any offset introduced here is added back at the end, so
MIDI note times always refer to the *original* file you passed in. Trim 4.2s of
dead air and the first note still reports at 4.2s, because in your file, it is.
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
- **basic-pitch**: the Python API (`predict`), not the CLI: the CLI pays the
  import cost per call.
- **Summing stems clips.** drums+bass+other overshoots 0dBFS where the original
  mix was already near full scale (measured 1.19 peak on real material).
  `mixdown.py` ends the chain with `alimiter`; `limit=0` opts out.
- **yt-dlp's `js_runtimes` takes a dict, not a list.** The CLI's `--js-runtimes`
  accepts a list, but the Python API wants `{runtime: {config}}` and raises
  `ValueError` otherwise. Pass `{}` as the config, not `None`: validation
  accepts `None`, then something downstream calls `.get()` on it.
- **`alimiter=level` defaults to true**, which auto-levels the limited signal
  back up to 0dBFS and silently undoes `limit`. It must be `level=disabled` or
  the limiter does nothing you can measure. An mp3 of a dense limited mix still
  decodes ~1.1 peak regardless: lossy reconstruction overshoots its source, and
  that's normal; use `--instrumental-format wav` if you need provable headroom.
- **numpy leaks into the manifest.** `pretty_midi` returns `instrument.program`
  as `int64`, and `statistics.median` over note times returns `float64`; both
  are unserializable. `core._jsonable` cleans the manifest as a data structure,
  not just on write: `res.manifest` is a public surface that other code
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
| In your browser: htdemucs ONNX (timcsy/demucs-web-onnx) | MIT (demucs-web) | **research only** (no licence on the export) | **no** |
| In your browser: ADT_STR int8 | CC BY-SA 4.0 | CC BY-SA 4.0 (an adaptation: same licence, with credit) | yes, with credit |
| In your browser: MuScriptor small int8 | MIT | CC-BY-NC 4.0, gated | no |
| In your browser: onnxruntime-web, TF.js, basic-pitch, protobufjs (bundled) | MIT, Apache-2.0, Apache-2.0, BSD-3-Clause | basic-pitch Apache-2.0 | yes |

| /try/ MIDI player: spessasynth, the design system's shared build (`vendor/design/sound/spessasynth/`, synced with `design/sync.sh --sound`) | Apache-2.0 | none | yes |
| /try/ MIDI sounds: `gm.sf3`, the design system's shared General MIDI soundfont (`vendor/design/sound/`, synced with `design/sync.sh --sound`) | none | GeneralUser GS License v2.0, S. Christian Collins (trimmed copy; `NOTICE` beside it) | yes |

The studio page credits every model at its foot (the Models list), with these
licences. The /try/ page plays each part's MIDI on its General MIDI program from
that soundfont (drums on channel 10's Standard kit), with spessasynth scheduling
the notes on the page's audio clock; `browser/verify/try_sound.mjs` checks it by ear
in headless Chrome (every part sounds, bass below melody, a real kit, silence after
switching back to the audio) and saves each part's recording to listen to.

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

**The public page and its Space are free and non-commercial.** The Online engine
(`space/`) runs htdemucs (research-only weights) and MuScriptor (CC BY-NC 4.0, gated:
the Space needs an `HF_TOKEN` of an account that accepted its terms). Neither is
bundled: the Space downloads every model at start-up from its official source.
Charging for the Space, or for anything built on its output, needs a different
separator and basic-pitch first.

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
