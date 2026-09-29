# stemscribe: song in, stems + multi-track MIDI out

A small Python library + CLI that takes an audio file (with or without vocals),
separates it into stems, and transcribes it into a labeled multi-track MIDI.
Extracted from the genre-bending project (renamed rearranged on 2026-09-25), which will import it as its
transcription front-end; useful standalone for any producer workflow.

## Why this exists (context)

genre-bending needs: song → instrumental MIDI (content/style inputs for an
arrangement engine) + the vocal melody as its own MIDI track (for melody
overlay). That logic currently lives as scattered scripts and one-off shell
commands; this project makes it one clean, tested, reusable module.

## Deliverable shape

- Python package `stemscribe` with a clean API
- Thin CLI: `stemscribe INPUT.mp3 -o OUTDIR [options]`
- Target: Python 3.11, macOS (Apple Silicon) first; keep Linux working

## Pipeline

```
audio in (mp3/wav/m4a)
  └─ 1. separate      demucs htdemucs → drums / bass / other / vocals (wav)
  └─ 2. instrumental  mix drums+bass+other → instrumental.(wav|mp3)
  └─ 3. transcribe    per-stem audio→MIDI via pluggable backend
  │                     bass  → bass track
  │                     other → harmony/comping track(s)
  │                     vocals → melody track (labeled "melody")
  │                     drums → drum backend (ADT_STR), or skipped without the extra
  └─ 4. cleanup       de-overlap + duration-trim pass (see Cleanup)
  └─ 5. merge         one multi-track MIDI, GM programs + named tracks
  └─ 6. realign       back onto the original file's timeline
  └─ 7. grid          tempo + bar lines fitted from the tightest track; optional snap
```

> **Amended 2026-09-25 (scope change, approved):** stemscribe now owns the beat
> grid and transcribes drums. Reason: rearranged is MIDI-only and copies a
> donor's timing, so a donor without a real grid had to be repaired by hand. See
> the README's "Beat grid" and "Drums" sections; the bake-off behind the design is
> in rearranged/research/bakeoff/.

## API sketch

```python
from stemscribe import process, Result

res: Result = process(
    "song.mp3",
    out_dir="out/",
    backend="basic-pitch",       # or "muscriptor"
    include_vocals_melody=True,  # transcribe vocal stem as melody track
    cleanup=True,                # de-overlap / trim pass
    keep_stems=True,             # write stem wavs to out_dir
    instrumental=True,           # write instrumental mix
)
res.midi_path          # out/song.mid  (multi-track)
res.stem_paths         # {"drums": ..., "bass": ..., "other": ..., "vocals": ...}
res.instrumental_path  # out/song_instrumental.mp3
res.track_map          # {"melody": 0, "bass": 1, "comping": 2, ...}
res.manifest_path      # out/manifest.json (all params + timings + qc stats)
```

CLI mirrors the API:
```
stemscribe song.mp3 -o out/ --backend basic-pitch --no-vocals-melody --midi-only
```

## Outputs (all under one out_dir)

| File | What |
|---|---|
| `song.mid` | multi-track MIDI: melody (from vocals), bass, comping/other; GM programs, named tracks |
| `song_instrumental.mp3` | stems-minus-vocals mix |
| `stems/{drums,bass,other,vocals}.wav` | the separated stems |
| `manifest.json` | input hash, params, backend, per-track note counts, quantization-error stats, timings |

## Backends (pluggable registry)

| Backend | License | Notes |
|---|---|---|
| `muscriptor` (default) | code MIT, weights CC-BY-NC (gated HF) | quality ceiling; NON-COMMERCIAL ONLY; needs `hf auth login` + accepted terms at huggingface.co/MuScriptor/muscriptor-medium; CLI: `muscriptor transcribe in.wav -o out.mid` |
| `basic-pitch` | Apache-2.0 | commercial-safe fallback; pitched stems only; run PER STEM, never on the full mix |

> **Default changed from the original spec** (was `basic-pitch`). muscriptor is
> the default because it transcribes better and is free for personal use. Since
> the default is now non-commercial, the commercial path is protected by the
> `STEMSCRIBE_COMMERCIAL=1` env var, which hard-fails any CC-BY-NC backend.
> rearranged's shipping build MUST set it (or pass `backend="basic-pitch"`).

Registry design: `BACKENDS = {"basic-pitch": fn, "muscriptor": fn}` where each
fn takes `(stem_wav_path, out_mid_path)`: copy the pattern from
`~/Playground/rearranged/test-harness/transcribe.py` (4-backend scaffold;
Klangio/YourMT3+ can slot in later, do NOT build them now).

Drums: pitched backends cannot transcribe drums. (Amended 2026-09-25: the drums
stem goes to a separate drum backend registry, `DRUM_BACKENDS`, with ADT_STR;
without the `[drums]` extra it is skipped with a warning, as before.)

## Cleanup pass (important, learned the hard way)

Raw transcriptions (esp. of reverby comping) come out as WALLS OF SUSTAINED
OVERLAPPING NOTES: downstream style engines then read the texture as "pad"
instead of "stabs". The cleanup pass, per track:

1. De-overlap: same-pitch overlapping notes → trim previous note's end to the
   next note's start.
2. Duration cap: notes longer than N beats (default 2) with re-onsets of other
   notes during their sustain get trimmed to the next onset (kill smear).
3. Velocity floor: drop notes below velocity ~15 (transcription noise).
4. Report before/after note counts + median duration in the manifest.

Make cleanup togglable and parameterized; defaults tuned on real material.

## Known gotchas (all hit in genre-bending, do not rediscover)

- **demucs save crashes** on torchaudio ≥2.9: `ModuleNotFoundError: torchcodec`.
  Do NOT use `demucs.separate` CLI's save path. Use the API:
  `demucs.pretrained.get_model('htdemucs')` + `demucs.apply.apply_model`,
  then save with `soundfile.write`. Working reference:
  the `sep.py` pattern: model.cpu(), AudioFile(...).read(streams=0,
  samplerate=model.samplerate, channels=model.audio_channels), normalize by
  ref mean/std, `apply_model(..., split=True, overlap=0.25)`, un-normalize,
  `sf.write(path, src.cpu().numpy().T, model.samplerate)`.
- **ffmpeg amix**: the installed ffmpeg doesn't know `normalize=0`; use
  `amix=inputs=3,volume=3` instead.
- **basic-pitch CLI import cost**: prefer the Python API
  (`basic_pitch.inference.predict_and_save` or `predict`) over shelling out.
- MuScriptor weights are ~2GB, downloaded on first run; slow. Cache is HF's.
- mp3 in via ffmpeg/audioread is fine; don't require wav input.

## Licenses (matters: rearranged ships commercially later)

demucs code MIT, but **its weights are research-only** (corrected 2026-09-25, see
the README's Licenses), basic-pitch Apache ✅, soundfile/pretty_midi ✅, ADT_STR CC BY-SA 4.0.
MuScriptor = NC: keep it an optional extra (`pip install stemscribe[muscriptor]`
or just a documented optional dep), clearly marked non-commercial.

## Acceptance test

Run on `~/Playground/rearranged/inputs/koprualti.mp3`:
- 4 stems written, instrumental mix listenable, vocals absent from it
- `song.mid` has ≥3 named tracks incl. "melody" (from vocals)
- cleanup reduces median note duration on the "other" track vs raw
- manifest sane (counts, params, timings)
Compare (informally) with `~/Playground/rearranged/phase0/koprualti_style.mid`:
the new output should be at least as clean.

## Non-goals for v1 (extension points only)

- Chord labels (BTC/BACHI), section detection: leave hooks, don't build. (The
  beat/downbeat grid and drum transcription moved in scope on 2026-09-25.)
- Melody identification inside instrumental tracks (skyline etc.): rearranged
  keeps that logic; here melody comes only from the vocal stem.
- Tempo changes within a song (the grid is one constant tempo).
- GPU/ROCm tuning.

## Integration note (for rearranged, later)

rearranged will replace: its scratchpad `sep.py`, the ffmpeg clip-mixing
commands, and its per-stem basic-pitch calls with `stemscribe.process()`.
Section cutting (by seconds) stays in rearranged; stemscribe processes
whole songs.
