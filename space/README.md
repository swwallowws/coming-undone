---
title: Coming Undone
short_description: simply split. Song in, stems and labeled MIDI out.
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.28.0
python_version: "3.10"
app_file: app.py
suggested_hardware: zero-a10g
license: mit
pinned: false
---

# Coming Undone

simply split. Song in, stems and labeled multi-track MIDI out.

This Space is the "Online" engine behind Coming Undone's page. The page calls its one
API endpoint, `split`, through Gradio's JavaScript client; the Gradio UI here is only
for trying it directly and for debugging.

**Free and non-commercial.** It runs models whose weights are licensed for research or
non-commercial use only (see Licences). Nothing here may be used commercially.

## Hardware

ZeroGPU (Settings, Hardware: "ZeroGPU"). Only the GPU work runs inside the one
`@spaces.GPU` function (`app._gpu_stage`, which runs `stemscribe.core.gpu_stage`):
demucs separation, then MuScriptor on each pitched stem and ADT_STR on the drums.
Decoding, the section cut, tempo, the instrumental mixdown, the basic-pitch fallback,
cleanup, merge, the grid and the mp3/MIDI files run outside it on the CPU and cost the
visitor no GPU time.

The request is sized to the prepared audio, measured before asking
(`split.gpu_seconds`): 8 s, plus 0.2 s per audio second for demucs (0.35 for 6 stems),
plus 6 s + 0.3 s per audio second for each MuScriptor stem, plus 4 s + 0.15 s per audio
second for the drums. The first real run (20 s, 4 stems) took 1.6 s to separate and
28.2 s to transcribe; the terms are that with about half again as margin, since a
request that runs out stops the run. basic-pitch runs on the CPU, so it only asks for
demucs and the drums.

| run (4 stems)       | request | runs a day, signed out (120 s) | free account (300 s) |
| ------------------- | ------- | ------------------------------ | -------------------- |
| muscriptor, 30 s    | 68 s    | 1                              | 3                    |
| muscriptor, 20 s    | 55 s    | 1                              | 3                    |
| muscriptor, 10 s    | 43 s    | 2                              | 4                    |
| basic-pitch, 30 s   | 23 s    | 3                              | 9                    |

Runs a day count what ZeroGPU charged the first run (1.5 times the request: a 93 s
request used up a signed-out 120 s) and that a run starts only while the quota left
covers its request (`split.runs_per_day`). The result JSON's `gpu` field has the
request and these counts; its `timings.gpu_call` is the whole GPU call, for tuning.

Every visitor has their own daily ZeroGPU quota: 2 minutes signed out, 5 minutes with a
free Hugging Face account. Calls from Coming Undone's own page (Gradio's JavaScript
client, from another site) carry no Hugging Face account, so they count as signed out.
A request that would not fit the visitor's remaining quota fails with ZeroGPU's quota
message, which the page turns into a friendly note offering the other engines.

One run takes at most 30 s of audio (`split.MAX_SECONDS`), from the section start the
page sends.

## API

`split(audio: file, options: str) -> (result: str, midi: file, parts: file[], files: file[])`

- `audio`: the audio file (mp3, wav, m4a, flac).
- `options`: a JSON object; every key is optional, unknown keys are refused. See
  `split.DEFAULTS`: `backend` (`muscriptor` or `basic-pitch`), `demucs_model`
  (`htdemucs`, `htdemucs_6s`), `meter` (`4/4`, `3/4`, `6/8`, `9/8:2+2+2+3`, `12/8`),
  `downbeat` (1 to the meter's pulses), `snap`, `tempo`, `start`, `duration`,
  `include_vocals_melody`, `mono_stems`, the cleanup knobs, and `stems_audio` (send
  the stems back as mp3).
- `result`: JSON text. Tempo, grid, tracks, warnings, and `parts` (every track with its
  notes, the page's by-part view). File references are base names that match the
  returned files' names.
- `midi`: the combined MIDI. `parts`: one MIDI per track on the same tempo map.
  `files`: the stems as mp3, the instrumental mix, and `manifest.json`.

## Secrets

- `HF_TOKEN` (required for MuScriptor): a read token of an account that accepted the
  terms at https://huggingface.co/MuScriptor/muscriptor-medium. Without it the
  `muscriptor` backend fails and `basic-pitch` still works.

## Licences

The code in this Space is Coming Undone's (formerly stemscribe), MIT. No model weights
are included: each model downloads at start-up from its official source and keeps its
own licence.

| Component | Code | Weights | Here |
|---|---|---|---|
| demucs (htdemucs, htdemucs_6s) | MIT | research only | separation; the reason this Space is non-commercial |
| MuScriptor | MIT | CC BY-NC 4.0 (gated) | default transcriber; non-commercial use only |
| basic-pitch | Apache-2.0 | Apache-2.0 | the fast transcriber |
| ADT_STR drums | CC BY-SA 4.0 | CC BY-SA 4.0 | drums; credit: Melucci, Merialdo, Akama 2026, https://github.com/pier-maker92/ADT_STR |
| Gradio, spaces | Apache-2.0 | none | the Space itself |

What visitors upload is their responsibility: only send audio you have the right to
process. Uploads and results are deleted from the Space within an hour.
