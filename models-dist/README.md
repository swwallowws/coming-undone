---
license: cc-by-sa-4.0
library_name: onnx
tags:
  - onnx
  - onnxruntime-web
  - audio
  - automatic-drum-transcription
  - music-transcription
base_model: Pierfrancesco/adt-str
---

# Coming Undone: browser models

Model files for the "In your browser" engine of
Coming Undone, which separates a song into stems and
transcribes them to MIDI entirely in the visitor's browser (onnxruntime-web, WebGPU
with a wasm fallback). The page reads `models.json` from this repo and downloads the
files it lists once, into the browser's Cache Storage. Nothing is uploaded: the audio
never leaves the machine.

## Files

| File | Size | What |
|---|---|---|
| `adt_encoder.int8.onnx` | 35.7 MB | ADT_STR drum transcription, encoder: log-mel (B, 246, 128) in, each decoder layer's cross-attention K/V out |
| `adt_decoder.int8.onnx` | 37.2 MB | ADT_STR drum transcription, decoder: token prefix + cross K/V in, next-token logits out (greedy decode runs in JS) |
| `models.json` | 1 KB | the manifest the page reads: file names, sizes, sha256, licences, and the pinned htdemucs URL |

Both ADT_STR graphs store their weights as int8 (per channel, weights only). The page
turns them back into float weights before it creates the session, because
onnxruntime-web 1.30's WebGPU backend computes DequantizeLinear feeding MatMul wrong.
Against the original PyTorch model on the same drum stem the int8 graphs give the same
hits (onset F1 1.000 on a 20 s test clip, 97/97 hits).

## Where they come from

- **ADT_STR** (Melucci, Merialdo, Akama 2026), variant `setting-tau-0.8` from
  [Pierfrancesco/adt-str](https://huggingface.co/Pierfrancesco/adt-str) at revision
  `a33c5c6b191a4ca1e0f6dc22140947485eb36ce8`. Code:
  [github.com/pier-maker92/ADT_STR](https://github.com/pier-maker92/ADT_STR).
  Exported to ONNX (attention written out by hand over the same weights) and quantized
  to int8 by Coming Undone's `browser/models/export_adt.py` and
  `browser/models/quantize.mjs`.

Not in this repo:

- **htdemucs** separation (Défossez et al., Meta,
  [facebookresearch/demucs](https://github.com/facebookresearch/demucs), MIT): the page
  loads the ONNX export by timcsy straight from
  [timcsy/demucs-web-onnx](https://huggingface.co/timcsy/demucs-web-onnx), pinned to
  revision `92e33df61cfc9eb820272aaa62d2ef6dcf4d950d` (180.5 MB). That repo states no
  licence of its own.
- **basic-pitch** (Spotify, Apache-2.0): its 0.9 MB TF.js model ships with the page.
- **MuScriptor small** (Mirelo and Kyutai): its weights are gated and CC BY-NC 4.0,
  so they are not shared here. MuScriptor runs in Coming Undone's Online and This
  computer engines instead.

## Licence

The ADT_STR weights are licensed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
by their authors: Melucci, Merialdo, Akama, "ADT_STR" (2026),
[github.com/pier-maker92/ADT_STR](https://github.com/pier-maker92/ADT_STR).

The files here are an adaptation of those weights (exported to ONNX and quantized to
int8). They are shared under the same licence, CC BY-SA 4.0: you may use them,
including commercially, as long as you credit the original authors, say that the
weights were changed, and share your own adaptations under CC BY-SA 4.0 too. The
quantized weights stay CC BY-SA 4.0 whatever format they are converted to.
