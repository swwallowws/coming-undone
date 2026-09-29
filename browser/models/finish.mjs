// Turn exported models into the browser engine's model folder: int8 downloads plus
// models.json, the manifest the page reads (file names, sizes, licences).
//   node browser/models/finish.mjs browser/models-out
//
// Expects export_adt.py's adt_encoder.onnx / adt_decoder.onnx in the folder and,
// optionally, export_muscriptor.py's ms-small/. htdemucs is taken from Hugging Face
// (timcsy/demucs-web-onnx) unless htdemucs_embedded.onnx sits in the folder too.
// Serve the folder with `stemscribe-web --browser-models <dir>` (it appears at
// /browser-models/), or host it anywhere with CORS and point the page at it.

import { execFileSync } from "node:child_process";
import { existsSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const dir = process.argv[2];
if (!dir) { console.error("usage: node browser/models/finish.mjs <models dir>"); process.exit(2); }
const quantize = fileURLToPath(new URL("./quantize.mjs", import.meta.url));
const q = (a, b) => {
  process.stdout.write(`${b}: `);
  process.stdout.write(execFileSync(process.execPath, [quantize, join(dir, a), join(dir, b)], { encoding: "utf8" }));
};
const size = (f) => statSync(join(dir, f)).size;

// pinned; the same revision as browser/src/models.js and scripts/stage_models.py
const HTDEMUCS_URL = "https://huggingface.co/timcsy/demucs-web-onnx/resolve/92e33df61cfc9eb820272aaa62d2ef6dcf4d950d/htdemucs_embedded.onnx";
const HTDEMUCS_BYTES = 180534758;

q("adt_encoder.onnx", "adt_encoder.int8.onnx");
q("adt_decoder.onnx", "adt_decoder.int8.onnx");
const manifest = {
  version: 1,
  htdemucs: existsSync(join(dir, "htdemucs_embedded.onnx"))
    ? { file: "htdemucs_embedded.onnx", bytes: size("htdemucs_embedded.onnx") }
    : { url: HTDEMUCS_URL, bytes: HTDEMUCS_BYTES },
  adt: {
    encoder: { file: "adt_encoder.int8.onnx", bytes: size("adt_encoder.int8.onnx") },
    decoder: { file: "adt_decoder.int8.onnx", bytes: size("adt_decoder.int8.onnx") },
    licence: "CC-BY-SA-4.0 (ADT_STR, Melucci, Merialdo, Akama 2026; int8 adaptation)",
  },
};
if (existsSync(join(dir, "ms-small", "lm.onnx"))) {
  q("ms-small/lm.onnx", "ms-small/lm.int8.onnx");
  manifest.muscriptorSmall = {
    dir: "ms-small/",
    files: { cond: "cond.onnx", lm: "lm.int8.onnx", emb: "emb.bin", meta: "meta.json" },
    bytes: ["cond.onnx", "lm.int8.onnx", "emb.bin", "meta.json"].reduce((a, f) => a + size(join("ms-small", f)), 0),
    licence: "CC-BY-NC-4.0 (MuScriptor weights, gated; non-commercial use only)",
  };
}
writeFileSync(join(dir, "models.json"), JSON.stringify(manifest, null, 1));
const mb = (b) => (b / 1e6).toFixed(1);
console.log(`models.json: htdemucs ${mb(manifest.htdemucs.bytes)} MB, drums ${mb(manifest.adt.encoder.bytes + manifest.adt.decoder.bytes)} MB` +
  (manifest.muscriptorSmall ? `, MuScriptor small ${mb(manifest.muscriptorSmall.bytes)} MB` : ""));
