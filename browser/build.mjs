// Build the browser engine into the studio's vendor folder:
//   npm --prefix browser install
//   npm --prefix browser run build
// Writes src/stemscribe/web/static/vendor/browser-engine/: engine.js (the bundle),
// basic-pitch/ (its model, Apache-2.0), VERSION and LICENSES.txt. The ONNX Runtime
// wasm files are not bundled: the page loads them from jsDelivr, pinned to the same
// onnxruntime-web version (engine config `ortWasm`). The big models are not here
// either: see browser/models/ and models.json.

import { build } from "esbuild";
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const out = here("../src/stemscribe/web/static/vendor/browser-engine/");
mkdirSync(`${out}basic-pitch`, { recursive: true });

await build({
  entryPoints: [here("src/engine.js")],
  bundle: true,
  format: "esm",
  minify: true,
  target: "es2022",
  outfile: `${out}engine.js`,
  legalComments: "eof",
  logLevel: "warning",
});

const bp = here("node_modules/@spotify/basic-pitch/model/");
for (const f of ["model.json", "group1-shard1of1.bin"]) copyFileSync(`${bp}${f}`, `${out}basic-pitch/${f}`);

const ver = (p) => JSON.parse(readFileSync(here(`node_modules/${p}/package.json`), "utf8")).version;
const pkg = JSON.parse(readFileSync(here("package.json"), "utf8"));
const { VERSION } = await import("./src/engine.js").catch(() => ({ VERSION: "?" }));
writeFileSync(`${out}VERSION`, [
  `coming-undone-browser-engine ${VERSION}`,
  `onnxruntime-web ${ver("onnxruntime-web")} (wasm from jsDelivr)`,
  `@tensorflow/tfjs ${ver("@tensorflow/tfjs")}`,
  `@spotify/basic-pitch ${ver("@spotify/basic-pitch")}`,
  `demucs-web ${ver("demucs-web")}`,
  `protobufjs ${ver("protobufjs")}`,
  `built from ${pkg.name}`,
  "",
].join("\n"));
writeFileSync(`${out}LICENSES.txt`, `The bundle in engine.js contains:
- onnxruntime-web: MIT, Microsoft
- @tensorflow/tfjs: Apache-2.0, Google
- @spotify/basic-pitch: Apache-2.0, Spotify (its model in basic-pitch/ too)
- demucs-web: MIT, timcsy
- protobufjs: BSD-3-Clause
- Coming Undone's own code: MIT (see the repository's LICENSE)

The models it downloads are not in this folder and keep their own licences:
- htdemucs (Defossez et al., Meta): research-only weights, ONNX export by timcsy
- ADT_STR drums (Melucci, Merialdo, Akama 2026): CC BY-SA 4.0; the int8 files are an
  adaptation and stay CC BY-SA 4.0
- MuScriptor small (Mirelo and Kyutai): CC-BY-NC 4.0, non-commercial use only
`);
console.log(`built ${out}engine.js`);
