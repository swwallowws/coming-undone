// Weight-only int8 quantization of an exported ONNX graph, in node (no Python onnx).
//   node browser/models/quantize.mjs <in.onnx> <out.onnx>
//
// Every float32 or float16 initializer of at least 4096 elements becomes uint8 codes
// (zero point 128) with a per-channel float scale, behind a DequantizeLinear node:
// per output column for MatMul weights, per row for anything else (embedding and
// positional tables). A float16 source is marked on its node (docString "float16") so
// the page restores float16. The page always expands these back to float weights
// before making a session (src/onnxwire.js expandDequant); the file is only the
// download format.

import { readFileSync, writeFileSync } from "node:fs";
import { DT, encodeNode, encodeTensor, rewriteGraph, tensorFloats } from "../src/onnxwire.js";

const [inPath, outPath] = process.argv.slice(2);
if (!inPath || !outPath) {
  console.error("usage: node browser/models/quantize.mjs <in.onnx> <out.onnx>");
  process.exit(2);
}

let before = 0, after = 0, count = 0;
const out = rewriteGraph(readFileSync(inPath), ({ nodes, inits }) => {
  const uses = new Map();
  for (const n of nodes) n.input.forEach((name, idx) => {
    if (!uses.has(name)) uses.set(name, []);
    uses.get(name).push({ n, idx });
  });
  const dropInits = new Set(), appendInits = [], prependNodes = [];
  for (const t of inits) {
    const size = t.dims.reduce((a, b) => a * b, 1);
    before += t.raw ? t.raw.length : 0;
    const float = t.dataType === DT.FLOAT || t.dataType === DT.FLOAT16;
    if (!float || size < 4096 || !t.raw) { after += t.raw ? t.raw.length : 0; continue; }
    const u = uses.get(t.name) || [];
    const matmulB = u.length > 0 && u.every(({ n, idx }) => n.opType === "MatMul" && idx === 1);
    const axis = matmulB ? t.dims.length - 1 : 0;
    const w = tensorFloats(t);
    const nAxis = t.dims[axis], inner = t.dims.slice(axis + 1).reduce((a, b) => a * b, 1);
    const scale = new Float32Array(nAxis);
    for (let k = 0; k < w.length; k++) {
      const c = Math.floor(k / inner) % nAxis;
      scale[c] = Math.max(scale[c], Math.abs(w[k]));
    }
    for (let c = 0; c < nAxis; c++) scale[c] = scale[c] / 127 || 1;
    const q = new Uint8Array(w.length);
    for (let k = 0; k < w.length; k++) q[k] = 128 + Math.round(w[k] / scale[Math.floor(k / inner) % nAxis]);
    const qn = `${t.name}_q8`, sn = `${t.name}_s`, zn = `${t.name}_z`;
    dropInits.add(t.name);
    appendInits.push(
      encodeTensor({ name: qn, dataType: DT.UINT8, dims: t.dims, raw: q }),
      encodeTensor({ name: sn, dataType: DT.FLOAT, dims: [nAxis], raw: new Uint8Array(scale.buffer) }),
      encodeTensor({ name: zn, dataType: DT.UINT8, dims: [nAxis], raw: new Uint8Array(nAxis).fill(128) }),
    );
    prependNodes.push(encodeNode({
      input: [qn, sn, zn], output: [t.name], name: `${t.name}_dq`, opType: "DequantizeLinear",
      intAttrs: { axis }, docString: t.dataType === DT.FLOAT16 ? "float16" : "",
    }));
    after += q.length + nAxis * 5;
    count++;
  }
  return { dropInits, appendInits, prependNodes };
});
writeFileSync(outPath, out);
console.log(`${count} weights to int8: ${(before / 1e6).toFixed(1)} MB -> ${(after / 1e6).toFixed(1)} MB, file ${(out.length / 1e6).toFixed(1)} MB`);
