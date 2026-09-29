// ONNX model rewriting at the protobuf wire level, on protobufjs 7's minimal
// Reader/Writer. Only the few fields that weight quantization touches are decoded
// (graph nodes and initializers); every other field is copied through byte for byte,
// so nothing the rewrite does not know about can get lost.
//
// Used at run time by expandDequant (int8 download -> float weights in the page) and
// at build time by browser/models/quantize.mjs (float weights -> int8).

import pb from "protobufjs/minimal.js";

const { Reader, Writer } = pb;
const dec = new TextDecoder();

export const DT = { FLOAT: 1, UINT8: 2, INT8: 3, FLOAT16: 10 };

// ModelProto.graph = 7; GraphProto.node = 1, initializer = 5
// NodeProto: input 1, output 2, name 3, op_type 4, attribute 5, doc_string 6, domain 7
// TensorProto: dims 1, data_type 2, name 8, raw_data 9
// AttributeProto: name 1, i 3, type 20 (INT = 2)

export function fields(buf) {
  const r = Reader.create(buf);
  const out = [];
  while (r.pos < r.len) {
    const start = r.pos;
    const tag = r.uint32();
    const field = tag >>> 3, wt = tag & 7;
    let dataStart = r.pos;
    if (wt === 2) {
      const len = r.uint32();
      dataStart = r.pos;
      r.skip(len);
    } else r.skipType(wt);
    out.push({ field, wt, start, end: r.pos, dataStart, dataEnd: r.pos });
  }
  return out;
}

const varint = (buf, f) => Reader.create(buf.subarray(f.dataStart, f.dataEnd)).int64();
const str = (buf, f) => dec.decode(buf.subarray(f.dataStart, f.dataEnd));

export function parseTensor(buf) {
  const t = { name: "", dims: [], dataType: 0, raw: null };
  for (const f of fields(buf)) {
    if (f.field === 8) t.name = str(buf, f);
    else if (f.field === 2) t.dataType = Number(varint(buf, f));
    else if (f.field === 9) t.raw = buf.subarray(f.dataStart, f.dataEnd);
    else if (f.field === 1) {
      if (f.wt === 0) t.dims.push(Number(varint(buf, f)));
      else {                                   // packed
        const r = Reader.create(buf.subarray(f.dataStart, f.dataEnd));
        while (r.pos < r.len) t.dims.push(Number(r.int64()));
      }
    } else if (f.field === 4 || f.field === 5 || f.field === 7) {
      throw new Error(`tensor ${t.name}: typed data fields are not supported, only raw_data`);
    }
  }
  return t;
}

export function parseNode(buf) {
  const n = { input: [], output: [], name: "", opType: "", domain: "", docString: "", attrs: {} };
  for (const f of fields(buf)) {
    if (f.field === 1) n.input.push(str(buf, f));
    else if (f.field === 2) n.output.push(str(buf, f));
    else if (f.field === 3) n.name = str(buf, f);
    else if (f.field === 4) n.opType = str(buf, f);
    else if (f.field === 6) n.docString = str(buf, f);
    else if (f.field === 7) n.domain = str(buf, f);
    else if (f.field === 5) {
      const ab = buf.subarray(f.dataStart, f.dataEnd);
      let name = "", i = null;
      for (const g of fields(ab)) {
        if (g.field === 1) name = str(ab, g);
        else if (g.field === 3) i = Number(varint(ab, g));
      }
      n.attrs[name] = i;
    }
  }
  return n;
}

export function encodeTensor({ name, dataType, dims, raw }) {
  const w = Writer.create();
  for (const d of dims) w.uint32((1 << 3) | 0).int64(d);
  w.uint32((2 << 3) | 0).int32(dataType);
  w.uint32((8 << 3) | 2).string(name);
  w.uint32((9 << 3) | 2).bytes(raw);
  return w.finish();
}

export function encodeNode({ input, output, name = "", opType, docString = "", intAttrs = {} }) {
  const w = Writer.create();
  for (const s of input) w.uint32((1 << 3) | 2).string(s);
  for (const s of output) w.uint32((2 << 3) | 2).string(s);
  if (name) w.uint32((3 << 3) | 2).string(name);
  w.uint32((4 << 3) | 2).string(opType);
  for (const [k, v] of Object.entries(intAttrs)) {
    const a = Writer.create();
    a.uint32((1 << 3) | 2).string(k);
    a.uint32((3 << 3) | 0).int64(v);
    a.uint32((20 << 3) | 0).int32(2);
    w.uint32((5 << 3) | 2).bytes(a.finish());
  }
  if (docString) w.uint32((6 << 3) | 2).string(docString);
  return w.finish();
}

const field = (num, bytes) => Writer.create().uint32((num << 3) | 2).bytes(bytes).finish();

function concat(parts) {
  const n = parts.reduce((a, p) => a + p.length, 0);
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

// Rewrite a model's graph. edit(graph) gets { nodes, inits } (parsed, each with its
// raw field bytes) and returns { dropNodes: Set<index>, dropInits: Set<name>,
// prependNodes: [bytes], appendInits: [bytes] }.
export function rewriteGraph(modelBytes, edit) {
  const buf = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
  const mf = fields(buf);
  const gf = mf.find((f) => f.field === 7);
  if (!gf) throw new Error("no graph in the model");
  const g = buf.subarray(gf.dataStart, gf.dataEnd);
  const gfs = fields(g);
  const nodes = [], inits = [];
  gfs.forEach((f) => {
    if (f.field === 1) nodes.push({ f, ...parseNode(g.subarray(f.dataStart, f.dataEnd)) });
    else if (f.field === 5) inits.push({ f, ...parseTensor(g.subarray(f.dataStart, f.dataEnd)) });
  });
  const e = edit({ nodes, inits });
  const dropNodes = e.dropNodes || new Set(), dropInits = e.dropInits || new Set();
  const dropStarts = new Set([
    ...nodes.filter((n, i) => dropNodes.has(i)).map((n) => n.f.start),
    ...inits.filter((t) => dropInits.has(t.name)).map((t) => t.f.start),
  ]);
  const parts = (e.prependNodes || []).map((b) => field(1, b));
  for (const f of gfs) if (!dropStarts.has(f.start)) parts.push(g.subarray(f.start, f.end));
  for (const b of e.appendInits || []) parts.push(field(5, b));
  const graph = concat(parts);
  const out = [];
  for (const f of mf) out.push(f.field === 7 ? field(7, graph) : buf.subarray(f.start, f.end));
  return concat(out);
}

// ---- float16 ------------------------------------------------------------------
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
export function toHalf(v) {
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const e = ((x >>> 23) & 0xff) - 127 + 15;
  let m = x & 0x7fffff;
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    m = (m | 0x800000) >> (1 - e);
    return sign | ((m + 0x1000) >> 13);
  }
  return sign | (e << 10) | ((m + 0x1000) >> 13);
}
export function fromHalf(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

// a copy in its own buffer, so typed views line up (node's Buffer.slice does not copy)
const aligned = (raw) => new Uint8Array(raw).buffer;

export function tensorFloats(t) {
  if (t.dataType === DT.FLOAT) return new Float32Array(aligned(t.raw));
  if (t.dataType === DT.FLOAT16) return Float32Array.from(new Uint16Array(aligned(t.raw)), fromHalf);
  throw new Error(`tensor ${t.name}: not a float tensor (${t.dataType})`);
}

// ---- run time: int8 weights -> float weights ------------------------------------
// Every DequantizeLinear whose inputs are all initializers becomes a plain float
// initializer again: float16 when the quantizer marked the source as float16
// (docString "float16"), else float32. onnxruntime-web 1.30's WebGPU backend gives
// wrong results for DequantizeLinear feeding MatMul, so the page never runs one.
export function expandDequant(modelBytes) {
  return rewriteGraph(modelBytes, ({ nodes, inits }) => {
    const byName = new Map(inits.map((t) => [t.name, t]));
    const dropNodes = new Set(), dropInits = new Set(), appendInits = [];
    nodes.forEach((n, i) => {
      if (n.opType !== "DequantizeLinear" || !n.input.every((x) => byName.has(x))) return;
      const [q, s, z] = n.input.map((x) => byName.get(x));
      const scale = new Float32Array(aligned(s.raw));
      const zp = z.raw;
      const qb = q.raw;
      const perTensor = scale.length === 1;
      const axis = perTensor ? 0 : (n.attrs.axis ?? 1);
      const inner = perTensor ? qb.length : q.dims.slice(axis + 1).reduce((a, b) => a * b, 1);
      const nAxis = perTensor ? 1 : q.dims[axis];
      const signed = q.dataType === DT.INT8;
      const half = n.docString === "float16";
      const out = half ? new Uint16Array(qb.length) : new Float32Array(qb.length);
      for (let k = 0; k < qb.length; k++) {
        const c = perTensor ? 0 : Math.floor(k / inner) % nAxis;
        const qv = signed ? (qb[k] << 24) >> 24 : qb[k];
        const zv = signed ? (zp[c] << 24) >> 24 : zp[c];
        const v = (qv - zv) * scale[c];
        out[k] = half ? toHalf(v) : v;
      }
      dropNodes.add(i);
      for (const t of [q, s, z]) dropInits.add(t.name);
      appendInits.push(encodeTensor({ name: n.output[0], dataType: half ? DT.FLOAT16 : DT.FLOAT,
                                      dims: q.dims, raw: new Uint8Array(out.buffer) }));
    });
    return { dropNodes, dropInits, appendInits };
  });
}
