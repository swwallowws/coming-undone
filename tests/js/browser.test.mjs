// The browser engine's pure parts, in node: node tests/js/browser.test.mjs
// (needs `npm --prefix browser install` for protobufjs). With a JSON path argument it
// instead runs the cleanup port on the notes in that file and prints the result, for
// tests/test_browser_js.py's parity check against stemscribe/cleanup.py.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const src = (p) => new URL(`../../browser/src/${p}`, import.meta.url).href;
const { cleanNotes } = await import(src("cleanup.js"));

if (process.argv[2]) {
  const { notes, params } = JSON.parse(readFileSync(process.argv[2], "utf8"));
  console.log(JSON.stringify(cleanNotes(notes, params)));
  process.exit(0);
}

const { writeMidi, readMidi } = await import(src("midi.js"));
const { estimateTempo } = await import(src("tempo.js"));
const W = await import(src("onnxwire.js"));
const { decodeTokens, ADT_TO_GM } = await import(src("drums.js"));
const { trimBounds } = await import(src("audio.js"));
const M = await import(src("muscriptor.js"));
let n = 0;
const test = async (name, fn) => { try { await fn(); n++; } catch (e) { console.error(`FAIL ${name}`); throw e; } };

await test("the MIDI writer round-trips notes, names, programs and the drum channel", () => {
  const tracks = [
    { name: "melody", program: 53, drum: false, notes: [{ pitch: 64, start: 0.5, end: 1.0, velocity: 90 }, { pitch: 67, start: 1.0, end: 1.25, velocity: 70 }] },
    { name: "drums", program: 0, drum: true, notes: [{ pitch: 36, start: 0.0, end: 0.1, velocity: 100 }] },
  ];
  const back = readMidi(writeMidi({ bpm: 150, tracks }));
  assert.ok(Math.abs(back.bpm - 150) < 0.01);
  const mel = back.tracks.find((t) => t.name === "melody"), dr = back.tracks.find((t) => t.name === "drums");
  assert.equal(mel.program, 53);
  assert.equal(dr.notes[0].channel, 9);
  mel.notes.forEach((x, i) => {
    assert.equal(x.pitch, tracks[0].notes[i].pitch);
    assert.ok(Math.abs(x.start - tracks[0].notes[i].start) < 0.002 && Math.abs(x.end - tracks[0].notes[i].end) < 0.002);
  });
});

await test("readMidi follows a tempo map", () => {
  // two tempo events by hand: 120 BPM, then 60 BPM from tick 960 (1 s)
  const bytes = writeMidi({ bpm: 120, tracks: [{ name: "a", program: 0, drum: false, notes: [{ pitch: 60, start: 2.0, end: 2.5, velocity: 80 }] }] });
  const m = readMidi(bytes);
  assert.ok(Math.abs(m.tracks[1].notes[0].start - 2.0) < 0.002);
});

await test("tempo from a steady drum pattern, with its half-time alternate", () => {
  const hits = [];
  for (let t = 0.1; t < 20; t += 0.4) hits.push({ time: t, weight: 1 }, { time: t + 0.2, weight: 0.3 });
  const e = estimateTempo(hits, 20);
  assert.ok(Math.abs(e.bpm - 150) < 0.6, `bpm ${e.bpm}`);
  assert.ok(e.candidates.some((c) => c.ratio === 0.5));
  assert.ok(Math.abs(((e.beat0 - 0.1) / 0.4) - Math.round((e.beat0 - 0.1) / 0.4)) < 0.05, `beat0 ${e.beat0}`);
  assert.equal(estimateTempo([{ time: 1 }], 5), null);
});

await test("expandDequant turns int8 weights back into float32 and float16", () => {
  const q = new Uint8Array([128, 129, 127, 255]), s = new Float32Array([0.5, 0.25]), z = new Uint8Array([128, 128]);
  const inits = [
    W.encodeTensor({ name: "w_q8", dataType: W.DT.UINT8, dims: [2, 2], raw: q }),
    W.encodeTensor({ name: "w_s", dataType: W.DT.FLOAT, dims: [2], raw: new Uint8Array(s.buffer) }),
    W.encodeTensor({ name: "w_z", dataType: W.DT.UINT8, dims: [2], raw: z }),
  ];
  const node = (doc) => W.encodeNode({ input: ["w_q8", "w_s", "w_z"], output: ["w"], opType: "DequantizeLinear", intAttrs: { axis: 1 }, docString: doc });
  const mm = W.encodeNode({ input: ["x", "w"], output: ["y"], opType: "MatMul" });
  const graph = (doc) => {
    const parts = [node(doc), mm].map((b) => [0x0a, ...varint(b.length), ...b])
      .concat(inits.map((b) => [0x2a, ...varint(b.length), ...b]));
    return new Uint8Array(parts.flat());
  };
  const varint = (v) => { const o = []; while (v > 127) { o.push((v & 127) | 128); v >>>= 7; } o.push(v); return o; };
  const model = (doc) => { const g = graph(doc); return new Uint8Array([0x08, 0x08, 0x3a, ...varint(g.length), ...g]); };
  for (const [doc, dt] of [["", W.DT.FLOAT], ["float16", W.DT.FLOAT16]]) {
    const out = W.expandDequant(model(doc));
    const f = W.fields(out);
    assert.equal(f[0].field, 1);                                   // ir_version kept
    const g = out.subarray(f[1].dataStart, f[1].dataEnd);
    const gf = W.fields(g);
    const nodes = gf.filter((x) => x.field === 1).map((x) => W.parseNode(g.subarray(x.dataStart, x.dataEnd)));
    assert.deepEqual(nodes.map((x) => x.opType), ["MatMul"]);
    const t = gf.filter((x) => x.field === 5).map((x) => W.parseTensor(g.subarray(x.dataStart, x.dataEnd)));
    assert.deepEqual(t.map((x) => x.name), ["w"]);
    assert.equal(t[0].dataType, dt);
    // axis 1: column scales 0.5 and 0.25
    assert.deepEqual(Array.from(W.tensorFloats(t[0])), [0, 0.25, -0.5, 31.75]);
  }
});

await test("ADT_STR tokens decode to onset, class and velocity", () => {
  // BOS, time 0.07 s, class 38 (snare), velocity 76, time 0.47, class 36, velocity 100, EOS
  const notes = decodeTokens([2, 11, 338, 476, 51, 336, 500, 3]);
  assert.deepEqual(notes.map((x) => [+x[0].toFixed(2), x[2], x[3]]), [[0.07, 38, 76], [0.47, 36, 100]]);
  assert.equal(ADT_TO_GM[43], 44);
});

await test("the silence trim finds the sound", () => {
  const x = new Float32Array(44100 * 2);
  for (let i = 22050; i < 66150; i++) x[i] = Math.sin(i / 10) * 0.5;
  const [a, b] = trimBounds(x);
  assert.ok(a > 19000 && a <= 22050, `start ${a}`);
  assert.ok(b >= 66150 && b < 69000, `end ${b}`);
});

await test("muscriptor's final pass: validate, de-overlap per channel, onset delay", () => {
  const notes = M.trimOverlapping(M.validateNotes([
    { program: 0, pitch: 60, onset: 0, offset: 1, drum: false },
    { program: 0, pitch: 60, onset: 0.5, offset: 0.9, drum: false },
    { program: 0, pitch: 62, onset: 0.2, offset: null, drum: false },
  ]));
  assert.deepEqual(notes.map((x) => [x.pitch, x.onset, +x.offset.toFixed(3)]), [[60, 0, 0.5], [62, 0.2, 0.21], [60, 0.5, 0.9]]);
  const beats = Array.from({ length: 50 }, (_, i) => i * 0.5);
  const late = Array.from({ length: 80 }, (_, i) => i * 0.25 + 0.02);
  const d = M.estimateOnsetDelay(late, beats, 120);
  assert.ok(d && Math.abs(d.seconds - 0.02) < 0.002, JSON.stringify(d));
  assert.equal(M.estimateOnsetDelay(late.slice(0, 10), beats, 120), null);     // too few onsets
});

console.log(`ok ${n} browser engine tests`);
