// MuScriptor small in the page: cond.onnx (log-mel -> prefix) + lm.onnx (one
// transformer step with an explicit KV cache), greedy decoding with prelude forcing,
// and ports of muscriptor 0.3.0's OpenNoteTracker (events.py), its mel front end
// (modules/mel_spectrogram.py) and its final note pass (validate_notes,
// trim_overlapping_notes, estimate_onset_delay). Mirrors TranscriptionModel.transcribe's
// defaults: greedy, cfg 1, 5 s chunks, batch 1, prelude forcing on.
//
// MuScriptor: Mirelo and Kyutai. Code MIT, weights CC-BY-NC 4.0 (non-commercial only).

import { fft, melFilterbank } from "./drums.js";

export const SR = 16000;
const CHUNK = 80000;
const MAX_GEN = 2000;
const EOS = 1;
const MIN_DUR = 0.01;

// ---- vocab (muscriptor.tokenizer.notes.build_event_vocab(1001)) ------------------
const VOCAB = [];
for (const t of ["PAD", "EOS", "UNK"]) VOCAB.push([t, 0]);
for (let v = 0; v <= 1000; v++) VOCAB.push(["shift", v]);
for (let v = 0; v <= 127; v++) VOCAB.push(["pitch", v]);
for (let v = 0; v <= 1; v++) VOCAB.push(["velocity", v]);
VOCAB.push(["tie", 0]);
for (let v = 0; v <= 129; v++) VOCAB.push(["program", v]);
for (let v = 0; v <= 127; v++) VOCAB.push(["drum", v]);
const TOKEN = new Map(VOCAB.map(([t, v], i) => [`${t}:${v}`, i]));

// ---- OpenNoteTracker (muscriptor/events.py) ------------------------------------------
export class Tracker {
  constructor(frameRate = 100) {
    this.fr = frameRate; this.open = new Map(); this.chunkStarted = false; this.notes = []; this.startIdx = new Map();
  }
  _start(prog, pitch, time) {
    this.startIdx.set(`${prog},${pitch}`, this.notes.length);
    this.notes.push({ program: prog, pitch, onset: time, offset: null, drum: false });
  }
  _end(key, time) {
    const i = this.startIdx.get(key);
    if (i !== undefined) { this.notes[i].offset = time; this.startIdx.delete(key); }
  }
  _endAll(time) { for (const k of this.open.keys()) this._end(k, time); this.open.clear(); }
  boundary(seek, nextSeek) {
    if (this.chunkStarted && this.inPrologue) this._endAll(this.seek);
    this.seek = seek; this.nextSeek = nextSeek;
    this.startTick = Math.round(seek * this.fr); this.tick = this.startTick;
    this.program = null; this.velocity = null; this.inPrologue = true; this.skipRest = false;
    this.tieSet = new Set(); this.chunkStarted = true;
  }
  feed(tok) {
    const [type, value] = VOCAB[tok];
    if (this.inPrologue) {
      if (type === "tie") {
        this.inPrologue = false; this.velocity = null;
        for (const k of [...this.open.keys()]) if (!this.tieSet.has(k)) { this.open.delete(k); this._end(k, this.seek); }
      } else if (type === "shift") {
        this.inPrologue = false; this.skipRest = true; this._endAll(this.seek);
      } else if (type === "program") this.program = value;
      else if (type === "pitch" && this.program !== null) this.tieSet.add(`${this.program},${value}`);
      return;
    }
    if (this.skipRest) return;
    if (type === "shift") { if (value > 0) this.tick = this.startTick + value; }
    else if (type === "program") this.program = value;
    else if (type === "velocity") this.velocity = value;
    else if (type === "drum") {
      const t = this.tick / this.fr;
      if (this.nextSeek === null || t < this.nextSeek) this.notes.push({ program: 128, pitch: value, onset: t, offset: t + MIN_DUR, drum: true });
    } else if (type === "pitch") {
      if (this.program === null || this.velocity === null) return;
      const t = this.tick / this.fr;
      if (this.nextSeek !== null && t >= this.nextSeek) return;
      const key = `${this.program},${value}`;
      if (this.open.has(key)) { this.open.delete(key); this._end(key, t); }
      if (this.velocity > 0) { this.open.set(key, t); this._start(this.program, value, t); }
    }
  }
  finish() {
    if (this.chunkStarted && this.inPrologue) return this._endAll(this.seek);
    for (const [k, onset] of this.open) this._end(k, onset + MIN_DUR);
    this.open.clear();
  }
  openKeys() {
    return [...this.open.keys()].map((k) => k.split(",").map(Number)).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  }
}

function tieSection(keys) {
  const toks = [];
  let prog = null;
  for (const [p, pitch] of keys) {
    if (p !== prog) { toks.push(TOKEN.get(`program:${p}`)); prog = p; }
    toks.push(TOKEN.get(`pitch:${pitch}`));
  }
  toks.push(TOKEN.get("tie:0"));
  return toks;
}

// ---- mel front end ---------------------------------------------------------------
let melCache = null;
function melSetup(m) {
  if (melCache && melCache.m === m) return melCache;
  const rows = melFilterbank(m.sampleRate, m.nFft, m.nMels, m.fMin, m.fMax).map((row) => {
    let a = 0, b = row.length - 1;
    while (a < row.length && row[a] === 0) a++;
    while (b > a && row[b] === 0) b--;
    return [a, row.slice(a, b + 1)];
  });
  // The window as stored in the weights (meta.json): not an exact Hann, and its
  // sidelobes decide the near-silent top bands.
  const win = m.window ? Float64Array.from(m.window)
    : Float64Array.from({ length: m.nFft }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / m.nFft));
  melCache = { m, rows, win };
  return melCache;
}

// One chunk -> log-mel [frames * nMels] (centered, reflect padding, magnitude).
export function logmelMS(wav, m) {
  const { rows, win } = melSetup(m);
  const N = m.nFft, hop = m.hop, pad = N / 2, n = wav.length;
  const frames = 1 + Math.floor(n / hop);
  const at = (i) => wav[i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i];
  const out = new Float32Array(frames * m.nMels);
  const re = new Float64Array(N), im = new Float64Array(N), mag = new Float64Array(N / 2 + 1);
  for (let f = 0; f < frames; f++) {
    const start = f * hop - pad;
    for (let i = 0; i < N; i++) { re[i] = at(start + i) * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k <= N / 2; k++) mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
    for (let b = 0; b < m.nMels; b++) {
      const [a, vals] = rows[b];
      let s = 0;
      for (let k = 0; k < vals.length; k++) s += vals[k] * mag[a + k];
      out[f * m.nMels + b] = Math.log(s + m.eps);
    }
  }
  return { data: out, frames };
}

// ---- the final note pass ---------------------------------------------------------
export function validateNotes(notes) {
  const out = [];
  for (const n of notes) {
    if (n.onset == null) continue;
    const x = { ...n };
    if (x.offset == null) x.offset = x.onset + MIN_DUR;
    else if (x.onset > x.offset) x.offset = Math.max(x.offset, x.onset + MIN_DUR);
    else if (!x.drum && x.offset - x.onset < 0.01) x.offset = x.onset + MIN_DUR;
    out.push(x);
  }
  return out;
}

export function trimOverlapping(notes) {
  if (notes.length <= 1) return notes;
  const channels = new Map();
  for (const n of notes) {
    const k = `${n.program},${n.pitch},${n.drum}`;
    if (!channels.has(k)) channels.set(k, []);
    channels.get(k).push(n);
  }
  const out = [];
  for (const ch of channels.values()) {
    ch.sort((a, b) => a.onset - b.onset);
    for (let i = 1; i < ch.length; i++) if (ch[i - 1].offset > ch[i].onset) ch[i - 1].offset = ch[i].onset;
    for (const n of ch) if (n.onset < n.offset) out.push(n);
  }
  out.sort((a, b) => a.onset - b.onset || (a.drum - b.drum) || a.program - b.program || a.pitch - b.pitch);
  return out;
}

const SUBDIVISIONS = [1, 2, 3, 4, 6, 8, 12, 16, 24];

// How late the onsets sit against the beat subdivision they are on (utils/beats.py),
// or null when that cannot be told reliably.
export function estimateOnsetDelay(onsets, beats, bpm) {
  if (!beats || beats.length < 2) return null;
  const period = 60 / bpm;
  const times = [...new Set(onsets.map((t) => Math.round(t * 1000) / 1000))].sort((a, b) => a - b)
    .filter((t) => t >= beats[0] && t <= beats[beats.length - 1]);
  if (times.length < 40) return null;
  const phase = times.map((t) => {
    let i = 1;
    while (i < beats.length - 1 && beats[i] < t) i++;
    return i - 1 + (t - beats[i - 1]) / (beats[i] - beats[i - 1]);
  });
  let best = null;
  for (const s of SUBDIVISIONS) {
    if (period / (2 * s) < 0.04) continue;
    let cx = 0, cy = 0;
    for (const p of phase) {
      const x = p * s, a = 2 * Math.PI * (x - Math.floor(x));
      cx += Math.cos(a); cy += Math.sin(a);
    }
    const conc = Math.hypot(cx, cy) / phase.length;
    if (!best || conc > best.conc) best = { s, conc, turns: Math.atan2(cy, cx) / (2 * Math.PI * s) };
  }
  if (!best || best.conc < 0.5) return null;
  const seconds = best.turns * period;
  return Math.abs(seconds) > 0.04 ? null : { seconds, concentration: best.conc, subdivision: best.s };
}

// ---- generation ------------------------------------------------------------------
function f16ToF32(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

export function embTable(buf, meta) {
  if (meta.dtype !== "fp16") return new Float32Array(buf);
  return Float32Array.from(new Uint16Array(buf), f16ToF32);
}

// wav: mono 16 kHz. Returns { notes, rawCount, tokens, onsetDelay } with pitched notes
// [{pitch, onset, offset, program}] (drum tokens dropped: the drums come from ADT_STR).
export async function transcribeMuScriptor(ort, { cond, lm }, meta, emb, wav, { ep, beats = null, bpm = null, onChunk = () => {} } = {}) {
  const { dim: D, heads: H, headDim: Dh, layers: L, card } = meta;
  const kvType = meta.dtype === "fp16" ? "float16" : "float32";
  const embRow = (tok, dst, off) => dst.set(emb.subarray(tok * D, (tok + 1) * D), off);
  const presentNames = [];
  for (let i = 0; i < L; i++) presentNames.push(`present.${i}.key`, `present.${i}.value`);
  const total = Math.ceil(wav.length / CHUNK);
  const tracker = new Tracker();
  let tokens = 0;
  for (let c = 0; c < total; c++) {
    tracker.boundary(c * 5, c + 1 < total ? (c + 1) * 5 : null);
    const prompt = c > 0 ? tieSection(tracker.openKeys()) : [];
    for (const tok of prompt) tracker.feed(tok);
    const w = new Float32Array(CHUNK);
    w.set(wav.subarray(c * CHUNK, Math.min((c + 1) * CHUNK, wav.length)));
    const { data, frames } = logmelMS(w, meta.mel);
    const pre = (await cond.run({ mel: new ort.Tensor("float32", data, [1, frames, meta.mel.nMels]) })).prefix;
    const P = pre.dims[1], T0 = P + 1 + prompt.length;
    const x = new Float32Array(T0 * D);
    x.set(pre.data);
    embRow(meta.bos, x, P * D);
    prompt.forEach((tok, i) => embRow(tok, x, (P + 1 + i) * D));
    const empty = kvType === "float16" ? new Uint16Array(0) : new Float32Array(0);
    const feeds = { x: new ort.Tensor("float32", x, [1, T0, D]) };
    for (let i = 0; i < L; i++) {
      feeds[`past.${i}.key`] = new ort.Tensor(kvType, empty, [1, H, 0, Dh]);
      feeds[`past.${i}.value`] = new ort.Tensor(kvType, empty, [1, H, 0, Dh]);
    }
    let res = await lm.run(feeds);
    const one = new Float32Array(D);
    let n = 0;
    for (;;) {
      const lg = res.logits.data;
      let best = 0, bv = -Infinity;
      for (let i = 0; i < Math.min(card, 1393); i++) if (lg[i] > bv) { bv = lg[i]; best = i; }
      if (best === EOS) break;
      tracker.feed(best);
      n++;
      if (prompt.length + n >= MAX_GEN) break;
      embRow(best, one, 0);
      const f = { x: new ort.Tensor("float32", one.slice(), [1, 1, D]) };
      for (let i = 0; i < L; i++) {
        f[`past.${i}.key`] = res[`present.${i}.key`];
        f[`past.${i}.value`] = res[`present.${i}.value`];
      }
      const prev = res;
      res = await lm.run(f);
      if (ep === "webgpu") for (const k of presentNames) { try { prev[k].dispose(); } catch (_) { /* cpu */ } }
    }
    if (ep === "webgpu") for (const k of presentNames) { try { res[k].dispose(); } catch (_) { /* cpu */ } }
    tokens += n;
    onChunk(c + 1, total);
  }
  tracker.finish();
  let notes = trimOverlapping(validateNotes(tracker.notes.filter((x) => x.offset !== null)));
  const delay = beats && bpm ? estimateOnsetDelay(notes.map((x) => x.onset), beats, bpm) : null;
  if (delay && delay.seconds) notes = notes.map((x) => ({ ...x, onset: x.onset - delay.seconds, offset: x.offset - delay.seconds }));
  return { notes: notes.filter((x) => !x.drum), tokens, onsetDelay: delay };
}
