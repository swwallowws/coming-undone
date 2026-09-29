// Coming Undone's "In your browser" engine. The whole run happens in the page:
// htdemucs separation (ONNX), basic-pitch on vocals / bass / other (TF.js), ADT_STR
// drums (ONNX), and MuScriptor small as the opt-in pitched transcriber (ONNX). WebGPU
// when the browser has it, wasm otherwise. Nothing is uploaded; models download once
// and stay in Cache Storage.
//
// It hands back the same Run as the other engines (static/js/engines.js): res, roll,
// parts, fileUrl, partUrl, restamp (null), shiftBar (null). Built into
// static/vendor/browser-engine/engine.js by browser/build.mjs.

import * as ort from "onnxruntime-web";
import * as tf from "@tensorflow/tfjs";
import { BasicPitch, addPitchBendsToNoteEvents, noteFramesToTime, outputToNotesPoly } from "@spotify/basic-pitch";
import { DemucsProcessor } from "demucs-web";
import { SR, decodeFile, mono, prepare, resampleOAC, resampleTA, rms, wavBlob } from "./audio.js";
import { SR as DRUM_SR, transcribeDrums } from "./drums.js";
import { SR as MS_SR, embTable, transcribeMuScriptor } from "./muscriptor.js";
import { cleanNotes, quantizationError } from "./cleanup.js";
import { estimateTempo } from "./tempo.js";
import { writeMidi } from "./midi.js";
import { expandDequant } from "./onnxwire.js";
import { fetchAll, loadManifest } from "./models.js";

export const VERSION = "1.0.0";

// stemscribe merge.py TRACK_SPEC / TRACK_ORDER
const TRACK_SPEC = { vocals: ["melody", 53], bass: ["bass", 33], other: ["comping", 0], drums: ["drums", 0] };
const TRACK_ORDER = ["melody", "bass", "guitar", "piano", "comping", "drums"];

// stemscribe backends.py basic-pitch defaults
const BP = { onset: 0.5, frame: 0.3, minFrames: Math.round((127.7 / 1000) * (22050 / 256)) };
const SILENT_RMS = 1e-3;

export class BrowserEngineError extends Error {
  constructor(kind, message, detail = "") { super(message); this.kind = kind; this.detail = detail; }
}

export async function probe() {
  let adapter = null;
  try { adapter = navigator.gpu ? await navigator.gpu.requestAdapter() : null; } catch (_) { adapter = null; }
  return {
    webgpu: !!adapter,
    adapter: adapter && adapter.info ? `${adapter.info.vendor} ${adapter.info.architecture}`.trim() : null,
    crossOriginIsolated: !!self.crossOriginIsolated,
    cores: navigator.hardwareConcurrency || 4,
  };
}

const mb = (b) => `${Math.round(b / 1e6)} MB`;

export function createBrowserEngine(config = {}) {
  const cfg = { models: "/browser-models/", ortWasm: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/",
                basicPitch: "/vendor/browser-engine/basic-pitch/model.json", ...config };
  return {
    id: "browser",
    probe,
    run: (input, opts, on) => run(cfg, input, opts, on),
  };
}

async function run(cfg, input, opts = {}, on = () => {}) {
  if (!input.file) throw new BrowserEngineError("input", "In your browser takes a file. Links work on This computer.");
  const t0 = performance.now();
  const timings = {};
  const lap = (k, t) => { timings[k] = +((performance.now() - t) / 1000).toFixed(2); };
  const warnings = [];
  const say = (stage, message) => on({ stage, message });

  // --- engine ---------------------------------------------------------------
  const env = await probe();
  const ep = cfg.ep && cfg.ep !== "auto" ? cfg.ep : env.webgpu ? "webgpu" : "wasm";
  ort.env.wasm.wasmPaths = cfg.ortWasm;
  ort.env.wasm.numThreads = env.crossOriginIsolated ? Math.min(env.cores, 8) : 1;
  say("engine", ep === "webgpu" ? `WebGPU${env.adapter ? ` (${env.adapter})` : ""}`
    : `wasm, ${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads === 1 ? "" : "s"} (no WebGPU here, so it is slower)`);
  const backend = opts.backend === "muscriptor-small" ? "muscriptor-small" : "basic-pitch";
  if (opts.demucs_model === "htdemucs_6s") warnings.push("6 stems are not available in the browser yet: this ran 4 stems (guitar and piano stay in \"other\")");
  if (opts.snap || (opts.meter && opts.meter !== "4/4") || opts.downbeat) {
    warnings.push("in the browser there is no beat-grid fit yet: snap, meter and bar one were not applied (This computer and Online apply them)");
  }

  // --- prepare ----------------------------------------------------------------
  let t = performance.now();
  say("prepare", `decoding ${input.file.name}`);
  let audio;
  try { audio = await decodeFile(input.file, SR); } catch (e) {
    throw new BrowserEngineError("input", "This browser could not decode that file. Try mp3 or wav.", String(e));
  }
  const prep = prepare(audio, { start: opts.start, duration: opts.duration, trimSilence: opts.trim_silence !== false });
  audio = null;
  lap("prepare", t);
  const seconds = prep.left.length / SR;
  if (seconds < 1) throw new BrowserEngineError("input", "That is under a second of audio after the section and trim.");

  // --- models -----------------------------------------------------------------
  const models = await loadManifest(new URL(cfg.models, location.href).href);
  if (backend === "muscriptor-small" && !models.muscriptorSmall) {
    throw new BrowserEngineError("input", "MuScriptor small's model files are not available on this site. Pick basic-pitch, or run it on This computer.");
  }
  const progress = (label) => {
    let last = 0, fetched = false;
    return (done, total, fromCache) => {
      fetched ||= !fromCache;
      const now = performance.now();
      if (now - last < 400 && done < total) return;
      last = now;
      if (fetched) say("download", `${label}: ${mb(done)} of ${mb(total)}`);
      else if (done >= total) say("models", `${label}: from this browser's cache`);
    };
  };
  const expand = (buf) => (ep === "webgpu" ? expandDequant(buf) : buf);   // see onnxwire.js
  const sessOpts = { executionProviders: [ep], graphOptimizationLevel: "all" };

  // --- 1. separate ------------------------------------------------------------
  t = performance.now();
  const [demucsBuf] = (await fetchAll([models.htdemucs], progress("separation model"))).bufs;
  say("separate", "separating into drums, bass, other and vocals");
  const proc = new DemucsProcessor({
    ort, sessionOptions: sessOpts,
    onProgress: (p) => say("separate", `segment ${p.currentSegment} of ${p.totalSegments}`),
  });
  await proc.loadModel(demucsBuf);
  const stems = await proc.separate(prep.left, prep.right);
  try { await proc.session.release(); } catch (_) { /* released */ }
  lap("separate", t);

  const files = new Map();
  const put = (name, blob) => { files.set(name, URL.createObjectURL(blob)); return name; };
  const stemFiles = {};
  for (const s of ["vocals", "drums", "bass", "other"]) stemFiles[s] = put(`stems/${s}.wav`, wavBlob(stems[s].left, stems[s].right, SR));
  t = performance.now();
  const instL = new Float32Array(prep.left.length), instR = new Float32Array(prep.left.length);
  for (const s of ["drums", "bass", "other"]) for (let i = 0; i < instL.length; i++) { instL[i] += stems[s].left[i]; instR[i] += stems[s].right[i]; }
  const base = (input.file.name || "song").replace(/\.[^.]+$/, "").replace(/[^\w.-]+/g, "_").slice(0, 60) || "song";
  const instrumental = put(`${base}_instrumental.wav`, wavBlob(instL, instR, SR));
  lap("instrumental", t);

  // --- 2. drums (ADT_STR) -------------------------------------------------------
  t = performance.now();
  const drumBufs = (await fetchAll([models.adtEncoder, models.adtDecoder], progress("drum model"))).bufs;
  say("drums", "transcribing the drums with ADT_STR");
  const enc = await ort.InferenceSession.create(expand(drumBufs[0]), ep === "webgpu" ? { ...sessOpts, preferredOutputLocation: "gpu-buffer" } : sessOpts);
  const dec = await ort.InferenceSession.create(expand(drumBufs[1]), sessOpts);
  const drumWave = resampleTA(mono(stems.drums.left, stems.drums.right), SR, DRUM_SR);
  const hits = await transcribeDrums(ort, enc, dec, drumWave, { onChunk: (i, n) => say("drums", `${i} of ${n} chunks`) });
  try { await enc.release(); await dec.release(); } catch (_) { /* released */ }
  lap("drums", t);

  // --- 3. tempo ---------------------------------------------------------------
  t = performance.now();
  let tempo;
  if (opts.tempo) {
    tempo = { bpm: +opts.tempo, source: "user", method: "user", analyzed_stem: null, candidates: [] };
    const p = 60 / tempo.bpm;
    tempo.beats = Array.from({ length: Math.ceil(seconds / p) + 1 }, (_, i) => i * p);
  } else {
    const est = estimateTempo(hits.map((h) => ({ time: h.time, weight: h.velocity / 127 })), seconds);
    tempo = est ? { ...est, source: "detected", method: "drum-hit autocorrelation (browser)", analyzed_stem: "drums" } : null;
  }
  lap("tempo", t);

  // --- 4. pitched parts ----------------------------------------------------------
  t = performance.now();
  const wanted = ["vocals", "bass", "other"].filter((s) => s !== "vocals" || opts.include_vocals_melody !== false);
  const raw = {};
  const fallbacks = {};
  let bp = null;
  const basicPitch = async (stem) => {
    if (!bp) {
      let ok = false;
      try { ok = await tf.setBackend("webgl"); } catch (_) { ok = false; }
      if (!ok) await tf.setBackend("cpu");
      await tf.ready();
      bp = new BasicPitch(new URL(cfg.basicPitch, location.href).href);
      await bp.model;
    }
    const buf = await resampleOAC(mono(stems[stem].left, stems[stem].right), SR, 22050);
    const frames = [], onsets = [], contours = [];
    await bp.evaluateModel(buf, (f, o, c) => { frames.push(...f); onsets.push(...o); contours.push(...c); }, () => {});
    const ev = noteFramesToTime(addPitchBendsToNoteEvents(contours, outputToNotesPoly(frames, onsets, BP.onset, BP.frame, BP.minFrames)));
    return ev.map((n) => ({ pitch: n.pitchMidi, start: n.startTimeSeconds, end: n.startTimeSeconds + n.durationSeconds,
                            velocity: Math.max(1, Math.min(127, Math.round(127 * n.amplitude))) }));
  };
  if (backend === "muscriptor-small") {
    const ms = models.muscriptorSmall;
    const bufs = (await fetchAll([{ url: ms.meta.url, bytes: 0 }, { url: ms.emb.url, bytes: 2.1e6 }, { url: ms.cond.url, bytes: 1.6e6 },
                                  { url: ms.lm.url, bytes: ms.bytes - 3.7e6 }], progress("MuScriptor small"))).bufs;
    const meta = JSON.parse(new TextDecoder().decode(bufs[0]));
    const emb = embTable(bufs[1], meta);
    const cond = await ort.InferenceSession.create(bufs[2], sessOpts);
    const L = meta.layers;
    const lmOpts = { ...sessOpts };
    if (ep === "webgpu") {
      lmOpts.preferredOutputLocation = { logits: "cpu" };
      for (let i = 0; i < L; i++) { lmOpts.preferredOutputLocation[`present.${i}.key`] = "gpu-buffer"; lmOpts.preferredOutputLocation[`present.${i}.value`] = "gpu-buffer"; }
    }
    // the int8 download always goes back to float16 here: its graph computes in float16
    const lm = await ort.InferenceSession.create(expandDequant(bufs[3]), lmOpts);
    for (const stem of wanted) {
      say("transcribe", `transcribing ${stem} with MuScriptor small`);
      const w16 = (await resampleOAC(mono(stems[stem].left, stems[stem].right), SR, MS_SR)).getChannelData(0);
      const out = await transcribeMuScriptor(ort, { cond, lm }, meta, emb, w16, {
        ep, beats: tempo && tempo.beats, bpm: tempo && tempo.bpm,
        onChunk: (i, n) => say("transcribe", `${stem}: ${i} of ${n} chunks`),
      });
      raw[stem] = out.notes.map((n) => ({ pitch: n.pitch, start: n.onset, end: n.offset, velocity: 100 }));
    }
    try { await cond.release(); await lm.release(); } catch (_) { /* released */ }
    // backends.is_sparse: a stem with sound but under one note per 4 bars came out
    // nearly empty; basic-pitch gets it again, as on the server
    const bpmForBars = tempo ? tempo.bpm : 120;
    for (const stem of wanted) {
      const bars = seconds * bpmForBars / 240, level = rms(mono(stems[stem].left, stems[stem].right));
      if (level > SILENT_RMS && bars >= 8 && raw[stem].length < bars / 4) {
        warnings.push(`stem '${stem}': muscriptor-small gave only ${raw[stem].length} notes; transcribed it again with basic-pitch`);
        say("transcribe", `transcribing ${stem} again with basic-pitch`);
        raw[stem] = await basicPitch(stem);
        fallbacks[stem] = "basic-pitch";
      }
    }
  } else {
    for (const stem of wanted) {
      say("transcribe", `transcribing ${stem} with basic-pitch`);
      raw[stem] = await basicPitch(stem);
    }
  }
  if (bp) { try { (await bp.model).dispose(); } catch (_) { /* gone */ } }
  lap("transcribe", t);

  if (!tempo) {
    const ons = Object.values(raw).flat().map((n) => ({ time: n.start, weight: n.velocity / 127 }));
    const est = estimateTempo(ons, seconds);
    tempo = est ? { ...est, source: "detected", method: "note-onset autocorrelation (browser)", analyzed_stem: "pitched" }
      : { bpm: 120, source: "default", method: "none", analyzed_stem: null, candidates: [], beats: [] };
    if (!est) warnings.push("tempo detection found no steady pulse; MIDI written at 120 BPM");
  }
  say("tempo", `tempo ${tempo.bpm.toFixed(2)} BPM${tempo.analyzed_stem ? ` (from ${tempo.analyzed_stem})` : ""}`);

  // --- 5. cleanup, merge ----------------------------------------------------------
  t = performance.now();
  const mono_stems = new Set(opts.mono_stems || []);
  const trackStats = {}, tracks = {};
  for (const stem of wanted) {
    const [name, program] = TRACK_SPEC[stem];
    if (opts.cleanup === false) {
      tracks[name] = { name, program, drum: false, notes: raw[stem] };
      trackStats[name] = { notes_before: raw[stem].length, notes_after: raw[stem].length, cleanup: "disabled" };
      continue;
    }
    const { notes, stats } = cleanNotes(raw[stem], {
      tempo: tempo.bpm, de_overlap: opts.de_overlap !== false, max_duration_beats: opts.max_duration_beats ?? 2,
      velocity_floor: opts.velocity_floor ?? 15, legato: !!opts.legato, legato_max_gap_beats: opts.legato_max_gap_beats ?? 0.25,
      monophonic: mono_stems.has(stem),
    });
    tracks[name] = { name, program, drum: false, notes };
    trackStats[name] = stats;
  }
  tracks.drums = { name: "drums", program: 0, drum: true,
                   notes: hits.map((h) => ({ pitch: h.pitch, start: h.time, end: h.time + 0.1, velocity: h.velocity })) };
  lap("cleanup", t);

  // back onto the original file's timeline
  const off = prep.info.offset;
  const order = TRACK_ORDER.filter((n) => tracks[n]);
  const track_map = Object.fromEntries(order.map((n, i) => [n, i]));
  const final = order.map((n) => ({ ...tracks[n], notes: tracks[n].notes.map((x) => ({ ...x, start: x.start + off, end: x.end + off })) }));
  for (const tr of final) {
    const s = trackStats[tr.name] || (trackStats[tr.name] = {});
    s.note_count = tr.notes.length;
    s.program = tr.program;
    s.quantization_error = quantizationError(tracks[tr.name].notes, tempo.bpm);
  }
  const nonEmpty = final.filter((x) => x.notes.length);
  const midi = put(`${base}.mid`, new Blob([writeMidi({ bpm: tempo.bpm, tracks: nonEmpty })], { type: "audio/midi" }));
  const partUrls = {};
  for (const tr of final) partUrls[tr.name] = URL.createObjectURL(new Blob([writeMidi({ bpm: tempo.bpm, tracks: [tr] })], { type: "audio/midi" }));
  timings.total = +((performance.now() - t0) / 1000).toFixed(2);

  const round = (x) => Math.round(x * 1e4) / 1e4;
  const noteRow = (n) => [n.pitch, round(n.start), round(n.end), Math.round(n.velocity)];
  const end = Math.max(off + seconds, ...final.flatMap((x) => x.notes.map((n) => n.end)));
  const parts = { end, parts: final.map((x) => ({ name: x.name, file: `${x.name}.mid`, drum: x.drum, notes: x.notes.map(noteRow) })) };
  const roll = { end, tracks: final.filter((x) => !x.drum && x.notes.length).map((x) => ({ name: x.name, notes: x.notes.map(noteRow) })) };
  const { beats, beat0, ...tempoOut } = tempo;
  const res = {
    tempo: tempoOut,
    grid: { fitted: false, disabled: true, reason: "no beat-grid fit in the browser yet" },
    prepared: prep.info,
    source: null,
    track_map,
    tracks: trackStats,
    timings,
    warnings,
    midi, instrumental,
    stems: stemFiles,
    manifest: null,
    engine: {
      id: "browser", version: VERSION, ep, adapter: env.adapter, crossOriginIsolated: env.crossOriginIsolated,
      wasmThreads: ort.env.wasm.numThreads, backend, drums: "adt-str", fallbacks,
    },
  };
  const manifestBlob = new Blob([JSON.stringify({ stemscribe_engine: "browser", ...res, parts: undefined }, null, 2)], { type: "application/json" });
  res.manifest = put("manifest.json", manifestBlob);
  say("done", `finished in ${timings.total.toFixed(1)} s`);
  return {
    res, roll, parts,
    fileUrl: (p) => (p ? files.get(p) || null : null),
    partUrl: (name) => partUrls[name] || null,
    restamp: null,
    shiftBar: null,
  };
}
