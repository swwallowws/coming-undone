// End-to-end check of the "In your browser" engine: drives the real studio page in
// headless Chrome (Runs: In your browser, pick a file, Transcribe) and compares what
// comes back with reference transcriptions.
//
//   node browser/verify/run.mjs <clip> [--port 8003] [--ep webgpu|wasm]
//        [--backend basic-pitch|muscriptor-small] [--drums-ref hits.json]
//        [--notes-ref server.mid] [--notes-track comping] [--tag name]
//
// Needs `stemscribe-web --port <port> --browser-models <models dir>` running.
// --drums-ref: {"hits_cpu": [{time, pitch}], ...} from PyTorch ADT_STR on the server's
//   drums stem (times on the server's trimmed timeline; the page's are shifted back
//   by its own trim, so both are compared on the trimmed timeline).
// --notes-ref: a MIDI file whose --notes-track is compared with the page's track
//   (onset F1, 50 ms, same pitch).
// Writes browser/verify/out/<tag>.json and prints a summary.

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { chromium } from "playwright-core";
import { readMidi } from "../src/midi.js";

const args = process.argv.slice(2);
const clip = args.find((a) => !a.startsWith("--") && !args[args.indexOf(a) - 1]?.startsWith("--"));
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : d; };
const port = opt("port", "8003"), ep = opt("ep", ""), backend = opt("backend", "basic-pitch");
const tag = opt("tag", `${clip.split("/").pop().replace(/\.\w+$/, "")}-${ep || "auto"}-${backend}`);

function chromeRssMB() {
  const txt = execFileSync("ps", ["-A", "-o", "pid=,ppid=,rss="], { encoding: "utf8", maxBuffer: 1 << 26 });
  const rows = txt.trim().split("\n").map((l) => l.trim().split(/\s+/).map(Number));
  const kids = new Map();
  for (const [pid, ppid] of rows) { if (!kids.has(ppid)) kids.set(ppid, []); kids.get(ppid).push(pid); }
  const rss = new Map(rows.map(([pid, , kb]) => [pid, kb]));
  let kb = 0;
  const stack = [...(kids.get(process.pid) || [])];
  while (stack.length) { const p = stack.pop(); kb += rss.get(p) || 0; stack.push(...(kids.get(p) || [])); }
  return kb / 1024;
}

// onset matching: greedy closest pairs within tol, optional same pitch
function match(ref, est, tol, samePitch) {
  const pairs = [];
  for (let i = 0; i < ref.length; i++) for (let j = 0; j < est.length; j++) {
    const d = Math.abs(ref[i].time - est[j].time);
    if (d <= tol && (!samePitch || ref[i].pitch === est[j].pitch)) pairs.push([d, i, j]);
  }
  pairs.sort((a, b) => a[0] - b[0]);
  const ri = new Set(), ej = new Set(), out = [];
  for (const [d, i, j] of pairs) { if (ri.has(i) || ej.has(j)) continue; ri.add(i); ej.add(j); out.push([i, j, d]); }
  return out;
}
const f1 = (m, r, e) => (r + e ? (2 * m) / (r + e) : 1);
const drumClass = (g) => (g === 35 || g === 36 ? "kick" : g >= 37 && g <= 40 ? "snare" : [42, 44, 46].includes(g) ? "hihat"
  : [41, 43, 45, 47, 48, 50].includes(g) ? "tom" : [49, 52, 55, 57].includes(g) ? "crash" : [51, 53, 59].includes(g) ? "ride" : "other");

const browser = await chromium.launch({
  channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--ignore-gpu-blocklist"],
});
let peak = 0;
const timer = setInterval(() => { peak = Math.max(peak, chromeRssMB()); }, 250);
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
const url = `http://localhost:${port}/?debug=1${ep ? `&ep=${ep}` : ""}`;
await page.goto(url);
await page.waitForFunction(() => { const b = document.querySelector('[data-engine="browser"]'); return b && !b.disabled; }, null, { timeout: 30000 });
const whereBefore = await page.evaluate(() => { document.querySelector('[data-engine="browser"]').click(); return document.getElementById("where-text").textContent; });
await page.setInputFiles("#file", clip);
if (backend !== "basic-pitch") await page.click(`#backend-choice [data-v="${backend}"]`);
const t0 = Date.now();
await page.click("#go");
await page.waitForFunction(() => window.__cuLastRun || !document.getElementById("err-panel").classList.contains("hidden"),
  null, { timeout: 30 * 60 * 1000, polling: 500 });
const wallMs = Date.now() - t0;
const out = await page.evaluate(() => {
  const r = window.__cuLastRun;
  const log = [...document.querySelectorAll("#log > div")].map((d) => d.textContent);
  if (!r) return { error: document.getElementById("err").textContent, log };
  const rows = [...document.querySelectorAll("#tracks tr")].map((tr) => [...tr.children].map((td) => td.textContent.trim()));
  return { res: r.res, parts: r.parts, log, tableRows: rows, crossOriginIsolated: self.crossOriginIsolated,
           resultsShown: !document.getElementById("results").classList.contains("hidden"),
           midiLink: !!document.querySelector("#dl a.main") };
});
// --twice: "New file" (a reload) and the same run again in this browser profile; the
// models must come from Cache Storage this time
let second = null;
if (args.includes("--twice") && !out.error) {
  await page.reload();
  await page.waitForFunction(() => { const b = document.querySelector('[data-engine="browser"]'); return b && !b.disabled; }, null, { timeout: 30000 });
  second = { whereAfter: await page.evaluate(() => { document.querySelector('[data-engine="browser"]').click(); return document.getElementById("where-text").textContent; }) };
  await page.setInputFiles("#file", clip);
  if (backend !== "basic-pitch") await page.click(`#backend-choice [data-v="${backend}"]`);
  const t1 = Date.now();
  await page.click("#go");
  await page.waitForFunction(() => window.__cuLastRun || !document.getElementById("err-panel").classList.contains("hidden"),
    null, { timeout: 30 * 60 * 1000, polling: 500 });
  Object.assign(second, await page.evaluate(() => ({
    total: window.__cuLastRun && window.__cuLastRun.res.timings.total,
    downloads: [...document.querySelectorAll("#log > div")].map((d) => d.textContent).filter((t) => t.startsWith("download")),
  })));
  second.wallMs = Date.now() - t1;
  second.planText = await page.evaluate(async () => {
    const c = await caches.open("coming-undone-models-v1");
    return (await c.keys()).map((k) => k.url.split("/").pop());
  });
}
clearInterval(timer);
await browser.close();

const summary = { tag, clip, ep: ep || "auto", backend, wallMs, peakChromeRssMB: Math.round(peak), whereBefore, errors };
const checks = [];
const check = (name, ok, detail) => checks.push({ name, ok: !!ok, detail });
if (out.error) {
  summary.error = out.error;
  check("run finished", false, out.error);
} else {
  const { res, parts } = out;
  Object.assign(summary, {
    engine: res.engine, timings: res.timings, tempo: { bpm: res.tempo.bpm, source: res.tempo.source, candidates: res.tempo.candidates },
    prepared: res.prepared, warnings: res.warnings, crossOriginIsolated: out.crossOriginIsolated,
    tracks: Object.fromEntries(Object.entries(res.tracks).map(([k, s]) => [k, { raw: s.notes_before, after: s.notes_after ?? s.note_count, count: s.note_count }])),
  });
  check("results rendered with a MIDI download", out.resultsShown && out.midiLink);
  check("every part has notes", parts.parts.every((p) => p.notes.length > 0), parts.parts.map((p) => `${p.name} ${p.notes.length}`).join(", "));
  const off = res.prepared.offset || 0;
  const drums = parts.parts.find((p) => p.drum);
  if (opt("drums-ref") && drums) {
    const ref = JSON.parse(readFileSync(opt("drums-ref"), "utf8")).hits_cpu;
    const est = drums.notes.map((n) => ({ time: n[1] - off, pitch: n[0] }));
    const m = match(ref, est, 0.05, false);
    const classes = [...new Set([...ref, ...est].map((h) => drumClass(h.pitch)))];
    let mc = 0;
    const perClass = {};
    for (const c of classes) {
      const r = ref.filter((h) => drumClass(h.pitch) === c), e = est.filter((h) => drumClass(h.pitch) === c);
      const k = match(r, e, 0.05, false).length;
      mc += k;
      perClass[c] = +f1(k, r.length, e.length).toFixed(3);
    }
    summary.drums = { hits: est.length, refHits: ref.length, onsetF1: +f1(m.length, ref.length, est.length).toFixed(3),
                      classF1: +f1(mc, ref.length, est.length).toFixed(3), perClass,
                      meanTimingMs: +(m.reduce((a, x) => a + x[2], 0) / Math.max(1, m.length) * 1000).toFixed(1) };
  }
  if (opt("notes-ref")) {
    const trackName = opt("notes-track", "comping");
    const ref = readMidi(readFileSync(opt("notes-ref"))).tracks.find((t) => t.name === trackName);
    const mine = parts.parts.find((p) => p.name === trackName);
    const R = ref.notes.map((n) => ({ time: n.start, pitch: n.pitch })), E = mine.notes.map((n) => ({ time: n[1], pitch: n[0] }));
    summary.notes = { track: trackName, notes: E.length, refNotes: R.length, onsetF1: +f1(match(R, E, 0.05, true).length, R.length, E.length).toFixed(3) };
  }
}
if (second) {
  summary.secondRun = second;
  check("second run reads the models from Cache Storage", second.planText.length >= 3 && second.downloads.length === 0
    && /already downloaded/.test(second.whereAfter),
    `${second.planText.join(", ")}; ${second.downloads.length} download lines, ${second.total} s; "${second.whereAfter.slice(0, 90)}..."`);
}
mkdirSync(new URL("./out/", import.meta.url), { recursive: true });
writeFileSync(new URL(`./out/${tag}.json`, import.meta.url), JSON.stringify({ summary, checks, log: out.log, tableRows: out.tableRows }, null, 1));
console.log(JSON.stringify(summary, null, 1));
for (const c of checks) console.log(`${c.ok ? "ok  " : "FAIL"} ${c.name}${c.detail ? `: ${c.detail}` : ""}`);
process.exit(checks.every((c) => c.ok) ? 0 : 1);
