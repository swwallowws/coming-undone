// Checks the public build of "In your browser": the studio page served as plain static
// files (as on GitHub Pages: no /browser-models/, no COOP/COEP headers), its models
// from the public URLs in index.html (CONFIG.publicModels, plus the pinned htdemucs in
// models.json). Run it after out/upload_models.py:
//
//   node browser/verify/public_models.mjs [clip] [--staged models-dist] [--port 8011]
//        [--ep webgpu|wasm] [--ref browser/verify/out/house-webgpu-bp.json]
//        [--drums-ref hits.json] [--tag name]
//
// 1. headers: every public model URL answers a cross-origin GET and a Range request
//    (CORS allowed for the page's origin, 206 with the manifest's size), the Hugging
//    Face redirect included.
// 2. the page: the browser engine turns on from the public models.json, offers
//    basic-pitch with MuScriptor small switched off, and runs the clip end to end in
//    headless Chrome, every model downloaded from the public URLs.
// 3. against an earlier run.mjs result (--ref, default house-webgpu-bp, whose clip is
//    the default clip): note counts, drum hits and tempo.
//
// --staged <dir>: before the upload. The model repo's URLs are answered from <dir>
// (scripts/stage_models.py) inside Chrome, at the same paths; htdemucs still comes
// from its real public URL, and the header checks cover it alone.
// --site <dir>: check a staged site (scripts/deploy-web.sh --stage <dir>) instead of
// web/static, opened under /coming-undone/ as on GitHub Pages.
// Writes browser/verify/out/<tag>.json.

import { createServer } from "node:http";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : d; };
const positional = args.filter((a, i) => !a.startsWith("--") && !args[i - 1]?.startsWith("--"));
const refPath = opt("ref", here("./out/house-webgpu-bp.json"));
const ref = existsSync(refPath) ? JSON.parse(readFileSync(refPath, "utf8")).summary : null;
const clip = positional[0] || ref?.clip;
if (!clip) { console.error("usage: node browser/verify/public_models.mjs <clip> (no --ref result to take the clip from)"); process.exit(2); }
const port = +opt("port", "8011"), ep = opt("ep", ""), staged = opt("staged", null);
const tag = opt("tag", `public-${staged ? "staged" : "live"}-${clip.split("/").pop().replace(/\.\w+$/, "")}-${ep || "auto"}`);
const site = opt("site", null);
const STATIC = site ? normalize(join(site, "/")) : here("../../src/stemscribe/web/static/");
const SUBPATH = site ? "/coming-undone" : "";
const PUBLIC_ORIGIN = "https://swwallowws.github.io";      // web/server.py PUBLIC_ORIGINS

// the page's own public models URL, so a pinned revision is what gets checked
const html = readFileSync(join(STATIC, "index.html"), "utf8");
const publicBase = html.match(/publicModels:\s*"([^"]+)"/)?.[1];
if (!publicBase) { console.error("no CONFIG.publicModels in index.html"); process.exit(2); }

const checks = [];
const check = (name, ok, detail) => { checks.push({ name, ok: !!ok, detail }); console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? `: ${detail}` : ""}`); };

// ---- 1. headers ----------------------------------------------------------------
async function corsGet(url, range) {
  const headers = { Origin: PUBLIC_ORIGIN, ...(range ? { Range: range } : {}) };
  const hops = [];
  let r = await fetch(url, { headers, redirect: "manual" });
  while (r.status >= 300 && r.status < 400) {
    hops.push({ status: r.status, acao: r.headers.get("access-control-allow-origin") });
    const next = new URL(r.headers.get("location"), url).href;
    await r.arrayBuffer().catch(() => {});
    r = await fetch(next, { headers, redirect: "manual" });
  }
  hops.push({ status: r.status, acao: r.headers.get("access-control-allow-origin"), contentRange: r.headers.get("content-range") });
  const body = range ? null : await r.text();
  if (range) await r.arrayBuffer().catch(() => {});
  return { hops, body, final: hops[hops.length - 1] };
}
const corsOk = (hops) => hops.every((h) => h.acao === "*" || h.acao === PUBLIC_ORIGIN);
async function checkFile(name, url, bytes) {
  const { hops, final } = await corsGet(url, "bytes=0-15");
  const total = +(final.contentRange || "").split("/")[1];
  check(`${name}: CORS for ${PUBLIC_ORIGIN}`, corsOk(hops), hops.map((h) => `${h.status} ${h.acao}`).join(" -> "));
  check(`${name}: range request`, final.status === 206 && total === bytes, `${final.status}, ${final.contentRange}, manifest ${bytes}`);
}

let manifest;
if (staged) {
  manifest = JSON.parse(readFileSync(join(staged, "models.json"), "utf8"));
  for (const part of ["encoder", "decoder"]) {
    const e = manifest.adt[part];
    const f = join(staged, e.file);
    check(`${e.file}: staged, size as in models.json`, existsSync(f) && statSync(f).size === e.bytes, existsSync(f) ? `${statSync(f).size}` : "missing");
  }
  check("staged models.json leaves MuScriptor out", !manifest.muscriptorSmall);
} else {
  const { hops, body, final } = await corsGet(new URL("models.json", publicBase).href);
  check("models.json: CORS", final.status === 200 && corsOk(hops), hops.map((h) => `${h.status} ${h.acao}`).join(" -> "));
  manifest = JSON.parse(body);
  check("public models.json leaves MuScriptor out", !manifest.muscriptorSmall);
  for (const part of ["encoder", "decoder"]) {
    const e = manifest.adt[part];
    await checkFile(e.file, new URL(e.file, publicBase).href, e.bytes);
  }
}
const htdemucsUrl = new URL(manifest.htdemucs.url || manifest.htdemucs.file, publicBase).href;
check("htdemucs pinned to a revision", /\/resolve\/[0-9a-f]{40}\//.test(htdemucsUrl), htdemucsUrl);
await checkFile("htdemucs_embedded.onnx", htdemucsUrl, manifest.htdemucs.bytes);

// ---- 2. the page, as GitHub Pages serves it -------------------------------------
const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json",
               ".bin": "application/octet-stream", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon",
               ".woff2": "font/woff2", ".sf3": "application/octet-stream", ".wasm": "application/wasm" };
const server = createServer((req, res) => {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (SUBPATH) {
    if (!p.startsWith(`${SUBPATH}/`)) { res.writeHead(404); res.end("not found"); return; }
    p = p.slice(SUBPATH.length);
  }
  if (p.endsWith("/")) p += "index.html";
  const f = normalize(join(STATIC, p));
  if (!f.startsWith(STATIC) || !existsSync(f) || statSync(f).isDirectory()) { res.writeHead(404); res.end("not found"); return; }
  res.writeHead(200, { "content-type": MIME[extname(f)] || "application/octet-stream" });
  res.end(readFileSync(f));
});
await new Promise((ok) => server.listen(port, ok));

const browser = await chromium.launch({
  channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage();
const errors = [], modelRequests = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("request", (r) => { if (/huggingface\.co|hf\.co/.test(r.url())) modelRequests.push(r.url()); });
if (staged) {
  await page.route(`${publicBase}**`, (route) => {
    const name = route.request().url().slice(publicBase.length).split("?")[0];
    const f = normalize(join(staged, name));
    if (!f.startsWith(normalize(staged)) || !existsSync(f)) return route.fulfill({ status: 404, headers: { "access-control-allow-origin": "*" }, body: "" });
    return route.fulfill({ status: 200, headers: { "access-control-allow-origin": "*", "content-type": MIME[extname(f)] || "application/octet-stream" }, body: readFileSync(f) });
  });
}
await page.goto(`http://localhost:${port}${SUBPATH}/?debug=1${ep ? `&ep=${ep}` : ""}`);
let turnedOn = true;
try {
  await page.waitForFunction(() => { const b = document.querySelector('[data-engine="browser"]'); return b && !b.disabled; }, null, { timeout: 30000 });
} catch (_) { turnedOn = false; }
check("In your browser turns on from the public models.json", turnedOn);
let out = { error: "engine did not turn on" }, wallMs = 0;
if (turnedOn) {
  const ui = await page.evaluate(() => {
    document.querySelector('[data-engine="browser"]').click();
    const bs = [...document.querySelectorAll("#backend-choice button")].map((b) => ({ v: b.dataset.v, disabled: b.disabled, title: b.title }));
    return { bs, hint: document.getElementById("backend-hint").textContent, where: document.getElementById("where-text").textContent };
  });
  const bp = ui.bs.find((b) => b.v === "basic-pitch"), ms = ui.bs.find((b) => b.v === "muscriptor-small");
  check("basic-pitch offered", bp && !bp.disabled);
  check("MuScriptor small switched off, pointing to Online / This computer", ms && ms.disabled && /Online or on This computer/.test(ms.title + ui.hint),
        ms ? `"${ms.title}"; hint "${ui.hint}"` : "no muscriptor-small button");
  await page.setInputFiles("#file", clip);
  const t0 = Date.now();
  await page.click("#go");
  await page.waitForFunction(() => window.__cuLastRun || !document.getElementById("err-panel").classList.contains("hidden"),
    null, { timeout: 30 * 60 * 1000, polling: 500 });
  wallMs = Date.now() - t0;
  out = await page.evaluate(() => {
    const r = window.__cuLastRun;
    const log = [...document.querySelectorAll("#log > div")].map((d) => d.textContent);
    if (!r) return { error: document.getElementById("err").textContent, log };
    return { res: r.res, parts: r.parts, log, resultsShown: !document.getElementById("results").classList.contains("hidden") };
  });
}
await browser.close();
server.close();

// ---- 3. results, against the earlier run -----------------------------------------
const summary = { tag, clip, publicBase, staged, ep: ep || "auto", wallMs, errors, modelRequests: [...new Set(modelRequests.map((u) => u.split("?")[0]))] };
if (out.error) {
  check("run finished", false, out.error);
} else {
  const { res, parts } = out;
  check("run finished, results shown", out.resultsShown);
  const fromPublic = (f) => summary.modelRequests.some((u) => u === new URL(f, publicBase).href);
  check("drum models fetched from the public repo URLs", fromPublic(manifest.adt.encoder.file) && fromPublic(manifest.adt.decoder.file));
  check("htdemucs fetched from its pinned URL", summary.modelRequests.includes(htdemucsUrl));
  check("every part has notes", parts.parts.every((p) => p.notes.length > 0), parts.parts.map((p) => `${p.name} ${p.notes.length}`).join(", "));
  Object.assign(summary, {
    engine: res.engine, timings: res.timings, tempo: res.tempo.bpm, warnings: res.warnings,
    tracks: Object.fromEntries(Object.entries(res.tracks).map(([k, s]) => [k, { raw: s.notes_before, count: s.note_count }])),
  });
  if (ref && ref.clip === clip) {
    const near = (a, b, rel, abs) => Math.abs(a - b) <= Math.max(abs, rel * b);
    for (const [k, s] of Object.entries(ref.tracks)) {
      const mine = summary.tracks[k];
      const a = mine ? (mine.raw ?? mine.count) : 0, b = s.raw ?? s.count;
      check(`${k}: notes near the earlier run`, mine && near(a, b, 0.1, 3), `${a} vs ${b}`);
    }
    check("tempo as in the earlier run", Math.abs(res.tempo.bpm - ref.tempo.bpm) < 0.5, `${res.tempo.bpm} vs ${ref.tempo.bpm}`);
    summary.earlier = { tag: ref.tag, ep: ref.engine?.ep, crossOriginIsolated: ref.crossOriginIsolated, timings: ref.timings, drums: ref.drums };
  } else if (ref) {
    console.log(`(no comparison: --ref ran ${ref.clip})`);
  }
  if (opt("drums-ref")) {
    const refHits = JSON.parse(readFileSync(opt("drums-ref"), "utf8")).hits_cpu;
    const off = res.prepared.offset || 0;
    const est = parts.parts.find((p) => p.drum).notes.map((n) => n[1] - off);
    const pairs = [];
    refHits.forEach((h, i) => est.forEach((t, j) => { const d = Math.abs(h.time - t); if (d <= 0.05) pairs.push([d, i, j]); }));
    pairs.sort((x, y) => x[0] - y[0]);
    const ri = new Set(), ej = new Set();
    for (const [, i, j] of pairs) if (!ri.has(i) && !ej.has(j)) { ri.add(i); ej.add(j); }
    const f1 = (2 * ri.size) / (refHits.length + est.length);
    summary.drums = { hits: est.length, refHits: refHits.length, onsetF1: +f1.toFixed(3) };
    check("drums onset F1 against the reference", f1 >= 0.95, `${f1.toFixed(3)} (${est.length}/${refHits.length} hits)`);
  }
  console.log(`timings now:     ${JSON.stringify(res.timings)} (${res.engine.ep}, isolated ${res.engine.crossOriginIsolated})`);
  if (summary.earlier) console.log(`timings earlier: ${JSON.stringify(ref.timings)} (${ref.engine?.ep}, isolated ${ref.crossOriginIsolated})`);
}
mkdirSync(here("./out/"), { recursive: true });
writeFileSync(here(`./out/${tag}.json`), JSON.stringify({ summary, checks, log: out.log }, null, 1));
const failed = checks.filter((c) => !c.ok);
console.log(failed.length ? `${failed.length} of ${checks.length} checks failed` : `all ${checks.length} checks passed`);
process.exit(failed.length ? 1 : 0);
