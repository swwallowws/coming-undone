// Sound check for the /try/ page's MIDI player: in headless Chrome, press Play, solo
// each part, switch it to MIDI and listen to the synth's output for a few seconds.
//   node browser/verify/try_sound.mjs [page url, default http://127.0.0.1:8011/try/]
// Serve a site frozen by scripts/freeze_try.py first. Checks that the General MIDI
// soundfont loads, that every part sounds, and that the parts sound like different
// instruments (bass low, drums broadband). Saves each part's recording as
// browser/verify/out/try-<part>.webm to listen to.

import { mkdirSync, writeFileSync } from "node:fs";
import { chromium } from "playwright-core";

const url = process.argv[2] || "http://127.0.0.1:8011/try/";
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
const page = await browser.newPage();
const errors = [], fetched = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "warning" || m.type() === "error") errors.push(m.text()); });
page.on("response", (r) => fetched.push(`${r.status()} ${r.url().split("/").slice(-2).join("/")}`));
await page.goto(url);
await page.waitForFunction(() => !document.getElementById("play").disabled, null, { timeout: 60000 });
await page.click("#play");
await page.waitForFunction(() => window.__trySound && window.__trySound().loaded, null, { timeout: 60000 });
const state = await page.evaluate(() => { const s = window.__trySound(); return { channels: s.channels, programs: s.programs }; });
const parts = await page.$$eval("#parts button[data-part]", (bs) => bs.map((b) => b.dataset.part).filter(Boolean));

const results = {};
mkdirSync(new URL("./out/", import.meta.url), { recursive: true });
for (const id of parts) {
  await page.click(`#parts button[data-part="${id}"]`);
  await page.click("#mode-midi");
  await page.waitForTimeout(400);                       // past the switch's quiet window
  const r = await page.evaluate(async (ms) => {
    const { ctx, out } = window.__trySound();
    const an = ctx.createAnalyser();
    an.fftSize = 4096;
    out.connect(an);
    const dest = ctx.createMediaStreamDestination();
    out.connect(dest);
    const rec = new MediaRecorder(dest.stream);
    const chunks = [];
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.start();
    const freq = new Float32Array(an.frequencyBinCount), time = new Float32Array(an.fftSize);
    const power = new Float64Array(an.frequencyBinCount);
    let sq = 0, n = 0, peak = 0;
    const t0 = performance.now();
    while (performance.now() - t0 < ms) {
      an.getFloatFrequencyData(freq);
      an.getFloatTimeDomainData(time);
      for (let i = 0; i < freq.length; i++) power[i] += 10 ** (freq[i] / 10);
      for (const x of time) { sq += x * x; peak = Math.max(peak, Math.abs(x)); }
      n += time.length;
      await new Promise((ok) => setTimeout(ok, 40));
    }
    rec.stop();
    await new Promise((ok) => { rec.onstop = ok; });
    out.disconnect(an); out.disconnect(dest);
    const hz = (i) => (i * ctx.sampleRate) / an.fftSize;
    let num = 0, den = 0, low = 0, high = 0;
    for (let i = 1; i < power.length; i++) {
      num += hz(i) * power[i]; den += power[i];
      if (hz(i) < 300) low += power[i];
      if (hz(i) > 4000) high += power[i];
    }
    const blob = new Blob(chunks, { type: rec.mimeType });
    const b64 = btoa(String.fromCharCode(...new Uint8Array(await blob.arrayBuffer())));
    return { rmsDb: +(10 * Math.log10(sq / n + 1e-12)).toFixed(1), peak: +peak.toFixed(3),
             centroidHz: Math.round(num / (den || 1)), lowShare: +(low / (den || 1)).toFixed(3),
             highShare: +(high / (den || 1)).toFixed(5), b64, type: rec.mimeType };
  }, 3000);
  writeFileSync(new URL(`./out/try-${id}.webm`, import.meta.url), Buffer.from(r.b64, "base64"));
  delete r.b64;
  results[id] = r;
}
// back to the audio stem: the synth must go quiet once its queued notes have passed
await page.click("#mode-audio");
await page.waitForTimeout(600);
const afterSwitch = await page.evaluate(async () => {
  const { ctx, out } = window.__trySound();
  const an = ctx.createAnalyser();
  out.connect(an);
  const t = new Float32Array(an.fftSize);
  let peak = 0;
  for (let i = 0; i < 20; i++) {
    an.getFloatTimeDomainData(t);
    for (const x of t) peak = Math.max(peak, Math.abs(x));
    await new Promise((ok) => setTimeout(ok, 50));
  }
  out.disconnect(an);
  return +peak.toFixed(4);
});
await browser.close();

const checks = [];
const quiet = afterSwitch;
const check = (name, ok, detail = "") => checks.push({ name, ok: !!ok, detail });
// the worklet processor loads through audioWorklet.addModule, not as a page response;
// `loaded` is only set once the synth and the soundfont are both ready
check("the soundfont and the synth load", fetched.some((f) => f.startsWith("200") && f.endsWith("gm.sf3")),
  fetched.filter((f) => /sf3|spessa/.test(f)).join(", "));
for (const [id, r] of Object.entries(results)) check(`${id} sounds`, r.peak > 0.01 && r.rmsDb > -60, `peak ${r.peak}, ${r.rmsDb} dB`);
if (results.bass && results.vocals) check("the bass sits lower than the melody", results.bass.centroidHz < results.vocals.centroidHz,
  `centroid bass ${results.bass.centroidHz} Hz, melody ${results.vocals.centroidHz} Hz`);
// channel 10 as a kit: the hi-hats and cymbals put energy above 4 kHz that the same
// notes (36, 42...) on a melodic instrument would not
if (results.drums && results.bass) check("the drums play the kit (energy above 4 kHz)",
  results.drums.highShare > 0.002 && results.drums.highShare > 10 * results.bass.highShare,
  `above 4 kHz: drums ${results.drums.highShare}, bass ${results.bass.highShare}`);
check("switching back to audio silences the synth", quiet < 0.005, `peak ${quiet} a second after the switch`);
check("no page errors", errors.length === 0, errors.join(" | "));
console.log(JSON.stringify({ url, state, results }, null, 1));
for (const c of checks) console.log(`${c.ok ? "ok  " : "FAIL"} ${c.name}${c.detail ? `: ${c.detail}` : ""}`);
process.exit(checks.every((c) => c.ok) ? 0 : 1);
