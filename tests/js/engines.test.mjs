// The page's engine layer, without a browser or network: node tests/js/engines.test.mjs
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL("../../src/stemscribe/web/static/js/engines.js", import.meta.url));
const E = await import(here);
let n = 0;
const test = async (name, fn) => { await fn(); n++; };

const result = {
  tempo: { bpm: 150 }, grid: { fitted: true }, prepared: { offset: 30 },
  midi: "song.mid", instrumental: "song_instrumental.mp3", manifest: "manifest.json",
  stems: { bass: "bass.mp3", drums: "drums.mp3" },
  parts: { end: 45, parts: [
    { name: "bass", file: "bass.mid", drum: false, notes: [[40, 31, 31.5, 80]] },
    { name: "drums", file: "drums.mid", drum: true, notes: [[36, 30, 30.1, 100]] },
    { name: "melody", file: "melody.mid", drum: false, notes: [] },
  ] },
};
const fd = (name, dir = "/tmp/gradio/abc") => ({ url: `https://space.hf.space/gradio_api/file=${dir}/${name}`, orig_name: name, path: `${dir}/${name}` });
const data = [JSON.stringify(result), fd("song.mid"),
  [fd("bass.mid", "/tmp/p"), fd("drums.mid", "/tmp/p"), fd("melody.mid", "/tmp/p")],
  [fd("bass.mp3"), fd("drums.mp3"), fd("song_instrumental.mp3"), fd("manifest.json")]];

await test("mapOnline finds every file the result names", () => {
  const run = E.mapOnline(data);
  assert.equal(run.fileUrl("song.mid"), data[1].url);
  assert.equal(run.fileUrl(result.instrumental), data[3][2].url);
  assert.equal(run.fileUrl(result.stems.bass), data[3][0].url);
  assert.equal(run.fileUrl("manifest.json"), data[3][3].url);
  assert.equal(run.partUrl("bass"), data[2][0].url);
  assert.notEqual(run.partUrl("bass"), run.fileUrl("bass.mp3"));   // parts and stems stay apart
  assert.equal(run.fileUrl(null), null);
  assert.equal(run.partUrl("vocals"), null);
});

await test("mapOnline's roll is the pitched parts with notes", () => {
  const run = E.mapOnline(data);
  assert.deepEqual(run.roll.tracks.map(t => t.name), ["bass"]);
  assert.equal(run.roll.end, 45);
  assert.equal(run.parts.parts.length, 3);
  assert.equal(run.restamp, null);
  assert.equal(run.shiftBar, null);
});

await test("quota messages become the quota kind, with the wait", () => {
  for (const m of [
    "You have exceeded your GPU quota (60s requested vs. 12s left). Try again in 3:12:05",
    "You have exceeded your free GPU quota. Sign up to get more quota.",
    "ZeroGPU quota exceeded",
  ]) assert.equal(E.classifyOnlineError(m).kind, "quota", m);
  assert.equal(E.classifyOnlineError("GPU quota exceeded. Try again in 3:12:05").detail, "3:12:05");
  const other = E.classifyOnlineError("ValueError: bad audio");
  assert.equal(other.kind, "failed");
  assert.equal(other.message, "ValueError: bad audio");
});

await test("online options drop the link format and keep types", () => {
  const o = E.onlineOptions({ audio_format: "native", tempo: null, downbeat: 2, snap: true, mono_stems: ["bass"], start: "" });
  assert.equal("audio_format" in o, false);
  assert.equal(o.stems_audio, true);
  assert.equal(o.downbeat, 2);
  assert.equal(o.start, null);
  assert.deepEqual(o.mono_stems, ["bass"]);
});

await test("online runs refuse a link before loading anything", async () => {
  let loaded = false;
  const eng = E.onlineEngine("x/y", async () => { loaded = true; return {}; });
  await assert.rejects(eng.run({ url: "https://a.b/c" }, {}), e => e.kind === "input");
  assert.equal(loaded, false);
});

await test("online run maps statuses and the quota error", async () => {
  const events = [];
  const fakeClient = (msgs) => ({
    Client: { connect: async () => ({ submit: () => (async function* () { yield* msgs; })() }) },
    handle_file: f => f,
  });
  const ok = E.onlineEngine("x/y", async () => fakeClient([
    { type: "status", stage: "pending", position: 2 },
    { type: "status", stage: "pending", progress_data: [{ desc: "waiting for a GPU" }] },
    { type: "data", data },
    { type: "status", stage: "complete" },
  ]));
  const run = await ok.run({ file: { name: "a.mp3" } }, {}, e => events.push(e));
  assert.equal(run.res.midi, "song.mid");
  assert.ok(events.some(e => e.stage === "queue" && /2 ahead/.test(e.message)));
  assert.ok(events.some(e => e.stage === "gpu"));
  const out = E.onlineEngine("x/y", async () => fakeClient([
    { type: "status", stage: "error", message: "You have exceeded your GPU quota. Try again in 1:00:00" },
  ]));
  await assert.rejects(out.run({ file: { name: "a.mp3" } }, {}), e => e.kind === "quota" && e.detail === "1:00:00");
  const down = E.onlineEngine("x/y", async () => ({ Client: { connect: async () => { throw new Error("503"); } }, handle_file: f => f }));
  await assert.rejects(down.run({ file: { name: "a.mp3" } }, {}), e => e.kind === "unreachable");
});

await test("local form data skips empty options and joins lists", () => {
  const f = E.localFormData({ file: new Blob(["x"]) }, { backend: "basic-pitch", tempo: null, start: "", mono_stems: ["bass", "vocals"], snap: false, downbeat: 2 });
  assert.equal(f.get("backend"), "basic-pitch");
  assert.equal(f.has("tempo"), false);
  assert.equal(f.has("start"), false);
  assert.equal(f.get("mono_stems"), "bass,vocals");
  assert.equal(f.get("snap"), "false");
  assert.equal(f.get("downbeat"), "2");
  assert.ok(f.has("file"));
});

await test("findLocal takes the first server that answers, and gives up quietly", async () => {
  const fetchFn = async (u) => {
    if (u.startsWith("http://gone")) throw new TypeError("failed to fetch");
    if (u.startsWith("http://html")) return { ok: true, json: async () => { throw new SyntaxError("<"); } };
    return { ok: true, json: async () => ({ version: "0.1.0", backends: [{ name: "basic-pitch" }] }) };
  };
  assert.equal((await E.findLocal(["http://gone", "http://html", "http://localhost:8002"], 50, fetchFn)).base, "http://localhost:8002");
  assert.equal(await E.findLocal(["http://gone"], 50, fetchFn), null);
});

console.log(`ok ${n} engine tests`);
