// The page's part names, the download and the status line, without a browser:
// node tests/js/parts.test.mjs
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL("../../src/stemscribe/web/static/js/parts.js", import.meta.url));
const P = await import(here);
let n = 0;
const test = async (name, fn) => { await fn(); n++; };

await test("one order, top to bottom like a score, for every place a part shows", () => {
  assert.deepEqual(["drums", "bass", "comping", "melody"].sort(P.byPart()), ["melody", "comping", "bass", "drums"]);
  assert.deepEqual(["drums", "piano", "bass", "guitar", "comping", "melody"].sort(P.byPart()),
    ["melody", "guitar", "piano", "comping", "bass", "drums"]);
  assert.deepEqual(["drums", "synth", "melody"].sort(P.byPart()), ["melody", "synth", "drums"]);
});

await test("stems are named after their parts", () => {
  assert.equal(P.partOf("vocals"), "melody");
  assert.equal(P.partOf("other"), "comping");
  assert.equal(P.partOf("synth"), "synth");
  assert.equal(P.partLabel("comping"), "Comping");
});

await test("every part has its own colour; the four match the lamp's", () => {
  const four = ["melody", "comping", "bass", "drums"].map(P.partToken);
  assert.equal(new Set(four).size, 4);
  assert.deepEqual(four, ["--drum-snare", "--drum-perc", "--drum-hats", "--drum-kick"]);
  assert.equal(new Set(P.PART_ORDER.map(P.partToken)).size, P.PART_ORDER.length);
});

await test("the melody plays as a flute, the rest on their own programs", () => {
  assert.equal(P.playProgram("melody"), 73);
  assert.equal(P.playProgram("bass"), 33);
  assert.equal(P.playProgram("unknown"), 0);
});

await test("the download button says what it takes, and the file says it too", () => {
  const all = ["melody", "comping", "bass", "drums"];
  assert.equal(P.downloadLabel(all, all), "Download all parts as MIDI");
  assert.equal(P.downloadLabel(["bass"], all), "Download the bass as MIDI");
  assert.equal(P.downloadLabel(["melody", "bass"], all), "Download the melody and bass as MIDI");
  assert.equal(P.downloadLabel(["melody", "comping", "bass"], all), "Download the melody, comping and bass as MIDI");
  const six = ["melody", "guitar", "piano", "comping", "bass", "drums"];
  assert.equal(P.downloadLabel(six.slice(0, 4), six), "Download 4 parts as MIDI");
  assert.match(P.downloadLabel([], all), /Switch a part on/);
  assert.equal(P.downloadName("out/My Song.mid", all, all), "My Song.mid");
  assert.equal(P.downloadName("out/My Song.mid", ["melody", "bass"], all), "My Song-melody-bass.mid");
});

// a small Standard MIDI File: a conductor (a tempo), then one track per [channel, program, pitch]
const vlq = v => { const o = [v & 0x7f]; while ((v >>= 7)) o.unshift((v & 0x7f) | 0x80); return o; };
const chunk = (id, body) => [...id].map(c => c.charCodeAt(0)).concat([body.length >>> 24 & 255, body.length >> 16 & 255, body.length >> 8 & 255, body.length & 255], body);
const smf = (parts, division = 220) => {
  const conductor = [0, 0xff, 0x51, 3, 0x07, 0xa1, 0x20, 0, 0xff, 0x2f, 0];
  const tracks = parts.map(([ch, prog, pitch]) => [0, 0xc0 | ch, prog, 0, 0x90 | ch, pitch, 90, ...vlq(220), pitch, 0, 0, 0xff, 0x2f, 0]);
  return Uint8Array.from([...chunk("MThd", [0, 1, 0, tracks.length + 1, division >> 8, division & 255]),
    ...chunk("MTrk", conductor), ...tracks.flatMap(t => chunk("MTrk", t))]).buffer;
};
// the channels a track's messages use, read event by event (deltas can be several bytes)
const channels = t => {
  const c = new Set();
  let i = 0, status = 0;
  while (i < t.length) {
    while (t[i] & 0x80) i++;
    i++;                                                            // the delta's last byte
    if (t[i] & 0x80) status = t[i++];
    if (status === 0xff) { i++; const len = t[i++]; i += len; continue; }
    c.add(status & 0x0f);
    i += (status & 0xf0) === 0xc0 || (status & 0xf0) === 0xd0 ? 1 : 2;
  }
  return [...c];
};

await test("merging parts keeps one conductor and gives each part its own channel", () => {
  // three single-part files, each on channel 0 as the server writes them, and the kit on 10
  const merged = P.mergeMidi([smf([[0, 53, 67]]), smf([[0, 33, 40]]), smf([[9, 0, 36]])]);
  const { division, tracks } = P.readTracks(merged.buffer);
  assert.equal(division, 220);
  assert.equal(tracks.length, 4);                                   // conductor + three parts
  assert.equal(tracks[0][1], 0xff);                                 // the tempo leads
  assert.deepEqual(tracks.slice(1).map(channels), [[0], [1], [9]]);
});

await test("merging skips channel 10 for pitched parts", () => {
  const files = Array.from({ length: 11 }, (_, i) => smf([[0, 0, 60 + i]]));
  const { tracks } = P.readTracks(P.mergeMidi(files).buffer);
  const used = tracks.slice(1).map(t => channels(t)[0]);
  assert.ok(!used.includes(9));
  assert.equal(used[9], 10);
});

await test("merging refuses files that count time differently", () => {
  assert.throws(() => P.mergeMidi([smf([[0, 0, 60]], 220), smf([[0, 0, 62]], 480)]), /count time/);
});

await test("the status line speaks plainly and names parts as the lanes do", () => {
  assert.equal(P.statusLine({ stage: "transcribe", message: "transcribing vocals stem with muscriptor ..." }), "writing down the melody");
  assert.equal(P.statusLine({ stage: "transcribe", message: "cached other transcription (basic-pitch)" }), "writing down the comping");
  assert.equal(P.statusLine({ stage: "separate", message: "separating with htdemucs ..." }), "separating the parts");
  assert.equal(P.statusLine({ stage: "gpu", message: "" }), "separating the parts on a free GPU");
  assert.equal(P.statusLine({ stage: "grid", message: "tempo map, median 150.000 BPM" }), "lining the notes up with the beat");
  assert.equal(P.statusLine({ stage: "instrumental", message: "mixing instrumental ..." }), null);
  assert.equal(P.statusLine({ stage: "queue", message: "waiting in line: 2 ahead" }), "waiting in line: 2 ahead");
});

await test("saveBytes saves under the given name from a blob", () => {
  const clicked = [];
  const doc = { body: { appendChild() {} }, createElement: () => { const a = { click() { clicked.push(a.download); }, remove() {} }; return a; } };
  P.saveBytes(new Uint8Array([1]), "x.mid", { doc, urls: { createObjectURL: () => "blob:x", revokeObjectURL() {} } });
  assert.deepEqual(clicked, ["x.mid"]);
});

console.log(`ok ${n} parts tests`);
