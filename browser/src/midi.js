// A Standard MIDI File writer (type 1, 480 ticks per quarter, constant tempo), for the
// browser engine's merged file and its per-part files. Same layout as stemscribe's
// merge: one named track per part with its GM program, drums on channel 10.

const PPQ = 480;

function vlq(n) {
  const out = [n & 0x7f];
  while ((n >>= 7)) out.unshift((n & 0x7f) | 0x80);
  return out;
}
const text = (s) => [...new TextEncoder().encode(s)];

function chunk(type, bytes) {
  const n = bytes.length;
  return [...text(type), (n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255, ...bytes];
}

function trackBytes(events) {
  events.sort((a, b) => a.tick - b.tick || a.order - b.order);
  const out = [];
  let last = 0;
  for (const e of events) {
    out.push(...vlq(e.tick - last), ...e.data);
    last = e.tick;
  }
  out.push(0x00, 0xff, 0x2f, 0x00);
  return out;
}

// tracks: [{ name, program, drum, notes: [{pitch, start, end, velocity}] }]; times in s.
export function writeMidi({ bpm, tracks, beatsPerBar = 4 }) {
  const toTick = (s) => Math.max(0, Math.round((s * bpm / 60) * PPQ));
  const usPerQ = Math.round(60000000 / bpm);
  const tempo = [
    { tick: 0, order: 0, data: [0xff, 0x51, 0x03, (usPerQ >> 16) & 255, (usPerQ >> 8) & 255, usPerQ & 255] },
    { tick: 0, order: 1, data: [0xff, 0x58, 0x04, beatsPerBar, 2, 24, 8] },
  ];
  const chunks = [chunk("MThd", [0, 1, 0, tracks.length + 1, (PPQ >> 8) & 255, PPQ & 255]), chunk("MTrk", trackBytes(tempo))];
  let channel = 0;
  for (const t of tracks) {
    let ch;
    if (t.drum) ch = 9;
    else { ch = channel++; if (ch === 9) ch = channel++; ch %= 16; }
    const name = text(t.name);
    const ev = [
      { tick: 0, order: 0, data: [0xff, 0x03, ...vlq(name.length), ...name] },
      { tick: 0, order: 1, data: [0xc0 | ch, t.program & 127] },
    ];
    for (const n of t.notes) {
      const on = toTick(n.start), off = Math.max(on + 1, toTick(n.end));
      const v = Math.max(1, Math.min(127, Math.round(n.velocity)));
      ev.push({ tick: on, order: 3, data: [0x90 | ch, n.pitch & 127, v] });
      ev.push({ tick: off, order: 2, data: [0x80 | ch, n.pitch & 127, 0] });   // offs before ons at a tick
    }
    chunks.push(chunk("MTrk", trackBytes(ev)));
  }
  return new Uint8Array(chunks.flat());
}

// Enough of a reader to check the writer and to read reference files (tests, verify):
// notes per track in seconds, through the file's whole tempo map.
export function readMidi(bytes) {
  let p = 0;
  const u32 = () => ((bytes[p++] << 24) | (bytes[p++] << 16) | (bytes[p++] << 8) | bytes[p++]) >>> 0;
  const u16 = () => (bytes[p++] << 8) | bytes[p++];
  const readVlq = () => { let n = 0, b; do { b = bytes[p++]; n = (n << 7) | (b & 0x7f); } while (b & 0x80); return n; };
  p = 8; u16(); const ntr = u16(), ppq = u16();
  // pass 1: raw events per track, tempo changes from any track
  const raw = [], tempos = [];
  for (let t = 0; t < ntr; t++) {
    p += 4;                                             // "MTrk"
    const len = u32(), stop = p + len;
    let tick = 0, status = 0;
    const tr = { name: "", program: 0, events: [] };
    while (p < stop) {
      tick += readVlq();
      if (bytes[p] & 0x80) status = bytes[p++];
      if (status === 0xff) {
        const type = bytes[p++], len2 = readVlq(), data = bytes.subarray(p, p + len2); p += len2;
        if (type === 0x51) tempos.push([tick, (data[0] << 16) | (data[1] << 8) | data[2]]);
        if (type === 0x03) tr.name = new TextDecoder().decode(data);
        if (type === 0x2f) break;
      } else if (status === 0xf0 || status === 0xf7) {
        p += readVlq();
      } else {
        const hi = status & 0xf0, a = bytes[p++], v = hi === 0xc0 || hi === 0xd0 ? 0 : bytes[p++];
        tr.events.push([tick, hi, a, v, status & 15]);
      }
    }
    p = stop;
    raw.push(tr);
  }
  tempos.sort((x, y) => x[0] - y[0]);
  if (!tempos.length || tempos[0][0] > 0) tempos.unshift([0, 500000]);
  const segs = [];                                      // [tick, seconds at tick, us per quarter]
  let sec = 0;
  tempos.forEach(([tk, us], i) => {
    if (i) sec += ((tk - segs[i - 1][0]) / ppq) * (segs[i - 1][2] / 1e6);
    segs.push([tk, sec, us]);
  });
  const toSec = (tk) => {
    let s = segs[0];
    for (const x of segs) if (x[0] <= tk) s = x; else break;
    return s[1] + ((tk - s[0]) / ppq) * (s[2] / 1e6);
  };
  const tracks = raw.map((tr) => {
    const out = { name: tr.name, program: tr.program, notes: [] }, open = new Map();
    for (const [tk, hi, a, v, ch] of tr.events) {
      if (hi === 0xc0) out.program = a;
      else if (hi === 0x90 && v > 0) open.set(`${ch},${a}`, { pitch: a, start: toSec(tk), velocity: v, channel: ch });
      else if (hi === 0x80 || (hi === 0x90 && v === 0)) {
        const o = open.get(`${ch},${a}`);
        if (o) { out.notes.push({ ...o, end: toSec(tk) }); open.delete(`${ch},${a}`); }
      }
    }
    return out;
  });
  return { ppq, bpm: 60e6 / segs[0][2], tempos: segs.map(([tk, s, us]) => ({ tick: tk, seconds: s, bpm: 60e6 / us })), tracks };
}
