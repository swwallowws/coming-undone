// Parts: one set of names, in one order and one colour each, for every place the page
// shows them (the lanes, the lamp, the download). Also the one MIDI file made from the
// parts that are switched on, and the status line in plain words.
//
// STEM_PART and PROGRAM mirror merge.TRACK_SPEC (stem -> track name, GM program);
// tests/test_parts_js.py checks they agree.

export const STEM_PART = {
  vocals: "melody",
  other: "comping",
  guitar: "guitar",
  piano: "piano",
  bass: "bass",
  drums: "drums",
};

// top to bottom, like a score: the tune, the harmony, the low end, the kit
export const PART_ORDER = ["melody", "guitar", "piano", "comping", "bass", "drums"];

// each part's colour token: the drum set's four, as the lamp colours them, and two
// mixes of them for 6 parts (defined in the page's :root)
export const PART_TOKEN = {
  melody: "--drum-snare",
  comping: "--drum-perc",
  bass: "--drum-hats",
  drums: "--drum-kick",
  guitar: "--part-guitar",
  piano: "--part-piano",
};

// the General MIDI program each part plays on (merge.TRACK_SPEC)
export const PROGRAM = {
  melody: 53,
  bass: 33,
  comping: 0,
  guitar: 25,
  piano: 0,
  drums: 0,
};
// a sung melody written for a voice patch plays as a flute here, as in Rearranged:
// the built-in voice sounds poorly
export const playProgram = name => (PROGRAM[name] === 53 ? 73 : PROGRAM[name] ?? 0);

const cap = s => s ? s[0].toUpperCase() + s.slice(1) : s;

export const partOf = stem => STEM_PART[stem] || stem;
export const partLabel = name => cap(name);
export const partToken = name => PART_TOKEN[name] || "--ink-mut";

// known parts in score order; anything unknown goes after them, still before the kit
export function partRank(name) {
  const i = PART_ORDER.indexOf(name);
  return i >= 0 ? i : PART_ORDER.indexOf("drums") - 0.5;
}

export const byPart = (key = x => x) => (a, b) => partRank(key(a)) - partRank(key(b));

// ---- what the download button says and saves

const base = p => String(p || "").split(/[\\/]/).pop();

/** The button's words for the parts that are on, out of all of them: it names what goes
 *  in the file, so switching a lane off visibly takes it out. */
export function downloadLabel(on, all) {
  if (!on.length) return "Switch a part on to download it";
  if (on.length === all.length) return "Download all parts as MIDI";
  if (on.length <= 3) {
    const names = on.length === 1 ? on[0] : `${on.slice(0, -1).join(", ")} and ${on[on.length - 1]}`;
    return `Download the ${names} as MIDI`;
  }
  return `Download ${on.length} parts as MIDI`;
}

/** The file's name: the song's own for every part, else the song and the parts. */
export function downloadName(midiPath, on, all) {
  const song = base(midiPath).replace(/\.midi?$/i, "") || "song";
  return on.length === all.length ? `${song}.mid` : `${song}-${on.join("-")}.mid`;
}

// ---- one Standard MIDI File from several parts' own files

const u32 = (b, i) => (b[i] << 24 | b[i + 1] << 16 | b[i + 2] << 8 | b[i + 3]) >>> 0;

/** A Standard MIDI File's division and its tracks' bytes. */
export function readTracks(buf) {
  const b = new Uint8Array(buf);
  if (b.length < 14 || u32(b, 0) !== 0x4d546864) throw new Error("not a MIDI file");
  const division = b[12] << 8 | b[13];
  const tracks = [];
  let at = 8 + u32(b, 4);
  while (at + 8 <= b.length) {
    const len = u32(b, at + 4);
    if (u32(b, at) === 0x4d54726b) tracks.push(b.slice(at + 8, at + 8 + len));
    at += 8 + len;
  }
  return { division, tracks };
}

// Walk a track's events: calls each(statusIndex, status) for every channel message, so
// a caller can read or rewrite its channel. Returns whether any note sounds.
function walk(t, each = () => {}) {
  let i = 0, status = 0, notes = false;
  const vlq = () => { let v = 0, c; do { c = t[i++]; v = (v << 7) | (c & 0x7f); } while (c & 0x80 && i < t.length); return v; };
  while (i < t.length) {
    vlq();
    let s = t[i];
    if (s & 0x80) { i++; } else s = status;                     // running status
    if (s === 0xff) { i++; const len = vlq(); i += len; continue; }
    if (s === 0xf0 || s === 0xf7) { const len = vlq(); i += len; continue; }
    status = s;
    if (t[i - 1] === s) each(i - 1, s);
    const kind = s & 0xf0;
    if (kind === 0x90 && t[i + 1] > 0) notes = true;
    i += kind === 0xc0 || kind === 0xd0 ? 1 : 2;
  }
  return notes;
}

/** One file from several parts' own files (each as the server, the Space and the browser
 *  engine write them: a conductor track, then the part's track). The first file's
 *  conductor keeps the tempo map, the meter and the bar lines; every part keeps its
 *  track, on a channel of its own (the kit stays on 10). */
export function mergeMidi(files) {
  const read = files.map(readTracks);
  const division = read[0].division;
  if (read.some(r => r.division !== division)) throw new Error("the parts' files count time differently");
  const conductor = read[0].tracks.filter(t => !walk(t));
  const parts = read.flatMap(r => r.tracks.filter(t => walk(t)));
  let next = 0;
  const out = parts.map(t => {
    const copy = t.slice();
    let drum = false;
    walk(copy, i => { if ((copy[i] & 0x0f) === 9) drum = true; });
    if (drum) return copy;
    if (next === 9) next++;
    const ch = next++ % 16;
    walk(copy, i => { copy[i] = (copy[i] & 0xf0) | ch; });
    return copy;
  });
  const tracks = [...conductor, ...out];
  const head = [0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 1, tracks.length >> 8, tracks.length & 255, division >> 8, division & 255];
  const size = head.length + tracks.reduce((n, t) => n + 8 + t.length, 0);
  const bytes = new Uint8Array(size);
  bytes.set(head, 0);
  let at = head.length;
  for (const t of tracks) {
    bytes.set([0x4d, 0x54, 0x72, 0x6b, t.length >>> 24 & 255, t.length >> 16 & 255, t.length >> 8 & 255, t.length & 255], at);
    bytes.set(t, at + 8);
    at += 8 + t.length;
  }
  return bytes;
}

// ---- save one file, only when asked

// A download attribute is ignored for a file on another origin (the Space, or the local
// server behind the public page), and the browser then opens the file in place of the
// page; saving from a blob keeps the page and the name.
export function saveBytes(bytes, filename, { doc = globalThis.document, urls = globalThis.URL } = {}) {
  const href = urls.createObjectURL(new Blob([bytes], { type: "audio/midi" }));
  const a = doc.createElement("a");
  a.href = href; a.download = filename;
  doc.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => urls.revokeObjectURL(href), 1000);
}

// ---- the run's status in plain words; null for a step not worth a line

export function statusLine({ stage, message = "" }) {
  const stem = (message.match(/(?:transcribing|cached) (\w+)/) || [])[1];
  switch (stage) {
    case "online": return message;
    case "upload": return "sending the file";
    case "queue": return message;
    case "gpu": return "taking the song apart on a free GPU";
    case "prepare": return "getting the audio ready";
    case "separate": return "taking the song apart";
    case "download": case "models": return "fetching the models, kept for next time";
    case "drums": return "writing down the drums";
    case "transcribe": return stem ? `writing down the ${partOf(stem)}` : "writing down the notes";
    case "tempo": return "finding the tempo";
    case "merge": case "grid": return "lining the notes up with the beat";
    default: return null;
  }
}
