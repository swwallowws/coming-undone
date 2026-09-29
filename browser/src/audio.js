// Audio in and out for the browser engine: decode, the prepare step (section and
// silence trim, as stemscribe's prepare.py does it), resampling, WAV encoding.

export const SR = 44100;

export async function decodeFile(file, sr = SR) {
  const ctx = new OfflineAudioContext(2, 1, sr);
  const buf = await ctx.decodeAudioData(await file.arrayBuffer());
  const left = buf.getChannelData(0);
  const right = buf.numberOfChannels > 1 ? buf.getChannelData(1) : left;
  return { left, right, sr, duration: buf.duration };
}

// librosa.effects.trim(mono, top_db) with its defaults (frame 2048, hop 512, RMS on
// zero-padded centered frames, dB against the loudest frame): [i0, i1) to keep.
export function trimBounds(mono, topDb = 50, frame = 2048, hop = 512) {
  const n = Math.floor(mono.length / hop) + 1;
  const ms = new Float64Array(n);
  let peak = 0;
  for (let f = 0; f < n; f++) {
    const c = f * hop - frame / 2;
    let s = 0;
    for (let i = 0; i < frame; i++) {
      const j = c + i;
      if (j >= 0 && j < mono.length) s += mono[j] * mono[j];
    }
    ms[f] = s / frame;
    peak = Math.max(peak, ms[f]);
  }
  if (peak <= 0) return [0, mono.length];
  const thr = peak * 10 ** (-topDb / 10);
  let a = -1, b = -1;
  for (let f = 0; f < n; f++) if (ms[f] > thr) { if (a < 0) a = f; b = f; }
  if (a < 0) return [0, mono.length];
  return [a * hop, Math.min(mono.length, (b + 1) * hop)];
}

export function mono(left, right) {
  const m = new Float32Array(left.length);
  for (let i = 0; i < m.length; i++) m[i] = 0.5 * (left[i] + right[i]);
  return m;
}

// The section, then the silence trim. offset: seconds from the original file's t=0 to
// the prepared audio's t=0, added back to every note at the end.
export function prepare(audio, { start = null, duration = null, trimSilence = true } = {}) {
  const sr = audio.sr;
  const a = Math.max(0, Math.round((start || 0) * sr));
  const b = duration ? Math.min(audio.left.length, a + Math.round(duration * sr)) : audio.left.length;
  let left = audio.left.subarray(a, b), right = audio.right.subarray(a, b);
  const applied = [`decoded to ${sr}Hz/2ch in the browser`];
  if (start || duration) applied.push(`section start=${start || 0}s duration=${duration || "end"}`);
  let head = 0, tail = 0;
  if (trimSilence) {
    const [i0, i1] = trimBounds(mono(left, right));
    if (i1 > i0 && (i0 > 0 || i1 < left.length)) {
      head = i0 / sr; tail = (left.length - i1) / sr;
      left = left.subarray(i0, i1); right = right.subarray(i0, i1);
      applied.push(`trimmed silence head=${head.toFixed(2)}s tail=${tail.toFixed(2)}s`);
    }
  }
  return {
    left, right, sr,
    info: {
      offset: +((start || 0) + head).toFixed(4),
      original_duration: +audio.duration.toFixed(3),
      duration: +(left.length / sr).toFixed(3),
      trimmed_head: +head.toFixed(4),
      trimmed_tail: +tail.toFixed(4),
      applied,
    },
  };
}

// torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width 6, rolloff
// 0.99): ADT_STR's cymbal classes flip on the browser resampler's small differences,
// so the drums path uses this exact port.
export function resampleTA(x, from, to) {
  const gcd = (p, q) => (q ? gcd(q, p % q) : p);
  const g = gcd(from, to);
  const orig = from / g, nw = to / g, lw = 6;
  const base = Math.min(orig, nw) * 0.99;
  const width = Math.ceil((lw * orig) / base);
  const K = 2 * width + orig;
  const kern = new Float32Array(nw * K);
  for (let p = 0; p < nw; p++) {
    for (let k = 0; k < K; k++) {
      let t = (-p / nw + (k - width) / orig) * base;
      t = Math.max(-lw, Math.min(lw, t));
      const win = Math.cos((t * Math.PI) / lw / 2) ** 2;
      const tp = t * Math.PI;
      kern[p * K + k] = (tp === 0 ? 1 : Math.sin(tp) / tp) * win * (base / orig);
    }
  }
  const n = x.length;
  const padded = new Float32Array(n + 2 * width + orig);
  padded.set(x, width);
  const outLen = Math.ceil((nw * n) / orig);
  const y = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const f = Math.floor(i / nw), p = i % nw, off = f * orig, kp = p * K;
    let s = 0;
    for (let k = 0; k < K; k++) s += kern[kp + k] * padded[off + k];
    y[i] = s;
  }
  return y;
}

// The browser's resampler, for the parts that do not need torchaudio's exact filter.
export async function resampleOAC(x, from, to) {
  const src = new AudioBuffer({ numberOfChannels: 1, length: x.length, sampleRate: from });
  src.copyToChannel(x, 0);
  const ctx = new OfflineAudioContext(1, Math.ceil((x.length * to) / from), to);
  const node = ctx.createBufferSource();
  node.buffer = src;
  node.connect(ctx.destination);
  node.start();
  return ctx.startRendering();
}

export function rms(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / Math.max(1, x.length));
}

// 16-bit stereo WAV.
export function wavBlob(left, right, sr) {
  const n = left.length, bytes = 44 + n * 4;
  const buf = new ArrayBuffer(bytes), v = new DataView(buf);
  const s = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  s(0, "RIFF"); v.setUint32(4, bytes - 8, true); s(8, "WAVE");
  s(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true);
  s(36, "data"); v.setUint32(40, n * 4, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    for (const ch of [left, right]) {
      const x = Math.max(-1, Math.min(1, ch[i]));
      v.setInt16(o, x < 0 ? x * 0x8000 : x * 0x7fff, true);
      o += 2;
    }
  }
  return new Blob([buf], { type: "audio/wav" });
}
