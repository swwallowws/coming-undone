// ADT_STR drum transcription around two ONNX graphs (browser/models/export_adt.py).
// JS ports of the model repo's adt_transcriber.py (2.56 s chunks, greedy decode), its
// ComputeMelSpectrogram (torchaudio MelSpectrogram) and MidiTokenizer.decode, plus
// stemscribe's ADT_STR_TO_GM mapping (backends.py). Checked against PyTorch hit for hit
// in the spike (coming-undone-browser-spike, drums/).
//
// ADT_STR: Melucci, Merialdo, Akama 2026, https://github.com/pier-maker92/ADT_STR,
// code and weights CC BY-SA 4.0.

export const SR = 24000;
export const CHUNK = 61440;        // 2.56 s
const N_FFT = 2048;
const HOP = 240;                   // 10 ms
const N_MELS = 128;
const F_MIN = 20;
const PAD_IDX = Math.floor(N_FFT / 2 / HOP) + 1;
export const FRAMES = 1 + CHUNK / HOP - PAD_IDX - (PAD_IDX + 1);   // 246
const BOS = 2, EOS = 3, VOCAB = 1400;

// stemscribe backends.py ADT_STR_TO_GM: ADT_STR "GM custom" class -> standard GM.
export const ADT_TO_GM = { 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42, 43: 44,
  44: 46, 45: 47, 46: 49, 47: 50, 48: 51, 49: 52, 50: 54, 51: 55, 52: 56,
  53: 58, 54: 60, 55: 69, 56: 71, 57: 73, 58: 75, 59: 78, 60: 80 };

// torchaudio.functional.melscale_fbanks, htk scale, norm=None. [mel][bin].
export function melFilterbank(sr = SR, nFft = N_FFT, nMels = N_MELS, fMin = F_MIN, fMax = sr / 2) {
  const nFreqs = nFft / 2 + 1;
  const hz2mel = (f) => 2595 * Math.log10(1 + f / 700);
  const mel2hz = (m) => 700 * (10 ** (m / 2595) - 1);
  const allFreqs = Float64Array.from({ length: nFreqs }, (_, i) => (i * Math.floor(sr / 2)) / (nFreqs - 1));
  const mMin = hz2mel(fMin), mMax = hz2mel(fMax);
  const fPts = Float64Array.from({ length: nMels + 2 }, (_, i) => mel2hz(mMin + ((mMax - mMin) * i) / (nMels + 1)));
  const fb = [];
  for (let m = 0; m < nMels; m++) {
    const row = new Float32Array(nFreqs);
    const dl = fPts[m + 1] - fPts[m], du = fPts[m + 2] - fPts[m + 1];
    for (let k = 0; k < nFreqs; k++) {
      row[k] = Math.max(0, Math.min(-(fPts[m] - allFreqs[k]) / dl, (fPts[m + 2] - allFreqs[k]) / du));
    }
    fb.push(row);
  }
  return fb;
}

// In-place radix-2 complex FFT.
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

let FB = null, WIN = null;

// ComputeMelSpectrogram.forward for one 61440-sample chunk -> Float32Array(246 * 128).
export function logmel(chunk) {
  FB ||= melFilterbank();
  WIN ||= Float64Array.from({ length: N_FFT }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT));
  const pad = N_FFT / 2, n = chunk.length;
  const at = (i) => chunk[i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i];
  const out = new Float32Array(FRAMES * N_MELS);
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT), pow = new Float64Array(N_FFT / 2 + 1);
  for (let f = 0; f < FRAMES; f++) {
    const start = (f + PAD_IDX) * HOP - pad;
    for (let i = 0; i < N_FFT; i++) { re[i] = at(start + i) * WIN[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k <= N_FFT / 2; k++) pow[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < N_MELS; m++) {
      const row = FB[m];
      let s = 0;
      for (let k = 0; k < pow.length; k++) if (row[k]) s += row[k] * pow[k];
      const v = Math.min(12, Math.max(-23, Math.log(s + 1e-10)));
      out[f * N_MELS + m] = (v + 23) / 35;
    }
  }
  return out;
}

// MidiTokenizer.decode, quirks included (insertion order, zip truncation).
export function decodeTokens(tokens) {
  const onsets = new Map(), pitches = new Map();
  let vel = new Map();
  tokens.forEach((t, i) => {
    if (t === BOS || t === EOS) return;
    if (t >= 4 && t < 300) onsets.set(i, (t - 4) / 100);
    else if (t >= 300 && t < 400) { if (onsets.has(i - 1)) pitches.set(i - 1, t - 300); }
    else if (t >= 400) { if (onsets.has(i - 2)) vel.set(i - 2, t - 400); }
  });
  if (vel.size === 0) { vel = new Map(); for (let i = 0; i < onsets.size; i++) vel.set(i, 100); }
  const o = [...onsets.values()], p = [...pitches.values()], v = [...vel.values()];
  const k = Math.min(o.length, p.length, v.length);
  const notes = [];
  for (let i = 0; i < k; i++) notes.push([o[i], o[i] + 0.1, p[i], v[i]]);
  return notes;
}

// Hits {time, pitch (standard GM), velocity} of a mono 24 kHz signal, sorted.
export async function transcribeDrums(ort, enc, dec, wave, { batch = 8, maxLength = 512, onChunk = () => {} } = {}) {
  const chunks = [];
  for (let s = 0; s < wave.length; s += CHUNK) {
    const c = new Float32Array(CHUNK);
    c.set(wave.subarray(s, Math.min(s + CHUNK, wave.length)));
    chunks.push([s, c]);
  }
  const rows = [];
  for (let b0 = 0; b0 < chunks.length; b0 += batch) {
    const part = chunks.slice(b0, b0 + batch), B = part.length;
    const feats = new Float32Array(B * FRAMES * N_MELS);
    part.forEach(([, c], i) => feats.set(logmel(c), i * FRAMES * N_MELS));
    const cross = await enc.run({ logmel: new ort.Tensor("float32", feats, [B, FRAMES, N_MELS]) });
    const seqs = Array.from({ length: B }, () => [BOS]);
    const done = new Array(B).fill(false);
    for (let step = 0; step < maxLength - 1; step++) {
      const L = seqs[0].length, tok = new BigInt64Array(B * L);
      seqs.forEach((s, i) => s.forEach((v, j) => { tok[i * L + j] = BigInt(v); }));
      const { logits } = await dec.run({ tokens: new ort.Tensor("int64", tok, [B, L]), ...cross });
      const lg = logits.data;
      for (let i = 0; i < B; i++) {
        let best = 0, bv = -Infinity;
        for (let k = 0; k < VOCAB; k++) if (lg[i * VOCAB + k] > bv) { bv = lg[i * VOCAB + k]; best = k; }
        const next = done[i] ? EOS : best;
        seqs[i].push(next);
        if (next === EOS) done[i] = true;
      }
      if (done.every(Boolean)) break;
    }
    for (const v of Object.values(cross)) { try { v.dispose(); } catch (_) { /* cpu tensor */ } }
    part.forEach(([start], i) => {
      for (const n of decodeTokens(seqs[i])) rows.push([n[0] + start / SR, n[2], n[3]]);
    });
    onChunk(Math.min(b0 + batch, chunks.length), chunks.length);
  }
  const seen = new Set(), hits = [];
  for (const r of rows) {                        // np.unique over rows
    const key = r.join(",");
    if (seen.has(key)) continue;
    seen.add(key);
    hits.push({ time: r[0], pitch: ADT_TO_GM[r[1]] ?? r[1], velocity: Math.max(1, Math.min(127, r[2])) });
  }
  hits.sort((a, b) => a.time - b.time || a.pitch - b.pitch);
  return hits;
}
