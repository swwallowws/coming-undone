// Tempo for the browser engine, from the drum hits (ADT_STR) or, without drums, from
// the pitched onsets: an onset train at 100 Hz, its autocorrelation over 60 to 200 BPM
// with a log-normal preference around 120 BPM (as librosa's beat tracker weighs
// tempi), then the beat phase that best fits the hits. The server's grid stage (bar
// lines, meter, tempo maps) is not ported; this gives a tempo, alternates and a
// constant beat grid.

const RATE = 100;

export function estimateTempo(events, seconds, { min = 60, max = 200, prior = 120 } = {}) {
  const n = Math.max(1, Math.ceil(seconds * RATE) + 1);
  const env = new Float64Array(n);
  for (const e of events) {
    const i = Math.round(e.time * RATE);
    if (i >= 0 && i < n) env[i] += 0.3 + 0.7 * (e.weight ?? 1);
  }
  // smooth (gaussian, sigma 2 frames) so near-misses still correlate
  const k = [0.05, 0.24, 0.4, 0.24, 0.05];
  const s = new Float64Array(n);
  for (let i = 0; i < n; i++) for (let j = -2; j <= 2; j++) if (i + j >= 0 && i + j < n) s[i] += k[j + 2] * env[i + j];
  const mean = s.reduce((a, b) => a + b, 0) / n;
  for (let i = 0; i < n; i++) s[i] -= mean;
  // autocorrelation at whole-frame lags, interpolated between them: rounding the lag
  // would give a range of tempi the same score and let the prior pick among them
  const acInt = new Map();
  const acAt = (L) => {
    if (L <= 0 || L >= n) return 0;
    if (!acInt.has(L)) {
      let v = 0;
      for (let i = 0; i + L < n; i++) v += s[i] * s[i + L];
      acInt.set(L, v / (n - L));
    }
    return acInt.get(L);
  };
  const ac = (lag) => {
    const L = Math.floor(lag), f = lag - L;
    return (1 - f) * acAt(L) + f * acAt(L + 1);
  };
  const score = (bpm) => {
    const p = (60 / bpm) * RATE;
    const periodic = ac(p) + 0.5 * ac(2 * p) + 0.25 * ac(3 * p);
    const w = Math.exp(-0.5 * (Math.log2(bpm / prior) / 1.0) ** 2);
    return periodic * w;
  };
  let best = null;
  for (let bpm = min; bpm <= max; bpm += 0.25) {
    const v = score(bpm);
    if (!best || v > best.v) best = { bpm, v };
  }
  if (!best || !(best.v > 0) || events.length < 8) return null;
  // refine to 0.01 BPM around the peak
  for (let bpm = best.bpm - 0.25; bpm <= best.bpm + 0.25; bpm += 0.01) {
    const v = score(bpm);
    if (v > best.v) best = { bpm, v };
  }
  const bpm = Math.round(best.bpm * 100) / 100;
  const period = 60 / bpm;
  // phase: circular mean of the hits on the beat
  let cx = 0, cy = 0;
  for (const e of events) {
    const a = (2 * Math.PI * e.time) / period;
    const w = 0.3 + 0.7 * (e.weight ?? 1);
    cx += w * Math.cos(a); cy += w * Math.sin(a);
  }
  let beat0 = ((Math.atan2(cy, cx) / (2 * Math.PI)) * period + period) % period;
  const beats = [];
  for (let t = beat0; t <= seconds + period; t += period) beats.push(+t.toFixed(4));
  const candidates = [{ bpm, score: 1, ratio: 1 }];
  for (const ratio of [0.5, 2, 2 / 3, 1.5]) {
    const b = bpm * ratio;
    if (b < 40 || b > 300) continue;
    candidates.push({ bpm: Math.round(b * 100) / 100, score: +Math.max(0, score(b) / best.v).toFixed(4), ratio });
  }
  candidates.sort((a, b) => b.score - a.score);
  return { bpm, beats, beat0: +beat0.toFixed(4), candidates };
}
