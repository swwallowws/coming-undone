/* Sound for the lamp, synthesised in the page: no files to fetch. Copied from the
   showcase (src/lamp/sound.ts). Audio starts only from a click or a key; where the
   browser has no Web Audio, audio() returns null and the lamp plays silently. */

import { hz } from "./core.js";

let ctx = null;
let noise = null;
let broken = false;

export function audio() {
  if (broken) return null;
  try {
    ctx ??= new AudioContext();
    if (ctx.state === "suspended") void ctx.resume().catch(() => {});
    if (!noise) {
      noise = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
      const d = noise.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    }
    return ctx;
  } catch {
    broken = true;
    return null;
  }
}

/** A note: `midi` at `at` seconds (the context's clock) for `dur` seconds. */
export function tone(a, midi, at, dur, gain = 0.1, type = "triangle") {
  const o = a.createOscillator(), g = a.createGain();
  o.type = type;
  o.frequency.setValueAtTime(hz(midi), at);
  g.gain.setValueAtTime(0, at);
  g.gain.linearRampToValueAtTime(gain, at + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, at + dur);
  o.connect(g).connect(a.destination);
  o.start(at);
  o.stop(at + dur + 0.05);
}

/** A drum hit: a falling sine for the kick, filtered noise for the rest. */
export function hit(a, kind, at, gain = 1) {
  const g = a.createGain();
  if (kind === "kick") {
    const o = a.createOscillator();
    o.frequency.setValueAtTime(130, at);
    o.frequency.exponentialRampToValueAtTime(42, at + 0.14);
    g.gain.setValueAtTime(0.5 * gain, at);
    g.gain.exponentialRampToValueAtTime(0.001, at + 0.3);
    o.connect(g).connect(a.destination);
    o.start(at);
    o.stop(at + 0.32);
    return;
  }
  if (!noise) return;
  const n = a.createBufferSource(), f = a.createBiquadFilter();
  n.buffer = noise;
  f.type = kind === "hats" ? "highpass" : "bandpass";
  f.frequency.value = kind === "hats" ? 7000 : kind === "snare" ? 1800 : 900;
  const len = kind === "hats" ? 0.05 : kind === "snare" ? 0.18 : 0.1;
  g.gain.setValueAtTime((kind === "hats" ? 0.12 : 0.3) * gain, at);
  g.gain.exponentialRampToValueAtTime(0.001, at + len);
  n.connect(f).connect(g).connect(a.destination);
  n.start(at);
  n.stop(at + len + 0.02);
}
