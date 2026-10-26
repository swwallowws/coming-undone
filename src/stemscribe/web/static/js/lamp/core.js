/* Pure helpers for the lamp. Copied from the showcase (src/lamp/core.ts), as plain
   JavaScript; keep the two in step until the lamp moves into the design system. */

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Smoothstep on 0..1: eases in and out, flat at both ends. */
export const smooth = (x) => x * x * (3 - 2 * x);

/** A colour as the browser reports it ("rgb(r, g, b)" or "rgba(...)"). */
export function parseRgb(css) {
  const m = css.match(/[\d.]+/g) ?? [];
  return [Number(m[0] ?? 0), Number(m[1] ?? 0), Number(m[2] ?? 0)];
}

export const mixRgb = (a, b, w) => [
  Math.round(a[0] + (b[0] - a[0]) * w),
  Math.round(a[1] + (b[1] - a[1]) * w),
  Math.round(a[2] + (b[2] - a[2]) * w),
];

export const rgba = (c, alpha) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`;

export const hz = (midi) => 440 * 2 ** ((midi - 69) / 12);

/** The lamp's radius on a card this wide: 180 px, at most a third of the card. */
export const lampRadius = (width) => Math.min(180, width * 0.34);
/** How strongly the lamp, centred `dx, dy` away, pulls a point taut: 1 at its centre, 0 at its edge. */
export const lampWeight = (dx, dy, r) => smooth(clamp(1 - Math.hypot(dx, dy) / r, 0, 1));

/** Where a click's play is at time `ms`. `start` is when it began (-1: never).
    p: progress 0..1 while playing; sweep: how far the reveal reaches (-1 none);
    keep: how firmly the reveal holds, easing from 1 to 0 over `relaxMs` after the end. */
export function playState(ms, start, durMs, relaxMs = 1200) {
  if (start < 0) return { p: -1, playing: false, sweep: -1, keep: 1 };
  const p = (ms - start) / durMs;
  const end = start + durMs;
  const relax = ms > end ? clamp((ms - end) / relaxMs, 0, 1) : 0;
  const sweep = ms >= start && relax < 1 ? clamp(p, 0, 1) : -1;
  return { p, playing: p >= 0 && p <= 1, sweep, keep: 1 - smooth(relax) };
}

/** How taut the playhead holds a point it reaches at `when`: 0 ahead of it, easing to `keep` just behind it. */
export const behind = (when, sweep, keep) =>
  sweep >= 0 && when <= sweep ? smooth(clamp((sweep - when) / 0.04, 0, 1)) * keep : 0;
