/* Coming Undone's lamp card: one song parts into its instruments, each written as
   notes. Copied from the showcase (src/lamp/cards/coming-undone.ts), with the parts
   named as the studio names them (js/parts.js) and in the same colours. */

import { hit, tone } from "./sound.js";

const PARTS = ["melody", "comping", "bass", "drums"];
const COL = ["snare", "perc", "hats", "kick"];
const LANE = [0.1, 0.36, 0.62, 0.88];
const VOC = [0, 2, 4, 2, 5, 4, 2, 0], CH = [0, 3, 1, 4], BASS = [0, 0, 3, 3, 5, 5, 3, 3];
const DR = new Set([0, 3, 4, 6, 8, 11, 12, 14]);
const nearestPart = (s, y) =>
  LANE.reduce((b, v, i) => (Math.abs(s.V(v) - y) < Math.abs(s.V(LANE[b]) - y) ? i : b), 0);
const rgba = (c, a) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;

export const comingUndone = {
  id: "stemscribe",
  category: "transcribe",
  still: [0.62, 0.5],
  strings: 12,
  colour: (k) => COL[k % 4],
  base: () => 0.3,
  rest: (s, k, t, sec) => {
    const env = 0.55 + 0.45 * Math.sin(Math.PI * t);
    const w = 0.5 * Math.sin(t * 23 + sec * 0.5) + 0.3 * Math.sin(t * 57 + 1) + 0.2 * Math.sin(t * 131 + 2);
    return [s.U(t), s.V(0.5 + 0.13 * w * env) + (k - 5.5) * 1.6 * env];
  },
  tight: (s, k, t) => {
    const part = k % 4, j = Math.floor(k / 4), lane = LANE[part];
    let v;
    if (part === 0) v = s.V(lane + 0.05 - VOC[Math.min(7, Math.floor(t * 8))] * 0.014) + (j - 1) * 1.2;
    else if (part === 1) v = s.V(lane + 0.05 - CH[Math.min(3, Math.floor(t * 4))] * 0.008 - j * 0.03);
    else if (part === 2) v = s.V(lane + 0.05 - BASS[Math.min(7, Math.floor(t * 8))] * 0.01) + (j - 1) * 1.2;
    else {
      const st = t * 16, near = st - Math.round(st), on = DR.has(Math.round(st) % 16);
      v = s.V(lane + 0.04) - (on ? Math.exp(-near * near * 60) : 0) * s.H * 0.08 + (j - 1) * 1.2;
    }
    return [s.U(t), v];
  },
  backdrop: (s, o) => {
    o.strokeStyle = rgba(s.C.line, 1);
    o.lineWidth = 1;
    LANE.forEach((v) => {
      o.beginPath();
      o.moveTo(0, s.V(v + 0.1));
      o.lineTo(s.W, s.V(v + 0.1));
      o.stroke();
    });
  },
  label: (s, _x, y) => `${PARTS[nearestPart(s, y)]} · written as notes`,
  playLabel: (s, p) => ({ text: "one song · four parts", x: s.U(p) + 8, y: s.V(0) }),
  audio: (a, at) => {
    const step = 3.2 / 8;
    [67, 69, 71, 69, 72, 71, 69, 67].forEach((m, i) => tone(a, m, at + i * step, step * 0.9, 0.07));
    [43, 43, 48, 48, 50, 50, 48, 48].forEach((m, i) => tone(a, m, at + i * step, step * 0.9, 0.1, "sine"));
    [[60, 64, 67], [57, 60, 64], [62, 65, 69], [59, 62, 67]].forEach((c, i) => c.forEach((m) => tone(a, m, at + (i * 3.2) / 4, 0.7, 0.03, "sine")));
    for (let i = 0; i < 16; i++) {
      const when = at + i * 0.2;
      if (i % 8 === 0 || i === 11) hit(a, "kick", when);
      if (i % 8 === 4) hit(a, "snare", when);
      if (i % 2 === 0) hit(a, "hats", when);
    }
  },
};
