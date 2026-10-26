/* The lamp: draws a card in strings, on a canvas. Copied from the showcase
   (src/lamp/engine.ts), as plain JavaScript.
   A card is a bundle of strings with a rest form (the music as a musician knows it)
   and a taut form (what the tool makes of it). The lamp, a soft light under the
   pointer, pulls the strings it touches taut and shows the card's backdrop beneath
   it; a click (or Enter) plays the card, and everything the playhead passes turns
   taut, then relaxes. Text lives only in the lamp's label or rides with the playhead. */

import { cssColor } from "../../vendor/design/tokens.js";
import { behind, clamp, lampRadius, lampWeight, mixRgb, parseRgb, playState, rgba } from "./core.js";
import { audio } from "./sound.js";

const TOKENS = {
  ground: "--ground", ground2: "--ground-2", ink: "--ink", mut: "--ink-mut", line: "--line", acc: "--acc",
  kick: "--drum-kick", snare: "--drum-snare", hats: "--drum-hats", perc: "--drum-perc",
};

const POINTS = 160, LEVELS = 8, PAD = 22, RELAX_MS = 1200, IDLE_MS = 2500;

const motionQuery = matchMedia("(prefers-reduced-motion: reduce)");
/** Whether the reader asked for reduced motion: then cards draw still and taut. */
export const calm = () => motionQuery.matches;

const mono = (c, px) => { c.font = `${px}px "Geist Mono", monospace`; };

const mounted = new Set();
let running = false;

/** Mounts `def` in `host`. Options: label (read out for the canvas: what a click
    and Enter do), silent (the picture only), onPress (in place of playing), still. */
export function mountCard(host, def, opts) {
  host.classList.add("lamp-host");
  host.dataset.category = def.category;
  host.dataset.label = "";
  host.dataset.playing = "0";

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%;touch-action:pan-y;cursor:pointer";
  canvas.tabIndex = 0;
  canvas.setAttribute("role", opts.onPress ? "link" : "button");
  canvas.setAttribute("aria-label", opts.label);
  host.append(canvas);
  const off = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const octx = off.getContext("2d");
  if (!ctx || !octx) throw new Error("canvas 2D is not available");

  const s = {
    el: host, W: 0, H: 0, x: 0, y: 0, inside: false, lx: 0, ly: 0, sec: 0,
    p: -1, playing: false, sweep: -1, keep: 1,
    C: {},
    seed: mounted.size * 1.7,
    U: (u) => PAD + u * (s.W - 2 * PAD),
    V: (v) => PAD + v * (s.H - 2 * PAD),
    u: (x) => (x - PAD) / (s.W - 2 * PAD),
    wAt: (x, y) => (calm() ? 1 : lampWeight(x - s.lx, y - s.ly, m.R)),
    lit: (x, y, when) => Math.max(s.wAt(x, y), s.sweep >= 0 && when <= s.sweep ? s.keep : 0),
  };
  const m = {
    s, def, canvas, ctx, off, octx, R: 0, visible: true, last: -Infinity, playStart: -1,
    still: !!opts.still, silent: !!opts.silent, label: "", playing: false, ready: false,
  };
  readColours(m);

  const at = (e) => {
    const r = canvas.getBoundingClientRect();
    s.x = e.clientX - r.left;
    s.y = e.clientY - r.top;
    s.inside = true;
    m.last = performance.now();
  };
  canvas.addEventListener("pointermove", at);
  canvas.addEventListener("pointerdown", at);
  canvas.addEventListener("pointerleave", () => { s.inside = false; });
  canvas.addEventListener("pointercancel", () => { s.inside = false; });
  const press = () => { if (opts.onPress) opts.onPress(); else play(m); };
  canvas.addEventListener("click", press);
  // Enter only: Space stays the page's
  canvas.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    press();
  });

  const size = () => {
    const r = canvas.getBoundingClientRect(), d = devicePixelRatio || 1;
    if (!r.width || !r.height) return;
    for (const c of [canvas, off]) {
      c.width = Math.round(r.width * d);
      c.height = Math.round(r.height * d);
    }
    ctx.setTransform(d, 0, 0, d, 0, 0);
    octx.setTransform(d, 0, 0, d, 0, 0);
    const first = !s.W;
    s.W = r.width;
    s.H = r.height;
    m.R = m.still ? s.W * 0.3 : lampRadius(s.W);
    if (first) {
      readColours(m);
      [s.lx, s.ly] = wander(m, 0);
    }
  };
  const ro = new ResizeObserver(size);
  ro.observe(canvas);
  const io = new IntersectionObserver(([e]) => { m.visible = !!e?.isIntersecting; });
  io.observe(canvas);

  mounted.add(m);
  start();
  return {
    play: () => play(m),
    destroy() {
      mounted.delete(m);
      ro.disconnect();
      io.disconnect();
      canvas.remove();
    },
  };
}

function readColours(m) {
  for (const key of Object.keys(TOKENS)) m.s.C[key] = parseRgb(cssColor(TOKENS[key], m.s.el));
}

function start() {
  if (running) return;
  running = true;
  const recolour = () => { for (const m of mounted) readColours(m); };
  new MutationObserver(recolour).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", recolour);
  const frame = (ms) => {
    for (const m of mounted) if (m.visible) draw(m, ms);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function play(m) {
  if (m.still) return;
  m.def.onPlay?.(m.s);
  const a = m.silent ? null : audio();
  if (a && m.def.audio) {
    try {
      m.def.audio(a, a.currentTime + 0.06, m.s);
    } catch {
      // the sound is a bonus: the picture plays regardless
    }
  }
  m.playStart = performance.now() + 60;
}

function wander(m, sec) {
  const { s, def } = m;
  if (def.wander) return def.wander(s, sec);
  const t = 0.5 + 0.4 * Math.sin(sec * 0.3 + s.seed);
  return def.rest(s, Math.floor(def.strings / 2), t, sec);
}

const sweepX = (s, p) => s.U(p);

function draw(m, ms) {
  const { s, def, ctx } = m;
  const { W, H, C } = s;
  if (!W || !H) return;
  const quiet = calm();
  s.sec = m.still ? 1 : quiet ? 0 : ms / 1000;

  // where the lamp is: easing toward the pointer, or wandering when idle
  if (m.still) {
    const [u, v] = def.still ?? [0.5, 0.5];
    s.lx = u * W;
    s.ly = v * H;
  } else {
    const idle = !s.inside || ms - m.last > IDLE_MS;
    if (!(quiet && idle)) {
      const [tx, ty] = idle ? wander(m, ms / 1000) : [s.x, s.y];
      const k = idle ? 0.04 : 0.25;
      s.lx += (tx - s.lx) * k;
      s.ly += (ty - s.ly) * k;
    }
  }

  const ps = m.still || quiet ? playState(ms, -1, 1) : playState(ms, m.playStart, (def.dur ?? 3.2) * 1000, RELAX_MS);
  s.p = ps.p;
  s.playing = ps.playing;
  s.sweep = ps.sweep;
  s.keep = ps.keep;
  if (ps.playing !== m.playing) {
    m.playing = ps.playing;
    s.el.dataset.playing = ps.playing ? "1" : "0";
  }

  ctx.fillStyle = rgba(C.ground2, 1);
  ctx.fillRect(0, 0, W, H);
  def.surface?.(s, ctx, s.sec);
  light(m, quiet);

  // the strings: each in a few paths by how taut they are; the drawn point nearest
  // the lamp is kept, it is what the lamp's label names
  const mid = (def.strings - 1) / 2, N = def.points ?? POINTS;
  let nearest = null, nd = m.R * 0.6;
  for (let k = 0; k < def.strings; k++) {
    const paths = Array.from({ length: LEVELS + 1 }, () => new Path2D());
    let px = 0, py = 0;
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1);
      const [rx, ry] = def.rest(s, k, t, s.sec);
      const [tx, ty] = def.tight(s, k, t, s.sec);
      let w = s.wAt(rx, ry);
      if (def.lift) w = Math.max(w, def.lift(s, k, t));
      if (s.sweep >= 0) w = Math.max(w, behind(def.when ? def.when(k, t) : t, s.sweep, s.keep));
      const x = rx + (tx - rx) * w, y = ry + (ty - ry) * w;
      const d = Math.hypot(x - s.lx, y - s.ly);
      if (d < nd) { nd = d; nearest = [x, y]; }
      if (i > 0) {
        const lv = Math.round(clamp(def.intensity ? def.intensity(s, k, t, w) : w, 0, 1) * LEVELS);
        paths[lv].moveTo(px, py);
        paths[lv].lineTo(x, y);
      }
      px = x;
      py = y;
    }
    const col = C[def.colour?.(k) ?? "acc"];
    const base = def.base ? def.base(k) : Math.abs(k - mid) < 0.6 ? 0.7 : 0.3;
    for (let lv = 0; lv <= LEVELS; lv++) {
      const w = lv / LEVELS;
      ctx.strokeStyle = rgba(mixRgb(C.mut, col, w), base + (1 - base) * w * 0.9);
      ctx.lineWidth = 1 + w * (def.thick ?? 0.4);
      ctx.stroke(paths[lv]);
    }
  }
  def.over?.(s, ctx, s.sec);

  // the playhead and its label while playing; otherwise the lamp's label
  let label = "";
  if (s.playing) {
    ctx.strokeStyle = rgba(C.acc, 0.55);
    ctx.lineWidth = 1;
    const x = (def.sweepX ?? sweepX)(s, s.p);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
    const pl = def.playLabel?.(s, s.p);
    if (pl) tag(s, ctx, pl.text, pl.x, pl.y, C.acc);
  } else if (def.label && !m.still && (s.inside || !quiet)) {
    label = def.label(s, s.lx, s.ly) ?? "";
    if (label) {
      const box = tag(s, ctx, label, s.lx + m.R * 0.45, s.ly - m.R * 0.45);
      const at = nearest ? [...nearest, 4] : null;
      if (at) pointAt(s, ctx, at[0], at[1], at[2], box);
    }
  }
  if (label !== m.label) {
    m.label = label;
    s.el.dataset.label = label;
  }
  if (!m.ready) {
    m.ready = true;
    s.el.dataset.ready = "1";
  }
}

/* the backdrop shows through the lamp, and everywhere the playhead has passed;
   under reduced motion it shows whole */
function light(m, quiet) {
  const { s, def, ctx, off, octx } = m;
  const backdrop = def.backdrop;
  if (!backdrop) return;
  const { W, H } = s;
  const pass = (mask) => {
    octx.globalCompositeOperation = "source-over";
    octx.clearRect(0, 0, W, H);
    backdrop(s, octx);
    if (mask) {
      octx.globalCompositeOperation = "destination-in";
      mask();
    }
    ctx.drawImage(off, 0, 0, W, H);
  };
  if (quiet) {
    pass(null);
    return;
  }
  pass(() => {
    const g = octx.createRadialGradient(s.lx, s.ly, m.R * 0.4, s.lx, s.ly, m.R * 1.1);
    g.addColorStop(0, "rgba(0,0,0,1)");
    g.addColorStop(1, "rgba(0,0,0,0)");
    octx.fillStyle = g;
    octx.fillRect(0, 0, W, H);
  });
  if (s.sweep >= 0 && s.keep > 0) {
    pass(() => {
      octx.fillStyle = `rgba(0,0,0,${s.keep})`;
      octx.beginPath();
      octx.rect(0, 0, (def.sweepX ?? sweepX)(s, s.sweep), H);
      octx.fill();
    });
  }
}

/** A small label on a ground-coloured strip, kept inside the card. */
function tag(s, c, text, x, y, colour = s.C.ink) {
  mono(c, 10);
  const tw = c.measureText(text).width;
  const cx = clamp(x, 4, Math.max(4, s.W - tw - 8));
  const cy = clamp(y, 12, s.H - 6);
  c.fillStyle = rgba(s.C.ground2, 0.92);
  c.fillRect(cx - 3, cy - 10, tw + 6, 14);
  c.fillStyle = rgba(colour, 1);
  c.fillText(text, cx, cy);
  return [cx - 3, cy - 10, cx + tw + 3, cy + 4];
}

/** A small ring on (x, y), radius r, and a hairline from it to the label's box. */
function pointAt(s, c, x, y, r, [x0, y0, x1, y1]) {
  c.strokeStyle = rgba(s.C.ink, 0.85);
  c.lineWidth = 1;
  c.beginPath();
  c.arc(x, y, r, 0, Math.PI * 2);
  c.stroke();
  const bx = clamp(x, x0, x1), by = clamp(y, y0, y1), d = Math.hypot(bx - x, by - y);
  if (d < r + 6) return;
  c.strokeStyle = rgba(s.C.ink, 0.35);
  c.beginPath();
  c.moveTo(x + ((bx - x) / d) * r, y + ((by - y) / d) * r);
  c.lineTo(bx, by);
  c.stroke();
}
