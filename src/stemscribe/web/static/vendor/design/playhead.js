/** Playhead: click or drag on a roll, a waveform or the playhead line itself to
 *  move the playhead, with a mouse or a finger.
 *
 *  seekable(el, { duration, onScrub, onSeek }) listens on `el` (the roll's box;
 *  the playhead line inside it is part of it). A press puts the head under the
 *  pointer and a drag carries it along: each move calls onScrub(t). Letting go
 *  calls onSeek(t) once. So a player that is playing can show the head following
 *  the pointer in onScrub and move the sound only in onSeek (a seek on every move
 *  would restart the notes over and over); a paused player can move its place in
 *  onScrub already, and Play starts from there.
 *
 *  With `touch-action: pan-y` (playhead.css) a sideways drag moves the head and
 *  an upward one still scrolls the page; when the browser takes a touch over for
 *  scrolling, onCancel() runs instead of onSeek. Pointer only: keep a slider or
 *  keys for the keyboard. Styles are in playhead.css. */

/** Clamp `t` into [0, duration]. A duration that is not a positive number gives 0. */
export function clampTime(t, duration) {
  if (!(duration > 0) || !(t > 0)) return 0;
  return Math.min(duration, t);
}

/** The time under `clientX` in a box that starts at `left` and is `width` wide,
 *  spanning 0 to `duration`: the left edge is 0, the right edge is the end, and
 *  anything past either edge is clamped to it. */
export function timeAt(clientX, left, width, duration) {
  if (!(width > 0)) return 0;
  return clampTime(((clientX - left) / width) * duration, duration);
}

const read = (v) => (typeof v === "function" ? v() : v);

/** Make `el` seekable. Options:
 *  - duration: seconds (or a function giving them, read on each press), mapped
 *    across the element's width; or
 *  - toTime(clientX, rect): your own mapping, for a zoomed or scrolled view.
 *  - onScrub(t): the head is at t while pressed or dragged (also on the press).
 *  - onSeek(t): the pointer was let go at t. Called once per gesture.
 *  - onCancel(): the gesture was taken over (a scroll) or cancelled.
 *  - enabled(): false ignores presses (nothing loaded yet). Default: always.
 *  Returns a function that removes the listeners. */
export function seekable(el, opts) {
  const map = opts.toTime
    ?? ((x, rect) => timeAt(x, rect.left, rect.width, read(opts.duration)));
  let active = null; // pointerId of the gesture under way
  let at = 0;

  const timeOf = (e) => map(e.clientX, el.getBoundingClientRect());
  const finish = () => {
    active = null;
    el.classList.remove("seeking");
  };

  const down = (e) => {
    if (active !== null || e.button > 0) return;
    if (opts.enabled && !opts.enabled()) return;
    if (!(el.getBoundingClientRect().width > 0)) return;
    if (e.pointerType === "mouse") e.preventDefault(); // no text selection while dragging
    active = e.pointerId;
    try { el.setPointerCapture(e.pointerId); } catch { /* a synthetic event has no pointer to capture */ }
    el.classList.add("seeking");
    at = timeOf(e);
    opts.onScrub?.(at);
  };
  const move = (e) => {
    if (e.pointerId !== active) return;
    at = timeOf(e);
    opts.onScrub?.(at);
  };
  const up = (e) => {
    if (e.pointerId !== active) return;
    at = timeOf(e);
    finish();
    opts.onSeek?.(at);
  };
  const cancel = (e) => {
    if (e.pointerId !== active) return;
    finish();
    opts.onCancel?.();
  };

  el.classList.add("seekable");
  el.addEventListener("pointerdown", down);
  el.addEventListener("pointermove", move);
  el.addEventListener("pointerup", up);
  el.addEventListener("pointercancel", cancel);
  el.addEventListener("lostpointercapture", cancel);

  return () => {
    el.removeEventListener("pointerdown", down);
    el.removeEventListener("pointermove", move);
    el.removeEventListener("pointerup", up);
    el.removeEventListener("pointercancel", cancel);
    el.removeEventListener("lostpointercapture", cancel);
    el.classList.remove("seekable", "seeking");
    active = null;
  };
}
