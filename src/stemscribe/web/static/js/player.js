// The parts as sound: the shared General MIDI soundfont through spessasynth, each part's
// notes queued 200 ms ahead on the AudioContext clock, as on the /try/ page. A part that
// is switched off is not queued. Times are on the original file's timeline; play runs
// from `from` to `to` (a section's start and end).
import { fetchSoundfont } from "../vendor/design/sound/soundfont.js";
import { heardTime } from "../vendor/design/playhead.js";
import { playProgram } from "./parts.js";

const VENDOR = new URL("../vendor/design/sound/", import.meta.url);
const LOOKAHEAD = 0.2, LEVEL = 0.8;

export function notePlayer({ onEnd } = {}) {
  let ctx = null, out = null, synth = null, loading = null;
  let parts = [], channel = {}, from = 0, to = 0;
  let playing = false, pos = 0, t0 = 0, scheduledTo = 0, timer = 0, my = 0;

  const songTime = () => (playing ? Math.max(pos, pos + (ctx.currentTime - t0)) : pos);

  // one channel per part, skipping 10, which is the kit's
  function assign() {
    channel = {};
    let ch = 0;
    for (const p of parts) {
      if (p.drum) { channel[p.name] = 9; continue; }
      if (ch === 9) ch++;
      channel[p.name] = ch % 16;
      if (synth) synth.programChange(channel[p.name], playProgram(p.name));
      ch++;
    }
  }

  // the synth and the soundfont (4.6 MB, kept by the browser for every studio) load on
  // the first Play, made inside the click as browsers require
  function load() {
    if (!ctx) {
      ctx = new AudioContext();
      out = ctx.createGain();
      out.gain.value = LEVEL;
      out.connect(ctx.destination);
    }
    if (ctx.state === "suspended") ctx.resume();
    loading ||= (async () => {
      const lib = await import(new URL("spessasynth/spessasynth_lib.min.js", VENDOR).href);
      await ctx.audioWorklet.addModule(new URL("spessasynth/spessasynth_processor.min.js", VENDOR).href);
      const s = new lib.WorkletSynthesizer(ctx);
      s.connect(out);
      await s.soundBankManager.addSoundBank(await fetchSoundfont(new URL("gm.sf3", VENDOR)), "gm");
      await s.isReady;
      synth = s;
      assign();
    })();
    loading.catch(e => { console.warn("the player did not load", e); loading = null; });
    return loading;
  }

  // notes already queued cannot be taken back: fade out until they have passed, release
  // everything, and queue nothing new before then
  function hush() {
    if (!ctx || !synth) return 0;
    const now = ctx.currentTime, until = now + LOOKAHEAD + 0.05, g = out.gain;
    g.cancelScheduledValues(now);
    g.setTargetAtTime(0, now, 0.01);
    g.setValueAtTime(LEVEL, until);
    setTimeout(() => synth.stopAll(true), (until - now) * 1000 - 10);
    scheduledTo = songTime() + (until - now);
    return until;
  }

  function note(p, [pitch, s, e, vel], at) {
    const ch = channel[p.name];
    const v = Math.max(1, Math.min(127, Math.round(vel)));
    // a drum hit is an onset; the kit's samples ring on by themselves
    const len = p.drum ? 0.1 : Math.min(Math.max(e - s, 0.06), 8);
    synth.noteOn(ch, pitch, v, { time: at });
    synth.noteOff(ch, pitch, { time: at + len });
  }

  function schedule() {
    if (!playing) return;
    const now = songTime();
    if (now >= to) { stop(to); pos = from; onEnd?.(); return; }
    const a = Math.max(scheduledTo, now), b = now + LOOKAHEAD;
    for (const p of parts) {
      if (!p.on) continue;
      for (const n of p.notes) if (n[1] >= a && n[1] < b) note(p, n, t0 + (n[1] - pos));
    }
    scheduledTo = Math.max(scheduledTo, b);
  }

  function stop(at = songTime()) {
    my++;
    if (!playing) { pos = at; return; }
    pos = Math.max(from, Math.min(at, to));
    playing = false;
    clearInterval(timer);
    hush();
  }

  return {
    /** the parts: [{ name, drum, notes: [[pitch, start, end, velocity]], on }] */
    setParts(list) { parts = list; assign(); },
    setSpan(a, b) { from = a; to = b; pos = Math.max(a, Math.min(pos, b)); },
    get playing() { return playing; },
    /** where to draw the head: what is heard, less the output's delay */
    time() { return playing ? heardTime(songTime(), pos, ctx) : pos; },
    async play(at = pos) {
      const me = ++my;
      await load();
      if (me !== my) return false;                   // stopped while the sounds loaded
      pos = at >= to - 0.05 ? from : Math.max(from, at);
      t0 = ctx.currentTime + 0.05;
      scheduledTo = pos;
      playing = true;
      schedule();
      timer = setInterval(schedule, 50);
      return true;
    },
    stop,
    // playing, the new place sounds once what was queued has faded; the head waits there
    seek(t) {
      const at = Math.max(from, Math.min(t, to));
      if (!playing) { pos = at; return; }
      const until = hush();
      pos = at;
      t0 = until;
      scheduledTo = at;
    },
    /** a part was switched on or off: what is queued already fades, the rest follows */
    refresh() { if (playing) hush(); },
  };
}
