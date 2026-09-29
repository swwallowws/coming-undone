// Port of stemscribe/cleanup.py's clean_instrument: the same steps, order, defaults and
// before/after stats, so the Tracks table reads the same whichever engine ran.
// Notes are {pitch, start, end, velocity}; returns { notes, stats }.

const median = (xs) => {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const r = (x, d) => Math.round(x * 10 ** d) / 10 ** d;
const medianDuration = (ns) => (ns.length ? r(median(ns.map((n) => n.end - n.start)), 4) : 0);

function polyphony(ns) {
  if (!ns.length) return 0;
  const span = Math.max(...ns.map((n) => n.end)) - Math.min(...ns.map((n) => n.start));
  return span > 0 ? r(ns.reduce((a, n) => a + n.end - n.start, 0) / span, 3) : 0;
}

function silenceRatio(ns) {
  if (ns.length < 2) return 0;
  const s = [...ns].sort((a, b) => a.start - b.start);
  const span = Math.max(...s.map((n) => n.end)) - s[0].start;
  if (span <= 0) return 0;
  let sounding = 0, cs = s[0].start, ce = s[0].end;
  for (const n of s.slice(1)) {
    if (n.start > ce) { sounding += ce - cs; cs = n.start; ce = n.end; } else ce = Math.max(ce, n.end);
  }
  sounding += ce - cs;
  return r(Math.max(0, 1 - sounding / span), 4);
}

function gridError(ns, tempo, division) {
  if (!ns.length) return 0;
  const step = 60 / tempo / division;
  const errs = ns.map((n) => { const off = ((n.start % step) + step) % step; return Math.min(off, step - off) / step; });
  return r(errs.reduce((a, b) => a + b, 0) / errs.length, 4);
}

function keepLowest(ns, keep = "lowest") {
  const lowest = keep === "lowest";
  const ordered = [...ns].sort((a, b) => a.start - b.start || (lowest ? a.pitch - b.pitch : b.pitch - a.pitch));
  const out = [];
  for (const n of ordered) {
    const beaten = out.some((m) => m.start < n.end && n.start < m.end && (lowest ? m.pitch < n.pitch : m.pitch > n.pitch));
    if (!beaten) out.push(n);
  }
  return out;
}

function nextOnsetAfter(onsets, excludeBefore) {
  let lo = 0, hi = onsets.length;                   // bisect_right
  while (lo < hi) { const mid = (lo + hi) >> 1; if (onsets[mid] <= excludeBefore) lo = mid + 1; else hi = mid; }
  return lo < onsets.length ? onsets[lo] : null;
}

export const DEFAULTS = {
  de_overlap: true, max_duration_beats: 2.0, duration_cap: true, velocity_floor: 15, min_duration: 0.02,
  tempo: 120, monophonic: false, mono_keep: "lowest", legato: false, legato_max_gap_beats: 0.25,
  legato_gap: 0.005, quantize: false, quantize_division: 4, quantize_strength: 0.5, quantize_keep_duration: true,
};

export function cleanNotes(input, params = {}) {
  const p = { ...DEFAULTS, ...params };
  const stats = {
    notes_before: input.length, median_duration_before: medianDuration(input),
    silence_ratio_before: silenceRatio(input), polyphony_before: polyphony(input),
    dropped_velocity_floor: 0, trimmed_de_overlap: 0, trimmed_duration_cap: 0, dropped_too_short: 0,
    dropped_monophonic: 0, extended_legato: 0, quantized: 0,
  };
  let notes = input.map((n) => ({ ...n }));
  if (p.velocity_floor > 0) {
    const kept = notes.filter((n) => n.velocity >= p.velocity_floor);
    stats.dropped_velocity_floor = notes.length - kept.length;
    notes = kept;
  }
  notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
  stats.grid_error_before = gridError(notes, p.tempo, p.quantize_division);
  if (p.monophonic) {
    const kept = keepLowest(notes, p.mono_keep);
    stats.dropped_monophonic = notes.length - kept.length;
    notes = kept;
  }
  if (p.quantize && notes.length && p.quantize_strength > 0 && p.quantize_strength <= 1) {
    const step = 60 / p.tempo / p.quantize_division;
    for (const n of notes) {
      const delta = (Math.round(n.start / step) * step - n.start) * p.quantize_strength;
      if (delta === 0) continue;
      n.start += delta;
      if (p.quantize_keep_duration) n.end += delta;
      else n.end += (Math.round(n.end / step) * step - n.end) * p.quantize_strength;
      if (n.end <= n.start) n.end = n.start + p.min_duration;
      stats.quantized++;
    }
    notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
  }
  if (p.de_overlap) {
    const byPitch = new Map();
    for (const n of notes) { if (!byPitch.has(n.pitch)) byPitch.set(n.pitch, []); byPitch.get(n.pitch).push(n); }
    for (const ns of byPitch.values()) {
      ns.sort((a, b) => a.start - b.start);
      for (let i = 1; i < ns.length; i++) if (ns[i - 1].end > ns[i].start) { ns[i - 1].end = ns[i].start; stats.trimmed_de_overlap++; }
    }
  }
  if (p.duration_cap && p.max_duration_beats > 0) {
    const maxS = p.max_duration_beats * (60 / p.tempo);
    const onsets = [...new Set(notes.map((n) => n.start))].sort((a, b) => a - b);
    for (const n of notes) {
      if (n.end - n.start <= maxS) continue;
      const nxt = nextOnsetAfter(onsets, n.start + 1e-6);
      if (nxt !== null && nxt < n.end) { n.end = nxt; stats.trimmed_duration_cap++; }
    }
  }
  const kept = notes.filter((n) => n.end - n.start >= p.min_duration);
  stats.dropped_too_short = notes.length - kept.length;
  notes = kept.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
  if (p.legato && notes.length) {
    const maxGap = p.legato_max_gap_beats * (60 / p.tempo);
    const onsets = [...new Set(notes.map((n) => n.start))].sort((a, b) => a - b);
    for (const n of notes) {
      const nxt = nextOnsetAfter(onsets, n.end - 1e-9);
      if (nxt === null) continue;
      const gap = nxt - n.end;
      if (gap > 0 && gap <= maxGap) { n.end = nxt - p.legato_gap; stats.extended_legato++; }
    }
  }
  Object.assign(stats, {
    notes_after: notes.length, median_duration_after: medianDuration(notes),
    silence_ratio_after: silenceRatio(notes), polyphony_after: polyphony(notes),
    grid_error_after: gridError(notes, p.tempo, p.quantize_division),
  });
  return { notes, stats };
}

// merge.py quantization_error, 4/4 (the browser engine fits no meter-aware grid).
export function quantizationError(notes, tempo, division = 4) {
  if (!notes.length) return { grid: `1/${division * 4}`, n: 0, mean: 0, median: 0 };
  const step = 60 / tempo / division;
  const errs = notes.map((n) => { const off = ((n.start % step) + step) % step; return Math.min(off, step - off) / step; });
  return { grid: `1/${division * 4}`, n: errs.length, mean: r(errs.reduce((a, b) => a + b, 0) / errs.length, 4), median: r(median(errs), 4) };
}
