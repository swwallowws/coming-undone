// Model files for the browser engine: the manifest (models.json), downloads with
// progress, and Cache Storage so every model downloads once and is reused on later
// runs (the cache survives reloads; the browser may still evict it under storage
// pressure, which navigator.storage.persist() asks it not to).

// the page reads the same cache to size the first download (index.html browserPlan)
const CACHE = "coming-undone-models-v1";

// timcsy/demucs-web-onnx, pinned to the commit the engine was checked against
export const HTDEMUCS_REV = "92e33df61cfc9eb820272aaa62d2ef6dcf4d950d";
const HTDEMUCS = { url: `https://huggingface.co/timcsy/demucs-web-onnx/resolve/${HTDEMUCS_REV}/htdemucs_embedded.onnx`, bytes: 180534758 };

export async function loadManifest(base) {
  let m = null;
  try {
    const r = await fetch(new URL("models.json", base));
    if (r.ok) m = await r.json();
  } catch (_) { /* no manifest: defaults below */ }
  m ||= {
    htdemucs: HTDEMUCS,
    adt: { encoder: { file: "adt_encoder.int8.onnx", bytes: 35709676 }, decoder: { file: "adt_decoder.int8.onnx", bytes: 37204260 } },
  };
  const url = (e) => new URL(e.url || e.file, base).href;
  const out = {
    htdemucs: { url: url(m.htdemucs), bytes: m.htdemucs.bytes },
    adtEncoder: { url: url(m.adt.encoder), bytes: m.adt.encoder.bytes },
    adtDecoder: { url: url(m.adt.decoder), bytes: m.adt.decoder.bytes },
    muscriptorSmall: null,
  };
  if (m.muscriptorSmall) {
    const dir = new URL(m.muscriptorSmall.dir, base).href;
    const f = m.muscriptorSmall.files;
    out.muscriptorSmall = {
      bytes: m.muscriptorSmall.bytes,
      cond: { url: new URL(f.cond, dir).href }, lm: { url: new URL(f.lm, dir).href },
      emb: { url: new URL(f.emb, dir).href }, meta: { url: new URL(f.meta, dir).href },
    };
  }
  return out;
}

async function openCache() {
  try { return await caches.open(CACHE); } catch (_) { return null; }   // no Cache Storage (http, private mode)
}

// Downloads (or reads from the cache) every url, reporting combined progress.
// entries: [{ url, bytes }]; on(done, total, fromCache) as bytes arrive.
export async function fetchAll(entries, on = () => {}) {
  const cache = await openCache();
  try { await navigator.storage?.persist?.(); } catch (_) { /* optional */ }
  const total = entries.reduce((a, e) => a + (e.bytes || 0), 0);
  let done = 0;
  const one = async (e) => {
    const hit = cache ? await cache.match(e.url).catch(() => null) : null;
    if (hit) { const b = await hit.arrayBuffer(); done += e.bytes || b.byteLength; on(done, total, true); return { buf: b, cached: true }; }
    const r = await fetch(e.url);
    if (!r.ok) throw new Error(`${e.url}: HTTP ${r.status}`);
    const parts = [];
    let n = 0;
    const reader = r.body.getReader();
    for (;;) {
      const { done: end, value } = await reader.read();
      if (end) break;
      parts.push(value); n += value.length; done += value.length;
      on(done, total);
    }
    const buf = new Uint8Array(n);
    let o = 0;
    for (const p of parts) { buf.set(p, o); o += p.length; }
    if (cache) { try { await cache.put(e.url, new Response(buf.slice())); } catch (_) { /* quota: works, just not kept */ } }
    return { buf: buf.buffer, cached: false };
  };
  const res = await Promise.all(entries.map(one));
  return { bufs: res.map((x) => x.buf), downloaded: res.filter((x) => !x.cached).length };
}
