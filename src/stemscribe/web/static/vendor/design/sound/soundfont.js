// The shared General MIDI bank (gm.sf3), fetched once for every studio on the site.
//
// Every studio lives on swwallowws.github.io, so one address for the bank means the browser
// downloads it once and keeps it in one Cache API entry they all read. A studio passes its own
// copy (vendor/design/sound/gm.sf3, from `sync.sh --sound`) as the fallback: used when the page
// is served anywhere else (localhost, a preview) or when the shared one doesn't answer.
//
//   const bytes = await fetchSoundfont(new URL('../vendor/design/sound/gm.sf3', import.meta.url),
//                                      { retire: ['starling-soundfont'] });

/** Bump with a new gm.sf3: the address changes, so every studio fetches the new bank. */
export const SOUNDFONT_VERSION = '1';
/** Where the one shared copy is served. */
export const SHARED_SOUNDFONT = `https://swwallowws.github.io/sound/gm.sf3?v=${SOUNDFONT_VERSION}`;
/** The one Cache API store every studio keeps the bank in. */
export const SOUNDFONT_CACHE = 'swwallowws-soundfont';

const samePath = (a, b) => {
  const x = new URL(a), y = new URL(b);
  return x.origin + x.pathname === y.origin + y.pathname;
};

/** The addresses to try, in order: the shared one first when this page is on its site,
 *  then the studio's own copy. */
export function soundfontSources(local, { shared = SHARED_SOUNDFONT, here = globalThis.location } = {}) {
  const own = new URL(String(local), here?.href).href;
  return here?.origin === new URL(shared).origin ? [shared, own] : [own];
}

/** The bank's bytes: from the shared cache, else fetched (shared address, then the studio's
 *  own copy) and cached. Options: retire, the names of caches a studio used before, deleted
 *  so their copy stops taking room; shared, here, fetch and caches for tests. */
export async function fetchSoundfont(local, opts = {}) {
  const { retire = [], fetch: get = globalThis.fetch, caches: store = globalThis.caches } = opts;
  let cache = null;
  try {
    cache = await store.open(SOUNDFONT_CACHE);
    for (const name of retire) await store.delete(name);
  } catch { /* no Cache API here: fetch every time */ }
  let failure = null;
  for (const url of soundfontSources(local, opts)) {
    try {
      let res = cache && (await cache.match(url));
      if (!res) {
        res = await get(url);
        if (!res.ok) throw new Error(`the sounds did not load (${res.status})`);
        if (cache) {
          // an older version of the same file goes; other files stay
          for (const old of await cache.keys()) if (old.url !== url && samePath(old.url, url)) await cache.delete(old);
          await cache.put(url, res.clone()).catch(() => {});
        }
      }
      return await res.arrayBuffer();
    } catch (e) {
      failure = e;
    }
  }
  throw failure;
}
