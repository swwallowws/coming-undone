/** Types for soundfont.js, for TypeScript projects. */

export declare const SOUNDFONT_VERSION: string;
export declare const SHARED_SOUNDFONT: string;
export declare const SOUNDFONT_CACHE: string;

export interface SoundfontOptions {
  /** Caches a studio used before; deleted so their copy stops taking room. */
  retire?: string[];
  /** For tests. */
  shared?: string;
  here?: { href: string; origin: string };
  fetch?: typeof fetch;
  caches?: CacheStorage;
}

/** The addresses to try, in order. */
export declare function soundfontSources(local: string | URL, opts?: SoundfontOptions): string[];

/** The bank's bytes: shared cache, else the shared address, else the studio's own copy. */
export declare function fetchSoundfont(local: string | URL, opts?: SoundfontOptions): Promise<ArrayBuffer>;
