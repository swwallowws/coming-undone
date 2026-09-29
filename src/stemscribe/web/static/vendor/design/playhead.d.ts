/** Types for playhead.js, for TypeScript projects. */

export declare function clampTime(t: number, duration: number): number;

export declare function timeAt(clientX: number, left: number, width: number, duration: number): number;

export interface SeekableOptions {
  /** Seconds mapped across the element's width; a function is read on each press. */
  duration?: number | (() => number);
  /** Your own mapping from a pointer's clientX to a time, for a zoomed or scrolled view. */
  toTime?: (clientX: number, rect: DOMRect) => number;
  /** The head is at `t` while pressed or dragged (also called on the press). */
  onScrub?: (t: number) => void;
  /** The pointer was let go at `t`. Once per gesture. */
  onSeek?: (t: number) => void;
  /** The gesture was taken over (a scroll) or cancelled. */
  onCancel?: () => void;
  /** False ignores presses. */
  enabled?: () => boolean;
}

/** Make `el` seekable by click or drag. Returns a function that removes the listeners. */
export declare function seekable(el: HTMLElement, opts: SeekableOptions): () => void;
