/** The one Formspree form every studio sends to. */
export declare const CONTACT_ENDPOINT: string;

/** Fill `box` with the collapsed "Get in touch" form; returns the <form>. */
export declare function contactForm(
  box: HTMLElement,
  opts?: { subject?: string; endpoint?: string },
): HTMLFormElement;
