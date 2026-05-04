import "@testing-library/jest-dom";

// Polyfill window.matchMedia for antd responsive components in jsdom
if (!window.matchMedia) {
  window.matchMedia = (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  });
}

// Polyfill window.getComputedStyle for antd scrollbar measurement in jsdom
if (!window.getComputedStyle) {
  window.getComputedStyle = () => ({
    getPropertyValue: () => "",
  } as unknown as CSSStyleDeclaration);
}
