export interface UrlSelection {
  window: string | null;
  start: string | null;
}

export function readSelectionFromUrl(search: string): UrlSelection {
  const params = new URLSearchParams(search);
  return {
    window: params.get("window"),
    start: params.get("start"),
  };
}

export function buildSearchString(selection: UrlSelection): string {
  const params = new URLSearchParams();
  if (selection.window) {
    params.set("window", selection.window);
  }
  if (selection.start) {
    params.set("start", selection.start);
  }
  const query = params.toString();
  return query ? `?${query}` : "";
}
