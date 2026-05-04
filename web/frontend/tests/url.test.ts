import { describe, expect, it } from "vitest";
import { buildSearchString, readSelectionFromUrl } from "../src/utils/url";

describe("readSelectionFromUrl", () => {
  it("returns nulls for empty search", () => {
    expect(readSelectionFromUrl("")).toEqual({ window: null, start: null });
  });

  it("parses window and start params", () => {
    expect(readSelectionFromUrl("?window=W1&start=S1")).toEqual({
      window: "W1",
      start: "S1",
    });
  });
});

describe("buildSearchString", () => {
  it("returns empty string when both null", () => {
    expect(buildSearchString({ window: null, start: null })).toBe("");
  });

  it("returns ?window=...&start=... when both present", () => {
    expect(
      buildSearchString({ window: "window_720h_to_72h", start: "start_2026_03_01_0000" })
    ).toBe("?window=window_720h_to_72h&start=start_2026_03_01_0000");
  });
});
