import { describe, expect, it } from "vitest";
import { formatStart, formatWindow } from "../src/utils/format";

describe("formatWindow", () => {
  it("formats 720h to 72h as 30 days to 3 days", () => {
    expect(formatWindow("window_720h_to_72h")).toBe(
      "30 天历史 → 3 天预测 (720h→72h)"
    );
  });

  it("formats 168h to 72h", () => {
    expect(formatWindow("window_168h_to_72h")).toBe(
      "7 天历史 → 3 天预测 (168h→72h)"
    );
  });

  it("falls back when not divisible by 24", () => {
    expect(formatWindow("window_5h_to_2h")).toBe("5h历史 → 2h预测 (5h→2h)");
  });

  it("returns input when pattern does not match", () => {
    expect(formatWindow("garbage")).toBe("garbage");
  });
});

describe("formatStart", () => {
  it("formats start_2026_03_01_0000", () => {
    expect(formatStart("start_2026_03_01_0000")).toBe(
      "2026-03-01 00:00 (北京时间)"
    );
  });

  it("formats start_2026_12_31_2359", () => {
    expect(formatStart("start_2026_12_31_2359")).toBe(
      "2026-12-31 23:59 (北京时间)"
    );
  });

  it("returns input when pattern does not match", () => {
    expect(formatStart("garbage")).toBe("garbage");
  });
});
