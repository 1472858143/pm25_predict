import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Header } from "../src/components/Header";

const windows = [
  {
    name: "window_720h_to_72h",
    input_window: 720,
    output_window: 72,
    starts: ["start_2026_03_01_0000"],
  },
  {
    name: "window_168h_to_72h",
    input_window: 168,
    output_window: 72,
    starts: [],
  },
];

describe("Header", () => {
  it("renders the dashboard title", () => {
    render(
      <Header
        windows={windows}
        selectedWindow="window_720h_to_72h"
        selectedStart="start_2026_03_01_0000"
        onChangeWindow={() => {}}
        onChangeStart={() => {}}
      />
    );
    expect(screen.getByText(/PM2.5 多模型预测对比仪表盘/)).toBeInTheDocument();
  });

  it("shows formatted window label as selected value", () => {
    render(
      <Header
        windows={windows}
        selectedWindow="window_720h_to_72h"
        selectedStart="start_2026_03_01_0000"
        onChangeWindow={() => {}}
        onChangeStart={() => {}}
      />
    );
    expect(screen.getByText(/30 天历史/)).toBeInTheDocument();
  });

  it("disables the start selector when window has no starts", () => {
    render(
      <Header
        windows={windows}
        selectedWindow="window_168h_to_72h"
        selectedStart={null}
        onChangeWindow={() => {}}
        onChangeStart={() => {}}
      />
    );
    const startSelect = screen.getByTestId("start-select");
    expect(startSelect.className).toContain("ant-select-disabled");
  });

  it("calls onChangeWindow when window selector changes", () => {
    const handler = vi.fn();
    render(
      <Header
        windows={windows}
        selectedWindow="window_720h_to_72h"
        selectedStart="start_2026_03_01_0000"
        onChangeWindow={handler}
        onChangeStart={() => {}}
      />
    );
    const trigger = screen.getByTestId("window-select").querySelector(".ant-select-selector");
    fireEvent.mouseDown(trigger as Element);
    const option = screen.getByText(/7 天历史/);
    fireEvent.click(option);
    expect(handler).toHaveBeenCalledWith(
      "window_168h_to_72h",
      expect.objectContaining({ value: "window_168h_to_72h" })
    );
  });
});
