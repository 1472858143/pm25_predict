import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("../src/api/client", () => ({
  fetchMetrics: vi.fn(),
}));

import { MetricsTable } from "../src/components/MetricsTable";
import { fetchMetrics } from "../src/api/client";

describe("MetricsTable", () => {
  beforeEach(() => {
    vi.mocked(fetchMetrics).mockReset();
  });

  it("renders model rows from API response", async () => {
    vi.mocked(fetchMetrics).mockResolvedValue({
      window: "window_720h_to_72h",
      start: "start_2026_03_01_0000",
      predict_start: "2026-03-01 00:00:00+08:00",
      models: [
        { model_name: "lstm", RMSE: 31, MAE: 30, MAPE: 35, SMAPE: 32, R2: 0.3, bias: -10 },
        { model_name: "attention_lstm", RMSE: 26, MAE: 25, MAPE: 30, SMAPE: 27, R2: 0.5, bias: -5 },
      ],
      missing_models: [],
    });
    render(<MetricsTable window="window_720h_to_72h" start="start_2026_03_01_0000" />);
    await waitFor(() => {
      expect(screen.getByText("lstm")).toBeInTheDocument();
    });
    expect(screen.getByText("attention_lstm")).toBeInTheDocument();
  });

  it("displays missing models tag", async () => {
    vi.mocked(fetchMetrics).mockResolvedValue({
      window: "window_720h_to_72h",
      start: "start_2026_03_01_0000",
      predict_start: "2026-03-01 00:00:00+08:00",
      models: [
        { model_name: "lstm", RMSE: 31, MAE: 30, MAPE: 35, SMAPE: 32, R2: 0.3, bias: -10 },
      ],
      missing_models: ["arima"],
    });
    render(<MetricsTable window="window_720h_to_72h" start="start_2026_03_01_0000" />);
    await waitFor(() => {
      expect(screen.getByText(/缺失: arima/)).toBeInTheDocument();
    });
  });

  it("shows error alert on failure", async () => {
    vi.mocked(fetchMetrics).mockRejectedValue(new Error("boom"));
    render(<MetricsTable window="window_720h_to_72h" start="start_2026_03_01_0000" />);
    await waitFor(() => {
      expect(screen.getByText(/加载失败/)).toBeInTheDocument();
    });
  });
});
