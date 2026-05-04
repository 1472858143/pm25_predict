import React, { useEffect, useMemo, useState } from "react";
import { Alert, Card, Col, Row, Select, Skeleton, Space } from "antd";
import ReactECharts from "echarts-for-react";
import { fetchModelPredictions } from "../api/client";
import type { ModelPredictionsResponse, PredictionRow } from "../types/api";
import { colorForModel } from "../utils/colors";

const DEFAULT_MODEL = "attention_lstm";
const HISTOGRAM_BINS = 20;

function buildHistogram(values: number[]): { x: string[]; y: number[]; mean: number; std: number } {
  if (values.length === 0) {
    return { x: [], y: [], mean: 0, std: 0 };
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const binWidth = span / HISTOGRAM_BINS;
  const counts = new Array(HISTOGRAM_BINS).fill(0);
  for (const v of values) {
    const idx = Math.min(HISTOGRAM_BINS - 1, Math.floor((v - min) / binWidth));
    counts[idx] += 1;
  }
  const x = counts.map((_, i) => (min + binWidth * (i + 0.5)).toFixed(1));
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
  return { x, y: counts, mean, std: Math.sqrt(variance) };
}

export interface ErrorAnalysisProps {
  window: string;
  start: string;
  availableModels: string[];
}

export const ErrorAnalysis: React.FC<ErrorAnalysisProps> = ({ window, start, availableModels }) => {
  const initialModel = availableModels.includes(DEFAULT_MODEL)
    ? DEFAULT_MODEL
    : availableModels[0] ?? null;
  const [selectedModel, setSelectedModel] = useState<string | null>(initialModel);
  const [data, setData] = useState<ModelPredictionsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedModel) return;
    setLoading(true);
    setError(null);
    fetchModelPredictions(selectedModel, { window, start })
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [selectedModel, window, start]);

  const scatterOption = useMemo(() => {
    if (!data) return null;
    const points = data.rows.map((r: PredictionRow) => [r.y_true, r.y_pred]);
    const all = points.flat();
    const min = Math.min(...all);
    const max = Math.max(...all);
    return {
      tooltip: {
        trigger: "item",
        formatter: (p: { value: number[] }) =>
          `y_true: ${p.value[0].toFixed(2)}<br/>y_pred: ${p.value[1].toFixed(2)}`,
      },
      grid: { top: 40, left: 60, right: 30, bottom: 50 },
      xAxis: { type: "value", name: "y_true", min, max },
      yAxis: { type: "value", name: "y_pred", min, max },
      series: [
        {
          type: "scatter",
          data: points,
          itemStyle: { color: colorForModel(selectedModel ?? "") },
          symbolSize: 6,
        },
        {
          type: "line",
          data: [
            [min, min],
            [max, max],
          ],
          lineStyle: { color: "#999", type: "dashed" },
          symbol: "none",
          tooltip: { show: false },
        },
      ],
    };
  }, [data, selectedModel]);

  const histogramOption = useMemo(() => {
    if (!data) return null;
    const errors = data.rows.map((r: PredictionRow) => r.error);
    const { x, y, mean, std } = buildHistogram(errors);
    return {
      tooltip: { trigger: "axis" },
      grid: { top: 40, left: 60, right: 30, bottom: 50 },
      title: {
        text: `mean=${mean.toFixed(2)}, std=${std.toFixed(2)}`,
        textStyle: { fontSize: 12, fontWeight: "normal" },
        left: "center",
        top: 0,
      },
      xAxis: { type: "category", data: x, name: "error" },
      yAxis: { type: "value", name: "count" },
      series: [
        {
          type: "bar",
          data: y,
          itemStyle: { color: colorForModel(selectedModel ?? "") },
        },
      ],
    };
  }, [data, selectedModel]);

  return (
    <Card title="单模型误差分析" style={{ marginTop: 16 }}>
      <Space style={{ marginBottom: 16 }}>
        <span>选择模型:</span>
        <Select
          style={{ minWidth: 200 }}
          value={selectedModel ?? undefined}
          onChange={setSelectedModel}
          options={availableModels.map((m) => ({ value: m, label: m }))}
        />
      </Space>
      {loading && <Skeleton active />}
      {error && <Alert type="error" message={`加载失败: ${error}`} showIcon />}
      {!loading && !error && data && scatterOption && histogramOption && (
        <Row gutter={16}>
          <Col span={12}>
            <Card type="inner" title="y_true vs y_pred">
              <ReactECharts option={scatterOption} style={{ height: 360 }} notMerge lazyUpdate />
            </Card>
          </Col>
          <Col span={12}>
            <Card type="inner" title="残差分布">
              <ReactECharts option={histogramOption} style={{ height: 360 }} notMerge lazyUpdate />
            </Card>
          </Col>
        </Row>
      )}
    </Card>
  );
};
