import React, { useEffect, useState } from "react";
import { Alert, Card, Skeleton } from "antd";
import ReactECharts from "echarts-for-react";
import { fetchPredictionsAggregate } from "../api/client";
import type { PredictionsAggregateResponse } from "../types/api";
import { Y_TRUE_COLOR, colorForModel } from "../utils/colors";

export interface PredictionCurveChartProps {
  window: string;
  start: string;
}

export const PredictionCurveChart: React.FC<PredictionCurveChartProps> = ({ window, start }) => {
  const [data, setData] = useState<PredictionsAggregateResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchPredictionsAggregate({ window, start })
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [window, start]);

  if (loading) {
    return (
      <Card title="预测曲线对比" style={{ marginTop: 16 }}>
        <Skeleton active />
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card title="预测曲线对比" style={{ marginTop: 16 }}>
        <Alert type="error" message={`加载失败: ${error ?? "no data"}`} showIcon />
      </Card>
    );
  }

  const modelNames = Object.keys(data.predictions);
  const series = [
    {
      name: "y_true",
      type: "line",
      data: data.y_true,
      lineStyle: { color: Y_TRUE_COLOR, width: 3 },
      itemStyle: { color: Y_TRUE_COLOR },
      smooth: false,
    },
    ...modelNames.map((name) => ({
      name,
      type: "line",
      data: data.predictions[name],
      lineStyle: { color: colorForModel(name) },
      itemStyle: { color: colorForModel(name) },
      smooth: false,
    })),
  ];

  const option = {
    tooltip: { trigger: "axis" },
    legend: { data: ["y_true", ...modelNames], top: 0 },
    grid: { top: 40, left: 60, right: 30, bottom: 60 },
    xAxis: {
      type: "category",
      data: data.timestamps,
      axisLabel: { rotate: 45, fontSize: 10 },
    },
    yAxis: {
      type: "value",
      name: "PM2.5 (μg/m³)",
    },
    dataZoom: [
      { type: "inside" },
      { type: "slider" },
    ],
    series,
  };

  return (
    <Card title="预测曲线对比" style={{ marginTop: 16 }}>
      <ReactECharts option={option} style={{ height: 480 }} notMerge lazyUpdate />
    </Card>
  );
};
