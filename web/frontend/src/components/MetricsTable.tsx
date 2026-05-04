import React, { useEffect, useState } from "react";
import { Alert, Card, Skeleton, Table, Tag } from "antd";
import type { ColumnsType } from "antd/es/table";
import { fetchMetrics } from "../api/client";
import type { ModelMetrics, MetricsResponse } from "../types/api";

const METRIC_KEYS: (keyof Omit<ModelMetrics, "model_name">)[] = [
  "RMSE",
  "MAE",
  "MAPE",
  "SMAPE",
  "R2",
  "bias",
];

const HIGHER_IS_BETTER: Set<string> = new Set(["R2"]);

function bestValueFor(metric: keyof Omit<ModelMetrics, "model_name">, models: ModelMetrics[]): number | null {
  if (models.length === 0) return null;
  const values = models.map((m) => m[metric]);
  if (HIGHER_IS_BETTER.has(metric)) {
    return Math.max(...values);
  }
  if (metric === "bias") {
    return values.reduce((best, v) => (Math.abs(v) < Math.abs(best) ? v : best), values[0]);
  }
  return Math.min(...values);
}

export interface MetricsTableProps {
  window: string;
  start: string;
}

export const MetricsTable: React.FC<MetricsTableProps> = ({ window, start }) => {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchMetrics({ window, start })
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [window, start]);

  if (loading) {
    return (
      <Card title="模型指标对比">
        <Skeleton active />
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card title="模型指标对比">
        <Alert type="error" message={`加载失败: ${error ?? "no data"}`} showIcon />
      </Card>
    );
  }

  const bestPerMetric: Record<string, number | null> = {};
  for (const k of METRIC_KEYS) {
    bestPerMetric[k] = bestValueFor(k, data.models);
  }

  const columns: ColumnsType<ModelMetrics> = [
    {
      title: "模型",
      dataIndex: "model_name",
      key: "model_name",
      render: (name: string) =>
        name === "attention_lstm" ? <strong>{name}</strong> : name,
    },
    ...METRIC_KEYS.map((key) => ({
      title: key,
      dataIndex: key,
      key,
      sorter: (a: ModelMetrics, b: ModelMetrics) => a[key] - b[key],
      render: (value: number) => {
        const best = bestPerMetric[key];
        const isBest = best !== null && value === best;
        const text = value.toFixed(3);
        return isBest ? <span style={{ background: "#d9f7be", padding: "2px 6px" }}>{text}</span> : text;
      },
    })),
  ];

  return (
    <Card title="模型指标对比">
      <Table<ModelMetrics>
        rowKey="model_name"
        columns={columns}
        dataSource={data.models}
        pagination={false}
      />
      {data.missing_models.length > 0 && (
        <div style={{ marginTop: 12 }}>
          {data.missing_models.map((m) => (
            <Tag key={m} color="warning">
              缺失: {m}
            </Tag>
          ))}
        </div>
      )}
    </Card>
  );
};
