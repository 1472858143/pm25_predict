import axios from "axios";
import type {
  MetricsResponse,
  ModelPredictionsResponse,
  PredictionsAggregateResponse,
  WindowsResponse,
} from "../types/api";

const http = axios.create({
  baseURL: "/api",
  timeout: 15000,
});

export interface SelectionParams {
  window?: string;
  start?: string;
}

export async function fetchWindows(): Promise<WindowsResponse> {
  const { data } = await http.get<WindowsResponse>("/windows");
  return data;
}

export async function fetchMetrics(params: SelectionParams = {}): Promise<MetricsResponse> {
  const { data } = await http.get<MetricsResponse>("/metrics", { params });
  return data;
}

export async function fetchPredictionsAggregate(
  params: SelectionParams = {}
): Promise<PredictionsAggregateResponse> {
  const { data } = await http.get<PredictionsAggregateResponse>("/predictions", { params });
  return data;
}

export async function fetchModelPredictions(
  modelName: string,
  params: SelectionParams = {}
): Promise<ModelPredictionsResponse> {
  const { data } = await http.get<ModelPredictionsResponse>(
    `/predictions/${encodeURIComponent(modelName)}`,
    { params }
  );
  return data;
}
