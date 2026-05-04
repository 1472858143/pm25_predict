export interface WindowInfo {
  name: string;
  input_window: number;
  output_window: number;
  starts: string[];
}

export interface WindowsResponse {
  windows: WindowInfo[];
}

export interface ModelMetrics {
  model_name: string;
  RMSE: number;
  MAE: number;
  MAPE: number;
  SMAPE: number;
  R2: number;
  bias: number;
}

export interface MetricsResponse {
  window: string;
  start: string;
  predict_start: string;
  models: ModelMetrics[];
  missing_models: string[];
}

export interface PredictionsAggregateResponse {
  window: string;
  start: string;
  horizons: number[];
  timestamps: string[];
  y_true: number[];
  predictions: Record<string, number[]>;
  missing_models: string[];
}

export interface PredictionRow {
  sample_id: number;
  origin_timestamp: string;
  target_end_timestamp: string;
  timestamp: string;
  horizon: number;
  y_true: number;
  y_pred_model: number;
  y_pred: number;
  error: number;
  abs_error: number;
  relative_error: number;
}

export interface ModelPredictionsResponse {
  window: string;
  start: string;
  model_name: string;
  rows: PredictionRow[];
}
