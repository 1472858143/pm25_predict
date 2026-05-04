export const MODEL_COLORS: Record<string, string> = {
  lstm: "#1677ff",
  attention_lstm: "#f5222d",
  xgboost: "#52c41a",
  random_forest: "#722ed1",
  arima: "#fa8c16",
  sarima: "#8c8c8c",
};

export const Y_TRUE_COLOR = "#000000";

export function colorForModel(name: string): string {
  return MODEL_COLORS[name] ?? "#13c2c2";
}
