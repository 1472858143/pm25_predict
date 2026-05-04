# PM2.5 Web Dashboard

本地 Web 仪表盘，用于可视化对比 PM2.5 多模型预测结果。

## 架构

```
浏览器 (localhost:5173)
    ↓ /api/*
Vite dev proxy → FastAPI (localhost:8000)
    ↓
pm25_forecast/outputs/  (JSON/CSV 文件)
```

**技术栈：**
- 后端：FastAPI + Pydantic + pytest
- 前端：React 18 + TypeScript + Vite + Ant Design + ECharts

## 快速启动

需要两个终端同时运行：

```powershell
# 终端 1：后端
conda activate pm25
cd web/backend
pip install -e .[test]
uvicorn app.main:app --reload --port 8000

# 终端 2：前端
cd web/frontend
npm install
npm run dev
```

打开 `http://localhost:5173`

## API 接口

所有接口均为 GET，前缀 `/api`。数据接口接受 `?window=<name>&start=<start_id>` 查询参数，缺省使用首个可用窗口/起点。

| 接口 | 说明 | 响应模型 |
|------|------|----------|
| `/api/health` | 健康检查 | `{"status": "ok"}` |
| `/api/windows` | 列出所有预测窗口及可用起点 | `WindowsResponse` |
| `/api/metrics` | 所有模型指标汇总（RMSE/MAE/MAPE/SMAPE/R²/bias） | `MetricsResponse` |
| `/api/predictions` | 所有模型预测值合集（用于曲线叠加） | `PredictionsAggregateResponse` |
| `/api/predictions/{model}` | 单模型完整预测明细 | `ModelPredictionsResponse` |
| `/api/horizon-metrics/{model}` | 单模型逐 horizon 指标 | `HorizonMetricsResponse` |

### 关键响应结构

**`/api/windows`**
```json
{
  "windows": [
    {
      "name": "window_720h_to_72h",
      "input_window": 720,
      "output_window": 72,
      "starts": ["start_2026_03_01_0000"]
    }
  ]
}
```

**`/api/metrics?window=window_720h_to_72h&start=start_2026_03_01_0000`**
```json
{
  "predict_start": "2026-03-01 00:00:00+08:00",
  "models": [
    {"model_name": "lstm", "RMSE": 36.78, "MAE": 27.91, "R2": 0.33, "bias": -12.57, "...": "..."}
  ],
  "missing_models": []
}
```

**`/api/predictions`**
```json
{
  "horizons": [1, 2, 3, "..."],
  "timestamps": ["2026-03-01 00:00:00+08:00", "..."],
  "y_true": [107.3, 107.6, "..."],
  "predictions": {
    "lstm": [132.1, 127.9, "..."],
    "attention_lstm": ["..."]
  }
}
```

### 错误处理

| 场景 | 状态码 | 说明 |
|------|--------|------|
| window/start 不存在 | 404 | `{"detail": "Window 'xxx' not found"}` |
| 模型目录缺失 | 200 | 跳过该模型，记入 `missing_models` |
| JSON 解析失败 | 500 | 包含文件路径与异常信息 |
| outputs 目录缺失 | 200 | `/api/windows` 返回 `{"windows": []}` |

## 前端页面

单页面三模块，自上而下：

1. **模型指标对比表** — Ant Design Table，6 模型横向对比，可排序，最优值绿底高亮，attention_lstm 行加粗
2. **预测曲线对比** — ECharts 折线图，y_true 粗黑线 + 各模型彩色线，支持区域缩放和图例切换
3. **单模型误差分析** — 模型选择下拉 + 散点图（y_true vs y_pred + y=x 参考线）+ 残差直方图

**模型调色板：**
| 模型 | 颜色 |
|------|------|
| lstm | 蓝 #1677ff |
| attention_lstm | 红 #f5222d（核心模型） |
| xgboost | 绿 #52c41a |
| random_forest | 紫 #722ed1 |
| arima | 橙 #fa8c16 |
| sarima | 灰 #8c8c8c |

**窗口选择：**
- 页面顶部两个下拉：时间窗口 + 预测起点
- 窗口名自动转换为中文：`window_720h_to_72h` → `30 天历史 → 3 天预测 (720h→72h)`
- 起点名自动转换：`start_2026_03_01_0000` → `2026-03-01 00:00 (北京时间)`
- URL 参数同步：`?window=...&start=...`，支持刷新恢复和分享

## 数据来源

后端读取 `pm25_forecast/outputs/` 下的预测结果目录结构：

```
outputs/
└── window_720h_to_72h/
    └── predictions/
        └── start_2026_03_01_0000/
            ├── lstm/
            │   ├── metrics.json
            │   ├── prediction_summary.json
            │   ├── predictions.csv
            │   └── horizon_metrics.csv
            ├── attention_lstm/
            └── ...（其他模型）
```

默认路径可通过 `OUTPUT_ROOT` 环境变量覆盖。

## 测试

```powershell
# 后端（26 个测试）
cd web/backend
pytest -v

# 前端（18 个测试）
cd web/frontend
npm test
```
