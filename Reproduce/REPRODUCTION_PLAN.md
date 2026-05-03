# PM2.5 多模型对比复现实验计划

## 1. 当前口径

默认实验：

```text
过去 720 小时 -> 未来 72 小时 PM2.5
```

预测策略：

- `LSTM`：6 特征直接多输出预测。
- `XGBoost`：6 特征窗口展平后直接多输出预测。
- `RandomForest`：6 特征窗口展平后直接多输出预测。
- `ARIMA`：只使用训练期历史 `pm25`。
- `SARIMA`：只使用训练期历史 `pm25`，默认季节周期为 24。

`ARIMA`、`SARIMA` 不使用外生特征，不使用验证期或预测窗口真实值拟合。

## 2. 样本规则

| 项目 | 规则 |
| --- | --- |
| 输入窗口 | `720` 小时 |
| 输出窗口 | `72` 小时 |
| 训练样本 | 目标窗口完整早于验证期 |
| 验证样本 | `--predict-start` 前 3 个月，目标窗口完整早于 `--predict-start` |
| 预测样本 | 目标窗口从 `--predict-start` 开始，长度为 `72` 小时 |
| 预测输出行数 | `72` |

硬约束：

- scaler 只在训练期 fit。
- 预测窗口真实值只用于最终评价。
- 验证期默认只用于 LSTM best model 选择。
- `ARIMA`、`SARIMA` 的训练输入只来自训练期 `pm25`。

## 3. 输出规划

```text
Reproduce/outputs/window_720h_to_72h/
├── data/
│   ├── windows.npz
│   ├── scaler.json
│   └── data_config.json
├── models/<model_name>/
│   ├── model.pt 或 model.pkl
│   └── training_config.json
├── predictions/start_2026_03_01_0000/<model_name>/
│   ├── predictions.csv
│   ├── metrics.json
│   ├── metrics_model_raw.json
│   ├── horizon_metrics.csv
│   ├── stage_metrics.csv
│   ├── prediction_summary.json
│   └── plots/
└── comparisons/start_2026_03_01_0000/
    ├── model_metrics.csv
    ├── model_metrics.json
    └── all_predictions.csv
```

`predictions.csv` 固定字段：

```text
model_name,sample_id,origin_timestamp,target_end_timestamp,timestamp,horizon,y_true,y_pred_model,y_pred,error,abs_error,relative_error
```

## 4. 执行命令

数据准备：

```powershell
python -m Reproduce.scripts.prepare_data --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

训练：

```powershell
python -m Reproduce.scripts.train_model --model lstm --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00" --device cuda --epochs 100
python -m Reproduce.scripts.train_model --model xgboost --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model random_forest --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model arima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model sarima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

预测：

```powershell
python -m Reproduce.scripts.predict_model --model lstm --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00" --device cuda
python -m Reproduce.scripts.predict_model --model random_forest --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

比较：

```powershell
python -m Reproduce.scripts.compare_models --models lstm xgboost random_forest arima sarima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

## 5. 验收标准

- 数据准备默认写入 `Reproduce/outputs/window_720h_to_72h/data/`。
- 每个模型训练产物写入 `Reproduce/outputs/window_720h_to_72h/models/<model_name>/`。
- 每个模型预测结果写入 `Reproduce/outputs/window_720h_to_72h/predictions/start_2026_03_01_0000/<model_name>/`。
- 每个模型的 `predictions.csv` 包含固定字段和 `72` 行。
- 汇总目录生成每个模型一行的 `model_metrics.csv`。
- `all_predictions.csv` 可按 `model_name` 区分不同模型结果。
- `ARIMA`、`SARIMA` 不读取验证期、预测窗口真实值或外生特征进行拟合。
