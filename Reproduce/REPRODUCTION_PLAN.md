# LSTM 720h -> 24h 直接多步预测复现计划

## 1. 当前口径

```text
过去 720 小时 -> 从指定起始时刻开始，一次性预测未来 24 小时 PM2.5
```

默认预测起始时刻：

```text
2026-03-01 00:00:00+08:00
```

## 2. 样本规则

| 项目 | 规则 |
| --- | --- |
| 输入窗口 | `720` 小时 |
| 输出窗口 | `24` 小时 |
| 训练样本 | 24 小时目标窗口完整早于验证期 |
| 验证样本 | `--predict-start` 前 3 个月，24 小时目标窗口完整早于 `--predict-start` |
| 预测样本 | 目标窗口从 `--predict-start` 开始，长度为 `24` 小时 |
| 预测样本数 | `1` |
| 预测输出行数 | `24` |

默认数据准备结果：

```text
X_train: [28408, 720, 6]
y_train: [28408, 24]
X_validation: [2137, 720, 6]
y_validation: [2137, 24]
X_predict: [1, 720, 6]
y_predict: [1, 24]
```

## 3. 输出规划

```text
Reproduce/outputs/lstm_720h_to_24h/
├── model.pt
├── model_best_val_loss.pt
├── model_best_train_loss.pt
├── calibration.json
├── calibration_horizon_stats.csv
├── training_history.csv
├── training_config.json
└── start_2026_03_01_0000/
    ├── predictions.csv
    ├── metrics.json
    ├── metrics_model_raw.json
    ├── horizon_metrics.csv
    ├── stage_metrics.csv
    └── plots/
        ├── prediction_curve.png
        ├── error_curve.png
        └── scatter.png
```

## 4. 执行命令

先激活 conda 环境：

```powershell
conda activate pm25
```

之后命令中统一使用 `python`，不要写完整解释器路径。

数据准备：

```powershell
python -m Reproduce.scripts.prepare_data --input-window 720 --output-window 24 --predict-start "2026-03-01 00:00:00+08:00"
```

训练：

```powershell
python -m Reproduce.scripts.train_lstm --input-window 720 --output-window 24 --predict-start "2026-03-01 00:00:00+08:00" --device cuda --epochs 100 --loss weighted_huber --peak-quantile 0.75 --peak-weight 3.0 --extreme-quantile 0.90 --extreme-weight 5.0 --variance-penalty 0.05 --calibration horizon_linear --calibration-fit train
```

训练过程每轮计算 `validation_loss`，并使用验证集最低损失保存 `model_best_val_loss.pt`。当前改进版默认使用峰值加权 Huber 损失、轻量方差惩罚，并在训练结束后用训练集拟合 `calibration.json`，用于缓解系统性低估、峰值响应不足和预测方差收缩。验证集仍只用于选择 best model。最终预测默认加载该模型并应用训练集校准。

预测：

```powershell
python -m Reproduce.scripts.predict_window --input-window 720 --output-window 24 --predict-start "2026-03-01 00:00:00+08:00" --device cuda
```

## 5. 验收标准

- `predictions.csv` 为 `24` 行。
- `horizon` 从 `1` 到 `24`。
- `predictions.csv` 中 `y_pred_model` 为原始 LSTM 反归一化输出，`y_pred` 为默认校准后的最终预测。
- `plots/prediction_curve.png` 横轴为小时级时间。
- 输出结果代表从 `--predict-start` 开始的未来 24 小时，不代表整个月预测。
- 预测开始后的数据只用于最终评价，不参与训练、验证、scaler fit 或模型选择。
- scaler 只在训练集上 fit；验证集和预测窗口只使用训练集 scaler transform。
- 默认 `calibration.json` 只允许使用训练集拟合，不能使用验证集或预测窗口真实值；若显式设置 `--calibration-fit validation`，需在实验报告中说明验证集被额外用于校准。
