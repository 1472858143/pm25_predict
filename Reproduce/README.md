# PM2.5 多模型 720h -> 72h 直接预测说明

## 1. 目标

本目录基于论文《基于机器学习的北京市 PM2.5 浓度预测模型及模拟分析》中的时间序列预测思路，使用北京小时级数据复现并扩展 PM2.5 预测实验。

默认实验为：

```text
过去 720 小时 -> 一次性预测未来 72 小时 PM2.5
```

`LSTM`、`XGBoost`、`RandomForest` 使用 6 个特征做直接多输出预测：

```text
temperature, humidity, wind_speed, precipitation, pressure, pm25
```

`ARIMA`、`SARIMA` 只使用训练期历史 `pm25`，不使用外生特征，不使用验证期或预测窗口真实值拟合。

## 2. 数据与时间切分

数据文件：

```text
data/processed_beijing.csv
```

默认预测起点：

```text
2026-03-01 00:00:00+08:00
```

时间切分规则：

- 训练期：验证集开始之前的全部历史数据。
- 验证期：`--predict-start` 前 3 个月，仅用于 LSTM best model 选择。
- 预测窗口：从 `--predict-start` 开始的未来 `output_window` 小时，默认 72 行。
- reserve：预测窗口之后的数据，不参与训练、验证、校准或调参。
- scaler 只在训练期 fit；验证期和预测窗口只 transform。

## 3. 输出结构

新实验结果统一写入窗口实验目录：

```text
Reproduce/outputs/window_720h_to_72h/
├── data/
├── models/
│   ├── lstm/
│   ├── xgboost/
│   ├── random_forest/
│   ├── arima/
│   └── sarima/
├── predictions/
│   └── start_2026_03_01_0000/
│       └── <model_name>/
└── comparisons/
    └── start_2026_03_01_0000/
```

单模型预测目录：

```text
Reproduce/outputs/window_720h_to_72h/predictions/start_2026_03_01_0000/<model_name>/
```

关键文件：

```text
predictions.csv
metrics.json
metrics_model_raw.json
horizon_metrics.csv
stage_metrics.csv
prediction_summary.json
plots/prediction_curve.png
plots/error_curve.png
plots/scatter.png
```

`predictions.csv` 固定字段：

```text
model_name,sample_id,origin_timestamp,target_end_timestamp,timestamp,horizon,y_true,y_pred_model,y_pred,error,abs_error,relative_error
```

跨模型比较目录：

```text
Reproduce/outputs/window_720h_to_72h/comparisons/start_2026_03_01_0000/
├── model_metrics.csv
├── model_metrics.json
└── all_predictions.csv
```

旧 LSTM 输出目录是历史产物，新脚本不再把旧结构作为默认写入目标。

## 4. 运行方式

先激活项目环境：

```powershell
conda activate pm25
python -c "import sys, torch; print(sys.executable); print(torch.cuda.is_available())"
```

准备共享窗口数据：

```powershell
python -m Reproduce.scripts.prepare_data --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

训练各模型：

```powershell
python -m Reproduce.scripts.train_model --model lstm --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00" --device cuda --epochs 100
python -m Reproduce.scripts.train_model --model xgboost --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model random_forest --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model arima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model sarima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

预测单个模型：

```powershell
python -m Reproduce.scripts.predict_model --model lstm --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00" --device cuda
python -m Reproduce.scripts.predict_model --model random_forest --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

汇总已有预测结果：

```powershell
python -m Reproduce.scripts.compare_models --models lstm xgboost random_forest arima sarima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

快速 smoke：

```powershell
python -m Reproduce.scripts.train_model --model random_forest --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00" --n-estimators 5 --n-jobs 1
python -m Reproduce.scripts.predict_model --model random_forest --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

## 5. 注意事项

- 推荐使用 `train_model.py`、`predict_model.py`、`compare_models.py` 作为新入口。
- `train_lstm.py`、`predict_window.py`、`predict_month.py` 保留为 LSTM 兼容入口，输出也进入新结构。
- LSTM 默认可使用训练集拟合的 horizon 线性校准；其它模型默认 `y_pred == y_pred_model`。
- 显式设置 `--calibration-fit validation` 时，需要在实验报告中说明验证集被额外用于校准。
- 预测窗口真实值只用于最终评价，不参与训练、校准、模型选择或调参。
