# LSTM 720h -> 24h 直接多步预测复现说明

## 1. 目标

本目录基于论文《基于机器学习的北京市 PM2.5 浓度预测模型及模拟分析》中的 LSTM 时间序列预测思想，结合本项目已有北京小时级数据，扩展实现直接多步预测实验。

当前主实验：

```text
过去 720 小时的 6 个特征 -> 一次性预测未来 24 小时 PM2.5
```

该方法是直接多步预测，不是连续单步预测，也不是递归预测。

## 2. 数据与时间口径

数据文件：

```text
data/processed_beijing.csv
```

输入特征：

```text
temperature, humidity, wind_speed, precipitation, pressure, pm25
```

默认预测起始时刻：

```text
2026-03-01 00:00:00+08:00
```

时间划分规则：

- 训练期：验证集开始之前的全部历史数据。
- 验证期：`--predict-start` 前 3 个月，用于选择 `best_model`。
- 预测期：从 `--predict-start` 开始的未来 `output-window` 小时。
- scaler 只在训练期拟合。
- 预测开始后的数据只能用于最终预测/测试评价，不得参与训练、验证、scaler fit、early stopping 或调参。

默认配置下：

| 项目 | 值 |
| --- | --- |
| 输入窗口 | `720` 小时 |
| 输出窗口 | `24` 小时 |
| 训练期 | `2022-08-04 09:00:00+08:00` 至 `2025-11-30 23:00:00+08:00` |
| 验证期 | `2025-12-01 00:00:00+08:00` 至 `2026-02-28 23:00:00+08:00` |
| 预测窗口 | `2026-03-01 00:00:00+08:00` 至 `2026-03-01 23:00:00+08:00` |

数据准备后主形状为：

```text
X_train: [28408, 720, 6]
y_train: [28408, 24]
X_validation: [2137, 720, 6]
y_validation: [2137, 24]
X_predict: [1, 720, 6]
y_predict: [1, 24]
```

## 3. 输出

主输出目录：

```text
Reproduce/outputs/lstm_720h_to_24h/
```

单次预测结果目录按起始时间命名：

```text
Reproduce/outputs/lstm_720h_to_24h/start_2026_03_01_0000/
```

关键产物：

```text
calibration.json
predictions.csv
metrics.json
metrics_model_raw.json
horizon_metrics.csv
stage_metrics.csv
plots/prediction_curve.png
plots/error_curve.png
plots/scatter.png
```

`predictions.csv` 只有 `24` 行，每一行对应未来一个小时：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 样本编号，单次预测时为 `0` |
| `origin_timestamp` | 预测窗口起始时刻 |
| `timestamp` | 当前 horizon 对应的目标小时 |
| `horizon` | 第几小时预测，`1-24` |
| `y_true` | 真实 PM2.5 |
| `y_pred_model` | LSTM 原始输出反归一化后的 PM2.5 |
| `y_pred` | 最终预测 PM2.5，默认是训练集校准后的结果 |
| `error` | `y_pred - y_true` |

图像按小时级绘制，横轴是未来 24 个小时的 `timestamp`，不是按天聚合。

## 4. 针对低估和峰值不足的改进

前一版普通 MSE 训练容易变成“保守平滑预测器”：整体低估、高污染峰值响应不足、预测方差小于真实方差。当前代码做了三处改进：

- 峰值加权损失：高于训练集 `75%` 分位数的 PM2.5 目标提高权重，高于 `90%` 分位数的目标进一步提高权重，减少高污染样本被均值化。
- 加权 Huber 损失：默认使用 `weighted_huber`，比纯 MSE 对异常噪声更稳，同时仍通过权重强调峰值。
- 方差惩罚与线性校准：训练时轻量约束 batch 预测振幅，训练结束后用训练集拟合 horizon 级线性校准，缓解系统性低估和方差收缩。

这些改进不使用预测开始后的数据。默认 `calibration.json` 只由训练集预测值和训练集真实值拟合，验证集仍只用于选择 `best_model`，预测窗口真实值仍只用于最终评价。

## 5. 运行方式

先激活项目使用的 conda 环境，确保命令中的 `python` 指向：

```text
E:\Enviroments\miniconda3\envs\pm25\python.exe
```

可在 PowerShell 中执行：

```powershell
conda activate pm25
python -c "import sys, torch; print(sys.executable); print(torch.cuda.is_available())"
```

确认环境正确后，先准备 `720h -> 24h` 窗口数据：

```powershell
python -m Reproduce.scripts.prepare_data --input-window 720 --output-window 24 --predict-start "2026-03-01 00:00:00+08:00"
```

训练命令如下。它会使用验证期之前的历史样本训练 LSTM，用预测起点前 3 个月验证集选择 `model_best_val_loss.pt`，同时保存最终模型。scaler 只在训练期 fit，验证期和预测窗口只 transform。默认训练已经启用峰值加权 Huber、方差惩罚和训练集校准；命令中显式写出这些参数，便于复现实验记录：

```powershell
python -m Reproduce.scripts.train_lstm --input-window 720 --output-window 24 --predict-start "2026-03-01 00:00:00+08:00" --device cuda --epochs 100 --loss weighted_huber --peak-quantile 0.75 --peak-weight 3.0 --extreme-quantile 0.90 --extreme-weight 5.0 --variance-penalty 0.05 --calibration horizon_linear --calibration-fit train
```

预测命令如下。它只预测从 `--predict-start` 开始的未来 `24` 小时，并按小时画图。默认会读取 `calibration.json`，把原始 LSTM 输出校准为最终 `y_pred`：

```powershell
python -m Reproduce.scripts.predict_window --input-window 720 --output-window 24 --predict-start "2026-03-01 00:00:00+08:00" --device cuda
```

如需查看未校准的 LSTM 原始输出效果，可以加 `--no-calibration`；同时 `predictions.csv` 中始终保留 `y_pred_model` 作为原始输出：

```powershell
python -m Reproduce.scripts.predict_window --input-window 720 --output-window 24 --predict-start "2026-03-01 00:00:00+08:00" --device cuda --no-calibration
```

快速检查流程可将训练轮数改为 `1`：

```powershell
python -m Reproduce.scripts.train_lstm --input-window 720 --output-window 24 --predict-start "2026-03-01 00:00:00+08:00" --device cuda --epochs 1
```

## 6. 参数说明

`--predict-start` 的作用是指定直接多步预测的起始目标时刻。

例如：

```text
--predict-start "2026-03-01 00:00:00+08:00"
```

表示模型输出：

```text
2026-03-01 00:00:00+08:00
到
2026-03-01 23:00:00+08:00
```

共 `24` 个小时的 PM2.5 预测值。

如果要预测下一天，可以改为：

```powershell
python -m Reproduce.scripts.predict_window --input-window 720 --output-window 24 --predict-start "2026-03-02 00:00:00+08:00" --device cuda
```

注意：更换预测起始时刻后，如果训练期也要随之改变，需要重新运行 `prepare_data` 和 `train_lstm`。

## 7. 注意

- 推荐使用 `predict_window.py` 入口；旧的 `predict_month.py` 仅保留兼容。
- 图像按小时级绘制，不做日均聚合。
- 当前方法一次性输出未来 24 小时，不使用递归滚动。
- 预测默认加载验证集损失最低的 `model_best_val_loss.pt`。
- 预测默认应用训练集拟合的 `calibration.json`；最终预测列为 `y_pred`，未校准原始模型列为 `y_pred_model`。
- 本实验任务难度高于原论文的未来 1 小时预测。
