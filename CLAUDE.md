# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目简介

PM2.5 多模型预测系统，基于北京小时级空气质量数据。默认实验：过去 720 小时 → 未来 72 小时直接多输出预测。支持 LSTM、AttentionLSTM、XGBoost、RandomForest、ARIMA、SARIMA 六种模型。

## 环境

```powershell
conda activate pm25
# Python 可执行文件: E:\Enviroments\miniconda3\envs\pm25\python.exe
# LSTM 训练需要 PyTorch + CUDA
```

## 常用命令

```powershell
# 准备共享窗口数据
python -m pm25_forecast.scripts.prepare_data --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"

# 训练（统一入口）
python -m pm25_forecast.scripts.train_model --model lstm --device cuda --epochs 100
python -m pm25_forecast.scripts.train_model --model attention_lstm --device cuda --epochs 100
python -m pm25_forecast.scripts.train_model --model xgboost
python -m pm25_forecast.scripts.train_model --model random_forest
python -m pm25_forecast.scripts.train_model --model arima
python -m pm25_forecast.scripts.train_model --model sarima

# 预测
python -m pm25_forecast.scripts.predict_model --model lstm --device cuda
python -m pm25_forecast.scripts.predict_model --model attention_lstm --device cuda

# 跨模型比较
python -m pm25_forecast.scripts.compare_models --models lstm attention_lstm xgboost random_forest arima sarima

# 运行测试（基于 unittest）
python -m unittest discover -s tests -v
```

## 架构

三阶段流水线：**prepare_data → train_model → predict_model**，compare_models 负责跨模型汇总。

**数据流**：`data/processed_beijing.csv` → `build_windows()` 滑动窗口 → `windows.npz`（X_train, y_train, X_validation, y_validation, X_predict, y_predict）。MinMaxScaler 仅在训练期 fit，验证期和预测窗口只做 transform。

**时间切分**（相对于 `--predict-start`）：
- 训练期：验证期之前的所有历史数据
- 验证期：predict-start 前 3 个月，仅用于 LSTM 最优模型选择
- 预测窗口：从 predict-start 开始的 `output_window` 小时
- reserve：预测窗口之后的数据，不参与训练、验证、校准或调参

**输出目录**：所有产物写入 `pm25_forecast/outputs/window_{input}h_to_{output}h/`。模型文件在 `models/<name>/`，预测结果在 `predictions/start_YYYY_MM_DD_HHMM/<name>/`，跨模型比较在 `comparisons/`。

**路径工具**（`utils/paths.py`）：集中管理路径构造。`SUPPORTED_MODEL_NAMES = ("lstm", "attention_lstm", "xgboost", "random_forest", "arima", "sarima")`。所有脚本通过 `validate_model_name()` 校验模型名。

**LSTM 特性**：直接多输出（单次前向传播输出所有时间步）。加权 Huber 损失 + 峰值分位数加权（75%/90% 分位阈值）。训练后按 horizon 拟合线性校准（默认在训练集上拟合）。保存三个模型权重：最终、最优训练损失、最优验证损失。

**AttentionLSTM 特性**：在 LSTM 基础上引入多头自注意力机制，用注意力加权聚合替代简单的最后时间步选择。架构：LSTM(input_window=720 → hidden_size=128) → Multi-Head Self-Attention(num_heads=4) → context[:, -1, :] → Linear(output_window=72)。同 LSTM 共享超参数（hidden_size, num_layers, dropout, learning_rate 等）及训练策略（加权 Huber 损失、ReduceLROnPlateau 调度、早停、梯度裁剪、线性校准）。核心模型，性能预期优于其他所有模型。

**统计模型**（ARIMA/SARIMA）：仅使用训练期 pm25 单变量，不使用外生特征，不读取验证期或预测窗口真实值。SARIMA 支持 `--sarima-auto` 通过 pmdarima 自动选参（需安装 pmdarima）。

**树模型**（XGBoost/RandomForest）：窗口展平为一维特征向量，通过 sklearn 兼容封装实现直接多输出。

## 约定

- 时间戳使用 Asia/Shanghai（UTC+8），默认 predict-start：`2026-03-01 00:00:00+08:00`
- 6 个特征：temperature, humidity, wind_speed, precipitation, pressure, pm25
- 目标列：pm25（同时作为特征输入）
- 所有脚本支持 `--prepare-data` 参数，窗口数据不存在时自动生成
- `train_lstm.py` / `predict_window.py` / `predict_month.py` 为历史 LSTM 入口，优先使用 `train_model.py` / `predict_model.py`
- LSTM 和 AttentionLSTM 训练后均使用 `hidden_size=128, num_layers=2, dropout=0.3` 默认值（经过调优），`predict_model.py` 的相关参数默认值已更新以匹配
- AttentionLSTM 可通过 `--attention-heads` 参数指定注意力头数（默认=4）
