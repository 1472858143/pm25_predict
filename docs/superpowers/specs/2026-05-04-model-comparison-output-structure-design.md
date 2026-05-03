# 多模型对比与输出结构规范设计

## 背景

当前项目已经实现 LSTM 对北京 PM2.5 的直接多步预测，但输出目录把窗口形状、模型名称、训练产物和预测产物混在同一层级，例如 `Reproduce/outputs/lstm_720h_to_24h/`。这会导致后续加入 XGBoost、RandomForest、ARIMA、SARIMA 后难以区分共享数据、单模型产物和跨模型对比结果。

本设计将实验窗口与模型运行拆分：窗口目录负责数据准备和时间切分，模型目录负责各自训练产物，预测目录按预测起点和模型分层，比较目录只保存横向汇总。

## 默认实验参数

默认输入窗口使用 `720` 小时，默认输出窗口使用 `72` 小时：

```text
过去 720 小时的历史信息 -> 一次性预测未来 72 小时 PM2.5
```

推荐 `720` 小时作为默认输入窗口的原因：

- 720 小时约等于 30 天，能覆盖 PM2.5 的日周期、多周天气变化和近期污染累积趋势。
- 对 LSTM 足够长，但不会明显增加序列训练负担。
- 对 XGBoost 和 RandomForest，展平后是 `720 * 6 = 4320` 个特征，比 2160 小时窗口更适合作为默认对比基线。
- 72 小时预测需要较长上下文，但不必默认使用 90 天窗口；更长窗口可作为后续实验变量。

默认预测起点保持 `2026-03-01 00:00:00+08:00`。

## 输出目录结构

新结果统一写入窗口实验目录：

```text
Reproduce/outputs/window_720h_to_72h/
├── data/
│   ├── windows.npz
│   ├── scaler.json
│   └── data_config.json
├── models/
│   ├── lstm/
│   ├── xgboost/
│   ├── random_forest/
│   ├── arima/
│   └── sarima/
├── predictions/
│   └── start_2026_03_01_0000/
│       ├── lstm/
│       ├── xgboost/
│       ├── random_forest/
│       ├── arima/
│       └── sarima/
└── comparisons/
    └── start_2026_03_01_0000/
        ├── model_metrics.csv
        ├── model_metrics.json
        └── all_predictions.csv
```

目录职责：

- `data/` 只与 `input_window`、`output_window`、`predict_start` 和时间切分有关，所有模型共享。
- `models/<model_name>/` 保存模型文件、训练历史、训练配置、校准文件等训练产物。
- `predictions/start_<YYYY_MM_DD_HHMM>/<model_name>/` 保存单个模型在指定预测窗口上的预测结果。
- `comparisons/start_<YYYY_MM_DD_HHMM>/` 保存多模型汇总结果。

旧目录如 `Reproduce/outputs/lstm_720h_to_24h/` 不迁移、不删除，新脚本不再把它作为默认写入目标。

## 模型口径

`lstm`：

- 使用 6 个特征 `[temperature, humidity, wind_speed, precipitation, pressure, pm25]`。
- 输入形状为 `[sample_count, input_window, 6]`。
- 输出形状为 `[sample_count, output_window]`。
- 保持当前直接多输出 LSTM，不改成递归预测。
- 默认继续支持训练集 horizon 线性校准。

`xgboost`：

- 使用同 LSTM 一致的 6 个特征。
- 将输入窗口展平成二维特征：`[sample_count, input_window * 6]`。
- 使用 `xgboost.XGBRegressor` 配合 `sklearn.multioutput.MultiOutputRegressor` 做直接多输出预测。
- 如果当前环境缺少 `xgboost`，脚本直接报出清晰错误，不静默降级为其它模型。

`random_forest`：

- 使用同 LSTM 一致的 6 个特征。
- 将输入窗口展平成二维特征：`[sample_count, input_window * 6]`。
- 使用 `sklearn.ensemble.RandomForestRegressor` 的多输出回归能力。

`arima`：

- 只使用训练期历史 `pm25`，不使用温度、湿度、风速、降水、气压等外生变量。
- 不使用验证期或预测窗口真实值参与拟合。
- 预测未来 `output_window` 步，输出形状统一为 `[1, output_window]`。

`sarima`：

- 只使用训练期历史 `pm25`，不使用外生变量。
- 不使用验证期或预测窗口真实值参与拟合。
- 默认季节周期使用 `24`，表示小时数据的日周期。
- 预测未来 `output_window` 步，输出形状统一为 `[1, output_window]`。

## 单模型预测输出

每个模型的预测目录统一包含：

```text
predictions.csv
metrics.json
horizon_metrics.csv
horizon_metrics.json
stage_metrics.csv
stage_metrics.json
prediction_summary.json
plots/prediction_curve.png
plots/error_curve.png
plots/scatter.png
```

`predictions.csv` 字段固定为：

```text
model_name
sample_id
origin_timestamp
target_end_timestamp
timestamp
horizon
y_true
y_pred_model
y_pred
error
abs_error
relative_error
```

字段含义：

- `model_name`：模型名称，用于跨模型拼接。
- `sample_id`：预测样本编号，单窗口预测时为 `0`。
- `origin_timestamp`：预测目标窗口起始时间。
- `target_end_timestamp`：预测目标窗口结束时间。
- `timestamp`：当前 horizon 对应的目标小时。
- `horizon`：预测步长，从 `1` 到 `output_window`。
- `y_true`：预测窗口真实 PM2.5，只用于最终评价。
- `y_pred_model`：模型原始预测值。LSTM 为反归一化后的未校准输出；其它模型为直接输出。
- `y_pred`：最终用于评价的预测值。LSTM 默认可为校准后输出；其它模型默认等于 `y_pred_model`。
- `error`：`y_pred - y_true`。
- `abs_error`：`abs(error)`。
- `relative_error`：`abs_error / max(abs(y_true), 1.0)`。

## 比较汇总输出

比较脚本读取已存在的模型预测目录，生成：

```text
model_metrics.csv
model_metrics.json
all_predictions.csv
```

`model_metrics.csv` 每行一个模型，至少包含：

```text
model_name,RMSE,MAE,MAPE,SMAPE,R2,bias,prediction_dir
```

`all_predictions.csv` 直接拼接各模型的 `predictions.csv`，要求每行都有 `model_name`，便于筛选、画图和后续分析。

如果指定模型缺少预测目录或关键文件，比较脚本应给出缺失模型和缺失文件路径，并停止生成不完整汇总。

## 脚本入口

共享数据准备：

```powershell
python -m Reproduce.scripts.prepare_data --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

单模型训练：

```powershell
python -m Reproduce.scripts.train_model --model lstm --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00" --device cuda
python -m Reproduce.scripts.train_model --model xgboost --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model random_forest --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model arima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model sarima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

单模型预测：

```powershell
python -m Reproduce.scripts.predict_model --model lstm --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00" --device cuda
```

模型对比：

```powershell
python -m Reproduce.scripts.compare_models --models lstm xgboost random_forest arima sarima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

兼容策略：

- 保留 `train_lstm.py`、`predict_window.py`、`predict_month.py`。
- 这些旧入口作为 LSTM 别名或薄封装，输出也应进入新结构。
- 新通用入口是后续文档和实验的推荐入口。

## 文件拆分

计划新增或调整：

```text
Reproduce/models/tree_models.py
Reproduce/models/statistical_models.py
Reproduce/utils/paths.py
Reproduce/utils/prediction_io.py
Reproduce/scripts/train_model.py
Reproduce/scripts/predict_model.py
Reproduce/scripts/compare_models.py
```

职责：

- `paths.py`：统一生成窗口目录、模型目录、预测目录、比较目录。
- `prediction_io.py`：统一生成 `predictions.csv`、指标文件、分 horizon 指标、阶段指标和图像。
- `tree_models.py`：封装 XGBoost 和 RandomForest 的训练、保存、加载、预测。
- `statistical_models.py`：封装 ARIMA 和 SARIMA 的训练、保存、加载、预测。
- `train_model.py`：按 `--model` 分发到对应训练流程。
- `predict_model.py`：按 `--model` 分发到对应预测流程，并统一写出预测结果。
- `compare_models.py`：读取多个模型预测结果，生成汇总指标和拼接预测表。

## 验证设计

新增轻量测试目录 `tests/`，默认测试不运行完整深度学习训练。

测试范围：

- 路径规范：验证 `window_720h_to_72h/models/lstm`、`predictions/start_2026_03_01_0000/lstm`、`comparisons/start_2026_03_01_0000` 的生成。
- 预测写入：用小数组验证 `prediction_io` 输出固定字段、指标 JSON/CSV 和图像状态文件。
- 汇总脚本：构造两个模拟模型预测目录，验证能生成 `model_metrics.csv`、`model_metrics.json`、`all_predictions.csv`。
- 树模型：用小样本验证展平输入后输出形状为 `[1, output_window]`。
- 统计模型：用小序列验证 ARIMA/SARIMA 输出形状为 `[1, output_window]`，并且拟合数据只来自训练期 PM2.5。
- LSTM 兼容：保留 `train_lstm.py` 与 `predict_window.py` 的入口测试，验证它们写入新结构或调用新入口。

可选人工 smoke：

```powershell
python -m Reproduce.scripts.train_model --model random_forest --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00" --n-estimators 5
python -m Reproduce.scripts.train_model --model lstm --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00" --device cuda --epochs 1
```

## 文档更新

需要更新：

- `Reproduce/README.md`
- `Reproduce/REPRODUCTION_PLAN.md`

文档必须明确：

- 默认实验为 `720h -> 72h`。
- LSTM、XGBoost、RandomForest 使用 6 特征。
- ARIMA、SARIMA 只使用训练期历史 `pm25`，不使用未来真实值或外生变量。
- 新输出结构以 `window_<input>h_to_<output>h/` 为根。
- 旧 LSTM 目录是历史产物，新实验结果不再写入旧结构。

## 验收标准

- 数据准备默认写入 `Reproduce/outputs/window_720h_to_72h/data/`。
- 每个模型训练产物写入 `Reproduce/outputs/window_720h_to_72h/models/<model_name>/`。
- 每个模型预测结果写入 `Reproduce/outputs/window_720h_to_72h/predictions/start_2026_03_01_0000/<model_name>/`。
- 每个模型的 `predictions.csv` 都包含固定字段和 `72` 行。
- 汇总目录能生成每个模型一行的 `model_metrics.csv`。
- `all_predictions.csv` 能按 `model_name` 区分不同模型预测结果。
- ARIMA/SARIMA 的训练输入不包含验证期、预测窗口真实值或任何外生特征。
- 旧入口仍可运行，并且不会写回旧的 `lstm_<input>h_to_<output>h/` 默认结构。
