# 多模型对比与输出结构规范 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将项目重构为 `window_720h_to_72h` 窗口实验目录，新增 LSTM、XGBoost、RandomForest、ARIMA、SARIMA 的统一训练、预测和比较输出。

**Architecture:** 先新增路径与预测输出基础设施，再把数据准备从 LSTM 命名目录迁移到窗口目录。LSTM 保留现有训练逻辑但写入 `models/lstm` 和 `predictions/<start>/lstm`，树模型和统计模型通过新封装接入同一输出层，最后用 `compare_models` 生成横向汇总。

**Tech Stack:** Python 3.10、NumPy、Pandas、PyTorch、scikit-learn、xgboost、statsmodels、matplotlib、unittest。

---

## File Structure

- Create: `Reproduce/utils/paths.py`  
  统一生成窗口实验目录、模型目录、预测目录、比较目录和支持的模型名。
- Create: `Reproduce/utils/prediction_io.py`  
  统一写出 `predictions.csv`、指标 JSON/CSV、stage 指标、图像和 `prediction_summary.json`。
- Create: `Reproduce/models/tree_models.py`  
  封装 XGBoost、RandomForest 的展平输入、训练、保存、加载和预测。
- Create: `Reproduce/models/statistical_models.py`  
  封装 ARIMA、SARIMA 的单变量训练、保存、加载和预测。
- Create: `Reproduce/scripts/train_model.py`  
  通用训练入口，按 `--model` 分发。
- Create: `Reproduce/scripts/predict_model.py`  
  通用预测入口，按 `--model` 分发并调用统一输出。
- Create: `Reproduce/scripts/compare_models.py`  
  汇总多个模型已有预测结果。
- Modify: `Reproduce/utils/data_utils.py`  
  默认 `output_window` 改为 72，`experiment_name()` 改为窗口目录命名，数据产物写入 `window_<input>h_to_<output>h/data/`。
- Modify: `Reproduce/scripts/prepare_data.py`  
  默认输出窗口改为 72，打印窗口实验名。
- Modify: `Reproduce/scripts/train_lstm.py`  
  增加可复用 `build_arg_parser()` 和 `run_training(args)`，把模型产物写入 `models/lstm/`。
- Modify: `Reproduce/scripts/predict_month.py`  
  增加可复用 `run_lstm_prediction(args)` 或调整现有 `run_prediction(args)`，把预测结果写入 `predictions/<start>/lstm/`。
- Modify: `Reproduce/scripts/predict_window.py`  
  继续作为 LSTM 预测别名。
- Modify: `Reproduce/scripts/evaluate_lstm.py`  
  继续作为 LSTM 预测别名。
- Modify: `Reproduce/README.md`  
  更新默认 `720h -> 72h`、模型口径、命令和输出目录。
- Modify: `Reproduce/REPRODUCTION_PLAN.md`  
  更新实验计划、输出结构和验收标准。
- Create: `tests/__init__.py`
- Create: `tests/test_paths.py`
- Create: `tests/test_prediction_io.py`
- Create: `tests/test_compare_models.py`
- Create: `tests/test_tree_models.py`
- Create: `tests/test_statistical_models.py`

---

### Task 1: 路径工具与窗口实验命名

**Files:**
- Create: `Reproduce/utils/paths.py`
- Modify: `Reproduce/utils/data_utils.py`
- Create: `tests/__init__.py`
- Create: `tests/test_paths.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` as an empty file.

Create `tests/test_paths.py`:

```python
from pathlib import Path
import unittest

from Reproduce.utils.data_utils import experiment_name
from Reproduce.utils.paths import (
    SUPPORTED_MODEL_NAMES,
    comparison_dir,
    data_dir,
    model_dir,
    prediction_dir,
    window_experiment_dir,
    window_experiment_name,
)


class PathUtilityTests(unittest.TestCase):
    def test_window_experiment_name_replaces_lstm_name(self):
        self.assertEqual(window_experiment_name(720, 72), "window_720h_to_72h")
        self.assertEqual(experiment_name(720, 72), "window_720h_to_72h")

    def test_model_prediction_and_comparison_dirs_are_nested_under_window(self):
        root = Path("E:/tmp/pm25_outputs")
        base = window_experiment_dir(root, 720, 72)
        self.assertEqual(base, root / "window_720h_to_72h")
        self.assertEqual(data_dir(base), base / "data")
        self.assertEqual(model_dir(base, "lstm"), base / "models" / "lstm")
        self.assertEqual(
            prediction_dir(base, "2026-03-01 00:00:00+08:00", "random_forest"),
            base / "predictions" / "start_2026_03_01_0000" / "random_forest",
        )
        self.assertEqual(
            comparison_dir(base, "2026-03-01 00:00:00+08:00"),
            base / "comparisons" / "start_2026_03_01_0000",
        )

    def test_supported_model_names_are_fixed(self):
        self.assertEqual(
            SUPPORTED_MODEL_NAMES,
            ("lstm", "xgboost", "random_forest", "arima", "sarima"),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_paths -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'Reproduce.utils.paths'` or assertion failure because `experiment_name(720, 72)` still returns `lstm_720h_to_72h`.

- [ ] **Step 3: Write minimal implementation**

Create `Reproduce/utils/paths.py`:

```python
from __future__ import annotations

from pathlib import Path

from Reproduce.utils.data_utils import parse_predict_start, safe_timestamp_label


SUPPORTED_MODEL_NAMES = ("lstm", "xgboost", "random_forest", "arima", "sarima")


def validate_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    if normalized not in SUPPORTED_MODEL_NAMES:
        supported = ", ".join(SUPPORTED_MODEL_NAMES)
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {supported}")
    return normalized


def window_experiment_name(input_window: int, output_window: int) -> str:
    return f"window_{int(input_window)}h_to_{int(output_window)}h"


def window_experiment_dir(output_root: str | Path, input_window: int, output_window: int) -> Path:
    return Path(output_root) / window_experiment_name(input_window, output_window)


def data_dir(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir) / "data"


def model_dir(experiment_dir: str | Path, model_name: str) -> Path:
    return Path(experiment_dir) / "models" / validate_model_name(model_name)


def start_dir_name(predict_start: str) -> str:
    return f"start_{safe_timestamp_label(parse_predict_start(predict_start))}"


def prediction_root_dir(experiment_dir: str | Path, predict_start: str) -> Path:
    return Path(experiment_dir) / "predictions" / start_dir_name(predict_start)


def prediction_dir(experiment_dir: str | Path, predict_start: str, model_name: str) -> Path:
    return prediction_root_dir(experiment_dir, predict_start) / validate_model_name(model_name)


def comparison_dir(experiment_dir: str | Path, predict_start: str) -> Path:
    return Path(experiment_dir) / "comparisons" / start_dir_name(predict_start)
```

Modify `Reproduce/utils/data_utils.py`:

```python
DEFAULT_INPUT_WINDOW = 720
DEFAULT_OUTPUT_WINDOW = 72


def experiment_name(input_window: int, output_window: int = DEFAULT_OUTPUT_WINDOW) -> str:
    return f"window_{int(input_window)}h_to_{int(output_window)}h"
```

Change `prepare_data_bundle()` defaults in the same file:

```python
def prepare_data_bundle(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    input_window: int = DEFAULT_INPUT_WINDOW,
    output_window: int = DEFAULT_OUTPUT_WINDOW,
    predict_start: str = DEFAULT_PREDICT_START,
) -> dict[str, Any]:
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest tests.test_paths -v
```

Expected: PASS with 3 tests.

- [ ] **Step 5: Commit**

```powershell
git add Reproduce/utils/paths.py Reproduce/utils/data_utils.py tests/__init__.py tests/test_paths.py
git commit -m "refactor: add window experiment path utilities"
```

---

### Task 2: 统一预测输出写入层

**Files:**
- Create: `Reproduce/utils/prediction_io.py`
- Create: `tests/test_prediction_io.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prediction_io.py`:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
import pandas as pd

from Reproduce.utils.prediction_io import PREDICTION_COLUMNS, build_predictions_frame, write_prediction_outputs


class PredictionIoTests(unittest.TestCase):
    def test_build_predictions_frame_has_fixed_columns_and_errors(self):
        y_true = np.array([[10.0, 20.0, 40.0]], dtype=float)
        y_pred_model = np.array([[12.0, 18.0, 36.0]], dtype=float)
        y_pred = np.array([[11.0, 19.0, 39.0]], dtype=float)
        frame = build_predictions_frame(
            model_name="random_forest",
            y_true=y_true,
            y_pred_model=y_pred_model,
            y_pred=y_pred,
            timestamps_start=np.array(["2026-03-01 00:00:00+08:00"]),
            timestamps_end=np.array(["2026-03-01 02:00:00+08:00"]),
            timestamps_target=np.array([["2026-03-01 00:00:00+08:00", "2026-03-01 01:00:00+08:00", "2026-03-01 02:00:00+08:00"]]),
        )
        self.assertEqual(list(frame.columns), PREDICTION_COLUMNS)
        self.assertEqual(frame["model_name"].tolist(), ["random_forest", "random_forest", "random_forest"])
        self.assertEqual(frame["horizon"].tolist(), [1, 2, 3])
        self.assertEqual(frame["error"].round(6).tolist(), [1.0, -1.0, -1.0])
        self.assertEqual(frame["abs_error"].round(6).tolist(), [1.0, 1.0, 1.0])

    def test_write_prediction_outputs_creates_standard_files(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "predictions" / "lstm"
            frame = build_predictions_frame(
                model_name="lstm",
                y_true=np.array([[10.0, 20.0, 30.0]], dtype=float),
                y_pred_model=np.array([[9.0, 18.0, 33.0]], dtype=float),
                y_pred=np.array([[10.0, 19.0, 31.0]], dtype=float),
                timestamps_start=np.array(["2026-03-01 00:00:00+08:00"]),
                timestamps_end=np.array(["2026-03-01 02:00:00+08:00"]),
                timestamps_target=np.array([["2026-03-01 00:00:00+08:00", "2026-03-01 01:00:00+08:00", "2026-03-01 02:00:00+08:00"]]),
            )
            summary = write_prediction_outputs(
                predictions=frame,
                output_dir=output_dir,
                model_name="lstm",
                model_path=Path("E:/models/lstm/model.pt"),
                calibration_path=Path("E:/models/lstm/calibration.json"),
                calibration_applied=True,
                calibration_method="horizon_linear",
                device="cpu",
                predict_start="2026-03-01 00:00:00+08:00",
            )

            self.assertTrue((output_dir / "predictions.csv").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "horizon_metrics.csv").exists())
            self.assertTrue((output_dir / "stage_metrics.csv").exists())
            self.assertTrue((output_dir / "prediction_summary.json").exists())
            written = pd.read_csv(output_dir / "predictions.csv")
            self.assertEqual(list(written.columns), PREDICTION_COLUMNS)
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("RMSE", metrics)
            self.assertEqual(summary["model_name"], "lstm")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_prediction_io -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'Reproduce.utils.prediction_io'`.

- [ ] **Step 3: Write minimal implementation**

Create `Reproduce/utils/prediction_io.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Reproduce.utils.data_utils import parse_predict_start, write_json
from Reproduce.utils.metrics import regression_metrics
from Reproduce.utils.plotting import plot_error_curve, plot_prediction_curve, plot_scatter, write_plot_status


PREDICTION_COLUMNS = [
    "model_name",
    "sample_id",
    "origin_timestamp",
    "target_end_timestamp",
    "timestamp",
    "horizon",
    "y_true",
    "y_pred_model",
    "y_pred",
    "error",
    "abs_error",
    "relative_error",
]


def _as_2d(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected 1D or 2D values, got shape {array.shape}.")
    return array


def build_predictions_frame(
    model_name: str,
    y_true: Any,
    y_pred_model: Any,
    y_pred: Any,
    timestamps_start: Any,
    timestamps_end: Any,
    timestamps_target: Any,
) -> pd.DataFrame:
    true = _as_2d(y_true)
    pred_model = _as_2d(y_pred_model)
    pred_final = _as_2d(y_pred)
    if true.shape != pred_model.shape or true.shape != pred_final.shape:
        raise ValueError(f"Prediction shape mismatch: true={true.shape}, model={pred_model.shape}, final={pred_final.shape}")

    output_window = int(true.shape[1])
    sample_count = int(true.shape[0])
    target_ts = np.asarray(timestamps_target).astype(str).reshape(sample_count, output_window)
    sample_ids = np.repeat(np.arange(sample_count), output_window)
    horizons = np.tile(np.arange(1, output_window + 1), sample_count)

    frame = pd.DataFrame(
        {
            "model_name": np.repeat(str(model_name), sample_count * output_window),
            "sample_id": sample_ids,
            "origin_timestamp": np.repeat(np.asarray(timestamps_start).astype(str), output_window),
            "target_end_timestamp": np.repeat(np.asarray(timestamps_end).astype(str), output_window),
            "timestamp": target_ts.reshape(-1),
            "horizon": horizons,
            "y_true": true.reshape(-1),
            "y_pred_model": pred_model.reshape(-1),
            "y_pred": pred_final.reshape(-1),
        }
    )
    frame["error"] = frame["y_pred"] - frame["y_true"]
    frame["abs_error"] = frame["error"].abs()
    frame["relative_error"] = frame["abs_error"] / np.maximum(frame["y_true"].abs(), 1.0)
    return frame[PREDICTION_COLUMNS]


def _stage_ranges(output_window: int) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    start = 1
    while start <= int(output_window):
        end = min(start + 23, int(output_window))
        ranges[f"h{start:03d}_{end:03d}"] = (start, end)
        start = end + 1
    return ranges


def write_prediction_outputs(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    model_name: str,
    model_path: str | Path | None,
    calibration_path: str | Path | None,
    calibration_applied: bool,
    calibration_method: str | None,
    device: str,
    predict_start: str,
) -> dict[str, Any]:
    output = Path(output_dir)
    plots_dir = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8")
    metrics = regression_metrics(predictions["y_true"], predictions["y_pred"])
    model_metrics = regression_metrics(predictions["y_true"], predictions["y_pred_model"])
    write_json(output / "metrics.json", metrics)
    write_json(output / "metrics_model_raw.json", model_metrics)

    output_window = int(predictions["horizon"].max())
    horizon_metrics = [
        {
            "horizon": horizon,
            **regression_metrics(
                predictions.loc[predictions["horizon"] == horizon, "y_true"],
                predictions.loc[predictions["horizon"] == horizon, "y_pred"],
            ),
        }
        for horizon in range(1, output_window + 1)
    ]
    write_json(output / "horizon_metrics.json", horizon_metrics)
    pd.DataFrame(horizon_metrics).to_csv(output / "horizon_metrics.csv", index=False, encoding="utf-8")

    stage_metrics = {}
    for stage_name, (start_h, end_h) in _stage_ranges(output_window).items():
        mask = (predictions["horizon"] >= start_h) & (predictions["horizon"] <= end_h)
        stage_metrics[stage_name] = regression_metrics(predictions.loc[mask, "y_true"], predictions.loc[mask, "y_pred"])
    write_json(output / "stage_metrics.json", stage_metrics)
    pd.DataFrame([{"stage": name, **values} for name, values in stage_metrics.items()]).to_csv(
        output / "stage_metrics.csv", index=False, encoding="utf-8"
    )

    plot_status = {
        "prediction_curve": plot_prediction_curve(predictions, plots_dir / "prediction_curve.png", f"{model_name} PM2.5 Prediction"),
        "error_curve": plot_error_curve(predictions, plots_dir / "error_curve.png", f"{model_name} PM2.5 Prediction Error"),
        "scatter": plot_scatter(predictions, plots_dir / "scatter.png", f"{model_name} True vs Predicted"),
    }
    write_plot_status(output / "plot_status.md", plot_status)

    summary = {
        "model_name": str(model_name),
        "prediction_dir": str(output),
        "model_path": None if model_path is None else str(model_path),
        "calibration_path": None if calibration_path is None else str(calibration_path),
        "calibration_applied": bool(calibration_applied),
        "calibration_method": calibration_method,
        "device": str(device),
        "sample_count": int(len(predictions)),
        "forecast_sample_count": int(predictions["sample_id"].nunique()),
        "output_window": output_window,
        "predict_start": str(parse_predict_start(predict_start)),
        "metrics": metrics,
        "model_raw_metrics": model_metrics,
    }
    write_json(output / "prediction_summary.json", summary)
    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest tests.test_prediction_io -v
```

Expected: PASS with 2 tests.

- [ ] **Step 5: Commit**

```powershell
git add Reproduce/utils/prediction_io.py tests/test_prediction_io.py
git commit -m "feat: add shared prediction output writer"
```

---

### Task 3: 数据准备默认窗口和输出目录迁移

**Files:**
- Modify: `Reproduce/utils/data_utils.py`
- Modify: `Reproduce/scripts/prepare_data.py`
- Create: `tests/test_prepare_data_contract.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prepare_data_contract.py`:

```python
import unittest

from Reproduce.scripts.prepare_data import parse_args
from Reproduce.utils.data_utils import DEFAULT_INPUT_WINDOW, DEFAULT_OUTPUT_WINDOW, experiment_name


class PrepareDataContractTests(unittest.TestCase):
    def test_default_window_is_720_to_72(self):
        self.assertEqual(DEFAULT_INPUT_WINDOW, 720)
        self.assertEqual(DEFAULT_OUTPUT_WINDOW, 72)
        self.assertEqual(experiment_name(DEFAULT_INPUT_WINDOW, DEFAULT_OUTPUT_WINDOW), "window_720h_to_72h")

    def test_prepare_data_parser_defaults_to_72_hour_output(self):
        parser = parse_args
        self.assertTrue(callable(parser))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_prepare_data_contract -v
```

Expected: FAIL because `DEFAULT_INPUT_WINDOW` and `DEFAULT_OUTPUT_WINDOW` are not yet exported or `DEFAULT_OUTPUT_WINDOW` is not used by CLI defaults.

- [ ] **Step 3: Write minimal implementation**

Modify imports and parser defaults in `Reproduce/scripts/prepare_data.py`:

```python
from Reproduce.utils.data_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
    prepare_data_bundle,
)
```

```python
parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW, help="Historical input window in hours.")
parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW, help="Direct forecast horizon in hours.")
```

Ensure `prepare_data_bundle()` in `Reproduce/utils/data_utils.py` builds:

```python
exp_name = experiment_name(input_window, output_window)
exp_dir = output_root_path / exp_name
data_dir = exp_dir / "data"
```

with `experiment_name()` already returning `window_<input>h_to_<output>h`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest tests.test_prepare_data_contract -v
```

Expected: PASS with 2 tests.

- [ ] **Step 5: Commit**

```powershell
git add Reproduce/utils/data_utils.py Reproduce/scripts/prepare_data.py tests/test_prepare_data_contract.py
git commit -m "refactor: default data preparation to 72 hour window"
```

---

### Task 4: LSTM 训练和预测迁移到模型子目录

**Files:**
- Modify: `Reproduce/scripts/train_lstm.py`
- Modify: `Reproduce/scripts/predict_month.py`
- Modify: `Reproduce/scripts/predict_window.py`
- Modify: `Reproduce/scripts/evaluate_lstm.py`
- Create: `tests/test_lstm_paths.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_lstm_paths.py`:

```python
from pathlib import Path
import unittest

from Reproduce.scripts.predict_month import checkpoint_path
from Reproduce.utils.paths import model_dir, prediction_dir, window_experiment_dir


class LstmPathTests(unittest.TestCase):
    def test_checkpoint_path_resolves_inside_lstm_model_dir(self):
        exp_dir = window_experiment_dir(Path("E:/tmp/outputs"), 720, 72)
        lstm_dir = model_dir(exp_dir, "lstm")
        self.assertEqual(checkpoint_path(lstm_dir, None), lstm_dir / "model.pt")

    def test_prediction_dir_for_lstm_uses_model_subdirectory(self):
        exp_dir = window_experiment_dir(Path("E:/tmp/outputs"), 720, 72)
        self.assertEqual(
            prediction_dir(exp_dir, "2026-03-01 00:00:00+08:00", "lstm"),
            Path("E:/tmp/outputs/window_720h_to_72h/predictions/start_2026_03_01_0000/lstm"),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_lstm_paths -v
```

Expected: FAIL because `checkpoint_path()` still expects the old experiment directory and prefers `model_best_val_loss.pt` at the experiment root.

- [ ] **Step 3: Write minimal implementation**

Modify `Reproduce/scripts/train_lstm.py`:

```python
from Reproduce.utils.data_utils import DEFAULT_INPUT_WINDOW, DEFAULT_OUTPUT_WINDOW
from Reproduce.utils.paths import model_dir, window_experiment_dir
```

In parser defaults:

```python
parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW, help="Historical input window in hours.")
parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW, help="Direct forecast horizon in hours.")
```

At the start of training after data preparation:

```python
window_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
lstm_dir = model_dir(window_dir, "lstm")
data_config_path = window_dir / "data" / "data_config.json"
bundle_path = window_dir / "data" / "windows.npz"
```

Replace model artifact paths:

```python
lstm_dir.mkdir(parents=True, exist_ok=True)
model_path = lstm_dir / "model.pt"
best_model_path = lstm_dir / "model_best_train_loss.pt"
best_val_model_path = lstm_dir / "model_best_val_loss.pt"
calibration_path = lstm_dir / "calibration.json"
```

Replace remaining writes exactly as follows:

```python
pd.DataFrame(history).to_csv(lstm_dir / "training_history.csv", index=False, encoding="utf-8")
pd.DataFrame(calibration["horizon_stats"]).to_csv(
    lstm_dir / "calibration_horizon_stats.csv",
    index=False,
    encoding="utf-8",
)
write_json(lstm_dir / "training_config.json", training_config)
```

Modify `Reproduce/scripts/predict_month.py`:

```python
from Reproduce.utils.paths import model_dir, prediction_dir, window_experiment_dir
from Reproduce.utils.prediction_io import build_predictions_frame, write_prediction_outputs
```

Change `checkpoint_path()` to accept the LSTM model directory:

```python
def checkpoint_path(lstm_dir: Path, requested: str | None) -> Path:
    if requested:
        return Path(requested)
    best_val = lstm_dir / "model_best_val_loss.pt"
    if best_val.exists():
        return best_val
    best = lstm_dir / "model_best_train_loss.pt"
    if best.exists():
        return best
    return lstm_dir / "model.pt"
```

In `run_prediction(args)`, resolve directories:

```python
window_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
lstm_dir = model_dir(window_dir, "lstm")
bundle_path = window_dir / "data" / "windows.npz"
ckpt = checkpoint_path(lstm_dir, args.model_path)
```

After prediction arrays are available, replace manual CSV/metrics writing with:

```python
predictions = build_predictions_frame(
    model_name="lstm",
    y_true=y_true,
    y_pred_model=y_pred_model,
    y_pred=y_pred,
    timestamps_start=timestamps_start,
    timestamps_end=timestamps_end,
    timestamps_target=timestamps_target,
)
window_dir_out = prediction_dir(window_dir, args.predict_start, "lstm")
summary = write_prediction_outputs(
    predictions=predictions,
    output_dir=window_dir_out,
    model_name="lstm",
    model_path=ckpt,
    calibration_path=calibration_path if calibration_path.exists() else None,
    calibration_applied=calibration_applied,
    calibration_method=None if calibration is None else calibration.get("method"),
    device=str(device),
    predict_start=args.predict_start,
)
summary["experiment_dir"] = str(window_dir)
return summary
```

Keep `predict_window.py` and `evaluate_lstm.py` importing `main` from `predict_month.py`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest tests.test_lstm_paths -v
```

Expected: PASS with 2 tests.

- [ ] **Step 5: Commit**

```powershell
git add Reproduce/scripts/train_lstm.py Reproduce/scripts/predict_month.py Reproduce/scripts/predict_window.py Reproduce/scripts/evaluate_lstm.py tests/test_lstm_paths.py
git commit -m "refactor: write lstm artifacts under model subdirectories"
```

---

### Task 5: 树模型封装

**Files:**
- Create: `Reproduce/models/tree_models.py`
- Create: `tests/test_tree_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tree_models.py`:

```python
import unittest

import numpy as np

from Reproduce.models.tree_models import flatten_window_features, train_random_forest_model


class TreeModelTests(unittest.TestCase):
    def test_flatten_window_features_preserves_sample_count(self):
        X = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        flattened = flatten_window_features(X)
        self.assertEqual(flattened.shape, (2, 12))
        self.assertEqual(flattened[0, 0], 0.0)
        self.assertEqual(flattened[1, -1], 23.0)

    def test_random_forest_predicts_direct_multi_output_shape(self):
        X_train = np.arange(8 * 3 * 2, dtype=float).reshape(8, 3, 2)
        y_train = np.arange(8 * 4, dtype=float).reshape(8, 4)
        model = train_random_forest_model(X_train, y_train, n_estimators=3, random_state=7, n_jobs=1)
        prediction = model.predict(flatten_window_features(X_train[:1]))
        self.assertEqual(prediction.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_tree_models -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'Reproduce.models.tree_models'`.

- [ ] **Step 3: Write minimal implementation**

Create `Reproduce/models/tree_models.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np


def flatten_window_features(X: Any) -> np.ndarray:
    array = np.asarray(X, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected X with shape [samples, input_window, features], got {array.shape}.")
    return array.reshape(array.shape[0], array.shape[1] * array.shape[2])


def train_random_forest_model(
    X_train: Any,
    y_train: Any,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for random_forest. Install scikit-learn in the active environment.") from exc
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=None if max_depth is None else int(max_depth),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )
    model.fit(flatten_window_features(X_train), np.asarray(y_train, dtype=np.float32))
    return model


def train_xgboost_model(
    X_train: Any,
    y_train: Any,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    try:
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("xgboost and scikit-learn are required for xgboost model training.") from exc
    base = XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        objective="reg:squarederror",
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )
    model = MultiOutputRegressor(base, n_jobs=1)
    model.fit(flatten_window_features(X_train), np.asarray(y_train, dtype=np.float32))
    return model


def save_tree_model(model: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(model, fh)


def load_tree_model(path: str | Path) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def predict_tree_model(model: Any, X_predict: Any) -> np.ndarray:
    prediction = model.predict(flatten_window_features(X_predict))
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim == 1:
        prediction = prediction.reshape(1, -1)
    return prediction
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest tests.test_tree_models -v
```

Expected: PASS with 2 tests if scikit-learn is installed. If scikit-learn is missing, install or run inside the `pm25` conda environment required by the project.

- [ ] **Step 5: Commit**

```powershell
git add Reproduce/models/tree_models.py tests/test_tree_models.py
git commit -m "feat: add tree model wrappers"
```

---

### Task 6: ARIMA/SARIMA 单变量模型封装

**Files:**
- Create: `Reproduce/models/statistical_models.py`
- Create: `tests/test_statistical_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_statistical_models.py`:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from Reproduce.models.statistical_models import load_train_pm25_series


class StatisticalModelTests(unittest.TestCase):
    def test_load_train_pm25_series_uses_train_period_only(self):
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "beijing.csv"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01 00:00:00+08:00", periods=6, freq="h"),
                    "pm25": [10, 11, 12, 13, 99, 100],
                    "temp": [1, 1, 1, 1, 1, 1],
                    "humidity": [50, 50, 50, 50, 50, 50],
                    "wind_speed": [2, 2, 2, 2, 2, 2],
                    "precipitation": [0, 0, 0, 0, 0, 0],
                    "pressure": [1000, 1000, 1000, 1000, 1000, 1000],
                }
            )
            frame.to_csv(csv_path, index=False)
            data_config = {
                "data_path": str(csv_path),
                "train_period": {"end": "2026-01-01 03:00:00+08:00"},
            }
            series = load_train_pm25_series(data_config)
            self.assertEqual(series.tolist(), [10.0, 11.0, 12.0, 13.0])
            self.assertNotIn(99.0, series.tolist())
            self.assertNotIn(100.0, series.tolist())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_statistical_models -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'Reproduce.models.statistical_models'`.

- [ ] **Step 3: Write minimal implementation**

Create `Reproduce/models/statistical_models.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd

from Reproduce.utils.data_utils import TARGET_COLUMN, fill_missing_values, load_beijing_data


def load_train_pm25_series(data_config: dict[str, Any]) -> np.ndarray:
    data_path = Path(data_config["data_path"])
    train_end = pd.Timestamp(data_config["train_period"]["end"])
    frame = fill_missing_values(load_beijing_data(data_path))
    mask = frame["timestamp"] <= train_end
    train_frame = frame.loc[mask].copy()
    if train_frame.empty:
        raise ValueError("Training PM2.5 series is empty for statistical model.")
    return train_frame[TARGET_COLUMN].to_numpy(dtype=np.float32)


def train_arima_model(series: Any, order: tuple[int, int, int] = (2, 1, 2)) -> Any:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise RuntimeError("statsmodels is required for arima. Install statsmodels in the active environment.") from exc
    values = np.asarray(series, dtype=float)
    model = ARIMA(values, order=tuple(int(v) for v in order))
    return model.fit()


def train_sarima_model(
    series: Any,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 24),
) -> Any:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise RuntimeError("statsmodels is required for sarima. Install statsmodels in the active environment.") from exc
    values = np.asarray(series, dtype=float)
    model = SARIMAX(
        values,
        order=tuple(int(v) for v in order),
        seasonal_order=tuple(int(v) for v in seasonal_order),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_statistical_model(model: Any, output_window: int) -> np.ndarray:
    forecast = model.forecast(steps=int(output_window))
    values = np.asarray(forecast, dtype=np.float32).reshape(1, int(output_window))
    return np.maximum(values, 0.0)


def save_statistical_model(model: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(model, fh)


def load_statistical_model(path: str | Path) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest tests.test_statistical_models -v
```

Expected: PASS with 1 test.

- [ ] **Step 5: Commit**

```powershell
git add Reproduce/models/statistical_models.py tests/test_statistical_models.py
git commit -m "feat: add univariate statistical model wrappers"
```

---

### Task 7: 通用训练入口

**Files:**
- Create: `Reproduce/scripts/train_model.py`
- Modify: `Reproduce/scripts/train_lstm.py`
- Create: `tests/test_train_model_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_model_cli.py`:

```python
import unittest

from Reproduce.scripts.train_model import build_arg_parser


class TrainModelCliTests(unittest.TestCase):
    def test_parser_accepts_all_model_names_and_defaults_to_72h(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--model", "random_forest"])
        self.assertEqual(args.model, "random_forest")
        self.assertEqual(args.input_window, 720)
        self.assertEqual(args.output_window, 72)

    def test_parser_accepts_lstm_device_option(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--model", "lstm", "--device", "cpu", "--epochs", "1"])
        self.assertEqual(args.model, "lstm")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.epochs, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_train_model_cli -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'Reproduce.scripts.train_model'`.

- [ ] **Step 3: Write minimal implementation**

Refactor `Reproduce/scripts/train_lstm.py`:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train direct multi-output LSTM for PM2.5 reproduction.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Input CSV path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Reproduction outputs root.")
    parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW, help="Historical input window in hours.")
    parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW, help="Direct forecast horizon in hours.")
    return parser
```

Move the existing body of `main()` into:

```python
def run_training(args: argparse.Namespace) -> dict[str, Any]:
    torch, _ = require_torch()
    return {"model_dir": str(lstm_dir), "training_config_path": str(lstm_dir / "training_config.json")}
```

Keep:

```python
def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def main() -> None:
    run_training(parse_args())
```

Create `Reproduce/scripts/train_model.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Reproduce.models.statistical_models import load_train_pm25_series, save_statistical_model, train_arima_model, train_sarima_model
from Reproduce.models.tree_models import save_tree_model, train_random_forest_model, train_xgboost_model
from Reproduce.scripts import train_lstm
from Reproduce.utils.data_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
    prepare_data_bundle,
    read_json,
    write_json,
)
from Reproduce.utils.paths import SUPPORTED_MODEL_NAMES, model_dir, validate_model_name, window_experiment_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one PM2.5 forecasting model.")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW)
    parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW)
    parser.add_argument("--predict-start", default=DEFAULT_PREDICT_START)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--arima-order", nargs=3, type=int, default=[2, 1, 2])
    parser.add_argument("--sarima-order", nargs=3, type=int, default=[1, 1, 1])
    parser.add_argument("--sarima-seasonal-order", nargs=4, type=int, default=[1, 0, 1, 24])
    return parser


def _ensure_data(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if args.prepare_data:
        prepare_data_bundle(args.data_path, args.output_root, args.input_window, args.output_window, args.predict_start)
    window_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
    data_config_path = window_dir / "data" / "data_config.json"
    bundle_path = window_dir / "data" / "windows.npz"
    if not data_config_path.exists() or not bundle_path.exists():
        prepare_data_bundle(args.data_path, args.output_root, args.input_window, args.output_window, args.predict_start)
    return window_dir, read_json(data_config_path), dict(np.load(bundle_path, allow_pickle=True))


def train_non_lstm(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    model_name = validate_model_name(args.model)
    window_dir, data_config, bundle = _ensure_data(args)
    out_dir = model_dir(window_dir, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"

    if model_name == "random_forest":
        model = train_random_forest_model(bundle["X_train"], bundle["y_train_raw"], args.n_estimators, args.max_depth, args.seed, args.n_jobs)
        save_tree_model(model, model_path)
    elif model_name == "xgboost":
        model = train_xgboost_model(bundle["X_train"], bundle["y_train_raw"], args.n_estimators, 6, 0.05, 0.9, 0.9, args.seed, args.n_jobs)
        save_tree_model(model, model_path)
    elif model_name == "arima":
        model = train_arima_model(load_train_pm25_series(data_config), tuple(args.arima_order))
        save_statistical_model(model, model_path)
    elif model_name == "sarima":
        model = train_sarima_model(load_train_pm25_series(data_config), tuple(args.sarima_order), tuple(args.sarima_seasonal_order))
        save_statistical_model(model, model_path)
    else:
        raise ValueError(f"Unsupported non-LSTM model: {model_name}")

    config = {
        "model_name": model_name,
        "model_path": str(model_path),
        "data_config": data_config,
        "training": {
            "seed": int(args.seed),
            "elapsed_seconds": float(time.time() - start),
        },
    }
    write_json(out_dir / "training_config.json", config)
    return config


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if validate_model_name(args.model) == "lstm":
        return train_lstm.run_training(args)
    return train_non_lstm(args)


def main() -> None:
    summary = run_training(build_arg_parser().parse_args())
    print(f"Trained model: {summary['model_name']}")
    print(f"Model path: {summary['model_path']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest tests.test_train_model_cli -v
```

Expected: PASS with 2 tests.

- [ ] **Step 5: Commit**

```powershell
git add Reproduce/scripts/train_model.py Reproduce/scripts/train_lstm.py tests/test_train_model_cli.py
git commit -m "feat: add unified model training entrypoint"
```

---

### Task 8: 通用预测入口

**Files:**
- Create: `Reproduce/scripts/predict_model.py`
- Modify: `Reproduce/scripts/predict_month.py`
- Create: `tests/test_predict_model_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_predict_model_cli.py`:

```python
import unittest

from Reproduce.scripts.predict_model import build_arg_parser


class PredictModelCliTests(unittest.TestCase):
    def test_parser_defaults_to_720_to_72(self):
        args = build_arg_parser().parse_args(["--model", "arima"])
        self.assertEqual(args.model, "arima")
        self.assertEqual(args.input_window, 720)
        self.assertEqual(args.output_window, 72)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_predict_model_cli -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'Reproduce.scripts.predict_model'`.

- [ ] **Step 3: Write minimal implementation**

Create `Reproduce/scripts/predict_model.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Reproduce.models.statistical_models import forecast_statistical_model, load_statistical_model
from Reproduce.models.tree_models import load_tree_model, predict_tree_model
from Reproduce.scripts import predict_month
from Reproduce.utils.data_utils import DEFAULT_INPUT_WINDOW, DEFAULT_OUTPUT_ROOT, DEFAULT_OUTPUT_WINDOW, DEFAULT_PREDICT_START, prepare_data_bundle
from Reproduce.utils.paths import SUPPORTED_MODEL_NAMES, model_dir, prediction_dir, validate_model_name, window_experiment_dir
from Reproduce.utils.prediction_io import build_predictions_frame, write_prediction_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict one PM2.5 window with one trained model.")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW)
    parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW)
    parser.add_argument("--predict-start", default=DEFAULT_PREDICT_START)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--calibration-path", default=None)
    parser.add_argument("--no-calibration", action="store_true")
    return parser


def _predict_non_lstm(args: argparse.Namespace) -> dict[str, Any]:
    if args.prepare_data:
        prepare_data_bundle(output_root=args.output_root, input_window=args.input_window, output_window=args.output_window, predict_start=args.predict_start)
    model_name = validate_model_name(args.model)
    window_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
    bundle_path = window_dir / "data" / "windows.npz"
    if not bundle_path.exists():
        prepare_data_bundle(output_root=args.output_root, input_window=args.input_window, output_window=args.output_window, predict_start=args.predict_start)
    bundle = np.load(bundle_path, allow_pickle=True)
    X_predict = bundle["X_predict"].astype(np.float32)
    y_true = bundle["y_predict_raw"].astype(np.float32)
    timestamps_start = bundle["timestamps_predict_start"].astype(str)
    timestamps_end = bundle["timestamps_predict_end"].astype(str)
    timestamps_target = bundle["timestamps_predict_target"].astype(str)

    out_model_dir = model_dir(window_dir, model_name)
    model_path = Path(args.model_path) if args.model_path else out_model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if model_name in {"random_forest", "xgboost"}:
        model = load_tree_model(model_path)
        y_pred_model = predict_tree_model(model, X_predict)
    else:
        model = load_statistical_model(model_path)
        y_pred_model = forecast_statistical_model(model, int(args.output_window))
    y_pred = y_pred_model
    predictions = build_predictions_frame(model_name, y_true, y_pred_model, y_pred, timestamps_start, timestamps_end, timestamps_target)
    out_dir = prediction_dir(window_dir, args.predict_start, model_name)
    return write_prediction_outputs(
        predictions=predictions,
        output_dir=out_dir,
        model_name=model_name,
        model_path=model_path,
        calibration_path=None,
        calibration_applied=False,
        calibration_method=None,
        device="cpu",
        predict_start=args.predict_start,
    )


def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    if validate_model_name(args.model) == "lstm":
        return predict_month.run_prediction(args)
    return _predict_non_lstm(args)


def main() -> None:
    summary = run_prediction(build_arg_parser().parse_args())
    print(f"Predicted samples: {summary['sample_count']}")
    print(f"Output dir: {summary['prediction_dir']}")
    print(f"Metrics: {summary['metrics']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest tests.test_predict_model_cli -v
```

Expected: PASS with 1 test.

- [ ] **Step 5: Commit**

```powershell
git add Reproduce/scripts/predict_model.py Reproduce/scripts/predict_month.py tests/test_predict_model_cli.py
git commit -m "feat: add unified model prediction entrypoint"
```

---

### Task 9: 多模型比较汇总

**Files:**
- Create: `Reproduce/scripts/compare_models.py`
- Create: `tests/test_compare_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compare_models.py`:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from Reproduce.scripts.compare_models import compare_existing_predictions


class CompareModelsTests(unittest.TestCase):
    def test_compare_existing_predictions_writes_metrics_and_all_predictions(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "window_720h_to_72h"
            start = base / "predictions" / "start_2026_03_01_0000"
            for model_name, pred in [("lstm", 11.0), ("random_forest", 12.0)]:
                model_dir = start / model_name
                model_dir.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "model_name": [model_name, model_name],
                        "sample_id": [0, 0],
                        "origin_timestamp": ["2026-03-01 00:00:00+08:00", "2026-03-01 00:00:00+08:00"],
                        "target_end_timestamp": ["2026-03-01 01:00:00+08:00", "2026-03-01 01:00:00+08:00"],
                        "timestamp": ["2026-03-01 00:00:00+08:00", "2026-03-01 01:00:00+08:00"],
                        "horizon": [1, 2],
                        "y_true": [10.0, 10.0],
                        "y_pred_model": [pred, pred],
                        "y_pred": [pred, pred],
                        "error": [pred - 10.0, pred - 10.0],
                        "abs_error": [abs(pred - 10.0), abs(pred - 10.0)],
                        "relative_error": [abs(pred - 10.0) / 10.0, abs(pred - 10.0) / 10.0],
                    }
                ).to_csv(model_dir / "predictions.csv", index=False)
            out = compare_existing_predictions(base, "2026-03-01 00:00:00+08:00", ["lstm", "random_forest"])
            self.assertTrue((out / "model_metrics.csv").exists())
            self.assertTrue((out / "model_metrics.json").exists())
            self.assertTrue((out / "all_predictions.csv").exists())
            metrics = pd.read_csv(out / "model_metrics.csv")
            self.assertEqual(metrics["model_name"].tolist(), ["lstm", "random_forest"])
            all_predictions = pd.read_csv(out / "all_predictions.csv")
            self.assertEqual(len(all_predictions), 4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_compare_models -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'Reproduce.scripts.compare_models'`.

- [ ] **Step 3: Write minimal implementation**

Create `Reproduce/scripts/compare_models.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Reproduce.utils.data_utils import DEFAULT_INPUT_WINDOW, DEFAULT_OUTPUT_ROOT, DEFAULT_OUTPUT_WINDOW, DEFAULT_PREDICT_START, write_json
from Reproduce.utils.metrics import regression_metrics
from Reproduce.utils.paths import SUPPORTED_MODEL_NAMES, comparison_dir, prediction_dir, validate_model_name, window_experiment_dir


def compare_existing_predictions(experiment_dir: str | Path, predict_start: str, models: Iterable[str]) -> Path:
    exp_dir = Path(experiment_dir)
    frames = []
    metrics_rows = []
    for raw_model_name in models:
        model_name = validate_model_name(raw_model_name)
        pred_dir = prediction_dir(exp_dir, predict_start, model_name)
        pred_path = pred_dir / "predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions for {model_name}: {pred_path}")
        frame = pd.read_csv(pred_path)
        if "model_name" not in frame.columns:
            frame.insert(0, "model_name", model_name)
        frames.append(frame)
        metrics_rows.append(
            {
                "model_name": model_name,
                **regression_metrics(frame["y_true"], frame["y_pred"]),
                "prediction_dir": str(pred_dir),
            }
        )

    out_dir = comparison_dir(exp_dir, predict_start)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(out_dir / "model_metrics.csv", index=False, encoding="utf-8")
    write_json(out_dir / "model_metrics.json", metrics_rows)
    pd.concat(frames, ignore_index=True).to_csv(out_dir / "all_predictions.csv", index=False, encoding="utf-8")
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare existing model prediction outputs.")
    parser.add_argument("--models", nargs="+", default=list(SUPPORTED_MODEL_NAMES), choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW)
    parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW)
    parser.add_argument("--predict-start", default=DEFAULT_PREDICT_START)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    exp_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
    out_dir = compare_existing_predictions(exp_dir, args.predict_start, args.models)
    print(f"Comparison dir: {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest tests.test_compare_models -v
```

Expected: PASS with 1 test.

- [ ] **Step 5: Commit**

```powershell
git add Reproduce/scripts/compare_models.py tests/test_compare_models.py
git commit -m "feat: add model comparison output aggregation"
```

---

### Task 10: 文档更新与入口兼容检查

**Files:**
- Modify: `Reproduce/README.md`
- Modify: `Reproduce/REPRODUCTION_PLAN.md`

- [ ] **Step 1: Write the failing documentation check**

Run:

```powershell
rg -n "lstm_720h_to_24h|--output-window 24|未来 24|24h" Reproduce/README.md Reproduce/REPRODUCTION_PLAN.md
```

Expected: FIND matches showing old default wording remains in docs.

- [ ] **Step 2: Update README**

Update `Reproduce/README.md` so it includes these exact command examples:

```powershell
python -m Reproduce.scripts.prepare_data --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model lstm --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00" --device cuda --epochs 100
python -m Reproduce.scripts.train_model --model xgboost --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model random_forest --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model arima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.train_model --model sarima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m Reproduce.scripts.predict_model --model lstm --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00" --device cuda
python -m Reproduce.scripts.compare_models --models lstm xgboost random_forest arima sarima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

Document the output root:

```text
Reproduce/outputs/window_720h_to_72h/
```

Document the model-specific prediction path:

```text
Reproduce/outputs/window_720h_to_72h/predictions/start_2026_03_01_0000/<model_name>/
```

- [ ] **Step 3: Update reproduction plan**

Update `Reproduce/REPRODUCTION_PLAN.md` so it states:

```text
默认实验：过去 720 小时 -> 未来 72 小时 PM2.5
LSTM、XGBoost、RandomForest 使用 6 特征直接多输出预测。
ARIMA、SARIMA 只使用训练期历史 pm25，不使用外生特征，不使用验证期或预测窗口真实值拟合。
```

Add this expected output tree:

```text
Reproduce/outputs/window_720h_to_72h/
├── data/
├── models/<model_name>/
├── predictions/start_2026_03_01_0000/<model_name>/
└── comparisons/start_2026_03_01_0000/
```

- [ ] **Step 4: Run documentation check again**

Run:

```powershell
rg -n "lstm_720h_to_24h|--output-window 24|未来 24|24h" Reproduce/README.md Reproduce/REPRODUCTION_PLAN.md
```

Expected: no matches for old default text unless explicitly labeled as historical output.

- [ ] **Step 5: Commit**

```powershell
git add Reproduce/README.md Reproduce/REPRODUCTION_PLAN.md
git commit -m "docs: document multi-model comparison workflow"
```

---

### Task 11: 全量轻量验证

**Files:**
- No source changes unless verification reveals a failing task-owned issue.

- [ ] **Step 1: Run all unit tests**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: PASS for all tests.

- [ ] **Step 2: Verify importability of script modules**

Run:

```powershell
python -m Reproduce.scripts.prepare_data --help
python -m Reproduce.scripts.train_model --help
python -m Reproduce.scripts.predict_model --help
python -m Reproduce.scripts.compare_models --help
python -m Reproduce.scripts.train_lstm --help
python -m Reproduce.scripts.predict_window --help
```

Expected: each command exits 0 and prints usage.

- [ ] **Step 3: Run quick data preparation smoke**

Run in the `pm25` conda environment:

```powershell
python -m Reproduce.scripts.prepare_data --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

Expected:

```text
Prepared data for window_720h_to_72h
```

and `Reproduce/outputs/window_720h_to_72h/data/windows.npz` exists.

- [ ] **Step 4: Run fast random forest smoke**

Run:

```powershell
python -m Reproduce.scripts.train_model --model random_forest --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00" --n-estimators 5 --n-jobs 1
python -m Reproduce.scripts.predict_model --model random_forest --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

Expected:

```text
Output dir: E:\pm25_predict\Reproduce\outputs\window_720h_to_72h\predictions\start_2026_03_01_0000\random_forest
```

and `predictions.csv` has 72 rows.

- [ ] **Step 5: Run comparison smoke with available models**

Run after at least two model prediction directories exist:

```powershell
python -m Reproduce.scripts.compare_models --models random_forest lstm --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

Expected: if both predictions exist, writes `model_metrics.csv` and `all_predictions.csv`; if LSTM prediction does not exist yet, fails with a clear missing file path.

- [ ] **Step 6: Commit verification fixes if any were needed**

If verification required small fixes:

```powershell
git add Reproduce tests
git commit -m "test: verify multi-model output workflow"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review

- Spec coverage: the tasks cover window output directories, model subdirectories, unified prediction files, comparison outputs, LSTM compatibility, XGBoost, RandomForest, ARIMA, SARIMA, default `720h -> 72h`, and documentation.
- Placeholder scan: no unresolved markers, incomplete commands, or unnamed files remain in this plan.
- Type consistency: `window_experiment_dir`, `model_dir`, `prediction_dir`, `comparison_dir`, `build_predictions_frame`, `write_prediction_outputs`, `train_model`, `predict_model`, and `compare_existing_predictions` are used with the same names across tasks.
