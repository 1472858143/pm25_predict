# AttentionLSTM 模型实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `attention_lstm` 模型，使用 Self-Attention 机制替代 LSTM 最后时间步选择，提升 R² 和 RMSE。

**Architecture:** LSTM 提取所有时间步隐藏状态 → Multi-Head Self-Attention 加权聚合 → Dropout → Linear 输出。完全独立于现有 `lstm` 模型，复用训练工具函数。

**Tech Stack:** PyTorch, argparse, unittest

---

## 文件清单

| 文件 | 职责 | 操作 |
|------|------|------|
| `pm25_forecast/models/attention_lstm.py` | AttentionLSTM 模型定义、AttentionConfig | 新建 |
| `pm25_forecast/scripts/train_attention_lstm.py` | 训练脚本（复用 train_lstm 工具函数） | 新建 |
| `pm25_forecast/scripts/predict_attention_lstm.py` | 预测脚本（复用 predict_month 模式） | 新建 |
| `pm25_forecast/utils/paths.py` | 模型名注册 | 修改 |
| `pm25_forecast/scripts/train_model.py` | 统一训练入口（添加分支） | 修改 |
| `pm25_forecast/scripts/predict_model.py` | 统一预测入口（添加分支） | 修改 |
| `tests/test_train_model_cli.py` | CLI 参数测试 | 修改 |

---

### Task 1: 创建 AttentionLSTM 模型定义

**Files:**
- Create: `pm25_forecast/models/attention_lstm.py`
- Modify: `pm25_forecast/utils/paths.py:8`

- [ ] **Step 1: 创建 attention_lstm.py 模型文件**

```python
from __future__ import annotations

from dataclasses import dataclass

from pm25_forecast.models.lstm_one_step import require_torch


@dataclass
class AttentionConfig:
    input_size: int = 6
    output_size: int = 1
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    num_heads: int = 4


def build_model(config: AttentionConfig):
    torch, nn = require_torch()

    class AttentionLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if int(config.hidden_size) % int(config.num_heads) != 0:
                raise ValueError(
                    f"hidden_size ({config.hidden_size}) must be divisible by num_heads ({config.num_heads})"
                )
            lstm_dropout = float(config.dropout) if int(config.num_layers) > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=int(config.input_size),
                hidden_size=int(config.hidden_size),
                num_layers=int(config.num_layers),
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.attention_query = nn.Linear(int(config.hidden_size), int(config.hidden_size))
            self.attention_key = nn.Linear(int(config.hidden_size), int(config.hidden_size))
            self.attention_value = nn.Linear(int(config.hidden_size), int(config.hidden_size))
            self.num_heads = int(config.num_heads)
            self.head_dim = int(config.hidden_size) // self.num_heads
            self.dropout = nn.Dropout(float(config.dropout))
            self.output = nn.Linear(int(config.hidden_size), int(config.output_size))

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            batch_size = lstm_out.size(0)
            seq_len = lstm_out.size(1)
            Q = self.attention_query(lstm_out)
            K = self.attention_key(lstm_out)
            V = self.attention_value(lstm_out)
            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn_weights, V)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            last_context = context[:, -1, :]
            prediction = self.output(self.dropout(last_context))
            return prediction

    return AttentionLSTM()
```

- [ ] **Step 2: 注册 attention_lstm 到 SUPPORTED_MODEL_NAMES**

修改 `pm25_forecast/utils/paths.py:8`：

```python
SUPPORTED_MODEL_NAMES = ("lstm", "attention_lstm", "xgboost", "random_forest", "arima", "sarima")
```

- [ ] **Step 3: 运行现有测试确认无破坏**

Run: `python -m unittest discover -s tests -v`
Expected: 全部通过（与之前一致）

- [ ] **Step 4: 提交**

```bash
git add pm25_forecast/models/attention_lstm.py pm25_forecast/utils/paths.py
git commit -m "feat: add AttentionLSTM model definition and register in SUPPORTED_MODEL_NAMES"
```

---

### Task 2: 创建 AttentionLSTM 训练脚本

**Files:**
- Create: `pm25_forecast/scripts/train_attention_lstm.py`

- [ ] **Step 1: 创建训练脚本**

```python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pm25_forecast.models.attention_lstm import AttentionConfig, build_model
from pm25_forecast.models.lstm_one_step import require_torch
from pm25_forecast.scripts.train_lstm import (
    build_target_weights,
    collect_model_predictions,
    loss_value,
    resolve_peak_thresholds,
    select_device,
)
from pm25_forecast.utils.calibration import apply_calibration, fit_horizon_linear_calibration
from pm25_forecast.utils.data_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
    TARGET_COLUMN,
    load_scaler,
    prepare_data_bundle,
    read_json,
    write_json,
)
from pm25_forecast.utils.metrics import regression_metrics
from pm25_forecast.utils.paths import model_dir, window_experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct multi-output AttentionLSTM for PM2.5 forecasting.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Input CSV path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Forecasting outputs root.")
    parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW, help="Historical input window in hours.")
    parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW, help="Direct forecast horizon in hours.")
    parser.add_argument("--predict-start", default=DEFAULT_PREDICT_START, help="Prediction start timestamp.")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attention-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", default="weighted_huber", choices=["mse", "weighted_mse", "weighted_huber"])
    parser.add_argument("--peak-quantile", type=float, default=0.75, help="Train-set quantile used as high-PM2.5 threshold.")
    parser.add_argument("--extreme-quantile", type=float, default=0.90, help="Train-set quantile used as extreme-PM2.5 threshold.")
    parser.add_argument("--peak-threshold", type=float, default=None, help="Optional absolute PM2.5 threshold for peak weighting.")
    parser.add_argument("--extreme-threshold", type=float, default=None, help="Optional absolute PM2.5 threshold for stronger weighting.")
    parser.add_argument("--peak-weight", type=float, default=3.0)
    parser.add_argument("--extreme-weight", type=float, default=5.0)
    parser.add_argument("--huber-delta", type=float, default=0.05, help="Huber delta in scaled PM2.5 space.")
    parser.add_argument("--variance-penalty", type=float, default=0.05, help="Penalty for batch-level prediction variance shrinkage.")
    parser.add_argument("--lr-patience", type=int, default=5, help="Epochs to wait before reducing LR on plateau.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Factor to reduce LR by on plateau.")
    parser.add_argument("--early-stopping-patience", type=int, default=15, help="Epochs to wait before early stopping.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping. 0 disables.")
    parser.add_argument("--calibration", default="horizon_linear", choices=["none", "horizon_linear"])
    parser.add_argument("--calibration-fit", default="train", choices=["train", "validation"])
    parser.add_argument("--calibration-slope-min", type=float, default=0.5)
    parser.add_argument("--calibration-slope-max", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--prepare-data", action="store_true", help="Regenerate prepared window data before training.")
    return parser.parse_args()


def resolve_attention_lstm_training_paths(output_root: str | Path, input_window: int, output_window: int) -> dict[str, Path]:
    window_dir = window_experiment_dir(output_root, input_window, output_window)
    attn_dir = model_dir(window_dir, "attention_lstm")
    return {
        "window_dir": window_dir,
        "attn_dir": attn_dir,
        "data_config_path": window_dir / "data" / "data_config.json",
        "bundle_path": window_dir / "data" / "windows.npz",
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    torch, _ = require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    if args.prepare_data:
        prepare_data_bundle(
            data_path=args.data_path,
            output_root=args.output_root,
            input_window=args.input_window,
            output_window=args.output_window,
            predict_start=args.predict_start,
        )

    paths = resolve_attention_lstm_training_paths(args.output_root, args.input_window, args.output_window)
    window_dir = paths["window_dir"]
    attn_dir = paths["attn_dir"]
    data_config_path = paths["data_config_path"]
    bundle_path = paths["bundle_path"]
    if not data_config_path.exists() or not bundle_path.exists():
        prepare_data_bundle(
            data_path=args.data_path,
            output_root=args.output_root,
            input_window=args.input_window,
            output_window=args.output_window,
            predict_start=args.predict_start,
        )

    data_config = read_json(data_config_path)
    bundle = np.load(bundle_path, allow_pickle=True)
    X_train = bundle["X_train"].astype(np.float32)
    y_train = bundle["y_train"].astype(np.float32)
    y_train_raw = bundle["y_train_raw"].astype(np.float32)
    X_validation = bundle["X_validation"].astype(np.float32)
    y_validation = bundle["y_validation"].astype(np.float32)
    y_validation_raw = bundle["y_validation_raw"].astype(np.float32)
    if len(X_train) == 0:
        raise ValueError("No training samples were prepared.")
    if len(X_validation) == 0:
        raise ValueError("No validation samples were prepared.")

    peak_threshold, extreme_threshold = resolve_peak_thresholds(
        y_train_raw=y_train_raw,
        peak_threshold=args.peak_threshold,
        extreme_threshold=args.extreme_threshold,
        peak_quantile=float(args.peak_quantile),
        extreme_quantile=float(args.extreme_quantile),
    )
    train_weights = build_target_weights(
        y_train_raw,
        peak_threshold=peak_threshold,
        extreme_threshold=extreme_threshold,
        peak_weight=float(args.peak_weight),
        extreme_weight=float(args.extreme_weight),
    )
    validation_weights = build_target_weights(
        y_validation_raw,
        peak_threshold=peak_threshold,
        extreme_threshold=extreme_threshold,
        peak_weight=float(args.peak_weight),
        extreme_weight=float(args.extreme_weight),
    )

    device = select_device(torch, args.device)
    model = build_model(
        AttentionConfig(
            input_size=int(X_train.shape[-1]),
            output_size=int(y_train.shape[-1]) if y_train.ndim > 1 else 1,
            hidden_size=int(args.hidden_size),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
            num_heads=int(args.attention_heads),
        )
    ).to(device)

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(train_weights))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    train_eval_loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
    validation_dataset = TensorDataset(
        torch.from_numpy(X_validation),
        torch.from_numpy(y_validation),
        torch.from_numpy(validation_weights),
    )
    validation_loader = DataLoader(validation_dataset, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(args.lr_patience),
        factor=float(args.lr_factor),
        min_lr=1e-6,
    )

    history: list[dict[str, float | int]] = []
    best_train_loss = float("inf")
    best_train_state = None
    best_val_loss = float("inf")
    best_val_state = None
    best_val_epoch = 0
    patience_counter = 0
    attn_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    print(
        f"loss={args.loss} "
        f"peak_threshold={peak_threshold:.3f} "
        f"extreme_threshold={extreme_threshold:.3f} "
        f"variance_penalty={float(args.variance_penalty):.4f} "
        f"attention_heads={int(args.attention_heads)}"
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_X, batch_y, batch_weight in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_weight = batch_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_X)
            loss = loss_value(
                torch,
                pred,
                batch_y,
                batch_weight,
                loss_name=str(args.loss),
                huber_delta=float(args.huber_delta),
                variance_penalty=float(args.variance_penalty),
            )
            loss.backward()
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.max_grad_norm))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses))
        model.eval()
        val_losses: list[float] = []
        val_sse = 0.0
        val_abs_error = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_X, batch_y, batch_weight in validation_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_weight = batch_weight.to(device, non_blocking=True)
                pred = model(batch_X)
                val_loss = loss_value(
                    torch,
                    pred,
                    batch_y,
                    batch_weight,
                    loss_name=str(args.loss),
                    huber_delta=float(args.huber_delta),
                    variance_penalty=float(args.variance_penalty),
                )
                val_losses.append(float(val_loss.detach().cpu().item()))
                diff = pred - batch_y
                val_sse += float(torch.sum(diff.pow(2)).detach().cpu().item())
                val_abs_error += float(torch.sum(torch.abs(diff)).detach().cpu().item())
                val_count += int(batch_y.numel())
        validation_loss = float(np.mean(val_losses))
        validation_rmse_scaled = float(np.sqrt(val_sse / max(val_count, 1)))
        validation_mae_scaled = float(val_abs_error / max(val_count, 1))

        scheduler.step(validation_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_rmse_scaled": validation_rmse_scaled,
                "validation_mae_scaled": validation_mae_scaled,
            }
        )
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_val_epoch = epoch
            best_val_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.8f} "
            f"val_loss={validation_loss:.8f} "
            f"val_rmse_scaled={validation_rmse_scaled:.8f} "
            f"best_val_loss={best_val_loss:.8f} "
            f"best_val_epoch={best_val_epoch} "
            f"lr={current_lr:.2e}"
        )
        if patience_counter >= int(args.early_stopping_patience):
            print(f"Early stopping at epoch {epoch}, best validation at epoch {best_val_epoch}")
            break

    model_path = attn_dir / "model.pt"
    best_model_path = attn_dir / "model_best_train_loss.pt"
    best_val_model_path = attn_dir / "model_best_val_loss.pt"
    torch.save(model.state_dict(), model_path)
    if best_train_state is not None:
        torch.save(best_train_state, best_model_path)
    if best_val_state is not None:
        torch.save(best_val_state, best_val_model_path)

    calibration_path = attn_dir / "calibration.json"
    calibration_metrics_before = None
    calibration_metrics_after = None
    if str(args.calibration) == "horizon_linear":
        scaler = load_scaler(window_dir)
        if best_val_state is not None:
            model.load_state_dict(best_val_state)
        calibration_loader = train_eval_loader if str(args.calibration_fit) == "train" else validation_loader
        calibration_true_raw = y_train_raw if str(args.calibration_fit) == "train" else y_validation_raw
        calibration_pred_scaled = collect_model_predictions(torch, model, calibration_loader, device)
        calibration_pred_raw = scaler.inverse_column(calibration_pred_scaled, TARGET_COLUMN)
        calibration = fit_horizon_linear_calibration(
            y_true=calibration_true_raw,
            y_pred=calibration_pred_raw,
            slope_min=float(args.calibration_slope_min),
            slope_max=float(args.calibration_slope_max),
            clip_min=0.0,
        )
        calibration_pred_calibrated = apply_calibration(calibration_pred_raw, calibration)
        calibration_metrics_before = regression_metrics(calibration_true_raw, calibration_pred_raw)
        calibration_metrics_after = regression_metrics(calibration_true_raw, calibration_pred_calibrated)
        calibration["fit_data"] = str(args.calibration_fit)
        calibration["source_model_path"] = str(best_val_model_path)
        calibration["metrics_before"] = calibration_metrics_before
        calibration["metrics_after"] = calibration_metrics_after
        write_json(calibration_path, calibration)
        pd.DataFrame(calibration["horizon_stats"]).to_csv(
            attn_dir / "calibration_horizon_stats.csv",
            index=False,
            encoding="utf-8",
        )
        print(
            f"Calibration {args.calibration_fit} metrics: "
            f"before={calibration_metrics_before} "
            f"after={calibration_metrics_after}"
        )
    else:
        write_json(
            calibration_path,
            {
                "method": "none",
                "fit_data": "none",
                "source_model_path": str(best_val_model_path),
            },
        )

    pd.DataFrame(history).to_csv(attn_dir / "training_history.csv", index=False, encoding="utf-8")
    training_config = {
        "data_config": data_config,
        "model": {
            "input_size": int(X_train.shape[-1]),
            "output_size": int(y_train.shape[-1]) if y_train.ndim > 1 else 1,
            "hidden_size": int(args.hidden_size),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "attention_heads": int(args.attention_heads),
        },
        "loss": {
            "name": str(args.loss),
            "peak_threshold": float(peak_threshold),
            "extreme_threshold": float(extreme_threshold),
            "peak_quantile": float(args.peak_quantile),
            "extreme_quantile": float(args.extreme_quantile),
            "peak_weight": float(args.peak_weight),
            "extreme_weight": float(args.extreme_weight),
            "huber_delta": float(args.huber_delta),
            "variance_penalty": float(args.variance_penalty),
        },
        "calibration": {
            "method": str(args.calibration),
            "fit_data": str(args.calibration_fit),
            "path": str(calibration_path),
            "slope_min": float(args.calibration_slope_min),
            "slope_max": float(args.calibration_slope_max),
            "metrics_before": calibration_metrics_before,
            "metrics_after": calibration_metrics_after,
        },
        "training": {
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "lr_patience": int(args.lr_patience),
            "lr_factor": float(args.lr_factor),
            "early_stopping_patience": int(args.early_stopping_patience),
            "max_grad_norm": float(args.max_grad_norm),
            "seed": int(args.seed),
            "device": str(device),
            "best_train_loss": float(best_train_loss),
            "best_validation_loss": float(best_val_loss),
            "best_validation_epoch": int(best_val_epoch),
            "total_epochs_run": epoch,
            "early_stopped": patience_counter >= int(args.early_stopping_patience),
            "elapsed_seconds": float(time.time() - start_time),
            "model_path": str(model_path),
            "best_train_loss_model_path": str(best_model_path),
            "best_validation_loss_model_path": str(best_val_model_path),
        },
    }
    write_json(attn_dir / "training_config.json", training_config)
    print(f"Saved model: {model_path}")
    print(f"Saved best-train-loss model: {best_model_path}")
    print(f"Saved best-validation-loss model: {best_val_model_path}")
    print(f"Saved calibration: {calibration_path}")
    return {
        "model_name": "attention_lstm",
        "model_dir": str(attn_dir),
        "model_path": str(model_path),
        "training_config_path": str(attn_dir / "training_config.json"),
    }


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本可被 Python 导入**

Run: `python -c "from pm25_forecast.scripts.train_attention_lstm import parse_args; print('import ok')"`
Expected: `import ok`

- [ ] **Step 3: 提交**

```bash
git add pm25_forecast/scripts/train_attention_lstm.py
git commit -m "feat: add train_attention_lstm training script"
```

---

### Task 3: 创建 AttentionLSTM 预测脚本

**Files:**
- Create: `pm25_forecast/scripts/predict_attention_lstm.py`

- [ ] **Step 1: 创建预测脚本**

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

from pm25_forecast.models.attention_lstm import AttentionConfig, build_model
from pm25_forecast.models.lstm_one_step import require_torch
from pm25_forecast.utils.calibration import apply_calibration
from pm25_forecast.utils.data_utils import (
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
    TARGET_COLUMN,
    load_scaler,
    prepare_data_bundle,
    parse_predict_start,
    read_json,
)
from pm25_forecast.utils.paths import model_dir, prediction_dir, window_experiment_dir
from pm25_forecast.utils.prediction_io import build_predictions_frame, write_prediction_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one direct multi-output window with a trained AttentionLSTM.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Forecasting outputs root.")
    parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW)
    parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW)
    parser.add_argument("--predict-start", default=DEFAULT_PREDICT_START, help="Prediction start timestamp.")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attention-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-path", default=None, help="Path to model checkpoint. Defaults to best-validation-loss model if available.")
    parser.add_argument("--calibration-path", default=None, help="Optional fitted calibration JSON path.")
    parser.add_argument("--no-calibration", action="store_true", help="Disable fitted prediction calibration.")
    parser.add_argument("--prepare-data", action="store_true", help="Regenerate prepared data before predicting.")
    return parser.parse_args()


def select_device(torch: Any, requested: str) -> Any:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(attn_dir: Path, requested: str | None) -> Path:
    if requested:
        return Path(requested)
    best_val = attn_dir / "model_best_val_loss.pt"
    if best_val.exists():
        return best_val
    best = attn_dir / "model_best_train_loss.pt"
    if best.exists():
        return best
    return attn_dir / "model.pt"


def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    torch, _ = require_torch()
    if args.prepare_data:
        prepare_data_bundle(
            output_root=args.output_root,
            input_window=args.input_window,
            output_window=args.output_window,
            predict_start=args.predict_start,
        )

    window_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
    attn_dir = model_dir(window_dir, "attention_lstm")
    bundle_path = window_dir / "data" / "windows.npz"
    if not bundle_path.exists():
        prepare_data_bundle(
            output_root=args.output_root,
            input_window=args.input_window,
            output_window=args.output_window,
            predict_start=args.predict_start,
        )
    bundle = np.load(bundle_path, allow_pickle=True)
    X_predict = bundle["X_predict"].astype(np.float32)
    y_predict_raw = bundle["y_predict_raw"].astype(np.float32)
    timestamps_start = bundle["timestamps_predict_start"].astype(str)
    timestamps_end = bundle["timestamps_predict_end"].astype(str)
    timestamps_target = bundle["timestamps_predict_target"].astype(str)
    if len(X_predict) != 1:
        raise ValueError(f"Expected one prediction sample, got {len(X_predict)}.")

    device = select_device(torch, args.device)
    model = build_model(
        AttentionConfig(
            input_size=int(X_predict.shape[-1]),
            output_size=int(y_predict_raw.shape[-1]) if y_predict_raw.ndim > 1 else 1,
            hidden_size=int(args.hidden_size),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
            num_heads=int(args.attention_heads),
        )
    ).to(device)
    ckpt = checkpoint_path(attn_dir, args.model_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    predictions_scaled: list[np.ndarray] = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, len(X_predict), batch_size):
            batch = torch.from_numpy(X_predict[start : start + batch_size]).to(device)
            pred = model(batch).detach().cpu().numpy()
            predictions_scaled.append(pred)
    y_pred_scaled = np.concatenate(predictions_scaled)

    scaler = load_scaler(window_dir)
    y_pred_model = scaler.inverse_column(y_pred_scaled, TARGET_COLUMN)
    calibration_applied = False
    calibration_path = Path(args.calibration_path) if args.calibration_path else attn_dir / "calibration.json"
    calibration: dict[str, Any] | None = None
    if not bool(args.no_calibration) and calibration_path.exists():
        calibration = read_json(calibration_path)
        y_pred = apply_calibration(y_pred_model, calibration)
        calibration_applied = calibration.get("method") not in {None, "none"}
    else:
        y_pred = y_pred_model
    y_true = y_predict_raw
    if y_pred_model.ndim == 1:
        y_pred_model = y_pred_model.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    output_window = int(y_true.shape[1])
    predictions = build_predictions_frame(
        model_name="attention_lstm",
        y_true=y_true,
        y_pred_model=y_pred_model,
        y_pred=y_pred,
        timestamps_start=timestamps_start,
        timestamps_end=timestamps_end,
        timestamps_target=timestamps_target,
    )

    prediction_start = parse_predict_start(args.predict_start)
    prediction_output_dir = prediction_dir(window_dir, args.predict_start, "attention_lstm")
    summary = write_prediction_outputs(
        predictions=predictions,
        output_dir=prediction_output_dir,
        model_name="attention_lstm",
        model_path=ckpt,
        calibration_path=calibration_path if calibration_path.exists() else None,
        calibration_applied=calibration_applied,
        calibration_method=None if calibration is None else calibration.get("method"),
        device=str(device),
        predict_start=args.predict_start,
    )
    summary["experiment_dir"] = str(window_dir)
    summary["predict_start"] = str(prediction_start)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_prediction(args)
    print(f"Predicted samples: {summary['sample_count']}")
    print(f"Output dir: {summary['prediction_dir']}")
    print(f"Metrics: {summary['metrics']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本可被 Python 导入**

Run: `python -c "from pm25_forecast.scripts.predict_attention_lstm import parse_args; print('import ok')"`
Expected: `import ok`

- [ ] **Step 3: 提交**

```bash
git add pm25_forecast/scripts/predict_attention_lstm.py
git commit -m "feat: add predict_attention_lstm prediction script"
```

---

### Task 4: 更新统一入口 train_model.py

**Files:**
- Modify: `pm25_forecast/scripts/train_model.py:49-51`（添加 --attention-heads 参数）
- Modify: `pm25_forecast/scripts/train_model.py:24`（添加 import）
- Modify: `pm25_forecast/scripts/train_model.py:165-168`（添加 dispatch 分支）

- [ ] **Step 1: 添加 import**

在 `pm25_forecast/scripts/train_model.py:24` 处添加：

```python
from pm25_forecast.scripts import train_attention_lstm, train_lstm
```

（替换原来的 `from pm25_forecast.scripts import train_lstm`）

- [ ] **Step 2: 添加 --attention-heads 参数**

在 `pm25_forecast/scripts/train_model.py` 的 `build_arg_parser()` 中，在 `--variance-penalty` 参数后添加：

```python
parser.add_argument("--attention-heads", type=int, default=4)
```

- [ ] **Step 3: 添加 dispatch 分支**

修改 `pm25_forecast/scripts/train_model.py` 的 `run_training()` 函数：

```python
def run_training(args: argparse.Namespace) -> dict[str, Any]:
    model_name = validate_model_name(args.model)
    if model_name == "lstm":
        return train_lstm.run_training(args)
    if model_name == "attention_lstm":
        return train_attention_lstm.run_training(args)
    return train_non_lstm(args)
```

- [ ] **Step 4: 运行现有测试确认无破坏**

Run: `python -m unittest discover -s tests -v`
Expected: 全部通过

- [ ] **Step 5: 提交**

```bash
git add pm25_forecast/scripts/train_model.py
git commit -m "feat: add attention_lstm dispatch to train_model.py"
```

---

### Task 5: 更新统一入口 predict_model.py

**Files:**
- Modify: `pm25_forecast/scripts/predict_model.py:106-109`（添加 dispatch 分支）
- Modify: `pm25_forecast/scripts/predict_model.py:41-44`（添加 --attention-heads 参数）

- [ ] **Step 1: 添加 --attention-heads 参数**

在 `pm25_forecast/scripts/predict_model.py` 的 `build_arg_parser()` 中，在 `--dropout` 参数后添加：

```python
parser.add_argument("--attention-heads", type=int, default=4)
```

- [ ] **Step 2: 添加 dispatch 分支**

修改 `pm25_forecast/scripts/predict_model.py` 的 `run_prediction()` 函数：

```python
def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    model_name = validate_model_name(args.model)
    if model_name == "lstm":
        return predict_month.run_prediction(args)
    if model_name == "attention_lstm":
        from pm25_forecast.scripts import predict_attention_lstm
        return predict_attention_lstm.run_prediction(args)
    return _predict_non_lstm(args)
```

- [ ] **Step 3: 运行现有测试确认无破坏**

Run: `python -m unittest discover -s tests -v`
Expected: 全部通过

- [ ] **Step 4: 提交**

```bash
git add pm25_forecast/scripts/predict_model.py
git commit -m "feat: add attention_lstm dispatch to predict_model.py"
```

---

### Task 6: 更新 CLI 测试

**Files:**
- Modify: `tests/test_train_model_cli.py`

- [ ] **Step 1: 添加 attention_lstm 默认值测试**

```python
def test_parser_attention_lstm_defaults(self):
    parser = build_arg_parser()
    args = parser.parse_args(["--model", "attention_lstm"])
    self.assertEqual(args.attention_heads, 4)
    self.assertEqual(args.hidden_size, 128)
    self.assertEqual(args.num_layers, 2)
    self.assertEqual(args.dropout, 0.3)
    self.assertEqual(args.lr_patience, 5)
    self.assertEqual(args.lr_factor, 0.5)
    self.assertEqual(args.early_stopping_patience, 15)
    self.assertEqual(args.max_grad_norm, 1.0)
```

- [ ] **Step 2: 添加 attention_heads 自定义参数测试**

```python
def test_parser_accepts_attention_heads(self):
    parser = build_arg_parser()
    args = parser.parse_args(["--model", "attention_lstm", "--attention-heads", "8"])
    self.assertEqual(args.attention_heads, 8)
```

- [ ] **Step 3: 运行全部测试**

Run: `python -m unittest discover -s tests -v`
Expected: 全部通过

- [ ] **Step 4: 提交**

```bash
git add tests/test_train_model_cli.py
git commit -m "test: add attention_lstm CLI tests"
```

---

### Task 7: 端到端验证

- [ ] **Step 1: 验证训练命令 help 输出**

Run: `python -m pm25_forecast.scripts.train_model --model attention_lstm --help`
Expected: 输出包含 `--attention-heads` 参数说明

- [ ] **Step 2: 验证预测命令 help 输出**

Run: `python -m pm25_forecast.scripts.predict_model --model attention_lstm --help`
Expected: 输出包含 `--attention-heads` 参数说明

- [ ] **Step 3: 验证 compare_models 支持 attention_lstm**

Run: `python -m pm25_forecast.scripts.compare_models --help`
Expected: `--models` choices 包含 `attention_lstm`

- [ ] **Step 4: 运行全部测试最终确认**

Run: `python -m unittest discover -s tests -v`
Expected: 全部通过

---

## 验收检查

完成所有 Task 后运行：

```powershell
# 验证模型注册
python -c "from pm25_forecast.utils.paths import SUPPORTED_MODEL_NAMES; print(SUPPORTED_MODEL_NAMES)"
# 预期: ('lstm', 'attention_lstm', 'xgboost', 'random_forest', 'arima', 'sarima')

# 验证 CLI 参数
python -m pm25_forecast.scripts.train_model --model attention_lstm --help

# 运行全部测试
python -m unittest discover -s tests -v
```

训练命令示例：

```powershell
python -m pm25_forecast.scripts.train_model --model attention_lstm --attention-heads 4 --device cuda --epochs 200
```

预测命令示例：

```powershell
python -m pm25_forecast.scripts.predict_model --model attention_lstm --device cuda
```

比较命令示例：

```powershell
python -m pm25_forecast.scripts.compare_models --models lstm attention_lstm xgboost random_forest
```
