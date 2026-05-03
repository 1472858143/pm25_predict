from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def plot_prediction_curve(predictions: pd.DataFrame, output_path: str | Path, title: str) -> bool:
    plt = _load_matplotlib()
    if plt is None:
        return False
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(frame["timestamp"], frame["y_true"], label="True PM2.5", linewidth=1.2)
    ax.plot(frame["timestamp"], frame["y_pred"], label="Predicted PM2.5", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("PM2.5")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def plot_error_curve(predictions: pd.DataFrame, output_path: str | Path, title: str) -> bool:
    plt = _load_matplotlib()
    if plt is None:
        return False
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(frame["timestamp"], frame["error"], label="Prediction error", linewidth=1.0, color="#b23b3b")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("y_pred - y_true")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def plot_scatter(predictions: pd.DataFrame, output_path: str | Path, title: str) -> bool:
    plt = _load_matplotlib()
    if plt is None:
        return False
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    min_value = float(min(y_true.min(), y_pred.min()))
    max_value = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=12, alpha=0.55)
    ax.plot([min_value, max_value], [min_value, max_value], color="#333333", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("True PM2.5")
    ax.set_ylabel("Predicted PM2.5")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def plot_loss_curve(history: pd.DataFrame, output_path: str | Path, title: str = "Training Loss") -> bool:
    plt = _load_matplotlib()
    if plt is None:
        return False
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["train_loss"], linewidth=1.2, label="Train loss")
    if "validation_loss" in history.columns:
        ax.plot(history["epoch"], history["validation_loss"], linewidth=1.2, label="Validation loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def write_plot_status(path: str | Path, plotted: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Plot Status", ""]
    for name, value in plotted.items():
        lines.append(f"- {name}: `{value}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
