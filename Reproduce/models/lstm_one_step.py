from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LSTMConfig:
    input_size: int = 6
    output_size: int = 1
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for LSTM reproduction. "
            "Run with E:\\Enviroments\\miniconda3\\envs\\pm25\\python.exe or install torch in the active environment."
        ) from exc
    return torch, nn


def build_model(config: LSTMConfig):
    torch, nn = require_torch()

    class OneStepLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            lstm_dropout = float(config.dropout) if int(config.num_layers) > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=int(config.input_size),
                hidden_size=int(config.hidden_size),
                num_layers=int(config.num_layers),
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.dropout = nn.Dropout(float(config.dropout))
            self.output = nn.Linear(int(config.hidden_size), int(config.output_size))

        def forward(self, x):
            outputs, _ = self.lstm(x)
            last_output = outputs[:, -1, :]
            prediction = self.output(self.dropout(last_output))
            return prediction

    return OneStepLSTM()
