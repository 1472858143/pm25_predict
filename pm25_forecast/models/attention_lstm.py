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
