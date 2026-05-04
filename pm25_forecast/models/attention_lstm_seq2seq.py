from __future__ import annotations

from dataclasses import dataclass

from pm25_forecast.models.lstm_one_step import require_torch


@dataclass
class Seq2SeqConfig:
    input_size_history: int = 16
    input_size_future: int = 9
    output_size: int = 1
    hidden_size: int = 128
    encoder_num_layers: int = 2
    decoder_num_layers: int = 1
    num_heads: int = 4
    dropout: float = 0.3
    output_window: int = 72


def build_seq2seq_model(config: Seq2SeqConfig):
    torch, nn = require_torch()
    if int(config.hidden_size) % int(config.num_heads) != 0:
        raise ValueError(f"hidden_size ({config.hidden_size}) must be divisible by num_heads ({config.num_heads})")

    class EncoderAttentionLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            lstm_dropout = float(config.dropout) if int(config.encoder_num_layers) > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=int(config.input_size_history),
                hidden_size=int(config.hidden_size),
                num_layers=int(config.encoder_num_layers),
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.attention_query = nn.Linear(int(config.hidden_size), int(config.hidden_size))
            self.attention_key = nn.Linear(int(config.hidden_size), int(config.hidden_size))
            self.attention_value = nn.Linear(int(config.hidden_size), int(config.hidden_size))
            self.num_heads = int(config.num_heads)
            self.head_dim = int(config.hidden_size) // self.num_heads
            self.dropout = nn.Dropout(float(config.dropout))

        def forward(self, history):
            lstm_out, (hidden, cell) = self.lstm(history)
            batch_size = lstm_out.size(0)
            seq_len = lstm_out.size(1)
            q = self.attention_query(lstm_out)
            k = self.attention_key(lstm_out)
            v = self.attention_value(lstm_out)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
            weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights, v)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            return self.dropout(context), hidden[-1], cell[-1]

    class MultiHeadCrossAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = int(config.num_heads)
            self.head_dim = int(config.hidden_size) // self.num_heads
            self.query = nn.Linear(int(config.hidden_size), int(config.hidden_size))
            self.key = nn.Linear(int(config.hidden_size), int(config.hidden_size))
            self.value = nn.Linear(int(config.hidden_size), int(config.hidden_size))
            self.out = nn.Linear(int(config.hidden_size), int(config.hidden_size))
            self.dropout = nn.Dropout(float(config.dropout))

        def forward(self, decoder_hidden, encoder_outputs):
            batch_size = encoder_outputs.size(0)
            seq_len = encoder_outputs.size(1)
            q = self.query(decoder_hidden).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.key(encoder_outputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.value(encoder_outputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
            weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights, v)
            context = context.transpose(1, 2).contiguous().view(batch_size, -1)
            return self.out(self.dropout(context))

    class AttentionLSTMSeq2Seq(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = EncoderAttentionLSTM()
            self.decoder_input = nn.Linear(int(config.input_size_future) + 1, int(config.hidden_size))
            self.decoder_cells = nn.ModuleList(
                [
                    nn.LSTMCell(
                        input_size=int(config.hidden_size),
                        hidden_size=int(config.hidden_size),
                    )
                    for _ in range(int(config.decoder_num_layers))
                ]
            )
            self.cross_attention = MultiHeadCrossAttention()
            self.output = nn.Sequential(
                nn.Dropout(float(config.dropout)),
                nn.Linear(int(config.hidden_size) * 2, int(config.hidden_size)),
                nn.ReLU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(int(config.hidden_size), int(config.output_size)),
            )
            self.output_window = int(config.output_window)

        def forward(
            self,
            history,
            future,
            first_pm25,
            teacher_forcing_targets=None,
            teacher_forcing_prob: float = 0.0,
        ):
            encoder_outputs, encoder_hidden, encoder_cell = self.encoder(history)
            batch_size = history.size(0)
            device = history.device
            hidden_states = [encoder_hidden for _ in self.decoder_cells]
            cell_states = [encoder_cell for _ in self.decoder_cells]
            prev_pm25 = first_pm25
            outputs = []
            steps = min(self.output_window, future.size(1))

            for step in range(steps):
                decoder_input = torch.cat([future[:, step, :], prev_pm25], dim=1)
                decoder_input = self.decoder_input(decoder_input)
                next_hidden = []
                next_cell = []
                cell_input = decoder_input
                for layer_index, cell in enumerate(self.decoder_cells):
                    h, c = cell(cell_input, (hidden_states[layer_index], cell_states[layer_index]))
                    next_hidden.append(h)
                    next_cell.append(c)
                    cell_input = h
                hidden_states = next_hidden
                cell_states = next_cell
                decoder_hidden = hidden_states[-1]
                context = self.cross_attention(decoder_hidden, encoder_outputs)
                pred = self.output(torch.cat([decoder_hidden, context], dim=1))
                outputs.append(pred)

                use_teacher = (
                    teacher_forcing_targets is not None
                    and float(teacher_forcing_prob) > 0
                    and bool((torch.rand(1, device=device) < float(teacher_forcing_prob)).item())
                )
                if use_teacher:
                    prev_pm25 = teacher_forcing_targets[:, step : step + 1]
                else:
                    prev_pm25 = pred.detach() if self.training else pred

            return torch.cat(outputs, dim=1)

    return AttentionLSTMSeq2Seq()
