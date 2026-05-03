# AttentionLSTM 模型设计

## 概述

在现有 LSTM 模型基础上，新增带 Self-Attention 机制的 LSTM 模型 `attention_lstm`。保留现有 `lstm` 模型不变，`attention_lstm` 作为独立模型类型注册到系统中。

**目标**：通过 Self-Attention 机制让模型自动学习哪些历史时间步对预测更重要，提升 R² 和 RMSE。

## 模型架构

```
输入: (batch, seq_len=720, features=6)
  → LSTM (hidden_size=128, num_layers=2, batch_first=True)
  → 所有时间步输出: (batch, seq_len, hidden_size)
  → Self-Attention (num_heads=4):
      Q = Linear(hidden_size → hidden_size)
      K = Linear(hidden_size → hidden_size)
      V = Linear(hidden_size → hidden_size)
      attention_scores = softmax(Q·K^T / sqrt(d_k))
      context = attention_scores · V  → (batch, seq_len, hidden_size)
      output = context[:, -1, :]  → 取最后时间步的上下文加权表示
  → Dropout(0.3)
  → Linear(hidden_size → output_size=72)
  → 输出: (batch, 72)
```

### 关键设计决策

1. **Self-Attention 作用于所有时间步**：LSTM 输出 720 个时间步的隐藏状态，Self-Attention 在这些隐藏状态之间计算注意力权重，自动学习哪些历史时间步对预测更重要。
2. **取最后时间步的上下文加权表示**：`context[:, -1, :]` 表示以最后时间步为查询，对所有时间步做注意力加权后的聚合表示。
3. **多头注意力**：默认 4 个注意力头，通过 `--attention-heads` 参数控制。
4. **与 LSTM 共享超参数**：`hidden_size`, `num_layers`, `dropout`, 学习率, 早停, 梯度裁剪等参数与 LSTM 共享。

## 文件结构

### 新增文件

| 文件 | 职责 |
|------|------|
| `pm25_forecast/models/attention_lstm.py` | AttentionLSTM 模型定义、AttentionConfig 数据类 |
| `pm25_forecast/scripts/train_attention_lstm.py` | 训练脚本（独立训练循环，复用 train_lstm.py 工具函数） |

### 修改文件

| 文件 | 改动 |
|------|------|
| `pm25_forecast/utils/paths.py` | `SUPPORTED_MODEL_NAMES` 添加 `"attention_lstm"` |
| `pm25_forecast/scripts/train_model.py` | `build_arg_parser()` 添加 `--attention-heads` 参数，`run_training()` 添加 `attention_lstm` 分支 |
| `pm25_forecast/scripts/predict_model.py` | `run_prediction()` 添加 `attention_lstm` 分支 |
| `tests/test_train_model_cli.py` | 添加 attention_lstm CLI 测试 |

## 模型定义

### `pm25_forecast/models/attention_lstm.py`

```python
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
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
            # Self-Attention
            Q = self.attention_query(lstm_out)  # (batch, seq_len, hidden_size)
            K = self.attention_key(lstm_out)
            V = self.attention_value(lstm_out)
            # Reshape for multi-head: (batch, num_heads, seq_len, head_dim)
            batch_size = Q.size(0)
            seq_len = Q.size(1)
            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
            context = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)
            # Reshape back: (batch, seq_len, hidden_size)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            # Take last time step
            last_context = context[:, -1, :]  # (batch, hidden_size)
            prediction = self.output(self.dropout(last_context))
            return prediction

    return AttentionLSTM()
```

## 训练脚本

### `pm25_forecast/scripts/train_attention_lstm.py`

- 复用 `train_lstm.py` 中的工具函数：
  - `loss_value()`, `build_target_weights()`, `resolve_peak_thresholds()`
  - `collect_model_predictions()`, `select_device()`
  - `resolve_lstm_training_paths()` → 改为调用 `resolve_attention_lstm_training_paths()`
- 训练循环与 `train_lstm.py` 完全相同（早停、学习率调度、梯度裁剪）
- 模型输出到 `models/attention_lstm/` 目录
- `training_config.json` 记录 `attention_heads` 参数

### CLI 参数

所有 LSTM 参数共享，新增：
- `--attention-heads`：注意力头数，默认 4

## 集成方式

### `pm25_forecast/utils/paths.py`

```python
SUPPORTED_MODEL_NAMES = ("lstm", "attention_lstm", "xgboost", "random_forest", "arima", "sarima")
```

### `pm25_forecast/scripts/train_model.py`

- `build_arg_parser()` 添加 `--attention-heads` 参数
- `run_training()` 添加分支：
  ```python
  if validate_model_name(args.model) == "attention_lstm":
      return train_attention_lstm.run_training(args)
  ```

### `pm25_forecast/scripts/predict_model.py`

- `run_prediction()` 添加分支：
  ```python
  if validate_model_name(args.model) == "attention_lstm":
      return predict_attention_lstm.run_prediction(args)
  ```
- 新建 `predict_attention_lstm.py` 或在 `predict_month.py` 中添加支持

## 测试

### 新增测试 (`tests/test_train_model_cli.py`)

```python
def test_parser_attention_lstm_defaults(self):
    parser = build_arg_parser()
    args = parser.parse_args(["--model", "attention_lstm"])
    self.assertEqual(args.attention_heads, 4)
    self.assertEqual(args.hidden_size, 128)
    self.assertEqual(args.num_layers, 2)
    self.assertEqual(args.dropout, 0.3)

def test_parser_accepts_attention_heads(self):
    parser = build_arg_parser()
    args = parser.parse_args(["--model", "attention_lstm", "--attention-heads", "8"])
    self.assertEqual(args.attention_heads, 8)
```

## 使用方式

```powershell
# 训练
python -m pm25_forecast.scripts.train_model --model attention_lstm --attention-heads 4 --device cuda --epochs 200

# 预测
python -m pm25_forecast.scripts.predict_model --model attention_lstm --device cuda

# 比较
python -m pm25_forecast.scripts.compare_models --models lstm attention_lstm xgboost random_forest
```

## 验收标准

1. 现有 `lstm` 模型完全不受影响
2. `attention_lstm` 可以独立训练和预测
3. 所有现有测试通过
4. 新增 CLI 测试通过
5. 训练日志包含学习率、早停信息
