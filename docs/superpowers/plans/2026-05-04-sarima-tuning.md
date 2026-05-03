# SARIMA 调参计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过 auto_arima 自动选择最优 SARIMA 参数，改善 SARIMA 作为比较基线的质量（当前 bias=-60, R²=-1.55 太差）。

**Architecture:** 在 `statistical_models.py` 中新增 `train_sarima_auto` 函数，使用 pmdarima.auto_arima 自动搜索最优 (p,d,q)(P,D,Q,s) 参数。在 `train_model.py` 中增加 `--sarima-auto` 开关。保留手动参数作为备选。

**Tech Stack:** pmdarima (需安装), statsmodels 0.14.6

---

## 文件变更

- Modify: `pm25_forecast/models/statistical_models.py` — 新增 `train_sarima_auto` 函数
- Modify: `pm25_forecast/scripts/train_model.py` — 新增 `--sarima-auto` CLI 标志
- Modify: `tests/test_statistical_models.py` — 新增 auto_arima 测试
- Modify: `tests/test_train_model_cli.py` — 新增 CLI 解析测试

---

### Task 1: 安装 pmdarima

- [ ] **Step 1: 安装 pmdarima 到 pm25 环境**

```powershell
conda activate pm25
pip install pmdarima
```

- [ ] **Step 2: 验证安装**

```powershell
python -c "import pmdarima; print(pmdarima.__version__)"
```

Expected: 版本号输出，无报错

---

### Task 2: 在 statistical_models.py 中添加 auto_arima 训练函数

**Files:**
- Modify: `pm25_forecast/models/statistical_models.py`

- [ ] **Step 1: 添加 train_sarima_auto 函数**

在 `train_sarima_model` 函数之后添加：

```python
def train_sarima_auto(
    series: Any,
    seasonal_period: int = 24,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    max_P: int = 2,
    max_D: int = 1,
    max_Q: int = 2,
) -> tuple[Any, dict[str, Any]]:
    try:
        from pmdarima import auto_arima
    except ImportError as exc:
        raise RuntimeError("pmdarima is required for auto SARIMA. Run: pip install pmdarima") from exc
    values = np.asarray(series, dtype=float)
    model = auto_arima(
        values,
        start_p=0, max_p=int(max_p),
        d=None, max_d=int(max_d),
        start_q=0, max_q=int(max_q),
        start_P=0, max_P=int(max_P),
        D=None, max_D=int(max_D),
        start_Q=0, max_Q=int(max_Q),
        m=int(seasonal_period),
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        enforce_stationarity=False,
        enforce_invertibility=False,
        information_criterion="aic",
    )
    order = model.order
    seasonal_order = model.seasonic_order if hasattr(model, "seasonic_order") else model.seasonal_order
    info = {
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "aic": float(model.aic()),
        "seasonal_period": int(seasonal_period),
    }
    return model, info
```

- [ ] **Step 2: 验证语法正确**

```powershell
python -c "from pm25_forecast.models.statistical_models import train_sarima_auto; print('OK')"
```

Expected: `OK`

---

### Task 3: 更新 train_model.py 添加 --sarima-auto 标志

**Files:**
- Modify: `pm25_forecast/scripts/train_model.py:71-73` (argparse 部分)
- Modify: `pm25_forecast/scripts/train_model.py:123-129` (sarima 训练分支)

- [ ] **Step 1: 添加新的 import**

在 `train_model.py` 第 17 行的 import 中添加 `train_sarima_auto`：

```python
from pm25_forecast.models.statistical_models import (
    load_train_pm25_series,
    save_statistical_model,
    train_arima_model,
    train_sarima_auto,
    train_sarima_model,
)
```

- [ ] **Step 2: 添加 CLI 参数**

在 `build_arg_parser` 中 `--sarima-seasonal-order` 之后添加：

```python
    parser.add_argument("--sarima-auto", action="store_true", help="Use auto_arima to select optimal SARIMA parameters.")
    parser.add_argument("--seasonal-period", type=int, default=24, help="Seasonal period for auto SARIMA (default: 24).")
```

- [ ] **Step 3: 修改 sarima 训练分支**

将 `train_non_lstm` 中的 sarima 分支（约第 123-129 行）改为：

```python
    elif model_name == "sarima":
        series = load_train_pm25_series(data_config)
        if args.sarima_auto:
            model, auto_info = train_sarima_auto(
                series,
                seasonal_period=args.seasonal_period,
            )
            save_statistical_model(model, model_path)
        else:
            model = train_sarima_model(
                series,
                tuple(args.sarima_order),
                tuple(args.sarima_seasonal_order),
            )
            save_statistical_model(model, model_path)
            auto_info = None
```

同时在 `config` 字典中记录 auto_info，将 `config["training"]` 改为：

```python
    config = {
        "model_name": model_name,
        "model_path": str(model_path),
        "data_config": data_config,
        "training": {
            "seed": int(args.seed),
            "elapsed_seconds": float(time.time() - start),
        },
    }
    if model_name == "sarima" and args.sarima_auto and auto_info is not None:
        config["training"]["auto_arima"] = auto_info
    write_json(out_dir / "training_config.json", config)
```

- [ ] **Step 4: 验证 CLI 解析**

```powershell
python -c "from pm25_forecast.scripts.train_model import build_arg_parser; args = build_arg_parser().parse_args(['--model', 'sarima', '--sarima-auto', '--seasonal-period', '12']); print(args.sarima_auto, args.seasonal_period)"
```

Expected: `True 12`

---

### Task 4: 添加测试

**Files:**
- Modify: `tests/test_statistical_models.py`
- Modify: `tests/test_train_model_cli.py`

- [ ] **Step 1: 添加 train_sarima_auto 的单元测试**

在 `tests/test_statistical_models.py` 中添加：

```python
class SarimaAutoTests(unittest.TestCase):
    def test_train_sarima_auto_returns_model_and_info(self):
        np.random.seed(42)
        series = np.random.randn(200).cumsum() + 50
        model, info = train_sarima_auto(series, seasonal_period=12, max_p=1, max_q=1, max_P=1, max_Q=1)
        self.assertTrue(hasattr(model, "forecast"))
        self.assertIn("order", info)
        self.assertIn("seasonal_order", info)
        self.assertIn("aic", info)
        self.assertEqual(len(info["order"]), 3)
        self.assertEqual(len(info["seasonal_order"]), 4)
```

同时在文件顶部 import 中添加 `train_sarima_auto`：

```python
from pm25_forecast.models.statistical_models import forecast_statistical_model, load_train_pm25_series, train_sarima_auto
```

- [ ] **Step 2: 添加 CLI 解析测试**

在 `tests/test_train_model_cli.py` 中添加：

```python
    def test_parser_accepts_sarima_auto_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--model", "sarima", "--sarima-auto", "--seasonal-period", "12"])
        self.assertTrue(args.sarima_auto)
        self.assertEqual(args.seasonal_period, 12)

    def test_parser_defaults_sarima_auto_off(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--model", "sarima"])
        self.assertFalse(args.sarima_auto)
        self.assertEqual(args.seasonal_period, 24)
```

- [ ] **Step 3: 运行所有测试**

```powershell
python -m unittest discover -s tests -v
```

Expected: 所有测试通过

- [ ] **Step 4: 提交**

```powershell
git add pm25_forecast/models/statistical_models.py pm25_forecast/scripts/train_model.py tests/test_statistical_models.py tests/test_train_model_cli.py
git commit -m "feat: add auto_arima support for SARIMA parameter tuning"
```

---

### Task 5: 使用 auto_arima 重新训练 SARIMA

- [ ] **Step 1: 用 auto_arima 训练 SARIMA（seasonal_period=24）**

```powershell
python -m pm25_forecast.scripts.train_model --model sarima --sarima-auto --seasonal-period 24 --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

记录输出中的 order、seasonal_order、aic 值。

- [ ] **Step 2: 用新模型进行预测**

```powershell
python -m pm25_forecast.scripts.predict_model --model sarima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

- [ ] **Step 3: 查看新指标并与旧结果对比**

```powershell
cat pm25_forecast/outputs/window_720h_to_72h/predictions/start_2026_03_01_0000/sarima/metrics.json
```

对比旧指标：
- 旧 RMSE: 71.59, MAE: 63.20, R²: -1.545, bias: -60.59
- 目标：bias 显著收窄（绝对值 < 30），R² 接近 0 或为正

- [ ] **Step 4: 如果 m=24 效果不理想，尝试 m=12**

```powershell
python -m pm25_forecast.scripts.train_model --model sarima --sarima-auto --seasonal-period 12 --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
python -m pm25_forecast.scripts.predict_model --model sarima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
cat pm25_forecast/outputs/window_720h_to_72h/predictions/start_2026_03_01_0000/sarima/metrics.json
```

选择 m=24 和 m=12 中指标更好的结果。

- [ ] **Step 5: 更新 ARIMA 预测结果（如需要重新预测以确保一致性）**

```powershell
python -m pm25_forecast.scripts.predict_model --model arima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

- [ ] **Step 6: 汇总比较**

```powershell
python -m pm25_forecast.scripts.compare_models --models arima sarima --input-window 720 --output-window 72 --predict-start "2026-03-01 00:00:00+08:00"
```

查看 `comparisons/start_2026_03_01_0000/model_metrics.json` 中两个模型的对比。

- [ ] **Step 7: 提交最终结果**

```powershell
git add pm25_forecast/outputs/
git commit -m "feat: re-train SARIMA with auto_arima tuned parameters"
```
