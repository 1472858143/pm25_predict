# PM2.5 多模型预测对比 Web 仪表盘设计

> **创建日期：** 2026-05-04
> **目标：** 在项目根目录新增 `web/` 子项目，提供基于浏览器的多模型预测结果可视化对比仪表盘。

## 1. 目标与范围

构建一个本地运行的 Web 仪表盘，读取 `pm25_forecast/outputs/` 下已生成的预测结果，提供三个核心视图：

1. **模型指标对比表** —— 横向比较 6 个模型的 RMSE / MAE / MAPE / SMAPE / R² / bias
2. **预测曲线对比图** —— 在同一时间轴上叠加真实值与各模型预测值
3. **单模型误差分析** —— 散点图 (y_true vs y_pred) + 残差直方图

**非目标（YAGNI）**：
- 不在 Web 上触发训练或预测
- 不做用户登录、权限管理
- 不做移动端响应式（桌面 1280px+ 优先）
- 不做 SSR、E2E 测试

## 2. 架构与目录结构

**双层结构**：FastAPI 后端 + React/TypeScript/Vite 前端，独立项目，HTTP/JSON 通信。

```
web/
├── backend/                  # FastAPI 服务
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI 入口 + CORS
│   │   ├── routes.py         # API 路由
│   │   ├── data_loader.py    # 读取 outputs/ 目录
│   │   ├── schemas.py        # Pydantic 模型
│   │   └── config.py         # 环境变量、路径配置
│   ├── tests/
│   │   ├── fixtures/         # 最小测试 outputs 目录
│   │   └── test_*.py
│   ├── pyproject.toml
│   └── README.md
└── frontend/                 # React + TS + Vite
    ├── src/
    │   ├── App.tsx
    │   ├── main.tsx
    │   ├── api/              # axios 客户端 + 接口封装
    │   ├── components/       # 复用 UI 组件
    │   ├── pages/            # Dashboard 页面 + 三模块
    │   ├── types/            # TypeScript 类型定义
    │   ├── utils/            # formatWindow, formatStart 等
    │   └── styles/
    ├── tests/
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    └── README.md
```

**外部依赖**：
- 后端读取项目根目录的 `pm25_forecast/outputs/`，路径通过环境变量 `OUTPUT_ROOT` 配置（默认相对路径 `../../pm25_forecast/outputs`）。
- 前端在 dev 阶段通过 Vite proxy 将 `/api/*` 转发到 `http://localhost:8000`。

## 3. 后端 API 设计

**只读接口**，6 个端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查，返回 `{"status": "ok"}` |
| GET | `/api/windows` | 扫描 outputs 目录，返回所有窗口及可用 starts |
| GET | `/api/metrics` | 返回所有模型的指标汇总（用于对比表） |
| GET | `/api/predictions` | 返回所有模型的合集预测（用于曲线图叠加） |
| GET | `/api/predictions/{model_name}` | 返回单个模型的完整预测（用于误差分析） |
| GET | `/api/horizon-metrics/{model_name}` | 返回单模型逐 horizon 指标（备用，便于扩展） |

所有数据接口均接受 `?window=<window_name>&start=<start_id>` 查询参数；缺省时使用首个可用窗口/起点。

### 3.1 响应示例

**`GET /api/windows`**：
```json
{
  "windows": [
    {
      "name": "window_720h_to_72h",
      "input_window": 720,
      "output_window": 72,
      "starts": ["start_2026_03_01_0000"]
    },
    {
      "name": "window_168h_to_72h",
      "input_window": 168,
      "output_window": 72,
      "starts": []
    }
  ]
}
```

**`GET /api/metrics?window=window_720h_to_72h&start=start_2026_03_01_0000`**：
```json
{
  "window": "window_720h_to_72h",
  "start": "start_2026_03_01_0000",
  "predict_start": "2026-03-01 00:00:00+08:00",
  "models": [
    {
      "model_name": "lstm",
      "RMSE": 36.78, "MAE": 27.91, "MAPE": 30.42,
      "SMAPE": 25.74, "R2": 0.33, "bias": -12.57
    }
  ],
  "missing_models": []
}
```

**`GET /api/predictions?window=...&start=...`**：
```json
{
  "window": "window_720h_to_72h",
  "start": "start_2026_03_01_0000",
  "horizons": [1, 2, ..., 72],
  "timestamps": ["2026-03-01 00:00:00+08:00", ...],
  "y_true": [107.30, 107.60, ...],
  "predictions": {
    "lstm": [132.10, 127.93, ...],
    "attention_lstm": [...],
    "xgboost": [...]
  }
}
```

**`GET /api/predictions/{model_name}?window=...&start=...`**：返回该模型的完整 predictions.csv 解析后的对象数组（含 y_true / y_pred / y_pred_model / error / abs_error / relative_error 等字段）。

### 3.2 错误处理

- 请求的 window/start 不存在 → 404，`{"detail": "Window 'xxx' not found"}`
- 模型目录缺失（如某模型未训练）→ 不报错，从 `models` 数组中跳过；同时在 `missing_models` 字段中列出
- CSV/JSON 解析失败 → 500，错误信息含文件路径与异常类型
- `outputs/` 根目录不存在 → 启动时打印警告，`/api/windows` 返回 `{"windows": []}`

## 4. 前端页面设计

**单页面 + 三模块自上而下滚动**，无路由切换。

### 4.1 整体布局

```
┌────────────────────────────────────────────────────────────────────────┐
│ Header                                                                  │
│   PM2.5 多模型预测对比仪表盘                                              │
│   时间窗口: [30 天历史 → 3 天预测 (720h→72h)  ▼]                        │
│   预测起点: [2026-03-01 00:00 (北京时间)       ▼]                       │
├────────────────────────────────────────────────────────────────────────┤
│ § 1. 模型指标对比                                                       │
│   [Ant Design Table]                                                   │
│   model | RMSE↑↓ | MAE | MAPE | SMAPE | R² | bias                      │
│   每列最优值高亮（绿底），attention_lstm 行加粗                         │
├────────────────────────────────────────────────────────────────────────┤
│ § 2. 预测曲线对比                                                       │
│   [ECharts 折线图]                                                     │
│   x: timestamp（横轴 72 小时）                                          │
│   y: PM2.5 (μg/m³)                                                     │
│   series: y_true（粗黑实线）+ 各模型 y_pred                            │
│   图例可点击切换显示/隐藏，dataZoom 区域缩放，tooltip 显示所有模型当前值 │
├────────────────────────────────────────────────────────────────────────┤
│ § 3. 单模型误差分析                                                     │
│   模型选择器 [attention_lstm ▼]（默认 attention_lstm）                  │
│   ┌────────────────────────────┬────────────────────────────┐         │
│   │ 散点图                      │ 残差直方图                   │         │
│   │ ECharts scatter            │ ECharts bar                │         │
│   │ x: y_true, y: y_pred       │ x: error 区间, y: 频次        │         │
│   │ 叠加 y=x 参考线              │ 叠加均值/标准差注释            │         │
│   └────────────────────────────┴────────────────────────────┘         │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 显示文本格式化

**窗口标签**（前端 `formatWindow()` 工具函数实现）：
- `window_720h_to_72h` → `30 天历史 → 3 天预测 (720h→72h)`
- `window_168h_to_72h` → `7 天历史 → 3 天预测 (168h→72h)`
- 规则：小时整除 24 显示为"X 天"，否则显示"Yh"；后缀附原始 ID 便于核对

**预测起点标签**（前端 `formatStart()` 工具函数实现）：
- `start_2026_03_01_0000` → `2026-03-01 00:00 (北京时间)`
- 规则：解析 `start_YYYY_MM_DD_HHMM` 字符串并重新格式化

**API 契约不变**：后端始终返回原始 `name` / `start` 字段，前端负责显示转换。

### 4.3 关键技术选型

| 用途 | 选型 | 理由 |
|------|------|------|
| 图表库 | **ECharts** | 中文友好、内置缩放/图例交互、性能好 |
| 组件库 | **Ant Design** | Table、Select、Card、Layout 一站式 |
| HTTP | **axios** | 简洁，配合 `useEffect` 即可（无需 React Query） |
| 状态 | React `useState` + `useEffect` | YAGNI，不引入 Redux/Zustand |

**模型固定调色板**：
- `lstm` 蓝、`attention_lstm` 红（核心模型突出）、`xgboost` 绿、`random_forest` 紫、`arima` 橙、`sarima` 灰
- `y_true` 粗黑实线

### 4.4 用户交互

- 启动时调 `/api/windows`，默认选第一个有 `starts` 的窗口 + 该窗口首个 start
- 切换窗口时自动选中该窗口下首个 start；若无 start，禁用 start 下拉并显示"该窗口暂无预测结果"
- 切换任一选择器后，三个模块自动重新加载数据
- URL 参数同步：`?window=window_720h_to_72h&start=start_2026_03_01_0000`，方便分享/刷新恢复
- API 请求期间，对应模块显示 Ant Design Skeleton loading
- API 失败 → `message.error()` + 模块内显示"加载失败，点击重试"

## 5. 错误处理总结

| 场景 | 后端 | 前端 |
|------|------|------|
| outputs 目录缺失 | 启动警告 + 空 `/api/windows` | 显示"未发现任何预测输出"空状态 |
| window 不存在 | 404 | message.error |
| 模型目录缺失 | 跳过 + `missing_models` | 表格中省略该行，添加提示 |
| CSV 解析失败 | 500 + 详情 | message.error，区域显示错误卡片 |
| 网络断开 | — | axios 超时提示 + 重试按钮 |

## 6. 测试策略

| 层级 | 工具 | 范围 |
|------|------|------|
| 后端单元 | `pytest` | data_loader 解析逻辑、windows 扫描、metrics 聚合 |
| 后端 API | `pytest` + `httpx.AsyncClient` | 接口响应结构、参数校验、404/500 分支 |
| 前端单元 | `vitest` | `formatWindow`、`formatStart` 等工具函数 |
| 前端组件 | `@testing-library/react` | 各模块渲染、API mock、空状态展示 |
| E2E | 不做 | YAGNI |

**测试 fixture**：`web/backend/tests/fixtures/outputs/` 放一个最小目录（一个窗口、一个 start、两个模型 lstm + attention_lstm），fixture 数据通过 pytest 的 `tmp_path` 配合环境变量注入，不依赖真实输出文件。

## 7. 启动与部署

**后端**：
```powershell
cd web/backend
pip install -e .
uvicorn app.main:app --reload --port 8000
```

**前端**：
```powershell
cd web/frontend
npm install
npm run dev   # http://localhost:5173
```

`web/frontend/vite.config.ts` 配置 dev proxy 将 `/api/*` 转发到 `http://localhost:8000`。

**生产部署**（暂不实现，仅记录）：可通过 `npm run build` 生成 dist，由 FastAPI StaticFiles 托管。

## 8. Git 与代码组织

- 新建 `web/.gitignore`：排除 `node_modules/`、`dist/`、`__pycache__/`、`*.egg-info/`、`.pytest_cache/`
- `web/backend` 和 `web/frontend` 各自独立 README 说明启动方式
- 不修改现有 `pm25_forecast/` 包的任何代码

## 9. 未来扩展（不在本次范围）

- 支持触发新预测/训练（POST 接口）
- 支持多预测起点对比
- 支持下载预测 CSV / 截图导出
- horizon metrics 视图（`/api/horizon-metrics` 接口已预留）
