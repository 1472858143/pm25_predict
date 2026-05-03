# Rename Reproduce Package Design

## 背景

`Reproduce/` 最初表示论文复现实验，但当前项目已经扩展为多模型 PM2.5 预测与对比工程。继续使用 `Reproduce` 会让 Python 模块入口、输出目录和文档语义都偏旧。

## 方案

将顶层包目录 `Reproduce/` 重命名为 `pm25_forecast/`。同步更新：

- Python 导入路径：`from Reproduce...` -> `from pm25_forecast...`
- 模块运行命令：`python -m Reproduce.scripts...` -> `python -m pm25_forecast.scripts...`
- 默认输出目录：`pm25_forecast/outputs/...`
- 测试和文档中的路径引用。

## 兼容性

不迁移旧的 `Reproduce/outputs` 实验产物，也不删除用户已有输出。新命令会写入 `pm25_forecast/outputs`。旧 `python -m Reproduce...` 入口不保留兼容别名，避免项目里同时存在两个包名。

## 验证

- 单元测试全部通过。
- 主要 CLI 的 `--help` 可运行。
- `prepare_data` 能生成 `720h -> 72h` 数据包。
