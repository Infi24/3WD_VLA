# 协调结论（当前采用）

当前只保留并采用你项目已有主链路：

1. 同学端发送两路图像（Top / Hand）
2. 你这边运行 `share/share_viewer.py` 接收并合成
3. 输出标准图：`brain_ready.jpg`（`Top | 灰分隔 | Hand`）
4. `bigbrain/pipeline_runner.py` 从 `monitor_dir` 读取最新图执行 `selector / 3dw / sequential`

## 说明

- 不再引入额外桥接脚本。
- 以你仓库现有脚本为准：`run_all.sh`、`run_all_offline.sh`、`run_3dw.sh`、`run_selector.sh`。
