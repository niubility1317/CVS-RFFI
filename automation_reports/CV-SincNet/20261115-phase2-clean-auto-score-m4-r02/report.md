# Phase 2 clean 自动评分等待进程 r02

接续 r01 观察器，保持相同固定矩阵、全局预测完成后逐项验证和独立 truth-last 评分，使用最多 4 个 CPU 评分进程。新增对新 scheduler 计划暂时半写 JSON 的处理：暂不计入覆盖，连续三次内容未变且仍无法解析则失败并保留日志。按精确 PID 验证 r01 只处于 WAITING 后停止其 owner，再使用独占 `runs/20261115-phase2-clean-auto-score-m4-r02` 启动 r02，不触碰任何预测 worker 或已有输出。

输入为项目 preparations 下的合法 clean 队列，自动发现后续队列；不接受已知失败旧 pilot。未完成任何方法、阶段或 K/N 时始终不打开 truth。每 300 秒检查一次，发生冲突或损坏时留存错误并停止，不自动声称完整评分。
