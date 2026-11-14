# Phase 2 clean 自动评分等待进程 r02

接续 r01 观察器，保持相同固定矩阵、全局预测完成后逐项验证和独立 truth-last 评分，使用最多 4 个 CPU 评分进程。新增对新 scheduler 计划暂时半写 JSON 的处理：暂不计入覆盖，连续三次内容未变且仍无法解析则失败并保留日志。按精确 PID 验证 r01 只处于 WAITING 后停止其 owner，再使用独占 `runs/20261115-phase2-clean-auto-score-m4-r02` 启动 r02，不触碰任何预测 worker 或已有输出。

输入为项目 preparations 下的合法 clean 队列，自动发现后续队列；不接受已知失败旧 pilot。未完成任何方法、阶段或 K/N 时始终不打开 truth。每 300 秒检查一次，发生冲突或损坏时留存错误并停止，不自动声称完整评分。

2026-11-15 约 03:10 CST：旧 PID 4185761 在 WAITING 且真值未打开时按 PID/CWD/命令核实停止；新 PID 4822 已在对应 r02 release 运行。独立读回 r02 状态 WAITING、32844 单元中 25316 单元不完整、truth_opened=false；检查频率 300 秒。代码 `9963ae1869ba9b821479708700b839b95aab7646`，release 归档本地与远端 SHA256 一致、远端 compile PASS。排队进程实际能否在缺失实验完成后评分尚待事件验证。
