# Phase 2 clean 自动评分等待进程 r03

接续 r02，其他方法、数据矩阵和 truth-last 屏障保持。ProtoNet→MoPC 主比较要求明确的 `mopc_hr_affine_v1` 标签，原二次桥 MoPC-only 行留作历史，不与新版本拼成一个方法；原 ProtoNet→Orthogonal/LoRa 不受影响。未完成的新版本不能用旧版本充数。对所有保留组的预测完成并预检后自动用最多 4 个 CPU 进程评分。

启动前准确核对 r02 仍为 WAITING 且未打开 truth，仅停止该 owner；新 owner 使用独占 `runs/20261115-phase2-clean-auto-score-m4-r03`。每 300 秒检查一次；单个完成 marker 或新 scheduler 计划短暂半写时暂不计入，稳定损坏则失败保留日志。原结果、未完成任务和其他 GPU/owner 进程不改变。
