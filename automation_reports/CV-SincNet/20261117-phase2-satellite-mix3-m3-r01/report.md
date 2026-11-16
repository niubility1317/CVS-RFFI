# Phase2 星地三场景对比

状态：准备发布。代码版本为本报告所在提交；调度配置和各方法实际输出记录在 N607 同名 preparation 与 run 中。

数据采用既有 `p2_min_v1`、`VALIDATED_ONCE` 三池星地接收信号及其 capsule/split，训练使用合法 source/support，固定预测后逐任务独立 truth-last 评分。对比方法仅 MIX01–03、七接收机、模型种子 2026092701/02/03；PAMF＋SCALR 的 MIX01–08 既有结果单独复用。取消 DADDA、ISSL、RadioNet、Hu 和 RA；不运行 C 源主干接 B 的组合。

原生 B：复用已完成预测和评分，仅对 MRIOR、Two-stage、POSTER 缺失的 K=1/10/20 分别补齐；ProtoNet 既有预测全部固定，原评分 owner 继续处理未评分项。Tweak M1 为逐条 query 判决，按 K=1/5/10/20 运行。

原生新类注册：Orthogonal 修复源权重、平衡样本库 LoRa 和 clip1000 MoPC 使用 K=5/10、N=5/10；原生 MoPC 和 LoRa 的既有同源 A0 经清点复用。B→C 当前发布五种 B 到 MoPC 的组合，同一 K 的 B 状态保存供其余注册方法复用；ProtoNet 使用 affine 桥。其余 B→LoRa/Orthogonal 组合仍须补全共享 B 接口和验证，不能算本次已发布。

N607 单 owner，物理 GPU 0–7 均可纳入、每卡本任务最多 6 个训练/预测进程；尊重其他 owner、启动前检查显存，配置剩余显存下限 6000 MiB。每任务冻结后立即进入独立评分队列。遇真实 MoPC 非有限则隔离该项并保留产物，其他系统性技术失败停止新增派发并排空；有限梯度裁剪不算失败。

执行目录和归档：`releases/20261117-phase2-satellite-mix3-m3-r01-release`；配置 `preparations/20261117-phase2-satellite-mix3-m3-r01/scheduler.json`；不可覆盖输出 `runs/20261117-phase2-satellite-mix3-m3-r01`。执行环境 N607 的 CVS-RFFI Python；入口 `python -m experiments.pamf_phase2_practical.satellite_mix3 dispatch --config <scheduler.json>`。预期包括逐任务预测、保存的 B 状态、评分、调度和错误记录；query 真值仅独立评分入口读取。
