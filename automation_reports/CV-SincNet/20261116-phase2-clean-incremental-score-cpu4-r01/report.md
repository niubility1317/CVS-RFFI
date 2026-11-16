# Phase 2 clean 逐任务独立评分

用户于 2026-11-16 要求不等待全矩阵，先评分已完成实验并持续评分后续完成任务。本 run 仅扫描现有 clean 对比的合法 scheduler 及 `mopc-stable-*` 接续计划；原生、PAMF、混合方法各按来源计划和方法版本分别记录。旧 ProtoNet 原桥为历史独立 arm，不并入 affine 或 `clip1000_v1`。不启动、重启、暂停或改变预测训练；不以目标结果修改方法、参数、队列或 checkpoint，未齐矩阵不得写成完整比较。

只对 `PREDICTIONS_COMPLETE` 且任务本身预登记 K/N 预测全部固定的输出执行：先校验任务身份、方法、split/K/N、预测文件、query 补集、类别范围和完整性，再独立读取对应 score-only 真值并计算 natural 与可用的 common-K 指标。每个任务写独立不可覆盖结果，保留 pool、RX、episode、seed、K、N、输入路径和版本；任务来源或 K 集不同不合并。损坏任务单独登记失败并继续其他任务；缺少 K 集时 common-K 标为不可用。输出 `runs/20261116-phase2-clean-incremental-score-cpu4-r01/jobs/<原计划>/<任务>`，进度与失败记录在同 run，最多 4 个 CPU worker，每 120 秒发现后续完成任务。

发布前核实两个既有本任务全矩阵 watcher 的 PID、CWD、命令及 WAITING／truth 未打开且无评分子进程，精确停止它们并保留输出；新 watcher 接替其自动评分职责，避免两套评分池叠加超过 4。其他 owner 及所有训练/预测 worker 保持。若旧 watcher 已进入评分或身份不符，不能盲停，先核实实际进程。数据和原始预测、真值、日志不进入 Git。此前诊断和 pilot 结果按实际版本及不完整范围报告，旧已暴露结果不声称新的独立盲测。

实际发布：旧 watcher PID 284205、1473723 在确认 WAITING、无子进程后以 pidfd 精确停止并核实退出；MoPC 控制器 1473542 保持运行。逐任务评分 owner PID 2726877，实际四个 CPU worker；首批扫描发现 27408 个可评分任务，初次进度已有 862 项评分完成、发现错误 0，队列持续处理，尚未完成全部评分。预测任务完成后新结果与存量结果交替入队。统计口径为每个独立来源任务及其现有 K/N，不跨方法版本合并。

首个已评分的指定检查例 `20261113-phase2-clean-hybrid-pilot-r01/hybrid-dadda_sda-00`：P01/MIX01、RX1-1、seed2026092701、clean raw、K5/N20、26471 条 query。DADDA→Orthogonal 的 overall accuracy 31.63%、旧类 macro accuracy 0%、新类 macro accuracy 49.07%、macro H 0%；DADDA→LoRa 分别为 55.71%、68.74%、50.70%、58.35%。两者属于同 B 的单个 pilot 任务，不代表全部实验；该 pilot 仅 K5，common-K 无法计算。不得用这些目标结果改变正在运行的固定配方。
