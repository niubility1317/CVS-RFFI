# ISSL 短预算学习率配对诊断

复用 `20261113-phase2-clean-native-pilot-r01` 已完成 ISSL 的源权重、五个 capsule、seed 2026092701 和 K5/N20 的 clean support/query；与既有原学习率短预算和 2000/300 长预算逐接收机配对。历史目标结果已暴露，本实验为开发评估，并非独立盲测。

固定四组：自监督／迁移轮数 50/25 或 100/50，各自监督初始学习率为 0.00003 或 0.0001；对照为原 0.00001。迁移学习率保持 0.03，优化器、StepLR、损失、数据、seed 与权重来源均保留。每组五个 capsule，共 20 项；仅用目标新类 support 以及授权源 replay，query 只用于预测。全部预测固定后自动调用独立 truth-last CPU 评分；不按新结果中途变更参数。

唯一运行根 `runs/20261114-phase2-issl-short-lr-pilot-m1-r01`，准备根同名，使用物理 GPU 5，最多两个本 run 进程，启动前至少 6000 MiB 可用显存，遇其他 owner 占用避让。每 worker 加载真实源权重后先进行 source-only 无 query smoke；故障停止派发并排空，保留全部输出。已有短预算和长预算预测不覆盖、不重训。本侧会话不能调用独立子代理，因此未进行独立 P0/P1 审查；用户此前在知悉此限制后明确要求直接启动该短预算对照。

调度命令为 `python -m experiments.pamf_phase2_practical.clean_scheduler dispatch --config preparations/20261114-phase2-issl-short-lr-pilot-m1-r01/scheduler.json`，工作目录为本 run 独立 release，实际代码版本以该 release 来源 Git commit 登记为准。预期每项生成预测与完成标记，全部完成后 `scores/` 保存逐项指标；GPU 负载变化会影响耗时。
