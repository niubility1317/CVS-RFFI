# Phase2 三池 clean 配对数据准备

该数据作业只生成归一化地面 IQ 和对应的原任务行映射，不启动模型训练、预测或评分。输入为 Full 地面 pool 和原三池 r10 的 capsule、partition、score-side 物理 ID。独立不可覆盖输出为 `runs/20261112-phase2-cleanpaired-data-r01`。逐 RX 复用一份 clean bank；每个 pool/episode/RX 建立记录位置映射，保留原 split、support 标签和 query 补集，模型侧不输出物理 ID 或 query 标签。检查 3×23×7 个 capsule 及各 20 个 K/N split、物理池、旧类 B/C support 和 K 前缀。

目标为 `/home/szu2070436088/2510044040/CV-SincNet` 下的两个 20261029 原始数据根；数据 builder 使用 Python 3、NumPy、CPU，GPU 不参与，单进程且不修改现有任务。代码以本文件所属 Git 提交为准，单独归档传输并在服务器编译后以 `python3 -m experiments.pamf_phase2_practical.clean_pairing --raw-root .../20261029-phase2-fullpool-episodes-s2026102901-r01/builder_only/pool --task-root .../20261029-phase2-threepool-episodes-s2026102901-r01 --output-root .../20261112-phase2-cleanpaired-data-r01` 执行。任何来源/ID/split/数值错误立即停止，保留部分输出，完成标记只在全部 483 capsule、9660 split 通过时写入。数据构建不授权启动实验。

## 制作结果与限制

服务器 `completion.json` 为 `CLEAN_PAIRED_COMPLETE`：7 RX、554823 条原始物理记录、483 个 capsule、9660 个 split，CPU 单进程用时约 140.2 秒。独立只读审计逐项重新核对 483 个 capsule 的观测 ID、物理 ID、池归属及完整性，并按原复数归一化公式对 896 条记录逐值核对，结果 VERIFIED；未读取 query 标签、训练或预测。clean bank 占用约 1.1 GiB，映射约 54 MiB。不同 episode 的同一地面记录复用一份原值，不作为新观测。

本次提供 clean 可选读取入口和原生外部/交叉 worker 的接入基础，并修正交叉 LoRa 的默认邻居数为 `min(15, 2K-1)`。历史已有结果未更改。新增单任务 SCALR 和外部 worker 入口；SCALR 仅读该任务的阶段合法 query，只输出 raw/raw 与恢复/恢复，已通过模拟测试，但尚未进行真实 checkpoint 的无 query smoke。外部入口对 MoPC、Tweak M1 和 ISSL 暂时拒绝启动，避免错误版本或已知未解决的数值故障。

`clean_comparison_plan.py` 只产生四组候选矩阵，明确为 `PLANNED_NOT_LAUNCH_READY`，不是完整调度器。Tweak M1 和长预算 ISSL 的完整绑定、Orthogonal 修复版权重、MoPC 的真实非有限故障、真实 checkpoint smoke、共享 GPU 并发和独立评分屏障仍须完成及验证；不能用历史星地输入或旧方法冒充这些准备已完成。
