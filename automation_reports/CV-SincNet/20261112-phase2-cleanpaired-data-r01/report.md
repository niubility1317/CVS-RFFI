# Phase2 三池 clean 配对数据准备

该数据作业只生成归一化地面 IQ 和对应的原任务行映射，不启动模型训练、预测或评分。输入为 Full 地面 pool 和原三池 r10 的 capsule、partition、score-side 物理 ID。独立不可覆盖输出为 `runs/20261112-phase2-cleanpaired-data-r01`。逐 RX 复用一份 clean bank；每个 pool/episode/RX 建立记录位置映射，保留原 split、support 标签和 query 补集，模型侧不输出物理 ID 或 query 标签。检查 3×23×7 个 capsule 及各 20 个 K/N split、物理池、旧类 B/C support 和 K 前缀。

目标为 `/home/szu2070436088/2510044040/CV-SincNet` 下的两个 20261029 原始数据根；数据 builder 使用 Python 3、NumPy、CPU，GPU 不参与，单进程且不修改现有任务。代码以本文件所属 Git 提交为准，单独归档传输并在服务器编译后以 `python3 -m experiments.pamf_phase2_practical.clean_pairing --raw-root .../20261029-phase2-fullpool-episodes-s2026102901-r01/builder_only/pool --task-root .../20261029-phase2-threepool-episodes-s2026102901-r01 --output-root .../20261112-phase2-cleanpaired-data-r01` 执行。任何来源/ID/split/数值错误立即停止，保留部分输出，完成标记只在全部 483 capsule、9660 split 通过时写入。数据构建不授权启动实验。

## 制作结果与限制

服务器 `completion.json` 为 `CLEAN_PAIRED_COMPLETE`：7 RX、554823 条原始物理记录、483 个 capsule、9660 个 split，CPU 单进程用时约 140.2 秒。独立只读审计逐项重新核对 483 个 capsule 的观测 ID、物理 ID、池归属及完整性，并按原复数归一化公式对 896 条记录逐值核对，结果 VERIFIED；未读取 query 标签、训练或预测。clean bank 占用约 1.1 GiB，映射约 54 MiB。不同 episode 的同一地面记录复用一份原值，不作为新观测。

本次提供 clean 可选读取入口和原生外部/交叉 worker 的接入基础，并修正交叉 LoRa 的默认邻居数为 `min(15, 2K-1)`。历史已有结果未更改。新增单任务 SCALR 和外部 worker 入口；SCALR 仅读该任务的阶段合法 query，只输出 raw/raw 与恢复/恢复，已通过模拟测试，但尚未进行真实 checkpoint 的无 query smoke。外部入口对 MoPC、Tweak M1 和 ISSL 暂时拒绝启动，避免错误版本或已知未解决的数值故障。

`clean_comparison_plan.py` 只产生四组候选矩阵，明确为 `PLANNED_NOT_LAUNCH_READY`，不是完整调度器。Tweak M1 和长预算 ISSL 的完整绑定、Orthogonal 修复版权重、MoPC 的真实非有限故障、真实 checkpoint smoke、共享 GPU 并发和独立评分屏障仍须完成及验证；不能用历史星地输入或旧方法冒充这些准备已完成。

## 2026-11-12 后续准备

只读权重清点及 CPU 严格加载/合成前向：PAMF 四 seed、DURA v1、ProtoNet/MRIOR/DADDA/Two-stage/POSTER/MoPC/LoRa 各四 seed、Tweak M1 和 ISSL fixed-v1 各四 seed 均通过相应检查；不含目标/query，DURA CPU eager 只作诊断，不能替代冻结 CUDA graph 的正式执行。Orthogonal 现有四个源权重保存的是旧伪标签配方，不满足 `group1_source_orthogonal_tempered_PL_v2`；未训练或冒用修复版权重。详见本机 `phase2_weight_inventory_20261112_v2.json` 和三个 `*_source_smoke_20261112*.json`。

新增 clean Tweak M1 与 ISSL fixed-v1 入口，前者 target support 可多条而 query 每次一条，后者固定 SSL 2000 epoch 与 transfer 300 epoch，直接 A→C 不消费 target-old support；均保留模型原预算及源状态加载核验。Tweak M1→LoRa 共享真实 B 编码器/旧 support gallery，并每条 query 独立判决；Tweak→MoPC/Orthogonal 的接口未绑定。`clean_prepare_pilot.py` 生成单 seed、K5/N20、最多五个目标的分组配置；`clean_scheduler.py` 使用单 owner、每 GPU 最多两条、启动前至少 6000 MiB 且避让其他进程，故障停止新增并保留输出，只关闭所选预测，不自动连接真值。完整矩阵的科学阈值与独立评分仍待实现与验证，计划中的 MoPC、修复版权重缺失的原生 Orthogonal 等必须单列，不能称完整矩阵可启动。

MoPC 增加显式可选诊断候选：在原非有限 guard 后用 FP64 聚合范数实施全局梯度裁剪；默认关闭，仓库原版训练路径不变。CPU 缩放线性编码器的三个合成 seed 上，原路径第 4 步真实非有限，裁剪阈值 1 的候选各完成 40 步且 40 步都触发裁剪。它实质改变优化器更新，不是作者仓库等价实现；真实 GPU 故障是否修复、结果影响和适用性均未知，正式 MoPC 仍拒绝启动。

新增 frozen DURA clean bank 构建器及 SCALR 缓存读取：同一物理记录只恢复一次，保持原三池映射；源数据按 batch 1/5/10/20/128 检查输出一致、冻结参数和非有限防护。合成缓存/现场恢复测试预测相等；服务器 CPU eager、8 条合成输入的真实 DURA 组件基准，重复三次恢复约 0.171–0.208 秒，一次恢复加三次缓存读取约 0.055–0.059 秒，输出逐值一致；不是 GPU 或完整任务加速。服务器每卡有其他 owner 的进程，因此尚未生成 clean DURA bank，也未验证目标 CUDA graph 的缓存等价或端到端吞吐。

服务器准备版 `releases/20261112-phase2-clean-preparation-r02` 从代码提交 `60f2524043a97cabb6462726bbafe8d96f727cf9` 归档，单归档本地/远端 SHA 一致，远端 compile PASS。`preparations/20261112-phase2-clean-pilot-r01` 已写出 75 个任务配置：一个 seed、五个目标、K5/N20，另独立标明 15 个阻塞方法/目标项。70 个非 PAMF 配置通过路径与绑定检查；5 个 PAMF 配置的源路径通过只读检查，但预设 DURA 缓存尚不存在，整体 `check` 因此按预期拒绝通过。未来数据缓存入口使用对应的 `runtime/pamf-00.json` 和独立输出 `runs/20261112-phase2-clean-dura-r01`，需按项目最小流程先预登记、核实空闲资源并完成真实 CUDA graph 等价检查；不得把配置生成当成实验启动。`runs/20261112-phase2-clean-pilot-r01` 尚未创建，未进行训练、目标预测或评分。

后继准备版 `releases/20261112-phase2-clean-preparation-r03` 固定提交 `c64329e687666ea78536d068f16e8034af3725c0`，增加 ISSL 两阶段 loss 记录；单归档传输校验及远端 compile 通过。新计划为 `preparations/20261112-phase2-clean-pilot-r02`，同样 75 项、70 个非 PAMF 路径检查通过、5 个 PAMF 源路径存在但恢复缓存缺失。独立读回确认两个计划均未创建正式 run root，也无本准备版调度/恢复/预测进程。原失败任务保存 B 的 CPU 对照诊断只完成脚本与登记，尚未执行；须满足项目规定的独立 P0/P1 后，才能以源/支持诊断形式评估，不据合成结果称 MoPC 已修复。
