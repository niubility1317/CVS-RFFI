# Phase2 星地原生域适应缺项接续

本次只补 MRIOR-SDA、DADDA-SDA、POSTER-FT 的原生 A0/B。三池、七接收机、四模型种子，MIX02–MIX08，每个 capsule/seed 的 K1、K10、K20；每 K 两条 A0/B 预测，每个方法 588 个任务、3528 条新预测。K5 及 MIX01 已评分行按实际历史文件保留，不重复执行。LS/MU/LU 维持原 K5 工作点并沿用历史结果。其他方法不包含在本队列。

数据为既有三池 `20261029-phase2-threepool-episodes-s2026102901-r01` 的固定星地 received；source 权重使用 `20261030-phase2-group1-concat-midPL-threepool-m4-r01` 对应方法/seed 的 source.pt。使用各原生 B 配方和预算，当前旧类 support 可见；query 不参与适应，预测固定并核验后逐任务自动独立 truth-last 评分，不等待矩阵收齐。历史来源和部分缺失的其他 profile 独立记录，不以低分筛选任务。

清点来源是旧 Group1 预测与独立评分的实际元数据和文件，执行清点 `phase2_satellite_native_reuse_inventory_20261116.py`；每个选中方法 4368 行已评分、3528 行缺失，校验对应 profile 的 1933 个评分任务且无清点错误。初始旧索引中的 ProtoNet 和 Two-stage 评分路径缺失；后来找到 20261101 恢复索引并重新逐行核实，两种方法各有 4368 行已评分、3528 行缺失，恢复索引的五种方法均无清点错误。本队列仅包含最初选中的三种方法。

N607：代码以包含本报告的 commit 标识；固定 source runtime 来自 `20261113-phase2-clean-native-pilot-r01`，准备时移除 clean/restored 指针并只使用 received。`python -m experiments.pamf_phase2_practical.satellite_native_da prepare` 创建唯一预备计划，`dispatch --config <plan>/scheduler.json` 发布。新运行及准备根为 `runs/20261116-phase2-satellite-native-da-m4-r01` 与同名 preparations；发布归档和远端 CWD 为同名 release。GPU 0、1、5、6、7，每 GPU 本队列最多 7 worker、启动前至少 6000 MiB、每新 worker 预留 3000 MiB，其他 owner 不动。预测 worker 首步运行真实源 checkpoint 无 query smoke；评分进程最多 1 个。技术/资源/输入故障停止新派发，自然排空并保留输出；低分不停止或调参。任务完成产出逐任务预测、独立评分、日志和资源记录。
