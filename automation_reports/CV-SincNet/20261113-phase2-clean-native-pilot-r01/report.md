# Phase2 clean 原生对照首轮

启动范围为已核验的六项原生 A→B 和 LoRa、ISSL fixed-v1 长预算两项原生 A→C。复用 20261113 clean-pilot-r03 已固定的五个 capsule、模型种子 2026092701 和 K5/N20，共 40 个作业；B 评测六旧类，直接 C 只使用目标新类支持。Tweak M1 每次只读一条 query。原生 Orthogonal 与 MoPC 待各自输入及故障解决后登记；全矩阵、PAMF 和交叉组合不在本次派发范围。

N607 单 owner，物理 GPU 1、2、3、4，每卡最多 5 个进程，启动前保留至少 6000 MiB 可用预算并计入在途进程尚未使用的预留。出现其他 GPU 进程时跳过该卡。正式输出固定为 `runs/20261113-phase2-clean-native-pilot-r01`；配置为 `preparations/20261113-phase2-clean-native-pilot-r01/scheduler.json`，使用此报告所属提交的 `releases/20261113-phase2-clean-native-pilot-r01-release`。

从 release 根以 N607 CVS-RFFI 解释器执行 `python -u -m experiments.pamf_phase2_practical.clean_scheduler dispatch --config <project>/preparations/20261113-phase2-clean-native-pilot-r01/scheduler.json`。每个作业首先用真实源权重和源样本执行无 query smoke。任何派发或 worker 故障停止新派发，排空已启动进程，保留日志与部分输出。作业完成写独立预测索引和完成标记，调度器只写本子集固定标记，保持全局预测和真值评分屏障关闭。
