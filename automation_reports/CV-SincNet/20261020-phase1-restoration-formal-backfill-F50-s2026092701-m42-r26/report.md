# 既有恢复器正式测试补齐 R26

状态：LOCAL_VERIFIED，未启动。方法 commit `12c1f236c9ac9d2f883340c9a3dfbb4564877930` 已推送、远端 OID 一致。52 项聚焦检查与一次独立 P0/P1 审查通过。


用户于 2026-10-20 明确要求“性能还要看测试集表现，前面没有跑测试集要补上”。本轮固定审计此前本对话全部已完成的 TX/F 无标签恢复器，覆盖 R01、R02、R03、R07、R10、R12、R15 至 R24，共 42 个原始训练行。6 行复用相同 F50、默认头、物理 ID、原始 received IQ 与七视图的完整正式结果；另外 36 行补测。完整逐行来源、代码提交、原 checkpoint、六类 seed、运行适配与复用指向保存在 `experiments/pamf_backfill_restore/design.py` 和本 run 登记中。

这次测试准入依据是用户明确的补测要求，不是历史源域筛选通过。源域失败的权重照常报告实际测试表现；不修改历史 source_decision，不把失败改写为通过，不重新排序或调整其他训练。42 行及采样方式在首次新增目标预测前固定。所有 checkpoint 沿用各自原始严格 loader、原合法 7/63/30 数据契约及完整继承来源；仅最终 E20/4940，L126000/U1134000，TX/F/target 训练访问为零。原 F50 的训练包含源域身份监督，整条识别链不称为完全无监督。

R06 是接收算子源诊断，没有新训练权重；R13/R14 是已知算子的源诊断，包含目标运行时不可获得的 oracle 信息；R25 是源域特征统计。它们不作为可部署的已训练恢复器补测。R04/R05/R09/R11 是既有正式测试；R08 被 R09 替代；R16 的 FFT reference、R17 的 single_shot_reference 是已有训练行的引用，避免重复计数。其他对话的 F 特征指导/语义监督恢复器不属于本轮“恢复训练零 F 访问”固定矩阵。

本轮不训练，原冻结 F50 seed2026092701 和原联合训练默认头直接预测。恢复器只输出一份 IQ，FFT/PA/统计等分支全部由原 F 推导。测试使用 R05 的原始 168000 个物理样本与原始信道 bytes，保留 clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。R09 同 F50 原始输入结果是配对基线。旧 AsF/cleanF 结果不充当同 F50 正式结果。

R01/02/03 的扩散与流复用原始按 opaque physical ID 固定的噪声与完整 16 步采样；R20/R21 复用各自原 primary seed、128 批大小、七视图顺序和原步数，不挑采样 seed。所有模型冻结，0 优化、0 分类头拟合、0 归一化状态更新。每行全部七视图有限预测及 ID 完整后，由独立 CUDA 关闭的 CPU scorer 首次连接 truth；逐行评分，不等待整批完成。报告准确率、Macro-F1、修复/损伤/净修复、TX/RX/day、IQ/FFT 误差、同样本 clean 特征距离及与其他 TX 的比较。识别提升支持统计层面的输入身份保留，不构成每一个生成细节的唯一归属证明。

验收保持：low urban ≥70%；另外五种 impaired 场景每个 ≥原始 clean84.67619047619047%−5 个百分点，即79.67619047619047%。另报恢复 clean 的损伤。已暴露测试基准，不宣称新的盲测；所有负结果保留，分数不回流训练、结构、参数、采样 seed 或选择性重跑。

资源：N607 普通账户，GPU pool0/1/5/7，最多2个本任务 worker、每卡1个、所有 owner 上限2个且空闲≥12GB；共享 capacity4 lock 持有至 PID 出现在 CUDA。新不可覆盖 root `20261020-phase1-restoration-formal-backfill-F50-s2026092701-m42-r26`。真实 checkpoint＋2个 L_s 的无 query smoke 是启动首步，通过立即进入固定补测。实际代码以 Git commit 固定，一次独立 P0/P1 审查、一次发布归档 SHA 对照和远端 compile。命令、环境/CWD、各路径和期望产物见登记。技术失败保留健康 worker 和输出、停止派发 pending，不自动重试；不停止其他 owner。


精确矩阵、命令、环境/CWD、路径、GPU、停止规则和预期产物见[登记](experiment.json)。
