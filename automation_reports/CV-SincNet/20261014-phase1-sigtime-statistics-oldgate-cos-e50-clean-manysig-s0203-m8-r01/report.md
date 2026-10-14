# SIG_TIME 统计组合：固定剩余两种子

本登记是首种子固定源条件 PASS 后的原样扩种子准备，首 seed 固定源资格及四行闭环已核实，现准备发布，仍未启动。用户优先要求完成 SIG_TIME + MEAN_VAR + COV 与单 SIG_TIME、同参数三路 SIG_TIME 控制的组合验证。原首 seed 的不可变运行保持。

矩阵：2026092702/03 × OLD_GATE_COS50、SIG_TIME、SIG_TIME_X3、SIG_TIME_MV_COV。两个历史 baseline 只读复用，六个新模型独立 scratch50/11100；不加载或续训首 seed 及历史最高 SIG_TIME checkpoint。新增 300 model-epochs / 66600 更新；完整三 seed 共九个新模型 450 / 99900，历史三个 baseline 150 单列复用。首 seed 不重训、替换或筛除。

机制与首 seed 完全一致：实际 t2 四窗、独立 6D 局部投影、每视图 28D 描述、统一投影 joint160、初始增益 0.05/上限 0.2。单分支新增 18976 参数；三路 SIG_TIME 与组合各新增 55968，初始 native 骨干/default W 一致，容量控制与组合所有初始可训练参数相同。MEAN_VAR 在同路径可由均值/协方差计算，不宣称原始信息独立；独立投影和描述方式是候选差异，不声称描述内在秩相同。

分类头保持既有 cosine6×160 scale30，随骨干按原流程联合训练，不冻结/另训/替换头。source 使用已审查的默认头-only native scope，RX/day/channel 拟合探针全部关闭；U 和自身 EMA teacher 使用相同特征机制。Old batch_neighbor gate + EMA0.999 + cos2e−4→1e−6、AdamW wd1e−4、33 label + 17 pseudo、lambda_U0.16/lambda_ent0.01/smoothing0.01 保持，无新星地训练增强。NoExplicitCFO 坐标屏蔽及内部逐包补偿保持；L6300/U56700/V27000、split392005、RX/day/TX/物理 ID 不变。U 真值隐藏，V 仅 eval 健康诊断。

发布条件仅为原首 seed 的固定源资格：三个新模型终轮 LCE≤0.5、新模块数值更新、真实导出与默认 joint160 一致且 W 独立，固定 L720 上组合与 X3 的 L2≥1e−6。条件和当前实现均在首 seed 新目标分数读取前冻结。发布还等待首 seed 四行完成既定预测/评分闭环；这里只读 completion 元数据，不用目标分数决定扩种子。资格 FAIL 则保留原结果，不启动这六个模型。

代码提交 a722e2593947ca804e2534ac75e4c081029d022c，分支 codex/phase1-feature-challenge-20261013。入口 python -u -m experiments.cvs_signature_stats_seeds.dispatch。普通 N607 既有 CVS-RFFI fullFP32 / 2 threads；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_signature_stats_seeds_20261014_r01；不可变 run/output /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-sigtime-statistics-oldgate-cos-e50-clean-manysig-s0203-m8-r01。共享容量锁，每 GPU 总实验进程（包含其他 owner）≤3、空闲≥12000 MiB，按实时容量分配。不停止、重启或热改健康旧任务。

已完成一次新增执行差异 P0/P1 审查 PASS，以及六个真实模型 context/初态/foreign-config 拒绝、固定三 seed 汇总、源资格拒绝与不同 seed 特征分组检查 PASS。原特征数学和训练内核不重复审查。发布时单个归档传输后一次 local/remote SHA 比较、remote compile、首步预flight读取原源资格并对历史 seed02 真实 checkpoint 作 zeroIQ no-query smoke，PASS 后直接运行；不增加授权文件、签名或数据重验。

Stop：运行自身的角色/来源/非有限数/预算/默认头/导出/物理 ID/预测或独立 truth-last 技术失败停止新增派发，保留健康进程与产物，不自动重跑；不依目标早停、换参数、换父权重或换种子。各模型固定 final50。

每行冻结后自动以 checkpoint 默认头生成 clean168000 逐包预测；预测固定后独立 truth-last 评分。八行共 1344000 次决策/144 组 CM、accuracy、Macro-F1；与原首 seed 合并 12 行、2016000 次/216 组。记录三 seed 平均/样本 SD、各 seed 配对差、正差数、最差 RX/day 与错误修正/引入；仅完整候选三 seed clean 均值≥84% 才能满足主指标。

预期产物：release 中 preflight.json/source_expansion_evidence.json；run launch/queue、每新行 initial/final/source_frozen/resolved/actual_args/initialization/50epoch/11100step/source_feature_witness、每行 prediction/complete、scores/metrics、source_features、analysis、combined_three_seed_analysis、completion。原数据/权重/原始预测/日志留产物目录；实现、报告与必要聚合 Git 上传并核对远端 OID。这里只验证暴露 ManySig clean 的旧六 TX；没有卫星部署、Phase2 或新 TX 声称。


首 seed 源条件 PASS：LCE 0.248702/0.227253/0.233990，固定 L720 组合 vs X3 L2=2.240102；实际新增参数数值更新及 W 独立导出通过。首 seed 四行全部独立评分完毕；此扩种子决定未读取其新目标分数。详见 source_expansion_decision.json。
