# 固定 SIG_TIME：置信度门控特征一致性 02/03 最小预登记

状态 LOCAL_VERIFIED / NOT_LAUNCHED。实际代码/config fe2d8287ef1f9c167c5b68ee395e8bfe01b08360，原机制复用 325d113dff08206cb2f174e74d33e48f3e0b2ad5；唯一新增适配器独立 P0/P1 PASS。真实 checkpoint 无 query smoke 为发布首步，PASS 立即继续，无另设审批。

## 固定依据与机制

首种子 20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s01-m3-r01 按事前源规则 PASS，完整源证据及固定扩展决定已在 3b0b38c708029570cd1d6327cd4db96e2752a6ad commit、push 并独立核对远端 OID，早于该候选首个新目标数值读回。决定为固定 2026092702/03、各 SIG_TIME 与 CONF_FEATURE_MT 从零 50 轮，不修改超参数、种子、父权重、排名或停止条件。首种子目标差为负也执行原决定，不选择性取消或重跑；首种子原产物复用。

保持 SIG_TIME/f_theta195041/实际 joint160 与既有默认 cosine6x160 scale30 头，原流程联合训练。原 old batch_neighbor 门控、ownEMA.999、L CE smoothing.01、U CE.16/entropy.01、原增强/采样/优化器/cos 日程均不变。新 feature loss 只使用已计算的 teacherweak/studentstrong；detach FP32 maxsoftmax teacher>=student，差阈值0、ties保留；原 half squaredL2(unitjoint160, eps1e-8) 乘 mask 后 sum/fullbatchN，绝不除保留数。原系数 .1 五轮 ramp：第34轮.02、第38轮起.1。对照系数0，所有原 U 损失对象不改。无新增模块/参数/头/投影器/前向，不叠加 Fishr，不另训/重训默认头或 RX/身份诊断分类器。

## 数据与矩阵

run 20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s0203-m6-r01，owner codex/root/confidence-feature-seeds-20261015，release cvs_confidence_feature_seeds_20261015_r01。每 seed 三行 OLD_GATE_COS50（核实历史同契约默认头，仅自动 clean 重预测/评分）、SIG_TIME ZERO、CONF_FEATURE_MT。四新模型各 scratch50=33label+17pseudo、222step/轮、11100更新，每模型 U 17x56700=963900包。两 seed 新增200模型epoch/44400更新，另共享历史100/22200；六 row clean1008000决策/108条分组指标。合并保留的首种子为9模型、6新300epoch/66600更新+3历史150/33300，1512000决策/162条指标。每模型不超过50轮，训练只自己的 EMA，不继承首种子或目标权重。

数据角色不变：ManySig/sourceRX1/3/4/6/8、day1/2/3、split392005、L6300/U56700隐藏TX/V27000；原物理ID/TX、NoExplicitCFO/内部补偿与 VALIDATED_ONCE 数据保持。V无梯度/拟合/模型更新；没有新增信道增强。旧门控+EMA+cos 是正式基线，不以纯CE替代。首源固定资格为 finalLCE<=.5、精确预算/默认头特征导出独立、L特征变化>=1e-6、34/50真实t1/t2/SIG梯度、最后5轮保留feature均值比<=.95、全U比<=1、候选覆盖>=.5；完整首源数据见 first_source_qualification.json。扩展不新增源阈值或目标反馈。

逐row训练→固定final50→现有默认 ORIGINAL_HEAD clean168000逐包预测→该row全部prediction固定→独立truth-last评分→同row报告。只clean；不追加卫星/practical，不进入Phase2或新TX。结果报告每seed/完整3seed mean±sampleSD、macroF1、最差RX/day、配对差、修正/引入错误与缺项。84%是固定3seed均值目标，n=1或未完矩阵不冒充达标；新机制选择不使用目标数字。

## 命令、资源与产物

本地 verified native Python，经原 native runner 执行 experiments.cvs_confidence_feature_seeds.publish --output local_artifacts/confidence_feature_seeds_delivery_r01。远端普通 N607 既有 CVS-RFFI 环境/fullFP32，线程 OMP/MKL/OpenBLAS=2，CWD PROJECT/releases/cvs_confidence_feature_seeds_20261015_r01，入口 python -u -m experiments.cvs_confidence_feature_seeds.dispatch。不可变根 PROJECT/runs/20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s0203-m6-r01，log位于其中logs，实际完整路径/逐row命令/config见experiment.json。共享 capacity lock 分配 GPU，每 GPU 全owner最多3个实验进程、allocation free>=12000MiB；保留所有健康任务，不停止/重启/热修改/迁移。

仅一次普通账户preflight、单release archive本地/远端SHA比较、remotecompile和真实CP noquery smoke；发布后独立核实 PID/CWD/argv/GPU/loggrowth。源角色/原头/feature边界、精确预算、数值有限、输出冲突、不可执行或合法预测/评分失败等 run-owned 系统技术失败为停止规则：保留partial和failure，不自动重试、不触碰其他任务；科学负结果不停止其他row或改配方。

预期：preflight/submit/launch/queue、各row final_ssdg.pth/source completion/50轮曲线和11100步日志，feature/SSL实际接线、33/34/50真实梯度与门控/全U/保留均值记录，默认头clean predictions/scores/completion，source_features/source_qualification/analysis/completion/combined_three_seed_analysis。原IQ/CP/PT/原始日志/预测留不可变根，Git仅实现/config/报告及必要聚合JSON/图。

## 已完成验证

14个源资格负例、四个真实native上下文、native参数仅seed/run/candidate/output变化、同seed初始化相同、实际feature梯度通路、辅助W梯度缺失、原U损失对象保留均PASS；检查不读取query、不拟合头、不更新optimizer。入口依赖闭包/生成脚本compile PASS。一次独立新增适配器 P0/P1 PASS，未重复审查未改核心。见local_checks.json和independent_p0p1.json。预登记模板残留的旧运行completion/三种子数字已移除，当前没有本run运行结果。


## 发布后实际状态

RUNNING / VERIFIED：实际执行 55ff29d676eb5f02afa8b3869ed4818122f09d9a 已推送核实，普通账户preflight、单archive传输比较、remotecompile和真实CPnoquery smoke PASS。四个worker实际PID/CWD/argv/GPU及跨时点loggrowth核实：SIG_TIME-s2026092702 PID2627597/GPU1/epoch5；CONF_FEATURE_MT-s2026092702 PID2627675/GPU2/epoch6；SIG_TIME-s2026092703 PID2627834/GPU3/epoch6；CONF_FEATURE_MT-s2026092703 PID2627914/GPU4/epoch5。每GPU当前1个本run实验进程，全owner上限3保持；两历史baseline默认头clean链完成，failure0。阶段仍在原label训练段，feature机制按原计划第34轮启用。详见release_launch_evidence.json；尚无完整3seed结果，未宣称性能提升。
