# 身份保护浅中层 MixStyle 首种子反证

固定t1/t2源L同TX/day跨RX统计混合，默认头正常联合训练；首种子2新模型+1历史对照，完整逐row clean/truth-last，不根据目标扩种子。

固定矩阵：OLD_GATE_COS50-s2026092701 只复用合法相同配方历史结果；MATCHED_CONTROL / IDENTITY_MIXSTYLE 两模型从零训练50/11100，33 label+17 pseudo。总新成本100 model-epochs/22200updates，历史50单列。三seed84%目标不由单seed完成。

候选只在主 L_s 监督前向 t1/t2 做 mean/std 混合，p=.5 alpha=.1 strength=1，无参数。真实物理eq/rx/day/sig/baseindex，同TX/day、不同RX和物理ID，缺配对保持identity；私有NumPy随机流。默认cosine6x160/scale30正常联合训练，不单训/替换/冻结头。不改原U排序/旧门控/EMA/NoExplicitCFO/输入策略，不叠加PAIRS/Sinc/TIME。

L6300、U56700真值隐藏、V27000只健康诊断，split392005原RX/day/TX及物理ID不变。候选来源是完整源t1/t2诊断 20261014-phase1-source-style-profile-manysig-s01-m1-r01，不是目标分组误差。已有暴露基准不是新独立盲测。

实际代码/config身份是Git release HEAD；没有额外设计/配置hash。普通N607既有CVS-RFFI环境，fullFP32、2threads。CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_identity_mixstyle50_20261014_r01；命令 python -u -m experiments.cvs_identity_mixstyle50.dispatch；run/output /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s01-m3-r01；GPU按sharedcapacity自动分配，每GPU计其他owner≤3且空闲≥12000MiB。

local negatives/实际adapter梯度及数值更新/teacher U eval隔离PASS，唯一独立P0/P1审查后发布；release首步real checkpoint zeroIQ no-query smoke PASS后直接继续。一个archive本地远端SHA核对及remote compile，一次preflight，owned PID/CWD/argv/GPU/日志增长核对。

固定终轮，不按V/target选模、改配方或重跑。技术 stop：Owned technical role/provenance/nonfinite/update count/mainL hook/default prediction/scoring failure stops dispatch with exact failure preserved, no retry or healthy/other owner mutation. Fixed50 final checkpoint; source thresholds determine only later independently registered seed expansion, current rows still fully predicted/scored. No target-feedback decisions.

固定源科学条件：两新模型终轮LCE≤.5；t1/t2全主L调用实际修改比例各≥.25；actual paired sourceL720 joint160平均L2≥1e-6。只用于后续另登记扩seed，当前3row仍完成自动clean预测/独立truth-last；不靠本轮目标分数扩seed。

每行source冻结和source_feature_witness后自动clean168000逐包预测，pred固定后独立truth-last，保留各RX/day/六TX Macro-F1/准确率及最差组。输出 launch.json；queue_state.json；analysis.json；completion.json；source_features.json；pernewrow initial_smoke.pt/final_ssdg.pth/source_frozen.json/resolved_native_args.json/resolved_config.json/source_contract.json/50epoch/11100step；pernewrow source_style_epoch_diagnostic.json/source_U_epoch_diagnostic.json/source_feature_witness.json；perrow prediction/predictions.npz/complete.json；perrow scores/metrics.json。

发布代码、预登记、最终分析和必要聚合表Git上传，checkpoint/IQ/私密配置/大型预测/原日志保留远端，禁止上传。

唯一独立 P0/P1 审查 PASS，无直接缺陷；未增加旧框架重复审查。登记、本地路径和特征边界检查 PASS。

发布 VERIFIED：实际代码 b2a4146b7571d8713ddbd98cd096fb98b2b4c00c；remote compile、real-checkpoint no-query smoke PASS。2新模型 PID1981402/GPU1 与 PID1981480/GPU2，CWD/argv/commit 与固定配置匹配，日志6614→22188、6630→17020增长；首轮222个主L调用闭合。源首轮证据与单archive传输登记见 launch_evidence.json，尚无新候选target结果。

冻结后源机制补充：source_moments.py 对两新 final50 模型使用完整 L6300，对照实际 t1/t2 mean/std 条件变化与固定 eval 身份保护干预进入 joint160 的效应。历史旧基线6300/12600包次直接复用已完成同checkpoint/同shuffle的源诊断，不重复前向。新成本25200源包次，Vcache materialize后丢弃，不前向/求导；无U/target/头拟合/参数更新。此分析不改变已冻结候选、任何源阈值或运行配方，不是新的训练结果或留RX泛化证明。

完整首seed源条件PASS：final LCE0.287843/0.289247；t1/t2实际L包调用修改48.6239%/48.9502%，各11100调用/1420800包次；固定L720 paired joint160 L2=6.010110、cos=0.883807。依据原阈值固定继续另外两个预定义modelseed的相同机制和scratch对照，决定时未读取target指标；见source_qualification.json和seed_expansion_decision.json。此PASS是源反证条件，不是识别提升/84%或新TX证明。当前3row已预测评分闭合，完整全量重算进行中。


**更正（2026-10-14）：** 此前“零拟合”的表述只适用于默认 TX 身份头和只读观察器。原 native 训练收尾为六个新模型各拟合 RX/day ridge 探针，共 12 次；channel 探针尝试因 invalid_input 失败。默认身份头未单独训练或替换，目标预测未使用探针，但这些源诊断仍违反用户当前禁止另训诊断分类头的要求。历史产物保留，不能标为完整满足当前零探针约束。新 SIG_TIME 组合入口已关闭这些拟合。详见剩余两种子报告中的 native_nuisance_probe_evidence.json。
