# 身份保护 MixStyle：固定剩余两种子验证

按首seed固定源条件PASS及先于target指标阅读的9e79dc413决策，剩余seed02/03四新scratch50、两历史基线；自动逐row clean/truthlast并与已完成firstseed合并三seed，不按目标改参数。

source资格：首seed run 20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s01-m3-r01 的固定条件PASS（LCE≤.5、t1/t2实际改包≥25%、真实source z差≥1e−6）。扩seed决定 9e79dc413967aee78fa87e5d0f6c7408c44e4efc 已在读取target指标前上传。所有参数/预算/层位不变，旧firstseed结果如实保留。

矩阵：seed2026092702/03 × OLD_GATE_COS50、MATCHED_CONTROL、IDENTITY_MIXSTYLE。4新模型从零训练各50/11100，33 label+17 pseudo；2历史对照只复用同seed固定final50、相同原物理数据角色和配方。新成本200model-epochs/44400updates，历史100单列复用，firstseed新训练0。无父权重/teacher checkpoint继承。

仅主L_s的t1/t2，p=.5 alpha=.1 strength=1，同TX/day、不同RX和真实物理ID，私有NumPy，缺donor保持identity。U/teacher/eval不干预，无新参数/额外训练forward/单独headfit；默认cosine6×160 scale30正常联合训练。旧门控/EMA.999/cos/NoExplicitCFO/内部补偿及SOURCE6300/56700/27000、split392005和允许RX/day/TX保持。V只eval健康诊断，不梯度/选模，U真值隐藏，不新增星地训练。

代码/config身份：一个Git release HEAD。入口 python -u -m experiments.cvs_identity_mixstyle_seeds.dispatch；普通N607既有CVS-RFFI fullFP32/2threads；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_identity_mixstyle_seeds_20261014_r01；不可变run/output /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s0203-m6-r01。共享capacity锁，计其他owner每GPU≤3，空闲≥12000MiB，按实时容量分配GPU。

新context本地真实native forward、两seed各paired初态191tensor一致、拒绝first/foreign seed、三seedmean/SD配对计算PASS；不重复原干预core审查。一个直接P0/P1新增执行差异审查后Git上传，单archive一次local/remoteSHA比、remotecompile、一次N607预flight：旧seed02真实CP zeroIQ no-query smoke PASS后直接运行。不得创建额外签名/数据重验/报告hash。

Stop：Owned role/provenance/nonfinite/update budget/mainL hook/prediction/scoring technical failure stops new dispatch, preserves other healthy processes and artifacts; no automatic retry. All rows fixed50/11100; no target ranking, selection, early stop, seed replacement, parent selection or hyperparameter change.

每行固定final50/11100并生成真实sourceL720 fθ与W独立证据后自动clean168000逐包prediction；prediction固定后独立truthlast评分。每行18组CM/F1/accuracy，各RX/day/旧6TX保留。所有6行完成后与旧firstseed3行按对应模型seed合并，最终9row/1512000决策/162metric，3seed平均/样本SD/配对差/最差RX/day。只有完整合并均值≥84%才设置goal_84，不使用单seed或两seed平均替代。

输出 launch.json；queue_state.json；analysis.json；completion.json；source_features.json；pernewrow initial_smoke.pt/final_ssdg.pth/source_frozen.json/resolved_native_args.json/resolved_config.json/source_contract.json/50epoch/11100step；pernewrow source_style_epoch_diagnostic.json/source_U_epoch_diagnostic.json/source_feature_witness.json；perrow prediction/predictions.npz/complete.json；perrow scores/metrics.json；combined_three_seed_analysis.json。原IQ、checkpoint、日志和大型prediction留既定目录；代码、预登记、实际结果/聚合报告Git上传并独立核对remoteOID。预登记时发布四个新模型；当前全部完成，完整结果见 final_analysis.md。

新增执行差异唯一直接P0/P1审查PASS，原干预core不重审。本地新context/三seed统计/登记/依赖与scriptcompile均PASS。

实际发布代码 2b8371027c11cab4b80c17a8c2d0797fac00524a，dispatcher PID2013525；独立进程/CWD/命令/GPU/配置/日志增长 VERIFIED。GPU1/2/3/4 的 MATCHED_CONTROL02/MIXSTYLE02/MATCHED_CONTROL03/MIXSTYLE03 已完成12/9/11/11轮，无失败；单归档传输比对、远端compile及真实旧seed02 checkpoint无查询smoke PASS。全部保持scratch50固定预算，两个历史baseline已只读复用；新目标结果尚未完成，三seed结果未产生。详见[发布证据](launch_evidence.json)。


## 完整结果回收

全部六行已训练或复用并完成默认头 clean 预测、独立 truth-last 评分；与首 seed 合并为 9 行、3 个固定种子。结果为 ANALYZED，详见 [完整结果](final_analysis.md)。

**更正（2026-10-14）：** 此前“零拟合”的表述只适用于默认 TX 身份头和只读观察器。原 native 训练收尾为六个新模型各拟合 RX/day ridge 探针，共 12 次；channel 探针尝试因 invalid_input 失败。默认身份头未单独训练或替换，目标预测未使用探针，但这些源诊断仍违反用户当前禁止另训诊断分类头的要求。历史产物保留，不能标为完整满足当前零探针约束。新 SIG_TIME 组合入口已关闭这些拟合。详见剩余两种子报告中的 native_nuisance_probe_evidence.json。
