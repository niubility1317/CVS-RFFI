# DURA默认版源分阶段身份特征保持与物理求解定位

- run_id：`20261030-diagnostic-DURA-source-stage-fidelity-s2026092701-r01`
- group_id：`phase1-dura-source-stage-fidelity`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen defaultR69+originalF50;source512x2views;received/coarse/stage1/prior/native/CG4-8-16/sourceoracleA-noise;teacherfeature/choicefidelity;0updates/TX/V/target

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 完整源分阶段身份特征定位

VERIFIED/ANALYZED_SOURCE_ONLY。CPU2/0GPU共13.895秒，固定512源位置（L128/U384）×cached01/20，32个B32批次×10路径完整，G/F状态不变、p.gradNone、0优化器/直接TX/V/target。真实source2Ls默认R69/F50 smoke PASS，原生最后CG2逐批精确重构；1项聚焦晋级规则检查与唯一独立P0/P1审查PASS，单次N607 preflight/归档两端SHA/compile完成。执行commit 6404a163a120272bac64f3a276096d1cba159157。

初次紧接提交读回时日志尚空、launch.json尚未生成，后续读回PID4031580已自然结束；launch内CWD/argv、两源缓存完整32批/3230bytes日志和complete/results已独立回收，未声称直接观测到存活期间PID。源原始npz逐个回收，两个缓存视图同512唯一位置及128L384U角色吻合，60列/视图有限。全部role/scene/state/lock统计、teacher修复/损伤与资格判定独立重算，合计7560标量核对一致。原始源位置/数组不入Git。

|源路径|cached01 IQNMSE|cached01 F余弦误差|cached01 teacher一致率|cached20 IQNMSE|cached20 F余弦误差|cached20 teacher一致率|
|---|---:|---:|---:|---:|---:|---:|
|received|1.68760777|0.12116517|56.640625%|1.69934714|0.11972755|57.812500%|
|coarse|0.49326968|0.18686235|15.234375%|0.48739827|0.18672061|15.234375%|
|stage1|0.02830368|0.03992907|90.039062%|0.02629589|0.03502721|92.578125%|
|prior|0.17499176|0.08390491|63.085938%|0.17472556|0.08518673|64.453125%|
|native|0.02021283|0.01282546|94.921875%|0.01943971|0.01097861|97.265625%|
|CG4|0.02190033|0.01485262|94.921875%|0.02121694|0.01255594|97.265625%|
|CG8|0.02203875|0.01497333|94.726562%|0.02136470|0.01265052|97.265625%|
|CG16|0.02203970|0.01496673|94.726562%|0.02136555|0.01264588|97.265625%|
|oracle_action_CG16|0.02266582|0.01604076|93.750000%|0.02035539|0.01376345|97.265625%|
|oracle_both_CG16|0.02514124|0.01742866|93.164062%|0.02287646|0.01485490|95.507812%|

这些一致率是恢复后F选择与同一源clean F选择的相同率，不是真实TX准确率。默认stage2比stage1分别提高4.8828125/4.6875pp；最后prior独立输出只有63.085938%/64.453125%，而经过原生物理一致性后为94.921875%/97.265625%。这支持保留两个恢复步骤和原生求解，不能据此把prior当独立clean恢复器或删掉第二步。一次联合训练与两个推理步骤是不同预算，用户要求减少多阶段训练不等于去掉有效推理步骤。

增加最后求解至4/8/16步均未通过预定源条件；CG4整体teacher一致数仍486/498，与native相同，但各自有1条teacher修复和1条损伤，逐场景仍须分别核对；CG8/16 cached01还净少1条一致。F余弦误差CG4较native约增加15.81%/14.37%，不是目标特征改善。全部候选两视图资格False，不发布增加CG的运行版，也不放宽条件。source真实A/噪声替换亦未稳定提高teacher保持，不能将它们称目标性能上界或部署输入。

新证据补充早先仅IQ的源求解定位，未使用任何target或TX真值，不改变R69权重的历史F-free身份，也不追溯更改其正式成绩。当前单阶段headFeature候选仍按原固定E20/4940等待GPU并自动完成七景；本诊断不取消任何已发布版本的正式测试。性能/训练效率验收继续，3pp目标ACTIVE、默认保持。
