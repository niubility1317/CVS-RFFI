# DURA单阶段源原头可见特征指导：源梯度标定

- run_id：`20261030-phase1-DURA-source-headFeature-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：QUEUED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

SourcefrozenFheadfeatureprojection rank5;alpha sourcecalibrated;GscratchE20/4940/noVarbase;noGancestors/directTX/V/target;automatic7formal

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

## 发布读回与实际资源等待

VERIFIED/QUEUED。执行commit d79a280d843696d680c82bf63b0138905cf1cb5d已提交/推送并比较远端OID一致，12项聚焦检查及该训练候选唯一独立P0/P1审查PASS，一次N607 preflight、单归档本地/远端SHA比较和远端compile通过。真实source2Ls原F50/G smoke通过：新投影特征到IQ梯度有限非零、F/原头参数状态不变、G严格保存重载、源物理角色匹配，0直接TX/target读取。

CPU controller PID4025316实际存活，独立核实CWD=`releases/20261030-DURA-source-headFeature-r01`、完整命令、空CUDA绑定。其日志从空增长至304bytes，包含smoke PASS及多条pending scratch_joint/active空/failed空的队列记录。各卡当前可见compute进程数均≥2，不满足预登记的启动时总可见PID<2条件；当前没有本run GPU worker、没有epoch完成或预测。控制器自然等资源，不干预其他owner，不把已发布或CPU存活称为训练已开始。

资源符合时原控制器自动启动一次G从零单阶段E20/4940，再串联全部source1800×7/target168000×7、同卡时延及独立truth-last。固定源系数/预算不热改，不重复提交；后续先读本root/PID/产物。正式性能未知，六星地3pp目标ACTIVE、默认DURA保持。
