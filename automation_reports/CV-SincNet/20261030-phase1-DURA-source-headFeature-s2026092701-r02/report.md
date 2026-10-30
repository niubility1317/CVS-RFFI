# DURA单阶段源原头可见特征：用户授权三进程卡资源切换

- run_id：`20261030-phase1-DURA-source-headFeature-s2026092701-r02`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Same source headFeature alpha/E20/scratch/data/seeds/7formal;authorizedresourceonly replacement ofoldunstartedCPUqueue;GPU4/6/7 total<=3/own1/free12GB;ownedFP32hostcopy

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

## 用户授权资源切换与实际训练读回

VERIFIED/RUNNING。用户明确允许单卡新增1本任务进程、总数最多3，只限旧队列仍未GPU启动时切换。执行commit 81dfc80a2d4687f9b9ab7bc9c28ede6b092f7c4a已提交/推送并比较远端OID一致；4项聚焦资源/停止条件/子进程路由及原科学配置检查PASS，该资源候选唯一独立P0/P1审查PASS；一次N607 preflight、单归档两端SHA和compile通过。科学配方/原F50与数据/E20预算保持，新增归档如实使用已验证的ownedFP32主机输出转换。

旧r01 CPUcontroller PID4025316在既有capacity锁内核对原d79 commit、精确CWD/命令、无子进程/row/submitted/complete后仅对该PID发送SIGTERM；随后独立/proc读回确认不再运行，旧root无GPU row，全部smoke/日志保留，resource_replaced.json与新r02绑定。没有停止、修改或重启其他owner进程。旧run登记为REPLACED，未冒充训练失败或已完成模型。

新CPUcontroller PID4059979、GPU4 worker PID4060125均实际存活；独立核实CWD=`releases/20261030-DURA-source-headFeature-capacity3-r02`、完整argv、worker CUDA_VISIBLE_DEVICES=4及nvidia-smi显存。读回时GPU4总可见计算进程3、本worker驻留2020MiB、剩余20965MiB。真实source2Ls G/F梯度非零/F冻结/G严格重载/source物理角色匹配smoke PASS。源clean特征63000×160/40.32MB缓存一次，3.450496秒完成；新增restored F前反向继续实际执行。

训练日志由空增长到543bytes，至少完成E1/247更新：L6300/U56700、F_inputs63000、F_updates0、直接TX0、G初始化checkpoint0。首轮wall82.045秒，IQNMSE0.46866796、head投影误差0.85258854、clip fraction0.465587；只是早期源收敛，不能推断正式性能或跨负载训练加速。F完整state不变将在训练结束再次核实。

固定最后E20/4940后自动同卡时延、source1800×7和target168000×7，各row全部预测固定后独立truth-last，不追加审批或依据target选择epoch/配方。六星地各距rawclean≤3pp、clean保持仍待完整结果，默认DURA不变。
