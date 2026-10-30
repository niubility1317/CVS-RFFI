# DURA从零单阶段E20：移除方差辅助校准

- run_id：`20261030-phase1-DURA-direct-noVarAux-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Sourcegradientmatchedablation:onlyremove.001logvarianceauxloss;sameordinaryscratchinit/E20/4940/LR1e-3;nativevariancelearnsthroughIQ;all7formal

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

## 预登记匹配对照

仅移除0.001×logvariance辅助校准，保留最终IQ、内部prior、谱和source action四项。依据固定fastLR源checkpoint的32批源梯度；不因恒等版正式成绩切换初始化或选择父权重。实际完全从普通随机初始化训练，累计E20/4940，source数据/seed/探针/学习率/FP32均匹配fastLR。方差仍通过恢复路径学习，只声明正值一致性参数，不声明物理噪声校准。

一个独立P0/P1审查PASS，两个聚焦测试PASS；首步真实checkpoint无query smoke后立即训练。完成后自动source1800×7、target168000×7、同卡时延及独立truth-last，所有负结果保留。无性能早停或候选重排，默认DURA不变。具体命令/不可覆盖root/GPU容量与技术失败停止规则见experiment.json。

## 启动后实际状态

VERIFIED/RUNNING。执行commit cf40edea90667d9645a724b1e8f333027cdf7869，N607一次只读preflight通过，唯一release归档本地/远端SHA匹配、远端compile通过；CWD为`releases/20261030-DURA-direct-noVarAux-r01`。首步两Ls真实输入、实际冻结F50 checkpoint无query smoke PASS，随后直接继续。CPUcontroller PID3641689，GPU7 worker PID3642320；独立/proc核实CWD/完整命令/CUDA可见设备，并在nvidia-smi中匹配实际worker。训练日志从0增长到467bytes、至少完成E1。其他owner后续启动的健康进程保持。

当前尚无正式结果；E20固定冻结后自动source/target各七景、独立truth-last和成本报告，不因源成绩改变测试矩阵。目标继续ACTIVE。
