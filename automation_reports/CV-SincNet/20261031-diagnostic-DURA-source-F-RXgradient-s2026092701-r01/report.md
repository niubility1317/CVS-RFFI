# 原头保护 RX 特征参考的源输出梯度证伪

- run_id：`20261031-diagnostic-DURA-source-F-RXgradient-s2026092701-r01`
- group_id：`phase1-dura-source-RX-canonical-feature-gradient`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定noVar source512×缓存01/20，比较IQ/旧Fcos/保护RX特征目标梯度；唯一权重及实质差异门槛，CPU2/无训练/TX/V/target。

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


## 最小研究说明

固定已通过的源RX几何（aee80ff、fit62488不含512 probe），.25特征参考收缩不变，不是波形相位规范化或部署校正。原noVar cf40源final20仅作2Ls G/F真实smoke；主计算复用其512条source缓存01/20已保存IQ。训练好F冻结，计算输入IQ对IQNMSE、旧cleanF参考cos损失和新RX参考cos损失的三条梯度，参考全部detach、无F/G更新，不读TX/V/target。

唯一lossweight=.25/全32批norm(gCanonical)/norm(gIQ)中位数。两个view各需中位校准压力[.125,.5]、与IQ平均cos>=0、负cos最多8/16；同时新旧Fcos梯度差范数相对旧梯度范数的中位数各至少10%。这项预定实质差异条件用于避免重跑近似旧失败项；任何失败不加大.25特征收缩、不放宽门槛或重选源父模型。通过仅允许后续独立登记单阶段scratchE20训练，不代表参数梯度方向、身份或目标收益。

记录完整1572864个三类梯度值供独立32批重算，并核对IQ闭式梯度。原F50知识和成本如实继承，非F-free，不产生硬伪TX标签或另训头。CPU2/0GPU，F1024主分析输入、G/F各2Ls smoke。默认DURA和逐景3pp/clean/效率目标未完成。
