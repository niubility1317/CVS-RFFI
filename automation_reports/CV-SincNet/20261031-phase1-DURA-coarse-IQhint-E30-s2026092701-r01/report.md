# DURA 从零单阶段30轮与早期粗恢复IQ教师指导

- run_id：`20261031-phase1-DURA-coarse-IQhint-E30-s2026092701-r01`
- group_id：`phase1-dura-scratch-early-coarse-IQ-hint`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

266951参数原图，从零联合E30/7410；前10轮原R17coarseIQ提示1→.1、后20轮0；先缓存630k教师IQ，无F/TX/V/target优化，自动七景正式。

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


## 最小候选说明

以原R17源coarse路径为IQ教师，目标是让从零单阶段网络早期得到明确的粗恢复函数指导。固定30轮/7410，所有G模块从首步共同更新；前10轮指导权重1线性降至.1、后20轮0，基础noVar四损失和原63000源L/U/20缓存保持。先生成对应first10原received的630k教师IQ，不复制权重或重新训练教师；生成耗时、数值1.29024GB缓存、教师历史coarse20/完整R17包60组件轮次与新G30轮分开报告。优化不用F、直接TX/V/target；F仅source无query smoke和冻结后评测。源路径选择使用F作机制诊断，旧R17使用过源V选择，不能称整条研究未利用F信息。

模型图266951参数/2prior4CG27FFT和原F50/defaulthead保持，固定final30后自动全七景预测、独立truth-last及实测成本。E30physical是优化控制参考，原整个进程含F指导head及成本不隐去。source资格不保证正式收益，每版测试均执行。当前代码检查和唯一审查通过，尚未发布/训练；实际PID/GPU/日志和产物后态另记。参数、矩阵、精确命令、环境/CWD、路径、资源与停止规则见登记。
