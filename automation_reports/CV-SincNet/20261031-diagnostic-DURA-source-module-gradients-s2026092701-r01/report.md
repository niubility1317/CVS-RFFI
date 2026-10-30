# 单阶段恢复器源域模块分工梯度诊断

- run_id：`20261031-diagnostic-DURA-source-module-gradients-s2026092701-r01`
- group_id：`phase1-dura-source-module-gradient-role`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

旧noVar源E20 checkpoint，固定512源样本×缓存01/20/B32，final IQ与coarse clean/action/prior/STFT模块梯度几何；0新训练/F/TX/V/target。

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


## 最小源诊断说明

仅旧noVar cf40源final20/4940，512源训练池样本（128L384U）×缓存01/20，32批B32/1024G输入+2Ls smoke，CPU2/CUDA空，无optimizer/F/TX/V/target。复用单次G前向，分别计算final IQ/prior.1/STFT.05/action.05/coarse对clean IQ的唯一参数梯度，coarse/action/prior精确全分区。旧诊断未覆盖coarseclean方向，本次只补该具体机制问题；不是另训头、预测目标或新恢复器。当前nativeIQhint健康训练保持。

先固定源假设规则：coarseclean或actionprobe在对应模块相对finalIQ的32个cos都可定义且至少24个为负，才支持进一步研究该模块分工；不自动训练、失败不放宽、不称真实身份收益。原多阶段分工可能有用只是待证假设，局部梯度不能证明完整训练轨迹。4聚焦检查/唯一P0P1 PASS，首步真实2Ls源checkpoint noquery后继续；完整state相等/gradNone收尾。后续若产生新G仍须独立从零单阶段E30和全七景正式测试。当前待发布，默认与完整目标ACTIVE。


## 源诊断完整结果

VERIFIED / ANALYZED，仅SOURCE_ONLY。32批/384组几何独立复算一致；coarseclean18/32负cos、actionprobe2/32，均未达到固定24/32规则，不启动梯度分离训练、不放宽。CPU循环28.349202秒，1024G输入+2Ls smoke，0F/TX/V/target/优化更新、model不变。真实PID/CWD/argv/CUDA空/log及自然结束已核实。详情见[分析](analysis.md)与results.json；默认保持，目标ACTIVE。
