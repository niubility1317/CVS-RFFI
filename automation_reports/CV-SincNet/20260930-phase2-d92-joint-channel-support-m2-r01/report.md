# LocalRidge与监督通道adapter联合适应及新类注册

- run_id：`20260930-phase2-d92-joint-channel-support-m2-r01`
- group_id：`d92-joint-channel-support`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

LocalRidge为最终分类器；支持集内折监督训练736个有界分块保范通道参数，B后C顺序继承，与原方法和C重置路径完整配对。

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

VERIFIED：joint-channel核心/入口/summary/编排76项不同合成检查通过；新结构独立P0/P1闭合；已生成并登记四row/160支持任务，preflight独立核实四缓存和practical residual绑定未改变。当前PLANNED，尚未启动，无真实性能结果，root唯一launch owner。

VERIFIED：joint-channel run唯一发布，runtime commit f18198054cd4ad66375ede114e4984349f3b31b4；独立读回核实supervisor37141及CPU worker37153/37154存活且argv/CWD匹配，rx3两row训练中、rx1两rowPENDING。已有完整实际训练STEP/FIT日志，尚无四row完整结果；不以partial评价/调参，无query/source样本读取。

全日志collector8项检查通过，方法与运行配置未改变，完整scan待四row终止后执行。

VERIFIED阶段进展：两个rx3 row各40任务完整完成，分别1872更新/7344头拟合/11232反向三角求解；rx1两row仍训练，当前累计113/160 parent。supervisor37141及当前CPU child46059/46486存活且argv/CWD匹配。没有完整四row结果，不读取或发布partial性能；运行代码/config不变。

VERIFIED行级状态：rx1两row独立核实仍TRAINING_ON_SUPPORT，CPU child46059/46486存活且argv/CWD一致，分别已完成19/40 parent；rx3两row完整40/40，当前118/160。未完成四row，不执行分析或报告partial性能。

VERIFIED：四row/160parent全部JOINT_CHANNEL_PROBE_COMPLETE，supervisor37141与所有worker均终止。实际936训练阶段、7488更新、8424内层目标、29376头拟合/分解、44928反向三角求解；无query/source样本/checkpoint加载/GPU。独立汇总与全日志扫描已各唯一启动，性能结论尚未完成，不将训练完成当改进成功。

VERIFIED：四row/160parent分析完成，独立support summary及完整日志scanner均下载核实。R_channel_seq相对R0的OOF/new-present：C旧+0.555556pp、新+0.062500pp、H+0.199110pp；B−B0+0.243056pp、注册旧类下降6.961806pp、逐任务绝对差11.334201pp。K5 B退化、新增2类部分新类退化，改善很小且混合，不晋级/不启动query。936阶段/7488非零更新，42.595679分钟CPU墙钟、峰值RSS536.472656MiB。A与B−A N/A；固定Phase1/source-free/practical residual。保留原BranchLocalRidge主线，目标ACTIVE。
