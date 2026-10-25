# 保留普通PL主项的恢复域软监督九学生匹配实验

- run_id：`20261025-phase1-PAMF-RestoredPL-m9-r01`
- group_id：`phase1-pamf-restored-IQ-ssl-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

PLControl/RawSoft/DURASoft各3seed，原ordinaryPL E30+20到E50；源规则定剂量0.12618073797689983，40dB后噪声按预定源负结果排除。

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

具体固定矩阵、源剂量、损失归一、数据/来源、成本、资源和自动七景测试验收见 [预登记](preregistration.md)。新的恢复域训练候选独立于默认模型；目标尚未达到。

## 实际启动核实

RUNNING，外部执行VERIFIED。科学release commit为c8a6efacfde29f8ced1c3bf684ea58d1aadffc28，一次archive传输比较/远端compile通过，独立PID/CWD/argv/GPU/loggrowth核实。共享源U56700缓存完成，52946条被三E30 raw教师接受（93.3792%，非U真值精度）；170100次F教师和170100次G前向，无V/U真值/target读取，全部注册buffers和参数不变。

真实ownE30+缓存的L768/U256/恢复学生512前向backward smoke通过，原6类默认头梯度有限，无optimizer/query。随后8个学生已在GPU0/1/2/3/4实际训练至E32/7104，第9条DURASoft seed03按全卡2worker上限等待空位。未停止其他owner。详见 [启动与来源证据](launch_evidence.json)、[实际源训练进度](launch_source_progress.csv)。

尚未产生本run目标测试结果，不能据缓存/源训练或smoke宣布目标达成。所有9条按固定配置到E50，再逐行raw/同G-LU全部视图预测与独立评分；原两对照和三项联合目标保持。

## 九学生实际源执行核对

9个学生均已真实启动，独立PID/CWD/argv和GPU映射核实；本次进度快照PLControl E43、RawSoft E42、DURASoft E42/E41/E33。

读取了当时全部可用完整源记录 **18995** 条（不是抽样；live gzip末尾可继续缓冲，尚非E50全日志）。逐条核对原L768/As KL0/跨RX权重、普通raw主项计数、新增两次前向平均与完整U batch归一、444步升温、实际own E30配置及6×160默认头。无非有限跳步、U真值或target训练。

三个seed的RawSoft/DURASoft在共同可用步数上，新增伪标签累计接受计数逐步完全一致；每完整U epoch均接受固定52946条，说明物理索引缓存与配对控制按预定执行。追加分支完整权重0.12618073797689983；seed03早期快照尚在升温也符合公式。

DURASoft seed01/02的额外源软CE分别从首epoch约0.2616/0.2577降至最近可用222步约0.2114/0.2134；这只说明恢复分支在学习，不能替代目标性能。全部已声明目标仍需E50后固定预测与独立truth-last。

[全部可用源记录汇总](available_source_execution.csv)、[配对及执行证据](available_source_evidence.json)、[九学生进度](current_source_progress.csv)。原目标criteria文件的“仅分析、未启动”状态已按实证更新为正在训练，三项数字门槛和数据权限保持。

## 八行评分完成的阶段结果

PARTIAL：8/9学生完成固定E50及独立raw/G-LU×三人群×七景truth-last，DURASoft尚缺seed03，不计算其三seed均值。以下为原目标RX全部日期168000记录/seed的完整3seed组均值±样本SD。

| 组别 | raw clean | raw六景 | G-LU clean | G-LU六景 |
| --- | ---: | ---: | ---: | ---: |
| PLControl | 84.967063±0.224833% | 70.334854±0.956618% | 85.022222±0.223596% | 78.061640±0.146305% |
| RawSoft | 84.872817±0.254680% | 70.262731±0.832628% | 84.926984±0.257182% | 78.011839±0.173580% |

PLControl相对原普通PL的raw clean/raw六景/G六景差分别−0.014683/−0.024504/−0.012996pp，支持基线复现接近。RawSoft相对本次匹配PLControl的对应差为−0.094246/−0.072123/−0.049802pp；相对原普通PL也未改善。因此额外raw固定软监督尚未达标，不能把源q精度高或源loss下降说成目标增益。恢复域组仍等待原seed03按冻结配置完成，不改参数/换seed/选择性重跑。

完整读取8个已完成学生的35520条新增E31–E50结构化记录和6392行stdout；actual final_only、自身E30父、原默认6×160头、native U/EMA关闭、所有新增mask/前向/权重均符合配置。B/C每行额外接受1058920次、每完整epoch52946条，零非有限跳步/U真值/target训练/G或教师更新。尚未完成的行不算入完整训练审计。

[主端点](partial8_primary_endpoints.csv)、[匹配控制差](partial8_matched_control.csv)、[完整已结束源训练](finished_source_execution.csv)、[阶段证据](partial8_result_evidence.json)。这些是阶段性已冻结测试结果，不改变剩余健康运行。
