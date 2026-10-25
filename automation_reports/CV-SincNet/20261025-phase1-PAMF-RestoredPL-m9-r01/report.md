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
