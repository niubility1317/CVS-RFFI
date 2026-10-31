# Phase3恢复器原L_s/U_s全量G10-clean配对缓存

- run_id：`20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01`
- group_id：`phase3-independent-restoration-source-pairs`；类别：`diagnostic`；阶段：`Phase3`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅制备63000条原源配对和可选实际F0 clean特征；0优化，不训练G，不打开OE/V/目标。

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

## 本次范围

独立全量源配对制备，0优化；OE只作将来冻结诊断。原6300Ls+56700Us，不读y/V/OE/目标。新架构先前实际F0成本已通过，本run不重跑成本、不训练新版G，因此无正式识别预测。完成后独立核对63000ID/角色、一次RMS和与既有512源配对一致性。

24项聚焦检查PASS；唯一独立P0/P1审查PASS。实际代码 53410ce889adc94c53c1783741f970fbdaf51453。原始IQ/特征/私有配置不提交Git。正式新G训练须接好自动冻结测试后另登记30轮匹配矩阵。
