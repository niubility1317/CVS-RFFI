# Phase3独立时频恢复器：实际F0与源G10零更新成本：单输出接口技术修复r02

- run_id：`20261101-diagnostic-P3Restore-TF-source-cost-s2026092701-r02`
- group_id：`phase3-independent-restoration-source-cost`；类别：`diagnostic`；阶段：`Phase3`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原L_s/U_s512条G10配对，独立TF和R69图的纯IQ/实际F0指导步骤交替计时，不更新任何权重。

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


## 固定技术恢复

r01独立poststate为FAILED：真实smoke误解包DURA生产接口返回值，0成本计时/0optimizer/0target，旧root保留。实际47c InferenceEngine内部验证valid后只返回网络输出，reader=None时(restored_IQ,)；本r02取[0]，与既有Phase3缓存_restore一致。新root/新执行commit，所有模型、512源ID/G10、归一化、损失、96步、阈值和权限保持。直接问题一次针对性P0P1 PASS；首步真实smoke后自动继续完整成本矩阵，无新性能承诺。
