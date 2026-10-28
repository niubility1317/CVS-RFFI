# PAMF第三阶段：源代理无标签域校正与匿名注册

- run_id：`20261029-phase3-source-joint-s2026092701-r01`
- group_id：`pamf-phase3-source-joint`；类别：`cvs`；阶段：`Phase3-source-joint-unlabeled-adaptation-discovery-registration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结四类PAMF，source L隐藏标签代理U1050，稳健公共平移与anchored DPmeans形成匿名注册；五状态在未来RX8V5400独立truth-last评分。

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

## 真实发布前版本与验证

实际code/config commit：`6570e891982d72bc317c9219ead3e97221d93951`。初始68项整体测试通过；P1修复后相关48项及审查者4项针对复核通过，其余23项未变。唯一独立P0/P1审查PASS；已修复“仅控制组通过也可联合推进”，仅D1R1同时满足全部任务可晋级。源校准、双U状态冻结、完整head输入校正、未来逐样本预测和U-only匿名匹配的合成全链通过，不当作真实性能。N607单次preflight通过，GPU2当时空闲；实际GPU由启动时容量选择，至多新增2/卡且不停止其他任务。真实own四类checkpoint两条合法Lsmoke为第一步，PASS立即继续source-only矩阵。

## 真实完成结果

VERIFIED/ANALYZED_SOURCE_ONLY，四stage完成且进程终止，全部5状态预测后独立评分，NO_PROMOTION。D0/D1第一外轮达到32组件上限，合格0；D1旧类开放准确率63.9167%，较D0少1.2222pp，后续未知拒识19/900=2.1111%，R1新类识别0/900。不得把空注册表的旧类零损伤当安全注册证据。完整19行日志无异常，真实2Lsmoke PASS，U1050/未来5400、原U_s与正式target0、零新神经训练。结果与组件/平移诊断见docs/PAMF_PHASE3_SOURCE_JOINT_RESULTS_20261029.md及本地analysis_summary.json，不扩target或seed，不增cap挽救旧run。
