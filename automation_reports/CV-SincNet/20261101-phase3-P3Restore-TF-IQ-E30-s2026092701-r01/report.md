# Phase3独立时频恢复器：纯IQ从零单阶段30轮

- run_id：`20261101-phase3-P3Restore-TF-IQ-E30-s2026092701-r01`
- group_id：`phase3-independent-restoration-TF-E30-pair`；类别：`cvs`；阶段：`Phase3`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同196898参数TF、初值、源顺序和15%clean保持；不读取指导缓存进行优化。

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

## 固定匹配矩阵

Fixedfinal30/7410/noearlystop/noVselection. SameTFinit/dataorder/masks/B256 botharms;AdamW.001 warm247 cosine2e-5/WD1e-4/clip1;IQNMSE+.05mean3STFT;no featureobjective.

NewGPhase3actualF0/defaulthead. Fixedsource-r03 E0/finalPCD/density/tau retained as input-shift diagnostic;PCDtargetUancestry disclosed,not newGteacher/init ormatchedcalibration. HistoricalF50seven separate/noPhase3claim.

全部科学差异、测试路径和资源规则已固定，训练结果不改变另一臂，不热改或停止其他owner。首个真实源smoke后立即继续；两类全部预测固定后独立truth-last。21项聚焦检查、唯一P0/P1两臂PASS及单一资源修复定点PASS。默认DURA不变。

## 已核实启动

{"status": "VERIFIED_RUNNING", "code_commit": "b33839dd1010dea36b68526043bbc4abe5eb1095", "controller_PID": 2167531, "row_PID": 2167548, "training_PID": 2167558, "GPU": 7, "admission": {"GPU": 7, "free_memory_MB": 22883, "existing_compute_PIDs": 1}, "independent_PID_CWD_cmdline_CUDA_NVML_match": true, "real_source_smoke_PASS": true, "log_bytes": 847, "epochs_seen": 2, "new_target_reads": 0}

真实源smoke通过后已进入原定训练，尚无冻结测试成绩。只有优化期纯IQ不读取F0目标/参数；smoke与冻结评测的F访问另计。阶段间重复核实原GPU资源，其他owner保持。

## 登记字段更正

目标代码 receiver=34（RX3-2），日期2021_03_15、2021_03_23；初版登记误写为[3,4]和2019日期，现按原冻结fresh_world_score及runtime更正。实际程序始终绑定同一个reference_runtime/Q_conditions，没有改变任何训练、测试输入、方法、预算或结果权限。
