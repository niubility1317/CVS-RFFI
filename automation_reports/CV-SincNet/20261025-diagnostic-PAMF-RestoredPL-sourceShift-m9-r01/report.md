# 恢复域软监督退步后的冻结源域表现与接收机几何诊断

- run_id：`20261025-diagnostic-PAMF-RestoredPL-sourceShift-m9-r01`
- group_id：`phase1-pamf-restored-IQ-source-shift-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（完整冻结源诊断已独立读回；不代表性能目标达成）

## 目的与对照

固定PLControl/RawSoft/DURASoft九份F50及同G-LU，完整sourceV比较raw/H/G分类、恢复误差、类内跨RX几何和原头方向变化；只诊断不拟合。

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

## 发布前范围

最小预登记见 [preregistration.md](preregistration.md)。仅冻结源域诊断；九份最终 F50 和同 G-LU，不训练、不拟合新头、不访问目标数据。两条聚焦测试和一次独立 P0/P1 审查通过；真实 checkpoint smoke 由启动器先执行。登记已完成，尚未发布。

## 启动核实（历史）

`VERIFIED_RUNNING`，实际科学代码 commit `b6260ca62520fa455fac36b3e7fa285389713ce2`。archive 传输与远端 compile 通过；独立进程/CWD/完整命令/GPU0/日志增长核对通过，真实 L2、九 F 和一 G 的 smoke 通过。完整 V 前向正在推进。参见 [启动证据](launch_evidence.json)。首次发布在本机连接配置路径解析处失败，尚未传输；核对远端 run/release 均不存在后复用原 archive 继续，没有重复计算或覆盖产物。

## 完整冻结源诊断完成

`ANALYZED`。九 F、八视图、27000条 source V 全部完成，参数/buffers不变，无新训练或目标访问。恢复域监督在源三个恢复场景提高0.316049/0.654321/1.150617 pp，但相同目标场景均退步；不能据此宣布方法有效。完整结论和限制见 [源分析](source_analysis.md) 与 [证据](source_result_evidence.json)。后续只固定可检验的新计划，未启动新的 F 训练。
