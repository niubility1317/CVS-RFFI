# 独立恢复器与 DURA 的源域波形互补

- run_id：`20261108-phase1-direct-dura-mix-source-s2026092701-r01`
- group_id：`phase1-direct-dura-source-complement`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结宽范围物理损失独立恢复器和 DURA，比较源域固定混合、物理步长与不可部署 clean oracle。

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

## 方案与运行

固定 DURA v1 和源训练 final20 的宽范围物理损失独立模型，使用相同源 received IQ；比较 DURA、独立模型、固定 0.5 混合、估计信道物理步长与需要 source clean 的不可部署 oracle。672 条已探索源 pilot 与独立模型拟合 ID 不重叠，DURA 历史重叠未知；不读取 F、TX 标签、U_s 真值、V 或目标，也不更新模型。

代码 `2a1ae2b54357e8acce4ee58daf92ff7b8cb909ec`；3 项聚焦测试、独立 P0/P1 审查、远端发布归档校验和编译通过。N607 CPU 两线程、PID 1029526，首批 8 条真实源记录 smoke 通过。波形诊断不是身份准确率；正式七景预测继续。

- 672 条源波形诊断完成；固定 0.5 混合在 RX8 六个受损组均低于 DURA，而 RX4/6 五组高于 DURA；clean 理想步长不可部署。详见 [analysis.md](analysis.md)。
