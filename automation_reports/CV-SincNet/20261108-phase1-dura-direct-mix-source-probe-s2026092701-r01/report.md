# DURA 与直接网络自适应融合源域诊断

- run_id：`20261108-phase1-dura-direct-mix-source-probe-s2026092701-r01`
- group_id：`phase1-dura-direct-mix-source-probe`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

诊断固定融合门控相对于 DURA 和大容量直接恢复器的源域波形误差与权重，不使用 TX、F、V 或目标。

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

## 诊断准备（2026-11-08）

源域配对波形诊断已登记：同一 672 条既有源 pilot 对比冻结 DURA、独立大容量直接恢复器和二者自适应融合。pilot 与直接模型和门控训练记录不重叠；原 DURA 是否见过这些记录未知。诊断不读取 TX 真值、分类器、验证集或目标集，不更新模型；只能说明波形误差与融合行为，不能代表识别准确率。

代码 commit `3bca790e085da53e931d8911e9ee58ab10ad2397`，四项针对性测试与独立 P0/P1 检查通过，发布包传输一致，远端编译通过。等待门控最终权重冻结后在 CPU 两线程后台运行，不影响正在进行的正式七场景预测。

后台源域诊断 PID 1052488 已启动，首批 8 条真实样本检查通过；完整结果待写入。

诊断已完成，672 条记录结果及分组分析见 analysis.md；源域波形误差不能代替正式识别准确率。
