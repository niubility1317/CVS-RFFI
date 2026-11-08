# 融合网络源域冻结识别器一致性

- run_id：`20261108-phase1-dura-direct-mix-source-proxy-s2026092701-r01`
- group_id：`phase1-dura-direct-mix-source-proxy`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

以固定源 clean 的冻结识别结果为伪类别，对照 DURA、direct、固定各半与两个融合门控；并非 TX 真实准确率。

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

## 后台诊断

两版融合器最终权重冻结后，使用源 pilot clean 预测作为冻结 Tri-SIG-TIME SSL 伪类别，与 DURA、direct 和固定各半对照。源 pilot 与 direct 及两门控拟合记录不重叠；原 DURA 是否见过 pilot 未核实。clean 伪类别可能有误，这不是 TX 真值准确率或目标测试。未使用目标、V 或 TX 标签，未更新模型。

代码 commit `6b13b88afd37f9277bc40f39dd5c4aac20998559`；七项针对性测试及独立 P0/P1 审查通过，发布包两端一致、远端编译通过。N607 CPU 两线程后台 PID 1097403，真实源前 8 条检查通过，预测日志已增长。

完整 672 条源 pilot 冻结识别器诊断已完成，分组及成对差异见 analysis.md；所有值为 clean 伪类别一致性，并非真实 TX 准确率。
