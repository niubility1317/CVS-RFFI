# 双门控共享 DURA 与直接恢复源域基准

- run_id：`20261108-phase1-direct-gates-shared-source-benchmark-s2026092701-r01`
- group_id：`phase1-direct-gates-shared-source-benchmark`；类别：`diagnostic`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定两版恢复门控，比较独立计算和共享 DURA/direct 的八类输出及相同 CPU 线程用时。

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

## 后台源域验证

冻结 DURA、独立直接恢复器、两版 final20 门控与 Tri-SIG-TIME SSL，在同一 672 条源 pilot 上比较独立与共享 DURA/direct 计算。每版八类中间输出和 logits 逐字节检查；CPU 两线程、两次交替顺序计时。该诊断不读取 TX/V/目标，不训练，不能替代七场景完整输出验证或并行吞吐实测。

代码 commit `35c9c7f68d090defeb26882b552a92d1a24e59b5`；七项针对性测试和独立 P0/P1 审查通过，发布包匹配、远端编译通过。后台 PID 1141775，真实源信号前 8 条逐字段检查通过。
