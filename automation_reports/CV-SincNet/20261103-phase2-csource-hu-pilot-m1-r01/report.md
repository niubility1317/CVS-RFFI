# Phase2 第二组：C 地面主干接 Hu 适应与 C 注册试运行

- run_id：`20261103-phase2-csource-hu-pilot-m1-r01`
- group_id：`phase2-group2-csource-hu`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Three fixed Hu/C transplant pairs, one exposed-development capsule and seed, K5/N20.

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

## 2026-11-03 启动读回

controller PID 2116882，3 种真实 C 源 checkpoint 与 Hu 源辅助权重的无 query smoke 3/3 PASS，GPU0/1/2 的 3 个 worker 实际 PID、CWD、命令已读回。预测和独立评分以实际后续产物为准；预测全部冻结后才进行独立评分。
