# Phase2 第二组：C 地面主干接 POSTER 适应与 C 注册试运行

- run_id：`20261103-phase2-csource-poster-pilot-m1-r01`
- group_id：`phase2-group2-csource-poster`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Three fixed POSTER/C transplant pairs, one exposed-development capsule and seed, K5/N20.

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

真实 controller PID 2107069；CWD、命令和 runtime 见 experiment.json。三种 C 地面权重的 POSTER source-only/no-query smoke 3/3 通过；实际 worker PID、GPU、CWD/argv 和至少 2/3 个闭合预测已读回。全部预测固定后自动独立评分，当前未宣称评分完成。

POSTER 适配保留冻结的 C 源编码器，重置并训练两层稠密投影与六类头 10 个 epoch；投影维度按 C 编码器匹配，投影进入 C 注册。该项目移植与作者原 POSTER 架构不同。Hu 跨主干绑定仍待实现；ISSL 保持停止、CSIL 作者绑定缺失。
