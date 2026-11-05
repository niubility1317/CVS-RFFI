# Phase2 DADDA→MoPC 支持集数值诊断（CPU）

- run_id：`20261106-phase2-dadda-mopc-support-cpu-s2026092703-r01`
- group_id：`phase2-group2-dadda-mopc-nonfinite`；类别：`diagnostic`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

使用 B r06 任务 5944 的原始源权重和 K1/N20 支持集，按原 B/C 配方单次记录首个非有限边界。

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

## 数值诊断

固定 B r06 任务 5944 的 K1/N20 和原始源权重。检查 B 终态、MoPC 当前支持集与原型分支 logits、损失、梯度、SGD 后参数与动量，以及阶段之间的原型校正。单个 CPU 进程执行一次。CPU 结果仅用于定位，不能判定 GPU2 故障是否已解决。预测文件、查询样本和真值均不读取，正式队列与旧产物由原运行保留。

本机 7 项定向测试通过。独立 P0/P1 核查通过，N607 实际 checkpoint smoke 在启动时先行验证。
