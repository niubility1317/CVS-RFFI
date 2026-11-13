# Phase2 第二组：C 地面主干跨七种 B 适应与原生 C 注册

- run_id：`20261103-phase2-csource-crosscil-7x3-m4-r01`
- group_id：`phase2-group2-csource-bc`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

All fixed 7 B by 3 native C-source/C-core transplants; matched four seeds and original K/N matrix.

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


## 2026-11-03 加速接管完成

旧 controller 仅在 16 条自有 worker 自然结束后退出，新 controller 2180731 接管。真实 source-only 无 query smoke 21/21 PASS；在真实 MoPC 源模型的 151 组参数上，新旧正则损失和梯度逐位一致。旧 run 370 个闭合任务、2770 行预测已逐行核实并硬链接复用；先导预测及评分引用保留。目标 truth 仍待全部预测冻结后独立使用。服务器端的实际速度需在新任务完成后按相同 K/N、seed 与 GPU 共享情况比较，不能引用本机合成倍数替代。
