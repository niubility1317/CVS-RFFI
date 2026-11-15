# Tri-SIG-TIME 十类与六源损失权重及梯度诊断

## 技术恢复与固定范围

替代 `20261115-phase1-pamf-repl-weight-clean-m14-r01`：14 行均完成原有 10 轮 E21–E30 并保存 E30 权重，随后原生 CoordinateSinc 的版本计数与源 V 代理 `inference_mode` 不兼容，全部在无目标预测前失败。恢复使用 `no_grad` 只读代理；针对性回归先复现失败再通过。r02 每行只继承同设置、seed、配方、源数据契约的原 E20 和自身 r01 E30，先计算 E30 V 代理，再独立续训 E31–E50（4440 步），不重训 E21–E30。E50 后全 V 代理，冻结后自动 raw clean 三集合预测和独立 truth-last 评分。

原 seed `2026092701` 是按已曝光 clean 目标上最大负差选取，因此仅作探索。14 候选的各项权重、单因素矩阵、准确数据角色、梯度遥测、5 进程每卡上限与 r01 不变，不依据新目标成绩改写训练；U 真 TX 始终隐藏，源 V 阈值正确率仅是代理。原产物保留；新 run 使用独立、不可覆盖目录和 owner，已完成 E30 的阶段来源清楚保留。

**VERIFIED / RUNNING**：执行 Git commit `6522307977daf9b9a15bbbd6bce4c69a66291604`；归档校验和远端编译通过。14 个候选均通过真实权重无 query smoke，并使用真实 E30 模型的原生 Sinc 路径检查两设置全部 V，验证修复。独立核实 14 个 E31–E50 worker、GPU、CWD、命令、E30 代理 artifact 和增长的 step 日志；所有预测与评分均未完成。十类 `weight3` E30 的 V 代理通过 22092 条、正确 22017 条；六源 `weight3` 为 30729 条通过、30697 条正确。二者是源 V 代理，不是训练 U 真值准确率。正式结果待各行冻结与 clean 独立评分。

- run_id：`20261115-phase1-pamf-repl-weight-clean-m14-r02`
- group_id：`phase1-pamf-repl-weight-clean`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

单种子十四个独立REPL变化；训练后自动raw clean预测与truth-last评分。

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
