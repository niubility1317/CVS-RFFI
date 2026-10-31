# Phase1 旧恢复锚定与梯度平衡生成轨迹

- run_id：`20261101-phase1-dura-anchored-balanced-E30-s2026092701-r01`
- group_id：`phase1-dura-anchor-balanced-flow`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结原 Phase1 DURA 和 Tri-SIG-TIME SSL，使用 DURA 输出为生成轨迹起点，源训练时对波形与语义梯度进行冲突投影和幅度限制。

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

## CPU smoke 技术失败与修复

状态 FAILED；未提交 GPU worker，未开始正式训练或测试。真实 F／DURA 与两条 L_s 的 CPU smoke 在 checkpoint 严格重载比较处失败，原 `smoke.log`、临时 checkpoint 和状态保留。

已用相同远端 PyTorch、真实检查点和相同两条源样本复现：保存前后全部参数逐项完全一致；由于参考生成器处于 train 模式、重载生成器处于 eval/frozen 模式，输出最大绝对差为 `1.1920929e-7`。将两边均置为 eval/frozen 后输出逐位一致。根因是比较路径模式不一致，不是 checkpoint 权重变化。

后继 r02 只将 smoke 参考路径切换为 eval/frozen，并启用新 run ID；正式源训练、损失、梯度规则、预算和预测配方保持不变。回归测试使用非零输出权重，检查相同模式下严格相等以及错误 DURA digest 被拒绝。
