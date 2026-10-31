# Phase1 既有恢复网络的 Tri-SIG-TIME 同基座对照

- run_id：`20261101-phase1-dura-trisigtime-baseline-s2026092701-r01`
- group_id：`phase1-dura-trisigtime-matched-baseline`；类别：`cvs`；阶段：`Phase1-baseline-evaluation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

将已有 Phase1 DURA 恢复 IQ 经冻结 Tri-SIG-TIME SSL 重新识别，与新候选的同输入原始预测按同物理 ID 对照。

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

## 已核实状态（2026-11-01）

已有 Phase1 DURA 的 `native_map_prior/prediction` 包含 clean 加六景恢复 IQ；其 168000 个物理 ID 与本次 Phase1 正式参考逐项同序。后续只取恢复 IQ，用原始 Phase1 三视图 Tri-SIG-TIME SSL 固定权重重新分类；旧识别器 logits、features 和分数均不作为本对照。原始信号决策取新候选已完整固定的同一 Tri-SIG-TIME 预测。

代码版本 `5dc285d0bbd44462178f3715f495bf32dc9ac918` 已发布，归档两端 SHA-256 一致。真实 Tri-SIG-TIME 权重与两条源训练记录的 CPU smoke 已通过；未训练或更新模型，未读测试 IQ 或 TX 标签。

控制器 PID 2197520 已独立读回，正等待第一条新候选控制器与 GPU worker 结束。七视图对照预测尚未开始；全部预测固定后才独立评分。当前没有同基座 DURA 新分数。

## 输入来源补充核验

旧 Phase1 DURA run 的原始 `launch.json` 将 `--formal-reference` 指向 `20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05/reference`；原预测代码从该目录逐视图读取 received IQ，再写出恢复 IQ。本次候选同样从该正式参考目录读取输入；两份恢复结果／参考的 168000 条物理 ID 已独立核实逐项同序。该检查支持输入对应关系，不替代即将进行的同基座识别和评分。

## 完整同基座对照结果

状态 VERIFIED／ANALYZED，七视图每视图 168000 条。DURA 旧恢复 IQ 与新候选同序物理 ID，均由原始 Phase1 三视图 Tri-SIG-TIME SSL 固定基座识别；没有使用旧分类器分数。独立 CPU 复算总体、TX、RX、day 和 Macro-F1，结果与原评分一致。

| Phase1 视图 | DURA＋Tri-SIG-TIME | 新轨迹候选＋Tri-SIG-TIME | 新候选减 DURA |
|---|---:|---:|---:|
| clean | 86.6155% | 86.2345% | -0.3810 pp |
| practical_high | 84.3435% | 75.5155% | -8.8280 pp |
| practical_mid | 83.0161% | 74.0077% | -9.0083 pp |
| practical_low_suburban | 81.3220% | 72.5411% | -8.7810 pp |
| practical_high_urban | 83.0494% | 72.1893% | -10.8601 pp |
| practical_mid_urban | 75.7060% | 60.5143% | -15.1917 pp |
| practical_low_urban | 68.8012% | 51.0363% | -17.7649 pp |

同视图逐样本配对验证了两方法原始预测完全一致，差值等于净修复数除以 168000。新轨迹候选六个场景全部低于 DURA；两个模型均未满足六场景距原始地面 clean 86.5387% 不超过 3 个百分点的目标。DURA 仅 practical_high 差距 2.1952 pp，low_urban 差距 17.7375 pp。单模型 seed，固定测试集已知，不构成独立盲测。
