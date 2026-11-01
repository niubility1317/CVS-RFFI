# Phase1 强正则残差 FIR 系数恢复网络

- run_id：`20261101-phase1-filter-residual-E30-s2026092701-r01`
- group_id：`phase1-regularized-residual-fir`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；七视图预测和独立复核均完成。

## 目的与对照

以强正则源域残差目标训练复数 FIR 系数预测网络，输入接收 IQ、DURA 输出及冻结地面基座特征。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整结果、对照和证据见下方“七视图测试结果”。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 启动与首轮核验

真实冻结 Tri-SIG-TIME SSL 与 DURA 的 CPU 2 条、GPU 128 条源记录检查均通过；GPU 4 的 worker PID 202717 已开始正式训练。独立重载首轮检查点，核对 493 次更新、训练配置、优化器及 GPU 随机状态，并在两条源记录检查确定输出、20% 校正约束和零残差身份。首轮检查点 SHA-256 为 `a73d88bafa6e9b8976c0ca00b8b2b092a057df9910969160e220077397c22445`。

这些检查不代表真实 GPU 中断续训已验证，也不代表 Phase1 测试性能。

## 固定训练完成

已完成预先登记的 30 轮和 14790 次更新。独立使用真实冻结 Tri-SIG-TIME SSL 和 DURA 在两条源记录上重载最终模型，确认输出确定且满足 20% 校正约束；最终检查点 SHA-256 为 `5cb013504f76e24c45f1648e4e4d62dd916bafde55e1e10d6609bb79960e7232`。训练未读取 TX 标签、V 或测试 IQ。

七个 Phase1 视图全部预测固定后才接入真值；该开发测试此前已被观察，不属于独立盲测。

## 七视图测试结果

地面 Tri-SIG-TIME SSL 干净基准为 86.5387%；恢复干净视图为 86.5863%。每个视图均为同序 168000 个物理 ID，预测全部固定后接入真值。三套独立检查重新评分、核对原始基座决策及逐条配对 DURA。

| 场景 | 本模型准确率 | DURA 准确率 | 相对地面基准下降（百分点） | 相对地面基准下降比例 |
| --- | ---: | ---: | ---: | ---: |
| practical_high | 83.8095% | 84.3435% | 2.7292 | 3.1537% |
| practical_mid | 82.5446% | 83.0161% | 3.9940 | 4.6153% |
| practical_low_suburban | 80.8917% | 81.3220% | 5.6470 | 6.5254% |
| practical_high_urban | 82.5613% | 83.0494% | 3.9774 | 4.5961% |
| practical_mid_urban | 75.3738% | 75.7060% | 11.1649 | 12.9016% |
| practical_low_urban | 68.4565% | 68.8012% | 18.0821 | 20.8949% |

六场景均低于 DURA。按下降不超过 3 个百分点的口径，只有 high 场景达标；按相对下降不超过 3% 的较严格口径，六个场景均未达标。该 Phase1 开发测试此前已被观察，单 seed 结果不属于独立盲测。

已保存训练完成、预测完成、测试汇总和结束标记，以及 [独立重评分](independent_rescore.txt)、[DURA 配对](paired_dura_comparison.txt)、[基座决策核对](baseline_decisions.txt)和 [文件哈希](sha256.json)。
