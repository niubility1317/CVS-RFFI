# Phase1 可学习复数滤波系数恢复网络

- run_id：`20261101-phase1-filter-student-E30-s2026092701-r01`
- group_id：`phase1-learned-widely-linear-fir`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；七视图预测和独立复核完成。

## 目的与对照

学习从接收 IQ、DURA 输出及冻结地面基座特征预测逐窗复数与共轭 FIR 校正系数。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整结果、对照和证据见下方。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## GPU 启动与首轮核验

GPU 4 的 worker 243045 已启动；真实冻结 Tri-SIG-TIME SSL 和 DURA 的 128 条源记录 GPU 检查通过。独立重载首轮检查点，核对 493 次更新、训练配置、优化器和 GPU 随机状态，并在两条源记录验证输出确定且校正不超过 20%。首轮检查点 SHA-256 为 `9021e3f1094f6d0773e28ffcd15d8963a912f4f5a9a2772eb999dce3ac8e6abc`。这不代表 GPU 中断续训已实测，也不代表已有 Phase1 测试结果。

## 固定训练完成

已完成预先固定的 30 轮、14790 次源域更新。使用真实冻结 Tri-SIG-TIME SSL 和 DURA 独立重载最终模型，在两条源记录核对输出确定性和 20% 校正限制；最终检查点 SHA-256 为 `743a8bc20106aa464390a955c9c80141df0fd1cbc870eb572ac3a784f7ef38ca`。七视图预测和评分已完成。

## 七视图测试结果

地面 Tri-SIG-TIME SSL 干净基准为 86.5387%，恢复干净视图为 86.4571%。每个视图均为同序 168000 个物理 ID，预测全部固定后接入真值。独立重新评分、核对基座决策并逐条配对 DURA。

| 场景 | 本模型准确率 | DURA 准确率 | 相对地面基准下降（百分点） | 相对地面基准下降比例 |
| --- | ---: | ---: | ---: | ---: |
| practical_high | 83.6821% | 84.3435% | 2.8565 | 3.3009% |
| practical_mid | 82.4536% | 83.0161% | 4.0851 | 4.7206% |
| practical_low_suburban | 80.7488% | 81.3220% | 5.7899 | 6.6905% |
| practical_high_urban | 82.4137% | 83.0494% | 4.1250 | 4.7667% |
| practical_mid_urban | 75.2190% | 75.7060% | 11.3196 | 13.0804% |
| practical_low_urban | 68.3042% | 68.8012% | 18.2345 | 21.0709% |

六场景均低于 DURA。按下降不超过 3 个百分点的口径，只有 high 场景达标；按相对下降不超过 3% 的较严格口径，六个场景均未达标。该 Phase1 开发测试此前已被观察，单 seed 结果不属于独立盲测。机器证据和 SHA-256 保存在本目录。
