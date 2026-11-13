# 10类 PAMF 分支原型温度源域内部开发试验

- run_id：`20261112-phase1-pamf-branch-prototype-source-tuning-s2026092701-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：ANALYZED（仅源域开发；实际状态按 events.jsonl 及独立证据更新）

## 目的与对照

从零训练三组相同划分的 Hybrid＋分支原型；仅在源 L 内部开发集选温度。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

[计算环境] 源域单种子三候选训练与开发集评分完成，未记录失败。执行代码 commit `fc7d30368628a9b2e284bebb9dbd44ce7b91cde8`。三个候选均使用同一 1305 条源 L_dev，模型均从零训练，未访问目标数据或 U 真值。

| 分支原型温度 | clean 准确率 | 模拟信道准确率 | 两者等权平均 |
|---|---:|---:|---:|
| 0.1 | 91.8774% | 59.9234% | 75.9004% |
| 0.2 | 92.1073% | 60.5364% | 76.3218% |
| 0.4 | 92.1073% | 59.6935% | 75.9004% |

模拟信道为对源 L_dev 的固定混合代理：low urban 437 条、low suburban 419 条、mid urban 449 条；不是用户表中的六景测试均值。温度 0.2 为三者中源域等权平均最高，但它是原温度，相对原温度的改进为零，不满足预登记的 0.25 个百分点晋级门槛。该单种子源域结果不能证明超过普通 PL；尚无本次目标预测或评分。三票门槛独立对照继续执行。

按预登记 artifact 逐项记录路径和缺项；保留每 row 与 RX/day/TX/scene/K/seed 的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
