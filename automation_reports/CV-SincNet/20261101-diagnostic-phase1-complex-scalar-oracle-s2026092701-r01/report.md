# Phase1 复数增益反事实源域诊断

- run_id：`20261101-diagnostic-phase1-complex-scalar-oracle-s2026092701-r01`
- group_id：`phase1-privileged-complex-scalar-source`；类别：`cvs`；阶段：`Phase1-source-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在源训练配对样本上，以干净 IQ 求最小二乘复数增益，诊断简单盲均衡路线的可能性。

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

## 源训练反事实结果

真实冻结 Tri-SIG-TIME SSL，对固定 512 条源记录（L_s 128、U_s 384）分别在两个源视图上计算；共 1024 条观察但只包含 512 个不同物理记录。复数标量每条都使用配对 clean IQ 拟合，属于特权反事实，推理时不可用。组计数和加权均值已独立复算。

| 源视图 | BLOCKED 样本 | 原始与 clean 教师一致 | DURA 与 clean 教师一致 | 特权复数标量与 clean 教师一致 |
|---|---:|---:|---:|---:|
| 01 | 83 | 35/83 | 65/83 | 22/83 |
| 20 | 62 | 23/62 | 48/62 | 15/62 |

两视图全样本特权标量的均值 IQ NMSE 为 0.8320、0.8378，DURA 为 0.0202、0.0194；标量在均方意义下优于原始输入，但 clean 教师一致性明显更差。这说明此单一复数比例族在当前源样本及该损失下不足以解释问题；它不排除时变、滤波或更复杂均衡。clean 教师一致性不是 TX 真实准确率，样本与训练重复，不能推断未见测试集。
