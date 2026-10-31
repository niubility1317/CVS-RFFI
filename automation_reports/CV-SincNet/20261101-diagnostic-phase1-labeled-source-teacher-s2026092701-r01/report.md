# Phase1 合法有标签源集的基座与恢复判断诊断

- run_id：`20261101-diagnostic-phase1-labeled-source-teacher-s2026092701-r01`
- group_id：`phase1-labeled-source-teacher-dura`；类别：`cvs`；阶段：`Phase1-source-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅使用 L_s 合法训练标签，比较 clean 基座与冻结 DURA 恢复后基座判断，核实物理 ID 对齐。

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

## Phase1 合法有标签源集诊断结果

VERIFIED／ANALYZED_SOURCE_ONLY。读取全部 6300 条 Phase1 L_s 合法训练标签；F 为冻结 Tri-SIG-TIME SSL，G 为冻结原 DURA。源视图 01、20 复用同一批物理 ID；未读取 U_s 标签、V 或测试 IQ，没有模型更新。

| 源视图 | clean F 正确 | DURA 后 F 正确 | clean 与 DURA 判断不同 | 仅 clean 正确 | 仅 DURA 正确 | 两者均错 |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 6271/6300 | 6012/6300 | 270 | 262 | 3 | 5 |
| 20 | 6271/6300 | 5996/6300 | 286 | 278 | 3 | 5 |

遮挡状态中，视图 01 的 clean 1045/1050、DURA 834/1050；视图 20 的 clean 1056/1059、DURA 825/1059。分场景和状态的计数汇总及配对恒等式已由独立进程核验。

标签 metadata、R07 训练源 IQ 和 F 源 IQ 实际存储顺序不同。按真实物理 ID 和底层存储索引重排后，6300 条 clean IQ 以 rtol=1e-5、atol=1e-6 匹配；预检的最大绝对差为 2.3842e-7。不得按原始行号关联标签。

当 clean 与 DURA 判断不一致时，clean 判断在该 L_s 训练子集上通常符合真实标签，为使用 clean 教师伪目标提供有限的源内依据；无法据此断言 U_s 或测试域同样可靠。clean F 和这些源样本属于训练数据，结果不是独立泛化评估。已启动的无 TX 标签边界学生训练保持原登记配置。
