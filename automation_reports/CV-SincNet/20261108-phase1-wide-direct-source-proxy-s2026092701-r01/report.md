# 宽范围独立恢复器源域一致性

- run_id：`20261108-phase1-wide-direct-source-proxy-s2026092701-r01`
- group_id：`phase1-wide-direct-source-proxy`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结窄范围、宽范围及宽范围加物理损失三个独立模型，以固定 clean 伪类别比较源域识别器一致性；并非 TX 真实准确率。

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

## 固定矩阵与运行

在 672 条已探索源 pilot 上比较窄范围、宽范围以及宽范围加源物理损失三个已固定第20轮的独立恢复器；三者拟合 ID 与 pilot 不相交，DURA 历史重叠未知。冻结 Tri-SIG-TIME SSL 的 clean 伪类别只在恢复器冻结后用于源域比较，不是 TX 真值或准确率，不改变已运行的七景测试。

代码 `37c6230fa7c7ae79b30872e92b12f720021e3c24`；3 项聚焦测试、独立 P0/P1 审查、远端归档校验与编译通过。N607 CPU 两线程、隐藏 GPU、PID 1020526；首批 8 条真实源记录的 raw／DURA／clean 固定参照一致。无 TX 真值、V 或目标读取，也无模型更新。

- 672 条源域比较完成：六受损景合计 DURA 546/576、窄范围 479/576、宽范围 534/576、宽范围＋源物理损失 541/576。为与 clean 伪类别一致数，不是 TX 真值准确率；[完整分组与逐条变化](analysis.md)。
