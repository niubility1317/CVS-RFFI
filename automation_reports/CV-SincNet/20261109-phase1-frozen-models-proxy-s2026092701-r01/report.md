# 冻结新恢复器源域识别一致性

- run_id：`20261109-phase1-frozen-models-proxy-s2026092701-r01`
- group_id：`phase1-frozen-models-proxy`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

对三个已经冻结的恢复模型比较源 clean 伪类别一致性，不是 TX 真值准确率。

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

## 固定对照

比较 DURA、累积量条件残差、累积量加同记录排序和独立多相盲恢复。四种恢复输出连接同一冻结 Tri-SIG-TIME SSL，依据 clean 伪类别计算源域一致性及与 DURA 的逐条增加/失去一致数；672 条源观测、14 组每组 48 条。每条源 clean/raw/DURA 决策复核既有固定预测。该指标并非 TX 真值准确率，源 pilot 已探索且 DURA 可能见过。

候选源训练未读 F 或 TX；F 只在本次冻结后源评估使用，不改变已有训练、模型或七场景测试。CPU 两线程；实际检查点首批 smoke，非有限结果和固定基准不一致即停止。代码 5b31f7559a0b11706fd03facb7c5378b2da6cb50；独立 P0/P1 静态检查无阻断问题；远端配对统计及非法输入检查通过。
