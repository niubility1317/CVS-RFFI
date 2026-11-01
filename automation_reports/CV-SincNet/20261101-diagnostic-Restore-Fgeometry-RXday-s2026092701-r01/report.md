# 恢复性能源端诊断：冻结F特征的接收机与日期相关变异

- run_id：`20261101-diagnostic-Restore-Fgeometry-RXday-s2026092701-r01`
- group_id：`restoration-source-Fgeometry-RXday-association`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅复用原63000条LsUs的冻结F0特征与RX/day元数据，分解组间变异并比较VarF固定尺度；不读取IQ/TX/V/OE/target，不加载神经网络或拟合分类头。

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

## 机制与边界

只分析已有原 L_s/U_s 63000 条 clean 的冻结 F0 fused160 特征。按公开 RX、日期和 RX×日期单元计算组均值的组间方差，对总体方差取比；再用 VarF 实际固定分母加权，比较所有维度和原四个方差分组。条件增量是联合分量减相应边际分量，包含交互，不能称纯主效应或把边际分量直接相加。

全部源行与维度固定，不按结果剔除。无 TX 条件信息，因此 RX/日期相关性不证明因果干扰或身份与环境可分；日期可能包含真实设备漂移。此处是 clean 特征能量，既不是恢复残差，也不是指导梯度能量。只用于检验一个源端机制假设，不直接选择删除维度、网络或超参数。

仅新 JSON 输出，0 神经前向、0 checkpoint 加载、0 训练、0 目标输入，不读取 IQ/V/OE/TX 或拟合诊断分类头。沿用 VALIDATED_ONCE 缓存及其原真实源2条 checkpoint smoke；本阶段首步只核对实际消费的数据句柄，PASS 后立即统计，不额外加载模型。3 项聚焦检查和唯一 P0/P1 审查 PASS。
