# Phase1 强正则残差 FIR 系数恢复网络

- run_id：`20261101-phase1-filter-residual-E30-s2026092701-r01`
- group_id：`phase1-regularized-residual-fir`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；实际状态按 events.jsonl 及远端记录更新。

## 目的与对照

以强正则源域残差目标训练复数 FIR 系数预测网络，输入接收 IQ、DURA 输出及冻结地面基座特征。

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

## 启动与首轮核验

真实冻结 Tri-SIG-TIME SSL 与 DURA 的 CPU 2 条、GPU 128 条源记录检查均通过；GPU 4 的 worker PID 202717 已开始正式训练。独立重载首轮检查点，核对 493 次更新、训练配置、优化器及 GPU 随机状态，并在两条源记录检查确定输出、20% 校正约束和零残差身份。首轮检查点 SHA-256 为 `a73d88bafa6e9b8976c0ca00b8b2b092a057df9910969160e220077397c22445`。

这些检查不代表真实 GPU 中断续训已验证，也不代表 Phase1 测试性能。固定 E30 训练和七视图预测仍待完成。
