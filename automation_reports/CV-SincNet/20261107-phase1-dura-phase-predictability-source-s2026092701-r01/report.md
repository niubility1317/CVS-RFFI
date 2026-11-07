# 源域相位可预测性记录互斥诊断

- run_id：`20261107-phase1-dura-phase-predictability-source-s2026092701-r01`
- group_id：`phase1-dura-phase-predictability`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：技术失败。真实 DURA 源上下文重放通过，但远端 NumPy 布尔数组到 PyTorch 掩码的转换失败；未开始 ridge 拟合，未生成结果。

## 目的与对照

固定源上下文，按物理记录分离 ridge 相位拟合与留出，比较可观测特征。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

N607 上 NumPy 2.2.5 和 PyTorch 2.1.0 的该转换可独立复现失败。输出目录未创建，原发布和失败日志保留。修复仅将布尔数组转换为 Python 布尔列表后显式构造 `torch.bool`，三视图的物理记录拆分保持不变；如继续运行，使用新 run 和不可覆盖输出，不把本次失败作为实验结果。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
