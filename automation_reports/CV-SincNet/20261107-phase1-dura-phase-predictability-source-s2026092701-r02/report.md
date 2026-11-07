# 源域相位可预测性记录互斥诊断

- run_id：`20261107-phase1-dura-phase-predictability-source-s2026092701-r02`
- group_id：`phase1-dura-phase-predictability`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：源域拟合和记录互斥的留出评估已完成；真实 DURA 源上下文重放通过。结果与限制见[分析](analysis.md)。

## 目的与对照

固定源上下文，按物理记录分离 ridge 相位拟合与留出，比较可观测特征。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

在合法源 IQ 中用 24579 个物理记录拟合预先固定的三种 ridge 相位预测器，以另外 8189 个物理记录留出；每条记录的三个模拟视图始终在同一侧。预先固定零相位、训练集常数均值及三种特征组合，所有行均报告。该工作拟合了源域线性诊断器，但没有更新 DURA、训练新的部署恢复器或读取 F、TX 标签、V、目标。留出仅对本次 ridge 拟合成立；DURA 或其他历史模型可能见过这些源记录。

真实 DURA checkpoint 源重放、六项聚焦测试和独立审查通过。首次 r01 因远端 NumPy 布尔转换兼容错误在拟合前失败，本次 r02 仅修复转换方式并保持原矩阵；r01 日志保留。预注册任务执行很短，源 smoke 与完成产物可核实，未捕获运行时 GPU 进程快照。源 IQ 和相位指标不能直接说明身份识别成绩。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
