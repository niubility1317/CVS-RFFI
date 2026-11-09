# 固定恢复专家的源域频率互补诊断

- run_id：`20261109-phase1-spectral-expert-complement-source-s2026092701-r01`
- group_id：`phase1-spectral-expert-complement-source`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在已探索源域 pilot 上比较 DURA、联合粒子、残差扩散和固定融合网络的 IQ 误差，并计算依赖 clean IQ 的不可部署频率选择 oracle。

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

## 固定诊断与科学边界

使用 672 条已探索源域 pilot 物理记录，包含 RX4/6、RX8 两组与七个视图，每单元 48 条。三个源训练模型的 fit 均排除 pilot，但固定 DURA 可能见过这些记录。逐条比较 DURA、联合粒子、残差扩散和频率融合的归一化 IQ 误差，统计融合相对 DURA 的改善及恶化条数。

理想频率选择器每个频点直接读取该源样本的 clean IQ，以最小复数平方误差从三个固定专家中选一个。这是依赖干净真值的不可部署源域诊断，只是这三个候选构成的逐频选择空间中的理想误差，不是任意网络的性能上界，也不是识别准确率。训练、选模或阈值不得读取目标结果。无 F 权重与输出、TX 标签、V 或目标读取，不更新模型。

GPU5 单任务 worker，至少 8192 MiB 空闲且少于三个 GPU 计算进程；真实源记录两条和实际权重 smoke 后自动处理全部 672 条。代码 6417da3a28ea8667510958310e880fdd272d5ea5，精确命令、来源、输出及停止规则见 experiment.json。两项数值检查与一次独立 P0/P1 静态审查通过；服务器执行尚未完成。

## 运行与诊断结果

发布归档本地与 N607 SHA-256 均为 63d72d818ac7b87694f9af1ae38ae7eca5c6ba2d23efbe773422c7c7bb85bdb9；远端编译及导入通过。GPU5 启动时空闲 24110 MiB、无计算进程；真实权重两条源记录 smoke 和 672 条源域诊断全部完成，诊断循环用时 8.96 秒。进程过快结束，未取得存活时的独立 PID/GPU 占用快照；完整结果与日志存在。详见 [源域诊断分析](analysis.md)。
