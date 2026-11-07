# 大容量恢复器源域初始梯度分组诊断

- run_id：`20261108-phase1-large-context-source-gradient-s2026092701-r01`
- group_id：`phase1-large-context-source-gradient`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；源域无更新诊断完成。

## 目的与对照

固定源缓存，比较互斥源物理记录两侧的输出头批次平均梯度；不更新模型。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

预登记：复用合法、已验证的 32768 源 ID × 三视图 DURA 上下文，固定随机种子选出互不重叠的两组各 1024 个源物理 ID，同一记录的三个视图始终在同一组。各组随机顺序、batch32，在独立新初始化的 14792706 参数大容量分支上计算输出头梯度；使用原 `dura_guard` 源 IQ 损失，无优化器或参数更新。统计每组平均梯度范数、单批梯度范数均值和两组平均梯度余弦。不读取 F、TX 标签、V 或目标，真实 DURA checkpoint 重放 32 条源上下文作为启动检查。

本诊断不读取当前正式训练模型，也不产生新的可部署恢复器；三个视图共享物理记录，不能当作独立样本。结果只能解释批次与分组平均梯度，不能证明单条记录梯度一致，更不能代表身份识别准确率。结果不会变更在跑模型或其七景测试。

两项统计与无效输入检查、一次独立 P0/P1 审查通过。启动时检查 GPU 7 的实际资源，保留其他 owner；输出冲突、源契约错误、非有限梯度或技术故障时终止本诊断并保留证据。

真实 DURA 源 checkpoint 重放及 192 个批次计算完成，两组平均输出头梯度余弦 0.9973；平均梯度范数约为单批范数均值的 59.4% 和 61.4%。仅适用于本次独立初始化与分组，不代表在跑模型后续梯度或身份准确率。详见[分析](analysis.md)。进程完成过快，未捕获运行中的 PID／GPU 快照；完成产物和 CUDA smoke 已核实。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
