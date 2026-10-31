# Phase1 锚定流对原恢复器的源样本配对诊断

- run_id：`20261101-diagnostic-phase1-anchor-source-paired-s2026092701-r02`
- group_id：`phase1-anchor-source-paired-diagnostic`；类别：`cvs`；阶段：`Phase1-source-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在相同源训练样本上比较固定 DURA 与已训练锚定流对干净基座判断的一致性及波形误差。

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

## 源训练样本配对结果

VERIFIED／ANALYZED_SOURCE_ONLY。固定 512 个 Phase1 源训练物理记录，在源视图 01 和 20 各评估一次，两个视图共享物理记录。原始 DURA 和最终锚定流均冻结，以 clean 信号在 Tri-SIG-TIME SSL 中的预测作为参照；这是一致性而非 TX 真实准确率。没有读取 TX 标签、V 或测试 IQ，也没有网络更新。

| 源视图 | DURA 与 clean 判断一致 | 锚定流与 clean 判断一致 | 修复／损伤 | DURA 平均 IQ NMSE | 锚定流平均 IQ NMSE |
|---|---:|---:|---:|---:|---:|
| 01 | 491/512 | 491/512 | 4／4 | 0.020213 | 0.019680 |
| 20 | 492/512 | 492/512 | 2／2 | 0.019440 | 0.018861 |

遮挡状态下，视图 01 的一致数为 DURA 65/83、锚定流 65/83；视图 20 为 DURA 48/62、锚定流 47/62。小组样本有限，不能据此推断总体或分布外性能。

诊断实现使用真实 DURA 逐批重算并与既有源 native IQ 比对；独立进程检查场景、状态、角色分组总数及修复／损伤恒等式。源内波形误差降低但判断一致数未净增加，支持继续研究针对困难样本识别边界的机制；它本身不能证明任何新方法有效。源样本参与过模型训练，此前 Phase1 开发测试已观察，均不是盲测。r01 统计接口失败日志已保留，r02 只修复 NumPy／PyTorch 布尔掩码转换。
