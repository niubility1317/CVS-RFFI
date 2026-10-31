# Phase1 源样本旧恢复器与 Tri-SIG-TIME 一致性诊断

- run_id：`20261101-diagnostic-phase1-dura-trisig-source-s2026092701-r01`
- group_id：`phase1-dura-source-trisigtime-consistency`；类别：`cvs`；阶段：`Phase1-source-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅重用 Phase1 源训练记录的原始接收和旧恢复器原始输出，以原 Tri-SIG-TIME SSL 比较对 clean 基座判断的保留程度。

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

## 源域同基座结果

VERIFIED／ANALYZED_SOURCE_ONLY。固定 512 个 Phase1 源训练物理记录（L128/U384），使用同一批物理记录在源视图 01/20 下的 1024 条观测。只读取旧恢复器原始 native IQ，不读取旧分类器输出、相位对齐波形或相关结果；clean、raw 和 native 均重新由冻结 Tri-SIG-TIME SSL 识别。没有读取 TX 真值、V 或测试 IQ，也没有网络更新。

| 源视图／分组 | 数量 | raw 对 clean 基座判断一致率 | 旧恢复输出对 clean 基座判断一致率 | 旧恢复输出 IQ NMSE |
|---|---:|---:|---:|---:|
| 01／全部 | 512 | 86.719% | 95.898% | 0.0202 |
| 01／low urban | 66 | 62.121% | 86.364% | 0.0667 |
| 01／BLOCKED | 83 | 42.169% | 78.313% | 0.0799 |
| 20／全部 | 512 | 89.062% | 96.094% | 0.0194 |
| 20／low urban | 70 | 65.714% | 85.714% | 0.0683 |
| 20／BLOCKED | 62 | 37.097% | 77.419% | 0.1009 |

一致率比较的是各源样本 clean 基座输出，并非真实 TX 准确率；两视图共享物理记录，不可当作 1024 个独立设备样本。完整分组数、取值范围和修复／损伤计数恒等式已独立核对。两视图均说明该源子集上的原生恢复输出是可研究的生成起点，并不证明后续生成模型必然改善。

本诊断启动时已有开发测试结果被研究过程观察；因此后续相关研究如继续使用该测试集，必须披露该测试集并非盲测，不能以此声称独立泛化。此前已登记的候选参数不变。
