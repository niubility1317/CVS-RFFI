# 普通PL冻结前缀的三模型一致性源域诊断

- run_id：`20261025-phase1-PAMF-PL-consensus-source-m6-r01`
- group_id：`phase1-pamf-source-pseudo-reliability-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

F0及1%标注的六份同契约普通PL E20冻结前缀；CPU完整sourceV比较单模型、三者一致、平均概率及双peer一致，检验误差相关性与预测类别校准，不训练、不访问target。

既有源诊断发现两种现象：多数被 As 门控拒绝的 raw 伪标签仍然正确，而 1% 标注的真实类别索引3存在集中错误（候选95.1028%、转移96.3082%），整体转移正确率99.5291%掩盖了该差异。该事实来自固定普通 PL E20 前缀的源 V，不是 U 真值或当前新学生的结果。本 run 检验独立 seed 的默认头预测能否提供更有用的错误识别依据，不以此预先断言集成有效。

固定阈值均为0.95，完整比较三个 single、三模型同类且全部达阈值、三模型 softmax 概率平均后达阈值，以及排除当前 seed 的另外两个模型同类且都达阈值。保存每个规则的正确数、错误数、覆盖率、两两错误重叠和三者共同错误。门控可以一致地犯错，所有负结果都保留，不据源 V 拟合新阈值或分类头。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

仅使用 F0、label01 各自原普通 PL E20 的 seed2026092701/02/03，共六份已核实源域前缀。每份在加载前核对真实 payload、seed、scratch 来源、source roles、无 target／teacher／native EMA／native U 标志。E20 已包含 E14–E20 的普通 PL，不称为纯监督前缀。模型默认头保持原结构，eval、无梯度、无 optimizer，读取前后检查 buffer 不变。

每模型首先用两条 L_s IQ 完成真实 checkpoint 无 query smoke，随后每个协议完整遍历27000条源 V。源 V 的真实类别用于诊断计数，预测类别用于可观察的精度分组，两者分开报告。U_s 不打开，不读取 U 真值、目标数据或 G 权重／输出。两个协议与模型 seed 之间可能重复物理观测，不声称54000个唯一样本或独立统计重复。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

CPU 单 worker、两线程，CUDA_VISIBLE_DEVICES 为空，allowed_gpus为空。该独立诊断不进入 GPU 训练队列；现有 ChannelPL 六条健康训练仍使用其不可变 release1fa92。新 root 独占，不覆盖既有产物；异常保留 partial、不自动重试。

10项聚焦检查已通过，覆盖6/10类、均值概率规则、排除自身的双peer、共同高置信错误、真实／预测类别分组区别，以及占用GPU／错误run在打开输入前拒绝。唯一独立 P0/P1 审查 PASS，未发现会让本次 CPU run 错、越权、占 GPU 或无法执行的缺陷。执行 commit 和独立启动证据在发布后记录。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

本 run 的三模型策略是一个源域联合诊断，不能作为三个独立 F 学生 seed 的新目标集结果。没有训练新 SSL、没有产生目标 prediction；原 clean≥+1 pp／六景>+1.5 pp 目标保持未达成。后续任何新训练方法另建独立固定矩阵，保留原识别器和恢复器默认版本。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
