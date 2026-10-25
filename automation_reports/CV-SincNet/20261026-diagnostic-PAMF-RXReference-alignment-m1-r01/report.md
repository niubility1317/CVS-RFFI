# 冻结模型的身份参照与默认头特征梯度一致性诊断

- run_id：`20261026-diagnostic-PAMF-RXReference-alignment-m1-r01`
- group_id：`phase1-pamf-receiver-reference-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（完整冻结源梯度诊断完成，性能目标仍未达成）

## 目的与对照

同seed E30/PLControl及三参照F50全部冻结，在12批可见L/隐藏U上比较参照目标与真实默认头soft/hard身份目标的特征梯度；不训练、不读V/target。

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

## 冻结源诊断发布前范围

参见 [最小预登记](preregistration.md)。五个同seed01 reader全部冻结，只在可见源L和隐藏源U上求特征叶节点梯度，不更新F/G/参照或读取V/目标。三项聚焦检查与compile通过，一次独立P0/P1审查PASS；实际L2默认头与特征梯度smoke由launcher先执行。U与冻结伪标签的一致性不能称为真实准确率，梯度相关性也不是目标收益证明。

## 完整冻结源诊断

`SOURCE_FEATURE_ALIGNMENT_COMPLETE`，独立完成产物读回VERIFIED。五reader同一个seed01；12批全部完成，L1536/U3072、F38400包前向、540次特征梯度，360统计单元均保留。真实L2/defaulthead/特征梯度smoke PASS，F所有参数及registered buffers不变，参数.grad、optimizer、新头拟合、V/目标/U真值读取均为0。耗时9.74秒，首次独立观察时已结束，因此没有运行中PID/CWD/GPU窗口；依据完成artifact核实，没有为捕获窗口重复执行。

下面只看与sourceHold实际新增监督对应的接受集合和参与RX（排除RX1）。身份目标为冻结raw伪标签，不能把不一致样本称为真错误。末列是已与伪标签一致的恢复样本上的中位范数比。

| 冻结F | G伪标签不一致数 | 参照/硬身份梯度cos中位数 | 不一致样本参照/硬身份范数比 | 一致样本范数比 |
| --- | ---: | ---: | ---: | ---: |
| E30 | 179 | 0.790370 | 0.073990 | 51.557451 |
| PLControl | 182 | 0.813690 | 0.074000 | 31.685495 |
| RawReference | 181 | 0.865297 | 0.070704 | 18.660222 |
| RestoredReference | 181 | 0.859807 | 0.071494 | 18.601705 |
| PairedReference | 177 | 0.855382 | 0.070853 | 18.921350 |

RestoredReference的181条不一致恢复记录，梯度方向全部为正，cos中位数0.859807；参照范数仅为默认头硬身份梯度的7.149%。乘训练系数0.071458后，中位比例约为 **0.511%**。同组已一致样本的参照/硬身份范数中位数为18.602。E30阶段也有相同量级差异。

这些结果支持“参照方向总体与身份纠正同向，但监督强度分配偏向已经一致的样本”的工作假设，弱化了“方向全面错误”的解释。它们只是特征层局部量，不等于整个backbone/AdamW更新，也不能证明加大权重一定提高源或目标性能。U伪标签仍可能有错，真实U精度未读取。

默认头soft-q与参照方向在容易raw U上可出现负相关，而hard-identity方向仍为正；这说明置信度校准与身份判别需要分开解释。所有软/硬目标、RX、场景及空单元见 [全部指标](source_all_alignment_cells.csv)，重点单元见 [焦点表](source_alignment_focus.csv)、[180批模型视图记录](source_alignment_batches.csv) 和 [独立执行证据](source_alignment_evidence.json)。原始norm来自batch平均loss；CSV另给乘L128/U256后的逐包loss导数范数，余弦与范数比不受该缩放影响。

下一步将先源验证“按默认头身份任务强度重新分配参照监督”，而非全局同时放大易/难样本。该方案尚未训练，旧三变体不扩展，原三种子目标仍未达成。

## 下一目标的实现准备

已实现独立的任务强度缩放损失函数，系数为 `stopgrad(min(32,||g_id||/(||g_ref||+1e-12)))`，固定L参照方向与默认头原联合训练保持。九项基础关系与新数学性质检查通过：已知相同目标退化为原目标、保留参照方向、单包特征梯度范数不超过硬身份任务范数、未接受样本为零，新增分支不向默认头参数写入梯度。原参照loss仅提取逐包项，旧行为回归通过。

这些是函数级检查，尚未完成新函数的真实checkpoint源剂量诊断或新训练发布；不把它们当作性能证据。后续先对普通G参照、任务缩放G参照和raw匹配控制做同批源校准，冻结新共同剂量和矩阵后再训练。
