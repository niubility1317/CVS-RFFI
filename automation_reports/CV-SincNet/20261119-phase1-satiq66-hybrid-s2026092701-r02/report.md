# SatIQ Oxford 66 Hybrid-SSL technical recovery

- run_id：`20261119-phase1-satiq66-hybrid-s2026092701-r02`
- group_id：`phase1-satiq-oxford-cross-time-hybrid`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Replace only U1 after E21 PyTorch 2.1 API failure; assemble three healthy fixed endpoints for coordinated truth-last evaluation

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

## Verified recovery launch

Owner PID 1786971 launched U1 replacement PID 1787274 on GPU 3. The N607 E21 real-data smoke completed one optimizer update and processed 132 L and 256 hidden-U exposures, with zero U above confidence 0.95. The accepted and empty U routes were also exercised locally on controlled synthetic inputs. S0/S1/U0 will be reused solely as frozen prediction endpoints. The watcher invokes the independent scorer after all four predictions are fixed.

The replacement crossed the original failure point and reached E22/2,200 updates. Through E22 it made 230,400 U exposures and accepted none at the fixed threshold. Controlled accepted-U branch verification also passed on the deployed PyTorch 2.1 runtime. The scheduled independent Q readout will run once the replacement reaches the fixed 5,000-update endpoint.

## 完成状态与未来时段对照

**VERIFIED · ANALYZED。** Hybrid-SSL 在 N607 完成 50 轮、5,000 次更新。S0、S1、U0 使用 r01 的原有冻结终点，U1 使用本次修复后的冻结终点。四组均预测了 4 月 24 至 30 日的同一批 293,371 条 IQ，逐条输出 66 类分数。独立评分在全部预测固定后执行。读回四份预测的 ID、分数形状和有限性，并独立复算准确率，均与评分产物一致。远端结果见该运行根目录的 `source_complete.json`、`predictions_complete.json` 和 `truth_last_metrics.json`。

| 方法 | 源验证宏 F1 | 未来正确数 | 未来准确率 | 未来宏 F1 | 未来宏召回 |
|---|---:|---:|---:|---:|---:|
| S0 · PAMF-Sup | 1.4606% | 4,642 | 1.5823% | 1.5104% | 1.5841% |
| S1 · Hybrid-Sup | 1.5270% | 4,480 | 1.5271% | 1.4634% | 1.5420% |
| U0 · Hybrid-Sup + 普通 PL | 1.4505% | 4,578 | 1.5605% | 1.4893% | 1.5679% |
| U1 · Hybrid-SSL | 1.5155% | 4,566 | 1.5564% | 1.4852% | 1.5621% |

均匀随机的 Top-1 和 Top-5 基线分别为 1.5152% 和 7.5758%，四组未来 Top-5 为 7.6436% 至 7.7475%。用每组预测的类别频率和 Q 真值频率计算独立情况下的预期准确率，范围为 1.5066% 至 1.5199%。模型均预测到全部 66 类，最常预测的单类占 2.44% 至 3.01%。未来多数类基线为 2.1829%，每类支持数为 2,003 至 6,404。结果与弱判别一致，且横跨多个日期和检测质量组。

相对 S0，S1 的未来准确率低 0.0552 个百分点，宏 F1 低 0.0470 个百分点，错变对 2,558 条、对变错 2,720 条，7 天中 6 天准确率下降。相对 S1，U0 的宏 F1 高 0.0259 个百分点。相对 U0，U1 的宏 F1 低 0.0040 个百分点，净少识别 12 条。两组半监督实际伪标签损失全程为 0，这些差异无法归因于伪标签。各行共享固定单种子与同一测试集，逐包结果还受日期和采集序列相关性影响。

## 损失是否正常优化

完整解析四行各 50 条轮日志、两个输入试跑、前后两次 U1 以及预测和评分日志。正式完成的四行均有连续的 100 至 5,000 次更新记录，损失有限。仅 r01 的 U1 有已记录的 PyTorch 2.1 接口错误。其替代版本跨过该点并完成。

| 方法 | 基础监督损失，第 1 → 50 轮 | 归一化后的加权平均 CE，第 1 → 50 轮 | 冻结模型在全部 L 上的准确率 |
|---|---:|---:|---:|
| S0 | 9.8926 → 6.1310 | 5.0616 → 3.5455 | 19.94% |
| S1 | 9.8976 → 6.1318 | 5.0627 → 3.5352 | 21.72% |
| U0 | 9.8976 → 6.1430 | 5.0627 → 3.5356 | 21.42% |
| U1 | 9.8976 → 6.1400 | 5.0627 → 3.5395 | 21.42% |

基础损失是 ordinary/raw 的平均 CE、对称 KL 以及权重为 0.68 的物理视图 CE 之和。加权平均 CE 由 `(base - symmetric_kl) / 1.68` 还原。随机初始化在同一 L 上准确率为 1.67%、CE 为 6.07。冻结 S0 至 U1 在 L 上的 CE 为 3.30 至 3.34，参数相对初始值的 L2 改变量为 28.7% 至 29.4%。这些证据确认优化有效降低了训练样本的分类损失。L 和 U 的复算采用本机 GPU、同一表示和冻结权重，源 V 和未来 Q 成绩使用 N607 的正式产物。

S1 的融合与分支同包对比到第 49 轮分别从 5.030 降至 3.445、从 4.145 降至 3.192。同期跨日关系从 4.990 到 5.117，原型损失从 4.411 到 4.478，基本无改善。S1 有 646,800 次同包对比和原型曝光，以及 660,000 次跨日关系曝光。物理扰动在第 50 轮按预定退火归零，相关同包和原型项此时关闭。扰动逐轮变弱也会让同包匹配变容易，因此同包损失下降不能单独证明时间不变的卫星身份表征已形成。四组最佳源 V 宏 F1 均出现在第 40 轮，范围为 1.5425% 至 1.6336%，依然接近随机。正式预测使用预登记的第 50 轮。 [损失实现](../../../experiments/pamf_satiq66/objectives.py)

## 近随机与零伪标签的原因

**已定位到训练样本能部分拟合、分辨能力无法跨日期复现。** S1 在训练池达到 21.72% 准确率，但只隔一个日期边界的 4 月 1 至 3 日 V 上剩 1.56%，未来 Q 为 1.53%。其 Q 的交叉熵为 4.671，仍高于均匀分配 66 类概率所对应的 4.190。分布覆盖全部 66 类，故整体低分不能由预测坍缩到少数类解释。监督目标约从第 10 轮的 6.94 降到第 50 轮的 6.13，验证准确率没有随训练损失同步改善。

**本轮表示限制和类别内变化都是候选解释。** 正式输入为 25 MS/s 下的 256 点短窗，时间长度 10.24 μs，约为 40 μs 的 Iridium 符号周期的四分之一。它针对公共同步字一次过渡，覆盖 8,000 点公共片段的 3.2%。两种输入只在源 V 上比较了各 300 次更新，原始短窗宏 F1 为 0.4946%，长窗降采样版为 0.2097%，均很弱。按规则选中短窗的证据只覆盖该小预算。L 中每个卫星编号实际覆盖 33 至 48 个波束，中位数 41.5，卫星标签可能混合随时间、波束和真实链路变化的观测。现有 66 类运行没有固定其他条件、分别识别短窗和波束混合的独立贡献。另一组 20 类站间实验用局部 logit 均值覆盖全部 11,000 点，仍然较弱，说明延长输入也未必充分。 [本轮输入设计](../../../docs/PAMF_SATIQ_HYBRID_SINGLE_SOURCE_20261119.md) · [20 类截段和完整包头对照](../20261119-phase1-pamf-satiq-header-s2026111902-r01/report.md)

| 半监督组 | E14–E50 无标签曝光次数 | 通过 0.95 | 冻结模型 U 置信度中位数 | P95 | P99 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| U0 · 普通 PL | 947,200 | 0 | 8.15% | 20.13% | 31.27% | 85.37% |
| U1 · Hybrid-SSL | 947,200 | 0 | 8.10% | 20.33% | 31.82% | 85.22% |

曝光计入重复采样。回放覆盖全部 95,040 条唯一源 U，U 的标签全程隐藏。训练首先筛选 66 类 softmax 的最大概率。U1 的五分支认证、物理对比和无标签原型都位于 0.95 筛选之后，因此零接收已经发生在投票之前。U1 从 E21 起正常经过保护和路由代码，任何 U 侧损失及有效配对均为 0。后续实查确认该图没有 BatchNorm。U 前向仍经过训练态 Dropout，随机状态和 GPU 数值差异可能使 U0、U1 和 S1 的轨迹有所不同。阈值保持预登记值。 [伪标签筛选与更新](../../../experiments/pamf_satiq66/train.py) · [后续训练链路检查](../20261119-phase1-satiq66-long-sup-s2026092701-r01/report.md)

原始日志、冻结权重、评分原件和只读诊断摘要保存在 `E:/type10-7/local_data/satiq66_20261119_r01/completion_20261119/`。本轮结论限于这套单站跨日 66 类设置、固定输入及单模型种子。
