# SDG 作者 WiSig 复现结果

**作者 WiSig 公开例程复现 VERIFIED：三个目标日的准确率与作者日志逐日一致，目标均值93.2444%。**运行 `20261019-phase1-sdg-author-wisig-s2023-r01` 完成200轮训练、源域选模、18600条冻结预测及独立truth-last评分，已取回产物复算。完整论文实验表仍未核准，`paper_effect_verified=false`。

## 固定实现与数据

论文为 *Avoiding Shortcuts: Enhancing Channel-Robust Specific Emitter Identification via Single-Source Domain Generalization*，IEEE TWC2025，DOI [10.1109/TWC.2025.3528568](https://doi.org/10.1109/TWC.2025.3528568)。[作者代码](https://github.com/BeechburgPieStar/SDG-for-Channel-Robust-SEI)固定提交`0090f4ba1d5cf76a4200a5be4830a9b08eb39699`的`2TWC_Version`，本次执行代码commit为`10bbde1396214177fc490d02c785291ce98c35c8`；[预登记](SDG_AUTHOR_WISIG_PREREG_20261019.md)在首次目标预测前固定。

使用作者公开的4个RX1-1日期pickle，6TX各1000包、256点。TX顺序为14-10、14-7、20-15、20-19、6-15、8-20。保留作者float64均功率归一化和原类顺序；作者文件的均衡状态未说明。day1为源，原两次seed2023划分得到train3780/val1620/source-holdout600；day2/3/4各6000包为目标。模型seed2023，匿名导出顺序seed20261019。

S MACNN共129894参数，原6类头与骨干联合训练；主/辅助增强深度2/1，类别/域对比损失权重1/100。直接调用原训练、验证、7种增强和损失函数，保留训练前torchsummary对RNG和BN的影响。Adam0.001、weight_decay0、batch32；固定200轮，无早停或调度，源验证NLL最小选模。没有作者权重或其他run父模型，没有目标拟合或另训头。

## 结果与作者日志

| 数据角色/日期 | 样本数 | 本次准确率 | 作者日志准确率 | 差值（百分点） | 本次Macro-F1 |
|---|---:|---:|---:|---:|---:|
| 源holdout day1 | 600 | 99.5000% | 99.5000% | 0.0000 | 99.5037% |
| 目标day2 | 6000 | 99.2167% | 99.2167% | 0.0000 | 99.2180% |
| 目标day3 | 6000 | 83.2667% | 83.2667% | 0.0000 | 79.6001% |
| 目标day4 | 6000 | 97.2500% | 97.2500% | 0.0000 | 97.2247% |
| 目标合并/准确率均值 | 18000 | **93.2444%** | **93.2444%** | **0.0000** | **93.1334%** |

目标各日样本数相同，三日准确率算术均值与合并准确率一致。表末Macro-F1为18000条合并后计算；三日Macro-F1先算后平均为92.0143%，两种口径分别保留。源holdout不混入目标平均数。

结果达到首次预测前固定的92.2444%工程下限。结论为“作者公开WiSig代码/数据/日志例程复现达到参照”，一致性范围是公开指标及loss打印精度；未将其追认为已核准论文全部表格、ORACLE实验或多seed稳定性。

day3保留了作者日志中的弱项：TX8-20的1000包中，893包被判为TX20-15，正确106包，召回率10.6%。因此93.24%目标均值不能解释为每类、每日期均稳定；这项负结果完整保留，未据此调参或重跑。

## 完整200轮核查

完整结构化epoch记录覆盖1至200，无缺轮；源NLL选择第191轮，最小NLL为0.0015611589748274397，与预测marker所用checkpoint轮次一致。200轮训练/验证计时累计1511.8349s（约25.20分钟），不含发布、数据读取、烟测、预测与评分，不作跨硬件加速比较。

已完整解析作者公开`2TWC_Version/log/WiSig`与本次200轮记录：所有训练loss在作者6位小数打印精度一致，所有源验证NLL在作者4位小数打印精度一致。源holdout及3个目标日准确率也全部一致。没有用目标指标选epoch或更改预算。

完整stdout未见Traceback、OOM、Killed或非有限loss。存在作者原实现的NumPy变长分段弃用警告及偶数卷积核padding警告；本次采用独占release的NumPy1.23.5保留原行为，未改增强公式或共享环境。

## 评分证据与结论范围

真实source checkpoint无query烟测通过，正式训练重新从零开始。200轮完成后冻结源NLL最佳模型，再准备匿名IQ/ID；预测器只用原头独立argmax。18600条prediction固定后，独立scorer才连接truth-side文件。

全部18600个物理包ID唯一。取回后独立重算4个日期的accuracy、Macro-F1和完整混淆矩阵，并核对物理RX/day/TX/row、匿名ID连接、18000条目标汇总和源选模轮次，逐值一致。证据在私有 `local_artifacts/rffi_dg_research_20261019/20261019-phase1-sdg-author-wisig-s2023-r01/verified_summary.json`、`retrieved/`及`author_public_full_log.txt`。数据、权重、预测和原始日志不进入Git。

这是单源跨天clean DG，不能作为跨接收机、CVS弱标签或星地部署成绩。IEEE[官方正文](https://ieeexplore.ieee.org/document/10847785)目前要求订阅访问，全文表格尚未核准；作者代码例程成功与严格论文表验收分别报告。
