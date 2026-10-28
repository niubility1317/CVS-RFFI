# PAMF Phase3：完整六类冻结源 OE 拒识结果

**结论：技术流程完整，科学结果为 `NO_PROMOTION`。** 当前完整六类 Tri-SIG-TIME SSL PAMF 保持冻结；源成员监督读出将辅助未知设备宏拒识率提高到 linear 68.4762%、MLP16 76.6032%，但整体与多数源 RX 均未达到预定95%。两种OE方法的旧类保护均通过；本次未读正式target池，不代表跨RX、星地、无标签适应或注册达标。

- run：`20261029-phase3-source-open-readout-s2026092701-r01`；row：`source-open-readout-s2026092701`。
- 实际code/config commit：`76951634c0901175a6d78c88caffbf49fdce8d04`。
- 状态：`VERIFIED / ANALYZED_SOURCE_ONLY`；`eligible_methods=[]`、`selected_method=null`、`final_phase3_complete=false`。
- 本报告完整读取4份执行日志与两条各200步损失，解析全部7方法、5RX、3未知TX及RX×TX结果；未重跑、扫参或修改代码／阈值。

## 基座、数据和权限

基座是当前三视图、固定时间重排的完整六类PAMF最终E50，232993参数；不是四类CE20代理。六类默认身份头始终原样冻结，只训练新增linear_oe（101参数）和mlp16_oe（1633参数）。这两个读出使用额外的“源旧六类／源非旧类”成员监督，没有辅助样本逐TX标签；不能称零源标签或目标端监督。原F自身合法SSL继承保持，本次不重新打开原U_s。

原source L6300只拟合8分支源几何；L6300＋辅助训练4200只拟合描述符normalizer和两个拒识读出。source V27000仅在两读出固定200步后，以99% retention、1% lower分位冻结每方法一个全局阈值。辅助验证3150来自与训练暴露TX互斥的3台源新设备（19-15、19-16、7-1），每台1050条；它们不参与几何、normalizer、梯度、阈值或早停。

数据均为原5源RX（1-19、18-2、19-2、2-19、3-19）、D1–D3的clean/eq1源端数据，辅助类别与旧六类、四compact TX以及正式四个unknown隔离。每RX已知5400条、辅助未知630条，每未知TX在每RX210条。5个RX均已属于原F源域；这项验证是未参与读出训练的TX泛化，不是留出接收机域泛化。

真实checkpoint smoke先在2条原L完成PASS，6类logits [2,6]、全部8分支完整；随后40650条四role完整导出，合计40652次冻结F前向。export原U_s、OE私有truth、target读取均为0；fit的V／novel validation／OEtruth读取为0；encoder和身份head更新为0。预测仅knownV校阈值，阈值完成标记先于novel读取；30150条×7方法全部固定后才独立truth-last评分。正式RX24-18的U768/Q1920本轮未读取、推理或评分。

## 固定200步的完整损失记录

损失为平衡成员softplus：0.5 mean softplus(-g_known)＋0.5 mean softplus(g_OE)。CPU AdamW lr0.001、wd0.0001、betas(0.9,0.999)、eps1e-8；每方法200次全批量6300＋4200特征更新，无V早停、无dropout。

| 方法 | 步数 | step1 | step50 | step100 | step150 | step200 | 上升步数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear_oe | 200 | 0.693147 | 0.334978 | 0.250089 | 0.210037 | 0.186006 | 0 |
| mlp16_oe | 200 | 1.140048 | 0.296108 | 0.198696 | 0.145002 | 0.116537 | 0 |

400个记录均有限，两条曲线全程单调下降。最后50步仍分别下降0.024031和0.028465，因此只能确认优化稳定、固定预算完成，不能称完全收敛。记录口径是每次更新前的训练loss，step200不是额外重算的更新后最终loss。没有逐步验证loss或训练分类准确率，不能据此确认／排除过拟合，也不能用训练loss替代独立新TX拒识。

## 七方法整体结果

旧六类闭集宏准确率统一为98.4667%（26586/27000）。全部方法共享原完整head的类别决策；只改变拒识。未知TX等权宏平均，旧类损失也采用宏准确率差。

| 方法 | 未知宏拒识 | 旧类开放宏准确率 | 旧类宏FRR | 较闭集损失/pp | AUROC | OSCR |
|---|---:|---:|---:|---:|---:|---:|
| msp | 32.5714% | 98.1556% | 0.9963% | 0.3111 | 0.974370 | 0.968310 |
| maxlogit | 35.2698% | 98.1704% | 0.9963% | 0.2963 | 0.963066 | 0.956431 |
| margin | 28.6984% | 98.0630% | 0.9963% | 0.4037 | 0.971843 | 0.965872 |
| energy | 37.8730% | 98.1481% | 0.9963% | 0.3185 | 0.962793 | 0.955942 |
| cosine | 40.2857% | 97.7111% | 0.9963% | 0.7556 | 0.968458 | 0.959647 |
| linear_oe | 68.4762% | 97.9148% | 0.9963% | 0.5519 | 0.985043 | 0.973137 |
| mlp16_oe | 76.6032% | 97.8852% | 0.9963% | 0.5815 | 0.990611 | 0.977672 |

两个OE方法均误拒269/27000条已知（0.9963%）；其中linear损失149条原本正确身份，MLP损失157条。它们的整体旧类损失为0.5519pp、0.5815pp，各RX也均≤2pp且FRR≤5%，旧类保护通过。当前“99%保留”的离散lower分位实际保留99.0037%，不要求恰好270条被拒。

静态控制中本次未知拒识最高是cosine的40.2857%；linear、MLP分别高28.1905pp与36.3175pp，MLP较linear高8.1270pp。这是同一源验证集上的描述性比较，OE增加了4200条源成员监督与训练开销；不把它称为相同标签预算的免费提升，也不据此替换已固定方法。AUROC最高0.990611仍不等价于固定操作点达到95%拒识。

## 全部源RX与未知TX结果

各RX未知宏拒识如下；每格都是3个TX等权，不按已知／未知样本数混平均。

| Main RX／名称 | MSP | Maxlogit | Margin | Energy | Cosine | Linear OE | MLP16 OE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 / 1-19 | 49.0476% | 56.3492% | 39.0476% | 58.4127% | 43.8095% | 96.9841% | 98.8889% |
| 3 / 18-2 | 36.1905% | 41.5873% | 33.3333% | 45.3968% | 45.7143% | 45.0794% | 66.0317% |
| 4 / 19-2 | 14.6032% | 13.1746% | 16.0317% | 13.4921% | 22.2222% | 81.2698% | 83.1746% |
| 6 / 2-19 | 38.5714% | 37.9365% | 35.0794% | 40.7937% | 49.5238% | 88.4127% | 87.9365% |
| 8 / 3-19 | 24.4444% | 27.3016% | 20.0000% | 31.2698% | 40.1587% | 30.6349% | 46.9841% |

两个OE方法只有RX1的宏拒识超过95%；RX3、RX4、RX6、RX8均失败。最弱是RX8（3-19）：30.6349%／46.9841%；其次RX3（18-2）：45.0794%／66.0317%。MLP的整体提升并不均匀，RX6还从88.4127%降至87.9365%。这些是已见source RX的差异，不是新target反馈。

| RX／名称 | Linear已知开放宏acc | Linear FRR | Linear损失/pp | MLP已知开放宏acc | MLP FRR | MLP损失/pp |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 1-19 | 99.4074% | 0.1481% | 0.0926 | 99.3889% | 0.1481% | 0.1111 |
| 3 / 18-2 | 94.4630% | 2.5000% | 1.1667 | 94.6667% | 2.2037% | 0.9630 |
| 4 / 19-2 | 98.2963% | 0.5926% | 0.1852 | 98.2778% | 0.5370% | 0.2037 |
| 6 / 2-19 | 99.0370% | 0.7037% | 0.6296 | 98.6481% | 1.0926% | 1.0185 |
| 8 / 3-19 | 98.3704% | 1.0370% | 0.6852 | 98.4444% | 1.0000% | 0.6111 |

五种静态控制的完整逐RX保护和置信区间见results.json及机器摘要。特别cosine在RX3的旧类损失为3.3148pp，整体看似合格不能掩盖该分组超限。

| 验证TX（每台1050） | MSP | Maxlogit | Margin | Energy | Cosine | Linear OE | MLP16 OE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 19-15 | 32.8571% | 31.4286% | 29.6190% | 32.7619% | 69.3333% | 61.6190% | 70.9524% |
| 19-16 | 45.3333% | 58.3810% | 38.0952% | 64.4762% | 0.1905% | 66.3810% | 70.3810% |
| 7-1 | 19.5238% | 16.0000% | 18.3810% | 16.3810% | 51.3333% | 77.4286% | 88.4762% |

MLP三个未知TX的整体拒识分别为70.9524%、70.3810%、88.4762%，没有一台达到95%；其95% Wilson区间分别为68.1340%–73.6180%、67.5491%–73.0643%、86.4030%–90.2689%。linear对应61.6190%、66.3810%、77.4286%，区间为58.6403%–64.5131%、63.4684%–69.1740%、74.8026%–79.8546%。区间单位是received source record；不把单TX或pooled区间称为宏均值区间，不把相关记录当独立实验重复。

两个OE的完整RX×TX拒识矩阵如下，每格分母210：

| RX／名称 | TX | Linear拒识条数／率 | MLP拒识条数／率 |
|---|---|---:|---:|
| 1 / 1-19 | 19-15 | 210/210 / 100.0000% | 208/210 / 99.0476% |
| 1 / 1-19 | 19-16 | 208/210 / 99.0476% | 208/210 / 99.0476% |
| 1 / 1-19 | 7-1 | 193/210 / 91.9048% | 207/210 / 98.5714% |
| 3 / 18-2 | 19-15 | 80/210 / 38.0952% | 127/210 / 60.4762% |
| 3 / 18-2 | 19-16 | 89/210 / 42.3810% | 124/210 / 59.0476% |
| 3 / 18-2 | 7-1 | 115/210 / 54.7619% | 165/210 / 78.5714% |
| 4 / 19-2 | 19-15 | 119/210 / 56.6667% | 136/210 / 64.7619% |
| 4 / 19-2 | 19-16 | 187/210 / 89.0476% | 179/210 / 85.2381% |
| 4 / 19-2 | 7-1 | 206/210 / 98.0952% | 209/210 / 99.5238% |
| 6 / 2-19 | 19-15 | 209/210 / 99.5238% | 210/210 / 100.0000% |
| 6 / 2-19 | 19-16 | 176/210 / 83.8095% | 164/210 / 78.0952% |
| 6 / 2-19 | 7-1 | 172/210 / 81.9048% | 180/210 / 85.7143% |
| 8 / 3-19 | 19-15 | 29/210 / 13.8095% | 64/210 / 30.4762% |
| 8 / 3-19 | 19-16 | 37/210 / 17.6190% | 64/210 / 30.4762% |
| 8 / 3-19 | 7-1 | 127/210 / 60.4762% | 168/210 / 80.0000% |

RX8上的19-15和19-16尤其薄弱：linear只有13.8095%和17.6190%，MLP均为30.4762%。RX3同两TX也分别只有38.0952%／42.3810%及60.4762%／59.0476%。不能通过剔除这两个设备或只报RX1宣布达标。

| 旧TX（每台4500） | 原closed acc | Linear开放acc | Linear FRR | MLP开放acc | MLP FRR |
|---|---:|---:|---:|---:|---:|
| 14-10 | 98.6222% | 98.2667% | 0.8222% | 98.4444% | 0.4889% |
| 14-7 | 98.4444% | 98.2667% | 0.6889% | 98.3111% | 0.6889% |
| 20-15 | 99.1333% | 98.9333% | 0.4222% | 98.6889% | 0.6889% |
| 20-19 | 96.6000% | 96.4667% | 0.7778% | 96.4889% | 0.7333% |
| 6-15 | 98.7556% | 96.4000% | 2.7556% | 96.2444% | 2.9111% |
| 8-20 | 99.2444% | 99.1556% | 0.5111% | 99.1333% | 0.4667% |

旧类6-15承担最多误拒：linear 2.7556%、MLP 2.9111%。表中逐类结果用于完整损伤说明；本次推进规则仍是事前固定的整体及每RX宏保护，不追加事后门槛。

## 决策、执行与下一步

两OE方法的失败检查完全一致：整体以及RX3/4/6/8的unknown宏拒识不足95%；其余已知保护、类别覆盖与非全拒检查均通过。5个静态控制不能触发OE候选推进。故没有选中方法、没有扩seed、没有访问正式Full池；本候选是科学门槛未通过，非技术失败。

完整pipeline用时16.732910 s：export 10.723000 s、fit 3.870515 s、predict 1.788136 s、score 0.348874 s；fit内部记录3.204812 s。仅导出使用GPU2，其余CPU；foreign_processes_stopped=0。独立post_state显示driver和子进程已退出、四job COMPLETE、failure=null。

四份完整日志共26行、2119字节，无Traceback、ERROR、OOM、Killed、NaN/Inf或Warning；训练记录中400个loss均有限。发布前64项测试＋35项subtests通过，一次独立P0/P1审查PASS且独立重跑同64＋35；发布前“smoke尚未执行”快照由本次真实export/smoke PASS完成。科学NO_PROMOTION没有被记为技术中断。

可以考虑的下一步均是建议，尚未实施：首先保留完整六类强基座，针对源RX3/8的拒识尾部与跨分支“是否支持同一个旧类”一致性设计新source-only候选，而不是仅扩大MLP；保持类置换不变、默认头冻结及独立源数据角色。其次用源RX留出／分组检查判断新读出是否学习了接收机相关捷径，现有结果只能提示这种风险，不能证明具体机制。任何新的操作点、目标或训练预算须另行事前固定；本次不改阈值、不延长200步、不把辅助验证改训练，也不打开正式target来选方案。

较低训练loss和更高AUROC说明本读出能拟合已开放的源成员区分，并不能证明对未暴露设备、未见RX或星地条件稳定。当前源拒识尚未通过95%，无标签适应、未知聚类、匿名注册和持续unknown拒识均未在本候选执行，Phase3总目标继续未完成。

## 完整证据入口

- [冻结方案](E:/type10-7/docs/PAMF_PHASE3_SOURCE_OE_REJECTION_PLAN_20261029.md)；[source-OE数据身份](E:/type10-7/docs/PAMF_PHASE3_SOURCE_OPEN_DATA_20261029.md)。
- 核心：[score/results.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/score/results.json)；[fit/training.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/fit/training.json)；[analysis_summary.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/analysis_summary.json)。
- 完成／独立后态：[pipeline_state.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/pipeline_state.json)；[post_state.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/post_state.json)；[landing_readback.txt](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/landing_readback.txt)。
- 四完整日志：[logs/export.log](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/logs/export.log)；[logs/fit.log](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/logs/fit.log)；[logs/predict.log](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/logs/predict.log)；[logs/score.log](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/logs/score.log)。
- 完成标记：[export/smoke.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/export/smoke.json)；[export/complete.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/export/complete.json)；[fit/complete.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/fit/complete.json)；[predict/calibration/complete.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/predict/calibration/complete.json)；[predict/complete.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/predict/complete.json)；[score/complete.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/score/complete.json)。
- 配置／验证：[configs/export.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/configs/export.json)；[runtime.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/runtime.json)；[local_verification.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/local_verification.json)；[focused_tests.log](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/focused_tests.log)；[package.json](E:/type10-7/local_artifacts/pamf_phase3_source_open_readout_20261029_r01/package.json)。
- [run登记](E:/type10-7/automation_reports/CV-SincNet/20261029-phase3-source-open-readout-s2026092701-r01/report.md)；[experiment.json](E:/type10-7/automation_reports/CV-SincNet/20261029-phase3-source-open-readout-s2026092701-r01/experiment.json)；[events.jsonl](E:/type10-7/automation_reports/CV-SincNet/20261029-phase3-source-open-readout-s2026092701-r01/events.jsonl)。
- N607原始root：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-source-open-readout-s2026092701-r01`。完整特征／几何／读出／阈值／预测NPZ保持原root；本次分析不重开这些数组。
