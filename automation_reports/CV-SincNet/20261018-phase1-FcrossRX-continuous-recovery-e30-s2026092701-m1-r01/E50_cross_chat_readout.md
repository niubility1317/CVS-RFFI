# 原完整 As 的 E50 单种子正式读出（完整分析已核实）

引用实验 `20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01`，唯一owner为引用对话。两行训练、全部七视图冻结预测、独立truth-last评分与全日志分析均完成，无failure；独立分析状态 `ANALYZED_VERIFIED`。dispatcher 的历史静态状态不替代分析完成证据。本对话只读取保存汇总，没有新增训练、预测、rawtruth或NN。

相对同预算同seed NoAs50，As50 clean **+0.365476pp**，urban均值 **+30.707738pp**，low **+22.508929pp**，mid **+29.864881pp**。满足clean下降不超过0.1绝对百分点；这是单模型种子2026092701的双提升观察，不是跨seed可重复证据。low52.3875%仍未达60%。

## 全部固定正式环境

| 环境 | NoAs50准确率% | As50准确率% | 配对变化pp | As50宏F1% | As50最差RX% |
| --- | --- | --- | --- | --- | --- |
| clean | 85.295 | 85.661 | +0.365 | 85.381 | 79.025 |
| practical_high | 39.952 | 82.239 | +42.287 | 82.031 | 72.279 |
| practical_mid | 39.140 | 79.557 | +40.416 | 79.342 | 70.288 |
| practical_low_suburban | 38.218 | 76.776 | +38.558 | 76.524 | 67.625 |
| practical_high_urban | 38.994 | 78.743 | +39.749 | 78.521 | 69.708 |
| practical_mid_urban | 34.037 | 63.902 | +29.865 | 63.536 | 57.246 |
| practical_low_urban | 29.879 | 52.387 | +22.509 | 51.849 | 47.233 |
| urban_mean | 34.303 | 65.011 | +30.708 | — | — |

## 预算增量与归因

固定F/跨RX同TX基座、默认cos头、完整practical配方、原L/U与所有源RX不变。每行自身原20+10父模型追加20，累计50/11100更新；严格自身student+EMA，重置AdamW/cos20/段计数与缓存。不是从零global-cos50，也不是本对话连续E30模型的扩展。

As50相对自身As30：clean **+0.319643pp**，mid **+1.095238pp**，low **+1.143452pp**，urban均值 **+1.071627pp**。NoAs50相对自身NoAs30：clean **+0.047619pp**，urban均值 **−1.267659pp**。因此urban配对预算交互 **+2.339286pp** 同时含As改善和NoAs退化，不全部归因As学习改善。

源域先于新正式汇总读出：固定V27000下As50−NoAs50 clean−.074074pp；As50相对自身As30 fullVclean+.118519pp、mid/low均值+1.805556pp。两固定draw仍只有1800个独立物理样本，不是两个模型seed。

新增2×20=40模型epoch/8880更新；复用2×30父成本60模型epoch/13320更新；两行累计100模型epoch/22200更新。原E30父模型也在连续E30报告作为参照，两个报告的累计成本不可直接相加。完整新增日志共8880个step、40个epoch，物理序列及普通增强MSE对齐无缺失；504个正式混淆矩阵和84个RX/day/TX分区重算一致。每新增行student-train／student-eval／teacher-eval分别4,543,920／594,002／1,134,000 NN，原生合计6,271,922 NN。两新增行连同源域／正式读出和共享smoke共14,922,586 NN；两父行历史18,298,572 NN单列，累计33,221,158 NN。新增NoAs／As训练段wall分别1602.126／2069.076 s，正式预测47.669／46.841 s。不能把并行wall差异当成算法加速。

As50相对NoAs50，clean修复4153包、新增错误3539包，净修复614包；low urban修复46642包、新增错误8827包。相对As30，low suburban绝对准确率提高0.914286pp，但最差RX下降0.116667pp，不能声称每个分层都改善。

## 边界与后续

当前完成结果只有一个seed，未获得可重复双提升证据。依据先于正式读出的源域预算收益，以及用户允许好配方总E50和原定三seed的授权，引用对话已登记固定02／03各自As／NoAs追加20的四行复现；新run为 `20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01`，唯一owner为 `codex/root/practical-budget50-replication`。截至本次核对为PLANNED_NOT_LAUNCHED。本对话尚未登记的重复实现已停止并归档，新增真实训练／预测／checkpoint读取均为0，不再另开同矩阵。原单seed run的automatic_seed_extension=false保持。低urban未达60%；三个suburban与clean仍有差距。当前结果不改变已先关闭的连续E30路线，不回流调参、选模或选择性重跑。

实际信道口径另经原始元数据核实：训练 `source/actual_channel.json` 的fc为2.462GHz；固定正式 `received_views/manifest.json` 六场景以及固定源面板 `observations/actual_engine.json` 的fc均为2.45GHz，fs均25MHz。源面板EQ=false、post_sync、reference_snr_db=30。这纠正了引用报告“训练／正式2.462GHz”的文字；没有修改信道、IQ或运行配置，也不据此推断性能原因。

保存读出：[独立评分汇总](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/E50_automatic_summary_observed.json>)；[先于正式读出的源域汇总](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/E50_source_before_formal_summary.json>)；[原owner登记与报告](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01/report.md>)。
