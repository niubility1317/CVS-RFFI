# Phase1非PAMF：拼接星地增强监督/普通PL三种子对比

- run_id：`20261019-phase1-external-dg-concat-pl-m219-r03`
- group_id：`phase1-nonpamf-concat-supervised-plain-pl`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

219次从零训练：192增强监督+27同时保留增强的普通PL；默认头、原生预算、七视图truth-last。

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

## 本次发布预登记

完整数据、接收机数量与预算依据见 [原执行设计](../../../docs/PHASE1_EXTERNAL_DG_EXECUTION_PLAN_20261019.md) 及 [当前增强与PL修订](../../../docs/PHASE1_DG_AUG_PL_REVISION_20261019.md)。当前矩阵由 experiments/phase1_external_dg_aug/matrix.json 固定；正式训练从零开始，不继承无增强历史权重。

所有正式训练均使用 clean＋satellite 拼接和 0.68 卫星 CE；PL 行在同一次训练额外使用阈值 0.95、权重 1、U/L=1 的普通 PL。监督与 PL 共用相同增强、模型种子、最大预算和源域选模规则。

RIEI/DRIFT E100（用户指定实现预算，阶段按比例缩放）；IB-RFF 两阶段各最多 E200、源域降学习率及早停；SDG E200；通用 DG 固定 5000 次更新；源数量固定预算组 10000 次更新；强 ResNet CE E200。IB 早停以实际执行轮数报告。训练首批启用卫星 CE，默认场景概率按相对最大预算推进。

仅物理 GPU 2、3，每卡一个本任务训练/预测 worker，其他 owner 占用时等待。每行全部七视图预测固定后独立 CPU truth-last 评分。两次相同技术故障暂停未启动行，健康行继续；低分、负收益或零 PL 接受率均不触发重跑、调参或停跑。

针对性检查：8 项通过；独立 P0/P1 审查 PASS，唯一 stages 字段缺陷已修复并完成针对性复核。实际 Git commit、发布包、运行路径和提交回执在发布后登记，真实 checkpoint 无 query 联合目标 smoke 为启动第一步，PASS 后立即继续。


用户最新指定使用另外两张空闲卡，已选物理GPU2、3；总数仍两卡。与r01逐行rows/protocols/views完全相同，仅run和资源设置变化。原独立P0/P1审查和针对性修复结论适用于未改动的方法实现；修改卡号后的8项针对性检查再次通过。


## 技术恢复边界

r02零GPU训练行，训练前数值错误原因未确定，诊断重放33项PASS。此run保持原219行所有科学配置不变，只增加异常定位后进行一次技术恢复。实际启动仍先执行真实源checkpoint无query smoke，失败则停止；不跳过finite检查，不调整模型、损失、seed或学习率。物理GPU2、3，总数两卡。

## 实际发布核实（2026-10-19T14:44:55+08:00）

VERIFIED：代码commit `cb9e19fbbbfee31704064be0ad4f6ec96d29b966`，远端分支OID独立核对一致；一次发布包传输MATCH、远端编译PASS，启动第一步33项真实源checkpoint无query联合目标与严格重载smoke全部通过。

实际训练位于物理GPU2、3，每卡一行，PID/CWD/argv与启动登记一致，完整源epoch记录及日志均持续增长。当前首批IB-RFF CE监督与CE＋PL配对行都记录了实际卫星增强样本，PL行同时记录U曝光；U缓存明确不导出TX标签，阈值0.95、默认头联合训练已核实。当前只是开始运行，未声称219性能行或96成本行完成；r02首次数值失败原因仍未确定，记录保留。

性能run：`20261019-phase1-external-dg-concat-pl-m219-r03`；成本run：`20261019-phase1-external-dg-concat-pl-cost-m96-r03`。实际私有运行路径、PID及日志证据位于本run runtime/release_result/postlaunch_verified 引用，不将私有运行配置或原始日志提交Git。成本队列等待全部性能行完成及进程实际退出。

## 首个完整四条件配对

[IB-RFF首seed增强与PL结果](ibrff_seed01_aug_pl_analysis.md)：4行七视图独立复算通过，只有seed2026092701。报告保留监督条件机制负差及PL条件机制正差，具体数字、实际早停轮数和全部场景在子报告；不据分数调整后续矩阵。整体仍RUNNING，219行与96成本未全部完成。

## IB-RFF主划分三seed完整结果

[IB-RFF增强监督与PL完整配对](ibrff_aug_pl_analysis.md)：12行七视图及全部分层CM独立复算通过，原生预算/源曲线/default-head核验齐全。负结果全部保留，后续固定矩阵继续运行，整体目标未完成。

## SDG CE 首种子监督／PL配对

VERIFIED：seed 2026092701 的 CE 监督与 CE＋PL 均完成 E200，默认头及完整源曲线核验通过，7视图独立复算共 588 个分层CM一致。全部条件保留默认拼接增强；当前仅1个seed，不能代替三seed结论。

| 视图 | 监督 Accuracy（%） | PL Accuracy（%） | PL−监督（百分点） | 监督 Macro-F1（%） | PL Macro-F1（%） |
|---|---:|---:|---:|---:|---:|
| clean | 72.8637 | 71.5851 | -1.2786 | 72.3386 | 71.7140 |
| practical_high | 42.9708 | 48.3268 | +5.3560 | 38.7514 | 41.3546 |
| practical_mid | 42.5202 | 47.1149 | +4.5946 | 38.8116 | 40.2923 |
| practical_low_suburban | 41.9280 | 45.9077 | +3.9798 | 38.5531 | 39.0563 |
| practical_high_urban | 42.0482 | 46.8137 | +4.7655 | 38.3105 | 39.9196 |
| practical_mid_urban | 37.9476 | 40.1173 | +2.1696 | 35.8745 | 33.7281 |
| practical_low_urban | 33.7958 | 34.6429 | +0.8470 | 32.4149 | 28.6628 |

按源验证 NLL 冻结，监督／PL 选中 epoch 分别为 101／66；两行均实际训练满200轮。监督不读取U，PL真值隐藏，阈值0.95、权重1、U/L=1，没有EMA或另训头。PL的clean负差如实保留，不据此关闭机制、调阈值或重跑。完整SDG监督/PL两行已在物理GPU2、3接续，整体14/219行完成，96成本尚未测量。

逐行独立复算与完整曲线证据：`local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/verified_rows/`；配对汇总：`local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/sdg_ce_seed01_pair_summary.json`。

## SDG首个完整四条件配对

[SDG首seed增强与PL结果](sdg_seed01_aug_pl_analysis.md)：4行七视图独立复算通过，只有seed2026092701。报告保留全部四条件的原始配对差，具体数字、实际早停轮数和全部场景在子报告；不据分数调整后续矩阵。整体仍RUNNING，219行与96成本未全部完成。

最新实际进度：**16/219行完成并独立复算**。SDG首seed四条件共1176个分层CM一致；clean原生组合−CE为监督+0.6339、PL+0.5851个百分点，PL clean两模型均下降。完整组合在本seed六种practical的Accuracy相对CE均为正；不把单seed或单指标结果冒充三seed结论。第二seed CE监督/PL已在GPU2、3接续，96成本仍等待性能全矩阵完成。

## SDG CE 第二种子监督／PL 配对

VERIFIED：seed 2026092702 的 CE 监督与 CE＋PL 均完成 E200，默认头及完整源曲线核验通过，7 视图独立复算共 588 个分层 CM 一致。全部条件保留默认拼接增强；本表仅该 seed，不能代替三 seed 结论。

| 视图 | 监督 Accuracy（%） | PL Accuracy（%） | PL−监督（百分点） | 监督 Macro-F1（%） | PL Macro-F1（%） |
|---|---:|---:|---:|---:|---:|
| clean | 73.6500 | 69.4714 | -4.1786 | 72.8879 | 67.8861 |
| practical_high | 30.3673 | 38.7190 | +8.3518 | 28.9815 | 37.6893 |
| practical_mid | 30.4643 | 37.7101 | +7.2458 | 29.2880 | 36.8528 |
| practical_low_suburban | 30.1452 | 36.5744 | +6.4292 | 28.9485 | 35.7552 |
| practical_high_urban | 30.4113 | 38.1131 | +7.7018 | 29.3385 | 37.1865 |
| practical_mid_urban | 29.8179 | 34.2393 | +4.4214 | 28.8436 | 33.3776 |
| practical_low_urban | 27.9940 | 30.5042 | +2.5101 | 26.7691 | 29.3335 |

按源验证 NLL 冻结，监督／PL 选中 epoch 分别为 77／55；两行均实际训练满 200 轮。监督不读取 U，PL 真值隐藏，阈值 0.95、权重 1、U/L=1，没有 EMA 或另训头。PL 的正负差如实保留，不据此关闭机制、调阈值或重跑。完整 SDG 监督/PL 两行已在物理 GPU2、3 接续，整体 18/219 行完成，96 成本尚未测量。

逐行独立复算与完整曲线证据：`local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/verified_rows/`；配对汇总：`local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/sdg_ce_seed02_pair_summary.json`。

## SDG第二个完整四条件配对

[SDG第二seed增强与PL结果](sdg_seed02_aug_pl_analysis.md)：4行七视图独立复算通过，只有seed2026092702。报告保留全部四条件的原始配对差，具体数字、实际早停轮数和全部场景在子报告；不据分数调整后续矩阵。整体仍RUNNING，219行与96成本未全部完成。

最新实际进度：**20/219 行完成并独立复算**。SDG 第二 seed 四条件共1176个分层CM一致；clean完整−CE为监督−3.5292、PL+2.9994个百分点，PL−监督在CE为−4.1786、完整SDG为+2.3500个百分点。所有负差保留；第三模型seed CE监督/PL已在GPU2、3接续，三seed家族结论和96行成本尚未完成。


## 2026-10-19 20:52：完成时间估算与低频查看

用户要求大批实验期间减少查询，并估算完成时间。最新计时证据为 50/219 行完成、2 行运行。两卡持续可用条件下，全套训练、七视图测试与评分、96 项成本及最终核验中心估计约剩余 18.1 小时，预计 10 月 20 日 15:00；敏感性窗口约 12:00 至 22:00（UTC+8），不构成保证。改为约每 30 分钟查看，阶段完成、异常或 ETA 明显偏移时汇报。计算口径、未知部分及更新规则见 [完成时间估算](completion_eta_analysis.md)。

## SDG主划分三seed完整结果

[SDG增强监督与PL完整配对](sdg_aug_pl_analysis.md)：12行七视图及全部分层CM独立复算通过，原生预算/源曲线/default-head核验齐全。负结果全部保留，后续固定矩阵继续运行，整体目标未完成。

## RIEI主划分三seed完整结果

[RIEI增强监督与PL完整配对](riei_aug_pl_analysis.md)：12行七视图及全部分层CM独立复算通过，原生预算/源曲线/default-head核验齐全。负结果全部保留，后续固定矩阵继续运行，整体目标未完成。

## DRIFT主划分三seed完整结果

[DRIFT增强监督与PL完整配对](drift_aug_pl_analysis.md)：12行七视图及全部分层CM独立复算通过，原生预算/源曲线/default-head核验齐全。负结果全部保留，后续固定矩阵继续运行，整体目标未完成。

## F0 四方法 48 行独立核验完成

四方法 full/自身 CE × 监督/普通 PL × 三种子，全部 7 视图及 14112 个分层混淆矩阵独立复算通过；完整源曲线、原生实际预算、默认头及增强/PL 执行核验通过。[四方法主表与负结果解释](f0_native_factorial_analysis.md)已形成。该阶段仅覆盖主 F0 原生方法，219 行性能及 96 行成本目标继续。

## 强 CE 参考九行完成

[ResNet18/50 与普通 PL 完整报告](strong_ce_aug_pl_analysis.md)：9 行三种子、七视图及 2646 个分层混淆矩阵独立复算通过；与前述原生方法 48 行一起补齐主 F0 的 57 行，整体 219＋96 目标继续。

## 2026-10-19 21:50：低频运行检查与 ETA

只读核对现有性能/成本 owner 存活、GPU 2/3 符合资源约束；65/219 行完成、2 行运行，无已登记失败。首批 ResNet50 已取得完整实际耗时，中心完成估计约 10 月 20 日 14:47，沟通仍取 15:00 左右；原 12:00 至 22:00 窗口暂保持。[更新后的估算依据](completion_eta_analysis.md)。继续约每 30 分钟检查。
