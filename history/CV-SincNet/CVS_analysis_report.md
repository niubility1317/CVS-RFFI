# CVS 项目版本迭代与成果分析报告

## 1. 项目背景与研究目标

本项目面向射频指纹识别（RFFI）中的发射机/设备识别任务。输入为 I/Q 时域信号，输出为设备类别。早期阶段使用 ORACLE/ORALCE 目录下 `X_train.npy / Y_train.npy / X_test.npy / Y_test.npy` 格式数据；后期扩展到 WiSig compact pkl，并把采集日期 day 与接收机 rx 显式作为域变量。

从代码和日志看，项目研究重点逐步从“在同一数据分布下提高分类准确率”转向“跨 day / rx 泛化”。后期方法包含 domain generalization、domain adversarial、物理缺陷增强、DAC/PA 物理分支、MixStyle 与多分支解耦。由于当前环境无 git 命令，版本顺序依据目录名、目录修改时间、日志时间、代码文件大小和模块增量推断。

## 2. 项目文件结构与代码组织

目录结构摘要：

| 目录 | 阶段定位 | 主要证据 |
|---|---|---|
| `type1` | 初始 SincNet + 数据增强 baseline | `dataset.py`、`model.py`、`train.py`，无日志/checkpoint |
| `type2` | 增强版模型与增强脚本 | `DataAugmentation_v2.py`、更大的 `model.py` |
| `type3` | 加入 SupCon / prototype / DAC loss 的首个有日志版本 | `contrastive_loss.py`、`train_run1_Aug_CL.log` |
| `type4` | 模型结构扩大 | `model.py` 从约 10 KB 增至约 13 KB |
| `type5` | 评估解释工具加入 | `eval_and_explain.py`、训练日志、checkpoint |
| `type6` | 训练框架重构 | argparse、AMP、rollback、稳定性保护 |
| `type7` | 自适应增强与 eval_out | `eval_out` 分类报告、混淆矩阵、聚类统计 |
| `type8` | DG 试验 | `train_type8_DG*.log` |
| `type9` | WiSig / day-rx / PA / sat-channel | `dataset_wisig.py`、`sat_channel.py`、`train.py` |
| `type10` | dual CVSincNet 原型 | `model_dual_cvsincnet.py`、random80 trainDays + day4 test |
| `type11`-`type15` | 多种 dual / rx-adc / cutloss / PCGrad 分支 | 对应 `train_V*.log` 与模型文件 |
| `type10-pa` | PA 分支消融 | `train_V10_g*.log`、`feature_diagnosis_summary_g*.csv` |
| `type10-2` | main/pa/dac 三输出消融 | `train_g*.log` |
| `type10-3` | stagewise 训练策略消融 | `train_s*.log` |
| `type10-4` | 不稳定试验 | 36 epoch 崩溃日志、5 epoch 短跑日志 |
| `type10-5` | 最终消融与推荐版本 | `train.py`、`model.py`、`model_dual_cvsincnet.py`、18 个 ablation 日志 |

关键入口文件：

- 早期训练入口：`D:\codefile\CV-SincNet\type1\train.py`、`type3\train.py`。
- 最终训练入口：`D:\codefile\CV-SincNet\type10-5\train.py`。
- 最终模型：`D:\codefile\CV-SincNet\type10-5\model.py` 与 `D:\codefile\CV-SincNet\type10-5\model_dual_cvsincnet.py`。
- 最终数据加载：`D:\codefile\CV-SincNet\type10-5\dataset_wisig.py`。
- 最终增强：`D:\codefile\CV-SincNet\type10-5\DataAugmentation.py`。
- 评估与诊断：`eval_and_explain.py`、`eval_feature_diagnosis.py`。

## 3. CVS 方法整体技术路线

技术路线可概括为：

`I/Q 信号 -> 物理/域增强视图 -> SincConv + 复值/频域物理特征 backbone -> ID 分支与 domain 分支解耦 -> 域对抗与一致性约束 -> DAC/PA 强度与选择性物理分支监督 -> 发射机类别预测`

最终版本的关键词：

- Cross-domain RFFI
- WiSig day/rx isolation
- CV-SincNet / DualCVSincNet
- DAC-aware / PA-aware augmentation
- domain adversarial learning
- feature disentanglement
- MixStyle
- stagewise training
- best-joint 与 best-test 双口径评估

## 4. 核心模块分析

### 4.1 数据处理模块

早期 `WiFiRFFIDataset` 读取 `Dataset_ORALCE/run*/X_*.npy` 与 `Y_*.npy`，见 `type1\dataset.py` 与 `type1\train.py`。这类设置主要衡量 run 内训练/测试性能，日志中准确率可达 98% 左右，但不能充分回答跨采集条件泛化问题。

最终 `type10-5\dataset_wisig.py` 使用 `WiSigCompactDataset`。关键逻辑包括：

- `_rms_normalize_iq` 做 RMS 归一化，见 `dataset_wisig.py:13`。
- `_pad_or_crop_2t` 把 I/Q 信号裁剪或补齐到固定长度，见 `dataset_wisig.py:19`。
- `WiSigCompactDataset` 支持 `domain="day" / "rx" / "rx_day"`，见 `dataset_wisig.py:96` 与 `dataset_wisig.py:108`。
- `__getitem__` 返回 `x, y, d, meta`，其中 `d` 是域标签，见 `dataset_wisig.py:208` 到 `dataset_wisig.py:220`。
- `make_wisig_trainval_test_by_day_rx` 构造显式 day/rx 隔离的 train/val/test 与 named test split，见 `dataset_wisig.py:434` 与 `dataset_wisig.py:705`。

最终默认 split 来自 `type10-5\train.py:1427` 到 `1436`：WiSig pkl、`rx_day` 域、训练 day `0,1`、测试 day `2,3`、训练 rx `0..6`、测试 rx `7..11`。日志确认 split 为 `train_days_label=['2021_03_01','2021_03_08']`，test days 为 `['2021_03_15','2021_03_23']`，并设置 unseen day seen rx、seen day unseen rx、unseen day unseen rx 三类测试。

### 4.2 模型结构模块

早期 `type1\model.py` 是 SincConv + CNN 的单分支 `CVSincNet`，`SincConv_Fast` 用可学习带通滤波器处理原始 I/Q。

最终 `type10-5\model.py` 包含：

- `SincConv1d`：前端可学习滤波，见 `model.py:68`。
- `MixStyle1D`：特征统计混合，见 `model.py:272`。
- `PhysicalAwareClassifier`：将 ID、DAC、PA 特征投影后融合，输出分类 logits 与 `dac_pred / pa_pred`，见 `model.py:494`、`model.py:559` 到 `585`。
- `CVSincNet`：主网络，返回 `feat_cls / feat_imp / feat_dac / feat_pa / feat_joint / feat_con` 等特征，见 `model.py:593`、`model.py:1081` 到 `1116`。

最终 `type10-5\model_dual_cvsincnet.py` 进一步引入 dual backbone：

- `id_backbone` 学习发射机身份相关特征，见 `model_dual_cvsincnet.py:113`。
- `dom_backbone` 学习域相关特征，见 `model_dual_cvsincnet.py:129`。
- `dom_head` 预测域，`adv_head` 通过 GRL 反向约束 `z_id` 去域，`probe_head` 诊断 `z_id` 是否泄露域信息，见 `model_dual_cvsincnet.py:143` 到 `145`。
- forward 中输出 `tx_logits`、`dom_logits`、`adv_dom_logits`、`probe_dom_logits`、`z_id`、`z_dom`，见 `model_dual_cvsincnet.py:188` 到 `205`。

### 4.3 损失函数模块

早期 `type3` 引入：

- CE：主分类损失。
- SupCon：多视图同类拉近。
- Prototype contrastive：原型级类别约束。
- DAC regression：预测 DAC 扰动强度，`type3\train.py` 中 `LAMBDA_CON=0.2`、`LAMBDA_PROTO=0.2`、`DAC_LAMBDA=0.5`。

最终 `type10-5\train.py` 的损失更完整：

- `loss_cls`：发射机分类 CE，见 `train.py:869`。
- `loss_dom`：domain 分支预测域，见 `train.py:882`。
- `loss_adv`：GRL 后的域对抗，见 `train.py:885`。
- `loss_probe`：域泄露诊断，见 `train.py:887`。
- `loss_cons`：同类跨域一致性，见 `train.py:890`。
- `loss_orth`：`z_id` 与 `z_dom` 协方差正交，见 `train.py:898`。
- PA/DAC 辅助损失：`cls_pa/cls_dac`、`pa_joint_inv`、`pa_imp_inv`、`pa_kl`、`dac_reg/pa_reg`、`select`、`mono`、`cross_zero`，见 `train.py:927` 到 `984`。
- loss 权重在 `train.py:1472` 到 `1500` 定义，默认包括 `lambda_dom=1.0`、`lambda_adv=0.5`、`lambda_orth=0.05`、`lambda_cons=0.1`、`lambda_probe=1.0`、PA/DAC 选择性与单调约束等。

### 4.4 训练流程模块

最终训练流程：

1. 读取 WiSig compact pkl，建立 train/val/named test。
2. 构建 dual model，可选 `torch.compile`，见 `train.py:1729`。
3. 使用 AdamW 与 CosineAnnealingLR，见 `train.py:1750` 到 `1751`。
4. 每个 epoch 根据 stage state 控制辅助视图与 loss gate，日志中显示 `phase=S1_core`、`S2_stabilize_aux`、`S3_selective_late`。
5. 构造 clean / PA / DAC / DAC+PA 视图，计算核心损失与辅助损失。
6. `safe_backward_step` 检查非有限梯度并跳过异常 step，见 `train.py:1028` 到 `1069`。
7. 每轮评估 val、overall test 与 named tests，并保存 best joint、best overall、best unseen 等多个 checkpoint，见 `train.py:2018` 到 `2069`。

### 4.5 测试与评价模块

早期指标为 `Test Acc`。后期指标更细：

- `val_tx`：验证集发射机准确率，用于 best-joint 选择。
- `overall_tx`：所有 named test 聚合准确率。
- `unseen_day_seen_rx`：新日期、已见接收机。
- `seen_day_unseen_rx`：已见日期、新接收机。
- `unseen_day_unseen_rx`：新日期、新接收机，最难。
- `dom/probe`：域分类与域泄露诊断。

最终报告中必须区分：

- Last epoch target acc：最后一轮 `[TEST] overall_tx`。
- Best-joint：按验证集最优保存时对应 test。
- Best-test-overall：测试集 overall 最优，适合分析上限，但不适合严格模型选择。

## 5. 版本迭代与演进过程

| 阶段 | 主要代码/日志 | 核心改动 | 改动动机 | 实验结果 | 结论 |
|---|---|---|---|---|---|
| Baseline | `type1` | SincConv + CE + 简单增强 | 建立 RFFI 分类基线 | 无日志 | 起步版本 |
| V1 多视图对比 | `type3` | SupCon + Prototype + DAC loss | 强化类内一致性和物理扰动监督 | last 98.23%，best 98.65% | ORACLE run 内效果很好 |
| V2 结构扩展 | `type5` | 更大模型 + eval explain | 增强表达和诊断 | Aug_CL3 last 97.53%，best 97.97% | 相比 type3 下降，可能因结构/训练策略改变 |
| V3 稳定训练 | `type6` | argparse、AMP、rollback、DAC-only view | 强增强下稳定训练 | last 95.97%，best 96.24%@E320 | 更复杂但 run 内准确率下降 |
| V4 自适应增强 | `type7` | augmentation controller、eval_out | 控制增强强度与解释 | last 96.75%，best 97.11%@E340 | 较 type6 有提升 |
| V5 DG 试验 | `type8` | DG / DGH 训练 | 面向泛化 | DG_h last 97.38%，best 97.57%@E343 | ORACLE 下略优 |
| V6 WiSig 跨域 | `type9` | WiSig pkl、day/rx 域、PA、sat_channel、DA | 转向跨采集条件泛化 | best 95.66%@E130 或 95.02%@E74，last 92.98 到 93.75 | 任务更难，出现 best 与 last 差距 |
| V7 dual 原型 | `type10` | dual CVSincNet 与 day4 测试 | 解耦 ID/domain | best_tx 96.12%@E27，last val_day4 92.17 | 早期 best 高，后期回落 |
| V8 dual 分支探索 | `type11`-`type15` | rx/adc、多视图、cutloss、PCGrad | 探索更强解耦 | type15 best 89.97%@E310，last val_tx 89.64；type11 best 91.59%@E128 但后期崩溃到 31.10 | 部分方法不稳定 |
| V9 PA 消融 | `type10-pa` | PA main/aux 分支 | 分析 PA 辅助价值 | g1 best-joint 87.90，g4 86.48 | PA 组合未必优于无 PA |
| V10 main/pa/dac 消融 | `type10-2` | main/pa/dac 三分支 | 分析物理分支 | g4 rerun main 84.69，pa 87.74，dac 86.19 | PA/DAC 分支本身泛化强，但 main 受制约 |
| V11 stagewise 消融 | `type10-3` | S1/S2/S3/S4 阶段训练 | 控制辅助 loss 启用时机 | best 82.63 到 84.02，last 83.42 到 84.34 | stagewise 有帮助但 split 更严或设置不同 |
| V12 最终消融 | `type10-5` | MixStyle、lite_c、分支/物理特征消融 | 统一 split 下比较模块贡献 | D02 best-joint 90.18，last 89.67，best-test 90.53 | 当前最佳与推荐版本 |

## 6. 实验设置与训练日志分析

最终统一设置依据 `type10-5\train.py` 与日志：

- 数据集：WiSig compact pkl `./Dataset_WigSig/ManySig.pkl`。
- 任务：16 类发射机识别。
- Source/train：2021-03-01、2021-03-08，rx 0..6。
- Target/test：2021-03-15、2021-03-23，rx 7..11，并拆分三类 target。
- batch size：128，eval batch size：256。
- optimizer：AdamW，lr=2e-4，wd=1e-4。
- scheduler：CosineAnnealingLR，lr_min=1e-6。
- epoch：100。

实验结果总表：

| 实验 | 阶段 | Last Epoch Overall | Best-joint Test | Best-test Overall | 最难 split best | 结论 |
|---|---|---:|---:|---:|---:|---|
| A00_s1_core_base | core baseline | 86.51 | 86.25@E83 | 89.51@E14 | 86.72@E10 | 最终消融基线 |
| A01_s4_base_no_mixstyle | full no MixStyle | 87.46 | 87.67@E93 | 89.72@E14 | 83.58@E7 | 较 A00 提升，但最难 split 降低 |
| B00_mixstyle_cd_td_t1 | crossdomain MixStyle | 88.03 | 88.08@E93 | 89.85@E11 | 85.37@E8 | MixStyle 有稳定增益 |
| B01_mixstyle_cd_td_t1_t2 | 多层 MixStyle | 48.07 | 86.61@E57 | 88.95@E42 | 84.04@E9 | 后期严重退化 |
| B02_mixstyle_random_td_t1 | random MixStyle | 16.67 | 88.22@E39 | 89.94@E8 | 86.71@E8 | early best 高但训练崩溃 |
| B03_mixstyle_cd_p015 | MixStyle p=0.15 | 87.72 | 86.78@E76 | 90.06@E15 | 86.08@E6 | 早期测试好，joint 一般 |
| B04_mixstyle_cd_alpha03 | alpha=0.3 | 87.94 | 88.12@E69 | 90.04@E10 | 85.64@E8 | 与 B00 接近 |
| C00_no_time | 去 time 分支 | 16.67 | 86.96@E42 | 88.05@E8 | 84.27@E8 | 后期崩溃，time 重要 |
| C01_no_dac | 去 DAC | 88.17 | 88.27@E86 | 90.04@E11 | 85.61@E11 | DAC 并非单独决定性能 |
| C02_no_pa | 去 PA | 86.45 | 86.99@E82 | 88.78@E7 | 83.04@E15 | PA 对稳定泛化有帮助 |
| C03_no_freq | 去 freq | 16.67 | 86.23@E36 | 86.71@E42 | 82.57@E9 | 频域/物理统计重要 |
| C04_no_stats | 去 stats | 85.83 | 86.01@E90 | 87.23@E55 | 82.22@E12 | 统计特征重要 |
| C05_no_dac_pa | 去 DAC+PA | 16.67 | 87.17@E31 | 88.26@E8 | 84.17@E22 | 后期不稳定 |
| C06_time_only | 仅 time | 86.59 | 86.37@E90 | 86.98@E56 | 81.36@E66 | 单 time 不够 |
| C07_freq_only | 仅 freq | 16.67 | 69.55@E47 | 70.43@E56 | 59.97@E22 | 单 freq 不足 |
| D00_mixstyle_no_dac | MixStyle 去 DAC | 88.84 | 88.84@E99 | 90.14@E25 | 85.04@E47 | 稳定性较好 |
| D01_mixstyle_no_pa | MixStyle 去 PA | 83.39 | 86.40@E40 | 88.99@E8 | 83.07@E8 | 去 PA 后下降且跳步多 |
| D02_litec_mixstyle | lite_c + MixStyle | 89.67 | 90.18@E86 | 90.53@E63 | 86.25@E10 | 最佳推荐版本 |

最好结果：

- 严格按 validation 选模型的 best-joint：`D02_litec_mixstyle`，test_overall 90.18%@E86。
- 仅看测试上限 best-test-overall：`D02_litec_mixstyle`，90.53%@E63。
- 最后一轮稳定性：`D02_litec_mixstyle`，last 89.67%，明显优于 A00 last 86.51%。

相比最终基线 A00：

- last epoch：+3.16 个百分点。
- best-joint test：+3.93 个百分点。
- best-test-overall：+1.02 个百分点。
- 最难 split best：D02 86.25%，略低于 A00 86.72%，说明 overall 提升不完全来自最难 split。

## 7. 失败实验与问题分析

1. `B02_mixstyle_random_td_t1`：best-joint 88.22%，但最后崩溃到 16.67%，`skipped_backward_batches=14428`。说明 random MixStyle 有早期收益，但会引入训练不稳定。
2. `B01_mixstyle_cd_td_t1_t2`：last 48.07%，多层 MixStyle 可能破坏身份特征。
3. `C00_no_time`、`C03_no_freq`、`C05_no_dac_pa`：均在后期退化到 16.67%，提示分支/物理信息删除后训练约束失衡。
4. `type11` 日志显示 best 91.59%@E128，但后期 val_tx 降到 31.10%，说明部分 dual/style/proxy 组合存在显著后期崩溃。
5. `type10-4\logs\train_20260422_174856.log` 到 E36 降到 16.67%，是不完整且失败的中间尝试。

## 8. 最终成果总结

最终方法建议定义为：`Dual CV-SincNet with physical-aware DAC/PA branches and cross-domain MixStyle`。最终代码位于 `D:\codefile\CV-SincNet\type10-5`，推荐实验为 `D02_litec_mixstyle`。

最终结论：

- 项目已经从 ORACLE run 内识别推进到 WiSig day/rx 双重跨域泛化。
- 最终版本在统一 split 下优于 core baseline，best-joint overall 从 86.25% 提升到 90.18%。
- MixStyle 与 lite_c 结构是最有效组合；PA 分支对稳定性有贡献；仅 time 或仅 freq 都不足。
- 训练稳定性仍是核心问题，多个实验出现 NaN、跳过 backward 或后期坍塌。
- best-test-overall 不能替代 validation-selected best-joint；组会中应同时报告 last、best-joint、best-test 三个口径。

## 9. 复现流程说明

推荐复现入口：

```powershell
cd D:\codefile\CV-SincNet\type10-5
python train.py --dataset wisig --wisig_pkl ./Dataset_WigSig/ManySig.pkl --model_variant lite_c --mixstyle_on
```

注意：

- 需要保证 `Dataset_WigSig/ManySig.pkl` 路径存在。
- 默认 split 已在 `train.py` 中设置为 day 0,1 训练，day 2,3 测试，rx 0..6 训练，rx 7..11 测试。
- 若复现 D02，需要对照 `D02_litec_mixstyle_20260424_000740.log` 中 run 配置，确认 `model_variant=lite_c`、MixStyle 开启、crossdomain 设置。
- 当前环境未安装 `git`，无法验证 commit；报告基于文件系统证据。

## 10. 导师可能提问与回答

| 问题 | 回答要点 | 可引用依据 |
|---|---|---|
| CVS 解决什么问题？ | 解决 RFFI 发射机识别在不同采集日期和接收机下泛化下降的问题 | `type10-5\dataset_wisig.py`、最终日志 split info |
| source / target 是什么？ | source 为训练 day/rx，target 分为 unseen day seen rx、seen day unseen rx、unseen day unseen rx | `type10-5\train.py:1433-1436` 与日志 `[TEST-SPLIT]` |
| CVS 的核心思想是什么？ | 用 Sinc/物理特征提取身份，显式建模 DAC/PA 与 domain，通过 dual backbone 和域对抗减少域泄露 | `model.py`、`model_dual_cvsincnet.py` |
| 为什么 MixStyle 有效？ | 它扰动或混合特征统计，使模型不依赖单一 day/rx 风格；B00/D02 相对 A00 有提升 | `type10-5` 日志表 |
| 哪个模块贡献最大？ | 从最终消融看，D02 lite_c + MixStyle 整体最佳；去 time/freq/stats 会明显退化 | `C00/C03/C04/C07/D02` 日志 |
| last 和 best 看哪个？ | 科研汇报应主报 validation-selected best-joint 与 last，best-test 只作为上限参考 | `type10-5` 日志同时保存三种口径 |
| 为什么有些版本没有提升？ | 辅助约束过强、多层/random MixStyle 或分支删除会导致梯度不稳定、NaN 或后期坍塌 | `B02`、`C00`、`C03` 的 skipped/nan 记录 |
| 如何复现实验？ | 进入 `type10-5`，使用 `train.py`，准备 `ManySig.pkl`，按日志配置选择 recipe/model_variant/mixstyle | `type10-5\train.py` |
| 当前不足？ | 稳定性、seed 复现、多 split 统计不足；最难 split 提升不稳定 | 日志中 NaN、skipped、best-last 差异 |

## 11. 不确定信息与待补充材料

- 无 git commit 历史，版本顺序为推断。
- 原项目 README 缺失，部分中文注释编码损坏。
- 未读取 `.pth` checkpoint 内部参数，只根据命名和日志判断。
- `type11` 到 `type15` 的若干日志缺少明确 target split 名称，按 day4/random80 阶段解释。
- 若需论文级结论，应补充多 seed、多 split 均值与标准差。

## 12. 质量检查

- 已扫描代码、日志、CSV、txt、checkpoint 命名。
- 已区分 last epoch、best-joint、best-test-overall。
- 已纳入失败实验。
- 所有关键结论均给出文件或日志依据。
- 不确定推断已单独标注。
