# CVS 项目模块实现与版本迭代分析报告

## 0. 说明

本报告基于 `D:\codefile\CV-SincNet` 项目目录中的代码、日志、模型命名和实验输出整理。由于未获得可用的 git commit 历史，版本顺序主要根据文件夹命名（`type1` 至 `type15`、`type10-*`）、日志时间、脚本命名、代码复杂度递增关系和实验任务迁移进行推断。凡涉及版本先后关系的判断，均按“根据文件夹命名/日志时间/代码差异推断”处理。

本报告重点回答三个问题：

1. 每个核心模块如何实现；
2. 每个模块在训练流程中承担什么功能；
3. 模块演进如何对应实验结果变化。

---

## 1. 项目总体任务与最终结论

### 1.1 研究任务

CVS 项目围绕无线射频指纹识别（RF fingerprinting）中的跨域泛化问题展开。模型输入为 I/Q 序列，输出为发射设备身份类别。项目后期核心任务从早期 ORACLE 数据集上的设备分类，逐步迁移到 WiSig 数据集中更困难的跨天、跨接收机 setting。

后期主任务可概括为：

`训练域 = 已见 day + 已见 rx`，测试域包含 `unseen day`、`unseen rx` 和 `unseen day + unseen rx`。模型需要学习与发射设备身份稳定相关的特征，同时抑制接收机、日期、信道、PA/DAC 等域因素带来的干扰。

### 1.2 最终推荐版本

最终推荐版本位于：

- `type10-5/train.py`
- `type10-5/model.py`
- `type10-5/model_dual_cvsincnet.py`
- `type10-5/dataset_wisig.py`
- `type10-5/DataAugmentation.py`
- `type10-5/logs/*.log`

最终推荐实验为：

- `type10-5/logs/D02_litec_mixstyle*.log`
- Last epoch target accuracy：`89.67%`
- Best joint accuracy：`90.18% @ Epoch 86`
- Best test overall：`90.53% @ Epoch 63`
- Best unseen-day-unseen-rx：`86.25% @ Epoch 10`

与 `type10-5` 内部核心 baseline `A00_s1_core_base` 相比：

| 对比项 | A00_s1_core_base | D02_litec_mixstyle | 提升 |
|---|---:|---:|---:|
| Last epoch | 86.51% | 89.67% | +3.16 |
| Best joint | 86.25% | 90.18% | +3.93 |
| Best test overall | 89.51% | 90.53% | +1.02 |
| Best unseen-day-unseen-rx | 86.72% | 86.25% | -0.47 |

结论：D02 在整体 last epoch 与 best joint 上最优，但最困难的 unseen-day-unseen-rx 子集并非全面优于 A00，说明最终方案提升了整体稳定性和平均泛化能力，但极端组合域泛化仍存在改进空间。

---

## 2. 版本演进总览

根据目录与日志推断，CVS 项目的演化链如下：

```text
type1/type2 基础 SincNet 分类
  -> type3/type5 对比学习与原型约束
  -> type6/type7 稳定训练、AMP、评估与增强策略
  -> type8 显式 domain generalization 实验
  -> type9 WiSig 跨 day/rx 任务与物理增强
  -> type10 双分支 CVSincNet 原型
  -> type10-pa / type10-2 / type10-3 消融与多辅助分支
  -> type10-4 不稳定强化尝试
  -> type10-5 最终模块化稳定版本
  -> type11-type15 并行探索：dual/rx-adc/cutloss/PCGrad 等
```

### 2.1 版本迭代表

| 阶段 | 主要代码/日志 | 核心改动 | 改动动机 | 实验结果 | 结论 |
|---|---|---|---|---|---|
| Baseline | `type1/dataset.py`, `type1/model.py`, `type1/train.py` | SincConv + CNN + CE 分类；简单增强 | 建立 I/Q 设备分类基线 | ORACLE 任务可训练 | 完成基础分类框架，但缺少跨域机制 |
| 对比学习阶段 | `type3/contrastive_loss.py`, `type3/train.py` | 加入 SupCon、prototype、DAC loss，多视图增强 | 提升类内聚合与设备身份判别性 | `train_run1_Aug_CL.log` last 98.23%，best 98.65% | 对 ORACLE 分类有效，但仍偏同分布 |
| 解释与精简阶段 | `type5/eval_and_explain.py`, `type5/logs` | 增加评估、可解释输出和 slim 变体 | 分析模型行为并降低复杂度 | full best 97.97%，slim best 97.56% | 精简有轻微损失，可解释评估开始成型 |
| 稳定训练阶段 | `type6/train.py` | argparse、AMP、checkpoint、训练保护 | 提高长训练可复现性和稳定性 | last 95.97%，best 96.24%@E320 | 工程化增强，但指标低于早期最佳 |
| 自适应增强阶段 | `type7/*`, `eval_out` | 增强策略与评估输出改进 | 尝试提高泛化并观察中间结果 | last 96.75%，best 97.11%@E340 | 比 type6 提升，说明增强策略有效 |
| DG 实验阶段 | `type8/train_type8_DG_h.log` | 引入更明确 domain generalization 训练 | 从分类转向跨域泛化 | last 97.38%，best 97.57%@E343 | 泛化实验方向明确 |
| WiSig 迁移阶段 | `type9/*` | 数据集切换为 WiSig/day/rx；加入 PA、sat_channel、DA | 面向真实跨天/跨接收机挑战 | best 95.66%@E130，last 92.98-93.75 | 任务更真实，last 与 best 出现差距 |
| 双分支原型 | `type10/model*.py`, `train_V1020260309_183806.log` | dual CVSincNet、day4 random80 split | 分离身份与域因素 | best_tx 96.12%@E27，last val_day4 92.17 | 早期有效但后期退化 |
| PA/辅助消融 | `type10-pa`, `type10-2`, `type10-3` | PA/DAC 辅助分支、阶段式训练、选择性约束 | 验证物理因素建模是否有益 | type10-2 g4 main 84.69 / pa 87.74 / dac 86.19；type10-3 s4 last 84.34 | 辅助分支有效，但权重和阶段敏感 |
| 不稳定强化 | `type10-4` | 更激进组合尝试 | 追求更高泛化 | 有 run last 16.67，也有短 run last 89.85 | 不稳定，不能作为最终版本 |
| 最终版本 | `type10-5/*` | 模块化 WiSig、MixStyle、dual disentangle、PA/DAC、safe backward | 平衡准确率、稳定性和可复现性 | D02 last 89.67，best-joint 90.18 | 当前最可靠推荐版本 |
| 并行探索 | `type11-type15` | dual/rx-adc/cutloss/PCGrad 等 | 探索更强解耦与优化 | type15 best 89.97，last 89.64；type11 曾 best 91.59 后崩到约 31.10 | 有探索价值，但稳定性不足或未超最终版本 |

---

## 3. 模块一：数据加载与预处理

### 3.1 早期 ORACLE 数据模块

代码位置：

- `type1/dataset.py`
- `type1/train.py`

功能：

- 读取 `.npy` 格式 I/Q 数据；
- 构造发射设备分类样本；
- 提供 `DataLoader` 给训练脚本；
- 训练目标为设备类别交叉熵分类。

实现原理：

早期数据模块主要服务于单一数据集上的监督分类。输入 I/Q 序列被组织为适配 SincNet/CNN 的张量，标签直接作为设备 ID。该阶段没有显式 source/target 域划分，重点是证明 SincConv 风格前端可以用于 RF 指纹识别。

局限：

- 域变量没有显式建模；
- train/test 划分更接近同分布验证；
- 无法直接回答跨 day、跨 rx 泛化问题。

### 3.2 WiSig 数据模块

代码位置：

- `type10-5/dataset_wisig.py`
- `_rms_normalize_iq`：约第 13 行
- `_pad_or_crop_2t`：约第 19 行
- `WiSigCompactDataset`：约第 96 行
- `_build_domain_lut`：约第 148-166 行
- `__getitem__`：约第 208-220 行
- `make_wisig_trainval_test_by_day_rx`：约第 434 行

功能：

- 读取 WiSig compact pkl 数据；
- 支持按 day、rx 或 day-rx 组合构造 domain；
- 对 I/Q 序列做 RMS 归一化；
- 对序列长度进行 pad/crop 到固定 `out_len=256`；
- `__getitem__` 返回 `(x, y, d, meta)`，其中 `y` 是发射机 ID，`d` 是域 ID，`meta` 包含 day/rx 等信息；
- 支持 train/val/test 按天和接收机划分。

实现原理：

WiSig 阶段把样本拆成“身份标签”和“域标签”。身份标签用于主分类，域标签用于 domain classifier、adversarial loss、MixStyle 分组和评估切片。`day_rx` 组合域能更细粒度描述环境变化，使训练时可以显式区分“同身份不同域”的变化来源。

模块作用：

数据模块是后期方法成立的基础。只有返回 `y` 和 `d`，训练阶段才能同时优化身份判别与域不变性；只有保留 `meta`，评估模块才能区分 unseen day、unseen rx 和 unseen day+rx。

演进意义：

从 ORACLE 到 WiSig，不是简单换数据集，而是研究问题从“分类准确率”升级为“跨域泛化能力”。这也是后续 MixStyle、domain adversarial、PA/DAC 辅助分支存在的前提。

---

## 4. 模块二：数据增强与物理扰动模拟

### 4.1 早期增强

代码位置：

- `type1/DataAugmentation.py`
- `type3/train.py`

功能：

- 为同一样本生成多个增强视图；
- 配合监督对比学习，使同类样本或同一设备的不同视图在特征空间中靠近。

实现原理：

增强后的多视图输入共享标签。模型通过 CE 保持分类能力，通过 contrastive/prototype loss 约束增强视图在特征空间的一致性。

效果证据：

- `type3/train_run1_Aug_CL.log`：last 98.23%，best 98.65%。

结论：

多视图增强 + 对比约束在早期 ORACLE 分类任务上非常有效，但该阶段增强主要服务于类内一致性，不足以覆盖真实接收机和日期域变化。

### 4.2 后期物理增强

代码位置：

- `type10-5/DataAugmentation.py`
- `DacParams`：约第 175 行
- `PaParams`：约第 226 行
- `RFFIAugmentor`：约第 261 行
- `simulate_dac`：约第 592 行
- `simulate_pa`：约第 780 行
- `simulate_dac_pa`：约第 826 行
- `build_augmentor`：约第 1050 行
- `apply_receiver_dg`：约第 1243 行

功能：

- 模拟 DAC 非线性、量化、失配等发射端/硬件扰动；
- 模拟 PA 非线性和饱和效应；
- 模拟接收端或信道相关扰动；
- 在训练阶段制造更丰富的跨域变化。

实现原理：

RF 指纹识别的困难来自身份因素与硬件/信道/接收机因素的耦合。物理增强模块通过人为注入 PA/DAC/接收端扰动，使模型在训练时见到更多“非身份变化”。理想情况下，身份主干应保留发射机稳定特征，而域相关分支或辅助头吸收可变物理因素。

模块作用：

- 为 supervised contrastive 和一致性约束提供不同视图；
- 为 PA/DAC 辅助分支提供有意义的预测目标；
- 提升模型对 unseen day/rx 的鲁棒性。

失败经验：

过强或组合不当的增强会导致训练崩溃。`type10-5` 中 `B02_mixstyle_random_td_t1` last 只有 16.67%，同时日志记录大量 `skipped_backward_batches=14428`，说明随机 MixStyle/扰动组合会使梯度异常或优化目标冲突。

---

## 5. 模块三：模型结构与特征提取

### 5.1 SincConv 前端

代码位置：

- `type10-5/model.py`
- `SincConv1d`：约第 68 行
- `CVSincNet`：约第 593 行

功能：

- 用可学习带通滤波器处理一维 I/Q 序列；
- 在原始波形层面提取频率相关特征；
- 作为 RF 信号分类的低层特征前端。

实现原理：

SincConv 使用参数化滤波器替代完全自由的一维卷积核，使前端更符合信号处理直觉。它倾向于学习不同频带的响应，对 RF 指纹中的频谱偏差、硬件非理想性和调制相关信息更敏感。

演进意义：

SincConv 是 CVS 项目的基础模型风格。早期版本主要依赖该前端完成分类，后期版本在该前端之上叠加 MixStyle、物理辅助分支和解耦结构。

### 5.2 MixStyle1D

代码位置：

- `type10-5/model.py`
- `MixStyle1D`：约第 272 行
- forward：约第 324 行

功能：

- 在特征层混合不同样本或不同域的统计量；
- 降低模型对特定域风格的依赖；
- 提升跨 day/rx 泛化。

实现原理：

MixStyle 的核心是对特征均值和方差进行混合。对 RF I/Q 特征而言，不同日期、接收机、信道会改变特征统计分布。通过在训练时混合统计量，模型被迫学习不依赖单一域统计的身份特征。

实验结论：

`type10-5` 最优实验 `D02_litec_mixstyle` 表明，合适的 MixStyle 配置能提升 overall 和 joint 指标。但失败实验 `B01/B02/D01` 也说明，MixStyle 的层位置、概率、混合方式和是否结合 PA/DAC 约束都很敏感。

### 5.3 PhysicalAwareClassifier

代码位置：

- `type10-5/model.py`
- `PhysicalAwareClassifier`：约第 494 行
- forward 输出：约第 559-585 行

功能：

- 将特征拆分或映射为身份相关、DAC 相关、PA 相关、联合特征等多个分量；
- 输出 `dac_pred`、`pa_pred` 等辅助预测；
- 支持主分类和物理辅助监督联合训练。

实现原理：

模型不再假设所有可判别信息都应进入同一分类头，而是把可能的物理因素显式建模。PA/DAC 辅助头的作用不是最终预测目标本身，而是通过辅助监督引导特征空间把部分硬件变化组织成可解释维度，从而降低主身份特征的混杂。

实验依据：

`type10-2` 消融显示：

| 实验 | main | pa | dac | 结论 |
|---|---:|---:|---:|---|
| g0 baseline_joint | 79.92 | 11.13 | 17.24 | 无辅助时 PA/DAC 表征弱 |
| g1 clean_coop | 82.28 | 83.73 | 84.26 | 辅助头显著提升 |
| g4 full balanced rerun | 84.69 | 87.74 | 86.19 | 平衡权重效果最好 |

结论：

PA/DAC 辅助分支是后期提升的重要来源，但需要与主分类、域对抗和增强强度平衡。

### 5.4 DualCVSincNetDisentangle

代码位置：

- `type10-5/model_dual_cvsincnet.py`
- `grad_reverse`：约第 53 行
- `DualCVSincNetDisentangle`：约第 73 行
- `id_backbone`：约第 113 行
- `dom_backbone`：约第 129 行
- heads：约第 143-145 行
- forward：约第 188 行
- `aux_id/aux_dom`：约第 196-197 行
- `dom/adv/probe logits`：约第 203-205 行

功能：

- 构建身份分支和域分支；
- 通过 domain classifier 与 gradient reversal 学习域不变身份特征；
- 用辅助 probe 监控或约束身份/域信息泄漏；
- 支持身份特征和域特征的解耦训练。

实现原理：

双分支结构的目标是让 `id_backbone` 保留设备身份信息，同时降低对 day/rx 域信息的依赖；`dom_backbone` 则吸收域变化。`grad_reverse` 在反向传播中反转梯度，使身份特征对域分类器“不可分”，从而逼近 domain-invariant representation。

局限：

双分支和对抗训练引入多个相互竞争目标，训练稳定性较差。`type11` 曾出现 best 91.59%@E128，但后期崩溃到约 31.10，说明 best 指标不能单独代表最终可靠性，必须同时看 last epoch 和训练曲线。

---

## 6. 模块四：损失函数系统

### 6.1 早期损失

代码位置：

- `type1/train.py`
- `type3/contrastive_loss.py`

组成：

- 交叉熵分类损失；
- supervised contrastive loss；
- prototype loss；
- DAC loss。

功能与原理：

CE 负责设备身份分类；contrastive loss 拉近同类/同设备增强视图特征；prototype loss 让特征靠近类别中心；DAC loss 尝试约束增强或域相关变化。该阶段的核心思想是先提升身份特征的紧致性与可分性。

效果：

`type3` 的 last 98.23%、best 98.65% 说明该策略在早期任务上有效。

### 6.2 后期损失

代码位置：

- `type10-5/train.py`
- `domain_loss_gates`：约第 316 行
- loss weights/gates：约第 597-613 行
- core loss：约第 869-898 行
- aux losses：约第 927-984 行
- safe backward：约第 1028-1069 行

主要组成：

| Loss | 功能 | 原理 |
|---|---|---|
| `loss_id` / CE | 主身份分类 | 直接优化发射机 ID 准确率 |
| `loss_dom` | 域分类 | 让域分支捕获 day/rx 信息 |
| `loss_adv` | 域对抗 | 通过梯度反转降低身份特征中的域可分性 |
| `loss_orth` | 正交约束 | 降低身份特征和域特征冗余 |
| `loss_cons` | 一致性约束 | 保持增强前后预测或特征稳定 |
| `loss_probe` | 信息泄漏监控/约束 | 检查身份/域信息是否错误流入另一分支 |
| `loss_pa` / `loss_dac` | 物理辅助监督 | 引导模型组织 PA/DAC 相关变化 |

加权方式：

后期训练通过 argparse 暴露多个 `lambda_*`，例如 `lambda_dom=1.0`、`lambda_adv=0.5`、`lambda_orth=0.05`、`lambda_cons=0.1`、`lambda_probe=1.0`。不同实验通过调节这些权重形成消融。

模块作用：

后期 loss 系统不再只是“让分类更准”，而是同时处理三个目标：

1. 身份可分；
2. 域因素可控或可剥离；
3. 物理扰动可解释并不破坏主分类。

关键结论：

辅助 loss 有效，但不是越多越好。`C05_no_dac_pa`、`D01_mixstyle_no_pa` 等实验显示，去掉 PA/DAC 辅助后整体稳定性下降；而过强 MixStyle 或随机混合又会导致训练崩溃。

---

## 7. 模块五：训练流程与工程稳定性

### 7.1 训练入口与参数系统

代码位置：

- `type10-5/train.py`
- argparse main：约第 1422 行以后
- data args：约第 1424-1436 行
- optimizer args：约第 1443-1450 行
- model/mixstyle args：约第 1451-1470 行
- core lambdas：约第 1472-1477 行
- aux/stage args：约第 1479-1500 行
- augmentation args：约第 1522-1560 行

默认关键设置：

| 参数 | 默认或典型值 |
|---|---|
| dataset | `wisig` |
| wisig_pkl | `./Dataset_WigSig/ManySig.pkl` |
| wisig_domain | `rx_day` |
| wisig_out_len | 256 |
| train days | `0,1` |
| test days | `2,3` |
| train rx | `0,1,2,3,4,5,6` |
| test rx | `7,8,9,10,11` |
| batch size | 128 |
| eval batch size | 256 |
| epochs | 100 |
| lr | `2e-4` |
| lr_min | `1e-6` |
| optimizer | AdamW |
| scheduler | Cosine |
| label smoothing | 0.01 |

### 7.2 训练循环

代码位置：

- core loss 计算：`type10-5/train.py` 约第 869-898 行
- aux loss 计算：约第 927-984 行
- optimizer/scheduler：约第 1750-1751 行
- checkpoint 保存：约第 2018-2069 行

流程：

```text
读取 batch: x, y, domain, meta
  -> 可选物理增强 / MixStyle
  -> model forward 输出身份、域、PA/DAC、特征
  -> 计算主 CE、域损失、对抗损失、正交损失、一致性损失、PA/DAC 辅助损失
  -> 加权求和
  -> safe backward
  -> optimizer step
  -> scheduler step
  -> 周期性评估 val/test/切片域
  -> 保存 best/last checkpoint
```

### 7.3 safe backward

代码位置：

- `type10-5/train.py`
- `safe_backward_step`：约第 1028-1069 行

功能：

- 检查梯度异常；
- 遇到数值不稳定时跳过 backward 或 step；
- 在复杂 loss/MixStyle/增强组合下防止训练直接中断。

实验意义：

日志中大量 `skipped_backward_batches` 不是可忽略细节，而是训练不稳定的重要证据。例如 `B02_mixstyle_random_td_t1` 发生大量跳过并最终 last 16.67%，说明该组合虽然早期 best 较高，但不适合作为稳定方案。

---

## 8. 模块六：测试与评价

### 8.1 评价函数

代码位置：

- `type10-5/train.py`
- `evaluate_loader`：约第 1246 行
- `format_epoch_block`：约第 1335 行

功能：

- 计算整体 accuracy；
- 记录 last epoch accuracy；
- 记录 best accuracy；
- 对不同 day/rx/domain split 输出结果；
- 区分 overall、joint、unseen-day、unseen-rx、unseen-day-unseen-rx。

### 8.2 为什么必须区分 last 和 best

CVS 项目多个实验存在“早期 best 高、后期退化”的现象：

| 实验 | best | last | 解释 |
|---|---:|---:|---|
| `type10` dual 原型 | best_tx 96.12%@E27 | val_day4 92.17%@E200 | 后期退化 |
| `type11` 某实验 | best 91.59%@E128 | 后期约 31.10 | 严重崩溃 |
| `type10-5 B02` | best-joint 88.22%@E39 | last 16.67 | 优化不稳定 |
| `type10-5 D02` | best-joint 90.18%@E86 | last 89.67 | best 与 last 接近，较稳定 |

结论：

组会汇报中应同时展示 best 和 last。best 说明方法潜力，last 说明训练稳定性和最终可复现状态。最终推荐应优先选择 best 高且 last 不崩溃的方案。

---

## 9. 实验结果总表

| 实验名称 | 版本阶段 | 数据集/任务 | 关键设置 | Last Epoch Target Acc | Best Acc | 结论 |
|---|---|---|---|---:|---:|---|
| `train_run1_Aug_CL` | type3 | ORACLE 分类 | Aug + SupCon + Proto + DAC | 98.23 | 98.65 | 早期最强，同分布分类有效 |
| `train_run1_Aug_CL3` | type5 | ORACLE 分类 | full + explain | 97.53 | 97.97 | 略低于 type3，但评估更完整 |
| `slim` | type5 | ORACLE 分类 | 精简模型 | 97.48 | 97.56 | 模型精简损失较小 |
| type6 stable | type6 | ORACLE/DG 过渡 | AMP + argparse + safe | 95.97 | 96.24 | 工程稳定，性能下降 |
| type7 adaptive | type7 | ORACLE/DG 过渡 | adaptive augmentation | 96.75 | 97.11 | 较 type6 提升 |
| `train_type8_DG_h` | type8 | DG | 显式泛化训练 | 97.38 | 97.57 | DG 方向有效 |
| type9 WiSig | type9 | WiSig day/rx | PA/channel/DA | 92.98-93.75 | 95.66 | 真实任务更难，best-last 差距增大 |
| `train_V1020260309_183806` | type10 | day4 random80 | dual CVSincNet | 92.17 | 96.12 | 早期 dual 有潜力，后期退化 |
| `g0 baseline_joint` | type10-2 | WiSig | 无有效 PA/DAC 辅助 | main 79.92 | - | 辅助物理表征弱 |
| `g4 full balanced rerun` | type10-2 | WiSig | PA/DAC 平衡辅助 | main 84.69 | - | 辅助分支明显有效 |
| `s4 full dual` | type10-3 | WiSig | full dual stage | 84.34 | 83.48 | 阶段式训练有改善 |
| `B02_mixstyle_random_td_t1` | type10-5 | WiSig day/rx | random MixStyle | 16.67 | 88.22 | 早期有效但严重崩溃 |
| `A00_s1_core_base` | type10-5 | WiSig day/rx | core baseline | 86.51 | 86.25 joint / 89.51 overall | 稳定 baseline |
| `D02_litec_mixstyle` | type10-5 | WiSig day/rx | lite_c + MixStyle | 89.67 | 90.18 joint / 90.53 overall | 当前推荐最终版本 |
| type15 cutloss | type15 | WiSig/dual | cutloss/PCGrad 探索 | 89.64 | 89.97 | 接近最终版，但未明显超过 D02 |

---

## 10. 模块贡献总结

| 模块 | 实现位置 | 功能 | 原理 | 对结果的影响 |
|---|---|---|---|---|
| WiSig 数据划分 | `type10-5/dataset_wisig.py` | 构造 day/rx/source-target 任务 | 显式建模域标签 | 使问题从分类升级为跨域泛化 |
| SincConv 前端 | `type10-5/model.py` | 提取频带相关 RF 特征 | 参数化带通滤波 | 构成全项目基础特征提取器 |
| MixStyle1D | `type10-5/model.py` | 混合特征统计 | 降低域统计依赖 | D02 提升明显，但配置敏感 |
| PA/DAC 增强 | `type10-5/DataAugmentation.py` | 模拟硬件扰动 | 用物理变化扩展训练分布 | 辅助泛化，但过强会不稳定 |
| PhysicalAwareClassifier | `type10-5/model.py` | 预测 PA/DAC 辅助任务 | 显式组织物理因素 | type10-2 消融显示有效 |
| Dual disentangle | `type10-5/model_dual_cvsincnet.py` | 分离身份/域特征 | 域对抗 + 双分支 | 有潜力但训练敏感 |
| 多 loss 系统 | `type10-5/train.py` | 联合身份、域、物理、正交、一致性约束 | 多目标优化 | 决定泛化与稳定性平衡 |
| safe backward | `type10-5/train.py` | 跳过异常梯度 | 防止训练中断 | 揭示失败实验的数值问题 |
| 分域评估 | `type10-5/train.py` | 输出 overall/joint/unseen split | 分析泛化来源 | 支撑最终结论 |

---

## 11. 最终方法概括

最终 CVS 方法可概括为：

```text
WiSig I/Q 序列
  -> RMS 归一化 + 固定长度裁剪
  -> SincConv/CVSincNet 特征提取
  -> 可选 MixStyle1D 域统计混合
  -> 身份分支 + 域分支 + PA/DAC 物理辅助分支
  -> CE + domain + adversarial + orthogonal + consistency + PA/DAC losses
  -> 按 day/rx/unseen split 评价泛化能力
```

核心思想：

CVS 并不是单纯堆叠更深模型，而是围绕“身份因素”和“非身份域因素”解耦展开。数据模块提供 day/rx 域标签，增强模块模拟硬件和接收端变化，模型结构将身份、域和物理因素分支化，loss 系统用对抗、正交和辅助监督约束特征空间，最终通过分域评估验证泛化能力。

---

## 12. 失败实验与可解释结论

| 失败/不稳定实验 | 现象 | 可能原因 | 价值 |
|---|---|---|---|
| `type10-5 B02_mixstyle_random_td_t1` | best-joint 88.22，但 last 16.67，跳过 batch 极多 | 随机 MixStyle 与增强/loss 冲突，梯度异常 | 证明 MixStyle 需要受控配置 |
| `type10-5 B01_mixstyle_cd_td_t1_t2` | last 48.07 | 多层/组合混合过强 | 提醒不能只看 early best |
| `type10-5 D01_mixstyle_no_pa` | last 83.39，跳过 batch 多 | 缺少 PA 辅助后增强扰动缺乏解释出口 | 说明 PA 分支有稳定作用 |
| `type11` 某实验 | best 91.59 后崩到约 31.10 | 对抗/解耦目标过强或权重不稳 | 说明高 best 不能直接作为最终结论 |
| `type10-4` | 一些 run last 16.67，一些短 run 89.85 | 训练未完整或设置不稳定 | 不适合作为最终版本 |

---

## 13. 复现建议

推荐复现入口：

```powershell
cd D:\codefile\CV-SincNet\type10-5
python train.py --dataset wisig --wisig_pkl ./Dataset_WigSig/ManySig.pkl --wisig_domain rx_day --epochs 100 --batch_size 128 --lr 2e-4 --model_variant lite_c --mixstyle_p 0.3 --mixstyle_alpha 0.1
```

复现时必须记录：

1. random seed；
2. train/test day 和 rx 划分；
3. MixStyle 层位置、概率、alpha；
4. PA/DAC 辅助 loss 权重；
5. last epoch accuracy；
6. best joint、best overall 和 hardest split。

---

## 14. 导师可能提问与回答

| 问题 | 回答要点 | 可引用依据 |
|---|---|---|
| CVS 解决什么问题？ | 解决 RF 指纹识别在跨 day、跨 rx 条件下泛化下降的问题。 | `type10-5/dataset_wisig.py` day/rx split；`type10-5/logs` |
| 为什么需要 domain label？ | 训练需要知道哪些变化来自 day/rx，才能做域对抗、域分类和分域评估。 | `WiSigCompactDataset.__getitem__` 返回 `(x,y,d,meta)` |
| SincConv 的作用是什么？ | 用参数化频带滤波器提取 RF I/Q 低层频谱特征。 | `type10-5/model.py:SincConv1d` |
| MixStyle 为什么有效？ | 混合特征统计，降低模型对固定 day/rx 风格的依赖。 | `MixStyle1D`；D02 vs A00 |
| PA/DAC 分支为什么有帮助？ | 将硬件扰动显式建模，减少其混入身份特征。 | `PhysicalAwareClassifier`；`type10-2 g0/g4` |
| 为什么一些 MixStyle 实验崩溃？ | 随机或过强混合会与多 loss 目标冲突，导致梯度异常。 | `B02` last 16.67 和大量 skipped batches |
| 最终该看 best 还是 last？ | best 反映潜力，last 反映稳定性；最终结论需二者同时报告。 | `type11`、`B02`、`D02` 对比 |
| 最终版本是否可复现？ | 有统一 train.py、argparse、checkpoint、日志和分域评估，具备复现基础；仍需固定 seed 和数据文件。 | `type10-5/train.py` |

---

## 15. 不确定信息与待补充材料

1. 未发现或未使用完整 git commit 历史，版本顺序为推断；
2. 部分日志不完整，无法确认所有实验是否完整跑完；
3. 部分 checkpoint 只有命名信息，未重新加载模型验证；
4. 数据文件 `ManySig.pkl` 的外部来源、清洗过程和样本统计需要进一步确认；
5. 若用于论文，需要补充多 seed 均值/方差；
6. 当前结论主要基于已有日志，未重新训练全部实验。

---

## 16. 质量自检

| 检查项 | 状态 |
|---|---|
| 是否扫描代码与日志 | 已基于主要目录与日志完成 |
| 是否识别明显版本 | 已覆盖 `type1-type15` 与 `type10-*` |
| 是否区分 last 与 best | 已区分 |
| 是否说明失败实验 | 已说明 |
| 是否聚焦模块实现与原理 | 已按数据、增强、模型、loss、训练、评价拆解 |
| 是否给出最终成果 | 已给出 D02 结果 |
| 是否标注不确定推断 | 已标注 |

