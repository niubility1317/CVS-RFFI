# 当前最优版本论文级技术报告

日期：2026-05-08

## 1. 最优版本定义

当前 5.8 训练日志中的开发集最优版本为：

- 实验名：`E2_residual_only_std_res001`
- 模型文件：`finalist_runs/E2_residual_only_std_res001/best_model_primary_ood.pth`
- 初始化来源：`finalist_runs/D1_domain_enhancer_off_seed1337/best_model_primary_ood.pth`
- 训练路线：`D1 source training -> E2 residual-only SGC fine-tuning`
- 选择指标：`FINAL-PRIMARY score = 0.35 * overall + 0.65 * strict_UDU`

核心结果：

| 指标 | 数值 |
|---|---:|
| Primary score | 88.24 |
| Overall TX accuracy | 90.70 |
| Strict unseen-day unseen-RX accuracy | 86.92 |
| Worst unseen-RX accuracy | 86.99 |
| Validation TX accuracy | 98.90 |
| SAT Avg on strict UDU | 41.58 |
| SAT Min on strict UDU | 37.27 |

严格说明：该版本是当前日志内的“开发选择最优”，不是最终无偏 test 结论。因为 `best_model_primary_ood.pth` 和 Phase-E 的 source 选择使用了 test-derived 指标。论文最终表格应使用 validation-only source/checkpoint selection 或 fresh holdout 重新确认。

## 2. 数据划分与任务设定

数据集为 WiSig `ManySig.pkl`，输入 IQ 序列长度为 `256`，类别数从默认 16 覆盖为 6 类发射机分类。域标签采用 `rx_day`，即 receiver 与日期的联合域。

训练域：

- Train days：`2021_03_01`, `2021_03_08`
- Train RX：`0,1,2,3,4,5,6`
- Train samples：`16800`
- Validation：同一 train days/train RX 的尾部切分，使用 `guard_gap=8` 避免相邻片段泄露，`val=66528`

测试域：

- Unseen-day seen-RX：新日期，旧接收机
- Seen-day unseen-RX：旧日期，新接收机
- Unseen-day unseen-RX：新日期，新接收机，是最严格 OOD 条件
- Per-RX tests：`rx7` 到 `rx11`，用于衡量弱接收机性能和 worst-RX

该任务的核心难点不是常规闭集 TX 分类，而是发射机身份特征在接收机链路、采集日期、卫星链路扰动下是否保持可分。

## 3. 总体方法概述

当前最优路线可以概括为：

> 使用轻量级 dual CV-SincNet 进行发射机身份与接收机/日期域因素解耦；主干保留时间、频域、PA 物理分支并去除 DAC 分支；域分支使用独立第二骨干提取域表征；训练时结合 GRL 域对抗、same-TX cross-domain consistency、hard-domain CE、conservative MixStyle、SAT mixed-orbit consistency、Fishr 梯度方差匹配；最后从 D1 source checkpoint 低学习率微调 residual-only SGC adapter。

从实验结果看，最终收益最大的是：

1. `rxrobust_lite_b_no_dac_mix015` 的保守 MixStyle + hard-domain CE source training。
2. D1 的 `domain_enhancer=off` source checkpoint。
3. E2 的 residual-only SGC adapter + `lambda_res=0.01` 微调。

需要特别注意：E2 构图时又按默认 slim group 重新启用了 `domain_enhancer=rcn_stats`。因此 E2 并非单纯“D1 + residual SGC”，而是：

> D1 权重初始化 + 新增 residual-only SGC adapter + 重新引入 RCN-stat domain enhancer + 全模型低学习率微调。

## 4. 主干模型结构

### 4.1 DualCVSincNetDisentangle

最优版本使用 `DualCVSincNetDisentangle`。该模型包含两个 CV-SincNet 骨干：

- `id_backbone`：发射机身份主干，输出 TX logits 与身份特征 `z_id`。
- `dom_backbone`：域特征主干，输出域表征 `z_dom`。
- `dom_head`：预测训练域 `rx_day`。
- `adv_head`：对 `z_id` 经过 GRL 后预测域，用于反向去域。

前向逻辑为：

1. 输入 IQ 先经过可选 SGC adapter。
2. `id_backbone(x)` 得到 `tx_logits` 和 `z_id`。
3. `dom_backbone(x)` 得到 `z_dom_raw`。
4. `DomainFeatureEnhancer` 可将 RCN 统计特征融合到 `z_dom`。
5. `dom_head(z_dom)` 执行显式域分类。
6. `adv_head(GRL(z_id))` 执行身份特征去域。

对于 Lite-B/Lite-D/Lite-E，两个骨干共享最低层 `sinc` 与 `hf` stem，但不共享后续表征块。这一设计使低层滤波器稳定，同时保留身份分支与域分支的解耦空间。

### 4.2 ID 主干：Lite-B no-DAC CV-SincNet

当前最优使用 `model_variant=lite_b`，`branch_ablation=no_dac`。Lite-B 的关键通道配置为：

- `sinc_out=24`
- `time_bottleneck=72`
- `emb_dim=192`
- `freq_bands=36`
- time path channels：`96 -> 144 -> 144`
- frequency path channels：`24 -> 48 -> 48`
- PA path channels：`72 -> 96 -> 96`
- DAC branch：禁用

ID 主干保留三类信息通道：

1. 时间路径：SincConv filterbank + 高频差分 + 非线性基，进入 depthwise-separable temporal CNN。
2. 频域路径：mirror-compressed frequency features，包含正/负频谱功率、谱比和不对称性。
3. PA 路径：memory-polynomial lift 与 envelope-aware dilated convolution，用于建模 PA 非线性。

DAC 分支被移除是当前路线的一个核心经验结论。原因是 WiSig 该 split 中 DAC-like 细节容易与接收机/采集链路纠缠，给 unseen RX 带来过拟合风险；保留 PA 分支则能提供更稳定的发射机非线性特征。

### 4.3 分类头：PhysicalAwareClassifier

分类头显式构造：

- `feat_id`：身份嵌入
- `feat_pa`：PA 相关嵌入
- `feat_dac`：DAC 相关嵌入，当前 ID 分支中禁用
- `feat_imp`：impairment 合并嵌入
- `feat_joint`：最终用于 CosFace 分类的 joint identity feature

TX 分类采用 CosFace head，即归一化特征与类别权重后加入 margin。它适合 RFFI 的小类数、高相似度类间区分，因为 margin 可以促使发射机嵌入形成更清晰的角度间隔。

### 4.4 第二骨干：域特征分支

当前配置中，第二骨干并不是完全同构主干，而是：

- `domain_branch_ablation=no_stats`
- `domain_enhancer=rcn_stats`
- `domain_enhancer_strength=0.35`

也就是说，域骨干内部移除了 handcrafted stats projection，但在骨干输出后又通过 `DomainFeatureEnhancer` 注入 RCN 统计编码。该设计有两个效果：

1. `dom_backbone` 仍可从原始时间/频率/PA/DAC-like 结构中学习域因素，避免完全依赖人工统计。
2. `RCNStatEncoder` 作为 gated residual enhancer，只在域路径中强化接收机/信道/噪声统计，使域特征更可分。

不过实验显示，source 阶段 D1 使用 `domain_enhancer=off` 反而是 PRE 最强，这说明 RCN stats enhancer 有过度域化风险。E2 的提升不能简单归因于 enhancer，因为它同时引入 residual SGC 和低学习率微调。因此论文中应把“第二骨干简化 + RCN enhancer”作为创新点，但必须用 ablation 支撑。

## 5. SGC Adapter 与残差设计

E2 使用 residual-only SGC：

```json
{
  "use_amp_norm": false,
  "use_freq_comp": false,
  "use_spectral_suppressor": false,
  "use_residual_comp": true,
  "residual_channels": 32,
  "residual_blocks": 2,
  "residual_kernel_size": 5,
  "residual_init_gamma": 0.0
}
```

ResidualChannelCompensator 为 depthwise-separable residual CNN：

```text
x_out = x + gamma * F_res(x)
```

其中 `gamma` 是可学习标量，初始化为 0。该设计有非常重要的科研意义：

- 初始化时严格等价于原模型输入，不破坏 source checkpoint。
- 训练只能逐步学习小幅通道补偿，降低 SGC 过拟合卫星模拟器的风险。
- `lambda_res=0.01` 约束 `gamma` 或 adapter delta，使补偿保持保守。

E2 相对 E0/no-adapter continue 的提升：

| 对比 | Primary | Overall | Strict UDU | Worst RX | SAT Avg |
|---|---:|---:|---:|---:|---:|
| E0 no adapter | 87.84 | 90.37 | 86.47 | 86.51 | 40.85 |
| E2 residual-only + res reg | 88.24 | 90.70 | 86.92 | 86.99 | 41.58 |
| 差值 | +0.40 | +0.33 | +0.45 | +0.48 | +0.73 |

结论：SGC 全模块没有稳定收益，但 residual-only SGC 有明确作用，尤其提升 strict UDU 和 worst-RX，这说明“保守残差补偿”比“显式频偏/幅度/谱抑制全修正”更适合当前数据。

## 6. 损失函数设计

训练总损失可写作：

```text
L =
  L_cls
  + w_dom L_dom
  + w_adv L_adv
  + w_orth L_orth
  + w_cons L_cons
  + w_group L_group
  + aux_scale * L_PA_aux
  + lambda_sat_cls L_sat_cls
  + lambda_sat_cons L_sat_cons
  + lambda_fishr L_fishr
  + lambda_res L_sgc_res
  + lambda_ecc(epoch) L_ecc
```

当前 E2 中实际有效的主要项为：

| Loss | 作用 | 基础权重/配置 |
|---|---|---:|
| `L_cls` | TX 分类 CE/CosFace logits 上的 CE | 1.0 |
| `L_dom` | 让 `z_dom` 可预测 `rx_day` | `lambda_dom=1.0` |
| `L_adv` | GRL 后让 `z_id` 去除域信息 | `lambda_adv=0.45` |
| `L_orth` | `z_id` 与 `z_dom` 协方差正交 | `lambda_orth=0.05` |
| `L_cons` | 同 TX 跨域身份中心一致性 | `lambda_cons=0.08` |
| `L_group` | hard-domain CE | `lambda_group_ce=0.10` |
| `L_PA_aux` | PA-only view 的分类/一致性/强度回归 | PA-only，DAC aux 为 0 |
| `L_sat_cls` | 卫星视图分类 CE | `lambda_sat_cls=0.08` |
| `L_sat_cons` | clean 与 SAT view 的 `z_id` cosine consistency | `lambda_sat_cons=0.04` |
| `L_fishr` | 域间 logit-gradient 方差匹配 | `lambda_fishr=0.02` |
| `L_sgc_res` | residual SGC 保守约束 | `lambda_res=0.01` |
| `L_ecc` | early confidence cap | E2 未启用，权重为 0 |

### 6.1 Hard-domain CE

Hard-domain CE 在训练 batch 内按 domain 分组计算 CE，取损失最高的 top 35% 训练域求均值。这里的 domain 是训练集中的 `rx_day`，不是测试集 RX。它的目的不是直接“知道 rx7/rx8 差”，而是在训练时持续压制最难的 source receiver/day 组合，从机制上提升未来 unseen RX 的 worst-case 表现。

数学形式：

```text
L_group = mean_topk({ CE(logits_i, y_i) | samples i in source domain d }, top_frac=0.35)
```

它提升差类别/差域性能的方式是：把 batch-average ERM 改成 worst-source-domain-biased ERM，使模型不能只优化易接收机/易日期组合。

### 6.2 域解耦损失

域解耦由三部分形成闭环：

1. `L_dom`：要求 `z_dom` 能预测 domain，使第二骨干确实捕获 receiver/date 变化。
2. `L_adv`：使用 GRL 要求 `z_id` 难以预测 domain，使身份特征去域。
3. `L_cons` / `L_orth`：同 TX 跨域特征拉近，且 `z_id` 与 `z_dom` 降低线性相关。

这比单纯 GRL 更稳，因为 `z_dom` 有显式去处，域信息被“引导到域分支”，而不是被迫从整个网络中消失。

### 6.3 SAT mixed consistency

训练时从主视图生成卫星链路视图：

- Scenario：`mixed_orbit`
- View source：`main`
- Start epoch：20
- `lambda_sat_cls=0.08`
- `lambda_sat_cons=0.04`

SAT simulator 包含 LEO/MEO/GEO 轨道采样、仰角-斜距耦合、路径损耗、天气/遮挡、多径、CFO、相位噪声、IQ imbalance 和 AGC residual。

但需要严谨区分：E2 最佳 checkpoint 出现在 epoch 5，早于 E2 阶段 SAT consistency 启动。因此 E2 最佳点的直接收益主要来自 D1 source 已学到的 SAT/Fishr 鲁棒性，以及 E2 早期 residual SGC/core loss 微调；不能说 E2 的后续 SAT consistency 是本 checkpoint 的直接提升来源。

### 6.4 Fishr

Fishr 使用一个轻量代理：计算 `prob - one_hot` 作为分类器 logit gradient proxy，然后匹配不同训练域上的梯度方差。该项降低某些 source domain 独有梯度方向对模型更新的支配，目标是提升域外一致性。

当前配置：

- `lambda_fishr=0.02`
- `fishr_min_domains=4`

### 6.5 PA-only auxiliary learning

由于主干 `no_dac`，DAC auxiliary 全部关闭。PA auxiliary 在 S2/S3 阶段启动：

- `lambda_cls_pa=0.20`
- `lambda_pa_joint_inv=0.06`
- `lambda_pa_kl=0.02`
- `lambda_pa_reg=0.10`
- normal view 中 `aug_p_pa=0.14`
- PA-only view 不叠加 channel 和 anti-shortcut，保持纯 PA 监督

目的：

1. 让模型识别 PA 非线性增强下的同一 TX。
2. 约束 PA-only view 的 `feat_joint` 接近 clean `feat_joint`。
3. 用 teacher clean logits 做 KL，减少增强视图决策漂移。
4. 回归 PA strength，迫使 PA 分支对物理扰动强度有响应。

## 7. 训练策略

### 7.1 两阶段路线

第一阶段 D1：

- 从零训练 200 epoch。
- 配置：`rxrobust_lite_b_no_dac_mix015`
- 额外：`--force_domain_enhancer off`
- 最优 source checkpoint：epoch 78 的 `best_model_primary_ood.pth`
- Primary：87.87

第二阶段 E2：

- 从 D1 `best_model_primary_ood.pth` 初始化。
- 训练 60 epoch。
- 学习率：`5e-5`
- SGC stage：`sgc_augment`
- Adapter：residual-only
- `lambda_res=0.01`
- 最优 checkpoint：epoch 5
- Primary：88.24

`sgc_augment` 不是 adapter-only freeze 训练；代码中只有 `sgc_adapt` 会冻结全模型只训 adapter。E2 的 `sgc_augment` 是全模型低学习率微调，因此收益来自 residual adapter 与主干小步联合调整。

### 7.2 Stagewise schedule

训练损失按 epoch 分三阶段：

- S1 core：只训练核心分类、域分支、GRL、正交和 hard-domain CE，辅助视图关闭。
- S2 stabilize aux：逐步打开 PA auxiliary，逐渐增强 consistency、aux CE、reg、KL。
- S3 refine aux：辅助项接近满权重，用于细化物理鲁棒性。

对于 E2 的 60 epoch：

- S1：epoch 1-16
- S2：epoch 17-56
- S3：epoch 57-60

由于最佳点在 epoch 5，E2 的 best checkpoint 实际处于 S1 core 阶段。这说明当前最优更接近“低学习率保守校准”而不是“长程辅助训练完全收敛”的结果。

### 7.3 Conservative MixStyle

MixStyle 配置：

- 开启层：`time_down,t1`
- `p=0.15`
- `alpha=0.10`
- `strength=0.65`
- pairing：`same_tx_crossdomain`
- fallback：`skip`
- late anneal：start 110, ramp 35, min p 0.05, min strength 0.35

设计逻辑：

- 只混合时间早期特征统计，不直接混合最终身份特征。
- 要求同 TX、不同 source domain，避免把不同发射机的指纹统计混在一起。
- fallback skip 避免 batch 中找不到合法 pairing 时做随机破坏。
- 保守 p/strength 防止 no-stats sensitive domain 下过强 style perturbation。

### 7.4 数据增强

主视图增强包括：

- PA normal augmentation：`p_pa=0.14`
- 时间平移
- 幅度缩放
- 随机相位旋转
- CFO
- 相位噪声
- AWGN
- 轻量多径
- DC offset / bandedge taper 等 anti-shortcut 变换

增强强度从 `0.10` ramp 到 `0.35`，warmup 3 epoch，ramp 15 epoch，curve 1.25。该 schedule 让模型先建立基本 TX 判别，再逐渐暴露于更强扰动。

## 8. 当前最优路线的创新点

### 创新点一：身份-域双骨干解耦，而不是单分支域对抗

传统 GRL 只惩罚身份特征里的域可分性，但没有明确告诉模型域信息应该去哪里。当前模型通过第二骨干显式承载 `rx_day` 域信息，再用 GRL/orth/cons 将身份特征与域特征分离。这是更完整的 disentanglement 框架。

### 创新点二：物理感知但保守的 no-DAC 路线

模型不是盲目堆叠所有硬件分支，而是根据 WiSig split 的泛化风险移除 DAC 分支，保留更稳定的 PA 非线性路径。该路线将“硬件物理先验”与“域泛化风险控制”结合起来。

### 创新点三：same-TX cross-domain MixStyle

常规 MixStyle 可能破坏 RFFI 中真实发射机指纹。当前 MixStyle 限制为同一 TX、不同 source domain 的 style mixing，把扰动限定在 receiver/date 风格层面，更符合 RFFI 的物理语义。

### 创新点四：residual-only SGC 作为保守通道校准器

完整 SGC 试图显式归一化幅度、补偿频偏、抑制频谱干扰，但容易过拟合模拟器假设。Residual-only SGC 使用零初始化残差补偿，仅学习必要的小幅输入修正，更适合 source checkpoint 继承和 OOD 稳定性。

### 创新点五：hard-domain CE + Fishr 的 worst-domain 优化

Hard-domain CE 聚焦高损 source domain，Fishr 匹配域间梯度方差。二者分别从风险目标和优化动态上抑制 easy-domain dominance，对 weak unseen RX 有直接价值。

## 9. 局限与必须补充的验证

1. 当前最优 checkpoint 是 test-derived primary 选择，不能直接作为最终无偏 test。
2. E2 的最佳点在 epoch 5，SAT consistency 和 PA auxiliary 在 E2 中尚未真正发挥；后续应单独报告 source inherited robustness 与 E2 微调贡献。
3. E2 同时改变 residual SGC、domain enhancer 和低学习率全模型微调，当前还不能完全隔离 residual SGC 的因果作用。
4. D1 source 显示 `domain_enhancer=off` 最强，而 E2 又 re-enable enhancer；这需要专门 ablation：
   - D1 source + residual SGC + enhancer off
   - D1 source + residual SGC + enhancer on
   - D1 source + no adapter + enhancer on
   - D1 source + adapter-only frozen SGC
5. 当前仅 seed 1337，论文主结论至少应补 3 seed。

## 10. 推荐论文表述

可以将当前方法命名为：

**Residual Satellite-Channel Calibration with Dual-Backbone Physical-Aware Disentanglement**

或更贴近现有命名：

**CVS-RFFI-RSGC: Receiver-Robust CV-SincNet with Residual Satellite-Ground Channel Adaptation**

建议摘要级表述：

> We propose a receiver-robust RFFI framework that disentangles transmitter identity and receiver-date domain factors using a dual CV-SincNet architecture. The identity path adopts a compact PA-aware no-DAC backbone, while a second domain path explicitly models receiver-date variations. Domain robustness is enforced through adversarial domain removal, same-transmitter cross-domain consistency, hard-domain cross entropy, conservative same-TX MixStyle, and Fishr-style gradient variance matching. To improve satellite-ground robustness without disrupting source-domain fingerprints, we introduce a zero-initialized residual SGC adapter that learns conservative IQ-domain compensation. On the 5.8 development run, this route achieves the best primary OOD score, improving strict unseen-day unseen-RX accuracy and worst-RX robustness over no-adapter continuation.

## 11. 当前可引用的关键文件

- `model_dual_cvsincnet.py`：dual backbone、GRL、second domain backbone、SGC adapter 插入位置。
- `model.py`：CVSincNet 单骨干、SincConv、time/frequency/PA/DAC 分支、CosFace physical-aware classifier。
- `train.py`：loss 组合、stagewise schedule、hard-domain CE、Fishr、SAT consistency、SGC training stage。
- `sgc_adapter.py`：residual-only SGC 与 full SGC 子模块。
- `sgc_losses.py`：residual regularization。
- `run_final_best_sgc_queue.sh`：D1/E2 实验启动配置。
- `5.8/metrics/experiment_metrics.csv`：所有实验的扁平指标。
- `5.8/logs/E2_residual_only_std_res001.log`：当前最优模型训练日志。

