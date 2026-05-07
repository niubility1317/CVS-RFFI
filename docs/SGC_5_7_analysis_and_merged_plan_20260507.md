# 5.7 SGC 日志综合分析与下一步训练路线

日期: 2026-05-07  
范围: `5.7/` 最新代码快照、`5.7/logs/` 训练日志、既有最佳模型报告 `docs/CVS_RFFI_model_route_report_20260506.md`、既有计划 `docs/superpowers/plans/2026-05-06-lite-b-sat-mixed-fishr.md`

## 结论摘要

`5.7` 的日志证明了 SGC-Adapter 训练链路已经能完整跑通，但还没有证明 SGC 是当前最优模型路线。当前最稳妥的一线方案仍应是:

```text
Lite-B no-DAC + conservative MixStyle + SAT mixed consistency + Fishr
```

其中 SAT mixed consistency 要用温和权重和延迟启动，而不是 5.7 SGC augment 日志里的 `lambda_cls=1.0`、`lambda_cons=1.0`、`start_epoch=1`。

本轮应该把目标拆成两层:

1. 先跑出一个第一最佳模型: `Lite-B no-DAC + MixStyle + Fishr + mild SAT mixed consistency`。
2. 再把 SGC 作为第二阶段增强路线验证，优先只试 `no_amp` adapter，并且显式使用小 SAT 权重。

不建议继续把 full SGC adapter 从 source 阶段直接训练成主线，也不建议把现有 `sgc_adapt` 当成最终适配路线。

## 证据范围

### 代码状态

`5.7` 内关键代码文件与根目录当前版本一致:

| 文件 | 结论 |
|---|---|
| `train.py` | 与根目录 `train.py` SHA256 一致 |
| `sgc_adapter.py` | 与根目录 `sgc_adapter.py` SHA256 一致 |
| `model_dual_cvsincnet.py` | 与根目录 `model_dual_cvsincnet.py` SHA256 一致 |

因此 `5.7` 不是一条新的代码分支，而是当前代码上的新一批 SGC 训练日志。

### 日志状态

`5.7/logs` 共有 32 个 `.log`，全部完整:

| stage | 数量 | 状态 |
|---|---:|---|
| source | 11 | complete |
| augment | 11 | complete |
| adapt | 10 | complete |

这批日志是目前 SGC 路线最有价值的实证材料。

## SAT mixed consistency 是什么

SAT mixed consistency 指的是在训练时构造一个“卫星信道视图”，让模型同时学习原始/常规增强样本和卫星信道增强样本，并约束两种视图的 ID 表征保持一致。

在当前代码里，它主要包含两项损失:

| 组成 | 参数 | 作用 |
|---|---|---|
| SAT classification | `lambda_sat_cls` | 对卫星信道增强后的样本继续做发射机 ID 分类，避免卫星信道下类别判别直接崩掉 |
| SAT feature consistency | `lambda_sat_cons` | 用 cosine consistency 约束卫星视图的 `z_id` 与主视图 `z_id` 接近，减少信道变化对 ID 表征的扰动 |

`mixed_orbit` 是训练用的卫星场景组合，比单一 `clear_leo` 更接近综合压力测试。它会混合不同轨道/低仰角/降雨/多径等信道扰动，使模型不要只适配一种干净卫星条件。

关键点是权重必须保守。RFFI 的目标不是通信接收机里的“尽量消除所有信道影响”，而是在增强鲁棒性的同时保留发射机硬件指纹。SAT consistency 如果过强，会把本该保留的细粒度 RF fingerprint 也压平。

## 5.7 结果总览

## SGC 模块到底有没有起作用

严谨结论: SGC 模块“有作用”，但当前不是稳定正作用。它确实改变了模型在卫星信道和 clean/OOD 上的行为；不过从 5.7 日志看，full SGC adapter 的净效果偏负，SAT Avg 的主要提升来自 SAT augment/consistency 训练本身，而不是 SGC adapter 本体。最值得保留的是 `no_amp` 形式的弱正向信号，而不是 full SGC。

这个判断要拆成四个对照。

### 对照 1: 无 SGC adapter，仅加入 SAT augment

`sgc_baseline_no_adapter` 的 source 到 augment:

| 对照 | Primary | strict UDU | overall | worst-RX | SAT Avg |
|---|---:|---:|---:|---:|---:|
| source | 88.22 | 85.91 | 90.53 | 85.07 | 38.31 |
| augment | 87.12 | 84.31 | 89.93 | 81.29 | 44.62 |
| 变化 | -1.10 | -1.60 | -0.60 | -3.78 | +6.31 |

这说明 SAT augment/consistency 本身已经能明显提高 SAT Avg，但代价是 clean/OOD 和 worst-RX 下滑。这个 +6.31 不能归功于 SGC，因为该实验没有 adapter。

### 对照 2: source 阶段启用 SGC adapter

与无 adapter baseline source 比较:

| preset | Primary 变化 | strict UDU 变化 | overall 变化 | worst-RX 变化 | SAT Avg 变化 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| full `sgc_lite_b_no_dac` | -1.92 | -1.96 | -1.89 | -7.92 | -3.34 | 明显负贡献 |
| `no_amp` | -1.23 | -0.64 | -1.83 | -11.21 | -2.94 | 仍为负，UDU 伤害较小 |
| `no_freq` | -2.06 | -2.14 | -1.97 | -2.63 | -2.90 | 负贡献 |
| `no_res` | -3.03 | -3.78 | -2.28 | -1.44 | -8.66 | 负贡献，SAT 很差 |
| residual-only | -2.50 | -2.67 | -2.34 | -3.14 | -0.14 | clean 负贡献 |
| Lite-D full | -0.99 | -1.29 | -0.69 | -4.04 | -4.59 | 比 Lite-B 轻，但仍负 |

结论: 在 source 阶段，SGC adapter 不是帮助模型学到更稳的发射机指纹，而是整体拉低 clean/OOD；full SGC 的 worst-RX 下降尤其明显。

### 对照 3: augment 阶段 SGC adapter 的边际贡献

把各个 SGC augment 与无 adapter augment 比较:

| preset | Primary 变化 | strict UDU 变化 | overall 变化 | worst-RX 变化 | SAT Avg 变化 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| full `sgc_lite_b_no_dac` | -1.69 | -1.49 | -1.89 | -5.87 | +1.49 | SAT 略增但 clean 代价过大 |
| `no_amp` | +0.24 | +0.63 | -0.15 | -4.32 | -0.58 | 有弱 clean/UDU 正信号，但 worst-RX 不合格 |
| `no_amp_freq` | -1.99 | -1.77 | -2.20 | -0.01 | +4.33 | SAT 上限探针，不适合主线 |
| `no_freq` | -1.65 | -1.31 | -2.00 | -1.17 | -0.62 | 负贡献 |
| `no_res` | -1.54 | -1.18 | -1.91 | +1.68 | +0.37 | worst 有改善但整体不足 |
| `no_spec` | -1.12 | -0.20 | -2.04 | -0.82 | +0.17 | 有 collapse 风险 |
| residual-only | -1.33 | -1.25 | -1.40 | +0.16 | +1.81 | SAT 小增但 clean 掉 |
| Lite-D full | +0.13 | +0.50 | -0.24 | -3.03 | -2.45 | compact 可观察，但非主线 |
| Lite-D light | -0.07 | +1.26 | -1.40 | -1.01 | -0.38 | UDU 有信号，overall 弱 |

这个对照说明:

1. full SGC 确实让 SAT Avg 比无 adapter augment 多了 +1.49，但 Primary、UDU、overall、worst-RX 全掉，净效果不划算。
2. `no_amp` 是唯一在 Primary 和 strict UDU 上略好于无 adapter augment 的 Lite-B SGC 变体，但 SAT Avg 没有更好，worst-RX 还低 4.32。
3. `no_amp_freq` 的 SAT Avg 多 +4.33，证明 SGC 子模块能强烈改变卫星压力表现；但 clean/OOD 同时掉，说明它不是“免费鲁棒性”。

### 对照 4: SGC adapt 是否有效

SGC adapt 的平均表现低于 augment:

| stage | SGC runs 数量 | Primary 均值 | strict UDU 均值 | overall 均值 | worst-RX 均值 | SAT Avg 均值 |
|---|---:|---:|---:|---:|---:|---:|
| source | 10 | 85.88 | 83.52 | 88.24 | 79.74 | 35.01 |
| augment | 10 | 85.98 | 83.61 | 88.35 | 79.44 | 44.94 |
| adapt | 10 | 84.10 | 80.96 | 87.23 | 76.89 | 42.59 |

adapt 阶段日志还显示:

```text
trainable=1,493
LOSS-SAT cls_sat=0.0000 sat_cons=0.0000
```

这意味着当前 adapt 只是在冻结主干后训练 adapter，并没有继续 SAT consistency。结果多数 preset 从 augment 到 adapt 是退化的。

结论: 当前 `sgc_adapt` 没有证明有效，应暂停作为最终阶段。

### 对 SGC 的最终判定

| 问题 | 判定 |
|---|---|
| SGC 有没有产生可观测影响？ | 有。不同 adapter 变体显著改变 Primary、worst-RX 和 SAT Avg。 |
| SAT 提升是否主要来自 SGC？ | 不是。无 adapter augment 已经让 SAT Avg +6.31，说明主要贡献来自 SAT augment/consistency。 |
| full SGC 是否是净正收益？ | 不是。source 与 augment 下 clean/OOD 和 worst-RX 代价过大。 |
| 哪个 SGC 方向最值得保留？ | `no_amp`，因为它在 augment 对照中 Primary +0.24、UDU +0.63，但必须解决 worst-RX 下滑。 |
| 当前 adapt 是否有效？ | 否。它没有 SAT loss，平均退化。 |
| SGC 能否进入下一轮？ | 可以，但只能作为第二阶段验证，不能作为第一最佳模型主线。 |

所以更准确的说法是:

```text
SGC 模块不是无效模块，而是当前配置下过度干预了 RFFI 指纹。
它对卫星域鲁棒性有可观测影响，但没有形成 clean/OOD + SAT 的稳定净增益。
下一步应验证 no_amp + mild SAT + 强主干 checkpoint，而不是继续 full SGC from scratch。
```

### Top by Primary OOD

| 排名 | preset | stage | Primary | strict UDU | overall | worst-RX | SAT Avg | 备注 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `sgc_baseline_no_adapter` | source | 88.22 | 85.91 | 90.53 | 85.07 | 38.31 | 5.7 clean/OOD 最强，不含 SGC adapter |
| 2 | `sgc_lite_b_no_dac_no_amp` | augment | 87.36 | 84.94 | 89.78 | 76.97 | 44.04 | SGC 中最接近可用的一条，但 worst-RX 明显掉 |
| 3 | `sgc_lite_d_no_dac` | augment | 87.25 | 84.81 | 89.69 | 78.26 | 42.17 | Lite-D 紧凑候选 |
| 4 | `sgc_lite_d_no_dac` | source | 87.23 | 84.62 | 89.84 | 81.03 | 33.72 | clean 尚可，SAT 弱 |
| 5 | `sgc_baseline_no_adapter` | augment | 87.12 | 84.31 | 89.93 | 81.29 | 44.62 | 无 adapter 的强 SAT augment 基线 |
| 6 | `sgc_lite_d_no_dac` | adapt | 87.06 | 84.63 | 89.49 | 78.20 | 46.54 | SAT 提升，clean/worst 不够 |
| 7 | `sgc_lite_d_no_dac_light` | augment | 87.05 | 85.57 | 88.53 | 80.28 | 44.24 | UDU 高但 overall 弱 |
| 8 | `sgc_lite_b_no_dac_no_amp` | source | 86.99 | 85.27 | 88.70 | 73.86 | 35.37 | no-amp 比 full SGC 更稳，但 worst 很低 |

### Top by SAT Avg

| 排名 | preset | stage | SAT Avg | Primary | strict UDU | overall | 备注 |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `sgc_lite_b_no_dac_no_amp_freq` | augment | 48.95 | 85.13 | 82.54 | 87.73 | SAT 压力下最强，但 clean/OOD 不可作为主线 |
| 2 | `sgc_lite_b_no_dac_no_amp` | adapt | 47.79 | 86.20 | 83.37 | 89.03 | SAT 好，clean/UDU/worst 掉 |
| 3 | `sgc_lite_d_no_dac` | adapt | 46.54 | 87.06 | 84.63 | 89.49 | 作为 compact SAT 候选可保留 |
| 4 | `sgc_lite_b_no_dac_residual_only` | augment | 46.43 | 85.79 | 83.06 | 88.53 | residual-only 有 SAT 帮助但 clean 不够 |
| 5 | `sgc_lite_b_no_dac` | augment | 46.11 | 85.43 | 82.82 | 88.04 | full SGC augment 不适合主线 |

## 与 5.6 最佳模型路线的对照

既有最佳日志中最重要的参照:

| run | Primary | strict UDU | overall | worst-RX | SAT Avg | 定位 |
|---|---:|---:|---:|---:|---:|---|
| `SAT37_r19_fishr` | 87.95 | 86.43 | 90.77 | 84.64 | 38.91 | 旧证据下的一线最优 |
| `SAT34_r19_groupdro_smooth` | 87.94 | 86.44 | 90.72 | - | - | clean/OOD 极强备选 |
| `SAT07_r25_compact_sat_mixed` | 87.85 | 86.27 | 90.79 | - | 41.98 | compact/SAT 候选 |
| `sgc_baseline_no_adapter source` | 88.22 | 85.91 | 90.53 | 85.07 | 38.31 | 5.7 最强 clean/OOD，但未用 0.65 UDU 权重 |
| `sgc_lite_b_no_dac_no_amp augment` | 87.36 | 84.94 | 89.78 | 76.97 | 44.04 | SGC 当前最强可用候选，但 clean/worst 低 |

解释:

1. `sgc_baseline_no_adapter source` 在 5.7 的 Primary 高，是因为当前日志默认 `primary_udu_weight=0.50`，而上一轮最佳计划要求 `0.65`。它不能直接替代上一轮 `SAT37_r19_fishr`，但说明当前代码下无 SGC 的 Lite-B no-DAC 基线仍然很强。
2. SGC 的价值主要体现在 SAT Avg 提升，例如 `no_amp` augment 从 source SAT Avg 35.37 提到 44.04。但它同时显著牺牲 worst-RX 和 strict UDU。
3. 第一最佳模型应该从已验证 clean/OOD 路线出发，把 SAT mixed consistency 温和并入，而不是把 SGC adapter 直接推到主线。

## 被日志证伪或应暂停的方向

### 1. full SGC adapter 从 source 阶段直接训练

`sgc_lite_b_no_dac source`:

```text
Primary 86.30, strict UDU 83.95, overall 88.64, worst-RX 77.15, SAT Avg 34.97
```

对比 `sgc_baseline_no_adapter source`:

```text
Primary 88.22, strict UDU 85.91, overall 90.53, worst-RX 85.07, SAT Avg 38.31
```

full SGC adapter 在 source 阶段就进入主干前端，会过早改变原始 IQ 的幅度、频偏和频谱结构。对 RFFI 来说，这些结构里既有信道扰动，也有发射机硬件指纹。source 阶段直接训练 full adapter 等于在 ID 表征稳定之前先动了指纹入口，方向不稳。

结论: 暂停 full SGC from scratch 主线。

### 2. SAT augment 权重 1.0/1.0 且 epoch 1 启动

5.7 launcher 在 augment 阶段传入:

```text
--lambda_feat 1.0
```

而 `train.py` 在 `sgc_augment` 下会把未显式指定的 SAT 权重推成:

```text
lambda_sat_cons = lambda_feat = 1.0
lambda_sat_cls = 1.0
sat_cons_start_epoch = 1
```

日志也显示:

```text
[SAT-TRAIN] scenario=mixed_orbit view_source=main lambda_cons=1.0000 lambda_cls=1.0000 start_epoch=1
```

这解释了为什么 SAT Avg 能上到 44-49，但 clean/OOD 掉得明显。上一轮最佳计划设定的是:

```text
lambda_sat_cls = 0.08
lambda_sat_cons = 0.04
sat_cons_start_epoch = 20
```

结论: `1.0/1.0 from epoch 1` 是过强正则，不能作为最佳模型路线。

### 3. 现有 sgc_adapt 作为最终适配阶段

当前 `sgc_adapt` 会冻结非 adapter 参数，只训练 adapter:

```text
trainable = 1,493
```

同时日志中 `LOSS-SAT` 始终为:

```text
cls_sat=0.0000 sat_cons=0.0000
```

也就是说 adapt 阶段并没有继续执行 SAT mixed consistency，只是在冻结主干后靠 adapter、prototype/residual 等项适配。结果多数路线 clean/OOD 下滑:

| preset | augment Primary | adapt Primary | 变化 |
|---|---:|---:|---:|
| `sgc_lite_b_no_dac` | 85.43 | 84.70 | -0.73 |
| `sgc_lite_b_no_dac_no_amp` | 87.36 | 86.20 | -1.16 |
| `sgc_lite_b_no_dac_no_res` | 85.58 | 82.26 | -3.32 |
| `sgc_lite_d_no_dac` | 87.25 | 87.06 | -0.19 |

结论: 暂时不要把 adapt 放入第一最佳模型路线。若后续要做，需要重新设计为 SAT-view adaptation 或目标域无标签 adaptation，而不是当前 source-only adapter 微调。

### 4. AmplitudeNormalizer 作为 RFFI 主线组件

no-amp 版本比 full SGC 明显更好:

| 对比 | full SGC | no_amp SGC | 结论 |
|---|---:|---:|---|
| source Primary | 86.30 | 86.99 | no_amp 更稳 |
| source strict UDU | 83.95 | 85.27 | no_amp 更稳 |
| augment Primary | 85.43 | 87.36 | no_amp 明显更好 |
| augment strict UDU | 82.82 | 84.94 | no_amp 明显更好 |

RMS 幅度归一化可能擦掉一部分发射机功放、增益链路、采集幅度统计中的有用指纹。通信任务里幅度归一化常用于消除信道强弱，但 RFFI 不能无差别消除。

结论: SGC 后续只把 `no_amp` 作为主候选。full amp_norm 放入停止组。

### 5. no_spec augment 与 collapse 风险

`sgc_lite_b_no_dac_no_spec augment` 被解析为 `collapse=True`，虽然 Primary 有 86.00、SAT Avg 44.79，但跳过 batch 达到 19，且训练稳定性不佳。

结论: 不把 no_spec 作为主线，只保留为诊断材料。

### 6. no_amp_freq 作为主模型

`sgc_lite_b_no_dac_no_amp_freq augment` 的 SAT Avg 最高:

```text
SAT Avg 48.95
```

但 clean/OOD 只有:

```text
Primary 85.13, strict UDU 82.54, overall 87.73
```

这说明它更像“卫星压力上限探针”，不是最佳模型。

结论: 只作为 SAT stress probe，不作为最终候选。

## 优化方向

### 方向 A: 主线回到已验证的 Lite-B no-DAC + Fishr

主线应继承上一轮最稳的结构:

```text
rxrobust_lite_b_no_dac_mix015
conservative MixStyle
Fishr lambda = 0.02
primary_udu_weight = 0.65
```

这样做的原因:

1. `SAT37_r19_fishr` 已证明 clean/OOD 强。
2. `sgc_baseline_no_adapter source` 再次证明不加 SGC adapter 的 Lite-B no-DAC 在当前代码下仍强。
3. Fishr 约束的是跨域梯度统计，比直接改前端信号更温和。

### 方向 B: SAT mixed consistency 温和加入

第一批不要使用 1.0/1.0。推荐:

```text
lambda_sat_cls = 0.08
lambda_sat_cons = 0.04
sat_cons_start_epoch = 20
sat_train_scenario = mixed_orbit
```

如果 SAT 不足，再做 0.12/0.06；如果 clean/OOD 掉，则降到 0.05/0.02 或把 start epoch 延后到 60。

### 方向 C: SGC 作为二阶段 no_amp adapter

在第一最佳模型跑出来后，再用 SGC no_amp 接入:

```text
preset = sgc_lite_b_no_dac_no_amp
stage = sgc_augment
source_ckpt = 第一最佳模型 best_model_primary_ood.pth
lambda_sat_cls = 0.08
lambda_sat_cons = 0.04
sat_cons_start_epoch = 20
```

当前 checkpoint 加载使用 `strict=False`，因此从无 SGC adapter 的 checkpoint 接入 no_amp adapter 是可尝试的；新 adapter 参数会随机初始化/默认初始化，主干加载已有权重。

这个实验的定位不是替代第一最佳，而是回答一个问题:

```text
在强 Fishr + mild SAT 的主干上，加 no_amp SGC adapter 是否还能提升 SAT Avg，同时不伤 strict UDU 和 worst-RX？
```

### 方向 D: 选择标准统一为 0.65 UDU 权重

5.7 launcher 默认没有传 `--primary_udu_weight 0.65`，导致最佳 checkpoint 选择与上一轮计划不一致。后续所有候选统一使用:

```text
--primary_udu_weight 0.65
```

并且报告至少记录:

```text
Primary OOD, strict UDU, overall, worst-RX, SAT Avg, skipped_backward_batches
```

## 合并后的下一步路线

### 主目标

先跑出一个“能作为当前最佳”的模型，而不是继续扩散 ablation:

```text
final_lite_b_fishr_sat_mild_v1
```

目标门槛:

| 指标 | 通过线 |
|---|---:|
| Primary OOD | >= 87.80 |
| strict UDU | >= 86.20 |
| overall | >= 90.50 |
| worst-RX | >= 84.50 |
| SAT Avg | >= 41.50 |
| skipped_backward_batches | <= 50 |

如果 V1 通过，就先把它定为第一最佳模型。SGC 只进入第二阶段验证，不抢主线。

## 预设实验组

### MAIN-A: 第一最佳模型组

| 实验名 | 结构 | SAT 权重 | Fishr | 目的 |
|---|---|---|---|---|
| `final_lite_b_fishr_sat_mild_v1` | Lite-B no-DAC mix015 | cls 0.08, cons 0.04, start 20 | 0.02 | 第一主候选 |

推荐先只跑这一组。

### WEIGHT-B: SAT 权重消融组

只在 MAIN-A 未达门槛时跑:

| 实验名 | 改动 | 使用条件 |
|---|---|---|
| `final_lite_b_fishr_sat_light_v2` | cls 0.05, cons 0.02, start 20 | V1 clean/OOD 掉 |
| `final_lite_b_fishr_sat_mid_v3` | cls 0.12, cons 0.06, start 20 | V1 clean/OOD 过线但 SAT Avg 不足 |
| `final_lite_b_fishr_sat_delayed_v4` | cls 0.08, cons 0.04, start 60 | V1 前期被 SAT 扰动拖累 |

### SGC-C: 二阶段 SGC 验证组

只在 MAIN-A 产出强 checkpoint 后跑:

| 实验名 | preset/stage | source_ckpt | 目的 |
|---|---|---|---|
| `sgc_noamp_from_best_fishr_mild` | `sgc_lite_b_no_dac_no_amp`, `sgc_augment` | MAIN-A best primary | 验证 no_amp adapter 是否能增 SAT 且少伤 clean/OOD |
| `sgc_lited_from_best_fishr_mild` | `sgc_lite_d_no_dac`, `sgc_augment` | MAIN-A best primary 或 Lite-D best | 紧凑部署候选 |

这组必须显式传小 SAT 权重，禁止走 launcher 默认 1.0/1.0。

### DIAG-D: 诊断组

| 实验名 | 定位 |
|---|---|
| `sgc_noamp_freq_sat_probe` | SAT 上限探针，不作为主模型 |
| `baseline_no_adapter_sat_mild` | 验证无 adapter + mild SAT 是否已足够 |

### STOP-X: 暂停/禁止组

| 路线 | 原因 |
|---|---|
| full `sgc_lite_b_no_dac` from source | clean/OOD 与 worst-RX 明显下降 |
| `sgc_augment` 默认 1.0/1.0 from epoch 1 | SAT 提升以 clean/OOD 损失换来，过强 |
| 当前 `sgc_adapt` 作为最终阶段 | 无 SAT loss，冻结主干后多数路线退化 |
| `no_spec` 作为主线 | 出现 collapse 风险 |
| `no_amp_freq` 作为主线 | SAT Avg 高但 Primary/UDU 太低 |

## 推荐命令

### MAIN-A: 第一最佳模型

```bash
mkdir -p finalist_runs/final_lite_b_fishr_sat_mild_v1 logs

CUDA_VISIBLE_DEVICES=${GPU_ID:-0} ${PYTHON_BIN:-python} -u train.py \
  --dataset wisig \
  --wisig_domain rx_day \
  --batch_size 256 \
  --slim_group rxrobust_lite_b_no_dac_mix015 \
  --epochs 200 \
  --wisig_train_ratio 0.2 \
  --primary_udu_weight 0.65 \
  --seed 1337 \
  --use_sat_consistency \
  --sat_train_scenario mixed_orbit \
  --sat_cons_start_epoch 20 \
  --lambda_sat_cls 0.08 \
  --lambda_sat_cons 0.04 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --eval_sat_channel \
  --eval_sat_on test_unseen_day_unseen_rx \
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_eval_max_batches -1 \
  --latest_save_path finalist_runs/final_lite_b_fishr_sat_mild_v1/latest_model.pth \
  --best_save_path finalist_runs/final_lite_b_fishr_sat_mild_v1/best_model_val.pth \
  --best_primary_save_path finalist_runs/final_lite_b_fishr_sat_mild_v1/best_model_primary_ood.pth \
  --best_test_save_path finalist_runs/final_lite_b_fishr_sat_mild_v1/best_model_test_overall.pth \
  --best_unseen_day_unseen_rx_save_path finalist_runs/final_lite_b_fishr_sat_mild_v1/best_model_strict_udu.pth \
  --best_worst_rx_save_path finalist_runs/final_lite_b_fishr_sat_mild_v1/best_model_worst_rx.pth \
  2>&1 | tee logs/final_lite_b_fishr_sat_mild_v1_seed1337.log
```

### SGC-C: no_amp adapter 二阶段验证

```bash
mkdir -p finalist_runs/sgc_noamp_from_best_fishr_mild logs

CUDA_VISIBLE_DEVICES=${GPU_ID:-0} ${PYTHON_BIN:-python} -u train.py \
  --dataset wisig \
  --wisig_domain rx_day \
  --batch_size 256 \
  --preset sgc_lite_b_no_dac_no_amp \
  --stage sgc_augment \
  --source_ckpt finalist_runs/final_lite_b_fishr_sat_mild_v1/best_model_primary_ood.pth \
  --train_sat_channel \
  --train_sat_scenario mixed_orbit \
  --sat_view_source main \
  --epochs 60 \
  --wisig_train_ratio 0.2 \
  --primary_udu_weight 0.65 \
  --sat_cons_start_epoch 20 \
  --lambda_sat_cls 0.08 \
  --lambda_sat_cons 0.04 \
  --lambda_res 0.01 \
  --eval_sat_channel \
  --eval_sat_on test_unseen_day_unseen_rx \
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_eval_max_batches -1 \
  --latest_save_path finalist_runs/sgc_noamp_from_best_fishr_mild/latest_model.pth \
  --best_save_path finalist_runs/sgc_noamp_from_best_fishr_mild/best_model_val.pth \
  --best_primary_save_path finalist_runs/sgc_noamp_from_best_fishr_mild/best_model_primary_ood.pth \
  --best_test_save_path finalist_runs/sgc_noamp_from_best_fishr_mild/best_model_test_overall.pth \
  --best_unseen_day_unseen_rx_save_path finalist_runs/sgc_noamp_from_best_fishr_mild/best_model_strict_udu.pth \
  --best_worst_rx_save_path finalist_runs/sgc_noamp_from_best_fishr_mild/best_model_worst_rx.pth \
  2>&1 | tee logs/sgc_noamp_from_best_fishr_mild_seed1337.log
```

注意: 这条命令显式设置了 `lambda_sat_cls` 和 `lambda_sat_cons`，避免 `sgc_augment` 自动回退到 1.0/1.0。

## 执行顺序

1. 跑 MAIN-A `final_lite_b_fishr_sat_mild_v1`。
2. 若 MAIN-A 通过全部门槛，记录为第一最佳模型。
3. 若 MAIN-A clean/OOD 未过，跑 WEIGHT-B light 或 delayed。
4. 若 MAIN-A clean/OOD 通过但 SAT Avg 不足，跑 WEIGHT-B mid。
5. 只有在 MAIN-A 或 WEIGHT-B 产生合格主模型后，才跑 SGC-C。
6. SGC-C 若不能同时保持 `strict UDU >= 86.0`、`worst-RX >= 84.0`，不进入最终模型，只作为 SAT robustness 辅助结果。

## 最终选择策略

优先级如下:

1. clean/OOD 不崩: strict UDU 和 worst-RX 是硬门槛。
2. Primary OOD 高: 用 `primary_udu_weight=0.65` 统一选择。
3. SAT Avg 有提升: 目标是超过 `SAT37_r19_fishr` 的 38.91，理想达到 41.50 以上。
4. 参数和复杂度可控: Lite-B 主线优先，Lite-D 作为 compact 候选，SGC adapter 只有在增益明确时加入。

当前最优路线判断:

```text
第一优先: Lite-B no-DAC + conservative MixStyle + Fishr + mild SAT mixed consistency
第二优先: Lite-D no-DAC SAT/mixed compact candidate
第三优先: no_amp SGC adapter from a strong Fishr+SAT checkpoint
暂停: full SGC from source, aggressive SGC augment, current sgc_adapt
```
