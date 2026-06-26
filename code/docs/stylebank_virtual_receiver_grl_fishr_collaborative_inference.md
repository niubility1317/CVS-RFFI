# StyleBank 虚拟接收机异构 + GRL/Fishr + 协同推理说明

生成时间: 2026-05-26

适用代码根目录: `E:\type10-7\code`

## 1. 一句话总览

这套方法的核心是: 在联邦学习中，每个 client 默认只看到自己的接收机数据，天然缺少跨接收机/跨域混合 batch，导致集中式训练里有效的 GRL、Fishr、MixStyle、跨域一致性等方法在 FL 下容易退化。StyleBank 的作用是让 server 只交换低维 RF 风格统计，而不是交换 IQ 样本；client 再用这些远端风格统计在本地构造“虚拟接收机视图”，从而重新制造多接收机异构 batch。GRL 用这些构造域压制接收机/信道捷径，Fishr 用这些构造域对齐不同虚拟接收机下的分类梯度方差，协同推理则把 clean 视图和多个虚拟接收机视图的概率进行融合，诊断虚拟多接收机是否真的带来互补信息。

可以把它理解成下面这条链:

```text
receiver-client FL
  -> client 上传 StylePacket 低维 RF 风格统计
  -> server 维护 FederatedStyleBank
  -> client 从 StyleBank 采样远端 receiver style
  -> 本地构造 clean + remote-style 虚拟接收机 batch
  -> GRL/Fishr/consistency 使用 d_style 做域泛化
  -> evaluation 时 clean + style views 概率融合，报告 rescue/harm/net gain
```

这不是物理多接收机协同推理的严格复现，而是用 StyleBank 生成的虚拟接收机视图去逼近“多接收机观察同一发射机”的协同机制。

## 2. 为什么普通 FL 里 GRL/Fishr 会失效

当前联邦设置通常使用 `--fl_client_key receiver`，也就是每个接收机作为一个 client。这个设置对隐私和部署语义合理，但会造成一个关键问题: 每个 client 本地 batch 往往只覆盖一个接收机域。

集中式训练中的 DG 技巧通常隐含一个前提:

- GRL 需要模型在同一训练过程中看到多个域，才能学习“发射机可分、接收机不可分”的表示。
- Fishr 需要多个域上的分类梯度统计，才能对齐不同域的梯度方差。
- MixStyle 的 cross-domain 版本需要跨域样本才能混合风格。
- same-TX cross-domain consistency 需要同一发射机在不同域下的特征配对。
- GroupDRO 或 hard-domain CE 需要 batch 或统计窗口中存在多个域损失。

如果每个 client 的本地数据只有一个 receiver 域，直接套用这些损失会出现三类退化:

1. 域数不足，loss 直接为 0 或被 gating 跳过。
2. 域标签虽然存在，但本地没有跨域对比，优化信号很弱。
3. 强行对单域 client 使用域对抗，可能只会制造噪声，破坏 TX 分类收敛。

所以关键不是“把 Fishr 加进 FedAvg”这么简单，而是先在 client 本地重建有意义的多域结构。StyleBank 就是这个结构来源。

## 3. 当前代码中的关键模块

| 模块 | 主要职责 |
|---|---|
| `federated/style_packet.py` | 定义 `StylePacket` 和 `StyleDomainBatch`。前者是上传给 server 的风格统计包，后者是本地 clean/style 扩展后的训练 batch。 |
| `federated/rf_style_extractor.py` | 从 IQ batch 中提取 class-marginalized RF 风格统计，包括幅度、相位、频谱、CFO、IQ imbalance、AGC、SNR proxy、multipath/lowpass proxy 等。 |
| `federated/style_bank.py` | server 侧 StyleBank，维护风格 centroid，支持 EMA 合并、容量裁剪、远端 style 采样、诊断统计。 |
| `federated/conditioned_receiver_dg.py` | 把 StylePacket 中的物理风格统计映射成保守的 receiver-chain IQ 扰动。 |
| `federated/virtual_domain_sampler.py` | 构造 clean + virtual style views 的 `StyleDomainBatch`，并生成显式 `d_style`。 |
| `federated/fed_trainer.py` | 联邦训练主控，负责 client 训练、FedAvg/FedProx 聚合、StyleBank 更新、GRL/Fishr loss、协同推理评估、日志与 metrics。 |
| `federated/reliability_fusion.py` | 概率融合工具，包括 ProtoBank 保守融合和 StyleBank collaborative fusion。 |
| `train.py` | CLI 参数入口，暴露 StyleBank、GRL、Fishr、协同推理、FL82 launcher 所需参数。 |
| `scripts/run_fed_fl82_validation_4gpu.sh` | N607 FL82 验证队列，包含 `FL82_10_fedprox_rx_ra_bex02_stylebank_collab_all5_r010` 这个虚拟异构协同推理实验变体。 |

## 4. StyleBank 如何制造“虚拟接收机异构”

### 4.1 StylePacket 不是样本，也不是高维特征

`StylePacket` 是一个轻量统计包，包含:

- `client_id`: 哪个 client 上传。
- `round_idx`: 哪一轮产生。
- `count`: 统计样本数。
- `stats`: RF 风格统计字典。
- `style_id`: server 侧 centroid id，可为空。
- `metadata`: 目标域、原始域、可靠性、age 等辅助信息。

重要的是，StylePacket 不上传原始 IQ 样本，也不默认上传大规模模型特征。它上传的是 class-marginalized 统计，因此通信量小，也降低了直接泄露样本内容的风险。

### 4.2 class-marginalized 的含义

`RFStyleExtractor.extract(x, y, ...)` 如果拿到标签 `y`，会先对每个 TX class 分别提取 style stats，再对出现过的 class 做等权平均。这样做有两个目的:

1. 减少类别分布不均对风格统计的污染。例如某个 client 恰好 TX0 很多、TX5 很少，直接平均可能把 TX 特征误当 receiver style。
2. 让 StyleBank 更接近“接收机/信道风格库”，而不是“某个 client 的 TX 类别组合库”。

当前提取的统计包括普通 IQ 统计和物理 receiver-chain 统计:

- 均值/方差/RMS: `iq_mean`, `iq_std`, `iq_rms`, `i_mean`, `q_mean`, `i_std`, `q_std`。
- 幅度与相位: `amp_mean`, `amp_std`, `phase_diff_mean`, `phase_diff_std`。
- 频谱: `spectrum_centroid`, `spectrum_bandwidth`, `spectrum_log_energy`。
- 物理近似: `phys_cfo_cycles_per_sample`, `phys_cfo_hz`, `phys_sro_ppm`, `phys_agc_gain_db`, `phys_iq_gain_imbalance_db`, `phys_iq_phase_imbalance_deg`, `phys_phase_noise_std`, `phys_awgn_snr_db`, `phys_multipath_strength`, `phys_lowpass_cutoff_frac`, `phys_lowpass_transition_frac`, `phys_softclip_level`。

这些统计不是为了重建原始信号，而是为了给本地样本施加一个“像远端接收机一样”的保守扰动。

### 4.3 server 侧 StyleBank 的更新逻辑

`FederatedStyleBank` 维护一组 `StyleCentroid`。每轮 client 训练时会收集本地 `StylePacket`，round 结束后 server 将所有 client 上传的 packets 合并到 StyleBank。

更新规则:

- 若 `merge_radius == 0`，每个 packet 默认作为一个新 centroid。
- 若 `merge_radius > 0`，会寻找 metadata 兼容且 L2 距离最近的 centroid，距离足够近时做 EMA 合并。
- 合并时数值统计按 `momentum` 平滑，metadata 中目标域标签会被保留，避免不同目标 receiver/domain 被错误合并。
- 超过 `max_centroids` 后按 count、age、round_idx 裁剪，保留更可靠/更新的 centroids。
- `sample_remote_styles(exclude_client_id=..., k=...)` 会排除本 client 的 style，再选择 count 高且彼此多样的远端 styles。

通信流可以写成:

```text
round r client i:
  local batch -> RFStyleExtractor -> StylePacket_i,r
  local training uses StyleBank from previous rounds

round r server:
  aggregate model updates
  collect StylePacket_i,r from clients
  update FederatedStyleBank

round r+1 client j:
  sample remote StylePacket from server StyleBank
  transform local clean x into virtual receiver views
```

因此，StyleBank 不是同一 batch 内即时交换，而是跨 round 逐步累积的 server-side style memory。

### 4.4 StyleConditionedReceiverDG 如何变换 IQ

`StyleConditionedReceiverDG.transform(x, style)` 会优先使用 `phys_*` 物理统计。如果 style 中没有物理统计，则退化到 legacy 的 gain/phase/noise 轻扰动。

物理路径包括:

- CFO: 根据 `phys_cfo_hz` 或 `phys_cfo_cycles_per_sample` 施加频偏。
- SRO: 根据 `phys_sro_ppm` 施加采样率偏移。
- phase noise: 根据 `phys_phase_noise_std` 施加相位噪声。
- IQ imbalance: 根据 `phys_iq_gain_imbalance_db` 和 `phys_iq_phase_imbalance_deg` 调整 I/Q。
- AGC/softclip: 根据 `phys_agc_gain_db` 和 `phys_softclip_level` 调整幅度链路。
- multipath: 根据 `phys_multipath_strength` 生成保守 FIR 多径。
- lowpass: 根据 `phys_lowpass_cutoff_frac` 和 `phys_lowpass_transition_frac` 做低通。
- AWGN: 当 `phys_awgn_snr_db` 较低时施加噪声。

当前推荐设置非常保守，例如:

- `fl_style_phys_jitter_scale=0.25`
- `fl_style_phys_max_cfo_hz=5000`
- `fl_style_phys_max_sro_ppm=25`
- `fl_style_phys_max_iq_gain_db=0.5`
- `fl_style_phys_max_iq_phase_deg=0.5`
- `fl_style_phys_max_phase_noise_std=0.0005`
- `fl_style_phys_p_lowpass=0.2`
- `fl_style_phys_p_multipath=0.2`

原因是 StyleBank 是虚拟 receiver 近似，不是精确信道仿真。扰动过强会先破坏 TX 判别，再谈不上域泛化。

### 4.5 StyleDomainBatch: clean + remote-style views

当 StyleBank 成熟后，本地 client 会构造类似下面的 batch:

```text
原始本地 batch:
  x, y, d_raw

StyleBank 扩展:
  x_clean      -> y, d_raw_clean, d_style_clean
  x_style_1    -> y, d_raw_remote_1, d_style_1
  x_style_2    -> y, d_raw_remote_2, d_style_2

拼接:
  x_cat = [x_clean, x_style_1, x_style_2]
  y_cat = [y, y, y]
  d_raw_cat = [raw clean receiver, target receiver of style 1, target receiver of style 2]
  d_style = constructed style-domain labels
```

`d_raw` 和 `d_style` 的角色不同:

- `d_raw`: 原始或目标 receiver/domain 标签，用于日志、诊断、目标域记录。
- `d_style`: 构造出来的 style-domain 标签，用于 GRL、Fishr、same-TX consistency、group CE 等 DG 损失。

这一区分很关键。Fishr/GRL 不应该盲目使用 client 原始域标签，因为 receiver-client FL 的本地原始域通常单一；它们应该使用 StyleBank 扩展后得到的构造域。

### 4.6 StyleBank 的成熟 gating

StyleBank 不应该从第 1 轮就强力接管训练。当前代码有多层 gating:

- `fl_style_replay_start_round`: 允许 replay 远端 style 的起始轮，默认 20。
- `fl_style_phys_start_round`: 允许物理 IQ style transform 的起始轮，默认 20。
- `fl_style_dg_start_round`: 允许 GRL/Fishr/consistency 使用 `d_style` 的起始轮，默认 40。
- `fl_style_min_remote_centroids`: 至少有多少远端 centroids 才 replay。
- `fl_style_replay_prob`: replay 概率，推荐 0.25。
- `fl_style_max_views`: 每个 clean batch 最多追加几个远端 style views，推荐先从 1 开始。
- `fl_style_dg_min_domains`: 至少多少构造 style domains 才允许 DG loss 使用 `d_style`。

推荐分阶段理解:

```text
round 1-19:
  只收集 StylePacket，不做 style replay。

round 20-39:
  可以做 clean + remote-style 的弱 replay，但 DG/Fishr 不急着使用 d_style。

round 40+:
  如果 style domains 足够，GRL/Fishr/consistency 才开始真正使用 d_style。
```

这个 schedule 是为了避免在 TX 分类器尚未稳定时，过早引入虚拟接收机扰动导致训练不收敛。

## 5. GRL 在这里到底做什么

### 5.1 模型内部路径

`model_dual_cvsincnet.py` 中的双分支模型大致有两条表示:

- `z_id`: 发射机身份分支，希望保留 TX fingerprint。
- `z_dom`: 接收机/信道/域分支，希望捕获 domain 信息。

对应 logits:

- `tx_logits`: TX 分类。
- `dom_logits`: 从 `z_dom` 预测 domain。
- `adv_dom_logits`: 从 `grad_reverse(z_id, grl_lambda)` 预测 domain。

GRL 的关键是 `grad_reverse(z_id)`:

```text
forward:
  adv_head 试图从 z_id 预测 receiver/style domain

backward:
  adv_head 参数被训练得更会预测 domain
  z_id 收到反向梯度，被迫去掉 domain 信息
```

因此 GRL 不是让 domain classifier 变差，而是让特征 `z_id` 对 domain classifier 不友好，同时仍要保持 TX 分类可用。

### 5.2 receiver_agnostic_bex02 目标

在 federated trainer 中，`fl_local_objective=receiver_agnostic_bex02` 会启用 receiver-agnostic 风格的域对抗目标。核心目标是:

```text
min TX CE
+ lambda_rx_adv * CE(domain_from_GRL(z_id), d_loss)
+ optional domain/DG losses
+ optional Fishr
+ optional satellite losses
+ optional FedProx
```

其中 `d_loss` 的来源:

- 无 StyleBank batch 时: 使用 remapped `d_raw`。
- 有 StyleBank batch 且 style DG 已成熟时: 使用 `d_style`。
- 有 StyleBank batch 但 style DG 未成熟时: 不让 DG losses 使用 style domains。

同时模型 forward 的 `domain_labels` 在有 style batch 时会收到 `d_style`，这使 MixStyle/模型内部 domain-aware 路径可以看到构造域；但真正的 loss 激活仍由 `style_dg_ready` 和 domain gates 控制。

### 5.3 避免双重计算同一个 GRL head

当前实现里还有一个重要保护: 如果模型没有单独的 `rx_logits`，receiver-agnostic loss 会复用 `adv_dom_logits` 作为 `loss_rx_adv`。这时不能再把同一个 `adv_dom_logits` 同时计入普通 `loss_adv`，否则一个 adversarial head 会被重复加权。

代码中使用 `rx_uses_adv_head` 保护这个情况:

```text
if objective is receiver_agnostic and rx_logits is missing:
  rx_logits = adv_dom_logits
  loss_rx_adv = CE(rx_logits, d_loss)
  skip generic loss_adv on the same adv_dom_logits
```

这点对稳定性很重要。否则 `lambda_rx_adv` 和 `lambda_adv` 叠加后，域对抗可能过强，直接压坏 TX 表示。

### 5.4 GRL 应该看哪些指标

重点日志/metrics:

- `loss_rx_adv`: receiver-agnostic GRL loss。
- `loss_adv`: generic adversarial loss，复用 head 时应避免重复。
- `grl_target_acc`: adversarial/domain head 对当前 domain target 的预测准确率。
- `diag_rx_adv_active`: receiver-agnostic GRL 是否激活。
- `diag_adv_active`: generic adv 是否激活。
- `zdom_target_acc`: `z_dom` 对 domain 的预测准确率。

解释时要小心:

- `grl_target_acc` 很高，说明 domain 信息仍容易从 `z_id` 中读出，去域还不够。
- `grl_target_acc` 接近随机，不一定代表好，也可能是 domain head 没学会。
- 判断 GRL 是否有效，必须同时看 TX accuracy、strict UDU、satellite metrics 和 loss 是否稳定。

## 6. Fishr 在这里到底做什么

### 6.1 Fishr 的核心思想

Fishr 不是普通的 feature alignment。它的核心是让不同域上的分类器梯度方差结构一致。直觉上:

```text
如果 TX 分类器在 receiver A、receiver B、receiver C 上的梯度方差类似，
说明模型在不同接收机风格下的决策边界调整方向更一致，
更可能学到 receiver-invariant 的 TX 判别规则。
```

当前代码中的实现是一个轻量 Fishr-style proxy:

```python
prob = softmax(logits)
one_hot = y 的 one-hot
grad_proxy = prob - one_hot

对每个 domain:
  V_domain = var(grad_proxy[domain])

target = mean(V_domain over domains)
loss_fishr = mean((V_domain - target)^2)
```

这不是原始 Fishr 论文里完整的逐参数梯度协方差匹配，而是用 logit-gradient proxy 做更轻量的近似。它适合当前代码的原因是计算便宜、容易接入 federated local objective，并且已有 gating 防止无效域数下误用。

### 6.2 为什么 Fishr 必须依赖 d_style

在 receiver-client FL 中，每个 client 的 raw domain 通常就是自己这个 receiver。如果直接用 raw `d_raw`:

```text
client rx0 local batch:
  d_raw = [rx0, rx0, rx0, rx0, ...]

Fishr:
  unique domains = 1
  vars_by_domain < min_domains
  loss_fishr = 0
```

这就是“Fishr + FedAvg 听起来合理，但实际可能没信号”的根源。

StyleBank 扩展后:

```text
client rx0 local batch:
  clean view       d_style=0
  remote style A   d_style=1
  remote style B   d_style=2

Fishr:
  unique constructed domains >= 2 或 >= 3
  可以计算各 style domain 的 grad_proxy variance
```

所以在当前设计里，Fishr 的正确域轴是 `d_style`，不是单 client 的 raw receiver 标签。

### 6.3 Fishr 的激活条件

当前 `_fishr_logit_gradient_variance_loss(logits, y, d, min_domains=...)` 会在这些情况下返回 0:

- `d is None`。
- batch size 太小。
- 没有有效 domain label。
- 每个 domain 内样本数不足，无法估计方差。
- 满足样本数条件的 domain 数少于 `max(2, fishr_min_domains)`。

要看 Fishr 是否真的生效，不能只看 `lambda_fishr > 0`，还要看:

- `diag_fishr_domain_count`
- `diag_fishr_active`
- `loss_fishr`
- `style_num_domains`
- `style_dg_ready`

推荐规则:

- clean + 1 个 remote style view: `fishr_min_domains=2`
- clean + 2 个 remote style views: `fishr_min_domains=3`
- clean + 3 个 remote style views: `fishr_min_domains=4`

如果 `fishr_min_domains=4` 但 `fl_style_max_views=1`，Fishr 大概率一直不激活。

### 6.4 Fishr 和 GRL 的分工

GRL 和 Fishr 都处理域泛化，但目标不同:

| 方法 | 作用对象 | 约束含义 | 风险 |
|---|---|---|---|
| GRL | `z_id` 表示 | 不让 `z_id` 携带 receiver/style domain 信息 | 过强会抹掉与 TX fingerprint 纠缠的有用信息 |
| Fishr | `tx_logits` 的梯度 proxy | 让不同 style domain 下分类梯度方差一致 | 域数不足时无效，扰动过强时对齐噪声 |
| same-TX consistency | `z_id` 特征 | 同一 TX 在不同 style views 下特征接近 | 需要正确配对，过强会压平细节 |
| GroupCE/DRO | domain-wise CE | 关注 hardest style/domain | 过早使用会追着噪声域优化 |

合理组合顺序应该是:

```text
先稳定 CE/FedProx/TX 分类
再引入弱 StyleBank replay
再打开 GRL/Fishr
最后才考虑更强 group loss 或 ProtoBank/协同融合
```

## 7. 本地训练目标的完整组成

当前 federated local objective 的结构可以概括为:

```text
L_total =
  L_cls
  + lambda_rx_adv   * L_rx_adv
  + lambda_dom      * L_dom
  + lambda_adv      * L_adv
  + lambda_orth     * L_orth
  + lambda_cons     * L_same_tx_consistency
  + lambda_group_ce * L_group_ce
  + lambda_fishr    * L_fishr
  + lambda_sat_cls  * L_sat_cls
  + lambda_sat_cons * L_sat_cons
  + lambda_fed_proto * L_fed_proto
  + FedProx_mu_term
```

各项含义:

- `L_cls`: TX 分类主损失，必须始终是核心。
- `L_rx_adv`: receiver/style adversarial loss，RA-BEX02 目标使用。
- `L_dom`: 让 `z_dom` 能预测 domain，帮助显式分离 domain 信息。
- `L_adv`: generic adversarial domain loss，避免和 `L_rx_adv` 重复使用同一个 head。
- `L_orth`: `z_id` 和 `z_dom` 的 covariance orthogonality。
- `L_same_tx_consistency`: 同一 TX 跨 domain/style 的特征一致性。
- `L_group_ce`: hard-domain CE 或 GroupDRO 风格项。
- `L_fishr`: style/domain 梯度方差对齐。
- `L_sat_cls`, `L_sat_cons`: 显式卫星信道路径，不应默认塞进 StyleBank。
- `FedProx`: 控制 client 本地漂移。

对当前 FL82 目标，最重要的是避免“所有东西一起开”。StyleBank、GRL、Fishr、satellite view、ProtoBank/fusion 同时增强，会让失败原因不可归因。

## 8. 协同推理如何做

### 8.1 真实物理协同 vs 当前虚拟协同

真实多接收机协同推理可以理解为:

```text
同一个发射机信号被多个物理 receiver 观测
receiver 1 -> p1(tx)
receiver 2 -> p2(tx)
receiver 3 -> p3(tx)
fusion(p1,p2,p3) -> final tx
```

当前实现没有真实多 receiver 同时观测同一条样本，因此使用 StyleBank 的远端 receiver style 生成虚拟视图:

```text
clean x                  -> p_base
StyleBank receiver style 1 -> p_style_1
StyleBank receiver style 2 -> p_style_2
fusion(p_base, p_style_1, p_style_2) -> final tx
```

这个机制在代码中由 `FederatedTrainer._evaluate_style_collab_fusion()` 实现，只在 evaluation 中运行。它不是训练损失，不会直接更新模型。

### 8.2 融合模式

`collaborative_probability_fusion()` 支持三种模式:

#### soft

无权平均:

```text
p_fused = mean(p_base, p_style_1, ..., p_style_K)
```

这是最接近“论文式简单概率平均”的形式，但因为这里的 style views 是虚拟生成的，不是真实独立 receiver，所以风险较高。

#### adaptive

按 view confidence 和 style reliability 加权:

```text
confidence = 1 - entropy(p_style) / log(num_classes)
aux_weight = clamp(style_reliability * confidence, 0, max_aux_weight)
p_fused = normalize(base_weight * p_base + sum(aux_weight_k * p_style_k))
```

`style_reliability` 默认来自:

- metadata 中的 `reliability` / `mean_reliability` / `style_reliability`。
- 如果 metadata 没有，则用 packet count 和 centroid age 估计。

推荐先用 `adaptive`，因为虚拟 style view 不一定总是可靠。

#### conservative

在 adaptive 基础上进一步限制辅助 view 权重，上限不超过 0.5。适合做安全诊断，避免虚拟 view 强行覆盖 clean view。

### 8.3 协同推理的诊断指标

协同推理不能只看 fused accuracy，要同时看 rescue 和 harm:

- `base_correct`: clean view 预测正确数。
- `fused_correct`: 融合后预测正确数。
- `rescue`: clean 错、fused 对的样本数。
- `harm`: clean 对、fused 错的样本数。
- `net_gain = rescue - harm`。
- `base_tx_acc`: clean view accuracy。
- `fused_tx_acc`: fused accuracy。

理想信号:

```text
fused_tx_acc > base_tx_acc
rescue > harm
net_gain > 0
style_collab_harm 不随 round 上升
```

危险信号:

```text
fused_tx_acc < base_tx_acc
harm 明显大于 rescue
adaptive fusion 的 style_reliability_mean 很低
style views 让模型置信但错误
```

metrics CSV 中应关注:

- `style_collab_rescue`
- `style_collab_harm`
- `style_collab_net_gain`
- `style_collab_base_tx_acc`
- `style_collab_fused_tx_acc`

## 9. ProtoBank 与 StyleBank 协同推理的关系

当前代码还存在 `ProtoEvidenceBank` 和 `proto_fusion_eval`。它和 StyleBank collaborative inference 是两条不同的 inference-time 诊断路径:

| 路径 | 输入 | 融合对象 | 目的 |
|---|---|---|---|
| ProtoBank fusion | 类原型 evidence | `p_base` + prototype posterior | 看可靠原型能否保守 rescue |
| StyleBank collaborative fusion | 虚拟 receiver views | `p_base` + style-view posteriors | 看虚拟异构 receiver 是否提供互补判断 |

推荐把 ProtoBank 当作保守辅助证据，不要让它替代 base classifier。当前配置里的 `proto_rho_max` 默认很小，就是为了避免 prototype 分支把已经稳定的 base logits 拉偏。

## 10. 推荐配置块

FL82 正式约束:

```bash
--wisig_train_ratio 0.1
--epochs 200
--fl_rounds 200
--fl_client_key receiver
```

StyleBank 保守设置:

```bash
--use_fl_style_bank_stats \
--fl_style_replay_start_round 20 \
--fl_style_phys_start_round 20 \
--fl_style_dg_start_round 40 \
--fl_style_dg_min_domains 2 \
--fl_style_max_views 1 \
--fl_style_replay_prob 0.25 \
--fl_style_phys_jitter_scale 0.25 \
--fl_style_phys_max_gain_delta 0.05 \
--fl_style_phys_max_noise_std 0.01 \
--fl_style_phys_max_cfo_hz 5000 \
--fl_style_phys_max_sro_ppm 25 \
--fl_style_phys_max_iq_gain_db 0.5 \
--fl_style_phys_max_iq_phase_deg 0.5 \
--fl_style_phys_max_phase_noise_std 0.0005 \
--fl_style_phys_min_awgn_snr_db 20 \
--fl_style_phys_p_lowpass 0.2 \
--fl_style_phys_p_multipath 0.2 \
--fl_style_phys_max_multipath_taps 3 \
--fishr_min_domains 2
```

StyleBank 虚拟协同推理:

```bash
--use_style_collab_eval \
--style_collab_views 2 \
--style_collab_fusion adaptive \
--style_collab_base_weight 1.0 \
--style_collab_max_aux_weight 0.75
```

推荐的整体 anchor:

```bash
--train_mode fedprox \
--wisig_domain rx \
--fl_client_key receiver \
--fedprox_mu 0.01 \
--fl_local_epochs 1 \
--fl_local_objective receiver_agnostic_bex02 \
--fl_sat_aug_mode baseline_view \
--use_aug \
--use_mixstyle \
--use_sat_consistency \
--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
--sat_view_prob 1.0 \
--sat_cons_start_epoch 1 \
--lambda_rx_adv 1.0 \
--grl_lambda 1.0 \
--lambda_fishr 0.02
```

当前 launcher 中对应的队列项是:

```text
FL82_10_fedprox_rx_ra_bex02_stylebank_collab_all5_r010
```

## 11. 指标与日志该怎么看

### 11.1 StyleBank 是否真的工作

关注:

- `global_style_summary.enabled`
- `style_bank_centroids`
- `style_bank_bytes`
- `style_bank_remote_sample_accept_rate`
- `style_replay_enabled`
- `diag_style_batch_active`
- `style_batch_views`
- `style_num_domains`
- `style_domain_entropy`

解释:

- `style_bank_centroids=0`: 没有可用 style。
- `diag_style_batch_active=0`: 本轮没有构造 style batch。
- `style_batch_views=1`: 只有 clean，没有 remote style view。
- `style_num_domains<2`: GRL/Fishr 无法获得有效构造域。

### 11.2 GRL/Fishr 是否真的工作

关注:

- `diag_rx_adv_active`
- `diag_adv_active`
- `grl_target_acc`
- `diag_fishr_domain_count`
- `diag_fishr_active`
- `loss_fishr`
- `loss_rx_adv`

解释:

- `lambda_fishr>0` 但 `diag_fishr_active=0`: Fishr 没有实际生效。
- `diag_rx_adv_active=1` 但 `loss_cls` 和 `loss_rx_adv` 同时爆炸: GRL 可能过强或 style 扰动过早。
- `grl_target_acc` 需要结合 TX acc 看，不能单独作为越低越好的指标。

### 11.3 协同推理是否有价值

关注:

- `global_style_collab_fusion.enabled`
- `style_collab_base_tx_acc`
- `style_collab_fused_tx_acc`
- `style_collab_rescue`
- `style_collab_harm`
- `style_collab_net_gain`

解释:

- `net_gain > 0`: 虚拟 receiver views 提供了净救援。
- `harm > rescue`: 虚拟 views 正在破坏 clean view。
- `fused_tx_acc` 只在部分 split 提升: 需要看是否正好是 strict UDU 或 satellite target split。

### 11.4 最终实验目标仍是双轴

不要只优化 clean strict UDU，也不要只优化卫星 clear-LEO。当前 FL82 目标至少要同时看:

- clean strict `test_unseen_day_unseen_rx >= 82.0%`
- clear-LEO satellite floors:
  - `test_unseen_day_seen_rx >= 84.30`
  - `test_seen_day_unseen_rx >= 60.10`
  - `test_unseen_day_unseen_rx >= 53.78`

如果一个方法只提高 clean strict UDU，却让 clear-LEO unseen RX 崩掉，不能算完整成功。

## 12. 常见失败模式

### 12.1 StyleBank 太早、太强

症状:

- `train_acc` 接近 0 或持续异常。
- `loss_cls` 上升，`loss_rx_adv` 也上升。
- strict UDU 尚未稳定时，style batch 大量插入。

原因:

- replay 太早。
- `fl_style_replay_prob` 太高。
- `fl_style_max_views` 太大。
- RF perturbation bounds 太强。
- `fl_local_epochs` 太高，client drift 被放大。

处理:

- 回到 `replay_start=20`, `dg_start=40`, `max_views=1`, `replay_prob=0.25`, `local_epochs=1`。

### 12.2 Fishr 看起来开启但实际无效

症状:

- `lambda_fishr=0.02`
- `loss_fishr=0`
- `diag_fishr_active=0`

原因:

- `d_style` 没构造出来。
- style views 不够。
- `fishr_min_domains` 高于实际 style domains。
- 每个 domain 内样本数不足。

处理:

- clean + 1 style view 时设 `fishr_min_domains=2`。
- 增大 batch size 或降低 domain 门槛。
- 先确认 `style_num_domains` 和 `style_batch_views`。

### 12.3 GRL 把 TX fingerprint 一起抹掉

症状:

- `loss_rx_adv` 有效，但 TX accuracy 降得很快。
- strict UDU 没升，seen split 也掉。

原因:

- receiver/style 信息与 TX fingerprint 部分纠缠。
- `lambda_rx_adv` 太大。
- style perturbation 过强。

处理:

- 降低 `lambda_rx_adv` 或延后 `fl_style_dg_start_round`。
- 先只开 StyleBank replay，不开 Fishr/GRL style DG。

### 12.4 把 satellite view 和 receiver style 混在一起

StyleBank 的默认职责是 receiver-style heterogeneity，不是卫星信道训练。卫星信道应该走显式 satellite path:

- `baseline_view`
- `cvs_consistency`
- `eval_sat_channel`

`--use_fed_style_sat_view` 默认应保持关闭。只有在明确做消融时，才把 satellite view 插进 StyleBank batch。

### 12.5 协同融合带来 harm

症状:

- `style_collab_harm` 大于 `style_collab_rescue`。
- `fused_tx_acc < base_tx_acc`。

原因:

- 虚拟 receiver views 不可靠。
- style transform 偏离真实分布。
- fusion 权重太大。

处理:

- 用 `adaptive` 或 `conservative`，不要先用 `soft`。
- 降低 `style_collab_max_aux_weight`。
- 检查 `style_reliability_mean`。

## 13. 当前实现边界和复查点

1. 这不是严格物理多接收机协同推理复现，而是 StyleBank virtual receiver approximation。
2. 当前 Fishr 是 logit-gradient variance proxy，不是完整逐参数 Fishr。
3. StyleBank 默认只应处理 receiver/style heterogeneity，卫星信道不要默认混入 StyleBank。
4. `d_raw` 和 `d_style` 必须保持语义分离。`d_raw` 是真实/目标 receiver 标签，`d_style` 是构造 style-domain 标签。
5. 对非 `rx0` client，应在正式 N607 启动前再审计一次默认 StyleBank batch 的 `d_style` 是否严格符合 clean/style 构造域预期。`VirtualDomainSampler` 的理想语义是 clean=0、remote style=1..K；默认 trainer 路径需要用日志或小测试确认所有 receiver-client 都没有 raw domain 混入 constructed `d_style`。
6. 协同推理只在 evaluation 里计算，不会自动改善训练。它首先是诊断指标，其次才可能成为未来方法贡献。
7. StyleBank 上传的是统计包，不是原始 IQ，但仍应把它视作可能带有设备统计指纹的信息，报告中应说明通信内容和 size。
8. 如果要上 N607，仍必须先本地改代码、验证、建 snapshot，再 `scp` 同步，并写 `automation_reports/CV-SincNet/<run-id>/report.md`。

## 14. 推荐消融顺序

不要从最复杂版本开始。推荐顺序:

1. `fedprox + receiver_agnostic_bex02 + baseline_view`，不启用 StyleBank，确认 clean strict UDU 和 satellite eval 基线。
2. 加 `--use_fl_style_bank_stats`，但保持保守 replay，确认 StyleBank 不破坏收敛。
3. 确认 `style_batch_views`, `style_num_domains`, `diag_rx_adv_active` 正常后，再看 Fishr 是否激活。
4. 开 `--use_style_collab_eval`，只做 evaluation fusion，先看 `rescue/harm/net_gain`。
5. 若协同推理 `net_gain > 0` 且 harm 低，再考虑更大 `style_collab_views` 或稍高 replay probability。
6. 若 clean strict UDU 达标但 satellite 不达标，优先调 explicit satellite path，而不是把 satellite 塞进 StyleBank。

## 15. 最小健康检查清单

正式实验进行中，每隔若干 round 检查:

```text
StyleBank:
  style_bank_centroids > 0
  style_bank_bytes > 0
  style_batch_views matches planned clean+style count
  style_replay_enabled becomes true after replay_start

GRL/Fishr:
  diag_rx_adv_active = 1 after d_style ready
  diag_fishr_active = 1 when lambda_fishr > 0 and domains sufficient
  loss_cls does not explode
  train_acc does not collapse

Collaborative inference:
  global_style_collab_fusion.enabled = true
  style_collab_net_gain >= 0 preferred
  style_collab_harm is low
  fused_tx_acc >= base_tx_acc preferred

Targets:
  clean strict test_unseen_day_unseen_rx
  clear_leo test_unseen_day_seen_rx
  clear_leo test_seen_day_unseen_rx
  clear_leo test_unseen_day_unseen_rx
```

## 16. 最终结论

StyleBank 虚拟接收机异构解决的是 federated receiver-client 设置下“本地缺少多域 batch”的根问题。GRL 负责把 `z_id` 中的 receiver/style shortcut 压掉，Fishr 负责让不同虚拟 receiver style 下的 TX 分类梯度方差更一致，协同推理负责在 evaluation 中验证这些虚拟 receiver views 是否真的提供了互补证据。

这套方法的成败不取决于单个 loss 名字，而取决于三件事:

1. StyleBank 生成的虚拟 receiver views 是否足够真实但不过强。
2. `d_style` 是否在正确时间、正确域数下喂给 GRL/Fishr。
3. clean view 和 style views 融合后是否 rescue 多于 harm。

如果这三点不成立，`StyleBank + GRL/Fishr + 协同推理` 只会变成复杂但无效的正则化堆叠。若三点成立，它才是一个有机的 federated domain generalization 方案。
