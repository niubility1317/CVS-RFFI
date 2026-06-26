# Few-shot cross-domain / cross-receiver RFFI literature survey

检索日期：2026-06-11

范围：本文把 `RF fingerprinting`、`radio-frequency fingerprint identification`、`RFFI`、`RFF`、`specific emitter identification`、`SEI` 作为同义/近义检索入口。纳入标准是论文明确处理 RF/RFF/SEI 的少样本、跨域、跨信道、跨接收机、开放集或增量识别问题。纯调制识别、纯 Wi-Fi sensing、纯 UAV detection 但不做发射机/设备指纹识别的工作未作为主线展开。

## 一句话结论

这个方向不是单一路线，而是四类问题的交叉：

1. 少样本闭集识别：已知设备集合内，每类只有 K 个标注样本。主流是 MAML、ProtoNet/MatchingNet、Siamese、度量学习、masked autoencoder/self-supervised pretraining 后微调。
2. 少样本跨域迁移：源域有较多数据，目标域只有少量标注。主流是原型网络、MMD/分布对齐、对比自监督预训练、数据/特征增强、少量 fine-tuning。
3. 跨接收机泛化：训练接收机和部署接收机不同，接收机硬件偏置污染发射机指纹。主流是 receiver-adversarial training、特征解耦/分离、domain adaptation、domain generalization、collaborative inference、federated receiver-invariant learning。
4. 开放集/增量部署：新设备加入、旧设备退出，或需要拒识未知设备。主流是 Gaussian prototype、open-set loss、meta-task adaptation、prototype/threshold/hypersphere 方法。

对 CVS-RFFI 这类 WiSig 低样本跨 receiver/day/domain 设置，最稳的工程路线是：预训练 backbone + domain-aware prototype/cosine head + 轻量 DG 正则 + best-checkpoint/early stopping + 多 seed。强 DANN/GRL 或重域不变正则在 K5/K10/K20 场景容易抹掉发射机细粒度差异，应该先弱化、分阶段启用。

## 代表论文矩阵

| 论文 | 场景 | 方法族 | 需要目标域数据吗 | 解决什么情况 | 局限 |
|---|---|---|---|---|---|
| Li et al., 2021, [A Survey of Few-Shot Learning for Radio Frequency Fingerprint Identification](https://link.springer.com/chapter/10.1007/978-3-030-90196-7_37) | RF 指纹少样本综述 | 数据增强、生成、fine-tuning、度量学习、元学习 | 视方法而定 | 建立少样本 RFFI 方法图谱 | 综述早于 2023-2026 的 cross-receiver/DG 主线 |
| Mackey et al., 2022, [Cross-Domain Adaptation for RF Fingerprinting Using Prototypical Networks](https://www.eng.auburn.edu/~szm0001/papers/sensys22.pdf) | 跨环境/跨时间/跨距离 RF fingerprinting | Prototypical Network, episodic few-shot | 需要少量目标域标注 | 新环境只有少量标注样本时快速适配 | poster 版，实验规模较小 |
| Zhao et al., 2024, [Cross-domain, Scalable, and Interpretable RF Device Fingerprinting](https://www.eng.auburn.edu/~szm0001/papers/infocom24.pdf) | 跨域、跨数据集、设备可扩展 | modified PTN, 自定义损失, LIME/XAI 辅助数据增强 | 通常需要 support set | 新设备/新域加入，且希望解释哪些片段支撑识别 | ORACLE 距离域仍是最难场景，跨数据集迁移受采集协议差异影响大 |
| Shi et al., 2025, [Towards a Unified Few-Shot Learning Evaluation Framework for RF Fingerprinting](https://www.eng.auburn.edu/~szm0001/papers/ICCCN2025.pdf) | 统一评测 few-shot TL | PTN、MatchingNet、MMD、fine-tuning、zero-shot | 比较不同假设 | 解决文献中 split、shot、baseline 不一致的问题 | 是评测框架，方法创新不是重点 |
| Yang et al., 2021/2022, [Specific Emitter Identification With Limited Samples: A Model-Agnostic Meta-Learning Approach](https://www.researchgate.net/publication/354427811_Specific_Emitter_Identification_With_Limited_Samples_A_Model-Agnostic_Meta-Learning_Approach) | 少样本 SEI | MAML | 目标任务少量标注 | 新设备类型/新任务需要快速微调 | MAML 训练复杂，对 task construction 敏感 |
| Xie et al., 2022, [Few-Shot SEI in Non-Cooperative Scenarios](https://www.researchgate.net/publication/391620401_Cross-Domain_Few-Shot_Specific_Emitter_Identification_via_Contrastive_Self-Supervised_Learning) 引用页中的相关条目 | 非协作少样本 SEI | bispectrum/Radon 特征 + 改进 meta-learning | 需要少量标注 | 信号难采集、非协作、传统特征仍有价值 | 前处理较重，不是端到端 raw IQ 路线 |
| Wang et al., 2022, [Few-Shot Specific Emitter Identification via Deep Metric Ensemble Learning](https://arxiv.org/abs/2207.06592) | ADS-B 飞机少样本 SEI | complex-valued CNN + deep metric learning + ensemble classifier | 每类少量标注 | 闭集少样本，尤其每类大于 5 个样本后表现强 | 不直接处理跨 receiver/domain |
| Wang et al., 2023, [Interpolative Metric Learning for Few-Shot Specific Emitter Identification](https://tohoku.elsevierpure.com/en/publications/interpolative-metric-learning-for-few-shot-specific-emitter-ident/) | Wi-Fi 少样本 SEI | InterML, 样本空间插值 + 特征距离约束 | 需要少量标注 | 无辅助数据时提升少样本泛化 | 仍偏闭集和单数据集 |
| Zhang et al., 2023/2024, [Real-World Aircraft Recognition Based on RF Fingerprinting With Few Labeled ADS-B Signals](https://trid.trb.org/View/2341446) | ADS-B 实测少样本 | 预处理 + Siamese few-shot training | 需要少量标注 | 实测飞机识别，小样本且噪声明显 | 场景专注 ADS-B，跨域不是主目标 |
| Yao et al., 2023, Few-Shot SEI Using AMAE | 少样本 SEI | asymmetric masked autoencoder | 可用无标注预训练 + 少量标注微调 | 标注少但有辅助无标注数据 | 依赖预训练域与目标域相近程度 |
| Liu et al., 2023/2024, SA2SEI | 少样本 SEI | self-supervised learning + adversarial augmentation + knowledge transfer | 无标注辅助数据 + 少量目标标注 | 辅助数据和目标设备不同，但希望迁移 RFF extractor | 极低 shot 下仍需关注正负样本构造 |
| Zhang et al., 2025, [Cross-Domain Few-Shot SEI via Contrastive Self-Supervised Learning](https://www.researchgate.net/publication/391620401_Cross-Domain_Few-Shot_Specific_Emitter_Identification_via_Contrastive_Self-Supervised_Learning) | ADS-B 到 LoRa/AIS 跨协议少样本 | CSEE, contrastive SSL, von Neumann entropy, EMA | 源域无标注辅助 + 目标少量标注 | 源/目标协议差异大，仍要少样本微调 | 不是 zero-shot DG，最终依赖目标域少量标注 |
| Fu et al., 2025/2026, [Cross-Channel SEI via MFA-FSL](https://www.researchgate.net/publication/398518833_Cross-Channel_Specific_Emitter_Identification_via_Meta-Feature_Augmentation-Enhanced_Few-Shot_Learning) | 跨信道少样本 SEI | meta-feature augmentation enhanced FSL | 需要少量目标/任务标注 | 发射机相同但信道扰动导致分布偏移 | 主要针对 channel，不等价于 receiver hardware shift |
| Sun et al., 2025, [Few-Shot SEI: KDM Fusion Framework](https://dl.acm.org/doi/10.1109/TIFS.2025.3550080) | IIoT 少样本 SEI | knowledge-data-model fusion, handcrafted + self-supervised + few-shot | 需要少量标注，常配合辅助数据 | 工业场景，手工物理特征和深度特征都可用 | 系统复杂度高，复现成本较大 |
| Xie et al., 2025, [Few-shot open-set RFFI via MLGPN](https://www.researchgate.net/publication/390651901_A_Novel_Radio_Frequency_Fingerprint_Identification_Scheme_for_Few-Shot_Open-set_Recognition) | LoRa few-shot open-set | Gaussian prototype, Mahalanobis distance, open-set loss, episodic meta-learning | 需要 few-shot support | 新设备加入/退出，且要拒识未知设备 | 重点是开放集，不是跨 receiver |
| Li et al., 2024/2025, [Meta-RFF](https://www.researchgate.net/publication/385778220_Meta-RFF_Few-Shot_Open-Set_Incremental_Learning_for_RF_Fingerprint_Recognition_via_Multi-phase_Meta_Task_Adaptation) | few-shot open-set incremental RFF | multi-phase meta-task adaptation | 需要增量 support | 部署期不断新增设备类别 | 更接近生命周期管理，跨域需另加机制 |
| Xie et al., 2022/2023, [Disentangled Representation Learning for RFF under Unknown Channel Statistics](https://arxiv.org/abs/2208.02724) | 未知信道统计、开放环境 | device-relevant/device-irrelevant disentanglement, adversarial learning, implicit augmentation | 不必看到所有未知信道 | 训练只覆盖简单信道，但部署到复杂多径/未知信道 | 处理 channel 更强，receiver 硬件偏置需额外建模 |
| Wang et al., 2023, [Semi-Supervised RF Fingerprinting with Consistency-Based Regularization](https://arxiv.org/abs/2304.14795) | 标注少但无标注多 | composite RF augmentation, consistency regularization, pseudo-labeling | 需要大量无标注同域/近域数据 | 标注瓶颈明显，但能持续收集无标注信号 | 不是严格 few-shot meta-learning；伪标签受 domain shift 影响 |
| Shen et al., 2022/2024, [Towards Receiver-Agnostic and Collaborative RFFI](https://arxiv.org/abs/2207.02999) | 多接收机 LoRa，跨接收机 | adversarial receiver-invariant training, collaborative inference, fine-tuning | DG 不需目标标注；fine-tuning 需少量目标标注 | 接收机硬件差异明显，多 receiver 可协作 | adversarial pressure 需平衡，强压可能损伤 TX 判别 |
| Liu et al., 2023, [Receiver-Agnostic RFFI via Feature Disentanglement](https://www.semanticscholar.org/paper/Receiver-Agnostic-Radio-Frequency-Fingerprint-via-Liu-Zhu/a2a784bc2d09c29e9ba16bd7d01241fefc9ab337) | 跨接收机 | feature disentanglement | 训练需多接收机数据 | 分离发射机特征和接收机特征 | 会议短文，公开细节有限 |
| Bao et al., 2023, [Two-stage UDA and Fine-tuning](https://www.researchgate.net/publication/378499765_Receiver-Agnostic_Radio_Frequency_Fingerprinting_Based_on_Two-stage_Unsupervised_Domain_Adaptation_and_Fine-tuning) | 跨接收机 | two-stage unsupervised DA + fine-tuning | 需要目标接收机无标注数据，fine-tune 阶段可能需要少量标注 | 新接收机可采无标注流量时 | 不是纯 DG，部署前需目标域数据 |
| Zhao et al., 2023, [GAN-RXA](https://arxiv.org/pdf/2303.14312) | receiver-agnostic transmitter fingerprinting | GAN-style receiver transformation/augmentation, RXA | 依方案而定 | 训练接收机有限，需要增强 receiver-agnostic 特征 | 生成质量和 receiver coverage 决定上限 |
| Yang et al., 2024, [Mitigating Receiver Impact via Domain Adaptation](https://arxiv.org/pdf/2404.08566) | 新 receiver 目标域 | DA, adaptive pseudo-labeling, class weighting, receiver impact mitigation | 需要目标 receiver 未标注数据 | 可采目标接收机样本但标注昂贵 | 伪标签错误会累积，类不均衡需处理 |
| Hu et al., 2024, [Few-shot cross-receiver RFFI based on feature separation](https://www.researchgate.net/publication/384969723_Few-shot_cross-receiver_radio_frequency_fingerprinting_identification_based_on_feature_separation) | few-shot cross-receiver Wi-Fi | 噪声/信道增强 + transmitter/receiver feature separation + similarity loss + fine-tuning | 新 receiver 每类少量样本 | 用户问题中最直接相关：跨接收机且 few-shot | 需要每个已训练发射机在新 receiver 上有少量样本 |
| Zhang et al., 2024, [Domain Generalization for Cross-Receiver RFFI](https://arxiv.org/abs/2411.03636) | 跨接收机 DG | Separable Condition, RIEI, emitter/receiver feature decoupling, FedRIEI | RIEI 不依赖目标 receiver 标注；FedRIEI 依赖多 receiver 联邦训练 | 新接收机未见，且不想集中原始数据 | 仍需要多源 receiver 训练覆盖足够变化 |
| Feng et al., 2025, Cross-Receiver RFFI with Dynamic Distribution Alignment | 跨接收机 DA | global/subdomain dynamic distribution alignment | 需要目标域数据 | source-target 和 class-conditional shift 同时存在 | 依赖目标域分布估计质量 |
| Pan et al., 2025, [Cross-Receiver Generalization via Feature Disentanglement and Adversarial Training](https://arxiv.org/abs/2510.09405) | 跨接收机泛化 | adversarial training + style transfer + feature disentanglement | 以多 receiver 训练为主 | receiver-induced shift 明显，需隔离 transmitter signature | arXiv 工作，需验证发表版本 |
| Ma et al., 2025, Length-Versatile Few-Shot RFFI | 输入长度变化 + few-shot | unsupervised self-distillation | 少量标注/无标注预训练 | 信号截断长度不固定，模型不想重训 | 长度鲁棒和跨 receiver 是两回事 |
| Liu et al., 2025, SRP-CBL few-shot RFFI | IoT few-shot RFFI | signal recurrence plot + convolutional broad learning | 需要少量标注 | 轻量、低样本、非深层大模型部署 | 跨域能力需单独验证 |

## 方法族归纳

### 1. Metric/prototype few-shot: ProtoNet, MatchingNet, Siamese, Gaussian prototype

这类方法把分类从固定 softmax 头变成“embedding 空间距离”。RF 指纹任务很适合这种建模，因为同一发射机在 receiver/day/channel 变化下应该围绕稳定身份中心聚集，不同发射机应该保持间隔。

适用情况：

- 新设备或新 receiver 有 K-shot support set。
- 类别会加入或减少，不希望每次重训固定分类头。
- 需要比 MAML 更简单、稳定、可复现的训练流程。

代表论文：

- Mackey et al. 使用 Prototypical Networks 做 RF fingerprinting 跨域适配。
- Zhao et al. 在 INFOCOM 2024 扩展为 modified PTN，并加入 XAI/LIME 辅助增强。
- Shi et al. 统一比较 PTN、MatchingNet、MMD、fine-tuning。
- Xie et al. 2025 的 MLGPN 用 Gaussian prototype 和 Mahalanobis distance 处理 few-shot open-set RFFI。

对 CVS-RFFI 的启发：优先考虑 domain-aware prototype 或 cosine classifier，而不是只靠 CE softmax。prototype 可以按 TX 聚合，也可以维护 receiver/day 条件下的局部 prototype，再用可靠性权重融合。

### 2. Optimization-based meta-learning: MAML/ANIL 类

MAML 的目标不是直接学一个通用分类器，而是学一个容易被少量样本快速更新的初始化。Yang et al. 将 MAML 引入 limited-sample SEI，显示训练任务和测试任务属于不同设备类型时仍可保持较高准确率。

适用情况：

- 目标域会频繁出现，每次只给少量样本，但允许做若干步梯度更新。
- 任务分布可以清晰构造，例如每个 episode 是不同设备集合、不同 receiver/day 组合。

局限：

- 对 episode 设计、内外循环学习率、batch 组织非常敏感。
- 在 RFFI 中 full MAML 往往比 ProtoNet/ANIL 工程成本高。
- 少样本跨接收机时，如果内循环样本本身被 receiver bias 主导，MAML 可能快速适配 receiver 而不是 TX identity。

对 CVS-RFFI 的启发：如果要做 meta-learning，ANIL 或 prototype-episode 比 full MAML 更实际。可以冻结低层 CV-SincNet/FFT 分支，仅让 adapter/head 快速适配。

### 3. Deep metric learning and interpolation: DMEL, InterML, triplet/contrastive family

DMEL、InterML 这类方法直接优化类内紧凑、类间分离。InterML 通过样本空间插值挖掘隐式样本，并在特征空间约束距离，减少对辅助数据的依赖。

适用情况：

- 主要挑战是每类样本少，而不是目标 receiver 完全未知。
- 可以构造正负样本对或 triplet。
- 需要比 softmax 更好的 embedding 几何。

局限：

- 负样本挖掘不当会把相似设备过度推开，损伤真实相邻发射机边界。
- 如果 domain shift 强，metric loss 可能把 receiver/day 差异学成“类间差异”。

对 CVS-RFFI 的启发：SupCon/triplet 应该轻量使用，并配合 domain-balanced sampler，避免同 TX 不同 receiver/day 被当作负样本。

### 4. Self-supervised / masked / consistency pretraining

这条线解决“标注少但无标注信号多”。SA2SEI、CSEE、AMAE、semi-supervised consistency RF fingerprinting 都属于这个方向。CSEE 还明确做跨协议 few-shot：ADS-B 作为源域，LoRa/AIS 作为目标域，源域无标注预训练后在目标域少量样本微调。

适用情况：

- 有大量无标注 RF 流量。
- 标注目标域样本昂贵，但可以拿到少量 K-shot。
- 目标协议或设备类型与源域不同，但底层 RF hardware impairment 有可迁移结构。

局限：

- 自监督增强必须符合 RF 物理，不然会破坏指纹。
- SSL 学到的可能是协议/调制/接收机统计，而不是 TX hardware signature。

对 CVS-RFFI 的启发：可以先做 masked/contrastive 预训练或 supervised pretraining，再小学习率微调 head/adapter。对 K5/K10/K20，预训练通常比“从零训练 + 强 DG loss”更可靠。

### 5. Domain adaptation: MMD、LMMD、KL/伪标签、adversarial DA

Domain adaptation 假设部署前能看到目标域数据，常见是无标注目标 receiver 样本。Yang et al. 的 receiver impact mitigation 使用 adaptive pseudo-labeling 和 class weighting，针对伪标签类不均衡做修正。动态分布对齐类方法则同时对齐全局分布和子域/类别条件分布。

适用情况：

- 新 receiver 已部署，可收集无标注数据。
- 不能或不想标注大量目标域样本。
- source/target 类集合基本一致。

局限：

- 这不是纯 domain generalization；如果目标域数据不可见，DA 不能直接用。
- RFFI 中伪标签错误会把 receiver artifact 固化为 emitter feature。
- 对类不均衡和 confidence threshold 很敏感。

对 CVS-RFFI 的启发：如果有新 receiver 无标注流量，可做 adapter/BN/prototype-only DA，而不宜全模型大幅更新。伪标签必须设置 coverage 和 accepted-only/full-denominator 双指标。

### 6. Domain generalization and feature disentanglement

DG 目标是在训练时只见源域，部署时不看目标域数据。Xie et al. 先把信号拆成 device-relevant 和 device-irrelevant 分量，再通过 adversarial learning 和隐式重组增强抑制信道统计过拟合。Zhang et al. 的 RIEI/FedRIEI 进一步聚焦 cross-receiver，提出 Separable Condition，并将接收机相关特征与发射机相关特征解耦。

适用情况：

- 目标 receiver/day/channel 部署前不可见。
- 多源训练域足够多，可以学到“哪些变化不是发射机身份”。
- 隐私或工程限制不允许集中所有 receiver 原始数据，此时 FedRIEI/Federated 变体有价值。

局限：

- 需要源域足够覆盖变化，否则 DG 会变成过拟合源域 receiver 组合。
- 解耦损失如果太强，会丢掉与 TX identity 相关的细微硬件特征。

对 CVS-RFFI 的启发：CVS 的 z_id / z_dom 思路与 RIEI/DRL 很接近。关键不是简单加 GRL，而是证明四件事：receiver/day 信息被压低、TX identity 没被抹掉、strict unseen receiver/day 提升、低 shot 不出现 best-vs-final rollback。

### 7. Receiver-agnostic collaborative / federated RFFI

Shen et al. 的 receiver-agnostic and collaborative RFFI 是跨接收机方向的关键基线：用 adversarial training 学 receiver-independent features，多 receiver 时做 collaborative inference，少量 fine-tuning 进一步提升弱 receiver。FedRIEI 则把 receiver-invariant 解耦放到联邦学习中，避免集中原始接收机数据。

适用情况：

- 多个接收机同时观测或可离线协作训练。
- 想提升 unseen/weak receiver 识别稳定性。
- 数据治理或隐私要求不允许集中原始 RF 数据。

局限：

- collaborative inference 与 federated training 是不同问题。前者融合证据，后者聚合训练知识。
- receiver adversarial 分支需控制强度，否则容易牺牲 TX separability。

对 CVS-RFFI 的启发：如果未来做多站点/卫星多接收机系统，可以先做 inference-time evidence fusion，再考虑 federated training。不要把两者混成一个不可诊断的大系统。

## 按部署问题选择方法

| 部署问题 | 推荐方法 | 不建议优先用 |
|---|---|---|
| 同 receiver/day，只是每类样本少 | ProtoNet/cosine head、DMEL/InterML、AMAE/SSL pretraining | 重 DANN/GRL |
| 新 receiver 有每类 5-30 个标注样本 | feature separation + fine-tuning、domain-aware prototypes、ANIL/head-only adaptation | 全模型大步长微调 |
| 新 receiver 只有无标注流量 | MMD/LMMD/伪标签 DA、adapter/BN/prototype-only TTA | 直接用伪标签全模型训练且无 coverage 监控 |
| 新 receiver 完全不可见 | RIEI/FedRIEI、receiver-adversarial DG、feature disentanglement、domain randomization | 依赖目标域数据的 DA |
| 跨 channel/SNR/distance | DR-RFF、MFA-FSL、物理一致增强、channel-invariant feature | 只按随机 train/test split 报结果 |
| 跨协议/跨数据集 | contrastive SSL + few-shot fine-tuning、modified PTN、统一 few-shot evaluation | 直接把源域分类头迁移 |
| 新设备持续加入且要拒识未知设备 | MLGPN、Meta-RFF、prototype + threshold/hypersphere | 固定 softmax 闭集分类 |
| 多 receiver 同时可用 | receiver-agnostic collaborative inference、FedRIEI、reliability-weighted fusion | 只报告单 receiver accepted-only accuracy |

## 对 CVS-RFFI 的具体落点

1. 论文定位：CVS-RFFI 更接近“cross-receiver/cross-day/domain generalization + few-shot adaptation”，不是单纯少样本闭集 SEI。因此相关工作应把 few-shot SEI、cross-domain RF fingerprinting、receiver-agnostic RFFI 三条线分开写，再说明 CVS 如何把它们合并。
2. 主实验对照：建议至少保留 zero-shot CE、fine-tuning、ProtoNet/cosine prototype、MMD/DA、receiver-adversarial/GRL、feature-disentanglement 或 RIEI-style baseline。Shi et al. 2025 的统一评测框架可作为 baseline 设计依据。
3. 低样本策略：K5/K10/K20/K30 应以预训练 + head/adapter 微调 + early stopping 为第一层。DG loss 要轻，尤其 satellite/adv/group/proto/supcon/fishr 这类正则不要全强度叠加。
4. 跨接收机策略：如果目标 receiver 完全不可见，用 RIEI/DRL/FedRIEI 语言描述为 DG；如果目标 receiver 有无标注流量，用 DA/TTA；如果目标 receiver 有每类少量标注，才称 few-shot cross-receiver fine-tuning。
5. 报告指标：不要只报平均 accuracy。至少区分 seen-day-seen-rx、unseen-day-seen-rx、seen-day-unseen-rx、strict UDU；如有开放集或拒识，还要同时报 full-denominator accuracy、coverage、accepted-only accuracy、unknown rejection AUC。

## 参考链接

- Li et al., 2021, A Survey of Few-Shot Learning for Radio Frequency Fingerprint Identification: https://link.springer.com/chapter/10.1007/978-3-030-90196-7_37
- Mackey et al., 2022, Cross-Domain Adaptation for RF Fingerprinting Using Prototypical Networks: https://www.eng.auburn.edu/~szm0001/papers/sensys22.pdf
- Zhao et al., 2024, Cross-domain, Scalable, and Interpretable RF Device Fingerprinting: https://www.eng.auburn.edu/~szm0001/papers/infocom24.pdf
- Shi et al., 2025, Towards a Unified Few-Shot Learning Evaluation Framework for RF Fingerprinting: https://www.eng.auburn.edu/~szm0001/papers/ICCCN2025.pdf
- Yang et al., 2021/2022, Specific Emitter Identification With Limited Samples: A Model-Agnostic Meta-Learning Approach: https://www.researchgate.net/publication/354427811_Specific_Emitter_Identification_With_Limited_Samples_A_Model-Agnostic_Meta-Learning_Approach
- Wang et al., 2022, Few-Shot Specific Emitter Identification via Deep Metric Ensemble Learning: https://arxiv.org/abs/2207.06592
- Wang et al., 2023, Interpolative Metric Learning for Few-Shot Specific Emitter Identification: https://tohoku.elsevierpure.com/en/publications/interpolative-metric-learning-for-few-shot-specific-emitter-ident/
- Zhang et al., 2023, Real-World Aircraft Recognition Based on RF Fingerprinting With Few Labeled ADS-B Signals: https://trid.trb.org/View/2341446
- Xie et al., 2022/2023, Disentangled Representation Learning for RF Fingerprint Extraction under Unknown Channel Statistics: https://arxiv.org/abs/2208.02724
- Wang et al., 2023, Semi-Supervised RF Fingerprinting with Consistency-Based Regularization: https://arxiv.org/abs/2304.14795
- Shen et al., 2022/2024, Towards Receiver-Agnostic and Collaborative RFFI: https://arxiv.org/abs/2207.02999
- Liu et al., 2023, Receiver-Agnostic RFFI via Feature Disentanglement: https://www.semanticscholar.org/paper/Receiver-Agnostic-Radio-Frequency-Fingerprint-via-Liu-Zhu/a2a784bc2d09c29e9ba16bd7d01241fefc9ab337
- Bao et al., 2023, Receiver-Agnostic RFF Based on Two-stage UDA and Fine-tuning: https://www.researchgate.net/publication/378499765_Receiver-Agnostic_Radio_Frequency_Fingerprinting_Based_on_Two-stage_Unsupervised_Domain_Adaptation_and_Fine-tuning
- Zhao et al., 2023, GAN-RXA: https://arxiv.org/pdf/2303.14312
- Yang et al., 2024, Mitigating Receiver Impact on RFFI via Domain Adaptation: https://arxiv.org/pdf/2404.08566
- Hu et al., 2024, Few-shot cross-receiver RFFI based on feature separation: https://www.researchgate.net/publication/384969723_Few-shot_cross-receiver_radio_frequency_fingerprinting_identification_based_on_feature_separation
- Zhang et al., 2024, Domain Generalization for Cross-Receiver RFFI: https://arxiv.org/abs/2411.03636
- Zhang et al., 2025, Cross-Domain Few-Shot SEI via Contrastive Self-Supervised Learning: https://www.researchgate.net/publication/391620401_Cross-Domain_Few-Shot_Specific_Emitter_Identification_via_Contrastive_Self-Supervised_Learning
- Fu et al., 2025/2026, Cross-Channel SEI via MFA-FSL: https://www.researchgate.net/publication/398518833_Cross-Channel_Specific_Emitter_Identification_via_Meta-Feature_Augmentation-Enhanced_Few-Shot_Learning
- Xie et al., 2025, Few-shot open-set RFFI via MLGPN: https://www.researchgate.net/publication/390651901_A_Novel_Radio_Frequency_Fingerprint_Identification_Scheme_for_Few-Shot_Open-set_Recognition
- Li et al., 2024/2025, Meta-RFF: https://www.researchgate.net/publication/385778220_Meta-RFF_Few-Shot_Open-Set_Incremental_Learning_for_RF_Fingerprint_Recognition_via_Multi-phase_Meta_Task_Adaptation
- Pan et al., 2025, Cross-Receiver Generalization via Feature Disentanglement and Adversarial Training: https://arxiv.org/abs/2510.09405
