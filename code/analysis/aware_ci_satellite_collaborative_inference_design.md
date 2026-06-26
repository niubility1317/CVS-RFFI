# AWARE-CI: 面向卫星群未见接收机的高效协同推理与星上轻量自适应设计

日期：2026-05-29  
工作区：`E:\type10-7`  
目标模型背景：SA33 / CVS-RFFI / WiSig / 星地信道评估  
设计目标：在多台未见接收机或卫星节点上，提高协同推理效率和性能，并支持可控的星上实时微调。

## 1. 背景与当前证据

当前 standalone collaborative inference 已验证：同一事件组按 `(tx_i, day_i, eq_i, sig_i)` 对齐多接收机观测，做概率级 soft fusion。SA33 全量 clean + 星地评估显示：

- Clean strict UDU：`K1=84.78` -> `K5=94.92`
- `clear_leo` strict UDU：`K1=36.54` -> `K5=53.88`
- `low_elev_leo` strict UDU：`K1=38.61` -> `K5=54.48`
- `rain_leo` strict UDU：`K1=36.45` -> `K5=52.76`
- `storm_mp` strict UDU：`K1=30.28` -> `K5=43.29`
- `mixed_orbit` strict UDU：`K1=30.08` -> `K5=44.91`

这说明普通 soft fusion 已经能显著提高多接收机稳定性，但仍有三个瓶颈：

1. 所有接收机同权，无法识别低 SNR、强漂移、错误相关的坏证据。
2. Full-K 通信和等待不适合卫星群实时部署。
3. 静态融合不能持续吸收未见接收机和星地信道的新域信息。

因此下一代算法不应只是“更多接收机平均”，而应是：

> 可信证据筛选 + 自适应 top-M 协同 + 不确定性加权融合 + 小模块 source-free 在线适配 + 轨道级轻量同步。

## 2. 文献与方法基础

### 2.1 RFFI / 多接收机泛化

- Receiver-Agnostic and Collaborative RFFI 强调接收机不变特征与多接收机协同推理，适合作为本项目真实多接收机 packet-level fusion 的直接参照。
- Federated RFFI powered by Unsupervised Contrastive Learning 使用联邦无监督对比学习，适合多节点不上传原始 IQ 的预训练与轨道级同步。
- Domain Generalization for Cross-Receiver RFFI / FedRIEI 提出 receiver-independent emitter features，并用 federated 版本避免集中收集多接收机原始数据。
- Cross-Receiver RFFI Source-Free Adaptation / CSCNet 表明只用源模型和目标接收机无标签数据做 source-free adaptation 是可行路线。

### 2.2 在线测试时适配

- Tent 只用测试 batch，通过熵最小化更新 BN 统计和通道仿射参数，计算轻，适合星上短窗口适配。
- CoTTA 面向连续变化目标域，用 EMA teacher、增强平均预测和随机恢复源权重降低错误累积和遗忘。
- EATA 不是每个样本都反传，而是筛掉高风险或冗余样本，并用 Fisher/重要性正则降低遗忘，适合星上省算力微调。
- SHOT / SHOT++ 的 source-free 思路提示：冻结分类假设或原型，只更新目标特征/小模块。

### 2.3 协同边缘推理与卫星网络

- Attention-aware Semantic Communications for Collaborative Inference 用 entropy-aware / attention-aware 机制减少通信，提示我们只传“语义证据”而不是全特征或原始数据。
- Joint Channel and Semantic-aware Grouping for Collaborative Edge Inference 表明在无线协同推理中，仅按语义或仅按信道选协作者都不稳，应联合语义相关性和物理链路状态。
- Trusted Multi-View Classification / evidential fusion 用不确定性动态融合多个 view，适合给低质量接收机降权。
- FedLEO / FELLO / satellite FL-SEC 等卫星 FL 工作提示：卫星网络连接间歇、链路受限，应该轨道内先聚合、地面窗口慢同步，不要高频全模型 FedAvg。
- LoRA-Edge / PEFT for edge devices 说明 CNN 小模块低秩或 adapter 微调可以显著降低训练参数量，符合星上热功耗和内存约束。

## 3. 算法名称与核心思想

算法名：**AWARE-CI**  
全称：**Adaptive Weighted Agreement and Receiver-Evidence Collaborative Inference**

一句话定义：

> 每颗卫星/接收机只上传轻量预测证据和域可靠性摘要；聚合器按置信度、一致性、域相似度、历史可靠性和接收机多样性动态选择 top-M 参与协同，用加权 log-opinion pool 做融合；在线阶段只更新 adapter / BN / calibration / prototype 等低风险模块。

## 4. 系统角色

### 4.1 单星 / 单接收机节点

本地保留：

- frozen CVS-RFFI backbone
- TX classifier / prototype head
- uncertainty / evidence head
- receiver-style adapter 或 FiLM adapter
- local prototype memory
- local calibration temperature
- drift / reliability tracker

每个事件本地计算：

```text
logits_r
top-L class ids and quantized log-probs
entropy_r, margin_r, energy_r
embedding sketch z_r, e.g. 16-32 dims random projection
style vector s_r, e.g. SNR/RSSI/CFO/IQ imbalance/channel stats
augmentation consistency a_r, e.g. JS divergence across weak views
historical reliability q_r
timestamp / orbit / elevation / link-quality bucket
```

### 4.2 星间或轨道内聚合器

聚合器可以是：

- 当前可见窗口中的 sink satellite
- 轨道内边缘节点
- 地面站窗口内的延迟聚合器

聚合器职责：

- 对齐 event_id
- 过滤坏证据
- 自适应选择 top-M 接收机
- 加权融合
- 输出 final label / abstain / request more receivers
- 维护轨道级 prototype 和 adapter delta 聚合

### 4.3 地面端

地面端不在实时闭环中承担每样本推理，而承担：

- 离线大模型训练
- 周期性模型审计
- adapter/prototype 汇总
- 长期漂移分析
- 真实标签回灌后的校准更新

## 5. 通信协议

### 5.1 每事件 evidence packet

默认不传 raw IQ，不传完整 feature，不传全模型梯度。

```text
event_id: 8 bytes
receiver_id_hash: 2-4 bytes
topL_class_ids: L * 2 bytes
topL_log_probs: L * 1-2 bytes
entropy/margin/energy/temp: 4-8 bytes
aug_js/aug_stability: 2-4 bytes
style_vector_int8: 8-32 bytes
embedding_sketch_int8: 16-32 bytes, optional
reliability_q: 1-2 bytes
timestamp/geometry/link bucket: 8-16 bytes
```

估算：`80-160 bytes / receiver / event`。  
若 `M=8`，约 `0.6-1.3 KB / event`，远低于上传 IQ 或中间大特征。

### 5.2 周期性统计包

每 `W` 个事件或每个地面可见窗口上传：

```text
class prototypes: active classes only
style moments: mean/var of style vector
entropy histogram
ECE/calibration proxy
pseudo-label acceptance / rejection counts
adapter delta or low-rank update summary
Fisher diagonal or importance sketch, optional
```

## 6. 自适应 top-M 接收机选择

### 6.1 候选过滤

接收机 `r` 进入候选池必须满足：

```text
event aligned
SNR / energy above minimum threshold
entropy below tau_entropy_max
margin above tau_margin_min or not clearly OOD
augmentation JS below tau_aug
recent drift score below tau_drift
packet timestamp within stale window
```

### 6.2 打分与选择

对候选接收机打分：

```text
score_r =
  alpha * reliability_r
+ beta  * confidence_r
+ gamma * domain_match_r
+ delta * diversity_gain_r
- eta   * redundancy_r
- lambda * latency_r
```

含义：

- `reliability_r`：历史融合采纳率、校准误差、漂移告警反向分数
- `confidence_r`：低 entropy、高 margin、高 energy separation
- `domain_match_r`：style vector 与可信域原型相似
- `diversity_gain_r`：不同接收机、不同几何、不同链路条件带来的互补性
- `redundancy_r`：与已选接收机高度相似的惩罚
- `latency_r`：等待该节点的时延成本

默认策略：

- `K_min = 3`
- `M_default = 5`
- `M_max = 8`
- full-K 只在低置信或离线复盘时使用
- 若 top-M 分歧大，允许 request-more 或 abstain

## 7. 可信融合公式

当前 soft fusion 可视为：

```text
p_fused = mean_r p_r
```

AWARE-CI 改成温度校准后的 weighted log-opinion pool：

```text
log P(y) = sum_r w_r * log softmax(logits_r / T_r)
P(y) = softmax(log P(y))
```

权重：

```text
w_r = normalize(
    R_r^alpha
  * C_r^beta
  * A_r^gamma
  * D_r^delta
  * V_r
)
```

其中：

- `R_r`：历史可靠性
- `C_r`：当前置信度
- `A_r`：增强一致性 / 群体一致性
- `D_r`：域相似度
- `V_r`：多样性修正，避免相关错误重复计票

冲突保护：

- 若 top-1 多数一致但 JS divergence 中等，降低融合温度，保守输出。
- 若 top-1 分裂且无多数簇，输出 abstain 或请求更多接收机。
- 若某接收机 KL 离群、历史可靠性低，trimmed aggregation 降权或剔除。

## 8. 星上实时微调

### 8.1 可更新模块

禁止默认全模型在线训练。优先级如下：

1. calibration temperature / classifier bias
2. BN / Norm affine 与统计
3. receiver-style adapter / FiLM adapter
4. LoRA / low-rank convolution adapter
5. prototype memory / class centroid

Backbone 和 TX classifier 默认冻结。只有离线地面复盘或长窗口有标注反馈时，才考虑更大范围更新。

### 8.2 伪标签接受门控

样本进入在线更新必须满足：

```text
selected_K >= K_min
fused_entropy < tau_entropy
fused_margin > tau_margin
agreement_rate >= rho
mean_pairwise_JS <= tau_js
no receiver drift alarm
class queue not saturated
scenario not in high-risk blocklist, e.g. early storm_mp if unstable
```

高置信样本用于 CE；中置信样本只用于一致性，不给硬标签；低置信样本只缓存或拒绝。

### 8.3 在线 loss

```text
L =
  CE(adapter(x), y_pseudo)
+ lambda_cons  * consistency_loss(weak_aug, strong_aug)
+ lambda_proto * prototype_alignment
+ lambda_prior * class_balance_regularizer
+ lambda_anchor * KL(student || frozen_anchor)
+ lambda_ewc   * Fisher / important-parameter regularizer
```

### 8.4 防崩溃机制

- EMA teacher：伪标签来自 fusion + teacher，不来自刚更新后的 student。
- Rollback buffer：保存最近 N 个 adapter checkpoint。
- Class-balanced queue：每类伪标签样本上限，防止坍缩到少数 TX。
- Negative update gate：若 entropy 降低但类别多样性也快速降低，停止更新。
- Cold anchor：周期性与 frozen base model 比较，KL 偏离过大则降学习率或回滚。
- Scenario-aware update：`storm_mp`、`mixed_orbit` 初期只做 calibration/prototype，不直接 CE 更新 adapter。

## 9. 轨道级轻量同步

星间或轨道内不做高频全模型 FedAvg。同步对象：

```text
adapter delta, quantized / top-k sparse
prototype delta
temperature / calibration stats
style moments
uncertainty histograms
Fisher / importance sketch
```

同步策略：

- 轨道内：短周期聚合，适合链路稳定时的 adapter/prototype sync。
- 轨道间：中周期同步，只传压缩统计。
- 地面站窗口：长周期审计与重训练，合并真实标签反馈。

聚合时按：

```text
sync_weight_r =
  data_freshness
* reliability
* coverage
* energy_budget
* link_quality
* drift_novelty
```

不按本地样本数简单平均。

## 10. 推理流程伪代码

```python
def aware_ci_infer(event_packets):
    candidates = []
    for packet in event_packets:
        if not event_aligned(packet):
            continue
        if is_stale(packet) or low_signal_quality(packet):
            continue
        if high_entropy(packet) and low_margin(packet):
            continue
        if packet.aug_js > tau_aug:
            continue
        candidates.append(packet)

    if len(candidates) < K_min:
        return local_or_conservative_result(candidates)

    selected = adaptive_top_m(candidates, M_default, M_max)
    weights = reliability_confidence_domain_diversity_weights(selected)
    fused = weighted_log_opinion_pool(selected, weights)

    if conflict_too_high(selected, fused):
        return abstain_or_request_more(selected)

    maybe_enqueue_for_online_update(selected, fused)
    return fused
```

```python
def online_update(window):
    accepted = []
    for item in window:
        if passes_pseudo_label_gate(item):
            accepted.append(item)

    if too_few_or_class_imbalanced(accepted):
        return "skip"

    old_state = adapter.snapshot()
    train_adapter_only(accepted, loss=ce + consistency + proto + anchor)

    if health_metric_degraded():
        adapter.restore(old_state)
        lower_update_rate()
```

## 11. 分阶段实现计划

### Phase 1：只做更好的融合，不做在线微调

目标：证明 AWARE-CI fusion 是否比 soft fusion 稳。

新增模块建议：

- `evaluation/aware_collaborative_fusion.py`
- 在现有 `collaborative_inference_eval.py` 增加：
  - `--collab_fusion aware`
  - `--collab_top_m`
  - `--collab_min_k`
  - `--collab_weight_confidence`
  - `--collab_weight_diversity`
  - `--collab_reject_conflicts`

优先实现：

- entropy / margin 权重
- receiver historical reliability 从当前 eval 的 per-rx proxy 估计
- top-M 选择
- trimmed outlier removal
- coverage / abstain 统计

### Phase 2：加入 prototype / style domain score

目标：在星地场景区分“可信接收机”和“坏视图”。

实现：

- 从模型倒数第二层导出 embedding
- class prototype memory
- style vector：先用可得的 SNR/energy/CFO proxy 或 RF summary
- domain_match 权重

### Phase 3：在线 adapter / calibration TTA

目标：只更新小模块，验证是否能让 strict UDU 星地指标进一步提升。

实现：

- temperature calibration
- classifier bias / BN affine
- small adapter
- pseudo-label gate
- rollback
- accepted-only 与 full-denominator 双指标

### Phase 4：轨道级同步仿真

目标：评估多卫星断续链路下是否仍可运行。

实现：

- link delay / packet loss / stale evidence 仿真
- adapter delta quantization
- prototype sync
- async aggregation

## 12. 实验设计

### 12.1 数据与 split

沿用当前协议：

- WiSig train ratio `0.1`
- 训练接收机与测试接收机严格不重叠
- clean splits：
  - `test_unseen_day_seen_rx`
  - `test_seen_day_unseen_rx`
  - `test_unseen_day_unseen_rx`
- satellite scenarios：
  - `clear_leo`
  - `low_elev_leo`
  - `rain_leo`
  - `storm_mp`
  - `mixed_orbit`

### 12.2 Baselines

必须比较：

1. Single receiver K=1
2. Current soft fusion K=all
3. Soft fusion top-M
4. AWARE-CI without online update
5. AWARE-CI + calibration only
6. AWARE-CI + adapter TTA
7. AWARE-CI with bad receiver injected
8. AWARE-CI under communication budget

### 12.3 Ablations

- 权重项：
  - confidence only
  - confidence + agreement
  - confidence + domain match
  - confidence + agreement + domain + diversity
- top-M：
  - fixed K
  - adaptive K
  - full-K
- 通信：
  - full logits
  - top-L logits
  - int8 quantized logits
  - no embedding sketch
  - with embedding sketch
- 在线更新：
  - no TTA
  - Tent-like BN only
  - calibration only
  - adapter only
  - adapter + prototype
  - with / without rollback
- 风险：
  - low SNR receiver injected
  - random bad receiver
  - stale evidence
  - packet loss
  - asynchronous window

### 12.4 指标

性能：

- full-denominator accuracy
- accepted-only accuracy
- coverage / reject rate
- rescue / harm / net gain
- macro-F1
- per-TX accuracy
- per-receiver accuracy
- strict UDU
- satellite scenario mean / worst-case

效率：

- bytes/event
- bytes/window
- latency p50 / p95
- selected K distribution
- GPU/CPU memory
- online update steps/window
- trainable parameter count
- energy proxy

稳定性：

- rollback count
- pseudo-label acceptance rate
- class entropy over accepted queue
- drift alarms
- negative transfer incidents

## 13. 预期收益与边界

预期收益：

- 在 clean 场景，AWARE-CI 应接近 full-K soft fusion，同时用更少接收机达到接近性能。
- 在星地 strict UDU，AWARE-CI 应优先减少 harm，把 low-quality receiver 降权，从而提升 full-denominator accuracy。
- 在通信受限场景，top-L + top-M 应显著降低通信量。
- 在线 adapter TTA 可能改善 `clear_leo/low_elev/rain`，但 `storm_mp/mixed_orbit` 必须谨慎门控。

不能提前宣称：

- 不能宣称已经“解决未见接收机泛化”，除非 unseen receiver / unseen day / unseen channel 都独立验证。
- 不能把 accepted-only accuracy 当总体 accuracy。
- 不能把仿真星地信道直接外推为真实星上链路。
- 不能声称全模型星上实时训练可行；本方案主张小模块在线适配。
- 不能声称协同一定优于单星；低质量接收机可能造成负迁移。

## 14. 多子 agent 结论整合

| Agent | 交付物 | 被采纳点 |
|---|---|---|
| 文献检索 | RFFI、TTA、卫星 FL、协同边缘推理方法 | 采用 Trusted Receiver Ensemble + Source-Free Adapter TTA + Orbit-Level Adapter Sync |
| 高效率算法 | AWARE-CI 算法草案 | 采用 evidence packet、adaptive top-M、weighted log-opinion pool、adapter-only update |
| 完成监督 | 需求清单与成功标准 | 文档覆盖目标、通信、融合、在线微调、部署、实验、风险 |
| 查漏补缺/review | 风险与最小 MVP | 强化 coverage/full accuracy、伪标签防崩溃、通信预算、异步与坏节点测试 |

## 15. 最小可实现版本

推荐从 MVP 开始：

> Frozen backbone + 每接收机本地推理 + top-L logits/evidence packet + adaptive top-M + uncertainty/domain weighted fusion + coverage/reject 统计。

不先做在线微调。先证明：

1. 在 `test_unseen_day_unseen_rx` 和五个星地场景下，AWARE-CI 是否比 soft fusion 降低 harm、提高 fused accuracy。
2. 在 `M < full-K` 时，是否能接近 full-K soft fusion。
3. 在低质量接收机注入时，是否自动降权而不是负迁移。

MVP 通过后，再加入：

1. calibration-only TTA
2. BN/adapter TTA
3. prototype sync
4. orbit-level adapter sync

## 16. 推荐下一步

1. 实现 `aware` fusion evaluator，不动训练主链路。
2. 在 SA33 checkpoint 上复跑 clean + satellite full eval：
   - baseline soft full-K
   - aware top-M
   - aware full-K
   - bad receiver injection
3. 若 aware fusion 提升明显，再设计 adapter-only TTA。
4. 全程报告 full-denominator accuracy、coverage、accepted-only accuracy、communication bytes/event，避免只报好看的 accepted 子集。

## 17. 参考来源

- Tent: Fully Test-time Adaptation by Entropy Minimization: https://arxiv.org/abs/2006.10726
- Continual Test-Time Domain Adaptation / CoTTA: https://arxiv.org/abs/2203.13591
- EATA: Efficient Test-Time Adaptation without Forgetting: https://proceedings.mlr.press/v162/niu22a.html
- Domain Generalization for Cross-Receiver RFFI / FedRIEI: https://arxiv.org/abs/2411.03636
- Federated RFFI powered by Unsupervised Contrastive Learning: https://livrepository.liverpool.ac.uk/id/eprint/3184752
- Attention-aware Semantic Communications for Collaborative Inference: https://arxiv.org/abs/2404.07217
- Joint Channel and Semantic-aware Grouping for Collaborative Edge Inference: https://arxiv.org/abs/2510.02191
- Cross-Device Collaborative Test-Time Adaptation: https://proceedings.neurips.cc/paper_files/paper/2024/file/de0e668df3fe63ec89e5a7e68f3d350f-Paper-Conference.pdf
- LoRA-Edge: Tensor-Train-Assisted LoRA for Practical CNN Fine-Tuning on Edge Devices: https://arxiv.org/abs/2511.03765
- Cross-Receiver RFFI Source-Free Adaptation / CSCNet: https://www.mdpi.com/1424-8220/25/14/4451
