# COSR-CI: ADV3B02 + qknn8 面向卫星群的协同开集 RFFI 设计草案

日期：2026-07-03  
底座模型：`ADV3B02_CORE90_SOFT_E200`  
在轨少样本方法：`qknn8`  
目标：通过现实可部署的卫星群协同推理，同时提升旧类、新类和未知类拒识。

## 1. 结论先行

推荐算法命名为 **COSR-CI**：Collaborative Open-Set Receiver Consensus Inference。

它不是把多接收机 softmax 简单平均，也不是把 qknn8 结果再投票。核心是把每个卫星/接收机变成一个低带宽 evidence node：

1. 本地用 `ADV3B02_CORE90_SOFT_E200` 冻结特征提取。
2. 本地用 `qknn8` 维护旧类 target support 与 seen-new support 的 int8 量化记忆。
3. 本地对每个 query 输出三类证据：old evidence、seen-new evidence、unknown risk。
4. 星间/星上聚合器按接收机质量、预测一致性、open-set 风险、链路时延选择 top-M 节点。
5. 聚合器输出四态决策：old label、seen-new label、unknown reject、defer/request-more。

这条路线直接针对当前失败面：

- qknn8 已能做少样本 old/new，但缺 unknown。
- Phase1 拒识能低 FAR，但旧类覆盖差。
- 闭集协同已有 `1..N` 接收机评估，但缺开集语义。

COSR-CI 将三者合并，但保留各自边界：unknown query 不调阈值，qknn8 不保存原始 support，协同只传低带宽证据。

## 2. 当前证据边界

当前不能声称目标已完成。

| 能力 | 当前证据 | 边界 |
|---|---|---|
| 闭集协同 | `code/evaluation/collaborative_inference_eval.py` 支持 `collab_counts all`。 | 只报闭集 base/fused/rescue/harm。 |
| qknn8 少样本新类 | `phase2_adv3b02_qknn_support_select_k10k15_no_unknown_20260703_1312` 最大 query K15 达 old 85.13%、new 86.15%。 | 明确不导出、不评估 unknown；离 99/97 目标很远。 |
| Phase1 unknown reject | `phase1_adv3b02_open_set_reject_20260702` 有低 FAR 安全门。 | 旧类 coverage/full accuracy 下降，不能作为完整部署方案。 |
| satellite-only reject | `phase1_adv3b02_satonly_open_set_reject_20260702` 证明单观测 LEO 下 FAR/旧类保留冲突仍存在。 | 需要协同而非单节点阈值继续硬压。 |

因此目标指标应作为后续研究 success gate，而不是已有结果：

| 指标 | 目标 |
|---|---:|
| old overall accuracy | 99% |
| old per-class floor | 95% |
| seen-new overall accuracy | 97% |
| seen-new per-class floor | 93% |
| unknown rejection | 99% |

## 3. 协议定义

集合约束沿用 `项目.md`：

```text
intersection(R_t, R_s) = empty
intersection(Y_new, Y_old) = empty
intersection(Y_unknown, union(Y_old, Y_new)) = empty
```

部署阶段每个目标 receiver domain `R_t` 可以包含多颗卫星/多台未见接收机节点，但每个节点必须来自训练未见接收机集合。所有 target-old、target-new、unknown query 必须处于同一 satellite/LEO target view 定义下。

未知类只用于最终评估，不参与：

- qknn8 support 建库；
- threshold fitting；
- receiver reliability 标定；
- top-M selector 监督训练。

## 4. 节点本地 evidence

每个节点 `r` 对同一事件 `e=(tx/day/eq/sig 或真实时间-频率对齐键)` 输出：

```text
base_topL_logits_r
qknn8_old_new_scores_r
best_old_label_r, best_old_margin_r
best_new_label_r, best_new_margin_r
unknown_risk_r
closed_confidence_r
open_set_margin_r
feature_quality_r
support_density_r
receiver_health_r
latency_estimate_r
```

其中：

- `qknn8_old_new_scores_r` 来自 int8 support memory，不保存 full precision support embedding。
- `unknown_risk_r` 由 source/proxy-only 或 support-only 背景风险估计得到，不能用 unknown query 拟合。
- `feature_quality_r` 可由 SNR/energy、entropy、top1-top2 margin、与 support 半径的关系给出。
- `receiver_health_r` 来自最近窗口的自一致性、defer 比例、与群体分歧率。

## 5. Evidence packet 与资源预算

每事件每接收机上传：

```text
event_id                  8 bytes
receiver_id_hash          2 bytes
topL class ids/logits     15-30 bytes, L=5
qknn8 best old/new ids     4 bytes
qknn8 margins/scores       8-16 bytes, int16/fp16
unknown risk fields        6-12 bytes
quality/reliability        6-12 bytes
timing/link buckets        6-12 bytes
optional z sketch          16-32 bytes int8
```

默认预算：

| 模式 | 每节点每事件 | M=5 | M=8 |
|---|---:|---:|---:|
| score-only | 64-96 bytes | 320-480 bytes | 512-768 bytes |
| score + z sketch | 96-144 bytes | 480-720 bytes | 768-1152 bytes |

星上持久状态：

| 状态 | 默认规模 |
|---|---:|
| qknn8 support code | `classes * K * dim` int8；当前 K=15、8 类、dim 若 128，则约 15 KB，不含小元数据 |
| per-class prototypes/radii | 数 KB |
| reliability window | 数 KB |
| adapter/calibration 可选 | 默认关闭；若开启应限制在 MB 级以内 |

必须报告：

- `participating_receivers`
- `eligible_receivers`
- `bytes_per_event`
- `fusion_latency_ms_p50/p95`
- `qknn_search_latency_ms`
- `unknown_gate_latency_ms`
- `state_size_bytes`
- `max_vram_mb` 或 CPU 内存峰值

## 6. Top-M 选择

不默认 full-K。每个事件从可见节点中选择 top-M：

```text
score_r =
  a * receiver_health_r
+ b * closed_or_open_confidence_r
+ c * support_density_r
+ d * group_diversity_gain_r
- e * unknown_conflict_penalty_r
- f * latency_penalty_r
```

默认配置：

```text
M_min = 1
M_default = 5
M_max = 8
full_K = offline audit only or low-confidence emergency
```

当 `M=1` 时就是单节点 qknn8/open-set baseline。报告必须从 `1` 到 `N` 全量输出，不能只挑 best-M。

## 7. 协同融合决策

COSR-CI 使用三层仲裁。

### 7.1 层一：节点级质量过滤

过滤掉：

- event 对齐失败；
- timestamp 超过 stale window；
- feature quality 过低；
- qknn8 support density 低于下限；
- recent disagreement 过高；
- unknown risk 无法计算。

### 7.2 层二：old/new/unknown 分数聚合

对每个候选标签 `y`：

```text
S_known(y) = sum_r w_r * log P_r(y)
```

对未知风险：

```text
R_unknown = weighted_quantile({risk_r}, q=0.75)
```

使用高分位 unknown risk，而不是平均 risk，原因是未知类 false accept 是安全问题；一个高质量节点强烈认为 unknown，应触发 defer/reject 复核。

### 7.3 层三：四态输出

```text
if R_unknown >= tau_reject and known_consensus < tau_rescue:
    output = unknown_reject
elif best_known_margin >= tau_accept and agreement >= tau_agree:
    output = best old/new label
elif M < M_max and latency budget remains:
    output = request_more_receivers
else:
    output = defer
```

关键点：`defer` 不能在最终精度中被悄悄排除。报告必须同时给：

- full accuracy；
- accepted-only accuracy；
- coverage；
- defer rate；
- unknown false accept rate；
- unknown reject rate。

## 8. qknn8 与协同的关系

每个节点本地执行 qknn8：

```text
z = ADV3B02(x)
zq = quantize_int8(z)
neighbors = topk_cosine(zq, support_codes)
local_known_score = radius_norm_knn(neighbors)
```

协同不共享 support 原样本，也不共享 full precision embedding。可共享：

- quantized support code hash；
- per-class support count；
- per-class radius；
- per-class reliability；
- query side top-k distances；
- optional int8 z sketch。

推荐第一版不做跨节点 support memory 合并。每节点独立建 qknn8 记忆，聚合器只融合 evidence。这样最容易定位错误：若协同失败，问题在 evidence fusion；若单节点 qknn8 失败，问题在 support 或特征。

## 9. 指标与验收门槛

每个 `K_receivers=1..N` 都要输出：

| 组 | 指标 |
|---|---|
| old | overall old acc、per-old-class acc、old floor、old coverage、old defer |
| seen-new | overall new acc、per-new-class acc、new floor、new coverage、new defer |
| unknown | unknown reject、unknown FAR、unknown->old、unknown->new |
| open-set | AUROC、FPR95、accepted-only 与 full denominator 对照 |
| 协同 | rescue、harm、net gain、eligible group count、excluded incomplete group count |
| 资源 | participating receivers、bytes/event、latency p50/p95、state size、memory/VRAM |

成功门槛严格定义为同一实验矩阵、同一阈值策略、同一 full denominator 下同时满足：

```text
old_acc >= 0.99
min_old_class_acc >= 0.95
seen_new_acc >= 0.97
min_seen_new_class_acc >= 0.93
unknown_reject >= 0.99
unknown_FAR <= 0.01
```

若通过 defer 降低 FAR，还必须满足部署覆盖约束。建议默认：

```text
old_coverage >= 0.95
seen_new_coverage >= 0.95
unknown_eval_coverage = 1.0
```

否则只能声明“安全筛查模式”，不能声明识别性能达标。

## 10. 实现阶段建议

### Phase A：离线 evidence evaluator

新增 evaluator，输入已导出的 feature/evidence，输出 `K=1..N` 协同开集指标。此阶段不跑模型前向，便于快速扫融合策略。

建议文件：

```text
code/evaluation/collaborative_open_set_qknn_eval.py
code/tests/test_collaborative_open_set_qknn_eval.py
```

### Phase B：接入 ADV3B02 feature export

复用已有 feature npz，要求包含：

```text
features
tx_logits
dataset_role
tx_ids
rx_ids
day_ids
sig_ids
sat_scenarios/channel_views
```

若缺少跨 receiver 同事件对齐键，则先只能做 receiver-domain ensemble，不得称为严格 event-level collaborative inference。

### Phase C：N607 全量矩阵

矩阵：

```text
target receivers: 20-1, 3-19, 7-14, 7-7, 8-8
K-shot: 5, 10, 15
receiver collaboration count: 1..N
views: leo_clear_weak, leo_low_elev_weak, leo_rain_weak
unknown pairs: at least 2 disjoint pairs
```

第一轮只跑 eval，不训练主干，不全模型微调。

## 11. 风险与约束

1. 99/97/99 目标非常高，当前单节点 evidence 明显不足；需要把它作为研究目标而不是承诺已有可达。
2. 如果多个接收机观测并非同一事件，协同只能叫 receiver-domain ensemble，不能叫 event-level 协同。
3. 如果 defer rate 很高，unknown reject 99% 没有部署意义。
4. 如果 resource constraint 原文找到后更严格，应优先收紧 evidence packet、M_max 和状态大小。
5. qknn8 对 support 选择敏感；协同不能掩盖某些 TX floor 极低的问题，例如当前报告中 `20-19` 旧类瓶颈。

## 12. 推荐下一步

先实现 Phase A。验收只看是否能在本地对 mock evidence 完成：

1. `K=1..N` 输出；
2. old/new/unknown full-denominator 指标；
3. per-class floor；
4. unknown FAR/reject；
5. bytes/event 与 latency 字段；
6. defer/coverage 分离；
7. unknown query 不参与阈值拟合的单元测试。

这一步通过后，再把已有 ADV3B02/qknn8 feature npz 接进来做真实离线扫描。
