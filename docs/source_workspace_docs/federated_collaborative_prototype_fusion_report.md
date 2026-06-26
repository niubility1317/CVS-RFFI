# 借鉴协作推理的多原型分类头联邦学习分析报告

## 1. 核心结论

RA-Collab 的协作推理给多原型分类头最大的启示是：不要把 prototype 当成一个必须独立赢过 base classifier 的新分类器，而要把它当成多源证据的软投票器、校正器和可拒绝的辅助判断来源。

在联邦学习场景下，这个启示更重要。每个客户端天然对应一个 receiver、receiver-day 或 receiver-channel domain view。多原型机制不应再被设计成一个中心化的强分类头，而应转向：

```text
Federated Collaborative Prototype Bank
```

也就是：

- 每个客户端维护本地 class/domain prototypes；
- 服务端保留多客户端、多模式 prototype mixture；
- 推理阶段做概率级 soft fusion 或 reliability-weighted fusion；
- prototype 只做低强度、可拒绝、可信度加权的校正；
- global/base classifier 始终作为 anchor；
- prototype 的目标是 rescue hard/boundary samples，而不是全面替代 base classifier。

一句话概括：多原型头过去失败，不一定说明 prototype 方向错误，更可能说明 prototype 被放在了错误的位置。它不适合当强分类头，更适合在联邦场景中作为跨客户端协作证据。

## 2. RA-Collab 协作融合的关键机制

RA-Collab 的协作推理不是训练一个额外分类器，而是在评估或推理阶段，把同一个信号在多个 receiver 上的 TX 预测概率融合。

普通 soft fusion：

```text
p_final = mean(p_rx1, p_rx2, ..., p_rxN)
```

带可靠性或 SNR 的 adaptive soft fusion：

```text
p_final = sum_i w_i * p_rxi
```

其中 `w_i` 表示第 `i` 个 receiver 观测的可信度。

这个机制有三个重要特征：

1. 融合的是概率证据，而不是直接强行修改 logits。
2. 融合发生在推理阶段，不把 fusion 变成容易过拟合源域的训练目标。
3. 它承认不同 receiver 的证据质量不同，因此允许软加权，而不是让某一个分支支配决策。

这对多原型头的直接启示是：prototype 也应该被看作多个证据源，而不是一个必须单独承担 CE 分类目标的主分类器。

## 3. 过去多原型分类头的失败机制

当前仓库中的 FJMP 主要包含以下部件：

- `code/FJMP/frozen_joint_prototype_head.py`：`FrozenJointPrototypeClassifier`、`MultiPrototypeHead`、`CalibratedFusion`、`ConfidenceGate`；
- `code/FJMP/base_protected_fusion.py`：`BaseProtectedFusion`；
- `code/FJMP/fjmp_v2_proto_head.py`：`SafeResidualProtoHead`；
- `code/FJMP/fjmp_v2_losses.py`：safe residual prototype 的分阶段损失；
- `code/FJMP/train_fjmp.py`：FJMP 训练、评估、融合和日志入口。

从已有设计和历史实验看，多原型头的失败主要来自以下几个机制。

### 3.1 Prototype Branch 本身弱于 Frozen Base

多原型头通常只包含一个轻量 projector、若干 class prototypes 和一个 fusion 模块，而 base classifier 来自完整训练好的 backbone。让一个小型 prototype head 独立学出比 base 更强的跨域判别能力，本身就很困难。

因此，prototype branch 更合理的角色是：

```text
边界校正信号 + hard sample rescue signal
```

而不是：

```text
替代 base classifier 的新主分类器
```

### 3.2 CE on Fused 短期有益但长期过拟合源域

历史结果中，A03/A06 这类配置在早期 epoch 出现过 UDU 提升，但继续训练后回落。这说明 `CE on fused` 确实能在短期调节决策边界，但它优化的是源域训练分布。

当训练继续推进时，prototype head 会逐步拟合源域里的 receiver/day/channel 残差噪声，导致：

```text
source validation 继续变好
unseen day + unseen receiver 逐渐变差
```

这是一种典型的 source-domain overfitting。

### 3.3 Logit 级强融合容易产生 Harm

旧 FJMP 的一些融合形式直接在 logits 层面修改 base 决策：

```text
calibrated_logit: fused = beta * base_logits + alpha * proto_logits
residual:         fused = base_logits + eta * centered(proto_logits)
```

如果 prototype 的跨域可靠性低于 base，强行修改 logits 会造成：

```text
base correct -> fused wrong
```

这就是 harm。只要 harm_rate 大于 rescue_rate，prototype 就会降低最终性能。

### 3.4 安全损失太重会让模型失去改善能力

为了解决 harm，后来加入了 DNH、margin preservation、KD、base-protected residual、SGV consistency 等安全约束。这些约束可以降低破坏性，但如果权重过大，就会出现另一个问题：

```text
不会伤害 base，但也不会改善 base
```

SGV-BP 这类设计的典型问题是安全损失项过多、量级过大，分类信号被淹没，prototype head 没有足够自由度产生有效 correction。

## 4. 联邦学习带来的新机会

联邦学习反而让 prototype 的价值更自然。

在集中式训练中，多原型头只是一个附加小头，很容易被 base 压制。但在 FL 中，每个客户端本身就是一个 domain source，例如：

- receiver client；
- receiver-day client；
- receiver-channel client；
- receiver-day-channel client。

因此，每个客户端的 prototype 不应该被简单平均掉，而应该被视为一个 domain-specific evidence source。

这和 RA-Collab 的思想非常接近：

```text
RA-Collab:
  多 receiver 预测 -> soft/adaptive fusion

Federated prototype:
  多 client prototype evidence -> soft/adaptive fusion
```

也就是说，prototype 在 FL 中不只是模型参数，而是跨客户端知识载体。

## 5. 推荐范式：Federated Collaborative Prototype Bank

建议将过去的多原型分类头改造成三层结构。

### 5.1 客户端本地 Prototype 统计

每个客户端 `i` 对每个 TX 类 `c` 维护若干 prototype：

```text
P_i,c,k
```

其中：

- `i` 是客户端；
- `c` 是发射机类别；
- `k` 是同类下第 `k` 个 prototype；
- `P_i,c,k` 表示 client `i` 上 class `c` 的一个局部特征中心。

客户端不应只上传 prototype 向量，还应上传可靠性统计：

```text
n_i,c,k          样本数
margin_i,c,k     本地分类 margin
entropy_i,c,k    prototype assignment entropy
acc_i            本地验证可靠性
drift_i          client local model 与 global model 的偏移
```

原因是：裸 prototype 没有可信度。服务端必须知道哪些 prototype 来自稳定客户端，哪些来自小样本、偏移大或噪声大的客户端。

### 5.2 服务端保留 Mixture，而不是粗暴平均

不建议这样做：

```text
P_global,c = average_i P_i,c
```

因为不同 receiver/domain 的 class center 可能本来就不一致，直接平均会抹平有用的 domain modes。

更合理的是保留 mixture：

```text
Bank_c = {P_i,c,k, reliability_i,c,k}
```

必要时可以做聚类压缩：

```text
client prototypes -> server clustering -> top-M representative prototypes
```

但不应把所有客户端的 prototype 强行压成一个均值。

### 5.3 推理阶段做 Prototype Evidence Fusion

给定一个测试样本特征 `z`，先由 global classifier 得到：

```text
p_base = softmax(global_logits)
```

再由 prototype bank 产生多个 client-prototype evidence：

```text
p_proto_i = softmax(sim(z, P_i,*,*) / tau)
```

最终融合：

```text
p_final = (1 - rho) * p_base + rho * sum_i w_i(z) * p_proto_i
```

其中：

- `rho` 是 prototype 介入强度；
- `w_i(z)` 是样本对 client `i` prototype evidence 的可信权重；
- `p_base` 始终保留为 anchor；
- `p_proto_i` 只做辅助校正。

这比旧 FJMP 的 logit 强融合更安全，因为它在概率层面融合，并且可以控制 prototype 的介入比例。

## 6. 推荐损失函数设计

联邦学习下不要让 prototype fused logits 承担全部 CE。推荐本地训练目标为：

```text
L_local =
  L_CE
+ mu * ||theta_i - theta_global||^2
+ lambda_pull * L_proto_pull
+ lambda_div * L_proto_diversity
+ lambda_cons * L_clean_sat_proto_consistency
+ lambda_kd * L_selective_KD
```

各项含义如下。

### 6.1 主分类 CE

```text
L_CE = CE(global_or_local_logits, y)
```

CE 主要训练主干和普通分类器，不建议直接让 prototype fused logits 承担主 CE。否则容易重复旧 FJMP 的问题，即短期提升、长期过拟合源域。

### 6.2 FedProx 约束

```text
L_prox = mu * ||theta_i - theta_global||^2
```

用于限制客户端本地更新偏离 global model 太远，尤其适合 receiver/day non-IID 场景。

### 6.3 Prototype Pull Loss

```text
L_proto_pull = distance(z, nearest P_i,y,k)
```

只要求同类特征靠近可靠 prototype，不要求 prototype branch 独立完成分类。

### 6.4 Prototype Diversity Loss

```text
L_proto_diversity
```

防止同一类的多个 prototype 坍缩到同一个点。历史实验中 K=3 的短期收益，很可能就来自更好的同类多模式覆盖。

### 6.5 Clean-Sat Prototype Consistency

星地信道训练需要保留，但建议作用在 prototype assignment 或 prototype evidence 上，而不是只做 logits consistency。

推荐：

```text
L_clean_sat_proto_consistency =
  KL(q_proto(clean) || q_proto(sat))
+ KL(q_proto(sat) || q_proto(clean))
```

其中 `q_proto` 是样本对同类 prototypes 的 assignment distribution。

这样做的好处是：它约束的是“同一个 TX 在 clean/sat view 下落到相近的 prototype mode”，更符合 prototype bank 的角色。

### 6.6 Selective KD

不要对所有样本都让 fused/prototype 模仿 base。否则 base 错误样本也会被锁死。

推荐只在 base 高置信正确样本上做 KD：

```text
if base_correct and base_conf high and base_margin high:
    apply KD
else:
    no KD
```

这保留 base 的强项，同时允许 prototype 在 hard samples 上提供 rescue。

## 7. 融合策略建议

### 7.1 避免强 Logit Fusion 作为默认

不建议默认使用：

```text
fused = beta * base_logits + alpha * proto_logits
```

或者过强的：

```text
fused = base_logits + eta * proto_residual
```

除非有非常严格的 gate 和 rho cap。

### 7.2 推荐概率级 Conservative Fusion

推荐从下面这个形式开始：

```text
p_final = (1 - rho) * p_base + rho * p_proto
```

初始设置：

```text
rho = 0.02, 0.05, 0.10
```

如果 `rho=0.02` 都会带来明显 harm，说明 prototype evidence 本身不可靠，应先修 prototype bank，而不是增强 fusion。

### 7.3 Reliability-Weighted Fusion

进一步可以做：

```text
p_final = (1 - rho) * p_base + rho * sum_i w_i * p_proto_i
```

`w_i` 可以来自：

- client 样本数；
- client validation accuracy；
- prototype assignment entropy；
- feature-prototype distance；
- client drift；
- clean/sat consistency；
- local loss moving average。

### 7.4 Hard Gate 只作为保护，不作为主机制

gate 应该回答：

```text
这个 prototype evidence 是否足够可靠，可以允许它参与融合？
```

而不是：

```text
这个 prototype 是否直接覆盖 base prediction？
```

推荐 gate 条件：

```text
base_margin low
prototype_entropy low
prototype_distance low
multi-client evidence agrees
```

只有这些条件同时满足时，prototype 才更可能提供 rescue。

## 8. 联邦场景中的 Prototype 聚合策略

### 8.1 FedAvg 参数聚合与 Prototype 聚合应分离

模型参数可以 FedAvg 或 FedProx 聚合，但 prototype 不应简单跟随参数平均。

推荐分离为：

```text
theta_global = FedAvg/FedProx(theta_i)
prototype_bank = reliability-aware mixture update(P_i,c,k)
```

### 8.2 Prototype Bank 更新

服务端可以使用动量更新：

```text
P_bank <- m * P_bank + (1 - m) * P_client
```

但必须按 class 和 reliability 做选择。

例如：

```text
accept P_i,c,k only if:
  n_i,c,k >= n_min
  entropy_i,c,k <= entropy_max
  drift_i <= drift_max
```

### 8.3 保留 Client Identity，但推理时不泄露标签

prototype bank 可以保留 `client_id` 或 domain type 作为统计来源，但推理时不要依赖测试样本真实 receiver 标签。权重应由样本特征到 prototype 的距离和可靠性自动决定。

## 9. 推荐实验路线

### 9.1 Stage 0：联邦基础对照

先确认 FL 主线是否稳定：

```text
FedAvg receiver
FedAvg receiver_day
FedProx receiver
FedProx receiver_day
```

观察：

- global validation；
- unseen day seen rx；
- seen day unseen rx；
- unseen day unseen rx；
- worst receiver group；
- client drift；
- FedProx prox loss。

### 9.2 Stage 1：Prototype 只做辅助训练，不参与推理

加入 prototype pull/diversity/clean-sat consistency，但最终仍用 global classifier logits。

目的：

```text
确认 prototype regularization 是否改善 representation
```

如果这一步都没有收益，不要进入复杂 fusion。

### 9.3 Stage 2：Prototype Inference Only

不训练新 fusion head，只在推理时做：

```text
p_final = (1-rho) p_base + rho p_proto
```

扫描：

```text
rho = 0.02, 0.05, 0.10
tau = 0.05, 0.10, 0.20
K = 1, 2, 3, 4
```

这一步的目标是验证 prototype evidence 是否有真实推理价值。

### 9.4 Stage 3：Client-Prototype Collaborative Fusion

保留 client-wise prototype mixture：

```text
p_proto = sum_i w_i(z) p_proto_i
```

比较：

```text
uniform client weights
sample-count weights
entropy weights
distance weights
validation-accuracy weights
drift-penalized weights
```

这一步直接借鉴 RA-Collab 的协作融合思想。

### 9.5 Stage 4：Gate 和 Rescue 机制

只允许 prototype 在 hard/boundary samples 介入：

```text
if base_margin < threshold and prototype_confidence high:
    use prototype fusion
else:
    use base prediction
```

重点观察：

- rescue_rate；
- harm_rate；
- net_gain_rate；
- changed_pred_rate；
- per-client worst accuracy。

### 9.6 Stage 5：星地信道一致性 Prototype Bank

在 clean 和 satellite-ground view 下约束 prototype assignment：

```text
q_proto(clean) ≈ q_proto(sat)
```

而不是只约束 logits。这样更符合 prototype 的“模式归属”角色。

## 10. 关键指标

除了常规 accuracy，必须记录以下指标。

### 10.1 Base/Prototype/Fused 三路指标

```text
base_tx_acc
proto_tx_acc
fused_tx_acc
```

分别在：

```text
val_source
test_unseen_day_seen_rx
test_seen_day_unseen_rx
test_unseen_day_unseen_rx
```

上记录。

### 10.2 Harm/Rescue 指标

```text
changed_pred_rate
rescue_rate
harm_rate
net_gain_rate = rescue_rate - harm_rate
```

这是判断 prototype fusion 是否值得继续的核心。

### 10.3 Prototype Bank 指标

```text
prototype_usage_entropy
dead_proto_rate
prototype_pairwise_cos_mean
prototype_pairwise_cos_max
assignment_entropy
client_prototype_drift
```

### 10.4 联邦特有指标

```text
client_drift_norm
client_update_norm
client_data_count
client_class_coverage
per_client_val_acc
per_client_proto_reliability
server_bank_accept_rate
```

### 10.5 星地视图指标

```text
clean_sat_proto_assignment_KL
clean_sat_prediction_gap
sat_scenario_worst_acc
sat_view_reliability
```

## 11. 最小可行实现建议

建议不要一开始就重写完整 FJMP，而是先做一个轻量版本：

### 11.1 Local Prototype Collector

每个 client 本地统计：

```text
class_mean_prototype
class_count
class_feature_var
class_margin_mean
```

先从 `K=1` 开始。

### 11.2 Server Prototype Bank

服务端保存：

```text
bank[class_id] = [
  {
    "client_id": ...,
    "prototype": ...,
    "count": ...,
    "reliability": ...
  }
]
```

### 11.3 Inference-Time Fusion

先不训练 fusion 参数，只用固定公式：

```text
p_final = 0.95 * p_base + 0.05 * p_proto
```

如果这个固定 5% prototype evidence 有收益，再考虑 gate 和 learnable rho。

## 12. 风险与注意事项

### 12.1 Prototype 泄露 Receiver Shortcut

如果 client 是 receiver，prototype 很容易携带 receiver-specific shortcut。解决方式：

- 不用 receiver label 参与推理；
- 使用 z_id 或 receiver-agnostic feature；
- 用 GRL 或 receiver adversarial loss 降低 receiver 信息；
- prototype reliability 对高 drift client 降权。

### 12.2 小样本 Client Prototype 不稳定

few-shot client 的 prototype 噪声会很大。解决方式：

- 设置 `n_min`；
- 使用 shrinkage：

```text
P = alpha * P_client + (1-alpha) * P_global_class
```

- 样本少的 client 只上传统计，不参与 fusion。

### 12.3 Prototype Bank 过大

client 数和 K 增加后 bank 会膨胀。解决方式：

- 每类 top-M；
- server clustering；
- 删除低可靠 prototype；
- 定期合并相似 prototypes。

### 12.4 Fusion 权重过大导致 Harm

默认 `rho` 必须非常小。建议从：

```text
rho = 0.02
```

开始，而不是直接 `0.1` 或更大。

## 13. 推荐优先级

最高优先级：

1. FedProx receiver / receiver_day baseline 稳定性；
2. client-wise prototype bank 统计，不参与 logits；
3. inference-only probability fusion；
4. reliability-weighted client prototype fusion；
5. clean/sat prototype assignment consistency。

暂时不建议优先做：

1. 复杂 learnable fusion head；
2. 强 CE on fused；
3. 大权重 margin preservation；
4. 大型 SGV-BP 多损失组合；
5. 直接让 prototype branch 独立替代 base classifier。

## 14. 总结

过去多原型分类头的失败，核心不是 prototype 概念本身失败，而是它被设计成了一个过重、过强、过容易源域过拟合的分类头。

RA-Collab 的协作推理提示我们：多源证据应该保留其来源差异，并在推理时以可靠性加权协作，而不是在训练时压成一个强分类器。

在联邦学习中，这一点尤其自然。每个 client 都可以贡献自己的 class prototype evidence。服务端不需要把这些 evidence 粗暴平均，而应构建一个可靠性加权的 collaborative prototype bank。

最终推荐方向是：

```text
Global classifier as anchor
+ Client-wise prototype bank as collaborative evidence
+ Conservative probability fusion
+ Reliability/gate-based rejection
+ Clean-sat prototype consistency
```

这条路线比复活旧式中心化多原型分类头更稳，也更符合你现在转向联邦学习的实验主线。
