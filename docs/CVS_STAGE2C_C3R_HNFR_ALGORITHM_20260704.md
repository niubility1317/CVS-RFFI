# CVS Stage2-C C3R-HNFR算法设计

## 目标

面向天基射频指纹识别卫星群协同推理，优先降低未知类误接收，同时旧类准确率和旧类逐类下限不能下降。该设计遵循CVS Stage2-C协议：`target_unknown`只用于最终评估，不能用于训练、阈值选择、候选挖掘或profile选择。

## 核心结论

当前ADV3B02特征空间中，部署侧COTE/C3R阈值融合存在强trade-off：拒识越强，旧类误拒越重。因此下一版算法必须先修复特征空间，再做星上轻量协同。推荐路线为`C3R-HNFR`：Conservative Collaborative Conformal Rejection with Hard-Negative Feature Repair。

## 数据边界

| 集合 | 用途 | 禁止事项 |
|---|---|---|
| `Y_old` source/target-old support | 旧类原型、旧类core radius、旧类回放保护 | 不允许被unknown veto压垮 |
| `Y_new` K-shot support | 新类注册原型 | 不允许用未知query聚类冒充seen-new |
| source-heldout `proxy_unknown` | hard-negative训练和拒识阈值校准 | 不得与`Y_old/Y_new/Y_unknown`重叠 |
| `Y_unknown` query | 最终拒识评估 | 禁止用于训练、阈值、候选筛选、profile选择 |
| source receiver | proxy挖掘和地面校准 | 不得混入target receiver统计 |

## 地面训练阶段

先用source-heldout TX构造hard negatives。候选必须满足：

```text
proxy_tx notin Y_old union Y_new union Y_unknown
proxy_rx intersect target_rx = empty
target_unknown_used_for_scoring = false
```

对冻结或半冻结的ADV3B02主干，只更新轻量投影头或adapter，目标是扩大旧类核心区与未知拒识区的间隔。

损失建议：

```text
L = L_ce + lambda_proto L_proto + lambda_evt L_evt + lambda_hn L_hn + lambda_old L_old_replay

L_proto = mean_i || normalize(z_i) - c_y ||_2^2
L_hn = mean_p softplus(m_hn - d_min(proxy_p, C_old_new))
L_old_replay = mean_o softplus(r_old(o) - tau_old_core)
```

其中`L_hn`推动proxy_unknown远离已知类包络，`L_old_replay`保护旧类support和旧类边界样本不被过度拒识。

## 星上推理阶段

每个接收机只保存轻量状态：

| 状态 | 内容 |
|---|---|
| 类原型 | `mu_c`、`n_c`、对角方差、core radius、tail radius |
| support统计 | distance/margin/energy/Mahalanobis分位数 |
| 拒识统计 | local reject rate、uncertain rate、unknown-risk直方图 |
| 漂移统计 | receiver/channel tag、prototype drift、adapter norm |
| 回滚状态 | old support replay acc、版本号、rollback trigger |

单接收机计算：

```text
S_accept(c)=w1 cos(z,mu_c)-w2 Maha(z,c)+w3 KNN_agree(c)+w4 margin(c)
S_reject=w5 energy+w6 proxy_boundary_risk+w7 tail_risk-w8 core_accept
```

判决顺序：

1. 若样本落入旧类核心包络，启用core accept保护，unknown veto降权。
2. 若`S_accept(c*)>=tau_accept(c*)`且`margin>=tau_margin`且`KNN_agree>=tau_knn`，输出已知类。
3. 若`S_reject>=tau_reject`且无core accept保护，输出`reject_unknown`。
4. 其他样本输出`uncertain/defer`，触发协同融合。

## 协同融合

每个接收机上传top-M轻量证据，不上传raw IQ，默认不上传逐样本embedding：

```text
packet_j = {
  top_labels, accept_scores, reject_score, margin,
  conformal_pvalue, support_density, reliability,
  receiver_id, channel_view, timestamp
}
```

多接收机融合：

```text
E_c = sum_j alpha_j S_accept_j(c)
R = sum_j beta_j S_reject_j

accept if max_c E_c >= Tau_c and R < Tau_reject and consensus_count >= m
reject if R >= Tau_reject and no core_accept_override
defer otherwise
```

`alpha_j`由support一致性、历史旧类保持、漂移小决定；`beta_j`由proxy hard-negative验证和近期误拒率决定。协同数量必须支持`1..N_receiver`可选，并记录`avg_participating`、`bytes_per_event`、`latency_ms`。

## 在线更新

优先更新原型，不优先LoRA。只有漂移持续且原型更新不足时，才启用rank 2-4的adapter或LoRA，并限制在投影头/BN affine。

回滚条件：

```text
old_support_acc_after < old_support_acc_before
old_core_reject_rate rises above guard
adapter_norm > norm_budget
unknown veto harms old_acc by more than guard_pp
```

被拒识或不确定样本进入quarantine，只做漂移监控和地面复核，不直接伪标训练。

## 验收指标

所有结论必须基于同一row联合指标：

| 指标 | 要求 |
|---|---|
| `old_acc` | 不低于基线，目标99% |
| `min_old` | 不低于基线，目标95% |
| `seen_new_acc` | 目标97% |
| `min_seen` | 目标93% |
| `unknown_reject` | 目标99% |
| `unknown_FAR` | 目标1%以内 |
| `target_unknown_training_count` | 必须为0 |
| `profile_selection_uses_target_unknown` | 必须为false |
| `resource_pass` | 按bytes/latency/GPU资源记录 |

当前proxy-mined+COTE/C3R远端复验未达标，只能作为hard-negative训练输入和负诊断证据。
