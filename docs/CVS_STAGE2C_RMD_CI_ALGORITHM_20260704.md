# CVS Stage2-C RMD-CI算法说明

## 定位

`RMD-CI`全称为`Receiver-conditioned Relative Mahalanobis Density Collaborative Inference`。它是APACE-CI之后的feature级协同推理诊断路线，用于检验接收机条件化relative Mahalanobis密度和support外壳负样本是否能改善`target_unknown`拒识。

该算法保持`ADV3B02_CORE90_SOFT_E200`特征底座冻结，在轨少样本域适应和新类学习仍使用`qknn8`/support memory。当前实现只做decision-layer评估，不做full-model fine-tuning。`target_unknown`只用于最终评估，不参与support、relative density拟合、外壳负样本生成、receiver reliability、profile选择或阈值拟合。

当前状态：候选诊断算法，尚非部署成功路线。

## 机制

1. 对每个目标接收机和每个`target_old/target_new`类别，用K-shot support拟合对角协方差类模型。
2. 对同一接收机的全部known support拟合背景密度，计算`relative Mahalanobis = class_md - background_md`。
3. 只使用known support prototype生成support外壳负样本，主要来自类间插值边界，不使用`target_unknown`。
4. 对每个query样本计算：
   - `rmd_score`：relative Mahalanobis密度分数；
   - `density_score`：最近support余弦密度；
   - `shell_risk`：样本接近support外壳负样本的风险；
   - `old_anchor_score`：旧类source anchor保护证据；
   - `margin`和`quality`：协同排序与选择性defer证据。
5. 协同融合时按`M=1..R`聚合接收机证据，先过旧类保护门，再过support shell未知拒识门，最后做已知类接受或`__defer__`。

## 决策边界

|决策|条件|
|---|---|
|旧类接受|`old_anchor_score/rmd_score/density/vote_fraction`达到profile门槛|
|已知类接受|`rmd_score/density/shell_risk/vote_fraction/receiver_count`达到known门槛|
|未知拒识|无旧类保护，且`high_shell_fraction`达到门槛，同时低RMD、低density或低margin至少一项成立|
|defer|证据不足或冲突但未达到拒识门|

## 资源字段

当前每个接收机上传RMD证据包的代理字段如下：

|字段|用途|
|---|---|
|`top_label/top_label_set`|候选类别和old/seen-new集合|
|`rmd_score`|接收机条件化relative Mahalanobis密度|
|`density_score`|最近support密度|
|`shell_risk`|support外壳负样本风险|
|`old_anchor_score`|旧类source anchor保护证据|
|`margin/quality`|协同排序和选择性defer|
|`bytes/latency_ms`|资源代理指标|

默认`evidence_packet_bytes=176`，`M=1..5`对应事件通信量约`176..880B`。该数值仍是`resource_proxy_pass`，不代表完整星间/星地链路预算。

## 协议边界

- 当前`receiver_domain_ranked`只是receiver-domain ensemble diagnostic，不等价于严格同事件多星观测。
- `target_unknown_training_count=0`、`threshold_uses_target_unknown=false`、`profile_selection_uses_target_unknown=false`、`reliability_uses_target_unknown=false`、`shell_negatives_use_target_unknown=false`必须保持为硬约束。
- `target_pass`必须同时满足性能门槛和`resource_proxy_pass=true`。
- `same_max_budget`下`event_count`会随`M`变化，因此`M=1..R`结果不能解释为同一事件分母上的严格收益曲线。

## 当前预期

若RMD-CI仍表现为relative Mahalanobis分数塌缩、unknown shell risk与seen-new/old重叠，则说明当前冻结特征中的known/unknown几何混叠无法通过轻量密度后处理解决。下一步应进入训练或轻量适配：接收机条件化adapter、source/target旧类锚点回放、support外壳负样本训练、prototype offset回滚、BN/FiLM/LoRA小参数更新。
