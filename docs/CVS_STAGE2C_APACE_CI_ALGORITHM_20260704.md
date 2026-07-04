# CVS Stage2-C APACE-CI算法说明

## 定位

`APACE-CI`全称为`Anchor-Protected Adaptive Conformal Ensemble Collaborative Inference`，即锚点保护的自适应保形协同推理。它是TCSR-CI和OPC-MECR之后的协同推理诊断路线，目标是在旧类保护优先的前提下，用保形p值、support密度、open energy和多接收机冲突证据改善`target_unknown`拒识。

该算法保持`ADV3B02_CORE90_SOFT_E200`特征底座冻结，在轨少样本域适应和新类学习仍使用`qknn8`/support memory。当前实现只做decision-layer评估，不做full-model fine-tuning。`target_unknown`只用于最终评估，不参与阈值、校准、receiver reliability、profile选择或adapter更新。

当前状态：候选诊断算法，尚非部署成功路线。

## 机制

1. 使用`target_old/target_new`的K-shot support构建目标域support memory和target prototype。
2. 使用`source`中的旧类样本构建`source_old_anchor`，作为旧类保护锚点。
3. 对support做leave-one-out非一致性分数，形成support-only保形校准器。
4. 对每个query样本和每个接收机计算轻量证据：
   - `target_proto_score`：到目标support prototype的相似度；
   - `old_anchor_score`：到source old anchor的相似度，只对旧类候选启用；
   - `density_score`：到同接收机support包络的最近相似度；
   - `conformal_p`：support-only非一致性p值；
   - `open_energy`：`1-max(support_score,target_proto_score)`；
   - `margin`：最佳类与次佳类的APACE综合分差；
   - `quality`：接收机本地证据质量。
5. 协同融合时按`M=1..R`选择参与接收机，先过旧类锚点保护门，再过未知多证据门，最后才允许已知类接受，否则输出`__defer__`。

## 决策顺序

1. **旧类锚点保护门**：若旧类候选同时满足`old_anchor_score/conformal_p/density/margin/vote_fraction`门槛，则输出旧类，不允许unknown门直接覆盖。
2. **未知多证据门**：若无旧类保护，且出现低p值、低density、高open energy、低margin或多接收机无共识，则输出`__unknown__`。
3. **已知类接受门**：若候选类的加权证据、投票比例和接收机数达到门槛，输出旧类或seen-new类。
4. **选择性defer**：其余情况输出`__defer__`，避免把不确定样本伪装成成功拒识。

## 资源字段

当前每个接收机上传APACE证据包的代理字段如下：

|字段|用途|
|---|---|
|`top_label/top_label_set`|协同候选类|
|`target_proto_score`|目标support prototype证据|
|`old_anchor_score`|旧类source anchor保护证据|
|`density_score/conformal_p/open_energy/margin`|未知拒识和已知接受证据|
|`quality`|接收机证据排序|
|`bytes/latency_ms`|资源代理指标|

默认`evidence_packet_bytes=160`，`M=1..5`对应事件通信量约`160..800B`。该数值仍为`resource_proxy_pass`，只覆盖本地证据包和融合代理，不代表完整星间/星地链路预算。正式资源声明必须补充协议头、时间戳、接收机ID、top-k label编码、float精度、重传、加密、调度和链路传播/排队时延。

## profile

|profile|用途|
|---|---|
|`apace_primary`|预注册主profile，旧类保护优先，unknown门保守|
|`apace_old_guard`|旧类保持诊断上限，通常不解决unknown|
|`apace_unknown_probe`|冲突拒识诊断，若旧类或seen-new下降则不能部署|

## 协议边界

- 当前`receiver_domain_ranked`只是receiver-domain ensemble diagnostic，不等价于严格同事件多星观测。
- 不得使用`best_posthoc_eval_row`选择部署profile或调阈值。
- `target_unknown_training_count=0`、`threshold_uses_target_unknown=false`、`profile_selection_uses_target_unknown=false`、`reliability_uses_target_unknown=false`必须保持为硬约束。
- 所有结果必须同row报告`old_acc/min_old/seen_new_acc/min_seen/unknown_reject/unknown_FAR/known_defer/unknown_defer/bytes_proxy/latency_proxy/participating/resource_proxy_pass/verdict`。

## 当前预期

若APACE-CI仍表现为“旧类保护时unknown FAR高、冲突拒识时旧类或seen-new下降”，则说明仅靠decision-layer协同仍不足以达成目标。下一步应进入训练/轻量适配层：接收机条件化Mahalanobis、relative density、支持集外壳负样本、旧类锚点回放guardrail、可回滚prototype offset或小adapter，而不是继续增加单一阈值。
