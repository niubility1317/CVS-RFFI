# CVS Stage2-C TCSR-CI算法说明

## 定位

`TCSR-CI`全称为`Target Class Support Reconstruction Collaborative Inference`。它是OPC-MECR之后的feature级诊断算法，目标是直接从`target_old/target_new`的K-shot support构建类条件支持包络，减少对PCET未知风险分数的依赖。

该算法保持`ADV3B02_CORE90_SOFT_E200`特征底座冻结，不进行full-model fine-tuning；在轨少样本域适应和新类学习仍以`qknn8`/support memory为核心。`target_unknown`只用于最终评估，不参与support、阈值、receiver reliability、profile选择或后验调参。

当前状态：候选诊断算法，尚非部署成功路线。

## 机制

1. 对每个目标接收机和每个已知类别，使用K-shot support形成support memory与prototype。
2. 用support leave-one-out相似度构造每类阈值，阈值来源只包含`target_old`和`target_new`support。
3. 对每个query样本计算：
   - `support_score`：到同接收机同类support向量的最大余弦相似度；
   - `prototype_score`：到同类prototype的余弦相似度；
   - `margin`：最佳类与次佳类的综合分数差；
   - `class_threshold`：support-only类包络阈值。
4. 协同融合时按`M=1..R`聚合接收机证据：
   - 已知类接受要求support分数超过类阈值、margin足够、投票占比足够；
   - 未知拒识可由低support分数一致触发；
   - 对`unknown_probe`，若多个接收机高相似但top label不一致，也可作为无稳定known共识拒识证据；
   - 其余输出`__defer__`，避免把不确定样本误报为成功拒识。

## 资源字段

TCSR-CI默认每个接收机上传轻量证据包：

|字段|用途|
|---|---|
|`top_label/top_label_set`|协同投票|
|`support_score/prototype_score/margin`|类条件包络|
|`class_threshold`|support-only阈值|
|`bytes/latency_ms`|资源代理指标|

默认`evidence_packet_bytes=128`，`M=1..5`对应事件通信量约`128..640B`。该数值仍是代理约束，需在找到`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`后再做逐条映射。

## profile

|profile|用途|
|---|---|
|`tcsr_support_tight`|保守support包络，验证support阈值拒识能力|
|`tcsr_old_guard`|旧类/seen-new更宽松接受，检查旧类和新类上限|
|`tcsr_unknown_probe`|更强无共识拒识探针，若旧类下降则只作诊断|

## 边界

- 不能将`best_posthoc_eval_row`用于profile或阈值选择。
- 不得把unknown query用于阈值拟合。
- `receiver_domain_ranked`是receiver-domain ensemble诊断，不等价于严格同事件多星观测。
- 所有结果必须同row报告`old_acc/min_old/seen_new_acc/min_seen/unknown_reject/unknown_FAR/defer/bytes/latency/participating/resource_pass`。

## 当前预期

TCSR-CI若仍出现“保旧类时unknown FAR高、拒识unknown时旧类下降”，则说明问题已经不主要在PCET风险分数，而在特征空间里target_unknown和known support包络本身不可分。下一步应转向特征适配或训练目标，而不是继续堆叠后处理阈值。
