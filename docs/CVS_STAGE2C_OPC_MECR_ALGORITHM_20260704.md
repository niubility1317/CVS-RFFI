# CVS Stage2-C OPC-MECR协同拒识算法说明

## 定位

`OPC-MECR`是面向天基RFFI卫星群协同推理的Stage2-C候选决策层算法，全称为`Old-Protected Class-conditional Multi-Envelope Collaborative Rejection`。它不改动`ADV3B02_CORE90_SOFT_E200`底座模型，不进行全模型微调；在轨少样本域适应和新类注册仍使用`qknn8`证据路径。该算法只增加轻量的类条件包络、旧类保护门、seen-new包络抢占和跨接收机协同一致性判断。

当前状态：候选算法已实现并进入诊断评估，不是部署成功结果。

## 协议边界

- `target_unknown`只用于最终评估，不参与阈值拟合、profile选择、receiver可靠性学习或包络构造。
- 支持集只来自`target_old`和`target_new`的K-shot样本。
- 输出语义限制为旧类标签、seen-new标签、`__unknown__`拒识或defer。
- 协同数量必须按`M=1..R`逐点报告，不能只报告最大参与接收机数。
- 结果必须同row报告`old_acc`、`min_old`、`seen_new_acc`、`min_seen`、`unknown_FAR`、defer、时延和字节数。

## 设计动机

上一轮COTE/C3R/HNFR和几何审计显示，问题不能简单归因为`proxy_unknown`与真实`target_unknown`不匹配。更准确的诊断是：`proxy_unknown`和`target_unknown`都会被`target_old/target_new`support形成的known包络大量吸收；HNFR虽然降低了proxy对已知原型的最大logit，但没有形成可迁移的类条件开放集边界。因此继续单纯推远proxy不足以解决未知拒识。

OPC-MECR把拒识逻辑拆成三个保守步骤：

1. 旧类保护：旧类高质量包络默认保留，只有极强多组件未知证据才能覆盖为defer或拒识。
2. seen-new抢占：当seen-new类条件包络显著强于旧类候选时，允许seen-new覆盖旧类保护，避免新类被旧类吞掉。
3. 无共识拒识：未知类拒识要求跨接收机高风险一致，同时不能存在稳定known包络共识。

## 证据包

每个接收机只需要上传小型证据包：

|字段|用途|
|---|---|
|`top_m`候选标签|类条件协同投票|
|score/margin/pvalue|known包络质量|
|support_count|K-shot支撑强度|
|receiver_class_reliability|接收机-类别可靠性|
|tail/mahalanobis/EVT风险|包络外证据|
|unknown_risk/class_negative_risk|未知风险辅助|
|latency_ms/bytes|资源约束报告|

默认每接收机证据包按128B估算，`M=1..5`时事件通信量约为`128..640B`，仍低于当前`1152B`预算。

## 决策流程

```text
输入：每个事件在M个接收机上的qknn8/PCET证据

1. 对每个接收机提取top_m候选。
2. 对每个候选类别计算class-conditional envelope：
   pvalue + score + margin + support_count + receiver_quality - tail风险。
3. 聚合每个候选类别的receiver_count、vote_fraction、mean_unknown_risk。
4. 若seen-new包络强于旧类包络并满足支撑条件，输出seen-new。
5. 否则若旧类包络满足安全门且无极强未知覆盖，输出旧类。
6. 若无稳定known共识且未知风险跨接收机一致，输出`__unknown__`。
7. 其余样本输出defer，避免为了低FAR牺牲旧类。
```

## 当前profile

|profile|用途|
|---|---|
|`opc_old_guard`|优先保护旧类，普通unknown_risk不直接拒绝旧类；用于检查旧类不下降边界|
|`mecr_balanced`|类条件包络与未知拒识折中|
|`mecr_unknown_probe`|更强未知拒识探针；若旧类下降则只能作为诊断|

## 新增指标

|指标|含义|
|---|---|
|`old_safe_accept_rate`|旧类安全门覆盖率|
|`old_reject_rate`|旧类被拒识或defer比例|
|`known_consensus_rate`|旧类/seen-new是否形成known共识|
|`unknown_no_consensus_rate`|未知样本缺少known共识的比例|
|`bytes_per_event`|参与协同后的平均事件通信量|
|`latency_ms_p95`|事件级p95时延|

## 当前风险

- 如果旧类安全门过宽，未知类会继续被known包络吸收。
- 如果未知拒识门过强，旧类准确率会下降。
- 如果seen-new注册不稳定，未知拒识提升可能来自误拒或误吞seen-new。
- 多接收机协同不保证随`M`单调提升，必须逐点报告。

当前算法仍需N607全量`query_per_class=12`评估。任何结果在未同时满足`old_acc>=0.99`、`min_old>=0.95`、`seen_new_acc>=0.97`、`min_seen>=0.93`、`unknown_reject>=0.99`前，均不得写成部署成功。
