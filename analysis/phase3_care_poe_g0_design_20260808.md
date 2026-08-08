# Phase3 CARE-PoE G0冻结设计

版本：2026-08-08

目标模式：`ACTIVE`

设计状态：`DESIGN_FROZEN`

候选：`CARE-PoE-G0`

证据等级：`TECHNICAL_SYNTHETIC_NO_PERFORMANCE_RESULT`

## 1.目标与边界

本轮实现Phase3的最小可运行技术闭环：真值无关的本地证据封存、相关性感知协同融合、`N_sat=1..5`、同输入A/B/C/D、独立真值评分，以及`anonymous→外部授权→fresh-K`状态机。它只证明接口、因果干预和失败关闭能够运行，不产生unknown FAR、安全拒绝率或旧类准确率的正式性能主张。

现有R8数据按`role/true_label`构造或排序事件，且没有标签可见前产生的`emission_event_id`、唯一`satellite_reception_id`和物理绑定receipt。因此R8不得进入本G0预测器，也不得被称为same-emission或真实多星协同。正式Phase3性能矩阵继续等待合法事件绑定数据。

## 2.冻结接口

`LocalEvidenceV2`只允许包含：

```text
schema_version,linkage_mode,emission_event_id或proxy_group_id,
satellite_reception_id,node_id,base_manifest_id,bundle_id,
class_handles,p_local,q,correlation_group_id,delay_ms,deadline_ms,
local_decision,local_label,reason_code,sealed_at_ms,evidence_hash
```

预测路径禁止读取`role`、`true_label`、`registration_authorized`和`credential`。`verified_physical`必须提供`emission_event_id`且不得提供`proxy_group_id`；`proxy_unverified`规则相反。`evidence_hash`是去掉自身后的canonical JSON SHA256。字段、哈希、event、reception、bundle、class handle或base manifest冲突时，整事件输出`defer/EVENT_INTEGRITY_FAILURE`。

真值只存在于独立`ScorerTruthSidecar`。预测artifact先封存，评分器后打开sidecar；替换或置换sidecar不得改变预测字节。

## 3.CARE-PoE

deadline前节点的可靠性为：

\[
r_m=\operatorname{clip}(q_m\pi_m\exp(-\lambda\Delta t_m),0,1)
\]

同一`correlation_group_id`只允许冻结node roster中排序最前的有效节点作为确定性代表，组强度取该代表的`r_m`；同组其它节点既不重加权组分布，也不增加组强度。不同组再以均匀类别先验执行加权PoE。任意新增同组相关证据不得改变融合结果。单有效节点走identity分支，逐项返回其`p_local/local_decision/local_label/reason_code`。零有效节点输出`defer/NO_VALID_RECEPTION`。

多节点unknown必须同时满足`P(U)>=tau_reject`、至少2个独立相关组和组质量和门；registered必须满足unknown上界、top margin和组间冲突门；其余均defer。缺失节点自然不进入事件，late节点被排除；事件最终始终只有1个prediction且`shot_count=1`。

## 4.同输入矩阵

冻结node roster为`SAT-01..SAT-05`，预算为`N_sat=1,2,3,4,5`：

| Arm | Phase1证据 | 部署读取 |
|---|---|---|
| A | base bundle | roster首节点 |
| B | new bundle | roster首节点 |
| C | base bundle | CARE-PoE |
| D | new bundle | CARE-PoE |

四臂读取同一`base_manifest_id`、event集合、deadline和roster。base/new按event×node逐项绑定同一`satellite_reception_id`、相关组、delay和deadline，任何不一致都拒绝运行。`N_sat=1`时强制`C=A,D=B`。节点输入顺序和class handle同步置换不得改变语义结果。

## 5.评分语义

registered真值中reject/defer均计识别错误；unknown真值中registered计false accept，unknown计safe reject，defer只计unresolved，不能伪装为safe reject。评分输出同排known accuracy、min-class old accuracy、unknown FAR、safe-reject rate和defer rate；合成fixture结果只作管线自检。

## 6.anonymous、授权与fresh-K

unknown决策只能生成不带语义身份的`anonymous_entity_id`。CARE-PoE不能自我授权。外部credential必须绑定anonymous实体和候选身份，包含至少2个独立来源、无冲突、有效期、置信度、显式授权和签名；缺失、过期或冲突均fail closed。

授权后必须重新采集恰好K个`verified_physical`独立emission event。support event ID和physical sample ID均唯一，且不得与历史unknown事件相交。历史unknown reception或其数学视图永不得转成support。合法fresh-K才生成新的`split_id`和Stage2-C桥接receipt。

## 7.G0发布门

1. focused tests覆盖truth/role禁入、hash tamper、ID冲突、单节点identity、相关复制、missing/late、one-event-one-shot、`N_sat=1..5`、A/B/C/D、独立评分和fresh-K。
2. 独立P0/P1审查为0/0。
3. Git commit、不可覆盖run ID和本地报告完成。
4. N607只运行确定性合成fixture与CPU入口，回收小artifact；不得读取为性能结果。
