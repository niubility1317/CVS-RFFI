# P2_CANONICAL_UNION_AUDIT_V1_20260828审计报告

## 结果摘要

Task 9完成N607四份WiSig compact资产的只读canonical union清点与8份deterministic split enumeration。1,268,812条source record合并为1,139,612条canonical record；129,200条重复来源被合并，identity conflict为0。22个new TX候选的完整排序、嵌套`Y_new5/10/20`和17个receiver在`K=1/5/10/20`下的资格已冻结。

本报告不声明`VALIDATED_ONCE`。本轮未运行训练、checkpoint smoke、prediction或scoring；N607现有`CVS-RFFI`环境仅依据用户对Task 9审计和后续Task 10 no-query smoke的明确授权使用，未安装或修改环境。

## 执行身份与交付面

|字段|值|
|---|---|
|Run ID|`P2_CANONICAL_UNION_AUDIT_V1_20260828`|
|审计代码提交|`3fd56b4ecd336b877c0cb74ce733d138f038b207`|
|release SHA256|`545aca71f852dd0ccde836c29dbd6ff359fd3d2066addf01820b423d76a2de0e`|
|remote root|`/home/szu2070436088/2510044040/CV-SincNet/runs/P2_CANONICAL_UNION_AUDIT_V1_20260828`|
|local artifact|`E:\type10-7\local_artifacts\P2_CANONICAL_UNION_AUDIT_V1_20260828`|
|远端解释器|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python`3.10.19`、NumPy`2.2.5`|
|本地解释器|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，Python`3.10.19`、NumPy`2.2.6`|

direct preflight、remote root absent/readback、release landed及唯一local/remote SHA比较均为`VERIFIED`。远端计划内唯一`py_compile`生成4个Python3.10 bytecode文件并完成独立readback。inventory约57秒退出0；split enumeration约247秒退出0，期间无stderr、超时或重复launch。

## Inventory结果

|资产|source record|preferred canonical record|
|---|---:|---:|
|ManySig|288,000|288,000|
|ManyTx|509,128|464,328|
|ManyRx|247,684|196,884|
|SingleDay|224,000|190,400|
|合计|1,268,812|1,139,612|

- `merged_duplicate_count=129,200`
- `conflict_count=0`
- `eligible_record_count=1,139,612`
- 覆盖150个TX、35个RX、4天和11,024个非空`TX×RX×day`cell；cell记录数之和与canonical总数一致。

## Class selection

完整22类排序为：

```text
11-1,7-11,10-11,10-7,11-4,11-7,15-1,16-16,2-19,20-12,20-7,3-13,5-5,6-1,7-10,8-18,8-3,13-3,4-11,3-18,11-17,1-11
```

- `Y_new5=[11-1,7-11,10-11,10-7,11-4]`
- `Y_new10=[11-1,7-11,10-11,10-7,11-4,11-7,15-1,16-16,2-19,20-12]`
- `Y_new20=[11-1,7-11,10-11,10-7,11-4,11-7,15-1,16-16,2-19,20-12,20-7,3-13,5-5,6-1,7-10,8-18,8-3,13-3,4-11,3-18]`

三组按前缀严格嵌套。

## Receiver资格

- MAXQ在`K=1/5/10/20`下均eligible：`1-1,14-7,2-1,20-1,7-14,7-7,8-8,13-13,2-20,8-13,1-20,18-19,19-1,20-19,8-14,8-7`。
- MAXQ在4个K下均ineligible：`13-7`。
- BAL4D在4个K下均eligible：`1-1,14-7,2-1,20-1,7-14,7-7,8-8`。
- BAL4D按定义只消费dense tier；其余10个profile receiver不是BAL4D候选。

## Split计数

8份manifest均包含26个registered TX，并共享`capsule_id=536fb610302e0298fe98b4708d2e6d51eb81aef676126c01d8de6ff1a67985f2`。

|policy|K|receiver|support|query|row|
|---|---:|---:|---:|---:|---:|
|MAXQ_ALL_UNIQUE|1|16|1,248|424,694|425,942|
|MAXQ_ALL_UNIQUE|5|16|6,240|419,702|425,942|
|MAXQ_ALL_UNIQUE|10|16|12,480|413,462|425,942|
|MAXQ_ALL_UNIQUE|20|16|24,960|400,982|425,942|
|BALANCED_4DAY_CORE|1|7|546|28,392|28,938|
|BALANCED_4DAY_CORE|5|7|2,730|28,392|31,122|
|BALANCED_4DAY_CORE|10|7|5,460|28,392|33,852|
|BALANCED_4DAY_CORE|20|7|10,920|28,392|39,312|

## 本地验证

- 精确拉取12个计划文件，总计564,514,594字节；本地文件集合与remote计划清单一致。
- 8份manifest的schema、protocol、profile、count identity、物理ID唯一性、support/query互斥和support公式均通过。
- query row中的`tx_id`字段计数为0。
- `conflicts.csv`的conflict row为0。
- 本地`ssr-gpu`聚焦测试：`tests/test_wisig_canonical_inventory.py tests/test_phase2_canonical_split.py tests/test_phase2_canonical_union_cli.py`以exit 0完成至100%。

## profile表示与限制

v1 profile只定义candidate receiver tier和candidate TX集合，不定义audit-derived selection/eligibility缓存。为避免扩展schema、破坏现有精确profile测试或改变已生成manifest的排序输入，`configs/phase2_canonical_union_profiles_v1.json`保持字节不变；完整排序、`Y_new5/10/20`和eligible-by-policy/K冻结在本报告与[inventory文档](../data/PHASE2_CANONICAL_UNION_INVENTORY_20260828.md)中。这是`NONBLOCKING`表示边界。

本次最高科学证据状态是inventory/split artifacts`VERIFIED`。Git交付以本文件所在提交的远端分支OID独立回读闭合。没有固定received IQ验证、`VALIDATED_ONCE`、模型性能、prediction closure或scorer结果。
