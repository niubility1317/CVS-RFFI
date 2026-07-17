# M6独立query不可达审计

## 结论

`PASS`。本审计只读取M6 runner源码、support-only enrollment manifest元数据、resource audit和query-unreachable proof；没有打开、枚举或解析任何query IQ、query token、truth sidecar、scorer输入或prediction artifact。

M6本身的support gate结论为`NO_GO_SUPPORT_GATE`。因此即使资源上界满足，也不得物化最终patch/head、不得打开query、不得执行预测或评分。

## 实际输入边界

- Runner CLI只有`--enrollment-root`和`--output-dir`，没有query、truth、scorer或宽泛capsule-root入口。
- Enrollment manifest必须精确为`schema=cvs.phase2.somph_predictor_bundle.v1`、`profile=enrollment_only`、`registration_state=after`。
- Manifest成员全集必须恰为6项：sealed runtime、method lock、overlay provenance和3个LEO_weak support文件。
- 任何额外成员、绝对路径、`..`、多层路径或包含`query/truth/scorer/apply_only/before`的成员都会fail closed。
- 实际manifest只含上述6项；审计器没有打开其中任何member。

## 独立负测

`31 passed`，覆盖：

- 合成query、truth、scorer、apply-only成员拒绝；
- 实际M6`_member_map`对额外query/truth/scorer成员拒绝；
- 绝对/遍历/宽泛capsule-root与query CLI拒绝；
- 缺失或置true的query fit/truth/role/count/quota/global-assignment guard拒绝；
- 非精确层名、通配层名、参数量超限拒绝；
- epoch>5、step>50、非SGD、momentum非0、持久化optimizer拒绝；
- FP16 patch+head超过256KB拒绝。

## 真实runner/resource复审

|检查项|结果|
|---|---:|
|原层精确白名单|`model.id_backbone.cls_head.id_proj.0.{weight,bias}`|
|合并后更新原参数|25,760|
|每fold epoch/step|5/5|
|optimizer|SGD，lr=0.05，momentum=0|
|optimizer持久化|否|
|NO-GO预注册rank2 FP16 patch上界|1,600B|
|11×160 int8 head+scale上界|1,782B|
|patch+head上界|3,382B≤262,144B|
|合并后新增MAC|0|
|最终patch/head物化|否|
|deployment export授权|否|

## Proof复审

17个字段逐项确认false：query access/fit/truth open、IQ/token access、truth sidecar、score、prediction、calibration、selection、early stop、rollback、candidate ranking，以及role Oracle、真实批类别数、类别quota和全批分配。`observed_equals_allowlist=true`、`input_manifest_member_allowlist_exact=true`、`input_manifest_extra_member_rejected=true`。

## 证据与哈希

- `audit_result.json`：实际manifest、runner、resource、proof四部分均PASS且`query_content_opened=false`。
- `audit_m6_query_unreachable.py`：`CD63A6288F80F25D0FC14443051739594868FA735A6CE9E6AC85CA67E294555B`
- `test_m6_query_unreachable.py`：`08F4734F936DC0DC0ED634BED37C13329A4978FFE07240B8AD0602F2DB4826F9`
- `audit_result.json`：`315A3CBCA05D00986BD1A2EDA9869C1182FDC8197E3C6E9593D75BE64BFF0776`

本目录及M6实现目录均未stage、未提交Git。本任务没有修改M6实现文件。
