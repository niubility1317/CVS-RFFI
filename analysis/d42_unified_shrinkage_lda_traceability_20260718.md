# D42统一自动收缩LDA设计追踪表

## 范围

- 权威设计：`automation_reports/CV-SincNet/d42_unified_shrinkage_lda_20260718/report.md`。
- 依据：当前active objective、`项目.md`、D37–D41真实support-held artifact及D42只读机制探针。
- 状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D42-01|只读同一固定received IQ；复用D18`p2_min_v1/VALIDATED_ONCE`capsule|Runner/support audit|pending|105行真实Runner及physical SHA待执行|
|D42-02|288维表征不变；old-only`log_diag`复用D38式full-batch B20且K变化保持20step|core/tests|pending|D38 before-state/trace匹配与K闭包待实现|
|D42-03|Stage2-B old-only自动收缩LDA；Stage2-C冻结metric后对old＋new统一重拟合|core/state|pending|生命周期、参数更新norm和trace测试待实现|
|D42-04|所有类等先验、同一Ledoit-Wolf收缩与判别公式；标签置换等变|core/tests|pending|显式反序registry测试待实现|
|D42-05|formal target coefficient两级residual-int8、intercept FP16、无FP32 sidecar|state/resource|pending|dtype/state反例待实现|
|D42-06|Phase1 ground int8只读且entry/exit真实重哈希相同|Runner/ground audit|pending|真实NPZ篡改反例待实现|
|D42-07|K1无伪物理样本；覆盖K1/5/10/20和new2/5/10/20|core/tests|pending|参数化闭包及K1 covariance fail-safe待实现|
|D42-08|query逐样本面对全部注册类；无role/quota/global assignment|score/predict|pending|batch拆分、行顺序和单行一致性待实现|
|D42-09|int8/FP32 matched outer argmax0变化、margin符号0翻转|core/Runner|pending|真实105行量化审计待执行|
|D42-10|保存最终argmax三类混淆与pairwise三类margin，口径分离|Runner/geometry|pending|artifact schema和反例待实现|
|D42-11|七候选×3场景×5fold=105行且held physical身份matched|candidate lock/Runner|pending|真实receipt待执行|
|D42-12|只有D42 int8可晋级，逐场景/逐类/floor/聚合门fail closed|selector|pending|逐门独立反例待实现|
|D42-13|full-batch B20固定20/20步、LDA闭式0步、state≤256KB、实际MAC/latency审计|resource|pending|current/K20规模及selected-only full-K10待验证|
|D42-14|FONR、ridge和单独qKNN路线不进入实现|design|rejected|真实15折分别出现old=0、H=68.97%和H=73.12%，不优于B3|
|D42-15|本地实现、独立审查、Git提交、真实artifact和报告闭环|repo/report|pending|待完成|
|D42-16|只有完整独立确认矩阵全门达标才能完成goal|confirmation report|deferred|D42 development不等于目标完成|

当前计数：`pending=14`、`implemented=0`、`verified=0`、`deferred=1`、`rejected=1`、`blocked=0`。
