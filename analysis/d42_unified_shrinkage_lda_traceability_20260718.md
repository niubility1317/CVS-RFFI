# D42统一自动收缩LDA设计追踪表

## 范围

- 权威设计：`automation_reports/CV-SincNet/d42_unified_shrinkage_lda_20260718/report.md`。
- 依据：当前active objective、`项目.md`、D37–D41真实support-held artifact及D42只读机制探针。
- 状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D42-01|只读同一固定received IQ；复用D18`p2_min_v1/VALIDATED_ONCE`capsule|Runner/support audit|verified|真实105行、跨候选physical SHA与support/query sealed闭合|
|D42-02|288维表征不变；old-only`log_diag`复用D38式full-batch B20且K变化保持20step|core/tests|verified|同seed同support与D38 before`log_diag`/20step trace逐bit匹配；K闭包通过|
|D42-03|无new参数的old-only B20 helper先物化before；之后才解析new并统一重拟合|core/state|verified|poison-new时序、不同new不改before及snapshot不可变测试通过|
|D42-04|所有类等先验；每类StandardScaler＋Ledoit-Wolf后等先验加权；sklearn1.7.2锁定；标签置换等变|core/tests|verified|C2/C4显式`Σw=μ`、版本漂移fail-close及registry反序测试通过|
|D42-05|precision-weighted target prototype`w_c`两级residual-int8、intercept FP16、无`mu/cov/FP32 w_c`sidecar|state/resource|verified|formal dtype/shape/state语义及非法sidecar反例通过|
|D42-06|Phase1 ground int8只读且entry/exit真实重哈希相同|Runner/ground audit|verified|真实临时NPZ entry/exit及篡改fail-close测试通过|
|D42-07|K1无伪物理样本；覆盖K1/5/10/20和new2/5/10/20|core/tests|verified|参数化闭包、K1/rank0单位协方差和无伪样本测试通过|
|D42-08|query逐样本面对全部注册类；无role/quota/global assignment|score/predict|verified|batch拆分、行顺序、单行与API禁用面测试通过|
|D42-09|int8/FP32 matched outer argmax0变化、margin符号0翻转|core/Runner|rejected|真实105行：before/final argmax变化1/3，margin翻转3，precision gate失败|
|D42-10|保存最终argmax三类混淆与pairwise三类margin，口径分离|Runner/geometry|verified|互斥error partition、独立字段和边界反例通过|
|D42-11|七候选×3场景×5fold=105行且held physical身份matched|candidate lock/Runner|verified|真实receipt105行；7候选各15行，held physical SHA matched|
|D42-12|只有D42 int8可晋级，逐场景/逐类/floor/聚合门fail closed|selector|verified|12个独立性能/协议/资源门反例通过|
|D42-13|full-batch B20固定20/20步、LDA闭式0步、state≤256KB、实际MAC/latency审计|resource|implemented|真实outer20/20、2016参数、8583B、65.44M MAC通过；selector失败故full-K10 latency未执行|
|D42-14|FONR、ridge和单独qKNN路线不进入实现|design|rejected|真实15折分别出现old=0、H=68.97%和H=73.12%，不优于B3|
|D42-15|本地实现、独立审查、Git提交、真实artifact和报告闭环|repo/report|verified|提交55a76bc1；44/68项主验证、独立复审、105行与六项SHA闭合|
|D42-16|只有完整独立确认矩阵全门达标才能完成goal|confirmation report|deferred|D42 development不等于目标完成|

当前计数：`pending=0`、`implemented=1`、`verified=12`、`deferred=1`、`rejected=2`、`blocked=0`。
