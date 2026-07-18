# D41 BEC设计追踪表

## 范围

- 权威设计：`automation_reports/CV-SincNet/d41_bec_20260718/report.md`。
- 依据：D40真实90行完整artifact、当前`项目.md`和active objective。
- 状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D41-01|只读同一固定received IQ；四个数学view不增加K/physical/LEO overlay|Runner/support audit|verified|Runner extraction与D41 integration验证1 physical observation→4 deterministic views、无额外physical/LEO|
|D41-02|full view严格复用D40 288维几何；三个固定block-erasure view置0后重归一|core/tests|verified|full逐bit与三个block golden通过|
|D41-03|固定`L_BEC=四view macro-CE均值＋三项JS均值`；无noise/anchor/HNBR/bias/radius|core/tests|verified|独立macro-CE/JS golden、极端logit finite及禁用项审计通过|
|D41-04|Stage2-B AdamW20只更新metric＋全部target-old；before int8 artifact不可变|core/state|verified|B20 trace、更新norm和C前后before字节快照通过|
|D41-05|Stage2-C SGD10从B继续，联合更新metric＋全部target-old/new|core/trace|verified|C10单参数组、old/new/logdiag非零更新与30步trace通过|
|D41-06|final target registry全部两级residual-int8；formal无FP32 target sidecar|state/resource|verified|formal dtype、空FP32方向与resident count测试通过|
|D41-07|Phase1 sealed ground int8不进入BEC更新且entry/exit逐bit相同|Runner/ground audit|verified|outer/full-K10真实NPZ逐次重哈希；入口与出口篡改反例fail closed|
|D41-08|K1无伪物理样本；覆盖K1/5/10/20和new2/5/10/20|core/tests|verified|K与new-count参数化资源/步数测试通过|
|D41-09|对注册类标签排列等变；无class ID/old-new query角色分支|core/tests|verified|实际registry反序重标、score列逆置与prediction handle映射通过|
|D41-10|query只用full view，逐样本面对全部注册类argmax|score/predict|verified|row split/order不变与query full-only审计通过|
|D41-11|保存actual old→new/new→old、new-new及两类最低margin/floor|Runner/geometry|verified|真实90行保存D41 142/2/32、B3 33/31/25及完整pairwise margin|
|D41-12|六候选×3场景×5fold=90行且held physical身份matched|candidate lock/Runner|verified|receipt90行；6候选每个15行，跨候选held physical SHA matched|
|D41-13|只有D41 int8可晋级，strict逐row/逐类/聚合门全部fail closed|selector|verified|12个独立门正例与逐门反例通过|
|D41-14|D41 int8/FP32共享参考方向，仅部署精度不同|matched ablation|verified|before/final matched argmax与trace一致性测试通过|
|D41-15|30/30步、C=10、current peak params=3456、state≤256KB，计入四view＋JS MAC|resource|verified|current/max规模、state cap与MAC反例通过|
|D41-16|本地验证、独立审查、Git提交、真实artifact和报告闭环|repo/report|verified|提交91894484/7fb47ad4；187项验证、独立复核、90行artifact和六项SHA闭合|
|D41-17|只有完整独立确认矩阵全门达标才能完成goal|confirmation report|deferred|D41 development不等于目标完成|

当前计数：`pending=0`、`deferred=1`、`implemented=0`、`verified=16`、`rejected=0`、`blocked=0`。

最高风险是BEC仅降低support数学view差异却不改善outer-held物理样本；第二风险是Stage2-C联合metric更新重新引入old遗忘。两者必须由真实before/after old逐类、seen-new逐类、实际双向侵入和同rowH/floor否证，不能用support loss或JS下降替代。
