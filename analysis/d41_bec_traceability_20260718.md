# D41 BEC设计追踪表

## 范围

- 权威设计：`automation_reports/CV-SincNet/d41_bec_20260718/report.md`。
- 依据：D40真实90行完整artifact、当前`项目.md`和active objective。
- 状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D41-01|只读同一固定received IQ；四个数学view不增加K/physical/LEO overlay|Runner/support audit|pending|实现与真实artifact待完成|
|D41-02|full view严格复用D40 288维几何；三个固定block-erasure view置0后重归一|core/tests|pending|view golden待完成|
|D41-03|固定`L_BEC=四view macro-CE均值＋三项JS均值`；无noise/anchor/HNBR/bias/radius|core/tests|pending|独立loss golden待完成|
|D41-04|Stage2-B AdamW20只更新metric＋全部target-old；before int8 artifact不可变|core/state|pending|参数spy与state测试待完成|
|D41-05|Stage2-C SGD10从B继续，联合更新metric＋全部target-old/new|core/trace|pending|更新范围与30步trace待完成|
|D41-06|final target registry全部两级residual-int8；formal无FP32 target sidecar|state/resource|pending|dtype/resident审计待完成|
|D41-07|Phase1 sealed ground int8不进入BEC更新且entry/exit逐bit相同|Runner/ground audit|pending|真实ground hash待完成|
|D41-08|K1无伪物理样本；覆盖K1/5/10/20和new2/5/10/20|core/tests|pending|参数化测试待完成|
|D41-09|对注册类标签排列等变；无class ID/old-new query角色分支|core/tests|pending|置换测试待完成|
|D41-10|query只用full view，逐样本面对全部注册类argmax|score/predict|pending|row split/order测试待完成|
|D41-11|保存actual old→new/new→old、new-new及两类最低margin/floor|Runner/geometry|pending|pairwise与真实artifact待完成|
|D41-12|六候选×3场景×5fold=90行且held physical身份matched|candidate lock/Runner|pending|integration与真实artifact待完成|
|D41-13|只有D41 int8可晋级，strict逐row/逐类/聚合门全部fail closed|selector|pending|独立反例待完成|
|D41-14|D41 int8/FP32共享参考方向，仅部署精度不同|matched ablation|pending|before/final prediction测试待完成|
|D41-15|30/30步、C=10、current peak params=3456、state≤256KB，计入四view＋JS MAC|resource|pending|规模与资源反例待完成|
|D41-16|本地验证、独立审查、Git提交、真实artifact和报告闭环|repo/report|pending|完成后记录命令/hash/结果|
|D41-17|只有完整独立确认矩阵全门达标才能完成goal|confirmation report|deferred|D41 development不等于目标完成|

当前计数：`pending=16`、`deferred=1`、`implemented=0`、`verified=0`、`rejected=0`、`blocked=0`。

最高风险是BEC仅降低support数学view差异却不改善outer-held物理样本；第二风险是Stage2-C联合metric更新重新引入old遗忘。两者必须由真实before/after old逐类、seen-new逐类、实际双向侵入和同rowH/floor否证，不能用support loss或JS下降替代。
