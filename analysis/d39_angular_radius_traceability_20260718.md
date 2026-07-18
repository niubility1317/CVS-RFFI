# D39 angular-radius追踪表

## 范围

- 权威设计：`automation_reports/CV-SincNet/d39_angular_radius_20260718/report.md`。
- 直接证据：D38真实90行screen及完整1200条trace。
- 声明：`pending`表示已锁定但未验证；技术实现与性能成功分别记录。

## 条款追踪

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D39-01|只读固定LEO弱support；无clean/source/query/role/quota|Runner/support audit|pending|真实90行后复核五项artifact与协议flags|
|D39-02|严格复用D38-B 20+10训练轨迹，不加bias/temperature或额外step|D39 fit/candidate lock|pending|D38/D39 trace逐row相等|
|D39-03|通过公开cosine/temperature接缝复用D38 scorer，不调用私有decode/transform|D38 seam/D39 core|pending|API与source审查|
|D39-04|`r0`及old radius只由注册前int8旧头和old support计算|D39 fit/state|pending|row-count与状态生命周期测试|
|D39-05|全部类统一`nu=4`、`(K-1)`收缩公式|D39 radius|pending|公式golden与标签置换测试|
|D39-06|K1严格`radius=r0`，无self-OOF或重选|D39 radius/budget|pending|K1边界测试|
|D39-07|new radius只由final int8新头和new support计算并append|D39 fit/state|pending|新类编译顺序与support来源测试|
|D39-08|旧base int8 prefix、旧radius prefix和`r0`注册后逐bit不变|D39 state|pending|append-only测试与geometry audit|
|D39-09|统一score为angular Gaussian，固定`epsilon=0.001`|D39 scorer|pending|golden score与finite测试|
|D39-10|正式state无FP32 target prototype；radius/`r0`为FP16|D39 state/resource|pending|dtype、serialization与resident FP32 count=0|
|D39-11|每个样本独立面对全部注册类，row split/order invariant|D39 score/predict|pending|API与批顺序测试|
|D39-12|保存support-held new-new/new-old及old→new尺度诊断|Runner/geometry audit|pending|pairwise字段和row-count测试|
|D39-13|矩阵为identity、ProtoNet、strong B3、D38-B、D39 int8、D39 FP32，共90行|candidate lock/Runner|pending|cardinality与matched physical ID测试|
|D39-14|只有D39 int8可晋级，严格比较old/new/floor/H/forgetting/intrusion|selector|pending|逐row逐类反例测试|
|D39-15|int8/FP32共享同一FP16 radius，仅替换base weight精度|D39 ablation|pending|state identity与outer argmax测试|
|D39-16|资源计入wrapper metadata、radius状态及`acos/log`标量操作|resource audit|pending|new2/5/10/20与K1/5/10/20边界测试|
|D39-17|本地验证、独立审查、Git提交、真实artifact与报告闭环|repo/report|pending|完成后记录命令、hash和结果|
|D39-18|只有完整独立确认矩阵全门达标才能完成goal|confirmation report|deferred|D39 development不等于目标完成|

## 当前计数与最高风险

`pending=17`、`deferred=1`、`implemented=0`、`verified=0`、`rejected=0`、`blocked=0`。

最高风险是同support拟合得到的radius过度奖励窄类，使旧→新侵入下降但new-new最低类继续失败；第二风险是radius校准恢复旧类后压低seen-new。两者都由outer-held同row旧→新侵入、全部逐类floor和pairwise margin直接否证，不通过额外超参数扫描补救。
