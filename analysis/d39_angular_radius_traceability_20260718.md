# D39 angular-radius追踪表

## 范围

- 权威设计：`automation_reports/CV-SincNet/d39_angular_radius_20260718/report.md`。
- 直接证据：D38真实90行screen及完整1200条trace。
- 声明：`pending`表示已锁定但未验证；技术实现与性能成功分别记录。

## 条款追踪

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D39-01|只读固定LEO弱support；无clean/source/query/role/quota|Runner/support audit|verified|真实90行`support_audit.json`确认query/clean/source/role/quota/global assignment均关闭，source closure不变|
|D39-02|严格复用D38-B 20+10训练轨迹，不加bias/temperature或额外step|D39 fit/candidate lock|verified|D38/D39 trace相等；before-Stage2C hook只见20条Stage2-B trace|
|D39-03|通过公开cosine/temperature接缝复用D38 scorer，不调用私有decode/transform|D38 seam/D39 core|verified|公开temperature与before-Stage2C hook API、source审查|
|D39-04|`r0`及old radius只由注册前int8旧头和old support计算|D39 fit/state|verified|hook次数1、trace长度20、new扰动不改变old radius|
|D39-05|全部类统一`nu=4`、`(K-1)`收缩公式|D39 radius|verified|K>1公式golden与标签置换测试|
|D39-06|K1严格`radius=r0`，无self-OOF或重选|D39 radius/budget|verified|K1逐bit退化测试|
|D39-07|new radius只由final int8新头和new support计算并append|D39 fit/state|verified|final int8 source、append顺序与来源测试|
|D39-08|旧base int8 prefix、旧radius prefix和`r0`注册后逐bit不变|D39 state|verified|三类prefix逐bit测试与geometry audit|
|D39-09|统一score为angular Gaussian，固定`epsilon=0.001`|D39 scorer|verified|score golden、finite与全类统一公式测试|
|D39-10|正式state无FP32 target prototype；radius/`r0`为FP16|D39 state/resource|verified|dtype、状态字节、B-arm拒绝、resident FP32 count=0|
|D39-11|每个样本独立面对全部注册类，row split/order invariant|D39 score/predict|verified|split/order/label-permutation测试|
|D39-12|保存support-held new-new/new-old及old→new尺度诊断|Runner/geometry audit|verified|pairwise、真实source token SHA与row-count测试|
|D39-13|矩阵为identity、ProtoNet、strong B3、D38-B、D39 int8、D39 FP32，共90行|candidate lock/Runner|verified|v17真实6×3×5=90行完成；held ranks及跨候选physical SHA一致|
|D39-14|只有D39 int8可晋级，严格比较old/new/floor/H/forgetting/intrusion|selector|verified|10类独立selector反例均回退identity|
|D39-15|int8/FP32共享同一FP16 radius，仅替换base weight精度|D39 ablation|verified|state identity、显式FP32候选预测/radius/r0/trace匹配|
|D39-16|资源计入wrapper metadata、radius状态及`acos/log`标量操作|resource audit|verified|new2/5/10/20、K1/5/10/20与3类full gate反例|
|D39-17|本地验证、独立审查、Git提交、真实artifact与报告闭环|repo/report|verified|实现提交`6098a3f0`；70/70测试；90行完成；5项artifact与receipt SHA一致；完成报告已记录|
|D39-18|只有完整独立确认矩阵全门达标才能完成goal|confirmation report|deferred|D39 development不等于目标完成|

## 当前计数与最高风险

`pending=0`、`deferred=1`、`implemented=0`、`verified=17`、`rejected=0`、`blocked=0`。

本地70/70项联合测试、`py_compile`、diff检查和第二次独立复审通过；真实90行已把两个风险直接否证为失败：D39只将侵入179/180降到174/180，after-old仍仅2.78%，new-new错序仍32/150，`09f8`最低新类准确率30%。另有`leo_low_elev_weak/fold0`注册前prediction SHA与D38-B不同，尽管同row总体/逐类精度及训练trace相同，仍按预注册逐样本同轨门失败。D39技术要求闭合，但路线结论为`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；D39-18继续deferred，goal保持active。
