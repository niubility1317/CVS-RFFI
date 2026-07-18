# D38 full-batch B3-geometry residual-int8追踪表

## 范围与权威

- 科学/数据协议：`E:\type10-7\项目.md`、`p2_min_v1`与goal objective。
- 直接失败证据：D37权威报告与本地105行support-only artifacts。
- 活动设计：`automation_reports/CV-SincNet/d38_strong_b3_quantized_20260718/report.md`。
- 声明：`implemented/verified`只表示代码与局部验证闭合，不表示真实support-held性能成立。

## 条款追踪

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D38-01|只使用固定单次LEO弱received-IQ support；不访问clean/source/query|runner/support audit|implemented|CLI/API无query/clean/source/role/quota入口；真实90行后复核artifact|
|D38-02|固定288D表征严格为`normalize([norm(z160);4*norm([FFT96;RF32])])`|D38 core/feature integration|verified|golden feature test验证joint auxiliary normalization和`1/17:16/17`块能量|
|D38-03|Stage2-B为无bias、20步full-batch AdamW，并与exact legacy strong B3区分命名|D38 core|verified|trace 20/20；candidate lock将legacy strong B3保留为独立row|
|D38-04|Stage2-B后先量化旧头；Stage2-C面对实际int8 decode旧头|D38 fit/state|verified|geometry、state和旧prefix测试验证编译顺序与冻结旧头|
|D38-05|Stage2-C loss覆盖全部old+new support，但只更新新权重|D38 core|verified|30步trace、参数面和旧prefix逐bit测试|
|D38-06|A=0步centroid；B=固定10步class-balanced CE＋worst-class＋anchor；只全局选一次|candidate lock/selector|verified|候选锁、A/B trace及15fold聚合selector测试|
|D38-07|K1/5/10/20执行同一K10锁定配置，K1无self-OOF或重选|budget/config|implemented|K1/5/10/20均验证B固定30step；正式非K10矩阵尚未运行|
|D38-08|target-old/new部署预测身份均为两级residual-int8，不存FP32 target prototype|D38 state/scorer|verified|dtype、只读state、FP32 resident count=0与显式ablation测试|
|D38-09|old/new独立编译，注册后旧code/scale/inverse-norm前缀逐bit不变|D38 state|verified|新类单独量化后直接append；全部旧数组逐bit测试|
|D38-10|decode后保持strong B3 weight normalization语义|D38 scorer|implemented|存储FP16 inverse norm并实施matched FP32/int8 outer argmax硬门|
|D38-11|每个样本独立面对全部注册类，无query图、quota或global assignment|score/predict API|verified|row split/order invariance与CLI/API禁用面测试|
|D38-12|新增合法support-held new-new/new-old pairwise诊断，不打开query|training/geometry audit|verified|held样本top competing new/top old及两类margin字段测试|
|D38-13|开发矩阵6候选×3场景×5fold=90行；direct ADV3B02为old-only旁路锚|runner|verified|精确候选锁、cardinality、ProtoNet equivalence和0-support anchor测试|
|D38-14|selector按scene×fold×class严格比较strong B3、identity/ProtoNet和FP32 ablation|selection|verified|逐row逐类matched门、B-int8唯一晋级与FP32 argmax反例测试|
|D38-15|formal资源≤80k params、≤30epoch、≤50step、≤256KB|resource audit|verified|new2/5/10/20、K1/5/10/20及含registry完整状态字节测试|
|D38-16|完整trace、selection/resource/geometry/support audit、receipt、stdout和哈希闭环|runner/report|pending|真实support screen后90/90行全量解析|
|D38-17|本地ssr-gpu窄验证、Git提交后才允许N607 sync|repo/report|pending|本地33/33通过并完成独立审查；Git提交与N607 sync待执行|
|D38-18|只有完整独立确认矩阵全门达标才能完成goal|confirmation report|deferred|D38 development设计不等于目标完成|

## 当前计数与高风险

`verified=11`、`implemented=4`、`pending=2`、`deferred=1`、`rejected=0`、`blocked=0`。

最高风险是20步full-batch Stage2-B无法复制exact legacy strong B3的旧域优势；第二风险是CE10降低new-new错序但通过更激进的新权重扩大旧→新侵入。任一风险在matched outer-held门出现即停止D38，不用query或额外参数扫描补救。
