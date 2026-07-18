# D38 full-batch B3-geometry residual-int8追踪表

## 范围与权威

- 科学/数据协议：`E:\type10-7\项目.md`、`p2_min_v1`与goal objective。
- 直接失败证据：D37权威报告与本地105行support-only artifacts。
- 活动设计：`automation_reports/CV-SincNet/d38_strong_b3_quantized_20260718/report.md`。
- 声明：`specified`仅表示编码前已锁定，不表示已实现或性能成立。

## 条款追踪

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D38-01|只使用固定单次LEO弱received-IQ support；不访问clean/source/query|runner/support audit|specified|复用D18 validated-once cell；真实运行90行后复核|
|D38-02|固定288D表征严格为`normalize([norm(z160);4*norm([FFT96;RF32])])`|D38 core/feature integration|specified|golden feature test，禁止FFT/RF各自重归一化|
|D38-03|Stage2-B为无bias、20步full-batch AdamW，并与exact legacy strong B3区分命名|D38 core|specified|trace必须20/20；matched strong B3独立row|
|D38-04|Stage2-B后先量化旧头；Stage2-C面对实际int8 decode旧头|D38 fit/state|specified|测试FP32旧头替换会被拒绝；trace记录量化先后|
|D38-05|Stage2-C loss覆盖全部old+new support，但只更新新权重|D38 core|specified|梯度/bitwise旧prefix测试|
|D38-06|A=0步centroid；B=固定10步class-balanced CE＋worst-class＋anchor；只全局选一次|candidate lock/selector|specified|候选锁、trace、15fold聚合选择|
|D38-07|K1/5/10/20执行同一K10锁定配置，K1无self-OOF或重选|budget/config|specified|A=20step、B=30step；K1测试|
|D38-08|target-old/new部署预测身份均为两级residual-int8，不存FP32 target prototype|D38 state/scorer|specified|dtype、serialization、FP32 resident count=0|
|D38-09|old/new独立编译，注册后旧code/scale/inverse-norm前缀逐bit不变|D38 state|specified|append-only测试与geometry audit|
|D38-10|decode后保持strong B3 weight normalization语义|D38 scorer|specified|FP32/int8 matched score/argmax测试|
|D38-11|每个样本独立面对全部注册类，无query图、quota或global assignment|score/predict API|specified|row split/order invariance和AST/API审计|
|D38-12|新增合法support-held new-new/new-old pairwise诊断，不打开query|training/geometry audit|specified|每个held new物理样本保存top competitor和两类margin|
|D38-13|开发矩阵6候选×3场景×5fold=90行；direct ADV3B02为old-only旁路锚|runner|specified|cardinality、同fold physical ID和baseline equivalence audit|
|D38-14|selector按scene×fold×class严格比较strong B3、identity/ProtoNet和FP32 ablation|selection|specified|反例测试禁止均值掩盖floor|
|D38-15|formal资源≤80k params、≤30epoch、≤50step、≤256KB|resource audit|specified|new2/5/10/20与K1/5/10/20边界测试|
|D38-16|完整trace、selection/resource/geometry/support audit、receipt、stdout和哈希闭环|runner/report|pending|真实support screen后90/90行全量解析|
|D38-17|本地ssr-gpu窄验证、Git提交后才允许N607 sync|repo/report|pending|当前未修改代码、未触碰N607|
|D38-18|只有完整独立确认矩阵全门达标才能完成goal|confirmation report|deferred|D38 development设计不等于目标完成|

## 当前计数与高风险

`specified=15`、`pending=2`、`deferred=1`、`implemented=0`、`verified=0`、`rejected=0`。

最高风险是20步full-batch Stage2-B无法复制exact legacy strong B3的旧域优势；第二风险是CE10降低new-new错序但通过更激进的新权重扩大旧→新侵入。任一风险在matched outer-held门出现即停止D38，不用query或额外参数扫描补救。
