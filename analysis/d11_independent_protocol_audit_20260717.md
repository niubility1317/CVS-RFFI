# D11独立协议与证据审计

日期：2026-07-17

审计角色：独立只读审计；未修改D11模块、runner、测试或原追踪表

最终结论：**NO-GO，禁止打开query**

## 1.审计范围与硬判据

本审计依据`AGENTS.md`与`项目.md`的Phase2正式边界，逐项检查：

1.仅允许sealed LEO_weak support；一个物理IQ只能对应一个已接收LEO_weak观测。
2.训练、注册与预测必须拒绝普通feature mapping，必须使用逐物理样本、实际received-IQ SHA、runtime、feature code与sealed Phase1 checkpoint绑定的验证artifact。
3.K10统一选择超参数；K1/K5不能读取K10余量。
4.注册评估必须在同一fold内对old/new各held2，只用old K8拟合adapter和old prototype，只用new K8注册new prototype；同时输出old/new/joint/H、逐类、floor与forgetting。
5.旧adapter与旧prototype冻结不等于无遗忘；新增prototype参与全注册类竞争后必须有old held非退化硬门。
6.adapter≤50k参数、适配≤20epoch、state≤256KiB；报告MAC、延迟、显存和相对identity-only single-qKNN Pareto。
7.正式query callback必须物理batch=1，并绑定sealed state、runtime、feature code、Phase1 checkpoint与实际物理IQ SHA；在support promotion失败或query运行时未闭合时禁止打开query。

## 2.版本化产物判定

|产物|状态|结论|
|---|---|---|
|`d11_trainable_lowrank_rank8_v5`|`SUPPORT_ONLY_D11_NOT_SELECTED_NO_QUERY_OPEN`|NO-GO。日志和资源证据完整，但注册L2O仅评估new held，缺少old held竞争与forgetting硬门。|
|`d11_trainable_lowrank_rank8_v6_joint`|`SUPPORT_ONLY_D11_NOT_SELECTED_NO_QUERY_OPEN`|NO-GO。joint old/new L2O已正确落地并实际检出旧类遗忘；三场景promotion门全部失败。|
|`d11_trainable_lowrank_rank8_v7_joint_hardened`|`RESOURCE_BENCHMARK_READONLY_ERROR_NOT_SELECTED`|NO-GO。状态不可变与content SHA已进入当前代码，但本轮在support-row Pareto基准中因readonly数组原地归一化错误停止，没有`COMMIT.json`，不能成为当前代码证据。|

v6的`COMMIT.json`绑定：

- D11 module SHA：`8bbc3dcc5580e0a5cca7edeef646afb61cda03bd0d5ef0dd94646db9df359d2f`
- D11 runner SHA：`76cb0ac747d2fd3cbbaf9cf59d2a15fbc51d089d4e5d425487b2b38e4224106a`
- combined feature code SHA：`a652f47835341df88624d1db037c515ee0ce429384ea8d3d5137236cc936d77e`
- sealed runtime SHA：`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`
- sealed Phase1 checkpoint SHA：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`

v6之后代码继续变化，因此v6只能作为其自身哈希绑定的NO-GO基线，不能替代当前工作树的完整落盘验证。

## 3.v6 joint old/new L2O结果

统一K10候选最终选择`d11_rank8_floor_seek`，但两个候选的三场景joint promotion门均未通过。

|LEO_weak场景|Before-adapter old overall/floor|After old overall/floor|After new overall/floor|H(old,new)|Old forgetting|old逐类非退化|
|---|---:|---:|---:|---:|---:|---|
|`leo_clear_weak`|0.7500/0.1000|0.6667/0.1000|0.5200/0.1000|0.5819|0.0833|False|
|`leo_low_elev_weak`|0.7333/0.3000|0.6667/0.2000|0.4600/0.2000|0.5025|0.0667|False|
|`leo_rain_weak`|0.7833/0.1000|0.6333/0.1000|0.6000/0.4000|0.6141|0.1500|False|

joint fold实现经代码审计确认：

- old/new每类均从K10中held2。
- adapter只使用old K8训练。
- old prototype只使用old K8，old held没有进入prototype。
- new prototype只使用new K8，new held没有进入prototype。
- old held与new held均在全部注册类上逐样本竞争。
- 输出`before_identity_old`、`before_adapter_old`、`after_old`、`after_new`、`joint_accuracy`、`h_old_new`、逐类准确率、floor及两种forgetting。
- promotion接入`old_per_class_non_degraded`，因此“旧参数逐位冻结但新prototype抢占旧类”的假安全状态不能通过。

## 4.负向攻击与修复状态

|攻击|早期结果|当前状态|审计判定|
|---|---|---|---|
|普通`Mapping[str,np.ndarray]`直接训练|接受|当前formal lifecycle API拒绝，要求`ValidatedFeatureArtifact`|当前代码已修复，但须由成功COMMIT重新证明。|
|无old feature fingerprint直接注册|接受|当前注册要求old artifact SHA，并核对runtime/code/checkpoint|当前代码已修复，但调用链仍需完整落盘。|
|同一未绑定query提交两份任意单行feature|两次均接受，score最大差0.220862|当前artifact核对实际received-IQ SHA并逐样本提取；公开raw-feature factory已移除|support侧明显改善；正式query wrapper仍不存在，不能据此开放query。|
|同一actual query IQ通过任意callback生成全`+1`/全`-1`feature|两次均接受，score最大差约2.0|构造函数已改为runner-internal命名，formal runner持有固定extractor|Python下划线不是安全边界；正式query必须由独立sealed runtime wrapper固定extractor并验证code hash。|
|324,832B状态超过256KiB|接受|当前构造、fit和注册均硬拒绝|当前代码已修复。|
|旧state原地修改prototype后预测|预测仍接受且同一query从`a`翻转为`b`|当前state数组复制为readonly并计算`state_content_sha256`；复测写入触发`ValueError: output array is read-only`|当前代码已修复，但v7未成功发布，因此仍待新COMMIT绑定。|
|旧adapter和old prototype逐位冻结后新增prototype抢占旧类|合成反例：`old0`→`new0`，旧状态仍逐位相同|joint old/new L2O已成为promotion硬门|v6实证三场景均检出旧类逐类退化。|

## 5.日志、资源与Pareto

v6证据：

- `training_log.jsonl`共1038行，epoch范围1—12。
- phases包含`selection`、`joint_registration_l2o`、`joint_registration_fold_summary`和`full_fit`。
- `support_audit.json`、`training_log.jsonl`和`report.md`的SHA均与`COMMIT.json`一致。
- trainable parameters：4,616。
- adapter MAC/view：4,616。
- after state：31,136B。
- GPU peak allocated：58,253,312B。
- GPU peak reserved：96,468,992B。
- Python tracemalloc peak：54,684,800B。
- 总wall time：30.4032s。

当前runner增加了单held-out support row的D11对identity-only single-qKNN MAC、延迟与状态对比，但v7在该基准中因readonly数组原地归一化错误停止。该测量即使修复，也只能标为`one_held_out_resource_probe_support_row_no_query_open`，不能冒充正式query延迟。正式query Pareto必须在promotion通过且query runtime完整绑定后另行测量。

## 6.K1/K5与query边界

- v5/v6均声明：`DEFERRED_UNTIL_INDEPENDENT_STRICT_SEALED_PACKAGES;NO_K10_PREFIX_SLICING_OR_SURPLUS_REACHABILITY`。
- 单测能拒绝将K10行直接作为K5输入。
- 尚无独立strict sealed K1/K5产物，因此不能声称K1/K5性能或资源结论。
- 所有v5/v6/v7记录均为`query_opened=false`。
- v6的support promotion明确为False；v7无COMMIT。

因此当前禁止打开query。缺少的不是一个布尔声明，而是：

1.绑定当前module/runner/test哈希的成功support-only COMMIT。
2.通过joint old/new非退化门的候选；当前三场景均有旧类遗忘。
3.正式query runner：只接受sealed state NPZ+metadata及其SHA，验证`state_content_sha256`、runtime、feature code、Phase1 checkpoint和实际received-IQ SHA；物理batch=1；不暴露任意feature mapping或任意extract callback。
4.成功的support-row Pareto审计，以及后续合法formal query per-sample MAC/延迟/显存审计。
5.独立strict sealed K1/K5验证。

## 7.最终判定

**D11维持NO-GO，不得作为单query候选，不得打开query，不得登记为Phase2正式性能证据。**

当前最有价值的结果不是性能提升，而是joint old/new L2O证明：旧adapter与旧prototype冻结仍会因为新增prototype竞争造成6.67—15.00pp旧类遗忘。该证据支持继续优化floor与新旧类分辨，但当前D11不满足support promotion，更未达到目标K10准确率门。

复审触发条件：出现绑定当前代码哈希的成功COMMIT，并同时通过joint old/new门、状态/runtime/code/checkpoint/physical IQ闭环、资源基准和独立K1/K5验证。在此之前保持support-only、no-query。
