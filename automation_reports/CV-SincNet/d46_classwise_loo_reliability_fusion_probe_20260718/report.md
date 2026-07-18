# D46类级head-only LOO可靠度融合探针报告

## 1.身份与目标

- 实验ID：`d46_classwise_loo_reliability_fusion_probe_20260718`
- 操作者：Codex`/root`
- 当前状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景、5个outer physical-rank held折。
- query sealed；不访问confirmation seeds，不生成125结果；当前不访问N607。

D45的全局support-LOO权重修复了量化翻转，却与D44在15/15个outer held预测上完全相同，且仍未修复rain旧类遗忘。D46只改变融合粒度：每个注册类以相同、类标签置换等变的公式获得full/block权重，从而允许不同类别跨过不同决策边界。B20、full与3-block LDA、support RMS、量化器、输入数据、outer folds和比较门保持不变。

## 2.机制与数据协议

对组件`g∈{full,block}`和匿名类`c`，在每个outer fit的合法support内部按physical-row rank做head-only leave-one-out。B20只在outer support训练一次并冻结；inner仅重拟合LDA/RMS。所有组件在RMS、CE和融合前先进入canonical affine gauge：每个特征的系数在类维均值为0，截距在类维均值为0。该规范消除不影响单组件argmax、却会污染类级异权融合的任意类公共仿射项。

逐类inner-held CE记为`CE_g,c`，锁定：

`log_evidence_g,c=-K×CE_g,c`，`w_g,c=softmax_g(log_evidence_g,c)`。

这里`K`必须是当前outer fit/inner分区的实际K，本development fit为K8，而不是capsule名义K10。无temperature、clip、阈值、权重扫描或class ID表。最终完整support fit的类`c`分数为：

`δ_c=w_full,c×δ_full,c/s_full+w_block,c×δ_block,c/s_block`。

公式不读取receiver、TX、old/new角色、handle、场景、outer-held或query。support label仅用于合法的support监督拟合与inner可靠度；outer-held仅在完整state冻结后评价。K1固定1:1等价回退；K2若两组件同CE证据不能闭合到1:1则fail closed。所有query仍独立在全部注册类上argmax，无truth、role Oracle、quota或global reassignment。

## 3.资源口径

LDA fit inventory沿用D45精确四组：before/final各2个main fit，K>1时before/final各`2K`个inner head fit，总数`4K+4`。另计：

- 可靠度打分MAC：K1仅有完整support RMS评分，为`2KD(C_old²+C_all²)`；K>1为`2K(K+1)D(C_old²+C_all²)`；
- 类级仿射融合MAC：`2(D+1)(C_old+C_all)`。

metric B20仍只训练一次、20 epoch/20 optimizer steps；最终只持久化一个融合int8/FP16 query state。host FP64 covariance峰值继续标记未实测，不能由CUDA峰值替代。

## 4.预注册性能门

相对D42 original固定基准必须同时满足：协议/lifecycle/source/ground/state/resource/artifact闭包；inner train-held互斥且support row exact-once；canonical gauge与类标签置换闭环；权重有限、严格为正且逐类和为1；before在首次new support读取前物化且不可变；聚合before-old、after-old、seen-new、H、最低before/after/new和joint均不退化，forgetting不增加，并至少一个final floor严格改善；clear、low-elev、rain各自before/after/new/H/joint不退化且forgetting不增加；before/final int8-FP32 argmax变化与margin翻转均为0；final old→new/new→old/new-new不超过D42的26/10/18。

报告必须保留全部匿名类×场景同row准确率和混淆，不能按类ID调参。目标协议没有要求每个单类准确率相对D42逐项不退化，因此晋级按预锁的通用minimum/lower-tail与逐场景门判断，不事后增加或删除单类门。

此外，D46的15个final held预测必须至少有1个与D45不同；若全部相同，即使汇总指标相同也判为`rejected`，不继续扫描温度或权重。探针强制identity并禁用full-K10；即使所有门通过，也只能进入另行正式候选实现和封闭开发验证，不能直接宣称正式性能或启动125。

## 5.文件、版本与执行计划

- 探针：`code/scripts/probe_d46_classwise_loo_reliability_fusion.py`。
- 单测：`tests/test_probe_d46_classwise_loo_reliability_fusion.py`及D42–D45回归。
- 追溯：`analysis/d46_classwise_loo_reliability_fusion_traceability_20260718.md`。
- 预期输出：`E:\type10-7\automation_reports\CV-SincNet\d46_classwise_loo_reliability_fusion_probe_20260718\classwise_inner_loo_likelihood`。
- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，本地串行，device=`auto`。
- Git：根目录`E:\type10-7`非Git；代码、测试、追溯与正式报告进入`github_publish/CVS-RFFI-repo`，只暂存本轮精确文件；根目录只保留报告镜像。

本地预运行验证：独立代码复核无P0；其发现的K1资源P1已用分段MAC和K1无likelihood指数语义修复，两项P2以逐fold匿名类归属、train补集重算和真实full＋block K2等证据测试加固。D42–D46定向回归最初`82 passed`；attempt0暴露真实support类块顺序并非数值class ID顺序后，verifier改为读取持久化匿名类归属，逐fold验证恰含全部类一次并重算train为held精确补集，回归增至`83 passed`。pytest退出码均为0；退出后本机临时目录`pytest-current`出现既知`WinError 5`清理噪声，不影响测试结论。

## 6.执行与闭包

- 预注册提交：`19c23a8a3dbb833525a63f12d32c215656eec603`。
- verifier修复提交：`66c68b5cf8f6618bb86d032fe2e916753eb1bebc`。
- detached clean worktree：`E:\type10-7\code\snapshots\d46wt`。
- runtime：`E:\type10-7\code\snapshots\d41wt`；Python=`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`；device=`auto`。
- 输入：D18 receiver`20-1`/seed`713101`/K10/new5密封capsule；实际outer fit K8；3场景×5折。
- attempt1输出：`E:\type10-7\automation_reports\CV-SincNet\d46_classwise_loo_reliability_fusion_probe_20260718\classwise_inner_loo_likelihood`。
- 完成：105/105行，elapsed`76.9772s`，query0，formal/performance claim均为false，N607未访问。
- receipt：`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；D46 verifier通过30条int8/FP32 fit row和全部source/lifecycle/resource闭包。

attempt0在提交`19c23a8a`上完成105/105行、elapsed`77.0287s`、query0，但末端D46 verifier因新增的`class_index×K+fold`行序假设拒绝。真实held索引显示每fold确实每匿名类一行且全局exact-once，只是support类块顺序不等于数值class ID顺序；因此这是verifier证据模型错误，不是训练或数据协议失败。attempt0输出未删除或覆盖，原样保存在`classwise_inner_loo_likelihood_attempt0_verifier_partition_assumption`，且没有D46 metadata成功标记。

## 7.同row候选结果

|Candidate|机制/精度|before-old|after-old|seen-new|H|forgetting|joint|min before|min after|min new|old→new/new→old/new-new|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D42-USLDA-INT8|D46类级层次前融合/int8|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|80.00%|53.33%|73.33%|25/8/15|负面，不晋级|
|D42-USLDA-FP32-MATCHED|同一D46解/FP32|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|80.00%|53.33%|73.33%|25/8/15|matched ablation|
|D42-D40-HNBR-INT8-NEGATIVE|old-heavy HNBR/int8|85.56%|85.00%|15.33%|25.16%|0.56pp|0%|66.67%|63.33%|0%|2/0/0|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|new-heavy BEC/int8|86.11%|20.56%|78.67%|31.50%|65.56pp|0%|76.67%|0%|36.67%|142/0/32|旧类崩溃|
|B3_SINGLE_IQ_DIAG_FFTRF|单IQ B3比较器|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|弱比较器|
|D42-PROTOnet-CDA-ZID160|ProtoNet CDA|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|0/0/0|负面|
|Z0_SUPPORT_ONLY|identity/support-only control|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|0/0/0|control|

固定TX切分为6 old＋5 new，receiver`20-1`、seed`713101`、K10 capsule、3场景、5折；每行指标来自同一candidate的15行。无rollback/defer分支；B20为20 epoch/20 optimizer step，closed-form LDA不增加optimizer step。

## 8.基准与场景对照

|版本|before-old|after-old|seen-new|H|forgetting|joint|min after|min new|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|D42 original|90.56%|81.67%|81.33%|80.63%|8.89pp|23.33%|50.00%|70.00%|26/10/18|
|D45 global LOO|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|
|D46 classwise LOO|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|53.33%|73.33%|25/8/15|

本报告`H`统一取matched row内`H_old_new`的算术均值。D45历史报告原先给出的83.10%是pooled-H；本轮已依据完整日志更正为同row均值82.16%，不以跨row汇总后的调和值抬高比较结果。

|场景|before-old|after-old|seen-new|H|forgetting|joint|min after|min new|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|98.33%|90.00%|98.00%|93.57%|8.33pp|40.00%|70.00%|90.00%|4/1/0|
|low-elev|88.33%|78.33%|76.00%|75.98%|10.00pp|20.00%|60.00%|50.00%|8/5/7|
|rain|90.00%|76.67%|80.00%|77.45%|13.33pp|10.00%|30.00%|70.00%|13/2/8|

D46相对D45有2/15个outer prediction SHA变化，共改变3/330个held预测，全部在low-elev；seen-new提高0.67pp、最低new提高3.33pp、new-new混淆减少1，但after-old下降0.56pp、forgetting增加0.56pp、old→new增加1。相对D42，D46的聚合seen-new/H/floor和混淆更好，但aggregate forgetting`10.56pp>8.89pp`；low-elev forgetting`10.00pp>8.33pp`；rain after-old`76.67%<78.33%`且forgetting`13.33pp>10.00pp`，预注册门失败。

## 9.全部匿名类×场景结果

类名按opaque class handle排序后匿名化为O0–O5和N0–N4，仅用于完整报告，不参与方法或调参。

|场景|类|before|after/new|
|---|---|---:|---:|
|clear|O0/O1/O2/O3/O4/O5|100/90/100/100/100/100%|100/90/90/70/90/100%|
|clear|N0/N1/N2/N3/N4|—|100/100/90/100/100%|
|low-elev|O0/O1/O2/O3/O4/O5|80/100/90/80/100/80%|80/80/90/60/70/90%|
|low-elev|N0/N1/N2/N3/N4|—|50/100/50/90/90%|
|rain|O0/O1/O2/O3/O4/O5|90/100/100/60/100/90%|90/100/90/30/60/90%|
|rain|N0/N1/N2/N3/N4|—|70/80/90/80/80%|

## 10.可靠度、量化与资源

- before类级`w_full`总体均值`0.4377`、范围`0.3087–0.7292`，每row类间range均值`0.2235`；clear/low-elev/rain均值为`0.4453/0.4270/0.4407`。
- final类级`w_full`总体均值`0.5096`、范围`0.2803–0.7967`，每row类间range均值`0.2659`；clear/low-elev/rain均值为`0.5109/0.5207/0.4971`。
- before/final/margin量化变化为`0/0/0`，max score error`0.0019154549`；int8与matched FP32同row指标完全一致。
- trainable parameters`2016`；20 epoch/20 optimizer steps；persistent state`8583B`；query MAC`6624`；CUDA peak`22,886,912B`。
- LDA fit`36`次；LDA MAC`1,065,830,400`；metric MAC`4,976,640`；可靠度评分MAC`6,511,104`；类级融合MAC`9,826`；总adaptation MAC`1,077,327,970`。host FP64 covariance peak未实测。
- 300条int8 B20 trace全部finite，epoch/step完整覆盖1–20，trace query rows为0。

## 11.artifact闭包

|Artifact|Bytes|SHA256|
|---|---:|---|
|training_log.jsonl|4,668,878|`deda8cff909ad68113c906eda8802d7083b1af1db84a1b2801e5434bc31c8252`|
|support_audit.json|313,481|`7a83f81fdb89c23b49ec4062a0e7f01f1f58e939fc3e01ae5df138e7568a20`|
|selection.json|2,990|`00b80ba8b80bc4d50c29298942d91bf08739bf1ea7cab7986b14575421a44378`|
|RECEIPT.json|4,845|`06c3943595f5b5565c047d065ff7a0baa80e4053b1d8396aea85b03891584659`|
|D46_PROBE_METADATA.json|2,271|`6f0ec33a616f9a78d5319a06355f2a62d3bb3cd9d9328912ae993fb20d3d3597`|
|geometry_audit.json|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|resource_audit.json|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|

## 12.判定与下一轮

D46为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。它证明类级同式证据能真实改变决策并提高new floor，同时保持量化零翻转，但类级证据方差过大，3个low-elev决策的收益伴随旧类遗忘增加，且rain仍未修复。D46不正式化、不生成125、不访问N607。

下一轮D47保持canonical full/block、B20、RMS、数据和外部门不变，仅对逐fold逐类log-likelihood差做无可调超参、类别置换等变的normal-normal empirical-Bayes层次收缩：不稳定类收回D45全局证据，稳定类保留D46差异。目标是保住D46的seen-new/min-new改善并恢复D45/D42旧类与forgetting；不得使用class ID、old/new角色、场景、receiver、outer-held或query，也不得扫描shrinkage。
