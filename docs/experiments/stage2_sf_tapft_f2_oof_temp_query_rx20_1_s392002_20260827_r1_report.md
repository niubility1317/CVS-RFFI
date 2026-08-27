# SF-TAPFT F2+OOF温度真实query评估

## 预登记

- run ID：`stage2_sf_tapft_f2_oof_temp_query_rx20_1_s392002_20260827_r1`。
- 当前状态：`ANALYZED`。
- 候选：F2=`S15-SCHED300+OOF全局温度`；固定300步完整cosine、30步warmup、`head+all time norm`、无Adapter，strict bundle中的推理温度为`T=1.07144044514`。
- 任务：不注册新类，只比较同一旧6类query上的`DA0_REG0`与`DA1_REG0`；REG0的新类准确率、old/new harmonic和注册效应均为`N/A`。
- 数据：`p2_min_v1/VALIDATED_ONCE`；receiver=`20-1`；场景=`leo_clear_weak`；K=10×6=60条support；query为rank10–19旧6类各10条，共60条。
- query capsule：`sf-erbt-oldonly-rx20-1-s713101-clear-k10-holdout10-v1`。
- query split：`p2_min_v1-rx20-1-s713101-clear-old6-k10-rank0_9-holdout-rank10_19`。
- 证据边界：该query是真实received-IQ独立holdout，但其truth已在历史16行闭合后揭示；F2在本次预测前未读取query/truth且未由该query选择，结果标记为`REUSED_VALIDATED_HOLDOUT_NOT_NEW_PROSPECTIVE_QUERY`，不得声称全新前瞻确认，不得反馈调参或选择性重跑。
- predictor边界：query NPZ只允许`received_iq/query_ids`；先生成不可覆盖的`DA0_REG0/DA1_REG0`prediction，完整闭合后才由独立scorer打开truth。
- 指标：accuracy、balanced accuracy、class floor、NLL、逐类准确率、`DA1_REG0-DA0_REG0`差值；另与同holdout历史M02结果作上下文对比。
- N607 CWD：`/home/szu2070436088/2510044040/CV-SincNet/<release-checkout>`。
- bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_s15plus5_rx20_1_s392002_20260827_r1/F2/output/sf_tapft_clean_single_bundle.pt`。
- 数据根：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/stage2_sf_d92e0_norf32_oldonly_rx20_1_s713101_20260826_r1`。
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_f2_oof_temp_query_rx20_1_s392002_20260827_r1`。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_f2_oof_temp_query_rx20_1_s392002_20260827_r1`。
- GPU：物理GPU0。
- 预期artifact：`prediction/{da0_reg0.npz,da1_reg0.npz,prediction_receipt.json}`、truth-last`score.json`、GNU time和GPU采样。
- 技术停止：仅限bundle/data绑定错误、query truth/role进入predictor、support/query重叠、错误checkout、输出碰撞、无法加载真实checkpoint、无prediction闭合或确定性系统异常；不得因性能低停止。

## 预注册判断

- 主要问题：F2的快速适配在真实holdout上是否相对自身`DA0_REG0`提升BA、floor并降低NLL。
- 快速档参考门槛：相对历史M02同holdout锚点BA≥86.17%、floor≥60%、NLL≤0.5394；因为本次holdout已复用，该门槛只作同row诊断，不构成新独立晋级。
- OOF温度只改变logit尺度，不改变argmax；因此它可改变NLL但不能单独改变accuracy、BA、floor或逐类准确率。

## 发布、修复与闭合证据

- 预登记提交：`de49aaca36a6851a24dfd627b662d59c039652c5`；最终运行提交：`b5b4683b1c3dbea8d6136a80565e52e4688d397d`，两者均已推送并独立核对GitHub远端OID与本地`HEAD`一致。
- 最终release归档：`release_b5b4683b.zip`；本地与N607 SHA256均为`8cc658155dc3ee0eac18396ba178bfd69a45a907f67b8a8f6233ecddf85e7901`，远端编译通过。
- 聚焦回归：27项通过；三个Python入口编译通过。
- 首次smoke因release环境未设置`PYTHONPATH=code`在导入前失败；随后发现历史F2 bundle的研究期`validation_steps`仍延伸到300，而最终状态只选择150步。两次均发生在query打开前，失败日志和旧release保留，未产生prediction。
- 定点修复：新clean-single bundle不再持久化研究验证日程；strict loader及DA0重建对历史clean-single bundle忽略该非推理字段，但继续严格校验`phase_steps/selected_phase_steps`、state audit、温度、model/head state、Phase1与target绑定。修复按TDD完成，独立定点复审无残留P0/P1。
- 最终真实checkpoint无query smoke：`PASS`；旧6类，`selected_phase_steps=[150,0,0]`，`DA0 scale=8.0`，`DA1 scale=7.466583921`，回算`T=1.071440445141165`，`query_opened=false`、`truth_opened=false`。
- prediction：两份60行状态NPZ与receipt完整，`query_truth_opened=false`、`query_role_opened=false`、`source_opened=false`；stderr只有PyTorch弃用警告，GNU time exit status=0。
- truth-last scorer：仅在prediction闭合后打开truth，`same_row_ids=true`、`truth_join_after_prediction_only=true`，score exit status=0。

## 适应前后真实query结果

|状态|accuracy|BA|class floor|NLL|类0|类1|类2|类3|类4|类5|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`DA0_REG0`|71.6667%|71.6667%|10.0000%|0.936688|70%|100%|70%|10%|100%|80%|
|`DA1_REG0`|85.0000%|85.0000%|60.0000%|0.471599|90%|90%|80%|60%|90%|100%|
|`DA1_REG0-DA0_REG0`|+13.3333pp|+13.3333pp|+50.0000pp|-0.465089|+20pp|-10pp|+10pp|+50pp|-10pp|+20pp|

本query每类10条、共60条，因此一个正确query对应总体BA约1.6667pp、对应类别准确率10pp。F2把正确数从43/60提高到51/60；困难类3从1/10提高到6/10，是floor提升的主要来源。类1和类4各回退1条，说明总体域适应有效但并非逐类无回退。

## 与M02同holdout比较

|候选|DA1 BA|floor|NLL|正确数|类0–5正确数|
|---|---:|---:|---:|---:|---|
|M02长程锚点|86.6667%|60.0000%|0.509438|52/60|9/10/8/6/9/10|
|F2+OOF温度|85.0000%|60.0000%|0.471599|51/60|9/9/8/6/9/10|
|F2-M02|-1.6667pp|0.0000pp|-0.037839|-1|0/-1/0/0/0/0|

F2相对M02只少1个类1正确query，floor完全保持，同时NLL降低0.037839。样本量只有60，当前不能把1条差异解释为稳定分类退化；但按预登记快速档门槛BA≥86.17%，F2仍以约1.17pp未通过，floor与NLL门槛通过，因此不晋级为默认快速档。

## OOF温度的真实query归因

F2 prediction中的DA1 logits已经使用`scale=8/T`。在truth-last之后，将同一logits乘回`T=1.071440445141165`可精确重构未温度缩放状态：

|口径|NLL|argmax|
|---|---:|---|
|重构未温度缩放|0.462502|与F2相同|
|F2+OOF温度|0.471599|与F2相同|
|温度净效应|+0.009097|完全不变|

因此，OOF温度在support OOF上改善NLL约0.003376，但在本query上反而使NLL恶化0.009097。F2较M02的NLL优势来自短程schedule所形成的logit/边界状态，不应归因于OOF温度；去掉温度时相对M02的NLL优势反而扩大到约0.046936。该结果否定“当前OOF全局温度可作为默认query校准”的假设，但不否定F2适配本身。

## 资源与工程数据

|阶段|wall-clock|最大RSS|关键工作量|备注|
|---|---:|---:|---|---|
|F2研究适配|50.13秒|1,825,924KiB|150个full-support选中步、150次backbone训练forward|1584/1584个可训练/实际变化元素；bundle 4,292,702B|
|query prediction|5.21秒|1,150,176KiB|60条×DA0/DA1逐样本全6类推理|进程在首次状态采样前已退出，未取得连续GPU显存峰值|
|truth-last评分|1.49秒|437,328KiB|两状态同ID连接与指标计算|CPU独立scorer，exit status=0|

F2的support OOF结果仍为BA=84.7222%、最低fold BA=72.2222%、NLL=0.542836，低于support晋级门槛；真实query的BA=85.00%、floor=60%、NLL=0.471599说明support OOF绝对值不能直接替代query，但两者都不支持把F2提升为当前默认最优分类工作点。

## 最终结论

1. `DA1_REG0-DA0_REG0`三项主指标方向均显著改善，证明F2快速域适应在该真实holdout上有效。
2. F2保持M02的60%floor并显著改善NLL，但分类正确数少1条，未通过预登记BA门槛；当前结论是`POSITIVE_DA_NOT_PROMOTED`。
3. OOF温度在query上方向反转，应从默认F2配置中移除或等待新的未揭示query重新确认；不得用本次已揭示holdout重新拟合T。
4. 本结果属于`REUSED_VALIDATED_HOLDOUT_NOT_NEW_PROSPECTIVE_QUERY`。它可回答F2在真实query上的适应前后效果并与M02同row比较，但不能代替新的前瞻query、其他receiver/seed或`leo_low_elev_weak/leo_rain_weak`确认。
5. 全程为`REG0`，未注册新类；新类准确率、old/new harmonic与注册效应均为`N/A`。
