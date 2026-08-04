# D138 D92-Lite-FULL288 Target125实验报告r4

## 状态

- 实验ID：`d138_d92_lite_full288_target125_20260804_r4`
- 登记时间：`2026-08-04`
- 当前状态：`ANALYZED / DIAGNOSTIC_RESULT_ONLY`
- 完成时间：`2026-08-04`（N607本地时间）
- 目标：用已闭合的D92 ground-fit runtime依赖完成冻结FULL288候选的125 outer、375 scene、750 surface及独立truth-side诊断评分。
- r1/r2/r3均在预测前的隔离source依赖阶段停止，未产生预测；r4为最后一次独立one-shot入口，不复用前三次输出。

## 冻结候选、协议与数据

- 候选：`D92-Lite-FULL288/r1`；method lock SHA256=`2bc4384f0a94f3be670a27738ee727db47d937332653bcf3f5ac2a06e02ba728`。
- 表示：sealed runtime完整`registered_feature_288`（z_id160+FFT96+RF32）；K1支持类质心，K5/K10支持-only共享对角OAS float64头；三级支持证据消歧，完全同证据fail-closed。
- 协议：`p2_min_v1`、LEO_weak-only、support/query物理ID互斥、query不fit/update/selection；矩阵125 outer、375 scene、750 before/after surface。
- 复用r6 prepared：plan SHA=`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`，context SHA=`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`。
- 当前sealed SOMP-H输入是诊断-only：`formal_launch_authority=false`、`formal_metric_claim_allowed=false`；最终只报告完整诊断结果。

## 本地版本与闭包

- Git工作树：`E:\type10-7\code\snapshots\d92_lite125_20260804_wt`；运行时提交将在r4登记commit。
- r4仅在r3已验证helper闭包上增加三个纯numpy core：`code/cvsrffi/stage2_d80_ground_commonmode_denoiser.py`、`code/cvsrffi/stage2_d81_ground_nuisance_cauchy_center.py`、`code/cvsrffi/stage2_d92_registration_balanced_covariance.py`；不改candidate、method lock、数据或矩阵。完整本地probe入口导入闭包已通过。
- 本地`ssr-gpu`相关probe/core编译、依赖回归、FULL288回归和diff-check已通过。

## N607发布与成功标准

- 新不可覆盖run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_full288_target125_20260804_r4`；source基底使用r3已验证closure，只覆盖上述两个core文件。
- 远端hash/compile/import和真实checkpoint smoke均通过后，已启动8个固定shard，一卡一shard，并完成merge、validate、build-truth、score。
- 成功条件：8/8 shard、125/125 outer、375/375 scene、750/750 surface、prediction/truth/score完整；不按accuracy、H、BA或中间值停止。

## r4启动证据

- 远端hash/py_compile/完整D92 ground-fit import通过；真实checkpoint smoke通过，receipt SHA=`d0c3bc745cd90c6f568ac56264e8c0fefb93f9210cae31593120ae3fc7229968`，smoke predictions SHA=`dc2840b8ce39c0656215d1beb5be2ba501362070a6947d2c1c15c7b66a04d4da`。
- 2026-08-04启动8个固定分片：shard0 PID1272005/GPU0，shard1 PID1272006/GPU1，shard2 PID1272007/GPU2，shard3 PID1272008/GPU3，shard4 PID1272010/GPU4，shard5 PID1272011/GPU5，shard6 PID1272012/GPU6，shard7 PID1272013/GPU7；CWD均为r4/source，日志在r4/logs/predict_shard_i.out。
- 首波检查：8/8进程存活，GPU进程各约556–558MiB，日志无Traceback/RuntimeError/CUDA-OOM/TIE_UNRESOLVED。

## 完整闭环证据

- 8/8分片正常退出；分片预测共750个，合并manifest包含125/125 outer、375/375 scene和750/750 surface。
- `validate`返回成功：`prediction_closure_verified=true`、`single_candidate_before_after_coverage_verified=true`、`before_after_old_query_matching_verified=true`；访问账本中`clean_source_runtime_access`、`query_fit`、`query_role`、`query_selection`、`query_truth`、`query_update`均为`false`。
- 真值目录：125个outer、375个scene、750个surface；文件SHA256=`0511e203c2addb5a5d6491a62876be1c76762c7a9fb13a72f67c7a9e1620ce0b`，catalog SHA256=`5a07c0e69154d908424c869f3dae43cb5055298f2086c91b81b6473d3dfdf597`。
- 评分完成：scene同一行375/375、scene-arm指标375/375、outer-arm聚合125/125；score manifest文件SHA256=`8f61b6c32b9acad8367fd7020acac32b5e98c8f8e0cb0b3df072b4499e41d76d`，canonical SHA256=`cacb4e5ad06301d0b2e0fd44bc22f20b0359c6b87bbb8c95d4fa90d2a79fb847`。
- 预测合并manifest文件SHA256=`87d8078719d8bedc8146c0753cc1821829db27be063c58e17127faae95232fa0`，canonical SHA256=`e5f0862e2e7c4b3c4d0186d0f9c6a1b9cb9bf0268c8bd5b781da538772fa8d2f`；truth-open事件文件SHA256=`7bc83c7b40513352469a85055286dc49cba7e7e76c9030985d3ec9fc40ae0305`。

## 同一候选、同一run结果

主表使用125个outer聚合行的同一候选结果；准确率为计数微平均，H和遗忘为同一行指标的均值，所有边际范围均附着在该候选/run上。

|候选/run|机制与类别|receiver/TX split、K、seed|before old|after old|seen new|H(old,new)均值|forgetting均值|after-old floor均值|loss/adapter摘要|结论|
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
|`D92-Lite-FULL288/r1` / `r4`|完整`registered_feature_288`；K1支持类质心，K5/K10支持-only共享对角OAS float64头；三级支持证据消歧|`20-1/3-19/7-14/7-7/8-8`；K=1/5/10；seed=`713102–713106`|78.59%|60.35%|41.96%|51.21%（17.00–71.43）|18.24个百分点（3.06–33.06）|32.49%（3.33–71.67）|无训练loss；无query适配|执行、truth-open、同一行score完整；诊断-only，不可晋升|

按K的同一run分组结果如下；K10的`new_count`按冻结矩阵实际取5/10/20。

|K-shot|outer行|new_count|before old微平均|after old微平均|seen new微平均|H均值|forgetting均值|after-old floor均值|post-registration total|
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
|1|25|20|67.14%|43.40%|26.62%|32.81%|23.74个百分点|14.53%|37.37%|
|5|25|20|80.04%|59.83%|43.24%|49.87%|20.21个百分点|29.53%|53.25%|
|10|75|5/10/20|81.92%|66.17%|49.98%|57.79%|15.75个百分点|39.47%|62.19%|

- 全矩阵计数：before-old=`35366/45000`，after-old=`27158/45000`，seen-new=`47202/112500`，post-registration total=`74360/157500`，总评分正确数/查询数=`109726/202500`（54.19%）；prediction query总数为202500。
- 资源封存：prediction artifact约30,179,875 bytes，prediction manifest约30,426,884 bytes，truth catalog约28,902,576 bytes，unique prediction artifact=`750`，registered support slot=`67650`。
- `target_verdict_summary`明确为`coverage_verdict=COMPLETE_125_TRUTH_OPEN_AND_SCORED`，但`target_verdict=NO_TARGET_THRESHOLD_DECLARED`、`system_diagnostic_only=true`、`formal_metric_claim_allowed=false`；本run没有按性能中间值早停或选择。
- 仅运行`M_JOINT`，所以`four_arm_causal_coverage_verified=false`；这不是四臂因果比较，也不构成正式性能晋升证据。逐outer的receiver、seed、scene、每类old计数和receipt保存在本地artifact：`E:\type10-7\automation_reports\CV-SincNet\d138_d92_lite_full288_target125_20260804_r4\artifacts\score_manifest.json`。

## 故障修复闭环

|run|故障位置|处置|结果|
|---|---|---|---|
|`r1`|隔离source缺少`probe_d92_registration_balanced_covariance`|停止并记录缺失依赖|预测前停止，无prediction/truth/score|
|`r2`|隔离source缺少`probe_d81_ground_nuisance_cauchy_center.py`|补齐D81 probe后建立新run|预测前停止，无prediction/truth/score|
|`r3`|隔离source缺少D80/D81纯numpy core|补齐D80、D81、D92三个core后建立新run|预测前停止，无prediction/truth/score|
|`r4`|依赖闭包完整|hash、compile、import、真实checkpoint smoke通过后直接启动125|完整生成prediction、truth和score结果|

## 解释与边界

之前其他对话中的`query prediction`不是“没有结果”，而是已经产生了真实checkpoint上的query推理artifact；其中一部分是smoke或部分技术链，另一部分是完整但诊断/负结果链。它们没有同时满足当前这次的完整125矩阵、独立truth-open、同一行score和本候选的证据闭合，因此不能直接当作正式性能表。

本次r4已经把“query能不能跑”修复并验证为可以：预测、合并、协议验证、真值构建和评分全部完成。但由于method lock仍明确标记`diagnostic_only=true`、没有声明目标阈值且只保留`M_JOINT`，当前结论是“完整执行并得到诊断结果”，不是“方法已晋升”。后续若继续优化，应针对K1低shot退化、after-old下降和seen-new不足设计下一候选，并按同样的完整125矩阵重新验证。
