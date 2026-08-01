# D108-CB-RRC-SMME/r2完整125实验报告

状态：`LOCAL_VERIFIED / RELEASE_READY`

## 实验登记

|字段|值|
|---|---|
|experiment ID|`d108_cbrrc_smme_target125_20260801_r2`|
|日期/operator|2026-08-01；主agent负责集成、数据与结果分析；Terra Max唯一运行子agent负责N607|
|目标|在不改D108方法、矩阵或资产的前提下，修复r1暴露的sklearn运行时硬锁兼容问题并完成完整125|
|假设|CB-RRC可改善D92表示的support/query稳定性，SMME可缓解低support-margin类；完整四臂同row直接验证|
|比较目标|完整125的D62、D92、SVRN-qKNN-BCRR；D91仅列15行development证据|
|claim scope|`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`|
|Git|本地分支`codex/stage2-da25-r1`；只提交，不push、不上传GitHub|

r1在首个pair构建前因N607`scikit-learn=1.7.0`与D42默认硬锁`1.7.2`不符而技术退出，零prediction、未启动shard、未开放truth，已封存为`NO_PERFORMANCE_RESULT`。r2不是性能重试：只复用项目已有D81正式路径的严格兼容集合`('1.7.0','1.7.2')`，在既有全局锁内同步临时注入实际版本、显式D42 config和D92 fit，并在所有正常/异常路径恢复两个全局；其他版本仍在fit前fail-closed。D92公式、288维表示、CB-RRC、SMME、四臂、125矩阵和数据资产完全不变。

## 本地实现与验证

|项目|证据|
|---|---|
|兼容修复|`code/cvsrffi/stage2_d108_d92_core.py`；SHA256=`d25a2f5a75476df944f519603900de9aae3450750989b81af3ff1e8991bb813f`|
|专项测试|`code/tests/test_stage2_d108_d92_core.py`；SHA256=`f94003af8fd7242453eab65fe810bf0209353992e759e18ca42dbb8a775772ce`|
|Git commit|`047223fde7a77c80fd3fab74f3bf459ee9eacbea`|
|验证|core`11 passed`；CB-RRC＋SMME＋core`42 passed`；全部D108联合`56 passed`；`py_compile`与`git diff --check`通过|
|独立复核|`P0=0,P1=0 / GO`；确认严格双版本白名单、显式config、双全局恢复和D92数学不变|
|release archive|`E:\type10-7\code\snapshots\d108_cbrrc_smme_target125_20260801_r2_source_047223fd.tar`；SHA256=`1028850a90c5fbbb91f4c661d09060ba03b1430c0258f1dc515d6017fc4ce54a`|

## 冻结方法、矩阵与资产

候选仍为`D108-CB-RRC-SMME/r1`方法身份；r2只表示新的不可覆盖运行。四臂固定：`M0=D92`、`M_DA=CB-RRC＋D92`、`M_HEAD=D92＋SMME`、`M_JOINT=CB-RRC＋D92＋SMME`。receiver=`20-1,3-19,7-14,7-7,8-8`；seed=`713102..713106`；slice=`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`；scene为三个`leo_*_weak`。闭包为125outer、375scene、1500arm pair、3000prediction surface、500outer-arm聚合行。

D92 matrix SHA256=`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`；checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；D19 ground manifest SHA256=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`；method lock工作树字节SHA256=`7e8b310eeffc5e56aa39d60ef3b66c652207c3d9c1004e04d4499e6073862845`。不使用RDCE，不重验`VALIDATED_ONCE`数据。

## N607预登记

远端run root固定为`/home/szu2070436088/2510044040/CV-SincNet/runs/d108_cbrrc_smme_target125_20260801_r2`，必须首次创建且不可覆盖；源码目录=`runroot/source`，Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD固定为源码目录。prepare写`prepared`；真实row0/clear no-truth smoke写`smoke`；8个GPU分别执行固定`shard-index=0..7`并写`shards/shard_i`；严格合并写`predictions`；封存后truth与score写`truth_catalog.json`和`score`。

发布链路只有：fresh direct preflight→archive/锁/编译核验→prepare→GPU0真实smoke→M0 before/after与历史D92参考按query ID和预测逐项完全一致→GPU0—7固定8shard→严格merge/validate 125/3000→prediction封存后build-truth/score→artifact回收与GPU/SSH清理。smoke parity通过后不得再增加gate。停止仅允许P0协议/安全故障，或至少两个不同outer row在prediction前出现相同确定性异常指纹；不得因任何中途性能值停止、调参、重启或选行。

期望artifact：`prepared/target125_plan.json`、`prepared/target125_context.json`、`smoke/smoke_receipt.json`、`smoke/smoke_predictions.json`、8个`prediction_shard_manifest.json`、`predictions/prediction_manifest.json`、`truth_catalog.json`、`score/score_manifest.json`和日志/PID/exit记录。完成后主表按125个outer-row均值报告before old、after old、before floor、after floor、seen-new、H、forgetting及全量post correct；D62、D92、SVRN保持同口径，D91单列development。
