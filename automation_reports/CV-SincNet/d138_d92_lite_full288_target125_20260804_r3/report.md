# D138 D92-Lite-FULL288 Target125实验报告r3

## 状态

- 实验ID：`d138_d92_lite_full288_target125_20260804_r3`
- 登记时间：`2026-08-04`
- 当前状态：`LOCAL_VERIFIED / REMOTE_LAUNCH_PENDING / DIAGNOSTIC_RESULT_ONLY_PENDING`
- 目标：一次性补齐r1/r2暴露的D92 ground-fit脚本闭包，完成冻结FULL288候选的125 outer、375 scene、750 surface及独立truth-side诊断评分。
- r1/r2均在smoke/import依赖阶段停止，未产生预测；r3为新的不可覆盖run，不续写前两次。

## 冻结候选与输入

- 候选：`D92-Lite-FULL288/r1`；method lock SHA256=`2bc4384f0a94f3be670a27738ee727db47d937332653bcf3f5ac2a06e02ba728`。
- 表示和决策：sealed runtime完整`registered_feature_288`（z_id160+FFT96+RF32）；K1支持类质心，K5/K10支持-only共享对角OAS float64头；唯一最终分数、support centroid、canonical support fingerprint三级消歧，完全同证据fail-closed。
- 协议：`p2_min_v1`、LEO_weak-only、support/query物理ID互斥、query不fit/update/selection；125 outer、375 scene、750 before/after surface。
- 复用r6 prepared：plan SHA=`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`，context SHA=`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`。
- 当前sealed SOMP-H输入明确为诊断-only：`formal_launch_authority=false`、`formal_metric_claim_allowed=false`；完成后只报告完整诊断结果。

## 本地闭包与验证

- Git工作树：`E:\type10-7\code\snapshots\d92_lite125_20260804_wt`；运行时提交：`b5b1ace6`。
- r3同步闭包：`probe_d92_registration_balanced_covariance.py`及其D81→D80→D66→D62→D61→D46→D45→D44→D43 helper链；均来自本地Git工作树，未改其内容。
- 本地`ssr-gpu`相关probe/core编译、依赖测试、FULL288回归和diff-check通过；远端r1/r2失败均为隔离source缺文件，不是方法失败。

## N607发布与成功标准

- 新不可覆盖run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_full288_target125_20260804_r3`；source基底使用r2已验证候选closure，只覆盖上述完整probe闭包。
- 先做远端hash/compile/import和真实checkpoint smoke；smoke通过后立即启动8个固定shard，一卡一shard，再merge、validate、build-truth、score。
- 只接受8/8 shard、125/125 outer、375/375 scene、750/750 surface和完整truth/score；不按accuracy、H、BA或中间值停止。
- 若在prediction前出现至少两个不同row同一确定性异常，停止并保留全部证据；不使用r1/r2任何输出作为输入。
