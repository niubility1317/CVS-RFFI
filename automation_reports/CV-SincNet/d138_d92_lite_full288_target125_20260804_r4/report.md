# D138 D92-Lite-FULL288 Target125实验报告r4

## 状态

- 实验ID：`d138_d92_lite_full288_target125_20260804_r4`
- 登记时间：`2026-08-04`
- 当前状态：`RUNNING / DIAGNOSTIC_RESULT_ONLY_PENDING`
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
- 先做远端hash/compile/import和真实checkpoint smoke；smoke通过后立即启动8个固定shard，一卡一shard，再merge、validate、build-truth、score。
- 成功条件：8/8 shard、125/125 outer、375/375 scene、750/750 surface、prediction/truth/score完整；不按accuracy、H、BA或中间值停止。

## r4启动证据

- 远端hash/py_compile/完整D92 ground-fit import通过；真实checkpoint smoke通过，receipt SHA=`d0c3bc745cd90c6f568ac56264e8c0fefb93f9210cae31593120ae3fc7229968`，smoke predictions SHA=`dc2840b8ce39c0656215d1beb5be2ba501362070a6947d2c1c15c7b66a04d4da`。
- 2026-08-04启动8个固定分片：shard0 PID1272005/GPU0，shard1 PID1272006/GPU1，shard2 PID1272007/GPU2，shard3 PID1272008/GPU3，shard4 PID1272010/GPU4，shard5 PID1272011/GPU5，shard6 PID1272012/GPU6，shard7 PID1272013/GPU7；CWD均为r4/source，日志在r4/logs/predict_shard_i.out。
- 首波检查：8/8进程存活，GPU进程各约556–558MiB，日志无Traceback/RuntimeError/CUDA-OOM/TIE_UNRESOLVED，尚无分片artifact（仍在运行）。
