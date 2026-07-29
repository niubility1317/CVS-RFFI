# CVS-RFFI Phase2 T1 fresh v10 feature运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2_t1_20260730_v10_d1f5e45c`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|只读复用v9闭合的50/50 package与v8 smoke三份scope cache，在全新output根补齐99次feature extraction|
|Git实现|`d1f5e45c72f20e6d81ea5d6fef5e05fcd5f56f0e`|
|状态|`PREREGISTERED / FEATURE_COMPLETION_PENDING / NO_PERFORMANCE_RESULT`|

## fresh修复与路径

|字段|值|
|---|---|
|N607 Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260730_v8_d1f5e45c`|
|package root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v7_d1f5e45c`，只读复用48个package；另从v5复用2个|
|feature output root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v8_d1f5e45c`，启动前必须完全不存在|
|driver log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2_t1_20260730_v10_d1f5e45c`，fresh且`noclobber`|
|controller|`<release>/artifacts/v10_feature_completion/feature_completion_controller.py`|

controller为单文件自包含，不依赖同目录helper或仓库`PYTHONPATH`才能启动；summary schema为`cvs.stage2.v10.feature_completion.v1`。controller明确分离`--package-root`与`--output-root`：96次调用的before/new20 package来自v9 package root，3次调用来自v5 reuse root；所有feature、逐行日志和summary仅写fresh v10 output root。每个feature子进程显式设置`PYTHONPATH=<release>/code`；若继承已有值，使用`os.pathsep`追加且release code优先。

## 并发、计数与停止门

GPU slots固定为`[1,2,2,3,3,4,4,5,5,6,6,7,7]`：GPU0不使用，GPU1仅增加1进程，GPU2-7各最多2进程。每波13行；同一非空确定性异常指纹出现≥2次即停止后续dispatch。

99次成功调用应新增297个scope caches/594个物理文件；加v8复用3个scope caches后为300个total，即A/B/C各100。states正式绑定只采用25个K=10 canonical Stage2-A cache并生成25个sidecar；100个Stage2-B combo跨3个arm服务300行，合计325行。其他75个Stage2-A cache是构建器伴生输出，不计为canonical A identity。

## 完成后回填

回填remote hashes/compile、fresh-root检查、50 package与v8 smoke可读性、runner/worker PID与GPU映射、first-wave计数、manifest/payload/loader、异常指纹和最终scope/file计数。完整结果前不作性能结论。
