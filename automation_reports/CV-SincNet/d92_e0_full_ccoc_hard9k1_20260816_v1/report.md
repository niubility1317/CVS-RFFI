# D92 E0 FULL CCOC Hard9+K1本地发布与N607交接报告

## 预发布状态

`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`

本报告对应`d92_e0_full_ccoc_hard9k1_20260816_v1`，操作方为Task3发布代理，记录时间为2026-08-17。当前工作树为`E:/type10-7/code/snapshots/d92_125wt`，科学基线与发布绑定提交为`07e69f1e4360eacec2b972f465c4c19dd1710fd8`。本轮只完成本地发布物、结构验证和交接准备，未SSH、未SCP、未启动N607、未读取或分析任何Hard9运行结果。

## 目标、比较与冻结边界

目标是为已通过Task1/Task2聚焦审查的`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`建立不可覆盖的Hard9+K1交接包。比较对象为同一冻结outer下的E0 FULL历史资源投影；Hard9包含9个performance outer，另含1个K1 liveness outer，3个LEO场景、30个scene-arm行、8个shard。G0 outer`rx_7_7__seed_713106__k_10__new_5`明确排除。

运行时继续遵守`protocol_schema=p2_min_v1`、query逐样本全注册类决策、query truth/fit/update/selection/role/quota/global-reassignment全禁用。注册资源门记录为hard上限`1048576B`（1MiB）和target`524288B`（512KiB）；query MAC、永久state、per-query latency以及其他冻结门不因本交接放宽。`fresh_run_retry=false`，无自动重试、无自动分析器调用。

`candidate_peak_hard_max_bytes=1048576`；`candidate_peak_target_max_bytes=524288`。

## 发布物与路径

|项目|值|
|---|---|
|本地发布根|`automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260816_v1/`|
|运行时归档|`runtime/d92_ccoc_hard9_k1_source_07e69f1e_20260816_v1.tar.gz`|
|归档SHA256|`62a8a1f8536e8df6d6cbbdc9314ae86519817d44b3982a7acaf42be8df949cdb`|
|归档大小|`308426`字节|
|归档成员|`49`（源成员`48`，含联合source manifest）|
|launch.sh|`5802`字节，SHA256=`9de104123c4eb622713940686eeafdc702cecd71ab45913d6a2fbebb633f7ef2`|
|method lock|`configs/stage2_d92_ccoc_hard9_k1_v1.json`|
|外部镜像|`E:/type10-7/automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260816_v1/`|

归档只包含runtime closure、runner、analyzer CLI和method lock config。Python/config成员按Git blob原始字节从`07e69f1e`封存，归档使用LF字节，并补入D80/D81动态依赖及D43/D44/D45/D46/D61/D62/D66/D80 probes；不携带G0 runner/core、data、checkpoint、truth sidecar、tests或docs。归档以整体SHA和安全成员边界核验，不恢复逐成员Git blob gate。

## N607交接接口（仅预登记）

|资源|不可覆盖路径|
|---|---|
|source archive|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_source_07e69f1e_20260816_v1.tar.gz`|
|source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_source_07e69f1e_20260816_v1`|
|driver|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_driver_d92_e0_full_ccoc_hard9k1_20260816_v1.sh`|
|output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_20260816_v1`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_hard9k1_20260816_v1`|
|local retrieval（预登记）|`E:/type10-7/local_artifacts/d92_e0_full_ccoc_hard9k1_20260816_v1`|

唯一detached启动命令：`cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_ccoc_hard9_k1_driver_d92_e0_full_ccoc_hard9k1_20260816_v1.sh >./d92_e0_full_ccoc_hard9k1_20260816_v1.launch.out 2>./d92_e0_full_ccoc_hard9k1_20260816_v1.launch.err </dev/null &`。

launch先检查source/output/log路径不存在，再解包、核验逐成员manifest、编译和import closure；之后按`prepare→truth-free smoke→8 shards`顺序执行。smoke只检查真实checkpoint的truth-free结构收据，只有smoke结构收据闭合后才会创建8个shard进程。runner只产生健康、prediction/COMMIT/fit/resource/score/summary计数和异常fingerprint等结构收据，launch不读取性能指标，也不自动运行`analyze_d92_ccoc_hard9_k1.py`。若共享systemic-stop receipt出现，外部协调只调用冻结`coordinator-stop`并以本run的active-process receipts核验归属；不使用宽泛kill，不重试。

## 本地验证记录

|检查|结果|
|---|---|
|发布测试RED|已观察发布面缺失导致的预期失败|
|archive整体身份、安全边界与运行时闭包|已验证，49成员、48源成员|
|archive/config/launch/report/manifest存在性与外部镜像|本地与外部镜像逐字节一致|
|Task1/Task2聚焦回归|`71 passed`（Task1矩阵6、runner26；Task2 analyzer39）|
|发布测试|`6 passed`，包含解包后runner/analyzer `--help`均rc0|
|`py_compile`、JSON语义读取、`bash -n`|均已验证|
|归档安全、49成员、48个源成员、G0 runner/core排除|均已验证|
|`git diff --check`|已验证|
|N607、SSH/SCP、运行和性能分析|未执行|

## 结论边界与后续

CCOC-16仅更新为release ready no performance：Hard9全门是否通过、是否进入Target125、任何性能verdict均留给主代理在真实N607完整artifact返回后裁决。本地发布完成不代表Hard9运行完成，也不构成性能、推广或论文结论。
