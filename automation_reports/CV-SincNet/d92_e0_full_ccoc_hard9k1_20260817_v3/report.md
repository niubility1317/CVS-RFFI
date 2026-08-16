# D92 E0 FULL CCOC Hard9+K1 v3实验发布报告

状态：`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT`

## 1.身份与目标

|字段|值|
|---|---|
|证据标签|`CCOC-16`|
|run ID|`d92_e0_full_ccoc_hard9k1_20260817_v3`|
|时间|2026-08-17|
|操作方|CCOC Hard9 v3发布代理|
|本地工作树|`E:/type10-7/code/snapshots/d92_125wt`|
|运行时代码提交|`2b844de5`|
|目标|以冻结的9个performance outer和1个K1 liveness outer取得CCOC相对同排E0的truth-last证据|
|假设|CCOC同时改善旧类、新类与遗忘指标，注册开销不超过放宽后的hard门，query实时推理面保持与E0一致|
|比较对象|`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`对同排`E0_FULL_ONLY`|

v2在pre-prepare阶段暴露第二个发布工程缺陷：解包归档没有Git对象，但runner把冻结commit/HEAD对象查询当成运行必要条件。v3仅删除这一冗余要求：普通归档执行使用`runtime_source.files[*].sha256`逐实际文件fail-closed校验，并在prepare回执报告`sha256_only`；`scientific_entry_commit`和`git_blob`继续作为追溯元数据。显式注入Git验证器的本地测试仍可检查可选Git绑定。v1/v2本地、远端和外部证据不覆盖、不重试。

## 2.本地变更与发布物

|项目|路径或结果|
|---|---|
|代码/config/tests提交|`2b844de5`；SHA-only archive mode及v3运行身份|
|配置|`configs/stage2_d92_ccoc_hard9_k1_v3.json`；相对v2仅`runtime.output_root`变化|
|运行时归档|`automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260817_v3/runtime/d92_ccoc_hard9_k1_source_2b844de5_20260817_v3.tar.gz`|
|归档闭包|48个实际源文件+1份source manifest；不含数据、checkpoint、truth、测试、文档和G0 runner/core|
|归档SHA256|`05d6199d724d96a596723524f34d4cbb67ff8cfbd6f0ada61deef5b7c0364cb1`|
|归档大小|308577B|
|启动脚本SHA256|`bf469f3f067c6f39b0fc2e50c1daa455bb4d9a4ed5d523ceaf357f8f851d2cb3`|
|外部镜像|`E:/type10-7/automation_reports/CV-SincNet/d92_e0_full_ccoc_hard9k1_20260817_v3/`|

本地验证聚焦于真实阻断：无`.git`的archive-mode prepare、锁定源字节漂移拒绝、matrix/runner受影响测试、精确v3归档导入与runner/analyzer help、归档prepare探针、`py_compile`、JSON、`bash -n`、tar安全和source manifest。未增加数据重验、额外签名或逐成员Git对象门。

## 3.N607预登记

|字段|值|
|---|---|
|远端project|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端archive|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_source_2b844de5_20260817_v3.tar.gz`|
|远端source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_source_2b844de5_20260817_v3`|
|远端driver|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_driver_d92_e0_full_ccoc_hard9k1_20260817_v3.sh`|
|远端output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_hard9_k1_20260817_v3`|
|远端log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_hard9k1_20260817_v3`|
|本地取回根|`E:/type10-7/local_artifacts/d92_e0_full_ccoc_hard9k1_20260817_v3`|
|Python环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|启动CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs`|
|PID|启动后由sole runner记录|
|GPU|smoke=`GPU0`；shard0–7分别映射`GPU0–7`|

唯一detached启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_ccoc_hard9_k1_driver_d92_e0_full_ccoc_hard9k1_20260817_v3.sh >./d92_e0_full_ccoc_hard9k1_20260817_v3.launch.out 2>./d92_e0_full_ccoc_hard9k1_20260817_v3.launch.err </dev/null &
```

启动顺序严格为`prepare→truth-free smoke→8 shards`，不自动运行analyzer，不允许同run重试。v3 archive、source、driver、output、log和本地取回路径必须在创建前全部不存在。

## 4.冻结矩阵、资源与推理门

- 协议为`p2_min_v1`，复用`VALIDATED_ONCE`数据，不重验数据。
- 三scene、9个performance outer+1个K1 liveness，共10 jobs、30 scene receipts、8 shards；G0 outer排除。
- `candidate_peak_hard_max_bytes=1048576`；`candidate_peak_target_max_bytes=524288`；wall hard≤150ms、candidate/E0 wall ratio≤1.50；目标仍为wall P90≤120ms、ratio P90≤1.25。
- 实时推理不放宽：query MAC和永久state bytes逐scene与同排E0精确相同；query truth/fit/update/selection/role/quota/global reassignment全部为false。
- 八项性能方向、幅度、稳定性和唯一裁决保持原冻结定义；K1不进入性能裁决。

预注册技术停止只允许协议/安全错误、覆盖风险、零prediction或至少两个不同outer在prediction前产生同一确定性异常fingerprint；不得读取准确率、H、BA、floor或其他性能值决定停止。

## 5.预期artifact、成功条件与风险

成功闭合需要`matrix_manifest.json`、truth-free smoke receipt、10份job receipt、30份scene closure、8份shard summary及完整日志。sole runner只检查PID/CWD/cmdline、GPU、日志增长、prediction/COMMIT/fit/resource/score/summary计数和异常fingerprint，不读取性能。完成后取回完整source/output/logs和10份manifest绑定truth sidecar，由主代理在本地运行冻结analyzer。

当前仅证明本地发布面可执行，不证明N607落地、Hard9健康或任何性能结论。主要风险是N607源package/E0资源路径的现场存在性和并发GPU资源；这些只在sole runner的短连接preflight中核对，不引入新的发布门。只有`ADVANCE_TO_TARGET125_CANDIDATE`才允许新建Target125发布。
