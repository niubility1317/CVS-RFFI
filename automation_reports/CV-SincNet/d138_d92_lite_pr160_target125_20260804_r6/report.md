# D138 D92-Lite-PR160 Target125实验报告r6

## 状态

- 实验ID：`d138_d92_lite_pr160_target125_20260804_r6`
- 登记时间：`2026-08-04T19:57:32+08:00`
- 当前状态：`LOCAL_VERIFIED / REMOTE_LAUNCH_PENDING / NO_PERFORMANCE_RESULT`
- 操作员：Codex主agent；N607发布、启动和短连接监控由本run唯一runner负责。
- 目标：验证D92-Lite-PR160/r3能否在完整125/375/750矩阵中消除r5真实float64 top tie技术故障，并完成独立truth-side score闭环。

## 冻结候选与比较

- 候选：`D92-Lite-PR160/r3`；method lock：`configs/d138_d92_lite_pr160_r3.json`；SHA256：`99647a633ff937d22e9ab5928ca2a1785757cd57e1445f3dc8245c534f89222e`。
- 协议：`p2_min_v1`；仅`leo_*_weak`；support/query物理ID互斥；查询不访问clean/source、truth、role、fit、update、selection、quota或global reassignment。
- 矩阵：125 outer、375 scene、750 before/after surface、8个固定modulo shard；单一运输臂`M_JOINT`，`DA0_REG0=before`、`DA0_REG1=after`。
- K1保持同一signed-PR160 qKNN主分数；仅当最终float64分数真实并列时，使用同一forward的raw signed-PR160 support-only类质心余弦作二级键；二级仍并列则fail-closed。K5/K10保持已冻结共享all-class diagonal OAS affine。
- 数据复用：r5已验证的received-IQ、物理ID、receiver/TX、scenario、K、support/query split和协议均不变，只复用`prepared`与`input`，不重做数据准备。

## 修复内容

r4在float32 top tie，r5确认最终float64分数本身仍真实并列。r3为support rows保留同一forward的raw signed-PR160，并在真实并列时计算support-only类质心余弦；该二级键不使用query truth/role、跨query状态、类别顺序、hash、quota或global reassignment。所有无法唯一解析的并列继续抛出确定性错误，避免伪造性能结果。

## 本地版本与验证

- Git工作树：`E:\type10-7\code\snapshots\d92_lite125_20260804_wt`；分支：`codex/next-r1-fabr-tsl-20260804`。
- 本轮代码提交：`02334ff48a3519e63e103870f0e60d93af3c8b74`（`fix: add support-only D92 tie resolution`）。r5报告的停止证据另行提交，不修改已有实验产物。
- 修改文件：`code/cvsrffi/stage2_d92_pr160_core.py`、`configs/d138_d92_lite_pr160_r3.json`、`tests/test_stage2_d92_pr160_core.py`、`tests/test_stage2_d92_pr160_target125.py`。
- 关键SHA256：core=`21aa028e7008093bfd5abe33622fa71cff4f6e10ca055de5efe2980453e8e85e`；qKNN=`6a3703d8068dec8f7dbe0c8185ac501bcda5f343642b1814ae7ceedc4469b86a`；method lock=`99647a633ff937d22e9ab5928ca2a1785757cd57e1445f3dc8245c534f89222e`；extractor=`56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`。
- 环境：Conda`ssr-gpu`。已完成`py_compile`和D138/D92/D108/qKNN窄回归，全部通过（68 tests）；`git diff --check`通过。

## N607登记

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`。
- 不可覆盖run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6`。
- source CWD：`RUN_ROOT/source`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；extractor：`RUN_ROOT/input/d92_pr160_extractor_runtime.pt`。
- 复用prepared：plan SHA256=`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`；context SHA256=`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`。
- 同步映射：r5已验证的31-file runtime closure、prepared和extractor复制到r6；本轮覆盖core、qKNN和r3 method lock，覆盖后必须逐文件验SHA，不覆盖r5。

## 实际启动命令

smoke命令固定为：

`CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u source/code/scripts/run_d92_pr160_target125.py smoke --method-lock source/configs/d138_d92_lite_pr160_r3.json --method-lock-sha256 99647a633ff937d22e9ab5928ca2a1785757cd57e1445f3dc8245c534f89222e --plan-manifest prepared/target125_plan.json --plan-manifest-sha256 13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348 --context-manifest prepared/target125_context.json --context-manifest-sha256 067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f --extractor-runtime input/d92_pr160_extractor_runtime.pt --output-dir smoke_row000_scene000 --row-index 0 --scene-index 0 --device cuda:0 --feature-batch-size 64`

smoke通过后立即固定启动8个分片；每个分片使用`CUDA_VISIBLE_DEVICES=i`、`--device cuda:0`和独立的`prediction_shard_i`及`control/shard_i.out`，i为0至7。分片命令除子命令和输出目录外与smoke使用同一method lock、plan/context hash和extractor。

## 健康停止与成功标准

- 只在本run的协议/安全/hash/覆盖错误，或至少两个不同outer row在产生prediction前出现同一确定性异常时停止；不按accuracy、H、BA、floor或中间结果停止。
- 8个分片完整成功、125/375/750闭合、merge/validate/truth-open/score全部通过后，才记录性能结果。
- 任一系统性技术停止均保留partial logs、manifest、PID/GPU/SSH清理证据，并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；不把部分结果送入性能分析。

## 完成后分析

结果表必须按同一candidate/run/receiver/TX/K/seed/scene绑定before old、after old、seen-new、`H_old_new`、forgetting、coverage及最终判定；不得使用跨row孤立极值。若r3仍因support-only二级键真实并列而系统停止，记录为D92 PR160路线的正式负证据，不再无边界增加tie-breaker。
