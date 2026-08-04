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

## r6运行证据与最终状态

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 未执行：`merge`、`validate`、`build-truth`、`score`。未打开truth，未产生125/375/750完整闭合，不得从任何partial prediction推导性能结论。
- r6是单一`M_JOINT`注册对照：`DA0_REG0=before`、`DA0_REG1=after`；`DA1_REG0`与`DA1_REG1`未纳入冻结矩阵，记为`N/A`。因此本run不报告DA主效应、注册效应或交互效应。

|检查项|已记录证据|结论|
|---|---|---|
|直连预检|`2026-08-04T20:09:54+08:00`通过；项目根可见；8张RTX3090空闲；r6根初始不存在|通过|
|r6落地|仅从r5复制`source`、`input`、`prepared`；未复制`control`或旧分片输出；随后仅覆盖r3 core、qKNN与method lock|不可覆盖性保持|
|远端SHA256|core=`21aa028e7008093bfd5abe33622fa71cff4f6e10ca055de5efe2980453e8e85e`；qKNN=`6a3703d8068dec8f7dbe0c8185ac501bcda5f343642b1814ae7ceedc4469b86a`；lock=`99647a633ff937d22e9ab5928ca2a1785757cd57e1445f3dc8245c534f89222e`；extractor=`56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`；plan=`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`；context=`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`|6项一致|
|远端硬门|`py_compile`通过；Torch`2.1.0+cu121`成功加载extractor；r3 import确认candidate、method lock与160维目标链|通过|
|真实checkpoint smoke|`D92_LITE160_REAL_CHECKPOINT_NO_QUERY_FIT_SMOKE_PASS`；`query_truth/fit/update/selection=false`；smoke receipt SHA=`44350c5b9ec3a3bc4bf0bece048f130913df5710a9744f5257088909a8903c3f`|通过|
|8个固定shard|PID=`1207327,1207328,1207329,1207330,1207331,1207335,1207338,1207339`；启动时均绑定r6 CWD、对应`CUDA_VISIBLE_DEVICES=i`和`--device cuda:0`|已唯一启动|
|首波健康|至少5个不同shard在其失败outer row形成prediction前出现同一确定性指纹；最终控制日志计数为6次：`D92PR160CoreError: TIE_UNRESOLVED: exact tie remains after raw support centroid`|触发系统性技术停止|
|partial保全|保留全部`control/shard_i.out`、smoke收据、partial prediction文件，以及shard2、shard6的分片manifest；未尝试合并partial artifacts|已保留，不可评分|
|清理|停止复核时8个r6 PID均已退出；无r6进程；GPU无计算进程；每次短连接后本机无遗留SSH/SCP或到N607/bridge的TCP22连接|已清理|

### 运行解释与后续边界

r6证明r3的support-only raw signed-PR160类质心二级键仍不能在所有真实并列中给出唯一类别。该结论是执行与可判定性负证据，不是accuracy、`H_old_new`、floor或其他性能比较。按照冻结stop rule，本run不重启、不覆盖、不创建r7；任何后续方法修改必须由主agent重新设计、独立复核、提交并使用新的不可覆盖run ID。
