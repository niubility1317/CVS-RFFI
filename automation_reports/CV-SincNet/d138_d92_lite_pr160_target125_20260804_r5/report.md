# D138 D92-Lite-PR160 Target125实验报告r5

## 状态

- 实验ID：`d138_d92_lite_pr160_target125_20260804_r5`
- 当前状态：`LOCAL_VERIFIED / REMOTE_LAUNCH_PENDING / NO_PERFORMANCE_RESULT`
- 操作员：Codex主agent；r5由主agent直接负责N607落地、启动和短连接监控。
- 目标：修复r4在完整125矩阵中重复出现的float32精确top tie技术故障，并执行完整125/375/750闭环。

## 冻结候选与比较

- 候选：`D92-Lite-PR160/r2`；method lock：`configs/d138_d92_lite_pr160_r2.json`；SHA256：`256aacf7b6f790ce213ac27c1bb496be1a964cbf4f21cdd46309630235fb3ca4`。
- 协议：`p2_min_v1`；LEO weak；support/query物理ID互斥；查询不访问truth、role、fit、update、selection、quota或global reassignment。
- 矩阵：125 outer、375 scene、750 before/after surface、8个固定modulo shard；单一运输臂`M_JOINT`，`DA0_REG0=before`、`DA0_REG1=after`。
- 复用依据：r4的validated prepared plan/context与received-IQ、物理ID、receiver/TX、scenario、K、support/query split和协议均不变；只变方法锁和评分实现，不重做数据准备。

## 修复内容

r4的异常是同一高精度分数在最终float32 cast后形成数值别名。r5保留qKNN和共享仿射的同一最终float64分数；若float32出现top tie且float64唯一最高，则只对该同一分数对应的类提升一个float32 ULP；float64仍并列则继续fail-closed。该修复不使用registry顺序、类别hash、argmax首项、query truth/role、跨query回退或class quota。

## 本地版本与验证

- Git工作树：`E:\type10-7\code\snapshots\d92_lite125_20260804_wt`；分支：`codex/next-r1-fabr-tsl-20260804`。
- Git提交：`eac43a1f8c1f901eca12354d04603776b0849afa`。
- 修改文件：`code/cvsrffi/stage2_d92_pr160_core.py`、`code/cvsrffi/stage2_zid_student_t_qknn.py`、`configs/d138_d92_lite_pr160_r2.json`及对应D138测试。
- 关键SHA256：core=`0d38e046b72fdaf45cd11e6c8fdf1995206f05d5d61a0222384a776a96e0a768`；qKNN=`6a3703d8068dec8f7dbe0c8185ac501bcda5f343642b1814ae7ceedc4469b86a`；extractor=`56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`。
- 环境：Conda`ssr-gpu`；py_compile通过；D138/D92/D108/qKNN窄回归`66 passed`；`git diff --check`通过。

## N607启动登记

- 直连`N607`先行预检；当前直连TCP/SSH不可用时仅使用已验证lab bridge，不改服务器账号、服务或系统配置。
- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`。
- 远端run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r5`，必须首次不存在且不可覆盖。
- source CWD：`RUN_ROOT/source`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；extractor：`RUN_ROOT/input/d92_pr160_extractor_runtime.pt`。
- r4 prepared输入：plan SHA256=`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`；context SHA256=`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`。r5先在新run root复制并验hash，不创建或覆盖旧r4产物。
- 启动命令：smoke使用`python -u code/scripts/run_d92_pr160_target125.py smoke --method-lock source/configs/d138_d92_lite_pr160_r2.json --method-lock-sha256 256aacf7b6f790ce213ac27c1bb496be1a964cbf4f21cdd46309630235fb3ca4 --plan-manifest prepared/target125_plan.json --plan-manifest-sha256 13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348 --context-manifest prepared/target125_context.json --context-manifest-sha256 067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f --extractor-runtime input/d92_pr160_extractor_runtime.pt --output-dir smoke_row000_scene000 --row-index 0 --scene-index 0 --device cuda:0 --feature-batch-size 64`。
- 分片命令：`CUDA_VISIBLE_DEVICES=i python -u code/scripts/run_d92_pr160_target125.py predict-shard --method-lock source/configs/d138_d92_lite_pr160_r2.json --method-lock-sha256 256aacf7b6f790ce213ac27c1bb496be1a964cbf4f21cdd46309630235fb3ca4 --plan-manifest prepared/target125_plan.json --plan-manifest-sha256 13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348 --context-manifest prepared/target125_context.json --context-manifest-sha256 067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f --extractor-runtime input/d92_pr160_extractor_runtime.pt --output-dir prediction_shard_i --shard-index i --device cuda:0 --feature-batch-size 64`，i=0..7；每卡一个本run进程。
- 预期产物：smoke receipt/predictions、8个shard manifest、merged prediction manifest、validate receipt、truth catalog/open receipt、score manifest及control/PID/GPU/SSH清理证据。

## 健康停止与成功标准

- 先验证source逐文件hash、compile、TorchScript load和r5 smoke；smoke通过后立即启动8 shard。
- 若两个不同outer row在prediction前出现同一确定性异常、出现协议/安全/覆盖/hash错误，停止且只保留本run partial artifacts；不因中间accuracy、H、BA或floor停止。
- 只有8 shard完整成功并闭合125/375/750，且独立truth-side score通过，才分析同row before/after性能；否则保持`NO_PERFORMANCE_RESULT`。

## 完成后分析

结果表必须逐candidate/run/receiver/TX/K/seed/scene保持before old、after old、seen-new、`H_old_new`、forgetting、coverage及最终判定同row绑定；不报告跨row孤立极值。
