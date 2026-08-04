# D138 D92-Lite-FULL288 Target125实验报告r1

## 状态

- 实验ID：`d138_d92_lite_full288_target125_20260804_r1`
- 登记时间：`2026-08-04`
- 当前状态：`LOCAL_VERIFIED / REMOTE_LAUNCH_PENDING / NO_PERFORMANCE_RESULT`
- 目标：修复PR160截断导致的query可判定性故障，完成完整125 outer、375 scene、750 before/after surface及truth-side score。
- 与r6关系：这是用户明确要求继续后的新candidate和新不可覆盖run，不续跑、不覆盖r6，也不把r6 partial prediction当输入。

## 冻结候选

- 候选：`D92-Lite-FULL288/r1`；method lock：`configs/d138_d92_lite_full288_r1.json`；SHA256：`2bc4384f0a94f3be670a27738ee727db47d937332653bcf3f5ac2a06e02ba728`。
- 表示：同一sealed D92 TorchScript runtime产生的完整`registered_feature_288`，包含z_id160、FFT96和RF32；不增加backbone forward。
- K1：support-only full288类质心余弦；K5/K10：support-only全类共享对角OAS float64仿射头。
- 并列规则：同一最终float64分数唯一胜者优先；否则full288 support-only类质心余弦；仍并列则canonical sorted full288 support fingerprint；支持证据完全相同才fail-closed。禁止registry顺序、class ID/hash、argmax首项、query truth/role、quota、global reassignment和跨query回退。
- 协议：`p2_min_v1`；仅`leo_*_weak`；support/query物理ID互斥；query batch=1；query不fit、不update、不selection。
- 矩阵：5 receiver×5 seed×5 slice=125 outer；3 scene=375 scene；before/after=750 surface；单一`M_JOINT`，`DA0_REG0=before`、`DA0_REG1=after`。

## 本地版本与验证

- Git工作树：`E:\type10-7\code\snapshots\d92_lite125_20260804_wt`。
- 新增文件：`code/cvsrffi/stage2_d92_full288_target125_core.py`、`code/cvsrffi/stage2_d92_full288_target125.py`、`code/scripts/run_d92_full288_target125.py`、`configs/d138_d92_lite_full288_r1.json`、`tests/test_stage2_d92_full288_target125_core.py`。
- 关键SHA256：core=`4d46ae943b3c60ea8250840af69d01bf3c1e0992747ecdd36c18de977da32313`；adapter=`c1a5efde69b9a8eaee5a71b29adbbfbd9fbe97b6566c6e8655af60e7e2e6f66b`；CLI=`b07026dc0594416c042f4b91db157775a4b203ce636128f493fcd30ad6d1ff71`；test=`281f86ed7cbfaa4f5d849409c43a93465724f99b74ca23834df2e020e2e4979b`。
- 环境：Conda`ssr-gpu`。
- 验证：`py_compile`通过；新full288核心、旧D92/D108/D129/D138及qKNN窄回归全部通过；`git diff --check`通过。

## N607发布登记

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`。
- 不可覆盖run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_full288_target125_20260804_r1`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD：`RUN_ROOT/source`。
- 数据复用：复用r6已验证的`prepared/target125_plan.json`和`prepared/target125_context.json`，plan SHA=`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`，context SHA=`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`。candidate只改变方法表示/头，不改变received-IQ、物理ID、receiver/TX、scenario、K、support/query split或协议。
- 远端source从r6已验证runtime closure复制后，仅覆盖本candidate的full288 core、adapter、CLI和method lock；不复制r6 control、shard或partial prediction。

## 实际启动命令

smoke：

`CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u source/code/scripts/run_d92_full288_target125.py smoke --method-lock source/configs/d138_d92_lite_full288_r1.json --method-lock-sha256 2bc4384f0a94f3be670a27738ee727db47d937332653bcf3f5ac2a06e02ba728 --plan-manifest prepared/target125_plan.json --plan-manifest-sha256 13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348 --context-manifest prepared/target125_context.json --context-manifest-sha256 067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f --output-dir smoke_row000_scene000 --row-index 0 --scene-index 0 --device cuda:0 --feature-batch-size 64`

smoke通过后立即启动8个固定分片：

`CUDA_VISIBLE_DEVICES=i /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u source/code/scripts/run_d92_full288_target125.py predict-shard --method-lock source/configs/d138_d92_lite_full288_r1.json --method-lock-sha256 2bc4384f0a94f3be670a27738ee727db47d937332653bcf3f5ac2a06e02ba728 --plan-manifest prepared/target125_plan.json --plan-manifest-sha256 13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348 --context-manifest prepared/target125_context.json --context-manifest-sha256 067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f --output-dir prediction_shard_i --shard-index i --device cuda:0 --feature-batch-size 64`，i=0..7。

## 健康停止与成功标准

- 只在协议/安全/hash/覆盖错误，或至少两个不同outer row在prediction前出现同一确定性异常时停止；不按accuracy、H、BA、floor或中间结果停止。
- 8个shard完整成功后执行merge、validate、build-truth、score；只有125/375/750闭合且truth-side score通过才产生性能结果。
- 若支持证据完全相同导致full288 fingerprint仍无法唯一判决，保留partial证据并标记`NO_PERFORMANCE_RESULT`，不使用非法类别顺序兜底。

## 完成后分析

按同一candidate/run/receiver/TX/K/seed/scene绑定before old、after old、seen-new、`H_old_new`、forgetting、coverage和资源，不报告跨run孤立极值。若完整结果为负，关闭该full288路线；若完整闭合，更新同一报告的逐candidate结果表。
