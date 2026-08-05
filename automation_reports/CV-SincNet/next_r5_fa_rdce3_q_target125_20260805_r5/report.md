# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r5）

## 1.身份与目标

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r5`
- 日期：2026-08-05；当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 目标：在同一sealed输入上完成FA-RDCE3+qKNN完整Target125四状态矩阵，验证域适应与新类注册的联合收益。
- 对照：同一行内`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`；不跨行拼接最优值。
- 主agent：`gpt-5.6-sol/high`；科学实现/独立复核：`Terra/max`；唯一N607 runner：`Luna/max`。
- Git worktree：`E:\fa125wt`；branch=`codex/next-r5-fa-q-target125-20260805`；commit=`c6fa99d1d82c6f22e380bec39dac84a571aa5083`。

## 2.r4失败与r5唯一修复

- r4状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。4个不同shard出现相同确定性异常`qKNN exact top tie fails closed`；422个partial prediction保留，但无manifest、truth或score。
- 根因：量化support和对称距离可合法产生完全相同最高logit，score层却在既有确定性裁决前错误中止。
- r5只移除“最高logit必须唯一”的错误要求，并统一采用`highest_logit_then_min_registered_class_index`。该规则仅读取当前行logits与冻结注册类顺序，不读取truth、role、quota、批量类计数或其他query。
- FA、RDCE、qKNN距离、logit、参数、矩阵和数据均不变；不调参，不续跑或覆盖r4。

## 3.本地验证与版本闭包

- `ssr-gpu`六入口`py_compile`：通过。
- 五份聚焦测试：`20 passed`；新增非词典序完全并列回归，确认选择冻结index0。
- `git diff --check`：通过；独立Terra/max复核：`P0=0，P1=0`。
- closure：`E:\type10-7\code\snapshots\next_r5_fa_rdce3_q_target125_20260805_r5_closure_c6fa99d1.tar`
- closure SHA256=`93c93f9d2c94a3a6c7a74053bcec2813e02ce8bcddf5cb545e0b912dbda2986e`；大小=`73134080`bytes。
- method lock SHA256=`7bcc22c86569861f64900a48c57fef46dea4c323c7fc9583923abf8336caa163`。
- builder/core/runtime/CLI SHA256=`9b6b938d87fdfa603f6e0c8be374c77ca7399430ea898cd37c0c37a530880e38`/`09d5a5c63e70575a4c2c2de78911cd83fcd5666385eb133fddef5eaeaffa7b50`/`0441e361fbb744b3d34af35b4cd4ac0b4609edabdbdab7e710e0a9e3425ffc27`/`d0811d699629aa71b75d9d6f111a48f2d2cfc0468788d14e95b3df62bcc0cca5`。

## 4.冻结矩阵、输入和指标

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed={713102,713103,713104,713105,713106}
(K,new)={(10,5),(10,10),(10,20),(5,20),(1,20)}
125 outer×3 leo_*_weak scene×4 state
=375 scene rows/1500 logical surfaces
=1350 unique predictions+150 K1 aliases
```

|输入|远端路径|SHA256|
|---|---|---|
|D106 strict tap|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|strict tap receipt|同目录`d106_ls_strict_tap.receipt.json`|`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|D108 plan|`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/prepared/target125_plan.json`|`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`|
|D108 context|同目录`target125_context.json`|`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`|

数据capsule/split未变化，按`p2_min_v1`复用`VALIDATED_ONCE`，不重新验数。REG0的seen-new/H=`N/A`。最终报告同一行计算DA前注册效应、DA后注册效应、无DA注册效应、有DA注册效应及difference-in-differences。

## 5.N607冻结执行与健康规则

- RUN_ROOT：`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5`，落地前必须`ABSENT`；r1至r4禁止触碰。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`RUN_ROOT/source`；GPU0至7各一个固定shard。
- 顺序：direct preflight→输入/GPU/RUN_ROOT核验→SCP closure→archive/member/source SHA→六入口py_compile→build asset→prepare→truth-free smoke→8 shards→merge→truth-open→score→只读取回。
- asset命令：`python -u code/scripts/build_next_r5_fa_target125_asset.py --strict-tap <上表strict tap> --strict-tap-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --method-lock $RUN_ROOT/source/configs/next_r5_fa_rdce3_q_target125_20260805.json --method-lock-sha256 7bcc22c86569861f64900a48c57fef46dea4c323c7fc9583923abf8336caa163 --output-dir $RUN_ROOT/input/fa_asset`。
- prepare/smoke/predict-shard/merge/truth-open/score均固定使用`python -u code/scripts/run_next_r5_fa_target125.py`对应子命令；asset/plan/context实产SHA须逐阶段固定并追加本报告，不能修改候选或矩阵。
- 成功闭合：125/125 outer、375/375 scene、1500/1500 logical、1350 unique、150 aliases、8/8 shard、完整manifest、truth-open和score。
- 立即停止仅限P0协议/安全/hash/覆盖故障，或至少两个不同row在prediction前出现相同确定性异常指纹。不得因中途accuracy、H、BA或floor停止。
- fresh retry authority：无；失败保留artifact并另建run ID。

## 6.结果待填

|状态|outer|scene|logical|unique|truth|score|结论|
|---|---:|---:|---:|---:|---|---|---|
|当前（停止）|未闭合|未闭合|未闭合|仅678个partial JSON|未打开|未产生|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|
## 7.落地、asset、prepare与smoke证据

- r5 closure已SCP至`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source_closure.tar`；archive SHA=`93c93f9d2c94a3a6c7a74053bcec2813e02ce8bcddf5cb545e0b912dbda2986e`、大小=`73134080`；member/source SHA与本地一致；远端六入口`py_compile`输出`R5_PY_COMPILE_OK`；每步SSH均为`SSH_CLEAN`。
- FA asset构建exit=`0`；semantic SHA=`8ada72d95a36ad435451acb21123f65c428cb05ae77b6ae9e82da3ca93589c85`；wire SHA=`33633242875ccf1556648ff6764a1de795de9a45938f965d4961e5a8ea31020c`；manifest SHA=`57f73851936ce54fbdb7dbfd9d63c8f4ca2fa4b5b1f109c3a4892572dbd97722`；source indices=`[0,1,2,3,4,5]`、old-class root=`f23394f508cae38116e7541b7954e647d4d47aa8f6abc69620cfa2813b873212`、target support/query rows=`0/0`。
- asset完整命令：`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source; /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/build_next_r5_fa_target125_asset.py --strict-tap /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz --strict-tap-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --method-lock /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source/configs/next_r5_fa_rdce3_q_target125_20260805.json --method-lock-sha256 7bcc22c86569861f64900a48c57fef46dea4c323c7fc9583923abf8336caa163 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/input/fa_asset`。
- prepare exit=`0`；plan SHA=`393263d239fc55593d384cade173d4656cb2cd9b437d666ead2b60324895b049`；context SHA=`6b170e0734e2a3000bd3b81d78747140925874cc0ee5a3d2c47422629ec0ccb7`；receipt file SHA=`759343d1b6c75476a82adac5747f1e54ade6da69d418169935b48dfad1fbc3ef`；canonical receipt SHA=`cb90650e0a7b4e2a5c9bc6d392f2104774d37c4105208c81392636bc93f33f70`；counts=`125/375/1500/1350/150`；五项query访问均`false`。
- prepare完整命令：`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source; /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py prepare --d108-plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/prepared/target125_plan.json --d108-plan-manifest-sha256 13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348 --d108-context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/prepared/target125_context.json --d108-context-manifest-sha256 067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f --fa-asset /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/input/fa_asset/fa_rdce3_target125.wire --fa-asset-sha256 33633242875ccf1556648ff6764a1de795de9a45938f965d4961e5a8ea31020c --method-lock /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source/configs/next_r5_fa_rdce3_q_target125_20260805.json --method-lock-sha256 7bcc22c86569861f64900a48c57fef46dea4c323c7fc9583923abf8336caa163 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared`。
- smoke exit=`0`；status=`REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS`；receipt file SHA=`546078963fafe2096403cceab66eaa8a37637140f47b7c5cf89569c49973ad51`；canonical SHA=`93241994ba9a2fe4b6445b0360729973564396f96bfd3746cb2765459a674aea`；四状态计数=`DA0_REG0:120, DA1_REG0:120, DA0_REG1:220, DA1_REG1:220`；query truth/selection/update/fit均`false`；truth/score目录尚未创建。
- smoke完整命令：`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source; CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py smoke --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_plan.json --plan-manifest-sha256 393263d239fc55593d384cade173d4656cb2cd9b437d666ead2b60324895b049 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_context.json --context-manifest-sha256 6b170e0734e2a3000bd3b81d78747140925874cc0ee5a3d2c47422629ec0ccb7 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/smoke --row-index 0 --scene-index 0 --device cuda:0`。

## 8.8-shard冻结命令与启动记录（已停止）

- 每条命令CWD=`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source`、`PYTHONUNBUFFERED=1`、固定GPU0–7、`--device cuda:0`；输出目录和日志叶子均此前不存在；启动后记录PID、CWD、cmdline、GPU和日志增长。
`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source; nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_plan.json --plan-manifest-sha256 393263d239fc55593d384cade173d4656cb2cd9b437d666ead2b60324895b049 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_context.json --context-manifest-sha256 6b170e0734e2a3000bd3b81d78747140925874cc0ee5a3d2c47422629ec0ccb7 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/shards/shard_0 --shard-index 0 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/logs/shard_0.log 2>&1 & echo $!`
`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source; nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_plan.json --plan-manifest-sha256 393263d239fc55593d384cade173d4656cb2cd9b437d666ead2b60324895b049 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_context.json --context-manifest-sha256 6b170e0734e2a3000bd3b81d78747140925874cc0ee5a3d2c47422629ec0ccb7 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/shards/shard_1 --shard-index 1 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/logs/shard_1.log 2>&1 & echo $!`
`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source; nohup env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_plan.json --plan-manifest-sha256 393263d239fc55593d384cade173d4656cb2cd9b437d666ead2b60324895b049 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_context.json --context-manifest-sha256 6b170e0734e2a3000bd3b81d78747140925874cc0ee5a3d2c47422629ec0ccb7 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/shards/shard_2 --shard-index 2 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/logs/shard_2.log 2>&1 & echo $!`
`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source; nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_plan.json --plan-manifest-sha256 393263d239fc55593d384cade173d4656cb2cd9b437d666ead2b60324895b049 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_context.json --context-manifest-sha256 6b170e0734e2a3000bd3b81d78747140925874cc0ee5a3d2c47422629ec0ccb7 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/shards/shard_3 --shard-index 3 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/logs/shard_3.log 2>&1 & echo $!`
`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source; nohup env CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_plan.json --plan-manifest-sha256 393263d239fc55593d384cade173d4656cb2cd9b437d666ead2b60324895b049 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_context.json --context-manifest-sha256 6b170e0734e2a3000bd3b81d78747140925874cc0ee5a3d2c47422629ec0ccb7 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/shards/shard_4 --shard-index 4 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/logs/shard_4.log 2>&1 & echo $!`
`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source; nohup env CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_plan.json --plan-manifest-sha256 393263d239fc55593d384cade173d4656cb2cd9b437d666ead2b60324895b049 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_context.json --context-manifest-sha256 6b170e0734e2a3000bd3b81d78747140925874cc0ee5a3d2c47422629ec0ccb7 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/shards/shard_5 --shard-index 5 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/logs/shard_5.log 2>&1 & echo $!`
`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source; nohup env CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_plan.json --plan-manifest-sha256 393263d239fc55593d384cade173d4656cb2cd9b437d666ead2b60324895b049 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_context.json --context-manifest-sha256 6b170e0734e2a3000bd3b81d78747140925874cc0ee5a3d2c47422629ec0ccb7 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/shards/shard_6 --shard-index 6 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/logs/shard_6.log 2>&1 & echo $!`
`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source; nohup env CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_plan.json --plan-manifest-sha256 393263d239fc55593d384cade173d4656cb2cd9b437d666ead2b60324895b049 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/prepared/target125_context.json --context-manifest-sha256 6b170e0734e2a3000bd3b81d78747140925874cc0ee5a3d2c47422629ec0ccb7 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/shards/shard_7 --shard-index 7 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/logs/shard_7.log 2>&1 & echo $!`

## 9.系统性技术停止与归档

- 触发时间：2026-08-05 12:17:28（CST）。shard_3与shard_7在不同worker路径、尚未产出prediction前，先后出现同一确定性异常，满足预注册的“至少两个不同row同指纹”停止规则：`cvsrffi.stage2_zid_student_t_qknn.ZIDStudentTQKNNError: z_id rows contain a zero-norm vector`，外层异常为`NextR5FATarget125RuntimeError: sealed z_id160 materialization failed`，发生于`REG1` support的sealed `z_id160` materialization。
- 运行绑定核验：shard_0至shard_7 PID依次为`1662376`至`1662383`；存活的`1662376、1662377、1662378、1662380、1662381、1662382`均核验CWD=`$RUN_ROOT/source`、cmdline含正确plan/context SHA及对应shard index。已对这6个仅属于本run的进程发送定向`SIGTERM`，等待3秒后全部退出；未使用广域终止。
- 停止后核验（2026-08-05 12:18:18 CST）：PID`1662376`至`1662383`均已停止；GPU0至7均为`1MiB/0%`且无compute app；SSH连接均为`SSH_CLEAN`。
- 日志证据：`shard_3.log`与`shard_7.log`各`2296`bytes，SHA256均为`e3baa52eeb3553994b1f69d08c69ece18eb7ecd336f75658c06d9d38e0b3b7c8`；其余6个日志均为`0`bytes，SHA256均为`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
- 部分输出保留：shard_0/1/2/3/4/5/6/7分别有`96/96/102/34/96/96/96/62`个partial JSON，共`678`个、`27330246`bytes。未生成shard manifest/receipt、merge manifest、truth或score；因此outer/scene/logical/unique矩阵均未闭合，不能作任何性能结论。
- 停止证据目录：`E:\fa125wt\automation_reports\CV-SincNet\next_r5_fa_rdce3_q_target125_20260805_r5\artifacts\stop_evidence_20260805_r5_zero_norm`；根目录镜像为`E:\type10-7\automation_reports\CV-SincNet\next_r5_fa_rdce3_q_target125_20260805_r5\artifacts\stop_evidence_20260805_r5_zero_norm`。目录含8份日志、runner停止证据及回收的小型sealed输入/receipt；partial JSON仅作保留证据，不纳入Git提交。
- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。按handoff约束不重试、不续跑、不覆盖；fresh retry authority=`无`。修复需由主agent另建非覆盖run ID后重新完成本地审查与发布门禁。
