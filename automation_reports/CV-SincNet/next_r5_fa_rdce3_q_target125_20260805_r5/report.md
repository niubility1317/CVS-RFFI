# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r5）

## 1.身份与目标

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r5`
- 日期：2026-08-05；当前状态：`LOCAL_VERIFIED`
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
|当前|0/125|0/375|0/1500|0/1350|未打开|未产生|`LOCAL_VERIFIED`|

