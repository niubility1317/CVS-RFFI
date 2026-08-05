# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r4）

## 1.身份与状态

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r4`
- 日期：2026-08-05
- 当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 候选：`NEXT-R5-FA-RDCE3-Q-TARGET125`
- 主agent：`gpt-5.6-sol/high`；唯一N607 runner：`Luna/max`
- Git worktree：`E:\fa125wt`；branch=`codex/next-r5-fa-q-target125-20260805`
- 科学实现与class-index bridge commit：`18db4cee`
- closure：`E:\type10-7\code\snapshots\next_r5_fa_rdce3_q_target125_20260805_r4_closure_18db4cee.tar`
- closure SHA256：`b96aa29c25ebf29f66ee0dfcb465841daac813ccc67cb082ac0bc3b9daf129ef`；大小=`73134080`bytes
- method lock v2 SHA256：`ff4731b38d390973fef9a61a862c823624ca090c55caf5c0aec1778c191b385c`
- builder/core/runtime/CLI SHA256：`9b6b938d87fdfa603f6e0c8be374c77ca7399430ea898cd37c0c37a530880e38`/`2695b942126eb0ecc40e6eca448438fa82126a30de399a7565ac66a285fa2049`/`57e5af4662746387db614899caf348d3dd2bcf00f450da4042eb93f6e1f6c7b3`/`d0811d699629aa71b75d9d6f111a48f2d2cfc0468788d14e95b3df62bcc0cca5`

## 2.历史技术失败与r4修复边界

|run|停止点|状态|性能结果|
|---|---|---|---|
|r1|closure method-lock CRLF字节漂移|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|无|
|r2|prepare产物写完后CLI `mappingproxy` JSON回传异常|同上|无|
|r3|truth-free row0/scene0 smoke发现raw TX标签与row-local `cls_*`句柄表示不一致|同上|无|
|r4|显式sealed class-index bridge修复后重新发布|`NOT_LANDED`|待跑|

r4不放宽类别校验。method lock v2固定source class indices=`0..5`和D106有序旧类根`f23394f508cae38116e7541b7954e647d4d47aa8f6abc69620cfa2813b873212`；asset builder必须实读同一绝对method-lock文件并验证其文件SHA、schema、candidate、protocol、indices和根。FA asset/wire封存source indices与有序根。运行时从每个sealed support/query package严格读取`{class_index,class_handle}`，要求REG0为`0..5`、REG1旧类前缀与REG0完全一致。core按class index取Phase1 center，按row-local handle分组support和输出qKNN标签。K1严格旁路、K5/K10闭式FA、REG1复用同一REG0状态对象均不变。

## 3.冻结125矩阵与四状态

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed={713102,713103,713104,713105,713106}
(K,new)={(10,5),(10,10),(10,20),(5,20),(1,20)}
125 outer × 3 leo_*_weak scene × 4 state
=375 scene rows / 1500 logical surfaces
=1350 unique predictions + 150 K1 aliases
```

四状态：`DA0_REG0=域适应前/新类注册前`、`DA1_REG0=域适应后/新类注册前`、`DA0_REG1=域适应前/新类注册后`、`DA1_REG1=域适应后/新类注册后`。REG0的seen-new/H为`N/A`。报告主效应`DA1_REG0-DA0_REG0`、`DA1_REG1-DA0_REG1`，注册效应`DA0_REG1-DA0_REG0`、`DA1_REG1-DA1_REG0`及定义于四状态指标的interaction。

## 4.本地验证与审查

- `ssr-gpu`六入口`py_compile`：通过。
- 五份聚焦测试：`19 passed`。
- 配置JSON解析与`git diff --check`：通过。
- 负测：method-lock文件SHA自身有效但外部旧类根被替换时，builder在strict tap/root比对处fail closed；旧v1 wire严格拒绝。
- 独立Terra/max复审：`P0=0，P1=0`。
- r4 archive成员与本地文件逐字节一致；method lock、builder、core、runtime、CLI成员SHA均已复核。

## 5.冻结输入

|输入|远端路径|SHA256|
|---|---|---|
|D106 strict tap|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|strict tap receipt|同目录`d106_ls_strict_tap.receipt.json`|`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|D108 plan|`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/prepared/target125_plan.json`|`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`|
|D108 context|同目录`target125_context.json`|`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`|

数据capsule/split未变化，按`p2_min_v1`复用`VALIDATED_ONCE`数据，不重验。

## 6.N607冻结执行

- RUN_ROOT：`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4`，落地前必须为`ABSENT`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`RUN_ROOT/source`。
- 顺序：preflight→输入/GPU/RUN_ROOT核验→sync closure→archive/member/source SHA→六入口py_compile→build asset→prepare→truth-free smoke→8 shards→merge→truth-open→score。
- asset入口必须含：`python -u code/scripts/build_next_r5_fa_target125_asset.py --strict-tap <frozen tap> --strict-tap-sha256 48b92... --checkpoint-sha256 2699... --method-lock $RUN_ROOT/source/configs/next_r5_fa_rdce3_q_target125_20260805.json --method-lock-sha256 ff4731... --output-dir $RUN_ROOT/input/fa_asset`。
- 其余入口固定使用`python -u code/scripts/run_next_r5_fa_target125.py {prepare|smoke|predict-shard|merge|truth-open|score}`，runner须在每阶段前把完全展开命令、SHA、CWD、环境、日志和输出路径追加到报告。
- GPU i固定`CUDA_VISIBLE_DEVICES=i`，子命令`--device cuda:0 --shard-index i`，i=0至7。
- 成功闭合：125/125 outer、375/375 scene、1500/1500 logical、1350 unique、150 aliases、8/8 shard、完整prediction manifest、truth-open与score。
- 停止仅限P0协议/安全/hash/覆盖故障或至少两个不同outer在prediction前出现相同确定性异常指纹；不得读取性能值决定停止。
- fresh retry authority：无。失败保留artifact并另建run ID，不覆盖、不续跑。

## 7.结果表

|状态|outer|scene|logical|unique|truth|score|结论|
|---|---:|---:|---:|---:|---|---|---|
|当前（停止）|未闭合|未闭合|未闭合|仅422个partial JSON|未打开|未产生|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

## 8.Runner landed/asset证据与prepare命令

- r4 closure已落地并完成archive/member/source SHA核验；六入口远端`py_compile`输出`R4_PY_COMPILE_OK`。r1-r3仅确认存在，未触碰。
- FA asset semantic SHA=`fa2e0205d8284c1c7b14865c6564c1120c5cc01d1985a8ac8e6ff9ec60c89016`；wire file SHA=`c81a34b4aa3964fbf44acbd99829ac98904f25aca0276c1730878bc96964fe49`；manifest SHA=`fc11e4f7b96edbdc4e3a019d021fcff66fe8d9a94b2a03c125d1fabef6b6989e`；manifest source indices=`[0,1,2,3,4,5]`、source old-class root=`f23394f508cae38116e7541b7954e647d4d47aa8f6abc69620cfa2813b873212`，query/support rows used均为0。
- prepare完整命令：`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source; /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py prepare --d108-plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/prepared/target125_plan.json --d108-plan-manifest-sha256 13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348 --d108-context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/prepared/target125_context.json --d108-context-manifest-sha256 067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f --fa-asset /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/input/fa_asset/fa_rdce3_target125.wire --fa-asset-sha256 c81a34b4aa3964fbf44acbd99829ac98904f25aca0276c1730878bc96964fe49 --method-lock /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source/configs/next_r5_fa_rdce3_q_target125_20260805.json --method-lock-sha256 ff4731b38d390973fef9a61a862c823624ca090c55caf5c0aec1778c191b385c --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared`。

## 9.Prepare收据与smoke命令

- prepare exit=`0`；状态=`D108_SEALED_INPUTS_AND_TARGET_FA_ASSET_PINNED`；`outer_job_count=125`、`scene_row_count=375`、`logical_state_surface_count=1500`、`unique_prediction_count=1350`、`alias_count=150`；`query_truth_access/query_role_access/query_selection_access/query_update_access/query_fit_access`均为`false`；SSH清理=`SSH_CLEAN`。
- prepared文件SHA：`target125_plan.json=94ca4dbeb5f4af3ee17f3c6a25ad48c644b1d18b8ce2d7ad015aafb715e6ab12`；`target125_context.json=504b743268e27e1a42d05c18e5aa1b1bbe9efa9178713021270c82ff9373ba69`；`prepare_receipt.json=c58b220f91516fa7cbe3037c64a83613f5243be5ef9a67e486dacf65e91c575b`。CLI返回的canonical receipt SHA=`56f0c6c156f694d5592123df86e8056bda6c399f150a6feb0b4c430c9b458018`，plan receipt SHA=`04ce2312e43754119f8c5e8a4026c636f07031f9875b122c813739d1bea752c3`，context receipt SHA=`3327a7ab2bb8cb7d5fd625f5a4b271ac67775d478b634d6d7de3113c208d1253`。
- 下一阶段唯一预注册truth-free smoke命令（CWD=`.../r4/source`，GPU0，row-index=`0`、scene-index=`0`，输出目录此前不存在）：`cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source; CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py smoke --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_plan.json --plan-manifest-sha256 94ca4dbeb5f4af3ee17f3c6a25ad48c644b1d18b8ce2d7ad015aafb715e6ab12 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_context.json --context-manifest-sha256 504b743268e27e1a42d05c18e5aa1b1bbe9efa9178713021270c82ff9373ba69 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/smoke --row-index 0 --scene-index 0 --device cuda:0`。预期仅生成`smoke/smoke_receipt.json`，不打开truth、不产生正式prediction/score；smoke失败则按r4单点审查规则停止，不自行重试或改动方法。
## 10.Smoke通过与8-shard启动命令

- 2026-08-05T11:49:52+08:00启动前GPU只读检查：GPU0–7均为`1MiB/0%`，无compute app；SSH清理=`SSH_CLEAN`。
- smoke exit=`0`；receipt file SHA=`546078963fafe2096403cceab66eaa8a37637140f47b7c5cf89569c49973ad51`；canonical SHA=`93241994ba9a2fe4b6445b0360729973564396f96bfd3746cb2765459a674aea`；四状态计数=`DA0_REG0:120, DA1_REG0:120, DA0_REG1:220, DA1_REG1:220`；正式prediction/truth/score尚未产生。
- 已执行的8条detached命令（每条均在CWD=`.../r4/source`，`PYTHONUNBUFFERED=1`，固定对应GPU；PID由远端shell返回并已独立核验）：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source; nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_plan.json --plan-manifest-sha256 94ca4dbeb5f4af3ee17f3c6a25ad48c644b1d18b8ce2d7ad015aafb715e6ab12 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_context.json --context-manifest-sha256 504b743268e27e1a42d05c18e5aa1b1bbe9efa9178713021270c82ff9373ba69 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/shards/shard_0 --shard-index 0 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/logs/shard_0.log 2>&1 & echo $!
cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source; nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_plan.json --plan-manifest-sha256 94ca4dbeb5f4af3ee17f3c6a25ad48c644b1d18b8ce2d7ad015aafb715e6ab12 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_context.json --context-manifest-sha256 504b743268e27e1a42d05c18e5aa1b1bbe9efa9178713021270c82ff9373ba69 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/shards/shard_1 --shard-index 1 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/logs/shard_1.log 2>&1 & echo $!
cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source; nohup env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_plan.json --plan-manifest-sha256 94ca4dbeb5f4af3ee17f3c6a25ad48c644b1d18b8ce2d7ad015aafb715e6ab12 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_context.json --context-manifest-sha256 504b743268e27e1a42d05c18e5aa1b1bbe9efa9178713021270c82ff9373ba69 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/shards/shard_2 --shard-index 2 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/logs/shard_2.log 2>&1 & echo $!
cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source; nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_plan.json --plan-manifest-sha256 94ca4dbeb5f4af3ee17f3c6a25ad48c644b1d18b8ce2d7ad015aafb715e6ab12 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_context.json --context-manifest-sha256 504b743268e27e1a42d05c18e5aa1b1bbe9efa9178713021270c82ff9373ba69 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/shards/shard_3 --shard-index 3 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/logs/shard_3.log 2>&1 & echo $!
cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source; nohup env CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_plan.json --plan-manifest-sha256 94ca4dbeb5f4af3ee17f3c6a25ad48c644b1d18b8ce2d7ad015aafb715e6ab12 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_context.json --context-manifest-sha256 504b743268e27e1a42d05c18e5aa1b1bbe9efa9178713021270c82ff9373ba69 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/shards/shard_4 --shard-index 4 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/logs/shard_4.log 2>&1 & echo $!
cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source; nohup env CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_plan.json --plan-manifest-sha256 94ca4dbeb5f4af3ee17f3c6a25ad48c644b1d18b8ce2d7ad015aafb715e6ab12 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_context.json --context-manifest-sha256 504b743268e27e1a42d05c18e5aa1b1bbe9efa9178713021270c82ff9373ba69 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/shards/shard_5 --shard-index 5 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/logs/shard_5.log 2>&1 & echo $!
cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source; nohup env CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_plan.json --plan-manifest-sha256 94ca4dbeb5f4af3ee17f3c6a25ad48c644b1d18b8ce2d7ad015aafb715e6ab12 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_context.json --context-manifest-sha256 504b743268e27e1a42d05c18e5aa1b1bbe9efa9178713021270c82ff9373ba69 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/shards/shard_6 --shard-index 6 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/logs/shard_6.log 2>&1 & echo $!
cd /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source; nohup env CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_next_r5_fa_target125.py predict-shard --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_plan.json --plan-manifest-sha256 94ca4dbeb5f4af3ee17f3c6a25ad48c644b1d18b8ce2d7ad015aafb715e6ab12 --context-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/prepared/target125_context.json --context-manifest-sha256 504b743268e27e1a42d05c18e5aa1b1bbe9efa9178713021270c82ff9373ba69 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/shards/shard_7 --shard-index 7 --device cuda:0 > /home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/logs/shard_7.log 2>&1 & echo $!
```
## 11.系统性技术停止与partial artifact

- 2026-08-05T11:54:14+08:00 first-wave核验发现shard0、shard3、shard5、shard7四个不同shard均出现同一确定性异常指纹：`cvsrffi.stage2_next_r5_fa_target125_core.NextR5FATarget125CoreError: qKNN exact top tie fails closed`，并以`FA-RDCE3/qKNN four-state execution failed`退出；满足预注册的“两不同row同指纹”系统性停止条件。该异常是技术失败，不是性能判断。
- 被停止PID在动作前逐一核验：1646282→shard1、1646283→shard2、1646285→shard4、1646287→shard6；四者CWD均为`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source`，cmdline均为r4 closure中的`run_next_r5_fa_target125.py predict-shard`并带对应`--shard-index`。随后仅对这四个run-owned PID发送`SIGTERM`，3秒内均停止；未使用广泛kill、未删除/覆盖/重启artifact。
- 2026-08-05T11:55:19+08:00停止后核验：GPU0–7均`1MiB/0%`且compute apps为空；SSH=`SSH_CLEAN`；run-owned Python PID均不存在。失败日志shard0/3/5/7各2329B，SHA均为`3c203634df9f9fc53d0d2f02d1cac8dac18662aa838991848f5616307fa2dbb`；SIGTERM分片shard1/2/4/6日志均为0B，SHA均为`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
- partial prediction已只读拉回并核对：422个JSON、总字节`16850416`；按shard计数为`36/68/70/26/64/40/68/50`（shard0至shard7）。未生成任何shard manifest、merged prediction manifest、truth catalog或score artifact；因此不得进入性能分析或promotion。
- 停止证据与拉回文件：`E:\fa125wt\automation_reports\CV-SincNet\next_r5_fa_rdce3_q_target125_20260805_r4\artifacts\stop_evidence_20260805_r4_qknn_top_tie\`（含8份日志、`shards/`部分prediction和`runner_stop_evidence.md`）。本run最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；fresh retry authority=`无`。
