# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r6）

## 身份与最终状态

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r6`；日期：2026-08-05。
- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 目标：在同一sealed输入下执行FA-RDCE3+qKNN完整125四状态矩阵，联合观察域适应与新类注册；本run未进入正式prediction、truth或score。
- 主agent：`gpt-5.6-sol/high`；科学实现/独立复核：`Terra/max`；唯一N607 runner：`Luna/max`。
- Git branch=`codex/next-r5-fa-q-target125-20260805`；科学commit=`8ae765fd2107db477233b8e27af5f91a69e633c6`；本次仅更新报告与证据。

## 冻结矩阵与协议

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed={713102,713103,713104,713105,713106}
(K,new)={(10,5),(10,10),(10,20),(5,20),(1,20)}
125 outer×3 leo_*_weak scene×4 state=375 scene rows/1500 logical surfaces
=1350 unique predictions+150 K1 aliases
```

- 四状态固定为`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`；REG0的seen-new与H为`N/A`。
- 协议固定为`p2_min_v1`；query truth/role/quota/global reassignment/fit/update/selection均不得访问。
- 数据继续复用`VALIDATED_ONCE`的sealed D108输入；本run未改变received IQ、物理ID、receiver/TX、K、support/query划分或协议schema。

## 版本闭包与远端落地

- closure：`E:\type10-7\code\snapshots\next_r5_fa_rdce3_q_target125_20260805_r6_closure_8ae765fd.tar`；大小=`73154560`bytes；SHA256=`4227bccfbd614ba5dd57f1bd75efe1738539c2f671c5e68be3e53908e742b113`。
- method lock/builder/core/runtime/CLI SHA256：`3b9059f545620bf2a47e8bd79b537ede15a1eb7fdce4be3fee952d4a27dcc6b9`/`9b6b938d87fdfa603f6e0c8be374c77ca7399430ea898cd37c0c37a530880e38`/`88777df6cdaf352bea37c9b4bd36a78a9b32a24947bf5404b32dad2037b7c2ac`/`994412c0c1bed7ede9c98b13132c1c3a22eed381c6698bdff9a2a6f4d7219336`/`d0811d699629aa71b75d9d6f111a48f2d2cfc0468788d14e95b3df62bcc0cca5`。
- 远端RUN_ROOT：`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r6`；落地前确认`ABSENT`；r1至r5确认存在但未触碰。
- direct N607 preflight：`PASS`；server=`dell-DSS8440`；Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；GPU0至7均为`0%/1MiB`；输入SHA全部匹配；每次SSH/SCP结束均`SSH_CLEAN`。
- closure远端SHA与大小匹配；六入口远端`py_compile`输出`R6_PY_COMPILE_OK`。

## 新asset与prepare证据

- r6未复用r4/r5 asset。新method-lock绝对路径为`.../r6/source/configs/next_r5_fa_rdce3_q_target125_20260805.json`。
- 独立新asset目录：`.../r6/source/input/fa_asset2`；wire SHA=`2a7c94fe218142e5773c6627c243d5724204aa789f0f50bb1bbcfb41a743e934`；semantic asset SHA=`9f1989fdd56db7d701a3edf1624928fe788d971f3ae2eea142906bb3337a67be`；manifest SHA=`232ab418b64e30c7ed867b5ddc93def3bf7717a352520e85f376c3e9b684358e`。
- asset封存`source_class_indices=[0,1,2,3,4,5]`、旧类根=`f23394f508cae38116e7541b7954e647d4d47aa8f6abc69620cfa2813b873212`、`target_support_rows_used=0`、`target_query_rows_used=0`。
- prepare状态=`D108_SEALED_INPUTS_AND_TARGET_FA_ASSET_PINNED`；`outer_job_count=125`、`scene_row_count=375`、`logical_state_surface_count=1500`、`unique_prediction_count=1350`、`alias_count=150`；五个query访问字段均为`false`。
- prepare产物SHA：plan=`17a7e2e5f6954a2470498ab05b295a60efb6d5d1e70b85c0ff314a94d6a44cd8`；context=`7f4c7794b39c8284030137fb5f50572deaa2a16665932c7d3102ceb474af15e5`；receipt=`1b8c277292cd24e70abb3db4202d5f63d192704f8ae1d4a2e91b3910dc4f656e`。

## truth-free smoke与停止证据

预注册row0/scene0的真实checkpoint truth-free smoke在正式prediction前失败，未读取truth、未生成prediction、未打开truth catalog。

|尝试|命令条件|退出|确定性指纹|结果|
|---|---|---:|---|---|
|GPU smoke|`--device cuda:0`，row-index=0，scene-index=0|1|`sealed runtime / same-IQ ReLU binding drift`|无smoke receipt|
|CPU smoke|`--device cpu`，同一row/scene|1|`Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)`，外层为`sealed z_id160 materialization failed`|无smoke receipt|

- GPU失败为单row真实运行证据；只做了一次预注册GPU smoke，未据此读取性能。CPU尝试仅为设备兼容性核验，未改变方法、输入或矩阵。
- 只读诊断显示GPU smoke的REG0 support同IQ绑定最大绝对误差=`6.9090724e-4`，冻结容差=`2e-6`；shape=`[60,160]`，sealed与重算均无全零行。这是r6 runtime binding技术故障，不是性能结果。
- smoke失败后未启动8个predict-shard；未生成shard manifest、merged prediction manifest、truth catalog或score artifact。GPU0至7回到`0%/1MiB`，无run-owned进程，SSH=`SSH_CLEAN`。
- 失败证据与取回的asset/prepare文件保存在本报告目录的`artifacts/`下；远端r6 RUN_ROOT保留source、asset和prepared，不删除、不覆盖。

## 运行闭合表

|状态|outer|scene|logical|unique|K1 alias|truth|score|结论|
|---|---:|---:|---:|---:|---:|---|---|---|
|最终停止|0/125|0/375|0/1500|0/1350|0/150|未打开|未产生|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

## 结论与后续

- 本run没有任何可报告的域适应前/后、新类注册前/后性能值，也没有正收益证据；不得进入候选排名、promotion或与D92结果的数值比较。
- 停止触发来自预注册的技术健康边界（真实checkpoint smoke在prediction前失败），不是因为中途准确率或其他性能指标弱。
- `fresh retry authority=无`。后续必须由主agent/Terra在本地修复并重新完成独立复核、生成新的non-overwriting closure和新的run ID；本runner不改代码、不绕过binding、不重试本run。

## 证据索引

- `artifacts/remote/fa_rdce3_target125.wire`
- `artifacts/remote/fa_target125_asset_manifest.json`
- `artifacts/remote/target125_plan.json`
- `artifacts/remote/target125_context.json`
- `artifacts/remote/prepare_receipt.json`
- `artifacts/runner_stop_evidence.md`
- `artifacts/smoke_cuda0.stderr.txt`
- `artifacts/smoke_cpu.stderr.txt`

