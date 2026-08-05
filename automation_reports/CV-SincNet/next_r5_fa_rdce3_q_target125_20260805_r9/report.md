# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r9）

## 身份与发布修复

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r9`；日期：2026-08-05；状态：`LOCAL_VERIFIED`。
- 目标：完成FA-RDCE3+qKNN完整125四状态矩阵，联合验证域适应和新类注册。
- r8科学代码、asset、prepare和truth-free smoke路径不变；r8唯一失败是8-shard启动命令漏写`cd RUN_ROOT/source`，导致8个PID的CWD为`/home/szu2070436088`。停止时无prediction/truth/score，严格为`NO_PERFORMANCE_RESULT`。
- r9不改代码、方法、参数、矩阵、输入或测试；只使用新不可覆盖run ID，并把每条detached命令冻结为`cd RUN_ROOT/source; nohup env CUDA_VISIBLE_DEVICES=i ...`，启动后逐PID核验CWD/cmdline。
- 科学commit=`99e84efa5109107a26e5e7bb010a5a0653663bbf`；主agent29项测试与独立Terra复核`P0=0，P1=0`均沿用。

## 闭包与矩阵

- closure=`E:\type10-7\code\snapshots\next_r5_fa_rdce3_q_target125_20260805_r8_closure_99e84efa.tar`；大小=`73164800`；SHA=`4688f746dfcf126980a13f2ea1bbba9ad0c4c61fb0e77f4491c938d8710d6e0a`。
- method lock/builder/core/runtime/CLI SHA=`8d4fd0e5d871e89d05abeabfdc39792ba5e760033bc9232f9dc5f7bb788478c1`/`9b6b938d87fdfa603f6e0c8be374c77ca7399430ea898cd37c0c37a530880e38`/`88777df6cdaf352bea37c9b4bd36a78a9b32a24947bf5404b32dad2037b7c2ac`/`38a69778a3131a144fcece1d7bd066f50c829852aba1c43d265f2c1a29ccbb38`/`6cd8c9c682e5cc4ac05b7c560c887858a5d5ec62147a4b3d535321806617c06c`。
- 矩阵：5 receiver×5 seed×`{(10,5),(10,10),(10,20),(5,20),(1,20)}`=125 outer；3 scene×4 state=1500 logical；1350 unique+150 K1 alias。
- PR160 extractor路径=`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/input/d92_pr160_extractor_runtime.pt`；SHA=`56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`；大小=`4618957`。
- D106 tap/receipt、checkpoint、D108 plan/context SHA沿用r8报告；数据为`p2_min_v1/VALIDATED_ONCE`，不重验。

## N607冻结执行

- RUN_ROOT=`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r9`，必须先`ABSENT`；r1至r8不可触碰。
- Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；所有命令CWD=`RUN_ROOT/source`。
- 顺序：preflight→输入/GPU/RUN_ROOT→closure SHA/compile→独立asset→prepare（PR160 path/SHA）→truth-free smoke→8 shards→merge→truth-open→score→取回。
- 每条shard命令必须在同一shell显式`cd RUN_ROOT/source`后再`nohup`；启动后PID、PPID、`/proc/PID/cwd`、cmdline、GPU和日志全部核验，CWD不符立即P0停止。
- 停止仅限P0/安全/hash/覆盖或至少两个不同row预测前同一确定性异常；不得因性能停止；fresh retry authority=`无`。

## 执行结果

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 14:00:35 direct N607 preflight通过；r9启动前为`ABSENT`，r1至r8保持存在且未触碰；GPU0至7均为`0%/1MiB`，冻结输入SHA全部匹配。
- closure已落地并独立解压；远端closure SHA=`4688f746dfcf126980a13f2ea1bbba9ad0c4c61fb0e77f4491c938d8710d6e0a`、大小=`73164800`bytes；五个科学入口SHA和`R9_PY_COMPILE_OK`均匹配本地闭包。
- 独立FA资产构建成功：wire=`asset/fa_rdce3_target125.wire`，wire SHA=`af971fb6829e0dd1ff7aed52df0841aed697d1d1c782f742d1918316f1e889b9`；semantic asset SHA=`cae219c47cf41c8b21c2b460f87388b3b9bdab525154ff34a8ed9e2c66250c0d`；manifest SHA=`7aca1cfe5323009fd507e99571e67d0822fb07fc7a46f42f639ba61d3d45f3f5`。
- 首次prepare因命令中wire SHA少字符在参数校验处退出，prepared目录保持`ABSENT`且未启动任何进程；修正为上述核验SHA后prepare成功：`D108_SEALED_INPUTS_AND_TARGET_FA_ASSET_PINNED`；125 outer、375 scene、1500 logical、1350 unique、150 alias；显式复用PR160 extractor路径及SHA；query truth/role/selection/update/fit均为`false`。最终prepared plan/context SHA=`571f4eaa51e6083edfb70a6e93d3a412a7258eb76422d1a718662d9ea5ff537c`/`1cc243d81af520a7f2210eeb562371c42b03b8637fcd5b671f6792a46bd6d849`。
- 真实checkpoint truth-free smoke成功：`REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS`；`DA0_REG0=120`、`DA1_REG0=120`、`DA0_REG1=220`、`DA1_REG1=220`，仅为技术健康证据，不是性能结果。
- 8个分片PID=`1725906`、`1725907`、`1725908`、`1725909`、`1725911`、`1725912`、`1725913`、`1725914`已启动；逐PID核验的CWD均精确为`RUN_ROOT/source`，`CUDA_VISIBLE_DEVICES`分别为0至7，命令均指向r9及固定shard输出路径。随后本runner的cmdline字符串核验表达式误报绑定失败并进入预注册P0停止路径；该内部校验误报不能把该批进程或任何中途状态计作实验结果。
- 已仅对上述8个经PID/run-root核验的r9进程发送`SIGTERM`；复核显示8个PID全部退出，GPU compute-app列表为空，8个shard日志均保留且为0bytes；shard目录仅为空目录，未产生prediction、merge、truth或score；SSH/SCP均已清理。
- 按handoff的`fresh retry authority=无`，r9不重试、不打开truth、不评分，不产生任何性能或推广结论。后续若需有效实验，必须由主agent明确授权新的不可覆盖run ID并修正runner检查。

|outer|scene|logical|unique|truth|score|结论|
|---:|---:|---:|---:|---|---|---|
|0/125|0/375|0/1500|0/1350|未打开|未产生|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

### 四状态性能字段

由于r9未形成任何完整prediction manifest，`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`的准确率、old/new harmonic、floor及其差分均为`N/A`；smoke计数不参与性能评分。
