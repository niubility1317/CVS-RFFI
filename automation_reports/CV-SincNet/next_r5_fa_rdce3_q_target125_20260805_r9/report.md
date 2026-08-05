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

## 结果待填

|outer|scene|logical|unique|truth|score|结论|
|---:|---:|---:|---:|---|---|---|
|0/125|0/375|0/1500|0/1350|未打开|未产生|`LOCAL_VERIFIED`|

