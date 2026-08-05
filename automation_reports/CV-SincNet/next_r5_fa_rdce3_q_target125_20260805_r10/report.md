# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r10）

## 身份与最小发布修复

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r10`；日期：2026-08-05；状态：`LOCAL_VERIFIED`。
- 目标：完成FA-RDCE3+qKNN完整125四状态矩阵。
- r9真实prepare和truth-free smoke均通过，四状态计数120/120/220/220；8个shard的CWD和GPU映射正确，但runner的cmdline case校验器误报后自动SIGTERM。日志0字节、无prediction/truth/score，因此r9为`NO_PERFORMANCE_RESULT`。
- r10复用完全相同科学commit、closure、method lock、输入和矩阵；只更换sole runner并冻结更小的一次性启动核验：直接读取每个`/proc/PID/{cwd,cmdline,status}`，检查工具自身异常不能作为P0，必须由原始字段证明实际不符后才停止。
- 科学commit=`99e84efa5109107a26e5e7bb010a5a0653663bbf`；29项测试、独立复核`P0=0，P1=0`沿用。

## 冻结闭包与执行

- closure=`E:\type10-7\code\snapshots\next_r5_fa_rdce3_q_target125_20260805_r8_closure_99e84efa.tar`；73164800bytes；SHA=`4688f746dfcf126980a13f2ea1bbba9ad0c4c61fb0e77f4491c938d8710d6e0a`。
- lock/builder/core/runtime/CLI SHA=`8d4fd0e5d871e89d05abeabfdc39792ba5e760033bc9232f9dc5f7bb788478c1`/`9b6b938d87fdfa603f6e0c8be374c77ca7399430ea898cd37c0c37a530880e38`/`88777df6cdaf352bea37c9b4bd36a78a9b32a24947bf5404b32dad2037b7c2ac`/`38a69778a3131a144fcece1d7bd066f50c829852aba1c43d265f2c1a29ccbb38`/`6cd8c9c682e5cc4ac05b7c560c887858a5d5ec62147a4b3d535321806617c06c`。
- RUN_ROOT=`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r10`，先为`ABSENT`；r1至r9不可触碰。
- 矩阵：125 outer×3 scene×4 state=1500 logical；1350 unique+150 K1 alias；GPU0至7各一shard。
- PR160 extractor路径=`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/input/d92_pr160_extractor_runtime.pt`；SHA=`56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`；4618957bytes。
- 全流程：preflight→closure/SHA/compile→asset→prepare→smoke→8shard→merge→truth-open→score。所有CLI output-dir必须ABSENT；prepare使用wire文件SHA并显式PR160 path/SHA。
- 每条launch必须显式`cd RUN_ROOT/source && nohup env CUDA_VISIBLE_DEVICES=i ...`。直接核验原始CWD精确等于source、cmdline含r10 plan/context/output/shard-index、GPU映射和日志；验证辅助脚本错误不能触发停止。
- 停止仅限原始证据证明P0/安全/hash/覆盖故障或≥2不同row预测前同一确定性异常；不得因性能停止；fresh retry authority=`无`。

## 结果待填

|outer|scene|logical|unique|truth|score|结论|
|---:|---:|---:|---:|---|---|---|
|0/125|0/375|0/1500|0/1350|未打开|未产生|`LOCAL_VERIFIED`|

