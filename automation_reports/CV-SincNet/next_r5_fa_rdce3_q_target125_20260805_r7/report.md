# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r7）

## 身份、修复与验证

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r7`；日期：2026-08-05；状态：`LOCAL_VERIFIED`。
- 目标：完成FA-RDCE3+qKNN完整125四状态矩阵，联合验证域适应和新类注册。
- r6真实smoke发现sealed TorchScript与eager checkpoint跨图ReLU绑定漂移，8分片未启动，无prediction/truth/score，严格为`NO_PERFORMANCE_RESULT`。
- r7删除r6双forward，直接复用D92-Lite-PR160已封存的单一graph-derived pre-ReLU extractor。正常行使用同图`ReLU(pre)`归一化；仅精确零ReLU行使用同一IQ非零signed pre归一化；非有限或pre精确零fail closed。
- PR160 source runtime SHA=`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`；extractor SHA=`56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`；大小=`4618957`bytes。
- `ssr-gpu`六入口编译、R5聚焦测试主agent`28 passed`、实现方扩展测试`29 passed`、`git diff --check`通过；独立Terra复核`P0=0，P1=0`。
- 主agent=`gpt-5.6-sol/high`；科学实现/复核=`Terra/max`；唯一N607 runner=`Luna/max`。

## 版本闭包

- 科学commit=`54802c7206e4516b1af5d65ed5125aa5c4d6c28e`。
- closure=`E:\type10-7\code\snapshots\next_r5_fa_rdce3_q_target125_20260805_r7_closure_54802c72.tar`；大小=`73154560`bytes；SHA=`0f18fdcba2343ddfb0d304ab6199a8fa48b40b4ee52c999007ce4f55c1583ddb`。
- method lock/builder/core/runtime/CLI SHA=`8d4fd0e5d871e89d05abeabfdc39792ba5e760033bc9232f9dc5f7bb788478c1`/`9b6b938d87fdfa603f6e0c8be374c77ca7399430ea898cd37c0c37a530880e38`/`88777df6cdaf352bea37c9b4bd36a78a9b32a24947bf5404b32dad2037b7c2ac`/`8c6a34ba6b2a820a054ba39e0980d1007a06323fb6c11941a0f12ee0e69c7296`/`6cd8c9c682e5cc4ac05b7c560c887858a5d5ec62147a4b3d535321806617c06c`。

## 冻结矩阵与输入

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed={713102,713103,713104,713105,713106}
(K,new)={(10,5),(10,10),(10,20),(5,20),(1,20)}
125 outer×3 scene×4 state=375 scene rows/1500 logical surfaces
=1350 unique predictions+150 K1 aliases
```

- D106 strict tap/receipt SHA=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`/`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`。
- checkpoint SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- D108 plan/context SHA=`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`/`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`；数据复用`p2_min_v1/VALIDATED_ONCE`，不重验。
- PR160 extractor绝对路径=`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/input/d92_pr160_extractor_runtime.pt`，prepare必须验证regular/non-symlink、SHA和大小并封存到新plan。

## N607执行

- RUN_ROOT=`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r7`，落地前必须`ABSENT`；r1至r6不可触碰。
- Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`RUN_ROOT/source`；GPU0至7各一固定shard。
- 顺序：direct preflight→输入/GPU/RUN_ROOT核验→SCP closure→SHA/compile→新method lock重建asset→prepare→truth-free smoke→8 shards→merge→truth-open→score→取回。
- prepare命令除既有D108 plan/context、FA asset和method lock参数外，必须新增`--pr160-extractor-runtime <上述绝对路径> --pr160-extractor-runtime-sha256 56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`。
- 成功闭合：125/125 outer、375/375 scene、1500/1500 logical、1350 unique、150 alias、8/8 shard、完整manifest、truth和score。
- 停止仅限P0/安全/hash/覆盖故障或至少两个不同row在prediction前同一确定性异常；不得因性能停止；fresh retry authority=`无`。

## 结果待填

|outer|scene|logical|unique|truth|score|结论|
|---:|---:|---:|---:|---|---|---|
|0/125|0/375|0/1500|0/1350|未打开|未产生|`LOCAL_VERIFIED`|

