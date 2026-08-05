# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r8）

## 身份、修复与验证

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r8`；日期：2026-08-05；状态：`LOCAL_VERIFIED`。
- 目标：完成FA-RDCE3+qKNN完整125四状态矩阵，联合验证域适应和新类注册。
- r7真实smoke发现同一旧support因REG0 60行和REG1 110行GPU batch形状不同而产生最大`8.1807375e-05`漂移；8分片未启动，无prediction/truth/score，严格为`NO_PERFORMANCE_RESULT`。
- r8保持r7单forward PR160图。REG0将support pre-ReLU和z_id160写入只读support-only缓存；REG1按sealed package/outer/scene、opaque physical ID、received-IQ shape+字节SHA及PR160 descriptor严格复用旧前缀，只forward新增support。query始终`batch_size=1`且不读写缓存。
- 合成60→110回归的forward序列为`60 support、2 query、50 new support、2 query`，不再重复110行support；错序或IQ字节漂移fail closed。
- 六入口编译、五套聚焦测试主agent/实现方均`29 passed`、`git diff --check`通过；独立Terra复核`P0=0，P1=0`。
- 主agent=`gpt-5.6-sol/high`；科学实现/复核=`Terra/max`；唯一runner=`Luna/max`。

## 版本闭包

- 科学commit=`99e84efa5109107a26e5e7bb010a5a0653663bbf`。
- closure=`E:\type10-7\code\snapshots\next_r5_fa_rdce3_q_target125_20260805_r8_closure_99e84efa.tar`；大小=`73164800`bytes；SHA=`4688f746dfcf126980a13f2ea1bbba9ad0c4c61fb0e77f4491c938d8710d6e0a`。
- method lock/builder/core/runtime/CLI SHA=`8d4fd0e5d871e89d05abeabfdc39792ba5e760033bc9232f9dc5f7bb788478c1`/`9b6b938d87fdfa603f6e0c8be374c77ca7399430ea898cd37c0c37a530880e38`/`88777df6cdaf352bea37c9b4bd36a78a9b32a24947bf5404b32dad2037b7c2ac`/`38a69778a3131a144fcece1d7bd066f50c829852aba1c43d265f2c1a29ccbb38`/`6cd8c9c682e5cc4ac05b7c560c887858a5d5ec62147a4b3d535321806617c06c`。

## 冻结矩阵与输入

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed={713102,713103,713104,713105,713106}
(K,new)={(10,5),(10,10),(10,20),(5,20),(1,20)}
125 outer×3 scene×4 state=375 scene rows/1500 logical surfaces
=1350 unique predictions+150 K1 aliases
```

- D106 strict tap/receipt SHA=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`/`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`；checkpoint SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- D108 plan/context SHA=`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`/`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`；复用`p2_min_v1/VALIDATED_ONCE`。
- PR160 extractor路径=`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/input/d92_pr160_extractor_runtime.pt`；SHA=`56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`；大小=`4618957`。

## N607执行

- RUN_ROOT=`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r8`，必须先为`ABSENT`；r1至r7不可触碰。
- Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`RUN_ROOT/source`；GPU0至7各一固定shard。
- 顺序：preflight→输入/GPU/RUN_ROOT→SCP closure→SHA/compile→新run asset→prepare（显式PR160路径/SHA）→truth-free smoke→8 shards→merge→truth-open→score→取回。
- 复用r7 method lock和FA科学资产语义，但必须在r8新目录重建独立asset；prepare须传`--pr160-extractor-runtime`和`--pr160-extractor-runtime-sha256`。
- 成功闭合：125/125 outer、375/375 scene、1500/1500 logical、1350 unique、150 alias、8/8 shard、完整manifest、truth和score。
- 停止仅限P0/安全/hash/覆盖故障或至少两个不同row预测前同一确定性异常；不得因性能停止；fresh retry authority=`无`。

## 执行结果

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 13:53 direct N607 preflight通过；r8在启动前为`ABSENT`，r1至r7保持存在且未触碰；GPU0至7均为`0%/1MiB`，冻结输入SHA全部匹配。
- closure已落地并独立解压；远端closure SHA=`4688f746dfcf126980a13f2ea1bbba9ad0c4c61fb0e77f4491c938d8710d6e0a`、大小=`73164800`bytes；五个科学入口SHA和`R8_PY_COMPILE_OK`均匹配本地闭包。
- 独立FA资产构建成功：wire=`asset/fa_rdce3_target125.wire`，wire SHA=`af971fb6829e0dd1ff7aed52df0841aed697d1d1c782f742d1918316f1e889b9`；semantic asset SHA=`cae219c47cf41c8b21c2b460f87388b3b9bdab525154ff34a8ed9e2c66250c0d`；manifest SHA=`c10a4d3c191111cd95194d3f855cbef32d26490f324f8891afd32549d7a6b73b`。
- `prepare`成功：`D108_SEALED_INPUTS_AND_TARGET_FA_ASSET_PINNED`；125 outer、375 scene、1500 logical、1350 unique、150 alias；显式复用PR160 extractor路径及SHA；query truth/role/selection/update/fit均为`false`。最终prepared plan/context SHA=`39ab00044339532c3e60f3c08dc7e2ec02db7db531bc7017edb092f8613dad9b`/`17c4472a0862d2f58583ccaebd2dae46509d8884257a2dee2e50cb95960cf88e`。
- 真实checkpoint truth-free smoke成功：`REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS`；`DA0_REG0=120`、`DA1_REG0=120`、`DA0_REG1=220`、`DA1_REG1=220`，仅为技术健康证据，不是性能结果。
- 8个分片PID=`1720239`至`1720246`已启动并分别占用GPU0至7（每卡约546MiB），命令、run-root和输出路径均指向r8；但启动后核验发现8个进程的`/proc/PID/cwd`均为`/home/szu2070436088`，不符合本报告冻结的`CWD=RUN_ROOT/source`绑定。该错误属于P0运行绑定故障，不能把该批进程或任何中途状态计作实验结果。
- 已仅对上述8个经命令行/run-root核验的PID发送`SIGTERM`；5秒后8个PID全部退出，GPU compute-app列表为空，8个shard日志保留且均为0bytes，未产生prediction、merge、truth或score；SSH/SCP均已清理。
- 首次asset命令末尾误校验不存在的`manifest.json`导致外层shell返回1，但builder本体已报告成功，实际manifest文件和SHA已只读复核；这不是实验运行故障。
- 按handoff的`fresh retry authority=无`，r8不重试、不打开truth、不评分，不产生任何性能或推广结论。后续若需有效实验，必须使用新的不可覆盖run ID并重新完成绑定核验。

|outer|scene|logical|unique|truth|score|结论|
|---:|---:|---:|---:|---|---|---|
|0/125|0/375|0/1500|0/1350|未打开|未产生|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

### 四状态性能字段

由于r8未形成任何完整prediction manifest，`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`的准确率、old/new harmonic、floor及其差分均为`N/A`；smoke计数不参与性能评分。
