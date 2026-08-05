# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r10）

## 身份与最小发布修复

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r10`；日期：2026-08-05；终态：`PREDICTIONS_COMPLETE / TRUTH_OPEN_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
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

## 执行结果

### 阶段闭环

|阶段|状态|证据|
|---|---|---|
|direct preflight|通过|N607直连、项目根目录可见、8卡空闲；SSH_CLEAN|
|closure落地|通过|73164800bytes；closure SHA=`4688f746dfcf126980a13f2ea1bbba9ad0c4c61fb0e77f4491c938d8710d6e0a`；五个入口SHA与py_compile通过|
|FA asset|通过|wire SHA=`af971fb6829e0dd1ff7aed52df0841aed697d1d1c782f742d1918316f1e889b9`；semantic SHA=`cae219c47cf41c8b21c2b460f87388b3b9bdab525154ff34a8ed9e2c66250c0d`；manifest SHA=`6d668488f59707f80abacaaa79d6f9248b59dd17705002e1f409f7b1b9d5839e`；source indices=`[0,1,2,3,4,5]`；target support/query使用=`0/0`|
|prepare|通过|`D108_SEALED_INPUTS_AND_TARGET_FA_ASSET_PINNED`；125 outer、375 scene、1500 logical、1350 unique、150 alias；五项query访问均为`false`；plan SHA=`0d28fc51548907843618524eb31a4b023971535ff861412d53033d3b20f1c292`；context SHA=`5c062dd13516b8a41b2d3ec9597345f2e792606021d2a9eebd13822ed08769d6`|
|真实checkpoint truth-free smoke|通过|`REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS`；四状态计数：`DA0_REG0=120`、`DA1_REG0=120`、`DA0_REG1=220`、`DA1_REG1=220`；receipt file SHA=`546078963fafe2096403cceab66eaa8a37637140f47b7c5cf89569c49973ad51`|
|8个predict-shard|通过|8/8进程启动并核验CWD/cmdline/PPid/GPU；outer计数=`16/16/16/16/16/15/15/15`；surface计数=`192/192/192/192/192/180/180/180`；合计125 outer、1500 logical；无异常指纹|
|merge|通过|`merged/prediction_manifest.json`；33027483bytes；SHA=`29982326a4c3130abea223a56c19362f9cf7e583da186e3498de2b80758c0c49`；1350 unique+150 alias闭合|
|truth-open|技术失败|未生成truth；原始错误为`KeyError: 'old_classes'`（`stage2_d108_truth_scorer._validate_d92_truth_sidecar`）；上层错误=`D92 truth-sidecar binding failed`|
|score|未启动|因truth-open未闭合，未打开score、未产生任何性能数值|

### 预测分片同一行证据

|shard|outer|logical surface|manifest SHA|
|---:|---:|---:|---|
|0|16|192|`d7f9e8f53a3eecf9e7c73469bef4ae4ad55effa56966f06266826fe073753fe3`|
|1|16|192|`b87bc74e9b18e71022bda014842ab8ca02c30bf0af6740f36c3950dcfd88faaf`|
|2|16|192|`f76b26a71affd79e987576150a1ba773bf1553f52b9777aecaf72b2cd64b56e2`|
|3|16|192|`cf9348867721740ab584fc720d4c2c6b6ded2706c24cd81f80a86b8d8606c9f6`|
|4|16|192|`f5606366eb1e0f8fcdfa3f90fbe8130db0b5bd05db554bf4016c2d7941fdbb35`|
|5|15|180|`0f92289f8c5ea13a6d733512926e224483408a54b1fc370f7ac7c63335aa70ea`|
|6|15|180|`0bb6f13e955659a4c2121db39f8afb4c1f1330bda596c86566f4cb07433f78cd`|
|7|15|180|`219038c65ebb750d33c7acffd7585cd5b780a2887a62e8bfc01a5427f2db8f87`|

### truth-open失败边界

`r10`的预测闭包保持完整且不可变；失败发生在预测之后的truth-sidecar绑定，不是预测过程、GPU、数据capsule或协议访问失败。truth-open调用构造的`d108_outer`只包含`receiver/seed/k_shot/new_count`，而D92 sidecar validator要求`old_classes/new_classes`，因此在读取truth sidecar前以`KeyError: 'old_classes'`退出。该run严格标记为`PREDICTIONS_COMPLETE / TRUTH_OPEN_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得从prediction artifact推导任何DA或注册性能。

四状态指标均为`N/A`：`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`没有truth连接，故不报告old accuracy、old floor、`seen_new_acc`、`H_old_new`、forgetting或差分。不存在“正收益”或“负收益”结论。

### 取回与清理

- 远端不可覆盖产物根：`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r10`；r1至r9未触碰。
- 已取回至：`E:\type10-7\automation_reports\CV-SincNet\next_r5_fa_rdce3_q_target125_20260805_r10\retrieved`。
- 取回内容包括merged manifest、8个shard manifest及其prediction JSON、prepared plan/context/receipt、smoke receipt、FA asset和全部shard/truth-open日志；关键文件与远端SHA一致。
- truth-open失败日志本地SHA=`51cb48ba338871fc724d20b994ab4274e37eb8827709d1f0980a3e09c6681d1`；远端GPU、run进程、SSH/SCP均已清理（SSH_CLEAN）。

### 结论与后续

本run完成了完整125矩阵的协议合法预测闭环，但没有形成truth/score，因此不进入性能排名或方法晋级。下一次若继续，必须先在本地修复truth-sidecar的旧/新类绑定，补充聚焦测试和独立P0/P1复核，创建新的不可覆盖run ID；不得修改、重标记或续跑r10的prediction产物。
