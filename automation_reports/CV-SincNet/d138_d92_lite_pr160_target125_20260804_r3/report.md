# D138 D92-Lite-PR160 Target125实验报告r3

## 状态

- 实验ID：`d138_d92_lite_pr160_target125_20260804_r3`
- 当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- r1终态：prepare前缺少`stage2_d108_matrix_protocol.py`，未启动smoke/shard。
- r2终态：补齐首层D108/D92闭包后，prepare前缺少`stage2_dssc_zdom_jg_qknn_r4_bcrr.py`，未启动smoke/shard。
- r1/r2均保留、不覆盖、不续跑；r3是第二次发布缺陷后的更小一次性独立入口，retry authority仍为`false`。

## r3修复边界

- D138方法、method lock、数据、checkpoint、extractor、矩阵和协议完全不变；只把递归导入审计发现的最后两个基线依赖加入固定source闭包。
- 依赖闭包静态审计已确认：`stage2_dssc_zdom_jg_qknn_r4_bcrr.py`只再依赖已同步的`stage2_svrn_bcr.py`和`stage2_zid_student_t_qknn.py`；`stage2_svrn_bcr.py`只依赖已同步的`stage2_zid_student_t_qknn.py`，不再引入新的本地`cvsrffi`模块。
- D138仍只测`DA0_REG0=before`与`DA0_REG1=after`；`DA1_REG0/DA1_REG1`为单运输臂候选范围外，不生成四状态DA因果结果。

## 冻结身份

- 基础D138代码commit：`b27e088302d6f6bf4a6ae88e8357d626a2336236`。
- 当前发布记录commit：`0f7e4f09bd336705d351e0b6c104b49eddf45bac`；r3只增加本报告、同步映射和目标状态，不修改D138代码。
- candidate：`D92-Lite-PR160/r1`；method lock SHA256：`019dd59780de735af3026b091ef88b600c07d75c48f96aad0c2de34d49e8cee7`。
- sealed source runtime SHA256：`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`。
- extractor SHA256：`56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`；大小`4618957`字节。
- protocol：`p2_min_v1`；矩阵：125 outer、375 scene、750 before/after surface、8 shard。

## r3完整source依赖闭包

除D138代码和r2的29项source文件外，仅新增以下两项：

|文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_dssc_zdom_jg_qknn_r4_bcrr.py`|`d5eddebb960b764647142570650e737e8e9ae08a1c8facd01feefe81426b3c539`|
|`code/cvsrffi/stage2_svrn_bcr.py`|`9b909cc88557f2d29be547ac9512211456d226dc2add99a8444ec6d30b6b6058`|

r3 source固定为31项hash闭包，明确排除整个项目树和所有`stage2_next_r2_*`文件。

## r3N607硬门

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r3`，首次创建且不可覆盖。
- directN607预检、逐文件hash、远端compile、Torch 2.1 extractor CPU load、唯一prepare、真实checkpoint row0/clear no-query smoke必须依次通过；任一失败即停止，不启动shard。
- smoke通过后固定8个shard，GPU`i`使用`CUDA_VISIBLE_DEVICES=i`，CLI统一`--device cuda:0 --shard-index i --feature-batch-size 64`；只有完整125/375/750闭包后才merge、validate、truth-open和score。
- 不因性能值停止；只按P0或两个不同outer row相同确定性故障停止本run；不重试、不调参、不从r1/r2partial产生性能结论。

## 本地证据与交接

- D138全套本地回归：`36 passed`；`py_compile`和`git diff --check`通过。
- r1远端Torch 2.1 load/compile通过但prepare失败；r2远端Torch 2.1 load/compile通过但prepare失败；两次均无GPU实验进程。
- r3完成后由主agent只分析完整同row truth-side结果；否则状态保持`NO_PERFORMANCE_RESULT`。

## N607唯一runner终态

- 执行角色：唯一N607 release runner；retry authority：`false`；未创建r4，未覆盖r3、r1或r2。
- 直接N607预检：首次通过；继续执行时直连TCP/22拒绝，按既有规则使用已验证lab bridge。普通N607账户为`szu2070436088`，未使用N607管理员账户。
- source闭包：31项逐项SHA匹配，`stage2_next_r2`数量为0；extractor SHA256=`56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`，大小`4618957`字节。
- compile：通过；CWD为`RUN_ROOT/source`，Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Torch=`2.1.0+cu121`。
- load：通过；CPU`torch.jit.load`、forward schema、`Tuple[Tensor,Tensor]`图输出均已记录。
- prepare：按冻结参数仅执行一次，退出码`1`。`control/prepare.out`记录：`FileExistsError: immutable prepare output already exists`，目标为`RUN_ROOT/prepared`。由于该目录按预注册步骤已预创建，launcher的不可变输出门拒绝继续；未删除目录、未覆盖文件、未重试。
- prepare之后阶段：未启动smoke、8 shard、merge、validate、build-truth或score；无plan/context、prediction、truth或score结果，状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 远端进程证据：prepare退出后无`run_d92_pr160_target125.py`进程；8张GPU均空闲；未产生PID或shard manifest。
- 本地回收：control证据位于`E:\type10-7\automation_reports\CV-SincNet\d138_d92_lite_pr160_target125_20260804_r3\artifacts\control\`，包括`source_hashes.out`、`compile.out`、`load.out`、`prepare_command.txt`、`prepare_start.txt`、`prepare.out`和`prepare_exit.txt`。本地`ssh.exe`、`scp.exe`及到`172.31.111.215:22`、`172.31.105.18:22`的连接均为0。

## r3远端终态证据（2026-08-04）

|阶段|结果|证据|
|---|---|---|
|direct N607预检|首次通过；后续直连出现`Connection refused`，桥接只读核验通过|本地预检输出；桥接主机`dell-DSS8440`|
|source闭包|通过|`source_hashes.out`：31项、`source_manifest_match=true`、`forbidden_stage2_next=0`|
|远端compile|通过|`control/compile.out`|
|Torch 2.1 extractor load|通过|`control/load.out`，`Torch 2.1.0+cu121`|
|唯一prepare|失败|`control/prepare_exit.txt`：`exit=1`|
|失败原因|run-owned空目录冲突|`FileExistsError: immutable prepare output already exists: .../r3/prepared`；该目录仅存在目录项，无文件|
|真实checkpoint smoke|未启动|无smoke receipt、无GPU进程|
|8 shard/125 outer|未启动|无shard日志、预测、score或merge产物|

远端run root为`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r3`。该目录及空的`prepared`目录保留，不删除、不覆盖；本次没有性能结果，也不从r1/r2/r3部分控制证据推导性能结论。SSH/SCP进程及到N607与桥接主机的22端口连接已清理。

## 结果表

|run/candidate|机制|矩阵|K/seed|性能结果|最终判定|
|---|---|---|---|---|---|
|`d138_d92_lite_pr160_target125_20260804_r3` / `D92-Lite-PR160/r1`|D138 PR160单运输臂|125 outer；375 scene；750 surface|冻结配置；完整矩阵未启动|未产生|`NO_PERFORMANCE_RESULT`|
