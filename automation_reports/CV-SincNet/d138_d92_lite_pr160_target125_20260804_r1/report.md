# D138 D92-Lite-PR160 Target125实验报告

## 状态

- 实验ID：`d138_d92_lite_pr160_target125_20260804_r1`
- 日期：2026-08-04
- 操作者：主agent负责科学集成、协议解释、结果分析与晋级决定；唯一N607 runner负责落地、启动、监控和回收。
- 当前状态：`LOCAL_VERIFIED / REVIEW_P0=0 / LOCAL_P1_CLOSED / REMOTE_TORCH2.1_SMOKE_PENDING / NO_TARGET_PERFORMANCE_RESULT`
- 目标：在用户明确要求下登记一个独立D138修复候选，修复D131的截断表示和K1精确并列执行缺陷，并尝试完成冻结的D92 125矩阵。
- 比较对象：同一outer row、同一scene、同一物理query集合的before锁定qKNN与after D92-Lite-PR160；D131历史partial仅作技术故障证据，不进入本次结果。

## 假设与冻结的唯一方法变更

- 从同一次sealed TorchScript前向读取`model.id_backbone.cls_head.joint_proj.0`的160维线性输出，保留pre-ReLU信息；不增加backbone前向。
- 使用冻结的signed totalization：正ReLU部分有正范数时使用其单位方向；全负有限行使用原始signed方向；精确零或非有限行fail-closed。
- before与K1 after保持精确qKNN路径；K5/K10 after使用所有注册类共享的对角OAS affine head，不做old/new分裂。
- support拟合只读support；query逐行独立、只读、零fit、零update、零selection；最终float32最高分精确并列统一fail-closed，不使用注册顺序、类别hash、argmax首项或跨query回退。
- 本候选不宣称历史D92 288维head等价；只有完整125 outer/375 scene/750 surface、独立truth-side score闭包后才可产生Target125性能结论。

## DA/注册状态映射

D138是单一`M_JOINT`运输臂的D92-Lite独立系统诊断，不执行或声称D108 joint domain-adaptation mechanism。因此本候选只测以下两个状态：

|状态|D138对应|是否测量|解释|
|---|---|---:|---|
|`DA0_REG0`|`before`|是|无独立DA干预、旧注册类、锁定qKNN|
|`DA0_REG1`|`after`|是|无独立DA干预、旧类加新类注册、PR160 head|
|`DA1_REG0`|无|否|单运输臂候选范围外|
|`DA1_REG1`|无|否|单运输臂候选范围外|

因此结果只能解释为同一`DA0`条件下的注册前后对照，不能填写四状态DA因果表，也不把`M_JOINT`名称误读为已执行DA机制。

## 冻结矩阵与协议

- 协议：`p2_min_v1`；LEO弱接收观测；support/query物理ID不相交；不访问clean/source、query truth、query role、class quota、true batch count或global reassignment。
- 接收机：`20-1,3-19,7-14,7-7,8-8`。
- 种子：`713102,713103,713104,713105,713106`。
- 切片：`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`。
- 场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- 覆盖：125个outer row、375个scene row、375个arm pair、750个before/after prediction surface、8个固定modulo shard。
- 数据复用依据：received-IQ bytes、physical IDs、receiver/TX、scenario、K、support/query split和`p2_min_v1`均不变，沿用已`VALIDATED_ONCE`的D92 sealed packages；不重复构建数据。

## 本地版本与验证

- Git工作树：`E:\type10-7\code\snapshots\d92_lite125_20260804_wt`；分支：`codex/next-r1-fabr-tsl-20260804`。
- 原有无关用户改动：`code/cvsrffi/stage2_next_r2_bssdg.py`、`code/cvsrffi/stage2_next_r2_cvfr.py`及对应测试未修改、未暂存。
- D138候选文件：`stage2_d92_pr160_core.py`、`stage2_d92_pr160_runtime.py`、`stage2_d92_pr160_target125.py`、`run_d92_pr160_target125.py`、`build_d92_pr160_extractor_runtime.py`、`d138_d92_lite_pr160_r1.json`、对应测试，以及D108 runner的160维类型扩展。
- 方法锁schema：`cvs.phase2.d138.d92_lite_pr160.method_lock.v1`。
- 候选：`D92-Lite-PR160/r1`。
- 方法锁SHA256：`019dd59780de735af3026b091ef88b600c07d75c48f96aad0c2de34d49e8cee7`。
- source sealed runtime SHA256：`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`。
- graph-derived PR160 extractor SHA256：`56612c66b49c8167b3fbed0be5aaa25649a3246a178903618274048d541d80a3`，大小`4618957`字节；artifact不进Git，按独立输入同步并验hash。
- 本地验证环境：Conda`ssr-gpu`。
- 已通过：D138 core、D108 typed-width、D138 PR160 adapter真实typed smoke与8-shard/750-surface merge、D92历史adapter、D108 runner、D108 Target125、D108 truth scorer共36项测试；结果`36 passed`。
- 已通过：新旧模块`py_compile`、`git diff --check`、本地TorchScript graph tap/hash检查、候选导入检查。
- 本地build依据：source runtime未被修改；derived extractor由`code/scripts/build_d92_pr160_extractor_runtime.py`生成。

## N607预注册路径与命令

- 访问：先执行`tools\\n607_ssh_preflight.ps1`，只允许direct `N607`；确认本run根目录不存在、GPU/磁盘可用后再同步。
- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`。
- 远端run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r1`，必须首次创建且不可覆盖。
- source CWD：`RUN_ROOT/source`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 输入：`RUN_ROOT/input/d92_pr160_extractor_runtime.pt`。
- 既有D92 matrix：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/matrix_manifest.json`，SHA=`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`。
- 既有checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`，SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- 既有D108 method lock：`/home/szu2070436088/2510044040/CV-SincNet/runs/d108_cbrrc_smme_target125_20260801_r3/source/configs/stage2_d108_cbrrc_smme_r1.json`，SHA=`7e8b310eeffc5e56aa39d60ef3b66c652207c3d9c1004e04d4499e6073862845`。
- 既有ground component：`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`，manifest SHA=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`。

执行顺序固定为：

1. 同步本地已验证文件，远端逐文件hash和compile检查；先检查N607的Torch版本是否能加载已绑定的extractor，失败则停止，不远端改代码、不强行启动。
2. `prepare`写入`RUN_ROOT/prepared`，生成plan/context后记录其实际文件SHA。
3. 使用真实checkpoint执行一次row0/clear的no-query smoke，必须同时验证两phase、query truth/fit/update/selection全为false、`truth_open=false`。
4. 在GPU`i`上以`CUDA_VISIBLE_DEVICES=i`运行固定`--device cuda:0 --shard-index i --feature-batch-size 64`的8个shard；每卡只允许本run一个进程。
5. 8个shard完整成功后才允许merge→validate→truth-open→score；任何partial不读性能、不替代完整矩阵。

## 健康停止与成功标准

- 停止条件：P0协议/安全违规、wrong hash/checkout、覆盖风险、extractor加载失败，或至少两个不同outer row在prediction前出现同一确定性异常指纹；只停止本run已核实的进程树。
- 不因accuracy、H、BA、floor或任何中间性能停止；不调参、不选择性补跑、不续用D131 partial。
- 成功标准：plan/context、smoke、8 shard、125/375/750闭包、独立truth catalog/open event、validate和score全部存在并通过hash/receipt核验。
- 性能结论边界：只有完整同row before/after score可用于解释；否则状态为`NO_PERFORMANCE_RESULT`。
- 若完整矩阵为负结果，本次候选按method lock关闭，不追加参数调优矩阵。

## 预期产物与完成后分析

- `prepared/target125_plan.json`、`prepared/target125_context.json`、prepare receipt。
- `smoke/real_checkpoint/`下的smoke receipt和smoke predictions。
- `shards/shard_0..7/`及各自manifest；`predictions/prediction_manifest.json`；`truth_catalog.json`；`score/score_manifest.json`；control/log/PID/GPU/hash/SSH清理证据。
- 完成后按同一candidate/run/receiver/TX/K/seed/scene行记录before old、after old、seen-new、`H_old_new`、forgetting、coverage、defer/rollback字段和最终判定；不报告跨row孤立最大值。

## 同步映射

详见`code/SYNC_MANIFEST.txt`中的D138条目；所有代码/config/test/report先在本地Git工作树验证和提交，再由唯一runner使用SCP同步到本run的`source`，extractor作为独立输入同步到`input`。

## 复核与版本状态

- 当前阶段：独立复核确认`P0=0`，新增typed adapter测试和hash/投影锁修正已闭合本地调用链P1；远端Torch 2.1加载和真实checkpoint no-query smoke仍是释放前硬门。
- Git commit：待独立复核通过后创建；不push、不上传、不覆盖历史D131 run。
- 远端结果：待唯一runner执行；在本报告更新前不宣称Target125性能结果。
