# D92-E0D-5arm-Hard12-v2发布修复实验报告

## 1.基本信息

|字段|内容|
|---|---|
|实验ID|`D92-E0D-5arm-Hard12-v2`|
|run ID|`d92_e0d_5arm_hard12v2_20260811_v2`|
|日期|2026-08-11|
|operator|Codex primary；N607唯一runner|
|当前状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|
|协议|`p2_min_v1`|
|证据范围|`DEVELOPMENT_ONLY_PSEUDO_BLIND_DISJOINT_STRESS_SCREEN`|
|唯一晋级候选|`E0_FULL_ONLY`|
|前序run|v1在prepare前因method-lock远端目录错误退出，`NO_PERFORMANCE_RESULT`|

## 2.目标与冻结科学规格

本run只修复交付路径，不改变v1的科学设计。目标仍是在固定288维A、B开启、task-balanced C、F0查询头和E关闭的路径内，比较D92_FULL、E0_FUSION、E0_FULL_ONLY、E0_BLOCK_ONLY、E0_FIXED50，验证删除K折LOO融合能否同时提高`H_old_new`并降低注册计算。

- selection SHA256：`2e3b3333a4a325bd0443a31065d3340d6a650a3e89620951a786637e6bce8d3a`；与Hard12-v1的outer交集为0。
- 12个outer、5臂、60个job；每个outer固定clear/low_elev/rain，共180个scene-arm。
- 10个K5/K10 outer进入性能门；2个K1 outer只检验strict alias和liveness。
- support/query物理ID与`p2_min_v1`已冻结；query逐样本在全部注册类上竞争，禁止truth、role、quota、fit、selection、update和global reassignment。
- 仅`E0_FULL_ONLY`可晋级；Hard12-v2通过后也只能进入完整Target125确认。

## 3.严格门

- 60/60 job、8/8 shard和全部prediction/score/audit闭合；五臂`DA0_REG0`状态与预测精确一致，K1的`DA0_REG1`严格别名一致。
- FULL_ONLY相对E0_FUSION：mean ΔH>0，10个performance outer至少8个ΔH非负；注册wall中位至少下降40%，增量peak不增加。
- FULL_ONLY相对D92_FULL：mean ΔH≥0.005（即0.5个百分点），至少8/10非负；old-balanced、old-floor、seen-new均不下降，forgetting不增加；注册wall中位至少下降60%。
- K5/K10 after two-state组件fit计数：FULL=48/88、FUSION=24/44、FULL_ONLY=2/2、BLOCK_ONLY=2/2、FIXED50=4/4；query MAC精确一致。

## 4.发布修复与本地验证

唯一修复是让launch读取同步到`source_root`根目录的冻结method lock，并把所有run路径改为全新v2。新增`require_file`失败信息，避免再次出现0B静默退出；方法代码、config内容、runtime archive、selection、矩阵和门限均未改变。

|交付物|本地路径与冻结身份|
|---|---|
|runtime archive|`E:\type10-7\code\snapshots\d92_e0d_runtime_closure_7d11a701.tar.gz`；3519772B；SHA256=`36fc9df5e174ecd87863dcb6663afb6875d5f07ca6d17282648adfa38a7f32df`|
|method lock|`configs/stage2_d92_e0d_5arm_hard12v2_v1.json`；2177B；SHA256=`b80f967e1fc070a730a7b193f691036339930af022682fe2fca81c2e4d229f86`|
|launch|`automation_reports/CV-SincNet/d92_e0d_5arm_hard12v2_20260811_v2/launch.sh`；3519B；SHA256=`a552e35c5ba25910d6b16a999cbead459e9fa5b8062fb447d24a159888ac1391`|
|代码版本|release repair commit=`0d8039b3`；release base=`3217a88a00e6ee36c46e48fc158c76fb6e4acb96`；runtime source commit=`7d11a7012ab62058db40f878f925c38160311311`|

本轮不重复数据验证、不做整树SHA或额外签名。`ssr-gpu`环境中`bash -n`、5项runner聚焦回归和`git diff --check`均已通过；独立复审确认P0=0、P1=0并给出`APPROVE_RELEASE`。

## 5.N607预注册

|项目|冻结值|
|---|---|
|python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|project|`/home/szu2070436088/2510044040/CV-SincNet`|
|source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_source_snapshot_20260811_v2`|
|working directory|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_source_snapshot_20260811_v2/code`|
|context|`/home/szu2070436088/2510044040/CV-SincNet/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json`|
|smoke|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_truthfree_smoke_20260811_v2`|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_5arm_hard12v2_20260811_v2`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0d_5arm_hard12v2_20260811_v2`|
|GPU|8个shard固定映射GPU0–7，每卡一个本run进程，进程内使用`cuda:0`|
|expected artifacts|60 job receipt、120 COMMIT、120 prediction artifact、60 score、180 fit audit、180 resource audit、8 shard summary|

同步映射固定为：

|本地文件|N607目标|
|---|---|
|`E:\type10-7\code\snapshots\d92_e0d_runtime_closure_7d11a701.tar.gz`|`runs/d92_e0d_source_snapshot_20260811_v2/d92_e0d_runtime_closure_7d11a701.tar.gz`|
|`configs/stage2_d92_e0d_5arm_hard12v2_v1.json`|`runs/d92_e0d_source_snapshot_20260811_v2/stage2_d92_e0d_5arm_hard12v2_v1.json`|
|本报告同目录`launch.sh`|`runs/d92_e0d_source_snapshot_20260811_v2/launch.sh`|

精确远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0d_source_snapshot_20260811_v2 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

runner须先执行只读preflight，确认四个v2路径不存在、GPU/进程容量安全，再同步三项。launch依次核对交付物、解包、import闭包、prepare 60任务、GPU0真实sealed-checkpoint truth-free smoke，再启动8个shard。

## 6.健康停止与成功判据

只在P0协议/安全违规、错误闭包/路径、输出覆盖、query泄漏、共享stop marker，或至少两个不同outer在prediction前出现同一确定性异常指纹时停止本run。绝不按H、accuracy或其他性能中间值停止。停止前必须绑定本run PID/CWD/cmdline，只终止本run进程树并保留全部partial artifacts；本run不授权fresh retry。

技术成功要求真实checkpoint smoke PASS、60/60 job、8/8 shard PASS、failed=0、异常指纹为空、最终GPU和SSH连接释放。只有完整artifact取回后才运行冻结分析器。

## 7.结果区

2026-08-11，v2在全新路径完成三项同步、运行闭包import和prepare；冻结manifest闭合为12个outer、60个job、180个scene-arm，SHA256=`99c25f000e64d6dc60fe13b74028ce33505fdccb404d658c518baaa37e3a6dc7`。真实checkpoint truth-free smoke固定执行`rx_7_7__seed_713104__k_1__new_20`的`D92_FULL`时，prediction在通用D81资源汇总读取`before_center_shift_l2_max`处触发`KeyError`。E0D审计行遗漏了该既有兼容字段。

driver在任何shard启动前退出；prediction stdout为0B，没有COMMIT、receipt、score或性能结果，8张GPU保持空闲，run-owned进程和SSH连接最终均为0。远端四个v2路径与partial artifacts原样保留；证据取回到`E:\type10-7\local_artifacts\d92_e0d_5arm_hard12v2_20260811_v2`。

因此v2禁止性能分析。修复范围仅为恢复E0D fit-audit与既有D81 evaluator的字段契约，不改变预测状态、方法、矩阵、门限或输入；后续必须使用全新run ID。
