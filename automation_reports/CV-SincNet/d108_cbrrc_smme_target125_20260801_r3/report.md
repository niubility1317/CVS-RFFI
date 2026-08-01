# D108-CB-RRC-SMME/r3完整125实验报告

状态：`STOPPED_BY_USER_SCOPE_REDUCTION / NO_PERFORMANCE_RESULT`

## 实验登记

|字段|值|
|---|---|
|experiment ID|`d108_cbrrc_smme_target125_20260801_r3`|
|日期/operator|2026-08-01；主agent负责集成、数据与结果分析；Terra Max唯一运行子agent负责N607|
|目标|保持D108方法和r2源码完全不变，只修正多GPU可见命名空间后完成完整125|
|比较目标|完整125的D62、D92、SVRN-qKNN-BCRR；D91仅列15行development证据|
|claim scope|`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`|
|Git|本地分支`codex/stage2-da25-r1`；只提交，不push、不上传GitHub|

r1因sklearn版本硬锁在零prediction处技术退出，r2已用严格1.7.0/1.7.2兼容修复通过真实smoke，且M0 before120条、after220条query ID和predicted handle与历史D92参考逐项一致。r2随后把8个进程分别传入`--device cuda:0..7`，但sealed TorchScript权重保持`cuda:0`，导致shard1—7输入/权重设备冲突并在零prediction处退出。r3只修正启动映射，不是方法或性能重试。

## 冻结源码、方法与资产

源码commit=`047223fde7a77c80fd3fab74f3bf459ee9eacbea`；archive=`E:\type10-7\code\snapshots\d108_cbrrc_smme_target125_20260801_r2_source_047223fd.tar`，SHA256=`1028850a90c5fbbb91f4c661d09060ba03b1430c0258f1dc515d6017fc4ce54a`。本地全部D108联合回归=`56 passed`，兼容修复独立复核`P0=0,P1=0`。四臂仍为`M0=D92`、`M_DA=CB-RRC＋D92`、`M_HEAD=D92＋SMME`、`M_JOINT=CB-RRC＋D92＋SMME`；完整矩阵仍为125outer、375scene、1500arm pair、3000prediction surface。D92 matrix、checkpoint、D19 ground和method lock SHA分别为`b70045e7…e4bc5c`、`2699eedc…d59c98`、`15b5e144…94629c`和`7e8b310e…62845`；不使用RDCE，不重验数据。

## 唯一启动修复与N607预登记

远端run root固定为`/home/szu2070436088/2510044040/CV-SincNet/runs/d108_cbrrc_smme_target125_20260801_r3`，必须首次创建且不可覆盖。每个shard的物理GPU绑定固定为：进程i设置`CUDA_VISIBLE_DEVICES=i`，CLI统一传`--device cuda:0 --shard-index i`。因此8个进程各自只看到一张物理卡，并把sealed模型与输入共同放在其局部`cuda:0`；严禁再传`cuda:i`。每个wrapper启动后必须核对环境中的`CUDA_VISIBLE_DEVICES`、Python cmdline、局部device、物理GPU PID和shard index一一对应。

执行链：fresh direct preflight→archive/锁/编译核验→prepare→沿用r2已证明的row0/clear真实smoke和M0 parity口径做一次r3落地健康smoke→立即启动8个固定shard→严格merge/validate 125/3000→prediction封存后build-truth/score→artifact回收与GPU/SSH清理。smoke通过后不得增加gate。停止仅允许P0协议/安全故障，或至少两个不同outer row在prediction前出现相同确定性异常指纹；不得按性能值停止、调参、重启或选择性补跑。

期望artifact：`prepared/target125_plan.json`、`prepared/target125_context.json`、`smoke/smoke_receipt.json`、`smoke/smoke_predictions.json`、8个shard manifest、`predictions/prediction_manifest.json`、`truth_catalog.json`、`score/score_manifest.json`及完整日志/PID/exit/GPU映射。完成后按125个outer-row均值报告before old、after old、before/after floor、seen-new、H、forgetting和全量post correct，并和D62、D92、SVRN同口径对比；D91单列development。

## 用户缩减矩阵后的收尾

2026-08-01用户明确要求性能验证只运行必要、重要的矩阵，以提升研发效率。最终目标本身只要求单seed Target25，因此五seed完整125不再是首轮必要证据。r3停止原因是用户改变实验范围，不是性能值、协议故障或执行异常；partial prediction不得评分或用于方法判断。

停止前唯一runner逐项确认8组wrapper/Python的run root、CWD、cmdline、`CUDA_VISIBLE_DEVICES`、shard和局部device绑定，`BINDING_OK=1`。随后仅向已确认属于本run的8个Python子PID发送`SIGTERM`；3秒内8个Python和8个wrapper全部退出，8个exit均为143，无需升级信号。停止时partial prediction共968个：S0=120、S1=120、S2=128、S3=120、S4=112、S5=120、S6=120、S7=128；异常指纹为0，8个shard manifest均未形成，`prediction_manifest/truth_catalog/score_manifest`均不存在，truth从未打开。

最终run-owned存活PID为0；GPU0—7均为0%利用率、1MiB占用且compute app为0；本地`ssh.exe=0`，TCP22连接为0。远端partial artifact原位保留，未删除、移动或覆盖。本地回收prepared、smoke、logs和remote-control证据至`artifacts/remote_r1`；`stop_summary.txt`SHA256=`23221279b874311785aa730a351269abd9d520885f6390c86cadee45ca60c3a2`。

本run永久终态为`STOPPED_BY_USER_SCOPE_REDUCTION / NO_PERFORMANCE_RESULT`。下一实验复用同一D108方法、commit和冻结输入，改为`seed=713102`的5receiver×5slice Target25，共25个outer、75个scene、300个arm pair、600个prediction surface；只有该完整新run才允许进入性能分析。
