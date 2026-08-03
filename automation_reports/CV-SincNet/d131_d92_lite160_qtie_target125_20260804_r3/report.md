# D131 D92-Lite160-QTIE Target125 r3实验报告

## 状态

- 实验ID：`d131_d92_lite160_qtie_target125_20260804_r3`
- 日期：2026-08-04
- 操作者：主agent负责科学集成、结果分析与晋级决定；Luna/max为唯一N607 runner。
- 当前状态：`LOCAL_VERIFICATION_IN_PROGRESS`
- 协议：`p2_min_v1`
- 目标：修复D131 r2在Lite最高分精确并列处的确定性执行缺陷，完成冻结的125 outer/375 scene/750 prediction surface矩阵。
- 比较对象：同一row的before qKNN与after D92-Lite160-QTIE；r2严格为`NO_PERFORMANCE_RESULT`，281个partial仅作执行故障证据。

## 假设与唯一科学变更

- K5/K10的after Lite logits若存在逐query精确最高分并列，仅在该Lite top集合内使用同一after support qKNN作二级判决。
- qKNN唯一胜者的Lite logit只向正无穷方向增加一个float32 ULP；非并列query、非top class及其他cell逐字节不变。
- 若qKNN在Lite top集合内仍精确并列，继续fail-closed；禁止按registry顺序、类别ID/hash或argmax首项裁决。
- before与K1保持精确qKNN路径；不改Lite拟合、INT8/FP16 wire、representation、矩阵、K、receiver、seed、scene或GPU分片。
- 该规则逐query独立、类标签置换等变；不读取query truth/role/quota、跨query状态或partial性能。

## 冻结矩阵

- 接收机：`20-1,3-19,7-14,7-7,8-8`
- 种子：`713102,713103,713104,713105,713106`
- 切片：`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`
- 场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- 覆盖：125个outer row、375个scene row、375个arm pair、750个before/after prediction surface、8个固定modulo shard。

## 本地版本与验证

- Git工作树：`E:\type10-7\code\snapshots\d92_lite125_20260804_wt`；分支`codex/d92-lite125-20260804`。
- 变更文件：`code/cvsrffi/stage2_d92_lite_target125_core.py`、`code/cvsrffi/stage2_d92_lite_target125.py`、`tests/test_stage2_d92_lite_target125_core.py`、`tests/test_stage2_d92_lite_target125.py`、`configs/d131_d92_lite160_qtie_target125_r2.json`、`.gitattributes`。
- 方法锁schema：`cvs.phase2.d131.d92_lite160_qtie_target125.method_lock.v2`。
- 方法锁SHA256：`e0f7f8623b4d53002206aca8575f8eadd2bca4150a7c5aed3d017b4827fa5dac`。
- 已通过：D92/D108/D129相关28项测试。
- 待记录：独立P0/P1复核、最终Git commit、确定性runtime archive与解包哈希。

## N607输入与发布冻结

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/runs/d131_d92_lite160_qtie_target125_20260804_r3/source`
- 远端run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d131_d92_lite160_qtie_target125_20260804_r3`，发布前必须确认不存在。
- D92 matrix：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/matrix_manifest.json`，SHA=`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`。
- D92 output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`。
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`，SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- D108 method lock：`/home/szu2070436088/2510044040/CV-SincNet/runs/d108_cbrrc_smme_target125_20260801_r3/source/configs/stage2_d108_cbrrc_smme_r1.json`，SHA=`7e8b310eeffc5e56aa39d60ef3b66c652207c3d9c1004e04d4499e6073862845`。
- ground：`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`，manifest SHA=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`。
- GPU：shard i使用`CUDA_VISIBLE_DEVICES=i`，CLI统一`--device cuda:0`；每卡最多本run一个进程。
- 顺序：prepare→真实checkpoint no-query smoke→8 shard→merge→validate→truth-open→score。
- 预期artifact：plan/context/prepare receipt、smoke receipt/predictions、8 shard manifests、prediction manifest、truth catalog/open event、score manifest、完整日志/PID/GPU/hash/SSH cleanup证据。

## 健康停止与结果边界

- 只在P0协议/安全违规、wrong hash/checkout、覆盖风险、launcher级确定性故障，或至少两个不同outer row在prediction前出现同一确定性异常指纹时停止。
- 不因accuracy、H、BA、floor或任何中间性能停止；不从partial结果选择receiver、seed、K、scene或方法。
- 只有完整125/375/750闭包且truth-side score匹配后才产生性能结论。
- 若完整结果为负，关闭D92-Lite路线，不追加调参矩阵。

## 完成后分析表

| candidate | receiver/TX | K/new | seed | scene | before old | after old | seen-new | H_old_new | forgetting | verdict |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| 待完整truth score | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

