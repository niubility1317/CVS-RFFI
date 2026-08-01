# D108-CB-RRC-SMME单seed Target25实验报告

状态：`PREREGISTERED / NOT_LANDED / NO_PERFORMANCE_RESULT`

## 实验登记

|字段|值|
|---|---|
|experiment ID|`d108_cbrrc_smme_target25_s713102_20260801_r1`|
|日期/operator|2026-08-01；主agent负责目标、协议和结果决策；Terra Max唯一runner负责N607落地、启动、监控、评分和回收|
|目标|用最小关键矩阵判断D108是否达到当前单seed Target25目标，避免五seed完整125的非必要计算|
|候选|`D108-CB-RRC-SMME/r1`，方法与已冻结D108完全相同|
|比较|`M0=D92`、`M_DA=CB-RRC＋D92`、`M_HEAD=D92＋SMME`、`M_JOINT=CB-RRC＋D92＋SMME`|
|证据边界|完整25-job单seed真实性能；达到目标后才考虑多seed确认|
|历史parent|`d108_cbrrc_smme_target125_20260801_r3`按用户范围缩减停止于968/3000，未开truth，永久`NO_PERFORMANCE_RESULT`|

## 假设与完整关键矩阵

D108假设CB-RRC能利用合法ground压缩知识和当前row旧类support改善接收机域偏移，SMME能在全注册类统一竞争中缓解注册后旧类／新类margin失衡；联合臂应同时提升旧类、new和floor，而不是只迁移错误。

固定矩阵：

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed=713102
slice={K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}
scene={leo_clear_weak,leo_low_elev_weak,leo_rain_weak}
arm={M0,M_DA,M_HEAD,M_JOINT}
phase={before,after}
```

闭包为25 outer、75 scene、300 arm pair、600 prediction surface。seed713102早于D108结果预登记，不是依据性能选择。不得删除任何receiver、slice或scene，不得按中间准确率停止。

## 性能目标

|切片|必须满足|
|---|---|
|K10/new5|`old_acc_after_increment≥92%`、`min_old_class_acc≥85%`、`seen_new_acc≥92%`|
|K10/new10|`old_acc_after_increment≥92%`、`min_old_class_acc≥85%`、`seen_new_acc≥90%`|
|K10/new20|`old_acc_after_increment≥92%`、`min_old_class_acc≥85%`、`seen_new_acc≥86%`|
|K5/new20|相对matched K10/new20的`A_old/F_old/N/H`下降均≤5pp|
|K1/new20|相对同row D92：`ΔH≥2pp、ΔF_old≥2pp、ΔA_old≥0、ΔN≥0`且old＋new总正确数严格增加|

结果必须按同一row同时报告receiver、slice、scene、arm、before/after old、old floor、seen-new、H、forgetting和correct count。D62、D92、SVRN只复用历史同row基线；D91仅列development证据。

## 本地版本与独立复审

|项目|值|
|---|---|
|Git分支|`codex/stage2-da25-r1`|
|实现commit|`437cb5a2fdc63db621743593fb0d0202e5cfeafe`|
|source archive|`E:\type10-7\code\snapshots\d108_cbrrc_smme_target25_s713102_20260801_r1_source_437cb5a2.tar`|
|archive SHA256|`d3f640108b796e3bc85856f6cca887f46dabe60a6bced99c6b86b74630a0e076`|
|Target25 module SHA256|`e7e107b5035f634132efc1102a6cd605629a6ac61fee9f0cfadf5f1f9b03c902`|
|CLI SHA256|`e858ee862eca26009bbfb51176ca87824ed6c29733e14ccc8242f36f597fd783`|
|独立review|`P0=0、P1=0 / RELEASE`；确认25-job笛卡尔积、D92／四臂逐值路径不变、query零fit/update/selection、8-shard不可覆盖、truth时序和异常恢复|
|本地验证|`ssr-gpu`中Target25＋D108 core／runner／truth聚焦回归24项通过；三个新文件`py_compile`通过|

根目录`E:\type10-7`不是Git仓库；本报告同时镜像到Git工作树并单独提交，不push、不上传GitHub。

## 冻结N607输入

|输入|路径／SHA256|
|---|---|
|D92 matrix manifest|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/matrix_manifest.json`；`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|D92 output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`；`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|ground component|`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`；manifest`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`|
|method lock|`<run-root>/source/configs/stage2_d108_cbrrc_smme_r1.json`；`7e8b310eeffc5e56aa39d60ef3b66c652207c3d9c1004e04d4499e6073862845`|

不重验`VALIDATED_ONCE`数据，不修改received-IQ、physical ID、receiver/TX、scenario、K或support/query split。

## N607路径、环境与命令

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d108_cbrrc_smme_target25_s713102_20260801_r1`
- CWD：`<run-root>/source`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 日志：`<run-root>/logs/`
- PID／exit：`<run-root>/remote_control/`
- expected：`prepared/target125_plan.json`、`prepared/target125_context.json`、`smoke/*`、8个shard manifest、`predictions/prediction_manifest.json`、`truth_catalog.json`、`score/score_manifest.json`

prepare命令：

```text
python code/scripts/run_d108_target25.py prepare --d92-matrix-manifest <D92-manifest> --d92-matrix-manifest-sha256 b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c --d92-output-root <D92-root> --checkpoint <checkpoint> --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --d108-method-lock configs/stage2_d108_cbrrc_smme_r1.json --d108-method-lock-sha256 7e8b310eeffc5e56aa39d60ef3b66c652207c3d9c1004e04d4499e6073862845 --ground-component-dir <ground-dir> --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c --output-dir <run-root>/prepared
```

smoke固定row0／scene0／GPU0，真实checkpoint、无truth；通过后立即启动8shard。每个物理GPU i使用：

```text
CUDA_VISIBLE_DEVICES=i python code/scripts/run_d108_target25.py predict-shard --plan-manifest <prepared-plan> --plan-manifest-sha256 <sha> --context-manifest <prepared-context> --context-manifest-sha256 <sha> --output-dir <run-root>/shards/shard_i --shard-index i --device cuda:0 --feature-batch-size 64
```

8shard全部退出0后严格merge和validate；prediction完整封存后才允许build-truth和score。唯一runner在实际执行时把所有占位符替换为本报告冻结路径并把完整命令、PID、GPU映射和SHA回填。

## 停止条件与成功条件

只在P0协议／安全问题、错误source／SHA／不可覆盖风险，或至少两个不同outer在prediction前产生同一确定性异常指纹时停止精确run-owned进程树。不得因accuracy、H、floor或任何中间性能值停止、选row、调参或重启。

技术完成要求：25/25 outer、75/75 scene、300/300 arm pair、600/600 prediction surface、8/8 shard、prediction manifest、150 truth surface、300 scene-arm score row、100 outer-arm聚合行全部闭合；truth在prediction封存后开启；异常为0。

性能完成要求：完整读取score和artifact，按本报告目标逐项判定。若未达目标，D108直接判弱并切换已release-ready的D109 Target25，不追加seed、不修D108阈值、不恢复完整125。

## 风险与完成后检查

- 主要性能风险：CB-RRC可能只产生小幅旧类改善；SMME可能伤害新类或floor；K1可能改变分数但没有联合收益。
- 主要执行风险：多GPU必须使用`CUDA_VISIBLE_DEVICES=i`配合统一`--device cuda:0`；严禁再次传`cuda:i`。
- 完成后回收完整prediction、score、truth-open event、shard manifests、日志、PID／exit、资源和SHA清单。
- 当前尚无D108 Target25性能结果；代码、测试和release状态不得写成性能正收益。

## 结果表（待实验完成）

|receiver|slice|arm|B-old|A-old|F-old|seen-new|H|forgetting|correct/query|判定|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|待运行||||||||||`NOT_RUN`|
