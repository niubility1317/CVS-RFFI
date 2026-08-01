# D109-SCRC单seed Target25实验报告

状态：`LOCAL_VERIFIED / THEORY_REVIEW_DEFERRED / NOT_LAUNCHED`

2026-08-02方向修订：根据用户最新要求，停止D109实验发布，先从统计可辨识性、接收机／信道物理模型和统一old/new决策原理研发轻量快速域适应方法。D109代码仅保留为已验证的比较实现；在新的理论候选完成交叉复审前，不连接N607、不启动Target25、不追加参数扫描。

## 实验登记

|字段|值|
|---|---|
|experiment ID|`d109_scrc_target25_s713102_20260802_r1`|
|日期／operator|2026-08-02；主agent负责目标、协议和最终性能决策；Terra Max唯一runner负责N607落地、启动、监控、评分和回收|
|目标|用最小必要矩阵判断D109是否产生重要、联合正收益；避免先跑五seed完整125|
|候选|`D109-SCRC/r1`|
|四臂|`M0=D92`、`M_DA=CB-RRC＋D92`、`M_HEAD=D92＋SCRC`、`M_JOINT=CB-RRC＋D92＋SCRC`|
|比较目标|同row的M0／M_DA，以及历史D92；不得用跨row单项最大值拼接结论|
|证据边界|单seed Target25真实性能；只有单seed达标后才考虑多seed确认|

## 最小必要矩阵

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed=713102
slice={K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}
scene={leo_clear_weak,leo_low_elev_weak,leo_rain_weak}
arm={M0,M_DA,M_HEAD,M_JOINT}
phase={before,after}
```

实际调度仅25个outer job，即5receiver×5关键slice。75个scene row、300个arm pair和600个prediction surface只是这25个任务内部的完整记账面，不是600次独立实验。不得追加receiver、seed、slice、参数扫描或完整125；也不得删除任一固定receiver、slice、scene或arm。

## 假设与判定

SCRC仅用当前phase合法support的D92 logits构造混淆矩阵，并以冻结、逐query、全注册类的互惠校正替代D108造成系统负收益的静态SMME偏置。假设M_HEAD相对M0、M_JOINT相对M_DA可同时改善注册后旧类、新类、H和弱类floor，而不是只迁移错误。

|切片|目标|
|---|---|
|K10/new5|`A-old≥92%`、`F-old≥85%`、`seen-new≥92%`|
|K10/new10|`A-old≥92%`、`F-old≥85%`、`seen-new≥90%`|
|K10/new20|`A-old≥92%`、`F-old≥85%`、`seen-new≥86%`|
|K5/new20|相对matched K10/new20的`A/F/N/H`下降均≤5pp|
|K1/new20|相对同row D92：`ΔH≥2pp、ΔF-old≥2pp、ΔA-old≥0、ΔN≥0`且old＋new正确数严格增加|

若D109没有重要联合正收益，完成25-job后立即判弱，不追加seed、不扫描SCRC参数、不恢复完整125，直接回到下一方法revision。

## 协议与资源边界

- 协议：`p2_min_v1`；复用已封存`VALIDATED_ONCE`数据，不改变received-IQ、physical ID、receiver/TX、scenario、K或support/query split。
- Phase2只读不可变Phase1 bundle、固定目标接收IQ、当前row合法support与注册表；无clean/source runtime访问。
- query逐样本面对全部注册类；零truth、role、quota、fit、update、selection和global reassignment。
- SCRC状态只由support构建；before／after及base／DA分别冻结，不跨臂复用。
- 最大注册类数26时SCRC状态约5412B；无训练epoch、阈值、温度、router或超参数搜索。

## 本地实现、验证与复审

|项目|值|
|---|---|
|Git分支|`codex/stage2-da25-r1`|
|实现文件|`code/cvsrffi/stage2_d109_target25.py`；SHA256=`5b5a7b1189ac7a413fc4c68543667cd21fa654e8b074f36bdca975313b18f678`|
|CLI|`code/scripts/run_d109_target25.py`；SHA256=`9298d52e26ef034da699289209df9079c8708d165270873e0d8e6bd99c9ddcfb`|
|聚焦测试|`tests/test_stage2_d109_target25.py`；SHA256=`959e5c93598ff8a0b3b1ec0ae4208157049fed601fa8b9a7c6b1ce80b4917f68`|
|本地验证|`ssr-gpu`中`py_compile`通过；D109 Target25／Target125／D92 core／SCRC共28项通过；仅既有PyTorch只读buffer警告|
|独立复审|`P0=0、P1=0 / RELEASE`；确认固定25-job、M0／M_DA逐值路径、M_HEAD／M_JOINT的SCRC注入、query权限、8shard、不可覆盖和truth时序|
|Git commit|以本地Git历史为准；不push、不上传GitHub|
|source archive|未生成；理论复审前不发布|

根目录`E:\type10-7`不是Git仓库；本报告镜像到Git工作树并与实现一起提交。

## 冻结N607输入

|输入|路径／SHA256|
|---|---|
|D92 matrix manifest|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/matrix_manifest.json`；`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|D92 output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`；`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|ground component|`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`；manifest`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`|
|D108 method lock|`<run-root>/source/configs/stage2_d108_cbrrc_smme_r1.json`；`7e8b310eeffc5e56aa39d60ef3b66c652207c3d9c1004e04d4499e6073862845`|

## 冻结的N607发布草案（当前不执行）

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d109_scrc_target25_s713102_20260802_r1`
- CWD：`<run-root>/source`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 日志：`<run-root>/logs/`
- PID／exit：`<run-root>/remote_control/`
- expected：`prepared/target125_plan.json`、`prepared/target125_context.json`、真实smoke、8个shard manifest、`predictions/prediction_manifest.json`、`truth_catalog.json`、`score/score_manifest.json`

prepare使用`code/scripts/run_d109_target25.py prepare`和上述冻结输入。真实row0／scene0无truth smoke通过后立即启动8个shard；每个物理GPU i固定：

```text
CUDA_VISIBLE_DEVICES=i python code/scripts/run_d109_target25.py predict-shard --plan-manifest <plan> --plan-manifest-sha256 <sha> --context-manifest <context> --context-manifest-sha256 <sha> --output-dir <run-root>/shards/shard_i --shard-index i --device cuda:0 --feature-batch-size 64
```

8shard全部退出0后merge并validate；600个prediction surface完整封存后才允许build-truth和score。唯一runner必须回填实际命令、PID、GPU映射、文件SHA和artifact计数。

## 停止、完成与分析规则

仅在P0协议／安全问题、错误source／SHA／不可覆盖风险，或至少两个不同outer在prediction前出现同一确定性异常指纹时停止精确run-owned进程树。不得因accuracy、H、floor或任何中间性能停止、选择row、调参或重启。

技术完成要求：25/25 outer、75/75 scene、300/300 arm pair、600/600 prediction surface、8/8 shard、150 truth surface、300 scene-arm score row和100 outer-arm聚合行闭合，异常为0。

完成后必须提供同row明细表，至少包含receiver、seed、K、新类数、scene、arm、B-old、A-old、F-old、seen-new、H、forgetting、correct/query、资源和判定；主结论基于M_HEAD−M0与M_JOINT−M_DA的配对联合收益。若弱，立即淘汰D109并研发下一方法；只有单seed目标通过才安排多seed确认。

## 执行记录

待唯一N607 runner回填。
