# D131 D92-Lite160-QTIE Target125 r3实验报告

## 状态

- 实验ID：`d131_d92_lite160_qtie_target125_20260804_r3`
- 日期：2026-08-04
- 操作者：主agent负责科学集成、结果分析与晋级决定；Luna/max为唯一N607 runner。
- 当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
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
- 独立Terra/max复核：`P0=0,P1=0,RELEASE_READY=yes`。
- Git commit：`00aa23620225d8453f5868ffb7e254c2ead20d48`。
- runtime archive：`E:\type10-7\code\snapshots\d131_d92_lite160_qtie_target125_20260804_r3_runtime_00aa2362_v1.tar`，SHA256=`f01e659b7ab223c6fe0c558576b3c27080e1f68a362579de914e37878ce86265`，大小17,192,960B；独立重生成包大小与SHA完全一致。
- 实际解包SHA：method lock=`e0f7f8623b4d53002206aca8575f8eadd2bca4150a7c5aed3d017b4827fa5dac`；core=`cfdd787ffb4b5c6a6e43435d268acacc2d5979cc272762d994a72d3efa1732e0`；adapter=`f736cbb809f525e775a52151a84935eb0a0c9c7c2d20ab728f1f45b62643850a`；CLI=`141a8713e2d7e96a2955d71baa028d44c3f407b3155835c3f1efe9701c625750`。

## N607 runner result

- preflight、远端五项SHA、`py_compile`、prepare及真实checkpoint no-query smoke均通过；smoke两phase存在，query truth/fit/update/selection均为false，truth未打开。
- 8个固定shard均启动并自然退出：shard2完成其partial shard；其余7个失败。总计393个partial prediction、1个partial manifest。
- 5个shard重复`D108 exact top tie must fail closed`；另2个shard重复`registered_feature_primary160 contains a zero or non-finite row`。满足预注册系统性技术停止规则。
- 未执行merge、validate、truth-open或score；GPU与SSH均已清理；53个主要artifact回收到根目录报告的`artifacts/`。
- `technical_stop.txt` SHA=`dbc6454933193acd4821647d71ea72636e609e4ea752f4161858e6f495eeb6b9`；r3不得重启、续跑或按partial晋级。

## 主agent故障定位与路线决定

根据冻结的shard取模顺序、每个outer固定6个surface的执行顺序和失败前prediction计数，不读取任何partial预测内容或性能，可精确定位如下：

|outer index|outer ID|scene|phase|K/new|故障|
|---:|---|---|---|---|---|
|19|`d108-rx-20-1__seed-713105__k-1__new-20`|`leo_low_elev_weak`|before|K1/new20|qKNN精确top tie|
|24|`d108-rx-20-1__seed-713106__k-1__new-20`|`leo_clear_weak`|before|K1/new20|qKNN精确top tie|
|29|`d108-rx-3-19__seed-713102__k-1__new-20`|`leo_rain_weak`|before|K1/new20|qKNN精确top tie|
|39|`d108-rx-3-19__seed-713104__k-1__new-20`|`leo_low_elev_weak`|before|K1/new20|qKNN精确top tie|
|49|`d108-rx-3-19__seed-713106__k-1__new-20`|`leo_low_elev_weak`|after|K1/new20|qKNN alias精确top tie|
|116|`d108-rx-8-8__seed-713105__k-10__new-10`|`leo_rain_weak`|after|K10/new10|完整288维有限，但primary z_id160为零|
|118|`d108-rx-8-8__seed-713105__k-5__new-20`|`leo_rain_weak`|after|K5/new20|完整288维有限，但primary z_id160为零|

- 5个tie全部位于K1；r3加入的K5/K10 Lite-top二级qKNN没有覆盖真正根因。
- 4个before K1与1个after K1都依赖同一qKNN表示。对完全对称的query/support证据，任何逐query且类置换等变的规则都无法保证唯一类别；registry顺序、类别ID/hash或argmax首项不合法。
- 两个zero-primary失败不是NaN或读取故障：现有288维registered feature仍由合法辅助128维保持有限，但D131只截取first160，丢失了这些样本唯一可用的辅助信息。
- full288、aux128或固定288→160投影在协议上可以另行设计，但必须对support/query统一定义，且不再是D131冻结的`first160 canonical z_id160`方法身份。
- 决定：关闭D131，不发布r4补丁，不重复Target125。任何多视图D92-Lite后继必须使用新candidate、新method lock和较小联合筛选；393个partial永久不评分。


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
| D131-D92-LITE160-QTIE/r2 | 未形成完整矩阵 | - | - | - | - | - | - | - | - | 关闭；NO_PERFORMANCE_RESULT |
