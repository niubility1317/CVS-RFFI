# CVS-RFFI Phase2-C T1完整消融v3运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2c_t1_20260730_v3_47212437`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|完成设计报告Stage2-C全部1425行screening矩阵，不基于中间性能缩小范围|
|正式release代码|`4721243770bee654bb7c41ad0bdd128d2dbfb863`|
|修复目标|package阶段使用source-sidecar loader读取合法v2 schema；最终发布的sidecar继续由formal scorer loader验收v3 schema|
|状态|`LOCAL_VERIFIED / INDEPENDENT_REVIEW_P0_0_P1_0 / PACKAGE_REUSE_PREREGISTERED / REMOTE_NOT_LAUNCHED / NO_PERFORMANCE_RESULT`|

## v1/v2边界与允许复用

v1与v2均固定为技术关闭且`NO_PERFORMANCE_RESULT`，不得patch、补写、覆盖或从原run继续执行。v3仅只读复用v2已成功发布的45个package及其不可变summary：

|复用项|固定值|
|---|---|
|package root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e`|
|package summary SHA256|`ed61d7c8403c203f4b1c73d4dd87251a67cf01b7da7f6f580143dfa852b026cb`|
|summary计数|expected/launched/completed/succeeded/validated均45；failed=0；systemic_stop=false|
|package制品|45个predictor manifest/seal、45个scoring manifest/source truth sidecar；package gate以predictor正式preflight及source-sidecar loader逐项重载|

该复用符合用户“前面的能复用就复用”和项目允许复用既有合法输入的要求。v3不重新审计D18数据，不要求不同启动的数据相同。

## 固定矩阵

|字段|值|
|---|---|
|source plan|`stage2c_screening_plan_47212437.json`|
|source plan SHA256|`6cc5fcba0194adaa65efb1c238479c7ac77d160228c2e1227db91d1852ffcbe4`|
|矩阵|1425 logical；19 arms×75 identities；预计1350 physical|
|states前置|325/325 `ARTIFACTS_COMPLETE`，失败0、缺件0|
|待fresh生成|75 feature identities、225 scope cache、30 formal v3 sidecar/60文件、75-entry index、1425-entry registry和sealed plan|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|GPU并发|8张GPU，每卡总compute进程最多2个；每波动态扣减既有compute进程|

## v3 fresh服务器路径

|字段|路径|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2c_t1_20260730_v3_47212437`|
|input/seal|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v3_47212437`|
|feature|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437`|
|sidecar|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437`|
|request|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2c_t1_20260730_v3_47212437`|
|run|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2c_t1_20260730_v3_47212437`|
|row log|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v3_47212437`|
|driver|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v3_47212437_driver`|

## 发布链与门禁

`package_launch_template.txt`在v3中只做v2 package只读验证，不创建或修改package。验证45/45后，`feature_launch_template.txt`和`sidecar_launch_template.txt`使用fresh v3输出根启动；feature完成门为75/75、225个scope cache且正式loader重载全部成功，sidecar完成门为30/30、60个文件且formal v3 loader逐项成功。`seal_launch_template.txt`生成75-entry index、1425-entry registry和1425 logical/1350 physical sealed plan。`launch_template.txt`再次执行states、package、feature、sidecar、source/sealed plan门后启动全部1425行。

启动前release必须精确HEAD=`47212437`且clean；所有v3目标根必须fresh不存在。首个feature wave、sidecar完成、seal完成、首个正式row和首个worker wave均记录计数、PID/CWD/cmdline、GPU槽、日志增长和异常指纹。仅P0或两个不同row在prediction前出现同一确定性故障时停止v3精确进程树；不得因性能值停止或缩小矩阵。

## 本地验证与复审

package gate保留predictor正式preflight，将scoring侧改为`cvsrffi.stage2_scoring_sidecar`读取source v2；sidecar controller随后发布并以`cvsrffi.stage2_metric_scorer`正式重载v3。新增loader分层focused test，完整相邻测试69项通过，compileall通过。独立复审结论`P0=0 / P1=0`，允许只读复用v2 package并创建fresh v3。

## 完成后检查

完成后追加75 feature、30 sidecar、sealed plan和1425行runner summary完整计数、异常、PID/GPU/SSH清理及同一实验结果表。结果解释基于同一candidate/run的old、seen-new、unknown、coverage、rollback/defer、loss/adapter和最终判定，不拼接跨行极值。
