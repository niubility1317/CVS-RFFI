# EPOC-ADV3B02教师蒸馏地面训练报告

## 基本信息

|字段|值|
|---|---|
|实验ID|`phase1_epoc_adv3b02_distill_20260705`|
|时间|2026-07-05|
|操作者|Codex|
|目标|在`ADV3B02_CORE90_SOFT_E200`基础上训练教师蒸馏学生模型，改善LEO星地信道下旧类表征稳定性和虚拟开集边界，为后续Stage2-C qknn8+协同推理评估准备新特征包|
|底座/教师|`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|训练环境|本地`ssr-gpu`验证；N607使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|结论状态|N607训练已完成，8个候选均产出checkpoint/metrics/prototype；Stage2-C qknn8+OSPR-CI协同评估待启动|

## 协议边界

|约束|本轮处理|
|---|---|
|地面训练不能接触未知类别|遵守。启动器只加载`ManySig.pkl`，不加载`ManyTx.pkl`，不传入`new_wisig_pkl`或真实unknown TX列表|
|`target_unknown`权限|不用于训练、阈值、profile或权重选择；仅后续Stage2-C评估可使用|
|拒识训练信号|仅来自`Y_old`源域内部leave-one-TX-out、soft inter-class mixup、虚拟低密度/边界outlier|
|旧类准确不能无约束下降|启动器仍启用`joint_safe`和PAIC守护；同时按用户要求放宽探索强度，允许更强虚拟边界候选进入诊断|
|声明边界|本轮不是部署成功，只是source-only地面训练修复实验|

## 算法设计

总损失由旧类CE、ADV3B02冻结教师clean KL、LEO视图teacher KL、`z_id`特征MSE、source-heldout proxy loss、soft unknown mixup、open-world feature loss、source episode半径约束、satellite strong-view CE组成。

关键实现：

|模块|实现|
|---|---|
|冻结教师|`train_ssdg.py`新增`--teacher_ckpt`，非零教师权重时加载并冻结教师模型|
|蒸馏项|`--lambda_teacher_clean_kl`、`--lambda_teacher_sat_kl`、`--lambda_teacher_zid_mse`|
|LEO视图|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|开集边界|`lambda_proxy_unknown`、`lambda_soft_unknown_mixup`、`lambda_open_world_feat`、`lambda_source_episode`，均不使用真实未知TX|
|部署包|每候选导出`phase2_zid_prototypes.pt`供后续Stage2-C评估|

## 候选矩阵

|候选|GPU|机制侧重|epoch|真实未知训练|
|---|---:|---|---:|---:|
|`EPOC_DISTILL_A_MILD`|0|轻蒸馏+轻虚拟边界|160|0|
|`EPOC_DISTILL_B_KDHI`|1|强teacher KL|170|0|
|`EPOC_DISTILL_C_OPENHI`|2|强source-heldout/open-world边界|180|0|
|`EPOC_DISTILL_D_SATHI`|3|强LEO旧类保持|180|0|
|`EPOC_DISTILL_E_SOFTMIX`|4|强soft inter-class mixup|190|0|
|`EPOC_DISTILL_F_RELAXED`|5|更激进虚拟边界，放宽保守旧类约束|190|0|
|`EPOC_DISTILL_G_BALANCED`|6|蒸馏/LEO/open-world均衡|200|0|
|`EPOC_DISTILL_H_AGGRESSIVE`|7|最强虚拟边界探索|200|0|

## 本地变更

|文件|目的|
|---|---|
|`code/SSDG/train_ssdg.py`|新增冻结教师蒸馏参数、加载、loss、telemetry|
|`code/scripts/launch_phase1_epoc_adv3b02_distill_20260705.sh`|新增EPOC教师蒸馏候选启动器|
|`code/tests/test_epoc_adv3b02_teacher_distill.py`|TDD覆盖parser和launcher dry-run协议约束|
|`docs/superpowers/plans/2026-07-05-epoc-adv3b02-distill.md`|实现计划与TDD证据|

## 本地验证

|命令|结果|
|---|---|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_epoc_adv3b02_teacher_distill.py -q`|RED:初次失败，缺少教师蒸馏参数和启动器；GREEN:2 passed，另有`.pytest_cache`权限warning|
|`bash -n code/scripts/launch_phase1_epoc_adv3b02_distill_20260705.sh`|PASS|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\SSDG\train_ssdg.py`|PASS|
|`bash code/scripts/launch_phase1_epoc_adv3b02_distill_20260705.sh --dry-run --only=EPOC_DISTILL_A_MILD`|PASS；dry-run含`--teacher_ckpt`、三种LEO视图、`real_unknown_classes_in_training=0`，不含`ManyTx.pkl`|

## 初始同步与启动计划

|项目|值|
|---|---|
|远端工作目录|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_epoc_adv3b02_distill_20260705`|
|远端log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_epoc_adv3b02_distill_20260705`|
|启动命令|`nohup bash code/scripts/launch_phase1_epoc_adv3b02_distill_20260705.sh > logs/phase1_epoc_adv3b02_distill_20260705/driver.out 2>&1 < /dev/null & echo $!`|
|健康检查|查看`driver.out`和各候选`.out`中的`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`、Traceback、OOM、NaN|

## 远端同步与启动记录

|项目|结果|
|---|---|
|N607 preflight|PASS；direct`N607`可用，host=`dell-DSS8440`，project root可见，GPU0-7均约`10/24576MiB`|
|远端环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|远端hash|`train_ssdg.py=5e7950dc...`，launcher=`8f18f0c2...`，test=`8e9264d6...`，plan=`ee0e2db4...`，report=`080d978c...`，manifest=`7dc5ec51...`|
|远端验证|`py_compile`PASS；`CVS-RFFI`环境无`pytest`模块，改用同环境等价Python断言验证parser和launcher dry-run，PASS；`bash -n`PASS|
|启动命令|`nohup bash code/scripts/launch_phase1_epoc_adv3b02_distill_20260705.sh > logs/phase1_epoc_adv3b02_distill_20260705/driver.out 2>&1 < /dev/null & echo driver_pid=$!`|
|driver PID|`2903082`|
|日志根目录|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_epoc_adv3b02_distill_20260705`|

## 启动健康检查

检查时间：N607 2026-07-05 18:28左右。

|候选|GPU|PID|健康状态|
|---|---:|---:|---|
|`EPOC_DISTILL_A_MILD`|0|`2903096`|日志含`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`|
|`EPOC_DISTILL_B_KDHI`|1|`2903921`|日志含`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`|
|`EPOC_DISTILL_C_OPENHI`|2|`2904343`|日志含`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`|
|`EPOC_DISTILL_D_SATHI`|3|`2904824`|日志含`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`|
|`EPOC_DISTILL_E_SOFTMIX`|4|`2905635`|日志含`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`|
|`EPOC_DISTILL_F_RELAXED`|5|`2906074`|日志含`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`|
|`EPOC_DISTILL_G_BALANCED`|6|`2906496`|日志含`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`|
|`EPOC_DISTILL_H_AGGRESSIVE`|7|`2907323`|日志含`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`|

GPU状态：GPU0-7约`2017-2327MiB/24576MiB`，利用率约`20-39%`。未检出`Traceback`、`RuntimeError`、`CUDA out of memory`、`NaN`或`unrecognized arguments`。

## 后续检查

训练结束后必须基于同一候选行读取`metrics_epoch.csv/jsonl`、`best_joint_safe_ssdg.pth`、`phase2_zid_prototypes.pt`，再导出Stage2-C特征并运行qknn8/AWARE/OSPR协同推理`M=1..5`。真实`ManyTx`新类/未知类只能在该评估阶段使用，不得回填为本轮地面训练证据。

## 完成状态与指标解析

检查时间：N607 2026-07-05 22:12-22:20左右。

|项目|结果|
|---|---|
|进程/GPU状态|`train_ssdg.py`与本run进程均已退出；GPU0-7均约`10MiB/24576MiB`，无本run残留训练负载|
|driver状态|`driver.out`结束于`[EPOC-DISTILL-SUBMIT-COMPLETE]`|
|日志错误|未检出`Traceback`、`RuntimeError`、`CUDA out of memory`、`out of memory`、`NaN`、`unrecognized arguments`或prototype导出失败|
|完成轮次|A=160，B=170，C=180，D=180，E=190，F=190，G=200，H=200，均达到计划轮次|
|关键产物|8个候选均存在`best_joint_safe_ssdg.pth`、`metrics_epoch.csv`、`metrics_epoch.jsonl`、`phase2_zid_prototypes.pt`|
|协议边界|地面训练只用`ManySig.pkl`源域；`real_unknown_classes_in_training=0`；真实`ManyTx`未知/新类未进入本轮训练|

## EPOC候选联合结果

排序口径：以同一候选行的`best_score`为主，结合`protected_sat_floor_tx`和unseen-day-unseen-rx准确率观察稳定性。以下指标均来自各候选`metrics_epoch.csv`中`best_epoch`对应行；百分数保留原始量纲。

|候选|final_epoch|best_epoch|best_score|best_test_tx|best_UDU|receiver_floor|sat_mean|sat_floor|clear_tx|low_tx|rain_tx|teacher_clean_kl|teacher_sat_kl|teacher_zid_mse|proxy_auc|proxy_vaccept|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`EPOC_DISTILL_B_KDHI`|170|100|86.648158|90.835294|86.808333|78.516667|79.083497|77.783333|81.043627|78.423529|77.783333|0.761722|4.737076|0.002191|0.528222|0.811859|
|`EPOC_DISTILL_G_BALANCED`|200|10|86.464849|91.120588|87.525000|80.091667|75.972549|74.710294|78.168137|75.039216|74.710294|1.059813|0.000000|0.002047|NA|NA|
|`EPOC_DISTILL_F_RELAXED`|190|70|86.347859|90.382843|86.470000|79.141667|78.178105|77.081373|79.974020|77.478922|77.081373|0.857566|5.816287|0.002249|0.523652|0.801923|
|`EPOC_DISTILL_C_OPENHI`|180|120|86.143068|90.316667|86.790000|77.075000|78.505719|77.417647|80.276961|77.822549|77.417647|1.042699|4.696401|0.002404|0.515870|0.809295|
|`EPOC_DISTILL_D_SATHI`|180|172|86.112630|89.825000|85.400000|78.516667|79.345588|78.268627|81.019608|78.748529|78.268627|0.765156|3.957130|0.002283|0.519628|0.806731|
|`EPOC_DISTILL_E_SOFTMIX`|190|184|86.101261|90.009804|86.316667|75.933333|80.116013|79.172059|81.700980|79.475000|79.172059|0.683914|3.976250|0.002252|0.521061|0.815385|
|`EPOC_DISTILL_A_MILD`|160|120|85.901657|90.316667|86.355000|75.008333|79.332680|78.325980|81.059314|78.612745|78.325980|0.762660|4.619915|0.002294|0.521934|0.812180|
|`EPOC_DISTILL_H_AGGRESSIVE`|200|188|85.899586|89.861275|85.243333|76.816667|79.863235|78.946078|81.430882|79.212745|78.946078|0.868992|3.754960|0.002292|0.520507|0.807372|

## 解释与下一阶段选择

|结论项|判断|
|---|---|
|主推底座|`EPOC_DISTILL_B_KDHI`，同一候选行`best_score`最高，clean/test和LEO sat_floor保持均衡，已导出Stage2 prototype|
|保底对照|`EPOC_DISTILL_G_BALANCED`旧类DG侧`best_test_tx`、`best_UDU`、`receiver_floor`最高，但LEO sat_floor低于B；适合作为旧类保持对照|
|LEO鲁棒对照|`EPOC_DISTILL_E_SOFTMIX`的`sat_mean/sat_floor`最高，但receiver_floor偏低；适合作为星地信道压力对照|
|未知拒识边界|本轮只证明源域代理未知/蒸馏训练完成，不能证明真实`target_unknown`拒识；真实未知拒识必须进入Stage2-C qknn8+协同推理评估|
|下一步|使用`EPOC_DISTILL_B_KDHI`作为首选底座，`G`和`E`作为必要对照，运行Stage2-C特征/qknn8/协同推理，`collab_counts=all`即M=1..全体target receiver数|

## Stage2-C评估启动约束

|约束|要求|
|---|---|
|底座模型|首选`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_epoc_adv3b02_distill_20260705/EPOC_DISTILL_B_KDHI/best_joint_safe_ssdg.pth`|
|prototype|首选`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_epoc_adv3b02_distill_20260705/EPOC_DISTILL_B_KDHI/phase2_zid_prototypes.pt`|
|协议|Stage2-C；`Y_old`来自ManySig旧类，`Y_new/Y_unknown`从ManyTx non-old TX互斥拆分|
|目标receiver|优先使用`项目.md`确认的`20-1,3-19,7-14,7-7,8-8`，与source receiver disjoint|
|qknn|使用qknn8作为基础在轨少样本方法|
|协同数量|必须覆盖`M=1..R`，其中`R`为目标receiver数量；报告参与推理数量、时延、通信字节和prototype存储|
|声明边界|未知query不得参与阈值拟合；clean view只能作为control；未达到Stage2-C三类指标前不得声明部署成功|

## Stage2-C启动器

|项目|值|
|---|---|
|本地启动器|`code/scripts/launch_phase2_epoc_b_ospr_qknn_collab_20260705.sh`|
|本地测试|`code/tests/test_phase2_epoc_b_ospr_qknn_collab_launcher.py`|
|首选候选|`EPOC_DISTILL_B_KDHI`|
|目标receiver|`20-1,3-19,7-14,7-7,8-8`|
|Stage2-C seen-new TX|`1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4`|
|Stage2-C unknown TX|`10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20`|
|OSPR proxy-unknown TX|`12-7,13-14,13-19,13-3,13-7,14-11,14-12,14-13`|
|数据覆盖审计|N607只读审计确认122个ManyTx non-old TX在5个目标receiver下均满足每receiver至少40条；所选三组TX均来自该覆盖池且互斥|
|本地验证|`bash -n code/scripts/launch_phase2_epoc_b_ospr_qknn_collab_20260705.sh`PASS；`ssr-gpu pytest code/tests/test_phase2_epoc_b_ospr_qknn_collab_launcher.py code/tests/test_phase2_ospr_ci_eval.py code/tests/test_phase2_collaborative_open_set_qknn_eval.py -q`为58 passed，另有`.pytest_cache`权限warning|
|远端计划|同步启动器和测试到N607，远端`bash -n`/dry-run/py_compile检查后，使用显存占用最低GPU导出Stage2-C LEO特征并运行OSPR-CI qknn8，`collab_counts=all`覆盖M=1..5|
|声明限制|默认`event_alignment_policy=receiver_domain_ranked`，因此本轮只能作为receiver-domain ensemble的`NON_DEPLOYMENT_DIAGNOSTIC`；若要作为严格卫星群同事件协同证据，必须另行使用`strict_event_key`并证明共享`role+tx+day+sig+scenario`|

## Stage2-C远端启动记录

检查时间：N607 2026-07-05 22:27-22:29左右。

|项目|结果|
|---|---|
|N607 preflight|PASS；直连`N607`可用，host=`dell-DSS8440`，项目根目录可见，GPU0-7均约`10MiB/24576MiB`|
|同步文件|`code/scripts/launch_phase2_epoc_b_ospr_qknn_collab_20260705.sh`、`code/tests/test_phase2_epoc_b_ospr_qknn_collab_launcher.py`、`code/SYNC_MANIFEST.txt`、本报告|
|远端hash|launcher=`dd428b57...`；test=`03dd47ab...`；manifest=`ab496b81...`；report=`fba548b3...`|
|远端验证|`bash -n`PASS；dry-run包含`EPOC_DISTILL_B_KDHI`、`collab_counts=all`、`qknn_k=8`、`protocol=Stage2-C`、`unknown_query_eval_only=true`、`verdict_scope=NON_DEPLOYMENT_DIAGNOSTIC`、`--event_alignment_policy receiver_domain_ranked`、三种LEO场景和`--target_unknown_reject 0.99`|
|启动命令|`cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/launch_phase2_epoc_b_ospr_qknn_collab_20260705.sh`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_epoc_b_ospr_qknn_collab_20260705/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5`|
|log path|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_epoc_b_ospr_qknn_collab_20260705/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5.out`|
|driver PID|`3009848`|
|OSPR PID|`3009925`|
|GPU选择|启动器选择GPU0；特征导出后GPU0约`12MiB/24576MiB`，OSPR-CI使用`--device cpu`继续运行|
|启动健康|`features_stage2c_leo_multirx.npz`已生成；manifest显示source旧类解析为`14-10,14-7,20-15,20-19,6-15,8-20`，target view为`satellite/LEO`，场景为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`；未检出Traceback/OOM/NaN/协议错误|
|当前状态|OSPR-CI qknn8协同评估运行中；完成后需解析`ospr_ci_summary.json`、ENPC/SLEV summary/evidence CSV，并按M=1..5同row报告old/seen-new/unknown/资源指标|

## Stage2-C首次启动故障与修复

首次启动的OSPR后端失败于评估分组阶段：

```text
ValueError: no evidence groups contain 1 receiver observations
```

根因不是特征导出失败、协议泄漏或OOM。远端只读诊断显示`ospr_ci_adapted_features.npz`可生成qknn evidence：

|诊断项|结果|
|---|---|
|evidence rows|`2200`|
|target receivers|`20-1,3-19,7-14,7-7,8-8`|
|event receiver count histogram|`{1:320,2:556,3:232,4:18}`|
|失败原因|默认`collab_group_policy=same_max_budget`要求所有M共享最大预算`M=5`的事件组；但receiver-domain ranked诊断中没有5-receiver事件组，因此k=1时也拿到空eligible集合|
|修复|启动器默认改为`collab_group_policy=available_up_to_k`、`partial_collab_min_receivers=1`，仍输出`collab_counts=all`的M=1..5预算表，但必须同时报告`actual_receiver_count_histogram`|
|声明边界|修复后仍是`NON_DEPLOYMENT_DIAGNOSTIC`；M=5是预算上限，不等价于每个事件真实5接收机同事件协同|

## Stage2-C retry1完成记录

运行ID：`phase2_epoc_b_ospr_qknn_collab_20260705_retry1`。  
完成时间：N607 2026-07-05 22:43 CST左右。  
结论：修复后的OSPR-CI/qknn8协同评估已完成，40行profile×M结果均`resource_pass=True`，但没有任何行达到目标门槛，`target_pass=True`行数为0。本轮不能写成部署成功，也不能作为严格同事件卫星群协同证据；它证明当前`EPOC_DISTILL_B_KDHI + qknn8 + OSPR-CI`在`receiver_domain_ranked`诊断设置下仍没有解决旧类/新类保持与未知拒识的冲突。

|项目|结果|
|---|---|
|远端hash校验|launcher=`52b0c05a...`；test=`6ef7b7f3...`；manifest=`f2561df0...`；report=`351a6cb9...`，与本地修复版一致|
|远端语法/dry-run|`bash -n`PASS；dry-run确认`collab_group_policy=available_up_to_k`、`partial_collab_min_receivers=1`、`collab_counts=all`、`event_alignment_policy=receiver_domain_ranked`、`verdict_scope=NON_DEPLOYMENT_DIAGNOSTIC`、`qknn_k=8`、`target_unknown_reject=0.99`|
|启动命令|`cd /home/szu2070436088/2510044040/CV-SincNet && RUN_ID=phase2_epoc_b_ospr_qknn_collab_20260705_retry1 bash code/scripts/launch_phase2_epoc_b_ospr_qknn_collab_20260705.sh`|
|driver PID|`3015202`|
|特征导出PID|`3015204`|
|OSPR PID|`3015285`|
|GPU/设备|GPU0用于短时特征导出，OSPR-CI使用CPU；健康检查时GPU0约`12MiB/24576MiB`|
|远端输出目录|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_epoc_b_ospr_qknn_collab_20260705_retry1/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5`|
|远端日志|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_epoc_b_ospr_qknn_collab_20260705_retry1/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5.out`|
|本地summary副本|`automation_reports/CV-SincNet/phase1_epoc_adv3b02_distill_20260705/stage2c_retry1_ospr_ci/`|
|完成产物|`ospr_ci_summary.json`、`ospr_ci_enpc_summary.csv`、`ospr_ci_slev_summary.csv`、ENPC/SLEV evidence CSV均已生成|

### 协议与泄漏审计

|检查项|证据|
|---|---|
|Stage2-C底座|`EPOC_DISTILL_B_KDHI`|
|旧类|`14-10,14-7,20-15,20-19,6-15,8-20`|
|seen-new|`1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4`|
|unknown|`10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20`|
|目标receiver|`20-1,3-19,7-14,7-7,8-8`|
|训练计数|`source_fit=13248`，`source_holdout_calibration=192`，`target_support=560`，`proxy_unknown=10527`，`target_unknown_eval_only=8000`|
|未知类泄漏|`target_unknown_training_count=0`，`target_unknown_eval_only=True`，`forbidden_roles=['target_unknown']`|
|LEO场景覆盖|`leo_clear_weak=779`，`leo_low_elev_weak=711`，`leo_rain_weak=710`条evidence|
|实际receiver覆盖|`strict_event_receiver_count`直方图为`{1:320,2:1112,3:696,4:72}`；没有5-receiver同事件组|
|声明边界|`event_alignment=receiver_domain_ranked_by_role_tx_scenario`，因此本轮为`NON_DEPLOYMENT_DIAGNOSTIC`；M=5是协同预算上限，不是每个事件真实5星同观测|

### 资源指标

|指标|值|
|---|---:|
|adapter参数|3840|
|adapter fp16字节|7680|
|prototype fp16字节|4480|
|qknn8 support int8字节|89600|
|verifier fp16字节|322|
|总状态字节|102082|
|bytes/event范围|128.00到250.09|
|latency_ms范围|5.23到5.42|
|资源约束|40/40行`resource_pass=True`|

### 完整profile×M结果表

表中准确率、拒识率和FAR均为百分比；`target=False`表示未达到`old_acc>=99%`、`min_old>=95%`、`seen_new_acc>=97%`、`min_seen>=93%`、`unknown_reject>=99%`的联合目标。

|backend|profile|M|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|known_defer|unknown_defer|bytes/event|latency_ms|resource|target|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|ENPC|`enpc_balanced`|1|58.62|6.67|31.55|19.51|16.62|83.38|0.00|0.00|128.00|5.42|True|False|
|ENPC|`enpc_balanced`|2|56.61|6.67|29.01|20.45|20.00|75.84|2.56|4.16|219.62|5.42|True|False|
|ENPC|`enpc_balanced`|3|56.32|6.67|28.50|20.45|23.12|74.81|1.48|2.08|248.04|5.42|True|False|
|ENPC|`enpc_balanced`|4|56.32|6.67|28.50|20.45|23.12|74.81|1.48|2.08|250.09|5.42|True|False|
|ENPC|`enpc_balanced`|5|56.32|6.67|28.50|20.45|23.12|74.81|1.48|2.08|250.09|5.42|True|False|
|ENPC|`enpc_known_anchor`|1|59.77|10.00|35.88|22.73|0.00|100.00|0.00|0.00|128.00|5.42|True|False|
|ENPC|`enpc_known_anchor`|2|59.48|8.33|36.13|22.73|0.00|100.00|0.00|0.00|219.62|5.42|True|False|
|ENPC|`enpc_known_anchor`|3|59.20|8.33|36.39|23.64|0.00|100.00|0.00|0.00|248.04|5.42|True|False|
|ENPC|`enpc_known_anchor`|4|59.20|8.33|36.39|23.64|0.00|100.00|0.00|0.00|250.09|5.42|True|False|
|ENPC|`enpc_known_anchor`|5|59.20|8.33|36.39|23.64|0.00|100.00|0.00|0.00|250.09|5.42|True|False|
|ENPC|`enpc_old80_unknown_probe`|1|46.55|5.00|23.66|9.76|48.05|51.95|0.00|0.00|128.00|5.42|True|False|
|ENPC|`enpc_old80_unknown_probe`|2|38.79|1.67|16.28|4.88|69.61|30.39|0.00|0.00|219.62|5.42|True|False|
|ENPC|`enpc_old80_unknown_probe`|3|34.77|1.67|11.96|2.27|79.74|20.26|0.00|0.00|248.04|5.42|True|False|
|ENPC|`enpc_old80_unknown_probe`|4|34.77|1.67|11.96|2.27|79.74|20.26|0.00|0.00|250.09|5.42|True|False|
|ENPC|`enpc_old80_unknown_probe`|5|34.77|1.67|11.96|2.27|79.74|20.26|0.00|0.00|250.09|5.42|True|False|
|ENPC|`enpc_unknown_strict`|1|0.00|0.00|0.00|0.00|70.65|0.00|46.96|29.35|128.00|5.42|True|False|
|ENPC|`enpc_unknown_strict`|2|24.14|0.00|6.62|1.82|82.86|14.55|12.28|2.60|219.62|5.42|True|False|
|ENPC|`enpc_unknown_strict`|3|20.40|0.00|3.56|0.00|88.31|9.09|12.15|2.60|248.04|5.42|True|False|
|ENPC|`enpc_unknown_strict`|4|20.40|0.00|2.54|0.00|88.57|8.83|12.15|2.60|250.09|5.42|True|False|
|ENPC|`enpc_unknown_strict`|5|20.40|0.00|2.54|0.00|88.57|8.83|12.15|2.60|250.09|5.42|True|False|
|SLEV|`slev_balanced`|1|54.89|6.67|30.03|19.51|28.05|71.95|0.00|0.00|128.00|5.23|True|False|
|SLEV|`slev_balanced`|2|50.29|3.33|24.94|12.00|45.97|53.77|0.13|0.26|219.62|5.23|True|False|
|SLEV|`slev_balanced`|3|50.00|3.33|24.17|12.00|50.65|49.09|0.00|0.26|248.04|5.23|True|False|
|SLEV|`slev_balanced`|4|50.00|3.33|23.92|12.00|51.17|48.57|0.00|0.26|250.09|5.23|True|False|
|SLEV|`slev_balanced`|5|50.00|3.33|23.92|12.00|51.17|48.57|0.00|0.26|250.09|5.23|True|False|
|SLEV|`slev_energy_strict`|1|0.00|0.00|0.00|0.00|92.99|0.00|19.57|7.01|128.00|5.23|True|False|
|SLEV|`slev_energy_strict`|2|19.25|0.00|2.04|0.00|94.03|5.71|5.80|0.26|219.62|5.23|True|False|
|SLEV|`slev_energy_strict`|3|15.52|0.00|1.27|0.00|97.66|2.08|5.80|0.26|248.04|5.23|True|False|
|SLEV|`slev_energy_strict`|4|15.52|0.00|1.02|0.00|97.92|1.82|5.80|0.26|250.09|5.23|True|False|
|SLEV|`slev_energy_strict`|5|15.52|0.00|1.02|0.00|97.92|1.82|5.80|0.26|250.09|5.23|True|False|
|SLEV|`slev_known_anchor`|1|59.77|10.00|35.88|22.73|0.00|100.00|0.00|0.00|128.00|5.23|True|False|
|SLEV|`slev_known_anchor`|2|59.48|8.33|36.39|22.73|0.00|100.00|0.00|0.00|219.62|5.23|True|False|
|SLEV|`slev_known_anchor`|3|59.20|8.33|36.39|23.64|0.00|100.00|0.00|0.00|248.04|5.23|True|False|
|SLEV|`slev_known_anchor`|4|59.20|8.33|36.39|23.64|0.00|100.00|0.00|0.00|250.09|5.23|True|False|
|SLEV|`slev_known_anchor`|5|59.20|8.33|36.39|23.64|0.00|100.00|0.00|0.00|250.09|5.23|True|False|
|SLEV|`slev_old80_energy_probe`|1|47.41|5.00|25.19|9.76|45.19|54.81|0.00|0.00|128.00|5.23|True|False|
|SLEV|`slev_old80_energy_probe`|2|41.67|1.67|17.30|4.88|66.75|33.25|0.00|0.00|219.62|5.23|True|False|
|SLEV|`slev_old80_energy_probe`|3|38.51|1.67|13.49|4.55|77.40|22.60|0.00|0.00|248.04|5.23|True|False|
|SLEV|`slev_old80_energy_probe`|4|38.51|1.67|13.49|4.55|77.92|22.08|0.00|0.00|250.09|5.23|True|False|
|SLEV|`slev_old80_energy_probe`|5|38.51|1.67|13.49|4.55|77.92|22.08|0.00|0.00|250.09|5.23|True|False|

### 结果解释与下一步

|观察|解释|
|---|---|
|最高旧类准确率|`ENPC/SLEV known_anchor,M=1`为59.77%，远低于OLD80阶段门槛，更低于最终99%目标；每类最低旧类仅10.00%|
|最高seen-new准确率|`ENPC/SLEV known_anchor,M>=2/3`约36.39%，每类最低seen-new约23.64%，远低于97%/93%目标|
|最高未知拒识|`SLEV energy_strict,M=4/5`达到97.92%拒识、1.82%FAR，但旧类只有15.52%、seen-new约1.02%，属于拒识强但已知类崩塌|
|低FAR条件下最好旧类|`SLEV energy_strict,M=3/4/5`满足`unknown_FAR<=5%`，但旧类只有15.52%，不可作为可用路线|
|协同数量影响|从M=1增加到M=4/5主要提高未知拒识或降低FAR，但没有提升旧类和seen-new；当前融合规则在已知类保持上反向伤害明显|
|工程资源|bytes/event和latency满足约束，瓶颈不是资源，而是表征与开集决策边界|
|路线判断|单纯在当前EPOC B特征上增加receiver-domain协同融合不足以达成目标。下一步应转向底层特征分离和训练目标：使用ADV3B02/EPOC教师指导的新地面模型，使叠加星地信道后的unknown特征远离`Y_old/Y_new`原型，同时保留旧类每类floor；训练阶段仍禁止接触真实`Y_unknown`，只能使用源侧proxy/open-set负样本或合成扰动负样本|

## OSPR-CI++最小实现与N607复评计划

记录时间：2026-07-06 02:20 CST。  
路线状态：`NON_DEPLOYMENT_DIAGNOSTIC`。本节只把OSPR-CI++作为已有Stage2-C特征上的qknn8协同复评分支，不改变`项目.md`协议，不使用真实`Y_unknown`参与地面训练、阈值拟合或校准。

### 本地代码改动

|文件|目的|SHA256|
|---|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增`ospr_ci_pp`融合策略名，并映射到support-protected unknown confirmation内部策略`scg_qknn_cvs`；保留请求策略名便于报告审计|`ED7999B7DF253D65FFEA1633BCA33C68A6CAC7EEA76BDD76021CCC69CB386968`|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|CLI允许`--fusion_policy ospr_ci_pp`，使远端复评脚本可直接调用|`754DA2CC5EAA9642F83CBEA6A77E0EFB916CFE0FCA13298C1D51CCB31BC9844D`|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|TDD覆盖OSPR-CI++在旧类双接收机一致高可靠时accept、未知多接收机高风险时reject，并检查资源预算|`61DADE646A81EF7F4C704A6AF9D90DD28A006D017AF0EFC4433AD7D1A5F39B89`|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|TDD覆盖CLI接受`ospr_ci_pp`枚举|`D95F6465463D8C6860FF6E9E278195DAFB8424F485302023605DCD75586DD609`|

### 本地验证

|命令|结果|
|---|---|
|`conda run --no-capture-output -n ssr-gpu python -m pytest -q code\tests\test_phase2_collaborative_open_set_qknn_eval.py -k ospr_ci_pp`|PASS，`1 passed,53 deselected`；仅`.pytest_cache`权限warning|
|`conda run --no-capture-output -n ssr-gpu python -m pytest -q code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py`|PASS，`122 passed`；仅`.pytest_cache`权限warning|
|`conda run -n ssr-gpu python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py`|PASS|

初始RED证据：`test_cli_accepts_ospr_ci_pp_fusion_policy`在CLI未允许`ospr_ci_pp`时因`argparse invalid choice`失败；加入枚举后变为GREEN。  
本地快照：`E:\type10-7\code\snapshots\phase2_ospr_ci_pp_20260706_022047\`。

### N607复评命令

复评复用retry1已有特征，不重新训练，不占用显存密集训练资源。输出目录计划为：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_epoc_b_ospr_qknn_collab_20260705_retry1/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5/ospr_ci_pp
```

计划命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_epoc_b_ospr_qknn_collab_20260705_retry1/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5/ospr_ci/ospr_ci_adapted_features.npz \
  --output_json runs/phase2_epoc_b_ospr_qknn_collab_20260705_retry1/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5/ospr_ci_pp/ospr_ci_pp.json \
  --output_evidence_csv runs/phase2_epoc_b_ospr_qknn_collab_20260705_retry1/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5/ospr_ci_pp/ospr_ci_pp_evidence.csv \
  --collab_counts all \
  --collab_group_policy same_max_budget \
  --partial_collab_min_receivers 1 \
  --k_shot 8 \
  --query_per_class 20 \
  --qknn_k 8 \
  --seed 4070505 \
  --support_selection_policy stable_first \
  --event_alignment_policy receiver_domain_ranked \
  --support_calibration_mode leave_one_out \
  --class_conformal_enabled \
  --class_evidence_top_m 2 \
  --class_verifier_policy support_quality \
  --class_verifier_top_m 2 \
  --class_shell_unknown_risk_enabled \
  --virtual_unknown_risk_enabled \
  --virtual_unknown_risk_samples_per_class 2 \
  --class_negative_risk_enabled \
  --class_negative_samples_per_class 2 \
  --fusion_policy ospr_ci_pp \
  --label_fusion_policy weighted_vote_margin \
  --receiver_class_reliability_policy support_calibrated \
  --candidate_set_min_receivers 2 \
  --candidate_set_min_top1_receivers 2 \
  --candidate_set_min_conformal_pvalue 0.50 \
  --candidate_set_min_label_receiver_class_reliability 0.75 \
  --candidate_set_max_label_unknown_risk 0.80 \
  --candidate_set_max_event_unknown_risk 0.80 \
  --candidate_set_max_label_risk_component_agreement 0.50 \
  --candidate_set_unknown_reject_risk 0.85 \
  --candidate_set_shell_reject_risk 0.85 \
  --candidate_set_max_receiver_pair_label_disagreement 0.25 \
  --candidate_set_max_receiver_pair_unknown_risk_range 0.25 \
  --unknown_risk_threshold 0.85 \
  --accept_margin_threshold 0.05 \
  --consensus_score_threshold 0.05 \
  --max_event_bytes 1152 \
  --max_event_latency_ms 20
```

验收边界：若OSPR-CI++仍不能在同一行同时达到`old_acc>=99%`、`min_old>=95%`、`seen_new_acc>=97%`、`min_seen>=93%`、`unknown_reject>=99%`，则继续作为协同推理负证据，并将主路线转入ADV3B02教师指导的source-only负原型壳层/特征分离蒸馏。

### N607复评完成记录

完成时间：2026-07-06 02:22 CST。  
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。  
执行方式：CPU/低显存复评，复用`ospr_ci_adapted_features.npz`，未重新训练；运行前GPU0/GPU1显存约`10MiB/24576MiB`，GPU2-7已有训练进程保持monitor-only。  
实际修正：原计划中的`same_max_budget`会因无5接收机同事件组而复现空组问题；本次实际使用`collab_group_policy=available_up_to_k`，因此M=5表示预算上限，不表示每个事件都有5个receiver观测。

|项目|结果|
|---|---|
|远端hash校验|4个同步文件hash均与本地一致：`ed7999...`、`754da2...`、`61dade...`、`d95f64...`|
|远端语法验证|`py_compile` PASS|
|远端聚焦测试|远端无`pytest`，改用`unittest`直接运行`Phase2CollaborativeOpenSetQknnEvalTest.test_cli_accepts_ospr_ci_pp_fusion_policy`，PASS|
|远端输出目录|`runs/phase2_epoc_b_ospr_qknn_collab_20260705_retry1/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5/ospr_ci_pp_20260706_022047/`|
|本地产物副本|`automation_reports/CV-SincNet/phase1_epoc_adv3b02_distill_20260705/stage2c_retry1_ospr_ci_pp_20260706_022047/`|
|`ospr_ci_pp.json` SHA256|`47C0807F6651F613242053C5642BE8524D0AF2819B704BE571A04235D86E528C`|
|`ospr_ci_pp_evidence.csv` SHA256|`D8A9BBB6AAA4CBD8BB56B54841BE09F64A2219D3E4B04C900D3E4057F0D11C91`|
|输入evidence|`receiver_count=5`，`group_count=1126`，`evidence_row_count=2200`|

#### OSPR-CI++ M=1..5结果

表中准确率、拒识率、FAR和资源违例率均为百分比。`unknown_FAR=0`在这里不是成功，因为旧类和seen-new几乎全部被拒识或错分。

|M|events|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|unknown_defer|bytes/event|latency_p95_ms|resource_violation|avg_rx|actual_rx_hist|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|1126|9.48|0.00|0.00|0.00|98.70|0.00|1.30|40.00|1.18|0.00|1.00|`{1:1126}`|
|2|1126|8.05|0.00|0.00|0.00|98.70|0.00|1.30|68.63|1.18|0.00|1.72|`{1:320,2:806}`|
|3|1126|8.05|0.00|0.00|0.00|99.48|0.00|0.52|77.51|1.18|0.00|1.94|`{1:320,2:556,3:250}`|
|4|1126|8.05|0.00|0.00|0.00|99.48|0.00|0.52|78.15|1.18|0.00|1.95|`{1:320,2:556,3:232,4:18}`|
|5|1126|8.05|0.00|0.00|0.00|99.48|0.00|0.52|78.15|1.18|0.00|1.95|`{1:320,2:556,3:232,4:18}`|

#### 结论

OSPR-CI++实现和远端复评均完成，但结果仍是负证据：它能把未知拒识推到`99.48%`，同时把旧类压到约`8%`、seen-new压到`0%`，并且每类floor为`0%`。这证明当前决策层/证据层协同无法在现有EPOC B特征上同时满足旧类、新类和未知拒识目标。下一步主路线应转入底层训练：以`ADV3B02_CORE90_SOFT_E200`为教师，训练source-only负原型壳层/reciprocal prototype shell/低密度能量约束蒸馏模型，使星地信道扰动后的未知类特征远离已知/seen-new原型；真实`Y_unknown`仍只允许用于最终Stage2-C评估。
