# phase2_r7_qknn8_collab_20260706

## 基本信息

|字段|内容|
|---|---|
|experiment_id|phase2_r7_qknn8_collab_20260706|
|timestamp|2026-07-06 06:58 CST|
|operator|Codex|
|objective|对R7两个source-only地面训练候选执行Stage2-C基础qknn8协同推理复评，覆盖协同接收机数量M=1..5，并记录星地信道、时延和资源约束。|
|base candidates|`EPOC_R7_FLOOR_LOCKED_SHELL`、`EPOC_R7_BALANCED_LOW_DENSITY`|
|route|`stage2c_qknn8_collab_base_m1_to_all`|
|status|completed_negative_stage2c_base_qknn8|

## 协议边界

- Stage2-C输入由`export_spaceborne_features.py`从ManySig target-old和ManyTx target-new/unknown导出。
- `target_old`和`target_new`support/query来自目标接收机域`R_t={20-1,3-19,7-14,7-7,8-8}`，与source receivers不相交。
- `Y_unknown`只作为query评估，不参与地面训练、support、阈值拟合或early stopping。
- 复评使用`target_channel_view=satellite`和`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`，clean view不作为部署成功证据。
- 本轮为基础qknn8协同复评；若指标不达标，结论只能作为路线证据或下一步底层修复依据，不能写作部署成功。

## 候选与资源

|case|GPU|ckpt|feature_npz|output_json|log|
|---|---:|---|---|---|---|
|`EPOC_R7_FLOOR_LOCKED_SHELL`|0|`runs/phase1_epoc_r7_floor_protected_shell_20260706/EPOC_R7_FLOOR_LOCKED_SHELL/best_joint_safe_ssdg.pth`|`runs/phase2_r7_qknn8_collab_20260706/EPOC_R7_FLOOR_LOCKED_SHELL/features_stage2c_leo_multirx.npz`|`runs/phase2_r7_qknn8_collab_20260706/EPOC_R7_FLOOR_LOCKED_SHELL/qknn8_collab_base.json`|`logs/phase2_r7_qknn8_collab_20260706/EPOC_R7_FLOOR_LOCKED_SHELL.out`|
|`EPOC_R7_BALANCED_LOW_DENSITY`|1|`runs/phase1_epoc_r7_floor_protected_shell_20260706/EPOC_R7_BALANCED_LOW_DENSITY/best_joint_safe_ssdg.pth`|`runs/phase2_r7_qknn8_collab_20260706/EPOC_R7_BALANCED_LOW_DENSITY/features_stage2c_leo_multirx.npz`|`runs/phase2_r7_qknn8_collab_20260706/EPOC_R7_BALANCED_LOW_DENSITY/qknn8_collab_base.json`|`logs/phase2_r7_qknn8_collab_20260706/EPOC_R7_BALANCED_LOW_DENSITY.out`|

## 关键配置

|字段|值|
|---|---|
|source_tx_ids|`0,1,2,3,4,5`|
|source_rxs|`0,1,2,3,4,5,6`|
|target_receivers|`20-1,3-19,7-14,7-7,8-8`|
|target_new_tx_ids|`1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4`|
|unknown_tx_ids|`10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20`|
|proxy_unknown_tx_ids|`12-7,13-14,13-19,13-3,13-7,14-11,14-12,14-13`|
|k_shot/query_per_class/qknn_k|`8/20/8`|
|collab_counts|`all`，即M=1..5|
|collab_group_policy|`available_up_to_k`|
|event_alignment_policy|`receiver_domain_ranked`，无同事件key时作为显式数据集诊断口径|
|max_event_bytes/max_event_latency_ms/evidence_packet_bytes|`1152/20/40`|

## 本地变更与验证

|文件|目的|sha256|
|---|---|---|
|`E:\type10-7\code\scripts\launch_phase2_r7_qknn8_collab_20260706.sh`|导出R7 Stage2-C LEO多接收机features，并运行基础qknn8协同复评。|`362759EC319C1D6CB6D846EFCFE92553A6E1BF26DA84568462DCCEE6BDB0F9A3`|

Snapshot:

`E:\type10-7\code\snapshots\phase2_r7_qknn8_collab_20260706`

|命令|结果|
|---|---|
|`bash -n code/scripts/launch_phase2_r7_qknn8_collab_20260706.sh`|PASS|
|`bash code/scripts/launch_phase2_r7_qknn8_collab_20260706.sh --dry-run --only=FLOOR`|PASS，确认`collab_counts=all`、`qknn_k=8`、`unknown_query_eval_only=true`、`ground_training_unknown_seen=false`。|
|`conda run -n ssr-gpu python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\export_spaceborne_features.py`|PASS|

## 启动计划

|字段|内容|
|---|---|
|remote_root|`/home/szu2070436088/2510044040/CV-SincNet`|
|python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|launch command|`cd /home/szu2070436088/2510044040/CV-SincNet; bash code/scripts/launch_phase2_r7_qknn8_collab_20260706.sh`|
|expected outputs|每case生成`features_stage2c_leo_multirx.npz`、`qknn8_collab_base.json`、`qknn8_collab_base_evidence.csv`。|
|startup checks|检查log进入feature export或qknn eval；扫描Traceback、RuntimeError、CUDA OOM、unrecognized arguments和Killed。|
|success target|旧类99%且每类不低于95%；seen-new 97%且每类不低于93%；unknown rejection 99%。未同row达标时记录负证据。|

## R7前置完成证据

2026-07-06 06:53 CST只读监控显示两个R7训练进程和driver均退出，GPU0-7约10MiB；两个候选均到E200/200并导出`phase2_zid_prototypes.pt/json`。`FLOOR`最终best_epoch=194、best_score=86.0011、best_test_tx=90.1167；`BALANCED`best_epoch=90、best_score=85.8023、best_test_tx=90.0270。错误扫描未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。

## N607启动与完成证据

|字段|内容|
|---|---|
|sync verify|2026-07-06 06:58 CST同步launcher、report和`code/SYNC_MANIFEST.txt`到N607；远端`sha256sum`显示launcher hash为`362759ec319c1d6cb6d846efcfe92553a6e1bf26da84568462dccee6bdb0f9a3`；远端`bash -n`和dry-run PASS。|
|launch command|`cd /home/szu2070436088/2510044040/CV-SincNet; bash code/scripts/launch_phase2_r7_qknn8_collab_20260706.sh`|
|PIDs|`EPOC_R7_FLOOR_LOCKED_SHELL`:PID`3276253` on GPU0；`EPOC_R7_BALANCED_LOW_DENSITY`:PID`3276257` on GPU1。|
|startup/final health|2026-07-06 07:02 CST复查时两个PID已退出，GPU0-7约10MiB；两个case均生成`features_stage2c_leo_multirx.npz`、`qknn8_collab_base.json`、`qknn8_collab_base_evidence.csv`；日志含`[R7-QKNN8-COLLAB-DONE]`，错误扫描为空。|
|local result copies|`E:\type10-7\local_artifacts\phase2_r7_qknn8_collab_20260706_0703\FLOOR_qknn8_collab_base.json`、`E:\type10-7\local_artifacts\phase2_r7_qknn8_collab_20260706_0703\BALANCED_qknn8_collab_base.json`。|

## 结果表

|case|M|groups|actual_avg_rx|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|known_coverage|known_full_acc|known_accepted_acc|defer|bytes/event|p95 latency ms|budget violation|verdict|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|FLOOR|1|1122|1.000|14.77|0.00|0.00|0.00|35.38|2.82|8.74|7.10|81.25|59.98|40.0|0.988|0.00|fail|
|FLOOR|2|1122|1.717|14.20|0.00|0.00|0.00|21.28|1.54|6.97|6.83|98.04|74.33|68.7|0.988|0.00|fail|
|FLOOR|3|1122|1.943|14.20|0.00|0.00|0.00|20.51|1.54|6.97|6.83|98.04|75.58|77.7|0.988|0.00|fail|
|FLOOR|4|1122|1.961|14.20|0.00|0.00|0.00|19.74|1.54|6.97|6.83|98.04|75.85|78.4|0.988|0.00|fail|
|FLOOR|5|1122|1.961|14.20|0.00|0.00|0.00|19.74|1.54|6.97|6.83|98.04|75.85|78.4|0.988|0.00|fail|
|BALANCED|1|1121|1.000|21.53|0.00|0.00|0.00|38.28|4.69|11.40|10.31|90.48|56.91|40.0|1.093|0.00|fail|
|BALANCED|2|1121|1.713|19.83|0.00|0.00|0.00|22.66|2.08|9.77|9.50|97.22|69.85|68.5|1.093|0.00|fail|
|BALANCED|3|1121|1.941|19.83|0.00|0.00|0.00|21.35|2.08|9.77|9.50|97.22|70.56|77.6|1.093|0.00|fail|
|BALANCED|4|1121|1.963|19.83|0.00|0.00|0.00|21.61|2.08|9.77|9.50|97.22|70.65|78.5|1.093|0.00|fail|
|BALANCED|5|1121|1.963|19.83|0.00|0.00|0.00|21.61|2.08|9.77|9.50|97.22|70.65|78.5|1.093|0.00|fail|

## 解释

- 资源约束不是失败原因：所有row的`budget violation`均为0，`p95 latency`约0.99-1.09ms，`bytes/event`最高约78.5，低于`max_event_bytes=1152`和`max_event_latency_ms=20`。
- 基础协同推理没有带来性能增益：M从1增至5后，旧类准确率没有提升，seen-new准确率仍为0，unknown reject从M=1的35.38%/38.28%下降到约19.74%/21.61%。
- 该结果与R7训练期proxy AUC低于随机的过程证据一致：R7没有修复LEO target feature geometry；qknn8协同在当前特征空间中只是更保守地产生defer，并未形成可用的新类或未知类边界。
- 下一步不应继续只调基础qknn8协同阈值；应转向更底层的表示学习/蒸馏路线，例如用ADV3B02指导的星地信道特征分离蒸馏、source-only open-set SSL、或显式旧类floor保护的新模型重训。真实`Y_unknown`仍不能进入地面训练。
