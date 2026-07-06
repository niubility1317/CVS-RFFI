# phase2_r8_r9_r10_qknn8_collab_20260706

## Objective

在R8/R9/R10全部完成并导出`phase2_zid_prototypes.pt/json`后，执行最终Stage2-C qknn8协同开集评估。评估覆盖R8/R9/R10六个候选，协同接收机数量按严格`exact_k`口径从`M=1`到全部5个目标接收机，目标视图为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。

## Protocol

|item|value|
|---|---|
|stage|Stage2-C old + seen-new enrollment|
|base/backbones|R8/R9/R10 source-only训练候选|
|few-shot head|qknn8|
|target receivers|`20-1,3-19,7-14,7-7,8-8`|
|source receivers|`1-1,1-19,14-7,18-2,19-2,2-1,2-19`；launcher内部索引`0,1,2,3,4,5,6`|
|K-shot|8|
|target-old TX|`14-10,14-7,20-15,20-19,6-15,8-20`；launcher内部索引`0,1,2,3,4,5`|
|target-new TX|`1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4`|
|unknown TX|`10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20`|
|unknown query use|eval-only|
|proxy unknown calibration|disabled；launcher不传`--proxy_unknown_tx_ids`|
|collab counts|all，`--collab_group_policy exact_k`，等价于严格`M=1..5`；高效预算语义`available_up_to_k`不用于本次基础全量测试|
|resource proxy fields|`max_event_bytes=1152`、`max_event_latency_ms=20`、`evidence_packet_bytes=40`|

## Protocol Audit

|item|evidence|verdict|
|---|---|---|
|`R_s/R_t` disjoint|`R_s={1-1,1-19,14-7,18-2,19-2,2-1,2-19}`；`R_t={20-1,3-19,7-14,7-7,8-8}`|pass|
|`Y_old/Y_new/Y_unknown`互斥|`Y_old={14-10,14-7,20-15,20-19,6-15,8-20}`；`Y_new={1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4}`；`Y_unknown={10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20}`|pass|
|真实unknown训练/阈值使用|Phase1候选为source-only；本launcher不传`--proxy_unknown_tx_ids`；评估器help声明unknown query rows are evaluation-only and never used to set thresholds|pass|
|support/query|`K=8`，`query_per_class=20`；target-old和seen-new support/query均来自`R_t`；unknown只进入query|remote export/eval后补充每receiver/每TX样本覆盖|
|LEO部署主视图|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`，`star_ground_channel_impl=simplified_leo_residual`|pass|
|协同数量|`--collab_counts all --collab_group_policy exact_k`|pass，严格`M=1..5`|

## Cases

|case|GPU|checkpoint|
|---|---:|---|
|`R8_RADIUS`|0|`runs/phase1_epoc_r8_paog_20260706/EPOC_R8_PAOG_RADIUS_ENERGY/best_joint_safe_ssdg.pth`|
|`R8_SHELL`|1|`runs/phase1_epoc_r8_paog_20260706/EPOC_R8_PAOG_SHELL_BALANCED/best_joint_safe_ssdg.pth`|
|`R9_ANCHOR`|2|`runs/phase1_epoc_r9_source_anchor_20260706/EPOC_R9_ANCHOR_NOPROXY/best_joint_safe_ssdg.pth`|
|`R9_GENTLE`|3|`runs/phase1_epoc_r9_source_anchor_20260706/EPOC_R9_GENTLE_VIRTUAL_LATE/best_joint_safe_ssdg.pth`|
|`R10_BOUNDARY`|4|`runs/phase1_epoc_r10_source_boundary_20260706/EPOC_R10_BOUNDARY_NOPROXY/best_joint_safe_ssdg.pth`|
|`R10_GENTLE`|5|`runs/phase1_epoc_r10_source_boundary_20260706/EPOC_R10_GENTLE_VOS_LATE/best_joint_safe_ssdg.pth`|

## Success Boundary

只有同一candidate在同一row中同时达到旧类99%、每类旧类不低于95%、seen-new97%、每类seen-new不低于93%、unknown拒识99%，才可声明目标完成。若未达标，本评估作为底层表征修复路线的直接反馈，不得写成部署成功。

## Local Verification

|time|command|result|
|---|---|---|
|2026-07-06 15:xx CST|`bash -n code/scripts/launch_phase2_r8_r9_r10_qknn8_collab_20260706.sh`|pass|
|2026-07-06 15:xx CST|`bash code/scripts/launch_phase2_r8_r9_r10_qknn8_collab_20260706.sh --dry-run --only=R10_GENTLE`|pass；prints strict`exact_k`、qknn8、unknown eval-only、LEO scenarios、GPU5、log/out_dir|
|2026-07-06 15:xx CST|`conda run -n ssr-gpu python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\export_spaceborne_features.py code\tests\test_phase2_r8_r9_r10_qknn8_collab_launcher.py`|pass|
|2026-07-06 15:xx CST|`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 CONDA_NO_PLUGINS=true conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase2_r8_r9_r10_qknn8_collab_launcher.py -q`|3 passed；only `.pytest_cache` permission warning|
|2026-07-06 15:xx CST|`bash -n code/scripts/launch_phase2_r8_r9_r10_qknn8_collab_budget_20260706.sh`|pass|
|2026-07-06 15:xx CST|`bash code/scripts/launch_phase2_r8_r9_r10_qknn8_collab_budget_20260706.sh --dry-run --only=R10_GENTLE`|pass；eval-only，复用strict exact run已导出的`features_stage2c_leo_multirx.npz`|
|2026-07-06 15:xx CST|`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 CONDA_NO_PLUGINS=true conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase2_r8_r9_r10_qknn8_collab_launcher.py code\tests\test_phase2_r8_r9_r10_qknn8_collab_budget_launcher.py -q`|6 passed；only `.pytest_cache` permission warning|

Note:一次普通`conda run` pytest因为Windows/GBK转发触发`UnicodeEncodeError`，不是项目测试失败；一次并行`conda run` help触发已知临时文件锁，之后按串行规则重跑通过。

## Pending Remote Evidence

同步到N607后需补齐：远端hash、远端`bash -n`、远端dry-run、远端`py_compile`、GPU/显存占用、启动命令、PID、日志路径、结果JSON/CSV路径、每candidate的`M=1..5`同row指标、`actual_receiver_count_histogram`、`unknown_query_eval_only`、`threshold_selection_label_scope`、时延/bytes/resource violation字段。

## Remote Launch Notes

|time|event|evidence|
|---|---|---|
|2026-07-06 15:00 CST|N607 preflight|direct N607 PASS；project root visible；GPU0-7 all about10MiB|
|2026-07-06 15:00 CST|remote sync/verify|launcher/test/snapshot/report synced；remote sha256 matched local hashes；remote`bash -n`、dry-run和`py_compile` PASS|
|2026-07-06 15:00 CST|strict exact launch|six candidates launched on GPU0-5: R8_RADIUS PID3500871, R8_SHELL PID3500875, R9_ANCHOR PID3500879, R9_GENTLE PID3500883, R10_BOUNDARY PID3500887, R10_GENTLE PID3500891|
|2026-07-06 15:02 CST|strict exact result|all six exported`features_stage2c_leo_multirx.npz` but failed in qknn eval with`ValueError: no evidence groups contain 5 receiver observations`|

Interpretation:严格`exact_k`同事件5接收机融合不可用于当前Stage2-C特征证据，因为没有5接收机共同观测的evidence group。这不是训练失败，也不是GPU问题；它是数据/事件对齐口径不满足。保留该失败作为协议边界证据。为继续满足“协同数量从1到全体接收机数量可选”的部署评估需求，新增`phase2_r8_r9_r10_qknn8_collab_budget_20260706`预算协同eval-only脚本，使用`--collab_group_policy available_up_to_k`并在结果中报告`actual_receiver_count_histogram`、`partial_group_count`和`exact_budget_group_count`，避免把预算M误写成严格同事件M。

## Budget Collaboration Result

预算协同评估已完成，六个候选均生成`qknn8_collab_budget.json`和`qknn8_collab_budget_evidence.csv`，日志无Traceback/RuntimeError/OOM/参数错误。结果复制到本地`local_artifacts/phase2_r8_r9_r10_qknn8_collab_budget_20260706/`，汇总CSV为`summary_budget_metrics.csv`。

协议字段：`stage2_success_claim=false`、`deployment_success_claim=false`、`unknown_query_eval_only=true`、`threshold_selection_label_scope=support_known_only`、`denominator_policy=per_k_available_receivers`、`collab_group_policy=available_up_to_k`。所有行`resource_budget_violation_rate=0`，但资源字段仍是offline proxy，不是真实星间链路实测。

|case|M预算|old_acc|min_old|seen_new|min_seen|unknown_reject|unknown_FAR|actual_rx_hist|exact_budget|partial|lat_p95_ms|bytes/event|resource_violation|
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
|`R10_BOUNDARY`|1|28.21%|0.00%|0.26%|0.00%|40.00%|10.00%|`{"1": 1132}`|1132|0|0.333|40.000|0.00%|
|`R10_BOUNDARY`|2|24.50%|0.00%|0.00%|0.00%|26.67%|2.82%|`{"1": 330, "2": 802}`|802|330|0.333|68.339|0.00%|
|`R10_BOUNDARY`|3|23.93%|0.00%|0.00%|0.00%|23.08%|2.56%|`{"1": 330, "2": 553, "3": 249}`|249|883|0.333|77.138|0.00%|
|`R10_BOUNDARY`|4|23.93%|0.00%|0.00%|0.00%|22.82%|2.56%|`{"1": 330, "2": 553, "3": 232, "4": 17}`|17|1115|0.333|77.739|0.00%|
|`R10_BOUNDARY`|5|23.93%|0.00%|0.00%|0.00%|22.82%|2.56%|`{"1": 330, "2": 553, "3": 232, "4": 17}`|0|1132|0.333|77.739|0.00%|
|`R10_GENTLE`|1|22.03%|0.00%|2.05%|0.00%|39.75%|9.87%|`{"1": 1140}`|1140|0|0.506|40.000|0.00%|
|`R10_GENTLE`|2|20.06%|0.00%|0.00%|0.00%|26.58%|4.81%|`{"1": 348, "2": 792}`|792|348|0.506|67.789|0.00%|
|`R10_GENTLE`|3|20.06%|0.00%|0.00%|0.00%|24.05%|4.81%|`{"1": 348, "2": 548, "3": 244}`|244|896|0.506|76.351|0.00%|
|`R10_GENTLE`|4|20.06%|0.00%|0.00%|0.00%|24.05%|4.81%|`{"1": 348, "2": 548, "3": 220, "4": 24}`|24|1116|0.506|77.193|0.00%|
|`R10_GENTLE`|5|20.06%|0.00%|0.00%|0.00%|24.05%|4.81%|`{"1": 348, "2": 548, "3": 220, "4": 24}`|0|1140|0.506|77.193|0.00%|
|`R8_RADIUS`|1|31.23%|0.00%|7.36%|0.00%|39.95%|20.37%|`{"1": 1126}`|1126|0|0.425|40.000|0.00%|
|`R8_RADIUS`|2|28.65%|0.00%|3.81%|0.00%|22.72%|7.57%|`{"1": 329, "2": 797}`|797|329|0.425|68.313|0.00%|
|`R8_RADIUS`|3|27.22%|0.00%|3.55%|0.00%|20.89%|5.74%|`{"1": 329, "2": 545, "3": 252}`|252|874|0.425|77.265|0.00%|
|`R8_RADIUS`|4|27.22%|0.00%|3.55%|0.00%|20.89%|5.74%|`{"1": 329, "2": 545, "3": 227, "4": 25}`|25|1101|0.425|78.153|0.00%|
|`R8_RADIUS`|5|27.22%|0.00%|3.55%|0.00%|20.89%|5.74%|`{"1": 329, "2": 545, "3": 227, "4": 25}`|0|1126|0.425|78.153|0.00%|
|`R8_SHELL`|1|35.41%|5.00%|2.37%|0.00%|40.94%|16.80%|`{"1": 1113}`|1113|0|0.276|40.000|0.00%|
|`R8_SHELL`|2|27.20%|0.00%|0.79%|0.00%|26.77%|7.61%|`{"1": 311, "2": 802}`|802|311|0.276|68.823|0.00%|
|`R8_SHELL`|3|27.20%|0.00%|0.53%|0.00%|23.62%|6.30%|`{"1": 311, "2": 541, "3": 261}`|261|852|0.276|78.203|0.00%|
|`R8_SHELL`|4|27.20%|0.00%|0.53%|0.00%|23.62%|6.56%|`{"1": 311, "2": 541, "3": 237, "4": 24}`|24|1089|0.276|79.066|0.00%|
|`R8_SHELL`|5|27.20%|0.00%|0.53%|0.00%|23.62%|6.56%|`{"1": 311, "2": 541, "3": 237, "4": 24}`|0|1113|0.276|79.066|0.00%|
|`R9_ANCHOR`|1|23.01%|0.00%|1.01%|0.00%|33.42%|9.07%|`{"1": 1133}`|1133|0|0.410|40.000|0.00%|
|`R9_ANCHOR`|2|21.31%|0.00%|0.51%|0.00%|24.87%|3.37%|`{"1": 338, "2": 795}`|795|338|0.410|68.067|0.00%|
|`R9_ANCHOR`|3|21.31%|0.00%|0.76%|0.00%|23.06%|2.85%|`{"1": 338, "2": 546, "3": 249}`|249|884|0.410|76.858|0.00%|
|`R9_ANCHOR`|4|21.31%|0.00%|0.76%|0.00%|22.80%|2.85%|`{"1": 338, "2": 546, "3": 226, "4": 23}`|23|1110|0.410|77.670|0.00%|
|`R9_ANCHOR`|5|21.31%|0.00%|0.76%|0.00%|22.80%|2.85%|`{"1": 338, "2": 546, "3": 226, "4": 23}`|0|1133|0.410|77.670|0.00%|
|`R9_GENTLE`|1|26.06%|0.00%|1.04%|0.00%|34.20%|13.73%|`{"1": 1125}`|1125|0|0.522|40.000|0.00%|
|`R9_GENTLE`|2|26.35%|0.00%|0.00%|0.00%|22.54%|3.37%|`{"1": 324, "2": 801}`|801|324|0.522|68.480|0.00%|
|`R9_GENTLE`|3|25.50%|0.00%|0.00%|0.00%|20.47%|3.11%|`{"1": 324, "2": 549, "3": 252}`|252|873|0.522|77.440|0.00%|
|`R9_GENTLE`|4|25.50%|0.00%|0.00%|0.00%|20.21%|3.11%|`{"1": 324, "2": 549, "3": 230, "4": 22}`|22|1103|0.522|78.222|0.00%|
|`R9_GENTLE`|5|25.50%|0.00%|0.00%|0.00%|20.21%|3.11%|`{"1": 324, "2": 549, "3": 230, "4": 22}`|0|1125|0.522|78.222|0.00%|

## Interpretation

最佳同row结果是`R8_SHELL`的`M预算=1`：old_acc35.41%、min_old5.00%、seen_new2.37%、min_seen0.00%、unknown_reject40.94%、unknown_FAR16.80%。这与目标旧类99%/min95%、seen-new97%/min93%、unknown拒识99%相差极大。预算M从1增加到5没有带来性能增益，反而由于可对齐多接收机组稀少，effective evidence偏向低M/partial group，旧类、seen-new和unknown拒识整体下降。

结论：R8/R9/R10表征修复后，基础qknn8协同仍不能解决星地信道未知类拒识，同时旧类和seen-new也未达到OLD80_FIRST门槛。下一步不应继续只调协同融合参数；需要回到底层表征/蒸馏路线，优先约束未知类附近的已知类边界，包括ADV3B02指导的source-only代理未知蒸馏、source overflow压缩、低密度接受抑制、radius_to_inter_ratio提升和z_id tail angle收缩。真实未知类仍只能评估，不能进入地面训练或阈值选择。

## Phase1 Rejection-Metric Snapshot

最终epoch训练侧拒识相关指标如下。R8/R9/R10的source overflow仍高达84.01%到96.28%；带proxy/virtual分支的候选`proxy_vaccept`仍在81.39%到86.83%，`bridge_accept_rate=1.0`，说明源侧边界仍允许大量代理/桥接样本落入已知类接受区域。`zid_p95/p99`仍约51.40到52.94度/72.19到76.91度，tail较宽。该证据支持下一步转向ADV3B02指导的source-only蒸馏和边界塑形，而不是继续单独调协同融合。

|case|epoch|best_score|test_tx|proxy_vaccept|source_overflow|bridge_accept|low_density_accept|tail_accept_loss|overflow_accept_loss|radius_to_inter|zid_p50|zid_p95|zid_p99|zid_tail_cvar|proxy_auc|nonfinite_grad|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`R8_RADIUS`|200|84.8666|89.57%|82.37%|89.07%|100.00%|0.04%|4.0835|3.7633|0.9722|30.09|52.94|76.75|56.80|0.4035|1.0|
|`R8_SHELL`|200|85.3407|89.47%|81.39%|84.01%|100.00%|0.04%|5.0367|2.8563|1.0076|30.56|52.55|76.91|57.23|0.3862|1.0|
|`R9_ANCHOR`|200|85.7999|89.27%|n/a|93.59%|n/a|n/a|0.0000|0.0000|n/a|30.12|50.10|72.79|52.92|n/a|0.0|
|`R9_GENTLE`|200|85.9079|89.55%|81.99%|91.24%|100.00%|0.09%|2.7757|1.6666|0.9854|30.16|51.41|72.19|53.46|0.5192|0.0|
|`R10_BOUNDARY`|200|85.7483|90.04%|n/a|96.28%|n/a|n/a|0.0000|0.0000|n/a|30.16|52.58|75.77|52.65|n/a|0.0|
|`R10_GENTLE`|200|85.9245|89.58%|86.83%|95.14%|100.00%|0.08%|1.7547|1.6081|0.9351|29.85|51.40|74.48|52.05|0.5633|0.0|

Next-route implication:优先降低`proxy_vaccept/source_overflow/bridge_accept_rate`，同时不牺牲source/test TX准确率；使`radius_to_inter_ratio`稳定低于当前R8_SHELL的1.0076并压缩`zid_p95/p99/tail_cvar`。训练仍必须source-only，不得接触真实`Y_unknown`或目标接收机样本。
