# Phase2 ADV3B02 SCG-qKNN协同开集诊断报告

## 基本信息

|字段|内容|
|---|---|
|实验ID|`phase2_adv3b02_scg_qknn_20260704`|
|时间|2026-07-04|
|操作者|Codex|
|目标|在`ADV3B02_CORE90_SOFT_E200`地面训练模型和qknn8少样本在轨适应基础上，验证新实现的`scg_qknn_cvs`是否能改善卫星群协同open-set识别。|
|底座权重|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|特征包|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`|
|远端环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|目标receiver domain|`20-1,3-19,7-14,7-7,8-8`|
|协同数量|`1..5`|
|星地信道|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|事件对齐边界|`receiver_domain_ranked`，不是strict same-event协同。|
|资源约束说明|按当前工作区文件名和关键词未找到`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`；本轮先报告`participating_receivers`、`bytes/event`、`latency_ms_p95`、GPU显存和状态大小代理字段。|

## 假设

既有ADV3B02诊断中，`dualroute_noguard`能保留较高known识别，但unknown FAR偏高；pairguard和class-negative路线能压低FAR，但会误伤known。`SCG-qKNN`使用支持集确认的known保护和unknown多源证据分离，可能在不使用unknown query调阈值的条件下给出更好的known/unknown折中。

## 协议与成功边界

本轮沿用`项目.md`：`R_t`与`R_s`不相交，`Y_old/Y_new/Y_unknown`互斥，unknown query只用于评估，不参与阈值拟合。成功必须同一行同时满足：

|指标|目标|
|---|---:|
|old_acc|>=0.99|
|min_old_class_acc|>=0.95|
|seen_new_acc|>=0.97|
|min_seen_new_class_acc|>=0.93|
|unknown_reject_rate|>=0.99|
|unknown_FAR|<=0.01|

若未同时满足，结果只作为diagnostic-only。

## 本地状态

当前工作树代码已包含`scg_qknn_cvs`策略和单元测试。计划先在本地/Git镜像检查，再同步或复用N607已同步代码运行。

## 计划远程命令

```bash
cd /tmp &&
PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code \
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --output_json /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_scg_qknn_20260704/collab_open_set_qknn_scg_adv3b02_qknn8_20260704.json \
  --output_evidence_csv /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_scg_qknn_20260704/collab_open_set_qknn_scg_adv3b02_qknn8_20260704_evidence.csv \
  --collab_counts all --event_alignment_policy receiver_domain_ranked \
  --fusion_policy scg_qknn_cvs --label_fusion_policy weighted_vote_margin \
  --receiver_class_reliability_policy support_calibrated --receiver_reliability_policy deployment_prior \
  --receiver_selection_policy fixed_receiver_order --support_selection_policy stable_first \
  --unknown_gate_mode support_envelope_evt --qknn_k 8 --k_shot 8 --query_per_class 20 \
  --prototype_score_blend 2.0 --mahalanobis_score_blend 1.0 --mahalanobis_score_temperature 0.2 \
  --class_conformal_enabled --class_conformal_min_support 2 --class_evidence_top_m 3 \
  --scenario_aware --radius_norm 0.3 \
  --unknown_risk_threshold 0.80 --accept_margin_threshold 0.10 --consensus_score_threshold 0.30 \
  --scorer_component_vote_threshold 0.50 \
  --candidate_set_min_receivers 2 --candidate_set_min_top1_receivers 2 \
  --candidate_set_min_conformal_pvalue 0.30 --candidate_set_min_label_receiver_class_reliability 0.75 \
  --candidate_set_max_label_risk_component_agreement 0.625 \
  --candidate_set_max_label_shell_risk 0.80 --candidate_set_shell_reject_risk 0.85 \
  --candidate_set_unknown_reject_risk 0.85 \
  --candidate_set_max_receiver_pair_label_disagreement 0.80 \
  --candidate_set_max_receiver_pair_unknown_risk_range 1.00 \
  --old_gate_min_support_density 0.50 --seen_new_gate_min_support_density 0.50 \
  --include_event_results
```

## 待回填

## N607执行记录

|项目|记录|
|---|---|
|N607 preflight|2026-07-04 07:05:24 CST通过；项目根可见；8张RTX3090均为`10/24576MiB`。|
|远端环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。|
|远端代码测试|`test_collaborative_open_set_qknn_eval.py`为65 tests OK；`test_phase2_collaborative_open_set_qknn_eval.py`为50 tests OK。|
|远端代码hash|`collaborative_open_set_qknn_eval.py=445bc2e803e7feef248dc60bcff964a32617364cad32c4455d9619601654114e`；`phase2_collaborative_open_set_qknn_eval.py=d0623fd37f34b98ed284c407b94f8721a1f505a3d5a4e16863b50cebb5e5202b`。|
|特征包hash|`features.npz=db559d78db305894307851750ef7d698db387f0984ff13c980fea99db85b8532`。|
|运行后GPU|8张RTX3090均为`10/24576MiB`，无持续显存占用。|

输出：

|产物|远端路径|SHA256|
|---|---|---|
|JSON|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_scg_qknn_20260704/collab_open_set_qknn_scg_adv3b02_qknn8_20260704.json`|`258c0bb8d4d86404bf6fdac1dc00733f17795fb8ffe64a049fa91f9c9c0687d7`|
|CSV|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_scg_qknn_20260704/collab_open_set_qknn_scg_adv3b02_qknn8_20260704_evidence.csv`|`de67f21298ab55cf8c8c5ee434d3f7020e8539e322a2f981d48271bd0f003385`|
|log|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_scg_qknn_20260704/scg_adv3b02_eval.out`|已拉回。|

本地产物目录：`E:\type10-7\remote_artifacts\phase2_adv3b02_scg_qknn_20260704\`

## 结果表

|k|total|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|defer|request_more|known_cov|scg_accept|support_protected|bytes/event|p95 ms|判定|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|307|0.4492|0.1000|0.2833|0.0500|0.6167|0.3333|0.1205|0.0000|0.5466|155|0|40.0|0.1564|FAR过高|
|2|250|0.3816|0.0000|0.2885|0.2812|0.6957|0.1304|0.3040|0.0000|0.3725|82|82|80.0|0.1564|FAR过高且known低|
|3|200|0.4750|0.1000|0.6500|0.6500|0.9750|0.0250|0.1100|0.0000|0.5312|86|86|120.0|0.1564|本轮最佳FAR/seen-new折中，但old远低于目标|
|4|150|0.7045|0.0000|0.4286|0.3500|0.9118|0.0294|0.1267|0.0000|0.6379|75|75|160.0|0.1564|old较高但seen-new/unknown不足|
|5|93|0.6792|0.0000|0.4000|0.0000|0.9000|0.1000|0.1505|0.0000|0.6027|46|46|200.0|0.1564|全receiver子集缺类/性能不足|

## 解释

1.`SCG-qKNN`在ADV3B02上优于SA33的known表现，但没有超过既有ADV3B02最佳诊断路线：`dualroute_noguard`的k=4曾达到`old_acc=0.8083`、`seen_new_acc=0.7500`、`unknown_FAR=0.0250`；本轮SCG k=4为`old_acc=0.7045`、`seen_new_acc=0.4286`、`unknown_FAR=0.0294`。
2.本轮最好open-set折中是k=3：`unknown_FAR=0.0250`且`unknown_reject=0.9750`，但`old_acc=0.4750`，不能作为OLD80_FIRST阶段推进依据。
3.k=1/k=2仍说明单节点和低协同数量下unknown证据不足；k=5受可用事件/类覆盖影响，不能解释为更多receiver必然提升。
4.资源上，证据包通信量按`40*k bytes/event`增长，p95融合延迟约`0.1564ms`，显存占用未增加；工程可运行，但算法指标未达标。

## 结论

本轮完成了`ADV3B02_CORE90_SOFT_E200 + qknn8 + scg_qknn_cvs`在N607`CVS-RFFI`环境下的k=1..5全量诊断。结果仍为diagnostic-only，不能声明旧类99%/每类95%、新类97%/每类93%或未知拒识99%。当前证据继续支持前序判断：瓶颈不是协同框架可运行性，而是ADV3B02特征空间中unknown与部分known/new的open-set风险不可分；下一步需要节点级独立open-set风险通道或轻量在轨adapter/prototype更新，而不是只改融合阈值。

## 待回填

## 持久化与断连检查

|项目|结果|
|---|---|
|本地报告|`E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_scg_qknn_20260704\report.md`|
|远程报告|`/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_scg_qknn_20260704/report.md`|
|Git镜像提交|`fd2ac17 Add ADV3B02 SCG qKNN diagnostic report`|
|SSH断连|最终检查：`ssh_processes=none`，`n607_established=none`，`bridge_established=none`。|
