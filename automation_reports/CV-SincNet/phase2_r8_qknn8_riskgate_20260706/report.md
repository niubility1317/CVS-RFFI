# phase2_r8_qknn8_riskgate_20260706

|字段|内容|
|---|---|
|experiment_id|phase2_r8_qknn8_riskgate_20260706|
|timestamp|2026-07-06 07:55 CST|
|operator|Codex|
|objective|在R8 PAOG proxy激活后出现早期负趋势时，提前准备一个不改地面训练、不接触真实未知类的Stage2-C风险门控复评入口，用已有qknn8协同评估器叠加Mahalanobis/EVT/virtual/class-shell风险，检验是否能在保护旧类的同时降低unknown误收。|
|base candidates|`EPOC_R8_PAOG_RADIUS_ENERGY`、`EPOC_R8_PAOG_SHELL_BALANCED`|
|status|completed_negative_non_deployment_diagnostic|

## 触发背景

R8 PAOG在07:46 CST进入proxy激活早期：

|candidate|epoch|proxy_auc|virtual_accept|soft_mix_accept|train_skipped_nonfinite_grad|结论|
|---|---:|---:|---:|---:|---:|---|
|`EPOC_R8_PAOG_RADIUS_ENERGY`|34/200|0.4033|0.8159|0.9995|1.0000|早期proxy分离弱且数值风险高。|
|`EPOC_R8_PAOG_SHELL_BALANCED`|32/200|0.3807|0.8094|1.0000|1.0000|连续E30-E32弱于随机分离。|

该证据不能证明R8最终失败，因为训练未完成、prototype未导出、真实`Y_unknown`仍未评估；但它说明当前“强proxy shell直接训练主干”的路线可能数值不稳。子agent和本地主线探查均建议把后续评估重心转为更稳的冻结特征风险门控：`qknn8 + prototype/Mahalanobis/EVT + virtual/class-shell risk + old-protected confirm`。

## 协议边界

- 本启动器只做Stage2-C评估入口，不做地面训练。
- `ManyTx.pkl`只作为Stage2-C target-new/unknown样本来源；真实`Y_unknown`只进入eval query。
- 阈值和风险门控来自target old/new support、source/proxy calibration、virtual unknown和class shell；不得用unknown query拟合阈值。
- 输出必须保持同row联合指标：old、seen-new、unknown reject/FAR、defer、资源字段，不得用单项最高值声明成功。
- 本报告不声明Stage2-C成功、部署成功或目标完成。

## 本地变更

|文件|目的|
|---|---|
|`E:\type10-7\code\scripts\launch_phase2_r8_qknn8_riskgate_20260706.sh`|新增R8完成后的Stage2-C qknn8风险门控协同评估启动器。|
|`E:\type10-7\automation_reports\CV-SincNet\phase2_r8_qknn8_riskgate_20260706\report.md`|记录算法动机、协议边界、验证与后续运行条件。|

## 方法配置

|模块|设置|目的|
|---|---|---|
|qknn8|`--qknn_k 8`、`--collab_counts all`、`M=1..target receiver count`|保持原目标要求的少样本部署头和协同数量扫描。|
|风险门控|`--unknown_gate_mode support_envelope_full`|联合score/radius/margin/Mahalanobis/EVT/oldness风险。|
|Mahalanobis/EVT|`mahalanobis_score_blend=0.35`、`mahalanobis_quantile=0.90`、`evt_tail_quantile=0.80`|以类条件分布边界替代R8不稳定的强训练壳层。|
|virtual/class-shell|`virtual_unknown_*`、`class_negative_risk_enabled`、`class_shell_unknown_risk_enabled`|只用support派生虚拟边界和class-shell风险，默认关闭真实`proxy_unknown` TX校准，避免“未知类不可见”叙事风险。|
|融合策略|`old_protected_unknown_confirm_cvs`|避免单纯提高拒识导致旧类和seen-new崩塌。|
|资源口径|`max_event_bytes=1152`、`max_event_latency_ms=20`、`evidence_packet_bytes=56`|继续记录代理资源预算，不能写成真实链路实测。|

## 运行条件

该启动器应在R8训练结束或至少目标checkpoint稳定后运行。若R8持续`train_skipped_nonfinite_grad=1.0`并且best checkpoint不再改进，可用当前`best_joint_safe_ssdg.pth`做诊断复评；否则优先等待训练完成。

预期远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
bash code/scripts/launch_phase2_r8_qknn8_riskgate_20260706.sh --dry-run
```

正式运行前仍需N607 preflight、local hash记录、scp同步、远端`bash -n`与dry-run验证。

## 成功/失败判据

|阶段|判据|
|---|---|
|local ready|`bash -n`、dry-run、`py_compile`通过。|
|Stage2-C useful|同一row至少先恢复`old_acc>=0.80`，再观察seen-new和unknown reject；未达OLD80不得声明主线成功。|
|目标完成|同row满足旧类99%且每类不低于95%，新类97%且每类不低于93%，未知拒识99%，资源代理预算通过。|
|失败|若old仍低于80%或unknown reject只能靠大规模defer/拒识换取，则该路线仍不成立。|

## 本地与远端验证

|检查|结果|
|---|---|
|`bash -n code/scripts/launch_phase2_r8_qknn8_riskgate_20260706.sh`|PASS|
|`bash code/scripts/launch_phase2_r8_qknn8_riskgate_20260706.sh --dry-run --only=RADIUS`|PASS；输出包含`unknown_query_eval_only=true`、`proxy_unknown_real_tx_calibration=0`、`stage2_success_claim=0`、`deployment_success_claim=0`、`support_envelope_full`、`old_protected_unknown_confirm_cvs`。|
|`conda run -n ssr-gpu python -m py_compile code/scripts/phase2_collaborative_open_set_qknn_eval.py code/evaluation/collaborative_open_set_qknn_eval.py`|PASS；首次使用Linux式环境变量前缀在PowerShell中失败，已按PowerShell env语法重跑通过。|
|N607 preflight|PASS；直连`N607`可用，项目根目录可见，GPU0/1有R8训练进程，GPU2-7低占用。|
|N607 sync|已同步launcher、report、snapshot和`code/SYNC_MANIFEST.txt`。|
|N607 hash|launcher与snapshot hash均为`4f098d5a4a212246d2b5f57d5157d7833c7fa85d839c1173c3c27f017dd9513b`；report hash为`eecafb9c6f35c3dfb875ebc14a3417c66cc994743fdb55c94d0a59c7c9aedaa3`。|
|N607 remote verify|远端`bash -n`PASS，远端dry-run字段PASS，远端`py_compile`PASS。|
|SSH cleanup|同步和验证后本地`ssh.exe=0`，N607/bridge 22端口ESTABLISHED连接均为0。|

## 2026-07-06 08:05 CST子agent review修正

合理性review指出：若默认把`ManyTx`中的额外真实非旧类作为`proxy_unknown`校准，会削弱“地面训练阶段不能接触未知类别”的课题叙事。该路径不等于使用target unknown query调阈值，但仍应避免作为默认主线。因此已修正启动器：

|修正项|结果|
|---|---|
|`PROXY_UNKNOWN_TX_IDS`默认值|改为空。|
|export命令|默认完全不传`--proxy_unknown_tx_ids`、`--proxy_unknown_rxs`、`--proxy_unknown_channel_view`、`--proxy_unknown_sat_scenarios`。|
|dry-run审计字段|新增`proxy_unknown_real_tx_calibration=0`。|
|新增测试|`code/tests/test_phase2_r8_qknn8_riskgate_launcher.py`确认默认不开启真实proxy unknown校准，并保留`unknown_query_eval_only=true`、`qknn_k=8`、`collab_counts all`、`support_envelope_full`。|

本地验证：

|检查|结果|
|---|---|
|`bash -n code/scripts/launch_phase2_r8_qknn8_riskgate_20260706.sh`|PASS|
|`bash code/scripts/launch_phase2_r8_qknn8_riskgate_20260706.sh --dry-run --only=RADIUS`|PASS；输出`proxy_unknown_real_tx_calibration=0`，且不含`--proxy_unknown_tx_ids`。|
|`conda run -n ssr-gpu python -m pytest code/tests/test_phase2_r8_qknn8_riskgate_launcher.py -q`|PASS，1 passed；仅有`.pytest_cache`权限警告。|
|`conda run -n ssr-gpu python -m py_compile code/scripts/phase2_collaborative_open_set_qknn_eval.py code/evaluation/collaborative_open_set_qknn_eval.py code/export_spaceborne_features.py`|PASS；第一次并行`conda run`遇到Windows临时文件锁，串行重跑通过。|

本地哈希：

|文件|SHA256|
|---|---|
|`code/scripts/launch_phase2_r8_qknn8_riskgate_20260706.sh`|`644C8682E05E3E9FD5A9E9802AE8DF2CF383A48865DCC9C1A81C62F9AB802BD6`|
|`code/tests/test_phase2_r8_qknn8_riskgate_launcher.py`|`3C1927DF114CF6B3B239F8F8DB49D56C9F1D9497A05E89777875FA49FCF41DC2`|
|`automation_reports/CV-SincNet/phase2_r8_qknn8_riskgate_20260706/report.md`|`80EFC2B1D5FE2E61BDC03ED4517FBDD8966D949BA08BF2288FA6CFDBBB8C9738`|

## 2026-07-06 08:00 CST启动前状态

用户目标要求服务器显卡存在其他低显存进程时仍启动实验。本轮N607 preflight通过，直连`N607`可用，项目根目录可见；GPU0/1运行R8 PAOG训练，显存约2.5GB/2.4GB，GPU2/3约10MiB且空闲，因此将Stage2-C风险门控评估分配到GPU2/3。

R8训练尚未完成，但`best_joint_safe_ssdg.pth`已稳定存在，且proxy激活后连续弱分离：

|candidate|远端状态|latest epoch|best epoch|best test_tx|latest proxy_auc|virtual_accept|proxy_accept|soft_mix_accept|nonfinite_grad|prototype export|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`EPOC_R8_PAOG_RADIUS_ENERGY`|running PID 3289110 GPU0|53/200|20|89.8152|0.4033|0.8196|0.0481|0.9995|1.0000|absent|
|`EPOC_R8_PAOG_SHELL_BALANCED`|running PID 3289536 GPU1|51/200|20|89.6681|0.3828|0.8061|0.0143|0.9990|1.0000|absent|

启动解释：本次不是宣布R8已完成，而是按“当前best checkpoint诊断复评”启动Stage2-C qknn8风险门控，以便在R8继续训练时并行获得早期真实`Y_unknown`评估证据。真实`Y_unknown`仍只用于query评估，不参与地面训练、阈值拟合或校准。根据子agent合理性review，启动器已改为默认`proxy_unknown_real_tx_calibration=0`，不再把`ManyTx`中的额外真实非旧类TX作为校准集；未知风险只来自support envelope、virtual unknown、class-negative和class-shell机制。

资源约束说明：未在本地找到精确文件名`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`，因此本run只记录launcher中的代理资源字段：`max_event_bytes=1152`、`max_event_latency_ms=20`、`evidence_packet_bytes=56`。这些字段不能写成真实星间链路实测。

正式远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
bash code/scripts/launch_phase2_r8_qknn8_riskgate_20260706.sh
```

预期输出：

|candidate|GPU|log|output json|evidence csv|
|---|---:|---|---|---|
|`EPOC_R8_PAOG_RADIUS_ENERGY`|2|`logs/phase2_r8_qknn8_riskgate_20260706/EPOC_R8_PAOG_RADIUS_ENERGY.out`|`runs/phase2_r8_qknn8_riskgate_20260706/EPOC_R8_PAOG_RADIUS_ENERGY/qknn8_riskgate.json`|`runs/phase2_r8_qknn8_riskgate_20260706/EPOC_R8_PAOG_RADIUS_ENERGY/qknn8_riskgate_evidence.csv`|
|`EPOC_R8_PAOG_SHELL_BALANCED`|3|`logs/phase2_r8_qknn8_riskgate_20260706/EPOC_R8_PAOG_SHELL_BALANCED.out`|`runs/phase2_r8_qknn8_riskgate_20260706/EPOC_R8_PAOG_SHELL_BALANCED/qknn8_riskgate.json`|`runs/phase2_r8_qknn8_riskgate_20260706/EPOC_R8_PAOG_SHELL_BALANCED/qknn8_riskgate_evidence.csv`|

## 2026-07-06 08:06 CST完成结果

远端运行已完成，两条case日志均包含`R8-QKNN8-RISKGATE-DONE`，错误扫描未发现`Traceback`、`RuntimeError`、`CUDA out of memory`、`unrecognized arguments`或`Killed`。结果已复制到本地`E:\type10-7\local_artifacts\phase2_r8_qknn8_riskgate_20260706_0806`。

协议审计：

|字段|结果|
|---|---|
|`verdict_scope`|`NON_DEPLOYMENT_DIAGNOSTIC`|
|`stage2_success_claim` / `deployment_success_claim`|均为`false`|
|`target_pass_count`|0|
|`unknown_query_eval_only`|`true`|
|`threshold_selection_label_scope`|`support_virtual_unknown`|
|`qknn_k` / `k_shot`|8 / 8|
|`target_receiver_count`|5，覆盖`M=1..5`|
|`target_channel_view`|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|`prototype_storage_bytes`|91840|
|`evidence_bytes_per_receiver_event`|56|

同row指标如下。百分比为JSON原值乘100；`min_old`与`min_seen`为每类最低准确率；`FAR=1-unknown_reject-defer`的评估口径来自JSON中的`unknown_FAR`。

|candidate|M|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|defer|known_coverage|known_accepted_acc|latency_p95_ms|bytes/event|resource_violation|verdict|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|RADIUS|1|16.62|0.00|0.00|0.00|95.30|4.70|0.00|14.48|77.78|0.30|56.00|0.00|FAIL|
|RADIUS|2|26.36|0.00|5.58|0.00|87.73|9.79|2.40|22.95|80.95|0.30|109.63|0.00|FAIL|
|RADIUS|3|27.79|0.00|6.85|0.00|84.60|12.79|2.40|24.45|80.22|0.30|110.55|0.00|FAIL|
|RADIUS|4|27.79|0.00|7.61|0.00|84.07|13.32|2.40|25.27|80.46|0.30|110.55|0.00|FAIL|
|RADIUS|5|27.79|0.00|7.61|0.00|84.07|13.32|2.40|25.27|80.46|0.30|110.55|0.00|FAIL|
|SHELL|1|17.00|0.00|0.00|0.00|95.01|4.99|0.00|15.03|78.18|0.30|56.00|0.00|FAIL|
|SHELL|2|29.18|1.72|6.07|0.00|88.71|9.84|1.98|23.54|84.30|0.30|110.15|0.00|FAIL|
|SHELL|3|31.16|1.72|9.76|0.00|86.35|11.02|1.98|25.27|80.46|0.30|110.69|0.00|FAIL|
|SHELL|4|31.16|1.72|10.03|0.00|85.83|11.02|1.98|25.27|80.00|0.30|110.69|0.00|FAIL|
|SHELL|5|31.16|1.72|10.03|0.00|85.83|11.02|1.98|25.27|80.00|0.30|110.69|0.00|FAIL|

解释：

- 相比R7 base qknn8，风险门控显著提高unknown拒识，M=1约95%，但旧类和seen-new仍远低于OLD80_FIRST和最终目标。
- 协同数量从1增至5没有带来目标级改善；SHELL在M=3..5旧类约31%、seen-new约10%，unknown拒识反而低于M=1。
- `known_accepted_acc`约78%到84%说明被接受的少量已知样本相对纯净，但`known_coverage`只有约14%到25%，大量旧类/新类被拒识。该路线不能作为部署成功。
- 结果支持当前判断：仅靠协同推理和风险门控不能达成目标，下一步应继续R8训练到完成并评估最终checkpoint；若仍失败，应转向更底层的source-only特征分离或轻量reject head训练，且真实未知类不得进入地面训练。
