# phase2_r8_qknn8_riskgate_20260706

|字段|内容|
|---|---|
|experiment_id|phase2_r8_qknn8_riskgate_20260706|
|timestamp|2026-07-06 07:55 CST|
|operator|Codex|
|objective|在R8 PAOG proxy激活后出现早期负趋势时，提前准备一个不改地面训练、不接触真实未知类的Stage2-C风险门控复评入口，用已有qknn8协同评估器叠加Mahalanobis/EVT/virtual/class-shell风险，检验是否能在保护旧类的同时降低unknown误收。|
|base candidates|`EPOC_R8_PAOG_RADIUS_ENERGY`、`EPOC_R8_PAOG_SHELL_BALANCED`|
|status|local_prepared_not_launched|

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
|virtual/class-shell|`virtual_unknown_*`、`class_negative_risk_enabled`、`class_shell_unknown_risk_enabled`|只用support/源域派生虚拟边界，避免真实未知训练泄漏。|
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
|`bash code/scripts/launch_phase2_r8_qknn8_riskgate_20260706.sh --dry-run --only=RADIUS`|PASS；输出包含`unknown_query_eval_only=true`、`stage2_success_claim=0`、`deployment_success_claim=0`、`support_envelope_full`、`old_protected_unknown_confirm_cvs`。|
|`conda run -n ssr-gpu python -m py_compile code/scripts/phase2_collaborative_open_set_qknn_eval.py code/evaluation/collaborative_open_set_qknn_eval.py`|PASS；首次使用Linux式环境变量前缀在PowerShell中失败，已按PowerShell env语法重跑通过。|
|N607 preflight|PASS；直连`N607`可用，项目根目录可见，GPU0/1有R8训练进程，GPU2-7低占用。|
|N607 sync|已同步launcher、report、snapshot和`code/SYNC_MANIFEST.txt`。|
|N607 hash|launcher与snapshot hash均为`4f098d5a4a212246d2b5f57d5157d7833c7fa85d839c1173c3c27f017dd9513b`；report hash为`eecafb9c6f35c3dfb875ebc14a3417c66cc994743fdb55c94d0a59c7c9aedaa3`。|
|N607 remote verify|远端`bash -n`PASS，远端dry-run字段PASS，远端`py_compile`PASS。|
|SSH cleanup|同步和验证后本地`ssh.exe=0`，N607/bridge 22端口ESTABLISHED连接均为0。|

当前未启动该评估。正式运行应等待R8训练完成或明确选择当前best checkpoint作为诊断复评。
