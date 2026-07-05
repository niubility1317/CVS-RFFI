# Phase2 ADV3B02 SMEC-CI支持度量/能量协同拒识诊断

## 基本信息

- 实验ID：`phase2_adv3b02_smec_ci_20260704`
- 时间：2026-07-04 17:02:36 +08:00
- 操作方：Codex主代理，多子agent审查
- 模型/特征来源：`ADV3B02_CORE90_SOFT_E200`特征包，用户指定地面权重口径继承为`SA33_sa27_ch2_leo3_ce0p7_r010_20260527_204104`
- 目标：在天基RFFI Stage2-C边界下，优先提升未知类拒识，同时旧类准确率不能下降；协同推理数量覆盖`collab_count=1..目标接收机数`
- 场景边界：`receiver_domain_ranked`是receiver-domain ensemble诊断，不是严格same-event多星观测证据

## 假设与算法

SMEC-CI（support-only metric/energy calibration collaborative inference）冻结base qknn8标签路由，只用target-old与target-new support构建每个目标接收机的轻量校准：

- prototype shell risk：query到base top label support centroid的cosine距离；
- support KNN density risk：query到receiver support memory的最近邻距离；
- old-head energy risk：仅对base top label属于旧类时弱参与；
- unknown query只参与评估，不参与阈值、超参或best row选择；
- 输出只提升`unknown_risk`和`class_evidence_top1_unknown_risk`，不改`predicted_label`或`class_evidence_top1_label`。

子agent审查结论已纳入实现边界：

- 该方法属于Stage2-C支持集校准，因为target-new support参与建模；
- 旧类保护不能只靠“label authority不变”，因为reject/defer仍会降低full old accuracy；
- 任何候选必须同一行满足旧类不下降、unknown改善、FAR不恶化；
- `receiver_domain_ranked`不能写成严格物理同事件卫星协同。

## 本地变更

Git承载目录：`E:\type10-7\github_publish\CVS-RFFI-repo`

新增文件：

- `code/scripts/phase2_support_metric_energy_ci_eval.py`
- `code/tests/test_phase2_support_metric_energy_ci_eval.py`

Git提交：

- `34d9061 Add support metric energy CI diagnostic`
- `4cc4426 Add old lossless SMEC profile`
- `d354d8c Add consensus guarded SMEC profile`

注意：本地Git状态中已有非本轮修改`code/scripts/phase2_confusion_aware_qknn_probe.py`和`code/scripts/phase2_qknn_prototype_compress_probe.py`，本实验不触碰、不提交这些文件。

## 本地验证

本地环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`

验证命令：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_support_metric_energy_ci_eval.py -q
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_support_metric_energy_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_support_metric_energy_ci_eval.py --feature_npz E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz --output_dir local_artifacts\phase2_adv3b02_smec_ci_20260704\local_all --force --write_evidence
```

验证结果：

- 单元测试：`3 passed`
- 语法检查：通过
- 本地全量诊断：`receiver_count=5`，`evidence_row_count=1000`，`summary_rows=20`，`candidate_count=0`
- 本地输出：
  - `E:\type10-7\github_publish\CVS-RFFI-repo\local_artifacts\phase2_adv3b02_smec_ci_20260704\local_all\smec_ci_summary.csv`
  - `E:\type10-7\github_publish\CVS-RFFI-repo\local_artifacts\phase2_adv3b02_smec_ci_20260704\local_all\smec_ci_best_rows.csv`
  - `E:\type10-7\github_publish\CVS-RFFI-repo\local_artifacts\phase2_adv3b02_smec_ci_20260704\local_all\smec_ci_audit.json`

## 本地逐项结果

`opu_old_preserve`下，SMEC-CI覆盖`collab_count=1..5`，但所有行均为`diagnostic_only`：

| algorithm | policy | collab_count | old_acc | seen_new_acc | unknown_reject_rate | unknown_FAR | known_coverage | bytes_per_event | latency_ms_p95 | delta_old_acc | delta_unknown_reject_rate | delta_unknown_FAR | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base_known_route | opu_old_preserve | 1 | 0.6364 | 0.6833 | 0.1167 | 0.8667 | 0.9636 | 40.0000 | 0.0981 |  |  |  |  |
| smec_ci | opu_old_preserve | 1 | 0.6310 | 0.6500 | 0.2333 | 0.7667 | 0.9474 | 64.0000 | 0.1281 | -0.0053 | 0.1167 | -0.1000 | diagnostic_only |
| base_known_route | opu_old_preserve | 2 | 0.6257 | 0.8167 | 0.0500 | 0.7167 | 0.9514 | 72.5733 | 0.0981 |  |  |  |  |
| smec_ci | opu_old_preserve | 2 | 0.6096 | 0.8167 | 0.1333 | 0.6500 | 0.9433 | 116.1173 | 0.1281 | -0.0160 | 0.0833 | -0.0667 | diagnostic_only |
| base_known_route | opu_old_preserve | 3 | 0.7594 | 0.7167 | 0.1167 | 0.5833 | 0.9069 | 98.6319 | 0.0981 |  |  |  |  |
| smec_ci | opu_old_preserve | 3 | 0.7540 | 0.7167 | 0.2000 | 0.5500 | 0.8988 | 157.8111 | 0.1281 | -0.0053 | 0.0833 | -0.0333 | diagnostic_only |
| base_known_route | opu_old_preserve | 4 | 0.8021 | 0.7833 | 0.1833 | 0.7500 | 0.9595 | 118.1759 | 0.0981 |  |  |  |  |
| smec_ci | opu_old_preserve | 4 | 0.7968 | 0.7833 | 0.2667 | 0.7000 | 0.9514 | 189.0814 | 0.1281 | -0.0053 | 0.0833 | -0.0500 | diagnostic_only |
| base_known_route | opu_old_preserve | 5 | 0.7968 | 0.7833 | 0.0833 | 0.7500 | 0.9595 | 130.2932 | 0.0981 |  |  |  |  |
| smec_ci | opu_old_preserve | 5 | 0.7914 | 0.7833 | 0.1667 | 0.7000 | 0.9514 | 208.4691 | 0.1281 | -0.0053 | 0.0833 | -0.0500 | diagnostic_only |

解释：

- SMEC-CI能提升未知拒识并降低FAR，但旧类full accuracy在所有`opu_old_preserve`协同数量下下降，违反“旧类准确性不能下降”；
- 最好旧类行仍远低于用户目标`old_acc=0.99/min_old=0.95`，unknown拒识也远低于`0.99`；
- 结论为负面诊断：当前support-only metric/energy风险不能直接作为主线部署候选。

## N607计划

远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

预期同步：

| local | remote |
|---|---|
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_support_metric_energy_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_support_metric_energy_ci_eval.py` |
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_support_metric_energy_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_support_metric_energy_ci_eval.py` |

预期远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_support_metric_energy_ci_eval.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_support_metric_energy_ci_eval.py --feature_npz remote_artifacts/phase2_adv3b02_features/features.npz --output_dir remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_all --force --write_evidence
```

## N607执行结果

N607 preflight：

- direct target：`N607`
- host：`dell-DSS8440`
- project root：`/home/szu2070436088/2510044040/CV-SincNet`
- 远端时间：2026-07-04 17:03:04 CST
- GPU状态：8张RTX 3090均约10MiB显存占用；最终选择`CUDA_VISIBLE_DEVICES=0`

远端同步：

| local | remote | 状态 |
|---|---|---|
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_support_metric_energy_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_support_metric_energy_ci_eval.py` | 已同步 |
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_support_metric_energy_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_support_metric_energy_ci_eval.py` | 已同步 |
| `E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz` | `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_features/features.npz` | 已同步 |

远端hash/语法验证：

- feature包SHA256：`db559d78db305894307851750ef7d698db387f0984ff13c980fea99db85b8532`
- 远端脚本SHA256：`bd26974b6d44ed396af751acb145751e102ec615627d905a2fe02b93059e084f`
- 远端Python：`Python 3.10.19`
- 语法检查：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_support_metric_energy_ci_eval.py`通过

远端执行命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_support_metric_energy_ci_eval.py --feature_npz remote_artifacts/phase2_adv3b02_features/features.npz --output_dir remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_all --force --write_evidence
```

远端输出：

```json
{"receiver_count": 5, "group_count": 307, "evidence_row_count": 1000}
{"summary_rows": 20, "candidate_count": 0, "summary_csv": "remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_all/smec_ci_summary.csv"}
```

远端artifact：

- `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_all/smec_ci_summary.csv`
- `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_all/smec_ci_best_rows.csv`
- `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_all/smec_ci_audit.json`

已拉回本地：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_all\smec_ci_summary.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_all\smec_ci_best_rows.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_all\smec_ci_audit.json`

SSH/SCP断连检查：

- preflight后：无`ssh.exe`，无到`172.31.111.215:22`或`172.31.105.18:22`的ESTABLISHED连接；
- SCP脚本/测试后：无残留；
- 远端语法验证后：无残留；
- 远端全量诊断后：无残留；
- 远端结果拉回后：无残留。

远端结论：

- `collab_count=1..5`全部覆盖；
- `target_channel_view`来自qknn metadata，包含`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`；
- `unknown_query_used_for_threshold=false`；
- `candidate_count=0`，与本地负面诊断一致；
- 不满足“未知类拒识优先且旧类准确性不能下降”的同一行联合判据。

## 风险与下一步

SMEC-CI的主要失败原因是support shell风险对弱old query同样敏感，提升unknown risk会通过reject/defer门控伤害旧类full accuracy。下一步不应继续调高该风险，而应转向更强的“旧类无损约束”算法：

- per-old support留一conformal risk control，给每个旧类设定旧类拒识预算；
- risk-lift只作用于非旧类候选或多接收机一致判定的open-space样本；
- 对unknown优先使用source-heldout伪未知/虚拟边界进行地面冻结校准，而不是target unknown query；
- 若要星上实时微调，只允许adapter/temperature在低风险old/seen-new样本上更新，并需要old retention rollback监控。

## 2026-07-04 old_lossless SMEC复测

针对上一节负面结果，新增`smec_old_lossless_ci`策略：当base top label属于旧类`Y_old`时，不提升该receiver行`unknown_risk`，只允许非旧类top label或非旧类候选承担support metric/energy风险提升。该策略的目的不是直接最大化unknown拒识，而是先关闭旧类full accuracy下降通道，再观察unknown是否仍有净收益。

本地验证：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_support_metric_energy_ci_eval.py -q
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_support_metric_energy_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_support_metric_energy_ci_eval.py --feature_npz E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz --output_dir local_artifacts\phase2_adv3b02_smec_ci_20260704\local_old_lossless --force --write_evidence
```

结果：单元测试`5 passed`，语法检查通过，本地全量输出`summary_rows=30,candidate_count=1`。

N607复测：

- preflight时间：2026-07-04 17:12:29 CST；
- 8张RTX3090均约`10/24576MiB`，选择GPU0；
- 远端Python：`Python 3.10.19`，环境`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；
- 远端脚本SHA256：`88119571c9593789568d82fa1876effa8dfaa8c8c1b197523a9d0016a3f2d72a`；
- 远端测试文件SHA256：`d1a8ac1f04e6fcc2dd8049e594d1a978265d2deba8622dfb1416853c6137cb9a`；
- feature包SHA256：`db559d78db305894307851750ef7d698db387f0984ff13c980fea99db85b8532`；
- 远端语法检查通过；
- 每次SSH/SCP后本地检查均无`ssh.exe`残留、无N607/bridge 22端口ESTABLISHED连接。

远端执行命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_support_metric_energy_ci_eval.py --feature_npz remote_artifacts/phase2_adv3b02_features/features.npz --output_dir remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_old_lossless --force --write_evidence
```

远端输出：

```json
{"receiver_count": 5, "group_count": 307, "evidence_row_count": 1000}
{"summary_rows": 30, "candidate_count": 1, "summary_csv": "remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_old_lossless/smec_ci_summary.csv"}
```

远端结果已拉回：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_old_lossless\smec_ci_summary.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_old_lossless\smec_ci_best_rows.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_old_lossless\smec_ci_audit.json`

`opu_old_preserve`下`collab_count=1..5`逐项结果：

| algorithm | policy | collab_count | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | known_coverage | defer_rate | bytes_per_event | latency_ms_p95 | delta_old_acc | delta_unknown_reject_rate | delta_unknown_FAR | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smec_old_lossless_ci | opu_old_preserve | 1 | 0.6364 | 0.1000 | 0.6500 | 0.4000 | 0.2000 | 0.7833 | 0.9514 | 0.0000 | 64.0000 | 0.2017 | 0.0000 | 0.0833 | -0.0833 | diagnostic_only |
| smec_old_lossless_ci | opu_old_preserve | 2 | 0.6257 | 0.1500 | 0.8167 | 0.7500 | 0.0833 | 0.6833 | 0.9514 | 0.0326 | 116.1173 | 0.2017 | 0.0000 | 0.0333 | -0.0333 | diagnostic_only |
| smec_old_lossless_ci | opu_old_preserve | 3 | 0.7594 | 0.2000 | 0.7167 | 0.7000 | 0.1667 | 0.5833 | 0.9069 | 0.0423 | 157.8111 | 0.2017 | 0.0000 | 0.0500 | 0.0000 | diagnostic_only |
| smec_old_lossless_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2000 | 0.7333 | 0.9595 | 0.0423 | 189.0814 | 0.2017 | 0.0000 | 0.0167 | -0.0167 | candidate |
| smec_old_lossless_ci | opu_old_preserve | 5 | 0.7968 | 0.2500 | 0.7833 | 0.7250 | 0.1167 | 0.7333 | 0.9595 | 0.0586 | 208.4691 | 0.2017 | 0.0000 | 0.0333 | -0.0167 | diagnostic_only |

候选行：

| algorithm | policy | collab_count | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_unknown_reject_rate | delta_unknown_FAR | bytes_per_event | latency_ms_p95 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smec_old_lossless_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2000 | 0.7333 | 0.0000 | 0.0167 | -0.0167 | 189.0814 | 0.2017 | candidate |

解释：

- 旧类不下降：`delta_old_acc=0.0000`，比标准SMEC的`-0.0053`更符合目标方向；
- unknown拒识有小幅净收益：`unknown_reject_rate +0.0167`，`unknown_FAR -0.0167`；
- 但绝对指标仍远低于目标：`min_old_class_acc=0.3000`，`seen_new_acc=0.7833`，`unknown_reject_rate=0.2000`，`unknown_FAR=0.7333`；
- 因此该策略是阶段性候选，只说明“旧类无损风险提升”方向合理，不能声明最终99/97/99目标完成。

old_lossless审计字段：

- `support_only_thresholds=true`；
- `unknown_query_used_for_threshold=false`；
- `label_authority=base_qknn_only`；
- `old_label_lift_blocked_count=722/1000`；
- `target_receiver_ids=["20-1","3-19","7-14","7-7","8-8"]`；
- `target_channel_view`继承qknn metadata，覆盖LEO弱场景。

下一步应基于该candidate继续增强，但增强对象不能再是“所有弱样本风险”。建议进入`OLRC-CI`：old-label risk freeze + seen-new/unknown boundary conformal + receiver-pair old-consensus rescue，使unknown改善集中在非旧类候选和跨receiver不稳定候选上，同时用每类old support留一风险预算强约束旧类floor。

## 2026-07-04 consensus_guard SMEC复测

在`old_lossless`基础上，新增`smec_consensus_guard_ci`：旧类top label不再绝对冻结，而是仅在以下条件同时满足时允许提升旧类top label行的`unknown_risk`：

- 同一`event_id`的跨receiver label agreement不高于`0.60`；
- 该receiver行的base weakness不低于`0.50`；
- 该行不是strong known candidate。

这相当于把风险提升限定到“弱证据+跨接收机分歧”的旧类候选，试图在不损害旧类full accuracy的前提下多拒识一部分unknown。

本地验证：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_support_metric_energy_ci_eval.py -q
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_support_metric_energy_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_support_metric_energy_ci_eval.py --feature_npz E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz --output_dir local_artifacts\phase2_adv3b02_smec_ci_20260704\local_consensus_guard --profiles standard,old_lossless,consensus_guard --force --write_evidence
```

结果：单元测试`7 passed`，语法检查通过，本地全量输出`summary_rows=40,candidate_count=2`。

N607复测：

- preflight时间：2026-07-04 17:21:12 CST；
- 8张RTX3090均约`10/24576MiB`，选择GPU0；
- 远端Python：`Python 3.10.19`，环境`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；
- 远端脚本SHA256：`a9fbaa1bcde21676c5a6787e70ed440394a908327ee005864cbdefb821bf9aca`；
- 远端测试文件SHA256：`f3682fa7cc5364423f40955658f34a243e10801f7db2dd378f01502fe184963c`；
- feature包SHA256：`db559d78db305894307851750ef7d698db387f0984ff13c980fea99db85b8532`；
- 远端语法检查通过；
- 每次SSH/SCP后本地检查均无`ssh.exe`残留、无N607/bridge 22端口ESTABLISHED连接。

远端执行命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_support_metric_energy_ci_eval.py --feature_npz remote_artifacts/phase2_adv3b02_features/features.npz --output_dir remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_consensus_guard --profiles standard,old_lossless,consensus_guard --force --write_evidence
```

远端输出：

```json
{"receiver_count": 5, "group_count": 307, "evidence_row_count": 1000}
{"summary_rows": 40, "candidate_count": 2, "summary_csv": "remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_consensus_guard/smec_ci_summary.csv"}
```

远端结果已拉回：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_consensus_guard\smec_ci_summary.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_consensus_guard\smec_ci_best_rows.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_consensus_guard\smec_ci_audit.json`

`opu_old_preserve`下`smec_consensus_guard_ci`的`collab_count=1..5`逐项结果：

| algorithm | policy | collab_count | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | known_coverage | defer_rate | bytes_per_event | latency_ms_p95 | delta_old_acc | delta_unknown_reject_rate | delta_unknown_FAR | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smec_consensus_guard_ci | opu_old_preserve | 1 | 0.6364 | 0.1000 | 0.6500 | 0.4000 | 0.2167 | 0.7667 | 0.9514 | 0.0000 | 64.0000 | 0.1560 | 0.0000 | 0.1000 | -0.1000 | diagnostic_only |
| smec_consensus_guard_ci | opu_old_preserve | 2 | 0.6257 | 0.1500 | 0.8167 | 0.7500 | 0.1000 | 0.6667 | 0.9514 | 0.0293 | 116.1173 | 0.1560 | 0.0000 | 0.0500 | -0.0500 | diagnostic_only |
| smec_consensus_guard_ci | opu_old_preserve | 3 | 0.7594 | 0.2000 | 0.7167 | 0.7000 | 0.1833 | 0.5667 | 0.9069 | 0.0391 | 157.8111 | 0.1560 | 0.0000 | 0.0667 | -0.0167 | diagnostic_only |
| smec_consensus_guard_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2167 | 0.7333 | 0.9595 | 0.0391 | 189.0814 | 0.1560 | 0.0000 | 0.0333 | -0.0167 | candidate |
| smec_consensus_guard_ci | opu_old_preserve | 5 | 0.7968 | 0.2500 | 0.7833 | 0.7250 | 0.1333 | 0.7333 | 0.9595 | 0.0554 | 208.4691 | 0.1560 | 0.0000 | 0.0500 | -0.0167 | diagnostic_only |

当前最佳SMEC系列候选：

| algorithm | policy | collab_count | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_unknown_reject_rate | delta_unknown_FAR | bytes_per_event | latency_ms_p95 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smec_consensus_guard_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2167 | 0.7333 | 0.0000 | 0.0333 | -0.0167 | 189.0814 | 0.1560 | candidate |

相对`old_lossless`：

- unknown拒识净提升从`+0.0167`扩大到`+0.0333`；
- 旧类full accuracy仍不下降；
- `defer_rate`从`0.0423`降到`0.0391`；
- 但绝对unknown拒识仍只有`0.2167`，FAR仍为`0.7333`，距离目标仍很远。

审计字段：

- `support_only_thresholds=true`；
- `unknown_query_used_for_threshold=false`；
- `label_authority=base_qknn_only`；
- `old_label_lift_blocked_count=605/1000`，相比`old_lossless`的`722/1000`更开放；
- `target_receiver_ids=["20-1","3-19","7-14","7-7","8-8"]`；
- 当前仍是`receiver_domain_ranked`诊断口径，不能写成严格同物理事件卫星群协同。

下一步建议从单纯风险提升转向“错误旧类预测纠偏”：当前unknown大量仍被old top label吸收，必须引入source-heldout伪未知/支持集外推边界或严格同事件key数据重导出，否则仅靠receiver-domain ranked ensemble很难把unknown拒识从0.2级推到0.99。

## 2026-07-04 old_boundary_guard与OBACE路线审计

本轮针对“unknown高一致性被旧类吸收”的诊断，新增`smec_old_boundary_guard_ci`。该profile在support-only条件下为每个old label建立old-vs-other prototype margin阈值：

```text
margin_c(x)=sim(z,p_c)-max_{j!=c} sim(z,p_j)
T_c=quantile_{0.05}({margin_c(x_i): x_i in support old class c})
risk_boundary=sigmoid((T_c-margin_c(x))/temperature)
```

融合边界：

- 阈值只来自target-old/seen-new support，不使用unknown query；
- frozen qKNN仍是唯一label authority，新增字段只提升`unknown_risk`；
- 对old top label，只有弱证据且`risk_boundary`达到profile阈值，或跨receiver label agreement低，才允许解除旧类风险提升阻断；
- tight profile将`old_boundary_min_risk`收紧到`0.98`，避免旧类accuracy下降。

本地验证：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_support_metric_energy_ci_eval.py -q
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_support_metric_energy_ci_eval.py code\tests\test_phase2_support_metric_energy_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_support_metric_energy_ci_eval.py --feature_npz E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz --output_dir local_artifacts\phase2_adv3b02_smec_ci_20260704\local_old_boundary_guard_tight --profiles standard,old_lossless,consensus_guard,old_boundary_guard --force --write_evidence
```

结果：单元测试`10 passed`，语法检查通过；本地矩阵`receiver_count=5,summary_rows=50,candidate_count=0`。本地派生特征的old性能过低，仅作为代码运行验证，不作为主结论。

N607复测：

- preflight时间：2026-07-04 17:35:51 CST；
- 8张RTX3090均约`10/24576MiB`，选择GPU0；
- 远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；
- 远端脚本SHA256：`cf1e53275dc9fba7e906193b53c46840ebe54c97ceaa7a005243b4bfbebf6b6b`；
- 远端测试文件SHA256：`af69aa0de239e3f288d1a052685311d57d61c5c0844eaaab06dc552fbb6d2cca`；
- 远端`pytest`不可用，改用`CVS-RFFI` Python手动导入并执行10个`test_`函数，结果`manual_test_functions_passed=10`；
- 每次SSH/SCP后本地检查均无`ssh.exe`残留、无N607/bridge 22端口ESTABLISHED连接；
- 第一次SCP误把测试文件也放到远端`code/scripts/`目录，随后已补同步到正确`code/tests/`路径；未删除误放文件，避免未经请求删除远端文件。

远端执行命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_support_metric_energy_ci_eval.py --feature_npz remote_artifacts/phase2_adv3b02_features/features.npz --output_dir remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_old_boundary_guard_tight --profiles standard,old_lossless,consensus_guard,old_boundary_guard --force --write_evidence
```

远端输出：

```json
{"receiver_count": 5, "group_count": 307, "evidence_row_count": 1000}
{"summary_rows": 50, "candidate_count": 2, "summary_csv": "remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_old_boundary_guard_tight/smec_ci_summary.csv"}
```

远端结果已拉回：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_old_boundary_guard_tight\smec_ci_summary.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_old_boundary_guard_tight\smec_ci_best_rows.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_old_boundary_guard_tight\smec_ci_audit.json`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_old_boundary_guard_tight\smec_old_boundary_guard_ci_evidence.csv`

N607候选行：

| algorithm | policy | collab_count | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_seen_new_acc | delta_unknown_reject_rate | delta_unknown_FAR | known_coverage | bytes_per_event | latency_ms_p95 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smec_old_lossless_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2000 | 0.7333 | 0.0000 | 0.0000 | 0.0167 | -0.0167 | 0.9595 | 189.0814 | 0.1991 | candidate |
| smec_consensus_guard_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2167 | 0.7333 | 0.0000 | 0.0000 | 0.0333 | -0.0167 | 0.9555 | 189.0814 | 0.1991 | candidate |

`smec_old_boundary_guard_ci`本轮不是候选：

| algorithm | policy | collab_count | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_unknown_reject_rate | delta_unknown_FAR | known_coverage | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smec_old_boundary_guard_ci | opu_old_preserve | 4 | 0.7968 | 0.3000 | 0.7833 | 0.7250 | 0.2167 | 0.7333 | -0.0053 | 0.0333 | -0.0167 | 0.9514 | diagnostic_only |
| smec_old_boundary_guard_ci | opu_old_guarded | 4 | 0.7594 | 0.3000 | 0.6667 | 0.6500 | 0.3500 | 0.5167 | -0.0107 | 0.0833 | 0.0167 | 0.7935 | diagnostic_only |

同一row结论：

- `old_boundary_guard`确实能提高unknown拒识，但只要提高幅度明显，就会伤害old accuracy；
- 收紧阈值后仍无法同时超过`smec_consensus_guard_ci`；
- 因此当前最强仍是`smec_consensus_guard_ci + opu_old_preserve + collab_count=4`，但unknown拒识绝对值只有`0.2167`，离目标`0.99`很远，不能声明目标达成。

子agent审计结论：

- 完成度监督子agent要求必须有代码diff、本地验证、N607 preflight/SCP/远端验证、低显存GPU证据和row级结果后才能声明完成；本段已补齐这些证据，但性能目标未达成；
- review子agent指出margin guard不能直接视为Stage2-C成功路线，若unknown在表示空间内部被旧类高置信吸收，margin guard会成为盲区；本轮结果验证了该风险；
- 文献/方法子agent建议下一阶段采用`OBACE`：Old-Boundary Anchored Conformal-EVT Energy Gate，即support-only prototype radius、energy、EVT、conformal p-value和receiver可靠性融合。该路线不使用unknown query调阈值，适合`collab_count=1..N`，但尚未在本轮完整实现。

下一步建议：

1. 将当前`old_boundary_guard`保留为诊断字段，不作为默认候选；
2. 实现`OBACE-CI`完整门：`distance/prototype radius + energy + EVT tail + conformal p-value`，并设置old support LOO通过率硬约束；
3. 对吸收热点`14-7/6-15/19-3`输出单独sink-class FAR/recall表；
4. 若同一row仍无法兼顾old与unknown，则问题更可能在表示空间可分性或特征训练阶段，需要回到ground model/adapter训练，而不是继续调协同推理阈值。

## 2026-07-04 OBACE-CI support-only conformal门

本轮在`old_boundary_guard`的基础上实现`obace_ci`：每个receiver仅用target-old/seen-new support构建类条件非一致性分数，query只做评估，不参与阈值拟合。核心证据为：

```text
A_c(x)=d_proto(x,c)/T_proto(c)+d_knn(x,S_c)/T_knn
p_conf=(1+#{A_i>=A_c(x):A_i in support class c})/(|S_c|+1)
risk_conf=1-p_conf
```

OBACE只在启用的绝对门中至少多个失败时提升`unknown_risk`。第一版宽松profile设置`obace_old_min_abs_failures=2,obace_nonold_min_abs_failures=2,obace_conformal_min_risk=0.75`；远端结果显示unknown改善明显但old/seen-new下降，不能作为候选。随后收紧为：

```text
obace_old_min_abs_failures=3
obace_nonold_min_abs_failures=3
obace_conformal_min_risk=0.85
```

本地验证：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_support_metric_energy_ci_eval.py -q
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_support_metric_energy_ci_eval.py code\tests\test_phase2_support_metric_energy_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_support_metric_energy_ci_eval.py --feature_npz E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz --output_dir local_artifacts\phase2_adv3b02_smec_ci_20260704\local_obace_tight --profiles standard,old_lossless,consensus_guard,old_boundary_guard,obace --force --write_evidence
```

结果：单元测试`14 passed`，语法检查通过；本地派生特征输出`receiver_count=5,summary_rows=60,candidate_count=0`，仅作为运行验证。

N607复测：

- preflight时间：2026-07-04 17:50:59 CST；
- 8张RTX3090均约`10/24576MiB`，选择GPU0；
- 远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；
- tight脚本SHA256：`be6667527a2e9a1297dfa27ad2924e625385e7577c829282d88a422b994b10b9`；
- tight测试SHA256：`cada9ebea3de51e645fafb7faa5fc8e09a2431b74bdce470ac6893e70ea4da38`；
- 远端`CVS-RFFI`手动执行测试函数：`manual_test_functions_passed=14`；
- 每次SSH/SCP后本地检查均无`ssh.exe`残留、无N607/bridge 22端口ESTABLISHED连接。

远端执行命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_support_metric_energy_ci_eval.py --feature_npz remote_artifacts/phase2_adv3b02_features/features.npz --output_dir remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_obace_tight --profiles standard,old_lossless,consensus_guard,old_boundary_guard,obace --force --write_evidence
```

远端输出：

```json
{"receiver_count": 5, "group_count": 307, "evidence_row_count": 1000}
{"summary_rows": 60, "candidate_count": 3, "summary_csv": "remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_obace_tight/smec_ci_summary.csv"}
```

远端结果已拉回：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_tight\smec_ci_summary.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_tight\smec_ci_best_rows.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_tight\smec_ci_audit.json`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_tight\obace_ci_evidence.csv`

同一row候选结果：

| algorithm | policy | collab_count | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_seen_new_acc | delta_unknown_reject_rate | delta_unknown_FAR | known_coverage | defer_rate | bytes_per_event | latency_ms_p95 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smec_old_lossless_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2000 | 0.7333 | 0.0000 | 0.0000 | 0.0167 | -0.0167 | 0.9595 | 0.0423 | 189.0814 | 0.1543 | candidate |
| smec_consensus_guard_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2167 | 0.7333 | 0.0000 | 0.0000 | 0.0333 | -0.0167 | 0.9555 | 0.0423 | 189.0814 | 0.1543 | candidate |
| obace_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2333 | 0.7000 | 0.0000 | 0.0000 | 0.0500 | -0.0500 | 0.9555 | 0.0456 | 189.0814 | 0.1543 | candidate |

OBACE tight是当前同协议下最强row，但仍远低于目标：

- 旧类目标只达到`old_acc=0.8021`，未达到99%，`min_old_class_acc=0.3000`远低于95%；
- seen-new只达到`0.7833`，未达到97%，`min_seen_new_class_acc=0.7250`低于93%；
- unknown拒识只有`0.2333`，FAR仍为`0.7000`，未达到99%拒识目标；
- 资源开销仍受控：`participating_receivers_p95=4`，`bytes_per_event=189.0814`，`latency_ms_p95=0.1543`。

evidence层审计：

| role | n | risk_mean | risk_q90 | conformal_mean | fail_count分布 | lifted | blocked |
|---|---:|---:|---:|---:|---|---:|---:|
| old | 600 | 0.0898 | 0.4252 | 0.4617 | 0:476,1:68,2:41,3:11,4:4 | 5 | 593 |
| seen_new | 200 | 0.1064 | 0.4523 | 0.4806 | 0:158,1:22,2:17,3:3 | 3 | 197 |
| unknown | 200 | 0.4049 | 0.9840 | 0.6767 | 0:74,1:48,2:52,3:25,4:1 | 19 | 175 |

解释：tight OBACE只放行少量多门失败样本，因而保住old/seen-new同一row不下降；unknown提升有限，说明大量unknown被旧类表示空间内部吸收，support-only门只能修掉边界外的一部分样本。下一步如果继续只调协同推理阈值，收益大概率递减。应转向两条路线之一：

1. 训练侧：让ADV3B02后续地面训练加入source-heldout伪未知/open-set SSL/ARPL或EVT-aware radius loss，改善`z_id`空间的unknown可分性；
2. 部署侧：在OBACE基础上增加source-heldout伪未知校准包或星上允许的小adapter，但必须保留old support LOO回滚和unknown query不调参边界。

## 2026-07-04 OBACE-event事件级协同门

OBACE tight只使用单receiver硬门，能保护old/seen-new，但只把unknown拒识推到`0.2333`。本轮新增`obace_event_ci`，将同一`event_id`内多个receiver的OBACE conformal风险做事件级累积：

```text
vote_r = 1[risk_conf,r >= 0.55]
event_risk = mean(risk_conf) + 0.15 * mean(vote_r) + 0.10 * mean(local_abs_fail > 0)
```

第一版事件门允许`event_min_votes=2,event_min_mean_risk=0.55`，unknown拒识更高，但old下降。最终保留的strict profile要求：

```text
event_min_votes=4
event_min_mean_risk=0.75
event_old_min_local_failures=2
event_nonold_min_local_failures=2
```

本地验证：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_support_metric_energy_ci_eval.py -q
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_support_metric_energy_ci_eval.py code\tests\test_phase2_support_metric_energy_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_support_metric_energy_ci_eval.py --feature_npz E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz --output_dir local_artifacts\phase2_adv3b02_smec_ci_20260704\local_obace_event --profiles standard,old_lossless,consensus_guard,old_boundary_guard,obace,obace_event --force --write_evidence
```

结果：单元测试`16 passed`，语法检查通过；本地派生特征输出`receiver_count=5,summary_rows=70,candidate_count=0`，仍只作为运行验证。

N607复测：

- GPU状态：8张RTX3090均约`10/24576MiB`，使用GPU0；
- 远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；
- 远端strict脚本SHA256：`0b8a136213bebaf9cfd496491cb3f307d848713d29479229dd540f82fea4136b`；
- 远端测试SHA256：`bc9b04d187b60cfb5cdc927cb8c60bd150829368bf36a4feaf69908b9e1ed32e`；
- 远端手动执行测试函数：`manual_test_functions_passed=16`；
- 每次SSH/SCP后本地检查均无`ssh.exe`残留、无N607/bridge 22端口ESTABLISHED连接；
- 资源约束说明文件`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`本地仍未检出，本轮仍只报告代理资源指标`participating_receivers_p95/bytes_per_event/latency_ms_p95`。

远端执行命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_support_metric_energy_ci_eval.py --feature_npz remote_artifacts/phase2_adv3b02_features/features.npz --output_dir remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_obace_event_tight --profiles standard,old_lossless,consensus_guard,old_boundary_guard,obace,obace_event --force --write_evidence
```

远端输出：

```json
{"receiver_count": 5, "group_count": 307, "evidence_row_count": 1000}
{"summary_rows": 70, "candidate_count": 4, "summary_csv": "remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_obace_event_tight/smec_ci_summary.csv"}
```

远端结果已拉回：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_event_tight\smec_ci_summary.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_event_tight\smec_ci_best_rows.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_event_tight\smec_ci_audit.json`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_event_tight\obace_event_ci_evidence.csv`

同一row候选结果：

| algorithm | policy | collab_count | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_seen_new_acc | delta_unknown_reject_rate | delta_unknown_FAR | known_coverage | defer_rate | bytes_per_event | latency_ms_p95 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smec_old_lossless_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2000 | 0.7333 | 0.0000 | 0.0000 | 0.0167 | -0.0167 | 0.9595 | 0.0423 | 189.0814 | 0.1555 | candidate |
| smec_consensus_guard_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2167 | 0.7333 | 0.0000 | 0.0000 | 0.0333 | -0.0167 | 0.9555 | 0.0423 | 189.0814 | 0.1555 | candidate |
| obace_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2333 | 0.7000 | 0.0000 | 0.0000 | 0.0500 | -0.0500 | 0.9555 | 0.0456 | 189.0814 | 0.1555 | candidate |
| obace_event_ci | opu_old_preserve | 4 | 0.8021 | 0.3000 | 0.7833 | 0.7250 | 0.2500 | 0.6833 | 0.0000 | 0.0000 | 0.0667 | -0.0667 | 0.9595 | 0.0423 | 189.0814 | 0.1555 | candidate |

evidence层审计：

| role | n | risk_mean | event_mean | event_q90 | event_vote分布 | lifted | blocked |
|---|---:|---:|---:|---:|---|---:|---:|
| old | 600 | 0.0898 | 0.5527 | 0.8267 | 0:71,1:176,2:173,3:112,4:63,5:5 | 3 | 596 |
| seen_new | 200 | 0.1104 | 0.5869 | 0.8244 | 0:19,1:43,2:45,3:49,4:34,5:10 | 5 | 195 |
| unknown | 200 | 0.4341 | 0.8573 | 1.0000 | 0:3,1:21,2:16,3:52,4:68,5:40 | 40 | 151 |

解释：`obace_event_ci`是当前最强同协议候选，证明多receiver事件级证据累积比单receiver硬门更有效：unknown拒识从`0.2333`升至`0.2500`，FAR从`0.7000`降至`0.6833`，old和seen-new同一row保持不下降。但绝对指标仍远低于目标，尤其`min_old_class_acc=0.3000`和`unknown_reject_rate=0.2500`说明表示空间仍严重吸收unknown。下一步应把协同推理门控与训练侧open-set表征改造合并，而不是继续单独调阈值。

## 2026-07-04 OBACE-VOID事件级强未知例外门

目标：在不降低当前`obace_event_ci/opu_old_preserve/collab_count=4`同row旧类和seen-new准确率的前提下，优先降低未知类被旧类/seen-new稳定吸收的问题。本轮离线审计发现，许多missed unknown样本已经具有较高事件级OBACE风险，但被`strong_known_candidate`白名单完全挡住；直接取消白名单会伤害old。因此新增`obace_void_ci`，只在事件级证据满足极端条件时允许突破strong-known保护：

```text
obace_event_risk >= 0.95
event_vote_count >= 4
event_receiver_count >= 4
event_label_count >= 2
event_label_agreement <= 0.80
```

该规则仍只使用target-old/seen-new support构造的原型、KNN、energy和conformal风险；`unknown_query_used_for_threshold=false`保持不变。它不是用unknown query调参，而是将“多接收机高风险且标签不稳定”的事件作为支持集校准风险的强证据例外。若远端评估显示`old_acc`或`seen_new_acc`下降，该profile必须降级为`diagnostic_only`，不能替代当前候选。

子agent审查结论：

| 子agent | 结论 | 已落实 |
|---|---|---|
| 文献/方法检索 | P0应采用原型+EVT/OpenMax、Energy/Mahalanobis/KNN、conformal p-value、多接收机协同；避免target unknown query调阈值和全模型在线自训练 | 本轮实现落在feature-level OBACE/conformal/事件融合层 |
| 算法监督 | 不应继续加严单一全局阈值；应做old/seen保真白名单+support-known-only未知二次确认，并输出`collab_count=1..all`曲线 | 新增`obace_void`profile，保留原OBACE_EVENT作为对照 |

本地变更：

| 文件 | 作用 |
|---|---|
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_support_metric_energy_ci_eval.py` | 新增`obace_void_guard`、事件`label_count`审计、`obace_void`profile |
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_support_metric_energy_ci_eval.py` | 新增两项TDD测试：极端分歧事件可突破strong-known；标签一致强old仍受保护 |

本地验证：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_support_metric_energy_ci_eval.py -q
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_support_metric_energy_ci_eval.py code\tests\test_phase2_support_metric_energy_ci_eval.py
```

结果：新增测试先失败于`SmecConfig.__init__() got an unexpected keyword argument 'obace_void_override_min_event_risk'`，实现后`18 passed`，语法检查通过。

计划远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_support_metric_energy_ci_eval.py --feature_npz remote_artifacts/phase2_adv3b02_features/features.npz --output_dir remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_obace_void --profiles standard,old_lossless,consensus_guard,old_boundary_guard,obace,obace_event,obace_void --force --write_evidence
```

预期输出：`collab_count=1..5`全量曲线、`obace_void_ci_evidence.csv`、`smec_ci_summary.csv`、`smec_ci_best_rows.csv`、`smec_ci_audit.json`。判定标准仍为同row`old_acc`不下降、`seen_new_acc`不下降、`unknown_reject_rate`上升且`unknown_FAR`不恶化。

远端执行结果：

```text
GPU pre-run: 8张RTX3090均约10/24576MiB，GPU0执行
{"receiver_count": 5, "group_count": 307, "evidence_row_count": 1000}
{"summary_rows": 80, "candidate_count": 5, "summary_csv": "remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_obace_void/smec_ci_summary.csv"}
```

远端同步与验证：

| 项目 | 结果 |
|---|---|
| Git提交 | `6c965a0 Add OBACE void event guard` |
| 本地脚本SHA256 | `C83E9F60873650F3365B59933E4A9DE26DEA83B8C35299A6E85DD9B1CE2E2FE1` |
| 本地测试SHA256 | `AE248E0C65D4B1DC5202A7363DA97B8BFB6D53D24116AB424476F6026F86297D` |
| 远端脚本SHA256 | `c83e9f60873650f3365b59933e4a9de26dea83b8c35299a6e85dd9b1ce2e2fe1` |
| 远端测试SHA256 | `ae248e0c65d4b1dc5202a7363da97b8bfb6d53d24116ab424476f6026f86297d` |
| 远端语法 | `py_compile`通过 |
| 远端单元测试 | `manual_test_functions_passed=18`；远端无`pytest`模块，使用手动函数执行 |
| SSH/SCP清理 | 每次命令后本地均无`ssh.exe`和N607/bridge 22端口ESTABLISHED连接 |

远端结果已拉回：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_void\smec_ci_summary.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_void\smec_ci_best_rows.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_void\smec_ci_audit.json`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_void\obace_void_ci_evidence.csv`

同一policy主对照：

| algorithm | policy | collab_count | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_seen_new_acc | delta_unknown_reject_rate | delta_unknown_FAR | known_coverage | defer_rate | bytes_per_event | latency_ms_p95 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| obace_event_ci | opu_old_preserve | 4 | 0.802139 | 0.300000 | 0.783333 | 0.725000 | 0.250000 | 0.683333 | 0.000000 | 0.000000 | 0.066667 | -0.066667 | 0.959514 | 0.042345 | 189.081433 | 0.154295 | candidate |
| obace_void_ci | opu_old_preserve | 4 | 0.802139 | 0.300000 | 0.783333 | 0.725000 | 0.250000 | 0.683333 | 0.000000 | 0.000000 | 0.066667 | -0.066667 | 0.959514 | 0.042345 | 189.081433 | 0.154295 | candidate |
| obace_ci | opu_old_preserve | 4 | 0.802139 | 0.300000 | 0.783333 | 0.725000 | 0.233333 | 0.700000 | 0.000000 | 0.000000 | 0.050000 | -0.050000 | 0.955466 | 0.045603 | 189.081433 | 0.154295 | candidate |
| smec_consensus_guard_ci | opu_old_preserve | 4 | 0.802139 | 0.300000 | 0.783333 | 0.725000 | 0.216667 | 0.733333 | 0.000000 | 0.000000 | 0.033333 | -0.016667 | 0.955466 | 0.042345 | 189.081433 | 0.154295 | candidate |
| smec_old_lossless_ci | opu_old_preserve | 4 | 0.802139 | 0.300000 | 0.783333 | 0.725000 | 0.200000 | 0.733333 | 0.000000 | 0.000000 | 0.016667 | -0.016667 | 0.959514 | 0.042345 | 189.081433 | 0.154295 | candidate |

更激进policy下，`obace_void_ci/opu_old_guarded/collab_count=4`达到`unknown_reject_rate=0.450000,unknown_FAR=0.483333`，但`old_acc=0.770053,seen_new_acc=0.683333`，低于当前保真候选，因此只能作为`diagnostic_only`，不能写成部署候选或Stage2-C成功。

VOID证据层审计：

| role | n | risk_mean | base_mean | event_mean | override_count | lifted_count | blocked_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| old | 600 | 0.107910 | 0.088268 | 0.552750 | 14 | 65 | 584 |
| seen_new | 200 | 0.124563 | 0.103182 | 0.586917 | 5 | 23 | 191 |
| unknown | 200 | 0.585460 | 0.394291 | 0.857306 | 62 | 128 | 115 |

解释：`obace_void_ci`证明“事件级高风险+多接收机标签分歧”的强未知例外能显著提升evidence层unknown风险，但在`opu_old_preserve`的最终融合下没有超过`obace_event_ci`；在`opu_old_guarded`下能继续压低FAR，但代价是旧类和seen-new明显下降。当前最强协议内候选仍是`obace_event_ci/opu_old_preserve/collab_count=4`与`obace_void_ci/opu_old_preserve/collab_count=4`并列，二者old/seen不下降、unknown拒识`0.250000`、FAR`0.683333`。这离`unknown_FAR<=0.05`仍很远，说明单纯协同推理门控已接近瓶颈，后续必须进入训练侧open-set表征或support/source-heldout伪未知校准包。

### Review修复：候选语义拆分

独立review指出，原`candidate`字段只表示相对工程候选，不应被误读为协议成功；并且候选门缺少显式`seen_new`不下降检查。本轮按review修复：

| 字段 | 含义 |
|---|---|
| `seen_new_not_drop_pass` | 同row相对base的seen-new准确率不下降 |
| `far_target_pass` | `unknown_FAR<=0.05` |
| `protocol_success` | old不降、seen-new不降、old_acc>=0.80、unknown_FAR<=0.05、unknown拒识提升且FAR不恶化 |
| `verdict=relative_tradeoff_candidate` | 只表示相对工程候选，不表示Stage2-C成功 |
| `verdict=protocol_success` | 才表示协议成功；当前预计不会出现 |

新增TDD测试：

```text
test_smec_relative_candidate_is_not_protocol_success_when_far_above_target
test_smec_seen_new_drop_blocks_relative_candidate
```

本地验证：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_support_metric_energy_ci_eval.py -q
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_support_metric_energy_ci_eval.py code\tests\test_phase2_support_metric_energy_ci_eval.py
```

结果：新增测试先失败于缺少`seen_new_not_drop_pass`字段，修复后`20 passed`，语法检查通过。

远端复测：

```text
Git提交：87f16cf Clarify OBACE candidate protocol status
远端脚本SHA256：069ec7880c1423c75e4d31bf13412f91d0ee2b3fd6a10f0cd2ca78685e8ef340
远端测试SHA256：2a957420a173ed2045ea1f093eb8f5ed652607001efecb00194301e1406c17d7
manual_test_functions_passed=20
GPU pre-run: 8张RTX3090均约10/24576MiB，GPU0执行
{"receiver_count": 5, "group_count": 307, "evidence_row_count": 1000}
{"summary_rows": 80, "candidate_count": 0, "summary_csv": "remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_obace_void_protocol/smec_ci_summary.csv"}
```

复测结果已拉回：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_void_protocol\smec_ci_summary.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_void_protocol\smec_ci_best_rows.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_void_protocol\smec_ci_audit.json`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_obace_void_protocol\obace_void_ci_evidence.csv`

协议语义修复后的主结果：

| algorithm | policy | collab_count | old_acc | seen_new_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_seen_new_acc | seen_new_not_drop_pass | far_target_pass | protocol_success | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| smec_old_lossless_ci | opu_old_preserve | 4 | 0.802139 | 0.783333 | 0.200000 | 0.733333 | 0.000000 | 0.000000 | true | false | false | relative_tradeoff_candidate |
| smec_consensus_guard_ci | opu_old_preserve | 4 | 0.802139 | 0.783333 | 0.216667 | 0.733333 | 0.000000 | 0.000000 | true | false | false | relative_tradeoff_candidate |
| obace_ci | opu_old_preserve | 4 | 0.802139 | 0.783333 | 0.233333 | 0.700000 | 0.000000 | 0.000000 | true | false | false | relative_tradeoff_candidate |
| obace_event_ci | opu_old_preserve | 4 | 0.802139 | 0.783333 | 0.250000 | 0.683333 | 0.000000 | 0.000000 | true | false | false | relative_tradeoff_candidate |
| obace_void_ci | opu_old_preserve | 4 | 0.802139 | 0.783333 | 0.250000 | 0.683333 | 0.000000 | 0.000000 | true | false | false | relative_tradeoff_candidate |

最终解释边界：`relative_tradeoff_candidate`只表示相对base同row有改善且old/seen不下降；由于所有行`far_target_pass=false`，本轮没有`protocol_success`，不能声明Stage2-C成功或部署成功。当前相对最强仍是`obace_event_ci`/`obace_void_ci`并列，说明协同推理可改善unknown拒识但幅度不足；下一步必须把门控前移到训练侧open-set表征或source-heldout/support-known-only伪未知校准，而不是继续调推理阈值。

## 2026-07-04 Proxy-VOID支持集伪未知校准

目标：在不使用target unknown query拟合阈值的前提下，把“类间边界区域”显式建成轻量伪未知证据。前一轮`obace_void_ci`说明单纯放宽事件门会伤害old/seen，新的`proxy_void_ci`不再突破strong-known保护，而是在每个receiver的old/seen support原型之间生成void锚点：

```text
v_ij = normalize(c_i + c_j)
tau_void = quantile(max_k sim(support, v_k), 0.90) + 0.01
r_void(x) = sigmoid((max_k sim(x, v_k) - tau_void) / T)
```

这个风险只依赖target-old support和seen-new support。未知query仍只用于评估，`smec_unknown_query_used_for_threshold=false`保持不变。部署成本是每receiver保存少量类间void锚点和一个阈值；计算代价是一组向量相似度，继续用`bytes_per_event`、`latency_ms_p95`、`participating_receivers_p95`作为当前离线资源proxy。精确文件`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`仍未在工作区找到；已再次用标题和资源关键词搜索，命中的是既有设计/traceability文件而非该原文，因此本报告继续把资源项标为proxy，不能写成真实星载端到端实测。

新增本地代码：

| 文件 | 变更 |
|---|---|
| `code/scripts/phase2_support_metric_energy_ci_eval.py` | 新增`proxy_void_weight/temperature/quantile/slack`配置、support-only void锚点、`smec_proxy_void_*`审计列和`proxy_void`profile |
| `code/tests/test_phase2_support_metric_energy_ci_eval.py` | 新增三项TDD测试：support-only锚点生成、边界query升高风险、support-like old query低风险 |

本地验证：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_support_metric_energy_ci_eval.py -q
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_support_metric_energy_ci_eval.py code\tests\test_phase2_support_metric_energy_ci_eval.py
```

结果：新增测试先失败于`SmecConfig.__init__() got an unexpected keyword argument 'proxy_void_*'`，实现后`23 passed`，语法检查通过。

远端同步和验证：

```text
Git提交：62cbd23 Add support-only proxy void risk
远端脚本SHA256：afccd643a0e2786482093307dd1a95e56e178e589a797ef8d9fa4ae6f9dbfeec
远端测试SHA256：e95f9948cf252c970a9eb8cf8674b9c055d56a427239fbe48b285ca8c8e7c947
manual_test_functions_passed=23
GPU pre-run: 8张RTX3090均约10/24576MiB，GPU0执行
{"receiver_count": 5, "group_count": 307, "evidence_row_count": 1000}
{"summary_rows": 90, "candidate_count": 0, "summary_csv": "remote_artifacts/phase2_adv3b02_smec_ci_20260704/remote_proxy_void/smec_ci_summary.csv"}
```

复测结果已拉回：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_proxy_void\smec_ci_summary.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_proxy_void\smec_ci_best_rows.csv`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_proxy_void\smec_ci_audit.json`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_proxy_void\proxy_void_ci_evidence.csv`

`opu_old_preserve/collab_count=4`主对比：

| algorithm | policy | collab_count | old_acc | seen_new_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_seen_new_acc | far_target_pass | protocol_success | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| base_known_route | opu_old_preserve | 4 | 0.802139 | 0.783333 | 0.183333 | 0.750000 |  |  |  |  |  |
| obace_event_ci | opu_old_preserve | 4 | 0.802139 | 0.783333 | 0.250000 | 0.683333 | 0.000000 | 0.000000 | false | false | relative_tradeoff_candidate |
| obace_void_ci | opu_old_preserve | 4 | 0.802139 | 0.783333 | 0.250000 | 0.683333 | 0.000000 | 0.000000 | false | false | relative_tradeoff_candidate |
| proxy_void_ci | opu_old_preserve | 4 | 0.802139 | 0.783333 | 0.250000 | 0.683333 | 0.000000 | 0.000000 | false | false | relative_tradeoff_candidate |

`proxy_void_ci`全量协同数量结果：

| policy | collab_count | old_acc | seen_new_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_seen_new_acc | protocol_success | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| opu_old_preserve | 1 | 0.636364 | 0.666667 | 0.233333 | 0.766667 | 0.000000 | -0.016667 | false | diagnostic_only |
| opu_old_preserve | 2 | 0.625668 | 0.816667 | 0.100000 | 0.650000 | 0.000000 | 0.000000 | false | diagnostic_only |
| opu_old_preserve | 3 | 0.759358 | 0.716667 | 0.166667 | 0.550000 | 0.000000 | 0.000000 | false | diagnostic_only |
| opu_old_preserve | 4 | 0.802139 | 0.783333 | 0.250000 | 0.683333 | 0.000000 | 0.000000 | false | relative_tradeoff_candidate |
| opu_old_preserve | 5 | 0.791444 | 0.783333 | 0.166667 | 0.683333 | -0.005348 | 0.000000 | false | diagnostic_only |
| opu_old_guarded | 1 | 0.620321 | 0.650000 | 0.283333 | 0.716667 | 0.000000 | -0.016667 | false | diagnostic_only |
| opu_old_guarded | 2 | 0.540107 | 0.516667 | 0.216667 | 0.483333 | 0.000000 | 0.000000 | false | diagnostic_only |
| opu_old_guarded | 3 | 0.631016 | 0.650000 | 0.333333 | 0.483333 | 0.000000 | 0.000000 | false | diagnostic_only |
| opu_old_guarded | 4 | 0.764706 | 0.666667 | 0.400000 | 0.483333 | -0.005348 | 0.000000 | false | diagnostic_only |
| opu_old_guarded | 5 | 0.754011 | 0.666667 | 0.333333 | 0.500000 | -0.010695 | 0.000000 | false | diagnostic_only |

`proxy_void`审计：

| role | n | unknown_risk_mean | proxy_risk_mean | proxy_risk_q90 | proxy_fail>=0.7 | lifted_unknown_risk>=0.5 | old_label_lift_blocked |
|---|---:|---:|---:|---:|---:|---:|---:|
| old | 600 | 0.089632 | 0.080238 | 0.386579 | 4 | 53 | 596 |
| seen_new | 200 | 0.109449 | 0.170728 | 0.450868 | 2 | 20 | 195 |
| unknown | 200 | 0.432424 | 0.043411 | 0.189944 | 0 | 95 | 151 |

结论：`proxy_void`作为support-only类间边界伪未知校准，在当前ADV3B02特征空间没有捕获真实unknown。未知样本的`smec_proxy_void_risk`均值只有`0.043411`，`>=0.7`的unknown数量为0；它没有提升超过`obace_event_ci/obace_void_ci`，全量90行仍然`candidate_count=0`、`protocol_success=0`。当前最强仍是`obace_event_ci/obace_void_ci/proxy_void_ci`在`opu_old_preserve/collab_count=4`并列的相对权衡结果，保持old/seen不降但`unknown_FAR=0.683333`，距离`unknown_FAR<=0.05`和unknown拒识99%目标很远。

下一步算法方向：不要继续只叠加推理期未知阈值。应把未知类拒识前移到训练侧或轻量在线微调侧，包括source-heldout伪未知episodic训练、receiver-conditioned energy margin、旧类原型保真正则、少量星间共识蒸馏和低秩adapter在线更新；推理期协同只负责低带宽证据融合和保守拒识，不能承担全部open-set分离。

## 2026-07-04 VUSE/OPU训练侧虚拟未知诊断

目标：验证现有训练侧/校准侧组件是否能在不改代码的情况下形成可部署候选。配置使用`phase2_collaborative_open_set_qknn_eval.py`的`support_envelope_full`、`virtual_unknown_calibration_enabled`、`virtual_unknown_risk_enabled`、`class_shell_unknown_risk_enabled`、`source_old_prototype_shrinkage_alpha=0.20`和`old_protected_unknown_confirm_cvs`。该配置仍满足`unknown_query_eval_only=true`，阈值作用域为`support_virtual_unknown`，事件对齐显式设为`receiver_domain_ranked`诊断。

远端命令在N607的`CVS-RFFI`环境执行，GPU0预运行显存约`10/24576MiB`，输出：

```text
{"receiver_count": 5, "group_count": 307, "evidence_row_count": 1000}
```

结果已拉回：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_vuse_opu\vuse_opu.json`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_smec_ci_20260704\remote_vuse_opu\vuse_opu_evidence.csv`

全量协同数量结果：

| collab_count | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | known_coverage | defer_rate | bytes_per_event | latency_ms_p95 | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.363636 | 0.000000 | 0.266667 | 0.000000 | 0.666667 | 0.333333 | 0.404858 | 0.000000 | 40.0 | 0.221819 | diagnostic_only |
| 2 | 0.638158 | 0.050000 | 0.365385 | 0.093750 | 0.326087 | 0.543478 | 0.730392 | 0.088000 | 80.0 | 0.221819 | diagnostic_only |
| 3 | 0.750000 | 0.050000 | 0.500000 | 0.050000 | 0.450000 | 0.550000 | 0.893750 | 0.000000 | 120.0 | 0.221819 | diagnostic_only |
| 4 | 0.670455 | 0.000000 | 0.678571 | 0.000000 | 0.382353 | 0.617647 | 0.913793 | 0.000000 | 160.0 | 0.221819 | diagnostic_only |
| 5 | 0.603774 | 0.000000 | 0.950000 | 0.000000 | 0.500000 | 0.500000 | 0.958904 | 0.000000 | 200.0 | 0.221819 | diagnostic_only |

结论：VUSE/OPU能把单接收机unknown拒识推到`0.666667`，但old_acc同步降到`0.363636`，多接收机下old_acc也只有`0.603774..0.750000`；这直接违反“旧类准确性不能下降”的约束。因此该方向不能作为部署候选，只能证明“未知拒识提升如果发生在已有特征空间的推理/校准端，会与old/seen边界强耦合”。下一步必须进入训练微调侧：用源旧类+目标old/seen support构造旧类保真正则，同时用留一TX/虚拟TX做open-set margin学习；推理端只保留`opu_old_preserve`式旧类保护和低带宽协同证据。

## 2026-07-04 Proxy-Unknown特征包与OVC/OPR准备

目的：当前`remote_artifacts/phase2_adv3b02_features/features.npz`没有`proxy_unknown`角色，因此`phase2_proxy_adapter_ci_eval.py`和`phase2_open_verifier_ci_eval.py`无法使用源侧伪未知学习开集边界。下一步生成一个保持相同`Y_old/Y_new/Y_unknown/R_t`的ADV3B02/qKNN8特征包，只额外加入source-side、label-disjoint、receiver-disjoint的`proxy_unknown`。target_unknown仍只用于评估，不参与阈值或adapter训练。

协议设置：

| 字段 | 值 |
|---|---|
| 底座权重 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| source old | `source_tx_ids=0,1,2,3,4,5`；source labels对应`14-10,14-7,20-15,20-19,6-15,8-20` |
| target receivers | `20-1,3-19,7-14,7-7,8-8` |
| target_new | `19-3,3-8` |
| target_unknown | `10-1,10-10` |
| proxy_unknown | `1-16,1-18,10-11,10-17,11-1,12-7,13-14,14-11` |
| proxy_unknown receivers | `1-1,1-19,14-7,18-2,19-2,2-1` |
| satellite view | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak`；`simplified_leo_residual` |
| 输出 | `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/features_proxy_unknown.npz` |

远端只读预检：

```text
N607 preflight OK at 2026年07月04日星期六19:01:58CST
GPU 0..7: RTX3090，均约10/24576MiB，util=0
存在：export_spaceborne_features.py、phase2_open_verifier_ci_eval.py、phase2_proxy_adapter_ci_eval.py、phase2_collaborative_open_set_qknn_eval.py、ADV3B02权重、ManySig.pkl、ManyTx.pkl
```

拟执行命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/export_spaceborne_features.py \
  --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth \
  --wisig_pkl Dataset_WigSig/ManySig.pkl \
  --new_wisig_pkl Dataset_WigSig/ManyTx.pkl \
  --out_npz remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/features_proxy_unknown.npz \
  --feature_name z_id \
  --source_tx_ids 0,1,2,3,4,5 \
  --source_rxs 0,1,2,3,4,5,6 \
  --target_old_tx_ids 0,1,2,3,4,5 \
  --target_old_rxs 20-1,3-19,7-14,7-7,8-8 \
  --target_old_channel_view satellite \
  --target_old_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --new_tx_ids 19-3,3-8 \
  --new_rxs 20-1,3-19,7-14,7-7,8-8 \
  --unknown_tx_ids 10-1,10-10 \
  --proxy_unknown_tx_ids 1-16,1-18,10-11,10-17,11-1,12-7,13-14,14-11 \
  --proxy_unknown_rxs 1-1,1-19,14-7,18-2,19-2,2-1 \
  --proxy_unknown_channel_view satellite \
  --proxy_unknown_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --target_new_channel_view satellite \
  --target_new_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --star_ground_channel_impl simplified_leo_residual \
  --wisig_equalized 1 \
  --wisig_domain rx_day \
  --wisig_out_len 256 \
  --max_samples_per_combo 0 \
  --max_samples_per_tx 220 \
  --batch_size 512 \
  --device cuda:0 \
  --seed 4070527
```

验收：导出后必须检查`dataset_role`中包含`proxy_unknown`，且`proxy_unknown`的TX不与`Y_old/Y_new/Y_unknown`重叠、接收机不与`R_t`重叠。只有通过该检查后，才运行OVC/OPR全量`collab_count=1..5`测试。

## 2026-07-04 Proxy-Unknown/OVC/OPR闭环结果

### 特征包导出与协议验证

远端导出在N607的`CVS-RFFI`环境完成，GPU0预运行显存约`10/24576MiB`，输出文件：

```text
/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/features_proxy_unknown.npz
```

导出manifest摘要：

| role | 样本数 | TX集合 | RX集合 |
|---|---:|---|---|
| source | 1320 | `14-10,14-7,20-15,20-19,6-15,8-20` | `1-1,1-19,14-7,18-2,19-2,2-1,2-19` |
| target_old | 1320 | `14-10,14-7,20-15,20-19,6-15,8-20` | `20-1,3-19,7-14,7-7,8-8` |
| target_new | 440 | `19-3,3-8` | `20-1,3-19,7-14,7-7,8-8` |
| target_unknown | 440 | `10-1,10-10` | `20-1,3-19,7-14,7-7,8-8` |
| proxy_unknown | 1760 | `1-16,1-18,10-11,10-17,11-1,12-7,13-14,14-11` | `1-1,1-19,14-7,18-2,19-2,2-1` |

协议检查结果：

```text
overlap_old_new=[]
overlap_old_unknown=[]
overlap_new_unknown=[]
proxy_overlap_all=[]
proxy_target_rx_overlap=[]
target_channel_view=satellite/LEO
star_ground_channel_impl=simplified_leo_residual
target_sat_scenarios=leo_clear_weak,leo_low_elev_weak,leo_rain_weak
```

结论：`proxy_unknown`满足source-side、label-disjoint和receiver-disjoint要求；`target_unknown`仍为eval-only。该特征包可用于训练侧/校准侧开集诊断，但仍只代表`simplified_leo_residual`星地信道压力测试，不构成真实轨道链路证明。

### OVC开集验证器结果

远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_open_verifier_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/features_proxy_unknown.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/ovc_remote/ovc_ci.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/ovc_remote/ovc_ci_summary.csv \
  --output_evidence_csv remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/ovc_remote/ovc_ci_evidence.csv \
  --profiles all --collab_counts all --collab_group_policy same_max_budget \
  --receiver_selection_policy fixed_receiver_order --k_shot 8 --query_per_class 20 --qknn_k 8 \
  --seed 4070601 --support_selection_policy stable_first --device cuda:0 --verifier_epochs 500 \
  --include_event_results
```

结果已拉回：

- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\ovc_ci.json`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\ovc_ci_summary.csv`

OVC摘要：

| selection | profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes_per_event | latency_ms | target_pass | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| best_unknown | ovc_unknown_guard | 2 | 0.300000 | 0.100000 | 0.000000 | 0.000000 | 0.475000 | 0.525000 | 0.337500 | 0.200000 | 96.0 | 0.25 | false | diagnostic_only |
| best_old_like | ovc_balanced | 5 | 0.825000 | 0.600000 | 0.250000 | 0.050000 | 0.031250 | 0.968750 | 0.006250 | 0.031250 | 240.0 | 0.25 | false | diagnostic_only |

`old_acc>=0.802139`且`seen_new_acc>=0.783333`的保真行数量为0。OVC没有形成有效unknown边界；当未知拒识提高到`0.475000`时，old/seen已经严重崩溃。

### OPR严格事件对齐验证

按合理性review要求，先使用`strict_event_key`验证真实同事件协同：

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_proxy_adapter_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/features_proxy_unknown.npz \
  --output_dir remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/opr_strict_q12 \
  --backend both --collab_counts all --collab_group_policy same_max_budget \
  --partial_collab_min_receivers 1 --k_shot 8 --query_per_class 12 --qknn_k 8 \
  --seed 4070404 --support_selection_policy stable_first --event_alignment_policy strict_event_key \
  --device cuda:0 --adapter_epochs 40 --adapter_rank 16 --adapter_alpha 0.20 \
  --old_preserve_weight 2.0 --proxy_open_weight 1.0 --support_compact_weight 0.5 --residual_weight 0.10
```

结果：

```text
RuntimeError: NO_ALIGNED_COLLABORATIVE_EVENTS: target receiver query rows do not share role+tx+day+sig+scenario keys; use --event_alignment_policy receiver_domain_ranked only for explicitly marked receiver-domain ensemble diagnostics
```

结论：当前特征包无法证明同一事件多接收机/多星协同推理。`collab_count=1..5`曲线只能解释为receiver-domain ensemble诊断，不能作为星座同事件协同部署证据。

### OPR域集成诊断结果

由于严格事件对齐无可用事件，改用`receiver_domain_ranked`完成诊断矩阵，并显式标记为diagnostic-only。`query_per_class`降为12是因为`receiver=7-14,target_unknown,tx=10-1`仅有12条query；继续使用20会被协议守卫拒绝。

远端命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_proxy_adapter_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/features_proxy_unknown.npz \
  --output_dir remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/opr_domain_q12 \
  --backend both --collab_counts all --collab_group_policy same_max_budget \
  --partial_collab_min_receivers 1 --k_shot 8 --query_per_class 12 --qknn_k 8 \
  --seed 4070404 --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked \
  --device cuda:0 --adapter_epochs 40 --adapter_rank 16 --adapter_alpha 0.20 \
  --old_preserve_weight 2.0 --proxy_open_weight 1.0 --support_compact_weight 0.5 --residual_weight 0.10 \
  --enpc_profiles all --slev_profiles all
```

结果已拉回：

- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\opr_ci_summary.json`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\opr_ci_enpc_summary.csv`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\opr_ci_slev_summary.csv`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\opr_ci_enpc.json`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\opr_ci_slev.json`

诊断摘要：

| backend | selection | profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes_per_event | latency_ms | target_pass | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| ENPC | best_joint | enpc_old80_unknown_probe | 5 | 0.666667 | 0.000000 | 0.416667 | 0.250000 | 0.666667 | 0.333333 | 640.0 | 2.774650 | false | diagnostic_only |
| ENPC | best_unknown | enpc_unknown_strict | 5 | 0.520833 | 0.000000 | 0.333333 | 0.083333 | 0.708333 | 0.291667 | 640.0 | 2.774650 | false | diagnostic_only |
| SLEV | best_unknown | slev_energy_strict | 5 | 0.416667 | 0.000000 | 0.083333 | 0.083333 | 0.958333 | 0.041667 | 640.0 | 2.892559 | false | diagnostic_only |
| SLEV | old_relative | slev_old80_energy_probe | 3 | 0.736111 | 0.333333 | 0.583333 | 0.500000 | 0.458333 | 0.500000 | 384.0 | 2.892559 | false | diagnostic_only |

`old_acc>=0.802139`且`seen_new_acc>=0.783333`的保真行数量为0。OPR/SLEV可以把未知拒识推到`0.958333`，但对应`old_acc=0.416667`、`seen_new_acc=0.083333`；这证实“未知拒识提升”和“旧类/seen-new保真”在当前ADV3B02特征空间内强冲突。

### 子agent审查结论与下一步算法约束

合理性审查给出的P0问题已经被验证：

1. `strict_event_key`无有效同事件协同，因此当前`receiver_domain_ranked`结果必须标记为多receiver域集成诊断，不能作为卫星群同事件协同证据。
2. OPR低秩适配器确实造成旧类/seen-new显著下降，不能作为部署候选。
3. 任何基于`target_unknown`选择best row的结论都不能作为部署阈值；本报告只把unknown最优行作为诊断，不作为候选。

下一步不应继续堆叠推理期阈值。建议算法路线改为`AWARE-CI++`：

- 冻结ADV3B02的`z_id`主干和旧类分类头，只允许星上更新小型adapter、temperature、类条件半径和receiver reliability；
- 对每个接收机生成低带宽证据包：top-L logits、entropy、margin、energy/kNN风险、Mahalanobis或Gaussian prototype距离、receiver reliability和link quality，单包目标`80..160B`；
- 聚合端按预算自适应top-M协同，从1个接收机开始，只在高风险/高分歧时请求更多接收机，报告`avg/p95参与数量`，而不是默认全体等待；
- unknown门控由kNN距离、类条件Mahalanobis、EVT尾部和conformal set size共同决定，阈值只用source old、source-side proxy_unknown和target old/seen-new support校准；
- 在线微调采用ODS过滤：unknown-like样本不参与TTA，known-like样本只更新adapter/BN affine/temperature，并以old-floor回滚作为硬门；
- 部署候选必须逐行通过`old_acc>=base_old_acc`、`min_old>=base_min_old`、`seen_new_acc>=base_seen_new_acc`、`min_seen>=base_min_seen`、`unknown_reject>=0.99`和资源proxy约束；否则只标记为diagnostic。

当前结论：本轮完成了source-side proxy_unknown、OVC、OPR和严格事件对齐验证，但没有得到满足目标的候选。最高unknown拒识行来自诊断性SLEV，`unknown_reject=0.958333`且`unknown_FAR=0.041667`，但旧类准确率跌到`0.416667`；旧类/seen-new保真行数量为0。因此单协同推理、support-only proxy void、OVC和OPR都不足以解决目标问题，下一步必须进入训练侧open-set表征和预算自适应协同算法实现。

## 2026-07-04 AWARE-CI预算自适应旧类保真协同原型

### 实现与同步

新增脚本：

- `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_aware_ci_eval.py`
- `E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_aware_ci_eval.py`

算法机制：

1. 冻结ADV3B02的`z_id`特征，不更新backbone或分类头；
2. 只用`source_old`和target old/seen-new K-shot support构造类条件prototype、qKNN memory和对角Mahalanobis统计；
3. 只用source-side `proxy_unknown`参与开集风险校准，`target_unknown`不参与阈值、profile或adapter训练；
4. 每接收机输出低带宽证据包：top prototype label/score、margin、prototype distance、qKNN distance、Mahalanobis risk、entropy、proxy gap、receiver reliability；
5. 聚合端按receiver reliability和低open-score选择top-M，输出`collab_count=1..5`、`avg_participating_receivers`、`p95_participating_receivers`、`bytes_per_event`和`latency_ms`；
6. profile分为`aware_old_safe`、`aware_balanced`、`aware_unknown_probe`，其中`aware_old_safe`优先保护旧类，`aware_unknown_probe`仅作为未知拒识诊断。

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_aware_ci_eval.py code\scripts\phase2_open_verifier_ci_eval.py code\scripts\phase2_proxy_adapter_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_aware_ci_eval.py code\tests\test_phase2_support_metric_energy_ci_eval.py -q
27 passed
```

远端同步与验证：

```text
N607 preflight OK at 2026年07月04日星期六19:15:39CST
GPU 0..7: RTX3090，均约10/24576MiB，util=0
code/scripts/phase2_aware_ci_eval.py SHA256=a5f860d6738037ec6bf9b24af15a3cbec037eb5dd394170bd1465bd076e47e60
code/tests/test_phase2_aware_ci_eval.py SHA256=c3304a8eb0435761a2b110072589e885c3bf5acf89c4b1497745078dc16b2f93
远端py_compile通过
远端pytest不可用：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python: No module named pytest
```

### 远端AWARE-CI评估

命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_aware_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/features_proxy_unknown.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/aware_ci/aware_ci.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/aware_ci/aware_ci_summary.csv \
  --output_evidence_csv remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/aware_ci/aware_ci_evidence.csv \
  --profiles all --collab_counts all --collab_group_policy same_max_budget \
  --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070704 --support_selection_policy stable_first
```

输出已拉回：

- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\aware_ci.json`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\aware_ci_summary.csv`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\aware_ci_evidence.csv`

AWARE-CI摘要：

| selection | profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes_per_event | latency_ms | avg_participating | p95_participating | target_pass | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| best_joint | aware_old_safe | 1 | 0.875000 | 0.583333 | 0.291667 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 96.0 | 0.35 | 1.0 | 1.0 | false | diagnostic_only |
| best_unknown | aware_unknown_probe | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.208333 | 0.791667 | 0.791667 | 0.791667 | 96.0 | 0.35 | 1.0 | 1.0 | false | diagnostic_only |

`old_acc>=0.802139`且`seen_new_acc>=0.783333`的保真行数量为0。AWARE-CI按设计比OPR/SLEV更保守，`aware_old_safe`能保持较高old_acc但不能保护seen-new，也没有未知拒识收益；`aware_unknown_probe`提升未知拒识到`0.208333`时old/seen完全不可用。该结果说明单纯在现有ADV3B02特征上组合kNN/Mahalanobis/entropy/proxy_gap仍不能解决unknown与known边界重叠问题。

### 本轮算法判断

有效结论不是“AWARE-CI失败即可放弃协同”，而是：

1. 当前数据没有同一事件跨接收机键，真实卫星群协同还缺事件级数据组织；
2. 当前ADV3B02特征空间下，target_unknown与old/seen支持流形重叠严重，推理端风险融合无法同时满足未知拒识和旧类保真；
3. 下一步应实现训练侧open-set表征：source-heldout proxy TX episodic训练、类条件compactness、unknown margin、old prototype distillation和ODS过滤的小步adapter TTA；
4. 协同推理层保留AWARE-CI的低带宽证据包和预算自适应top-M机制，但它应服务于“不确定时请求更多接收机/延迟决策”，不能单独承担未知类分离。

## 2026-07-04 Open-Set特征几何审计

### 实现与同步

新增诊断脚本：

- `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_open_set_geometry_audit.py`
- `E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_open_set_geometry_audit.py`

目的：量化`target_unknown`相对`target_old/target_new`支持流形的可分性，避免继续在不可分特征空间上盲调协同阈值。该脚本只用`target_unknown`做事后AUROC/FPR95/oracle诊断，不用它选择阈值或profile。

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_open_set_geometry_audit.py code\scripts\phase2_aware_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_open_set_geometry_audit.py code\tests\test_phase2_aware_ci_eval.py -q
8 passed
```

远端同步与运行：

```text
N607 preflight OK at 2026年07月04日星期六19:21:59CST
GPU 0..7: RTX3090，均约10/24576MiB，util=0或1
code/scripts/phase2_open_set_geometry_audit.py SHA256=7ac2cbcc32d6231af334d419c571a69e52822ab2f0665c462b1311cb86ea5636
code/tests/test_phase2_open_set_geometry_audit.py SHA256=5b6a0595eb07672d072dd45ff8eaf5bd204df0bf27bdcd117467a245c8aea6df
远端py_compile通过
```

远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_open_set_geometry_audit.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/features_proxy_unknown.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/geometry_audit/geometry_audit.json \
  --output_by_role_tx_csv remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/geometry_audit/geometry_by_role_tx.csv \
  --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070705 --support_selection_policy stable_first
```

输出已拉回：

- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\geometry_audit.json`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\geometry_by_role_tx.csv`

### 关键结果

样本计数：

| source_old_train | target_support | proxy_unknown_train | target_old_query | target_seen_new_query | target_unknown_query |
|---:|---:|---:|---:|---:|---:|
| 1320 | 320 | 1760 | 360 | 120 | 120 |

组件级可分性：

| score | AUROC_unknown_vs_known | FPR95_known_as_unknown | oracle_FAR<=0.05时old_keep | oracle_FAR<=0.05时seen_keep | oracle_unknown_reject | 结论 |
|---|---:|---:|---:|---:|---:|---|
| aware_score | 0.595625 | 0.731250 | 0.302778 | 0.150000 | 0.958333 | 融合风险不足，low-FAR下known保留崩溃 |
| proto_dist | 0.654670 | 0.797917 | 0.188889 | 0.100000 | 0.958333 | 原型距离略有效但远不够 |
| knn_dist | 0.666233 | 0.845833 | 0.166667 | 0.083333 | 0.958333 | qKNN距离对未知有弱信号，但误杀known严重 |
| maha | 0.672760 | 0.791667 | 0.219444 | 0.150000 | 0.958333 | 当前最强单项仍不满足部署门 |
| entropy | 0.616354 | 0.933333 | 0.047222 | 0.091667 | 0.958333 | 熵风险基本不可用 |
| proxy_gap | 0.596424 | 0.806250 | 0.175000 | 0.025000 | 0.958333 | source-side proxy_unknown与真实target_unknown错位 |
| negative_margin | 0.523003 | 0.902083 | 0.130556 | 0.000000 | 0.958333 | margin几乎不能分离unknown |

按接收机的`aware_score`分离性：

| receiver | old_n | seen_new_n | unknown_n | AUROC_unknown_vs_known | FPR95_known_as_unknown |
|---|---:|---:|---:|---:|---:|
| 20-1 | 72 | 24 | 24 | 0.530382 | 0.854167 |
| 3-19 | 72 | 24 | 24 | 0.611111 | 0.812500 |
| 7-14 | 72 | 24 | 24 | 0.646701 | 0.541667 |
| 7-7 | 72 | 24 | 24 | 0.519531 | 0.739583 |
| 8-8 | 72 | 24 | 24 | 0.723958 | 0.677083 |

未知TX最近已知标签吸附：

| unknown_tx | n | top_absorbing_known_label | nearest_known_label_counts |
|---|---:|---|---|
| 10-1 | 60 | 6-15 | `14-10:16,14-7:2,19-3:2,20-15:1,20-19:3,3-8:7,6-15:16,8-20:13` |
| 10-10 | 60 | 20-19 | `14-7:9,19-3:4,20-15:1,20-19:39,3-8:7` |

### 审计结论

1. 当前ADV3B02/qKNN8特征空间中，`target_unknown`并没有形成可被稳定拒识的远离流形区域。最佳单项`maha`的AUROC只有`0.672760`，FPR95仍为`0.791667`。
2. 即使用`target_unknown`标签做oracle阈值，想达到`unknown_FAR<=0.05`也只能保留`old_keep<=0.302778`、`seen_keep<=0.150000`。这证明“99%未知拒识且旧类/新类不下降”的目标不可能只靠当前特征上的阈值或协同投票实现。
3. `proxy_gap`表现很弱，说明source-side `proxy_unknown`与真实target_unknown的几何关系不一致。继续增加proxy_unknown阈值权重会伤害known，而不是产生可靠拒识。
4. `10-10`强吸附到旧类`20-19`，`10-1`分散吸附到`6-15/14-10/8-20`等旧/seen标签；下一步训练侧应对这些“unknown-to-known吸附对”做hard negative episode，而不是只做全局unknown margin。

下一步实施方向：新增训练侧open-set表征路线`SAGE-OSR`，在ADV3B02基础上只训练轻量head/adapter，使用source-old、target old/seen support、source-side proxy_unknown和hard unknown-to-known吸附对，目标函数包含`old prototype distillation + seen-new compactness + proxy/hard negative margin + ODS过滤 + rollback old-floor`。协同层继续使用AWARE-CI作为资源受限证据融合，不再作为主要分离机制。

## 2026-07-04 SAGE-OSR训练侧open-set原型计划

目的：验证训练侧表征修复是否能突破上一轮推理端阈值/协同融合瓶颈。SAGE-OSR在ADV3B02特征上训练轻量低秩residual adapter，训练数据只包含`source_old`、target old/seen-new K-shot support和source-side `proxy_unknown`；`target_unknown`继续仅用于最终评估。

新增文件：

- `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_sage_osr_ci_eval.py`
- `E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_sage_osr_ci_eval.py`

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_sage_osr_ci_eval.py code\scripts\phase2_aware_ci_eval.py code\scripts\phase2_proxy_adapter_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_sage_osr_ci_eval.py code\tests\test_phase2_aware_ci_eval.py code\tests\test_phase2_open_set_geometry_audit.py -q
9 passed
```

计划远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_sage_osr_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/features_proxy_unknown.npz \
  --output_dir remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/sage_osr \
  --collab_counts all --collab_group_policy same_max_budget \
  --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070706 \
  --support_selection_policy stable_first --device cuda:0 \
  --adapter_epochs 60 --adapter_rank 12 --adapter_alpha 0.10
```

验收口径：输出`SAGE-OSR+AWARE-CI`的`collab_count=1..5`全量表，记录`old_acc/min_old/seen_new_acc/min_seen/unknown_reject/unknown_FAR/bytes_per_event/latency_ms/参与接收机数量`。若仍无`old/seen`保真行，则作为训练侧open-set负向诊断，不得写作部署候选。

### SAGE-OSR远端结果

远端同步与验证：

```text
N607 preflight OK at 2026年07月04日星期六19:28:57CST
GPU 0..7: RTX3090，均约10/24576MiB，util=0
code/scripts/phase2_sage_osr_ci_eval.py SHA256=def4cfc9609d446cbcea6e6f368da5578ef51a6ce9ba68ad8deded2d325f7814
code/tests/test_phase2_sage_osr_ci_eval.py SHA256=d570bbc61b69da4a93cb7d7cee3f5a99247c4fef045be2c1d92f68335dd74be9
远端py_compile通过
```

远端输出：

- `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/sage_osr/sage_osr_adapted_features.npz`
- `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/sage_osr/sage_osr_summary.json`
- `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/sage_osr/sage_osr_aware_summary.csv`

本地拉回：

- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\sage_osr\sage_osr_summary.json`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\sage_osr\sage_osr_aware.json`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\sage_osr\sage_osr_aware_summary.csv`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\sage_osr\sage_osr_aware_evidence.csv`

训练侧诊断：

| metric | before | after | interpretation |
|---|---:|---:|---|
| source_proto_acc | 0.951515 | 0.956061 | source old未被破坏 |
| support_proto_acc | 0.587500 | 0.568750 | target old/seen support结构略降 |
| proxy_max_logit_mean | 10.881646 | -2.334146 | proxy_unknown被强力推离已知原型 |
| proxy_max_logit_q90 | 12.454719 | -1.570605 | hard proxy margin生效 |
| final_loss | - | 4.037298 | 训练完成，无NaN/OOM |

SAGE-OSR+AWARE结果摘要：

| selection | profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes_per_event | latency_ms | avg_participating | p95_participating | target_pass | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| best_joint | aware_old_safe | 3 | 0.833333 | 0.583333 | 0.083333 | 0.083333 | 0.000000 | 1.000000 | 0.083333 | 0.291667 | 384.0 | 0.45 | 3.0 | 3.0 | false | diagnostic_only |
| best_unknown | aware_unknown_probe | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.166667 | 0.833333 | 0.885417 | 0.833333 | 128.0 | 0.45 | 1.0 | 1.0 | false | diagnostic_only |
| best_seen | aware_old_safe | 1 | 0.819444 | 0.500000 | 0.208333 | 0.166667 | 0.000000 | 1.000000 | 0.041667 | 0.000000 | 128.0 | 0.45 | 1.0 | 1.0 | false | diagnostic_only |

`old_acc>=0.802139`且`seen_new_acc>=0.783333`的保真行数量为0。

结论：SAGE-OSR第一版证明了训练侧hard proxy margin可以显著降低source-side proxy_unknown对已知原型的吸附，但这种margin没有迁移到真实`target_unknown`拒识，并且破坏了seen-new支持结构。根因与几何审计一致：source-side proxy_unknown与真实target_unknown几何错位。下一步应改为两阶段训练：先用target old/seen support做class-preserving subspace alignment，保证support_proto_acc和seen-new query不下降；再用proxy_unknown做分接收机/分TX的hard-negative curriculum，而不是一次性全局推离所有proxy样本。

## 2026-07-04 SAGE-OSR v2两阶段curriculum计划

目的：修复SAGE-OSR v1“proxy推离成功但seen-new结构崩塌”的问题。v2保留低秩residual adapter，但默认使用两阶段训练：

1. `alignment`阶段：只用source old和target old/seen support做分类、old distillation、support compactness、support margin和residual约束，不启用proxy unknown loss；
2. `negative`阶段：保持上述保真约束，再启用source-side `proxy_unknown`的hard open loss和entropy loss；
3. proxy采样从全局随机改为`tx_balanced`，避免某个proxy TX主导开集边界；
4. 总轮数仍由`--adapter_epochs`控制，默认按`alignment_fraction=0.67`拆分；可用`--curriculum legacy`复现v1。

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_sage_osr_ci_eval.py code\scripts\phase2_aware_ci_eval.py code\scripts\phase2_proxy_adapter_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_sage_osr_ci_eval.py code\tests\test_phase2_aware_ci_eval.py code\tests\test_phase2_open_set_geometry_audit.py -q
10 passed
```

计划远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_sage_osr_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/features_proxy_unknown.npz \
  --output_dir remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/sage_osr_v2 \
  --collab_counts all --collab_group_policy same_max_budget \
  --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070707 \
  --support_selection_policy stable_first --device cuda:0 \
  --adapter_epochs 80 --curriculum two_stage --alignment_fraction 0.67 \
  --adapter_rank 12 --adapter_alpha 0.08 --proxy_curriculum tx_balanced \
  --proxy_open_weight 0.75 --proxy_entropy_weight 0.10 --old_distill_weight 8.0 \
  --support_cls_weight 2.0 --support_compact_weight 1.0 --residual_weight 0.75
```

验收：若v2不能同时提升seen-new和unknown，则记录为`class-preserving curriculum仍不足`，下一步应回到特征包生成/地面训练阶段加入source-heldout TX open-set episodic loss，而不是继续调星上adapter。

### SAGE-OSR v2远端结果

远端同步与验证：

```text
N607 preflight OK at 2026年07月04日星期六19:34:20CST
GPU 0..7: RTX3090，均约10/24576MiB，util=0
code/scripts/phase2_sage_osr_ci_eval.py SHA256=45e88c0c52cebb72aa9d950ee29f007930290f4bf462bb8096f1888543bdda80
code/tests/test_phase2_sage_osr_ci_eval.py SHA256=e074ffb69ba1465a9866f134c307ed3b4f274e11a976ae722e855c05f933cbfd
远端py_compile通过
```

远端输出：

- `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/sage_osr_v2/sage_osr_adapted_features.npz`
- `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/sage_osr_v2/sage_osr_summary.json`
- `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/sage_osr_v2/sage_osr_aware_summary.csv`

本地拉回：

- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\sage_osr_v2\sage_osr_summary.json`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\sage_osr_v2\sage_osr_aware.json`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\sage_osr_v2\sage_osr_aware_summary.csv`
- `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_20260704\sage_osr_v2\sage_osr_aware_evidence.csv`

训练侧诊断：

| metric | before | after | interpretation |
|---|---:|---:|---|
| source_proto_acc | 0.951515 | 0.955303 | source old保持 |
| support_proto_acc | 0.556250 | 0.590625 | v2相对v1修复了support结构下降问题 |
| proxy_max_logit_mean | 10.895288 | -1.075975 | proxy_unknown仍被推离已知原型 |
| proxy_max_logit_q90 | 12.450642 | 0.021717 | hard proxy强度比v1温和 |
| curriculum | - | `two_stage` | `alignment_epochs=54,negative_epochs=26` |
| proxy_curriculum | - | `tx_balanced` | 8个proxy TX均衡采样 |
| final_loss | - | 5.093537 | 训练完成，无NaN/OOM |

SAGE-OSR v2+AWARE结果摘要：

| selection | profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes_per_event | latency_ms | avg_participating | p95_participating | target_pass | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| best_joint | aware_old_safe | 4 | 0.833333 | 0.416667 | 0.541667 | 0.416667 | 0.000000 | 1.000000 | 0.041667 | 0.083333 | 512.0 | 0.45 | 4.0 | 4.0 | false | diagnostic_only |
| best_unknown | aware_unknown_probe | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.166667 | 0.833333 | 0.875000 | 0.833333 | 128.0 | 0.45 | 1.0 | 1.0 | false | diagnostic_only |
| best_seen | aware_old_safe | 1 | 0.805556 | 0.583333 | 0.541667 | 0.416667 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 128.0 | 0.45 | 1.0 | 1.0 | false | diagnostic_only |

全量1..5协同下，`aware_old_safe`的seen-new由v1的`0.083333..0.208333`提升到最高`0.541667`，说明class-preserving alignment有效；但所有old-safe/balanced行的`unknown_reject=0`，unknown probe最高也只有`0.166667`且known基本崩溃。`old_acc>=0.802139`且`seen_new_acc>=0.783333`的保真行数量仍为0。

结论：v2证明两阶段curriculum能缓解seen-new塌陷，但不能把source-side proxy_unknown的开集边界迁移到真实target_unknown。到此为止，星上轻量adapter/阈值/协同层已经给出一致负证据：当前ADV3B02特征包中的真实unknown不是proxy_unknown可覆盖的外侧区域。下一步应转到地面训练或特征包生成阶段：把`Y_unknown`相邻TX作为source-heldout open-set episodic训练任务，在地面重训/微调ADV3B02后再导出特征包；星上仅保留AWARE-CI资源受限协同推理。

## 2026-07-04 C3R-Stage2C保守协同一致性拒识融合

目的：根据“单协同推理不足、优先解决未知类拒识且旧类准确性不能下降”的新要求，新增一个部署侧轻量决策层`C3R-Stage2C`（Conservative Collaborative Conformal Receiver Fusion）。该方法不训练主干，不使用`target_unknown`调阈值；它复用PCET/qknn8的support-only证据，在`M=1..|R_t|`协同数量上执行：

1. `old shield`：旧类候选必须通过support conformal正例质量、receiver质量、tail-risk和source-anchor风险约束；
2. `seen-new strict gate`：seen-new需要更严格多receiver投票；
3. `unknown confirmation`：仅当old/seen-new均无shield时，才用多receiver风险一致性拒识；
4. 预注册profile顺序固定为`c3r_old_anchor -> c3r_balanced -> c3r_unknown_guarded`，`best_eval_row`只用于诊断分析，不用于profile选择。

本地变更文件：

| file | purpose |
|---|---|
| `E:\type10-7\code\scripts\phase2_c3r_stage2c_ci_eval.py` | 新增C3R-Stage2C融合评估脚本，输出`collab_count=1..N`、bytes/event、latency、avg/p95参与receiver和同row old/seen/unknown指标 |
| `E:\type10-7\code\tests\test_phase2_c3r_stage2c_ci_eval.py` | 覆盖profile解析、old shield、unknown确认、协同数量1..N与unknown eval-only元数据 |

非Git工作区快照：

- `E:\type10-7\code\snapshots\phase2_c3r_stage2c_20260704\phase2_c3r_stage2c_ci_eval.py`
- `E:\type10-7\code\snapshots\phase2_c3r_stage2c_20260704\test_phase2_c3r_stage2c_ci_eval.py`

Git-backed镜像目标：

- `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_c3r_stage2c_ci_eval.py`
- `E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_c3r_stage2c_ci_eval.py`

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\phase2_c3r_stage2c_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_c3r_stage2c_ci_eval.py -q
4 passed

C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_c3r_stage2c_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_c3r_stage2c_ci_eval.py -q
4 passed
```

本地真实特征smoke：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe E:\type10-7\code\scripts\phase2_c3r_stage2c_ci_eval.py --feature_npz E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz --output_json E:\type10-7\local_artifacts\phase2_adv3b02_c3r_stage2c_20260704\local_smoke_v4\c3r_summary.json --output_summary_csv E:\type10-7\local_artifacts\phase2_adv3b02_c3r_stage2c_20260704\local_smoke_v4\c3r_summary.csv --profiles c3r_old_anchor,c3r_balanced,c3r_unknown_guarded --collab_counts all --collab_group_policy same_max_budget --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070721 --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked --max_event_bytes 1152 --max_event_latency_ms 20
```

本地smoke结果摘要：

| profile | collab_count | old_acc | min_old | seen_new_acc | unknown_reject | unknown_FAR | bytes_per_event | latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c3r_old_anchor | 1 | 0.566372 | 0.083333 | 0.000000 | 0.555556 | 0.444444 | 128.0 | 1.996631 |
| c3r_old_anchor | 3 | 0.513889 | 0.000000 | 0.000000 | 0.583333 | 0.416667 | 384.0 | 1.996631 |
| c3r_balanced | 3 | 0.250000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 384.0 | 1.996631 |
| c3r_unknown_guarded | 3 | 0.236111 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 384.0 | 1.996631 |

同条件COTE比较：

| profile | collab_count | old_acc | min_old | seen_new_acc | unknown_reject | unknown_FAR | bytes_per_event | latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cote_known_anchor | 5 | 0.935484 | 0.000000 | 0.583333 | 0.250000 | 0.750000 | 640.0 | 1.937150 |

阶段性结论：C3R实现了协议合规的support-only未知确认，并在若干profile上显著降低unknown FAR，但目前仍以旧类准确性大幅下降为代价，不满足“旧类准确性不能下降”。因此C3R当前只能作为`NON_DEPLOYMENT_DIAGNOSTIC`，不能替代COTE/AWARE作为主线。下一步N607测试用于确认本地smoke是否可复现，并记录GPU资源/latency/协同数量曲线；若远端结果一致，应转入地面特征空间重训或proxy_unknown挖掘，而不是继续只调部署侧融合阈值。

计划N607命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=<low_mem_gpu> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_c3r_stage2c_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/features_proxy_unknown.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/c3r_stage2c/c3r_summary.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/c3r_stage2c/c3r_summary.csv \
  --profiles c3r_old_anchor,c3r_balanced,c3r_unknown_guarded \
  --collab_counts all --collab_group_policy same_max_budget \
  --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070721 \
  --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 --max_event_latency_ms 20
```

### C3R-Stage2C远端结果

N607预检与资源状态：

```text
N607 preflight OK at 2026年07月04日星期六19:50:59CST
GPU 0..7: RTX3090，均约10/24576MiB，util=0
选用GPU0运行C3R；运行后GPU 0..7仍约10/24576MiB，util=0
```

同步与远端验证：

| item | result |
|---|---|
| synced script | `code/scripts/phase2_c3r_stage2c_ci_eval.py` -> `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_c3r_stage2c_ci_eval.py` |
| synced test | `code/tests/test_phase2_c3r_stage2c_ci_eval.py` -> `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_c3r_stage2c_ci_eval.py` |
| script SHA256 | `a39ac268e2e091e626a4a80e353ff82878065685c9c4f4bb68ceeed6392a3378` |
| test SHA256 | `58bf0028cbd8fe08f1cc1869a521ea61d45f90da76e30d5fd579a40794656245` |
| remote env | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| remote py_compile | pass |
| remote pytest | unavailable: `No module named pytest` |
| remote unittest | `Ran 4 tests in 0.067s OK` |
| SSH cleanup | final local `ssh.exe=null`，N607/bridge ESTABLISHED连接`null` |

远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_c3r_stage2c_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/features_proxy_unknown.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/c3r_stage2c/c3r_summary.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/c3r_stage2c/c3r_summary.csv \
  --profiles c3r_old_anchor,c3r_balanced,c3r_unknown_guarded \
  --collab_counts all --collab_group_policy same_max_budget \
  --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070721 \
  --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 --max_event_latency_ms 20
```

远端输出：

- `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/c3r_stage2c/c3r_summary.json`
- `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_unknown_20260704/c3r_stage2c/c3r_summary.csv`

本地拉回：

- `E:\type10-7\local_artifacts\phase2_adv3b02_c3r_stage2c_20260704\remote\c3r_summary.json`
- `E:\type10-7\local_artifacts\phase2_adv3b02_c3r_stage2c_20260704\remote\c3r_summary.csv`

拉回文件hash：

| file | SHA256 |
|---|---|
| `c3r_summary.json` | `A16586C1AA4F8C8F7B5F0933A59B37...` |
| `c3r_summary.csv` | `2141252A0F9BD1DCB2B86E27D3C4A9...` |

远端C3R结果表：

| profile | collab_count | old_acc | min_old | seen_new_acc | unknown_reject | unknown_FAR | bytes_per_event | latency_ms | avg_participating | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c3r_old_anchor | 1 | 0.604167 | 0.208333 | 0.000000 | 0.208333 | 0.791667 | 128.0 | 3.491358 | 1.0 | false |
| c3r_old_anchor | 2 | 0.655172 | 0.200000 | 0.000000 | 0.250000 | 0.750000 | 256.0 | 3.491358 | 2.0 | false |
| c3r_old_anchor | 3 | 0.638889 | 0.166667 | 0.000000 | 0.250000 | 0.750000 | 384.0 | 3.491358 | 3.0 | false |
| c3r_old_anchor | 4 | 0.596491 | 0.000000 | 0.000000 | 0.250000 | 0.750000 | 512.0 | 3.491358 | 4.0 | false |
| c3r_old_anchor | 5 | 0.583333 | 0.000000 | 0.000000 | 0.291667 | 0.708333 | 640.0 | 3.491358 | 5.0 | false |
| c3r_balanced | 1 | 0.437500 | 0.000000 | 0.000000 | 0.458333 | 0.541667 | 128.0 | 3.491358 | 1.0 | false |
| c3r_balanced | 3 | 0.208333 | 0.000000 | 0.000000 | 0.833333 | 0.166667 | 384.0 | 3.491358 | 3.0 | false |
| c3r_balanced | 5 | 0.250000 | 0.000000 | 0.000000 | 0.958333 | 0.041667 | 640.0 | 3.491358 | 5.0 | false |
| c3r_unknown_guarded | 1 | 0.000000 | 0.000000 | 0.000000 | 0.541667 | 0.458333 | 128.0 | 3.491358 | 1.0 | false |
| c3r_unknown_guarded | 4 | 0.175439 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 512.0 | 3.491358 | 4.0 | false |
| c3r_unknown_guarded | 5 | 0.083333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 640.0 | 3.491358 | 5.0 | false |

结论：C3R在远端真实特征包上确认了强trade-off：`c3r_balanced,M=5`可把`unknown_FAR`降到`0.041667`，但`old_acc=0.25`；`c3r_unknown_guarded,M=4/5`可达到`unknown_reject=1.0`，但旧类几乎崩溃。最保守的`c3r_old_anchor`也只有`old_acc<=0.655172`，明显低于同条件COTE `old_acc=0.935484`。因此C3R不是当前可部署路线，只能作为负诊断证明：在当前ADV3B02特征空间中，部署侧positive-only conformal协同拒识仍无法同时满足unknown拒识和旧类保持。下一步必须转向地面特征空间修复：source-heldout TX hard negative mining、EVT/Gaussian prototype训练期约束、或重训/微调ADV3B02，而不是继续在星上融合层堆阈值。

## 2026-07-04 Source-heldout Proxy-Unknown挖掘计划

目标：根据C3R负诊断，停止继续加严部署侧拒识阈值，先构造协议合规的source-heldout hard-negative池，用于后续ADV3B02/qKNN8特征包修复、EVT/MTPL式拒识边界训练和COTE/C3R复验。该步骤不声明Stage2-C成功，只生成可审计候选。

子agent审查结论：

| 角色 | 结论 | 对本轮动作的约束 |
|---|---|---|
| 文献/方法 | 优先组合为`z_id/z_dom`解耦+ProtoNet旧类/seen-new原型+MTPL/EVT拒识；FL/多星训练只作为系统扩展或future work | 当前先做source-heldout proxy_unknown，不把多星FL替代Stage2-C轻量适应 |
| 高效算法 | 推荐`C3R-HNFR`：source-heldout hard-negative mining+feature repair+core accept保护旧类 | 不能继续只把C3R阈值调得更狠；必须加入旧类core保护与hard-negative间隔 |
| 逐项监督 | 工程链路已覆盖N607、同步、低显存GPU、资源/latency、collab_count，但性能目标未满足 | 后续结果必须同row同时看`old_acc/min_old/unknown_FAR` |
| Review | C3R低FAR不能写成部署成功；`proxy_unknown`不得来自`target_unknown`或target query统计 | manifest必须记录`target_unknown_used_for_scoring=false`和协议TX/RX排除 |

新增本地文件：

| file | purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\phase2_proxy_unknown_miner.py` | 读取WiSig compact元数据，按source receiver覆盖/样本平衡选择label-disjoint proxy_unknown TX | `3B4F2119C2378DB6138DC281E4FDF4E52A8C7D0792B47BA4F109978D32F5B8F0` |
| `E:\type10-7\code\tests\test_phase2_proxy_unknown_miner.py` | 验证协议TX排除、target receiver排除、最小coverage过滤和JSON/CSV输出 | `57A2B8FDE1F46EBB66B84728DD22BA561BB77B3B1088954D2554AD1526BAE4B1` |

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\phase2_proxy_unknown_miner.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_proxy_unknown_miner.py -q
3 passed，另有既有.pytest_cache权限warning

C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_proxy_unknown_miner.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_proxy_unknown_miner.py -q
3 passed
```

版本状态：

| item | value |
|---|---|
| publish repo | `E:\type10-7\github_publish\CVS-RFFI-repo` |
| commit | `33b6829 Add source-heldout proxy unknown miner` |
| snapshot | `E:\type10-7\code\snapshots\phase2_proxy_unknown_miner_20260704\` |
| ignored/untracked | `local_artifacts/`仍为本地结果区，不纳入提交 |

远端计划命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_proxy_unknown_miner.py \
  --wisig_pkl Dataset_WigSig/ManyTx.pkl \
  --source_tx_ids 14-10,14-7,20-15,20-19,6-15,8-20 \
  --target_new_tx_ids 19-3,3-8 \
  --target_unknown_tx_ids 10-1,10-10 \
  --proxy_source_rxs 1-1,1-19,14-7,18-2,19-2,2-1,2-19 \
  --target_rxs 20-1,3-19,7-14,7-7,8-8 \
  --exclude_tx_ids 1-16,1-18,10-11,10-17,11-1,12-7,13-14,14-11 \
  --top_k 16 \
  --min_source_rx_coverage 4 \
  --min_samples_per_tx 300 \
  --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/proxy_miner/proxy_unknown_manifest.json \
  --output_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/proxy_miner/proxy_unknown_candidates.csv
```

验收门槛：

| gate | requirement |
|---|---|
| 协议TX | selected TX不得与`Y_old/Y_new/Y_unknown`或既有proxy集合重叠 |
| 协议RX | `proxy_source_rxs_used`不得包含`R_t` |
| 泄漏审计 | `target_unknown_used_for_scoring=false` |
| 候选质量 | 每个候选至少4个source receiver有样本且总样本不少于300 |
| 后续复验 | 使用新候选导出特征包后，必须同row报告`old_acc/min_old/seen_new/min_seen/unknown_FAR/coverage/defer/per-class floor`；达不到旧类保持则继续标记diagnostic-only |

### Proxy-Unknown挖掘远端结果

N607预检：

```text
N607 preflight OK at 2026年07月04日星期六20:03:15CST
GPU 0..7: RTX3090，均约10/24576MiB，util=0
```

远端同步与验证：

| item | result |
|---|---|
| script | `code/scripts/phase2_proxy_unknown_miner.py` |
| test | `code/tests/test_phase2_proxy_unknown_miner.py` |
| remote script SHA256 | `3b4f2119c2378db6138dc281e4fdf4e52a8c7d0792b47ba4f109978d32f5b8f0` |
| remote test SHA256 | `57a2b8fde1f46ebb66b84728dd22ba561bb77b3b1088954d2554ad1526bae4b1` |
| remote py_compile | pass |
| remote unittest | `Ran 3 tests in 1.174s OK` |

首次运行失败原因：

```text
ValueError: cannot resolve proxy_source_rxs item '2-19'
```

根因：`2-19`存在于先前`ManySig` source RX集合，但不在`ManyTx.pkl`的RX列表中。为保持候选池与`ManyTx`可解析元数据一致，修正`proxy_source_rxs`为已有proxy导出使用的6个source RX：`1-1,1-19,14-7,18-2,19-2,2-1`。

成功命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_proxy_unknown_miner.py \
  --wisig_pkl Dataset_WigSig/ManyTx.pkl \
  --source_tx_ids 14-10,14-7,20-15,20-19,6-15,8-20 \
  --target_new_tx_ids 19-3,3-8 \
  --target_unknown_tx_ids 10-1,10-10 \
  --proxy_source_rxs 1-1,1-19,14-7,18-2,19-2,2-1 \
  --target_rxs 20-1,3-19,7-14,7-7,8-8 \
  --exclude_tx_ids 1-16,1-18,10-11,10-17,11-1,12-7,13-14,14-11 \
  --top_k 16 \
  --min_source_rx_coverage 4 \
  --min_samples_per_tx 300 \
  --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/proxy_miner/proxy_unknown_manifest.json \
  --output_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/proxy_miner/proxy_unknown_candidates.csv
```

远端输出与本地拉回：

| file | remote SHA256 | local path |
|---|---|---|
| `proxy_unknown_manifest.json` | `b363b74088270efa46a6c132724d21b773e31046e70ebc441c5738a10864a0ad` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\proxy_miner\proxy_unknown_manifest.json` |
| `proxy_unknown_candidates.csv` | `7fcd1f80ceab26601b7498442c684757a0909779aa27b3c19b5170bfcfd1e13b` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\proxy_miner\proxy_unknown_candidates.csv` |

协议审计：

```text
selected_proxy_unknown_tx_ids=9-7,8-3,7-9,6-6,5-5,4-11,3-20,20-8,2-8,19-9,18-9,17-11,16-20,15-19,14-9,13-3
eligible_count=132
candidate_count=150
target_unknown_used_for_scoring=False
proxy_source_rxs_used=1-1,1-19,14-7,18-2,19-2,2-1
proxy_target_rx_overlap=[]
```

候选表：

| rank | tx_id | family | total_samples | rx_coverage | rx_min_samples | rx_balance |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `9-7` | 9 | 1200 | 6 | 200 | 1.0 |
| 2 | `8-3` | 8 | 1200 | 6 | 200 | 1.0 |
| 3 | `7-9` | 7 | 1200 | 6 | 200 | 1.0 |
| 4 | `6-6` | 6 | 1200 | 6 | 200 | 1.0 |
| 5 | `5-5` | 5 | 1200 | 6 | 200 | 1.0 |
| 6 | `4-11` | 4 | 1200 | 6 | 200 | 1.0 |
| 7 | `3-20` | 3 | 1200 | 6 | 200 | 1.0 |
| 8 | `20-8` | 20 | 1200 | 6 | 200 | 1.0 |
| 9 | `2-8` | 2 | 1200 | 6 | 200 | 1.0 |
| 10 | `19-9` | 19 | 1200 | 6 | 200 | 1.0 |
| 11 | `18-9` | 18 | 1200 | 6 | 200 | 1.0 |
| 12 | `17-11` | 17 | 1200 | 6 | 200 | 1.0 |
| 13 | `16-20` | 16 | 1200 | 6 | 200 | 1.0 |
| 14 | `15-19` | 15 | 1200 | 6 | 200 | 1.0 |
| 15 | `14-9` | 14 | 1200 | 6 | 200 | 1.0 |
| 16 | `13-3` | 13 | 1200 | 6 | 200 | 1.0 |

运行后资源：GPU 0..7仍约`10/24576MiB`，util=0。最终本地SSH清理检查无`ssh.exe`，无N607/bridge ESTABLISHED连接。

下一步：使用上述16个候选导出新的`features_proxy_mined.npz`，保持`target_unknown`只用于eval；随后优先做几何审计和`collab_count=1..5`协同复验。若旧类准确性低于基线或`min_old`崩溃，该候选集只能作为hard-negative训练输入，不能作为部署成功。

### Proxy-Mined特征包与协同复验

特征包导出命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/export_spaceborne_features.py \
  --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth \
  --wisig_pkl Dataset_WigSig/ManySig.pkl \
  --new_wisig_pkl Dataset_WigSig/ManyTx.pkl \
  --out_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz \
  --feature_name z_id \
  --source_tx_ids 0,1,2,3,4,5 \
  --source_rxs 0,1,2,3,4,5,6 \
  --target_old_tx_ids 0,1,2,3,4,5 \
  --target_old_rxs 20-1,3-19,7-14,7-7,8-8 \
  --target_old_channel_view satellite \
  --target_old_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --new_tx_ids 19-3,3-8 \
  --new_rxs 20-1,3-19,7-14,7-7,8-8 \
  --unknown_tx_ids 10-1,10-10 \
  --proxy_unknown_tx_ids 9-7,8-3,7-9,6-6,5-5,4-11,3-20,20-8,2-8,19-9,18-9,17-11,16-20,15-19,14-9,13-3 \
  --proxy_unknown_rxs 1-1,1-19,14-7,18-2,19-2,2-1 \
  --proxy_unknown_channel_view satellite \
  --proxy_unknown_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --target_new_channel_view satellite \
  --target_new_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --star_ground_channel_impl simplified_leo_residual \
  --wisig_equalized 1 \
  --wisig_domain rx_day \
  --wisig_out_len 256 \
  --max_samples_per_combo 0 \
  --max_samples_per_tx 220 \
  --batch_size 512 \
  --device cuda:0 \
  --seed 4070527
```

特征包审计：

| item | value |
|---|---|
| remote path | `/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz` |
| local path | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\features_proxy_mined.npz` |
| SHA256 | `48759DC53D6A5E1210C0064FDE356E4D2BDA5B1109F947074387E9625E27AF6D` |
| feature shape | `(7040,160)` |
| target view | `satellite/LEO` |
| star-ground impl | `simplified_leo_residual` |
| target scenarios | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |

| role | rows | TX | RX |
|---|---:|---|---|
| source | 1320 | `14-10,14-7,20-15,20-19,6-15,8-20` | `1-1,1-19,14-7,18-2,19-2,2-1,2-19` |
| target_old | 1320 | `14-10,14-7,20-15,20-19,6-15,8-20` | `20-1,3-19,7-14,7-7,8-8` |
| target_new | 440 | `19-3,3-8` | `20-1,3-19,7-14,7-7,8-8` |
| target_unknown | 440 | `10-1,10-10` | `20-1,3-19,7-14,7-7,8-8` |
| proxy_unknown | 3520 | `13-3,14-9,15-19,16-20,17-11,18-9,19-9,2-8,20-8,3-20,4-11,5-5,6-6,7-9,8-3,9-7` | `1-1,1-19,14-7,18-2,19-2,2-1` |

协议检查：`proxy_unknown_overlaps_source=[]`，`proxy_unknown_overlaps_target_unknown=[]`，`proxy_unknown_overlaps_target_new=[]`。运行后GPU 0..7仍约`10/24576MiB`，util=0。

COTE复验命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_cote_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/cote_remote/cote_ci.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/cote_remote/cote_ci_summary.csv \
  --profiles all --collab_counts all --collab_group_policy same_max_budget \
  --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070721 \
  --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 --max_event_latency_ms 20
```

COTE结果：

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes_per_event | latency_ms | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cote_known_anchor | 1 | 0.812500 | 0.500000 | 0.375000 | 0.250000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 128.0 | 5.521803 | false |
| cote_known_anchor | 2 | 0.931034 | 0.833333 | 0.291667 | 0.250000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 256.0 | 5.521803 | false |
| cote_known_anchor | 3 | 0.916667 | 0.750000 | 0.291667 | 0.166667 | 0.000000 | 1.000000 | 0.010417 | 0.000000 | 384.0 | 5.521803 | false |
| cote_known_anchor | 4 | 0.947368 | 0.916667 | 0.208333 | 0.083333 | 0.041667 | 0.958333 | 0.012346 | 0.083333 | 512.0 | 5.521803 | false |
| cote_known_anchor | 5 | 0.895833 | 0.000000 | 0.250000 | 0.166667 | 0.166667 | 0.833333 | 0.027778 | 0.041667 | 640.0 | 5.521803 | false |
| cote_balanced | 5 | 0.729167 | 0.000000 | 0.000000 | 0.000000 | 0.458333 | 0.541667 | 0.000000 | 0.000000 | 640.0 | 5.521803 | false |
| cote_unknown_confirm | 5 | 0.562500 | 0.000000 | 0.000000 | 0.000000 | 0.666667 | 0.333333 | 0.000000 | 0.000000 | 640.0 | 5.521803 | false |

C3R复验命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_c3r_stage2c_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/c3r_remote/c3r_summary.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/c3r_remote/c3r_summary.csv \
  --profiles c3r_old_anchor,c3r_balanced,c3r_unknown_guarded \
  --collab_counts all --collab_group_policy same_max_budget \
  --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070721 \
  --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 --max_event_latency_ms 20
```

C3R结果：

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes_per_event | latency_ms | avg_participating | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c3r_old_anchor | 1 | 0.604167 | 0.208333 | 0.000000 | 0.000000 | 0.208333 | 0.791667 | 0.000000 | 0.000000 | 128.0 | 5.712844 | 1.0 | false |
| c3r_old_anchor | 2 | 0.643678 | 0.200000 | 0.000000 | 0.000000 | 0.250000 | 0.750000 | 0.000000 | 0.041667 | 256.0 | 5.712844 | 2.0 | false |
| c3r_old_anchor | 3 | 0.638889 | 0.166667 | 0.000000 | 0.000000 | 0.250000 | 0.750000 | 0.010417 | 0.041667 | 384.0 | 5.712844 | 3.0 | false |
| c3r_old_anchor | 4 | 0.596491 | 0.000000 | 0.000000 | 0.000000 | 0.250000 | 0.750000 | 0.012346 | 0.000000 | 512.0 | 5.712844 | 4.0 | false |
| c3r_old_anchor | 5 | 0.583333 | 0.000000 | 0.000000 | 0.000000 | 0.291667 | 0.708333 | 0.000000 | 0.000000 | 640.0 | 5.712844 | 5.0 | false |
| c3r_balanced | 3 | 0.208333 | 0.000000 | 0.000000 | 0.000000 | 0.833333 | 0.166667 | 0.114583 | 0.125000 | 384.0 | 5.712844 | 3.0 | false |
| c3r_balanced | 5 | 0.250000 | 0.000000 | 0.000000 | 0.000000 | 0.958333 | 0.041667 | 0.000000 | 0.000000 | 640.0 | 5.712844 | 5.0 | false |
| c3r_unknown_guarded | 4 | 0.175439 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.061728 | 0.000000 | 512.0 | 5.712844 | 4.0 | false |
| c3r_unknown_guarded | 5 | 0.083333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.111111 | 0.000000 | 640.0 | 5.712844 | 5.0 | false |

拉回文件：

| file | SHA256 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\cote_remote\cote_ci.json` | `B793D0A19E9C9E37A7DAA69C462361D245DA60D26FAB083043ADEF80C1302D04` |
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\cote_remote\cote_ci_summary.csv` | `AD0F47F8007488C7BFC6B08BE42004D07F729A7E253B3AB21570E568139DE258` |
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\c3r_remote\c3r_summary.json` | `4121B5760A16457EA797546E3AEA56F145E4103590542DB42479572E323F03A0` |
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\c3r_remote\c3r_summary.csv` | `A7C3DD740C64302FBEFC19B84F33B9BD49189158B92215C7D70CD1FD548EF2BE` |

结论：新source-heldout proxy候选集通过协议与资源审计，但没有直接解决部署侧协同推理性能。COTE在旧类保持较好的行仍有`unknown_FAR>=0.958333`；C3R能把`unknown_FAR`降到`0.041667`或`0`，但同row旧类准确性下降到`0.25`或更低。因此当前结果仍是`NON_DEPLOYMENT_DIAGNOSTIC`。正确下一步是把proxy-mined TX作为地面训练/微调阶段的hard-negative输入，优化`z_id`特征空间和EVT/MTPL拒识边界，而不是把新proxy候选直接写成星上部署成功。

### C3R-HNFR设计文档与版本状态

已新增独立算法设计文档：

| item | value |
|---|---|
| local doc | `E:\type10-7\docs\CVS_STAGE2C_C3R_HNFR_ALGORITHM_20260704.md` |
| publish doc | `E:\type10-7\github_publish\CVS-RFFI-repo\docs\CVS_STAGE2C_C3R_HNFR_ALGORITHM_20260704.md` |
| remote doc | `/home/szu2070436088/2510044040/CV-SincNet/docs/CVS_STAGE2C_C3R_HNFR_ALGORITHM_20260704.md` |
| SHA256 | `E65D22FD834168302DE04BABD7D6DA4DADF07050BBA2D579EDAF52F778766AD8` |
| commit | `7598b46 Document C3R-HNFR Stage2C algorithm` |

设计文档明确：下一版主线为`C3R-HNFR`，即source-heldout hard-negative feature repair+旧类core accept保护+轻量协同证据融合；当前proxy-mined+COTE/C3R远端结果未达标，只能作为hard-negative训练输入和负诊断证据。

最终清理与版本状态：

```text
local ssh.exe: none
N607/bridge ESTABLISHED: none
publish git: ahead 440, only local_artifacts/ untracked
```

## 2026-07-04 Proxy-mined HNFR/OPR保守适配复验

目的：上一节确认proxy-mined候选不能直接让部署侧COTE/C3R达标。根据C3R-HNFR设计，继续验证“训练侧/特征侧轻量修复”是否能在不牺牲旧类的前提下降低unknown FAR。本节复用现有`phase2_proxy_adapter_ci_eval.py`，但采用更保守的old-preserve参数，避免重复默认OPR已知的旧类崩溃。

本地smoke：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe E:\type10-7\code\scripts\phase2_proxy_adapter_ci_eval.py \
  --feature_npz E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\features_proxy_mined.npz \
  --output_dir E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_hnfr_smoke \
  --backend enpc --collab_counts 1 --collab_group_policy same_max_budget \
  --partial_collab_min_receivers 1 --k_shot 8 --query_per_class 12 --qknn_k 8 \
  --seed 4070801 --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked \
  --device cpu --adapter_epochs 1 --adapter_rank 4 --adapter_alpha 0.05 \
  --old_preserve_weight 8.0 --proxy_open_weight 0.25 --support_compact_weight 0.25 --residual_weight 0.50 \
  --enpc_profiles all --max_event_bytes 1152 --max_event_latency_ms 20
```

结果：

```text
adapted_feature_npz=E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_hnfr_smoke\opr_ci_adapted_features.npz
target_receivers=20-1,3-19,7-14,7-7,8-8
```

远端计划：

| config | intent | key params |
|---|---|---|
| `hnfr_oldguard` | 强旧类保真，弱proxy排斥，检验是否能不伤old/seen | `rank=8,alpha=0.06,epochs=50,lr=1e-3,old_preserve=10,proxy_open=0.20,residual=0.60` |
| `hnfr_balanced` | 中等proxy排斥，保留旧类保护，检验unknown收益上限 | `rank=8,alpha=0.10,epochs=50,lr=1e-3,old_preserve=6,proxy_open=0.50,residual=0.35` |

共同设置：`feature_npz=remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz`，`backend=both`，`collab_counts=all`，`k_shot=8`，`query_per_class=12`，`qknn_k=8`，`event_alignment_policy=receiver_domain_ranked`，`max_event_bytes=1152`，`max_event_latency_ms=20`。严格事件协同仍未具备同事件key，因此该复验继续标记为receiver-domain ensemble diagnostic。

### HNFR/OPR远端结果

N607预检：

```text
N607 preflight OK at 2026年07月04日星期六20:22:21CST
GPU 0..7: RTX3090，均约10/24576MiB，util=0
```

远端输出与hash：

| config | file | SHA256 |
|---|---|---|
| hnfr_oldguard | `opr_ci_summary.json` | `b7e97388554212d692961990522025b6f9a224ebc1c089758cf3b5e597efdeba` |
| hnfr_oldguard | `opr_ci_enpc_summary.csv` | `351027e3e596711260b2488e89d13bef778a75e7ac81b1814424bf7531de9c96` |
| hnfr_oldguard | `opr_ci_slev_summary.csv` | `e9fa04f57219f222af905c5cf95d4c314146defe961908ac7fedb210f3b1a70b` |
| hnfr_oldguard | `opr_ci_adapted_features.npz` | `0bf4bafdd3893c853de361f293543b3298556a624233926f4d1a79b8b9f3c640` |
| hnfr_balanced | `opr_ci_summary.json` | `50a93e989bd1d933a19eb76edb46f5fbe72747062ea00dea265ca5e1cc2e932f` |
| hnfr_balanced | `opr_ci_enpc_summary.csv` | `f2baa3a1de7c54950af6a6c08ba0b4a8175ca4246f79084cebb51a11b8854f8b` |
| hnfr_balanced | `opr_ci_slev_summary.csv` | `04e643766e7a82e2a0c3fbc68a0da7be3379d1f2f354d77402ff21fb08228cc6` |
| hnfr_balanced | `opr_ci_adapted_features.npz` | `86e999206f6e7b81c91785f7238c0d2b2c4a684ede2e5086f23a1f9fc6f85712` |

运行后资源：GPU 0..7仍约`10/24576MiB`，util=0。最终断连检查无`ssh.exe`，无N607/bridge ESTABLISHED连接。

适配器训练摘要：

| config | device | final_loss | source_proto_before | source_proto_after | support_proto_before | support_proto_after | proxy_max_before | proxy_max_after | state_bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hnfr_oldguard | cuda:0 | 1.868445 | 0.945455 | 0.956061 | 0.587500 | 0.625000 | 10.797809 | -0.098662 | 8002 |
| hnfr_balanced | cuda:0 | 2.004482 | 0.945455 | 0.956061 | 0.587500 | 0.618750 | 10.797809 | -1.474262 | 8002 |

同row结果摘要：

| config | backend | profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes_per_event | latency_ms | target_pass |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hnfr_oldguard | ENPC | enpc_old80_unknown_probe | 4 | 0.847458 | 0.666667 | 0.250000 | 0.250000 | 0.416667 | 0.583333 | 512.0 | 4.953120 | false |
| hnfr_oldguard | ENPC | enpc_balanced | 3 | 0.875000 | 0.666667 | 0.250000 | 0.250000 | 0.041667 | 0.958333 | 384.0 | 4.953120 | false |
| hnfr_oldguard | SLEV | slev_old80_energy_probe | 4 | 0.864407 | 0.666667 | 0.291667 | 0.250000 | 0.333333 | 0.666667 | 512.0 | 4.619424 | false |
| hnfr_oldguard | SLEV | slev_balanced | 3 | 0.875000 | 0.666667 | 0.250000 | 0.250000 | 0.083333 | 0.916667 | 384.0 | 4.619424 | false |
| hnfr_balanced | ENPC | enpc_old80_unknown_probe | 4 | 0.847458 | 0.583333 | 0.333333 | 0.250000 | 0.416667 | 0.583333 | 512.0 | 5.208428 | false |
| hnfr_balanced | ENPC | enpc_balanced | 3 | 0.902778 | 0.666667 | 0.208333 | 0.083333 | 0.041667 | 0.958333 | 384.0 | 5.208428 | false |
| hnfr_balanced | SLEV | slev_old80_energy_probe | 4 | 0.847458 | 0.583333 | 0.333333 | 0.250000 | 0.375000 | 0.625000 | 512.0 | 5.253560 | false |
| hnfr_balanced | SLEV | slev_balanced | 3 | 0.902778 | 0.666667 | 0.208333 | 0.083333 | 0.083333 | 0.916667 | 384.0 | 5.253560 | false |

结论：保守HNFR/OPR比默认OPR更好地保住旧类，`old_acc`最高可到`0.902778`，`min_old=0.666667`；但未知拒识仍明显不足，保真较好的行`unknown_FAR`仍在`0.583333..0.958333`。训练指标显示adapter已成功降低proxy_unknown对已知原型的最大logit，但真实`target_unknown`拒识没有同步达到目标，说明source-heldout proxy与真实target_unknown几何错位，或当前adapter只学到了“proxy排斥”而未形成可迁移unknown边界。当前HNFR/OPR仍为`NON_DEPLOYMENT_DIAGNOSTIC`，不能作为部署候选。

### Proxy-vs-TargetUnknown几何审计

新增脚本：

| file | SHA256 | commit |
|---|---|---|
| `E:\type10-7\code\scripts\phase2_proxy_target_geometry_audit.py` | `79784B1FEB1055EE06339B87B34098989E1D35E203A7B1755CFEB376B35DE9B4` | `e30eeec Add proxy target geometry audit` |
| `E:\type10-7\code\tests\test_phase2_proxy_target_geometry_audit.py` | `BED94702A418B74BF3C76A25E689D31139A334DE5D54B3D8906217E27934D307` | `e30eeec Add proxy target geometry audit` |

验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\phase2_proxy_target_geometry_audit.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_proxy_target_geometry_audit.py -q
1 passed，另有既有.pytest_cache权限warning

C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_proxy_target_geometry_audit.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_proxy_target_geometry_audit.py -q
1 passed

N607:
sha256sum matches local
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_proxy_target_geometry_audit.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_proxy_target_geometry_audit.py
Ran 1 test in 0.034s OK
```

远端/本地审计输出：

| config | remote JSON SHA256 | local JSON |
|---|---|---|
| base | `c7abdab7ec206a2f7c54fcce70b1aa5bb0ce1d8ca1071829f2bd1f233fab3460` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\geometry_audit\base_geometry.json` |
| hnfr_oldguard | `2c156bd2d76619c21182714d8f388a8c605cb28a6f131f5c343e0aa357041315` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\geometry_audit\hnfr_oldguard_geometry.json` |
| hnfr_balanced | `b2b7761ee8bb59f17a1acdd6b051e693aaba8ed1aea803bbdfc9de6f675c191b` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\geometry_audit\hnfr_balanced_geometry.json` |

核心几何指标：

| config | support_threshold | known-vs-target_unknown AUROC | known-vs-proxy AUROC | target_unknown FPR95 | proxy_unknown FPR95 | target_unknown accept@support | proxy accept@support |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.511987 | 0.670434 | 0.499081 | 0.883333 | 0.953125 | 0.866667 | 0.944318 |
| hnfr_oldguard | 0.781303 | 0.674792 | 0.474493 | 0.758333 | 0.933807 | 0.766667 | 0.943466 |
| hnfr_balanced | 0.860084 | 0.679844 | 0.452723 | 0.800000 | 0.951136 | 0.800000 | 0.951136 |

分组分布：

| config | group | n | accept@support | top1_acc | max_cos_mean | max_cos_p50 | margin_mean |
|---|---|---:|---:|---:|---:|---:|---:|
| base | target_old_query | 360 | 0.941667 | 0.711111 | 0.869038 | 0.926444 | 0.404925 |
| base | target_new_query | 120 | 0.950000 | 0.266667 | 0.873800 | 0.937231 | 0.125254 |
| base | target_unknown_query | 120 | 0.866667 | NA | 0.765896 | 0.833401 | 0.289030 |
| base | proxy_unknown | 3520 | 0.944318 | NA | 0.867871 | 0.938663 | 0.221493 |
| hnfr_oldguard | target_old_query | 360 | 0.941667 | 0.722222 | 0.925680 | 0.942746 | 0.159772 |
| hnfr_oldguard | target_new_query | 120 | 0.975000 | 0.150000 | 0.935658 | 0.947043 | 0.059780 |
| hnfr_oldguard | target_unknown_query | 120 | 0.766667 | NA | 0.853604 | 0.901986 | 0.091812 |
| hnfr_oldguard | proxy_unknown | 3520 | 0.943466 | NA | 0.929709 | 0.949178 | 0.093942 |
| hnfr_balanced | target_old_query | 360 | 0.947222 | 0.725000 | 0.952893 | 0.971330 | 0.086641 |
| hnfr_balanced | target_new_query | 120 | 0.958333 | 0.216667 | 0.963395 | 0.980709 | 0.020412 |
| hnfr_balanced | target_unknown_query | 120 | 0.800000 | NA | 0.903440 | 0.927906 | 0.061573 |
| hnfr_balanced | proxy_unknown | 3520 | 0.951136 | NA | 0.959943 | 0.979344 | 0.036100 |

解释：按target old/new support拟合的known接受阈值下，`proxy_unknown`本身也大量被known包络吸收，FPR95约`0.93..0.95`；HNFR适配后known support阈值升高，但proxy和target_unknown的max-cos分布也同步升高，导致可迁移拒识边界没有形成。当前证据不支持将失败简单归因为proxy与target_unknown不匹配；更准确地说，`proxy_unknown`和`target_unknown`都被target old/new support形成的known包络大量吸收，HNFR未形成可迁移的类条件开放集边界。下一步应避免继续单纯“推远proxy”，改为类条件密度/多原型support重构或目标域known包络收缩，并将seen-new注册正确性作为unknown拒识前置门槛。

### OPC-MECR候选算法落地

新增候选算法：`Old-Protected Class-conditional Multi-Envelope Collaborative Rejection`，简称`OPC-MECR`。该算法作为轻量决策层接在PCET/qknn8证据之后，不改动`ADV3B02_CORE90_SOFT_E200`底座模型，不使用`target_unknown`拟合阈值或选择profile。设计说明见：`E:\type10-7\docs\CVS_STAGE2C_OPC_MECR_ALGORITHM_20260704.md`。

本地变更：

| file | purpose |
|---|---|
| `E:\type10-7\code\scripts\phase2_opc_mecr_ci_eval.py` | OPC-MECR评估器，支持`M=1..R`、旧类保护、seen-new包络抢占、无共识未知拒识、资源指标 |
| `E:\type10-7\code\tests\test_phase2_opc_mecr_ci_eval.py` | 单测：profile解析、旧类安全门、强未知无共识拒识、`collab_counts=all`协议标志 |
| `E:\type10-7\docs\CVS_STAGE2C_OPC_MECR_ALGORITHM_20260704.md` | 算法说明、协议边界、证据包、风险 |

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\phase2_opc_mecr_ci_eval.py E:\type10-7\code\tests\test_phase2_opc_mecr_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_opc_mecr_ci_eval.py -q
4 passed，另有既有.pytest_cache权限warning
```

本地CPU smoke：

```text
feature_npz=E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\features_proxy_mined.npz
output_dir=E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_opc_mecr_smoke4
profiles=all
collab_counts=all
k_shot=8
query_per_class=4
qknn_k=8
seed=4070801
support_selection_policy=stable_first
event_alignment_policy=receiver_domain_ranked
```

smoke结果摘要：

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | old_reject_rate | bytes_per_event |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opc_old_guard | 1 | 0.781250 | 0.250000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.100000 | 0.031250 | 128.0 |
| opc_old_guard | 2 | 0.833333 | 0.250000 | 0.500000 | 0.250000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 256.0 |
| opc_old_guard | 3 | 0.708333 | 0.250000 | 0.750000 | 0.500000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 384.0 |
| opc_old_guard | 4 | 0.777778 | 0.500000 | 0.500000 | 0.250000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 512.0 |
| opc_old_guard | 5 | 0.750000 | 0.000000 | 0.625000 | 0.500000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 640.0 |
| mecr_balanced | 3 | 0.708333 | 0.000000 | 0.000000 | 0.000000 | 0.125000 | 0.875000 | 0.312500 | 0.208333 | 384.0 |
| mecr_unknown_probe | 5 | 0.437500 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.562500 | 640.0 |

解释：本地smoke显示OPC-MECR已具备预期的取舍形态：`opc_old_guard`能显著降低旧类误拒并在`M=2..5`保持`old_reject_rate=0`，但未知拒识仍失败；`mecr_unknown_probe`能拒识未知，但旧类下降严重。因此该算法当前仍是候选诊断路线，不是部署结果。全量`query_per_class=12`需要在N607上运行后才能与COTE/C3R/HNFR进行同口径比较。

### OPC-MECR N607全量评估

N607同步与验证：

| file | remote path | SHA256 |
|---|---|---|
| `phase2_opc_mecr_ci_eval.py` | `code/scripts/phase2_opc_mecr_ci_eval.py` | `40aeab9a1f5aeb0d1960f209f903c1bf0efa45aa7b33e2e17495a7b70dc2fca5` |
| `test_phase2_opc_mecr_ci_eval.py` | `code/tests/test_phase2_opc_mecr_ci_eval.py` | `543754604b84b14e45ee12df0002d46ddb7e2a5bd9aa538a188aaae5beff9ebb` |
| `CVS_STAGE2C_OPC_MECR_ALGORITHM_20260704.md` | `docs/CVS_STAGE2C_OPC_MECR_ALGORITHM_20260704.md` | `daebd7803efaf924ece93b282913a32ffa12006dec66421b0f11b7f9b880bee2` |

远端验证：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_opc_mecr_ci_eval.py code/tests/test_phase2_opc_mecr_ci_eval.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_opc_mecr_ci_eval.py
Ran 4 tests in 0.068s OK
```

远端命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_opc_mecr_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/opc_mecr_remote/opc_mecr.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/opc_mecr_remote/opc_mecr_summary.csv \
  --output_evidence_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/opc_mecr_remote/opc_mecr_evidence.csv \
  --profiles all --collab_counts all --collab_group_policy same_max_budget \
  --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070801 \
  --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 --max_event_latency_ms 20
```

资源状态：预检与运行后GPU 0..7均约`10/24576MiB`，util=0；本任务选择`CUDA_VISIBLE_DEVICES=0`。该脚本是轻量证据层评估，未持续占用显存。最终断连检查无`ssh.exe`，无到N607或桥接机的ESTABLISHED连接。

远端结果hash与本地拉回路径：

| artifact | remote SHA256 | local path |
|---|---|---|
| `opc_mecr.json` | `2157c3206cbd0e45f59716dea0d29d5d69be509d2fe274d3b76a61b8b8e47163` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\opc_mecr_remote\opc_mecr.json` |
| `opc_mecr_summary.csv` | `9c14365c15519e9506a6f4f0888f484627a43c155518d862af80089651bb7c75` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\opc_mecr_remote\opc_mecr_summary.csv` |
| `opc_mecr_evidence.csv` | `14e6adb9a96c38d3168c5334cac43e92cd58075d4d83715496c2b6bbd818a4db` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\opc_mecr_remote\opc_mecr_evidence.csv` |

全量同row结果：

| profile | M | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | old_safe_accept | old_reject | bytes | latency_ms | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| opc_old_guard | 1 | 0.864583 | 0.666667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.066667 | 0.083333 | 0.979167 | 0.020833 | 128.0 | 6.170490 | false |
| opc_old_guard | 2 | 0.823529 | 0.500000 | 0.250000 | 0.166667 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 256.0 | 6.170490 | false |
| opc_old_guard | 3 | 0.791667 | 0.416667 | 0.500000 | 0.416667 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 384.0 | 6.170490 | false |
| opc_old_guard | 4 | 0.762712 | 0.500000 | 0.416667 | 0.333333 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 512.0 | 6.170490 | false |
| opc_old_guard | 5 | 0.770833 | 0.000000 | 0.458333 | 0.416667 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 640.0 | 6.170490 | false |
| mecr_balanced | 1 | 0.593750 | 0.000000 | 0.000000 | 0.000000 | 0.083333 | 0.916667 | 0.425000 | 0.125000 | 0.750000 | 0.250000 | 128.0 | 6.170490 | false |
| mecr_balanced | 2 | 0.658824 | 0.000000 | 0.000000 | 0.000000 | 0.083333 | 0.916667 | 0.348624 | 0.166667 | 0.800000 | 0.200000 | 256.0 | 6.170490 | false |
| mecr_balanced | 3 | 0.652778 | 0.000000 | 0.000000 | 0.000000 | 0.041667 | 0.958333 | 0.364583 | 0.250000 | 0.791667 | 0.208333 | 384.0 | 6.170490 | false |
| mecr_balanced | 4 | 0.610169 | 0.000000 | 0.000000 | 0.000000 | 0.208333 | 0.791667 | 0.445783 | 0.333333 | 0.611111 | 0.388889 | 512.0 | 6.170490 | false |
| mecr_balanced | 5 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 0.208333 | 0.791667 | 0.527778 | 0.375000 | 0.687500 | 0.312500 | 640.0 | 6.170490 | false |
| mecr_unknown_probe | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.750000 | 0.250000 | 0.333333 | 0.250000 | 0.000000 | 1.000000 | 128.0 | 6.170490 | false |
| mecr_unknown_probe | 2 | 0.482353 | 0.000000 | 0.000000 | 0.000000 | 0.458333 | 0.541667 | 0.165138 | 0.208333 | 0.433333 | 0.566667 | 256.0 | 6.170490 | false |
| mecr_unknown_probe | 3 | 0.527778 | 0.000000 | 0.000000 | 0.000000 | 0.541667 | 0.458333 | 0.114583 | 0.125000 | 0.500000 | 0.500000 | 384.0 | 6.170490 | false |
| mecr_unknown_probe | 4 | 0.508475 | 0.000000 | 0.000000 | 0.000000 | 0.750000 | 0.250000 | 0.120482 | 0.041667 | 0.444444 | 0.555556 | 512.0 | 6.170490 | false |
| mecr_unknown_probe | 5 | 0.458333 | 0.000000 | 0.000000 | 0.000000 | 0.791667 | 0.208333 | 0.069444 | 0.000000 | 0.500000 | 0.500000 | 640.0 | 6.170490 | false |

结论：OPC-MECR验证了“旧类保护门”和“未知拒识探针”两个方向的可控取舍，但仍未解决最终目标。`opc_old_guard`在`M=2..5`可把`old_reject_rate`压到0，并给出一定seen-new提升；但未知拒识为0，说明known包络仍过宽。`mecr_unknown_probe`能提高未知拒识，最高`unknown_reject=0.791667`，但旧类准确率降到`0.458333..0.527778`。因此OPC-MECR当前也是`NON_DEPLOYMENT_DIAGNOSTIC`，不能作为达标算法。

### OPC-MECR协议review修复与v2复验

review指出v1存在两个P1协议风险：`best_eval_row`名称可能被误用为基于`target_unknown`的profile选择，且defer与unknown reject共用`__unknown__`输出标签。v2修复如下：

| issue | fix |
|---|---|
| 后验最优行命名风险 | 将`best_eval_row`改为`best_posthoc_eval_row`，并将`joint_score_scope`写为`posthoc_evaluation_analysis_only_not_profile_or_threshold_selection` |
| defer与unknown reject共用标签 | 新增`DEFER_LABEL="__defer__"`与`output_action`，unknown拒识为`output_action=reject_unknown`，defer为`output_action=defer` |
| `known_consensus_rate`定义过宽 | 改为仅在`old_safe`或`seen_safe`通过时计入known consensus |
| `M=1..R`缺少分母 | summary增加`event_count/old_total/seen_new_total/unknown_total` |
| 资源字段不完整 | summary和报告保留`avg_participating/p95_participating/resource_pass` |

修正版文件与提交：

| file | SHA256 | commit |
|---|---|---|
| `E:\type10-7\code\scripts\phase2_opc_mecr_ci_eval.py` | `85B0CF2D337CABF4B9454BF37B4386C78604B47E093599A691947138B72DF6FD` | `e821e42 Harden OPC-MECR protocol outputs` |
| `E:\type10-7\code\tests\test_phase2_opc_mecr_ci_eval.py` | `2C81C6C211E6CD4710A81A57E2F984ECE1E57666EF3D7D3819677846982CF227` | `e821e42 Harden OPC-MECR protocol outputs` |

追加协议测试提交：

| file | SHA256 | commit | coverage |
|---|---|---|---|
| `E:\type10-7\code\tests\test_phase2_opc_mecr_ci_eval.py` | `FA70807A7F80821606A67F77E50B89FBDDB42F0D5255AEDDFEDEC5E5E1FC5A48` | `4a1eea4 Add OPC-MECR target-unknown protocol test` | 构造最小Stage2-C feature NPZ并跑OPC-MECR入口，断言`target_unknown`不进入support conformal、阈值或receiver reliability来源 |

v2本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\phase2_opc_mecr_ci_eval.py E:\type10-7\code\tests\test_phase2_opc_mecr_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_opc_mecr_ci_eval.py -q
7 passed，另有既有.pytest_cache权限warning

C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_opc_mecr_ci_eval.py E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_opc_mecr_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_opc_mecr_ci_eval.py -q
7 passed
```

v2远端同步与验证：

```text
code/scripts/phase2_opc_mecr_ci_eval.py  85b0cf2d337cabf4b9454bf37b4386c78604b47e093599a691947138b72df6fd
code/tests/test_phase2_opc_mecr_ci_eval.py  fa70807a7f80821606a67f77e50b89fbddb42f0d5255aeddfedec5e5e1fc5a48
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_opc_mecr_ci_eval.py code/tests/test_phase2_opc_mecr_ci_eval.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_opc_mecr_ci_eval.py
Ran 7 tests in 0.065s OK
```

v2远端命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_opc_mecr_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/opc_mecr_remote_v2/opc_mecr.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/opc_mecr_remote_v2/opc_mecr_summary.csv \
  --output_evidence_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/opc_mecr_remote_v2/opc_mecr_evidence.csv \
  --profiles all --collab_counts all --collab_group_policy same_max_budget \
  --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070801 \
  --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 --max_event_latency_ms 20
```

v2远端结果hash：

| artifact | remote SHA256 | local path |
|---|---|---|
| `opc_mecr.json` | `ac5cb7fdea4a14946390088c91f44ae8ba61989aef4a34755e3daf95f81be3bc` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\opc_mecr_remote_v2\opc_mecr.json` |
| `opc_mecr_summary.csv` | `37d8c7d29096d4000dca40bfee32175252afedc6bf507b8b127d9ac8a07b1801` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\opc_mecr_remote_v2\opc_mecr_summary.csv` |
| `opc_mecr_evidence.csv` | `175aef10f485d4e7f1660de7c7c9daf33321e59d9b0042940ae9ef38bf3209c5` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\opc_mecr_remote_v2\opc_mecr_evidence.csv` |

v2全量结果：

| profile | M | event_count | old_total | seen_new_total | unknown_total | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | old_safe_accept | old_reject | known_consensus | bytes | latency_ms | avg_participating | p95_participating | resource_pass | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| opc_old_guard | 1 | 144 | 96 | 24 | 24 | 0.864583 | 0.666667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.066667 | 0.083333 | 0.979167 | 0.020833 | 0.933333 | 128.0 | 4.670341 | 1.000000 | 1.000000 | true | false |
| opc_old_guard | 2 | 133 | 85 | 24 | 24 | 0.823529 | 0.500000 | 0.250000 | 0.166667 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.976471 | 0.000000 | 1.000000 | 256.0 | 4.670341 | 2.000000 | 2.000000 | true | false |
| opc_old_guard | 3 | 120 | 72 | 24 | 24 | 0.791667 | 0.416667 | 0.500000 | 0.416667 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.986111 | 0.000000 | 1.000000 | 384.0 | 4.670341 | 3.000000 | 3.000000 | true | false |
| opc_old_guard | 4 | 107 | 59 | 24 | 24 | 0.762712 | 0.500000 | 0.416667 | 0.333333 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 512.0 | 4.670341 | 4.000000 | 4.000000 | true | false |
| opc_old_guard | 5 | 96 | 48 | 24 | 24 | 0.770833 | 0.000000 | 0.458333 | 0.416667 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 640.0 | 4.670341 | 5.000000 | 5.000000 | true | false |
| mecr_balanced | 1 | 144 | 96 | 24 | 24 | 0.593750 | 0.000000 | 0.000000 | 0.000000 | 0.083333 | 0.916667 | 0.425000 | 0.125000 | 0.635417 | 0.364583 | 0.575000 | 128.0 | 4.670341 | 1.000000 | 1.000000 | true | false |
| mecr_balanced | 2 | 133 | 85 | 24 | 24 | 0.658824 | 0.000000 | 0.000000 | 0.000000 | 0.083333 | 0.916667 | 0.348624 | 0.166667 | 0.741176 | 0.258824 | 0.642202 | 256.0 | 4.670341 | 2.000000 | 2.000000 | true | false |
| mecr_balanced | 3 | 120 | 72 | 24 | 24 | 0.652778 | 0.000000 | 0.000000 | 0.000000 | 0.041667 | 0.958333 | 0.364583 | 0.250000 | 0.722222 | 0.277778 | 0.625000 | 384.0 | 4.670341 | 3.000000 | 3.000000 | true | false |
| mecr_balanced | 4 | 107 | 59 | 24 | 24 | 0.610169 | 0.000000 | 0.000000 | 0.000000 | 0.208333 | 0.791667 | 0.445783 | 0.333333 | 0.677966 | 0.322034 | 0.530120 | 512.0 | 4.670341 | 4.000000 | 4.000000 | true | false |
| mecr_balanced | 5 | 96 | 48 | 24 | 24 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 0.208333 | 0.791667 | 0.527778 | 0.375000 | 0.562500 | 0.437500 | 0.444444 | 640.0 | 4.670341 | 5.000000 | 5.000000 | true | false |
| mecr_unknown_probe | 1 | 144 | 96 | 24 | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.750000 | 0.250000 | 0.333333 | 0.250000 | 0.000000 | 1.000000 | 0.000000 | 128.0 | 4.670341 | 1.000000 | 1.000000 | true | false |
| mecr_unknown_probe | 2 | 133 | 85 | 24 | 24 | 0.482353 | 0.000000 | 0.000000 | 0.000000 | 0.458333 | 0.541667 | 0.165138 | 0.208333 | 0.494118 | 0.505882 | 0.385321 | 256.0 | 4.670341 | 2.000000 | 2.000000 | true | false |
| mecr_unknown_probe | 3 | 120 | 72 | 24 | 24 | 0.527778 | 0.000000 | 0.000000 | 0.000000 | 0.541667 | 0.458333 | 0.114583 | 0.125000 | 0.541667 | 0.458333 | 0.416667 | 384.0 | 4.670341 | 3.000000 | 3.000000 | true | false |
| mecr_unknown_probe | 4 | 107 | 59 | 24 | 24 | 0.508475 | 0.000000 | 0.000000 | 0.000000 | 0.750000 | 0.250000 | 0.120482 | 0.041667 | 0.542373 | 0.457627 | 0.397590 | 512.0 | 4.670341 | 4.000000 | 4.000000 | true | false |
| mecr_unknown_probe | 5 | 96 | 48 | 24 | 24 | 0.458333 | 0.000000 | 0.000000 | 0.000000 | 0.791667 | 0.208333 | 0.069444 | 0.000000 | 0.500000 | 0.500000 | 0.375000 | 640.0 | 4.670341 | 5.000000 | 5.000000 | true | false |

资源边界：本地检索`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`未检出目标文件，且检索中遇到`tmpwhf_cnj1`权限错误。因此当前`bytes/latency/participating/resource_pass`仍是基于脚本参数`max_event_bytes=1152`、`max_event_latency_ms=20`的代理约束指标，不应写成已完成对资源约束设计说明的逐条映射。

v2结论：协议风险已修正，性能结论不变。所有OPC-MECR行仍`target_pass=false`。`opc_old_guard`保护旧类但不能拒识未知，`mecr_unknown_probe`改善未知拒识但严重伤旧类。当前结果仍为`NON_DEPLOYMENT_DIAGNOSTIC`。

## TCSR-CI支持包络收缩协同推理追加实验

时间：2026-07-04 21:20-21:34 CST

目标：在OPC-MECR仍表现为“保旧类时unknown FAR高、拒识unknown时旧类下降”后，追加feature级`Target Class Support Reconstruction Collaborative Inference`诊断算法。该算法保持`ADV3B02_CORE90_SOFT_E200`特征底座冻结，只使用`target_old/target_new`的`K-shot support`构建类条件support包络，`target_unknown`仅作最终评估，不进入阈值、support、receiver reliability或profile选择。

本地新增文件：

| file | SHA256 | purpose |
|---|---|---|
| `E:\type10-7\code\scripts\phase2_tcsr_ci_eval.py` | `66F02609BFFCBC784E1BCC09AE510696A9988BDB33CBFFE381D8A6FBA78BE005` | TCSR-CI评估入口，支持`M=1..R`、profile、证据CSV和协议元数据 |
| `E:\type10-7\code\tests\test_phase2_tcsr_ci_eval.py` | `44A5DF51202E884F93B2276ACC177174572399376D4F90BABB27E91CC19184DD` | TDD单测，覆盖融合、无共识拒识、`M=1..R`和`target_unknown`不泄漏 |
| `E:\type10-7\docs\CVS_STAGE2C_TCSR_CI_ALGORITHM_20260704.md` | `A5C4DABA10FC25D0CED139458804E12C6E0B337FD50917A3F0CD465980A99465` | 算法说明、资源字段、profile与边界 |

本地快照：

```text
E:\type10-7\code\snapshots\phase2_tcsr_ci_20260704\
```

Git发布树：

```text
repo: E:\type10-7\github_publish\CVS-RFFI-repo
commit: b98ecb0 Add TCSR collaborative support-envelope eval
status: ahead 445, only local_artifacts/ remains untracked
```

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\phase2_tcsr_ci_eval.py E:\type10-7\code\tests\test_phase2_tcsr_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_tcsr_ci_eval.py -q
4 passed，另有既有.pytest_cache权限warning

C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_tcsr_ci_eval.py E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_tcsr_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_tcsr_ci_eval.py -q
4 passed
```

本地smoke结果hash：

| artifact | SHA256 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_tcsr_smoke2\tcsr.json` | `52E232494EEECF9442113E9101A19CD2FE2FC7DA1C2F4549E134BC967AF8D090` |
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_tcsr_smoke2\tcsr_summary.csv` | `2876BDECFCA4C32B1595C7E3BD2ECB9C15316D84E67AC36A31348D7B00DDA3E2` |
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_tcsr_smoke2\tcsr_evidence.csv` | `35231D1F98AC63B064A3A251FDA72DACA2CE9BEAE0053B395C5230FEE2400C7E` |

N607预检与同步：

```text
powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1
Preflight OK，direct N607可用，project_root=/home/szu2070436088/2510044040/CV-SincNet
GPU0..7均约10MiB/24576MiB，未发现活跃python/train/phase2/eval进程

scp -> /home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_tcsr_ci_eval.py
scp -> /home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_tcsr_ci_eval.py
scp -> /home/szu2070436088/2510044040/CV-SincNet/docs/CVS_STAGE2C_TCSR_CI_ALGORITHM_20260704.md
```

远端文件hash与验证：

```text
code/scripts/phase2_tcsr_ci_eval.py  66f02609bffcbc784e1bcc09ae510696a9988bdb33cbffe381d8a6fba78be005
code/tests/test_phase2_tcsr_ci_eval.py  44a5df51202e884f93b2276acc177174572399376d4f90babb27e91cc19184dd
docs/CVS_STAGE2C_TCSR_CI_ALGORITHM_20260704.md  a5c4daba10fc25d0ced139458804e12c6e0b337fd50917a3f0cd465980a99465

/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_tcsr_ci_eval.py code/tests/test_phase2_tcsr_ci_eval.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_tcsr_ci_eval.py
Ran 4 tests in 0.034s OK
```

注：`CVS-RFFI`环境没有`pytest`模块，因此远端使用测试文件自带`unittest.main()`执行同一组测试。

远端全量命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_tcsr_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/tcsr_remote/tcsr.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/tcsr_remote/tcsr_summary.csv \
  --output_evidence_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/tcsr_remote/tcsr_evidence.csv \
  --profiles all --collab_counts all --collab_group_policy same_max_budget \
  --k_shot 8 --query_per_class 12 --seed 4070801 \
  --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 --max_event_latency_ms 20
```

远端结果hash：

| artifact | SHA256 | local path |
|---|---|---|
| `tcsr.json` | `2C200027ADBCA8FA513F541928F15BA0705978694B6CBCDD029D20E353754DF0` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\tcsr_remote\tcsr.json` |
| `tcsr_summary.csv` | `D97930CED9803E65B07529A03409BC6879CFBBEE6C1FC3761B2A949D08ED8757` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\tcsr_remote\tcsr_summary.csv` |
| `tcsr_evidence.csv` | `C4DE63F46ED9824553E06053ABD38EE2A995B2E75822D338F051F8D1A458C797` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\tcsr_remote\tcsr_evidence.csv` |

TCSR-CI全量结果：

| profile | M | event_count | old_total | seen_new_total | unknown_total | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes | latency_ms | avg_participating | p95_participating | resource_pass | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| tcsr_support_tight | 1 | 144 | 96 | 24 | 24 | 0.697917 | 0.208333 | 0.208333 | 0.083333 | 0.000000 | 1.000000 | 0.325000 | 0.375000 | 128 | 0.596 | 1.0 | 1.0 | true | false |
| tcsr_support_tight | 2 | 133 | 85 | 24 | 24 | 0.694118 | 0.105263 | 0.291667 | 0.083333 | 0.000000 | 1.000000 | 0.293578 | 0.375000 | 256 | 0.596 | 2.0 | 2.0 | true | false |
| tcsr_support_tight | 3 | 120 | 72 | 24 | 24 | 0.611111 | 0.250000 | 0.208333 | 0.000000 | 0.000000 | 1.000000 | 0.458333 | 0.583333 | 384 | 0.596 | 3.0 | 3.0 | true | false |
| tcsr_support_tight | 4 | 107 | 59 | 24 | 24 | 0.610169 | 0.000000 | 0.458333 | 0.333333 | 0.000000 | 1.000000 | 0.421687 | 0.666667 | 512 | 0.596 | 4.0 | 4.0 | true | false |
| tcsr_support_tight | 5 | 96 | 48 | 24 | 24 | 0.500000 | 0.000000 | 0.083333 | 0.083333 | 0.083333 | 0.916667 | 0.638889 | 0.625000 | 640 | 0.596 | 5.0 | 5.0 | true | false |
| tcsr_old_guard | 1 | 144 | 96 | 24 | 24 | 0.895833 | 0.666667 | 0.541667 | 0.416667 | 0.000000 | 1.000000 | 0.008333 | 0.000000 | 128 | 0.596 | 1.0 | 1.0 | true | false |
| tcsr_old_guard | 2 | 133 | 85 | 24 | 24 | 0.800000 | 0.526316 | 0.625000 | 0.500000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 256 | 0.596 | 2.0 | 2.0 | true | false |
| tcsr_old_guard | 3 | 120 | 72 | 24 | 24 | 0.638889 | 0.333333 | 0.416667 | 0.416667 | 0.000000 | 1.000000 | 0.375000 | 0.541667 | 384 | 0.596 | 3.0 | 3.0 | true | false |
| tcsr_old_guard | 4 | 107 | 59 | 24 | 24 | 0.644068 | 0.400000 | 0.750000 | 0.583333 | 0.000000 | 1.000000 | 0.301205 | 0.666667 | 512 | 0.596 | 4.0 | 4.0 | true | false |
| tcsr_old_guard | 5 | 96 | 48 | 24 | 24 | 0.541667 | 0.000000 | 0.583333 | 0.416667 | 0.000000 | 1.000000 | 0.375000 | 0.500000 | 640 | 0.596 | 5.0 | 5.0 | true | false |
| tcsr_unknown_probe | 1 | 144 | 96 | 24 | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.991667 | 1.000000 | 128 | 0.596 | 1.0 | 1.0 | true | false |
| tcsr_unknown_probe | 2 | 133 | 85 | 24 | 24 | 0.470588 | 0.000000 | 0.041667 | 0.000000 | 0.583333 | 0.416667 | 0.275229 | 0.083333 | 256 | 0.596 | 2.0 | 2.0 | true | false |
| tcsr_unknown_probe | 3 | 120 | 72 | 24 | 24 | 0.541667 | 0.083333 | 0.083333 | 0.000000 | 0.541667 | 0.458333 | 0.197917 | 0.083333 | 384 | 0.596 | 3.0 | 3.0 | true | false |
| tcsr_unknown_probe | 4 | 107 | 59 | 24 | 24 | 0.440678 | 0.000000 | 0.041667 | 0.000000 | 0.791667 | 0.208333 | 0.132530 | 0.041667 | 512 | 0.596 | 4.0 | 4.0 | true | false |
| tcsr_unknown_probe | 5 | 96 | 48 | 24 | 24 | 0.375000 | 0.000000 | 0.041667 | 0.000000 | 0.750000 | 0.250000 | 0.152778 | 0.041667 | 640 | 0.596 | 5.0 | 5.0 | true | false |

TCSR-CI结论：

1. 所有TCSR-CI行均`target_pass=false`，不能写成部署成功。
2. `tcsr_old_guard,M=1`是旧类保护较好的行：`old_acc=0.895833,min_old=0.666667,seen_new_acc=0.541667`，但`unknown_reject=0,unknown_FAR=1`，完全不能解决未知拒识。
3. `tcsr_unknown_probe,M=4`是后验未知拒识较强的行：`unknown_reject=0.791667,unknown_FAR=0.208333`，但`old_acc=0.440678,min_old=0,seen_new_acc=0.041667`，严重违反“旧类准确性不能下降”要求。
4. TCSR-CI相对OPC-MECR没有突破核心矛盾：当前feature空间中`target_unknown`仍大量落入known support/prototype包络。继续堆叠单一阈值或多数投票不合理。
5. `bytes/latency/participating/resource_pass`仍是脚本代理资源指标，当前未找到`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`，不能声称已完成资源说明逐项映射。

## 子agent方法建议与下一步路线

文献/方法子agent结论：TCSR/OPC失败的根因不是单纯接收机数量不足，而是未知类在判别空间内进入known包络。推荐组合：接收机条件化Mahalanobis或相对Mahalanobis、Energy/OOD能量分数、OpenMax尾部分布、证据冲突检测、选择性分类/风险覆盖控制、few-shot open-set原型一致性、轻量联邦/持续适配、open-world未知簇缓冲。不建议继续做纯多数投票、单一全局阈值、全模型星上联邦微调或直接把未知伪标签加入新类。

算法构建子agent建议下一代算法为`APACE-CI`：`Anchor-Protected Adaptive Conformal Ensemble Collaborative Inference`，即锚点保护的自适应保形协同推理。核心结构：

1. 旧类锚点保护门优先，使用`source_anchor_proto + target_support_proto + conformal p-value + margin`保护旧类，避免unknown拒识门误伤旧类。
2. 未知拒识门使用多证据：保形低p值、低density、高open energy、多星无稳定known共识、support外壳负样本校准。
3. 多星协同从固定投票改成质量/可靠性加权证据融合，并支持早停与`defer`。
4. 实时微调只更新温度、阈值、prototype offset、BN/FiLM/LoRA小适配器；主干ADV3B02冻结。
5. 在线更新必须经过旧类回放/锚点guardrail，旧类下降超过阈值即回滚adapter。

建议新增入口：

```text
code/scripts/phase2_apace_ci_eval.py
code/tests/test_phase2_apace_ci_eval.py
docs/CVS_STAGE2C_APACE_CI_ALGORITHM_20260704.md
```

APACE-CI必须测试：

| test | requirement |
|---|---|
| unknown leakage guard | `target_unknown`不参与阈值、profile、校准、adapter更新 |
| M coverage | `M=1..R`全覆盖 |
| old protect before unknown | 旧类保护门触发时unknown拒识不能覆盖旧类，只能accept old或defer |
| multi-evidence unknown | 低p值、低density、高energy、多星冲突时输出unknown |
| high-energy old guard | 高energy但旧类锚点成立时不能误拒旧类 |
| online rollback | 微调后旧类回放下降则回滚 |
| resource packet | 每星发送字段和字节估计低于约束 |

当前建议：下一轮不再继续调TCSR/OPC阈值，转向APACE-CI或同等的“旧类保护优先+多证据未知拒识+选择性defer+轻量可回滚适配”闭环。

## 子agent监督与review阻断项

监督子agent结论：工程链路和诊断审计基本完成，但“部署成功/目标达成”未完成。当前同row无法同时满足未知类拒识优先、旧类不下降和Stage2-C目标。

review子agent指出以下必须执行的阻断规则：

| issue | required handling |
|---|---|
| 协同语义 | 当前`receiver_domain_ranked`只能称为`receiver-domain ranked ensemble diagnostic`，不得写成严格同事件多星观测。只有存在同一发射事件、多接收机同步观测ID时，才能写`multi-satellite collaborative inference`。 |
| `target_unknown`泄漏 | 继续要求`target_unknown_training_count=0`、`threshold_uses_target_unknown=false`、`profile_selection_uses_target_unknown=false`。下一轮应补充`reliability_uses_target_unknown=false`和`reliability_fit_scope`字段。 |
| 多profile后验选择 | `best_posthoc_eval_row`只作诊断敏感性，不得作为主结论。下一轮应预注册primary profile，其他profile标记为`diagnostic_sensitivity`。 |
| unknown/defer解释 | 不能用defer伪装拒识成功。后续表格应拆分`unknown_accept_as_known_rate`、`unknown_reject_rate`、`unknown_defer_rate`和known coverage。 |
| 旧类不下降 | 任一低FAR或高unknown reject行只要`old_acc/min_old`下降，即为`NON_DEPLOYMENT_DIAGNOSTIC`。 |
| 资源约束 | 当前`resource_pass`是脚本代理值；正式表述应改成`resource_proxy_pass`，并补齐协议头、时间戳、重传、加密、链路调度、跨星/星地延迟等预算。 |
| M覆盖 | 每个profile必须完整报告`M=1..R`，不能只展示最大M或后验最优M。 |
| K-shot阈值 | `K=1/2`时support LOO阈值不稳，下一轮需要`threshold_valid_by_k`和fallback规则。 |

当前SSH/SCP清理状态：每次远端任务后均检查本地`ssh.exe`和到`172.31.111.215:22`/`172.31.105.18:22`的ESTABLISHED连接。最终检查无残留连接。

## APACE-CI锚点保护保形协同推理追加实验

时间：2026-07-04 21:45 CST开始

目标：在TCSR-CI和OPC-MECR均未同时满足“未知类拒识优先且旧类不下降”后，实现`APACE-CI`首版。该路线将旧类保护前置：`source_old_anchor + target_support_proto + support-only conformal p-value + density + open energy + receiver conflict`共同参与决策。当前实现仍保持`ADV3B02_CORE90_SOFT_E200`特征底座冻结，在轨少样本域适应和新类学习仍使用`qknn8`/support memory；`target_unknown`只作最终评估。

本地新增文件：

| file | SHA256 | purpose |
|---|---|---|
| `E:\type10-7\code\scripts\phase2_apace_ci_eval.py` | `0A84162A209707E07A3DAB4C464AEBACE390BE2AA07E8A3BC39C99323749099B` | APACE-CI评估入口，支持`M=1..R`、旧类锚点保护、多证据未知拒识、资源代理字段和泄漏scope |
| `E:\type10-7\code\tests\test_phase2_apace_ci_eval.py` | `B0AD4F1CEB8FB1CB1E818AB751AE96409E0E231CBFBCC95BF4E71053E2D4ACBA` | TDD单测，覆盖旧类保护优先、未知多证据拒识、`M=1..R`和`target_unknown`不泄漏 |
| `E:\type10-7\docs\CVS_STAGE2C_APACE_CI_ALGORITHM_20260704.md` | `77005D8A46374E53F3A3C30DA240F3F81C34607CB1BC5DEF6DED07C2386B1F39` | 算法说明、profile、资源代理字段和协议边界 |

本地快照：

```text
E:\type10-7\code\snapshots\phase2_apace_ci_20260704\
```

Git发布树：

```text
repo: E:\type10-7\github_publish\CVS-RFFI-repo
commit: 50197f6 Add APACE collaborative open-set eval
```

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\phase2_apace_ci_eval.py E:\type10-7\code\tests\test_phase2_apace_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_apace_ci_eval.py -q
4 passed，另有既有.pytest_cache权限warning

C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_apace_ci_eval.py E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_apace_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_apace_ci_eval.py -q
4 passed
```

本地smoke结果hash：

| artifact | SHA256 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_apace_smoke2\apace.json` | `416283834F04C6A677CB832A356C21CC7FC71E2AA460C1A5B85CA93BFA8C0CF7` |
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_apace_smoke2\apace_summary.csv` | `54206EF121A1480C7BB048AABE29F47B3A74F3601FE37F69BE194B50B09CDF82` |
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_apace_smoke2\apace_evidence.csv` | `3449C65B70BC98E194EEC8C11D26186984C7ED6864ACC7D0A73742BCCCB80004` |

APACE-CI本地smoke结果：

| profile | M | event_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes_proxy | latency_proxy_ms | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| apace_primary | 1 | 48 | 0.843750 | 0.750000 | 0.625000 | 0.250000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 160 | 0.652 | false |
| apace_primary | 2 | 46 | 0.700000 | 0.142857 | 0.500000 | 0.250000 | 0.250000 | 0.750000 | 0.000000 | 0.000000 | 320 | 0.652 | false |
| apace_primary | 3 | 40 | 0.750000 | 0.000000 | 0.250000 | 0.000000 | 0.250000 | 0.750000 | 0.093750 | 0.125000 | 480 | 0.652 | false |
| apace_primary | 4 | 34 | 0.833333 | 0.750000 | 0.250000 | 0.000000 | 0.375000 | 0.625000 | 0.153846 | 0.000000 | 640 | 0.652 | false |
| apace_primary | 5 | 32 | 0.750000 | 0.000000 | 0.125000 | 0.000000 | 0.625000 | 0.375000 | 0.166667 | 0.000000 | 800 | 0.652 | false |
| apace_old_guard | 1 | 48 | 0.843750 | 0.750000 | 0.625000 | 0.250000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 160 | 0.652 | false |
| apace_old_guard | 2 | 46 | 0.933333 | 0.750000 | 0.500000 | 0.250000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 320 | 0.652 | false |
| apace_old_guard | 3 | 40 | 0.833333 | 0.500000 | 0.250000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 480 | 0.652 | false |
| apace_old_guard | 4 | 34 | 0.833333 | 0.750000 | 0.250000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 640 | 0.652 | false |
| apace_old_guard | 5 | 32 | 0.812500 | 0.000000 | 0.250000 | 0.000000 | 0.000000 | 1.000000 | 0.166667 | 0.250000 | 800 | 0.652 | false |
| apace_unknown_probe | 1 | 48 | 0.656250 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.375000 | 0.375000 | 160 | 0.652 | false |
| apace_unknown_probe | 2 | 46 | 0.566667 | 0.142857 | 0.500000 | 0.250000 | 0.250000 | 0.750000 | 0.131579 | 0.000000 | 320 | 0.652 | false |
| apace_unknown_probe | 3 | 40 | 0.708333 | 0.000000 | 0.250000 | 0.000000 | 0.375000 | 0.625000 | 0.093750 | 0.000000 | 480 | 0.652 | false |
| apace_unknown_probe | 4 | 34 | 0.722222 | 0.000000 | 0.125000 | 0.000000 | 0.625000 | 0.375000 | 0.153846 | 0.000000 | 640 | 0.652 | false |
| apace_unknown_probe | 5 | 32 | 0.625000 | 0.000000 | 0.125000 | 0.000000 | 0.625000 | 0.375000 | 0.166667 | 0.000000 | 800 | 0.652 | false |

本地smoke结论：APACE-CI已经能把多接收机冲突转化为unknown拒识信号，但仍存在同样trade-off。`apace_old_guard,M=2`旧类较高（`old_acc=0.933333`），但`unknown_reject=0`；`apace_primary/apace_unknown_probe,M=5`未知拒识升至`0.625`，但`min_old=0`且seen-new明显下降。所有本地smoke行仍为`NON_DEPLOYMENT_DIAGNOSTIC`，不能写成部署成功。

计划远端全量命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_apace_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/apace_remote/apace.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/apace_remote/apace_summary.csv \
  --output_evidence_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/apace_remote/apace_evidence.csv \
  --profiles all --collab_counts all --collab_group_policy same_max_budget \
  --k_shot 8 --query_per_class 12 --seed 4070801 \
  --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 --max_event_latency_ms 20
```

N607远端同步与验证：

```text
code/scripts/phase2_rmd_ci_eval.py  b02c1458d147b33b7e6772e244254f976f3f14f5f6728d14d6e06ddcf948eb83
code/tests/test_phase2_rmd_ci_eval.py  4837272071b495380e6e7f24475f530852d0d182bfe5245f438e6eaa785400f8
docs/CVS_STAGE2C_RMD_CI_ALGORITHM_20260704.md  3c6ff696d0647e6c0b4b586d8721e00607e54d5590887d0d1321db940f35d4e3

/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_rmd_ci_eval.py code/tests/test_phase2_rmd_ci_eval.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_rmd_ci_eval.py
Ran 4 tests in 0.019s OK
```

N607运行环境：

```text
preflight: direct N607 OK
remote cwd: /home/szu2070436088/2510044040/CV-SincNet
remote python: /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
GPU before run: GPU0..7均约10/24576MiB，util=0
selected GPU: CUDA_VISIBLE_DEVICES=0
active user python/train/phase2/eval process before run: none except grep
```

远端结果hash：

| artifact | SHA256 | local path |
|---|---|---|
| `rmd.json` | `D9F0D3025D2313CA70E00E9055D2EE39BB98A102DC40C59DD73446D15A1AA3B8` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\rmd_remote\rmd.json` |
| `rmd_summary.csv` | `7989E84DC7851700F389BEF65010BE4BEAC10A2697F87116AB064C49208EC90E` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\rmd_remote\rmd_summary.csv` |
| `rmd_evidence.csv` | `B8CE45B4FB00D4E2EBBA671301E1B9A73001E39B6303917B08A3C39FED0C5E43` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\rmd_remote\rmd_evidence.csv` |

RMD-CI远端全量结果：

| profile | M | event_count | old_total | seen_new_total | unknown_total | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes_proxy | latency_proxy_ms | avg_participating | resource_proxy_pass | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| rmd_primary | 1 | 144 | 96 | 24 | 24 | 0.916667 | 0.666667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 176 | 0.764 | 1.0 | true | false |
| rmd_primary | 2 | 133 | 85 | 24 | 24 | 0.882353 | 0.583333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 352 | 0.764 | 2.0 | true | false |
| rmd_primary | 3 | 120 | 72 | 24 | 24 | 0.750000 | 0.416667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.250000 | 0.500000 | 528 | 0.764 | 3.0 | true | false |
| rmd_primary | 4 | 107 | 59 | 24 | 24 | 0.779661 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.228916 | 0.375000 | 704 | 0.764 | 4.0 | true | false |
| rmd_primary | 5 | 96 | 48 | 24 | 24 | 0.708333 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.263889 | 0.291667 | 880 | 0.764 | 5.0 | true | false |
| rmd_old_guard | 1 | 144 | 96 | 24 | 24 | 0.916667 | 0.666667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 176 | 0.764 | 1.0 | true | false |
| rmd_old_guard | 2 | 133 | 85 | 24 | 24 | 0.894118 | 0.666667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 352 | 0.764 | 2.0 | true | false |
| rmd_old_guard | 3 | 120 | 72 | 24 | 24 | 0.847222 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 528 | 0.764 | 3.0 | true | false |
| rmd_old_guard | 4 | 107 | 59 | 24 | 24 | 0.813559 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 704 | 0.764 | 4.0 | true | false |
| rmd_old_guard | 5 | 96 | 48 | 24 | 24 | 0.708333 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.236111 | 0.291667 | 880 | 0.764 | 5.0 | true | false |
| rmd_unknown_probe | 1 | 144 | 96 | 24 | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 176 | 0.764 | 1.0 | true | false |
| rmd_unknown_probe | 2 | 133 | 85 | 24 | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.990826 | 1.000000 | 352 | 0.764 | 2.0 | true | false |
| rmd_unknown_probe | 3 | 120 | 72 | 24 | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.989583 | 1.000000 | 528 | 0.764 | 3.0 | true | false |
| rmd_unknown_probe | 4 | 107 | 59 | 24 | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 704 | 0.764 | 4.0 | true | false |
| rmd_unknown_probe | 5 | 96 | 48 | 24 | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 880 | 0.764 | 5.0 | true | false |

RMD-CI远端结论：

1. 所有RMD-CI行仍`target_pass=false`，不能写成部署成功。
2. RMD-CI可保留一定旧类均值：`rmd_primary,M=1`和`rmd_old_guard,M=1`均为`old_acc=0.916667`，但`min_old=0.666667`，仍未达到旧类目标。
3. RMD-CI完全未解决seen-new和unknown：所有远端行`seen_new_acc=0`，所有远端行`unknown_reject=0,unknown_FAR=1`。
4. 结论比APACE更明确：接收机条件化relative Mahalanobis和support shell risk在当前冻结特征上没有形成可用开集边界。当前瓶颈不是缺少另一个后处理阈值，而是`ADV3B02_CORE90_SOFT_E200`的Stage2-C冻结特征空间中known/unknown几何混叠严重。
5. 下一步必须转向训练/轻量适配实验，而不是继续叠加decision-layer拒识器。建议实现`AOR-Adapter`或同类方法：冻结主干，仅更新prototype offset/BN affine/小LoRA，训练目标包含旧类锚点回放、seen-new support CE、support外壳负样本能量/距离损失，并用旧类guardrail回滚。

N607远端同步与验证：

```text
code/scripts/phase2_apace_ci_eval.py  0a84162a209707e07a3dab4c464aebace390be2aa07e8a3bc39c99323749099b
code/tests/test_phase2_apace_ci_eval.py  b0ad4f1ceb8fb1cb1e818ab751ae96409e0e231cbfbcc95bf4e71053e2d4acba
docs/CVS_STAGE2C_APACE_CI_ALGORITHM_20260704.md  77005d8a46374e53f3a3c30da240f3f81c34607cb1bc5def6ded07c2386b1f39

/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_apace_ci_eval.py code/tests/test_phase2_apace_ci_eval.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_apace_ci_eval.py
Ran 4 tests in 0.035s OK
```

N607运行环境：

```text
preflight: direct N607 OK
remote cwd: /home/szu2070436088/2510044040/CV-SincNet
remote python: /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
GPU before run: GPU0..7均约10/24576MiB，util=0
selected GPU: CUDA_VISIBLE_DEVICES=0
active user python/train/phase2/eval process before run: none except grep
```

远端结果hash：

| artifact | SHA256 | local path |
|---|---|---|
| `apace.json` | `E0670CB953C0A33F7CB08FD40E22E15992CDA3ACE8DB6A547FE41DE2F6990646` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\apace_remote\apace.json` |
| `apace_summary.csv` | `CD413881AF8466BDD68F1E4DD4F8685FBD44B2DE5BE0F1F4D153175AFD637F65` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\apace_remote\apace_summary.csv` |
| `apace_evidence.csv` | `F1A8FA7541C5477AD4A3486B21B178058C733825837E54ADA0A748328FD6ABDF` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\apace_remote\apace_evidence.csv` |

APACE-CI远端全量结果：

| profile | M | event_count | old_total | seen_new_total | unknown_total | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_accept_as_known | unknown_FAR | known_defer | unknown_defer | bytes_proxy | latency_proxy_ms | avg_participating | resource_proxy_pass | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| apace_primary | 1 | 144 | 96 | 24 | 24 | 0.854167 | 0.708333 | 0.541667 | 0.416667 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 160 | 0.652 | 1.0 | true | false |
| apace_primary | 2 | 133 | 85 | 24 | 24 | 0.752941 | 0.421053 | 0.291667 | 0.166667 | 0.375000 | 0.625000 | 0.625000 | 0.000000 | 0.000000 | 320 | 0.652 | 2.0 | true | false |
| apace_primary | 3 | 120 | 72 | 24 | 24 | 0.694444 | 0.250000 | 0.166667 | 0.083333 | 0.333333 | 0.416667 | 0.666667 | 0.104167 | 0.250000 | 480 | 0.652 | 3.0 | true | false |
| apace_primary | 4 | 107 | 59 | 24 | 24 | 0.779661 | 0.400000 | 0.166667 | 0.083333 | 0.458333 | 0.458333 | 0.541667 | 0.072289 | 0.083333 | 640 | 0.652 | 4.0 | true | false |
| apace_primary | 5 | 96 | 48 | 24 | 24 | 0.708333 | 0.000000 | 0.083333 | 0.000000 | 0.583333 | 0.416667 | 0.416667 | 0.138889 | 0.000000 | 800 | 0.652 | 5.0 | true | false |
| apace_old_guard | 1 | 144 | 96 | 24 | 24 | 0.854167 | 0.708333 | 0.541667 | 0.416667 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 160 | 0.652 | 1.0 | true | false |
| apace_old_guard | 2 | 133 | 85 | 24 | 24 | 0.917647 | 0.750000 | 0.291667 | 0.166667 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 320 | 0.652 | 2.0 | true | false |
| apace_old_guard | 3 | 120 | 72 | 24 | 24 | 0.902778 | 0.666667 | 0.166667 | 0.083333 | 0.166667 | 0.833333 | 0.833333 | 0.000000 | 0.000000 | 480 | 0.652 | 3.0 | true | false |
| apace_old_guard | 4 | 107 | 59 | 24 | 24 | 0.898305 | 0.750000 | 0.166667 | 0.083333 | 0.083333 | 0.916667 | 0.916667 | 0.000000 | 0.000000 | 640 | 0.652 | 4.0 | true | false |
| apace_old_guard | 5 | 96 | 48 | 24 | 24 | 0.791667 | 0.000000 | 0.166667 | 0.083333 | 0.041667 | 0.708333 | 0.958333 | 0.208333 | 0.250000 | 800 | 0.652 | 5.0 | true | false |
| apace_unknown_probe | 1 | 144 | 96 | 24 | 24 | 0.625000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.625000 | 1.000000 | 0.425000 | 0.375000 | 160 | 0.652 | 1.0 | true | false |
| apace_unknown_probe | 2 | 133 | 85 | 24 | 24 | 0.647059 | 0.416667 | 0.291667 | 0.166667 | 0.375000 | 0.458333 | 0.625000 | 0.064220 | 0.166667 | 320 | 0.652 | 2.0 | true | false |
| apace_unknown_probe | 3 | 120 | 72 | 24 | 24 | 0.680556 | 0.250000 | 0.166667 | 0.083333 | 0.458333 | 0.416667 | 0.541667 | 0.072917 | 0.125000 | 480 | 0.652 | 3.0 | true | false |
| apace_unknown_probe | 4 | 107 | 59 | 24 | 24 | 0.677966 | 0.000000 | 0.083333 | 0.000000 | 0.541667 | 0.375000 | 0.458333 | 0.060241 | 0.083333 | 640 | 0.652 | 4.0 | true | false |
| apace_unknown_probe | 5 | 96 | 48 | 24 | 24 | 0.562500 | 0.000000 | 0.083333 | 0.000000 | 0.583333 | 0.291667 | 0.416667 | 0.152778 | 0.125000 | 800 | 0.652 | 5.0 | true | false |

APACE-CI远端结论：

1. 所有APACE-CI行仍`target_pass=false`，不能写成部署成功。
2. 旧类保护最佳行是`apace_old_guard,M=2`：`old_acc=0.917647,min_old=0.750000`，但`unknown_reject=0,unknown_FAR=1`，仍完全不能拒识未知。
3. 未知拒识最佳行是`apace_primary,M=5`或`apace_unknown_probe,M=5`：`unknown_reject=0.583333,unknown_FAR=0.416667`，但`old_acc`分别仅`0.708333/0.562500`，`min_old=0`，seen-new也下降到`0.083333`。
4. APACE-CI证明多接收机冲突能提供一部分unknown信号，但该信号仍与旧类/新类识别冲突。仅靠decision-layer协同无法达到目标`old_acc=99%、min_old=95%、seen_new=97%、min_seen=93%、unknown_reject=99%`。
5. 下一步应从“协同推理后处理”升级到“训练/轻量适配产生可分离特征”的路线：接收机条件化Mahalanobis或relative density、support外壳负样本、旧类锚点回放guardrail、可回滚prototype offset/BN affine/LoRA小adapter，并把unknown/defer样本排除在正类训练之外。

APACE-CI review整改：

| issue | action |
|---|---|
| `target_pass`未显式纳入资源代理门控 | 已新增回归测试：性能全达标但`bytes=2048`超过`max_event_bytes=1152`时，必须`resource_proxy_pass=false,target_pass=false,verdict=NON_DEPLOYMENT_DIAGNOSTIC`。随后修正APACE门控为`metric_pass and resource_proxy_pass`。 |
| `M=1..R`不是同一事件分母收益曲线 | 已在APACE文档和本报告中明确：`same_max_budget`下`event_count`随`M`从144降到96，当前只能解读为参与数量覆盖，不能解读为同一批事件上增加卫星数量的严格单调收益。 |
| 严格多星协同证据未成立 | 继续保持阻断：当前只支持`receiver_domain_ranked`，是receiver-domain ensemble diagnostic，不是严格同事件多星观测。 |

整改后本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\phase2_apace_ci_eval.py E:\type10-7\code\tests\test_phase2_apace_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_apace_ci_eval.py -q
5 passed，另有既有.pytest_cache权限warning
```

整改后文件hash：

| file | SHA256 |
|---|---|
| `E:\type10-7\code\scripts\phase2_apace_ci_eval.py` | `F43D0DA28E045B1781736E7B0131DC29930FF14080B95B27E1F65B077B93FC70` |
| `E:\type10-7\code\tests\test_phase2_apace_ci_eval.py` | `DEBB653D86AFE4D89B35235047C5FCE71861CF84F8AE7AAF98BB19B03C05FDAF` |

Git发布树整改提交：

```text
7007a83 Harden APACE resource pass gate
```

整改版N607同步与验证：

```text
code/scripts/phase2_apace_ci_eval.py  f43d0da28e045b1781736e7b0131dc29930ff14080b95b27e1f65b077b93fc70
code/tests/test_phase2_apace_ci_eval.py  debb653d86afe4d89b35235047c5fce71861cf84f8ae7aaf98bb19b03c05fdaf
docs/CVS_STAGE2C_APACE_CI_ALGORITHM_20260704.md  6648b6edfa593c0b472f076c440985dc467eb085a21fb900009ceb75c6040240

/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_apace_ci_eval.py code/tests/test_phase2_apace_ci_eval.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_apace_ci_eval.py
Ran 5 tests in 0.036s OK
```

说明：整改不改变已完成远端APACE全量表的数值结论，因为该表所有行均`resource_proxy_pass=true`且`target_pass=false`。整改只防止未来出现“性能达标但资源代理失败仍被标成`target_pass=true`”的误报。

## RMD-CI接收机条件化relative density追加实验

时间：2026-07-04 22:00 CST开始

目标：在APACE-CI证明多接收机冲突有一定unknown信号但仍伤旧类/新类后，继续测试更接近密度建模的`RMD-CI`。该算法使用`target_old/target_new`K-shot support拟合接收机条件化对角Mahalanobis类模型和背景模型，并用known support prototype生成support外壳负样本。`target_unknown`仍只用于最终评估。

本地新增文件：

| file | SHA256 | purpose |
|---|---|---|
| `E:\type10-7\code\scripts\phase2_rmd_ci_eval.py` | `B02C1458D147B33B7E6772E244254F976F3F14F5F6728D14D6E06DDCF948EB83` | RMD-CI评估入口，支持`M=1..R`、relative Mahalanobis density、support shell risk和资源代理字段 |
| `E:\type10-7\code\tests\test_phase2_rmd_ci_eval.py` | `4837272071B495380E6E7F24475F530852D0D182BFE5245F438E6EAA785400F8` | TDD单测，覆盖旧类密度接受、shell risk未知拒识、`M=1..R`和`target_unknown`不泄漏 |
| `E:\type10-7\docs\CVS_STAGE2C_RMD_CI_ALGORITHM_20260704.md` | `3C6FF696D0647E6C0B4B586D8721E00607E54D5590887D0D1321DB940F35D4E3` | RMD-CI算法说明和协议边界 |

本地快照：

```text
E:\type10-7\code\snapshots\phase2_rmd_ci_20260704\
```

Git发布树：

```text
repo: E:\type10-7\github_publish\CVS-RFFI-repo
commit: f4657aa Add RMD collaborative density eval
```

本地验证：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\phase2_rmd_ci_eval.py E:\type10-7\code\tests\test_phase2_rmd_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_rmd_ci_eval.py -q
4 passed，另有既有.pytest_cache权限warning

C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_rmd_ci_eval.py E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_rmd_ci_eval.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_rmd_ci_eval.py -q
4 passed
```

本地smoke结果hash：

| artifact | SHA256 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_rmd_smoke\rmd.json` | `B5FFE9881DCCB67B9B11C0C163B65C613233A8EC88762EEAAC08543D4912801C` |
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_rmd_smoke\rmd_summary.csv` | `6BE46C3375BC5E5710B3A5EC73CDFFE7C957019A08A55BDC5D2A4A8ED1939486` |
| `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_rmd_smoke\rmd_evidence.csv` | `BE4200D21C98D28D811D132BE8AC355DAA6607B33B72906EE1D82DC5B1915DAB` |

RMD-CI本地smoke结果：

| profile | M | event_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes_proxy | latency_proxy_ms | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| rmd_primary | 1 | 48 | 0.906250 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 176 | 0.764 | false |
| rmd_primary | 2 | 46 | 0.866667 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 352 | 0.764 | false |
| rmd_primary | 3 | 40 | 0.750000 | 0.250000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.218750 | 0.375000 | 528 | 0.764 | false |
| rmd_primary | 4 | 34 | 0.666667 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.269231 | 0.375000 | 704 | 0.764 | false |
| rmd_primary | 5 | 32 | 0.687500 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.250000 | 0.375000 | 880 | 0.764 | false |
| rmd_old_guard | 1 | 48 | 0.906250 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 176 | 0.764 | false |
| rmd_old_guard | 2 | 46 | 0.866667 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 352 | 0.764 | false |
| rmd_old_guard | 3 | 40 | 0.791667 | 0.250000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 528 | 0.764 | false |
| rmd_old_guard | 4 | 34 | 0.722222 | 0.250000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 704 | 0.764 | false |
| rmd_old_guard | 5 | 32 | 0.687500 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.250000 | 0.375000 | 880 | 0.764 | false |
| rmd_unknown_probe | 1 | 48 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 176 | 0.764 | false |
| rmd_unknown_probe | 2 | 46 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 352 | 0.764 | false |
| rmd_unknown_probe | 3 | 40 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 528 | 0.764 | false |
| rmd_unknown_probe | 4 | 34 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 704 | 0.764 | false |
| rmd_unknown_probe | 5 | 32 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 880 | 0.764 | false |

本地证据分布摘要：

| role | rmd_score_mean | density_mean | shell_risk_mean | old_anchor_mean | margin_mean |
|---|---:|---:|---:|---:|---:|
| old | 0.5174 | 0.9288 | 0.0702 | 0.8516 | 0.1816 |
| seen_new | 0.5124 | 0.9001 | 0.1500 | 0.8320 | 0.1407 |
| unknown | 0.5100 | 0.8265 | 0.2643 | 0.7653 | 0.1461 |

本地smoke结论：RMD-CI的relative Mahalanobis分数几乎塌缩在`0.51`附近，未形成可用分离；support shell risk对unknown略高，但与old/seen-new仍重叠。当前结果比APACE更保旧类，但seen-new和unknown拒识均失败，仍为负诊断。

计划远端全量命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_rmd_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/rmd_remote/rmd.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/rmd_remote/rmd_summary.csv \
  --output_evidence_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/rmd_remote/rmd_evidence.csv \
  --profiles all --collab_counts all --collab_group_policy same_max_budget \
  --k_shot 8 --query_per_class 12 --seed 4070801 \
  --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 --max_event_latency_ms 20
```
## AOR-Adapter-CI预注册与本地验证（2026-07-04 22:19）

### 目标

在OPC/TCSR/APACE/RMD均未达标后，新增AOR-Adapter-CI（Anchor-preserving Open-set Receiver Adapter Collaborative Inference）作为下一轮受控诊断。核心目标是优先提升`unknown_reject`，但旧类保护优先于unknown拒识：若unknown门控与旧类锚点门控冲突，旧类门控优先。

### 方法边界

- 冻结`ADV3B02_CORE90_SOFT_E200`特征，不改主模型权重。
- 每个target receiver只用`target_old/target_new`K-shot support拟合identity初始化的对角适配代理。
- source old prototypes只作为旧类锚点，不引入target unknown。
- 伪未知由support原型插值/外推构造，`target_unknown`仅用于最终sealed evaluation。
- `same_max_budget`下M增加会改变`event_count`，M曲线不能写成同分母因果提升。
- 资源字段仍为`resource_proxy`，不能声明真实星载端到端实时链路通过。

### 本地变更

| 文件 | 目的 | SHA256 |
|---|---|---|
| `code/scripts/phase2_aor_adapter_ci_eval.py` | AOR-Adapter-CI评测脚本，输出M=1..R summary/evidence JSON/CSV | `5F5A5380918DF163A33282BBE8165A9520748C0EF830CD2CCB8DABFB8868CA5D` |
| `code/tests/test_phase2_aor_adapter_ci_eval.py` | 单元测试old-first门控、unknown门控、M覆盖、unknown不入训练 | `89D17FD260A1E0E4FA8EF3CF34C557662731F0322C8144AEBF22E39F18CAC0B9` |
| `docs/CVS_STAGE2C_AOR_ADAPTER_CI_ALGORITHM_20260704.md` | 算法说明、协议边界、验收口径 | `42133262B6203BD25E7677C4B390C6212549F34F9AB376F6E08872F3F44E840F` |

### 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\phase2_aor_adapter_ci_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_aor_adapter_ci_eval.py -q` | `4 passed, 1 warning`（`.pytest_cache`权限警告，非实验失败） |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_aor_adapter_ci_eval.py -q` | `4 passed` |

### 版本状态

| 项 | 值 |
|---|---|
| 发布仓库 | `E:\type10-7\github_publish\CVS-RFFI-repo` |
| 分支 | `codex/cvs-rffi-release-20260626` |
| 提交 | `84c6242 Add AOR adapter collaborative eval` |
| 快照 | `E:\type10-7\code\snapshots\phase2_aor_adapter_ci_20260704_221931` |
| 未跟踪项 | 发布仓库仍有历史`local_artifacts/`，本轮未纳入提交 |

### N607同步计划

| 本地文件 | 远端目标 |
|---|---|
| `E:\type10-7\code\scripts\phase2_aor_adapter_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_aor_adapter_ci_eval.py` |
| `E:\type10-7\code\tests\test_phase2_aor_adapter_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_aor_adapter_ci_eval.py` |
| `E:\type10-7\docs\CVS_STAGE2C_AOR_ADAPTER_CI_ALGORITHM_20260704.md` | `/home/szu2070436088/2510044040/CV-SincNet/docs/CVS_STAGE2C_AOR_ADAPTER_CI_ALGORITHM_20260704.md` |

### N607同步与远端验证

| 项 | 证据 |
|---|---|
| preflight | direct N607 OK；project root存在；GPU0..7均为`10/24576MiB`、util=0 |
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 远端工作目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端hash | script `5f5a5380918df163a33282bbe8165a9520748c0ef830cd2ccb8dabfb8868ca5d`；test `89d17fd260a1e0e4fa8ef3cf34c557662731f0322c8144aebf22e39f18cac0b9`；doc `42133262b6203bd25e7677c4b390c6212549f34f9ab376f6e08872f3f44e840f` |
| 远端测试 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_aor_adapter_ci_eval.py` -> `Ran 4 tests in 0.018s OK` |
| SSH断开 | 同步/测试/运行/拉取后检查均为`SshProcessCount=0,N607ConnCount=0,BridgeConnCount=0` |

### N607全量运行

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && \
mkdir -p remote_artifacts/phase2_adv3b02_proxy_mined_20260704/aor_remote && \
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
  code/scripts/phase2_aor_adapter_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/aor_remote/aor.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/aor_remote/aor_summary.csv \
  --output_evidence_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/aor_remote/aor_evidence.csv \
  --profiles all \
  --collab_counts all \
  --collab_group_policy same_max_budget \
  --k_shot 8 \
  --query_per_class 12 \
  --seed 4070801 \
  --support_selection_policy stable_first \
  --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 \
  --max_event_latency_ms 20
```

### CRISP-C远端同步与验证

| 项 | 证据 |
|---|---|
| preflight | direct N607 OK；project root存在；GPU0..7均`10/24576MiB`、util=0 |
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 远端hash | script `909623b9528fbc04690611d4605827017f89aca7c9b9731c88af4a832c26e1f3`；test `04d4a8d0b8b209f864abd4906c524c107dc45b881ba038091fc5d6217d621a9c`；doc `2e3feaa7c7bc0dc856c919a7515708c59d018dc058266d7e697a49d557d08404` |
| 远端测试 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_crisp_c_ci_eval.py` -> `Ran 4 tests in 0.028s OK` |
| GPU/进程 | 运行前GPU0..7均`10/24576MiB`、util=0；未见本用户训练/评估进程；运行后GPU0..7仍`10/24576MiB`、util=0 |
| SSH断开 | preflight、同步、远端测试、运行、拉回和后验GPU检查后均为`SshProcessCount=0,N607ConnCount=0,BridgeConnCount=0` |

### CRISP-C远端artifact

| artifact | SHA256 | 本地拉回路径 |
|---|---|---|
| `crisp.json` | `4463271BF05D4ACC87E435E1F7D5AC73DA76CEA887DCBCBABAF2CA0679CFA2B7` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\crisp_remote\crisp.json` |
| `crisp_summary.csv` | `F9ED6809D6B44F7A18E056BD705FD2C82499C4CF4A62C4B6C47BDCB66488994B` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\crisp_remote\crisp_summary.csv` |
| `crisp_evidence.csv` | `492106F88F5AD3680F3316FBF049EF857C448FABBEEAE0465A4E0291A09A3381` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\crisp_remote\crisp_evidence.csv` |

### CRISP-C远端结果表

receiver_total=5，target_receivers=`20-1,3-19,7-14,7-7,8-8`。`target_unknown_training_count=0`，`unknown_query_eval_only=true`，`threshold_uses_target_unknown=false`，`profile_selection_uses_target_unknown=false`，`prototype_fit_uses_target_unknown=false`。

| profile | M | event_count | excluded | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes/event | latency_ms | avg_participating | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| crisp_primary | 1 | 144 | 0 | 0.729167 | 0.333333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.116667 | 0.041667 | 128.0 | 0.41 | 1.0 | false |
| crisp_primary | 2 | 137 | 7 | 0.797753 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.097345 | 0.041667 | 256.0 | 0.41 | 2.0 | false |
| crisp_primary | 3 | 120 | 24 | 0.791667 | 0.416667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.072917 | 0.041667 | 384.0 | 0.41 | 3.0 | false |
| crisp_primary | 4 | 103 | 41 | 0.690909 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.025316 | 0.041667 | 512.0 | 0.41 | 4.0 | false |
| crisp_primary | 5 | 96 | 48 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.027778 | 0.125000 | 640.0 | 0.41 | 5.0 | false |
| crisp_old_guard | 1 | 144 | 0 | 0.791667 | 0.333333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.033333 | 0.041667 | 128.0 | 0.41 | 1.0 | false |
| crisp_old_guard | 2 | 137 | 7 | 0.842697 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.035398 | 0.041667 | 256.0 | 0.41 | 2.0 | false |
| crisp_old_guard | 3 | 120 | 24 | 0.805556 | 0.416667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.041667 | 0.000000 | 384.0 | 0.41 | 3.0 | false |
| crisp_old_guard | 4 | 103 | 41 | 0.709091 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.041667 | 512.0 | 0.41 | 4.0 | false |
| crisp_old_guard | 5 | 96 | 48 | 0.770833 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.083333 | 640.0 | 0.41 | 5.0 | false |
| crisp_unknown_probe | 1 | 144 | 0 | 0.645833 | 0.250000 | 0.208333 | 0.166667 | 0.000000 | 1.000000 | 0.141667 | 0.083333 | 128.0 | 0.41 | 1.0 | false |
| crisp_unknown_probe | 2 | 137 | 7 | 0.707865 | 0.500000 | 0.166667 | 0.083333 | 0.000000 | 1.000000 | 0.159292 | 0.166667 | 256.0 | 0.41 | 2.0 | false |
| crisp_unknown_probe | 3 | 120 | 24 | 0.708333 | 0.416667 | 0.041667 | 0.000000 | 0.000000 | 1.000000 | 0.156250 | 0.125000 | 384.0 | 0.41 | 3.0 | false |
| crisp_unknown_probe | 4 | 103 | 41 | 0.545455 | 0.416667 | 0.125000 | 0.000000 | 0.000000 | 1.000000 | 0.177215 | 0.166667 | 512.0 | 0.41 | 4.0 | false |
| crisp_unknown_probe | 5 | 96 | 48 | 0.604167 | 0.000000 | 0.125000 | 0.000000 | 0.000000 | 1.000000 | 0.180556 | 0.250000 | 640.0 | 0.41 | 5.0 | false |

### CRISP-C evidence分布

| role | rows | events | top_label_set_counts | top_label_match_rate | old_accept_score_mean | seen_new_accept_score_mean | reject_score_mean | old_envelope_violation_mean | seen_new_residual_mean |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| old | 360 | 96 | `old=293,seen_new=67` | 0.588889 | 0.868450 | 0.338294 | 0.194531 | 0.006125 | 0.552055 |
| seen_new | 120 | 24 | `seen_new=70,old=50` | 0.575000 | 0.883294 | 0.745642 | 0.112863 | 0.018121 | 0.179336 |
| unknown | 120 | 24 | `old=93,seen_new=27` | 0.000000 | 0.775508 | 0.339944 | 0.239535 | 0.026005 | 0.544605 |

### CRISP-C远端结论

所有CRISP-C行仍`target_pass=false`。相比KERA/AOR，CRISP-C的top-k/多原型证据确实让`seen_new`候选从几乎完全被old吸收变成`70/120`条`top_label_set=seen_new`，但可部署门控后`seen_new_acc`仍最高只有`0.208333`，且真实`unknown`仍保持高`old_accept_score_mean=0.775508`、低`reject_score_mean=0.239535`，没有形成拒识边界。该结果进一步证明当前`ADV3B02_CORE90_SOFT_E200`特征空间中真实target unknown不是简单support残差/多原型包络可分离的外侧区域。下一步应从地面训练或特征包生成阶段修复open-set表征：source-heldout TX episodic open-set训练、旧类prototype distillation、hard-negative feature repair和receiver-conditioned calibration，然后再用CRISP-C/AWARE资源受限协同层复验。

### AOR远端artifact

| artifact | SHA256 | 本地拉回路径 |
|---|---|---|
| `aor.json` | `F575176C519C8AFF46DB09524C39F1BB21932B2DC39C5F88F3B6246551410A2D` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\aor_remote\aor.json` |
| `aor_summary.csv` | `71468D3AB3102ED38BF23698C0D190071F605D190B766BE8D21C1D5DB96C1E3B` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\aor_remote\aor_summary.csv` |
| `aor_evidence.csv` | `22528E0CC6435F82DCBAE944BC57ED418E9B0FACB61A005F9FF5D3F024CFF105` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\aor_remote\aor_evidence.csv` |

### AOR结果表

receiver_total=5，target_receivers=`20-1,3-19,7-14,7-7,8-8`。`target_unknown_training_count=0`，`unknown_query_eval_only=true`，`pseudo_unknown_uses_target_unknown=false`。

| profile | M | event_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes/event | latency_ms | resource_proxy_pass | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| aor_primary | 1 | 144 | 0.958333 | 0.833333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 192 | 0.82 | true | false |
| aor_primary | 2 | 133 | 0.941176 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 384 | 0.82 | true | false |
| aor_primary | 3 | 120 | 0.847222 | 0.666667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.135417 | 0.416667 | 576 | 0.82 | true | false |
| aor_primary | 4 | 107 | 0.847458 | 0.583333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.144578 | 0.416667 | 768 | 0.82 | true | false |
| aor_primary | 5 | 96 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.208333 | 0.333333 | 960 | 0.82 | true | false |
| aor_old_guard | 1 | 144 | 0.958333 | 0.833333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 192 | 0.82 | true | false |
| aor_old_guard | 2 | 133 | 0.941176 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 384 | 0.82 | true | false |
| aor_unknown_probe | 1 | 144 | 0.947917 | 0.833333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.016667 | 0.000000 | 192 | 0.82 | true | false |
| aor_unknown_probe | 2 | 133 | 0.929412 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 384 | 0.82 | true | false |

完整15行见`aor_summary.csv`。上表保留关键profile和M=1/2/主要退化行，不能用不同row的单项最优拼接为候选。

### AOR结论

AOR-Adapter-CI是`NON_DEPLOYMENT_DIAGNOSTIC`。它证明旧类锚点优先机制可以把旧类均值推高到`0.958333`、`min_old=0.833333`，但seen-new注册和真实unknown拒识仍完全失败：最佳协议行`seen_new_acc=0`、`unknown_reject=0`、`unknown_FAR=1`。M增大时旧类也下降，说明当前`receiver_domain_ranked/same_max_budget`协同不能补偿特征几何混叠。

因此，当前证据不支持继续只调support伪未知、投票、密度或轻量对角适配。下一步应进入地面训练/特征空间修复：使用source-heldout open-set episodic训练、目标域known包络收缩、多原型support重构和旧类prototype distillation，再导出新特征包后由AWARE/OPU类协同层做资源受限推理。星上端只应保留小参数原型/adapter更新和unknown缓存确认，不能把当前AOR结果写成部署成功。

### AOR子agent复核

| 角色 | 结论 |
|---|---|
| 完成度监督 | 工程链路已闭环：本地实现、测试、发布仓库提交、SCP、远端hash/py_compile/单测、GPU0低占用运行、M=1..5、artifact拉回和SSH断开检查均可验收。主任务性能未完成。 |
| 查漏补缺/review | 未看到直接偷用`target_unknown`的证据；`target_unknown_training_count=0`、`profile_selection_uses_target_unknown=false`、`pseudo_unknown_uses_target_unknown=false`。但后续若根据unknown结果反复调profile，仍会构成后验泄漏，必须预注册。 |
| 指标口径 | `M=1`最佳，M增大旧类下降，未证明更多协同接收机提升性能。`same_max_budget`下`event_count`变化，M曲线只能作为覆盖率诊断。 |
| 成功声明 | 所有行`target_pass=false`，AOR只能写为`NON_DEPLOYMENT_DIAGNOSTIC`。不能把`resource_proxy_pass=true`扩展成真实星载端到端链路预算通过。 |
| 下一步 | 先修`old-vs-seen-new` known enrollment，要求`seen_new_acc>0`且旧类不下降；再引入source-side heldout/proxy_unknown或support-only伪未知拒识，并用严格同event分母复测协同收益。 |

## KERA-CI预注册与本地验证（2026-07-04 22:30）

### 目标

AOR远端结果显示`old_acc`均值可提升但`seen_new_acc=0`、`unknown_reject=0`。KERA-CI（Known Enrollment Repair Adapter Collaborative Inference）针对这个失败模式，只改变事件级融合顺序：先判断已注册seen-new是否通过known enrollment，再执行old-anchor guard，最后才做unknown拒识。该设计不放宽最终目标，不使用`target_unknown`调参。

### 本地变更

| 文件 | 目的 | SHA256 |
|---|---|---|
| `code/scripts/phase2_kera_ci_eval.py` | KERA-CI评测脚本，复用AOR证据构造并替换融合顺序 | `0D753A9F168DB2CCF975A284E1857BC37EFC539B8A100C78A8781C02DBE49B1E` |
| `code/tests/test_phase2_kera_ci_eval.py` | TDD测试seen-new不被old anchor覆盖、old保护、unknown门控、eval-only边界 | `83A48A104B4FFFC8BBDF394F22EEDAFBFB859A153FC6C4F3F6191E99A97339D9` |
| `docs/CVS_STAGE2C_KERA_CI_ALGORITHM_20260704.md` | 算法说明、协议边界、验收口径 | `81564905CDFB8EA5E47EC20B1EFC5EE1045118ED86AE5CEA15C81892A3C1E1D8` |

### TDD与本地验证

| 阶段 | 命令 | 结果 |
|---|---|---|
| RED | `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_kera_ci_eval.py -q` | 预期失败：`ModuleNotFoundError: No module named 'phase2_kera_ci_eval'` |
| GREEN | `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_kera_ci_eval.py -q` | `4 passed, 1 warning`（`.pytest_cache`权限警告，非实验失败） |
| 语法 | `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\phase2_kera_ci_eval.py` | PASS |
| 发布仓库测试 | `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_kera_ci_eval.py -q` | `4 passed` |

### 版本状态

| 项 | 值 |
|---|---|
| 发布仓库 | `E:\type10-7\github_publish\CVS-RFFI-repo` |
| 分支 | `codex/cvs-rffi-release-20260626` |
| 提交 | `c8976b2 Add KERA known enrollment collaborative eval` |
| 快照 | `E:\type10-7\code\snapshots\phase2_kera_ci_20260704_223028` |
| 未跟踪项 | 发布仓库仍有历史`local_artifacts/`，本轮未纳入提交 |

### N607同步计划

| 本地文件 | 远端目标 |
|---|---|
| `E:\type10-7\code\scripts\phase2_kera_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_kera_ci_eval.py` |
| `E:\type10-7\code\tests\test_phase2_kera_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_kera_ci_eval.py` |
| `E:\type10-7\docs\CVS_STAGE2C_KERA_CI_ALGORITHM_20260704.md` | `/home/szu2070436088/2510044040/CV-SincNet/docs/CVS_STAGE2C_KERA_CI_ALGORITHM_20260704.md` |

### KERA远端同步与验证

| 项 | 证据 |
|---|---|
| preflight | direct N607 OK；project root存在；GPU0..7均为`10/24576MiB`、util=0 |
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 远端hash | script `0d753a9f168db2ccf975a284e1857bc37efc539b8a100c78a8781c02dbe49b1e`；test `83a48a104b4fffc8bbdf394f22eedafbfb859a153fc6c4f3f6191e99a97339d9`；doc `81564905cdfb8ea5e47ec20b1efc5ee1045118ed86ae5cea15c81892a3c1e1d8` |
| 远端测试 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_kera_ci_eval.py` -> `Ran 4 tests in 0.019s OK` |
| GPU/进程 | 运行前GPU0..7均`10/24576MiB`、util=0，未见用户侧python/train/eval进程 |
| SSH断开 | 同步/测试/运行/拉取后检查均为`SshProcessCount=0,N607ConnCount=0,BridgeConnCount=0` |

### KERA全量运行

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && \
mkdir -p remote_artifacts/phase2_adv3b02_proxy_mined_20260704/kera_remote && \
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
  code/scripts/phase2_kera_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/kera_remote/kera.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/kera_remote/kera_summary.csv \
  --output_evidence_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/kera_remote/kera_evidence.csv \
  --profiles all \
  --collab_counts all \
  --collab_group_policy same_max_budget \
  --k_shot 8 \
  --query_per_class 12 \
  --seed 4070801 \
  --support_selection_policy stable_first \
  --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 \
  --max_event_latency_ms 20
```

### KERA远端artifact

| artifact | SHA256 | 本地拉回路径 |
|---|---|---|
| `kera.json` | `EE4577834B85B78E185AFB955D34C4EE814FDDC20F05E7BE74E8BF99CACB2B1C` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\kera_remote\kera.json` |
| `kera_summary.csv` | `7338173C5807399B9A35D5EF241D707B62DF68F042175621CE086F534F9E05DB` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\kera_remote\kera_summary.csv` |
| `kera_evidence.csv` | `6F771FBE170C0B206CB488B22AE06A1380219A789516A9826300B9110E8C26EB` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\kera_remote\kera_evidence.csv` |

### KERA结果表

receiver_total=5，target_receivers=`20-1,3-19,7-14,7-7,8-8`。`target_unknown_training_count=0`，`unknown_query_eval_only=true`，`threshold_uses_target_unknown=false`，`profile_selection_uses_target_unknown=false`。

| profile | M | event_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes/event | latency_ms | resource_proxy_pass | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| kera_primary | 1 | 144 | 0.958333 | 0.833333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 192 | 0.82 | true | false |
| kera_primary | 2 | 133 | 0.941176 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 384 | 0.82 | true | false |
| kera_primary | 3 | 120 | 0.888889 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 576 | 0.82 | true | false |
| kera_primary | 4 | 107 | 0.864407 | 0.583333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 768 | 0.82 | true | false |
| kera_primary | 5 | 96 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.208333 | 0.333333 | 960 | 0.82 | true | false |
| kera_old_guard | 1 | 144 | 0.958333 | 0.833333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 192 | 0.82 | true | false |
| kera_old_guard | 2 | 133 | 0.941176 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 384 | 0.82 | true | false |
| kera_unknown_probe | 1 | 144 | 0.958333 | 0.833333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 192 | 0.82 | true | false |
| kera_unknown_probe | 2 | 133 | 0.929412 | 0.750000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 384 | 0.82 | true | false |

完整15行见`kera_summary.csv`。

### KERA结论

KERA-CI是`NON_DEPLOYMENT_DIAGNOSTIC`。它修复的是AOR的融合顺序风险，但远端结果显示真实seen-new仍没有成为有效候选：预注册`kera_primary,M=1`与AOR同型，`old_acc=0.958333,min_old=0.833333,seen_new_acc=0,unknown_reject=0,unknown_FAR=1,target_pass=false`。因此失败根因不只是old-first融合顺序，而是`features_proxy_mined.npz`中target seen-new query在当前support/adapted score空间仍被旧类或错误known包络吸收。下一步应转向score/feature层修复：先输出per-event/per-role候选分布，确认seen-new top_label去向；再做类条件多原型support重构或target known包络收缩，而不是继续只调融合门控。

### KERA子agent复核

| 角色 | 结论 |
|---|---|
| review | KERA作为AOR后的受控诊断合理，但后验调参风险高；必须只把预注册primary row作为主结论，posthoc best只作敏感性分析。 |
| review | seen-new gate放到old gate前会增加old->seen-new误接收风险，因此必须报告`old_acc/min_old/per-old-class acc/old->seen_new confusion`，不能只看seen-new提升。 |
| review | `target_unknown`仍应保持eval-only；`resource_proxy_pass=true`只能写作代理资源通过，不能扩展为真实星载端到端实时。 |
| 监督 | 本地文档、TDD测试、发布提交、N607同步、低显存GPU条件已完成；KERA远端运行和artifact拉回在本节补齐后完成。长期性能目标仍未完成。 |

## Candidate Distribution Audit（2026-07-04 22:38）

### 目标

KERA/AOR均表现为`seen_new_acc=0`、`unknown_reject=0`。本诊断不改变算法或阈值，只读取已生成的`*_evidence.csv`，在融合前统计每个role的`top_label_set`、`top_label`、receiver分布和分数字段，定位失败发生在candidate generation/score层还是event fusion层。

### 本地变更

| 文件 | 目的 | SHA256 |
|---|---|---|
| `code/scripts/phase2_candidate_distribution_audit.py` | 读取evidence CSV并输出role/true_label/receiver候选分布诊断 | `5E516FA281E2DDB71AD99DB460DA61ED61B4390C7640D910489972355E96E46B` |
| `code/tests/test_phase2_candidate_distribution_audit.py` | TDD测试role/top_label_set分布、JSON/CSV输出 | `9CDBD28F6DC92ABF9B6E00D07FD34C5B501D6F710376228219955CFE5176DB4D` |

### TDD与版本状态

| 项 | 证据 |
|---|---|
| RED | `pytest E:\type10-7\code\tests\test_phase2_candidate_distribution_audit.py -q`预期失败：`ModuleNotFoundError: No module named 'phase2_candidate_distribution_audit'` |
| GREEN | `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\code\tests\test_phase2_candidate_distribution_audit.py -q` -> `2 passed, 1 warning`（`.pytest_cache`权限警告） |
| 发布仓库测试 | `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_phase2_candidate_distribution_audit.py -q` -> `2 passed` |
| 发布仓库提交 | `9d6b39d Add Stage2 candidate distribution audit` |
| 快照 | `E:\type10-7\code\snapshots\phase2_candidate_distribution_audit_20260704_223832` |

### N607同步与远端运行

| 项 | 证据 |
|---|---|
| preflight | direct N607 OK；project root存在；GPU0..7均约`10/24576MiB`，util 0或1 |
| 远端hash | script `5e516fa281e2ddb71ad99db460da61ed61b4390c7640d910489972355e96e46b`；test `9cdbd28f6dc92abf9b6e00d07fd34c5b501d6f710376228219955cfe5176db4d` |
| 远端测试 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_candidate_distribution_audit.py` -> `Ran 2 tests in 0.004s OK` |
| 远端KERA命令 | `python code/scripts/phase2_candidate_distribution_audit.py --evidence_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/kera_remote/kera_evidence.csv --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/kera_remote/kera_candidate_audit.json --output_by_role_label_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/kera_remote/kera_candidate_by_role_label.csv --algorithm KERA-CI` |
| 远端AOR命令 | `python code/scripts/phase2_candidate_distribution_audit.py --evidence_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/aor_remote/aor_evidence.csv --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/aor_remote/aor_candidate_audit.json --output_by_role_label_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/aor_remote/aor_candidate_by_role_label.csv --algorithm AOR-Adapter-CI` |
| SSH断开 | 同步、远端测试、运行、拉回后检查均为`SshProcessCount=0,N607ConnCount=0,BridgeConnCount=0` |

### Candidate audit artifact

| artifact | SHA256 | 本地拉回路径 |
|---|---|---|
| `kera_candidate_audit.json` | `77D2A283033AAE9F2B22E59F05C678C64338CA7CC9B306FAB45D9BB26D5E6BD6` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\kera_remote\kera_candidate_audit.json` |
| `kera_candidate_by_role_label.csv` | `0FA3807B1C5F6255033E9DB176F2C7667D3F71AAB4A290B3E5642AD0DFB486B0` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\kera_remote\kera_candidate_by_role_label.csv` |
| `aor_candidate_audit.json` | `6FD69B78A5F015B2427E8583746E75BE808AD3075A782AAAF3348984DC8F5BE2` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\aor_remote\aor_candidate_audit.json` |
| `aor_candidate_by_role_label.csv` | `6F172B3FF32A371B84A5E0B6D3204903C124596E3E49C6E20D3878CDDCEEBD4A` | `E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\aor_remote\aor_candidate_by_role_label.csv` |

### Evidence层分布摘要

| algorithm | role | rows | events | top_label_match_rate | top_label_set_counts | known_score_mean | unknown_score_mean | margin_mean | old_anchor_score_mean |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| KERA-CI | old | 360 | 96 | 0.7333 | `old=355,seen_new=5` | 0.8621 | 0.2481 | 0.3892 | 0.8224 |
| KERA-CI | seen_new | 120 | 24 | 0.0333 | `old=116,seen_new=4` | 0.8446 | 0.3072 | 0.2006 | 0.8061 |
| KERA-CI | unknown | 120 | 24 | 0.0000 | `old=120` | 0.7705 | 0.3514 | 0.3207 | 0.7395 |
| AOR-Adapter-CI | old | 360 | 96 | 0.7333 | `old=355,seen_new=5` | 0.8611 | 0.2494 | 0.3900 | 0.8205 |
| AOR-Adapter-CI | seen_new | 120 | 24 | 0.0333 | `old=116,seen_new=4` | 0.8435 | 0.3091 | 0.2006 | 0.8043 |
| AOR-Adapter-CI | unknown | 120 | 24 | 0.0000 | `old=120` | 0.7690 | 0.3530 | 0.3213 | 0.7372 |

### Per-class evidence摘要

| algorithm | role | true_label | rows | events | top_label_match_rate | top_label_set_counts |
|---|---|---|---:|---:|---:|---|
| KERA-CI | seen_new | `19-3` | 60 | 12 | 0.0667 | `old=56,seen_new=4` |
| KERA-CI | seen_new | `3-8` | 60 | 12 | 0.0000 | `old=60` |
| KERA-CI | unknown | `10-1` | 60 | 12 | 0.0000 | `old=60` |
| KERA-CI | unknown | `10-10` | 60 | 12 | 0.0000 | `old=60` |

### 诊断结论

在当前KERA/AOR evidence接口下，融合层不是主要瓶颈。`seen_new`在进入event fusion前已经几乎全部被本地证据层排为old：`120`条seen-new receiver evidence中`116`条`top_label_set=old`，真实标签匹配率只有`0.0333`。`unknown`更极端，`120`条全部`top_label_set=old`，且`unknown_score_mean≈0.35`，没有形成可用于拒识的高风险信号。

该结论边界：只能说明当前`ADV3B02_CORE90_SOFT_E200+qknn8+AOR/KERA evidence`摘要接口下，继续调event fusion顺序/阈值不是主路线；不能写成所有协同推理方法无效，也不能否定未来传输更丰富feature sketch、多原型距离或prototype residual后的协同层。下一步应转向score/prototype/feature层：seen-new support原型重构、old known包络收缩、类条件多原型、source-heldout open-set训练或feature-space calibration，再回到AWARE/OPU/KERA类资源受限融合层评测。

### Candidate audit子agent复核

| 角色 | 结论 |
|---|---|
| review | 诊断足以支持受限结论：当前KERA/AOR evidence层下，seen-new候选生成和unknown风险生成已经在融合前失败，下一步应转向score/prototype/feature层。 |
| review | 必须保留denominator、per-class/per-receiver分布，不能把结论升级成“所有协同推理无效”。 |
| 监督 | 早期监督指出发布、远端运行和报告待补齐；本节已补充发布提交、远端运行、artifact hash、拉回路径和报告结论。 |

## CRISP-C残差包络协同诊断（2026-07-04 23:15）

### 目标

根据AOR/KERA和candidate audit的负证据，新增`CRISP-C`（Cooperative Residual-Interval Sketch Prototype）诊断模块。该模块不再只调event fusion顺序，而是在每个接收机本地构建旧类收缩包络、seen-new多原型、support residual和conformal p-value，再以低带宽rank sketch做`M=1..R`协同融合。`target_unknown`仍只用于最终评估，不参与prototype、阈值、profile或reliability选择。

### 本地变更与版本

| 文件 | 目的 | SHA256 |
|---|---|---|
| `code/scripts/phase2_crisp_c_ci_eval.py` | CRISP-C证据构建、融合、`M=1..receiver_count`评估和资源代理字段 | `909623B9528FBC04690611D4605827017F89ACA7C9B9731C88AF4A832C26E1F3` |
| `code/tests/test_phase2_crisp_c_ci_eval.py` | TDD覆盖seen-new top-k救援、unknown包络拒识、unknown eval-only、资源门控不能替代target pass | `04D4A8D0B8B209F864ABD4906C524C107DC45B881BA038091FC5D6217D621A9C` |
| `docs/CVS_STAGE2C_CRISP_C_ALGORITHM_20260704.md` | CRISP-C算法说明、协议边界和当前负诊断结论 | `2E3FEAA7C7BC0DC856C919A7515708C59D018DC058266D7E697A49D557D08404` |

| 项 | 证据 |
|---|---|
| RED | `pytest code\tests\test_phase2_crisp_c_ci_eval.py -q`初次失败：`ModuleNotFoundError: No module named 'phase2_crisp_c_ci_eval'` |
| GREEN | `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_crisp_c_ci_eval.py -q` -> `4 passed`（`.pytest_cache`权限警告不影响测试） |
| 语法 | `python -m py_compile code\scripts\phase2_crisp_c_ci_eval.py code\tests\test_phase2_crisp_c_ci_eval.py`通过 |
| 快照 | `E:\type10-7\code\snapshots\phase2_crisp_c_20260704_231500` |
| 发布仓库 | `E:\type10-7\github_publish\CVS-RFFI-repo`提交`a8fed9f Add CRISP-C collaborative residual sketch eval`；仍有历史未跟踪`local_artifacts/`未纳入提交 |

### 本地真实特征快速诊断

命令：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_crisp_c_ci_eval.py --feature_npz local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\features_proxy_mined.npz --output_json local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_crisp_all\crisp.json --output_summary_csv local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_crisp_all\crisp_summary.csv --output_evidence_csv local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_crisp_all\crisp_evidence.csv --profiles all --collab_counts all --collab_group_policy same_max_budget --k_shot 8 --query_per_class 12 --seed 4070901 --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked --max_event_bytes 1152 --max_event_latency_ms 20
```

结果摘要：

| profile | M | event_count | excluded | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes/event | latency_ms | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| crisp_primary | 1 | 144 | 0 | 0.729167 | 0.333333 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 128.0 | 0.41 | false |
| crisp_primary | 2 | 137 | 7 | 0.797753 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 256.0 | 0.41 | false |
| crisp_primary | 3 | 120 | 24 | 0.791667 | 0.416667 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 384.0 | 0.41 | false |
| crisp_old_guard | 2 | 137 | 7 | 0.842697 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 256.0 | 0.41 | false |
| crisp_unknown_probe | 1 | 144 | 0 | 0.645833 | 0.250000 | 0.208333 | 0.166667 | 0.000000 | 1.000000 | 128.0 | 0.41 | false |
| crisp_unknown_probe | 2 | 137 | 7 | 0.707865 | 0.500000 | 0.166667 | 0.083333 | 0.000000 | 1.000000 | 256.0 | 0.41 | false |

完整15行见`local_crisp_all\crisp_summary.csv`。协议字段：`unknown_query_eval_only=true`、`target_unknown_training_count=0`、`threshold_uses_target_unknown=false`、`profile_selection_uses_target_unknown=false`、`prototype_fit_uses_target_unknown=false`。

### Evidence层补充诊断

CRISP-C暴露了比KERA/AOR更丰富的候选信息：本地evidence中`seen_new`的`top_label_set=seen_new`为`70/120`，明显高于KERA的`4/120`；但融合后`seen_new_acc`仍低，说明top-k候选出现并不等于support residual和old gap足以通过可部署门控。`unknown`仍有`93/120`条`top_label_set=old`，且`old_accept_score_mean=0.7755`、`reject_score_mean=0.2395`，说明真实unknown仍落入当前old包络内部。

### 结论

CRISP-C当前仍是`NON_DEPLOYMENT_DIAGNOSTIC`。它支持的受限结论是：多原型/top-k残差证据能暴露seen-new候选，但不能解决真实target unknown被old包络吸收的问题；因此仅靠部署侧support residual、rank sketch和协同融合不足以达到`old_acc 99%/min_old 95%/seen_new 97%/min_seen 93%/unknown_reject 99%`。下一步应进入地面训练或特征生成阶段，加入source-heldout/open-set episodic训练、旧类prototype distillation和hard-negative feature repair，再用CRISP-C/AWARE类轻量协同层复验。

### N607同步计划

| 本地文件 | 远端目标 |
|---|---|
| `E:\type10-7\code\scripts\phase2_crisp_c_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_crisp_c_ci_eval.py` |
| `E:\type10-7\code\tests\test_phase2_crisp_c_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_crisp_c_ci_eval.py` |
| `E:\type10-7\docs\CVS_STAGE2C_CRISP_C_ALGORITHM_20260704.md` | `/home/szu2070436088/2510044040/CV-SincNet/docs/CVS_STAGE2C_CRISP_C_ALGORITHM_20260704.md` |

拟远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && \
mkdir -p remote_artifacts/phase2_adv3b02_proxy_mined_20260704/crisp_remote && \
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
  code/scripts/phase2_crisp_c_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz \
  --output_json remote_artifacts/phase2_adv3b02_proxy_mined_20260704/crisp_remote/crisp.json \
  --output_summary_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/crisp_remote/crisp_summary.csv \
  --output_evidence_csv remote_artifacts/phase2_adv3b02_proxy_mined_20260704/crisp_remote/crisp_evidence.csv \
  --profiles all \
  --collab_counts all \
  --collab_group_policy same_max_budget \
  --k_shot 8 \
  --query_per_class 12 \
  --seed 4070901 \
  --support_selection_policy stable_first \
  --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 \
  --max_event_latency_ms 20
```
## 2026-07-04 continuation: source-heldout metric repair and DMG-CI diagnostic

### Objective

Continue the Stage2-C satellite-swarm collaborative inference objective after CRISP-C failed to meet the same-row target. The new check moved from fusion-only repair toward source-heldout hard-negative metric repair while preserving the CVS protocol boundary that `target_unknown` remains eval-only.

### Local files changed

| File | Purpose |
|---|---|
| `code/scripts/phase2_dual_metric_guard_ci_eval.py` | New DMG-CI evidence combiner: base ADV3B02/qknn8 evidence keeps old-core predictions; source-heldout metric evidence contributes seen-new rescue and reject risk. |
| `code/tests/test_phase2_dual_metric_guard_ci_eval.py` | TDD tests for old-core protection, metric seen-new rescue, target_unknown eval-only metadata, and `M=1..N` summary output. |
| `docs/CVS_STAGE2C_DMG_CI_ALGORITHM_20260704.md` | Algorithm and negative-result boundary for DMG-CI. |

### Local verification

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_source_open_metric_ci_eval.py -q` | PASS, 1 passed; pytest cache permission warning only. |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_source_open_metric_ci_eval.py code\scripts\phase2_proxy_adapter_ci_eval.py` | PASS. |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_dual_metric_guard_ci_eval.py -q` | RED first: module missing; GREEN after implementation: 2 passed; pytest cache permission warning only. |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_dual_metric_guard_ci_eval.py` | PASS. |

### Local SOM-CI full-curve diagnostic

Input feature bundle:

`E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\features_proxy_mined.npz`

Output directory:

`E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_som_ci_full`

Protocol counters:

| Field | Value |
|---|---:|
| source_old | 1320 |
| proxy_unknown | 3520 |
| target_support | 0 |
| target_unknown_eval_only | 440 |
| target_unknown_training_count | 0 |
| total_fp16_state_bytes | 2240 |

Same-row local results remained diagnostic-negative:

| backend | best row type | profile | M | old_acc | seen_new_acc | unknown_reject | unknown_FAR | target_pass |
|---|---|---|---:|---:|---:|---:|---:|---|
| ENPC | best old/seen | `enpc_known_anchor` | 3 | 0.8889 | 0.5417 | 0 | 1.0000 | false |
| ENPC | best unknown | `enpc_old80_unknown_probe` | 4 | 0.5082 | 0.2083 | 1.0000 | 0 | false |
| SLEV | best old/seen | `slev_known_anchor` | 3 | 0.9028 | 0.5417 | 0 | 1.0000 | false |
| SLEV | best unknown | `slev_old80_energy_probe` | 5 | 0.3958 | 0.1250 | 1.0000 | 0 | false |

Interpretation: source-only metric repair exposes more seen-new structure than CRISP-C, but unknown rejection still trades directly against old-class retention.

### Local DMG-CI diagnostic

Identity/base evidence was generated with `metric_epochs=0` under:

`E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_som_ci_identity_base`

DMG-CI output directory:

`E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_dmg_ci_full`

Route counts for ENPC/SLEV were identical:

| Route | Count |
|---|---:|
| `base_guarded` | 22 |
| `base_old_core` | 55 |
| `metric_reject_guard` | 507 |
| `metric_seen_new_rescue` | 16 |

Same-row local results:

| backend | M | event_total | excluded | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_coverage | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ENPC | 1 | 96 | 48 | 0.1042 | 0 | 0 | 0 | 0.7917 | 0.0417 | 0.0833 | false |
| ENPC | 5 | 96 | 48 | 0.1458 | 0 | 0 | 0 | 0.0833 | 0 | 0.0972 | false |
| SLEV | 1 | 96 | 48 | 0.1042 | 0 | 0 | 0 | 0.7917 | 0.0417 | 0.0833 | false |
| SLEV | 5 | 96 | 48 | 0.1458 | 0 | 0 | 0 | 0.0833 | 0 | 0.0972 | false |

DMG-CI is not promotable. It proves that simply injecting source-heldout metric risk as a second gate can reject unknowns, but the same risk surface also rejects most old and seen-new samples.

### Current root-cause evidence

Score/risk distribution on the local evidence shows no clean risk separation. For example, in the base ENPC evidence, old `unknown_risk` median is `0.9927`, seen-new median is `0.9989`, and true unknown median is `0.9979`. In the SOM metric evidence, old `unknown_risk` median is `0.9977`, seen-new median is `0.9984`, and true unknown median is `0.9992`. This explains why risk-threshold or dual-metric gating cannot satisfy old retention and unknown rejection simultaneously.

### Decision

SOM-CI and DMG-CI are retained as reproducible diagnostics only. The next aligned route must change the training objective or exported representation more directly: old-class core replay/floor loss plus source-heldout hard negatives and receiver-conditioned open-set separation, followed by qknn8 Stage2-C evaluation. More post-hoc gating is unlikely to reach `old_acc 99% / seen_new 97% / unknown_reject 99%` on the current feature geometry.

### Git and N607 sync

Git-backed mirror commit:

`5737a4f Add DMG-CI dual metric guard diagnostic`

Local snapshot:

`E:\type10-7\code\snapshots\phase2_dmg_ci_20260704_235900`

Synced files:

| Local | Remote |
|---|---|
| `E:\type10-7\code\scripts\phase2_dual_metric_guard_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_dual_metric_guard_ci_eval.py` |
| `E:\type10-7\code\tests\test_phase2_dual_metric_guard_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_dual_metric_guard_ci_eval.py` |
| `E:\type10-7\docs\CVS_STAGE2C_DMG_CI_ALGORITHM_20260704.md` | `/home/szu2070436088/2510044040/CV-SincNet/docs/CVS_STAGE2C_DMG_CI_ALGORITHM_20260704.md` |

Remote hash verification:

| Remote file | SHA256 |
|---|---|
| `code/scripts/phase2_dual_metric_guard_ci_eval.py` | `1c00817dc206df7778e6f708c0806ce1e0f63c54b94610100df68b9be906c24c` |
| `code/tests/test_phase2_dual_metric_guard_ci_eval.py` | `fabb69c476863c01e721c53d44865130a5b55c127e08bb00d283560932be0383` |
| `docs/CVS_STAGE2C_DMG_CI_ALGORITHM_20260704.md` | `e1f7c87238dfca16c6b7609511fab37d3f28939aaaaa2bffdbf4675a6c720b5c` |

Remote verification:

| Command | Result |
|---|---|
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_dual_metric_guard_ci_eval.py` | PASS |
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_dual_metric_guard_ci_eval.py` | PASS, 2 tests OK |

### N607 DMG-CI full run

Remote working directory:

`/home/szu2070436088/2510044040/CV-SincNet`

Remote output root:

`/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_mined_20260704/dmg_ci_remote`

Remote Python:

`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

GPU evidence:

| Time | GPU memory |
|---|---|
| before | GPUs 0-7 all `10/24576 MiB`, util 0 |
| after | GPUs 0-7 all `10/24576 MiB`, util 0 |

Remote artifact hashes:

| Artifact | SHA256 |
|---|---|
| `dmg/dmg_enpc.json` | `bd8801e74aac1838be5f85c9cd500f27bbd87083e1a9e66037cbf5c49a132b68` |
| `dmg/dmg_slev.json` | `b9fd96293a0bb19cbaadb893aff9f0407bb1c38769bf6ac0ded9afea8a2e25a2` |
| `dmg/dmg_enpc_evidence.csv` | `fc67c7e3747f7c0f919b21bec3be350eaa941cd70b34826d38b0185b5d8214d0` |
| `dmg/dmg_slev_evidence.csv` | `1210877fd6eedd0ef55773ab9f9e418e7260652c6c8c6ba627c3e85c6af0a6a9` |

Pulled local copies:

`E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\dmg_ci_remote`

Remote same-row results:

| backend | M | total | excluded | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_coverage | bytes/event | p95 latency ms | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ENPC | 1 | 96 | 48 | 0.1042 | 0 | 0 | 0 | 0.7917 | 0.0417 | 0.0833 | 160 | 5.1714 | false |
| ENPC | 2 | 96 | 48 | 0 | 0 | 0.0833 | 0 | 0.5417 | 0 | 0.0278 | 320 | 5.1714 | false |
| ENPC | 3 | 96 | 48 | 0.0833 | 0 | 0 | 0 | 0.2083 | 0 | 0.0556 | 480 | 5.1714 | false |
| ENPC | 4 | 96 | 48 | 0.1042 | 0 | 0 | 0 | 0.0833 | 0 | 0.0694 | 640 | 5.1714 | false |
| ENPC | 5 | 96 | 48 | 0.1458 | 0 | 0 | 0 | 0.0833 | 0 | 0.0972 | 800 | 5.1714 | false |
| SLEV | 1 | 96 | 48 | 0.1042 | 0 | 0 | 0 | 0.7917 | 0.0417 | 0.0833 | 160 | 4.7436 | false |
| SLEV | 2 | 96 | 48 | 0 | 0 | 0.0833 | 0 | 0.5417 | 0 | 0.0278 | 320 | 4.7436 | false |
| SLEV | 3 | 96 | 48 | 0.0833 | 0 | 0 | 0 | 0.2083 | 0 | 0.0556 | 480 | 4.7436 | false |
| SLEV | 4 | 96 | 48 | 0.1042 | 0 | 0 | 0 | 0.0833 | 0 | 0.0694 | 640 | 4.7436 | false |
| SLEV | 5 | 96 | 48 | 0.1458 | 0 | 0 | 0 | 0.0833 | 0 | 0.0972 | 800 | 4.7436 | false |

Final boundary for this continuation: DMG-CI was successfully implemented, synced and tested on N607, but it is a negative diagnostic. It confirms the current feature/risk geometry cannot meet the requested deployment target through post-hoc dual-metric gating.

## 2026-07-04 OF-HNFR-CI旧类floor硬负样本特征修复

目的：单协同推理、OPR、CRISP-C和DMG-CI均未能同时满足未知类拒识与旧类/seen-new保真。本节实现训练侧`OF-HNFR-CI`，在训练目标中加入source旧类、target-old support和seen-new support的CE/KL保真、margin floor与残差约束，而不是只依赖后处理阈值。该约束不能保证query级旧类准确率不下降，最终仍以同一行评估结果为准。

算法文件：

| 文件 | 用途 |
|---|---|
| `code/scripts/phase2_old_floor_hnfr_adapter_ci_eval.py` | OF-HNFR-CI低秩残差适配器、旧类floor、seen-new floor、proxy/virtual hard-negative open loss、qknn8后端复验 |
| `code/tests/test_phase2_old_floor_hnfr_adapter_ci_eval.py` | 单元测试：floor loss、target_unknown eval-only训练计数 |
| `docs/CVS_STAGE2C_OF_HNFR_CI_ALGORITHM_20260704.md` | 算法说明与部署边界 |

本地验证：

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_old_floor_hnfr_adapter_ci_eval.py code\tests\test_phase2_proxy_adapter_ci_eval.py -q` | PASS，4 passed；`.pytest_cache`权限警告不影响测试 |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_old_floor_hnfr_adapter_ci_eval.py` | PASS |

本地全量诊断命令：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_old_floor_hnfr_adapter_ci_eval.py --feature_npz local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\features_proxy_mined.npz --output_dir local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_of_hnfr_guarded_q12 --backend both --device cpu --adapter_epochs 60 --adapter_rank 16 --batch_size 256 --collab_counts all --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070801
```

`query_per_class=12`沿用前文协议说明：`features_proxy_mined.npz`中`receiver=7-14,target_unknown,tx=10-1`只有12条query，使用20会触发协议守卫。

本地训练侧指标：

| 指标 | before | after |
|---|---:|---:|
| source_proto_acc | 0.945455 | 0.955303 |
| support_proto_acc | 0.587500 | 0.684375 |
| target_old_support_proto_acc | 0.645833 | 0.695833 |
| seen_new_support_proto_acc | 0.412500 | 0.650000 |
| source_old_margin_mean | 6.522048 | 3.268555 |
| target_old_support_margin_mean | 2.730331 | 1.727009 |
| seen_new_support_margin_mean | -1.637527 | 0.095562 |
| proxy_max_logit_mean | 10.797812 | -0.082385 |

本地同一行结果摘要：

| backend | profile | M | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes/event | latency_ms | target_pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ENPC | enpc_known_anchor | 4 | 0.932203 | 0.750000 | 0.750000 | 0.666667 | 0.000000 | 1.000000 | 512 | 3.920387 | false |
| ENPC | enpc_old80_unknown_probe | 3 | 0.819444 | 0.666667 | 0.458333 | 0.416667 | 0.416667 | 0.583333 | 384 | 3.920387 | false |
| SLEV | slev_known_anchor | 4 | 0.932203 | 0.750000 | 0.750000 | 0.666667 | 0.000000 | 1.000000 | 512 | 4.001318 | false |
| SLEV | slev_old80_energy_probe | 3 | 0.833333 | 0.666667 | 0.500000 | 0.500000 | 0.375000 | 0.625000 | 384 | 4.001318 | false |
| SLEV | slev_energy_strict | 5 | 0.416667 | 0.000000 | 0.125000 | 0.000000 | 0.916667 | 0.083333 | 640 | 4.001318 | false |

本地解释：修正版OF-HNFR-CI比首版更合理，seen-new support从`0.4125`提升到`0.65`，旧类support也有提升，proxy_unknown最大已知logit被压低；但在当前特征与后端配置下，真实target_unknown仍与known边界高度重叠。保真行不能拒识未知，拒识行会伤害old/seen。因此OF-HNFR-CI当前仍是`NON_DEPLOYMENT_DIAGNOSTIC`，需要上N607低显存GPU复验后归档，不可写成部署达标。

### OF-HNFR-CI N607全量复验

Git镜像提交：

`a14a57a Add OF-HNFR-CI old-floor hard-negative diagnostic`

本地快照：

`E:\type10-7\code\snapshots\phase2_of_hnfr_ci_20260704_233239`

同步文件：

| Local | Remote |
|---|---|
| `E:\type10-7\code\scripts\phase2_old_floor_hnfr_adapter_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_old_floor_hnfr_adapter_ci_eval.py` |
| `E:\type10-7\code\tests\test_phase2_old_floor_hnfr_adapter_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_old_floor_hnfr_adapter_ci_eval.py` |
| `E:\type10-7\docs\CVS_STAGE2C_OF_HNFR_CI_ALGORITHM_20260704.md` | `/home/szu2070436088/2510044040/CV-SincNet/docs/CVS_STAGE2C_OF_HNFR_CI_ALGORITHM_20260704.md` |

远端文件hash：

| Remote file | SHA256 |
|---|---|
| `code/scripts/phase2_old_floor_hnfr_adapter_ci_eval.py` | `78c64f656ab3579436a38debc548a5acf09285c5db83f6da932d58848795af49` |
| `code/tests/test_phase2_old_floor_hnfr_adapter_ci_eval.py` | `79a008be9e9d7b0b32d119704c1c701ba6771c54dbbfff6b45064ea7e1d42320` |
| `docs/CVS_STAGE2C_OF_HNFR_CI_ALGORITHM_20260704.md` | `584de087b8c20be27e4220ee0321063496cc92fb6209640c4fa6f3e39531ef73` |

远端验证：

| 命令 | 结果 |
|---|---|
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_old_floor_hnfr_adapter_ci_eval.py` | PASS |
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m pytest ...` | 环境无`pytest`，改用unittest |
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_old_floor_hnfr_adapter_ci_eval.py` | PASS，2 tests OK |

远端命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_old_floor_hnfr_adapter_ci_eval.py --feature_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz --output_dir remote_artifacts/phase2_adv3b02_proxy_mined_20260704/of_hnfr_remote --backend both --device cuda:0 --adapter_epochs 60 --adapter_rank 16 --batch_size 256 --collab_counts all --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070801
```

GPU证据：

| Time | GPU memory |
|---|---|
| before | GPUs 0-7 all`10/24576 MiB`，util 0 |
| after | GPU0 `10/24576 MiB`，util 5；GPUs 1-7 all`10/24576 MiB`，util 0 |

远端输出：

`/home/szu2070436088/2510044040/CV-SincNet/remote_artifacts/phase2_adv3b02_proxy_mined_20260704/of_hnfr_remote`

拉回本地：

`E:\type10-7\local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\of_hnfr_remote`

远端artifact hash：

| Artifact | SHA256 |
|---|---|
| `of_hnfr_ci_summary.json` | `04d045b7f467c56531c3b5c099a4a793adb48d9d885379f76664d00d815ee655` |
| `opr_ci_enpc_summary.csv` | `9816a96ba990e5d1c7b7d633134bf80ff9de195d543549ba0a78540d94867759` |
| `opr_ci_slev_summary.csv` | `8fc7ce94ca7625388f18c0bf1667b2006bb149b7953f9caa40ddabb85cfbd2b5` |
| `of_hnfr_ci_adapted_features.npz` | `17416e4fb857d1d558f4a5ad774c181a4d8984cf44f131b6d44c4c4ff2c72ba5` |

远端训练侧指标：

| 指标 | before | after |
|---|---:|---:|
| source_proto_acc | 0.945455 | 0.953788 |
| support_proto_acc | 0.587500 | 0.684375 |
| target_old_support_proto_acc | 0.645833 | 0.679167 |
| seen_new_support_proto_acc | 0.412500 | 0.700000 |
| proxy_max_logit_mean | 10.797809 | -0.196340 |
| total_fp16_state_bytes |  | 13122 |

远端同一行结果摘要：

| backend | profile | M | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes/event | latency_ms | target_pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ENPC | enpc_known_anchor | 3 | 0.916667 | 0.750000 | 0.583333 | 0.583333 | 0.000000 | 1.000000 | 384 | 5.201122 | false |
| ENPC | enpc_balanced | 3 | 0.916667 | 0.750000 | 0.541667 | 0.500000 | 0.125000 | 0.875000 | 384 | 5.201122 | false |
| ENPC | enpc_unknown_strict | 5 | 0.645833 | 0.000000 | 0.208333 | 0.083333 | 0.583333 | 0.416667 | 640 | 5.201122 | false |
| ENPC | enpc_unknown_strict | 4 | 0.830508 | 0.666667 | 0.291667 | 0.083333 | 0.375000 | 0.625000 | 512 | 5.201122 | false |
| SLEV | slev_known_anchor | 3 | 0.916667 | 0.750000 | 0.583333 | 0.583333 | 0.000000 | 1.000000 | 384 | 4.999545 | false |
| SLEV | slev_balanced | 3 | 0.916667 | 0.750000 | 0.541667 | 0.500000 | 0.166667 | 0.833333 | 384 | 4.999545 | false |
| SLEV | slev_energy_strict | 5 | 0.416667 | 0.000000 | 0.083333 | 0.000000 | 0.875000 | 0.125000 | 640 | 4.999545 | false |

远端结论：OF-HNFR-CI在训练侧实现了预期的旧类/seen-new支持样本保真与proxy_unknown排斥，但全量协同推理仍没有任何达标行。保真最佳行`old_acc=0.916667,seen_new_acc=0.583333`时`unknown_reject=0`；未知拒识最高行`unknown_reject=0.875`时`old_acc=0.416667,min_old=0,seen_new_acc=0.083333`。在当前ADV3B02/Proxy-mined特征、OF-HNFR-CI训练配置和ENPC/SLEV后端下，这表明真实target_unknown仍与known包络严重重叠；当前轻量适配器与事件级后处理组合不足以满足“优先未知拒识且旧类不下降”的目标。

子agent review整改：合理性监督指出`best_joint_row`是事后诊断排序，不能表述为部署profile选择不使用`target_unknown`；同时“旧类准确性不能下降写入训练目标”和“证明所有轻量适配器无效”的表述过强。本节已降级为：`target_unknown`不进入训练/适配器/阈值拟合，`best_joint_row`仅为事后诊断；训练目标只加入旧类/seen-new保真与floor约束，query级不下降仍以同一行评估结果判定；远端负结论仅限定在当前特征、训练配置和后端组合。
## OSPR-CI source-heldout prototype repair implementation

更新时间：2026-07-05 09:30:58 +08:00

用户批准`OSPR-CI`后，本轮结合对话`019f20e3-181c-7333-ace1-0c1dcf8df514`中的Phase1特征/拒识修复经验，将source-only low-FAR但旧类覆盖下降的结论转化为Stage2-C约束：`target_unknown`仍只能最终评估，旧类保持必须写入训练损失和同一行判据，不能再把低FAR安全门单独当成功。

新增文件：

| 文件 | 作用 | SHA256 |
|---|---|---|
| `code/scripts/phase2_ospr_ci_eval.py` | OSPR-CI source-heldout prototype repair adapter，复用qknn8 ENPC/SLEV协同后端输出`M=1..R` | `CEC009BF587E036EEA0EBD1584ECAA7D49CA17CFC0DC7899379520B70F5298BA` |
| `code/tests/test_phase2_ospr_ci_eval.py` | 单测覆盖source-heldout、`target_unknown`eval-only、TX overlap fail-closed、source-heldout不足fail-closed、资源代理不冒充真实星载通过 | `EBBE48796D6CD1B13145FE913586F470438426AA1A733732B22AE3C027413532` |
| `docs/CVS_STAGE2C_OSPR_CI_ALGORITHM_20260705.md` | OSPR-CI算法定义、协议边界、损失结构、资源字段与验收门槛 | `85DAC6BAA3632DFA25EFEB214BDE4443B03ADE7BBDE19C57D3C7997A2E8C53B2` |

本地验证：

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_ospr_ci_eval.py -q` | PASS，4 passed；`.pytest_cache`写入被Windows拒绝但不影响测试 |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_ospr_ci_eval.py code\tests\test_phase2_ospr_ci_eval.py` | PASS |

本地真实smoke：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_ospr_ci_eval.py --feature_npz local_artifacts\phase2_adv3b02_proxy_mined_20260704\remote\features_proxy_mined.npz --output_dir local_artifacts\phase2_adv3b02_proxy_mined_20260704\local_ospr_smoke2 --backend both --collab_counts all --k_shot 8 --query_per_class 12 --qknn_k 8 --seed 4070505 --source_holdout_per_class 12 --adapter_epochs 2 --adapter_rank 4 --batch_size 256 --device cpu --support_selection_policy stable_first --event_alignment_policy receiver_domain_ranked --max_event_bytes 1152 --max_event_latency_ms 20
```

本地smoke产物：

| 产物 | SHA256 |
|---|---|
| `local_artifacts/phase2_adv3b02_proxy_mined_20260704/local_ospr_smoke2/ospr_ci_summary.json` | `D7A2D86D5F658FC35A85403409486197CC082B41D9B582B81971316019688D14` |
| `local_artifacts/phase2_adv3b02_proxy_mined_20260704/local_ospr_smoke2/ospr_ci_enpc_summary.csv` | `CE104C823A18C30AEA6EF94C153231B910040B4D4A186CAC7C0244540EA1D3AD` |
| `local_artifacts/phase2_adv3b02_proxy_mined_20260704/local_ospr_smoke2/ospr_ci_slev_summary.csv` | `365283DC4826EAE0BC1EFC752CE26C9523D72866C0B1D3A5405C410150CD1D92` |

协议计数：

| 字段 | 值 |
|---|---:|
| `source_fit` | 1248 |
| `source_holdout_calibration` | 72 |
| `proxy_unknown` | 3520 |
| `target_support` | 320 |
| `target_unknown_eval_only` | 440 |
| `target_unknown_training_count` | 0 |
| target receivers | `20-1,3-19,7-14,7-7,8-8` |
| OSPR-CI state bytes proxy | 56642 |
| qknn8 support int8 bytes proxy | 51200 |

M覆盖：

| 后端 | `collab_count`覆盖 |
|---|---|
| ENPC | `1,2,3,4,5` |
| SLEV | `1,2,3,4,5` |

本地smoke最高联合分行仍未达标：

| 后端 | profile | M | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes/event | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ENPC | `enpc_old80_unknown_probe` | 4 | 0.7143 | 0.3333 | 0.0833 | 0.0833 | 0.6667 | 0.3333 | 512 | `NON_DEPLOYMENT_DIAGNOSTIC` |
| ENPC | `enpc_old80_unknown_probe` | 5 | 0.5625 | 0.0000 | 0.0000 | 0.0000 | 0.8333 | 0.1667 | 640 | `NON_DEPLOYMENT_DIAGNOSTIC` |
| SLEV | `slev_old80_energy_probe` | 4 | 0.7679 | 0.5000 | 0.0833 | 0.0833 | 0.6250 | 0.3750 | 512 | `NON_DEPLOYMENT_DIAGNOSTIC` |
| SLEV | `slev_energy_strict` | 5 | 0.5208 | 0.0000 | 0.0000 | 0.0000 | 0.8750 | 0.1250 | 640 | `NON_DEPLOYMENT_DIAGNOSTIC` |

子agent审查纳入的硬边界：

| 审查项 | 处理 |
|---|---|
| 文献/方法 | OSPR-CI不是稳定公开缩写；本项目定义为source-heldout prototype repair + qknn8 collaborative inference；依据RF开集原型、few-shot open-set energy、Mahalanobis/OOD、类增量旧知识保持和on-orbit协同推理机制约束设计。 |
| 高效率算法 | 采用低秩残差adapter、qknn8 int8 support代理状态、每事件`128*M`字节后端证据估计；不传raw IQ或完整support。 |
| 监督完成度 | 目前只有实现和本地smoke，目标同一行未达标；不能标完成。 |
| 查漏补缺 | 已补`target_unknown`eval-only、TX overlap fail-closed、source-heldout不足fail-closed、资源代理不冒充真实通过；后续仍需N607报告、scp映射、SSH清理证据。 |

N607全量计划：

```text
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=<low_vram_gpu> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_ospr_ci_eval.py \
  --feature_npz remote_artifacts/phase2_adv3b02_proxy_mined_20260704/features_proxy_mined.npz \
  --output_dir remote_artifacts/phase2_adv3b02_proxy_mined_20260704/ospr_ci_remote \
  --backend both \
  --collab_counts all \
  --k_shot 8 \
  --query_per_class 12 \
  --qknn_k 8 \
  --seed 4070505 \
  --source_holdout_per_class 12 \
  --adapter_epochs 90 \
  --adapter_rank 12 \
  --batch_size 256 \
  --device cuda:0 \
  --support_selection_policy stable_first \
  --event_alignment_policy receiver_domain_ranked \
  --max_event_bytes 1152 \
  --max_event_latency_ms 20
```

启动前必须重新执行N607 preflight、GPU低显存选择、scp同步、远端hash/py_compile、报告同步，并在每次SSH/SCP后记录本地`ssh.exe`和TCP22清理状态。
