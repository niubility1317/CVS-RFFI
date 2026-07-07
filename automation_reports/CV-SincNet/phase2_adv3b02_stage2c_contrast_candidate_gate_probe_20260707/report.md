# qKNNV42 Stage2-C contrast candidate gate probe

## 基本信息

- experiment_id:phase2_adv3b02_stage2c_contrast_candidate_gate_probe_20260707
- timestamp:2026-07-07
- operator:Codex
- status:local_verified,pending_n607_sync_launch
- diagnostic_only:true
- deployment_success_claim:false
- stage2_success_claim:false

## 目标和假设

上一轮`seen_new_old_contrast`能恢复seen-new注册，但unknown_FAR最高接近0.97，说明无门控接受会把unknown误收为known。本轮不改变CVS协议，不改target unknown eval-only边界，只把contrast注册分数接入已有`candidate_set_cvs`低FAR候选门控，扫`candidate_set_unknown_reject_risk`为0.50/0.60/0.70，观察是否存在非全拒绝且FAR显著下降的同一行结果。

## 本地变更

| file | purpose |
|---|---|
| `code/scripts/launch_phase2_adv3b02_stage2c_contrast_candidate_gate_probe_20260707.sh` | 新增N607冻结特征诊断启动器，复用contrast打分和`candidate_set_cvs`门控 |

## 计划验证

| command | expected |
|---|---|
| `bash -n code/scripts/launch_phase2_adv3b02_stage2c_contrast_candidate_gate_probe_20260707.sh` | PASS |
| `ROOT=/tmp/cvs-rffi-dryrun PYTHON=python bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_candidate_gate_probe_20260707.sh --dry-run` | PASS,展开16个诊断组合 |

## N607计划

| item | value |
|---|---|
| remote_root | `/home/szu2070436088/2510044040/CV-SincNet` |
| source_run_id | `phase2_adv3b02_stage2c_normsep_protocol_20260707` |
| case_id | `PHASE2_STAGE2C_RX7_14` |
| run_id | `phase2_adv3b02_stage2c_contrast_candidate_gate_probe_20260707` |
| variants | `STAGE2C_NORM_SEP`,`STAGE2C_HEAD_SEP` |
| K-shot | `5`,`10` |
| profiles | `CAND_U050_W025`,`CAND_U060_W025`,`CAND_U070_W025`,`CAND_U070_W050M02` |
| output_root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_contrast_candidate_gate_probe_20260707` |
| log_root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_candidate_gate_probe_20260707` |

Planned remote command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_candidate_gate_probe_20260707.sh
```

## 判据

| criterion | interpretation |
|---|---|
| `unknown_FAR<=0.05`且known不全为0 | 可进入更细门控扫参 |
| FAR下降但seen_new坍塌 | 说明candidate_set可控但门控过硬，需要contrast-aware分布门控 |
| FAR仍高 | 说明现有risk字段不足以区分unknown误收，需增加事件级contrast margin证据 |

## N607完成状态

- launch_pid:4122465
- remote_status:completed
- remote_json_count:16/16
- remote_command:`cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_candidate_gate_probe_20260707.sh > logs/phase2_adv3b02_stage2c_contrast_candidate_gate_probe_20260707/launch_background.out 2>&1 &`
- summary_json:`E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_contrast_candidate_gate_probe_20260707\stage2c_contrast_candidate_gate_probe_summary.json`
- summary_csv:`E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_contrast_candidate_gate_probe_20260707\stage2c_contrast_candidate_gate_probe_summary.csv`
- ssh_cleanup:checked,no local`ssh.exe`,no ESTABLISHED TCP22 after sync,launch,monitor,pull

## 结果表

| variant | profile | K | old_acc | min_old | seen_new | min_seen_new | unknown_FAR | known_coverage | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HEAD | CAND_U050_W025 | 10 | 0.2976 | 0.0000 | 0.0000 | 0.0000 | 0.0375 | 0.1459 | FAR可行但seen-new全拒绝 |
| NORM | CAND_U070_W025 | 10 | 0.4548 | 0.0000 | 0.0000 | 0.0000 | 0.0786 | 0.2214 | old较高但FAR不可行,seen-new全拒绝 |
| NORM | CAND_U070_W050M02 | 10 | 0.4548 | 0.0000 | 0.0000 | 0.0000 | 0.0786 | 0.2214 | old较高但FAR不可行,seen-new全拒绝 |
| NORM | CAND_U060_W025 | 10 | 0.3905 | 0.0000 | 0.0000 | 0.0000 | 0.0607 | 0.1898 | 接近FAR门槛但seen-new全拒绝 |
| HEAD | CAND_U070_W025 | 10 | 0.4024 | 0.0000 | 0.0000 | 0.0000 | 0.0661 | 0.1980 | FAR不可行,seen-new全拒绝 |
| HEAD | CAND_U070_W050M02 | 10 | 0.4024 | 0.0000 | 0.0000 | 0.0000 | 0.0661 | 0.1980 | FAR不可行,seen-new全拒绝 |
| HEAD | CAND_U060_W025 | 10 | 0.3429 | 0.0000 | 0.0000 | 0.0000 | 0.0571 | 0.1694 | 接近FAR门槛但seen-new全拒绝 |
| NORM | CAND_U050_W025 | 10 | 0.3286 | 0.0000 | 0.0000 | 0.0000 | 0.0571 | 0.1582 | 接近FAR门槛但seen-new全拒绝 |
| all K5 rows | candidate_set profiles | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 全拒绝 |

## 解释

- `candidate_set_cvs`确实能把FAR压下来:HEAD K10`CAND_U050_W025`达到unknown_FAR=0.0375。
- 该低FAR路径把seen-new完全拒绝，seen_new_acc和min_seen_new_class_acc均为0，不满足本轮优化目标。
- K5全部全拒绝，说明现有risk字段在少shot下过硬。
- 结论:旧candidate risk门控只能证明“可控FAR”，不能证明“低FAR+新类注册”。下一步必须把`seen_new_old_contrast_delta`作为事件级证据输出，并在seen-new gate中要求contrast margin，而不是仅靠unknown risk。
