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
