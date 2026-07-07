# qKNNV42 Stage2-C Contrast Floor Probe

- experiment_id:phase2_adv3b02_stage2c_contrast_floor_probe_20260707
- timestamp:2026-07-07
- operator:Codex
- objective:在contrast risk relief能恢复seen-new但低FAR行old全拒绝、K=10行FAR过高的基础上，引入seen-new class support floor，验证能否降低unknown->seen-new误接收，同时保留K=10旧类域适应收益，并观察最低seen-new类是否脱离0。
- protocol:CVS Stage2-C，K=5/K=10目标域old+seen-new support，target receiver为LEO叠加信道，target unknown只做evaluation，不参与训练、校准、阈值选择或模型选择。
- verdict_scope:NON_DEPLOYMENT_DIAGNOSTIC。不得写作部署成功或Stage2成功证据。

## Motivation

上一轮`phase2_adv3b02_stage2c_contrast_relief_probe_20260707`显示：

| route | K | old_acc | seen_new | min_seen | unknown_FAR | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| low-FAR NORM relief | 5 | 0.0000 | 0.1143 | 0.0000 | 0.0982 | seen-new恢复但old全拒绝 |
| HEAD relief | 10 | 0.6048 | 0.1357 | 0.0000 | 0.3286 | old与seen-new同时恢复但FAR过高 |
| NORM relief | 10 | 0.6357 | 0.1196 | 0.0000 | 0.3321 | old较好但FAR过高 |

结论是contrast不能只作为全局relief开关。新probe将relief限制为support count、conformal pvalue和receiver-class reliability达标的seen-new标签，目标是减少弱类/弱support导致的unknown误接收。

## Local Version State

- 根目录`E:\type10-7`不是Git仓库；代码变更进入Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`，并同步到根目录运行面后再发往N607。
- Git branch:`codex/cvs-rffi-release-20260626`
- base result commit:`1e679f4 Record Stage2-C contrast relief results`

| changed file | purpose |
| --- | --- |
| `code/evaluation/collaborative_open_set_qknn_eval.py` | 为默认关闭的`seen_new_contrast_risk_relief_*`增加support count、pvalue和receiver-class reliability下限；下限不达标时不应用risk relief，并写入事件审计 |
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | 主qKNN CLI透传新增floor参数 |
| `code/scripts/phase2_frozen_manytx_unknown_diagnostic.py` | ManyTx诊断wrapper透传新增floor参数，并补齐`candidate_set_min_label_receiver_class_reliability` |
| `code/scripts/launch_phase2_adv3b02_stage2c_contrast_floor_probe_20260707.sh` | 新增N607冻结特征诊断启动器，覆盖NORM/HEAD、K=5/10和4个support-floor profile |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | 增加RED/GREEN测试：弱support seen-new不得享受contrast risk relief |
| `code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py` | 验证ManyTx wrapper接受新增floor CLI参数 |

## Local Verification

| command | workspace | result |
| --- | --- | --- |
| `python -m pytest code\tests\test_collaborative_open_set_qknn_eval.py -k "risk_relief_requires_class_support_floor"` | Git承载面 | RED后GREEN，最终PASS |
| `python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\scripts\phase2_frozen_manytx_unknown_diagnostic.py` | Git承载面 | PASS |
| `bash -n code/scripts/launch_phase2_adv3b02_stage2c_contrast_floor_probe_20260707.sh` | Git承载面 | PASS |
| `bash -lc 'ROOT=/tmp/cvs-rffi-floor-dryrun PYTHON=python bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_floor_probe_20260707.sh --dry-run'` | Git承载面 | PASS，展开16个冻结特征诊断组合 |
| `python -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py` | Git承载面 | PASS，134 passed |

## N607 Plan

| field | value |
| --- | --- |
| source_run_id | `phase2_adv3b02_stage2c_normsep_protocol_20260707` |
| run_id | `phase2_adv3b02_stage2c_contrast_floor_probe_20260707` |
| variants | `STAGE2C_NORM_SEP`、`STAGE2C_HEAD_SEP` |
| K | `5`、`10` |
| profiles | `FLOOR_U095_W050M02_D008_R035_S2P060Q060`、`FLOOR_U095_W050M02_D008_R050_S2P060Q060`、`FLOOR_U095_W050M02_D008_R035_S3P070Q070`、`FLOOR_U095_W050M02_D005_R050_S3P070Q070` |
| expected jobs | 16冻结特征诊断组合，不训练 |
| output_root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_contrast_floor_probe_20260707` |
| log_root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_floor_probe_20260707` |
| summary_json | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_floor_probe_20260707/stage2c_contrast_floor_probe_summary.json` |

Remote command after sync:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && setsid bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_floor_probe_20260707.sh > logs/phase2_adv3b02_stage2c_contrast_floor_probe_20260707/launch_background.out 2>&1 &
```

## Success Criteria

- Primary:K=10行在`old_acc>=0.60`附近时把`unknown_FAR`从0.3286-0.4018显著压低，最好进入`unknown_FAR<=0.10`。
- Secondary:`seen_new_acc`保持高于上一轮gate-only最高值0.0107，且`min_seen_new_class_acc>0`。
- Failure mode:若FAR降低但seen-new/min_seen回到0，则support floor过强；若FAR仍高，则需要class-balanced quota或unknown shell veto，而不是继续调单个relief scale。

## Launch Record

- preflight:pending
- sync:pending
- remote_verification:pending
- remote_command:pending
- pid:pending
- status:pending

## Results

pending
