# qKNNV42 Stage2-C Contrast Risk Relief Probe

- experiment_id:phase2_adv3b02_stage2c_contrast_relief_probe_20260707
- timestamp:2026-07-07
- operator:Codex
- objective:在contrast delta gate诊断显示“低FAR行全拒绝known、非零seen-new行FAR仍高”的基础上，验证seen-new contrast证据作为candidate-set风险缓释项是否能恢复seen-new接收，同时维持unknown FAR边界。
- protocol:CVS Stage2-C，K=5/K=10目标域old+seen-new support，target receiver为LEO叠加信道，target unknown只做evaluation，不参与训练、校准、阈值选择或模型选择。
- verdict_scope:NON_DEPLOYMENT_DIAGNOSTIC。不得写作部署成功或Stage2成功证据。

## Motivation

上一轮`phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707`结论为：contrast gate单独作为过滤器不是可推广路线。`unknown_FAR<=0.10`时所有行`known_coverage=0`；K=10的非零seen-new行最高`seen_new_acc=0.0107`且`min_seen_new_class_acc=0`，FAR约0.2143。由此判断candidate risk仍把seen-new压成unknown，下一步应将positive contrast作为seen-new专用risk relief，而不是继续收紧gate。

## Local Version State

- 根目录`E:\type10-7`不是Git仓库；代码变更进入Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`，并同步到根目录运行面。
- Git branch:`codex/cvs-rffi-release-20260626`
- implementation commit:`1f4053d Add Stage2-C seen-new contrast risk relief`
- previous result commits:`fa8ef06`、`0af6869`、`295f1ea`

| changed file | purpose |
| --- | --- |
| `code/evaluation/collaborative_open_set_qknn_eval.py` | 增加默认关闭的`seen_new_contrast_risk_relief_*`，仅对seen-new候选用contrast达标证据缩放candidate-set接收/拒绝风险，并保留raw风险审计字段 |
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | 主qKNN CLI透传risk relief参数 |
| `code/scripts/phase2_frozen_manytx_unknown_diagnostic.py` | ManyTx诊断wrapper透传risk relief参数 |
| `code/scripts/launch_phase2_adv3b02_stage2c_contrast_relief_probe_20260707.sh` | 新增N607冻结特征诊断启动器，覆盖NORM/HEAD、K=5/10和6个risk relief组合 |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | 验证contrast risk relief可在contrast达标时把高raw-risk seen-new从reject救回accept |
| `code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py` | 验证ManyTx wrapper接受risk relief CLI参数 |

## Local Verification

| command | workspace | result |
| --- | --- | --- |
| `python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\scripts\phase2_frozen_manytx_unknown_diagnostic.py` | Git承载面 | PASS |
| `python -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py` | Git承载面 | PASS，133 passed |
| `python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\scripts\phase2_frozen_manytx_unknown_diagnostic.py` | 根目录运行面 | PASS |
| `python -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py` | 根目录运行面 | PASS，133 passed；仅`.pytest_cache`写入权限warning |
| `bash -n code/scripts/launch_phase2_adv3b02_stage2c_contrast_relief_probe_20260707.sh` | 根目录运行面 | PASS |
| `bash -lc 'ROOT=/tmp/cvs-rffi-dryrun PYTHON=python bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_relief_probe_20260707.sh --dry-run'` | 根目录运行面 | PASS，展开24个冻结特征诊断组合 |

## N607 Plan

| field | value |
| --- | --- |
| source_run_id | `phase2_adv3b02_stage2c_normsep_protocol_20260707` |
| run_id | `phase2_adv3b02_stage2c_contrast_relief_probe_20260707` |
| variants | `STAGE2C_NORM_SEP`、`STAGE2C_HEAD_SEP` |
| K | `5`、`10` |
| profiles | `RELIEF_U090_W025_D002_R050`、`RELIEF_U090_W025_D005_R050`、`RELIEF_U095_W025_D005_R050`、`RELIEF_U095_W050M02_D005_R050`、`RELIEF_U095_W050M02_D008_R050`、`RELIEF_U095_W050M02_D008_R035` |
| expected jobs | 24冻结特征诊断组合，不训练 |
| output_root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_contrast_relief_probe_20260707` |
| log_root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_relief_probe_20260707` |
| summary_json | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_relief_probe_20260707/stage2c_contrast_relief_probe_summary.json` |

Remote command after sync:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_relief_probe_20260707.sh > logs/phase2_adv3b02_stage2c_contrast_relief_probe_20260707/launch_background.out 2>&1 &
```

## Success Criteria

- Primary:出现同一行`seen_new_acc`显著高于0.0107、`min_seen_new_class_acc>0`且`unknown_FAR<=0.10`。
- Secondary:若`unknown_FAR`略高但`min_seen_new_class_acc>0`，保留为下一步更紧contrast delta/receiver-count的候选。
- Failure mode:若risk relief只提高old/coverage但seen-new最低类仍0，下一步需从class-level imbalance或per-new-TX receiver reliability入手。

## Launch Record

- preflight:pending
- sync:pending
- remote_command:pending
- pid:pending
- status:pending

## Results

pending
