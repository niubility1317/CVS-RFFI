# qKNNV42 Stage2-C Contrast Delta Gate Probe

- experiment_id:phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707
- timestamp:2026-07-07
- operator:Codex
- objective:在上一轮contrast提升seen-new但FAR失控、candidate gate压FAR但seen-new归零的基础上，增加event/class evidence级`seen_new_old_contrast_delta`，并用默认关闭的seen-new contrast gate验证是否可在释放unknown gate后保留seen-new注册性能同时压低unknown FAR。
- protocol:CVS Stage2-C，K=5/K=10目标域old+seen-new support，target receiver为LEO叠加信道，target unknown只做evaluation，不参与训练、校准、阈值选择或模型选择。
- verdict_scope:NON_DEPLOYMENT_DIAGNOSTIC。不得写作部署成功或Stage2成功证据。

## Local Version State

- 根目录`E:\type10-7`不是Git仓库；本次代码变更先进入Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`，并同步同名文件到根目录运行面。
- Git branch:`codex/cvs-rffi-release-20260626`
- pre-change relevant commits:`973078c`、`19b5846`、`ddc5d7f`、`eb45e41`
- untracked unrelated local artifacts remain untouched:`local_artifacts/phase2_adv3b02_proxy_mined_20260704/`、`local_artifacts/phase2_adv3b02_smec_ci_20260704/`

| changed file | purpose |
| --- | --- |
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | 写出event级与class evidence top-M级`seen_new_old_contrast_*`诊断字段；主CLI透传contrast gate参数 |
| `code/evaluation/collaborative_open_set_qknn_eval.py` | 增加默认关闭的`seen_new_contrast_gate_*`融合门控，接入candidate/scg/support/selective/orbit接收路径，并写出审计字段 |
| `code/scripts/phase2_frozen_manytx_unknown_diagnostic.py` | ManyTx诊断wrapper透传`seen_new_contrast_gate_*`参数 |
| `code/scripts/launch_phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707.sh` | 新增N607冻结特征诊断启动器，覆盖NORM/HEAD、K=5/10和6个contrast-delta gate组合 |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | 验证low contrast seen-new在gate开启时被阻断，低阈值时可接收 |
| `code/tests/test_phase2_collaborative_open_set_qknn_eval.py` | 验证evidence行写出contrast delta与class evidence top-M诊断字段 |
| `code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py` | 验证ManyTx wrapper接受contrast gate CLI参数 |

## Local Verification

| command | workspace | result |
| --- | --- | --- |
| `python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\scripts\phase2_frozen_manytx_unknown_diagnostic.py code\evaluation\collaborative_open_set_qknn_eval.py` | Git承载面 | PASS |
| `python -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py` | Git承载面 | PASS，132 passed |
| `python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\scripts\phase2_frozen_manytx_unknown_diagnostic.py code\evaluation\collaborative_open_set_qknn_eval.py` | 根目录运行面 | PASS |
| `python -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py` | 根目录运行面 | PASS，132 passed；仅`.pytest_cache`写入权限warning |
| `bash -n code/scripts/launch_phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707.sh` | 根目录运行面 | PASS |
| `bash -lc 'ROOT=/tmp/cvs-rffi-dryrun PYTHON=python bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707.sh --dry-run'` | 根目录运行面 | PASS，展开24个冻结特征诊断组合 |

## N607 Plan

| field | value |
| --- | --- |
| source_run_id | `phase2_adv3b02_stage2c_normsep_protocol_20260707` |
| source feature path | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_normsep_protocol_20260707/PHASE2_STAGE2C_RX7_14/<variant>/features_stage2c_leo_repaired.npz` |
| run_id | `phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707` |
| variants | `STAGE2C_NORM_SEP`、`STAGE2C_HEAD_SEP` |
| K | `5`、`10` |
| profiles | `DELTA_U085_W025_D002`、`DELTA_U090_W025_D002`、`DELTA_U090_W025_D005`、`DELTA_U095_W025_D005`、`DELTA_U095_W050M02_D005`、`DELTA_U095_W050M02_D008` |
| expected jobs | 24冻结特征诊断组合，不训练 |
| output_root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707` |
| log_root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707` |
| summary_json | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707/stage2c_contrast_delta_gate_probe_summary.json` |

Remote command after sync:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707.sh > logs/phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707/launch_background.out 2>&1 &
```

## Success Criteria

- Primary:在`unknown_FAR<=0.10`约束下，找到`seen_new_acc>0`且`min_seen_new_class_acc>0`的同一行结果，并保持old_acc/min_old不低于上一轮candidate gate可比行。
- Secondary:若FAR仍高，使用同一行event gate审计字段判断是否contrast delta不足、unknown gate仍过宽，或candidate risk仍主导seen-new失败。
- Failure mode:所有低FAR行仍`seen_new_acc=0`，则contrast delta gate只是过滤器，下一步需要将positive contrast作为seen-new专用risk relief或receiver-level class reliability修正，而不是继续收紧gate。

## Launch Record

- preflight:pending
- sync:pending
- remote_command:pending
- pid:pending
- status:pending

## Results

pending
