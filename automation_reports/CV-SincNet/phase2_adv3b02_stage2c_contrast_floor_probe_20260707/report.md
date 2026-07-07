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
- implementation commit:`653c6cc Add Stage2-C contrast floor probe`
- follow-up commit:`f5f85a1 Expose contrast floor config in qKNN results`

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

- preflight:PASS，`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`，直连`N607`可用，项目根目录与GPU可见。
- sync:PASS，已同步4个运行文件到`/home/szu2070436088/2510044040/CV-SincNet`。
- synced_files:
  - `code/evaluation/collaborative_open_set_qknn_eval.py`，sha256=`a4b5b4704d4f944d3ea6bba8abbd595297fc47a94780efc76fe7764f3cb77eba`
  - `code/scripts/phase2_collaborative_open_set_qknn_eval.py`，sha256=`18213f8effe4674956002a75b76323d6010e82b5029bb41fa4b25b4fbeafe3b6`
  - `code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`，sha256=`679f1f477319c7fdb31f043e8c5e7c05649b9db9ad343b52e76edb4430c84b7c`
  - `code/scripts/launch_phase2_adv3b02_stage2c_contrast_floor_probe_20260707.sh`，sha256=`d678ff9ef0146cbd49b331fcf1142ff7e6dbfdba31433bf4abbd847cdf32d441`
- remote_verification:PASS，远端hash一致；远端`py_compile`、`bash -n`和`--dry-run`通过，dry-run展开16个冻结特征诊断组合。
- remote_prelaunch_context:source NORM/HEAD NPZ存在；`/home`剩余约7.6T；同run_id目录由dry-run创建但启动前未发现结果文件。
- remote_command:`cd /home/szu2070436088/2510044040/CV-SincNet && setsid bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_floor_probe_20260707.sh > logs/phase2_adv3b02_stage2c_contrast_floor_probe_20260707/launch_background.out 2>&1 &`
- landed_processes:launcher PID`4155635`，观察到Python子任务PID`4156350`和后续`4157068`执行。
- launch_log:`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_floor_probe_20260707/launch_background.out`
- local_ssh_cleanup:启动命令超时后清理本地残留`ssh.exe` PID`30368`；后续检查无本地`ssh.exe`和无ESTABLISHED TCP22连接。
- status:COMPLETED，summary生成于2026-07-07 13:45 CST。

## Results

summary已拉回本地：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_contrast_floor_probe_20260707\stage2c_contrast_floor_probe_summary.json`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_contrast_floor_probe_20260707\stage2c_contrast_floor_probe_summary.csv`

说明：本run使用`f5f85a1`之前的远端顶层输出，summary中的`seen_new_contrast_risk_relief_min_support_count/min_pvalue/min_receiver_class_reliability`字段显示为0，这是输出记录字段遗漏，不代表命令未传入floor。实际floor以launcher profile和launch log中的`floors=...`为准；该记录字段遗漏已由`f5f85a1`修复并同步至N607，后续run会直接在summary中显示。

### Result Table

| variant | profile group | K | old_acc | min_old | seen_new | min_seen | unknown_FAR | unknown_reject | known_coverage | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HEAD | all floor profiles | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | floor过强，全拒绝known |
| NORM | all floor profiles | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | floor过强，全拒绝known |
| HEAD | all floor profiles | 10 | 0.6048 | 0.1429 | 0.0107 | 0.0000 | 0.2143 | 0.7839 | 0.3173 | old恢复但seen-new退回gate-only，FAR仍高 |
| NORM | all floor profiles | 10 | 0.6357 | 0.0857 | 0.0089 | 0.0000 | 0.2107 | 0.7839 | 0.3276 | old恢复但seen-new退回gate-only，FAR仍高 |

### Interpretation

- support-floor relief是负向诊断：它把上一轮低FAR K=5的`seen_new_acc=0.1143`压回0，同时old仍为0。
- K=10下旧类保持`old_acc=0.6048-0.6357`，但seen-new仅`0.0089-0.0107`，与上一轮contrast gate-only水平相同；unknown FAR约0.21，仍高于`<=0.10`约束。
- 所有16行`min_seen_new_class_acc=0`，最低seen-new类坍塌未解决。
- 结论：单纯support-floor会把relief变成过滤器，不能解决“old保留+seen-new增加+低FAR”的三目标冲突。下一步应转向class-balanced quota/最低类曝光补偿或unknown shell veto，而不是继续提高support floor。
