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

- preflight:PASS，`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`，直连`N607`可用，项目根目录与GPU可见。
- sync:PASS，已同步4个运行文件到`/home/szu2070436088/2510044040/CV-SincNet`。
- synced_files:
  - `code/evaluation/collaborative_open_set_qknn_eval.py`，sha256=`fb331fd3ded748e1a8f55cde159fbe7f950e1624d8279320076d5801b9226c68`
  - `code/scripts/phase2_collaborative_open_set_qknn_eval.py`，sha256=`70edfe5ea10a52cd91919542ae565c6866e9e282881cd58fee8eb085bc2ba420`
  - `code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`，sha256=`56832b0cdaac1b707cabd59ae127643002a95ebb212c8a5a1ff0119dfd329fad`
  - `code/scripts/launch_phase2_adv3b02_stage2c_contrast_relief_probe_20260707.sh`，sha256=`e82ab1fe87be02a1319fb5121a96542aa9d5458d9c01852f2fdfb6fb028a486d`
- remote_verification:PASS，远端`py_compile`、`bash -n`和`--dry-run`通过，dry-run展开24个冻结特征诊断组合。
- remote_prelaunch_context:source NORM/HEAD NPZ存在；`/home`剩余约7.6T；同run_id启动前无既有任务。
- remote_command:`cd /home/szu2070436088/2510044040/CV-SincNet && setsid bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_relief_probe_20260707.sh > logs/phase2_adv3b02_stage2c_contrast_relief_probe_20260707/launch_background.out 2>&1 &`
- landed_processes:launcher PID`4143375`，观察到Python子任务PID`4143636`开始执行。
- launch_log:`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_relief_probe_20260707/launch_background.out`
- local_ssh_cleanup:启动命令超时后清理本地残留`ssh.exe` PID`30256`；后续检查无本地`ssh.exe`和无ESTABLISHED TCP22连接。
- status:COMPLETED，summary生成于2026-07-07 13:20 CST。

## Results

summary已拉回本地：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_contrast_relief_probe_20260707\stage2c_contrast_relief_probe_summary.json`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_contrast_relief_probe_20260707\stage2c_contrast_relief_probe_summary.csv`

### Result Table

| variant | profile | K | old_acc | min_old | seen_new | min_seen | unknown_FAR | unknown_reject | known_coverage | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NORM | RELIEF_U095_W050M02_D008_R035 | 5 | 0.0000 | 0.0000 | 0.1143 | 0.0000 | 0.0982 | 0.9018 | 0.1224 | low-FAR下seen-new恢复，但old全拒绝，min_seen仍0 |
| NORM | RELIEF_U095_W050M02_D008_R050 | 5 | 0.0000 | 0.0000 | 0.1143 | 0.0000 | 0.0982 | 0.9018 | 0.1224 | 同上，label scale 0.35/0.50无差异 |
| HEAD | RELIEF_U095_W050M02_D008_R035 | 5 | 0.0000 | 0.0000 | 0.1054 | 0.0000 | 0.1000 | 0.9000 | 0.1214 | FAR刚好0.10，但old全拒绝 |
| HEAD | RELIEF_U095_W050M02_D008_R050 | 5 | 0.0000 | 0.0000 | 0.1054 | 0.0000 | 0.1000 | 0.9000 | 0.1214 | FAR刚好0.10，但old全拒绝 |
| HEAD | RELIEF_U095_W050M02_D008_R035 | 10 | 0.6048 | 0.1429 | 0.1357 | 0.0000 | 0.3286 | 0.6696 | 0.4520 | old与seen-new同时恢复，但FAR过高 |
| HEAD | RELIEF_U095_W050M02_D008_R050 | 10 | 0.6048 | 0.1429 | 0.1357 | 0.0000 | 0.3286 | 0.6696 | 0.4520 | 同上 |
| NORM | RELIEF_U095_W050M02_D008_R035 | 10 | 0.6357 | 0.0857 | 0.1196 | 0.0000 | 0.3321 | 0.6625 | 0.4571 | old较好但FAR过高 |
| HEAD | RELIEF_U095_W050M02_D005_R050 | 10 | 0.6048 | 0.1429 | 0.1500 | 0.0000 | 0.4018 | 0.5964 | 0.4898 | seen-new最高但FAR不可接受 |

### Interpretation

- risk relief相对上一轮有明确正向：低FAR可行行不再全拒绝known，`seen_new_acc`从0提升到0.1143，同时`unknown_FAR=0.0982`。
- 但该可行行`old_acc=0`，说明单一路径的seen-new relief牺牲了旧类域适应，不能作为qKNNV42当前最优路线。
- K=10下old_acc可回到0.6048-0.6357，seen_new_acc也可到0.1196-0.1500，但unknown_FAR升至0.3286-0.4018，远超可接受边界。
- 所有24行`min_seen_new_class_acc=0`，新类最低类坍塌没有解决。下一步必须显式做seen-new类均衡/最低类保护，而不是只做全局risk relief。

### Next Route

建议下一步实现双保护融合：old路径保持原candidate risk guard；seen-new路径只对contrast达标候选做relief，并叠加per-seen-new class minimum exposure/receiver evidence约束，目标是在`unknown_FAR<=0.10`下避免old全拒绝，同时让`min_seen_new_class_acc>0`。
