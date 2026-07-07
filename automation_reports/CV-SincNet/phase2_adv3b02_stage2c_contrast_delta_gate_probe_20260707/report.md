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

- preflight:PASS，`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`，直连`N607`可用，项目根目录与GPU可见。
- sync:PASS，已同步4个运行文件到`/home/szu2070436088/2510044040/CV-SincNet`。
- synced_files:
  - `code/evaluation/collaborative_open_set_qknn_eval.py`，sha256=`ff455f81d4c5cd8fc6f91727542ec6bf9a88b7004939b6910381b2c7171e0f68`
  - `code/scripts/phase2_collaborative_open_set_qknn_eval.py`，sha256=`982ba014cb6a25b23caecd8a079378a99a28be03274634ad1f98c15cb942e254`
  - `code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`，sha256=`7fb2513ce38a3a22f53bb6c17878c3661bde5b5e3020b66b993c93507845e869`
  - `code/scripts/launch_phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707.sh`，sha256=`020c2bc50427950f7248d6378b97811c9a60e6fe9a46aa144b57c6b463945024`
- remote_verification:PASS，远端`py_compile`、`bash -n`和`--dry-run`通过，dry-run展开24个冻结特征诊断组合。
- remote_prelaunch_context:source NORM/HEAD NPZ存在；`/home`剩余约7.6T；同run_id启动前无既有进程。
- remote_command:`cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707.sh > logs/phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707/launch_background.out 2>&1 &`
- landed_processes:launcher PID`4133484`，观察到Python子任务PID`4134196`正在执行`DELTA_U095_W025_D005_k5`。
- launch_log:`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707/launch_background.out`
- local_ssh_cleanup:启动命令超时后清理本地残留`ssh.exe` PID`8280`；后续检查无本地`ssh.exe`和无ESTABLISHED TCP22连接。
- status:RUNNING

## Results

completed:2026-07-07 13:04 CST。summary已拉回本地：

- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707\stage2c_contrast_delta_gate_probe_summary.json`
- `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_contrast_delta_gate_probe_20260707\stage2c_contrast_delta_gate_probe_summary.csv`

### Result Table

| variant | profile | K | old_acc | min_old | seen_new | min_seen | unknown_FAR | unknown_reject | known_coverage | defer | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HEAD | DELTA_U085_W025_D002 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | low-FAR但全拒绝known |
| HEAD | DELTA_U090_W025_D002 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | low-FAR但全拒绝known |
| HEAD | DELTA_U090_W025_D005 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | low-FAR但全拒绝known |
| HEAD | DELTA_U095_W025_D005 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | low-FAR但全拒绝known |
| HEAD | DELTA_U095_W050M02_D005 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | low-FAR但全拒绝known |
| HEAD | DELTA_U095_W050M02_D008 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | low-FAR但全拒绝known |
| NORM | DELTA_U085_W025_D002 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | low-FAR但全拒绝known |
| NORM | DELTA_U090_W025_D002 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | low-FAR但全拒绝known |
| NORM | DELTA_U090_W025_D005 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | low-FAR但全拒绝known |
| NORM | DELTA_U095_W025_D005 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | low-FAR但全拒绝known |
| HEAD | DELTA_U085_W025_D002 | 10 | 0.5286 | 0.0000 | 0.0018 | 0.0000 | 0.1286 | 0.8714 | 0.2633 | 0.0000 | seen-new非零但FAR超0.10且最低类仍0 |
| HEAD | DELTA_U090_W025_D002 | 10 | 0.5548 | 0.0000 | 0.0018 | 0.0000 | 0.1554 | 0.8446 | 0.2816 | 0.0000 | seen-new非零但FAR升高 |
| NORM | DELTA_U095_W025_D005 | 10 | 0.6357 | 0.0857 | 0.0089 | 0.0000 | 0.2107 | 0.7839 | 0.3276 | 0.0032 | old略恢复但FAR高，min_seen仍0 |
| HEAD | DELTA_U095_W050M02_D005 | 10 | 0.6048 | 0.1429 | 0.0107 | 0.0000 | 0.2143 | 0.7839 | 0.3173 | 0.0013 | 本轮seen-new最高但FAR高，min_seen仍0 |

### Interpretation

- `seen_new_contrast_gate`单独作为接收过滤器不是可推广路线：当`unknown_FAR<=0.10`时，全部可行行都是`known_coverage=0`，没有旧类或新类接收。
- K=10比K=5有少量恢复，但同一行最高`seen_new_acc`只有0.0107，`min_seen_new_class_acc=0`，不能解决新类增多下最低类坍塌。
- 负向证据说明问题不是“缺少一个更紧的gate”，而是candidate risk仍把seen-new当unknown压死；contrast证据需要作为seen-new专用risk relief或校准项进入接收风险，而不是只作为过滤器。

### Next Route

下一步实现默认关闭的`seen_new_contrast_risk_relief_*`：仅当输出label属于seen-new且contrast delta/receiver-count达标时，降低candidate-set接收使用的label/event unknown risk和component agreement；同时保留contrast gate用于unknown FAR控制。该路线是诊断扩展，不改变`项目.md`协议，不使用target unknown训练或校准。
