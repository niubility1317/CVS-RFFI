# Phase A ADV3B02 qknn8 collaborative open-set evaluator N607 report

## Run metadata

| Field | Value |
|---|---|
| run_id | `phaseA_adv3b02_collab_open_set_qknn_eval_20260703` |
| timestamp | 2026-07-03 |
| operator | Codex |
| objective | Sync and verify the hardened collaborative open-set qknn evidence evaluator on N607, with collaboration count selectable from 1 to all observed receivers. |
| scope boundary | Phase A remote evaluator verification only; not a full ADV3B02/qknn8 Stage2-C performance claim. |

## Hypothesis and comparison target

The hardened evaluator should fail closed on protocol-unsafe evidence, record threshold and denominator boundaries, and report old/seen-new/unknown metrics plus latency and communication telemetry without breaking the existing closed-set collaborative evaluator.

Comparison target:

- Existing `code/evaluation/collaborative_inference_eval.py` closed-set collaborative receiver fusion tests remain passing.
- New `code/evaluation/collaborative_open_set_qknn_eval.py` tests pass locally and remotely.

## Local files changed

| File | Purpose | SHA256 |
|---|---|---|
| `code/evaluation/collaborative_open_set_qknn_eval.py` | Hardened offline evidence-level collaborative open-set qknn evaluator. | `8926013E39AD9707037E702DEBD7E3CAD4E65AD09BCA19EECADFC5334C0D2898` |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | Tests for 1..N receiver counts, protocol fail-closed behavior, threshold source, high-risk defer, and protocol metadata. | `9A4FFEDB859145D722A16D601EDBF1779A44158F3BD8F2645F3BD2E7372B33D4` |
| `analysis/adv3b02_qknn8_collab_open_set_traceability_20260703.md` | Requirement traceability and review fix record. | `01C600844E5D29F286C7B1E80E3C14CF46768B46AA90291263D1A787C9EDA7F5` |
| `analysis/adv3b02_qknn8_satellite_collab_open_set_design_20260703.md` | Algorithm design note and literature-aligned boundary. | `6900AE049AD20458A68B7743D75ECBADAC592B1A41FF1C525DA8CBCDECF5CF14` |

## Version state

Root workspace `E:\type10-7` is not a Git repository. Git-backed mirror:

- Repository: `E:\type10-7\github_publish\CVS-RFFI-repo`
- Branch: `codex/cvs-rffi-release-20260626`
- Commits:
  - `96bc3a5 Add collaborative open-set qknn evaluator`
  - `87632c4 Harden collaborative open-set qknn evaluation`
- Status after commit: branch ahead of origin by 195 commits; no uncommitted mirror changes for this task.

Local snapshot before N607 sync:

`E:\type10-7\code\snapshots\phaseA_adv3b02_collab_open_set_qknn_eval_20260703\`

## Local verification

Commands were run with `ssr-gpu` through `conda run` because this shell is non-interactive.

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; $env:CONDA_REPORT_ERRORS='false'
conda run -n ssr-gpu python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py
conda run -n ssr-gpu python -m pytest -q code\tests\test_collaborative_open_set_qknn_eval.py
conda run -n ssr-gpu python -m pytest -q code\tests\test_collaborative_inference_eval.py
```

Results:

| Check | Result |
|---|---|
| open-set evaluator `py_compile` | passed |
| open-set evaluator tests | `7 passed, 1 warning` |
| closed-set collaborative evaluator tests | `5 passed, 1 warning` |

Windows note: parallel `conda run` commands can fail with temporary file lock errors; verification was rerun serially and passed.

## N607 preflight and GPU context

Preflight:

```powershell
powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1
```

Result:

- Direct `N607` target OK.
- Host: `dell-DSS8440`
- User: `szu2070436088`
- Project root: `/home/szu2070436088/2510044040/CV-SincNet`
- Initial GPU memory: GPUs 4 and 5 had the lowest memory among active processes at 647 MiB.

Post-review occupancy probe before launch:

- GPU memory readout: all GPUs reported approximately 10 MiB used.
- `nvidia-smi --query-compute-apps` returned no active compute apps.
- A PowerShell quoting issue caused the remote `date '+%F %T'` field to be empty in one probe; the GPU evidence itself was returned by `nvidia-smi`.

Selected GPU for the bounded remote verification job: `CUDA_VISIBLE_DEVICES=4`.

## Sync plan

Remote root: `/home/szu2070436088/2510044040/CV-SincNet`

| Local path | Remote path |
|---|---|
| `E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/collaborative_open_set_qknn_eval.py` |
| `E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_collaborative_open_set_qknn_eval.py` |
| `E:\type10-7\analysis\adv3b02_qknn8_collab_open_set_traceability_20260703.md` | `/home/szu2070436088/2510044040/CV-SincNet/analysis/adv3b02_qknn8_collab_open_set_traceability_20260703.md` |
| `E:\type10-7\analysis\adv3b02_qknn8_satellite_collab_open_set_design_20260703.md` | `/home/szu2070436088/2510044040/CV-SincNet/analysis/adv3b02_qknn8_satellite_collab_open_set_design_20260703.md` |
| this report | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phaseA_adv3b02_collab_open_set_qknn_eval_20260703/report.md` |

## Remote command plan

Environment:

- Python: `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- Working directory: `/home/szu2070436088/2510044040/CV-SincNet`
- GPU allocation: `CUDA_VISIBLE_DEVICES=4`

Bounded verification command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
RUN_ID=phaseA_adv3b02_collab_open_set_qknn_eval_20260703
mkdir -p logs/$RUN_ID runs/$RUN_ID automation_reports/CV-SincNet/$RUN_ID
nohup env PYTHONPATH=$PWD/code:$PWD CUDA_VISIBLE_DEVICES=4 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m pytest -q code/tests/test_collaborative_open_set_qknn_eval.py code/tests/test_collaborative_inference_eval.py > logs/$RUN_ID/remote_eval_smoke.out 2>&1 &
echo $!
```

Expected outputs:

- `logs/phaseA_adv3b02_collab_open_set_qknn_eval_20260703/remote_eval_smoke.out`
- Optional pytest cache only; no dataset, checkpoint, or training output should be modified.

## Metrics to watch

| Metric | Expected |
|---|---|
| remote open-set evaluator tests | pass |
| remote closed-set collaborative tests | pass |
| GPU memory | low; this is a bounded test and should not allocate training-scale VRAM |
| side effects | no dataset/checkpoint deletion or modification |

## Risks and assumptions

| Risk | Boundary |
|---|---|
| This is not a full Stage2-C run. | Results only prove evaluator correctness on test evidence. |
| True ADV3B02/qknn8 open-set evidence is not yet connected. | Follow-up must export or generate real per-receiver qknn evidence with old/seen-new/unknown under LEO view. |
| Resource constraints file by exact Chinese title was not found locally. | Current resource telemetry is bytes/latency only; prototype storage and VRAM require later measurement. |
| User permits launch despite other low-memory GPU processes. | Current occupancy is even lower/no active compute apps, so GPU4 is acceptable. |

## Remote execution status

Pending sync and remote launch.
