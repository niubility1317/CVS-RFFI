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
| `code/evaluation/__init__.py` | Ensures N607 imports `code/evaluation` before the root-level `evaluation` package. | `A1BF5887A4EF35178CE57117AC9BAC013543C57A55FBACDEE3B3C59E8A57A341` |
| `code/evaluation/collaborative_inference_eval.py` | Existing closed-set collaborative evaluator synced as a remote regression dependency. | `A01B8A95070D52F00993047FCD75F76E6364CA8FC2714FF6D693D0063671F489` |
| `code/tests/test_collaborative_inference_eval.py` | Existing closed-set collaborative evaluator tests synced as a remote regression dependency. | `1BD574819DC3CE1C88D9A092DC3BE53B7902677565D142C75EA0C49ED44699D1` |

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
| `E:\type10-7\code\evaluation\__init__.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/__init__.py` |
| `E:\type10-7\code\evaluation\collaborative_inference_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/collaborative_inference_eval.py` |
| `E:\type10-7\code\tests\test_collaborative_inference_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_collaborative_inference_eval.py` |

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
- `logs/phaseA_adv3b02_collab_open_set_qknn_eval_20260703/remote_unittest_final.out`
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

Final status: completed.

### Sync verification

Remote hashes matched local for the four new/changed Phase A files and report. Additional regression dependencies were synced and verified:

| Remote path | SHA256 |
|---|---|
| `code/evaluation/__init__.py` | `a1bf5887a4ef35178ce57117ac9bac013543c57a55fbacdee3b3c59e8a57a341` |
| `code/evaluation/collaborative_inference_eval.py` | `a01b8a95070d52f00993047fcd75f76e6364ca8fc2714ff6d693d0063671f489` |
| `code/tests/test_collaborative_inference_eval.py` | `1bd574819dc3ce1c88d9a092dc3be53b7902677565d142c75ea0c49ed44699d1` |

### Remote command outcome

The first planned pytest command failed because the remote `CVS-RFFI` Conda environment does not include `pytest`:

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python: No module named pytest
```

The verification was rerun with standard-library `unittest` instead. One intermediate attempt using `python -m unittest code.tests...` failed because the root-level `evaluation` package and stdlib `code` module conflicted with the intended `code/evaluation` import path. This was fixed by syncing the already tracked `code/evaluation/__init__.py` package marker to N607 and executing test files by path.

Final remote command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
RUN_ID=phaseA_adv3b02_collab_open_set_qknn_eval_20260703
{
  env PYTHONPATH=$PWD/code:$PWD CUDA_VISIBLE_DEVICES=4 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_collaborative_open_set_qknn_eval.py
  echo open_set_rc=$?
  env PYTHONPATH=$PWD/code:$PWD CUDA_VISIBLE_DEVICES=4 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_collaborative_inference_eval.py
  echo closed_rc=$?
} > logs/$RUN_ID/remote_unittest_final.out 2>&1
```

Final remote log:

```text
Ran 7 tests in 0.004s
OK
open_set_rc=0
Ran 5 tests in 0.056s
OK
closed_rc=0
```

### Resource and cleanup status

| Item | Result |
|---|---|
| GPU used | `CUDA_VISIBLE_DEVICES=4` |
| Final GPU memory | all GPUs reported 10 MiB used at 2026-07-03 16:09:28 CST |
| Remote test processes | none left after completion |
| Local SSH cleanup | transient `ssh.exe` exited; no ESTABLISHED connection to `172.31.111.215:22` or `172.31.105.18:22` remained |

### Interpretation

This completes N607 synchronization and low-load remote verification for the Phase A evaluator module. It does not complete the full scientific experiment: real ADV3B02/qknn8 per-receiver evidence under satellite/LEO old/seen-new/unknown splits is still required before reporting old 99%, seen-new 97%, or unknown reject 99% claims.
