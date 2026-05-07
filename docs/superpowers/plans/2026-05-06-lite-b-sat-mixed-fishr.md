# Lite-B SAT Mixed Fishr Best Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and select the first best checkpoint for `Lite-B no-DAC + SAT mixed consistency + conservative MixStyle + Fishr`.

**Architecture:** Use the already validated `rxrobust_lite_b_no_dac_mix015` preset as the backbone and training-control base. Add mixed-orbit SAT consistency from epoch 20 and Fishr domain-gradient alignment at weight `0.02`; select the best model primarily by Primary OOD, with strict UDU, worst-RX, and SAT scenario average as guardrails.

**Tech Stack:** Python/PyTorch training via `train.py`, WiSig `rx_day` split, CUDA GPUs, shell launch commands, log-based checkpoint selection.

---

## File Structure

- Read-only evidence:
  - `C:\Users\lh594\Desktop\CVS-RFFI\docs\CVS_RFFI_model_route_report_20260506.md`
  - `C:\Users\lh594\Desktop\CVS-RFFI\type10-7\CV-SincNet\logs\SAT05_r19_sat_cons_mixed_20260428_002835.log`
  - `C:\Users\lh594\Desktop\CVS-RFFI\type10-7\CV-SincNet\logs\SAT37_r19_fishr_20260428_203517.log`
- Training entry:
  - `C:\Users\lh594\Desktop\CVS-RFFI\train.py`
- Planned output directory:
  - `C:\Users\lh594\Desktop\CVS-RFFI\finalist_runs\lite_b_sat_mixed_fishr_v1`
- Planned log:
  - `C:\Users\lh594\Desktop\CVS-RFFI\logs\final_lite_b_sat_mixed_fishr_v1_seed1337.log`
- Best model checkpoint target:
  - `C:\Users\lh594\Desktop\CVS-RFFI\finalist_runs\lite_b_sat_mixed_fishr_v1\best_model_primary_ood.pth`

## Model Route

Use this exact route:

```text
Backbone preset: rxrobust_lite_b_no_dac_mix015
Model variant: lite_b
Branch ablation: no_dac
Domain enhancer: rcn_stats
MixStyle: conservative same_tx_crossdomain from preset
SAT train scenario: mixed_orbit
SAT consistency start epoch: 20
SAT classification weight: 0.08
SAT feature consistency weight: 0.04
Fishr weight: 0.02
Fishr minimum domains: 4
Primary OOD UDU weight: 0.65
Epochs: 200
Seed: 1337
```

## Success Gates

The first best model is acceptable only if all gates pass:

```text
Primary OOD score >= 87.80
strict unseen-day unseen-RX >= 86.20
overall test accuracy >= 90.50
worst-RX >= 84.50
SAT final-primary average >= 41.50
No final NaN collapse
skipped_backward_batches <= 50
```

The expected reference band comes from prior logs:

```text
SAT37 Fishr clean/OOD: Primary 87.95, strict UDU 86.43, overall 90.77, SAT Avg 38.91
SAT05 mixed consistency: Primary 86.94, strict UDU 85.35, overall 89.88, SAT Avg 42.97
SAT13 high SAT weight: Primary 87.68, strict UDU 86.14, overall 90.53, SAT Avg 43.91
```

The target of this run is to keep most of SAT37 clean/OOD strength while recovering part of SAT05/SAT13 satellite robustness.

---

### Task 1: Environment And Output Prep

**Files:**
- Read: `C:\Users\lh594\Desktop\CVS-RFFI\train.py`
- Create directory: `C:\Users\lh594\Desktop\CVS-RFFI\finalist_runs\lite_b_sat_mixed_fishr_v1`
- Create directory: `C:\Users\lh594\Desktop\CVS-RFFI\logs`

- [ ] **Step 1: Confirm Python command on the training machine**

Run:

```bash
python - <<'PY'
import sys
print(sys.executable)
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
```

Expected:

```text
Python executable path is printed.
torch version is printed.
cuda True and at least 1 GPU are printed.
```

If `python` is unavailable, run:

```bash
python3 - <<'PY'
import sys
print(sys.executable)
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
```

Expected:

```text
python3 works with torch and CUDA.
Use PYTHON_BIN=python3 in all later commands.
```

- [ ] **Step 2: Prepare output directories**

Run:

```bash
mkdir -p finalist_runs/lite_b_sat_mixed_fishr_v1 logs
```

Expected:

```text
No error.
Both directories exist.
```

- [ ] **Step 3: Verify the training script exposes required arguments**

Run:

```bash
${PYTHON_BIN:-python} train.py --help | grep -E "use_sat_consistency|sat_train_scenario|lambda_sat_cls|lambda_sat_cons|lambda_fishr|slim_group|primary_udu_weight"
```

Expected:

```text
The output includes all listed argument names.
```

---

### Task 2: One-Epoch Smoke Run

**Files:**
- Execute: `C:\Users\lh594\Desktop\CVS-RFFI\train.py`
- Output: `C:\Users\lh594\Desktop\CVS-RFFI\finalist_runs\lite_b_sat_mixed_fishr_v1\smoke_latest.pth`
- Log: `C:\Users\lh594\Desktop\CVS-RFFI\logs\smoke_lite_b_sat_mixed_fishr_v1.log`

- [ ] **Step 1: Run one epoch with reduced SAT evaluation**

Run:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID:-0} ${PYTHON_BIN:-python} -u train.py \
  --dataset wisig \
  --wisig_domain rx_day \
  --batch_size 256 \
  --slim_group rxrobust_lite_b_no_dac_mix015 \
  --epochs 1 \
  --wisig_train_ratio 0.2 \
  --primary_udu_weight 0.65 \
  --use_sat_consistency \
  --sat_train_scenario mixed_orbit \
  --sat_cons_start_epoch 20 \
  --lambda_sat_cls 0.08 \
  --lambda_sat_cons 0.04 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --eval_sat_channel \
  --eval_sat_on test_unseen_day_unseen_rx \
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_eval_max_batches 2 \
  --latest_save_path finalist_runs/lite_b_sat_mixed_fishr_v1/smoke_latest.pth \
  --best_save_path finalist_runs/lite_b_sat_mixed_fishr_v1/smoke_best_val.pth \
  2>&1 | tee logs/smoke_lite_b_sat_mixed_fishr_v1.log
```

Expected:

```text
[MODEL] includes variant=lite_b branch_ablation=no_dac.
[MIXSTYLE] shows p=0.150 strength=0.650.
[SAT-TRAIN] scenario=mixed_orbit lambda_cons=0.0400 lambda_cls=0.0800 start_epoch=20.
[LOSS-DG] includes fishr.
[SAT-TEST] lines appear for configured scenarios.
Training reaches Epoch 001/001 and exits normally.
```

- [ ] **Step 2: Reject the setup if the smoke log has fatal failures**

Run:

```bash
grep -E "Traceback|RuntimeError|ModuleNotFoundError|No such file|nan%" logs/smoke_lite_b_sat_mixed_fishr_v1.log || true
```

Expected:

```text
No Traceback, RuntimeError, ModuleNotFoundError, No such file, or nan% lines.
```

---

### Task 3: Full Candidate Run V1

**Files:**
- Execute: `C:\Users\lh594\Desktop\CVS-RFFI\train.py`
- Output directory: `C:\Users\lh594\Desktop\CVS-RFFI\finalist_runs\lite_b_sat_mixed_fishr_v1`
- Log: `C:\Users\lh594\Desktop\CVS-RFFI\logs\final_lite_b_sat_mixed_fishr_v1_seed1337.log`

- [ ] **Step 1: Start the full 200-epoch run**

Run:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID:-0} ${PYTHON_BIN:-python} -u train.py \
  --dataset wisig \
  --wisig_domain rx_day \
  --batch_size 256 \
  --slim_group rxrobust_lite_b_no_dac_mix015 \
  --epochs 200 \
  --wisig_train_ratio 0.2 \
  --primary_udu_weight 0.65 \
  --seed 1337 \
  --use_sat_consistency \
  --sat_train_scenario mixed_orbit \
  --sat_cons_start_epoch 20 \
  --lambda_sat_cls 0.08 \
  --lambda_sat_cons 0.04 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --eval_sat_channel \
  --eval_sat_on test_unseen_day_unseen_rx \
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_eval_max_batches -1 \
  --latest_save_path finalist_runs/lite_b_sat_mixed_fishr_v1/latest_model.pth \
  --best_save_path finalist_runs/lite_b_sat_mixed_fishr_v1/best_model_val.pth \
  --best_primary_save_path finalist_runs/lite_b_sat_mixed_fishr_v1/best_model_primary_ood.pth \
  --best_test_save_path finalist_runs/lite_b_sat_mixed_fishr_v1/best_model_test_overall.pth \
  --best_unseen_day_unseen_rx_save_path finalist_runs/lite_b_sat_mixed_fishr_v1/best_model_strict_udu.pth \
  --best_worst_rx_save_path finalist_runs/lite_b_sat_mixed_fishr_v1/best_model_worst_rx.pth \
  2>&1 | tee logs/final_lite_b_sat_mixed_fishr_v1_seed1337.log
```

Expected:

```text
The run completes 200 epochs.
The final log includes Training finished lines for best_primary_ood_score, best_test_overall_tx_acc, best_unseen_day_unseen_rx_tx_acc, and best_worst_rx_tx_acc.
The checkpoint finalist_runs/lite_b_sat_mixed_fishr_v1/best_model_primary_ood.pth exists.
```

- [ ] **Step 2: Monitor during training every 10 epochs**

Run:

```bash
tail -n 120 logs/final_lite_b_sat_mixed_fishr_v1_seed1337.log
```

Expected healthy signals:

```text
VAL tx rises toward 98-99%.
TEST overall rises toward 89-91%.
SAT-TEST lines are present after each epoch.
skipped backward warnings remain occasional, not continuous.
No [LOSS-CORE] total=nan lines.
```

Stop the run if this pattern appears after epoch 40:

```text
[VAL] tx=16.67%
[TEST] overall_tx=16.67%
[LOSS-CORE] total=nan
```

---

### Task 4: Parse And Select The First Best Model

**Files:**
- Read: `C:\Users\lh594\Desktop\CVS-RFFI\logs\final_lite_b_sat_mixed_fishr_v1_seed1337.log`
- Select: `C:\Users\lh594\Desktop\CVS-RFFI\finalist_runs\lite_b_sat_mixed_fishr_v1\best_model_primary_ood.pth`

- [ ] **Step 1: Extract final-best metrics**

Run:

```bash
grep -E "Training finished|\\[FINAL-PRIMARY\\]|\\[FINAL-BEST\\]|\\[BEST-PRIMARY\\]|\\[BEST-WORST-RX\\]" \
  logs/final_lite_b_sat_mixed_fishr_v1_seed1337.log
```

Expected:

```text
best_primary_ood_score is printed.
FINAL-PRIMARY has test_overall_tx, strict_udu, and score.
FINAL-PRIMARY SAT-TEST lines are printed for clear_leo, low_elev_leo, rain_leo, storm_mp, mixed_orbit.
```

- [ ] **Step 2: Compute SAT final-primary average**

Run:

```bash
${PYTHON_BIN:-python} - <<'PY'
import re
from pathlib import Path
log = Path("logs/final_lite_b_sat_mixed_fishr_v1_seed1337.log").read_text(errors="ignore")
vals = [float(m.group(1)) for m in re.finditer(r"\[FINAL-PRIMARY\] \[SAT-TEST\].*?strict_udu=([0-9.]+)%", log)]
print("sat_final_primary_values", vals)
print("sat_final_primary_avg", round(sum(vals) / len(vals), 2) if vals else "missing")
PY
```

Expected:

```text
Five SAT values are printed.
sat_final_primary_avg is numeric.
```

- [ ] **Step 3: Accept or reject the checkpoint**

Accept:

```text
Use finalist_runs/lite_b_sat_mixed_fishr_v1/best_model_primary_ood.pth as the first best model if all Success Gates pass.
```

Reject:

```text
Do not promote the checkpoint if Primary OOD < 87.80, strict UDU < 86.20, SAT Avg < 41.50, final collapse occurred, or skipped_backward_batches > 50.
```

---

### Task 5: Fallback Run V2 If V1 Misses Satellite Robustness

**Files:**
- Output directory: `C:\Users\lh594\Desktop\CVS-RFFI\finalist_runs\lite_b_sat_mixed_fishr_v2_sat_heavy`
- Log: `C:\Users\lh594\Desktop\CVS-RFFI\logs\final_lite_b_sat_mixed_fishr_v2_sat_heavy_seed1337.log`

Use V2 only if V1 clean/OOD passes but SAT Avg is below `41.50`.

- [ ] **Step 1: Run SAT-heavy Fishr variant**

Run:

```bash
mkdir -p finalist_runs/lite_b_sat_mixed_fishr_v2_sat_heavy

CUDA_VISIBLE_DEVICES=${GPU_ID:-0} ${PYTHON_BIN:-python} -u train.py \
  --dataset wisig \
  --wisig_domain rx_day \
  --batch_size 256 \
  --slim_group rxrobust_lite_b_no_dac_mix015 \
  --epochs 200 \
  --wisig_train_ratio 0.2 \
  --primary_udu_weight 0.65 \
  --seed 1337 \
  --use_sat_consistency \
  --sat_train_scenario mixed_orbit \
  --sat_cons_start_epoch 20 \
  --lambda_sat_cls 0.16 \
  --lambda_sat_cons 0.08 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --eval_sat_channel \
  --eval_sat_on test_unseen_day_unseen_rx \
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_eval_max_batches -1 \
  --latest_save_path finalist_runs/lite_b_sat_mixed_fishr_v2_sat_heavy/latest_model.pth \
  --best_save_path finalist_runs/lite_b_sat_mixed_fishr_v2_sat_heavy/best_model_val.pth \
  --best_primary_save_path finalist_runs/lite_b_sat_mixed_fishr_v2_sat_heavy/best_model_primary_ood.pth \
  --best_test_save_path finalist_runs/lite_b_sat_mixed_fishr_v2_sat_heavy/best_model_test_overall.pth \
  --best_unseen_day_unseen_rx_save_path finalist_runs/lite_b_sat_mixed_fishr_v2_sat_heavy/best_model_strict_udu.pth \
  --best_worst_rx_save_path finalist_runs/lite_b_sat_mixed_fishr_v2_sat_heavy/best_model_worst_rx.pth \
  2>&1 | tee logs/final_lite_b_sat_mixed_fishr_v2_sat_heavy_seed1337.log
```

Expected:

```text
The run completes 200 epochs.
SAT Avg improves over V1.
Primary OOD remains >= 87.50 and strict UDU remains >= 86.00.
```

---

### Task 6: Fallback Run V3 If V1 Misses Clean/OOD

**Files:**
- Output directory: `C:\Users\lh594\Desktop\CVS-RFFI\finalist_runs\lite_b_sat_mixed_fishr_v3_delayed_sat`
- Log: `C:\Users\lh594\Desktop\CVS-RFFI\logs\final_lite_b_sat_mixed_fishr_v3_delayed_sat_seed1337.log`

Use V3 only if V1 clean/OOD is below gate, because the historical delayed-SAT run `SAT15` kept clean/OOD strong.

- [ ] **Step 1: Run delayed SAT-start Fishr variant**

Run:

```bash
mkdir -p finalist_runs/lite_b_sat_mixed_fishr_v3_delayed_sat

CUDA_VISIBLE_DEVICES=${GPU_ID:-0} ${PYTHON_BIN:-python} -u train.py \
  --dataset wisig \
  --wisig_domain rx_day \
  --batch_size 256 \
  --slim_group rxrobust_lite_b_no_dac_mix015 \
  --epochs 200 \
  --wisig_train_ratio 0.2 \
  --primary_udu_weight 0.65 \
  --seed 1337 \
  --use_sat_consistency \
  --sat_train_scenario mixed_orbit \
  --sat_cons_start_epoch 60 \
  --lambda_sat_cls 0.08 \
  --lambda_sat_cons 0.04 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --eval_sat_channel \
  --eval_sat_on test_unseen_day_unseen_rx \
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_eval_max_batches -1 \
  --latest_save_path finalist_runs/lite_b_sat_mixed_fishr_v3_delayed_sat/latest_model.pth \
  --best_save_path finalist_runs/lite_b_sat_mixed_fishr_v3_delayed_sat/best_model_val.pth \
  --best_primary_save_path finalist_runs/lite_b_sat_mixed_fishr_v3_delayed_sat/best_model_primary_ood.pth \
  --best_test_save_path finalist_runs/lite_b_sat_mixed_fishr_v3_delayed_sat/best_model_test_overall.pth \
  --best_unseen_day_unseen_rx_save_path finalist_runs/lite_b_sat_mixed_fishr_v3_delayed_sat/best_model_strict_udu.pth \
  --best_worst_rx_save_path finalist_runs/lite_b_sat_mixed_fishr_v3_delayed_sat/best_model_worst_rx.pth \
  2>&1 | tee logs/final_lite_b_sat_mixed_fishr_v3_delayed_sat_seed1337.log
```

Expected:

```text
Primary OOD recovers toward the SAT37 reference band.
strict UDU is >= V1 strict UDU if V1 underperformed.
```

---

### Task 7: Final Promotion Record

**Files:**
- Create: `C:\Users\lh594\Desktop\CVS-RFFI\docs\lite_b_sat_mixed_fishr_run_result.md`

- [ ] **Step 1: Record the selected checkpoint**

Create `docs/lite_b_sat_mixed_fishr_run_result.md` with this exact structure:

```markdown
# Lite-B SAT Mixed Fishr Run Result

Date:

## Selected Checkpoint

Checkpoint:
Log:
Run directory:

## Metrics

| Metric | Value |
|---|---:|
| Primary OOD |  |
| strict UDU |  |
| overall |  |
| worst-RX |  |
| SAT Avg |  |
| skipped backward batches |  |

## Decision

Selected because:

Rejected alternatives:
```

- [ ] **Step 2: Fill the record from log output**

Use values from:

```bash
grep -E "Training finished|\\[FINAL-PRIMARY\\]|\\[BEST-WORST-RX\\]" logs/final_lite_b_sat_mixed_fishr_v1_seed1337.log
```

Expected:

```text
The markdown file identifies exactly one selected checkpoint and exactly one log.
The metrics table contains numeric values for every row.
```

## Execution Choice

Recommended first execution:

```text
Run Task 1, Task 2, Task 3, and Task 4 only.
Do not run V2 or V3 unless V1 misses a gate.
```

