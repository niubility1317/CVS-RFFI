# Fedbase CVS-RFFI Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the four `Fedbase/` RFFI papers as paper-named, testable CVS-RFFI adaptations without re-labeling existing approximate project modules as strict reproductions.

**Architecture:** Keep paper-specific algorithms in small modules under `code/federated/`, then wire only the verified modules into the existing `code/train.py` federated route. The first implementation layer proves the exact losses, aggregation rules, client selection, and pretrain/fine-tune boundaries on tiny tensors; the second layer provides CLI and launcher reachability for CVS-RFFI/N607 experiments.

**Tech Stack:** Python, PyTorch, pytest/unittest, existing CVS-RFFI `dataset_wisig.py`, existing federated client splitting and state aggregation helpers.

---

## File Structure

- Create `analysis/fedbase_cvsrffi_reproduction_traceability_20260603.md`: source-to-code traceability table for all four papers.
- Create `code/federated/fedriei.py`: FedRIEI alternating local step and receiver-client aggregation helpers.
- Create `code/federated/feature_alignment.py`: FedFA complex CNN blocks and pairwise full-covariance CORAL loss.
- Create `code/federated/contrastive_fl.py`: FUCL two-view augmentation contract, NT-Xent loss, encoder-only aggregation helpers.
- Create `code/federated/receiver_agnostic_fl.py`: RAFL GRL loss helpers and Label Loss Driven client selection.
- Create `tests/test_fedbase_paper_methods.py`: tiny tensor tests for the four paper-specific modules.
- Modify `code/train.py`: add paper-named `train_mode` choices and dispatch placeholders only after module tests pass.
- Modify `code/federated/fed_trainer.py`: accept paper-named train modes and route to paper-specific local objectives only after module tests pass.
- Add `run_fedbase_paper_queue.sh`: dry-run first; commands must include `--wisig_train_ratio 0.1`, `--fl_rounds 200`, `--epochs 200`, `--fl_client_key receiver`.
- Update `code/SYNC_MANIFEST.txt` or an N607 run report after local verification, because this workspace is not a git repository.

## Task 1: Traceability Record

**Files:**
- Create: `analysis/fedbase_cvsrffi_reproduction_traceability_20260603.md`

- [ ] **Step 1: Add the initial traceability table**

Record all paper requirements extracted by the subagents. Each row must use one of `pending`, `implemented`, `verified`, `deferred`, `rejected`, or `blocked`.

- [ ] **Step 2: Mark known boundaries**

Mark CVS-RFFI data protocol differences as `pending` or `deferred`, not as verified strict original-dataset reproduction.

## Task 2: Red Tests For Paper Method Kernels

**Files:**
- Create: `tests/test_fedbase_paper_methods.py`

- [ ] **Step 1: Write failing import tests**

Add tests that import:

```python
from federated.fedriei import fedriei_loss_terms, fedriei_alternating_step
from federated.feature_alignment import pairwise_coral_alignment_loss
from federated.contrastive_fl import nt_xent_loss, encoder_only_state_dict
from federated.receiver_agnostic_fl import label_loss_driven_client_selection, receiver_agnostic_loss
```

- [ ] **Step 2: Run red tests**

Run:

```powershell
conda activate ssr-gpu; python -m pytest tests/test_fedbase_paper_methods.py -q
```

Expected: fail because the new modules do not exist.

## Task 3: FedRIEI Kernel

**Files:**
- Create: `code/federated/fedriei.py`
- Test: `tests/test_fedbase_paper_methods.py`

- [ ] **Step 1: Add loss term tests**

Test that FedRIEI returns `loss_ce`, `loss_mi`, `loss_ie`, and total `loss = loss_ce + lambda_mi * loss_mi - lambda_ie * loss_ie`.

- [ ] **Step 2: Add alternating-step test**

Use a tiny `baselines.riei_fd.model.RIEIModel` and assert the step returns two phases: `ce_phase` and `disentangle_phase`.

- [ ] **Step 3: Implement minimal FedRIEI helpers**

Implement helpers by reusing `baselines.riei_fd.losses` and preserving the paper-specific phase names.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
conda activate ssr-gpu; python -m pytest tests/test_fedbase_paper_methods.py -q
```

## Task 4: FedFA Kernel

**Files:**
- Create: `code/federated/feature_alignment.py`
- Test: `tests/test_fedbase_paper_methods.py`

- [ ] **Step 1: Add CORAL hand-check test**

Construct two tiny feature matrices and assert the loss uses `1/(4d^2)` full covariance distance.

- [ ] **Step 2: Add complex-convolution shape test**

Instantiate a minimal complex block and assert `[B,2,L] -> embedding_dim` remains reachable.

- [ ] **Step 3: Implement FedFA helpers**

Implement pairwise covariance loss and minimal paper-named model components. Do not reuse class-conditional/server-bank CORAL as the strict paper loss.

## Task 5: FUCL Kernel

**Files:**
- Create: `code/federated/contrastive_fl.py`
- Test: `tests/test_fedbase_paper_methods.py`

- [ ] **Step 1: Add NT-Xent mask test**

For `2B` embeddings, assert positives are same-sample paired views, not same-label positives.

- [ ] **Step 2: Add encoder-only aggregation test**

Assert classifier/head keys are excluded from aggregation.

- [ ] **Step 3: Implement FUCL helpers**

Implement cosine NT-Xent with temperature `0.05` default and encoder state filtering.

## Task 6: RAFL Kernel

**Files:**
- Create: `code/federated/receiver_agnostic_fl.py`
- Test: `tests/test_fedbase_paper_methods.py`

- [ ] **Step 1: Add GRL loss behavior test**

Use tiny logits and assert transmitter and receiver CE terms are separately logged; GRL is represented by the model path, not by subtracting receiver CE from the receiver-head update.

- [ ] **Step 2: Add Label Loss Driven selection test**

Given candidate client label-wise losses, assert selected clients correspond to highest-loss labels and not overall-loss Power-of-Choice.

- [ ] **Step 3: Implement RAFL helpers**

Implement label-wise aggregation, selected-label ranking, and deterministic fallback fill when fewer than `S` clients are selected.

## Task 7: CLI And Launcher Reachability

**Files:**
- Modify: `code/train.py`
- Modify: `code/federated/fed_trainer.py`
- Create: `run_fedbase_paper_queue.sh`
- Test: `tests/test_fedbase_launcher.py`

- [ ] **Step 1: Write launcher dry-run test**

Assert four commands are emitted for `fedriei`, `fedfa`, `fucl`, and `rafl`, each with CVS-RFFI hard constraints.

- [ ] **Step 2: Add train-mode parse test**

Assert `code/train.py --help` includes the four paper-named modes.

- [ ] **Step 3: Implement minimal CLI wiring**

Add choices and paper-route logging without changing existing default `centralized/fedavg/fedprox/fedcvs_vmb/split_bex02` behavior.

## Task 8: Review And Local Verification

**Files:**
- Update: `analysis/fedbase_cvsrffi_reproduction_traceability_20260603.md`
- Update: `code/SYNC_MANIFEST.txt` or future N607 report

- [ ] **Step 1: Run focused local tests**

Run:

```powershell
conda activate ssr-gpu; python -m pytest tests/test_fedbase_paper_methods.py tests/test_fedbase_launcher.py -q
```

- [ ] **Step 2: Dispatch reviewer subagents**

Ask one reviewer to check paper-method parity and one reviewer to check code/test quality. Fix critical and important findings before remote sync.

- [ ] **Step 3: Prepare N607 gate**

Only after local verification, run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1
```

Then create/update an experiment report before any `scp` or launch.

## Self-Review Notes

- The plan separates strict paper kernels from existing project enhancements.
- The first local target is testable without N607 or the full WiSig dataset.
- Full N607 experiments remain gated by local tests, traceability status, preflight, sync manifest/report, and startup health check.
