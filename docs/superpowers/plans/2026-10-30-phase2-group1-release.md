# Phase2 Group1 Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the approved Group1 comparison matrix with default source clean/satellite concatenation, midpoint PL and complete stage resource records; launch only correctly bound rows on N607.

**Architecture:** New Group1 entry points leave the stopped legacy queues untouched. Source-only jobs freeze independent native models; DA jobs execute A→B and registration jobs execute A→C. PAMF uses two support fits and four query paths; scoring is a separate dependent process.

**Tech Stack:** Python/PyTorch, NumPy/SciPy, existing practical residual channel, source cache, r10 three-pool capsules and frozen DURA.

**Spec:** `docs/PAMF_PHASE2_GROUP1_DESIGN_20261030.md`, amended by the user's 2026-10-30 source concatenation and release authorization.

## Global Constraints

- Source training retains native complete supervised losses and uses clean＋satellite concatenation from the first epoch; PL threshold0.95/weight1 starts at the midpoint.
- Pure metric methods use distance softmax and native metric losses; no new trainable classifier.
- Query is never fitted; independent truth-side scoring follows completed predictions.
- Each GPU admits at most two own workers, other owners additional; check actual free memory and preserve other processes.
- One immutable root, one committed release archive/transfer SHA comparison, one no-query real-checkpoint smoke and one independent P0/P1 review.
- CSIL author Fisher and Tweak M10 remain explicit unavailable bindings; no substitution by corefix/M1.

## Review Focus

- U TX truth is absent from loader outputs and never used to form pseudo labels.
- Source satellite views genuinely enter the optimized loss, with actual exposure recorded.
- CIL C begins at A and never calls legacy generic B10.
- PAMF support/query flags are independent and query inference cannot mutate F/G/decision state.
- Failed source/protocol tasks block descendants and cannot masquerade as completed jobs.

## Task 1: Source recipe and resource instrumentation

**Files:** `experiments/pamf_phase2_practical/group1.py`, `group1_source.py`, `group1_resources.py`; `tests/test_phase2_group1.py`.

- [ ] Add focused failing tests for midpoint PL, distance confidence, source boundary and optimization/resource recording.
- [ ] Implement native source bindings with full supervised objectives, source-only practical concatenation and no early stop.
- [ ] Verify a saved real checkpoint using source IQ only before target jobs.

## Task 2: Native target paths and matrix dispatcher

**Files:** `experiments/pamf_phase2_practical/group1_target.py`, `group1_dispatch.py`; Group1 release config and run report.

- [ ] Test and implement DA-only B and direct native C, with no inherited legacy combinations.
- [ ] Implement PAMF four cells with two support fits, shared C terminal and per-component resource records.
- [ ] Generate main/curve/pure scopes, pilot dependencies and independent scoring; preserve exact r10 handles.

## Task 3: Publish and verify

- [ ] Amend scientific protocol and design with latest source augmentation/release authorization; register exact run.
- [ ] Run focused tests and one independent launch-critical review; fix direct P0/P1 issues.
- [ ] Commit only owned changes, push/read back, transfer one archive and compare its SHA once, compile remotely.
- [ ] Launch new owner/root; independently verify PID/CWD/argv/GPU/log growth and report exact running/pending rows.
