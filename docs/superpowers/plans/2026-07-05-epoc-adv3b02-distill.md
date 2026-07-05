# EPOC-ADV3B02 Distill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在`ADV3B02_CORE90_SOFT_E200`基础上训练一个source-only教师蒸馏学生模型，增强LEO星地信道下旧类表征稳定性和虚拟开集边界，同时不接触真实未知类别。

**Architecture:** `ADV3B02`作为冻结教师；学生从`ADV3B02`初始化并全量微调。训练只使用`ManySig`源域`Y_old`和源域派生LEO视图，拒识压力来自旧类内部leave-one-TX-out、soft inter-class mixup和虚拟低密度outlier，不加载`ManyTx`或`target_unknown`。

**Tech Stack:** Python、PyTorch、`code/SSDG/train_ssdg.py`、bash launcher、pytest、N607 CVS-RFFI Conda环境。

---

### Task 1: 教师蒸馏参数与训练损失

**Files:**
- Modify:`code/SSDG/train_ssdg.py`
- Test:`code/tests/test_epoc_adv3b02_teacher_distill.py`

- [x] **Step 1: Write the failing test**

```python
args = build_arg_parser().parse_args([
    "--output_dir","out",
    "--teacher_ckpt","/runs/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth",
    "--lambda_teacher_clean_kl","0.35",
    "--lambda_teacher_sat_kl","0.20",
    "--lambda_teacher_zid_mse","0.04",
])
assert args.lambda_teacher_clean_kl == 0.35
```

- [x] **Step 2: Run test to verify it fails**

Run:`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_epoc_adv3b02_teacher_distill.py -q`

Expected:FAIL with unrecognized teacher distillation arguments.

- [x] **Step 3: Write minimal implementation**

Add parser args, load frozen teacher when any teacher weight is non-zero, compute clean KL, sat-view KL and normalized`z_id`MSE.

- [x] **Step 4: Run test to verify it passes**

Run:`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_epoc_adv3b02_teacher_distill.py -q`

Expected:PASS.

### Task 2: EPOC启动器

**Files:**
- Create:`code/scripts/launch_phase1_epoc_adv3b02_distill_20260705.sh`
- Test:`code/tests/test_epoc_adv3b02_teacher_distill.py`

- [x] **Step 1: Write the failing launcher test**

```python
result = subprocess.run([
    "bash","code/scripts/launch_phase1_epoc_adv3b02_distill_20260705.sh",
    "--dry-run","--only=EPOC_DISTILL_A_MILD",
], check=True, capture_output=True, text=True)
assert "--teacher_ckpt" in result.stdout
assert "ManyTx.pkl" not in result.stdout
```

- [x] **Step 2: Run test to verify it fails**

Expected:FAIL because launcher file does not exist.

- [x] **Step 3: Write minimal launcher**

Create 8 candidates across GPUs0-7 with wider teacher/open-set/satellite weights. Use only`ManySig.pkl`; log`real_unknown_classes_in_training=0`.

- [x] **Step 4: Verify launcher**

Run:`bash -n code/scripts/launch_phase1_epoc_adv3b02_distill_20260705.sh`

Run:`bash code/scripts/launch_phase1_epoc_adv3b02_distill_20260705.sh --dry-run --only=EPOC_DISTILL_A_MILD`

Expected:syntax PASS and dry-run command contains teacher distillation,source-only,LEO scenarios and no`ManyTx.pkl`.

### Task 3: N607运行与报告

**Files:**
- Create/Modify:`automation_reports/CV-SincNet/phase1_epoc_adv3b02_distill_20260705/report.md`
- Modify:`code/SYNC_MANIFEST.txt`

- [ ] **Step 1: Record local verification**

Record RED/GREEN evidence, changed files, protocol boundary and exact local commands.

- [ ] **Step 2: Run N607 preflight**

Run:`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`

Expected:direct N607 access, project root and GPU visibility.

- [ ] **Step 3: Sync files**

Use`scp` for changed code, tests, launcher, plan and report.

- [ ] **Step 4: Remote verify**

Run remote`py_compile`,`pytest`,`bash -n` and dry-run under`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`.

- [ ] **Step 5: Launch and health check**

Launch bounded launcher with independent logs under`logs/phase1_epoc_adv3b02_distill_20260705/`; after startup inspect`[CONFIG-TEACHER]`,`[CONFIG-LOSS]`,`[CONFIG-SAT]`,`[EPOCH-BEGIN]`,traceback/OOM/NaN.

