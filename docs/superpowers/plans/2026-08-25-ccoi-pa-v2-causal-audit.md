# CCOI-PA-V2 Causal Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修正CCOI-PA-V2报告中的因果解释和审计缺口，并发布一个复用冻结Core90与现有C4状态、不会重复C0–C4训练的source-only因果审计run。

**Architecture:** 将纯统计计算放入独立的`ccoi_causal_audit.py`，训练runner只负责冻结特征提取、小探针/小头训练和不可覆盖artifact写入。现有`train_phase1_ccoi_pa.py`仅补充未来训练历史的负样本/anchor记录，不改变V2模型、损失或旧run。

**Tech Stack:** Python、PyTorch、pytest、现有SSDG/WiSig数据管线、N607 CUDA、独立JSON artifact。

**Spec:** `E:/codex/home/attachments/31afbf77-fc4f-4ffe-8a1a-da9c1207ecb3/pasted-text.txt`

## Global Constraints

- Phase1严格使用`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，探针只在`L_s`拟合并在`V_select`评估。
- 不读取target/query真值进行训练、校准、选模或候选重排。
- Core90与现有C4 sidecar均冻结；不重复C0–C4，不修改旧run。
- 新run ID与output root不可覆盖；N607只使用普通账户并保护无关进程。
- 训练日志新增字段不追溯伪造旧run的历史计数。
- 报告将`1-NMSE`称为“归一化能量拟合分数”，不称标准`R²`。

---

### Task 1: Pure causal-audit metrics

**Files:**
- Create: `code/cvsrffi/ccoi_causal_audit.py`
- Test: `code/tests/test_ccoi_causal_audit.py`

**Interfaces:**
- Produces: `normalized_energy_fit_score`, `token_code_audit`, `pair_relation_counts`, `complementarity_table`, `factorized_holdout_metrics`。
- Consumes: CPU或CUDA Tensor；所有返回值必须可JSON序列化。

- [ ] **Step 1: Write the failing metric tests.**

```python
import torch
from cvsrffi.ccoi_causal_audit import (
    complementarity_table,
    normalized_energy_fit_score,
    pair_relation_counts,
    token_code_audit,
)


def test_energy_fit_score_is_not_named_r_squared():
    assert normalized_energy_fit_score(0.12593) == pytest.approx(0.87407)


def test_token_code_audit_separates_token_hard_from_packet_dominant():
    probs = torch.tensor([[[.9, .1], [.4, .6]], [[.8, .2], [.3, .7]]])
    out = token_code_audit(probs)
    assert out["token_hard_observed"] == 2
    assert out["packet_dominant_observed"] == 1


def test_pair_relation_counts_are_global_and_include_cross_tx_negatives():
    q = torch.tensor([[1., 0.], [.99, .10], [1., 0.]])
    out = pair_relation_counts(q, torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0]), .90)
    assert out["same_tx_cross_rx_matched"] == 1
    assert out["cross_tx_same_rx_matched"] == 1


def test_complementarity_reports_rescue_harm_and_oracle():
    out = complementarity_table(
        torch.tensor([0, 1, 1, 0]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 0, 1, 0]),
    )
    assert out["base_wrong_side_correct"] == 1
    assert out["base_correct_side_wrong"] == 1
    assert out["oracle_accuracy"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run `python -m pytest code/tests/test_ccoi_causal_audit.py -q`; expect import failure for `cvsrffi.ccoi_causal_audit`.**
- [ ] **Step 3: Implement only the imported functions plus shape/finite validation.**
- [ ] **Step 4: Re-run the same command; expect all tests to pass.**

### Task 2: Training-history observability fix

**Files:**
- Modify: `code/train_phase1_ccoi_pa.py:453-483`
- Test: `code/tests/test_phase1_ccoi_pa_runner.py`

**Interfaces:**
- Produces: every trained epoch records `positive_pairs`, `negative_pairs`, `anchor_count`, and `anchor_fraction`.
- Consumes: existing `CCOILossOutput` without changing loss behavior.

- [ ] **Step 1: Add the failing aggregation regression test.**

```python
from cvsrffi.ccoi_losses import CCOILossOutput
from train_phase1_ccoi_pa import accumulate_pair_audit


def test_pair_audit_records_negative_and_anchor_coverage():
    pair = CCOILossOutput(torch.tensor(0.), positive_count=3, negative_count=5, anchor_count=4)
    sums = {}
    accumulate_pair_audit(sums, pair, batch_size=8)
    assert sums == {
        "positive_pairs": 3.0,
        "negative_pairs": 5.0,
        "anchor_count": 4.0,
        "anchor_fraction": .5,
    }
```

- [ ] **Step 2: Run the single test; expect import failure for `accumulate_pair_audit`.**
- [ ] **Step 3: Implement `accumulate_pair_audit` and replace the old positive-only accumulation.**
- [ ] **Step 4: Run `python -m pytest code/tests/test_phase1_ccoi_pa_runner.py code/tests/test_ccoi_losses.py -q`; expect zero failures.**

### Task 3: Frozen V2 causal-audit runner

**Files:**
- Create: `code/audit_phase1_ccoi_pa_v2.py`
- Test: `code/tests/test_phase1_ccoi_pa_causal_audit_runner.py`

**Interfaces:**
- Consumes: Core90 checkpoint、C4 `sidecar.pth`、WiSig PKL、不可覆盖output dir。
- Produces: `protocol_and_smoke.json`、`feature_audit.json`、`probe_audit.json`、`pair_geometry.json`、`holdout_factorization.json`、`complementarity.json`、`audit_manifest.json`。

- [ ] **Step 1: Add failing parser/protocol tests.**

```python
from audit_phase1_ccoi_pa_v2 import build_arg_parser, validate_sidecar_payload


def test_runner_requires_c4_v2_sidecar_and_source_only_roles(tmp_path):
    args = build_arg_parser().parse_args([
        "--output_dir", str(tmp_path / "new"),
        "--checkpoint", "base.pt",
        "--sidecar", "sidecar.pth",
        "--wisig_pkl", "wisig.pkl",
    ])
    assert args.fit_role == "L_s"
    assert args.eval_role == "V_select"
    validate_sidecar_payload({"schema": "cvs.phase1.ccoi_pa_sidecar.v2", "row": "C4"})
    with pytest.raises(ValueError):
        validate_sidecar_payload({"schema": "cvs.phase1.ccoi_pa_sidecar.v2", "row": "C3"})


def test_runner_refuses_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    args = build_arg_parser().parse_args([
        "--output_dir", str(output),
        "--checkpoint", "base.pt",
        "--sidecar", "sidecar.pth",
        "--wisig_pkl", "wisig.pkl",
    ])
    with pytest.raises(FileExistsError):
        validate_output_root(args)
```

- [ ] **Step 2: Run `python -m pytest code/tests/test_phase1_ccoi_pa_causal_audit_runner.py -q`; expect missing-module failure.**
- [ ] **Step 3: Implement parser、immutable-output guard、sidecar schema guard and no-query smoke.**
- [ ] **Step 4: Implement complete `L_s`/`V_select` extraction for q、code、theta、holdout、base/operator logits and TX/RX/day metadata.**
- [ ] **Step 5: Implement bounded linear、MLP、kNN and flattened-token sequence probes, with fit/eval roles hard-coded to `L_s`/`V_select`.**
- [ ] **Step 6: Implement capacity-matched H0–H6 and HR heads using the same optimizer、epochs、hidden width and target normalization.**
- [ ] **Step 7: Implement group bootstrap keyed by TX×receiver×day and fixed stop-rule fields.**
- [ ] **Step 8: Run the focused test plus `--synthetic_smoke`; expect zero failures and all seven JSON artifacts.**

### Task 4: Report correction and preregistration

**Files:**
- Modify: `docs/experiments/PHASE1_CCOI_PA_V2_S20260824_20260825A_REPORT.md`
- Create: `docs/experiments/PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825A_REPORT.md`
- Mirror: `E:/type10-7/automation_reports/CV-SincNet/PHASE1_CCOI_PA_V2_S20260824_20260825A/report.md`
- Create mirror: `E:/type10-7/automation_reports/CV-SincNet/PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825A/report.md`

**Interfaces:**
- Produces: 逐项回应、事实/假设边界、候选矩阵、命令、路径、GPU、停止规则和预期artifact。

- [ ] **Step 1: Correct the old report at each affected claim and append a traceable response table.**
- [ ] **Step 2: Create the minimal new-run preregistration report.**
- [ ] **Step 3: Re-open both UTF-8 files and verify required wording and paths.**

### Task 5: Local verification, review, and Git publication

**Files:**
- Modify only files listed above plus the reviewed launcher/release helper required by Task 6.

- [ ] **Step 1: Run focused tests in serialized `ssr-gpu`.**
- [ ] **Step 2: Run production Python compilation and the existing CCOI regression suite.**
- [ ] **Step 3: Perform the one allowed independent P0/P1 correctness review.**
- [ ] **Step 4: Inspect exact diff, stage only intended files, commit, push, and independently compare remote branch OID with local `HEAD`.**

### Task 6: N607 release and audit execution

**Files:**
- Create: `code/scripts/launch_phase1_ccoi_pa_v2_causal_audit_20260825.sh`
- Update: new-run report only.

**Interfaces:**
- Consumes: committed release archive、existing Core90 checkpoint、existing immutable C4 sidecar、WiSig PKL。
- Produces: one new immutable N607 run root and bounded audit artifacts; no C0–C4 relaunch.

- [ ] **Step 1: Run the required direct read-only N607 preflight and inspect process/GPU/path state.**
- [ ] **Step 2: Build one release archive, compare its local/remote SHA once, and compile remotely.**
- [ ] **Step 3: Launch exactly one audit owner after the launcher smoke; record PID/CWD/cmdline/GPU/log growth.**
- [ ] **Step 4: Monitor with short read-only SSH checks until artifacts complete or a preregistered systemic technical stop occurs.**

### Task 7: Result analysis and final publication

**Files:**
- Modify: both causal-audit report copies.

- [ ] **Step 1: Parse complete audit logs and every structured artifact.**
- [ ] **Step 2: Apply q-leakage、pair coverage、H0–H6/HR and oracle-gain stop rules without changing them post hoc.**
- [ ] **Step 3: Record implementation coverage、results、limitations、exposed problems and next-route verdict.**
- [ ] **Step 4: Commit/push the final report and independently verify the remote OID.**
