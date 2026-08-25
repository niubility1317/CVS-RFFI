# Slow-Fast P0.5 Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可在地面source receiver-held-out episode上冻结、在Phase2仅凭target support运行的P0.5门控，并补齐旧V2的truth-last score诊断。

**Architecture:** `slow_fast_selection.py`只负责support统计和状态选择；新增`slow_fast_calibration.py`负责source receiver-held-out episode和冻结校准bundle；`stage2_slow_fast_runner.py`只消费冻结参数并生成query只读prediction；`slow_fast_scorer.py`在全部prediction闭合后计算旧/新query分解、raw-cosine诊断和响应面。

**Tech Stack:** Python3.10、PyTorch、NumPy、pytest、JSON、现有CVS `p2_min_v1` runner/scorer。

**Spec:** `docs/superpowers/specs/2026-08-25-slow-fast-p05-calibration-design.md`

## Global Constraints

- Phase2只允许target support更新/选择；query truth、role和score不能回流。
- source receiver-held-out query只用于Phase1.5地面校准。
- 不重用receiver20-1、seed392002的target truth调参后重跑。
- `VALIDATED_ONCE`数据不因方法改动重验。
- 不新增逐support-token hash、seal或额外发布门。
- 所有行为变更先执行RED测试，再写最小生产实现。

---

### Task 1: Support统计、唯一cross-fit和P0.5门控

**Files:**
- Modify: `code/cvsrffi/slow_fast_selection.py`
- Modify: `tests/test_slow_fast_selection.py`

**Interfaces:**
- Produces: `SupportTrustPolicy`、`support_state_diagnostics(...)`、扩展后的`select_support_only_state(...)`。
- Consumes: `features/labels/prototypes/physical_ids`、冻结policy和row seed。

- [x] **Step 1: 写唯一fold与physical ID互斥失败测试**

```python
def test_crossfit_removes_duplicate_partitions_and_keeps_physical_ids_disjoint():
    splits = _stratified_crossfit_splits(labels, k_shot=2, seed=7, repeats=8,
                                         physical_ids=physical_ids)
    assert len({canonical_split(split) for split in splits}) == len(splits) // 2
    assert all(train_ids.isdisjoint(valid_ids) for train_ids, valid_ids in split_ids(splits))
```

- [x] **Step 2: 运行RED**

Run: `python -m pytest tests/test_slow_fast_selection.py -q`
Expected: FAIL，现有API不接收`physical_ids`且重复partition未去除。

- [x] **Step 3: 最小实现唯一重复分层2-fold**

```python
def _stratified_crossfit_splits(labels, *, k_shot, seed, repeats, physical_ids=None):
    # 每个repeat生成互补两折；canonical key去除相同/互补重复；直接检查ID集合互斥。
```

- [x] **Step 4: 写分位数、相对trust和support归一化强度RED测试**

```python
def test_support_normalized_strength_hits_q90_target_without_exceeding_hard_cap():
    policy = SupportTrustPolicy(q90_move=0.10, hard_move=0.30,
                                q90_relative_move=0.8, minimum_positive_folds=5)
    diagnostics = support_state_diagnostics(..., nominal_lambda=0.125, policy=policy)
    assert diagnostics["effective_lambda"] == pytest.approx(0.05)
    assert diagnostics["q90_feature_move"] <= 0.10 + 1e-6
```

- [x] **Step 5: 运行RED并实现support统计**

实现字段：`q50/q90/max_feature_move`、`q90_relative_move`、逐fold`risk_gain`、`positive_fold_count`、`fold_gain_std`、`fold_gain_lcb90`、`effective_lambda`。

- [x] **Step 6: 修复审计语义和计算量字段**

```python
audit.update({
    "selection_protocol": "repeated_stratified_2fold",
    "crossfit_fit_count": fit_count,
    "loo_fit_count": 0,
    "deployment_candidate_updates": steps,
    "crossfit_updates": fit_count * steps,
})
```

- [x] **Step 7: GREEN与回归**

Run: `python -m pytest tests/test_slow_fast_selection.py -q`
Expected: PASS。

### Task 2: Truth-last scorer的旧/新query与raw-cosine诊断

**Files:**
- Modify: `code/cvsrffi/slow_fast_scorer.py`
- Modify: `tests/test_slow_fast_scorer.py`
- Modify: `tests/test_stage2_slow_fast_runner.py`

**Interfaces:**
- Produces: `_score_shadow_row`中的`flip_diagnostics`、`score_diagnostics`、`new_class_intrusion`。
- Consumes: 已校验的ordered query IDs、predicted IDs、raw cosine scores和truth sidecar。

- [x] **Step 1: 写旧类flip和新类变化RED测试**

```python
assert state_diag == {
    "old_query_decision_changes": 2,
    "old_positive_flips": 1,
    "old_negative_flips": 1,
    "new_query_decision_changes": 1,
    "net_old_correct_change": 0,
}
```

- [x] **Step 2: 运行RED并最小实现truth分组计数**

必须先验证全部prediction路径、score形状和ordered IDs，再首次读取truth。

- [x] **Step 3: 写raw-cosine手算RED测试**

```python
assert diag["mean_true_class_cosine_delta"] == pytest.approx(0.05)
assert diag["mean_top1_top2_margin_delta"] == pytest.approx(-0.02)
assert diag["mean_score_vector_l2_change"] == pytest.approx(0.1)
assert diag["new_class_intrusion_delta"] == pytest.approx(0.03)
```

- [x] **Step 4: 实现score诊断并GREEN**

Run: `python -m pytest tests/test_slow_fast_scorer.py tests/test_stage2_slow_fast_runner.py -q`
Expected: PASS。

### Task 3: 响应面、Spearman和移动—收益汇总

**Files:**
- Create: `code/cvsrffi/slow_fast_diagnostics.py`
- Create: `tests/test_slow_fast_diagnostics.py`
- Modify: `code/cvsrffi/slow_fast_scorer.py`

**Interfaces:**
- Produces: `build_shadow_response_surface(row_scores, support_receipts)`。
- Returns: scene/state rows、Spearman、move-gain rows和P0停止信号。

- [x] **Step 1: 写手算Spearman和完整状态轴RED测试**

```python
summary = build_shadow_response_surface(scores, receipts)
assert summary["spearman_support_query"] == pytest.approx(-1.0)
assert summary["state_count"] == 4
assert summary["p0_stop_signal"] is True
```

- [x] **Step 2: 运行RED并实现无第三方依赖的rank/Spearman**

平分rank取平均名次；状态按`shadow_state_specs`连接，不读取query选择新状态。

- [x] **Step 3: GREEN**

Run: `python -m pytest tests/test_slow_fast_diagnostics.py -q`
Expected: PASS。

### Task 4: Source receiver-held-out校准与冻结bundle

**Files:**
- Create: `code/cvsrffi/slow_fast_calibration.py`
- Create: `code/scripts/calibrate_slow_fast_p05.py`
- Create: `tests/test_slow_fast_calibration.py`
- Modify: `code/cvsrffi/slow_fast_bundle.py`

**Interfaces:**
- Produces: `build_receiver_heldout_episodes(cache, ...)`、`calibrate_p05_gate(...)`和`cvs.slow_fast.p05.calibration.v1`JSON。
- Consumes: `GroundFeatureCache`、FILM bundle、frozen prototypes。

实现优化：冻结校准保持为独立、严格JSON，不回写或复制Phase1.5 bundle。该JSON只在地面侧读取；发布前仅把最终纯deployment参数抄入预登记row config。Phase2 runner明确拒绝`calibration_path`，避免运行时接触任何source receiver或episode统计。

- [x] **Step 1: 写episode物理ID互斥、K=10和receiver-held-out RED测试**

```python
episodes = build_receiver_heldout_episodes(cache, k_shot=10, seed=17)
assert all(ep.support_ids.isdisjoint(ep.query_ids) for ep in episodes)
assert all(torch.bincount(ep.support_labels).tolist() == [10] * classes for ep in episodes)
assert all(ep.heldout_receiver not in ep.fit_receivers for ep in episodes)
```

- [x] **Step 2: 实现episode builder并GREEN**

若cache无法为每类提供K=10与独立query，该receiver episode明确跳过并记录原因，不降低K。

- [x] **Step 3: 写冻结唯一FILM配置RED测试**

```python
calibration = calibrate_p05_gate(episodes, candidates)
assert calibration["candidate_id"] == "FAST_FILM_R8"
assert calibration["target_query_used"] is False
assert set(calibration) == CALIBRATION_SCHEMA_KEYS
```

- [x] **Step 4: 实现小型可审计校准**

使用source held-out query标记candidate是否改善，比较预注册规则，按worst-receiver mean、floor、侵入风险和计算量冻结一个配置；不保存样本级source派生物。

- [x] **Step 5: CLI负测与GREEN**

Run: `python -m pytest tests/test_slow_fast_calibration.py -q`
Expected: PASS。

### Task 5: Phase2 runner消费冻结deployment参数与审计schema

**Files:**
- Modify: `code/cvsrffi/stage2_slow_fast_runner.py`
- Modify: `code/cvsrffi/stage2_slow_fast_matrix.py`
- Modify: `tests/test_stage2_slow_fast_runner.py`
- Modify: `tests/test_stage2_slow_fast_matrix.py`
- Create: `configs/stage2_slow_fast_p05_template_s392003_20260825.json`

**Interfaces:**
- Consumes: row config中的纯`p05_*`deployment参数和`crossfit_seed`；禁止`calibration_path`。
- Produces: 仅`DA0_REG0/DA1_REG0`的正式P0.5receipt。

实现优化：正式P0.5先走单row runner，不为早期可证伪实验扩建9-row matrix；`stage2_slow_fast_matrix.py`继续只承担既有V2诊断矩阵。独立新capsule确认存在后再生成可运行config，不提交可能误启动的占位配置。

- [x] **Step 1: 写seed传播、schema和query只读RED测试**

```python
assert receipt["crossfit_seed"] == 392003
assert receipt["decision_rule"] == "frozen_prototype_cosine_slow_fast_v2"
assert receipt["adapter_schema"] == "cvs.cached_slow_fast.v2"
assert receipt["selection_schema"] == "cvs.slow_fast.p05.selection.v1"
assert receipt["prediction_schema"] == "raw_cosine.v1"
assert receipt["query_state_update_count"] == 0
```

- [x] **Step 2: 实现严格config allowlist与冻结calibration加载**

禁止source path、source校准path、truth path和旧V2 shadow网格进入正式P0.5 config。

- [x] **Step 3: 实现计算量拆分**

旧V2诊断fixture必须得到`21+183+76=280`；正式部署receipt只报告P0.5实际消费量。

- [x] **Step 4: GREEN与协议负测**

Run: `python -m pytest tests/test_stage2_slow_fast_runner.py tests/test_stage2_slow_fast_matrix.py -q`
Expected: PASS。

### Task 6: Capsule审计、N607地面校准和正式报告

**Files:**
- Create/Update: `E:\type10-7\automation_reports\CV-SincNet\cvs_slow_fast_p05_calibration_s392002_20260825_r1\report.md`
- Create/Update: `docs/experiments/cvs_slow_fast_p05_calibration_s392002_20260825_r1_report.md`
- Update: `analysis/slow_fast_p05_traceability_20260825.md`

**Interfaces:**
- Produces: source校准artifact、capsule可用性结论、可选的一次独立target结果。

- [x] **Step 1: 本地全量聚焦验证**

Run: `python -m pytest tests/test_slow_fast_*.py tests/test_stage2_slow_fast_*.py -q`
Expected: 0 failures。

- [ ] **Step 2: 唯一一次独立P0/P1正确性审查**

只接受会导致N607跑错、协议越权、覆盖输出、无法启动或非法prediction的发现。

- [ ] **Step 3: Git提交、push和远端OID核对**

只stage本轮代码、测试、配置和报告。

- [ ] **Step 4: 审计新capsule**

只读检查本地/远端现有`p2_min_v1`、`VALIDATED_ONCE`数据。不存在新receiver／seed时记录`MISSING_INDEPENDENT_TARGET_CAPSULE`，不得重用旧truth。

- [ ] **Step 5: N607最小发布**

执行预检、一次release SHA、远端编译、真实checkpoint无query smoke、source receiver-held-out校准。若新capsule合法存在，立即运行唯一`DA0_REG0/DA1_REG0`目标验证；否则在地面校准artifact闭合后停止性能声明。

- [ ] **Step 6: truth-last评分与报告**

报告逐scene mean/floor/worst-class、新类侵入、McNemar或bootstrap和P0停止条件。反向审计23项需求后提交、push并核对远端OID。
