# WISER-RF v2/P3-Primary实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将WISER-RF阶段A重构为P3/D92原生优化，并通过三场景pilot及门槛后的Target25大query矩阵验证跨receiver、跨seed稳定性。

**Architecture:** 复用现有`stage2_binova_d92.py`可微D92基础，先以精确D92同构测试锁定数值语义，再在独立`stage2_wiser_p3.py`中实现cross-fit风险、共享域流形、P3主导梯度投影和identity–FFT约束。旧WISER A只作为N1对照；N2～N6使用新的隔离训练入口，所有prediction完成后由独立truth-last scorer报告绝对query指标与变化。

**Tech Stack:** Python 3、PyTorch、NumPy、pytest、ADV3B02 checkpoint、`p2_min_v1`、N607 CUDA、Git。

**Spec:** `docs/superpowers/specs/2026-08-31-wiser-rf-p3-primary-design.md`

## Global Constraints

- 阶段A只读取旧类target support；训练、阶段选择、插值和超参数选择的`query_rows_used`必须为0。
- query只在模型与support状态完全冻结后由predictor打开；truth只由独立scorer在prediction完整后连接。
- 复用匹配的`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`，不得因方法或超参数变化重验数据。
- 正式可微D92在正常双模态、零identity、零FFT、极小范数和高条件数输入上与精确D92logits最大误差小于`1e-4`。
- Sinc、domain分支和冻结源分类头保持冻结；低性能不得作为技术停进程理由。
- pilot使用`rx_3_19__seed_713102__k_10__new_5`的三个`leo_*_weak`场景；只有科学门槛通过才运行Target25。
- Target25固定5receiver×5seed×`K10/new5`×3scene，共25outer/75scene unit；query使用完整包，不抽样。
- 本地测试串行使用`conda.exe run -n ssr-gpu`；N607只同步本地已验证Git提交对应的唯一release归档。
- 只stage本计划涉及的文件；不得加入现有`conversation_index/`或`local_artifacts/`。

---

## 文件职责

- `code/cvsrffi/stage2_binova_d92.py`：正式可微D92与精确old-only D92的共享数值语义。
- `code/cvsrffi/stage2_wiser_p3.py`：cross-fit、P3风险、域流形、梯度投影、互补性和插值纯函数。
- `code/cvsrffi/wiser_source_summary.py`：从现有量化摘要暴露保持域身份的`[domain,class,feature]`固定点。
- `code/cvsrffi/stage2_wiser_rf.py`：time-first渐进解冻参数白名单。
- `code/cvsrffi/stage2_wiser_runner.py`：N2～N6训练生命周期与support-only阶段选择。
- `code/cvsrffi/stage2_wiser_pilot.py`：N0～N6registry及pilot晋级门槛。
- `code/cvsrffi/stage2_wiser_scoring.py`：单row绝对query指标、NLL、per-class及配对delta/help-harm。
- `code/cvsrffi/stage2_wiser_target25.py`：Target25/K10扩展矩阵和跨单元晋级汇总。
- `code/scripts/run_stage2_wiser_pilot.py`：N0～N6真实checkpoint smoke、pilot与score-pilot命令。
- `code/scripts/run_stage2_wiser_target25.py`：Target25 prepare/run-shard/score-shard/analyze入口。
- `configs/wiser_rf_p3_primary_20260831.json`：冻结arm、loss、阶段、pilot和Target25配置。
- `tests/test_stage2_wiser_p3.py`：P3核心数学与边界条件。
- `tests/test_stage2_wiser_runner.py`：训练可达性、冻结、阶段与回滚。
- `tests/test_stage2_wiser_pilot.py`：新arm registry及三场景门槛。
- `tests/test_stage2_wiser_scoring.py`：truth-last绝对指标与配对变化。
- `tests/test_stage2_wiser_target25.py`：25outer/75scene覆盖及聚合门槛。
- `tests/test_run_stage2_wiser_pilot.py`、`tests/test_run_stage2_wiser_target25.py`：CLI、package路由与不可覆盖输出。

---

### Task 1: 锁定可微D92与精确D92数值同构

**Files:**
- Modify: `code/cvsrffi/stage2_binova_d92.py`
- Create: `tests/test_stage2_wiser_p3.py`

**Interfaces:**
- Consumes: `d92_geometry_features(identity160, fft96)`、`fit_differentiable_d92(rows, labels, old_class_count=6)`、`exact_d92_fit(identity160, fft96, labels, class_ids=range(6), old_class_count=6, seed=seed)`。
- Produces: `differentiable_old_d92_logits(fit_identity, fit_fft, fit_labels, eval_identity, eval_fft) -> torch.Tensor`。

- [ ] **Step 1: 编写五类输入同构失败测试**

```python
@pytest.mark.parametrize("case", ["normal", "zero_identity", "zero_fft", "tiny", "ill_conditioned"])
def test_differentiable_old_d92_matches_exact_logits(case: str) -> None:
    fit_id, fit_fft, labels, eval_id, eval_fft = make_d92_case(case)
    exact = exact_d92_fit(
        fit_id.detach().numpy(), fit_fft.detach().numpy(), labels.numpy(),
        class_ids=range(6), old_class_count=6, seed=713102, device="cpu",
    )
    expected = torch.tensor(exact.score(eval_id.numpy(), eval_fft.numpy()))
    actual = differentiable_old_d92_logits(
        fit_id, fit_fft, labels, eval_id, eval_fft,
    )
    assert torch.max(torch.abs(actual.double() - expected.double())).item() < 1.0e-4
```

- [ ] **Step 2: 运行测试并确认当前实现至少一类输入失败**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_p3.py::test_differentiable_old_d92_matches_exact_logits -q`

Expected: FAIL，差异超过`1e-4`或缺少`differentiable_old_d92_logits`。

- [ ] **Step 3: 实现统一的old-only可微评分桥**

```python
def differentiable_old_d92_logits(
    fit_identity: torch.Tensor,
    fit_fft: torch.Tensor,
    fit_labels: torch.Tensor,
    eval_identity: torch.Tensor,
    eval_fft: torch.Tensor,
) -> torch.Tensor:
    fit_rows = d92_geometry_features(fit_identity, fit_fft)
    eval_rows = d92_geometry_features(eval_identity, eval_fft)
    state = fit_differentiable_d92(fit_rows, fit_labels, old_class_count=6)
    return state.score(eval_rows)
```

若测试暴露OAS、jitter或dtype差异，只在`stage2_binova_d92.py`中把公式改为与`fit_old_only_erbt`实际消费的D92语义一致，不改变精确路径。

- [ ] **Step 4: 增加双模态同时退化拒绝和反向传播测试**

```python
def test_differentiable_d92_rejects_both_modalities_zero() -> None:
    with pytest.raises(BiNOVAD92Error, match="both modalities"):
        differentiable_old_d92_logits(zero_id, zero_fft, labels, zero_id[:2], zero_fft[:2])

def test_cross_fit_logits_backpropagate_to_fit_and_held_out_identity() -> None:
    logits = differentiable_old_d92_logits(fit_id, fit_fft, labels, eval_id, eval_fft)
    logits.square().mean().backward()
    assert fit_id.grad is not None and torch.isfinite(fit_id.grad).all()
    assert eval_id.grad is not None and torch.isfinite(eval_id.grad).all()
```

- [ ] **Step 5: 运行D92聚焦回归并提交**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_p3.py tests/test_stage2_binova_d92.py tests/test_stage2_sf_erbt_oldonly.py -q`

Expected: PASS。

Commit: `git add code/cvsrffi/stage2_binova_d92.py tests/test_stage2_wiser_p3.py && git commit -m "feat: align differentiable old D92"`

---

### Task 2: 实现五折P3主损失和类别风险/floor约束

**Files:**
- Create: `code/cvsrffi/stage2_wiser_p3.py`
- Modify: `tests/test_stage2_wiser_p3.py`

**Interfaces:**
- Consumes: Task 1的`differentiable_old_d92_logits`。
- Produces: `stratified_crossfit_indices`、`frozen_class_risk`、`cross_fitted_p3_loss`、`update_nonnegative_duals`。

- [ ] **Step 1: 编写确定性5-fold覆盖测试**

```python
def test_five_fold_crossfit_is_eight_fit_two_valid_per_class() -> None:
    labels = torch.arange(6).repeat_interleave(10)
    folds = stratified_crossfit_indices(labels, fold_count=5, seed=713102)
    assert len(folds) == 5
    seen = torch.zeros(60, dtype=torch.long)
    for fold in folds:
        seen[fold.validation_indices] += 1
        for class_id in range(6):
            assert int((labels[fold.fit_indices] == class_id).sum()) == 8
            assert int((labels[fold.validation_indices] == class_id).sum()) == 2
    assert torch.equal(seen, torch.ones_like(seen))
```

- [ ] **Step 2: 运行测试确认函数缺失**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_p3.py::test_five_fold_crossfit_is_eight_fit_two_valid_per_class -q`

Expected: FAIL with import error。

- [ ] **Step 3: 实现fold数据结构和固定physical-order折分**

```python
@dataclass(frozen=True)
class CrossFitFold:
    fit_indices: torch.Tensor
    validation_indices: torch.Tensor

def stratified_crossfit_indices(labels: torch.Tensor, *, fold_count: int, seed: int) -> Sequence[CrossFitFold]:
    values = labels.view(-1).long()
    classes = torch.unique(values, sorted=True)
    validation_parts: list[list[torch.Tensor]] = [[] for _ in range(fold_count)]
    for class_id in classes.tolist():
        indices = torch.where(values == int(class_id))[0]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 1_000_003 * int(class_id))
        order = torch.randperm(len(indices), generator=generator).to(indices.device)
        for fold_index, chunk in enumerate(torch.tensor_split(indices[order], fold_count)):
            validation_parts[fold_index].append(chunk)
    all_indices = torch.arange(len(values), device=values.device)
    folds = []
    for parts in validation_parts:
        validation = torch.sort(torch.cat(parts)).values
        fit_mask = torch.ones(len(values), dtype=torch.bool, device=values.device)
        fit_mask[validation] = False
        folds.append(CrossFitFold(all_indices[fit_mask], validation))
    return folds
```

- [ ] **Step 4: 编写P3风险、soft floor与dual更新测试**

```python
def test_p3_loss_reports_class_risk_and_penalizes_only_violations() -> None:
    result = cross_fitted_p3_loss(
        identity, fft, labels, folds=folds,
        baseline_class_risk=torch.full((6,), 0.5),
        class_duals=torch.ones(6), epsilon=torch.zeros(6), rho=2.0, beta=0.25, tau=0.1,
    )
    assert result.class_risk.shape == (6,)
    assert torch.all(result.violation >= 0)
    assert result.total >= result.mean_risk

def test_duals_remain_nonnegative() -> None:
    updated = update_nonnegative_duals(torch.tensor([0.1, 0.0]), torch.tensor([-1.0, 0.4]), rate=0.5)
    assert torch.equal(updated, torch.tensor([0.0, 0.2]))
```

- [ ] **Step 5: 实现`P3LossResult`与完整cross-fit损失**

```python
@dataclass(frozen=True)
class P3LossResult:
    total: torch.Tensor
    mean_risk: torch.Tensor
    soft_floor: torch.Tensor
    class_risk: torch.Tensor
    violation: torch.Tensor
    oof_logits: torch.Tensor
    oof_predictions: torch.Tensor
```

实现时每折只用fit索引拟合D92、只对validation索引求CE；最后按原row索引还原60条OOF logits。

- [ ] **Step 6: 运行P3核心测试并提交**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_p3.py -q`

Expected: PASS。

Commit: `git add code/cvsrffi/stage2_wiser_p3.py tests/test_stage2_wiser_p3.py && git commit -m "feat: add cross-fitted P3 risk"`

---

### Task 3: 实现共享域流形锚并移除VSW层级错配

**Files:**
- Modify: `code/cvsrffi/wiser_source_summary.py`
- Modify: `code/cvsrffi/stage2_wiser_p3.py`
- Modify: `tests/test_wiser_source_summary.py`
- Modify: `tests/test_stage2_wiser_p3.py`

**Interfaces:**
- Consumes: `QuantizedSourceSummary`现有dense或low-rank摘要。
- Produces: `domain_class_points() -> torch.Tensor[D,C,F]`、`infer_shared_domain_weights`、`shared_domain_manifold_loss`。

- [ ] **Step 1: 编写域身份保持测试**

```python
def test_dense_summary_exposes_domain_class_points() -> None:
    summary = load_quantized_source_summary(dense_summary_path)
    points = summary.domain_class_points()
    assert points.ndim == 3
    assert points.shape[1:] == (6, 160)
    assert torch.isfinite(points).all()
```

- [ ] **Step 2: 运行测试确认当前summary只暴露class-first虚拟点**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_wiser_source_summary.py::test_dense_summary_exposes_domain_class_points -q`

Expected: FAIL with missing method。

- [ ] **Step 3: 在摘要对象中实现保持域registry的重建**

```python
def domain_class_points(self) -> torch.Tensor:
    if self.direct_points is not None:
        return F.normalize(self.direct_points.detach().float(), dim=-1).transpose(0, 1)
    residual = torch.einsum("dcr,crf->dcf", self.coefficients.float(), self.basis.float())
    return F.normalize(self.centers.float()[None, :, :] + residual, dim=-1).detach()
```

低秩格式的center domain补一行零残差，使返回domain数与封存registry一致。

- [ ] **Step 4: 编写共享单纯形权重测试**

```python
def test_shared_domain_weights_are_one_simplex_for_all_classes() -> None:
    weights = infer_shared_domain_weights(target_features, labels, source_points, steps=80, learning_rate=0.1, l2=0.01)
    assert weights.shape == (source_points.shape[0],)
    assert torch.all(weights >= 0)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1.0e-6)
    loss = shared_domain_manifold_loss(target_features, labels, source_points, weights)
    assert loss.ndim == 0 and torch.isfinite(loss)
```

- [ ] **Step 5: 实现固定`w_t`推断和稳健流形损失**

权重用softmax参数化并在冻结初始特征上优化一次；返回值`detach()`。训练期间只有target类中心对model保留梯度，source points和weights均冻结。

- [ ] **Step 6: 运行摘要/P3测试并提交**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_wiser_source_summary.py tests/test_stage2_wiser_p3.py -q`

Expected: PASS。

Commit: `git add code/cvsrffi/wiser_source_summary.py code/cvsrffi/stage2_wiser_p3.py tests/test_wiser_source_summary.py tests/test_stage2_wiser_p3.py && git commit -m "feat: add shared domain manifold"`

---

### Task 4: 实现P3梯度投影、互补性与激活安全

**Files:**
- Modify: `code/cvsrffi/stage2_wiser_p3.py`
- Modify: `tests/test_stage2_wiser_p3.py`

**Interfaces:**
- Produces: `project_auxiliary_gradients`、`identity_fft_diagnostics`、`identity_fft_penalties`。

- [ ] **Step 1: 编写冲突梯度投影测试**

```python
def test_projected_auxiliary_gradient_cannot_oppose_primary() -> None:
    primary = (torch.tensor([1.0, 0.0]),)
    auxiliary = (torch.tensor([-1.0, 2.0]),)
    projected, audit = project_auxiliary_gradients(primary, auxiliary)
    assert torch.dot(primary[0], projected[0]) >= -1.0e-7
    assert audit["raw_dot"] < 0 and audit["projected_dot"] >= -1.0e-7
```

- [ ] **Step 2: 运行测试确认投影函数缺失**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_p3.py::test_projected_auxiliary_gradient_cannot_oppose_primary -q`

Expected: FAIL with import error。

- [ ] **Step 3: 实现None梯度安全的投影与加权合成**

```python
def project_auxiliary_gradients(primary, auxiliary, *, eps=1.0e-12):
    primary_safe = tuple(torch.zeros_like(a) if p is None else p for p, a in zip(primary, auxiliary))
    auxiliary_safe = tuple(torch.zeros_like(p) if a is None else a for p, a in zip(primary_safe, auxiliary))
    raw_dot = sum((p * a).sum() for p, a in zip(primary_safe, auxiliary_safe))
    primary_norm = sum(p.square().sum() for p in primary_safe)
    coefficient = torch.minimum(raw_dot, raw_dot.new_zeros(())) / (primary_norm + float(eps))
    projected = tuple(a - coefficient * p for p, a in zip(primary_safe, auxiliary_safe))
    projected_dot = sum((p * a).sum() for p, a in zip(primary_safe, projected))
    return projected, {"raw_dot": float(raw_dot.detach()), "projected_dot": float(projected_dot.detach())}
```

- [ ] **Step 4: 编写互补性、条件数和zero-id测试**

```python
def test_identity_fft_diagnostics_detect_duplication_and_zero_identity() -> None:
    diagnostics = identity_fft_diagnostics(identity, fft, labels, zero_tolerance=1.0e-12)
    assert diagnostics.zero_identity_count == 1
    assert diagnostics.cross_covariance_frobenius >= 0
    assert len(diagnostics.canonical_correlations) == 5
    assert diagnostics.joint_condition_number >= 1
```

- [ ] **Step 5: 实现`IdentityFFTDiagnostics`及只惩罚增加量的loss**

```python
@dataclass(frozen=True)
class IdentityFFTDiagnostics:
    zero_identity_count: int
    identity_norm_q01: float
    dead_activation_ratio: float
    cross_covariance_frobenius: float
    joint_condition_number: float
    canonical_correlations: tuple[float, float, float, float, float]
```

`identity_fft_penalties`返回`duplication_loss`和`energy_loss`，并接受冻结baseline的交叉协方差范数。

- [ ] **Step 6: 运行P3测试并提交**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_p3.py -q`

Expected: PASS。

Commit: `git add code/cvsrffi/stage2_wiser_p3.py tests/test_stage2_wiser_p3.py && git commit -m "feat: guard P3 gradient and modality geometry"`

---

### Task 5: 接入time-first训练、阶段分支和support-only插值

**Files:**
- Modify: `code/cvsrffi/stage2_wiser_rf.py`
- Modify: `code/cvsrffi/stage2_wiser_runner.py`
- Modify: `tests/test_stage2_wiser_rf.py`
- Modify: `tests/test_stage2_wiser_runner.py`

**Interfaces:**
- Consumes: Tasks 2～4的P3损失、流形、投影和诊断。
- Produces: `WISERP3TrainingConfig`、`train_wiser_p3_arm`、`select_support_safe_interpolation`。

- [ ] **Step 1: 编写time-first冻结白名单测试**

```python
def test_p3_stage1_opens_time_late_and_keeps_frequency_frozen() -> None:
    audit = configure_p3_time_first_update(model, branch="stage1_time")
    assert any(name.startswith("id_backbone.t3.") for name in audit.trainable_parameter_names)
    assert not any(name.startswith("id_backbone.f") for name in audit.trainable_parameter_names)
    assert audit.sinc_frozen and audit.source_head_frozen and audit.domain_branch_frozen
```

- [ ] **Step 2: 运行冻结测试并确认新入口缺失**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_rf.py::test_p3_stage1_opens_time_late_and_keeps_frequency_frozen -q`

Expected: FAIL with import error。

- [ ] **Step 3: 实现`stage1_time/stage2_time/stage2_frequency/stage2_joint/stage3`白名单**

保留旧`configure_progressive_identity_update`供N1使用；新入口不得改变旧A路径。

- [ ] **Step 4: 编写N2～N6loss可达性、query零使用和梯度审计测试**

```python
@pytest.mark.parametrize("arm", ["N2", "N3", "N4", "N5", "N6"])
def test_p3_arm_uses_only_support_and_refreezes(arm: str) -> None:
    audit = train_wiser_p3_arm(model, support_iq, labels, support_tokens=tokens, source_summary=summary, arm=arm, config=tiny_config)
    assert audit.query_rows_used == 0
    assert audit.optimizer_steps > 0
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
```

- [ ] **Step 5: 实现新的训练配置和arm递增机制**

```python
@dataclass(frozen=True)
class WISERP3TrainingConfig:
    fold_count: int = 5
    stage_steps: tuple[int, int, int] = (1500, 2000, 2500)
    diagnostic_interval: int = 100
    risk_rho: float = 2.0
    floor_beta: float = 0.25
    floor_tau: float = 0.1
    interpolation_grid: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.0)
    seed: int = 713102
```

N2只用P3 mean risk；N3增加类别风险/floor；N4增加共享域流形；N5增加辅助梯度投影；N6增加互补和energy约束。

- [ ] **Step 6: 编写内部三分支选择和`alpha=0`回退测试**

```python
def test_stage_branch_selection_is_support_only_and_falls_back_to_alpha_zero() -> None:
    result = select_support_safe_interpolation(base_state, candidate_state, evaluator=always_unsafe, grid=(1.0, 0.5, 0.0))
    assert result.alpha == 0.0
    assert result.query_rows_used == 0
```

- [ ] **Step 7: 实现固定字典序分支选择和插值，并记录P3轨迹**

分支排序键固定为`(-oof_p3_ba,-oof_p3_floor,joint_condition_number,branch_name)`；同分时选择参数自由度更低的分支。

- [ ] **Step 8: 运行runner回归并提交**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_rf.py tests/test_stage2_wiser_runner.py tests/test_stage2_wiser_p3.py -q`

Expected: PASS。

Commit: `git add code/cvsrffi/stage2_wiser_rf.py code/cvsrffi/stage2_wiser_runner.py tests/test_stage2_wiser_rf.py tests/test_stage2_wiser_runner.py && git commit -m "feat: train WISER P3-primary arms"`

---

### Task 6: 扩展N0～N6pilot和正式三场景门槛

**Files:**
- Modify: `code/cvsrffi/stage2_wiser_pilot.py`
- Modify: `code/scripts/run_stage2_wiser_pilot.py`
- Modify: `tests/test_stage2_wiser_pilot.py`
- Modify: `tests/test_run_stage2_wiser_pilot.py`
- Create: `configs/wiser_rf_p3_primary_20260831.json`

**Interfaces:**
- Consumes: N1旧`train_wiser_arm`和N2～N6新`train_wiser_p3_arm`。
- Produces: `P3_ARMS`、`formal_p3_primary_decision(rows, arm)`、支持`smoke/pilot/score-pilot`的N0～N6CLI。

- [ ] **Step 1: 编写arm registry和baseline强制包含测试**

```python
def test_p3_pilot_registry_is_n0_through_n6() -> None:
    assert P3_ARMS == ("N0", "N1", "N2", "N3", "N4", "N5", "N6")
    assert normalize_p3_arms(("N4",)) == ("N0", "N4")
```

- [ ] **Step 2: 编写新P3门槛正反测试**

```python
def test_p3_gate_requires_cross_scene_floor_and_flip_safety() -> None:
    decision = formal_p3_primary_decision(make_three_scene_rows(delta_ba=(0.04, 0.03, 0.035), floor=(0.0, 0.01, 0.0), net_flips=(3, 2, -1)), arm="N6")
    assert decision["passed"] is True
    failed = formal_p3_primary_decision(make_three_scene_rows(delta_ba=(0.08, -0.01, 0.05), floor=(0.0, -0.01, 0.0), net_flips=(3, -2, 3)), arm="N6")
    assert failed["passed"] is False
```

- [ ] **Step 3: 运行pilot测试确认旧A/B registry不满足**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_pilot.py tests/test_run_stage2_wiser_pilot.py -q`

Expected: FAIL on new imports/registry。

- [ ] **Step 4: 实现不覆盖旧schema的新P3 pilot schema**

使用`cvs.phase2.wiser_rf.p3_primary.pilot.v1`，每个arm fresh-load同一checkpoint；N0不训练，N1走旧A，N2～N6走新训练入口。保存的support audit必须先于第一次query package读取。

- [ ] **Step 5: 写入冻结配置**

```json
{
  "schema": "cvs.phase2.wiser_rf.p3_primary.config.v1",
  "protocol_schema": "p2_min_v1",
  "phase2_data_status": "VALIDATED_ONCE",
  "pilot_outer_key": "rx_3_19__seed_713102__k_10__new_5",
  "arms": ["N0", "N1", "N2", "N3", "N4", "N5", "N6"],
  "scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
  "fold_count": 5,
  "query_policy": "full_package_read_only_after_support_freeze"
}
```

- [ ] **Step 6: 运行CLI与pilot测试并提交**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_pilot.py tests/test_run_stage2_wiser_pilot.py tests/test_stage2_wiser_runner.py -q`

Expected: PASS。

Commit: `git add code/cvsrffi/stage2_wiser_pilot.py code/scripts/run_stage2_wiser_pilot.py configs/wiser_rf_p3_primary_20260831.json tests/test_stage2_wiser_pilot.py tests/test_run_stage2_wiser_pilot.py && git commit -m "feat: add WISER P3-primary pilot"`

---

### Task 7: 输出详细query绝对指标、delta和help/harm

**Files:**
- Modify: `code/cvsrffi/stage2_wiser_scoring.py`
- Modify: `tests/test_stage2_wiser_scoring.py`

**Interfaces:**
- Produces: `_probe_metrics(prediction, logits, truth)`、`compare_wiser_score_rows(control, candidate)`。

- [ ] **Step 1: 编写Accuracy/BA/floor/NLL绝对值测试**

```python
def test_truth_last_score_reports_absolute_query_metrics_and_nll(tmp_path: Path) -> None:
    result = score_wiser_predictions(pred_path, receipt_path, truth_path)
    p3 = result["probes"]["P3_OLD_D92"]
    assert set(("accuracy", "balanced_accuracy", "floor", "nll", "per_class_accuracy")) <= set(p3)
    assert result["query_rows"] == 12
```

- [ ] **Step 2: 编写配对百分点和翻转测试**

```python
def test_paired_comparison_reports_pp_and_help_harm() -> None:
    comparison = compare_wiser_score_rows(control_row, candidate_row)
    assert comparison["accuracy_delta_pp"] == pytest.approx(8.333333, abs=1.0e-5)
    assert comparison["help_count"] == 2
    assert comparison["harm_count"] == 1
    assert comparison["unchanged_count"] == 9
```

- [ ] **Step 3: 运行测试确认当前scorer缺NLL与配对比较**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_scoring.py -q`

Expected: FAIL on missing fields/functions。

- [ ] **Step 4: 保存完整logits并实现稳定NLL**

scorer必须从prediction NPZ读取`p1_logits/p2_logits/p3_logits`，用`logsumexp`计算NLL；不能重算模型或打开support。

- [ ] **Step 5: 实现同token、同truth、同scene的配对比较**

comparison输出绝对control/candidate值、`delta_pp`、per-class delta、help/harm/unchanged和正向净翻转。任何token registry不一致直接拒绝。

- [ ] **Step 6: 运行scorer与pilot回归并提交**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_scoring.py tests/test_stage2_wiser_pilot.py tests/test_run_stage2_wiser_pilot.py -q`

Expected: PASS。

Commit: `git add code/cvsrffi/stage2_wiser_scoring.py tests/test_stage2_wiser_scoring.py && git commit -m "feat: report paired WISER query gains"`

---

### Task 8: 实现Target25大query矩阵与跨单元确认门槛

**Files:**
- Create: `code/cvsrffi/stage2_wiser_target25.py`
- Create: `code/scripts/run_stage2_wiser_target25.py`
- Create: `tests/test_stage2_wiser_target25.py`
- Create: `tests/test_run_stage2_wiser_target25.py`

**Interfaces:**
- Consumes: `stage2_bisage_target125.canonical_target125_rows()`、pilot冠军marker和Task 7配对score rows。
- Produces: `canonical_target25_rows`、`build_wiser_target25_manifest`、`target25_promotion_decision`及5个CLI子命令。

- [ ] **Step 1: 编写25outer/75scene精确覆盖测试**

```python
def test_target25_is_all_receivers_all_seeds_fixed_k10_new5() -> None:
    rows = canonical_target25_rows()
    assert len(rows) == 25
    assert {row["receiver"] for row in rows} == {"20-1", "3-19", "7-14", "7-7", "8-8"}
    assert {row["seed"] for row in rows} == {713102, 713103, 713104, 713105, 713106}
    assert {(row["k_shot"], row["new_class_count"]) for row in rows} == {(10, 5)}
    assert sum(len(row["scenarios"]) for row in rows) == 75
```

- [ ] **Step 2: 运行测试确认模块不存在**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_target25.py -q`

Expected: FAIL with import error。

- [ ] **Step 3: 实现历史Target125的确定性K10/new5过滤与不可覆盖manifest**

manifest保留原`capsule_id/split_id/packages/truth_sidecar`，只新增新的output root、冠军commit/arm和shard index；不得打开query或truth。

- [ ] **Step 4: 编写Target25确认门槛测试**

```python
def test_target25_gate_requires_receiver_seed_and_scenario_coverage() -> None:
    decision = target25_promotion_decision(make_75_safe_paired_rows())
    assert decision["passed"] is True
    unsafe = make_75_safe_paired_rows(low_elev_delta=-0.01)
    assert target25_promotion_decision(unsafe)["passed"] is False
```

- [ ] **Step 5: 实现75单元聚合**

输出overall median、每个scenario median、10%分位、overall/low-elev floor median、5receiver和5seed正向覆盖、help/harm总计及配对bootstrap区间。固定seed生成bootstrap，不参与训练。

- [ ] **Step 6: 实现`prepare/run-shard/score-shard/analyze`CLI**

`run-shard`只生成prediction；`score-shard`独立读取truth；`analyze`要求25outer×3scene的N0与冠军全部存在。任何缺格不得标记`ANALYZED`。

- [ ] **Step 7: 运行Target25聚焦测试并提交**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_target25.py tests/test_run_stage2_wiser_target25.py tests/test_stage2_wiser_scoring.py -q`

Expected: PASS。

Commit: `git add code/cvsrffi/stage2_wiser_target25.py code/scripts/run_stage2_wiser_target25.py tests/test_stage2_wiser_target25.py tests/test_run_stage2_wiser_target25.py && git commit -m "feat: add WISER Target25 validation"`

---

### Task 9: 完成本地集成验证、追踪更新和正式代码提交

**Files:**
- Modify: `docs/experiments/wiser_rf_p3_primary_20260831_traceability.md`
- Create: `automation_reports/CV-SincNet/wiser_rf_p3_primary_hist_e0_20260831_v1/report.md`（根目录正式报告，随后镜像到Git承载面）

**Interfaces:**
- Consumes: Tasks 1～8全部实现。
- Produces: 可发布Git提交、真实checkpoint无query smoke命令和最小预登记报告。

- [ ] **Step 1: 运行全部WISER聚焦测试**

Run: `conda.exe run -n ssr-gpu python -m pytest tests/test_stage2_wiser_p3.py tests/test_wiser_source_summary.py tests/test_stage2_wiser_rf.py tests/test_stage2_wiser_runner.py tests/test_stage2_wiser_pilot.py tests/test_stage2_wiser_scoring.py tests/test_stage2_wiser_target25.py tests/test_run_stage2_wiser_pilot.py tests/test_run_stage2_wiser_target25.py tests/test_stage2_binova_d92.py tests/test_stage2_sf_erbt_oldonly.py -q`

Expected: PASS with zero failures。

- [ ] **Step 2: 运行语法、CLI和diff检查**

Run: `conda.exe run -n ssr-gpu python -m py_compile code/cvsrffi/stage2_wiser_p3.py code/cvsrffi/stage2_wiser_runner.py code/cvsrffi/stage2_wiser_pilot.py code/cvsrffi/stage2_wiser_scoring.py code/cvsrffi/stage2_wiser_target25.py code/scripts/run_stage2_wiser_pilot.py code/scripts/run_stage2_wiser_target25.py`

Run: `conda.exe run -n ssr-gpu python code/scripts/run_stage2_wiser_pilot.py --help`

Run: `git diff --check`

Expected: all exit 0。

- [ ] **Step 3: 反向更新追踪表**

把每个已实现ID改为`verified`并记录具体测试名；`SUM-01`保持`deferred`，`STAGEB-01`保持`blocked`，真实实验项保持`pending`。

- [ ] **Step 4: 进行一次独立P0/P1正确性审查**

审查范围仅包括会导致真实实验跑错、query越权、输出覆盖、错误停止、不能启动或不能产生合法prediction的问题。若发现P0/P1，定点修复后只复审原问题一次。

- [ ] **Step 5: 创建最小预登记报告**

报告冻结run ID、Git commit、N0～N6矩阵、pilot outer、三个场景、命令、`ssr-gpu`、CWD、checkpoint/manifest/source-summary、run/log root、具体GPU、prediction和score artifact、技术停止规则及科学晋级门槛。

- [ ] **Step 6: 精确提交、push并核对远端OID**

Run: `git status -sb`

Run: `git add code/cvsrffi/stage2_binova_d92.py code/cvsrffi/stage2_wiser_p3.py code/cvsrffi/wiser_source_summary.py code/cvsrffi/stage2_wiser_rf.py code/cvsrffi/stage2_wiser_runner.py code/cvsrffi/stage2_wiser_pilot.py code/cvsrffi/stage2_wiser_scoring.py code/cvsrffi/stage2_wiser_target25.py code/scripts/run_stage2_wiser_pilot.py code/scripts/run_stage2_wiser_target25.py configs/wiser_rf_p3_primary_20260831.json tests/test_stage2_wiser_p3.py tests/test_wiser_source_summary.py tests/test_stage2_wiser_rf.py tests/test_stage2_wiser_runner.py tests/test_stage2_wiser_pilot.py tests/test_stage2_wiser_scoring.py tests/test_stage2_wiser_target25.py tests/test_run_stage2_wiser_pilot.py tests/test_run_stage2_wiser_target25.py docs/experiments/wiser_rf_p3_primary_20260831_traceability.md`

Run: `git commit -m "feat: implement WISER P3-primary"`

Run: `git ls-remote origin refs/heads/codex/binova-d92-20260829`

Expected: remote OID与`git rev-parse HEAD`完全一致。

---

### Task 10: 发布N607 pilot、独立评分并按门槛放量

**Files:**
- Modify: 根`automation_reports/CV-SincNet/wiser_rf_p3_primary_hist_e0_20260831_v1/report.md`
- Modify: Git镜像实验报告与追踪表

**Interfaces:**
- Consumes: Task 9固定的commit、release和预登记报告。
- Produces: pilot `ANALYZED`结果；通过时继续Target25，否则发布科学负结果。

- [ ] **Step 1: 运行N607只读preflight并盘点GPU训练任务**

Run: `powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`

Expected: ordinary `N607`身份、服务器时间、项目根和GPU可见；每GPU不超过用户授权的3个训练实验。

- [ ] **Step 2: 构建唯一release归档并完成一次本地/远端SHA比对**

归档只包含Git提交已固定的运行所需代码/config。通过`scp`同步到唯一release路径，比较归档SHA一次，远端解压后只运行一次`py_compile`。

- [ ] **Step 3: 运行真实ADV3B02 checkpoint无query smoke**

smoke必须报告`query_opened=false/query_rows_used=0`、N0～N6可构造、D92五类同构和所有模型参数在结束时冻结；PASS后立即继续pilot。

- [ ] **Step 4: 启动不可覆盖pilot并回读进程绑定**

按资源盘点为arms分配具体GPU；每张卡总训练实验不超过3。启动后一次性核对主PID、CWD、cmdline、物理GPU、run root和log增长。不得因低性能停止。

- [ ] **Step 5: prediction完整后独立运行score-pilot**

确认7arms×3scenes共21组prediction及receipt完整、`truth_opened=false`，再另起scorer连接truth。输出每row query count、Accuracy/BA/floor/NLL、per-class、help/harm、P1/P2/P3和资源数据。

- [ ] **Step 6: 应用pilot科学门槛**

若所有正式候选失败，记录`next_experiment_authorized=false`，不启动Target25/Stage B；完成报告、commit、push和远端OID核验。若至少一个候选通过，按预登记顺序选择门槛内P3 BA中位提升最高者作为唯一Target25冠军。

- [ ] **Step 7: 启动Target25的不可覆盖shards**

使用冠军与N0覆盖25outer×3scene；完整query package不采样。每个shard prediction闭合后独立`score-shard`，全部75对齐单元完成后运行`analyze`。

- [ ] **Step 8: 发布Target25详细结果和后续决定**

报告至少包含实际query总数、每receiver/seed/scene绝对Accuracy/BA/floor/NLL、百分点变化、per-class、help/harm、median/10%分位/worst-scenario、配对bootstrap区间和资源数据。只有Target25门槛通过才授权K10扩展；Stage B仍等待K10扩展结果。

- [ ] **Step 9: 提交正式报告、push并核对远端OID**

精确stage Git镜像报告和追踪表，提交后自动push；独立比较远端branch OID与本地`HEAD`。完成后暂停相关监控心跳。

---

## 计划自检

- 规格第1～14节均映射到Task 1～10。
- P3数值同构在Task 1先闭合，避免后续损失建立在近似语义上。
- query绝对准确率、百分点变化、大query和多scene要求分别由Task 7、Task 8和Task 10闭合。
- 低秩源类内协方差摘要明确不在首轮pilot实现范围，追踪状态保持`deferred`；现有26×6聚合中心足以验证共享域流形。
- 阶段B不在当前实现任务中，只有K10扩展通过后才解除`blocked`。
- 所有新接口在产生它们的Task中定义，后续Task只消费已定义接口。
