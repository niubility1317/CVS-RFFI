# Phase1 HCF-DG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 严格实现HCF-DG V1/V2/V3、A0–A12分阶段矩阵、source-only选择、四场景闭合和N607正式发布入口。

**Architecture:** 在独立`phase1_hcfdg`包中复用一次现有`lite_d`identity backbone，新增48D环境因子编码器、矩形episode采样、LODO分类、rank-4公共—特定头、反事实环境传输、HDRO、诊断指标和单前向星地增强。旧`train_ssdg.py`、ADV3B02和ADV3B03入口保持原行为；新launcher直接驱动HCF-DG trainer并复用现有WiSig source-role builder与最终四场景评估器。

**Tech Stack:** Python3.10+、PyTorch、NumPy、pytest、现有WiSig/LEO增强代码、Git、N607 CUDA环境。

**Spec:** `docs/superpowers/specs/2026-08-30-phase1-hcfdg-design.md`

## Global Constraints

- 只使用`Dataset_WigSig/ManySig.pkl`中的source receivers`1,3,4,6,8`和day`1,2,3`训练。
- source角色固定为`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，四类物理样本ID两两不交。
- 固定筛选seed为`392001/392002/392003`。
- HCF-DG主batch固定为`6TX×4domain×4sample=96`。
- 星地训练为每个样本独立`p=0.30`选择`mixed_orbit`、否则clean；每个batch只执行一次共享身份主干前向。
- V1预算为4000 optimizer updates；V2预算为6300 updates。
- HCF-DG不读取Phase2 capsule、support、query、truth、prototype或split。
- 最终checkpoint必须分别闭合clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。
- 代码测试串行使用`conda.exe run -n ssr-gpu`，不得并发启动Conda包装命令。
- 所有文件先在本地工作树修改和验证；N607只接收已提交、已推送的release归档。
- 每个实现任务结束后更新`analysis/phase1_hcfdg_traceability.md`对应ID。

## File Structure

- Create `code/cvsrffi/phase1_hcfdg/__init__.py`: 只导出稳定公共接口。
- Create `code/cvsrffi/phase1_hcfdg/config.py`: 冻结候选、阶段、预算、seed和矩阵schema。
- Create `code/cvsrffi/phase1_hcfdg/sampler.py`: source-only矩形episode与fold选择。
- Create `code/cvsrffi/phase1_hcfdg/satellite.py`: 单视图`mixed_orbit`替换和channel factor标签。
- Create `code/cvsrffi/phase1_hcfdg/model.py`: 单主干、环境编码器、公共—特定头、对抗头和反事实传输。
- Create `code/cvsrffi/phase1_hcfdg/losses.py`: LODO、内容条件原型、CF、HDRO、CSD和FAC损失。
- Create `code/cvsrffi/phase1_hcfdg/trainer.py`: Stage0–4训练、优化日程、冻结、checkpoint和资源遥测。
- Create `code/cvsrffi/phase1_hcfdg/metrics.py`: 泄漏、漂移、margin、反事实和same-row指标。
- Create `code/scripts/launch_phase1_hcfdg_matrix_20260830.py`: A0–A12矩阵、不可覆盖路径、并发调度和四场景评估。
- Create `code/tests/phase1_hcfdg/`: 按模块组织聚焦测试。
- Modify `analysis/phase1_hcfdg_traceability.md`: 每组实现后写入文件和验证证据。
- Create `automation_reports/CV-SincNet/<run-id>/report.md`: 正式实验最小预登记与终态结果。

---

### Task 1: Freeze Configuration and Matrix Schema

**Files:**

- Create: `code/cvsrffi/phase1_hcfdg/__init__.py`
- Create: `code/cvsrffi/phase1_hcfdg/config.py`
- Test: `code/tests/phase1_hcfdg/test_config.py`
- Modify: `analysis/phase1_hcfdg_traceability.md`

**Interfaces:**

- Produces: `HCFDGConfig`、`StageBudget`、`MatrixRow`、`candidate_config(candidate_id)`、`quick_screen_rows(folds)`、`deep_screen_rows(folds)`、`residual_rows(folds)`。
- Consumes: 无生产代码依赖；只使用Python标准库。

- [ ] **Step 1: Write failing tests for frozen defaults and all candidate transitions**

```python
def test_quick_screen_uses_frozen_three_seeds_and_36_rows():
    rows = quick_screen_rows((1, 8))
    assert {row.seed for row in rows} == {392001, 392002, 392003}
    assert {row.candidate_id for row in rows} == {f"A{i}" for i in range(6)}
    assert len(rows) == 36
    assert all(row.optimizer_updates == 4000 for row in rows)

def test_candidate_activation_is_cumulative_and_report_ordered():
    assert candidate_config("A4").use_lodo is True
    assert candidate_config("A4").use_csd is False
    assert candidate_config("A9").use_content_conditioning is True
    assert candidate_config("A9").use_hdro is True
    assert candidate_config("A12").residual_mode == "phasedelta_dsq"
```

- [ ] **Step 2: Run RED**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_config.py -q`

Expected: collection fails because`cvsrffi.phase1_hcfdg.config`does not exist.

- [ ] **Step 3: Implement immutable schemas and exact candidate factory**

```python
@dataclass(frozen=True)
class HCFDGConfig:
    candidate_id: str
    optimizer_updates: int
    use_dual_control: bool = False
    use_environment_encoder: bool = False
    use_rectangular_batch: bool = False
    use_lodo: bool = False
    use_csd: bool = False
    counterfactual_mode: str = "off"
    use_hdro: bool = False
    use_content_conditioning: bool = False
    residual_mode: str = "off"

@dataclass(frozen=True)
class MatrixRow:
    candidate_id: str
    heldout_receiver: int
    seed: int
    optimizer_updates: int
    gpu: int | None = None
```

Implement explicit candidate definitions forA0–A12; reject unknown IDs and rejectA10–A12 construction unless the caller passes`v2_passed=True`.

- [ ] **Step 4: Run GREEN and full config tests**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_config.py -q`

Expected: all config tests pass.

- [ ] **Step 5: Update traceability HCF-009/HCF-013/HCF-018/HCF-022/HCF-023/HCF-024/HCF-025/HCF-033**

Set implemented rows to`implemented`and record the test path; keep them below`verified`until launcher integration proves reachability.

- [ ] **Step 6: Commit**

Run: `git.exe add -- code/cvsrffi/phase1_hcfdg/__init__.py code/cvsrffi/phase1_hcfdg/config.py code/tests/phase1_hcfdg/test_config.py analysis/phase1_hcfdg_traceability.md`

Run: `git.exe commit -m "feat: define HCF-DG candidates and budgets"`

---

### Task 2: Implement Source-Only Fold Selection and Rectangular Episodes

**Files:**

- Create: `code/cvsrffi/phase1_hcfdg/sampler.py`
- Test: `code/tests/phase1_hcfdg/test_sampler.py`
- Modify: `analysis/phase1_hcfdg_traceability.md`

**Interfaces:**

- Consumes: NumPy arrays`tx_ids`、`receiver_ids`、`day_ids`、`channel_ids`、`q_phys`。
- Produces: `select_center_and_far_receivers(q_phys, receiver_ids)`、`EpisodeDescriptor`、`HCFDGEpisodeBatchSampler`。

- [ ] **Step 1: Write failing tests for center/far selection, 96-sample geometry and deterministic replay**

```python
def test_episode_has_six_tx_four_domains_and_four_samples_per_cell():
    sampler = HCFDGEpisodeBatchSampler(metadata, seed=392002)
    episode = next(iter(sampler))
    assert len(episode.indices) == 96
    assert len(set(episode.tx_ids)) == 6
    assert len(set(episode.domain_ids)) == 4
    assert len(set(episode.receiver_ids)) >= 3
    assert episode.query_mask.sum() > 0
    assert not np.any(episode.support_mask & episode.query_mask)

def test_center_and_far_receivers_use_source_qphys_only():
    center, far = select_center_and_far_receivers(q_phys, receiver_ids)
    assert (center, far) == (3, 8)
```

- [ ] **Step 2: Run RED**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_sampler.py -q`

Expected: import failure for missing sampler module.

- [ ] **Step 3: Implement exact source-only selector and episode sampler**

```python
@dataclass(frozen=True)
class EpisodeDescriptor:
    indices: tuple[int, ...]
    tx_ids: tuple[int, ...]
    receiver_ids: tuple[int, ...]
    day_ids: tuple[int, ...]
    channel_ids: tuple[int, ...]
    domain_ids: tuple[int, ...]
    episode_type: str
    query_domain: int
    support_mask: np.ndarray
    query_mask: np.ndarray
    valid_tx_mask: np.ndarray
    episode_seed: int
```

Use episode probabilities`0.65/0.225/0.125`; sample complete cells first, then mark incomplete cells without borrowing anotherTX. Make`set_epoch(epoch)`derive a stable generator from`seed+epoch*1000003`.

- [ ] **Step 4: Run GREEN plus existing balanced sampler regression**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_sampler.py code\tests\test_balanced_tx_rx_sampler.py -q`

Expected: all tests pass and legacy sampler behavior remains unchanged.

- [ ] **Step 5: Update traceability HCF-011/HCF-012 and commit**

Run: `git.exe add -- code/cvsrffi/phase1_hcfdg/sampler.py code/tests/phase1_hcfdg/test_sampler.py analysis/phase1_hcfdg_traceability.md`

Run: `git.exe commit -m "feat: add HCF-DG source episodes"`

---

### Task 3: Implement Single-Forward Mixed-Orbit Batch Construction

**Files:**

- Create: `code/cvsrffi/phase1_hcfdg/satellite.py`
- Test: `code/tests/phase1_hcfdg/test_satellite.py`
- Modify: `analysis/phase1_hcfdg_traceability.md`

**Interfaces:**

- Consumes: cleanIQ tensor、现有`apply_satellite_channel_batch`兼容augmentor、PyTorch generator。
- Produces: `ChannelFactors`、`SingleViewBatch`、`build_single_view_batch(x, augmentor, generator, p_sat=0.30)`。

- [ ] **Step 1: Write failing tests for one-view shape, deterministic Bernoulli mask and factor labels**

```python
def test_single_view_batch_never_concatenates_samples():
    result = build_single_view_batch(x, fake_augmentor, generator, p_sat=0.30)
    assert result.iq.shape == x.shape
    assert result.satellite_mask.shape == (x.shape[0],)
    assert result.channel_factors.shape[0] == x.shape[0]

def test_empirical_satellite_fraction_tracks_point_three():
    count = sum(build_single_view_batch(x, aug, g, 0.30).satellite_mask.sum().item() for _ in range(200))
    assert abs(count / (200 * len(x)) - 0.30) < 0.02
```

- [ ] **Step 2: Run RED**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_satellite.py -q`

- [ ] **Step 3: Implement immutable factor schema and masked replacement**

```python
@dataclass(frozen=True)
class SingleViewBatch:
    iq: torch.Tensor
    satellite_mask: torch.Tensor
    channel_labels: torch.Tensor
    channel_factors: torch.Tensor

mask = torch.rand(batch_size, generator=generator, device="cpu") < float(p_sat)
out = clean.clone()
out[mask] = augmentor(clean[mask], scenario="mixed_orbit", generator=generator)
```

Clean rows receivechannel label0 and zeroed physical factor bins; satellite rows receiveaugmentor-emittedCFO、phase-noise、SNR、multipath和elevation bins. Reject`p_sat`outside`[0,1]`.

- [ ] **Step 4: Run GREEN and verify no concat helper is called**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_satellite.py -q`

- [ ] **Step 5: Update HCF-016/HCF-017 and commit**

Run: `git.exe add -- code/cvsrffi/phase1_hcfdg/satellite.py code/tests/phase1_hcfdg/test_satellite.py analysis/phase1_hcfdg_traceability.md`

Run: `git.exe commit -m "feat: add HCF-DG single-view satellite batches"`

---

### Task 4: Build the Single Backbone and Factorized Training Heads

**Files:**

- Create: `code/cvsrffi/phase1_hcfdg/model.py`
- Test: `code/tests/phase1_hcfdg/test_model.py`
- Modify: `analysis/phase1_hcfdg_traceability.md`

**Interfaces:**

- Consumes: existingidentity backbone returning`logits`and`feat_joint`、`q_phys`、receiver/day/channel labels。
- Produces: `HCFDGModel`、`HCFDGOutput`、`FactorizedEnvironmentEncoder`、`CommonSpecificLowRankHead`、`CounterfactualTransport`。

- [ ] **Step 1: Write failing tests for dimensions, stop-gradient, rank and inference pruning**

```python
def test_hcfdg_output_has_single_backbone_and_48d_environment():
    out = model(x, tx_labels=y, env_meta=meta, training_aux=True)
    assert backbone.forward_calls == 1
    assert out.z_id.shape == (96, 160)
    assert out.z_env.shape == (96, 48)
    assert out.specific_logits.shape == out.common_logits.shape

def test_inference_uses_only_common_head():
    model.eval()
    logits = model.inference_logits(x)
    assert torch.equal(logits, model.common_head(model.identity_features(x)))
```

- [ ] **Step 2: Run RED**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_model.py -q`

- [ ] **Step 3: Implement typed outputs and low-rank heads**

```python
@dataclass
class HCFDGOutput:
    common_logits: torch.Tensor
    specific_logits: torch.Tensor | None
    z_id: torch.Tensor
    z_rx: torch.Tensor
    z_day: torch.Tensor
    z_channel: torch.Tensor
    z_env: torch.Tensor
    receiver_logits: torch.Tensor | None
    day_logits: torch.Tensor | None
    channel_logits: torch.Tensor | None
    tx_from_env_logits: torch.Tensor | None
    conditional_receiver_logits: torch.Tensor | None
    fused_feature: torch.Tensor
```

Use`stop_gradient(h_early)`only on the environment path. Implement`W_e=W0+U diag(a_rx+a_day+a_channel)V^T`withrank4 andspecific dropout0.5. `inference_logits`must not call environment, specific, adversarial or counterfactual modules.

- [ ] **Step 4: Implement bounded counterfactual transport in the same module**

```python
gamma = self.gamma_head(delta_env).clamp(-self.gamma_cap, self.gamma_cap)
beta = self.beta_head(delta_env).clamp(-self.beta_cap, self.beta_cap)
h_cf = (1.0 + gamma) * F.layer_norm(h, h.shape[1:]) + beta
```

Pair onlysame-TX rows for receiver swap; expose explicit target environment labels.

- [ ] **Step 5: Run GREEN and finite-gradient checks**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_model.py -q`

- [ ] **Step 6: Update HCF-001/HCF-002/HCF-003/HCF-004/HCF-007/HCF-014 and commit**

Run: `git.exe add -- code/cvsrffi/phase1_hcfdg/model.py code/tests/phase1_hcfdg/test_model.py analysis/phase1_hcfdg_traceability.md`

Run: `git.exe commit -m "feat: add HCF-DG factorized model"`

---

### Task 5: Implement LODO, Counterfactual and Hierarchical Risk Losses

**Files:**

- Create: `code/cvsrffi/phase1_hcfdg/losses.py`
- Test: `code/tests/phase1_hcfdg/test_losses.py`
- Modify: `analysis/phase1_hcfdg_traceability.md`

**Interfaces:**

- Consumes: `HCFDGOutput`、episode masks、TX/receiver/day/channel labels、content keys。
- Produces: `lodo_prototype_loss`、`content_conditioned_lodo_loss`、`counterfactual_losses`、`hierarchical_dro_loss`、`compose_hcfdg_loss`。

- [ ] **Step 1: Write a RED test proving query rows cannot enter prototypes**

```python
def test_lodo_prototypes_exclude_every_query_domain_row():
    loss, info = lodo_prototype_loss(z, y, domain, query_domain=9, temperature=0.10)
    assert 9 not in info.support_domains
    assert info.query_count == int((domain == 9).sum())
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Write RED tests for content fallback, parent shrinkage and finite gradients**

```python
def test_content_conditioning_falls_back_without_close_support():
    _, info = content_conditioned_lodo_loss(z, y, d, keys, query_domain=2, max_distance=0.01)
    assert info.fallback_classes == frozenset({0, 1})

def test_hdro_shrinks_small_child_group_to_parent():
    loss, info = hierarchical_dro_loss(per_sample_loss, groups, kappa=8.0, tau=0.25, min_group=4)
    assert info.shrunk_risks["tx_rx:0:1"] < info.raw_risks["tx_rx:0:1"]
    loss.backward()
    assert torch.isfinite(per_sample_loss.grad).all()
```

- [ ] **Step 3: Run RED**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_losses.py -q`

- [ ] **Step 4: Implement all loss functions and explicit metrics dataclasses**

Implement normalized cosine logits withtemperature0.10, content-key soft weights, fallback to unweighted prototypes, CF-ID/INV/ENV/style components, HDRO groups`receiver/day/channel/tx_receiver/tx_day/tx_channel`, and weighted total:

```python
total = id_loss + 0.40*lodo + 0.15*cf + 0.10*hdro + 0.15*csd + 0.05*fac
```

Disabled components must return an exact scalar zero on the same device; no legacyGroup CE、FISHR、REx oropen-world CVaR calls are permitted.

- [ ] **Step 5: Run GREEN**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_losses.py -q`

- [ ] **Step 6: Update HCF-005/HCF-006/HCF-008/HCF-010/HCF-013/HCF-015 and commit**

Run: `git.exe add -- code/cvsrffi/phase1_hcfdg/losses.py code/tests/phase1_hcfdg/test_losses.py analysis/phase1_hcfdg_traceability.md`

Run: `git.exe commit -m "feat: add HCF-DG domain generalization losses"`

---

### Task 6: Implement Stage0–4 Training and Exact Optimizer Schedules

**Files:**

- Create: `code/cvsrffi/phase1_hcfdg/trainer.py`
- Test: `code/tests/phase1_hcfdg/test_trainer.py`
- Modify: `analysis/phase1_hcfdg_traceability.md`

**Interfaces:**

- Consumes: `HCFDGConfig`、`HCFDGModel`、labeled/unlabeled/validation loaders、`build_single_view_batch`。
- Produces: `HCFDGTrainer.train()`、`TrainState`、`CheckpointPayload`、JSONL/CSV metrics and resource telemetry。

- [ ] **Step 1: Write RED tests for exact update counts and one backbone call per main update**

```python
def test_v1_runs_exactly_4000_optimizer_updates(fake_trainer):
    state = fake_trainer.train(candidate_config("A5"))
    assert state.optimizer_updates == 4000
    assert state.backbone_forward_calls == 4000

def test_v2_stage_counts_total_6300_and_freeze_at_half():
    state = fake_trainer.train(candidate_config("A9"))
    assert state.stage_updates == {"stage0": 700, "stage1": 1200, "stage2": 2100, "stage3": 1700, "stage4": 600}
    assert state.freeze_update == 3150
```

- [ ] **Step 2: Write RED tests for source-only access andU_s restrictions**

Use sentinels that raise if target/query keys are opened. AssertStage0 consumes onlyIQ plusreceiver/day metadata and never computesTX CE on`U_s`.

- [ ] **Step 3: Run RED**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_trainer.py -q`

- [ ] **Step 4: Implement stage controller, two parameter groups and schedules**

```python
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-4},
    {"params": new_head_params, "lr": 3e-4},
], weight_decay=1e-4)
```

Warm up for5% of total updates, cosine decay to`1e-6`, and rampCosFace margin from0 to0.30 during the first20%. Atupdate3150 freezeSinc and the first time-domain block. KeepAMP enabled whenCUDA is available.

- [ ] **Step 5: Implement checkpoint and telemetry schema**

Checkpoint must include`phase1_method=hcfdg`、candidate ID、source split、fold、seed、update、config、model/optimizer/scaler states and common-head-only inference metadata. Metrics must include step time、samples/s、dataloader wait、peak memory、forward/backward time and totalGPU-hours.

- [ ] **Step 6: Run GREEN**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_trainer.py -q`

- [ ] **Step 7: Update HCF-009/HCF-018/HCF-019/HCF-020/HCF-021/HCF-027/HCF-028 and commit**

Run: `git.exe add -- code/cvsrffi/phase1_hcfdg/trainer.py code/tests/phase1_hcfdg/test_trainer.py analysis/phase1_hcfdg_traceability.md`

Run: `git.exe commit -m "feat: add HCF-DG staged trainer"`

---

### Task 7: Implement Source Diagnostics and Same-Row Selection Metrics

**Files:**

- Create: `code/cvsrffi/phase1_hcfdg/metrics.py`
- Test: `code/tests/phase1_hcfdg/test_metrics.py`
- Modify: `analysis/phase1_hcfdg_traceability.md`

**Interfaces:**

- Consumes: frozen embeddings/logits and source-only labels/domain metadata。
- Produces: `conditional_receiver_leakage`、`environment_tx_leakage`、`specific_gap`、`counterfactual_effectiveness`、`domain_drift_ratio`、`SameRowMetrics`、`rank_source_rows`。

- [ ] **Step 1: Write RED tests for all five report diagnostics**

```python
def test_domain_drift_ratio_matches_closed_form_example():
    value = domain_drift_ratio(class_domain_centers, class_centers)
    assert value == pytest.approx(0.25)

def test_source_ranking_never_combines_different_rows():
    ranked = rank_source_rows(rows)
    assert ranked[0].row_id == "A5-F8-S392002"
    assert ranked[0].clean == rows_by_id[ranked[0].row_id].clean
```

- [ ] **Step 2: Run RED**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_metrics.py -q`

- [ ] **Step 3: Implement deterministic source-only probes and metrics**

Use regularized linear probes fitted only onsource train embeddings and scored onsource validation. Returnmacro accuracy, per-TX leakage,`Delta_spec`, CF identity retention, CF environment switch,`R_drift`, minimum class margin, clean/LEO mean/floor and harmonic selection score.

- [ ] **Step 4: Run GREEN and serialization round-trip**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_metrics.py -q`

- [ ] **Step 5: Update HCF-026/HCF-034 and commit**

Run: `git.exe add -- code/cvsrffi/phase1_hcfdg/metrics.py code/tests/phase1_hcfdg/test_metrics.py analysis/phase1_hcfdg_traceability.md`

Run: `git.exe commit -m "feat: add HCF-DG source diagnostics"`

---

### Task 8: Build the Formal Matrix Launcher and Four-Scenario Closure

**Files:**

- Create: `code/scripts/launch_phase1_hcfdg_matrix_20260830.py`
- Test: `code/tests/phase1_hcfdg/test_launcher.py`
- Modify: `code/cvsrffi/phase1_hcfdg/__init__.py`
- Modify: `analysis/phase1_hcfdg_traceability.md`

**Interfaces:**

- Consumes: all newHCF-DG modules、existing WiSig role builder、existing final satellite evaluator。
- Produces: `build_plan`、`build_train_command`、`validate_output_root`、`run_row`、`evaluate_final_checkpoint` and immutable plan/final status JSON。

- [ ] **Step 1: Write RED tests for all 36 quick rows and exact command bindings**

```python
def test_quick_plan_binds_report_split_and_seeds():
    rows = build_plan(stage="quick", folds=(1, 8))
    assert len(rows) == 36
    assert {row.seed for row in rows} == {392001, 392002, 392003}
    command = build_train_command(rows[0], roots)
    assert value_after(command, "--wisig_train_rxs") == "1,3,4,6,8"
    assert value_after(command, "--wisig_train_days") == "1,2,3"
    assert "phase2" not in " ".join(command).lower()
```

- [ ] **Step 2: Write RED negative tests**

Rejectnonempty target receiver fields、Phase2 paths、existing output roots、unknown candidate IDs、A10–A12 without a recordedV2 pass and any row lacking all four final scenarios.

- [ ] **Step 3: Run RED**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_launcher.py -q`

- [ ] **Step 4: Implement launcher and exact per-GPU concurrency limit**

Assign at mosttwo active rows perGPU. Preserve each row under`<run-root>/<candidate>-F<fold>-S<seed>`and never reuse a directory. Write`plan.json`before dispatch and`final_status.json`only after every row reaches a terminal state.

- [ ] **Step 5: Implement final checkpoint evaluation**

For each completed row, strictly reload the final checkpoint and invoke clean plus`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`. Require separate metrics/log artifacts and zero missing/unexpected/shape mismatch keys. Training exit alone must not write`ARTIFACTS_COMPLETE`.

- [ ] **Step 6: Run GREEN plus existing ADV3B03 launcher regression**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg\test_launcher.py code\tests\test_phase1_adv3b03_src5_day123_seedscan.py -q`

- [ ] **Step 7: Update HCF-022/HCF-023/HCF-024/HCF-025/HCF-028/HCF-029/HCF-030/HCF-031/HCF-032/HCF-033 and commit**

Run: `git.exe add -- code/scripts/launch_phase1_hcfdg_matrix_20260830.py code/cvsrffi/phase1_hcfdg/__init__.py code/tests/phase1_hcfdg/test_launcher.py analysis/phase1_hcfdg_traceability.md`

Run: `git.exe commit -m "feat: add HCF-DG formal matrix launcher"`

---

### Task 9: Complete Local Integration, Real-Checkpoint Smoke and Reverse Traceability Audit

**Files:**

- Test: all `code/tests/phase1_hcfdg/*.py`
- Modify: `analysis/phase1_hcfdg_traceability.md`
- Create: `automation_reports/CV-SincNet/<smoke-run-id>/report.md`

**Interfaces:**

- Consumes: completed HCF-DG package and one existing real`lite_d`checkpoint。
- Produces: local smoke artifact、strict reconstruction evidence、final traceability states。

- [ ] **Step 1: Run the complete focused test suite**

Run: `conda.exe run -n ssr-gpu python -m pytest code\tests\phase1_hcfdg code\tests\test_balanced_tx_rx_sampler.py code\tests\test_phase1_adv3b03_src5_day123_seedscan.py -q`

Expected: zero failures and no warnings caused byHCF-DG.

- [ ] **Step 2: Compile every new Python entrypoint**

Run: `conda.exe run -n ssr-gpu python -m py_compile code\cvsrffi\phase1_hcfdg\__init__.py code\cvsrffi\phase1_hcfdg\config.py code\cvsrffi\phase1_hcfdg\sampler.py code\cvsrffi\phase1_hcfdg\satellite.py code\cvsrffi\phase1_hcfdg\model.py code\cvsrffi\phase1_hcfdg\losses.py code\cvsrffi\phase1_hcfdg\trainer.py code\cvsrffi\phase1_hcfdg\metrics.py code\scripts\launch_phase1_hcfdg_matrix_20260830.py`

- [ ] **Step 3: Run one real-checkpoint no-query smoke**

Load an existingADV3B02/ADV3B03`lite_d`identity backbone through the HCF wrapper, execute one clean/single-view training update and one inference batch, then strictly save/reload the HCF checkpoint. The smoke must prove`backbone_forward_calls=1`、no target/query access、finite loss and common-head inference.

- [ ] **Step 4: Perform the one allowed independent P0/P1 review**

Review only defects that could misroute receiver/day/seed/candidate, access target/Phase2 data, overwrite output, break launch, corrupt checkpoint reconstruction or prevent four-scenario artifacts. Fix directP0/P1 findings with a failing test, then run one scoped re-review.

- [ ] **Step 5: Reverse-audit all 34 traceability rows**

Every implementation row must name its actual file and test evidence. Mark code-complete rows`verified`; leave experiment-dependent rows`implemented`until N607 artifacts exist. No row may remain`pending`without an explicit remaining action.

- [ ] **Step 6: Commit and push local implementation**

Run: `git.exe diff --check`

Run: `git.exe add --`with only the HCF-DG implementation、tests、traceability and smoke report paths.

Run: `git.exe commit -m "feat: complete Phase1 HCF-DG implementation"`

Verify local`HEAD`equals the exact remote branchOID.

---

### Task 10: Prepare and Launch the N607 A0–A5 Formal Matrix

**Files:**

- Create: `automation_reports/CV-SincNet/<formal-run-id>/report.md`
- Mirror: the same report under the Git worktree.

**Interfaces:**

- Consumes: pushed implementation commit、single release archive、N607 ordinary account。
- Produces: immutable36-row formal run、PID/GPU/log binding and monitoring state。

- [ ] **Step 1: Write the minimal preregistration report**

Record run ID、commit、36 rows、center/far fold IDs、seeds`392001/392002/392003`、4000 updates、commands、CWD、paths、GPU allocation、technical stop rules and expected artifacts. Do not add extra gates.

- [ ] **Step 2: Run the project-owned N607 preflight with the ordinary account**

Use`powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`. If direct connectivity alone fails and identity remains valid, use the documented lab bridge. Never use`N607-admin`.

- [ ] **Step 3: Build one release archive, compare one local/remote SHA and compile once remotely**

Archive only committed code/config/report inputs. Upload to a unique release root, compare the archive SHA once, extract without overwriting an existing root and compile the HCF entrypoint once.

- [ ] **Step 4: Execute the real one-row smoke as launcher step one**

RunA4、fold center、seed392002 for one bounded update and all four strict final evaluations. OnPASS, the same launcher proceeds to the formal matrix; no smoke authorization artifact is created.

- [ ] **Step 5: Launch the36-row matrix and perform one binding readback**

Verify dispatcher and main trainingPIDs、exactCWD/cmdline/run-root、GPU mapping、log growth and no deterministic fatal fingerprint. Keep at mosttwo training processes perGPU and do not touch unrelated jobs.

- [ ] **Step 6: Commit/push the launch record and verify remote OID**

Precisely stage only the formal report mirror, commit, allow automatic push and independently compare remoteOID with local`HEAD`.

---

### Task 11: Close, Analyze and Advance the Report-Ordered Matrix

**Files:**

- Modify: `automation_reports/CV-SincNet/<formal-run-id>/report.md`
- Modify: `analysis/phase1_hcfdg_traceability.md`
- Create later: A6–A9、confirmation、A10–A12 and final8-seed reports when their report-defined stage begins.

**Interfaces:**

- Consumes: complete training/evaluation artifacts only。
- Produces: same-row source ranking、resource analysis、next-stage frozen configuration and final target confirmation when authorized by report order。

- [ ] **Step 1: Hold every row below ARTIFACTS_COMPLETE until all four evaluations close**

Require4000 updates、final checkpoint、strict reconstruction and separateclean/clear/low/rain artifacts. Technical failures retainpartial artifacts and use the registered failure state; low performance never stops a healthy row.

- [ ] **Step 2: Parse every log and artifact, not tails or samples**

Validate update sequence、finite metrics、checkpoint identity、scenario identity、per-fold/per-seed metrics and resource telemetry for all36 rows.

- [ ] **Step 3: Rank source rows and publish A0–A5 conclusions**

Compare same-rowLODO mean/floor、clean、threeLEO scenarios、five diagnostics andGPU-hours. Freeze the next report-ordered candidate without target feedback.

- [ ] **Step 4: Run A6–A9 and source LORO confirmation in the same governed sequence**

Use the same three seeds for2-fold screens, then5 folds×3 seeds×6300 updates for the top two source candidates. A10–A12 start only ifA8 orA9 passes the frozen source gate.

- [ ] **Step 5: Run final8-seed source training and one target confirmation**

After structure and hyperparameters freeze, train all preregistered8 seeds on all source receivers. Freeze the final seed fromsource-only evidence, then perform exactly one zero-adaptation、prediction-first evaluation over all target receivers and days; target results cannot feed back.

- [ ] **Step 6: Complete traceability and formal publication**

Set everyHCF-001–HCF-034 row to`verified`or an explicit intentional non-implementation state supported by the report. Mirror final reports, precisely stage, commit, automatically push and independently verify remoteOID.
