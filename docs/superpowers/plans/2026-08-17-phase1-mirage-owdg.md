# Phase1 MIRAGE-OWDG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从头实现、训练并一次性确认MIRAGE-OWDG，使其在`7% L_s/63% U_s/30% V_s`的source-only研发条件下完成半监督跨接收机DG、source proxy开放集训练和身份互斥target unknown拒识。

**Architecture:** 新代码位于独立的`cvsrffi.phase1_mirage`命名空间，依次实现数据权限、角色化proxy、轻量IQ encoder、开放世界头、B0/A/B/C训练、source校准、不可变bundle和truth-blind target推理。source六fold完整闭合并通过Gate 1-3后，才允许从头训练唯一`M*`和`B0*`、通过审查B并运行一次固定target capsule。

**Tech Stack:** Python 3.10、PyTorch、NumPy、SciPy、scikit-learn、pytest、ManySig/WiSig compact loader、现有LEO weak cache verifier、Git Bash、N607八GPU。

## Global Constraints

- 科学规格为`docs/superpowers/specs/2026-08-17-phase1-mirage-owdg-design.md`，批准commit为`8ec5540b`。
- Windows所有命令使用`C:\Program Files\Git\bin\bash.exe`作为外层shell，`login:false`；禁止`pwsh`/`pwsh.exe`。
- 本地Python检查串行运行`conda run -n ssr-gpu`，每次先打印`sys.executable`和`CONDA_PREFIX`；不得并发启动多个Conda wrapper。
- source物理样本固定为`0.07/0.63/0.30`，`L_s/U_s/V_s`可以共享TX身份但物理ID两两不交。
- `proxy_train`只从`L_s`生成并可训练拒识机制；`P_cal/P_select`只从对应source validation生成并可校准/选模。
- target unknown TX身份与source train/validation TX身份互斥；target数据不得训练、校准、选模、重排候选或触发选择性重跑。
- target-known和target-unknown采用相同单物理样本单LEO weak观测、预处理、前向和决策规则。
- B0/A/B/C共享主干容量、optimizer、step预算和正式`epochs=200`；主干参数不超过3M，bundle不超过16MiB。
- source六fold选择方法而不选择最佳fold；target只运行冻结`M*`和`B0*`各一次。
- N607每GPU最多两个训练进程；一个run ID只有一个runner，runner不得改方法、阈值、arm、fold、seed或矩阵。
- 所有代码、配置、协议和报告变更先本地验证、精确提交，再同步N607；禁止远端改码。
- 根目录`E:\type10-7`不是Git仓库；根协议和根报告必须镜像到Git承载面并记录未版本化边界。

---

## File Structure

### Protocol and traceability

- Modify `E:\type10-7\项目.md`: replace the old fixed TX-disjoint/non-training proxy rule with the approved split-relative proxy rule.
- Modify `docs/PROJECT_PROTOCOL.md`: mirror the same Phase1 data/proxy/target semantics.
- Create `tests/phase1_mirage/test_protocol_docs.py`: protect the Git protocol text from regressing to the old semantics.

### Core package

- Create `code/cvsrffi/phase1_mirage/__init__.py`: public exports only.
- Create `code/cvsrffi/phase1_mirage/data.py`: inventory rows, `7/63/15/15` physical split, role-safe training/validation views.
- Create `code/cvsrffi/phase1_mirage/proxy.py`: deterministic class-role proxy episodes and class-mask construction.
- Create `code/cvsrffi/phase1_mirage/model.py`: IQ preprocessing, multiscale patch encoder, Transformer fusion and quality output.
- Create `code/cvsrffi/phase1_mirage/head.py`: prototypes, radii, covariance, energy, unknown risk and tri-state decision.
- Create `code/cvsrffi/phase1_mirage/losses.py`: pseudo-label gate, masked/consistency/proxy/open-set losses and Group-CVaR.
- Create `code/cvsrffi/phase1_mirage/config.py`: frozen arm/budget schemas and config hashes.
- Create `code/cvsrffi/phase1_mirage/trainer.py`: EMA/SWAD lifecycle, fold training, metrics and checkpoint receipts.
- Create `code/cvsrffi/phase1_mirage/calibration.py`: source-only threshold search.
- Create `code/cvsrffi/phase1_mirage/scoring.py`: source fold metrics, Gates 1-4 and arm selection.
- Create `code/cvsrffi/phase1_mirage/bundle.py`: exact-member immutable package export/load.
- Create `code/cvsrffi/phase1_mirage/target.py`: target prediction records and independent scoring helpers.

### Entrypoints, config and tests

- Create `configs/phase1_mirage_owdg/{b0,a,b,c,source_matrix,final_refit}.json`.
- Create `code/scripts/run_phase1_mirage_source_matrix.py`.
- Create `code/scripts/run_phase1_mirage_final_refit.py`.
- Create `code/scripts/build_phase1_mirage_bundle.py`.
- Create `code/scripts/predict_phase1_mirage_target.py`.
- Create `code/scripts/score_phase1_mirage_target.py`.
- Create `scripts/launchers/run_phase1_mirage_source_matrix.sh`.
- Create `scripts/launchers/run_phase1_mirage_final_refit.sh`.
- Create `tests/phase1_mirage/test_{data,proxy,model,head,losses,trainer,calibration,scoring,bundle,target,cli}.py`.

### Evidence

- Create/mirror `automation_reports/CV-SincNet/phase1_mirage_source6_20260817_v1/report.md` before source release.
- Create/mirror `automation_reports/CV-SincNet/phase1_mirage_final_refit_20260817_v1/report.md` before final refit.
- Create/mirror `automation_reports/CV-SincNet/phase1_mirage_target_confirm_20260817_v1/report.md` before target access.

---

### Task 1: Synchronize the Approved Scientific Protocol

**Files:**
- Modify: `E:\type10-7\项目.md`
- Modify: `docs/PROJECT_PROTOCOL.md`
- Create: `code/cvsrffi/phase1_mirage/protocol.py`
- Create: `tests/phase1_mirage/test_protocol_policy.py`
- Create: `analysis/phase1_mirage_owdg_traceability.md`

**Interfaces:**
- Consumes: approved spec section 2-3.
- Produces: synchronized authoritative text and machine-executable`Phase1DataPolicy`for`L_s/U_s/V_s`、`proxy_train`、`P_cal/P_select`与身份互斥target unknown。

- [x] **Step 1: Write the failing policy-behavior test**

```python
def test_proxy_train_is_labeled_training_only_and_can_receive_rejection_gradients():
    policy = Phase1DataPolicy()
    assert policy.proxy_origin_is_allowed(ProxyRole.PROXY_TRAIN, SourcePartition.L_S)
    assert policy.allows(ProxyRole.PROXY_TRAIN, Permission.REJECTION_GRADIENT)
```

- [x] **Step 2: Run the test and verify the policy interface is initially absent**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_protocol_policy.py -q`

Expected: FAIL because`cvsrffi.phase1_mirage.protocol`is absent.

- [x] **Step 3: Implement the minimum policy interface and synchronize both Phase1 protocol sections**

The behavior tests must cover`0.07/0.63/0.15/0.15`、physical-ID互斥但TX身份可共享、`proxy_train`的`L_s`来源与拒识梯度、`P_cal/P_select`的独立validation来源和唯一用途、target unknown身份互斥，以及所有target角色的零训练/校准/选模/选择性重跑权限。

Delete the old fixed-TX-disjoint and proxy-all-training-ban semantics. Preserve all Phase2/Phase3 permissions unchanged.

- [x] **Step 4: Verify both documents and report the root non-Git boundary**

Run:

```bash
conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_protocol_policy.py -q
rg -n "proxy_train|P_cal/P_select|target unknown TX身份" /e/type10-7/项目.md docs/PROJECT_PROTOCOL.md
git diff --check
```

Expected: PASS; both documents show matching semantics. Record in the implementation report that`项目.md`is outside Git and mirrored by`docs/PROJECT_PROTOCOL.md`.

- [x] **Step 5: Commit the Git-backed protocol change**

```bash
git add docs/PROJECT_PROTOCOL.md docs/superpowers/plans/2026-08-17-phase1-mirage-owdg.md code/cvsrffi/phase1_mirage/protocol.py tests/phase1_mirage/test_protocol_policy.py analysis/phase1_mirage_owdg_traceability.md
git commit -m "feat: enforce MIRAGE Phase1 data policy"
```

### Task 2: Build Role-Safe Source Splits

**Files:**
- Create: `code/cvsrffi/phase1_mirage/__init__.py`
- Create: `code/cvsrffi/phase1_mirage/data.py`
- Create: `tests/phase1_mirage/test_data.py`

**Interfaces:**
- Consumes: `Sequence[SourceInventoryRow]`, where the builder sees TX truth once.
- Produces: `SourceSplitManifest`, `LabeledView`, `UnlabeledView`, `ValidationView` and deterministic ID lists.

- [ ] **Step 1: Write failing split and label-hiding tests**

```python
def test_split_is_7_63_15_15_and_hides_u_truth(rows_100_per_group):
    split = build_source_split(rows_100_per_group, seed=817001)
    assert tuple(map(len, (split.l_ids, split.u_ids, split.v_cal_ids, split.v_select_ids))) == (7, 63, 15, 15)
    assert not (set(split.l_ids) & set(split.u_ids))
    assert not hasattr(materialize_unlabeled(rows_100_per_group, split.u_ids)[0], "tx_label")


def test_target_receiver_is_rejected_before_split(rows_with_target_receiver):
    with pytest.raises(SourceProtocolError, match="target receiver"):
        build_source_split(rows_with_target_receiver, seed=817001, forbidden_receivers={"20-1"})
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_data.py -q`

Expected: FAIL with missing`cvsrffi.phase1_mirage.data`.

- [ ] **Step 3: Implement immutable split types and deterministic grouping**

```python
class SourceProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class SourceInventoryRow:
    physical_sample_id: str
    tx_label: int
    receiver_id: str
    day_id: str
    iq_index: int


@dataclass(frozen=True)
class LabeledView:
    physical_sample_id: str
    tx_label: int
    receiver_id: str
    day_id: str
    iq_index: int


@dataclass(frozen=True)
class UnlabeledView:
    physical_sample_id: str
    receiver_id: str
    day_id: str
    iq_index: int


@dataclass(frozen=True)
class ValidationView(LabeledView):
    split_role: Literal["val_cal", "val_select"]


@dataclass(frozen=True)
class SourceSplitManifest:
    l_ids: Sequence[str]
    u_ids: Sequence[str]
    v_cal_ids: Sequence[str]
    v_select_ids: Sequence[str]
    split_schema: str


def _partition_counts(size: int) -> tuple[int, int, int, int]:
    fractions = (0.07, 0.63, 0.15, 0.15)
    raw = [size * value for value in fractions]
    counts = [math.floor(value) for value in raw]
    order = sorted(range(4), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[: size - sum(counts)]:
        counts[index] += 1
    return tuple(counts)


def build_source_split(
    rows: Sequence[SourceInventoryRow], *, seed: int,
    forbidden_receivers: Collection[str] = (),
) -> SourceSplitManifest:
    if len({row.physical_sample_id for row in rows}) != len(rows):
        raise SourceProtocolError("duplicate physical_sample_id")
    if {row.receiver_id for row in rows} & set(forbidden_receivers):
        raise SourceProtocolError("target receiver present in source inventory")
    groups: dict[tuple[int, str, str], list[SourceInventoryRow]] = defaultdict(list)
    for row in rows:
        groups[(row.tx_label, row.receiver_id, row.day_id)].append(row)
    buckets: list[list[str]] = [[], [], [], []]
    for group_rows in groups.values():
        ordered = sorted(
            group_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{row.physical_sample_id}".encode("utf-8")
            ).digest(),
        )
        counts = _partition_counts(len(ordered))
        cursor = 0
        for bucket, count in zip(buckets, counts):
            bucket.extend(row.physical_sample_id for row in ordered[cursor : cursor + count])
            cursor += count
    return SourceSplitManifest(
        *(tuple(sorted(bucket)) for bucket in buckets),
        split_schema="phase1_mirage_source_7_63_15_15_v1",
    )
```

The returned manifest must include ID SHA256 values, group counts, receiver/TX registries and`split_schema="phase1_mirage_source_7_63_15_15_v1"`. The training materializer must construct`UnlabeledView`without a label field.

- [ ] **Step 4: Run focused tests**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_data.py -q`

Expected: PASS, including deterministic same-seed and changed-seed tests.

- [ ] **Step 5: Commit**

```bash
git add code/cvsrffi/phase1_mirage/__init__.py code/cvsrffi/phase1_mirage/data.py tests/phase1_mirage/test_data.py
git commit -m "feat: add MIRAGE source split contract"
```

### Task 3: Generate Balanced Proxy Episodes

**Files:**
- Create: `code/cvsrffi/phase1_mirage/proxy.py`
- Create: `tests/phase1_mirage/test_proxy.py`

**Interfaces:**
- Consumes: labeled batch labels plus`split_role in {"train_l","val_cal","val_select"}`.
- Produces: `ProxyEpisode(proxy_class, registered_class_mask, registered_rows, proxy_rows, schedule_receipt)`.

- [ ] **Step 1: Write failing provenance, masking and permutation tests**

```python
def test_train_proxy_accepts_only_labeled_training_role():
    with pytest.raises(ProxyProtocolError, match="train_l"):
        build_proxy_episode(labels, split_role="train_u", seed=9, episode_index=0)


def test_proxy_class_is_absent_from_registered_mask():
    episode = build_proxy_episode(torch.tensor([0, 0, 1, 1, 2, 2]), split_role="train_l", seed=9, episode_index=0)
    assert not episode.registered_class_mask[episode.proxy_class]
    assert set(torch.tensor([0, 0, 1, 1, 2, 2])[episode.proxy_rows].tolist()) == {episode.proxy_class}
```

- [ ] **Step 2: Verify tests fail**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_proxy.py -q`

Expected: FAIL with missing proxy module.

- [ ] **Step 3: Implement a deterministic balanced schedule**

```python
class ProxyProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ProxyEpisode:
    proxy_class: int
    registered_class_mask: torch.Tensor
    registered_rows: torch.Tensor
    proxy_rows: torch.Tensor
    schedule_receipt: Mapping[str, int | str]


def proxy_class_for_episode(class_ids: Sequence[int], *, seed: int, episode_index: int) -> int:
    ordered = sorted(set(map(int, class_ids)))
    if len(ordered) < 3:
        raise ProxyProtocolError("proxy episode requires at least three classes")
    offset = int(hashlib.sha256(f"{seed}:proxy".encode()).hexdigest(), 16) % len(ordered)
    return ordered[(offset + episode_index) % len(ordered)]
```

`build_proxy_episode`must reject`train_u`, accept only`train_l/val_cal/val_select`, mask the proxy class before logits/prototypes are formed, and emit a class-permutation-invariant receipt using counts rather than raw class names.

- [ ] **Step 4: Run focused tests**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_proxy.py -q`

Expected: PASS; one full cycle assigns each class exactly once as proxy.

- [ ] **Step 5: Commit**

```bash
git add code/cvsrffi/phase1_mirage/proxy.py tests/phase1_mirage/test_proxy.py
git commit -m "feat: add role-balanced source proxy episodes"
```

### Task 4: Implement the Lightweight MIRAGE Encoder

**Files:**
- Create: `code/cvsrffi/phase1_mirage/model.py`
- Create: `tests/phase1_mirage/test_model.py`

**Interfaces:**
- Consumes: IQ tensor`[B,2,T]`with finite float values.
- Produces: `MIRAGEFeatures(z_id:[B,160], z_dom:[B,32], quality:[B], tokens:[B,N,192])`.

- [ ] **Step 1: Write failing shape, normalization and budget tests**

```python
def test_encoder_outputs_finite_normalized_features_under_budget():
    model = MIRAGEEncoder(MIRAGEConfig())
    out = model(torch.randn(4, 2, 256))
    assert out.z_id.shape == (4, 160)
    assert out.z_dom.shape == (4, 32)
    assert torch.allclose(out.z_id.norm(dim=1), torch.ones(4), atol=1e-5)
    assert torch.isfinite(out.quality).all()
    assert sum(p.numel() for p in model.parameters()) <= 3_000_000
```

- [ ] **Step 2: Verify tests fail**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_model.py -q`

Expected: FAIL with missing model module.

- [ ] **Step 3: Implement preprocessing and encoder**

```python
@dataclass(frozen=True)
class MIRAGEConfig:
    patch_kernel: int = 32
    patch_stride: int = 16
    token_dim: int = 192
    transformer_layers: int = 4
    transformer_heads: int = 4
    z_id_dim: int = 160
    z_dom_dim: int = 32


@dataclass
class MIRAGEFeatures:
    z_id: torch.Tensor
    z_dom: torch.Tensor
    quality: torch.Tensor
    tokens: torch.Tensor


def preprocess_iq(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = x - x.mean(dim=-1, keepdim=True)
    rms = centered.square().sum(dim=1).mean(dim=-1).clamp_min(1e-8).sqrt()
    normalized = centered / rms[:, None, None]
    peak = centered.square().sum(dim=1).sqrt().amax(dim=-1)
    quality_aux = torch.stack([rms.log(), peak / rms.clamp_min(1e-8)], dim=1)
    return normalized, quality_aux
```

Build a Conv1d patch stem, parallel depthwise kernels`3/7/15`, four pre-norm Transformer layers, quality-gated local/global fusion and separate identity/domain heads. Apply`torch.nan_to_num`only at the external input boundary; raise on non-finite internal outputs in formal mode.

- [ ] **Step 4: Run focused tests**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_model.py -q`

Expected: PASS on CPU for lengths256 and512.

- [ ] **Step 5: Commit**

```bash
git add code/cvsrffi/phase1_mirage/model.py tests/phase1_mirage/test_model.py
git commit -m "feat: add lightweight MIRAGE IQ encoder"
```

### Task 5: Implement the Open-World Geometry and Decision Head

**Files:**
- Create: `code/cvsrffi/phase1_mirage/head.py`
- Create: `tests/phase1_mirage/test_head.py`

**Interfaces:**
- Consumes: normalized`z_id`, registered class geometry and optional episode class mask.
- Produces: class scores, Mahalanobis distances, radius margins, energy, unknown risk, quality and tri-state decision.

- [ ] **Step 1: Write failing geometry and decision tests**

```python
def test_defer_is_not_unknown_and_proxy_mask_removes_class():
    head = MIRAGEOpenHead(num_classes=3, feature_dim=160, covariance_rank=8)
    output = head(torch.randn(5, 160), class_mask=torch.tensor([True, False, True]))
    assert torch.all(output.class_scores[:, 1] < -1e5)
    thresholds = DecisionThresholds(tau_q=.5, tau_reg=.2, tau_unk=.8)
    decision = decide(output, quality=torch.tensor([.1, .9, .9, .9, .9]), thresholds=thresholds)
    assert decision.labels[0] == DEFER_LABEL
    assert not decision.explicit_unknown[0]
```

- [ ] **Step 2: Verify tests fail**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_head.py -q`

Expected: FAIL with missing head module.

- [ ] **Step 3: Implement exact head outputs**

```python
@dataclass(frozen=True)
class DecisionThresholds:
    tau_q: float
    tau_reg: float
    tau_unk: float


@dataclass
class OpenHeadOutput:
    class_scores: torch.Tensor
    class_distances: torch.Tensor
    radius_margins: torch.Tensor
    energy: torch.Tensor
    unknown_risk: torch.Tensor


@dataclass
class DecisionResult:
    labels: torch.Tensor
    explicit_unknown: torch.Tensor
    registered: torch.Tensor
    deferred: torch.Tensor
```

Normalize prototypes, parameterize radii with`softplus(log_radius)`, covariance diagonals with`softplus(log_diag)+1e-4`, and an optional rank-8 factor. Define unknown risk as a learned monotonic non-negative combination of normalized minimum distance, positive radius margin and energy. Reject any`tau_reg>tau_unk`.

- [ ] **Step 4: Run head tests including class-label permutation**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_head.py -q`

Expected: PASS; permuting class rows only permutes class scores and leaves unknown risk unchanged.

- [ ] **Step 5: Commit**

```bash
git add code/cvsrffi/phase1_mirage/head.py tests/phase1_mirage/test_head.py
git commit -m "feat: add MIRAGE open-world geometry head"
```

### Task 6: Freeze B0/A/B/C Configuration and Losses

**Files:**
- Create: `code/cvsrffi/phase1_mirage/config.py`
- Create: `code/cvsrffi/phase1_mirage/losses.py`
- Create: `tests/phase1_mirage/test_losses.py`

**Interfaces:**
- Consumes: arm ID, model outputs, labels, teacher outputs, proxy episode and domain-group IDs.
- Produces: frozen`ArmConfig`, accepted pseudo mask and named loss dictionary with scalar`total`.

- [ ] **Step 1: Write failing arm-diff and pseudo-gate tests**

```python
def test_all_arms_share_budget_and_change_only_declared_mechanisms():
    configs = [arm_config(name) for name in ("B0", "A", "B", "C")]
    assert {c.epochs for c in configs} == {200}
    assert {c.encoder for c in configs} == {MIRAGEConfig()}
    assert arm_diff("A", "B0") == {"masked_latent", "cross_receiver", "prototype_pseudo"}
    assert arm_diff("B", "A") == {"proxy_open_loss", "radius_energy", "boundary_mixup"}
    assert arm_diff("C", "B") == {"group_cvar"}


def test_pseudo_label_requires_all_four_conditions():
    mask = pseudo_accept_mask(top1=.96, margin=.21, views_agree=True, inside_radius=True)
    assert mask.item()
    assert not pseudo_accept_mask(top1=.96, margin=.19, views_agree=True, inside_radius=True).item()
```

- [ ] **Step 2: Verify tests fail**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_losses.py -q`

Expected: FAIL with missing config/loss modules.

- [ ] **Step 3: Implement frozen configs and loss functions**

```python
def pseudo_accept_mask(top1, margin, views_agree, inside_radius):
    return (top1 >= 0.95) & (margin >= 0.20) & views_agree & inside_radius


def group_cvar(losses: torch.Tensor, groups: torch.Tensor, *, tail_fraction: float = 0.30) -> torch.Tensor:
    group_means = torch.stack([losses[groups == g].mean() for g in groups.unique(sorted=True)])
    k = max(1, math.ceil(group_means.numel() * tail_fraction))
    return group_means.topk(k).values.mean()
```

Implement CE, weak/strong consistency, masked latent prediction, cross-receiver consistency, prototype-aware pseudo loss, proxy BCE/radius/energy losses and boundary mixup. Proxy rows must never enter registered CE in that episode. Group fallback order is`receiver×day×scene -> receiver×scene -> receiver -> global`when a group has fewer than16 rows.

- [ ] **Step 4: Run focused tests**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_losses.py -q`

Expected: PASS; backward tests show proxy gradient only forB/C and Group-CVaR only forC.

- [ ] **Step 5: Commit**

```bash
git add code/cvsrffi/phase1_mirage/config.py code/cvsrffi/phase1_mirage/losses.py tests/phase1_mirage/test_losses.py
git commit -m "feat: freeze MIRAGE causal arms and losses"
```

### Task 7: Implement the EMA/SWAD Fold Trainer

**Files:**
- Create: `code/cvsrffi/phase1_mirage/trainer.py`
- Create: `tests/phase1_mirage/test_trainer.py`

**Interfaces:**
- Consumes: `TrainConfig`, role-safe loaders and output directory.
- Produces: checkpoint, epoch JSONL/CSV, split/proxy/resource receipts and immutable completion receipt.

- [ ] **Step 1: Write failing no-target and lifecycle tests**

```python
def test_formal_training_requires_200_epochs_and_rejects_target_keys(tmp_path):
    with pytest.raises(TrainingProtocolError, match="200"):
        TrainConfig(arm="B", epochs=2, formal=True)
    with pytest.raises(TrainingProtocolError, match="target"):
        train_fold(config_200, loaders={"l": l, "u": u, "target": target}, output_dir=tmp_path)


def test_cpu_smoke_writes_checkpoint_and_completion(tmp_path):
    result = train_fold(smoke_config(epochs=2), tiny_role_safe_loaders(), output_dir=tmp_path)
    assert result.checkpoint_path.is_file()
    assert result.completion_receipt["status"] == "COMPLETED"
```

- [ ] **Step 2: Verify tests fail**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_trainer.py -q`

Expected: FAIL with missing trainer.

- [ ] **Step 3: Implement the three-stage trainer**

```python
class TrainingProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class TrainConfig:
    arm: Literal["B0", "A", "B", "C"]
    epochs: int = 200
    warmup_end: int = 40
    joint_end: int = 160
    formal: bool = True


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: Path
    completion_receipt: Mapping[str, object]


def train_fold(config: TrainConfig, loaders: Mapping[str, DataLoader], output_dir: Path) -> TrainResult:
    if config.formal and config.epochs != 200:
        raise TrainingProtocolError("formal MIRAGE training requires 200 epochs")
    if any("target" in key.lower() for key in loaders):
        raise TrainingProtocolError("target loader is forbidden")
    required = {"l", "u", "v_cal", "v_select"}
    if set(loaders) != required:
        raise TrainingProtocolError(f"loader roles must equal {sorted(required)}")
    model, head, optimizer = build_models_and_optimizer(config)
    ema = make_ema_copy(model, head)
    swad = make_swad_accumulator(model, head)
    metrics_path = output_dir / "metrics_epoch.jsonl"
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_train_epoch(
            model=model, head=head, ema=ema, optimizer=optimizer,
            labeled_loader=loaders["l"], unlabeled_loader=loaders["u"],
            epoch=epoch, config=config,
        )
        update_ema(ema, model, head, decay=0.999)
        validation = run_source_validation(
            ema=ema, v_cal=loaders["v_cal"], v_select=loaders["v_select"]
        )
        if epoch >= 161:
            update_swad(swad, ema)
        append_epoch_metrics(metrics_path, epoch, train_metrics, validation)
    checkpoint_path = write_final_checkpoint(
        output_dir=output_dir, model=model, head=head, ema=ema, swad=swad,
        config=config,
    )
    receipt = write_completion_receipt(
        output_dir=output_dir, checkpoint_path=checkpoint_path,
        status="COMPLETED", epochs=config.epochs,
    )
    return TrainResult(checkpoint_path=checkpoint_path, completion_receipt=receipt)
```

Implement the named private helpers in the same file:`build_models_and_optimizer`,`make_ema_copy`,`make_swad_accumulator`,`run_train_epoch`,`update_ema`,`run_source_validation`,`update_swad`,`append_epoch_metrics`,`write_final_checkpoint`,`write_completion_receipt`. Update EMA after optimizer steps only; run validation without gradients or state updates; average epochs161-200 into SWAD; select checkpoint from`V_cal`known macro/worst-scene with a predeclared lexicographic rule. Persist RNG states, config hash, split hash, Git commit and state-dict SHA256.

- [ ] **Step 4: Run trainer smoke and state-update tests**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_trainer.py -q`

Expected: PASS; validation leaves model/EMA/prototype bytes unchanged.

- [ ] **Step 5: Commit**

```bash
git add code/cvsrffi/phase1_mirage/trainer.py tests/phase1_mirage/test_trainer.py
git commit -m "feat: add MIRAGE source fold trainer"
```

### Task 8: Implement Source Calibration, Metrics and Gates

**Files:**
- Create: `code/cvsrffi/phase1_mirage/calibration.py`
- Create: `code/cvsrffi/phase1_mirage/scoring.py`
- Create: `tests/phase1_mirage/test_calibration.py`
- Create: `tests/phase1_mirage/test_scoring.py`

**Interfaces:**
- Consumes: immutable`V_cal/P_cal`score tables and`V_select/P_select`score tables.
- Produces: frozen`DecisionThresholds`, same-row metrics, Gate receipts and unique arm ID.

- [ ] **Step 1: Write failing threshold and non-compensation tests**

```python
def test_calibration_enforces_known_frr_and_returns_no_solution():
    with pytest.raises(NoDeployableSeparation):
        calibrate_thresholds(known_scores=inseparable_known, proxy_scores=inseparable_proxy, max_known_frr=.10)


def test_gate_failure_cannot_be_compensated_by_other_metrics():
    receipt = evaluate_source_gates(candidate=great_macro_bad_proxy, baseline=b0)
    assert receipt.gate2_pass
    assert not receipt.gate3_pass
    assert not receipt.promoted
```

- [ ] **Step 2: Verify tests fail**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_calibration.py tests/phase1_mirage/test_scoring.py -q`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement exact calibration and same-row metrics**

```python
class NoDeployableSeparation(RuntimeError):
    pass


def calibrate_thresholds(known_scores, proxy_scores, *, max_known_frr: float = 0.10) -> DecisionThresholds:
    candidates = empirical_threshold_grid(known_scores, proxy_scores)
    feasible = [t for t in candidates if known_false_rejection(known_scores, t) <= max_known_frr]
    if not feasible:
        raise NoDeployableSeparation("NO_DEPLOYABLE_SEPARATION")
    return max(feasible, key=lambda t: (proxy_rejection(proxy_scores, t), registered_coverage(known_scores, t), -defer_rate(known_scores, t)))
```

Use`sklearn.metrics.roc_auc_score`. Compute macro, per-class, minimum-class, receiver, day, scene, worst-scene, FRR, explicit unknown rejection, false accept, coverage and defer from one score table. Aggregate sixfold values with equal fold/scene weights and implement the exact Gate2/3 deltas from the design spec.

- [ ] **Step 4: Verify focused tests**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_calibration.py tests/phase1_mirage/test_scoring.py -q`

Expected: PASS; a5/6 non-degradation boundary test passes exactly and4/6 fails.

- [ ] **Step 5: Commit**

```bash
git add code/cvsrffi/phase1_mirage/calibration.py code/cvsrffi/phase1_mirage/scoring.py tests/phase1_mirage/test_calibration.py tests/phase1_mirage/test_scoring.py
git commit -m "feat: add source-only MIRAGE calibration and gates"
```

### Task 9: Add Frozen Configs and Source/Refit Entrypoints

**Files:**
- Create: `configs/phase1_mirage_owdg/b0.json`
- Create: `configs/phase1_mirage_owdg/a.json`
- Create: `configs/phase1_mirage_owdg/b.json`
- Create: `configs/phase1_mirage_owdg/c.json`
- Create: `configs/phase1_mirage_owdg/source_matrix.json`
- Create: `configs/phase1_mirage_owdg/final_refit.json`
- Create: `code/scripts/run_phase1_mirage_source_matrix.py`
- Create: `code/scripts/run_phase1_mirage_final_refit.py`
- Create: `scripts/launchers/run_phase1_mirage_source_matrix.sh`
- Create: `scripts/launchers/run_phase1_mirage_final_refit.sh`
- Create: `tests/phase1_mirage/test_cli.py`

**Interfaces:**
- Consumes: frozen JSON config, ManySig path, run ID and output root.
- Produces: dry-run matrix,24 source rows,2 final-refit rows and structured completion manifests.

- [ ] **Step 1: Write failing matrix and parser tests**

```python
def test_source_dry_run_has_exactly_24_unique_rows():
    rows = build_source_rows(load_matrix("configs/phase1_mirage_owdg/source_matrix.json"))
    assert len(rows) == 24
    assert len({row.row_id for row in rows}) == 24
    assert {(row.arm, row.fold) for row in rows} == {(a, f) for a in ("B0", "A", "B", "C") for f in range(1, 7)}


def test_final_refit_accepts_only_source_promoted_arm_and_b0():
    assert [row.role for row in build_refit_rows(frozen_arm="C")] == ["B0_STAR", "M_STAR"]
```

- [ ] **Step 2: Verify tests fail**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_cli.py -q`

Expected: FAIL with missing files.

- [ ] **Step 3: Implement exact configs and CLIs**

All arm JSON files must contain`epochs=200`,`labeled_ratio=.07`,`unlabeled_ratio=.63`,`validation_ratio=.30`,`proxy_source=train_l_or_validation_only`,`target_access=false`,`encoder_schema=mirage_owdg:z_id:l2:160:v1`. The source CLI must refuse an existing output directory, a short Git commit, config-hash mismatch or any argument containing`target`.

The source launcher must map24 rows deterministically acrossGPU0-7 with at most two live rows per GPU and support`DRY_RUN=1`; final refit must launch only`B0_STAR`and`M_STAR`.

- [ ] **Step 4: Run parser, dry-run and shell checks**

```bash
conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_cli.py -q
bash -n scripts/launchers/run_phase1_mirage_source_matrix.sh
bash -n scripts/launchers/run_phase1_mirage_final_refit.sh
DRY_RUN=1 bash scripts/launchers/run_phase1_mirage_source_matrix.sh --dry-run | wc -l
```

Expected: tests PASS, both shell scripts parse, dry-run line count is24.

- [ ] **Step 5: Commit**

```bash
git add configs/phase1_mirage_owdg code/scripts/run_phase1_mirage_source_matrix.py code/scripts/run_phase1_mirage_final_refit.py scripts/launchers/run_phase1_mirage_source_matrix.sh scripts/launchers/run_phase1_mirage_final_refit.sh tests/phase1_mirage/test_cli.py
git commit -m "feat: add MIRAGE source and refit runners"
```

### Task 10: Export and Reload the Immutable Deployment Bundle

**Files:**
- Create: `code/cvsrffi/phase1_mirage/bundle.py`
- Create: `code/scripts/build_phase1_mirage_bundle.py`
- Create: `tests/phase1_mirage/test_bundle.py`

**Interfaces:**
- Consumes: final checkpoint, registered geometry, thresholds and opaque class handles.
- Produces: one zip bundle with exact members and a verified production runtime.

- [ ] **Step 1: Write failing exact-member and round-trip tests**

```python
def test_bundle_roundtrip_is_finite_small_and_source_free(tmp_path):
    path = export_bundle(tiny_final_state(), tmp_path / "mirage.zip")
    assert path.stat().st_size <= 16 * 1024 * 1024
    loaded = load_bundle(path)
    before = tiny_iq()
    assert_prediction_equal(loaded.predict(before), reference_predict(before), atol=1e-5)
    manifest = loaded.manifest
    assert not any(token in json.dumps(manifest).lower() for token in ("dataset", "source_path", "target_truth", "cache"))
```

- [ ] **Step 2: Verify tests fail**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_bundle.py -q`

Expected: FAIL with missing bundle module.

- [ ] **Step 3: Implement exact-member zip export/load**

Freeze these members only:

```text
manifest.json
runtime/mirage_runtime.torchscript.pt
geometry/registered_geometry.npz
locks/decision_thresholds.json
locks/class_handles.json
```

Trace a production wrapper that returns tensors in this order:`identity_embedding,registered_class_scores,class_distance_or_radius,unknown_score,quality,decision_code`. Validate member allowlist, per-member SHA256, total size, finite geometry, unique opaque class handles and`allow_pickle=False`forNPZ. Reject symlinks, path traversal and extra members.

- [ ] **Step 4: Run bundle tests and TorchScript parity**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_bundle.py -q`

Expected: PASS; production reload predictions match reference within`1e-5`and bundle≤16MiB.

- [ ] **Step 5: Commit**

```bash
git add code/cvsrffi/phase1_mirage/bundle.py code/scripts/build_phase1_mirage_bundle.py tests/phase1_mirage/test_bundle.py
git commit -m "feat: add immutable MIRAGE deployment bundle"
```

### Task 11: Separate Truth-Blind Target Prediction from Scoring

**Files:**
- Create: `code/cvsrffi/phase1_mirage/target.py`
- Create: `code/scripts/predict_phase1_mirage_target.py`
- Create: `code/scripts/score_phase1_mirage_target.py`
- Create: `tests/phase1_mirage/test_target.py`

**Interfaces:**
- Predictor consumes: verified LEO cache, bundle and output path only.
- Scorer consumes: sealed prediction JSONL plus independent truth JSONL.
- Produces: same-row target-known/unknown metrics and Gate4 receipt.

- [ ] **Step 1: Write failing CLI-boundary and state-immutability tests**

```python
def test_predictor_parser_has_no_truth_or_role_arguments():
    options = {action.dest for action in build_predict_parser()._actions}
    assert not options & {"truth", "label", "known_role", "unknown_role", "class_quota"}


def test_prediction_does_not_mutate_bundle_state(tmp_path):
    runtime = load_bundle(tiny_bundle(tmp_path))
    before = runtime_state_sha256(runtime)
    predict_cache(runtime, tiny_role_blind_cache(), tmp_path / "predictions.jsonl")
    assert runtime_state_sha256(runtime) == before
```

- [ ] **Step 2: Verify tests fail**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_target.py -q`

Expected: FAIL with missing target module.

- [ ] **Step 3: Implement sealed prediction and independent scoring**

```python
@dataclass(frozen=True)
class PredictionRecord:
    opaque_query_id: str
    scene: str
    registered_scores: Sequence[float]
    min_class_distance: float
    unknown_score: float
    quality: float
    decision: Literal["registered", "unknown", "defer"]
    predicted_class_handle: str | None
```

The predictor must use`load_verified_leo_weak_cache_set`, process every query independently against all registered classes, write canonical JSONL and a seal containing row count/file SHA/bundle SHA. The scorer must verify the seal first, join truth by opaque ID, and compute global/per-scene known macro, minimum-class, FRR, unknown rejection, false accept, coverage and defer. It must not import`trainer`or write into the bundle directory.

- [ ] **Step 4: Run target tests**

Run: `conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_target.py -q`

Expected: PASS; duplicate/missing query IDs, extra truth rows and modified prediction bytes fail closed.

- [ ] **Step 5: Commit**

```bash
git add code/cvsrffi/phase1_mirage/target.py code/scripts/predict_phase1_mirage_target.py code/scripts/score_phase1_mirage_target.py tests/phase1_mirage/test_target.py
git commit -m "feat: add blind MIRAGE target prediction and scoring"
```

### Task 12: Local Integration, Reports and Review A

**Files:**
- Create: `automation_reports/CV-SincNet/phase1_mirage_source6_20260817_v1/report.md`
- Mirror: `E:\type10-7\automation_reports\CV-SincNet\phase1_mirage_source6_20260817_v1\report.md`
- Modify only files identified by focused failures fromTasks1-11.

**Interfaces:**
- Consumes: all core implementation commits.
- Produces: `LOCAL_VERIFIED`, source report preregistration and independentP0/P1 review with`P0=0,P1=0`.

- [ ] **Step 1: Write the report before any N607 handoff**

Record objective, hypothesis, B0/A/B/C matrix,6 folds, seeds, dataset path/hash, exact local files, verification commands, commit, config hashes, N607 release/run/log paths, GPU plan, expected artifacts, Gate1-3, systemic-stop rule and target access=`0`.

- [ ] **Step 2: Run the full focused suite serially in`ssr-gpu`**

```bash
conda run -n ssr-gpu python -c "import os,sys; print(sys.executable); print(os.environ.get('CONDA_PREFIX'))"
conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage -q
conda run -n ssr-gpu python -m py_compile code/cvsrffi/phase1_mirage/*.py code/scripts/*phase1_mirage*.py
bash -n scripts/launchers/run_phase1_mirage_source_matrix.sh
bash -n scripts/launchers/run_phase1_mirage_final_refit.sh
git diff --check
```

Expected: interpreter and prefix contain`ssr-gpu`; all tests PASS; compile/shell/diff checks PASS.

- [ ] **Step 3: Run a real-data no-target smoke and checkpoint forward**

Run a two-epoch smoke on a small real ManySig source slice, save its checkpoint, reload that checkpoint and execute one no-target forward. Confirm inputs contain only`L_s/U_s/V_s`, output schema is complete and model/bundle forward is finite. This is technical evidence only and must not be interpreted as performance.

- [ ] **Step 4: Obtain independent Review A**

The reviewer checks protocol docs, split/proxy provenance, no-target negative tests, arm parity, class-permutation symmetry, checkpoint/bundle forward and allP0/P1 findings. Fix each concreteP0/P1 with a focused failing test, then rerun only affected tests plus the full focused suite. Do not addP2 release machinery.

- [ ] **Step 5: Commit the verified implementation and report**

```bash
git add code/cvsrffi/phase1_mirage code/scripts/*phase1_mirage*.py configs/phase1_mirage_owdg scripts/launchers/run_phase1_mirage_*.sh tests/phase1_mirage automation_reports/CV-SincNet/phase1_mirage_source6_20260817_v1/report.md docs/PROJECT_PROTOCOL.md
git commit -m "feat: implement MIRAGE Phase1 source pipeline"
```

### Task 13: Release and Complete the N607 Source Matrix

**Files:**
- Update: `automation_reports/CV-SincNet/phase1_mirage_source6_20260817_v1/report.md`
- Retrieve under: `automation_reports/CV-SincNet/phase1_mirage_source6_20260817_v1/artifacts/`

**Interfaces:**
- Consumes: frozen implementation commit and24-row source matrix.
- Produces: complete same-row source artifacts, Gate1-3 receipt and unique promoted arm or valid source failure.

- [ ] **Step 1: Freeze release identity and exact remote paths**

```bash
RUN_ID=phase1_mirage_source6_20260817_v1
IMPLEMENTATION_COMMIT=$(git rev-parse HEAD)
RELEASE=/home/szu2070436088/2510044040/CV-SincNet/releases/${RUN_ID}_${IMPLEMENTATION_COMMIT:0:8}
RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/${RUN_ID}
LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/${RUN_ID}
DATASET=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl
REMOTE_PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
```

Record full commit and SHA256 for every synced member. Expected dataset SHA256 is`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`.

- [ ] **Step 2: Delegate one sole N607 runner with the complete frozen handoff**

The handoff includes role, objective, local/remote paths, exact file ownership,24 rows, sixfold seeds, command, environment, GPU map, expected artifacts, no-tuning list, health schedule, deterministic exception fingerprint rule and fresh-run retry=`not authorized without main-agent review`. The primary agent must not launch the same run.

- [ ] **Step 3: Run direct preflight, land exact files and verify remotely**

The runner uses the reviewed local preflight, direct`N607`first and the verified bridge only if direct transport fails. It verifies GPU/process occupancy, all new paths`ABSENT`, dataset hash, archive/member hashes, remote`py_compile`, CLI help, both`bash -n`checks and exact24-line dry-run. Every SSH/SCP command disconnects and local`ssh.exe`/TCP22 cleanup is verified after ambiguity.

- [ ] **Step 4: Launch once and monitor technical health**

Remote launch command shape:

```bash
cd "$RELEASE/code" && nohup env RUN_ID="$RUN_ID" PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT="$RELEASE/code" DATASET="$DATASET" REMOTE_PYTHON="$REMOTE_PYTHON" bash scripts/launchers/run_phase1_mirage_source_matrix.sh > "/home/szu2070436088/2510044040/CV-SincNet/logs/${RUN_ID}_launcher.out" 2>&1 < /dev/null &
```

Verify PID/CWD/cmdline/run-root/GPU binding and first-wave launched/completed/succeeded/failed/prediction counts. Stop only forP0 protocol/safety fault or two distinct rows with the same deterministic pre-prediction exception fingerprint; never stop for poor accuracy.

- [ ] **Step 5: Retrieve artifacts, score all rows and update the report**

Require24/24 terminal rows, checkpoints, score tables, threshold receipts, resource receipts and no-target access receipts. Evaluate Gate1-3 from same-row data, list everyfold/arm, and either freeze one promoted arm or record`SOURCE_GATE_FAIL / TARGET_NOT_ACCESSED`. Commit the report and small artifacts; do not download forbidden large datasets or delete remote outputs.

### Task 14: Final Refit, Bundle and Review B

**Files:**
- Create/update: `automation_reports/CV-SincNet/phase1_mirage_final_refit_20260817_v1/report.md`
- Retrieve under: `automation_reports/CV-SincNet/phase1_mirage_final_refit_20260817_v1/artifacts/`

**Interfaces:**
- Consumes: one Gate1-3-approved arm and frozen source-only selection receipt.
- Produces: `M*`, `B0*`, two frozen thresholds/geometries, two valid bundles and Review B approval.

- [ ] **Step 1: Preregister the two-row refit**

Set`RUN_ID=phase1_mirage_final_refit_20260817_v1`. Record the promoted arm, source evidence hashes, final seed, exact two rows`B0_STAR/M_STAR`,200 epochs, source-only checkpoint selection, paths, GPU map, expected bundles and target access=`0`.

- [ ] **Step 2: Use a sole runner to launch the exact two-row N607 refit**

Repeat direct preflight, new-path`ABSENT`, commit/member/dataset hashes, compile/help/shell/dry-run and process binding. The runner cannot change the promoted arm or reuse a source-fold checkpoint.

- [ ] **Step 3: Require natural completion and build both bundles**

Each row must produce a final checkpoint, registered geometry, source thresholds, checkpoint receipt, score tables and resource receipt. Run`build_phase1_mirage_bundle.py`for each; require production reload parity, finite outputs, exact member allowlist and≤16MiB.

- [ ] **Step 4: Run Review B**

The independent reviewer verifies one frozen candidate, B0 comparator, real checkpoint forwards, target predictor has no truth/role inputs, thresholds come only from source, target command is exact, output paths are absent/non-overwriting and no third review layer is added. Require`P0=0,P1=0`before target access.

- [ ] **Step 5: Commit frozen artifacts and report**

Commit small manifests, SHA lists, review receipt and updated report. Large checkpoints/bundles remain on preserved N607 paths with hashes and retrieval instructions unless their size fits repository policy.

### Task 15: Run the One-Time Target Confirmation and Final Gate Audit

**Files:**
- Create/update: `automation_reports/CV-SincNet/phase1_mirage_target_confirm_20260817_v1/report.md`
- Retrieve under: `automation_reports/CV-SincNet/phase1_mirage_target_confirm_20260817_v1/artifacts/`

**Interfaces:**
- Consumes: Review-B-approved`M*`,`B0*`, their frozen thresholds and one validated target capsule.
- Produces: sealed predictions, independent scores, Gate4/5 receipts and final five-Gate verdict.

- [ ] **Step 1: Preregister the immutable target command before opening data**

Set`RUN_ID=phase1_mirage_target_confirm_20260817_v1`. Record target receiver, known/unknown TX registries, capsule/split IDs, three scene physical-ID roots, bundle hashes, prediction and score paths, and exact commands forB0*andM*. Confirm target unknown TX intersection with source train/validation TX is empty.

- [ ] **Step 2: Run the predictor once for each frozen bundle**

The sole runner verifies all output paths`ABSENT`, then runs`predict_phase1_mirage_target.py`forB0*andM*with the same capsule. Predictor arguments contain no truth/role path. Verify row counts, scene counts, prediction seals and unchanged bundle hashes before any scorer starts.

- [ ] **Step 3: Score only after both prediction artifacts are sealed**

Run`score_phase1_mirage_target.py`with independent truth. Produce same-row global/per-scene known macro, minimum-class, worst-scene, FRR, unknown rejection, false accept, coverage and defer forB0*andM*. Preserve confusion counts and per-class known accuracy.

- [ ] **Step 4: Apply Gate4 and Gate5 without feedback**

Require target-known macro gain≥2pp, minimum-class/worst-scene nonlower, global/clear/low-elev/rain explicit unknown rejection each≥70%, known FRR≤10%, bundle reload/output/resource closure. If any item fails, record`VALID_TARGET_FAILURE / NO_RETUNE_NO_RERUN`.

- [ ] **Step 5: Complete the final report and commit evidence**

The report includes full same-row tables, anomalies, target-blind receipts, bundle/resource data, all five non-compensating Gate decisions and the permitted claim boundary. Commit the report and small immutable artifacts. Do not modify the method, thresholds, bundles, target capsule or predictions after scoring.

---

## Plan Self-Review

- Spec coverage: Tasks1-3 cover protocol,split和proxy；Tasks4-7 cover encoder、head、loss和trainer；Tasks8-9 cover source calibration、Gates和矩阵；Tasks10-11 cover bundle与target盲评接口；Tasks12-15 cover本地验证、两次审查、N607 source矩阵、final refit和一次性target确认。
- Placeholder scan: the plan contains no unresolved implementation token. Runtime commit/path values are derived by executable shell variables; run IDs, dataset path/hash, remote Python and report paths are fixed.
- Type consistency: `MIRAGEConfig`、`MIRAGEFeatures`、`DecisionThresholds`、`OpenHeadOutput`、`TrainConfig`、`PredictionRecord`and all producer/consumer relationships are named once and reused consistently.
- Scope: protocol, code, source evidence and target confirmation form one sequential dependency chain. Target tasks remain unreachable unless the source Gate and Review B tasks succeed.
