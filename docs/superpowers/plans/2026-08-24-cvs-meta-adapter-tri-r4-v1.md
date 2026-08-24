# CVS_META_ADAPTER_TRI_R4_V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在ADV3B02基础上实现Phase1 source-only元训练和Phase2 adapter-only三步快速域适应，使support梯度由独立source meta-query训练，并以同row`DA0_REG0/DA1_REG0`验证真实目标域收益。

**Architecture:** 在CVSincNet的time、freq和fusion三个256维表示点插入rank-4残差adapter；Phase1以分层source episode执行FOMAML和模块级Meta-SGD，Phase2只更新adapter权重与门。数据episode、元目标、内循环、训练器、checkpoint、Phase2运行器和scorer分别放入小型模块，`train.py`只保留参数与入口路由。

**Tech Stack:** Python 3、PyTorch、NumPy、pytest、现有CV-SincNet/ADV3B02、JSON/NPZ、N607 CUDA环境、Git。

**Spec:** `docs/CVS_META_ADAPTER_TRI_R4_V1_DESIGN_20260824.md`

## Global Constraints

- Phase1数据角色固定为`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；有监督meta support/query只能来自`L_s`。
- target receiver不得进入Phase1训练、校准或选择；`U_s`真值始终不可见；`V_cal/V_select`不得反向传播。
- support/query physical sample ID严格不相交；同一物理样本的不同clean/LEO视图不能跨边界。
- Phase1卫星增强严格复用ADV3B02 Core90 LEO_WEAK日程和`lambda_sat_cls=0.68`、`lambda_sat_cons=0`。
- Phase2只接受`p2_min_v1`、`VALIDATED_ONCE`及匹配的`capsule_id/split_id`，不得因adapter或checkpoint变化重验数据。
- Phase2只读取target support标签；source/clean/cache和query truth/role不可达；query不得更新任何参数或buffer。
- 不新增或训练协方差、LDA或持久分类头；CosFace头、原型和类别映射冻结。
- V1固定三个rank-4 adapter、FOMAML、模块级Meta-SGD、Phase2主更新3步；Phase2可训练参数≤总参数1%，V1部署最多5步、硬上限40步；Phase1 source诊断允许报告A(10)。
- V1不实现二阶MAML、每参数Meta-SGD、`z_dom`条件初始化、多专家、IQ输入adapter或新类support联合encoder适配。
- 所有本地代码测试先进入`ssr-gpu`环境；本计划中的`python`命令均假定当前shell已执行`conda activate ssr-gpu`。
- 每个任务只stage列出的文件，提交后自动push并回读远端OID；不得stage工作树中既有`.pytest_tmp/`、`local_artifacts/`或其他无关文件。
- 代码任务完成后只进行一次独立P0/P1审查；P2和额外形式gate不得阻塞发布。

---

## File Map

### 数据和episode

- Modify: `code/dataset_wisig.py`——规范physical sample ID和capture block元数据。
- Replace/extend: `code/cvsrffi/meta_episodes.py`——类型化分层episode生成器。
- Create: `code/tests/test_meta_episode_sampler_v1.py`——角色、物理隔离、任务混合和类别guard测试。

### 模型和元优化

- Create: `code/cvsrffi/meta_adapter.py`——rank-4残差adapter、参数白名单和步长查询。
- Modify: `code/model.py`——time/freq/fusion adapter hook。
- Modify: `code/model_dual_cvsincnet.py`——向identity backbone传递adapter构造参数。
- Create: `code/cvsrffi/meta_checkpoint.py`——旧ADV3B02到meta模型的受控迁移、严格meta checkpoint读写和参数预算审计。
- Create: `code/cvsrffi/meta_objectives.py`——support、adapt、guard、floor、topology和zero-step目标。
- Create: `code/cvsrffi/meta_inner_loop.py`——FOMAML adapter-only函数式内循环。
- Create: `code/cvsrffi/meta_trainer.py`——Phase1-B/Phase1-C训练、评估与checkpoint选择。

### Phase1入口

- Create: `code/cvsrffi/meta_phase1_entry.py`——数据、episode、模型、训练器和artifact编排。
- Modify: `code/train.py`——薄CLI入口。
- Create: `configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824.json`——冻结训练配置。
- Create: `code/scripts/launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py`——本地dry-run和N607启动命令生成器。

### Phase2运行和评分

- Create: `code/cvsrffi/stage2_meta_adapter_adaptation.py`——support-only三步适配核心。
- Create: `code/cvsrffi/stage2_meta_adapter_runner.py`——同row DA0/DA1 prediction闭合。
- Create: `code/cvsrffi/stage2_meta_adapter_scorer.py`——truth-last配对评分和晋级判断。
- Create: `code/cvsrffi/stage2_meta_adapter_handoff.py`——`DA1_REG0`到Stage2-C的冻结状态交接。
- Create: `code/scripts/run_stage2_meta_adapter.py`——单row CLI。
- Create: `code/scripts/smoke_stage2_meta_adapter_no_query.py`——真实checkpoint无query smoke。
- Create: `code/scripts/score_stage2_meta_adapter.py`——单rowtruth-last scorer CLI。
- Create: `code/scripts/summarize_stage2_meta_adapter_matrix.py`——Target5/Target25聚合。

### 测试、报告和追踪

- Create: `code/tests/test_meta_adapter_model_v1.py`
- Create: `code/tests/test_meta_checkpoint_v1.py`
- Create: `code/tests/test_meta_objectives_v1.py`
- Create: `code/tests/test_meta_inner_loop_v1.py`
- Create: `code/tests/test_meta_trainer_v1.py`
- Create: `code/tests/test_meta_phase1_entry_v1.py`
- Create: `tests/test_stage2_meta_adapter_adaptation.py`
- Create: `tests/test_stage2_meta_adapter_runner.py`
- Create: `tests/test_stage2_meta_adapter_scorer.py`
- Create: `tests/test_stage2_meta_adapter_handoff.py`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`
- Create during execution: `automation_reports/CV-SincNet/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1/report.md`
- Create during execution: `automation_reports/CV-SincNet/adv3b02_stage2b_meta_adapter_tri_r4_t5t25_s713101_20260824_r1/report.md`
- Create mirror: `docs/experiments/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1_report.md`
- Create mirror: `docs/experiments/adv3b02_stage2b_meta_adapter_tri_r4_t5t25_s713101_20260824_r1_report.md`

---

### Task 1: WiSig Physical Sample Identity and Capture Blocks

**Traceability:** `META-01`、`META-04`

**Files:**
- Modify: `code/dataset_wisig.py:101-243`
- Create: `code/tests/test_meta_episode_sampler_v1.py`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Consumes: `WiSigIndex(tx_i,rx_i,day_i,eq_i,sig_i)`。
- Produces: `wisig_physical_sample_id(item) -> str`、`wisig_capture_block_id(item, block_size) -> int`；dataset metadata新增`physical_sample_id`和`capture_block_i`。

- [ ] **Step 1: 写physical ID和capture block的失败测试**

```python
from dataset_wisig import WiSigIndex, wisig_capture_block_id, wisig_physical_sample_id


def test_wisig_physical_id_is_complete_and_stable():
    item = WiSigIndex(tx_i=2, rx_i=3, day_i=1, eq_i=0, sig_i=19)
    assert wisig_physical_sample_id(item) == "tx2|rx3|day1|eq0|sig19"


def test_capture_block_uses_sig_index_without_claiming_real_channel():
    item = WiSigIndex(tx_i=2, rx_i=3, day_i=1, eq_i=0, sig_i=19)
    assert wisig_capture_block_id(item, block_size=8) == 2
```

- [ ] **Step 2: 运行测试并确认RED**

Run: `python -m pytest code/tests/test_meta_episode_sampler_v1.py -q`

Expected: collection或import失败，指出两个函数尚不存在。

- [ ] **Step 3: 实现规范ID并写入metadata**

```python
def wisig_physical_sample_id(item: WiSigIndex) -> str:
    return (
        f"tx{int(item.tx_i)}|rx{int(item.rx_i)}|day{int(item.day_i)}|"
        f"eq{int(item.eq_i)}|sig{int(item.sig_i)}"
    )


def wisig_capture_block_id(item: WiSigIndex, block_size: int = 8) -> int:
    if int(block_size) <= 0:
        raise ValueError("capture block_size must be positive")
    return int(item.sig_i) // int(block_size)
```

在`WiSigCompactDataset.__getitem__`的`meta`中加入：

```python
"physical_sample_id": wisig_physical_sample_id(it),
"capture_block_i": wisig_capture_block_id(it, self.capture_block_size),
"capture_block_semantics": "sig_index_time_block_proxy",
```

- [ ] **Step 4: 验证GREEN和邻近dataset回归**

Run: `python -m pytest code/tests/test_meta_episode_sampler_v1.py tests/test_baseline_training_behaviors.py -q`

Expected: 新测试通过；既有WiSig索引和split测试无回归。

- [ ] **Step 5: 更新追踪表并提交**

将`META-04`状态更新为`implemented`，verification写入测试命令。

```text
git add code/dataset_wisig.py code/tests/test_meta_episode_sampler_v1.py docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md
git commit -m "feat: add stable WiSig meta sample identities"
```

---

### Task 2: Hierarchical Meta-Episode Sampler

**Traceability:** `META-02`、`META-03`、`META-04`

**Files:**
- Modify: `code/cvsrffi/meta_episodes.py`
- Modify: `code/tests/test_meta_episode_sampler_v1.py`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Consumes: `Sequence[MetaSampleRef]`，每项包含dataset index、TX、receiver、day、eq、capture block、physical ID、role和view。
- Produces: `HierarchicalMetaEpisodeSampler.sample(seed: int) -> MetaEpisode`；训练sampler固定`allowed_roles=("L_s",)`，无梯度评估sampler固定`allowed_roles=("V_cal","V_select")`。

- [ ] **Step 1: 写类型、隔离和任务比例失败测试**

```python
from cvsrffi.meta_episodes import (
    EpisodeKind,
    HierarchicalMetaEpisodeSampler,
    MetaEpisodeSamplerConfig,
    MetaSampleRef,
)


def test_episode_never_overlaps_physical_ids_or_non_labeled_roles():
    sampler = HierarchicalMetaEpisodeSampler(
        refs=make_balanced_refs(),
        config=MetaEpisodeSamplerConfig(k_choices=(1, 2), query_per_class=2),
    )
    episode = sampler.sample(seed=73)
    support_ids = {row.physical_sample_id for row in episode.support}
    query_ids = {row.physical_sample_id for row in episode.query_adapt + episode.query_guard}
    assert support_ids.isdisjoint(query_ids)
    assert {row.role for row in episode.support + episode.query_adapt + episode.query_guard} == {"L_s"}


def test_query_only_classes_are_guard_not_adapt():
    episode = make_guard_episode(seed=11)
    assert episode.adapt_class_ids == frozenset({0, 1, 2})
    assert episode.guard_class_ids == frozenset({3, 4, 5})
    assert all(row.tx_i in episode.guard_class_ids for row in episode.query_guard)
```

- [ ] **Step 2: 运行测试并确认RED**

Run: `python -m pytest code/tests/test_meta_episode_sampler_v1.py -q`

Expected: `EpisodeKind`和sampler接口不存在。

- [ ] **Step 3: 实现类型化episode API**

```python
class EpisodeKind(str, Enum):
    SAME_DOMAIN = "Q_SAME_DOMAIN"
    RX_HOLDOUT = "Q_RX_HOLDOUT"
    DAY_CHANNEL_HOLDOUT = "Q_DAY_CHANNEL_HOLDOUT"
    CLEAN_TO_LEO = "Q_CLEAN_TO_LEO"
    LEO_CROSS = "Q_LEO_CROSS"


@dataclass(frozen=True)
class MetaSampleRef:
    dataset_index: int
    tx_i: int
    rx_i: int
    day_i: int
    eq_i: int
    capture_block_i: int
    physical_sample_id: str
    role: str
    view: str


@dataclass(frozen=True)
class MetaEpisode:
    kind: EpisodeKind
    support: tuple[MetaSampleRef, ...]
    query_adapt: tuple[MetaSampleRef, ...]
    query_guard: tuple[MetaSampleRef, ...]
    adapt_class_ids: frozenset[int]
    guard_class_ids: frozenset[int]
    k_shot: int
    seed: int
```

`MetaEpisodeSamplerConfig`同时包含`allowed_roles: tuple[str,...]`和`training: bool`。`training=True`时只接受`("L_s",)`；`training=False`时只接受`("V_cal","V_select")`，且trainer必须在`torch.no_grad()`评估路径消费。

`MetaEpisodeSamplerConfig`固定默认混合：

```python
episode_weights = {
    EpisodeKind.SAME_DOMAIN: 0.40,
    EpisodeKind.RX_HOLDOUT: 0.20,
    EpisodeKind.DAY_CHANNEL_HOLDOUT: 0.15,
    EpisodeKind.CLEAN_TO_LEO: 0.15,
    EpisodeKind.LEO_CROSS: 0.10,
}
```

sampler在最终返回前必须显式断言角色、physical ID和类别路由。

- [ ] **Step 4: 增加五类episode、固定seed和标签置换测试**

Run: `python -m pytest code/tests/test_meta_episode_sampler_v1.py -q`

Expected: 五类任务均可构造；相同seed完全一致；不同seed产生不同合法episode；label permutation后任务计数不变。

- [ ] **Step 5: 保留旧API兼容并运行Meta-SSL回归**

旧`sample_rxday_episode()`保留为兼容包装，不进入新训练主路径。

Run: `python -m pytest code/tests/test_meta_ssl_split.py code/tests/test_meta_ssl_train_loop.py code/tests/test_meta_episode_sampler_v1.py -q`

Expected: 全部通过。

- [ ] **Step 6: 更新追踪表并提交**

```text
git add code/cvsrffi/meta_episodes.py code/tests/test_meta_episode_sampler_v1.py docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md
git commit -m "feat: add hierarchical source meta episodes"
```

---

### Task 3: Rank-4 Adapter and ADV3B02 Model Hooks

**Traceability:** `META-05`、`META-06`、`META-11`

**Files:**
- Create: `code/cvsrffi/meta_adapter.py`
- Modify: `code/model.py:897-957,1462-1538,1692-1730`
- Modify: `code/model_dual_cvsincnet.py`
- Create: `code/tests/test_meta_adapter_model_v1.py`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Produces: `ResidualMetaAdapter(dim: int, rank: int, init_step_size: float)`。
- Produces: `iter_inner_adapter_parameters(model)`、`adapter_step_size_by_parameter(model)`、`adapter_parameter_budget(model)`。
- `CVSincNet`新增构造参数`meta_adapter_rank: int=0`和`meta_adapter_sites: str=""`；默认值完全保持旧模型结构和state dict。

- [ ] **Step 1: 写模型结构和默认兼容失败测试**

```python
def test_rank_zero_preserves_legacy_state_dict_keys():
    legacy = build_model(dataset="wisig", input_len=256, meta_adapter_rank=0)
    assert not any("meta_adapter" in key for key in legacy.state_dict())


def test_tri_r4_has_only_three_adapter_sites_and_near_identity_step0():
    model = build_model(
        dataset="wisig",
        input_len=256,
        meta_adapter_rank=4,
        meta_adapter_sites="time,freq,fusion",
    ).eval()
    names = [name for name, _ in model.named_parameters() if "meta_adapter" in name]
    assert {name.split(".")[0] for name in names} == {
        "meta_adapter_time", "meta_adapter_freq", "meta_adapter_fusion"
    }
```

- [ ] **Step 2: 运行测试并确认RED**

Run: `python -m pytest code/tests/test_meta_adapter_model_v1.py -q`

Expected: builder不接受`meta_adapter_rank`。

- [ ] **Step 3: 实现adapter和参数白名单**

```python
class ResidualMetaAdapter(nn.Module):
    def __init__(self, dim: int, rank: int = 4, init_step_size: float = 1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.down = nn.Linear(dim, rank, bias=True)
        self.up = nn.Linear(rank, dim, bias=True)
        self.gate = nn.Parameter(torch.tensor(0.01))
        self.log_step_size = nn.Parameter(torch.log(torch.tensor(init_step_size)))
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.up(F.silu(self.down(self.norm(z))))
        return z + torch.tanh(self.gate) * delta

    def step_size(self) -> torch.Tensor:
        return F.softplus(self.log_step_size).clamp(1e-6, 5e-2)
```

`iter_inner_adapter_parameters()`必须排除`log_step_size`，但包含`down/up/gate`。

- [ ] **Step 4: 在三个明确位置接入hook**

```python
# time branch hook
t_emb = self.meta_adapter_time(self.t_proj(t))

# frequency branch hook after the existing statistics residual
f_emb = self.f_proj(f)
f_emb = f_emb + self.freq_stats_proj(dac_stats)
f_emb = self.meta_adapter_freq(f_emb)

# fusion hook immediately before cls_head
base = self.meta_adapter_fusion(self.fuse(base_in))
```

rank=0或site未启用时对应成员为`nn.Identity()`，且不产生adapter state key。

- [ ] **Step 5: 验证预算、梯度白名单和旧模型forward**

Run: `python -m pytest code/tests/test_meta_adapter_model_v1.py code/tests/test_advb02_crra_model.py -q`

Expected: tri-R4 adapter总状态约6930参数；inner可训练部分占真实ADV3B02总参数≤1%；旧模型输出shape与state key保持兼容。

- [ ] **Step 6: 更新追踪表并提交**

```text
git add code/cvsrffi/meta_adapter.py code/model.py code/model_dual_cvsincnet.py code/tests/test_meta_adapter_model_v1.py docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md
git commit -m "feat: add tri-site rank4 meta adapters"
```

---

### Task 4: Controlled Checkpoint Migration and Strict Meta Bundles

**Traceability:** `META-05`、`META-06`、`META-10`、`META-11`

**Files:**
- Create: `code/cvsrffi/meta_checkpoint.py`
- Create: `code/tests/test_meta_checkpoint_v1.py`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Consumes: legacy ADV3B02 checkpoint或meta checkpoint。
- Produces: `load_legacy_base_for_meta(model, payload) -> CheckpointLoadAudit`。
- Produces: `save_meta_bundle(path, model, config, selection) -> None`、`load_meta_bundle_strict(path, device) -> tuple[nn.Module, MetaBundleAudit]`。

- [ ] **Step 1: 写允许缺失adapter但拒绝其他差异的失败测试**

```python
def test_legacy_migration_allows_only_meta_adapter_keys():
    audit = load_legacy_base_for_meta(meta_model, legacy_payload)
    assert audit.unexpected_keys == ()
    assert audit.missing_keys
    assert all("meta_adapter" in key for key in audit.missing_keys)


def test_meta_bundle_requires_strict_state_and_frozen_head():
    save_meta_bundle(path, meta_model, config=valid_config(), selection=valid_selection())
    loaded, audit = load_meta_bundle_strict(path, device=torch.device("cpu"))
    assert audit.checkpoint_load_strict
    assert audit.trainable_fraction <= 0.01
```

- [ ] **Step 2: 运行并确认RED**

Run: `python -m pytest code/tests/test_meta_checkpoint_v1.py -q`

Expected: module不存在。

- [ ] **Step 3: 实现双路径加载规则**

meta bundle schema固定为：

```python
META_BUNDLE_SCHEMA = "cvs.meta_adapter.tri_r4.v1"
required = {
    "schema", "model_state", "model_args", "meta_adapter_config",
    "selection", "base_checkpoint", "class_mapping", "prototypes",
}
```

legacy迁移仅在Phase1初始化时允许adapter key缺失；Phase2只能使用`load_meta_bundle_strict()`，不得静默补adapter。

- [ ] **Step 4: 加入预算和禁用头检查**

加载后断言：

```python
if audit.trainable_fraction > 0.01:
    raise ValueError("Phase2 adapter trainable fraction exceeds 1%")
if any(token in name for name in trainable_names for token in ("cls_head", "classifier", "lda", "cov")):
    raise ValueError("classifier-like state is not allowed in the adapter budget")
```

- [ ] **Step 5: 运行测试并提交**

Run: `python -m pytest code/tests/test_meta_checkpoint_v1.py code/tests/test_meta_adapter_model_v1.py -q`

Expected: 全部通过。

```text
git add code/cvsrffi/meta_checkpoint.py code/tests/test_meta_checkpoint_v1.py docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md
git commit -m "feat: add strict meta adapter checkpoint bundles"
```

---

### Task 5: Support and Outer Meta Objectives

**Traceability:** `META-03`、`META-08`、`META-11`

**Files:**
- Create: `code/cvsrffi/meta_objectives.py`
- Create: `code/tests/test_meta_objectives_v1.py`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Produces: `support_objective(outputs, labels, frozen_prototypes, initial_adapter, current_adapter, config) -> LossBreakdown`。
- Produces: `outer_objective(pre_outputs, post_outputs, labels, adapt_mask, guard_mask, frozen_prototypes, config) -> LossBreakdown`。

- [ ] **Step 1: 写adapt/guard隔离和置换不变失败测试**

```python
def test_guard_rows_never_enter_adapt_ce():
    losses = outer_objective(
        pre_outputs=pre,
        post_outputs=post,
        labels=torch.tensor([0, 1, 4, 5]),
        adapt_mask=torch.tensor([True, True, False, False]),
        guard_mask=torch.tensor([False, False, True, True]),
        frozen_prototypes=prototypes,
        config=MetaObjectiveConfig(),
    )
    assert losses.adapt_count == 2
    assert losses.guard_count == 2


def test_floor_and_guard_are_class_permutation_invariant():
    original = outer_objective_from_fixture(permutation=None)
    permuted = outer_objective_from_fixture(permutation=[3, 1, 5, 0, 2, 4])
    torch.testing.assert_close(original.total, permuted.total)
```

- [ ] **Step 2: 运行并确认RED**

Run: `python -m pytest code/tests/test_meta_objectives_v1.py -q`

Expected: module不存在。

- [ ] **Step 3: 实现固定头CE、原型锚和L2-SP**

```python
@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    adapt: torch.Tensor
    guard: torch.Tensor
    floor: torch.Tensor
    topology: torch.Tensor
    zero_step: torch.Tensor
    prototype: torch.Tensor
    l2sp: torch.Tensor
    adapt_count: int
    guard_count: int
```

floor使用每类平均CE的平滑最大值：

```python
class_losses = torch.stack([row_ce[labels == cls].mean() for cls in labels.unique(sorted=True)])
loss_floor = tau * torch.logsumexp(class_losses / tau, dim=0)
```

- [ ] **Step 4: 实现topology和zero-step保留项**

topology比较pre/post类中心的pairwise cosine矩阵；zero-step使用pre-update query CE。没有guard行时`L_guard`返回同device的标量零，不得产生NaN。

- [ ] **Step 5: 运行目标和数值回归测试并提交**

Run: `python -m pytest code/tests/test_meta_objectives_v1.py -q`

Expected: adapt/guard计数正确，空mask有限，标签置换不变。

```text
git add code/cvsrffi/meta_objectives.py code/tests/test_meta_objectives_v1.py docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md
git commit -m "feat: add guarded meta adaptation objectives"
```

---

### Task 6: First-Order Adapter Inner Loop and Module-Level Meta-SGD

**Traceability:** `META-06`、`META-07`、`META-08`、`META-11`

**Files:**
- Create: `code/cvsrffi/meta_inner_loop.py`
- Create: `code/tests/test_meta_inner_loop_v1.py`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Consumes: model、support tensor、label、`support_loss_fn`和固定步数。
- Produces: `first_order_adapt(model: nn.Module, x: Tensor, y: Tensor, support_loss_fn: Callable, steps: int) -> FastAdapterState`。
- Produces: `functional_forward(model, fast_state, x, y=None) -> Mapping[str,Tensor]`。

- [ ] **Step 1: 写真实梯度、白名单和Meta-SGD失败测试**

```python
def test_inner_loop_changes_only_adapter_weights_and_gate():
    before = clone_named_state(model)
    fast = first_order_adapt(model, support_x, support_y, support_loss_fn, steps=3)
    assert fast.steps == 3
    assert set(fast.parameters) == set(iter_inner_adapter_parameter_names(model))
    assert_state_equal_outside(before, model, allowed_prefix="meta_adapter")


def test_query_gradient_reaches_initialization_and_module_step_size():
    fast = first_order_adapt(model, support_x, support_y, support_loss_fn, steps=1)
    loss = functional_forward(model, fast, query_x)["logits"].square().mean()
    loss.backward()
    assert model.meta_adapter_time.log_step_size.grad is not None
    assert model.meta_adapter_time.up.weight.grad is not None
```

- [ ] **Step 2: 运行并确认RED**

Run: `python -m pytest code/tests/test_meta_inner_loop_v1.py -q`

Expected: inner loop module不存在。

- [ ] **Step 3: 实现一阶函数式更新**

```python
@dataclass(frozen=True)
class FastAdapterState:
    parameters: OrderedDict[str, torch.Tensor]
    steps: int
    support_losses: tuple[float, ...]


def first_order_adapt(model, x, y, support_loss_fn, steps: int) -> FastAdapterState:
    if steps < 0 or steps > 10:
        raise ValueError("V1 source meta inner steps must be in [0, 10]")
    fast = OrderedDict(iter_inner_adapter_parameters(model))
    history = []
    for _ in range(steps):
        loss = support_loss_fn(functional_forward(model, fast, x, y), y, fast)
        grads = torch.autograd.grad(loss, tuple(fast.values()), create_graph=False)
        history.append(float(loss.detach()))
        fast = OrderedDict(
            (name, value - adapter_step_size(model, name) * grad.detach())
            for (name, value), grad in zip(fast.items(), grads)
        )
    return FastAdapterState(fast, steps, tuple(history))
```

`functional_forward`使用`torch.func.functional_call(model, fast.parameters, args=(x,), kwargs={"y": y, "return_aux": True}, strict=False)`；partial state只能包含已验证adapter参数名。

- [ ] **Step 4: 加入非有限梯度和步数保护**

任一loss/gradient非有限时抛出`MetaInnerLoopError`并保留原模型不变；不得静默跳过后继续生成checkpoint。Phase2封装在调用本函数前另行限制`steps<=5`，正式V1严格要求`steps==3`。

- [ ] **Step 5: 运行测试和模型邻近回归并提交**

Run: `python -m pytest code/tests/test_meta_inner_loop_v1.py code/tests/test_meta_adapter_model_v1.py -q`

Expected: 真实backward发生；主干/头/buffer未变；step size收到outer梯度。

```text
git add code/cvsrffi/meta_inner_loop.py code/tests/test_meta_inner_loop_v1.py docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md
git commit -m "feat: add first order adapter meta inner loop"
```

---

### Task 7: Phase1 Meta-Trainer, Phase1-C and Source-Only Selection

**Traceability:** `META-07`、`META-08`、`META-10`

**Files:**
- Create: `code/cvsrffi/meta_trainer.py`
- Create: `code/tests/test_meta_trainer_v1.py`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Produces: `MetaTrainerConfig`、`run_meta_train_step()`、`evaluate_adaptation_curve()`、`select_source_checkpoint()`。
- Consumes: Tasks 2、5、6的episode、objective和inner-loop API。

- [ ] **Step 1: 写优化器白名单和选择规则失败测试**

```python
def test_phase1b_optimizer_contains_only_adapter_and_log_step_sizes():
    optimizer = build_phase1b_optimizer(model, MetaTrainerConfig())
    names = optimizer_parameter_names(model, optimizer)
    assert names
    assert all("meta_adapter" in name for name in names)


def test_source_selection_rejects_zero_step_or_guard_regression():
    candidates = [
        candidate("bad_clean", clean_delta_pp=-0.6, guard_floor_delta_pp=0.0, worst_a3_delta_pp=2.0),
        candidate("bad_guard", clean_delta_pp=0.0, guard_floor_delta_pp=-0.1, worst_a3_delta_pp=3.0),
        candidate("valid", clean_delta_pp=-0.2, guard_floor_delta_pp=0.0, worst_a3_delta_pp=1.1),
    ]
    assert select_source_checkpoint(candidates).candidate_id == "valid"
```

- [ ] **Step 2: 运行并确认RED**

Run: `python -m pytest code/tests/test_meta_trainer_v1.py -q`

Expected: trainer module不存在。

- [ ] **Step 3: 实现Phase1-B训练步**

每个meta batch循环4个episode，分别计算fast state和outer loss，求平均后只对adapter初始化和`log_step_size`执行outer optimizer step。日志必须包含：

```python
{
    "episode_kind": episode.kind.value,
    "k_shot": episode.k_shot,
    "inner_steps": fast.steps,
    "loss_adapt": float(losses.adapt.detach()),
    "loss_guard": float(losses.guard.detach()),
    "loss_floor": float(losses.floor.detach()),
    "grad_cos_support_query": grad_cos,
}
```

- [ ] **Step 4: 实现Phase1-C受控optimizer**

`build_phase1c_optimizer`只包含adapter及`t_proj/f_proj/fuse`，主干三层learning rate=`0.05*adapter_outer_lr`；`cls_head`、原型和其他模块必须不在optimizer中。

- [ ] **Step 5: 实现A(0/1/3/5/10)和source-only选择**

`evaluate_adaptation_curve`只接收标记为`V_cal`或`V_select`的source episode，并在函数入口拒绝任何target receiver标识。选择严格执行clean step0、guard floor和worst-task A3三层规则。

- [ ] **Step 6: 运行trainer测试并提交**

Run: `python -m pytest code/tests/test_meta_trainer_v1.py code/tests/test_meta_inner_loop_v1.py code/tests/test_meta_objectives_v1.py -q`

Expected: optimizer白名单正确、指标有限、选择规则确定性通过。

```text
git add code/cvsrffi/meta_trainer.py code/tests/test_meta_trainer_v1.py docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md
git commit -m "feat: add source only meta adapter trainer"
```

---

### Task 8: Phase1 Entry, Frozen Config and Dry-Run Launcher

**Traceability:** `META-01`、`META-02`、`META-09`、`META-10`、`META-14`

**Files:**
- Create: `code/cvsrffi/meta_phase1_entry.py`
- Modify: `code/train.py`
- Create: `configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824.json`
- Create: `code/scripts/launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py`
- Create: `code/tests/test_meta_phase1_entry_v1.py`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Produces CLI mode`--use_cvs_meta_adapter`和`run_meta_phase1(args, ds_w)`。
- Launcher读取冻结JSON并生成不可覆盖run root。

- [ ] **Step 1: 写CLI默认值、角色比例和dry-run失败测试**

```python
def test_meta_adapter_cli_defaults_are_v1_locked():
    args = parse_args_for_test(["--use_cvs_meta_adapter"])
    assert args.meta_adapter_rank == 4
    assert args.meta_adapter_sites == "time,freq,fusion"
    assert args.meta_inner_steps == 3
    assert args.meta_inner_max_steps == 5


def test_phase1_entry_rejects_noncanonical_source_ratios():
    config = valid_config()
    config["source_roles"]["L_s"] = 0.10
    with pytest.raises(ValueError, match="0.07"):
        validate_meta_phase1_config(config)
```

- [ ] **Step 2: 运行并确认RED**

Run: `python -m pytest code/tests/test_meta_phase1_entry_v1.py -q`

Expected: CLI参数和entry module不存在。

- [ ] **Step 3: 实现薄入口和冻结配置**

JSON关键字段固定为：

```json
{
  "schema": "cvs.phase1.meta_adapter.tri_r4.v1",
  "run_id": "phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1",
  "seed": 392002,
  "base_checkpoint": "runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth",
  "wisig_pkl": "Dataset_WigSig/ManySig.pkl",
  "source_roles": {"L_s": 0.07, "U_s": 0.63, "V_cal": 0.15, "V_select": 0.15},
    "adapter": {"rank": 4, "sites": ["time", "freq", "fusion"], "inner_steps": 3, "deployment_max_steps": 5, "source_diagnostic_max_steps": 10},
  "episode_weights": {"Q_SAME_DOMAIN": 0.40, "Q_RX_HOLDOUT": 0.20, "Q_DAY_CHANNEL_HOLDOUT": 0.15, "Q_CLEAN_TO_LEO": 0.15, "Q_LEO_CROSS": 0.10},
  "k_choices": [1, 2, 5, 10],
  "meta_batch_size": 4,
  "phase1c_backbone_lr_ratio": 0.05,
  "evaluate_steps": [0, 1, 3, 5, 10]
}
```

- [ ] **Step 4: 让`train.py`只路由，不复制训练循环**

在WiSig数据加载完成后加入：

```python
if bool(getattr(args, "use_cvs_meta_adapter", False)):
    from cvsrffi.meta_phase1_entry import run_meta_phase1
    run_meta_phase1(args, ds_w)
    return
```

- [ ] **Step 5: 实现dry-run输出和不可覆盖检查**

Run: `python code/scripts/launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py --config configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824.json --dry-run`

Expected: 输出run ID、base checkpoint、数据路径、GPU、命令、output root、expected artifacts；不创建训练进程；若output root已存在则失败。

- [ ] **Step 6: 运行入口和Meta-SSL邻近回归并提交**

Run: `python -m pytest code/tests/test_meta_phase1_entry_v1.py code/tests/test_meta_ssl_cli_defaults.py code/tests/test_meta_ssl_train_loop.py -q`

Expected: 全部通过。

```text
git add code/cvsrffi/meta_phase1_entry.py code/train.py configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824.json code/scripts/launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py code/tests/test_meta_phase1_entry_v1.py docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md
git commit -m "feat: wire phase1 meta adapter training entry"
```

---

### Task 9: Phase2 Adapter-Only Adaptation Core

**Traceability:** `META-01`、`META-06`、`META-11`

**Files:**
- Create: `code/cvsrffi/stage2_meta_adapter_adaptation.py`
- Create: `tests/test_stage2_meta_adapter_adaptation.py`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Produces: `MetaAdapterPhase2Config`、`MetaAdapterAdaptAudit`。
- Produces: `adapt_meta_adapter_on_support(model, support_iq, support_labels, frozen_prototypes, class_ids, context, config) -> MetaAdapterAdaptAudit`。
- Produces: `predict_with_frozen_meta_adapter(model, query_iq, frozen_prototypes, class_ids) -> Tensor`。

- [ ] **Step 1: 写协议、参数和query冻结失败测试**

```python
def test_phase2_updates_only_adapter_and_exactly_three_steps():
    before = clone_named_state(model)
    learned_steps_before = clone_meta_step_sizes(model)
    audit = adapt_meta_adapter_on_support(
        model, support_iq, labels, prototypes, class_ids,
        context=valid_p2_context(),
        config=MetaAdapterPhase2Config(steps=3),
    )
    assert audit.backward_count == 3
    assert audit.trainable_fraction <= 0.01
    assert all("meta_adapter" in name for name in audit.updated_parameter_names)
    assert_meta_step_sizes_equal(learned_steps_before, model)
    assert_state_equal_outside(before, model, allowed_prefix="meta_adapter")


def test_phase2_rejects_source_or_query_keys():
    bad = {**valid_p2_context(), "query_role": "old_query"}
    with pytest.raises(ValueError, match="context allowlist"):
        adapt_meta_adapter_on_support(
            model,
            support_iq,
            labels,
            prototypes,
            class_ids,
            context=bad,
            config=MetaAdapterPhase2Config(steps=3),
        )
```

- [ ] **Step 2: 运行并确认RED**

Run: `python -m pytest tests/test_stage2_meta_adapter_adaptation.py -q`

Expected: module不存在。

- [ ] **Step 3: 实现严格context和固定步数**

```python
_PHASE2_CONTEXT_ALLOWLIST = frozenset({
    "protocol_schema", "phase2_data_status", "capsule_id", "split_id"
})

@dataclass(frozen=True)
class MetaAdapterPhase2Config:
    steps: int = 3
    max_steps: int = 5
    hard_step_limit: int = 40
```

入口要求`steps==3`用于正式V1；测试可使用0～5。任何大于5的V1配置直接失败，40只作为上层硬保护。

- [ ] **Step 4: 实现support-only更新和query预测**

support loss使用冻结原型余弦logits和固定class mapping；调用`first_order_adapt()`读取meta bundle中冻结的模块级步长，将返回的fast adapter权重写回模型。Phase2不得创建optimizer或更新`log_step_size`。适配结束执行：

```python
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
```

query预测函数加`@torch.no_grad()`，调用前后比较完整state dict。

- [ ] **Step 5: 运行适配测试和APSTA邻近回归并提交**

Run: `python -m pytest tests/test_stage2_meta_adapter_adaptation.py tests/test_stage2_structured_late_block_adaptation.py -q`

Expected: 全部通过；新方法不改变现有late-block行为。

```text
git add code/cvsrffi/stage2_meta_adapter_adaptation.py tests/test_stage2_meta_adapter_adaptation.py docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md
git commit -m "feat: add stage2 meta adapter support updates"
```

---

### Task 10: Same-Row DA0/DA1 Runner and Real No-Query Smoke

**Traceability:** `META-01`、`META-06`、`META-10`、`META-12`、`META-14`

**Files:**
- Create: `code/cvsrffi/stage2_meta_adapter_runner.py`
- Create: `code/scripts/run_stage2_meta_adapter.py`
- Create: `code/scripts/smoke_stage2_meta_adapter_no_query.py`
- Create: `tests/test_stage2_meta_adapter_runner.py`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Produces: `run_meta_adapter_stage2_row(config, output_dir, device) -> Mapping[str,Any]`。
- 输出一个row内的`predictions_DA0_REG0.npz`、`predictions_DA1_REG0.npz`和`receipt.json`。

- [ ] **Step 1: 写query打开顺序、状态和不可覆盖失败测试**

```python
def test_runner_adapts_before_query_is_opened_and_emits_two_states(tmp_path):
    receipt = run_meta_adapter_stage2_row(valid_config(tmp_path), tmp_path / "out", "cpu")
    assert receipt["query_opened_before_adaptation"] is False
    assert receipt["states"] == ["DA0_REG0", "DA1_REG0"]
    assert receipt["source_opened"] is False
    assert receipt["query_state_update_count"] == 0


def test_runner_refuses_existing_output_root(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        run_meta_adapter_stage2_row(valid_config(tmp_path), out, "cpu")
```

- [ ] **Step 2: 运行并确认RED**

Run: `python -m pytest tests/test_stage2_meta_adapter_runner.py -q`

Expected: runner module不存在。

- [ ] **Step 3: 实现最小payload allowlist和执行顺序**

配置只允许：

```python
_CONFIG_ALLOWLIST = frozenset({
    "protocol_schema", "phase2_data_status", "capsule_id", "split_id",
    "checkpoint_path", "support_path", "query_path", "prototype_path",
    "receiver", "scenario", "operating_point", "seed", "k_shot", "steps",
})
_SUPPORT_KEYS = frozenset({"received_iq", "support_labels"})
_QUERY_KEYS = frozenset({"received_iq", "query_ids"})
_PROTOTYPE_KEYS = frozenset({"prototypes", "class_ids"})
```

顺序必须是：严格加载bundle→生成DA0状态→打开support→执行3步→冻结DA1状态→打开query→分别预测→状态回读。

- [ ] **Step 4: 实现无query smoke**

smoke配置不得包含`query_path`；输出状态固定为`REAL_META_CHECKPOINT_NO_QUERY_SMOKE_PASS`，并记录：

```json
{
  "query_opened": false,
  "source_opened": false,
  "backward_count": 3,
  "checkpoint_load_strict": true,
  "query_state_update_count": 0
}
```

- [ ] **Step 5: 运行runner、smoke和late-block邻近回归并提交**

Run: `python -m pytest tests/test_stage2_meta_adapter_runner.py tests/test_stage2_structured_late_block_runner.py -q`

Expected: 全部通过。

```text
git add code/cvsrffi/stage2_meta_adapter_runner.py code/scripts/run_stage2_meta_adapter.py code/scripts/smoke_stage2_meta_adapter_no_query.py tests/test_stage2_meta_adapter_runner.py docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md
git commit -m "feat: add paired meta adapter stage2 runner"
```

---

### Task 11: Truth-Last Scoring, Promotion Logic and Stage2-C Handoff

**Traceability:** `META-12`、`META-13`、`META-15`

**Files:**
- Create: `code/cvsrffi/stage2_meta_adapter_scorer.py`
- Create: `code/cvsrffi/stage2_meta_adapter_handoff.py`
- Create: `code/scripts/score_stage2_meta_adapter.py`
- Create: `code/scripts/summarize_stage2_meta_adapter_matrix.py`
- Create: `tests/test_stage2_meta_adapter_scorer.py`
- Create: `tests/test_stage2_meta_adapter_handoff.py`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Produces: `score_meta_adapter_pair(da0_path, da1_path, truth_path) -> PairedStage2BScore`。
- Produces: `summarize_meta_adapter_matrix(scores) -> MatrixDecision`。
- Produces: `freeze_da1_reg0_handoff(model, binding) -> FrozenMetaAdapterHandoff`。

- [ ] **Step 1: 写同row配对、REG0 N/A和晋级失败测试**

```python
def test_scorer_requires_identical_query_ids_and_reports_reg0_na():
    score = score_meta_adapter_pair(da0_path, da1_path, truth_path)
    assert score.da0.state == "DA0_REG0"
    assert score.da1.state == "DA1_REG0"
    assert score.da0.seen_new_acc is None
    assert score.da1.h_old_new is None


def test_promotion_requires_both_mean_and_floor_thresholds():
    assert summarize_rows(mean_delta_pp=1.1, floor_delta_pp=0.4).promote is False
    assert summarize_rows(mean_delta_pp=0.9, floor_delta_pp=1.0).promote is False
    assert summarize_rows(mean_delta_pp=1.0, floor_delta_pp=0.5).promote is True
```

- [ ] **Step 2: 写Stage2-C冻结交接失败测试**

```python
def test_handoff_contains_no_optimizer_head_or_new_class_update():
    handoff = freeze_da1_reg0_handoff(model, valid_binding())
    assert handoff.state == "DA1_REG0"
    assert handoff.optimizer_state is None
    assert handoff.new_class_support_consumed is False
    assert all("cls_head" not in name for name in handoff.adapted_state)
```

- [ ] **Step 3: 运行并确认RED**

Run: `python -m pytest tests/test_stage2_meta_adapter_scorer.py tests/test_stage2_meta_adapter_handoff.py -q`

Expected: scorer和handoff module不存在。

- [ ] **Step 4: 实现truth-last评分和矩阵决策**

scorer先验证两份prediction完整、query ID完全一致、state合法，再打开truth。矩阵汇总按旧类等权mean和全row最小class floor计算：

```python
promote = mean_delta_pp >= 1.0 and floor_delta_pp >= 0.5
verdict = "PROMOTE_TO_TARGET25" if promote else "SCIENTIFIC_FAILURE_NO_PROMOTION"
```

- [ ] **Step 5: 实现Stage2-C只读交接对象**

handoff只序列化adapter参数、checkpoint/bundle ID、capsule/split和`DA1_REG0`状态，不含optimizer、梯度、分类头副本或new-class状态。

- [ ] **Step 6: 运行评分、handoff和既有scorer回归并提交**

Run: `python -m pytest tests/test_stage2_meta_adapter_scorer.py tests/test_stage2_meta_adapter_handoff.py tests/test_stage2_structured_late_block_scorer.py -q`

Expected: 全部通过。

```text
git add code/cvsrffi/stage2_meta_adapter_scorer.py code/cvsrffi/stage2_meta_adapter_handoff.py code/scripts/score_stage2_meta_adapter.py code/scripts/summarize_stage2_meta_adapter_matrix.py tests/test_stage2_meta_adapter_scorer.py tests/test_stage2_meta_adapter_handoff.py docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md
git commit -m "feat: score and hand off meta adapter states"
```

---

### Task 12: Local Integration, Real Base-Checkpoint Smoke and Single P0/P1 Review

**Traceability:** `META-01`～`META-15`

**Files:**
- Modify only if review finds direct P0/P1: files introduced in Tasks 1～11
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`
- Create: `automation_reports/CV-SincNet/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1/report.md`

**Interfaces:**
- Produces local state`LOCAL_VERIFIED`和一次审查结论。

- [ ] **Step 1: 运行聚焦RED→GREEN测试集合**

Run:

```text
python -m pytest code/tests/test_meta_episode_sampler_v1.py code/tests/test_meta_adapter_model_v1.py code/tests/test_meta_checkpoint_v1.py code/tests/test_meta_objectives_v1.py code/tests/test_meta_inner_loop_v1.py code/tests/test_meta_trainer_v1.py code/tests/test_meta_phase1_entry_v1.py tests/test_stage2_meta_adapter_adaptation.py tests/test_stage2_meta_adapter_runner.py tests/test_stage2_meta_adapter_scorer.py tests/test_stage2_meta_adapter_handoff.py -q
```

Expected: 全部通过，0 skipped-by-error。

- [ ] **Step 2: 运行邻近回归和静态检查**

Run:

```text
python -m pytest code/tests/test_meta_ssl_split.py code/tests/test_meta_ssl_train_loop.py code/tests/test_advb02_crra_model.py tests/test_stage2_structured_late_block_adaptation.py tests/test_stage2_structured_late_block_runner.py tests/test_stage2_structured_late_block_scorer.py -q
python -m py_compile code/cvsrffi/meta_episodes.py code/cvsrffi/meta_adapter.py code/cvsrffi/meta_checkpoint.py code/cvsrffi/meta_objectives.py code/cvsrffi/meta_inner_loop.py code/cvsrffi/meta_trainer.py code/cvsrffi/meta_phase1_entry.py code/cvsrffi/stage2_meta_adapter_adaptation.py code/cvsrffi/stage2_meta_adapter_runner.py code/cvsrffi/stage2_meta_adapter_scorer.py code/cvsrffi/stage2_meta_adapter_handoff.py
git diff --check
```

Expected: 全部退出0。

- [ ] **Step 3: 用真实ADV3B02 checkpoint执行受控迁移smoke**

使用`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`构建未训练adapter bundle，并执行无query smoke。该smoke只证明checkpoint、真实backward、预算和隔离链，不产生性能结论。

Expected receipt状态：`REAL_BASE_CHECKPOINT_ADAPTER_INIT_NO_QUERY_SMOKE_PASS`；`query_opened=false`、`source_opened=false`、`backward_count=3`、`trainable_fraction<=0.01`。

- [ ] **Step 4: 进行唯一一次独立P0/P1审查**

审查范围仅限会导致真实实验跑错、越权、覆盖输出、误杀进程、无法启动或无法产生合法prediction的问题。P2记录为非阻断；额外gate记录为`REJECTED_EXTRA_GATE`。

- [ ] **Step 5: 若有P0/P1，完成一次定点修复和定点复审**

只重跑直接覆盖该问题的测试和受影响邻近测试，不做第二次全量审查。

- [ ] **Step 6: 反向审计追踪表并提交**

所有已实现软件条目更新为`verified`；实验条目保持`pending`。报告记录变更文件、验证命令、真实smoke和审查结果。

```text
git add docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md automation_reports/CV-SincNet/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1/report.md
git commit -m "test: verify meta adapter local release"
```

独立执行：`git rev-parse HEAD`和`git ls-remote origin refs/heads/work/cvs-active`，两者必须一致。

---

### Task 13: N607 Release and Phase1 Minimal Screen

**Traceability:** `META-07`、`META-09`、`META-10`、`META-14`

**Files:**
- Modify: `automation_reports/CV-SincNet/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1/report.md`
- Create mirror: `docs/experiments/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1_report.md`

**Interfaces:**
- Produces N607 run`phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1`。
- Expected artifacts: P0～P4 source-only metrics、A(0/1/3/5/10)、clean和三个LEO_WEAK结果、候选checkpoint和训练日志。

- [ ] **Step 1: 写最小预登记报告**

报告只记录：候选P0～P4、commit、命令、N607 CWD、输入输出路径、GPU、停止规则和expected artifacts。P5 Phase1-C只在P4 source晋级后单独追加，不作为首轮gate。

- [ ] **Step 2: 执行N607只读preflight**

Run: `powershell.exe -ExecutionPolicy Bypass -File E:\type10-7\tools\n607_ssh_preflight.ps1`

Expected: direct`N607`目标、身份、服务器时间、项目根和GPU可见。若直接路径失败且身份无歧义，按AGENTS规定使用lab bridge；不得尝试交互密码或临时relay。

- [ ] **Step 3: 生成release归档并只校验一次SHA**

归档只包含本次commit所需代码、配置和launcher。记录本地归档路径、远端路径和单一archive SHA；不得计算成员SHA或额外seal。

- [ ] **Step 4: SCP落地并远端编译**

使用普通`N607`账户将release归档复制到：

`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1/`

远端解压到不可覆盖checkout，使用`$HOME/.conda/envs/ssr-gpu/bin/python -m py_compile`编译Task 12列出的模块。

- [ ] **Step 5: 记录GPU占用并启动首轮screen**

首轮只启动P1随机adapter、P2普通监督adapter、P3 FOMAML固定LR、P4 FOMAML＋Meta-SGD；P0直接由step0评估产生。每GPU最多两个训练进程，且不干预已有N607任务。

技术停止仅限协议越权、错误checkout/output root、无prediction闭合、launcher-wide故障或至少两个row相同的确定性pre-prediction异常；不得因低性能停止。

- [ ] **Step 6: 启动后做一次归属和日志增长检查**

使用短连接核对主PID、CWD、完整cmdline、run root、GPU映射和日志增长；完成后立即断开SSH。

- [ ] **Step 7: 更新报告、提交并回读远端OID**

状态写为`RUNNING`，不能写成训练完成或性能结果。

```text
git add docs/experiments/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1_report.md
git commit -m "exp: launch phase1 meta adapter screen"
```

---

### Task 14: Phase1 Completion, Source Selection and Strict Meta Bundle

**Traceability:** `META-08`、`META-09`、`META-10`、`META-14`

**Files:**
- Modify: Phase1 report
- Modify: `docs/experiments/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1_report.md`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Consumes completedP0～P4 artifacts。
- Produces selectedmeta bundle或`SCIENTIFIC_FAILURE_NO_PROMOTION`。

- [ ] **Step 1: 等待训练和四场景评估全部完成**

完成标准包括checkpoint identity、clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`及对应日志。训练结束但缺场景评估时状态仍不是`ARTIFACTS_COMPLETE`。

- [ ] **Step 2: 汇总source adaptation曲线**

每个候选按任务类型和类别模式输出A(0/1/3/5/10)、mean、floor、per-class、`Y_adapt/Y_guard`、梯度余弦、adapter范数、步长、显存和延迟。

- [ ] **Step 3: 执行冻结选择规则**

依次检查：clean step0下降≤0.5pp、guard floor不下降、最大worst-task A3增益。同分时选择参数和延迟更低者。

- [ ] **Step 4: 仅在P4 source晋级后运行P5 Phase1-C**

P5仍使用相同source角色和episode，不接触target数据。若P5未超过P4或违反zero-step/guard约束，最终bundle选择P4。

- [ ] **Step 5: 导出并严格回读meta bundle**

使用`load_meta_bundle_strict()`回读，随后用真实meta checkpoint再次执行无query smoke。Expected：`REAL_META_CHECKPOINT_NO_QUERY_SMOKE_PASS`。

- [ ] **Step 6: 更新报告和追踪状态并提交**

若无候选通过source选择，记录科学失败并进入下一个结构候选，不启动Target5。若通过，状态写`ARTIFACTS_COMPLETE/ANALYZED`并记录选中bundle路径。

```text
git add docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md docs/experiments/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1_report.md
git commit -m "report: select phase1 meta adapter bundle"
```

---

### Task 15: Phase2 Target5, Conditional Target25 and Final Decision

**Traceability:** `META-01`、`META-06`、`META-12`、`META-13`、`META-14`、`META-15`

**Files:**
- Create: `configs/stage2b_meta_adapter_tri_r4_target5_s713101_20260824.json`
- Create only after Target5 promotion: `configs/stage2b_meta_adapter_tri_r4_target25_s713101_20260824.json`
- Create: `automation_reports/CV-SincNet/adv3b02_stage2b_meta_adapter_tri_r4_t5t25_s713101_20260824_r1/report.md`
- Create mirror: `docs/experiments/adv3b02_stage2b_meta_adapter_tri_r4_t5t25_s713101_20260824_r1_report.md`
- Modify: `docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md`

**Interfaces:**
- Target5：receiver=`20-1`×operating point=`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`，seed=`713101`，三个LEO_WEAK场景。
- Target25：receiver=`20-1,3-19,7-14,7-7,8-8`×相同5个operating point，seed=`713101`，三个LEO_WEAK场景。
- 每个row输出配对`DA0_REG0/DA1_REG0`prediction和truth-last score。

- [ ] **Step 1: 从同一VALIDATED_ONCE manifest生成Target5配置**

只核对`protocol_schema/capsule_id/split_id/phase2_data_status`；adapter、checkpoint和预算变化不得触发数据重验。配置固定`steps=3`，query payload只允许`received_iq/query_ids`。

- [ ] **Step 2: 完成真实meta checkpoint无query smoke**

smoke PASS后立即继续Target5，不创建smoke授权token或额外receipt链。

- [ ] **Step 3: 发布并启动Target5**

按Task 13的direct N607 preflight、单archive SHA、远端编译和一次启动检查执行。run root固定为：

`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_stage2b_meta_adapter_tri_r4_t5t25_s713101_20260824_r1/target5`

- [ ] **Step 4: prediction完整后独立连接truth**

先验证每个row的DA0/DA1 query ID完全一致、prediction完整和query状态未更新，再由独立scorer读取truth。REG0的new accuracy和`H_old_new`写`N/A`。

- [ ] **Step 5: 执行Target5晋级判断**

按所有Target5同row旧类等权均值和全矩阵class floor计算：

```text
mean_delta_pp = DA1_REG0.mean_old_acc - DA0_REG0.mean_old_acc
floor_delta_pp = DA1_REG0.old_class_floor - DA0_REG0.old_class_floor
PROMOTE iff mean_delta_pp >= 1.0 and floor_delta_pp >= 0.5
```

未晋级时记录`SCIENTIFIC_FAILURE_NO_PROMOTION`，不创建Target25配置；下一候选顺序为`FUSION_R8→TIME_FUSION_R4→SCALE_SHIFT`。

- [ ] **Step 6: 仅在晋级后生成并运行Target25**

Target25沿用同一bundle、seed、协议句柄、更新步数和scorer，不调参、不使用Target5 query结果改变方法。

- [ ] **Step 7: 完成最终报告和Stage2-C交接artifact**

报告逐row保留receiver、scenario、operating point、old mean、old floor、更新延迟、状态大小和verdict。若Target25仍达标，生成`FrozenMetaAdapterHandoff`供后续Stage2-C注册实验使用；该artifact不含optimizer、头或新类更新。

- [ ] **Step 8: 反向审计、提交、push和OID回读**

追踪表中每项必须为`verified/deferred/rejected/blocked`之一；不得留下`implemented`而无验证。最终报告说明V1是严格设计实现，唯一有意调整是三个rank-8收缩为三个rank-4以满足≤1%预算。

```text
git add configs/stage2b_meta_adapter_tri_r4_target5_s713101_20260824.json docs/CVS_META_ADAPTER_TRI_R4_V1_TRACE_20260824.md docs/experiments/adv3b02_stage2b_meta_adapter_tri_r4_t5t25_s713101_20260824_r1_report.md
git commit -m "report: record meta adapter target decision"
git rev-parse HEAD
git ls-remote origin refs/heads/work/cvs-active
```

Expected: 本地HEAD与远端branch OID完全一致；报告状态与真实artifact一致。

---

## Plan Completion Criteria

计划执行完成必须同时满足：

1.Tasks 1～11的聚焦测试和Task 12邻近回归全部通过；
2.真实ADV3B02迁移smoke和真实meta checkpoint无query smoke均有明确receipt；
3.一次P0/P1审查闭合，未增加白名单外gate；
4.Phase1至少形成P0～P4完整source-only结果及四场景评估；
5.只有source选择通过时才进入Phase2 Target5；
6.Target5只有同时达到mean`+1.0pp`和floor`+0.5pp`才进入Target25；
7.全部正式交付物提交、push并远端OID回读一致；
8.任何未完成性能阶段保持`UNKNOWN`或明确科学失败，不把smoke、RUNNING或部分prediction描述为正收益。
