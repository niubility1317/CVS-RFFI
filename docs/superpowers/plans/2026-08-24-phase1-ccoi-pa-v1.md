# Phase1 CCOI-PA-V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结`ADV3B02_CORE90_SOFT_E200`真实checkpoint的前提下，实现挑战条件化PA算子sidecar，完成C0–C4单seed最小矩阵、clean与三个LEO_WEAK场景prediction及同row科学诊断。

**Architecture:** 现有CV-SincNet只新增`pa_token_map`辅助输出，不新增checkpoint参数。`code/cvsrffi/ccoi_pa.py`实现双视图、固定内容统计、挑战编码、FiLM条件响应、集合池化和非循环holdout预测；`ccoi_losses.py`实现普通/挑战匹配SupCon、DiD和三距离诊断；独立runner严格加载并冻结Core90，训练sidecar后输出不可覆盖artifact。

**Tech Stack:** Python、PyTorch、pytest、现有WiSig/SSDG数据划分、现有LEO_WEAK信道评估、Git、N607 CUDA环境。

**Spec:** `docs/CVS_PHASE1_CCOI_PA_V1_DESIGN_20260824.md`

## Global Constraints

- Phase1只使用source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，四者物理ID不交，`rho_label≤0.1`。
- target、query、query truth、query role和跨query全局状态均不得进入训练、校准或选择。
- 冻结Core90主干；C0–C4使用同一checkpoint、split、seed、预算和评估器。
- 最终评估必须分别包含clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。
- 同一物理样本的多个视图不增加K；CCOI bundle不得保存source样本级embedding或token缓存。
- 默认关闭CCOI时，旧checkpoint结构和原始logits保持不变。
- V1不实现Soft-DTW、partial OT、生成器、多机制融合或状态子空间。
- 所有本地测试串行使用`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`。
- 不改`train_ssdg.py`默认Core90训练语义；CCOI使用独立runner。

---

### Task 1: 暴露冻结PA时序图且保持旧模型兼容

**Files:**
- Modify: `code/model.py`
- Create: `code/tests/test_ccoi_pa_backbone_interface.py`

**Interfaces:**
- Consumes: `CVSincNet.forward(...,return_aux=True)`。
- Produces: `aux["pa_token_map"]: Tensor[B,C,L]`；禁用PA分支时为`Tensor[B,0,0]`或明确零张量；state_dict键不变。

- [ ] **Step 1: Write the failing test**

```python
def test_cvsincnet_exposes_prepool_pa_map_without_new_state_keys():
    model = build_model(num_classes=6, dataset="wisig", input_len=256)
    keys_before = tuple(model.state_dict())
    out = model(torch.randn(2, 2, 256), return_aux=True)
    assert out["pa_token_map"].ndim == 3
    assert out["pa_token_map"].shape[0] == 2
    assert tuple(model.state_dict()) == keys_before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_ccoi_pa_backbone_interface.py -q`
Expected: FAIL with missing `pa_token_map`.

- [ ] **Step 3: Write minimal implementation**

Store `p` immediately after`pa_b3`; return it only in the existingaux dictionary. Do not addmodules, parameters orconstructor flags.

- [ ] **Step 4: Run test to verify it passes**

Run the Task 1 test plus`test_exact_ssdg_checkpoint_loading.py`and`test_advb02_crra_model.py`.

---

### Task 2: 双视图、固定内容统计与挑战编码器

**Files:**
- Create: `code/cvsrffi/ccoi_pa.py`
- Create: `code/tests/test_ccoi_pa_views_and_challenge.py`

**Interfaces:**
- Produces: `make_dual_iq_views(x)->(content,fingerprint)`。
- Produces: `tokenize_iq(x,64,16)->Tensor[B,T,2,64]`，256点输入`T=13`。
- Produces: `fixed_content_statistics(tokens)->Tensor[B,T,S]`，目标无梯度。
- Produces: `PAChallengeEncoder.forward(content)->ChallengeOutput(q,code_prob,content_stats)`。
- Produces: `challenge_pretrain_losses(clean,satellite,mask)->dict[str,Tensor]`。

- [ ] **Step 1: Write failing tests**

```python
def test_tokenization_has_thirteen_tokens_and_fingerprint_is_unchanged():
    x = torch.randn(3, 2, 256)
    content, fingerprint = make_dual_iq_views(x)
    assert torch.equal(fingerprint, x)
    assert tokenize_iq(content, 64, 16).shape == (3, 13, 2, 64)

def test_fixed_content_targets_are_detached_and_finite():
    x = torch.randn(2, 2, 256, requires_grad=True)
    stats = fixed_content_statistics(tokenize_iq(x, 64, 16))
    assert not stats.requires_grad
    assert torch.isfinite(stats).all()
```

- [ ] **Step 2: Verify RED**

Run the Task 2 test file; expected import failure because`ccoi_pa.py`does not exist.

- [ ] **Step 3: Implement minimal components**

Use package-level mean removal andRMS normalization for content view; never normalize eachtoken independently. Use a small sharedConv1d encoder,32-dimensionalq and48-way soft codebook. Masked reconstruction predicts fixed statistics from neighboringq; temporal prediction targets the next non-overlap anchor statistic.

- [ ] **Step 4: Verify GREEN**

Run Task 2 tests and assert non-zero gradients reach challenge encoder while fixed targets remain detached.

---

### Task 3: 条件响应、集合算子与非循环holdout

**Files:**
- Modify: `code/cvsrffi/ccoi_pa.py`
- Create: `code/tests/test_ccoi_pa_operator.py`

**Interfaces:**
- Produces: `PAConditionalResponseHead(pa_map,q,conditioned=True)->Tensor[B,T,R]`。
- Produces: `OperatorPool(response,q,valid_mask)->OperatorOutput(theta,attention,coverage,entropy)`。
- Produces: `nonoverlap_anchor_indices(T,token_length,stride)`和确定性support/holdout mask。
- Produces: `HeldoutChallengePredictor(theta,q_holdout)->frozen PA target shape`。

- [ ] **Step 1: Write failing tests**

```python
def test_operator_pool_is_permutation_invariant():
    out1 = pool(r, q, mask).theta
    out2 = pool(r[:, perm], q[:, perm], mask[:, perm]).theta
    torch.testing.assert_close(out1, out2)

def test_holdout_anchors_do_not_overlap_in_raw_sample_ranges():
    support, holdout = nonoverlap_holdout_masks(13, token_length=64, stride=16, fold=0)
    assert raw_intersection_count(support, holdout, 64, 16) == 0
```

- [ ] **Step 2: Verify RED**

Expected missing class/function failures.

- [ ] **Step 3: Implement minimal operator path**

Adaptively pool the frozenPA map to13 challenge tokens. C1 uses a learned constant condition through the sameFiLM network; C2–C4 use frozenq. Holdout target is the adaptive-pooled frozenPA map with`detach()` and uses only stride64 anchors.

- [ ] **Step 4: Verify GREEN**

Run Task 3 tests, including empty mask, all-invalid mask, permutation and target-gradient isolation.

---

### Task 4: 条件pair、DiD和三距离否证

**Files:**
- Create: `code/cvsrffi/ccoi_losses.py`
- Create: `code/tests/test_ccoi_losses.py`

**Interfaces:**
- Produces: `challenge_pair_masks(q_summary,y_tx,domain,min_cosine)`。
- Produces: `ccoi_supcon_loss(theta,masks,temperature)`。
- Reuses: `tx_rx_rectangle_identity_loss`from`tx_rx_geometry.py`。
- Produces: `conditional_distance_diagnostics(response,q,y_tx,domain)->dict`。

- [ ] **Step 1: Write failing literal-geometry tests**

```python
def test_challenge_masks_select_cross_domain_positive_and_same_domain_negative():
    y = torch.tensor([0, 0, 1, 1])
    d = torch.tensor([0, 1, 0, 1])
    q = torch.tensor([[1.,0.],[1.,0.],[1.,0.],[0.,1.]])
    masks = challenge_pair_masks(q, y, d, min_cosine=0.99)
    assert masks.positive[0,1]
    assert masks.negative[0,2]
    assert not masks.negative[0,3]
```

- [ ] **Step 2: Verify RED**

Expected missing module failure.

- [ ] **Step 3: Implement minimal losses**

Use the existing normalizedSupCon implementation where possible. Empty valid pairs return differentiable zero and explicit counts. Three-distance diagnostics must report counts andNaN when a relation is unobservable; never fabricate zero.

- [ ] **Step 4: Verify GREEN**

Run Task 4 tests plus`test_tx_rx_geometry.py`and`test_balanced_tx_rx_sampler.py`.

---

### Task 5: 冻结checkpoint的C0–C4 runner与协议负测

**Files:**
- Create: `code/train_phase1_ccoi_pa.py`
- Create: `code/tests/test_phase1_ccoi_pa_runner.py`
- Create: `code/tests/test_ccoi_protocol.py`

**Interfaces:**
- Consumes: exactCore90 checkpoint and`_build_ssdg_wisig_data`source roles。
- Produces: per-row`prediction.jsonl`、`metrics.json`、`sidecar.pth`、challenge audit and final matrix summary。
- Produces: wrapper with base-compatible`tx_logits/dom_logits/z_id/z_dom`and additionalCCOI telemetry。

- [ ] **Step 1: Write failing protocol and dry-run tests**

```python
def test_runner_requires_current_source_role_ratios():
    with pytest.raises(ValueError, match="0.07/0.63/0.15/0.15"):
        validate_source_roles(bad_args, split_info)

def test_c0_has_no_trainable_sidecar_and_c1_to_c4_share_capacity():
    specs = build_matrix_specs()
    assert specs["C0"].train_sidecar is False
    assert len({spec.parameter_profile for name, spec in specs.items() if name != "C0"}) == 1
```

- [ ] **Step 2: Verify RED**

Expected missing runner APIs.

- [ ] **Step 3: Implement runner**

Strictly rebuild checkpoint through`build_exact_ssdg_model_from_checkpoint`; freeze every base parameter. Pretrainq once from`L_s+U_s`withoutU labels, then freeze and reuse it forC2–C4. C1–C4 start sidecar weights from the same seeded initialization. `V_cal`only calibratesfixedfusion alpha;`V_select`selects fixed epoch output without updating state. Evaluation writesprediction before metrics aggregation.

- [ ] **Step 4: Add four-scenario evaluation**

Reuse`evaluate_sat_scenarios`with exactly`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`; clean remains separate. Add challenge distribution slices and`d1/d2/d3`diagnostics.

- [ ] **Step 5: Verify GREEN**

Run runner/protocol tests, a CPU`--dry_run`, and one synthetic one-batch smoke.

---

### Task 6: Launcher、配置和最小预登记报告

**Files:**
- Create: `code/scripts/launch_phase1_ccoi_pa_v1_20260824.sh`
- Create: `code/tests/test_phase1_ccoi_pa_launcher.py`
- Create: `docs/experiments/PHASE1_CCOI_PA_V1_CONFIG_20260824.json`
- Create outside Git root then mirror after launch: `automation_reports/CV-SincNet/<run-id>/report.md`

**Interfaces:**
- Launcher first runs real checkpoint no-query smoke; PASS后立即进入C0–C4。
- Immutable runID/output root; no overwrite。

- [ ] **Step 1: Write failing launcher test**

Assert exact source ratios,Core90 checkpoint, threeLEO场景, one seed, unique row roots, smoke-first and no target/query arguments.

- [ ] **Step 2: Verify RED**

Expected launcher missing.

- [ ] **Step 3: Implement launcher and config**

UseN607 paths from the active report. Defaulthead epochs andq pretrain epochs remain explicit experiment values, not globalPhase1 defaults.

- [ ] **Step 4: Verify GREEN**

Runpytest launcher test and verifiedGit Bash`bash -n`.

---

### Task 7: 本地验证、P0/P1审查、Git与N607发布

**Files:**
- Update: `docs/CVS_PHASE1_CCOI_PA_V1_TRACE_20260824.md`
- Update: `docs/CVS_PHASE1_CCOI_PA_V1_DESIGN_20260824.md`
- Update after evidence: active experiment`report.md`and itsGit mirror。

- [ ] **Step 1: Run focused full verification**

Run all new tests plus baseline checkpoint,PA/CRRA,balanced sampler,TX/RX geometry,Phase1 protocol andsatellite evaluation tests. Run`py_compile`for all changedPython files.

- [ ] **Step 2: Run real checkpoint no-query smoke**

Strict load must report0missing/0unexpected; oneC2 batch must produce finite losses andprediction without target/query access.

- [ ] **Step 3: Perform one independent P0/P1 read-only review**

Review only direct run-break, protocol-leak, overwrite, process-safety, launch andprediction-closure defects. Fix at most once and perform one scoped re-review if needed.

- [ ] **Step 4: Commit and push exact files**

Stage only this plan's files and report mirror. Verify remote branchOID equalslocalHEAD.

- [ ] **Step 5: N607 minimal release**

Run local direct preflight, record occupancy, create one release archive, compare local/remoteSHA once, compile remotely once, then launch immutable run.

- [ ] **Step 6: Post-launch binding**

VerifyPID、CWD、cmdline、GPU and log growth once. Status remains`RUNNING`until all predictions complete.

- [ ] **Step 7: Independent scoring and final report**

After predictions close, scorer connects truth and writes same-row clean/threeLEO metrics,operator diagnostics,coverage and verdict. Commit/push the completed report and verify remoteOID.

## Self-review

- Spec coverage: T08–T22 andT34–T39 each map toTasks1–7; T33 remains blocked by absent verified semantic-content metadata.
- Deferred scope: Soft-DTW、partial OT、多机制、生成器和状态子空间 do not enter any code task.
- Type consistency: challenge output、operator output、matrix spec and result artifact names are defined once and consumed unchanged.
- Placeholder scan: no`TBD`or`TODO`; all commands, files, interfaces and expected failures are explicit.
