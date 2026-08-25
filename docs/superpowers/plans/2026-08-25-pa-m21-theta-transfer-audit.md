# PA-M2.1 Theta Transfer Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现权重独立的C1′/C4′多fold theta迁移审计，并仅在阶段A通过后运行truth-blind安全gate。

**Architecture:** 新增独立`ccoi_pa_m21`纯函数模块承载V3 sidecar契约、block split、relation bank、F矩阵、M0/LOTO和gate；新增单一runner复用现有Core90、WiSig和C4训练路径，但不修改历史A/B产物。runner先生成阶段A全部artifact和verdict，只有`A_PASS`才冻结gate配置并执行阶段B。

**Tech Stack:** Python、PyTorch、NumPy、scikit-learn、pytest、现有SSDG/WiSig与CCOI-PA代码、N607 CUDA。

**Spec:** `docs/superpowers/specs/2026-08-25-pa-m21-theta-transfer-audit-design.md`

## Global Constraints

- 保持`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`和Phase1 source-only边界。
- `V_audit_retro`只称权重独立，不称研究历史完全未见。
- 旧A/B、08d3提交、旧C4和旧artifact只读。
- 主F关系无fallback，support只来自`V_select_fit_support_bank`，候选映射不读取q。
- 只在`A_PASS`后执行阶段B；低性能不触发技术停止。
- 不实现Soft-DTW、OT、强制码本均衡、多机制融合或Core90解冻。
- 所有项目测试使用`ssr-gpu`环境串行执行。

---

### Task 1: Sidecar V3配置与严格迁移

**Files:**
- Create: `code/cvsrffi/ccoi_pa_m21.py`
- Create: `code/tests/test_ccoi_pa_m21.py`

**Interfaces:**
- Produces: `SidecarArchitectureConfig`、`build_sidecar_v3_payload(...)`、`load_sidecar_v3(...)`、`migrate_v2_challenge_encoder(...)`。
- Consumes: `CCOIPASidecar`、`PAChallengeEncoder`和旧V2 payload。

- [ ] **Step 1: 写V3 round-trip和无参数语义错误的失败测试**

```python
def test_v3_sidecar_rejects_stride_drift():
    payload = make_v3_payload(stride=16)
    with pytest.raises(ValueError, match="stride"):
        load_sidecar_v3(payload, expected=make_config(stride=8), device=torch.device("cpu"))

def test_v2_requires_explicit_legacy_migration():
    with pytest.raises(ValueError, match="legacy_migration_mode"):
        migrate_v2_challenge_encoder(make_v2_payload(), legacy_migration_mode=False)
```

- [ ] **Step 2: 运行测试并确认因模块或接口缺失而失败**

Run: `python -m pytest code/tests/test_ccoi_pa_m21.py -q`

- [ ] **Step 3: 实现配置dataclass、字段验证、strict load和显式V2迁移**

- [ ] **Step 4: 运行测试，确认V3 round-trip、token/stride/contract/类别/域负测全部通过**

Run: `python -m pytest code/tests/test_ccoi_pa_m21.py -q`

### Task 2: Block split与近重复聚合审计

**Files:**
- Modify: `code/cvsrffi/ccoi_pa_m21.py`
- Modify: `code/tests/test_ccoi_pa_m21.py`

**Interfaces:**
- Produces: `split_v_select_retro(metadata, seed, block_candidates=(10,20,25)) -> RetroSplit`。
- Produces: `duplicate_audit(iq, metadata, split) -> dict[str, Any]`。

- [ ] **Step 1: 写block不跨role、guard排除、seed复现和base_index不交的失败测试**

```python
def test_retro_split_is_block_disjoint_and_reproducible():
    split1 = split_v_select_retro(meta_fixture(), seed=17)
    split2 = split_v_select_retro(meta_fixture(), seed=17)
    assert split1 == split2
    assert set(split1.fit_base_indices).isdisjoint(split1.audit_base_indices)
    assert set(split1.guard_base_indices).isdisjoint(split1.fit_base_indices)
    assert split1.role_by_group[(0, 0, 0, 1, 3)] != "BOTH"
```

- [ ] **Step 2: 运行聚焦测试并确认split接口缺失**

- [ ] **Step 3: 实现仅依赖metadata的B选择、65/35或70/30、guard和覆盖统计**

- [ ] **Step 4: 写exact/near duplicate聚合字段测试并确认失败**

- [ ] **Step 5: 实现固定投影、相似度分位数、0.999/0.995比例和sig间隔统计，不返回样本级特征**

- [ ] **Step 6: 运行Task 1–2全部测试**

### Task 3: 四fold与独立relation bank

**Files:**
- Modify: `code/cvsrffi/ccoi_pa_m21.py`
- Modify: `code/tests/test_ccoi_pa_m21.py`

**Interfaces:**
- Produces: `build_fold_records(...) -> FoldRecords`，每条记录带`base_index/fold_id/q/theta/target/support_raw_mask/holdout_raw_mask`。
- Produces: `build_relation_indices(audit_meta, bank_meta, relation, seed, physical_features=None) -> RelationMapping`。
- Produces: `common_anchor_mask(mappings, required=("F2","F3","F5"))`。

- [ ] **Step 1: 写四fold全部出现且raw mask不相交的失败测试**

```python
def test_all_nonoverlap_folds_are_emitted():
    records = build_fold_records(sidecar_fixture(), packet_fixture())
    assert sorted(records.fold_id.unique().tolist()) == [0, 1, 2, 3]
    assert not bool((records.support_raw_mask & records.holdout_raw_mask).any())
```

- [ ] **Step 2: 运行测试并确认旧fold0行为无法满足**

- [ ] **Step 3: 实现四fold记录和macro聚合**

- [ ] **Step 4: 写F2/F3/F4/F5严格关系、无fallback、bank-only和q不变性测试**

```python
def test_f3_is_same_tx_cross_rx_same_day_without_fallback():
    mapping = build_relation_indices(audit_meta(), bank_meta(), "F3", seed=3)
    valid = mapping.valid
    assert torch.all(bank_tx[mapping.index[valid]] == audit_tx[valid])
    assert torch.all(bank_rx[mapping.index[valid]] != audit_rx[valid])
    assert torch.all(bank_day[mapping.index[valid]] == audit_day[valid])
    assert torch.all(mapping.index[~valid] == -1)
```

- [ ] **Step 5: 实现稳定键候选映射、3个mapping seed、F6固定PA统计和F7不可用状态**

- [ ] **Step 6: 实现common-anchor mask与all-valid/common统计**

- [ ] **Step 7: 运行Task 1–3全部测试**

### Task 4: F矩阵、判据和敏感性聚合

**Files:**
- Modify: `code/cvsrffi/ccoi_pa_m21.py`
- Modify: `code/tests/test_ccoi_pa_m21.py`

**Interfaces:**
- Produces: `run_factor_matrix(train_records, audit_records, relation_mappings, head_seeds, ...) -> dict`。
- Produces: `evaluate_stage_a(c1, c4, coverage, sensitivity) -> StageAVerdict`。

- [ ] **Step 1: 写F0–F9 schema、共同anchor和fold macro手算测试**

- [ ] **Step 2: 运行测试并确认factor runner缺失**

- [ ] **Step 3: 复用同容量holdout head实现3个head seed和逐fold训练/评估**

- [ ] **Step 4: 写5%、5%、10%、80%、C4′对C1′3%边界值测试**

```python
def test_stage_a_partial_when_both_transfer_but_conditioning_gain_is_small():
    verdict = evaluate_stage_a(c1=passing_rows(f3=0.80), c4=passing_rows(f3=0.79), coverage=0.9, sensitivity=stable())
    assert verdict.status == "A_PARTIAL"
    assert verdict.next_route == "KEEP_PA_OPERATOR_STOP_CURRENT_CHALLENGE_CONDITIONING"
```

- [ ] **Step 5: 实现分组bootstrap、seed方向一致性和A_PASS/PARTIAL/FAIL状态机**

- [ ] **Step 6: 运行Task 1–4全部测试**

### Task 5: q条件probe、M0与leave-one-TX residual

**Files:**
- Modify: `code/cvsrffi/ccoi_pa_m21.py`
- Modify: `code/tests/test_ccoi_pa_m21.py`

**Interfaces:**
- Produces: `conditional_q_probe(...)`、`m0_exact_pair_retrieval(...)`、`run_loto_residual(...)`。

- [ ] **Step 1: 写ordered/shuffled/DeepSets与条件子集schema测试**

- [ ] **Step 2: 实现真实特征probe，不把固定条件字段作为输入**

- [ ] **Step 3: 写M0手算rank、Recall@1/5、MRR和候选池约束测试**

- [ ] **Step 4: 实现M0 clean/satellite同物理样本检索和theta距离**

- [ ] **Step 5: 写6折LOTO训练集合绝不包含held-out TX的失败测试**

```python
def test_loto_common_model_never_sees_held_out_tx():
    audit = run_loto_residual(loto_fixture())
    assert len(audit["folds"]) == 6
    for fold in audit["folds"]:
        assert fold["held_out_tx"] not in fold["train_txs"]
```

- [ ] **Step 6: 实现LOTO公共响应、residual probe和距离指标**

- [ ] **Step 7: 运行Task 1–5全部测试**

### Task 6: Truth-blind有界gate

**Files:**
- Modify: `code/cvsrffi/ccoi_pa_m21.py`
- Modify: `code/tests/test_ccoi_pa_m21.py`

**Interfaces:**
- Produces: `bounded_residual_fusion(...)`、`fit_truth_blind_gate(v_cal, groups, ...)`、`evaluate_stage_b(...)`。

- [ ] **Step 1: 写gate feature allowlist、g=0、全拒绝、全接受clip和role隔离失败测试**

```python
def test_zero_gate_is_exactly_core90():
    final = bounded_residual_fusion(base, operator, gate=torch.zeros(4), eta=0.2, clip_norm=0.5)
    torch.testing.assert_close(final, base, rtol=0, atol=0)
```

- [ ] **Step 2: 运行测试并确认gate接口缺失**

- [ ] **Step 3: 实现有界logit correction和显式feature allowlist**

- [ ] **Step 4: 实现V_cal block-group CV、多项logistic、rescue/harm代价和参数冻结**

- [ ] **Step 5: 写B的0.20pp、CI、clean、worst-RX、utility、coverage和LOO-RX门槛测试**

- [ ] **Step 6: 实现B verdict；非A_PASS必须输出`NOT_RUN_A_GATE`**

- [ ] **Step 7: 运行Task 1–6全部测试**

### Task 7: 真实runner与聚合artifact

**Files:**
- Create: `code/audit_phase1_ccoi_pa_m21.py`
- Create: `code/tests/test_phase1_ccoi_pa_m21_runner.py`
- Modify: `docs/CVS_PHASE1_CCOI_PA_M21_TRACE_20260825.md`

**Interfaces:**
- Consumes: Core90 checkpoint、旧C4 V2 sidecar、WiSig、不可覆盖output dir。
- Produces: 14个聚合artifact和source-only decision manifest。

- [ ] **Step 1: 写parser、不可覆盖、target/query拒绝、14 artifact闭合和A-gated-B测试**

- [ ] **Step 2: 运行runner测试并确认脚本缺失**

- [ ] **Step 3: 实现数据加载、旧challenge迁移、C1′/C4′同模板replay和V_select_fit选模**

- [ ] **Step 4: 接入split、duplicate、四fold、F矩阵、q/M0/LOTO和条件阶段B**

- [ ] **Step 5: 所有JSON使用UTF-8、`allow_nan=False`并记录`target_or_query_access=false`**

- [ ] **Step 6: 运行新测试和既有55项CCOI回归套件**

### Task 8: Launcher、最小预登记与本地发布

**Files:**
- Create: `code/scripts/launch_phase1_ccoi_pa_m21_20260825.sh`
- Create: `code/tests/test_phase1_ccoi_pa_m21_launcher.py`
- Create: `docs/experiments/PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C_REPORT.md`
- Create mirror: `E:\type10-7\automation_reports\CV-SincNet\PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C\report.md`

**Interfaces:**
- Produces: real-checkpoint no-query smoke后连续执行正式新run；拒绝已有smoke/output/log。

- [ ] **Step 1: 写launcher行为测试，验证不可覆盖和smoke失败不继续**

- [ ] **Step 2: 实现launcher，固定新run ID、GPU、路径、技术停止规则和预期artifact**

- [ ] **Step 3: 写最小预登记报告；先记录候选、矩阵、命令、CWD、路径、GPU和停止规则，实施提交完成后、发布前写入精确commit**

- [ ] **Step 4: 运行shell语法检查、parser help、dry-run和完整相关测试**

- [ ] **Step 5: 更新追踪表状态，精确stage本轮文件，提交、push并核对远端OID**

### Task 9: N607最小发布、运行、评分和最终报告

**Files:**
- Modify: 两份实验报告镜像
- Add after analysis: `docs/experiments/artifacts/PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C/*.json`

- [ ] **Step 1: 执行N607只读preflight和资源/路径检查，不干预无关进程**

- [ ] **Step 2: 构建一次release归档，执行一次本地/远端SHA比较和一次远端编译**

- [ ] **Step 3: 启动唯一新run并检查PID/CWD/cmdline/GPU/log增长**

- [ ] **Step 4: 短连接监控直到阶段A和条件阶段B闭合；只按预登记系统技术失败规则停止**

- [ ] **Step 5: 独立读取聚合artifact，生成A/B verdict和逐cell/seed/scene分析**

- [ ] **Step 6: 更新完整中文报告：方法、落地实现、结果、问题、负向效果、后续路线和追踪计数**

- [ ] **Step 7: 镜像非样本级JSON，精确stage、提交、push并核对远端OID**

## Plan self-review

- 规格中的42个追踪项均映射到Task 1–9。
- 代码接口在首次出现处定义，后续任务只消费已定义接口。
- 计划没有将逐文件SHA、seal、环境锁、完整多seed或多机制设为发布gate。
- 阶段B明确受阶段A约束；阶段A负结果仍产生完整科学artifact。
