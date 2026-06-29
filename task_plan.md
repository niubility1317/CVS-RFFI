# Task Plan

## Goal
全面分析当前工作区内各版本代码、实验配置与训练日志，筛选最适合继续推进的模型与训练路线，并生成中文报告。

## Phases

| Phase | Status | Notes |
|---|---|---|
| 1. Inventory sources | complete | Main evidence: `type10-4`, `type10-7`, docs, old `unkown` report; root SGC logs are mostly dry-run/start failures. |
| 2. Extract experiment metrics | complete | Parsed 183 `.log` files; 110 contain training epochs. |
| 3. Compare code routes | complete | Compared `type10-4`, `type10-7`, root SGC/SSDG, and `unkown` satellite-hybrid branch. |
| 4. Decide model route | complete | Recommended R19 Lite-B no-DAC + Fishr; R25 Lite-D no-DAC as compact candidate; SGC as next experiment. |
| 5. Generate report | complete | Wrote `docs/CVS_RFFI_model_route_report_20260506.md`. |

## Decision Criteria
- Prefer validated logs over intent-only scripts.
- Prefer routes with high target/generalization performance and stable training behavior.
- Penalize routes with only failed/empty logs or unresolved integration risk.
- Preserve distinctions between completed results, partial runs, and planned experiments.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| `rg --files` access denied | Initial file inventory | Switched to PowerShell recursive listing. |
| `python -m pytest ...` failed: `No module named pytest` | Verification | Used `py_compile`; attempted manual tests, but runtime imports fail because `torch` is not installed in current Python. |
| `rg` access denied | Current SGC search | Use PowerShell `Get-ChildItem` and `Select-String` instead. |

## Current Task: SGC Channel Processing Analysis

### Goal
Study the local SGC satellite-ground channel processing implementation and explain its distinctive mechanisms, especially residual links, then compare common satellite communication channel processing methods with RFFI-friendly adaptations.

### Phases

| Phase | Status | Notes |
|---|---|---|
| 1. Locate SGC implementation | complete | Root code/docs/tests found; `rg` unavailable so PowerShell search used. |
| 2. Extract SGC mechanisms | complete | Found four adapter blocks, residual blending/compensation, residual loss, staged training. |
| 3. Compare communication vs RFFI processing | complete | Separate code recovery objectives from fingerprint preservation. |
| 4. Produce Chinese explanation | complete | Ground claims in local files and give practical recommendations. |

## CODEX-PHASE2-OPENWORLD: Current Handoff

### Goal
把 Phase 2 在轨部署诊断转成 Codex 可执行工程计划：少样本域适应、新类注册、未知类拒识、TX/RX 双向解耦、域骨干使用方式、SGC no_amp 路线和角空间诊断。

### Primary Docs

Read first:

```text
docs/PHASE2_OPEN_WORLD_DIAGNOSIS_AND_CODEX_PLAN_20260626.md
docs/PHASE2_FULL_PROTOTYPE_MASK_OPENWORLD_IMPLEMENTATION_20260626.md
docs/PHASE2_PROTOTYPE_DISTANCE_BOUNDS_AND_NEW_CLASS_PRIORS_20260626.md
```

### Key Takeaways

- 当前 `source/sgc_augment/sgc_adapt` 不是完整 open-world Phase 2。
- 当前 `sgc_adapt` 只训练小 adapter，不应被视为最终在轨适配路线。
- 当前解耦主要是 `z_id` 去 RX/day，缺少 `z_dom/z_rx` 去 TX、TX×RX 四角约束和 open-set 校准。
- 地面基模需要按角空间诊断：class radius、inter-class angle、effective rank、TX/RX ANOVA、known-vs-unknown AUROC/FPR95/OSCR。
- 地面训练必须导出每类训练样本到原型的 p95/p99/max/robust_max/top-k 最大距离，供阶段二未知拒识使用。
- 新类 support 很少时，可以借助地面原型库提供半径、尾部分布、最近旧类邻域和域漂移先验，但不能把新类原型强行拉向旧类。
- SGC 后续优先 `no_amp`；full per-channel amplitude normalization 可能抹掉 IQ imbalance/PA/RX 线索。
- 目标 RX 必须包含旧已知 TX 锚点，否则 TX 与 RX 效应不可分。
- 完整实现蓝图已拆成：TX/RX/TX-domain prototype banks、feature masks、TX×RX geometry losses、balanced sampler、multi-prototype open-world head、`phase2_adapt.py`。

### Codex Phases

| Phase | Status | Notes |
|---|---|---|
| P0. Diagnostics | planned | Extend `eval_feature_diagnosis.py` with angular geometry, effective rank, TX/RX ANOVA, domain leakage probes, and open-set metrics. |
| P1. Prototype banks | planned | Add `phase2_prototypes.py` with TX, domain, TX-domain local prototype banks, radius tracker, and `PrototypeDistanceTracker`. |
| P2. Prototype bounds | planned | Export `phase2_proto_bounds.json/.pt` with per-class p95/p99/max/robust_max/top-k distances and class-domain bounds. |
| P3. Sampler/loss | planned | Add balanced TX×RX sampler, rectangle losses, real-gradient prototype margin, and TX adversary on domain feature. |
| P4. Feature masks | planned | Add `feature_masks.py`; learn `M_tx/M_rx/M_int` with overlap/coverage/binary/balance regularizers. |
| P5. Multi-prototype head | planned | Add `open_world_head.py` for old/new class prototypes, radii, energy, multiview agreement, and unknown rejection. |
| P6. New-class priors | planned | In `phase2_adapt.py`, use source radius priors, nearest-old-class priors, target-domain drift priors, and shrinkage for few-shot new class registration. |
| P7. Phase2 entry | planned | Add independent `phase2_adapt.py`, `open_world_head.py`, `open_world_memory.py`, and `eval_open_world.py`. |

### Do Not Do

- Do not directly concatenate `z_dom` into TX classifier.
- Do not let domain prototypes act as direct TX evidence.
- Do not use raw maximum distance as the only unknown threshold; keep it for audit and default to p99/robust_max modes.
- Do not force new-class prototypes toward old-class prototypes; old prototypes provide radius/uncertainty/domain priors, not identity direction.
- Do not rely only on max-softmax threshold for unknown rejection.
- Do not continue full SGC from source as the default route.
- Do not apply ordinary entropy minimization to uncertain/unknown target samples.
- Do not ignore target RX old-known anchor coverage.
