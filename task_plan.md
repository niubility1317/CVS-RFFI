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
