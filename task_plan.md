# Model Slimming Plan

## Goal

Build a root-level CVS-RFFI model slimming plan that preserves performance while reducing parameters, training cost, and inference overhead.

## Key Questions

- Should DAC branch/view remain when the main route is `no_dac`?
- Which modules are safe to remove, compress, or keep?
- Which experiment presets should be available directly from `train.py` and launch scripts?

## Phases

| Phase | Status | Notes |
| --- | --- | --- |
| Read experiment analysis and current code | complete | Analysis identifies R19/R25 no-DAC routes as best performance/efficiency tradeoff. |
| Decide slimming principles | complete | Mainline is no-DAC; DAC-only auxiliary view is not needed for no-DAC models. |
| Add documentation plan | complete | Added `docs/MODEL_SLIMMING_PLAN.md`. |
| Add preset experiment groups | complete | Extended `slim_group` with anchor, compact, robust, and boundary presets. |
| Add launch script entries | complete | Added `run_model_slimming_experiments.sh`. |
| Verify and commit | complete | Compile and script checks passed; changes committed locally. |

## Constraints

- Do not touch child workspaces such as `type10-4/`, `type10-6-sat/`, `type10-7/`, or `unkown/`.
- Preserve default behavior for existing non-slimming commands.
- Prefer root project patterns: `slim_group`, `branch_ablation`, staged training, and existing launch scripts.
