# Progress

## 2026-05-06
- Started comprehensive version/log analysis.
- Read required skills: `using-superpowers`, `planning-with-files`.
- Ran session catchup; no unsynced root planning context was reported.
- Root inventory found active code, docs, run folders, logs, and older `unkown/` planning/report materials.
- `rg --files` failed with access denied; switched to PowerShell listing.
- Read docs for all-preset, SGC, slimming, SSDG, and prior satellite-upgrade report.
- Sampled representative `type10-4` and `type10-7` logs; confirmed metric formats and NaN/collapse signatures.
- Ran an inline parser across all `.log` files; current top clean/OOD routes are SAT37 Fishr, SAT34 GroupDRO, and compact SAT07 R25.
- Checked root launcher logs; 2026-05-06 SGC/slim/SSDG logs mostly failed because `python` was unavailable or were dry-runs.
- Generated `docs/CVS_RFFI_model_route_report_20260506.md`.
- Verified report file exists and contains key conclusions for `SAT37`, `SAT34`, `SAT07`, SGC-Adapter, and root launcher failure caveat.
- `python -m py_compile` passed for main SGC/report-relevant Python files and tests.
- Could not run pytest because current Python lacks `pytest`; could not manually invoke tests because current Python also lacks `torch`.
- Prepared next-stage training plan for `Lite-B no-DAC + SAT mixed consistency + conservative MixStyle + Fishr`.
- Added `docs/superpowers/plans/2026-05-06-lite-b-sat-mixed-fishr.md` with smoke run, full run, success gates, fallback variants, and promotion record steps.
- Started current SGC channel processing analysis requested by user.
- Read existing planning files and appended a scoped current-task section to `task_plan.md`.
- `rg` search failed with access denied again; logged it and switched to PowerShell search.
- Read current SGC core files: `sgc_adapter.py`, `sat_channel.py`, `sgc_losses.py`, SGC design docs, train wiring, model integration, scenario configs, and tests.
- Logged key SGC mechanisms in `findings.md`.
- Completed communication-vs-RFFI comparison and prepared final Chinese explanation.

## 2026-05-07
- Analyzed complete `5.7/logs` SGC run set and generated `5.7/sgc_log_summary.json`.
- Wrote `docs/SGC_5_7_analysis_and_merged_plan_20260507.md` with wrong-route analysis, SGC effect decomposition, and merged best-model direction.
- Wrote `docs/final_best_sgc_experiment_groups_20260507.md` defining Phase A best-model exploration and Phase B SGC/residual mechanism checks.
- Added `run_final_best_sgc_queue.sh`, an 8-GPU queue launcher. Default is now `PHASES=A`, so it runs best-model exploration first; SGC/residual runs only when `PHASES=B` or `PHASES=A,B` is set.
- Verified `run_final_best_sgc_queue.sh` with `bash -n`, Phase A dry-run, and full `PHASES=A,B` dry-run.
