# Progress

## 2026-05-06

- Started model slimming plan based on `CVS-RFFI_Experiment_Analysis.md`.
- Confirmed current repo is on `main`, one local commit ahead because previous GitHub push failed due network.
- Confirmed `train.py` already disables DAC aux path for `branch_ablation=no_dac`.
- Added model slimming presets to `train.py`.
- Added `docs/MODEL_SLIMMING_PLAN.md` and `run_model_slimming_experiments.sh`.
- Verified Python compile, bash syntax, and `train.py --help` visibility for new presets.
- Started SSDG SSL extension after user clarified it must not affect the existing architecture.
- Added `WiSigUnlabeledSubsetDataset`, unlabeled-pool construction, and conservative pseudo-label memory.
- Wired SSDG pseudo-label CE plus weak/strong consistency as opt-in losses in `train.py`.
- Added SSDG presets, `docs/SSDG_SSL_EXPERIMENTS.md`, `run_ssdg_experiments.sh`, and unit/smoke coverage.
- Verified with `py_compile`, `bash -n`, direct SSDG tests, train help visibility, and pseudo-label loss smoke test.
- Committed SSDG work locally. Push to `https://github.com/niubility1317/CVS-RFFI.git` failed because github.com:443 could not be reached from this machine.
