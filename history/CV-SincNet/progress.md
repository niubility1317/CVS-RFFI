# Paper Baseline Progress

## 2026-05-07
- Started implementation of four paper baselines requested by the user.
- Loaded requested skills: `using-superpowers`, `brainstorming`, and `planning-with-files`.
- Read project structure and latest WiSig/model files.
- Read supplied Codex plans; the TIFS2025 plan decoded cleanly with UTF-8.
- Replaced old planning files with the current baseline implementation plan.
- Added `baselines/` shared common utilities: GRL, ResNet1D/2D, spectrogram, RF augmentation, datasets, metrics, config helpers.
- Added TIFS2025 baseline model/loss/data/train/eval files.
- Added RIEI model/loss/train/eval files.
- Added DRIFT model/loss/train/eval files.
- Added receiver-agnostic collaborative model/loss/train/eval/fusion files.
- Added configs under `configs/` and launch scripts under `scripts/`.
- Ran smoke tests with `D:\App\miniconda3\envs\rff_std\python.exe`; import/forward/loss backward tests pass.
- Ran synthetic one-epoch train/eval entries for all four method families and saved metrics/checkpoints under `outputs/`.
