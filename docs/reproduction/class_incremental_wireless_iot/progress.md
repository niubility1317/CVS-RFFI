# Progress Log

## 2026-07-08

- Read `E:\type10-7\AGENTS.md` and `E:\type10-7\项目.md`.
- Read skills: `using-superpowers`, `dispatching-parallel-agents`, `pdf-extraction`, `planning-with-files`, `academic-writing`, `cv-sincnet-n607-automation`.
- Searched memory for prior paper reproduction and multi-agent supervision rules.
- Spawned four read-only subagents: paper mechanism extraction, repo implementation mapping, experiment protocol/N607 boundary, and paper-only audit checklist.
- Confirmed `pdfplumber 0.11.9` is available and the PDF exists at `C:\Users\lh594\Desktop\Class-Incremental_Learning_for_Wireless_Device_Identification_in_IoT.pdf`.
- Confirmed root workspace is not Git; selected `E:\type10-7\github_publish\CVS-RFFI-repo` for Git-backed deliverables.
- Added RED tests in `tests/test_paper_reproduction_csil_class_incremental.py`; first valid RED was `ModuleNotFoundError: No module named 'paper_reproduction.CSIL'`.
- Added `paper_reproduction/CSIL/` with protocol, model, losses, metrics, train dry-run and README.
- Added `paper_reproduction/configs/csil_adsb_paper_faithful.json`.
- Added `paper_reproduction/CSIL/paper_checklist.md`.
- Ran `conda run -n ssr-gpu python -m pytest tests/test_paper_reproduction_csil_class_incremental.py -q`; after fixing float32 metric rounding, result was `6 passed`.
- Post-implementation audit found `embedding.bias` was not masked. Added RED regression test `test_csil_gradient_masks_lock_old_embedding_bias_and_weights`, verified it failed, then masked old embedding bias gradients in `CSILClassifier.apply_gradient_masks`.
- Re-ran `conda run -n ssr-gpu python -m pytest tests/test_paper_reproduction_csil_class_incremental.py -q`; result was `7 passed`.
- Re-ran `conda run -n ssr-gpu python -m compileall -q paper_reproduction/CSIL`; result exit code 0 after a serial retry.
- Ran `conda run -n ssr-gpu python -m paper_reproduction.CSIL.train --config paper_reproduction/configs/csil_adsb_paper_faithful.json --dry-run --formal`; result exit code 0.
- Searched public sources and cloned official `pcwhy/CSIL` to `E:\type10-7\local_artifacts\external_refs\pcwhy_CSIL`; local HEAD is `8ce8637daf4dc60eeb1c56bff64c050c5b2353e9`.
- Spawned three read-only subagents for official repo structure, MATLAB mechanism migration, and CVS protocol alignment.
- Added RED tests for official zero-bias `5*cosine+5`, expansion dtype/device preservation, EWC old-block slicing, KD shape/detach, and masked SGD over full momentum/L2 update.
- Verified RED failures, then updated `model.py` and `losses.py` to match the official MATLAB semantics.
- Ran `conda run -n ssr-gpu python -m pytest tests/test_paper_reproduction_csil_class_incremental.py -q`; result was `12 passed`.
- Added `public_sources.md` and `official_repo_migration.md` under `docs/reproduction/class_incremental_wireless_iot/`.
- Renamed the reproduction package from `paper_reproduction/csil_class_incremental_iot` to `paper_reproduction/CSIL` and updated imports/docs to use `paper_reproduction.CSIL`.
