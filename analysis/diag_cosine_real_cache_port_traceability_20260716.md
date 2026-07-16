# Diag-cosine real-cache port traceability

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|----|----------------|-------------|--------------|--------|--------------|-------|
| DC-01 | 项目.md 7.1, 7.4, 10.3 | Predictor input is limited to sealed, pre-overlaid `leo_*_weak` support/query packages and the sealed Phase1 runtime; no raw/clean dataset or legacy loader input is exposed. | `code/cvsrffi/stage2_diag_cosine_exploration.py`, `code/scripts/run_cvs_stage2_diag_cosine_exploration.py` | verified | CLI help exposes only package/seal/output/device/candidate; focused package fixture test passes. | Reuses the existing one-time package admission evidence; no second control plane was added. |
| DC-02 | 项目.md 7.3, 10.3 | Adaptation is target-support-only and imports no source sample/cache/feature/logit/prototype/statistic/adapter. | `code/cvsrffi/stage2_diag_cosine_exploration.py` | verified | Fit API has no query or source argument; focused tests pass. | The historical runner is not reused as a whole because it contains source-bank and source-logit branches. |
| DC-03 | 项目.md 7.2 | Query truth, role, quota, ordering and batch-global assignment are absent from predictor inputs; each query is scored over all registered classes independently. | `code/cvsrffi/stage2_diag_cosine_exploration.py`;`code/cvsrffi/stage2_diag_cosine_scorer.py` | verified_real_dev | D1/D2 query-extension invariance和prediction NPZ exact-member测试PASS；N607真实D1/D2 prediction先冻结，独立scorer随后才连接truth。 | predictor receipt固定query truth/role/count/quota/global assignment均不可达。 |
| DC-04 | 项目.md 9.2, 9.3 | The same implementation supports Stage2-B old-only enrollment and Stage2-C old+seen-new enrollment without role-specific scoring. | `code/cvsrffi/stage2_diag_cosine_exploration.py` | verified | Matched package stage/state/registry validation and fixture execution pass. | Registered class handles come only from labeled support package indices. |
| DC-05 | 项目.md 10.3 | Port D1 `el_diag_aug3_fftrf_w4p0_e20` and add D2 fixed-prototype mode: three registered support LEO views, one query view, FFT-RF weight 4.0, at most 20 epochs, <=50k trainable parameters, <=256KB state, no dense query graph. | `code/cvsrffi/stage2_diag_cosine_exploration.py` | verified | 16 focused/legacy adapter tests pass; resource assertions cover both modes. | D2 trains only the shared diagonal scale and fixes the current registry's support prototypes with zero class bias. It is not a cross-Stage2-B/C frozen-state or true registry-stable method. FFT-RF is computed from the already-overlaid IQ row itself. |
| DC-06 | 项目.md 10.3 | Preserve complete loss/resource diagnostics and immutable prediction SHA binding. | `code/cvsrffi/stage2_diag_cosine_exploration.py`;`code/cvsrffi/stage2_diag_cosine_scorer.py`;`code/scripts/score_cvs_stage2_diag_cosine_exploration.py` | verified_real_dev | D1 before/after prediction SHA=`336247...`/`5ecd18...`，score SHA=`22d784...`；20epoch完整loss trace、MAC、状态、时延、峰值显存receipt已回收。 | 真实开发结果D1为before old 98.61%、after old/new/H 96.94%/90.67%/93.70%、old floor 95%、forgetting 1.67pp；new5仍差目标1.33pp。 |
| DC-07 | User scope | Avoid edits to currently conflicting SOMP-H bundle/runtime files and provide narrow tests only. | `tests/test_stage2_diag_cosine_exploration.py` | verified | `git diff --check` and focused pytest pass. | No Git commit in this subtask. |

## Reverse audit

- Verified: 7
- Deferred: 0
- Rejected: 0
- Blocked: 0
- Design parity: D1 is a strict mathematical port of the locked historical hyperparameters and same-row FFT96+RF32 construction. D2 is an explicitly named current-registry fixed-prototype approximation, not historical parity and not a true cross-state registry-stable method.
- Highest remaining risk: D1当前仅覆盖`20-1/713101/K10/new5`开发行；必须继续验证同seed new10/new20、K5和5receiver×确认seed矩阵。当前query时延为可分离向量化总时长均值，仍需singleton端到端p50/p95。
