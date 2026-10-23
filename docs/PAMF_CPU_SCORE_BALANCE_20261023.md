# Remaining CPU scoring: bounded parallel resource handover

User authorization: on 2026-10-23 the user requested distributing the remaining tasks more evenly to accelerate completion. This changes resource scheduling only, not scientific settings or the experiment matrix.

At the 10:26 Hong Kong preflight, all model training, frozen prediction, G IQ caches and both exclusive GPU timing panels were complete. The effective remaining queue contained 49 CPU score/analysis stages. CPU busy was about 28%, available RAM 392 GiB and I/O wait below 0.01%. The old dispatcher admitted one scorer and then slept 30 seconds even when that scorer had already finished. GPUs are not consumers of these remaining stages, so filling them with extra inference or retraining would not accelerate this queue and is outside scope.

## Fixed resource change

- Resource owner: `20261023-phase1-PAMF-cpu-balance-control-r01`.
- Four independent CPU score workers maximum; polling interval one second. Each worker retains the existing two-thread numerical environment and `CUDA_VISIBLE_DEVICES=''`.
- Additional admission requires valid telemetry with CPU busy <75%, RAM available >=64 GiB and I/O wait <5%. Pressure pauses new admission only; no healthy worker is killed.
- The existing readiness rule still requires every registered prediction/dependency completion before its score or analysis can start. Each existing row retains its own output/config/log identity; no row is repeated.
- The current complete resolved job list, statuses, excluded historical rows, failure counters and per-job scientific releases are inherited unchanged. All one-time extension preparation flags are disabled for this handover because their jobs already exist.
- The existing dispatcher alone is retired after exact PID/CWD/argv validation. Existing healthy score workers are adopted, not stopped or relaunched. No new training, inference, target prediction, model selection or experiment hypothesis is introduced. Frozen truth-last scoring stays separate from all predictors.

## Release and validation

The controller and publisher are fixed by the task Git commit. The publisher uses one immutable release/output root, one archive SHA comparison and one native remote compile. It reuses existing validated data/predictions and published scientific workers; no model/checkpoint loading or extra data validation is performed by this resource-only release.

Focused checks cover capacity, active-job duplicate protection, missing prediction dependencies, exclusion of GPU jobs, host-pressure admission, unchanged defaults, invalid configuration and exact resolved-list/worker-release retention. The existing GPU load policy regression checks remain applicable. One independent P0/P1 review is limited to launch/ownership/dependency/output correctness. A whole-repository test suite is not an additional release gate for this resource-only change (`REJECTED_EXTRA_GATE`).

Exact local command: verified Python `-m experiments.cric_dura_comparison.publish_cpu_balance --repo <registered worktree> --output <fresh local artifact folder> --previous <current continuation_runtime.json> --ssh-config <project n607_ssh_config>`. N607 uses the unchanged ordinary-user environment. Private runtime and launch evidence belong under `local_artifacts/cpu_score_balance_20261023/`, not Git.

Stop rule remains the existing two-identical-unresolved-technical-fingerprints rule. No automatic retry of old failures or stopped input-invalid rows. Post-release verification must independently confirm dispatcher identity, actual CPU worker concurrency, log/completion growth, no CUDA score workers and preservation of the scientific job count/configuration. Actual throughput, not occupancy, determines the reported speedup.

Local verification: five new CPU scheduling/handover tests and eleven existing GPU admission regressions PASS. The new tests were first observed failing on the missing behavior. One independent P0/P1 review returned PASS, checking ownership, dependencies, capacity, original scientific releases and output preservation; it made no edits or remote changes.
