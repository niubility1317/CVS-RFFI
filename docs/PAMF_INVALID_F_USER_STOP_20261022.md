# Stop input-invalid historical F workers

Authorization: user explicitly requested stopping the old input-misaligned F experiments on 2026-10-22. Repaired binding runs, all valid comparison workers, all G workers, and other owners are outside the stop scope.

The bounded operation is implemented by `tools/stop_pamf_invalid_f.py`. It follows the existing controller's recorded handover to the current live owner, identifies only the 69 original F0 training rows listed by `F_binding_recovery.replacements`, and intersects those rows with exact live PID/start-time/CWD/argv identities. It never selects the repaired runs by name prefix alone.

The existing controller classifies missing completion after process termination as technical failure. To prevent this deliberate stop from triggering its repeated-failure guard, the operation retires only the verified dispatcher, terminates the selected historical F worker trees, records `STOPPED_BY_USER` in a separate preserved handover snapshot, and reuses the already-published current controller code. Valid workers retain their PIDs; the full resolved jobs, dependencies, per-job code releases, numerical settings, matrix, seeds, budgets, and resource policy remain unchanged. One-time extension preparation is not repeated because its resolved jobs already exist.

No model output is deleted or overwritten. No synthetic training-completion artifact is created. No old invalid row becomes a valid result, and old excluded rows cannot be resubmitted. Existing failure counters are preserved rather than cleared.

Preflight at 21:47 Hong Kong time found seven remaining historical workers: F13, F14, F16, F18, F19, F20 and F0 matched ordinary PL, all seed 2026092703. Eighteen valid workers were protected. Earlier F12/F17 had finished naturally before this stop. The live source controller had moved to the supervised-supplement owner; its complete resolved queue is retained.

Private operational evidence is stored under `local_artifacts/stop_invalid_F_20261022/`; remote control paths and runtime state stay outside Git. Execution requires an explicit `--execute` option after the read-only preflight. Final completion must be independently checked against live process identities, CUDA process IDs, new controller state, unchanged failure counters, and continued valid-worker progress.
