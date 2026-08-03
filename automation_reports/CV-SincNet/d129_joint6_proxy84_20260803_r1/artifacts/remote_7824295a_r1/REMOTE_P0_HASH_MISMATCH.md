# D129 r1 partial remote evidence

Status: `STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`.

## Immutable-run evidence

- Remote run root: `/home/szu2070436088/2510044040/CV-SincNet/runs/d129_joint6_proxy84_20260803_r1`
- Direct-N607 preflight route: `ssh -F E:\type10-7\tools\n607_ssh_config N607`
- The run root was absent before creation. Only `input/`, `source/`, `logs/`, and `smoke/` were created before the failure gate.
- Remote bundle SHA256: `a0068428b2d537a3d073031639fd704a03b760886f3157f928e3fada947bec3b`
- Remote fixture SHA256: `d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669`
- Remote D104 archive SHA256: `dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`
- Remote checkpoint SHA256: `2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`

## P0 trigger

After extracting the verified bundle to `source/`, the required D129 Python source files matched their frozen SHA256 values. The extracted method lock did not:

- Expected / frozen commit `7824295a2f4d7897d6ba4cd9370e97bce5988171`: `fd47cd9f52d4ae29100ebcaff5e2a64c5397294b72394990e2f2040a16cbedd7`
- Extracted remote `source/configs/d129_joint6_method_lock_20260803.json`: `73da38b66319ee69bf2076da698ada55b59e9569d671f4097fbdd80a45a8cd9f`
- Retrieved evidence file: `remote_method_lock.json` (SHA256 `73da38b66319ee69bf2076da698ada55b59e9569d671f4097fbdd80a45a8cd9f`)

This is a wrong-checkout/hash P0 gate. Smoke, prepare, predict, and score were not started. The final run-root check found `smoke/` and `logs/` empty and `prepare/`, `predict/`, and `score/` absent. No D129 process existed at 2026-08-03T19:33:30+08:00; all eight GPUs were 0% utilization and 1 MiB used. The noninteractive remote shell also reported `conda: command not found`; this was observed only after the hash gate and did not affect the P0 decision.

Every bounded SSH/SCP task finished with local `ssh.exe` count 0 and local N607 port-22 connection count 0. Remote artifacts are preserved in place; no remote content was deleted or overwritten.
