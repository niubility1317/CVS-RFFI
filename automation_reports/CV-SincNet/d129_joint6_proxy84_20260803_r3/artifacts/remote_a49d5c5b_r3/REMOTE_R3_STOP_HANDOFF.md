# D129 r3 remote runner handoff

Status: STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT.

## Landing evidence

- Remote run root: /home/szu2070436088/2510044040/CV-SincNet/runs/d129_joint6_proxy84_20260803_r3
- Release commit: a49d5c5b220a490beebea53a4d5da08b80820113
- Release bundle SHA256: 4d6fe5a6de6e4d497fa3cdf6ebc71543301581b01be5703fc04649967cc4ca5c
- Fixture SHA256: d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669
- D104 archive SHA256: dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d
- Checkpoint SHA256: 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98
- Seven D129 source files and archive-byte method lock SHA256 73da38b66319ee69bf2076da698ada55b59e9569d671f4097fbdd80a45a8cd9f matched.
- Absolute CVS-RFFI Python py_compile passed under the frozen CPU-only environment.

## Executed stages

- Smoke completed: smoke.json SHA256 7902bc4e0f9f9988258604baec117cc75bbf3da5ab868deee64a3518209effcf; both candidates PASS_REAL_NO_TRUTH_SMOKE, truth_loaded=false, performance_result=false.
- Prepare completed in its own Python process. Receipt SHA256 ebf6172545563c85df9f9bccde0b5090534c0f81d7aed3ecec504ddc4573cd96; package SHA256 402f223eab2705c003d76b62a4ad39920c249bfa08bcfc380cdeb2627d5da691; plan SHA256 f69fcd7c1dd7f487164f568603cc3fcc70e155a8f1faada81165b384307522b9. Truth SHA remained receipt-only and truth was not retrieved.

## Stop trigger

Detached predict PID 517676 exited before creating predict/ or any prediction artifact. Complete count was 0/168 and score was not invoked. The normalized exception was:

cvsrffi.stage2_d129_joint6_heads.D129Joint6HeadsError: affine FP16 intercept is not representable

The preceding warning was RuntimeWarning: overflow encountered in cast at stage2_d129_joint6_heads.py:647. No D129 process remained, GPUs returned to 0%/1MiB, and no restart, tuning, code change or r4 launch occurred.

Every SSH/SCP action finished with local ssh.exe count 0 and local N607 port-22 connection count 0. The remote run root is preserved without deletion or overwrite.
