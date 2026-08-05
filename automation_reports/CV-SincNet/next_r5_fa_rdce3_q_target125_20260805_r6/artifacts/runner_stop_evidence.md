# r6 runner停止证据

- 时间：2026-08-05；run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r6`。
- preflight、closure落地、六入口compile、新asset构建、prepare均成功；asset和prepare文件已取回。
- GPU truth-free smoke：row0/scene0、`--device cuda:0`、exit=1；fingerprint=`sealed runtime / same-IQ ReLU binding drift`。
- CPU兼容性核验：同一row/scene、`--device cpu`、exit=1；fingerprint=`Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)`，外层=`sealed z_id160 materialization failed`。
- 只读绑定诊断：REG0 support shape=`[60,160]`，`max_abs_error=6.9090724e-4`，容差=`2e-6`，sealed/recomputed均无全零行。
- 8个predict-shard未启动；无prediction/shard-manifest/merged-manifest/truth/score。
- 停止后GPU0至7均`0%/1MiB`，run-owned进程为0，SSH=`SSH_CLEAN`；r1至r5未触碰。
- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；`fresh retry authority=无`。
