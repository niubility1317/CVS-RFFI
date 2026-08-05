# r5 runner stop evidence

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r5`
- final status：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- trigger：2026-08-05T12:17:28+08:00监控发现shard3与shard7两个不同shard均在prediction前退出，均出现确定性指纹`cvsrffi.stage2_zid_student_t_qknn.ZIDStudentTQKNNError: z_id rows contain a zero-norm vector`，runtime包装为`NextR5FATarget125RuntimeError: sealed z_id160 materialization failed`。不涉及性能值。
- binding：停止前逐一核验PID 1662376(shard0)、1662377(shard1)、1662378(shard2)、1662380(shard4)、1662381(shard5)、1662382(shard6)；CWD均为`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r5/source`，cmdline均为r5 closure中的`run_next_r5_fa_target125.py predict-shard`，带正确plan/context SHA及对应`--shard-index`。
- stop action：仅对上述6个run-owned PID发送定向`SIGTERM`；3秒后全部停止。未使用广泛kill，未删除、覆盖、重启任何artifact；shard3/7原生失败进程未重复干预。
- post-stop：2026-08-05T12:18:18+08:00，PID1662376–1662383均STOPPED；GPU0–7均`1MiB/0%`且compute apps为空；SSH=`SSH_CLEAN`。
- partial inventory：无shard manifest、merged prediction manifest、truth catalog或score artifact；JSON共678个、总字节`27330246`，按shard为`96/96/102/34/96/96/96/62`（shard0至shard7）。
- logs：shard3/7各2296B，SHA=`e3baa52eeb3553994b1f69d08c69ece18eb7ecd336f75658c06d9d38e0b3b7c8`；shard0/1/2/4/5/6为空，SHA=`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
- retrieved immutable evidence：本目录包含8份日志、678个partial prediction JSON、asset wire/manifest、prepared plan/context/receipt和smoke receipt；remote/local SHA已核对。该目录仅为停止证据，不是性能结果；fresh retry authority=`无`。
