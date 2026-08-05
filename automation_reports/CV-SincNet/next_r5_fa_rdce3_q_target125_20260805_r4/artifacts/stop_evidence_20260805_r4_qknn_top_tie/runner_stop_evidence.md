# r4 runner stop evidence

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r4`
- stop class：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- trigger observation：2026-08-05T11:54:14+08:00；shard0、shard3、shard5、shard7四个不同shard均出现同一确定性异常指纹：`cvsrffi.stage2_next_r5_fa_target125_core.NextR5FATarget125CoreError: qKNN exact top tie fails closed`，随后由runtime包装为`FA-RDCE3/qKNN four-state execution failed`。
- stop action：对仍存活且已核验属于本run的PID 1646282（shard1）、1646283（shard2）、1646285（shard4）、1646287（shard6）发送定向`SIGTERM`；3秒后四者均停止。未使用广泛`pkill`，未删除、覆盖或重启任何artifact。
- binding evidence：四个被停止PID的CWD均为`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r4/source`；cmdline均为同一closure中的`run_next_r5_fa_target125.py predict-shard`，分别带`--shard-index 1/2/4/6 --device cuda:0`和r4 prepared plan/context SHA。
- post-stop verification：2026-08-05T11:55:19+08:00，GPU0–7均为`1MiB/0%`，`nvidia-smi --query-compute-apps`为空；SSH清理=`SSH_CLEAN`。`pgrep`输出中的唯一匹配为当前只读检查shell自身，run-owned Python PID均已不存在。
- remote log SHA：失败日志shard0/3/5/7均为`3c203634df9f9fc53d0d2f02d1cac8dac18662aa838991848f5616307fa2dbb2`；被SIGTERM的空日志shard1/2/4/6均为`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
- partial prediction inventory：remote与本地拉回均为422个JSON、总字节`16850416`；按shard计数为`shard0=36, shard1=68, shard2=70, shard3=26, shard4=64, shard5=40, shard6=68, shard7=50`。没有生成任何shard manifest、merged prediction、truth或score artifact。
- retrieved files：本目录的`shard_0.log`至`shard_7.log`和`shards/`部分prediction JSON均为远端r4 run的只读副本；本地拉回后SHA/计数已核对。该目录及其内容仅用于停止证据，不是性能结果。
