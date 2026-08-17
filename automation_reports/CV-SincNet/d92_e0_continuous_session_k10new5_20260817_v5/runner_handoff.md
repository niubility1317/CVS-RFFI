# v5 N607 Runner Handoff

## 终态

`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

冻结启动命令只提交一次。技术失败发生后没有存活的run-owned PID或GPU compute进程，因此没有执行进程终止。`fresh_run_retry=false`。

## 冻结身份与落地证据

- run ID：`d92_e0_continuous_session_k10new5_20260817_v5`
- science commit：`2ccfcfd1c83e7012a3821faaf3d453dad231ccdf`
- release commit：`04a2f05cb898852026eec6667afd1fb576520b22`
- archive：`d92_e0_continuous_session_v5.tar.gz`，171051B，SHA256=`4295d20c9f7841d1f9116e08a101b38c47d9770645e84df1287e36b6528d050c`，31成员、安全tar验证通过。
- launch：`launch.sh`，2294B，SHA256=`562524a13dea3c8aee9e2df6da2dd63699b139cf83c5b802ad9dbcbf64ffc8aa`，`bash -n`通过。
- 远端Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CUDA可见。初始run/log根、PID和本地retrieval均为ABSENT，GPU0–4无compute占用。
- 唯一提交命令：`cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v5 && nohup bash ./launch.sh >./launch.out 2>./launch.err </dev/null &`。

## 技术触发与资源记录

- 健康核验时间：`2026-08-17T14:58:07+08:00`。
- 规范化异常指纹：`ContinuousSessionPredictionError: registration wall hard gate failed`，来源：`smoke_gpu0.out`。
- 位置绑定：`smoke/leo_clear_weak/DA1_REG0`。fit receipt：`session_index=0`；scene字段=`ABSENT`，schedule字段=`ABSENT`。
- resource receipt仅记录的技术字段如下；不作性能或因果解释。

| 字段 | 值 |
|---|---:|
| `registration_wall_time_ns` | 267672490 |
| `registration_wall_hard_max_ns` | 300000000 |
| `registration_incremental_peak_working_set_bytes` | 7524352 |
| `registration_incremental_peak_hard_max_bytes` | 4194304 |
| `registration_peak_rss_bytes` | 1469804544 |

## 保全与取回

| artifact | count |
|---|---:|
| `matrix_manifest.json` | 1 |
| `prepared_manifest.json` | 5 |
| `delta_receipt.json` | 5 |
| `COMMIT.json` | 5 |
| `*.npz` | 75 |
| `fit_audit.json` | 1 |
| `resource_audit.json` | 1 |
| `prediction_manifest.json` | 0 |
| `job_receipt.json` | 0 |
| `execution_receipt.json` | 0 |

- 完整取回根：`E:/type10-7/local_artifacts/d92_e0_continuous_session_k10new5_20260817_v5`。
- `run_root`包含release archive、runtime/source、output、launch、`launch.out`与`launch.err`；180文件，canonical tree SHA256=`4e17d62bc2f54979b51ea012e9e6b1bd135c69e8c17c423556c303085256e8ce`。
- `log_root`包含3个日志文件，canonical tree SHA256=`423fd4a5b698becafe0677488e9633f5efdced3c2c6e0f38118c6faf721a29bf`。
- 上述两份canonical路径加逐文件hash清单均与远端匹配。

## 分析边界与清理

- 未复制truth sidecar或truth闭合receipt，`content_read=false`；未运行analyzer。
- 未读取或报告accuracy、H、BA、floor或其他性能字段。正式prediction/receipt闭合不存在，不能分析或推广。
- 结束时run-owned PID=0、GPU compute=0；本地`ssh.exe`=0、`scp.exe`=0、至N607 TCP22 `ESTABLISHED`=0。
