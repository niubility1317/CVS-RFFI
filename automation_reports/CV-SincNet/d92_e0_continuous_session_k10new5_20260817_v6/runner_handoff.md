# v6 N607 Runner Handoff

## 终态

`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

冻结启动命令只提交一次。首次健康核验时已无run-owned PID或GPU compute进程，无需执行进程终止。`fresh_run_retry=false`。

## 冻结身份与落地证据

- run ID：`d92_e0_continuous_session_k10new5_20260817_v6`
- science commit：`e5fcb515de750bfdfd129996c66892d2af003f68`
- release commit：`3fd775e74015a50f01a92418c6588007ed5f7522`
- archive：`d92_e0_continuous_session_v6.tar.gz`，171110B，SHA256=`9514ef2dab5ba0f64c91e0e37e73bcc514c3bf8e8b4cdea5300ea88adc5a2b40`，31成员、安全tar验证通过。
- launch：`launch.sh`，2263B，SHA256=`c99254e0097c7c1727d560ae0973cc4100024bb78ecb372463f0e863417a4c30`，`bash -n`通过。
- 远端Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CUDA可见。初始run/log根、PID和本地retrieval均为ABSENT，GPU0–4无compute占用。
- 唯一提交命令：`cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v6 && nohup bash ./launch.sh >./launch.out 2>./launch.err </dev/null &`。

## 健康触发

- 首次健康核验时间：`2026-08-17T15:15:51+08:00`。
- 规范化异常指纹：`ContinuousSessionPredictionError: terminal schedule closure drift`，来源：`smoke_gpu0.out`。
- 类型：确定性执行故障。停止不是由wall/peak资源收据超限触发；资源数值未用于停机决定。
- 未观察到协议/安全/query泄漏或输出覆盖证据；但正式prediction前的smoke执行没有形成prediction闭合，因此按冻结规则封口而不重试。

## 保全与取回

| artifact | count |
|---|---:|
| `matrix_manifest.json` | 1 |
| `prepared_manifest.json` | 5 |
| `delta_receipt.json` | 5 |
| `COMMIT.json` | 5 |
| `fit_audit.json` | 21 |
| `resource_audit.json` | 21 |
| `prediction_manifest.json` | 0 |
| `job_receipt.json` | 0 |
| `execution_receipt.json` | 0 |

- 完整取回根：`E:/type10-7/local_artifacts/d92_e0_continuous_session_k10new5_20260817_v6`。
- `run_root`包含release archive、runtime/source、output、launch、`launch.out`与`launch.err`；220文件。
- `log_root`包含3个日志文件。远端与本地run/log文件计数一致。

## 分析边界与清理

- 未复制truth sidecar或truth receipt，`content_read=false`；未运行analyzer。
- 未读取或报告accuracy、H、BA、floor或其他性能字段。prediction/receipt闭合不存在，不能分析或推广。
- 结束时run-owned PID=0、GPU compute=0；本地`ssh.exe`=0、`scp.exe`=0、至N607 TCP22 `ESTABLISHED`=0。
