# D92 E0连续session实验发布报告

## 运行身份与状态

- 实验ID：`d92_e0_continuous_session_k10new5_20260817_v5`
- 方法：`D92_E0_CUMULATIVE_REPLAY_SESSION_V1`
- 目标：直接验证D92 E0按单类或少数新类连续注册时的状态、精度、资源与终端等价性。
- 当前状态：`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_PERFORMANCE_RESULT`
- 科学代码提交：`2ccfcfd1c83e7012a3821faaf3d453dad231ccdf`
- 结论边界：`DEVELOPMENT_ONLY_CONTINUOUS_SESSION_SCREEN`。

## 本轮唯一修复

v4已完成全部truth-free Phase A注册状态封存，但Phase B错误地要求`DA1_REG0`与`DA1_REG1`使用相同查询令牌，并在smoke停止。提交`2ccfcfd1`改为：`DA1_REG0`绑定注册前sealed apply查询，所有`DA1_REG1`session绑定注册后sealed apply查询；不改方法、数据、矩阵、阈值、预测核或评分规则。连续session相关测试51项、`py_compile`和diff检查通过。

## 冻结矩阵

| 维度 | 冻结值 |
|---|---|
| receiver outer | `20-1`,`3-19`,`7-14`,`7-7`,`8-8` |
| scene | `leo_clear_weak`,`leo_low_elev_weak`,`leo_rain_weak` |
| seed / K | `713106` / `10` |
| 注册轨迹 | `batch_5`、`singleton_forward`、`singleton_reverse`、`chunk_2_2_1` |
| 规模 | 5 outer×3 scene×4 schedule=60 trajectories；session fit=210 |
| 状态 | `DA1_REG0`、`DA1_REG1_S1..S5` |
| query | 每样本在全部已注册类中独立判决；truth/fit/update/selection/role/quota/global-reassignment均false |
| 资源门 | 每个`DA1_REG1`注册wall≤300ms、增量working-set≤4MiB；query state/MAC与原F0类数等价 |

## N607运行交接

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- 运行CWD：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v5`
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_continuous_session_k10new5_20260817_v5`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：GPU0先做truth-free smoke；正式5个outer分别使用GPU0–GPU4并发。

唯一detached启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v5 && nohup bash ./launch.sh >./launch.out 2>./launch.err </dev/null &
```

完整预测封存后，主代理独立取回五份truth sidecar并运行truth-last分析。发布脚本和runner不读取性能。

## 停止与成功规则

- P0协议/安全越界、错误运行字节、输出覆盖、query泄漏或launcher-wide确定性故障：仅停止精确run-owned进程并保留partial artifacts。
- 两个outer在prediction前出现同一确定性异常：停止后续dispatch；不重试、不覆盖。
- 不因中间性能停止。`fresh_run_retry=false`。
- 成功需要5个outer、210次session fit、全部不可覆盖预测闭合；随后才允许truth-last分析连续轨迹性能。

## Runner封口（2026-08-17）

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 启动：冻结的`nohup bash ./launch.sh`命令仅提交一次；首次健康核验时run-owned PID为0、GPU compute进程为0，无需执行进程终止。
- 技术触发：`smoke_gpu0.out`记录`ContinuousSessionPredictionError: registration wall hard gate failed`。这是技术失败指纹，不作性能解释、方法修复或重试。
- 仅限技术资源记录：路径绑定阶段为`smoke/leo_clear_weak/DA1_REG0`；fit receipt中`session_index=0`，receipt中的scene与schedule字段均为`ABSENT`。resource receipt记录`registration_wall_time_ns=267672490`、`registration_wall_hard_max_ns=300000000`、`registration_incremental_peak_working_set_bytes=7524352`、`registration_incremental_peak_hard_max_bytes=4194304`、`registration_peak_rss_bytes=1469804544`。
- 已保全的truth-free产物：`matrix_manifest=1`、`prepared_manifest=5`、`delta_receipt=5`、`COMMIT=5`、`NPZ=75`、`fit_audit=1`、`resource_audit=1`；`prediction_manifest=0`、`job_receipt=0`、`execution_receipt=0`。正式矩阵未形成可分析闭合。
- 取回：`E:/type10-7/local_artifacts/d92_e0_continuous_session_k10new5_20260817_v5/run_root`（180文件，canonical tree SHA256=`4e17d62bc2f54979b51ea012e9e6b1bd135c69e8c17c423556c303085256e8ce`）与`log_root`（3文件，SHA256=`423fd4a5b698becafe0677488e9633f5efdced3c2c6e0f38118c6faf721a29bf`）均与远端逐文件canonical hash清单匹配；其中包含runtime/source、output、release archive、launch、launch.out/err和日志。
- 未复制truth sidecar或receipt用于truth闭合，`content_read=false`；未运行analyzer，未读取accuracy、H、BA、floor或其他性能字段。`fresh_run_retry=false`，不可进入性能分析。
- 封口清理：本地`ssh.exe`、`scp.exe`及至N607 TCP22连接均为0。
