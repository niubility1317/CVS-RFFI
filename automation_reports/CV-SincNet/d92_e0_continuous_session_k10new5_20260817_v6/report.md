# D92 E0连续session实验发布报告

## 运行身份与状态

- 实验ID：`d92_e0_continuous_session_k10new5_20260817_v6`
- 方法：`D92_E0_CUMULATIVE_REPLAY_SESSION_V1`
- 目标：直接完成单类/少数类连续注册的冻结5-outer性能与资源实验。
- 当前状态：`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_PERFORMANCE_RESULT`
- 科学代码提交：`e5fcb515de750bfdfd129996c66892d2af003f68`
- 结论边界：`DEVELOPMENT_ONLY_CONTINUOUS_SESSION_SCREEN`。

## 本轮唯一运行控制修复

v5在truth-free smoke的注册资源检查中提前停止，因异常发生在资源收据发布前而没有形成可分析的session结果。v6保留用户批准的300ms/4MiB阈值，但将其作为truth-last分析中的资源裁决，不再作为启动期异常；因此即使注册资源不合格，也能完成预测并同时得到性能与资源结论。实时推理的全类独立决策、零query更新、state与`C×288`MAC校验均未放宽。连续session相关测试51项通过。

## 冻结矩阵

| 维度 | 冻结值 |
|---|---|
| receiver outer | `20-1`,`3-19`,`7-14`,`7-7`,`8-8` |
| scene | `leo_clear_weak`,`leo_low_elev_weak`,`leo_rain_weak` |
| seed / K | `713106` / `10` |
| 注册轨迹 | `batch_5`、`singleton_forward`、`singleton_reverse`、`chunk_2_2_1` |
| 规模 | 5 outer×3 scene×4 schedule；210次`DA1_REG1`注册 |
| 状态 | `DA1_REG0`、`DA1_REG1_S1..S5` |
| 注册资源裁决 | wall≤300ms、增量working-set≤4MiB；超限保留结果并在分析中`REJECT_RESOURCE` |
| 实时推理 | 每样本全注册类独立判决；truth/fit/update/selection/role/quota/global均false；state/MAC按类数闭合 |

## N607运行交接

- CWD：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v6`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_continuous_session_k10new5_20260817_v6`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU0先做truth-free smoke；正式5个outer固定GPU0–GPU4。

唯一启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v6 && nohup bash ./launch.sh >./launch.out 2>./launch.err </dev/null &
```

技术健康完成后才取回5份truth sidecar并由主代理运行truth-last分析；runner不读性能、不调参、不重试。协议/安全、query泄漏、错误运行字节、覆盖或确定性执行故障仍触发精确停机；资源超限本身不再是运行健康停机条件。

## Runner封口（2026-08-17）

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 启动：冻结`nohup bash ./launch.sh`命令仅提交一次。首次健康核验时run-owned PID=0、GPU compute=0，因此无需执行进程终止。
- 技术触发：`smoke_gpu0.out`记录`ContinuousSessionPredictionError: terminal schedule closure drift`。这是确定性执行故障；不作性能解释、不调参、不修复或重试。
- 资源门边界：本次停止并非wall/peak资源收据超限导致；未读取任何资源或性能数值作为停机决策。
- 已保全truth-free产物：`matrix_manifest=1`、`prepared_manifest=5`、`delta_receipt=5`、`COMMIT=5`、`fit_audit=21`、`resource_audit=21`；`prediction_manifest=0`、`job_receipt=0`、`execution_receipt=0`。正式prediction闭合不存在。
- 取回：`E:/type10-7/local_artifacts/d92_e0_continuous_session_k10new5_20260817_v6/run_root`（220文件）及`log_root`（3文件）已从对应远端根完整复制，文件计数与远端一致；包含runtime/source、output、release archive、launch、`launch.out`/`launch.err`和日志。
- 未复制truth sidecar或truth receipt，`content_read=false`；未运行analyzer，未读取accuracy、H、BA、floor或其他性能字段。`fresh_run_retry=false`，不可进入性能分析。
- 封口清理：本地`ssh.exe`、`scp.exe`及至N607 TCP22连接均为0。
