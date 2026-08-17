# D92 E0连续session实验发布报告

## 运行身份与状态

- 实验ID：`d92_e0_continuous_session_k10new5_20260817_v4`
- 方法：`D92_E0_CUMULATIVE_REPLAY_SESSION_V1`
- 目标：验证D92 E0按单类或少数新类连续注册时的状态、性能、资源与终端等价性。
- 当前状态：`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_PERFORMANCE_RESULT`
- 代码HEAD：`c909135f`
- 结论边界：`DEVELOPMENT_ONLY_CONTINUOUS_SESSION_SCREEN`。

## 两轮真实启动故障与冻结修复

- v2在首个`DA1_REG0`旧类基线处错误执行新类注册4MiB门而停止；`b3c28e5a`将300ms/4MiB硬门限定到`DA1_REG1`session，基线仍完整记录资源。
- v3把合法`stage2b(before)→stage2c(after)`转换误判为stage漂移；`c909135f`固定after apply必须是`stage2c/after/apply_only`，同时继续绑定receiver、seed、K、checkpoint、runtime、method lock与注册类集合。

本v4不修改方法、阈值、数据、矩阵、query路径或性能裁决。连续session聚焦回归50项通过。

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
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_continuous_session_k10new5_20260817_v4`
- 运行CWD：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v4`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：GPU0先做truth-free smoke；正式5个outer分别使用GPU0–GPU4并发。

唯一detached启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v4 && nohup bash ./launch.sh >./launch.out 2>./launch.err </dev/null &
```

完整预测封存后，主代理再独立取truth并运行truth-last分析。本发布脚本和runner不读取性能。

## 停止规则

- P0协议/安全越界、错误checkout、输出覆盖、query泄漏或launcher-wide确定性故障：停止精确run-owned进程并保留partial artifacts。
- 两个outer在prediction前出现同一确定性异常：停止后续dispatch；不重试、不覆盖。
- 不因中间性能停止。`fresh_run_retry=false`。

## Runner失败态封口（2026-08-17）

- 终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 触发：GPU0 truth-free smoke出现确定性技术异常`ContinuousSessionPredictionError: fixed apply query token/IQ identity drift`。
- formal dispatch：未启动；job receipt和prediction manifest计数均为0。
- 接管runner未执行新的detached launch、未重试、未修改方法、矩阵、配置或远端已有artifact。
- 封口核验：无本run绑定PID，GPU compute进程为空；异常触发后仅保全现有run/log内容。
- 完整性：远端与本地runtime archive SHA-256均为`dec0db85258945b66b465d695bddfee33443a7d385ba86d1863b7ce3c8ef5b12`，launch SHA-256均为`b952ada7c94ae6158daeced54082fc1bb6c86ceafb8492b311b38d1f3984f211`；归档31条目且无不安全路径；两端`bash -n launch.sh`通过。
- 取回：`E:/type10-7/local_artifacts/d92_e0_continuous_session_k10new5_20260817_v4/run_root`和`.../log_root`中的相对路径与文件字节数分别和远端一致。
- 受限诊断：仅检查指定source job中`offline/predictor/before/apply_only_staging/manifest.json`与`offline/predictor/after/apply_only_staging/manifest.json`；两者均ABSENT，两个精确root的maxdepth1列表均为空，因此未复制apply manifest且未扩大扫描。此前创建的非目标delta manifest重复副本保留为未读取、不可用的本地冗余件，等待主代理处置。
- 边界：未复制或读取truth sidecar，未读取NPZ/query/truth，未运行analyzer，未读取或解释任何性能值。
