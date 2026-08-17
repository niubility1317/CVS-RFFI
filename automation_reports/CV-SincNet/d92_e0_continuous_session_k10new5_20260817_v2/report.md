# D92 E0连续session实验发布报告

## 运行身份与状态

- 实验ID：`d92_e0_continuous_session_k10new5_20260817_v2`
- 方法：`D92_E0_CUMULATIVE_REPLAY_SESSION_V1`
- 目标：验证D92 E0在固定DA1旧类状态下按单类或少数新类连续注册时的状态、预测、资源与终端等价性。
- 当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 代码HEAD：`8b41e7f3`
- 结论边界：本轮为`DEVELOPMENT_ONLY_CONTINUOUS_SESSION_SCREEN`，未连接truth、未产生性能结论。

## 冻结矩阵

| 维度 | 冻结值 |
|---|---|
| receiver outer | `20-1`,`3-19`,`7-14`,`7-7`,`8-8` |
| scene | `leo_clear_weak`,`leo_low_elev_weak`,`leo_rain_weak` |
| seed / K | `713106` / `10` |
| 注册轨迹 | `batch_5`、`singleton_forward`、`singleton_reverse`、`chunk_2_2_1` |
| 规模 | 5 outer×3 scene×4 schedule=60 trajectories；session fit=210 |
| 状态命名 | `DA1_REG0`、`DA1_REG1_S1..S5` |
| query策略 | `per_sample_all_registered_classes`；truth/fit/update/selection/role/quota/global-reassignment均为false |
| 资源门 | 注册wall target/hard max 300ms；增量working-set hard max 4MiB；query state/MAC保持类数等价 |

## 本地版本与验证

本发布只包含当前Git HEAD中的既有实现和冻结配置，不修改生产代码、配置、测试或设计文件。运行时归档仅收录预测/分析/矩阵入口及其必要Python闭包，不收录数据、checkpoint、runs结果或truth sidecar。

验证命令（`ssr-gpu`环境，串行）：

```text
MSYSTEM=MINGW64 conda run -n ssr-gpu python -m pytest -q tests/test_stage2_d92_continuous_session.py tests/test_stage2_d92_continuous_session_matrix.py tests/test_stage2_d92_continuous_session_prediction.py tests/test_stage2_d92_continuous_session_analysis.py tests/test_run_d92_e0_continuous_session.py tests/test_analyze_d92_e0_continuous_session.py
MSYSTEM=MINGW64 conda run -n ssr-gpu python -m py_compile <runtime Python files>
bash -n launch.sh
tar -tzf runtime/d92_e0_continuous_session_v2.tar.gz
```

本地验证结果与归档SHA写入`DELIVERY_MANIFEST.json`。本报告不读取或解释任何性能值。

## N607运行交接

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- 远端runs根：`/home/szu2070436088/2510044040/CV-SincNet/runs`
- 远端日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_continuous_session_k10new5_20260817_v2`
- 运行CWD：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v2`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 远端输出：`.../runs/d92_e0_continuous_session_k10new5_20260817_v2/output`
- GPU：GPU0先做真实checkpoint无truth smoke；正式5个outer job分别使用GPU0–GPU4并发。

交接顺序：解包runtime→`prepare`生成5-job manifest→`prepare-deltas`全部5 job→GPU0执行`smoke`（仅`batch_5`+`singleton_forward`）→正式5个`run-job`并发→只做技术状态计数。分析器和truth sidecar由后续主代理在预测封存后独立执行，本发布脚本不读取truth。

唯一detached启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v2 && nohup bash ./launch.sh >./launch.out 2>./launch.err </dev/null &
```

预期每个session目录包含`prediction_artifact.npz`、`fit_audit.json`、`resource_audit.json`、`execution_receipt.json`和`COMMIT.json`，每个job包含`prediction_manifest.json`及job receipt。

## 停止规则与风险

- P0协议/安全越界、错误checkout/输出覆盖、query泄漏或launcher-wide确定性故障：停止该run并保留partial artifacts，标记`NO_PERFORMANCE_RESULT`。
- 两个不同outer job在产生prediction前出现相同确定性异常fingerprint：停止后续dispatch，不重试、不覆盖。
- 不因中间准确率、H、BA、floor或遗忘值停止。
- `fresh_run_retry=false`；本run不可恢复续跑。
- 若资源receipt超过冻结门，保留技术artifact并由主代理判定，不读取性能作解释。

## Runner技术交接（2026-08-17）

- runner：`Luna/max`，唯一N607 launch owner；未修改方法、配置、测试或矩阵。
- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 停止原因：GPU0真实checkpoint无truth smoke在产生prediction前触发`ContinuousSessionPredictionError: registration wall hard gate failed`；launcher的`set -e`按冻结顺序停止，未进入formal outer dispatch。未重试、未调参、未读取accuracy、H、BA、floor或forgetting。

### Preflight与落地证据

| 项目 | 结果 |
|---|---|
| 直连账户/主机 | `N607`普通账户` szu2070436088` / `dell-DSS8440`，直连`VERIFIED` |
| 项目根 | `/home/szu2070436088/2510044040/CV-SincNet`，`VERIFIED` |
| GPU0–4启动前占用 | 每卡`0%`利用率、`1MiB/24576MiB`，满足占用门 |
| 同RUN_ID进程/路径 | RUN_ROOT、LOG_ROOT和本地retrieval启动前均不存在；无同run进程 |
| archive | `171009`B，SHA256=`854fb1f6b4f0139230ef858c2b8ce53433df176b4805922cbefc2898214c4903` |
| launch.sh | `2294`B，SHA256=`284422bc2224958af3bcfe7e42588208ab6a09de9ba86725bcf3b678a17f951e` |
| 远端验证 | archive/launch大小与SHA匹配，`bash -n=PASS` |

### 启动与技术计数

唯一detached命令已提交一次：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v2 && nohup bash ./launch.sh >./launch.out 2>./launch.err </dev/null &
```

| 阶段/产物 | 状态或计数 |
|---|---:|
| `prepare` | `COMPLETED`，5 jobs，210 session fits |
| `prepare-deltas` | `COMPLETED`，5 delta roots |
| GPU0 smoke | `FAILED`，registration wall hard gate，未产生prediction |
| formal outer启动/完成 | `0/0` |
| formal prediction manifests/job receipts | `0/0` |
| formal state artifact sets | `0`（预期225，未进入formal） |
| partial output | `116` files / `2154890`B |
| partial logs | `3` files / `3177`B |
| runtime snapshot | `58` files / `1475530`B |
| truth sidecars | `0`（formal未启动，未取） |
| analyzer | 未运行 |

### 清理与retrieval

- 停止后按固定run-root扫描：run-owned进程`NONE`；`nvidia-smi`计算进程为空。
- 本地SSH/SCP短连接均已退出，未保留`ssh.exe`、连接、端口转发或后台监控。
- partial artifacts已取回：`E:/type10-7/local_artifacts/d92_e0_continuous_session_k10new5_20260817_v2/`，包含`runtime/`、`output/`、`logs/`、`launch.out`、`launch.err`、`launch.sh`和远端`release/`副本。
- 远端run/log/output/runtime保留原状；未删除、覆盖或清理任何远端产物。
- 结构化交接：同目录`runner_handoff.md`与`runner_handoff.json`。

本run不产生性能结果；`fresh_run_retry=false`，后续如需修复必须由主代理在本地完成修复、聚焦验证并创建新的不可覆盖RUN_ID。
