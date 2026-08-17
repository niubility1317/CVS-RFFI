# D92 E0连续session实验发布报告

## 运行身份与状态

- 实验ID：`d92_e0_continuous_session_k10new5_20260817_v1`
- 方法：`D92_E0_CUMULATIVE_REPLAY_SESSION_V1`
- 目标：验证D92 E0在固定DA1旧类状态下按单类或少数新类连续注册时的状态、预测、资源与终端等价性。
- 当前状态：`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_PERFORMANCE_RESULT`
- 代码HEAD：`ad483e5d`
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
| 资源门 | 注册wall target/hard max 150ms；增量working-set hard max 4MiB；query state/MAC保持类数等价 |

## 本地版本与验证

本发布只包含当前Git HEAD中的既有实现和冻结配置，不修改生产代码、配置、测试或设计文件。运行时归档仅收录预测/分析/矩阵入口及其必要Python闭包，不收录数据、checkpoint、runs结果或truth sidecar。

验证命令（`ssr-gpu`环境，串行）：

```text
MSYSTEM=MINGW64 conda run -n ssr-gpu python -m pytest -q tests/test_stage2_d92_continuous_session.py tests/test_stage2_d92_continuous_session_matrix.py tests/test_stage2_d92_continuous_session_prediction.py tests/test_stage2_d92_continuous_session_analysis.py tests/test_run_d92_e0_continuous_session.py tests/test_analyze_d92_e0_continuous_session.py
MSYSTEM=MINGW64 conda run -n ssr-gpu python -m py_compile <runtime Python files>
bash -n launch.sh
tar -tzf runtime/d92_e0_continuous_session_v1.tar.gz
```

本地验证结果与归档SHA写入`DELIVERY_MANIFEST.json`。本报告不读取或解释任何性能值。

## N607运行交接

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- 远端runs根：`/home/szu2070436088/2510044040/CV-SincNet/runs`
- 远端日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_continuous_session_k10new5_20260817_v1`
- 运行CWD：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v1`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 远端输出：`.../runs/d92_e0_continuous_session_k10new5_20260817_v1/output`
- GPU：GPU0先做真实checkpoint无truth smoke；正式5个outer job分别使用GPU0–GPU4并发。

交接顺序：解包runtime→`prepare`生成5-job manifest→`prepare-deltas`全部5 job→GPU0执行`smoke`（仅`batch_5`+`singleton_forward`）→正式5个`run-job`并发→只做技术状态计数。分析器和truth sidecar由后续主代理在预测封存后独立执行，本发布脚本不读取truth。

精确命令入口见同目录`launch.sh`。预期每个session目录包含`prediction_artifact.npz`、`fit_audit.json`、`resource_audit.json`、`execution_receipt.json`和`COMMIT.json`，每个job包含`prediction_manifest.json`及job receipt。

## 停止规则与风险

- P0协议/安全越界、错误checkout/输出覆盖、query泄漏或launcher-wide确定性故障：停止该run并保留partial artifacts，标记`NO_PERFORMANCE_RESULT`。
- 两个不同outer job在产生prediction前出现相同确定性异常fingerprint：停止后续dispatch，不重试、不覆盖。
- 不因中间准确率、H、BA、floor或遗忘值停止。
- `fresh_run_retry=false`；本run不可恢复续跑。
- 若资源receipt超过冻结门，保留技术artifact并由主代理判定，不读取性能作解释。

