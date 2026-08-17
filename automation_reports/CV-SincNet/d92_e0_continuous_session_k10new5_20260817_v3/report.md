# D92 E0连续session实验发布报告

## 运行身份与状态

- 实验ID：`d92_e0_continuous_session_k10new5_20260817_v3`
- 方法：`D92_E0_CUMULATIVE_REPLAY_SESSION_V1`
- 目标：验证D92 E0在固定DA1旧类状态下按单类或少数新类连续注册时的状态、预测、资源与终端等价性。
- 当前状态：`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_PERFORMANCE_RESULT`
- 代码HEAD：`b3c28e5a`
- 结论边界：本轮为`DEVELOPMENT_ONLY_CONTINUOUS_SESSION_SCREEN`，未连接truth、未产生性能结论。

## v2故障与v3唯一修复

v2已真实启动并在GPU0 smoke的首个`DA1_REG0`旧类基线资源检查处停止，异常为`registration peak hard gate failed`；没有进入任何`DA1_REG1`新类注册，也没有启动正式outer。v3只修正门控作用域：`DA1_REG0`仍记录wall/peak，但不占用连续注册session的300ms/4MiB预算；每个`DA1_REG1`session继续严格执行原资源门。方法、数据、矩阵、query路径和资源阈值均未变化。

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
| 资源门 | 仅`DA1_REG1`注册wall hard max 300ms、增量working-set hard max 4MiB；query state/MAC保持类数等价 |

## 本地版本与验证

v3运行时归档仅收录Git commit`b3c28e5a`的预测/分析/矩阵入口及必要Python闭包，不收录数据、checkpoint、run结果或truth sidecar。TDD先复现旧入口不支持基线免除门控，再以最小修复转绿；连续session聚焦回归49项通过，Python编译与diff检查通过。

## N607运行交接

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- 远端日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_continuous_session_k10new5_20260817_v3`
- 运行CWD：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v3`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：GPU0先做真实checkpoint无truth smoke；正式5个outer job分别使用GPU0–GPU4并发。

唯一detached启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v3 && nohup bash ./launch.sh >./launch.out 2>./launch.err </dev/null &
```

预期每个session目录包含`prediction_artifact.npz`、`fit_audit.json`、`resource_audit.json`、`execution_receipt.json`和`COMMIT.json`，每个job包含`prediction_manifest.json`及job receipt。分析器和truth sidecar由主代理在完整预测封存后独立运行，本发布脚本不读取truth。

## 停止规则与风险

- P0协议/安全越界、错误checkout/输出覆盖、query泄漏或launcher-wide确定性故障：停止该run并保留partial artifacts，标记`NO_PERFORMANCE_RESULT`。
- 两个不同outer job在产生prediction前出现相同确定性异常fingerprint：停止后续dispatch，不重试、不覆盖。
- 不因中间accuracy、H、BA、floor或遗忘值停止。
- `fresh_run_retry=false`；本run不可恢复续跑。
- `DA1_REG1`资源receipt超过300ms/4MiB时保留技术artifact并停止，不读取性能作解释。
