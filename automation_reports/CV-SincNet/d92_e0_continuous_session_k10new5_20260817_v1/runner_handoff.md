# Runner Handoff

- Run ID：`d92_e0_continuous_session_k10new5_20260817_v1`
- Runner：`Luna/max`，sole N607 launch owner
- Final status：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- Performance result：`false`
- Fresh-run retry：`false`

## Trigger

`prepare`与`prepare-deltas`完成后，GPU0真实checkpoint无truth smoke在prediction前抛出：

```text
cvsrffi.stage2_d92_continuous_session_prediction.ContinuousSessionPredictionError: registration wall hard gate failed
```

Launcher按`set -e`停止，formal 5 outer未dispatch。未重试、未运行analyzer、未读取性能指标。

## Remote evidence

- Remote run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v1`
- Remote log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_continuous_session_k10new5_20260817_v1`
- Archive landing：`runs/d92_e0_continuous_session_k10new5_20260817_v1/release/d92_e0_continuous_session_v1.tar.gz`，`171172`B，SHA256=`1d8a91e5c7a1332c44289915de6d082ce22743cf155f13f461eb6230aa1da4b8`
- Launch landing：`runs/d92_e0_continuous_session_k10new5_20260817_v1/launch.sh`，`2294`B，SHA256=`cf6af9203533d26a143e2b3713c49971948b827ea180efe8c61fdcd9789142f7`
- Remote `bash -n`：`PASS`

## Technical counts

| Item | Count/status |
|---|---:|
| prepare jobs / session fits | `5 / 210` |
| delta roots | `5` |
| formal outer launched / completed | `0 / 0` |
| formal prediction manifests / job receipts | `0 / 0` |
| formal state artifact sets | `0` |
| partial output files / bytes | `116 / 2154890` |
| partial log files / bytes | `3 / 3177` |
| runtime snapshot files / bytes | `58 / 1475530` |
| truth sidecars retrieved | `0` |

## Cleanup and retrieval

- Post-stop run-owned PIDs：`[]`；post-stop GPU compute apps：`[]`。
- SSH/SCP：短连接已退出，无持久连接/端口转发/后台监控。
- Local partial retrieval：`E:/type10-7/local_artifacts/d92_e0_continuous_session_k10new5_20260817_v1/`
- Retrieved subtrees：`runtime/`、`output/`、`logs/`、`launch.out`、`launch.err`、`launch.sh`、`release/`
- Remote artifacts remain preserved; no delete/overwrite.

## Frozen references

- Runtime science/integration commit：`ad483e5d`
- Release commit：`1c3412f7`
- Report command commit：`7cf13cf7`
