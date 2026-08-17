# D92 E0连续session v4 runner handoff

- run_id：`d92_e0_continuous_session_k10new5_20260817_v4`
- runner role：唯一N607 runner
- terminal_state：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- trigger：GPU0 truth-free smoke出现`ContinuousSessionPredictionError: fixed apply query token/IQ identity drift`。
- formal_dispatch：`NOT_STARTED`
- retry_authority：`false`

## 封口证据

- 远端run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v4`
- 远端log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_continuous_session_k10new5_20260817_v4`
- 末次健康核验：无本run绑定PID；GPU compute进程为空；formal job receipt和prediction manifest均为0。
- runtime archive SHA-256：`dec0db85258945b66b465d695bddfee33443a7d385ba86d1863b7ce3c8ef5b12`
- launch SHA-256：`b952ada7c94ae6158daeced54082fc1bb6c86ceafb8492b311b38d1f3984f211`
- 归档31条目且无不安全路径；远端与本地`bash -n launch.sh`均通过。
- 本地取回：`E:/type10-7/local_artifacts/d92_e0_continuous_session_k10new5_20260817_v4/run_root`和`.../log_root`；两份清单的相对路径及字节数均与远端一致。
- 受限诊断：指定source job的`offline/predictor/before/apply_only_staging/manifest.json`与`offline/predictor/after/apply_only_staging/manifest.json`均ABSENT；两个精确root的maxdepth1列表均为空，未复制apply manifest且未扩大扫描。非目标delta manifest重复副本保留为未读取、不可用的本地冗余件，等待主代理处置。

## 不做事项

- 未执行第二次launch，未重试，未覆盖或删除artifact。
- 未复制或读取truth sidecar，`content_read=false`。
- 未读取NPZ/query/truth，未运行analyzer，未读取或解释任何性能指标。
