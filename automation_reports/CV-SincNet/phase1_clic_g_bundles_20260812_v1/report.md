# Phase1 CLIC G bundle六fold v1预注册报告

## 状态与目标

- 实验ID：`phase1_clic_g_bundles_20260812_v1`。
- 当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / FORMAL_INVOCATION=1 / RETRY=NO / NO_PERFORMANCE_RESULT`。
- 操作者：主控Codex；N607唯一runner：`Luna/max`。
- 目标：用training v5、clean v4和source-LEO v4为F1—F6的G臂各生成一份不可变deployment bundle；不重复已在predictor artifacts v2成功产生的6份C descriptor，不读取target、不计算性能。
- v2已封存为系统性技术失败，但6C descriptor／6C train config有效保留；G=0。真实checkpoint只读证据表明`sample_rate_hz=0.0,dataset=wisig,wisig_out_len=256`，而训练入口会把非正占位值规范化为WiSig 25MHz。旧bundle重建错误地封存了占位0。

## 最小修复与冻结矩阵

- 本地修复仅在`_runtime_rebuild_from_checkpoint`镜像训练入口既有规则：`sample_rate_hz<=0`时WiSig→25MHz、其他dataset→5MHz；正值原样保留，非数值／非有限仍fail-closed。模型state、架构、source policy、阈值、数据和fold均未改变。
- 真实G bundle同型RED在旧实现精确触发`sample_rate too low or min_band_hz too large`；修复后GREEN，并验证bundle内effective sample rate=`25000000.0`及真实reload forward成功。
- 聚焦G重建／reload／标量／一次重建多row测试`4／4`通过；postfreeze全量`140／140`通过；py_compile和diff-check通过。
- launcher：`code/scripts/launch_phase1_clic_g_bundles6_v1_20260812.sh`，SHA-256=`AF36EB5BAFC32D7975203C20DF2FC8885818EAFBAFA42BCE25E2C883E6670864`；bundle实现SHA-256=`F826163A2F55FAF7CE3C9CD7F856AFE4452EAA75E52BC1364C1EECC02541C982`。dry-run精确6行，只含F1G—F6G；训练=`phase1_clic12_20260812_v5`、clean=`phase1_clic_postfreeze_20260812_v4`、source-LEO=`phase1_clic_source_leo_20260812_v4`；禁止C／target／truth／score／role／query／package参数为0。
- 输出：`runs/phase1_clic_g_bundles_20260812_v1/F{1..6}G_CLIC12/g_deployment_bundle.zip`；日志：`logs/phase1_clic_g_bundles_20260812_v1`；启动前必须不存在。6个CPU worker、每worker线程2，CUDA禁用；正式launch唯一1次，retry=`NO`。

## 健康与后续

- 至少2fold在完整bundle前出现同一确定性异常即封存技术失败；不得按性能停止，不得远端修代码或重试。
- 成功QA只做production verifier重开、bundle member／SHA／local4／training config／zero-fit-update-threshold-selection检查，不打开target cache、不读性能。
- 6G bundle闭合后，目标prediction v1使用predictor artifacts v2中的C工件和本run的G工件，同一target confirmation v2 IQ-only package生成12份prediction；独立评分仍必须同时给target LEO-weak、unknown rejection和scene／RX／class／day域泛化。
- fresh独立审查结论：`P0=0，P1=0，ALLOW`。实际覆盖WiSig0→25MHz、其他dataset0→5MHz、正值原样、bad／NaN／±Inf拒绝；G聚焦`5／5`、postfreeze`140／140`、launcher测试`1／1`、bash-n和6行dry-run均通过。

|候选|输入配置|预期工件|target LEO-weak|unknown rejection|域泛化|当前结论|
|---|---|---|---|---|---|---|
|F1G—F6G|training v5＋clean v4＋source-LEO v4|预期6份G bundle，实际0/6|本run不读target；无prediction工件|未进入unknown rejection|未进入域泛化审计|STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE|

## N607落地、静态核验与正式运行证据

- 冻结commit：`9065e2036f1ddbd540191ca0e9b6275b8ad018c9`。`git archive`物理tar为267612160 bytes，SHA-256=`D81B5A83EF63CA0614DE213170BA7099DB5A56A3228C9739546E16BA4A0ACEE3`；未纳入共享`conversation_index/`。
- 唯一SCP恰1次，远端tar SHA/bytes与本地闭合。原子release：`/home/szu2070436088/2510044040/releases/phase1_clic_g_bundles_20260812_v1_9065e203`。launcher远端physical/canonical SHA均为`AF36EB5BAFC32D7975203C20DF2FC8885818EAFBAFA42BCE25E2C883E6670864`；exporter远端physical SHA=`FFF7B5111D6F82E437AD001AFA90CE8D5F3C749F3E14C0A6A6D772BC637C9638`，CRLF归一化canonical SHA=`F826163A2F55FAF7CE3C9CD7F856AFE4452EAA75E52BC1364C1EECC02541C982`。
- 静态核验：exporter`py_compile`、`--help`、launcher`bash -n`均PASS；`bash launcher --dry-run`精确6行（F1G—F6G），C/target/truth/score/role/query/package禁用参数计数为0。
- F1G只读checkpoint/terminal binding及临时runtime smoke PASS：checkpoint`sample_rate_hz=0.0`，effective runtime重建为25000000.0Hz；输出维度`z_id=160,z_dom=160,q_clic=4`。未写正式run工件，未读取target或性能。
- 正式命令唯一调用：
  `nohup bash /home/szu2070436088/2510044040/releases/phase1_clic_g_bundles_20260812_v1_9065e203/code/scripts/launch_phase1_clic_g_bundles6_v1_20260812.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_g_bundles_20260812_v1_outer.out 2>&1 &`
  `FORMAL_INVOCATION=1`，outer PID=`2768813`，`RETRY=NO`。

## 系统性技术停止证据

- 六个worker均在生成完整bundle前以同一确定性故障退出：launcher`line 90`的`Segmentation fault (core dumped)`；因此满足“至少2fold同一异常”的预注册systemic-stop规则，停止本run且不重试、不修远端、不删除证据。
- outer全文（6行，SHA-256=`7C936F35E5D458252391A95E6C6938592D1D2AC32336EEABB651FAB69ABDF563`）：

```text
/home/szu2070436088/2510044040/releases/phase1_clic_g_bundles_20260812_v1_9065e203/code/scripts/launch_phase1_clic_g_bundles6_v1_20260812.sh: line 90: 2768825 Segmentation fault      (core dumped) PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${G_CMD[@]}" > "${log_path}" 2>&1
/home/szu2070436088/2510044040/releases/phase1_clic_g_bundles_20260812_v1_9065e203/code/scripts/launch_phase1_clic_g_bundles6_v1_20260812.sh: line 90: 2768831 Segmentation fault      (core dumped) PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${G_CMD[@]}" > "${log_path}" 2>&1
/home/szu2070436088/2510044040/releases/phase1_clic_g_bundles_20260812_v1_9065e203/code/scripts/launch_phase1_clic_g_bundles6_v1_20260812.sh: line 90: 2768833 Segmentation fault      (core dumped) PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${G_CMD[@]}" > "${log_path}" 2>&1
/home/szu2070436088/2510044040/releases/phase1_clic_g_bundles_20260812_v1_9065e203/code/scripts/launch_phase1_clic_g_bundles6_v1_20260812.sh: line 90: 2768835 Segmentation fault      (core dumped) PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${G_CMD[@]}" > "${log_path}" 2>&1
/home/szu2070436088/2510044040/releases/phase1_clic_g_bundles_20260812_v1_9065e203/code/scripts/launch_phase1_clic_g_bundles6_v1_20260812.sh: line 90: 2768837 Segmentation fault      (core dumped) PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${G_CMD[@]}" > "${log_path}" 2>&1
/home/szu2070436088/2510044040/releases/phase1_clic_g_bundles_20260812_v1_9065e203/code/scripts/launch_phase1_clic_g_bundles6_v1_20260812.sh: line 90: 2768841 Segmentation fault      (core dumped) PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${G_CMD[@]}" > "${log_path}" 2>&1
```

- `F1G_CLIC12.out`—`F6G_CLIC12.out`均0 bytes，SHA-256均为`E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`。PID表`logs/phase1_clic_g_bundles_20260812_v1/pids_g_bundles6.tsv`记录6行：`2768825,2768831,2768833,2768835,2768837,2768841`，加outer PID`2768813`均已退出。run限定目录存在F1—F6空目录，`g_deployment_bundle.zip`计数为0；限定目录内未发现core文件（未打开任何core）。
- 停止后GPU0—7均`0%/1MiB`；本地SSH/SCP进程与TCP22连接均为0。未读取dmesg、target、IQ truth或任何性能字段。

## 最终封存

- `ARTIFACTS_COMPLETE`未达到；本run最终状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，G bundle工件0/6。下一步若需修复segfault，必须使用全新run ID并重新完成本地审查；本run不可重启或重标记为性能结果。
