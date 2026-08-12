# Phase1 CLIC G bundle六fold串行v2预注册报告

## 状态与目标

- 实验ID：`phase1_clic_g_bundles_20260812_v2_serial`。
- 当前状态：`LOCAL_VERIFIED / REVIEW_ALLOW / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
- v1正式launch=1后，单F1G真实runtime smoke成功，但6个CPU worker并发启动时6／6在任何bundle前统一native segfault。v1已封存、不得重试。
- v2是按两轮release repair上限拆出的最小独立one-shot入口：同一commit模型修复、同一F1G—F6G矩阵、同一training v5／clean v4／source-LEO v4、同一CPU线程上限，仅把6fold并发改为单进程逐fold串行；不调参、不改方法、不选fold、不接触target或性能。

## 入口与技术门

- launcher：`code/scripts/launch_phase1_clic_g_bundles6_v2_serial_20260812.sh`，SHA-256=`5F0DF6E0CB9B3F888CFF9EB24AA0C4899FC7DC7AEB6046B5D89D7286AE51BACC`。每个fold启动一个G exporter后立即wait并记录exit code，成功才进入下一fold；任何非0立即停止后续fold并保留已完成工件。
- dry-run仍精确6行F1G—F6G，禁止C／target／truth／score／role／query／package参数为0；bash-n通过；串行结构及矩阵测试`2／2`通过。
- sample-rate真实同型修复、G reload／标量／一次重建多row聚焦`5／5`，postfreeze`140／140`；fresh G-only v1代码审查已`P0=0，P1=0，ALLOW`，v2仅减少并发。
- 输出：`runs/phase1_clic_g_bundles_20260812_v2_serial/F{1..6}G_CLIC12/g_deployment_bundle.zip`；日志：`logs/phase1_clic_g_bundles_20260812_v2_serial`；启动前必须不存在。CUDA禁用，OMP／MKL／OpenBLAS各2线程；formal launch唯一1次，retry=`NO`。

## 后续与结果边界

- 成功只做production verify/reload技术QA，不读target、不报告性能。6G闭合后，target prediction v1改为C取predictor artifacts v2、G取本串行v2，仍对同一IQ-only target LEO-weak package生成12份prediction。
- target LEO-weak、unknown rejection、scene／RX／class／day DG是后续独立scorer的强制共同输出；本G-only构建run仅提供冻结predictor，不产生这些性能指标。
- fresh独立复审结论：`P0=0，P1=0，ALLOW`。确认相对v1只有run ID／TSV和6并发→逐fold串行变化；每fold立即wait、记录exit、成功才下一fold；bash-n、dry-run6和专测`2／2`通过，科学矩阵与输入未漂移。

|候选|矩阵是否保留|并发变化|预期工件|target／unknown／DG|当前结论|
|---|---|---|---|---|---|
|F1G—F6G|是，6／6全部执行|6并发→逐fold串行|6份G bundle|本run不读取；后续12路prediction＋scorer统一报告|待运行|
