# Phase1 CLIC G bundle六fold v1预注册报告

## 状态与目标

- 实验ID：`phase1_clic_g_bundles_20260812_v1`。
- 当前状态：`LOCAL_VERIFIED / FRESH_REVIEW_PENDING / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
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

|候选|输入配置|预期工件|target LEO-weak|unknown rejection|域泛化|当前结论|
|---|---|---|---|---|---|---|
|F1G—F6G|training v5＋clean v4＋source-LEO v4|6份G bundle|本run不读target；下一阶段逐行LEO-weak预测|下一阶段由冻结source rule输出|下一阶段按三scene及分层审计|待运行|
