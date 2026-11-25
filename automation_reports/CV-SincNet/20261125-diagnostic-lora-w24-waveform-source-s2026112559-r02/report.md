# LoRa W24 冻结波形的独立评分

复用上次已经完成且保留的两路线七景、300 个同包中心 IQ 波形。在评分端沿用固定 W24 输入的 float32 实部和虚部表示复核实收中心窗。参考信号仍按原实收中心 RMS 归一化。保持原频带、误差定义及冻结识别对照，在新的输出根只运行 CPU 评分。 [原运行、诊断矩阵及原因](../20261125-diagnostic-lora-w24-waveform-source-s2026112559-r01/report.md)

入口为 `python -m experiments.pamf_lora_main.w24_waveform_audit publish --rescore-frozen --repo . --context <源域运行上下文> --ssh-config <直连配置> --output <新发布产物目录>`。N607 普通账号，原 W24 发布工作目录，八个 CPU 工人，本次独立输出根为 `runs/20261125-diagnostic-lora-w24-waveform-source-s2026112559-r02`。验证旧波形冻结覆盖及已登记物理包顺序，评分前逐景逐点核对实收 IQ。失败时保留两次运行产物。期望 `state.json`、`score-source.log` 和 `waveform-scores.json`。 [执行登记](experiment.json)
