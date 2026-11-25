# LoRa W24 实际恢复波形的双路线频带归因

本次诊断复用已登记 `20261125-diagnostic-lora-w24-readout-sweep-source-s2026112559-r01` 的 300 个源 V 物理包、五台接收机、原生 Full 和 Residual 信道设置，以及各自 clean 和六景。固定 40,000 步的已冻结 W24 DURA，种子 `2026112991`，对每包实际执行一次完整恢复。按同一实收中心 RMS 保存实收、半量和全量三个中心窗读出。识别证据引用原先同包、同 W24 E104 识别器的独立源评分。

两路预测全部冻结后，独立 CPU 进程重放隔离的无新增噪声参考，逐点校验实收视图。用同一时间坐标和归一化尺度测量居中 384 点窗口的复数 NMSE、幅度 NMSE 和信号能量加权的相位弦损失。固定频带为绝对频率 0 至 100、100 至 160、160 至 192 kHz，端点归入后一带，最后一带含 192 kHz。报告总体及去除八条 chirp 共同均值后的逐包误差中位数与相对实收的配对差。原生边缘读出沿用实测 IQ。评分参考仅用于隔离归因。

N607 普通账号，沿用冻结 W24 发布运行时和工作目录，单个有余量的 GPU 和八个 CPU 信道工人。入口为 `python -m experiments.pamf_lora_main.w24_waveform_audit publish --repo . --context <源域运行上下文> --ssh-config <直连配置> --output <新发布产物目录>`。使用独立输出根 `runs/20261125-diagnostic-lora-w24-waveform-source-s2026112559-r01`。首先在合法源 U 执行真实权重的无标签检查。识别面板物理 ID、来源、逐点接收 IQ、冻结覆盖或数值不匹配即停止本诊断下游并保留产物。预期输出为 `smoke.json`、两条路线的七景波形、`waveforms-frozen.json` 和 `waveform-scores.json`。 [执行登记](experiment.json) · [同包识别得分](../../../docs/analysis/LORA_W24_READOUT_SWEEP_20261125.md)

提交 `83c91f57659b913e22afc2951a11b34c378b11ec` 已在 N607 核实发布和真实权重检查。两路线的 14 个波形视图全部冻结。首轮评分在 clean 逐点核对处失败。旧版 SciPy 重采样生成 complex128，而预测按原 W24 合同转为 complex64。按原合同 cast 后，300 个实收 clean 中心窗与已冻结文件逐点一致。原运行及产物保留，修正后由独立输出根继续评分。 [评分恢复](../20261125-diagnostic-lora-w24-waveform-source-s2026112559-r02/report.md)
