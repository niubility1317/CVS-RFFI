# WiSig 实际恢复波形与源域识别归因

复用已验证的 360 个源 V 物理包，覆盖 5 台接收机与 3 天，每层 24 包。两路线保留各自的 clean 和六种常规 practical。相同 256 点实收窗分别做直通、原冻结 Graph-RR R69，以及原居中 FIR 的粗恢复、第一次 MAP 和末步读出。R69 和居中 FIR 每包均运行完整恢复。冻结种子 `2026092701` 的 Main split Tri-SIG-TIME 对各读出分类，逐包复现原 1,800 包源域预测中相同物理 ID 的类别。

N607 普通账号，沿用 `20261125-phase1-wisig-native-fir-source-m4-r01` 发布工作目录和环境，一张有资源余量的 GPU。入口为 `python -m experiments.pamf_unrolled_restore.wisig_waveform_audit publish --repo . --context <源域上下文> --ssh-config <直连配置> --output <发布目录>`。新输出根为 `runs/20261125-diagnostic-wisig-waveform-source-s2026112501-r01`。真实权重检查先运行 4 个合法源 U 包。若来源、模型输出、范围、既有决策或冻结覆盖失败，停止下游并保留产物。

两路线七景均冻结后，由独立 CPU scorer 读源 V 标签以及隔离重放的无新增噪声参考。在实收坐标和尺度核对同包波形，报告 0–4、4–8 和 8–12.5 MHz 的复数 NMSE、幅度 NMSE 和参考能量加权相位弦损失，并给出各景、各接收机的正确计数和相对直通改对／改错。该参考保留了无新增噪声时的信道作用，频带指标衡量相对该接收波形的保真。预测进程仅可见已登记的实收 IQ，目标读取、训练和校准均为零。预期产物为 `smoke.json`、14 份冻结读出、`waveforms-frozen.json` 和 `waveform-scores.json`。 [预登记](experiment.json) · [原源域识别面板](../../../docs/analysis/WISIG_NATIVE_FIR_SOURCE_RESULTS_20261125.md)
