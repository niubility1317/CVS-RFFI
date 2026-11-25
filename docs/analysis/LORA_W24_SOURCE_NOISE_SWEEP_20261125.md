# LoRa 原生 Full 源域去噪幅度诊断

运行 `20261125-diagnostic-lora-w24-noise-removal-source-s2026112559-r01`。固定原生 Full、星地增强 W24 E104，复用既有因子诊断的 300 个源 V 包、五台源接收机、物理顺序及信道随机流。每个 clean 和六景实收窗口固定六档，分别扣除仿真新增噪声的 0%、25%、50%、75%、90% 和 100% 幅度。中间读出为原无新增噪声分量加剩余噪声，信号分量与时标保持相同。0% 使用原实收输入，100% 使用原无新增噪声分量。该诊断使用隔离的精确分量，供估计识别需要的降噪幅度。候选训练、推理、校准和目标测试不消费这些分量。

N607 普通账号，单 GPU、8 个 CPU 信道工人，归档原生 W24 工作目录。入口为 `python -m experiments.pamf_lora_main.w24_factor_audit publish --noise-sweep --repo <checkout> --context <registered-LoRa-context> --ssh-config <direct-config> --output <new-release>`。不可变输出为 `runs/20261125-diagnostic-lora-w24-noise-removal-source-s2026112559-r01`。源 U 真权重及信道重放检查先于源 V。各景两端的盲分类必须与既有诊断逐包一致。全部预测固定后独立 CPU 才读取源 V 真值。零目标访问，零新训练，零选模。预测、端点或数据契约错误停止本运行并保留产物。预期 `smoke.json`、`predictions.npz`、`source-predictions.json`、`source-scores.json`。 [既有诊断](../../automation_reports/CV-SincNet/20261125-diagnostic-lora-w24-distortion-source-s2026112559-r01/report.md)
