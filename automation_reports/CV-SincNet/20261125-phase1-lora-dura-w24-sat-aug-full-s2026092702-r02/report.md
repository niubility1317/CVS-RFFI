# LoRa 星地增强 W24 与重训 DURA 配对复测 r02

r01 在源 U 真实权重检查中遇到 Sinc 缓存张量版本错误，未写出目标预测。r02 以 `no_grad` 构建和运行识别器，并显式绑定原归档的可变宽度描述符。权重仍为星地增强 W24 E104 和重训 W24 DURA 第 40,000 步。沿用 r01 固定原生视图，15 台接收机、30,000 包及全部七景，直通和旧版半量复响应均维持原归档配对。 [原矩阵和负结果](../20261125-phase1-lora-dura-w24-sat-aug-full-s2026092702-r01/report.md)

从当前工作树以 `python -m experiments.pamf_lora_main.dura_w24_augmented_target publish` 绑定同一训练发布记录。N607 使用本次 r02 不可覆盖目录及空闲优先的最多四张 GPU，源 U 检查通过后才分片预测。直通逐包对齐和矩阵闭合失败时保留故障产物。预期 105 个双行预测数组、四个分片记录和 `smoke.json`。全量冻结以后由 `tools.lora_dura_w24_score score --augmented-reference <frozen-archive-predictions>` 独立评分。
