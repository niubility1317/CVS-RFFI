# 同包 raw/H 软参照的源域错误诊断

状态：ANALYZED，同一 RXMean run 的既有六份 E50 source V logits 后续分析。**高置信 raw 不能保证其星地 H 派生物自身预测正确。当前 H 自参照的一致性没有独立的正确身份信息，因此不能单凭一致性下降宣布修正了身份错误。** 这不是对当前目标损伤的因果证明。

固定两方法 RXMean-U、RXMean-LU，各 seed 2026092701/02/03；原 V27000、同一文件的 raw＋low suburban/mid urban/low urban，共 648000 份已存在 logits。没有新增 F 前向、训练、源/目标预测、U TX 真值或 target 访问。源 V 标签只用于只读诊断，不能外推为实际 U 正确率。未跨设备拼接 raw/H；六份文件的 27000 个 ID、标签、RX/day 顺序全部一致。

以下限定 raw 的 max probability≥0.95，均值±样本标准差，单位 %。每 seed 原接受子集为 25870 至 25931 条（U）或 25874 至 25930 条（LU），不拿接受子集当全部 V。raw 判对→H 判错率以整个 raw 接受子集作分母。

| 方法 | H 场景 | H 准确率 | raw 判对→H 判错率 | H 高置信子集准确率 |
| --- | --- | ---: | ---: | ---: |
| RXMean-U | practical_low_suburban | 91.830602 ± 0.062044 | 8.052335 ± 0.075766 | 99.780615 ± 0.033597 |
| RXMean-U | practical_mid_urban | 75.517585 ± 0.238574 | 24.367917 ± 0.248219 | 99.806325 ± 0.058462 |
| RXMean-U | practical_low_urban | 61.116382 ± 0.445462 | 38.766543 ± 0.456526 | 99.767006 ± 0.040863 |
| RXMean-LU | practical_low_suburban | 91.842713 ± 0.063747 | 8.040231 ± 0.072211 | 99.775311 ± 0.033608 |
| RXMean-LU | practical_mid_urban | 75.510152 ± 0.273630 | 24.375357 ± 0.280170 | 99.794129 ± 0.068808 |
| RXMean-LU | practical_low_urban | 61.095754 ± 0.450852 | 38.787178 ± 0.458474 | 99.770394 ± 0.030459 |

RXMean-U 的 low urban 中，约 38.77% 从 raw 判对变为 H 判错，但 H 自身达到 0.95 且判错的比例仅约 0.0785%（分母仍是整个 raw 接受子集）。这说明多数错误 H 处于较低置信，并非大量高置信错标签；不能把 soft-target 约束直接等同于这些错 argmax 的硬监督。一致性可能保留有用的不确定性，也可能仅维持当前模糊/错误关系，实际作用须由固定正式结果验证。

这给 FastTrust 借鉴提供了具体问题：raw 可信度、H 证据强度、同包对应和已知 RX 可分别决定新增分支的监督类型。高置信 raw 的身份信息可以可信，但强 H 的唯一身份判决未必有证据；候选集合或选择性参照仍需独立检验。该分析没有据此更改已运行 Max 版的阈值、参照、剂量、seed 或预算，也没有证明一种新路由有效。

指标定义：原始/受损预测的 argmax 与 V 标签比较；高置信阈值固定 0.95；`KL(p_raw||p_H)` 及 clean soft target 对 H logits 的梯度范数 `||p_H−p_raw||` 仅为解析统计。后者不是当前 RXMean 一致性项的网络梯度，也不是实际 AdamW 更新。保存的原 H 是固定 V 模拟实现，不等于当前 U 的全部三实现缓存。

按 overall/TX/RX/day/RX-day，以及 all/raw_accepted/raw_rejected 三个范围，共 1620 个逐 seed 分层记录。见 [全部分层](per_seed_strata.csv)、[三 seed 汇总](three_seed_summary.csv) 和 [分析边界](analysis.json)。四项针对性数值/权限/三 seed 完整性检查通过。原 raw clean/六景失败结论保持，原全范围性能目标 active。
