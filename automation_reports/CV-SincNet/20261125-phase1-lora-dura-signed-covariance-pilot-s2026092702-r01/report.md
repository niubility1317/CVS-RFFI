# LoRa 扩容半量 DURA 分侧协方差门控消融

固定原扩容 DURA 从零训练的第 40,000 步 `expanded` 权重，训练提交 `3de50180aaaa1179587a753d4f6dde201fc989c6`。对照沿用原外带门控；消融仅替换是否恢复的判据，两者共用接收坐标、观测方差、两阶段求解、半量输出和冻结 Tri-set `2026092702`。损失、权重及训练预算在两个读出中完全匹配。原生 W24 的独立重训及测试另见其自己的登记。 [消融入口](../../../experiments/pamf_lora_main/dura_signed_covariance_target.py) · [权重来源](../20261124-phase1-lora-dura-capacity-source-m4-r01/report.md)

每包对六条去扫频 chirp 的 Hann 外带复谱按正负频侧分别形成协方差。分别取最小三个特征值的均值开方，再取较大值，用于门控。阈值为源 U 的 `0.013684820942580701`，在 2,500 个 clean 包的两种采样率上一次冻结，两个 practical 路线共用。门控值不覆盖网络内的方差估计。输入是当包实收 IQ，六窗到齐后计算，无附加参数。源 V 的覆盖证据中，与另外一个共享外带统计相比，Full 增加 1,818 个激活，clean 零激活。本轮识别对照用真正的扩容 DURA 原门控。 [冻结门限和源覆盖](../20261125-phase1-lora-dura-dual-aperture-s2026112991-r01/report.md)

真实权重无 query 检查后，依次冻结 900 包源 V 的 clean 和六景双路线预测，独立评分。源结果作为参考。固定目标矩阵为同一个 Tri-set 种子和 15 台未见接收机。每台沿用登记物理顺序前 200 包，全程使用同一次登记的 Full 和 Residual IQ。目标预测固定为直通和新门控两臂，每路线各有 3,000 个物理包、七个视图。全量 420 个预测数组冻结后，在独立本机评分端逐值核对直通，并连接同包原 DURA 半量、扩容半量及既有对照的归档预测，再关联真实标签。报告两路线的六景均值、低强城区、clean 正确数、逐景逐机结果、纠错及改错和按接收机分层的物理包配对区间。采集 B1 门控旁路与恢复用时。 [独立评分入口](../../../tools/lora_dual_aperture_pilot_score.py)

N607 普通账号，按实际空闲容量分配预测 GPU。发布点为已推送提交，远程运行根为 `${N607_PROJECT}/runs/20261125-phase1-lora-dura-signed-covariance-pilot-s2026092702-r01`，工作目录为同名 `-release`。运行 `python -m experiments.pamf_lora_main.dura_signed_covariance_target publish --repo <checkout> --output <artifact-root>/release --context <runtime_source.json> --ssh-config <config>`。源契约、真权重检查、有限性或预测矩阵不闭合时停止此运行并保留产物。预期产物为 `smoke.json`、两套源预测和独立评分、`predictions/complete.json`、每个分片完整记录与本机 `readout/scores.json`。本次单种子固定子集比较独立门控贡献，原三种子正式验收仍按完整目标登记。
