# LoRa 弱复谱保真重训的双路线目标子集

固定从随机初始化训练的第 40,000 步扩容 DURA、一份源 U 门限、原前端及末步 0.5 回写。源训练发布为 `20261125-phase1-lora-dura-balanced-transfer-source-s2026112991-r01`，提交 `35530da3ffe3d5d1634a30810dc600d7c809e08b`。候选的 Full 和 Residual 共用同一份模型、规则与冻结 Tri-set 种子 `2026092702`。 [源重训及损失设计](../20261125-phase1-lora-dura-balanced-transfer-source-s2026112991-r01/report.md)

调度器等待 40,000 步产物、真实权重检查和两路源 V 独立评分闭合，将完整源预测和评分作为参考随结果保存。源 V 正确数不参与目标启停或候选选择。独立真实权重检查使用源 U，一次完成。目标在两条已登记 practical 缓存中，分别取 15 台未见接收机的原盲序前 200 个包。两路线各 3,000 个物理包、七景、直通与候选各一行，合计 420 个决策数组。候选仅输入当前实收六窗 IQ。

冻结全部文件后，本机独立评分先核对每路线、每接收机、每场景的直通决策与原归档逐值一致。对照包括直通、两条路线各自的原冻结 DURA 半量、原扩容半量、`guard_half`，以及 Full 中已归档的复响应半量。之后以匹配物理顺序关联原始测试真值，逐场景和接收机报告正确数、同包纠错、改错、净变化及 1,600 次物理包分层配对重采样区间。三种子 30,000 包的正式验收矩阵单独保留。

N607 普通账号与已核实的项目 Python。命令 `python -m experiments.pamf_lora_main.dura_balanced_transfer_target publish --repo <checkout> --output <artifact-root>/release --context <runtime_source.json> --ssh-config <config> --training-receipt <source-release>/submit.json`。输出目录 `${N607_PROJECT}/runs/20261125-phase1-lora-dura-balanced-transfer-pilot-s2026092702-r01`，工作目录为对应的不可覆盖 `-release`。候选和旧对照的身份、测试物理前缀和两路线固定。健康源训练继续，测试开始后按实时可用 GPU 分配八个接收机分片。训练／权重身份不符、盲预测或原数据契约失败、输出未闭合时停止本链并保留已有产物。预期有双路线源参考、`smoke.json`、八份分片完成记录、`predictions/complete.json` 与独立 `scores.json`。评分入口为 `python -m tools.lora_dual_aperture_pilot_score {collect,score} --method balanced-transfer`。
