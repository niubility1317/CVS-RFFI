# LoRa DURA 保留共享非载波形状

完整 12000 包源 V 分量对照中，只改六条 chirp 的共享非载波形状净错化 102 次。保留共享形状，只用载波与每条 chirp 特有的修正净纠正 183 次，完整固定 DURA 为 137 次。该结果登记为待独立目标验证的方向。

沿用完整 1 MHz 对齐 LoRa 的源与盲目标合同、已经冻结的三个 LoRa G 及源 IQ 旁路。三种子分别连接 `20261122-phase1-lora-extended-supervision-m12-r01` 中同种子第 168 轮自适应 Tri。四路依次为直接读出、原固定 DURA、保留共享非载波形状并修载波及单 chirp 残差、仅修单 chirp 非载波残差。后三路共用同一个 DURA 提案及判断，投影后实际修正的 RMS 上限设为该包原恢复量的 RMS。共享非载波分量在对齐相位、去啁啾和单 chirp RMS 归一化的坐标中定义。旁路包逐点复制。

每种子每路固定 15 个未见接收机 × 2000 包，全部七景，共 84 个评分单元。预测固定后独立评分，报告 clean、六景、逐接收机与 TX、净纠错和相对原固定恢复。源波形诊断与真实 checkpoint 无目标 smoke 不读取身份信息。恢复器、识别器和门控全部冻结，目标真值不用于改配方或重排已登记结果。

N607 普通账户，沿用现有 LoRa Python。发布命令 `python -m experiments.pamf_lora_main.full_dura_publish --shape`，传入已验证数据、原 G 和第 168 轮 Tri 的 release receipt，实际 Git commit 写入 release。CWD 为 `CV-SincNet/releases/20261122-phase1-lora-dura-preserve-shape-m9-r01-release`，输出为 `CV-SincNet/runs/20261122-phase1-lora-dura-preserve-shape-m9-r01`。GPU0–6 每卡最多一个本 run worker。检查真实源权重、源 clean 旁路与 practical 活跃输出；源契约不匹配、非有限数、旁路改动与盲预测不完整均为技术停止条件。预计产物为 `smoke.json`、逐 seed `source_wave`、冻结 `predictions` 与独立 `scores.json`。
