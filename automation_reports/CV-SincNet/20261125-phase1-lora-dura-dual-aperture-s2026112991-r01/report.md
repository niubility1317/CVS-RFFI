# LoRa 双采样率 DURA 源训练

一份新权重直接处理已登记 Full 和 Residual 中 `6×2×256` 的实收 IQ，也接受有真实首尾的 `8×2×432` W24 输入。两种输入各在自己的接收采样网格上对齐。使用共用中央六条 chirp 的 72–112 kHz Hann 外带量标定唯一门限，并将其按采样率换算为各自网格的单点观测方差。边界缺失时，在条件通道中提供明确的不可用标记，数据一致性求解仅作用于实收中心。宽输入保留原生首尾。源域结果已经显示共享带数值可对齐，同时存在 RTL 留出 clean 误触发。本轮把同一门控纳入原定评估，干净正确数由冻结后同包测试判定。 [模型与统一入口](../../../experiments/pamf_lora_main/dura_apertures.py) · [门控源证据](../20261125-phase1-lora-dura-w24-io-s2026092702-r01/report.md)

按原 W24 预算从零训练 40,000 步，种子 `2026112991`。拟合、校准、留出分别为同合法盲 U 角色的 20,000、2,500、500 包。每个 batch 的 16 个源包分为两个等大的采样网格组。两组分别经原生捕获速率上的通用扰动得到输入，并共用原两阶段 DURA 的权重、优化器和半量读出。在每组保留 1 个原样包，匹配既有两包的身份保持损失预算。一次 AdamW 联合更新粗恢复、两步 CG 的观测估计和后续细化。初始学习率 `2e-4`，余弦下降至 `2e-5`，总梯度范数上限 1。锁定第 40,000 步。拟合、校准、留出波形检验均无 TX、识别器、源 V、目标或仿真轨迹读取。 [训练入口](../../../experiments/pamf_lora_main/dura_dual_aperture_train.py)

N607 普通账号和空闲优先 GPU 策略。单次真实源 IQ 冒烟检查通过后进入完整日程。路径为 `${N607_PROJECT}/runs/20261125-phase1-lora-dura-dual-aperture-s2026112991-r01`，工作目录为同名 `-release`。运行 `python -m experiments.pamf_lora_main.dura_dual_aperture_train publish --repo <checkout> --output <artifact-root>/release --context <runtime_source.json> --ssh-config <config>`。实际提交由推送的 HEAD 固定并写入 `submit.json`。输入不符、数值异常、运行身份不符或产物无法闭合时停止本运行。预期有 `smoke-dual_aperture.json`、`logs/fit.log`、`dual_aperture-complete.json`、`dual_aperture-model.pt` 和独立进程终态读回。

冻结后在已登记的两路线源 V 上，用同一固定读出与直通、原冻结 DURA 和扩容 DURA 做同包比较，观察 clean、六景和低城区。首轮正式目标验证固定 Tri-set 种子 `2026092702`，沿用 15 台接收机和两路原登记 IQ，每台取同一盲序前 200 个物理包。预测和评分分别运行，源 V 结果用于解释与研判，固定子集照常完成。晋级三种子 30,000 包矩阵的科学阈值依据独立源 V，在两路线的六景正确数均严格超过同次配对原 DURA，且 clean 正确数至少持平。正式硬指标仍以原目标为准。W24 单种子四组测试使用其已发布的独立权重和独立矩阵。 [双路线评测登记](../20261125-phase1-lora-dura-dual-aperture-pilot-s2026092702-r01/report.md)

## 启动核验

14 项本地聚焦测试和独立 P0/P1 审查通过。N607 已独立核对归档、远程编译、代码提交 `760d7466b64ae89b6806972bdc52792ae6b49f1a`、GPU 5、进程工作目录和命令。真实源 U 启动检查通过，测得 1,273,150 个参数，读取识别器、TX、源 V、目标和仿真轨迹均为零。已读到完整训练的第一步有限日志，仍以第 40,000 步冻结为准。原 W24 训练继续在 GPU 0 运行。
