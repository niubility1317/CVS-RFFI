# LoRa 四分量方向的独立噪声重抽训练

待证伪的假设是，每次见到同一源波形时重抽加性噪声，可以让恢复器更好地保留固定的复数细节。对照复用此前固定噪声源库训练的合规 48,000 步四分量方向头。候选从相同种子重新初始化，同一模型结构、同一 Practical 父恢复器、同一接收相位跟踪和四分量保范数读出。旧方向头只作冻结后对照。 [原四分量训练](../20261128-phase1-lora-w24-component-directions-s2026112991-r01/report.md) · [独立源噪声库](../20261128-data-lora-w24-online-additive-source-s2026112991-r01/report.md)

每条源 U_s 波形沿用原传播和接收机作用。候选每步用源加噪前波形和两个已登记的噪声算子重抽加性扰动，并对受扰输入与同条训练参考施加相同新 AGC。模型只看受扰后的包。原 20,000 个合法源 U_s 拟合包和 500 个互斥留出包保持一致。拟合索引与原版本使用同一 RNG 和顺序，训练噪声另用种子 `2026113891`，冒烟用 `2026113892`。留出波形复用原固定噪声。每批 16 包含 2 个原恒等视图，固定 48,000 步，模型种子 2026112991，AdamW 初始 `2e-4`、余弦终点 `2e-5`、衰减 `1e-4`，梯度裁剪 1。保留原双分支损失与 0.50 倍四分量角度损失。模型训前、训练和冻结均隔离 W24、TX、源 V 与目标。

命令 `python -m experiments.pamf_lora_main.w24_peer_spectral_direction_source publish --online-additive-redraw --repo . --context <原 Practical 源上下文> --ssh-config <N607 配置> --output <独占本地发布目录>`。代码以本次登记提交为准。N607 用原源上下文 Python 和发布工作目录，按 7、0、3、2、6、5 优先选择一张符合并发且有 12,288 MiB 可用显存的卡。真实源 U 前向、反向检查通过后继续。候选采用固定末步，与旧方向头同步走此前确定的 300 包源 V，先封存全部七景预测再独立评分。两个版本都进入正式 Full。文件包括源训练日志、`fit-smoke.json`、`fit-complete.json`、`direction.pt`、源预测及独立分数，位于 `runs/20261128-phase1-lora-w24-online-redraw-directions-s2026112991-r01`。技术或权限错误时停止当次运行并保留结果。 [正式配对登记](../20261128-phase1-lora-w24-formal-online-redraw-cpu-s2026092702-r01/report.md)
