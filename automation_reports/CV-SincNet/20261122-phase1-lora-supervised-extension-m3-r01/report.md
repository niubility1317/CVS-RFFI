# LoRa 监督基座 6000 步续训

用户要求先修复监督拟合，再进入自适应 Tri 半监督。此前 900 步的联合 Tri 在同一 1600 包 L 和 800 包源 V 上为 52.75% 和 43.88%。本轮只训练阶段性监督 F，U 和目标不进入配置、训练或选择。沿用 LoRa 输入修正试验的物理角色、1 MHz practical 模拟和六个 chirp 的缓存。全部候选从匹配同源角色、未用 U 的旧版 E30 `packet_sup` 开始，独立重建 AdamW。使用固定种子 2026092701，不改变标签权限。

## 对照

| 监督候选 | 0–1800 步 | 1801–6000 步 |
|---|---|---|
| spectral_clean | 原 IQ + STFT 分支，逐类加权 clean CE | 同左 |
| spectral_practical | 同上 | 加 practical 真标签 CE，前 10 轮从 0.035 平滑到 0.35 |
| coherent_practical | 原 IQ + 公共载频与相位去除后的幅相残差分支，逐类加权 clean CE | 同步加入相同 practical CE |

三组每轮 50 批、120 轮。每轮 L 1600 个物理包恰好各曝光一次，每批 8 类，每类 4 个接收机。L 顺序和场景抽样使用独立的固定随机数，三候选共享样本和场景序列。clean 目标保持完整 6000 步。调度固定为前 1800 步从 3e-4 余弦衰减，随后重启到 1.8e-4，再衰减至 2.4e-5，以观察后续训练边际收益。记录步数、完整 L/V、逐类、逐 RX、CE 和学习率。1800、3000、4200、6000 步保存 checkpoint，记录六种 practical 源 V。源选择指标从源 clean、六景 practical 均值和最差源 RX 得分合成，权重固定为 1、0.5、0.3。每条臂全部跑完并保留；源诊断无目标评分。本轮的 V 是训练 RX 上物理包分离的源验证，后续需要独立留源 RX 的证据再校准伪标签。

## 发布前核对

真实父权重与真实源 IQ 的一次完整采样覆盖核对、带标签反向、源协议负向检查均通过。原始 IQ 叠加已知相位和载频平移时，24 条真实输入的新残差分支输出 MSE 约为 1.24e-6。1200 步源域小对照中，谱域纯 CE 为 L 64.25%、源 clean 48.00%、六景 32.94%。分阶段谱域 practical 组为 L 63.88%、源 clean 47.00%、六景 35.25%。残差 practical 组为 L 60.69%、源 clean 53.25%、六景 42.17%。三个短训练同数据同预算，practical 小对照从 400 步启用增广，长训练固定在 1800 步后。残差组短训练设备 33 仅 5/80，尚未达到稳定监督基座。 [短程源域证据](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/repair_fixture/supervised_diagnostic.json)

## 运行

运行 `python -m experiments.pamf_lora_main.refine_publish --supervised --repo REPO --output OUT --context CONTEXT --ssh-config SSH_CONFIG --parent-submit R02_RECEIPT --data-submit R01_RECEIPT`。代码身份为本报告所在同一 Git commit。N607 普通用户，既有 CVS-RFFI Python，工作目录为 `CV-SincNet/releases/20261122-phase1-lora-supervised-extension-m3-r01-release`。不可覆盖输出为 `CV-SincNet/runs/20261122-phase1-lora-supervised-extension-m3-r01`。GPU 0 至 6 按可用容量每任务一张独立显卡，排除 GPU 7。源 cache 使用 R02 已准备的 `source_views/L_s.npy` 与 `V.npy`，不打开 U 或目标文件。源协议、真实父权重 smoke、非有限梯度、训练步数或系统性 CUDA 错误阻止继续派发并保留已有产物。预期逐臂 `epochs.jsonl`、四个 checkpoint、`complete.json` 及 `source_decision.json`。
