# LoRa DURA 残差保持与多 chirp 共识试验

以旧 LoRa DURA 的逐 chirp 配准版本为同源权重起点，只用合法 L/U IQ，保持双阶段展开骨架。L 提供已经生成的 6 条 practical 视图，每视图 1600 包。U 使用原 G 配对的 2560 包，共 12160 对。G 不读取 TX 标签、识别器、目标或源 V。两个训练行都用同数据、固定 seed 2026092701、每批 24 对及 10 轮。旧逐 chirp DURA 只作已冻结基准。

| 读出 | 差异 |
|---|---|
| direct | 同一冻结 adaptive_tri 直接识别 |
| old_chirp | 旧逐 chirp 配准 DURA |
| consensus | 去啁啾和频偏校正后，用同包其余 chirp 的加权残差取共识 |
| projection | 旧骨架只恢复与公共去啁啾载频正交的分量并学习每包剂量 |
| coherent | 同一投影和剂量，再由模型混合跨 chirp 共识方向 |

恢复量在接收频偏坐标中零均值，映回原始 IQ 后保留观测公共载频分量。旧限幅在满幅时消除强度更新，新版在平滑方向归一化后学习剂量，限制在输入 RMS 的 35% 以下。训练目标为配准后残差误差、可重复残差误差和微弱波形改动惩罚。保护以当前 IQ 可观察的相干度决定。源 clean L 最低粗相干度为 0.8278，0.94 旧门槛会触发 3 条。新门槛固定 0.82，极坏包不尝试恢复。2400 条真实源 clean IQ 全部逐点透传。真实 practical smoke 的公共载频变化约 3.42e-9，剂量梯度 −0.0441。无识别器的受控前导加噪中，共识 MSE 0.15125→0.13051。

全 G 冻结后连接同一个已经冻结的 LoRa adaptive_tri。源 V 只用于审计，正式开发目标固定 15 台未见 RX × 128 包，clean 加六种 practical，五个读出共 35 个评分单元。报告逐场景相对 direct 净纠错、F1、最差 RX、clean 旁路以及 G 训练损失。目标早期已历史曝光，以同一物理包进行开发比较，目标成绩不回流当前训练或选模。

命令为 `python -m experiments.pamf_lora_main.refine_publish --dura-repair --repo REPO --output OUT --context CONTEXT --ssh-config SSH_CONFIG --parent-submit R02 --data-submit R01`。N607 既有 CVS-RFFI 普通环境，发布目录 `CV-SincNet/releases/20261122-phase1-lora-dura-residual-m2-r01-release`，不可覆盖输出 `CV-SincNet/runs/20261122-phase1-lora-dura-residual-m2-r01`。按 GPU 0–6 空闲实际使用，每卡本任务一进程、全卡至多 4 个，排除 GPU7。相同物理源契约、真实 G smoke 与健康 CUDA 更新是运行条件，非有限或缺行时停止后续派发。真值侧 `packet_score --dura-repair` 等完整预测冻结后评分。
