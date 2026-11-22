# LoRa 原 DURA 的可分离物理轨迹监督

在 600 个固定源 V 包的六景上，同一冻结强 Tri 使用原 practical 为 58.0278%，精确减去模拟器新增噪声后为 67.8611%，clean 为 78.1667%。城区低从 41.8333% 升到 63.5000%。无噪声仍残留 deterministic 信道失真。基于该对照校准原 DURA 的两个既有组件，让粗恢复看见无新增噪声的 practical，让作用算子用同一源 clean 解释无噪 practical，同时校准已注入的噪声方差。

固定本 LoRa 数据合同，生成已注册独立源 practical realization 的 2,800 L 与 6,720 U 的物理轨迹。U 每五条保留一条不参加任何恢复训练，其余固定每三条采一条供 trace。生成时重新核对原输出与已有源 IQ 完全相同。仅保存按原观测 RMS 定标的无噪 practical，观察与无噪之差给出注入噪声。V 与目标均不参与训练或门控校准。推理仅使用当时的接收 IQ。没有 F、TX 真值或 WiSig 权重进入恢复训练。

继承同合同原 R65 教师。前 450 步单独校准原粗恢复，接着 450 步只校准原接收作用估计器，联合 1,500 步后得到同源老师。原 R69 学生复制最后两阶段并蒸馏 2,400 步，保留两 prior、四 CG、FFT/IFFT、原配对波形、相位、谱和 clean 恒等机制。每步物理轨迹 4 L、8 U。联合阶段额外启用既有 20 次 practical 配对及 4 次 clean 恒等样本，轨迹损失权重 0.12。单独校准时另外组件不接收梯度。

固定种子 2026092702 与冻结强 Tri-set。六路对照复用原 R69 配对实验，新增轨迹训练后全量和半量，观测 IQ RMS 上限 15% 和 7.5%。同 30,000 包、15 台未见 RX、clean 和六种 practical、840 份盲预测文件、56 个评分单元。目标标签仅在整组预测固定后由独立评分侧读取。报告逐景绝对表现、和同方案 clean 的差值、对照增益、净纠错和 clean 保持。

使用 `python -m experiments.pamf_lora_main.full_dura_publish --draw-submit ... --stack-submit ... --r69-submit ... --r69-trace`，N607 普通账户、GPU0–7 容量调度。Git commit 定义代码，CWD `CV-SincNet/releases/20261122-phase1-lora-dura-r69-physical-trace-m2-r01-release`，运行根 `CV-SincNet/runs/20261122-phase1-lora-dura-r69-physical-trace-m2-r01`。源轨迹生成后完成真实 IQ 与已冻结权重 smoke，再训练、源核对、盲预测和独立评分。异常数据合同、物理对齐、权重祖先、梯度、非有限、限幅及矩阵错误才停止。

[源域去噪反事实](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_fingerprint_audit/r69_channel_noise_oracle_source.json)
