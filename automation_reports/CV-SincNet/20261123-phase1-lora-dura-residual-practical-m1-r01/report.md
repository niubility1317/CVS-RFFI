# LoRa DURA 在 residual practical 下重训

## 观测与数据

沿用 LoRa 原有 5 台源接收机、2,800 个 L 包、25,200 个 U 包和 15 台未见目标接收机的 30,000 个物理包。保留六个原有前导 chirp 和 clean 处理。Residual 引擎支持最多 1 ms 的冻结快照，本试验把带真实邻接上下文的 1 MHz 原始前导抗混叠重采样，再在每个已登记的 256 kHz、256 点 chirp 上运行一次 residual practical。每包六个快照共享同一个几何抽样、同一接收机硬件会话，各 chirp 的残差实现独立。使用 post-sync、无 EQ、868.1 MHz，参考 SNR 为 30 dB。以快照提前 8 个低速率采样点的真实上下文补偿已登记因果滤波数值延迟。所有 chirp 输出统一按包 RMS 定标。本方案与 full practical 的 1 MHz 连续波形路径存在处理顺序和快照相关性的差异，两者分开记录。

## 训练和正式验证

种子固定为 2026092702。源 L/U 每个物理包两套确定的 residual 观测，保持同包 clean 配对。作用训练使用同一观测的无噪波形拟合局部复增益，另由同轨迹的新增噪声计算方差。局部复增益是对角近似。源训练使用 15% clean 恒等曝光、配对观测相位对齐和去载波形状损失。粗恢复从零初始化，后续各阶段仅继承本轮直接父权重。保持原 DURA 八阶段、R65 延迟剪枝、R69 两阶段四步 CG 及 266,951 参数。各阶段 20 轮、4,940 步，共 39,520 步。恢复训练 worker 未收到目标或 F 权重路径，也不读 TX 真值或 V 标签。

G 权重冻结后才生成 residual 盲测视图，再接同种子冻结 LoRa Tri-set。固定比较直通、R69 原生波形、保留观测载波的全量限幅及半量。每个读出覆盖 clean 和六种 practical，15 台接收机共 420 份预测数组。独立评分程序验证全矩阵冻结后才读取目标真值，逐景和逐接收机记录纠对、改错、clean 变化和六景均值。另读已有同种子的 full practical 直通以及复响应半量作为跨路线比较。当前门控不跨路线继承。

## 发布和判定

在 N607 普通账户的独立运行目录 `CV-SincNet/runs/20261123-phase1-lora-dura-residual-practical-m1-r01` 执行。使用现有 LoRa 普通账户 Python 和冻结依赖，工作目录是同名 release。用 `python -m experiments.pamf_lora_main.residual_dura publish` 传入 repo、输出目录、普通账户运行环境、SSH 配置、现有物理数据和 Tri-set 发布回执。GPU 优先序为 3、0、1、4、5、6、2，源训练和预测各自最多一张卡。源配对或盲视图生成用 16 个 CPU 工作者。源真实数据 smoke 核对对齐、快照重放、无噪成分重构以及粗恢复、作用和 R17 梯度。运行中遇到输入、观测、非有限、checkpoint 或矩阵错误时停止本 run 并保留产物。

独立评分入口为 `python -m experiments.pamf_lora_main.residual_dura_score`，输入为新发布回执、SSH 配置、原数据集、独立输出根及已有复响应三种子结果。训练日志位于 `G/lineage/<stage>/training.jsonl`，盲预测位于 `predictions/`，正式读出为 `scores.json`。本次先完成单种子重训验证，三种子总体性能另外计入。
