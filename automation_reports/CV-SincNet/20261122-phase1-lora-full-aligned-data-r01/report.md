# 完整 LoRa 主划分的 1 MHz 前导与 practical 视图

本数据运行支撑三种子自适应 Tri 正式对照，保留 `lora_rx5_main_v1` 原注册物理切分和权限。五台源接收机共有 2800 L、25200 U、12000 V。目标是原来 15 台未见接收机，每台 2000 包，共 30000 包。源 ID 用原冻结元数据和原始接收序号复原。U 元数据保持盲标签。目标只读取官方归档的 `data` 列，按原 seed/receiver 物理顺序导出 IQ，完全不读取 label。向 N607 复制的目标目录不含可用于评分的 TX 信息。

重新从 1 MHz 的 8192 复采样前导生成原样与六种 practical 视图。卷积使用已在可比小试核对过的完整模型，去除数值滤波延迟并使用实际输入上下文，再提取六个 256 点 chirp。L/V 共用已登记的物理样本和各自的模拟命名空间。另对每条源域 U 保存一份按物理序号分配的 practical 视图，供后续检验 practical 一致性使用。是否纳入候选以源域数据为依据。目标采用既有七视图固定矩阵，测试真值晚于识别器冻结和完成全套预测。

命令为 `python -m experiments.pamf_lora_main.full_packet_publish --repo REPO --output OUT --inputs DATA --context CONTEXT --ssh-config SSH_CONFIG`，使用普通账号与原 CVS-RFFI Python。运行和 release 分别为 `CV-SincNet/runs/20261122-phase1-lora-full-aligned-data-r01` 与 `CV-SincNet/releases/20261122-phase1-lora-full-aligned-data-r01-release`。原生源、目标输出不可覆盖。source 和 TX-blind target 构建互相隔离，分别最多 16 个 CPU worker，源 smoke 与两条 trace 均不使用 GPU。真实输入取样、dtype/物理角色不合或信道缓存进程异常时，登记明确失败并保留已有产物。预期成功标记为 `FULL_VIEWS_READY`，带独立的源 NPY 和 15 个盲接收机完整视图。此 data run 不训练、不选模、不评分。

本地已复算小试中同身份的全部 6880 包。源 L 1600、源 U 2560、源 V 800 和盲目标 1920 的 IQ 均与既有 1 MHz 缓存逐点相等。实际源包的 practical 输出为 float32，确认补偿 24 个 1 MHz 数值延迟采样。源 smoke 无目标读取。 [物理样本兼容与真源烟雾检查](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/local_view_smoke)
