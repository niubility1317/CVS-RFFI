# LoRa 留源接收机教师验证

在保持 LoRa 1600 条标注训练与 800 条源验证原始物理角色的前提下，预先固定留出 B210-1、PLUTO-1、RTL-1 三折，各覆盖一类不同来源的设备或接收链。每折真实训练样本仅来自其余四台源接收机，TX/RX 格各 32 条，总计 1280 条。留出 RX 的 160 条源 V 在本折冻结后才预测和读取真实 TX。模型从随机初始化开始，不加载含该接收机的任何权重或优化器状态。全源 E30/E120 仅是历史参照，不进入折训练。U 与目标封闭。各折共享既定对齐后的 LoRa 六 chirp 输入与去公相位和载频的监督模型。

每折 180 轮 × 40 批，总计 7200 步。每轮每条 L 恰好出现一次，每批为 8 个类在四台训练 RX 上各 1 条。前 2160 步纯 clean CE，随后 practical 真标签 CE 用 10 轮从 0.035 升至 0.35。种子固定 2026092701，末轮统一冻结。记录 L、步数和各折 held clean 及六种 practical 指标。固定门槛 0.95 与整包、奇偶 chirp 三预测一致分别统计。合并三个留出折后按 TX/RX 将 V 等分，用第一半在 0.85、0.90、0.95 中选最低合格阈值，另一半只验证一次。每半至少接纳 8 条，精度至少 87.5%，至少两个 RX 各判对 2 条，Wilson 下界至少 0.55；逐类公布接纳与正确数。门槛证据用于之后的 LoRa 源域迁移试验，不替代新接收机的正式评分。

命令为 `python -m experiments.pamf_lora_main.refine_publish --holdout --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG --parent-submit R02_RECEIPT --data-submit R01_RECEIPT`。单一发布 commit。N607 既有 CVS-RFFI 环境，运行目录 `CV-SincNet/releases/20261122-phase1-lora-supervised-rxholdout-m3-r01-release`，输出 `CV-SincNet/runs/20261122-phase1-lora-supervised-rxholdout-m3-r01`。GPU 0–6 空闲容量调度，各任务单卡，GPU7 排除。复用既有 source_views 物理包，输入协议错误、真实输入反向失败、非有限值或任何训练/冻结文件错误均停止后续派发。预期产物为源烟雾验证、三份进度日志和冻结权重及 `holdout_audit.json`。
