# LoRa W24 冻结识别器与 DURA 输入输出测试

固定原 W24 `phase_native24` 种子 `2026092702` 的 E108 源冻结权重。主读出为八窗扩容 W24 DURA 从头训练 40,000 步的同包结果。对照包含 W24 直通、原扩容半量 DURA 的中央六窗修正回写、原权重的两组六窗覆盖全部八窗。冻结权重的两个对照共用原中央六窗门限，重训版本使用自己的合法源 U 门限。四条读出共用同一实收 IQ、包序和冻结识别器。 [训练登记](../20261125-phase1-lora-dura-w24-train-s2026112991-r01/report.md) · [盲预测入口](../../../experiments/pamf_lora_main/dura_w24_target.py)

第一轮覆盖 15 台未见接收机。Clean 30,000 包，与已归档的 W24 直通预测逐值匹配。六景各 3,000 个物理包，每台固定取原盲序前 200 个，所有扰动共用这批物理包。使用 Full practical 原场景生成器在 1 MHz 连续构造完整八条 chirp，记录外只在信道生成时用静默前缀和后缀补偿已知数值延迟。试验用独立 `w24-target-full` 命名空间，四臂接收同一份生成视图。该 Full W24 视图有自己的匹配口径。现有 residual 缓存只提供六条 256 点窗口，首尾原生信息缺失。

N607 普通账号，四个按空闲优先分配的 GPU 分片，每片三 CPU 子进程负责构造 receive-only 视图。调度器等待 40,000 步终点冻结，再分配 GPU、执行真实源 U 检查并做盲预测。异常按运行根停止。运行目录为 `${N607_PROJECT}/runs/20261125-phase1-lora-dura-w24-io-s2026092702-r01`，发布目录以同名 `-release` 结尾。以 `python -m experiments.pamf_lora_main.dura_w24_target publish --repo <checkout> --output <artifact-root>/release --context <runtime_source.json> --ssh-config <config> --training-receipt <fit-release>/submit.json` 绑定训练提交与本次预测代码。每一接收机和场景写独立四行决策数组，共 105 份。预测全量闭合后，在独立评分侧才使用原始测试真值。

报告 clean 正确数、六景均值、低城区、逐场景和逐接收机正确数、同包纠错和改错，以及接收机内按物理包分层重采样的 95% 区间。本轮为单种子 W24 输入输出试验。原冻结 Tri-set 三种子全量双路线验收单独保留。 [独立评分](../../../tools/lora_dura_w24_score.py)

## 调度核验

已独立核对 N607 归档、远程编译及提交 `2c1facd2a15aea1b34a30676743429a961e1d8b4`。调度器处于 `WAITING_FOR_FROZEN_TRAINING`，未生成目标预测。评分端已核对可用的原始真值归档位置，等待完整盲预测后才开始关联标签。
