# LoRa 双采样率 DURA 单种子双路线配对测试

固定双采样率 DURA 从零训练的第 40,000 步、源 U 门限、末步半量读出和同一冻结 Tri-set 种子 `2026092702`。两路线均在原登记的 `6×2×256` 实收输入上运行同一模型。候选推理只接收当包 IQ。Full、Residual、接收机和场景用于调度、索引和汇总。 [源训练登记](../20261125-phase1-lora-dura-dual-aperture-s2026112991-r01/report.md) · [盲预测入口](../../../experiments/pamf_lora_main/dura_dual_aperture_evaluate.py)

先在合法源 U 做真实权重冒烟，记录恢复器 clean 和通用受扰单包时延，再在同一 900 包固定源 V 面板上分路线处理 clean 和六景。独立 CPU 评分在两路线源预测固定后读取源真值，比较直通、对应路线原冻结 DURA 半量、原扩容半量和候选。原 Full 读出为 `coherent_half`，原 Residual 为冻结 R69 经形状投影后的 `dura_half`。源 V 晋级条件为两路各自六景正确数严格超过原 DURA，clean 正确数至少持平。源 V 的得分高低不影响本次固定子集测试。

正式子集使用 15 台未见接收机，每台原盲序前 200 个物理包，两路线各 3,000 包，种子 `2026092702`，均测 clean 和六景。全量闭合 420 个决策数组后，在评分侧逐值匹配各路线归档直通。复用匹配物理包的原 DURA、原扩容和 `guard_half` 冻结预测，Full 另对照冻结复响应半量。直通对齐及完整产物都核实后，评分器才按相同物理序读取原始标签。输出逐路线、逐景、逐接收机正确数、纠错、改错、净变化及 1,600 次按 RX 分层的物理包配对区间。 [独立评分入口](../../../tools/lora_dual_aperture_pilot_score.py)

在 N607 普通账号下，训练提交 `760d7466b64ae89b6806972bdc52792ae6b49f1a` 和源运行路径固定绑定。评测工作目录为 `${N607_PROJECT}/releases/20261125-phase1-lora-dura-dual-aperture-pilot-s2026092702-r01-release`。运行 `python -m experiments.pamf_lora_main.dura_dual_aperture_evaluate publish --repo <checkout> --output <artifact-root>/release --context <runtime_source.json> --ssh-config <config> --training-receipt <fit-release>/submit.json`。调度器等待训练成功冻结后再分配 GPU，源两路分别调度、目标最多八个独立接收机分片。技术停止条件为终步或身份不符、真实权重冒烟失败、数据矩阵或有限性检查失败、预测集不闭合。保留全部产物。

预期产物包括 `smoke.json`、两条路线各自的源预测及源评分、`source-promotion.json`、八个分片状态、闭合的 `predictions/complete.json` 与独立评分 `scores.json`。本次单种子子集用于首次匹配测试。正式硬指标沿用预登记的三种子和 30,000 个物理包。

## 调度核验

13 项聚焦检查已通过，覆盖终点依赖、旧三种子矩阵、子集下标、直通对齐和配对统计算法。N607 已独立核对评测提交 `619febc9421ffaf3cc28df0b22a5793655db086f`、归档传输、远程编译、进程工作目录和命令。调度器已进入 `WAITING_FOR_FROZEN_TRAINING`，目标预测文件为零。原 W24 调度器仍在运行。独立评分端已绑定原始标签归档和四个匹配对照目录，确认目标预测完全闭合后才开始评分。
