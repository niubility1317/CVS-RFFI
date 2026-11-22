# LoRa DURA 高相干六 chirp 细节恢复

在原门控旁路的高相干源配对中，同包 clean 提供的共享与独有非载波形状可额外纠正 1,892 次，限制校正到 15% 观测 RMS 后可额外纠正 1,785 次。理想 CFO 校正只多判对 28 次。本实验验证仅靠观测 IQ 和合法源配对，能否学到这种方向。

固定种子 2026092702 和同种子的 Tri-set，沿用现有 protected DURA 分支。对其旁路且六 chirp 中位相干度至少 0.94 的包，新增一个六 chirp 卷积分支。先在实测载波坐标消去平均载波，再分开学习共享和各 chirp 独有的残差。预测校正无载波分量，应用时使用实测载波相位、频率和尺度，限于观测 RMS 的 15%。训练 1,800 步。每步原 L、原 U、独立 L、独立 U 分别 6、8、4、6 个 practical 配对，另 8 个 clean 恒等样本。均以同物理包 clean 的归一化形状为监督。固定保留每五个 U 中的一个，这些包的两个 realization 和 clean 均排除在恢复网络训练之外。

恢复器冻结后，用保留 U 的一半，学习给定观测与所提议的校正是否能将源配对形状误差至少降低 5%。使用所有源 L/U clean IQ 确定零源 clean 触发的分界。在保留 U 的另一半 practical 配对和独立源 V 记录实际触发数、受益比例和波形误差。门控使用可观测量，训练数据契约之外的 F、TX 真值、目标数据读取均为零。V 波形诊断只在恢复器与门控冻结后运行。

正式矩阵固定为同一批 30,000 包、15 目标 RX、clean 和六场景 practical，以及 Tri-set 直通、固定 DURA、形状保护和新增高相干分支四条路径。复用先前冻结的三条对照预测。仅为通过新门控的额外样本重新执行同一个冻结 F。共 420 个盲预测数组、28 个评分单元，完整冻结后独立读取真值。分别记录原分支和新分支的激活、净纠错、clean 变化及逐 RX、逐场景指标。单种子阶段按新增分支相对原保护版的六景净改善且 clean 无净损失评估，扩大种子前报告逐场景和接收机分布。

用原 full packet 数据、原 G、Tri-set、protected G、独立源 draw 和联合对照的发布记录调用 `python -m experiments.pamf_lora_main.full_dura_publish --draw-submit ... --stack-submit ... --coherent-detail`。使用普通 N607 账户和本次 Git commit，发布 CWD 为 `CV-SincNet/releases/20261122-phase1-lora-dura-coherent-detail-m1-r01-release`，唯一运行根为 `CV-SincNet/runs/20261122-phase1-lora-dura-coherent-detail-m1-r01`。允许所有 GPU，由既有容量调度选择训练与预测绑定。发布包括实际源 IQ 和旧 G 权重 smoke、一次源训练、源波形诊断、盲预测和独立评分。数据合同、非有限输出、限幅、对照身份或矩阵完整性出错即停止。
