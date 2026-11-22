# LoRa 留源校准的自适应 Tri 配对试验

## 受控比较

预登记单种子，同一 LoRa 六 chirp practical cache 和物理身份。三个识别器均从 `20261122-phase1-lora-supervised-extension-m3-r01` 的 `coherent_practical` 源选择 E120 冻结权重出发。继承链为同契约且只用标签的 packet_sup/E30 加 6000 步监督续训，训练和选择均未读取 U 真值及目标。三组使用相同 48 轮×50 步、完整 L 曝光、源 practical 增广序列、真实标签加权、优化器、余弦学习率和末轮冻结。各半监督方法的每轮 1600 个盲 U 候选 ID 完全相同且不重复，增广使用单独随机流。比较匹配优化步数；Tri 的教师缓存和额外视图有计算开销。

| 行 | 训练差异 |
|---|---|
| continued_sup | 真实标签 clean CE + 0.35 practical CE，无 U |
| plain_pl | 同一 L 目标，加普通在线 0.95 硬 PL，余下样本无伪标签损失 |
| adaptive_tri | 使用经留源验证的静态教师规则和多视图跨 RX，仅对可靠类别开启硬 CE 与跨 RX 锚点；所有候选 U 还接受弱的同包跨 chirp 特征一致性 |

`20261122-phase1-lora-supervised-rxholdout-m3-r01` 使用三折真实留源、冷启动监督教师，在独立源包验证阈值。候选类须通过折校准与另一半验证，再在实际 E120 的源 V 上以不低于 0.90 的三预测置信度通过二次校验。至少两个合格类才开启硬标签，每个 batch 每类至多 3 条，按当前合格类数调硬 CE 权重，最大 0.40。跨 RX 项只用有真标签的不同 RX 同类 L 作为正对，实际有接纳时执行。模糊样本以 0.06 弱剂量优化不同奇偶 chirp 的同包特征方向，避免强行指定 TX。每行记录加权前 CE、接纳物理 ID、预测类别、损失及源分组指标。只有最终权重固定后才打开盲缓存和离线真值。

## 评分与执行

源使用 L=1600、V=800、训练可见 U 波形=2560。目标开发读出固定 15 台未见 RX × 128 条 × clean 和六种 practical 共每行七景，沿用已历史曝光的盲物理顺序，不据此选择或调参。三条行共 21 个评分单元。相对同一运行的 `plain_pl` 报告净纠错、准确率、F1、逐 RX、源 L/V 及离线 U 准确率。pilot 用于验证方法和挑选后续源域复核方向，目标得分不回流当前训练，也不宣称完整主矩阵已验证。

命令为 `python -m experiments.pamf_lora_main.refine_publish --adaptive --repo REPO --output OUTPUT --context CONTEXT --ssh-config SSH_CONFIG --parent-submit R02_RECEIPT --data-submit R01_RECEIPT`。固定 N607 普通账号 CVS-RFFI 环境，CWD `CV-SincNet/releases/20261122-phase1-lora-adaptive-repair-m3-r01-release`，产物 `CV-SincNet/runs/20261122-phase1-lora-adaptive-repair-m3-r01`，GPU0–6 每任务一张卡，排除 GPU7。源物理门槛审计缺失、父模型不合、非法数据、非有限 loss 或不完整预测停止进一步派发。真值侧 `experiments.pamf_lora_main.packet_score --adaptive` 自动等待冻结后完成全部评分。预期为 `source_policy.json`、逐轮 L/V 与 U 接纳记录、冻结模型/预测与独立分数。

发布前用真实 E120 父权重、真实源 IQ 验证三视图一致性反向，梯度范数约 0.536。验证每类最多三条、单类禁用硬标签和多类跨 RX 支持。发布烟雾测试再次只用有标签的同源 IQ。
