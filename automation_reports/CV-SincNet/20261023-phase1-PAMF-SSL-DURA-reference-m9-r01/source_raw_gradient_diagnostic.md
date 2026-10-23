# 原 E30 raw 身份梯度与频谱干预诊断

两个诊断仅在 CPU 使用原 RXPSM 同 seed E30／6660，未更新模型、教师、归一化或分类头，未读取目标或 U 真值。UGP／CMR 健康正式任务保持。本节在读取这两版任何目标成绩前完成；不使用其测试成绩确定下一方案。

梯度部分复用每 seed 前 8 个固定 L128／U256 源批次，每次还原模型和自身 EMA／L-only 原型／置信状态。原受控首个 L 输入仍为 raw IQ；内部完整六次 T1 L 前向和原 U 损失保持。新增只读取原 T1 最后两个 raw L 前向已有的平均真实标签 CE 图，不添加网络前向或损失更新。临时 L-only 统计丢弃；参数严格保持原权重。

| 参数组 | raw CE 与原 U 冲突／24 | raw CE 与原 full-L 保护后 U 冲突／24 | 原 full-L 与 raw CE 冲突／24 |
|---|---:|---:|---:|
| all | 14 | 12 | 0 |
| t2 | 14 | 13 | 0 |
| cls_head | 11 | 8 | 0 |
| other | 14 | 13 | 0 |

整体在 full-L 保护后仍有 12／24 次 U 梯度与 raw CE 负内积，最小余弦约 −0.291。full-L 与 raw CE 在全部 24 次均为正内积，不能将两种监督参考当作相同方向。这里只说明当前欧氏梯度空间的缺口，不是原生外部 ordinary 增强轨迹、Adam 实际步长、目标损伤因果或后续收益证明。time_down 无自身具名权重，其上游仍在 other。

另一个固定频谱干预仅由全部 6300 条合法 L_s、90 个 TX×day×RX 单元估计 16-bin 平滑 log-PSD；先在同类／day 消除跨 RX 总均值，再平均类别／day，得到与 TX 无关的名义 RX 样式。变换仅用来源 RX 与其他源 RX，幅度增益限制 0.5 至 2，并保持每条 IQ 的 RMS。没有用 V 或 U 拟合变换；对原固定 V720／U960 全部四个循环 donor RX 做冻结检查，不选择 donor。

该 pooled RX 项仅描述约 55.51% 的条件频谱变化；留一 TX 外估计与对应 TX 的 RX 频谱差余弦为 0.469 至 0.912，仍有显著 TX 交互。V 四个 donor 的正确数均值为 seed1 700.5／720、seed2 697.75／720、seed3 699／720，原输入为 704、703、703；不能把它认定成纯硬件接收机变换。U 预测一致性也不是 U 标签精度。当前暂不发布频谱变换训练，不搜索强度或根据目标重试。

参考射频文献中的通道／身份分离动机：[Disentangled Representation Learning for RF Fingerprint Extraction under Unknown Channel Statistics](https://arxiv.org/abs/2208.02724)。本文只借用问题动机，未复现其对抗分解网络；本源诊断的 class-balanced PSD 估计是独立操作假设。
