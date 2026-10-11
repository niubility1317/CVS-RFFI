# 无标签样本与浅中层机制的历史覆盖核对

本次只核对已有代码和完整源训练报告，不读取目标成绩来决定新机制，不更改当前冻结实验。

## 已做过的无标签特征增强

`20261012-phase1-e20-routed-fishr-manysig-m24-r01` 的 COUPLED、STYLE、JOINT 三个方案均在无标签 U 的强视图主前向中调用 `main_style`，作用位置为 `time_down` 与 `t2.pool`。每个方案保留三个种子，沿用 E20 后追加 20 轮。

其概率来自当步学生的弱视图，并与计算图分离；供体要求同伪身份、同 day、跨 RX、不同物理 ID，置信度至少 0.95。主 KL 另有接受门槛。没有读取 U 的真实 TX。

该批已完成的源报告覆盖 24 行、106,560 次主更新。9 个 U 风格变体各层均记录了全部 3,996 次调用。合法供体比例约 95.11% 至 95.88%，实际特征改变比例约 46.80% 至 48.02%，相对特征变化均值约 0.0110 至 0.0152。分母是每层被调用的样本出现次数；两层不能相加后称为不同样本覆盖率。相对变化也不能直接解释成有效身份变化或泛化收益。

这说明旧 U 风格分支确实执行过，并非开关存在但梯度路径未生效。单独再次启用同样的 U MixStyle 不构成新的方法比较。

## 当前批次的准确边界

当前 `scratch-calibrated-style m36` 与 `scratch-identity-style m9` 将合法 U 通过另外两个原始 E20 模型的固定概率均值纳入主 KL；学生从头训练，不载入教师权重。U 样本没有被排除，但这两批的 `mixstyle_u=False`，不能宣称 U 同时接受了浅中层风格增强。

当前 Fishr 类项使用合法 L 上的身份 margin 对浅中层虚拟通道坐标的敏感度；旧 routed Fishr 使用辅助读出头的梯度方差。二者统计对象不同，不能笼统当作同一种 Fishr 的重复运行。

当前困难风格仅在合法 L 上搜索。m9 在 clean 与四个完整风格候选之间选择整批 CE 最大者；clean 被选中时不会安装风格 hook，也不计入实际特征改变。只有完整源日志和固定测试完成后，才能分别讨论覆盖率、训练信号和测试结果。

## 可追溯来源

- [旧批完整源训练报告](../20261012-phase1-e20-routed-fishr-manysig-m24-r01/source_trajectory.md)
- [旧批配置](../../../experiments/cvs_e20_routed_fishr/design.py)
- [旧批 U 主损失路径](../../../experiments/cvs_e20_routed_fishr/objective.py)
- [旧批 U 风格 hook](../../../experiments/cvs_e20_budget_fishr/multilevel.py)
- [当前设计](../../../experiments/cvs_scratch_identity_style/DESIGN.md)

本核对不发布额外训练，也不据此变更在训配置。后续若研究 U 特征干预，应明确它相对于已有供体、监督来源、扰动规则及梯度路径的实质差异，并保持固定对照。
