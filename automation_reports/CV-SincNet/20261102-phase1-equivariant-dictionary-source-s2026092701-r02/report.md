# Phase1 隐藏 TX 标签的复数字典恢复器 r02

## 方法与协议
独立复数短片段字典从接收 IQ、估计后验均值和盲估噪声预测局部残差，不加载 DURA。r01 源训练入口会直接读取源 L_s 设备身份真值，不符合恢复器 TX 标签隐藏约束；r01 已在训练前停止，相关队列保留并不得恢复。r02 去除该标签加载及分类交叉熵，改为仅对合法 L_s/U_s 配对源 IQ 计算波形与冻结 Tri-SIG-TIME SSL 特征损失。冻结基座本身由原半监督识别训练获得，来源开销应披露；现有物理先验的混合分量使用冻结基座预测伪类别构造，而非直接读取 TX 标签，属于 F-guided 方法。

源训练与 prior fit、源留出记录互不重叠。固定源预测后才连接合法 L_s 真值评分；训练不读任何 TX 身份、V、测试或 U_s 身份真值。模型冻结后 Phase1 clean 与六个 practical 场景逐记录预测，同一组预测固定后独立 truth-last 评分。此次评估沿用已多次探索的地面开发集，不称为盲测。

## 固定矩阵和运行
代码/config 使用本报告所在 Git commit；N607 普通用户，release 工作目录，2 线程，最高 2 个本任务 GPU 计算进程且最少 12 GB 可用显存，采用既有共享 GPU 租约；CPU 负载超过 72 等待。固定 seed 2026110317，训练 8192 条源记录、3 个既定模拟视图、5 轮；source 真实权重 smoke 先运行，再执行完整源预测和独立评分。命令 `python -X utf8 -u -m experiments.phase1_equivariant_dictionary.source_controller --commit <本报告所在 commit>`；source、七景预测和独立评分分别由各自控制器执行。

不可覆盖 run 为 `20261102-phase1-equivariant-dictionary-source-s2026092701-r02`，ground 为 `20261102-phase1-equivariant-dictionary-ground-test-s2026092701-r02`；预期输出模型、源预测、七景预测及 truth-last 评分。发生协议越界、执行失败、输出冲突或无法合法预测时停止并保留产物；成绩低不停止。源诊断不能替代地面测试。
