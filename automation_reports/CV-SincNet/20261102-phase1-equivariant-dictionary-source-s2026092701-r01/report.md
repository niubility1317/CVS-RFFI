# 独立复数字典恢复网络：Phase1 源训练与地面测试

## 科学方案

独立候选以源训练的盲信道估计和六分量高斯混合后验均值为初值，用可学习复数短时字典作信号先验。256 点 IQ 被分成重叠的 32 点片段；每个片段与 128 个复数原子计算最小二乘复系数，按相位不变的相对残差软分配，再重叠合成局部校正。门网络读取接收波形与初值的相对功率、复相关、拟合误差、字典不确定度和估计噪声；输出有界残差。字典模块按共同相位旋转设计为等变；联合改变幅度与噪声尺度时，在数值稳定范围内近似保持对应关系。零初始化严格保持物理后验初值。

字典模块本身 30,177 个参数，不包含上游盲信道、源先验、保护门和冻结 Tri-SIG-TIME SSL 的成本。模型训练与推理不加载或调用 DURA；源域评分与正式评分仅将已固定 DURA 预测作为对照。模型不使用推理时不可得的干净 IQ、类别真值或仿真真实信道。

## 训练与预先固定评测

使用固定 8,192 条 Phase1 源训练记录、原有 21/22/23 三个模拟视图，包含 clean 与受损视图；源划分与物理先验训练记录、源留出记录分离。训练 5 轮、batch 16，物理后验预计算 batch 8；AdamW 学习率 3e-4、weight decay 1e-4、梯度裁剪 1。仅字典和其残差门训练，冻结盲信道、0.8 保护门与 Tri-SIG-TIME SSL。训练损失结合源配对波形、干净 F 特征与软输出，以及合法 L_s 身份交叉熵；U_s 身份真值、V 与地面测试均不读取。六类 L_s 元数据为物理编号核对而读取 6,300 条，只有训练记录内的合法 L_s 标签可参与更新，实际数量由输出记录。

先做 64 条混合源角色、1 轮与 28 条源预测的真实权重预检；随后使用固定的 5 轮最终 checkpoint 预测 672 条源留出记录。固定全部预测后，先校验同物理编号的原始基座、保护门、物理后验与既有源记录一致，再只连接合法 L_s 身份；对比物理后验和固定 DURA 的正确数、配对修复／损伤，以及波形误差和干净 F 高置信伪标签一致性。

在训练前固定完整地面测试：clean 与六种模拟星地视图各 168,000 条。测试前仅重放 112 条固定源预测；七视图预测全部固定后才连接真值，并独立核对共同基座、物理编号、119 个 TX／RX／日期分组与 cross-RX、cross-RX+day。源域或地面结果不用于选择 checkpoint、改变另一个健康实验或取消七景。六个受损场景相对干净 Tri-SIG-TIME SSL 86.538690% 的下降均不超过 3% 的阈值为 83.942530%；另报告下降 3 个百分点的 83.538690% 门槛及 clean 损伤。

已有源记录、基座和地面开发集此前已被反复用于探索，结果是单种子模拟地面开发结果，不是独立盲测或真实卫星链路。目标持续 active，不能根据模型结构或源代理宣称达标。

## 执行及停止规则

- 代码身份为包含本登记与实现的 Git 提交；launch owner 为本对话主 agent。
- N607 项目 CVS-RFFI Python，TF32 关闭，CPU 2 线程；与本对话已登记的 DURA 双候选共用 GPU 工作锁，单次最多一个本任务 GPU worker。获得锁后重新检查 CPU 负载不超过核心数减 24、至少一张卡剩余显存不少于 12,288 MiB 且可见计算进程少于 2 条；训练前在 N607 定向验证锁的跨进程继承。不得停止或修改其他任务。
- 源训练：`python -m experiments.phase1_equivariant_dictionary.train_source --root ROOT --out RUN --commit COMMIT --device cuda:0`，预检另加 `--smoke`；源真值评分为 `python -m experiments.phase1_equivariant_dictionary.score_source --root ROOT --source RUN --out SCORE --commit COMMIT`，CPU only。
- 七景入口：`python -m experiments.phase1_equivariant_dictionary.ground smoke|predict --root ROOT --out GROUND --commit COMMIT --device cuda:0`；独立评分为 `score --device cpu`。
- 源训练根：`runs/20261102-phase1-equivariant-dictionary-source-s2026092701-r01`，预检加 `-training-smoke`，源评分加 `-score`，控制器加 `-controller`。地面根：`runs/20261102-phase1-equivariant-dictionary-ground-test-s2026092701-r01`，控制器加 `-controller`，独立评分加 `-auto-audit`；全部输出唯一，不覆盖。
- 只因协议越界、非有限值、路径碰撞、源重放不一致、无合法预测或技术故障停止本实验并保留 partial，低准确率不触发停止。发布前完成局部协议负测、一次聚焦独立 P0/P1 审查、单归档校验和远端编译；实际源权重预检通过后再自动训练。

当前为本机原型与实验准备，不表示真实源训练或七景评估已完成。
