# 报告投影机制：三种子 clean 固定矩阵

用户最新要求每模型 30 轮、普通 cos 分类头、取消源 V 性能筛选，只测试 clean。本轮使用实际合法从零 CE30 的三个对应种子父权重，每个父模型保持固定末轮，训练 L6300 仅闭式拟合表示变换，无新增 SGD 或训练 epoch。父训练累计 90 模型轮；新的特征提取、矩阵分解和 24 行推理成本另计，不冒充从零训练 24 个模型。

固定方法 NATIVE、COS_MEAN、CENTERED_COS、GEIV、ORTHOGONAL、OBLIQUE、RIDGE、SOFT_RIDGE，各配 primary 2026092701/02/03。所有变换维持 160 维和普通 30×cos 分类公式，无 RX 分类器、专家路由或特殊 margin。GEIV 用条件 RX 协方差加 0.1×类内协方差及相对 ridge0.001，固定谱权重下限0.5并保留全宽，避免仅凭六个源类均值将新身份表示截断至五维。正交和斜投影消除全部非零条件均值 RX 张成空间；15 个类对保留率、PD/P²残差作为机制诊断，不据此筛掉 clean 测试。softlambda1、ridge0.001均提前固定。Gaussian 有效 Fisher 仅在报告明确的连续均值路径、高斯噪声工作模型下计算，不能解释为离散 TX 的真实 Fisher 信息。

实际 L/U/V 为 7/63/30，训练池标签占10%；源 RX1/3/4/6/8，原物理角色保持。U不打开，V无前向；源fit只用L标签和RX。严格核验三个CE30父checkpoint的实际来源、训练数据、epoch及无目标训练/选模，不继承oracle80。源协方差与样本特征不进入预测状态；仅固定A、bias与普通cos类别权重。Phase2可重新拟合类别权重，但本轮不声称这些源拟合表示状态已通过Phase2权限审计。

各种子全部8个映射固定后，共享一次冻结骨干 clean 前向，形成8组各168000独立全类决策。三个种子共4032000预测。独立exec评分进程仅在同seed全部预测及物理ID/count检查通过后连接truth，产生24×18=432条same-row指标，保留每seed、RX、TX、day及三seed均值/标准差。NATIVE使用原tx_logits作精确决策对照，映射等价公式另核验。目标三seed clean均值≥84%仍待实测。已暴露benchmark，不声称新独立盲测；所有预登记方法均报告，不按目标分数改变本轮配方或选择性重跑。

执行commit读取release_commit.txt；命令、环境/CWD、输入/输出路径、矩阵与产物见experiment.json。普通N607账号，一个不可覆盖run/release，一个archive SHA比对、remotecompile与真实checkpoint无querysmoke。单launchowner，共享capacitylock，每GPU总实验进程≤3/free≥12000MB。仅runowned系统技术失败、数据/权限错误、输出碰撞、不可执行或非法预测停止，保留产物；不按性能停止，不改健康旧任务。完成后独立评分、同row分析、登记与Git交付。
