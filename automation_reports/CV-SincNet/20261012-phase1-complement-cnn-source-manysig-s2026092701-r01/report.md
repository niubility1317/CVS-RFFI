# 独立全波形CNN：30轮源端互补检验

目标clean三seed均值84%尚未达成。原native3冻结确认81.3254%，比同recipe单模型高1.8692pp，成本3倍。前一源EMA/handoff完整10模型未满足固定源条件，保留负结果、无目标测试。本候选检验具有独立学习路径的全波形骨干，而非继续修饰同一native特征；收益是否来自更多参数及架构互补需分开报告，不预先称有效。

5个heldRX1/3/4/6/8各从零CNN30轮/6660更新，seed2026092701，L5040、heldV5400，U/目标关闭，原20%物理划分不变。原源加载器真实契约复用；训练前整RX排除，最终30后才预测heldV，固定L轮转seed+101、AdamW0.0002至0.000001的原cos30、WD0.0001、CE及label_smoothing0，无任何增强/teacher/EMA/辅助loss。实际代码读不到旧权重做训练初始化。

CNN逐packet先RMS规范化及固定80:160窗口lag20斜率数值补偿；只送补偿后256IQ，classifier无CFO标量。此窗口不声明已知激励/验证前导，也不声明任意RX或全物理不变。4个Conv1d通道32/64/128/128、kernel9/5/5/3、每packet GroupNorm8/GELU/AvgPool2，保留最终16时间位置；flatten2048->256/dropout0.2->160/LayerNorm，cosine固定scale16六类头。实际668928参数，梯度/零输入/冻态不变/批内packet独立CUDA检查PASS。

固定混合为0.5 CNN softmax温度1 +0.5已冻结同fit4RX native3的softmax概率均值温度1。全部类别、packet同一公式，不学习权重、温度、不挑成员。旧native源概率由原10新/5复用完整source分析核验，并与CNN的5400物理ID逐个对齐；先持久化混合预测再连接源V真值。CNN单模型、native3、混合全部报告。资格固定：混合sourceV均值比native3至少+1pp、4/5RX非负、最差RX非劣1pp；未通过不启动目标。

新增5CNN33300更新/150模型轮，native15历史99900更新/450模型轮复用；混合4模型每组1200003参数，训练和推理新增成本完整披露，不冒充相同成本控制。若资格通过，primary2701/02/03各从零全源CNN30轮，复用各自已验证同物理角色无CFO fixedfinal30的native3作自身推理成员；不蒸馏或给CNN初始化。每row立刻自动clean+六practical，各168000；七视图预测固定后独立truth-last，报告全部三seed/RX/day/TX和缺项。目标结果不回流选骨干、权重或重新训练，现有目标基准已暴露，不称新盲测或卫星部署。

一次P0/P1、聚焦负测/实际checkpoint noquery smoke、一次Git代码config提交、不可覆盖run/release/output、一次N607preflight、archive本地远端SHA比对、compile、启动PID/CWD/argv/GPU/loggrowth核验。普通账户，唯一owner，每GPU至多3个总实验进程含其他owner、free>=12000MB；仅按系统技术/权限/输出碰撞故障停止所属run，不按分数停止、重排或选择性重跑。精确矩阵/环境/命令/CWD/路径/stoprule见experiment.json。没有额外阻塞review/seal/hash层。
