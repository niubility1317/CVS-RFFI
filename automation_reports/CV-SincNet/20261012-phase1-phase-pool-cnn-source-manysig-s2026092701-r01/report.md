# 全局相位与统计池化：源端30轮消融预登记

原CNN完整5row源证据为训练100%、heldV均值38.5852%、固定混合54.7333%，native3为61.7778%，全部RX混合下降。这个直接源失败提示时间位置与局部相位模式可能产生过拟合；尚不能将失败原因写成已证明。新15row固定3arms×5RX，只检验此假设，不读取target。

POOL_ONLY移除flatten2048，保留4stage卷积后每channel time mean+populationstd共256，再256->160；参数210176。FLAT_PHASE保留原668928参数位置展平，训练每L包独立uniform[-pi,pi]共同复数unit相位旋转。POOL_PHASE同时采用两者。原FLAT/noaug的5份固定final30控制已完成，保留原权重、源预测和负结果。全局相位乘法保持功率、TX硬件身份；它不是频率扰动或星地信道，不声称任意接收链不变性。原RMS/lag20数值补偿、无CFO标量、cos头、GroupNorm逐packet保留。CUDA已经验证单位旋转功率和补偿等变、所有实际梯度、推理不增广及packet独立。

每row随机学生seed2701、固定30×222=6660；fitL5040/heldV5400，整heldRX排除训练L，U和target关闭。相同原AdamW WD0.0001、coslr0.0002至0.000001、CE、label_smoothing0，不加载旧checkpoint作初始化，不增teacher/EMA/星地。启用相位arm必须实际训练增广852480个Lpacket，POOL_ONLY为0；完整日志记录6660更新，final30固定后eval关闭增广、一次heldV预测/评分，冻态检查不变。

每个新CNN与相同fit4RX的native3已冻结概率按预定0.5/0.5融合，全部6类同公式，ID精确对齐，混合预测先保存后source评分，无拟合权重、温度或成员选择。每arm独立资格：混合比native3源均值至少+1pp，至少4/5RX非负，最差RX非劣1pp。完成全部15后只在合格arms中按source混合均值最高冻结，完全并列参数成本低者优先，再按名字确定；没有合格者不访问目标。合格者才三primary2701/02/03全源同recipe CNN30+已冻结合法native3，每row自动clean+六practical，各168000，全视图固定后独立truth-last，完整三seed/RX/day/TX结果保留，不依目标改recipe/融合或重跑。

新增15CNN99900更新/450模型轮；已完成FLAT5历史33300/native15历史99900另计；不同pool参数与推理成本完整报告，不把减少参数当成性能目标，也不冒充同成本胜出。当前clean84%尚未达成，基准已暴露，不称新盲测或卫星部署。一次候选family P0/P1、聚焦负测及匹配真实checkpoint noquerysmoke、实际代码/config Git提交、一次不可覆盖run/release、N607preflight/archiveSHA/compile、启动PID/CWD/argv/GPU/loggrowth检查。普通账户唯一owner，GPU总进程<=3含其他owner/free>=12000MB；只系统技术/权限/输出碰撞失败停止所属run，不按性能停止或选择性重跑。矩阵/命令/环境/CWD/路径/stoprule见experiment.json，无新增门槛。
