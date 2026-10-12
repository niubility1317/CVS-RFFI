# 有限实际作用间隔：源端可证伪预登记

仍以 84% clean 三种子为最终目标。前一轮新表示先导完整返回：原IQ 51.1519%、RAW_REL 46.3704%、COMPLEX_REL 45.2296%、GL2_REL 45.8741%（五个留出源RX V均值；单seed、从零MLP）。两个不变表示均未过既定条件，保留负结果，不扩大强不变分支。该先导不能证明一般不可用，也不替代CNN互补特征研究。

## 独立假设与矩阵

报告另提议直接约束难类判别间隔。本轮不使用冻结旧响应读出或小扰动一阶代理，而直接执行四种有限数字操作。五个源RX各从零训练三个原CVS控制：CLEAN_REPLAY、ACTION_CE、FINITE_MARGIN。一个model_seed=2026092701，15模型各固定20epoch×222=4440更新，无早停。完整身份骨干、分类头及参考分支均随机初始化；留出RX完全不参与训练，不能借用全源E20权重。真实E20仅作无query预检，绝不进入新模型。

每次主训练为clean L128 CE；extra使用同batch前32包。三个方案都生成±1.5 dB固定频谱FIR、±0.25 sample插值四种实际操作，额外前向均为eval160：REPLAY输入五份clean，ACTION_CE输入clean+四实际操作求平均CE，FINITE_MARGIN输入相同五视图。

对每个竞争类c计算各实际视图m_yc=f_y-f_c，取各竞争类跨视图最小间隔，再计算logsumexp_c(-min_a m_yc)。正确类间隔恒为0。此损失上界五个实际视图的最大CE，但多个竞争类可来自不同最坏视图，因此不是某一个共同最坏操作，也不是连续椭球/FIR证书。主CE+0.5×extra_loss固定；类别全部同一公式，不指定TX难类权重。

extra eval保留模型梯度，不更新dropout或BN持久状态；主clean train分支正常训练。全部方案同样本顺序、同总前向包数，无U、EMA、信道模拟或辅助优化器。数字操作不是物理换RX；真正整机留出检验能否提供迁移收益。

## 源规则与结果闭合

固定20epoch后先保存身份权重与source_frozen，再预测heldRX L1260、V5400的clean/四操作，冻结源预测后计分。V从不反向更新、不拆新角色；全部15 TX类对、RX/day、混淆、NLL、间隔及完整4440步日志保留。

FINITE_MARGIN晋级条件：相对ACTION_CE平均clean至少+0.5个百分点，5RX至少4非负；相对REPLAY平均clean最多下降0.5个百分点；最差RX不低于ACTION_CE最差超过1个百分点；四操作压力均值不低于ACTION_CE。全部源条件同时通过才进入原物理契约三种子匹配训练，逐row自动clean+六practical预测与独立truth-last评分。本轮source-only从不访问目标；不将既有多次观察的开发集合重新称盲测。

## 执行

命令、环境、commit读回、CWD、输出、GPU、owner和产物以experiment.json为准。不可覆盖root，一次P0/P1审查、一次真实无query smoke、archive一次比对和远端编译后直接继续；每GPU最多3个总CUDA进程。仅权限/划分/输出冲突或系统技术故障停止所属run，低性能不停止，不重启健康旧任务，全部负结果保留。
