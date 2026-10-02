# CVS 因果包络耦合：源域实验预登记

状态：PLANNED。本轮沿用基础网络优化目标：交叉熵唯一、无信道增强、身份骨干、从零训练，最终只测clean。六环境评估是独立交付，不用于本轮设计或选模。

只替换共享能量核心行为分支的12项输入：P_bar[n]=(P[n]+P[n-lag])/2，输入为z[n-m](P_bar[n-m]/4)^q，m=0至3、q=0至2，lag固定1或4，缺失历史位置补零。三阶与五阶项包含滞后包络交叉乘积；原位置功率读出、网络深宽、embedding、分类头保留，两种模型均202553参数，与energy控制同seed初始state完全一致。原residual_fusion为164225参数，不称与其等参数。

数学上扩大的是输入lift的线性张成空间，不宣称旧整网无法表达这些交互。功率对共同常相位不变，新复数基函数维持相位charge1。两候选均原received输入、alpha0，不校正CFO。通信和射频依据是因果包络记忆项；固定平均只是受限结构假设，不是完整GMP、DPD或PA系数辨识。25MHz下1/4点对应40ns/160ns设计延迟，未测量真实器件时间常数。RX、信道与噪声也作用于包络，不宣称唯一TX硬件或TX/RX因果分离；平滑可能损害有用瞬态，全部负结果保留。[数学、通信、物理与RFF论证](../../../docs/CVS_COUPLED_ENVELOPE_HYPOTHESIS_20261002.md)。

两个候选各四seed，共8个新scratch模型；model/loader seed2026092701至2026092704，split392005。原L/U/V=6300/56700/27000，U不使用；sourceRX1、3、4、6、8，day1、2、3，六TX，equalized1/中心256/单位RMS。没有历史checkpoint、resume、teacher、EMA或其他继承状态。复用已验证物理数据，构建时核对完整角色，不因方法变化重验数据。AdamWlr2e-4、wd1e-4、cosine最低1e-6；batch128保留末批；每模型200轮×50步=10000步。完整FP32，cuDNN/matmul TF32关闭，无裁剪或额外loss。

四个既有energy_equivariant源控制只读取源元数据和完整源曲线，不加载权重、不读取目标成绩。8新加4控制齐全后，按四seed均值(0.5源V准确率+0.5最差源RX准确率)最高选中；完全同分才依次比较V、最差RX、MAC、参数和固定顺序。checkpoint固定E200，不挑最佳轮，不以公共物理误差筛选。

若新候选选中，冻结后默认完成4行新clean预测，并复用24行既有固定控制；全部28行预测完整后独立truth-last评分。若energy源控制保留，沿用其原已验证clean结果，两个新候选测试记N/A，不访问未选query。条件测试run为20261002-phase1-cvs-coupled-clean-manysig-m28-r01，每row168000条、6类、7目标RX；capsule/物理ID/truth路径、控制行和指标见experiment.json。无LEO、SFT、support或新增类，目标评分不回流研发或选择性重跑。

保留详细文本、10000步JSONL、完整epochJSONL、紧凑JSONL/CSV；测量CE/权重/LR/全参数梯度、执行旗标、源V/最差RX、alpha0及固定标量梯度N/A、实际behavior.0.conv输入的三阶/五阶变化和公式误差、六block能量比例、耗时/显存。冻结后保存90个源TX×RX×day单元和30组公共TX/RX级联诊断；公共数据不训练，物理误差不作为排名或停机门槛。ConvLinear MAC不包含额外逐元素lift运算，不能据相同MAC宣称总计算量相同，实际时间另外测量。

56项聚焦检查通过，8个丢弃CPU模型完成24次CE更新，独立P0/P1审查PASS。远端发布后使用已验证CVS Python冷启动检查；每个正式模型先执行真实checkpoint往返无query smoke，再进入source训练。

一个launch owner，独占输出；每GPU最多两个总训练任务、至少12GB空闲。保留历史和partial；不干预其他健康任务，不以低性能停机，不自动重复启动。源分数和公共数学性质不等于clean提升，原目标尚未完成。

[逐行登记](experiment.json) · [资源和路径preflight](evidence/preflight.json) · [本地验证](evidence/local_validation.json) · [独立审查](evidence/independent_review.json)。
