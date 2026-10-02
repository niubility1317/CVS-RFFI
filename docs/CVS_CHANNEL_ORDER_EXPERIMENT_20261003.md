# CVS信道补偿与非线性交互：Phase1结构实验

2026-10-03。用户明确授权依据[架构设计](CVS_PHASE1_CHANNEL_ROBUST_ARCHITECTURE_20261003.md)实现并运行实验。本轮已完成16份E200训练及冻结候选的独立clean测试。以下设计与规则保留发布前约定；结果为未取得测试提升。

## 实现范围

第一轮采用保留现有RFF主干的增量结构：每个候选从零构建`neural_residual_shallow`，保留原始IQ主干、频率路径和分类头，再增加全采样率复数分支。它不是把主干输入全部替换为均衡结果，不声称已完成信道或TX/RX分离。原attention读出计划继续保持未启动，本轮不混入该改变量。

所有候选都只用原单一发射机CE。相同L6300、U56700不使用、V27000、RX1/3/4/6/8、day1/2/3、split392005，原E200×50steps、batch128不丢尾批、AdamW学习率0.0002/weight_decay0.0001、cosine终值0.000001、FP32/TF32关闭。四个modelseed为2026092701至2026092704；splitseed固定，无增强/support/evaluation随机性。没有额外loss、增强、重采样、分阶段训练、teacher、EMA或目标适应。

## 四个固定候选

|候选|原始主干之外的分支|目的|
|---|---|---|
|channel_capacity|F(x)及普通额外复卷积输出|检验额外容量，不使用动态补偿|
|channel_compensated|F(Gₓx)|只增加受限补偿后的辅助特征|
|channel_dual|共享F的F(x)与F(Gₓx)|检验原始/补偿双路|
|channel_order|共享双路，加F(Gₓx)−GₓF(x)交互|检验运算次序交互的增量价值|

已有shallow四份源元数据只作为固定控制；不加载其checkpoint。16份新模型加4份旧源记录构成20行选择矩阵。capacity也有资格胜出，不因论文偏好重排。既有控制若胜出，复用其已完成的测试，不对未选候选访问query。

Gₓ由当前包的logpower均值/标准差和lag1/2/4归一化复自相关共8维上下文，经8→16→8的MLP生成4个复系数。四个5tap复FIR基各自限制复模长ℓ₁范数≤1，系数的复模长和≤1，修正幅度ρ=0.25。因此每个固定输入算子的‖Gₓ−I‖≤0.25；这个界不保证真实信道恢复、跨输入动态网络的全局稳定性或身份信息保留。

F为全采样率复卷积1→8、k5，接真正非线性径向激活，再接8→8、k3及径向激活。算子内部没有RMS归一化、bias或下采样。Gₓ从原x计算一次，在原IQ和F输出的每个通道使用同一组滤波器。两种顺序的公共有效区左右各裁5点，排除padding造成的伪交互。

u/v采用原统计读出，各80维映射到160维。d采用与u的归一化复交叉积实部/虚部均值共16维映射到160维；它在d=0附近仍有线性梯度，避免只读|d|²造成一阶梯度消失。G的系数出口为零、F包含实际非线性、基非零、d出口为非零小权重；u/v新增出口为零。新增初始化隔离RNG，保留同seed主干参数与初始分类函数。最终沿用原分类头的一次L2归一化。

参数接近匹配不等于计算匹配。实际参数、参与梯度的参数、卷积/线性MACs、训练/推理耗时、显存、常驻状态均记录；MACs不当作包含FFT、归一化和逐元素计算的总FLOPs。实测capacity/compensated/dual/order分别为247451/234587/247387/249947参数；capacity较完整结构少约1.00%，不是完全相同的参数或计算预算。

## 选择、测试和声明边界

固定E200，按四seed均值`0.5×源V准确率+0.5×最差源RX准确率`最大者选择，性能完全相同后才依原规则比较成本。不能使用最佳epoch、历史目标分数或新测试分数重排；低分不触发停止或选择性重跑。

新候选胜出才冻结它的4个模型，完成4份新clean预测，与36份既有冻结预测组成40行矩阵。每模型使用相同168000物理query、6个注册TX和7个目标RX；所有预测固定后才独立truth-last评分，完整复算320条ALL/RX评分、80组汇总、72组配对，并核对旧288条评分不变。本轮只测clean，不加LEO、新类、support/SFT。

结构消融在源域完整报告；未选候选不访问目标，不能把源域消融写成目标域模块消融。源V含已见RX，源分数不能证明未见RX泛化。当前输入为equalized=1、center256、unitRMS；结果只支持这份预处理契约下的剩余失真/跨接收条件表现，不直接证明原始多径信道鲁棒性。历史benchmark已有暴露，新结果不是首次盲测。

冻结的公开合成输入仅检验算子性质和运行正确性，不作为新训练样本。若在接收后IQ上施加卷积，明确为等效扰动代理，不能解释为真实改变TX与非线性RX之间的传播信道。任何测试结果不回流本轮结构、超参数或选择规则。

## 执行与证据

源run：`20261003-phase1-cvs-channel-order-identity-manysig-m16-r01`；release：`cvs_channel_order_identity_20261003_r01`。

条件clean run：`20261003-phase1-cvs-channel-order-clean-manysig-m40-r01`；release：`cvs_channel_order_clean_eval_20261003_r01`。

唯一launch owner为主Agent。新输出独占，保护所有旧产物。每GPU最多2个训练进程，已运行健康任务不热改、不停止、不重启。

本地验证包含：参数/实际contract、初始主干与logits、两步CE梯度、补偿幅度界、同采样率线性F在公共有效区的d≈0、非线性F的非零交互、逐包独立和全局相位性质、严格state回读、源选择/输入权限负测和条件truth-last矩阵。依最小流程进行一次独立P0/P1正确性审查。

每步/每epoch保留测量到的CE、LR、梯度、实际模块启用情况、G幅度与交互输出、源V/RX和资源；capacity无G、非order无d的相应字段为N/A，不能用虚构的0冒充测量。诊断作用于最后一个源训练batch，明确口径，不当作整个V分布。

执行状态与真实证据保存在[源run报告](../automation_reports/CV-SincNet/20261003-phase1-cvs-channel-order-identity-manysig-m16-r01/report.md)，本设计不追加目标评分。

## 完成结果（2026-10-03）

双路按固定源规则以97.1301%的综合分入选，比shallow高0.0083个百分点；完整order为97.0963%，未入选。入选双路clean为69.4347%±1.7912%，同核心控制为69.8567%±1.1881%，配对差−0.4220±0.7680个百分点，仅1/4个seed为正。原native为76.2272%，冻结residual_fusion为78.4543%。实现、训练、评分和记录完成；本轮架构性能假设未得到支持，不据目标结果换测候选或重跑。

[源报告](../automation_reports/CV-SincNet/20261003-phase1-cvs-channel-order-identity-manysig-m16-r01/report.md) · [源机制分析](../automation_reports/CV-SincNet/20261003-phase1-cvs-channel-order-identity-manysig-m16-r01/mechanism_report.md) · [完整独立测试](../automation_reports/CV-SincNet/20261003-phase1-cvs-channel-order-clean-manysig-m40-r01/report.md)。
