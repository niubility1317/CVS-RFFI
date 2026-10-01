# GroupBarrierJoint：LocalRidge 与 support 微调的下一步结构

当前交付范围是数学设计和独立 gate 求解组件；完整 B→C 方法尚待集成与验证，没有新的准确率结论。候选名称为 `D92-GroupBarrierJointLocalRidge-v1`。它使用合法目标 support，继续固定合规 Phase1 和 practical residual 信道，不开展性能参数网格。

当前不可变 MarginJoint 重复基准已有一行技术失败：C 的普通 KKT 隐式梯度不满足独立紧约束和严格互补的判定。其他健康行继续原排程。失败文字还不能判定具体是哪一项条件不成立；本次修正依据的是求导的适用范围，未读取 query 预测、truth 或成绩。原 run 不能改写为这个新候选，也不能用成功子集报告完整矩阵。

## 三个部件如何联合

| 部件 | 训练输入与更新 | 作用与局限 |
|---|---|---|
| B 旧类适应 | 当前旧类 support；沿用解析 LocalRidge 与小 adapter 的监督目标 | 旧类适应提升由 B 承担；本次 C 的结构修正不会自动改善 B。 |
| 新类条件头 | 只用当前新类 inner-train；固定原旧类尺度的 PSD kernel、解析 Ridge 和自由截距 | 学习新类内部分类，避免重新拟合全部旧类列。 |
| 新旧组 gate 与 C adapter | 全部合法 inner-train；Bernoulli loss、RKHS 正则及旧 support 的逐新类障碍约束 | 学习组间竞争；梯度通过 gate、新头和 kernel 的两端传回继承实际 B 的小 adapter。 |

C 对旧类使用 `logsigmoid(g) + logsoftmax(f_B)`，对新类使用 `logsigmoid(-g) + logsoftmax(h_new)`。所有注册类的概率总和为 1，每个 query 都比较全部注册类。gate 的概率由当前模型预测，注册分组只是已知类 ID；它不读取 query 的旧/新真实身份、真实类数、配额或批量组成。

对任意两个旧类，C 的数学分数差等于 B 的原分数差。因此旧类内部排序得以保留。它不能保证旧类最终准确率不下降，因为新类仍能在统一竞争中胜出；也不能修复 B 原来的旧类内部错误。浮点实现需要保留组内差和跨组比较的代数结构，不能将数学恒等式冒称逐位相同。

## 为什么使用有限障碍

每个旧训练物理点 i 和每个新类 j 都保留约束。下界为 `a_ij = d_i - log p_B(y_i) + log p_new(j)`；`d_i` 是实际 B 补零到新类后的原最小真实类 margin，负值和零都不截断。要求 `g_i > a_ij` 能保留这些训练点相对于新类的原 margin；并不保护未参与训练的 query。

新增 gate 最小化原 Bernoulli 加 RKHS 目标，再减去 `ζ Σ_ij log(g_i-a_ij)`。逐条保留约束可避免 max 在新类并列处的非光滑问题。严格可行区的 Hessian 为正，重复物理特征、并列下界和奇异 PSD kernel 都不要求独立活动约束或严格互补。求解使用 `I + sqrt(D_eff) K sqrt(D_eff)` 的 SPD 分解和自由截距 Schur 补，不求逆 K，不加入 jitter，不使用伪逆。

预先固定平均原 head 目标差预算为 `1e-4`。对于 N 个训练物理点、m 个旧点和 q 个新类，固定 `ζ = N*1e-4/(m*q)`。在精确中心路径驻点，原硬约束目标差不超过 `N*1e-4`。这是目标函数的近似界，不能解释为准确率、参数距离或 query 遗忘界。有限障碍改变了内层方法，须使用独立名称、记录和版本。

实际求解必须另外报告全部正余量、函数驻点残差、截距残差、条件数和实际原始/候选对偶证据。`Σ μ_ij*slack_ij` 的代数值不能单独证明浮点输出已达到理论差界。极端 logit 下的曲率下溢或无法解析的数值条件仍是明确的技术失败，不通过夹紧曲率或零梯度隐藏。

## 从历史源码和论文吸取什么

SCV 已拆分过旧组概率，但它的保护对象是地面 Phase1，部分概率混合不能保证保留适应后的 B 排序。SFHead 的全类重初始化和有限 KD 也没有函数排序保证。D28 的门控来自平方损失，D30 则保持原组赢家。新候选保留实际 B，独立学习有明确概率目标的 gate，并通过新类 Ridge 联合更新 C adapter；它不是给旧门控更换几个参数。

[BiC（CVPR 2019）](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf)指出增量分类中的旧新分数偏置，并使用独立的偏置修正层。这提供了显式处理组间校准的背景；该论文的 exemplar/validation 使用规则不能直接套用本项目。

[R2D2](https://robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf)提供可微 Ridge 学习器的依据；[OptNet](https://proceedings.mlr.press/v70/amos17a/amos17a.pdf)提供通过优化层求导的背景；[凸优化教材](https://www.seas.ucla.edu/~vandenbe/cvxbook/bv_cvxbook.pdf)支撑严格可行障碍与对偶间隙关系。本项目的实际 B 冻结、逐新类 margin、概率分解和完整联合梯度是针对当前协议的推导，不是论文已经证明过的卫星准确率收益。

完整公式、可行性、唯一函数证明、kernel/RHS/自由截距伴随及退化边界见[数学推导](D92_GROUP_FACTORIZED_JOINT_DERIVATION_20261001.md)。

## 验证与报告范围

先验证独立组件的 PSD/奇异输入、全部约束、完整 K/L/下界梯度、自由截距和失败计账，再集成实际 B→C。inner 旧头仅用 old inner-train 重新拟合；实际继承的 `U_B` 已使用合法完整旧 support，所以 C inner-held 是优化用 support，不能称为独立验证。final 才使用当次完整实际 B。新增 0 类直接复用实际 B。

正式候选必须先固定源码和逐行预登记，之后按相同物理 support/query 报告 A、B、C 的完整 K×新增类数表。当前这些准确率、H、适应提升、旧类下降、新旧差距均为 N/A。训练/推理耗时、可训练参数、峰值内存、常驻状态和传输字节也需实际测量；单个求解组件通过测试不代表完整方法省算力或性能提高。


## 已核实的组件验证

独立 gate 组件已在本机原生激活的 `ssr-gpu` 中通过 20 项合成测试，见[完整验证记录](D92_GROUP_BARRIER_GATE_VALIDATION_20261001.json)。包括独立 finite-feature 优化器对照、K/L/每个下界与自由截距的差分、奇异/零核、失败计费，以及 6 旧类、20 新类、K=20 时 520 个物理点和 2400 条并列约束的独立标量对照。

最大规模回归首先发现伴随式相消。现改用缩放坐标直接计算 `D_eff*lambda` 和有界下界曲率比例，保留原残差容差；修订后全部 20 项通过。原失败输出仍保存。这里验证的是 gate 组件，完整新类 Ridge→gate→U/Z 的联合链仍待集成测试，未启动这个候选的真实实验。
