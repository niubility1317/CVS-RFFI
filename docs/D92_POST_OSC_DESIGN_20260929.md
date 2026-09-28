# OSC 之后的后备设计：单步共享 Cauchy 几何校准

日期：2026-09-29。状态：仅设计，未实现、未启动、未登记正式方法；不修改 OSC 或其他运行中方法。本设计保持对所有历史及当前 target 成绩、结果解释和评分索引的盲态。以下依据来自当前协议与代码结构，不是任何实验反馈。

## 1. 结论与适用范围

只保留一个可检验机制：在固定 D92 identity160+FFT96 特征上，使用全部注册类的真实 support，做**一次联合 Cauchy 重加权**，同时估计各类中心与共同正则化协方差，然后编译统一等先验判别头。该机制没有视图、随机投影、门控网络、候选网格、温度搜索或地面摘要依赖。

它有明确但有限的依据：若部分合法 support 的类内残差过大，共同 Gaussian 均值/协方差会受其影响；同一个重尾目标允许同时降低它们对中心和协方差的影响。原 D81/D92 已使用一次 Cauchy 类中心校准，因此重尾残差处理有现有结构依据，不是凭目标成绩新添任意组件。

**没有依据保证它满足每个 K 的新旧类全面改善。** K1 无法区分类内正常样本与异常样本，闭式退化为球形最近中心；K2 对称初值下，类中心也不会移动。该方向的主要可检验作用在多 support 的估计稳健性。若后续必须先证明一个机制本身为 K1 增加可辨识信息，应该否决本设计作为下一次主实验，而不是再拼接其他组件包装成全面方案。

## 2. 实际核对的代码和权限

已只读核对：

- `code/cvsrffi/stage2_d81_ground_nuisance_cauchy_center.py:96` 的 `translate_to_robust_centers`。第 134 行附近将 shots<=2 固定为 identity；第 137 至 154 行从类均值出发，以 ground nuisance energy 作一次 Cauchy 权重；后续明确检查平移前后 within-class residual 不变。
- `code/cvsrffi/stage2_d92_registration_balanced_covariance.py` 的 `_group_covariance` 和 `build_registration_balanced_equal_lda`。第 186 至 229 行附近独立计算类均值、old/new 组 covariance，再编译等先验共享判别头。
- `code/cvsrffi/stage2_ablation_executors.py:96` 的 `_project_feature_profile`。FFT-only 256 维配置保持 `unit(concat(unit(identity160),4*unit(FFT96)))`。
- 本机 `E:/type10-7/项目.md` §5.3.1、§5.3.2：普通地面原型不能被用于在线拟合 covariance/LDA/持久头；已绑定冻结的量化聚合摘要具有独立且有限的使用权限，不能据普通原型扩展出该权限。

本候选完全不读取地面原型或摘要，只用固定模型和当前 row 的合法 support。所有 source 样本、逐样本 source embedding、source loader、pseudo-source、query 反馈和跨 row 学习状态都不需要也不允许。ground 新增载荷为 0 B。允许使用摘要并不构成必须使用它的理由。

本机制与已有方向的区别是**联合改变类中心和残差协方差估计**：D81 保持 residual covariance 不变；SGJoint 是固定摘要归一化加普通 support covariance；SFHead 是监督判别优化；BNNA 是视图扰动方向上的非线性 gate；MVKME/OSC 保留多视图信息。本机制不声称增加表示信息，也不把单纯可逆线性变换视为创新。

## 3. 唯一固定目标

每类有 n 个真实物理 support，共 C 类，N=Cn，d=256。按物理 ID 固定归约顺序，类 ID 只决定注册标签。输入向量：

`x_i = unit(concat(unit(z_i), 4*unit(f_i)))`，其中 `unit(v)=v/max(||v||_2,1e-12)`。

z_i 是固定 Phase1 对原 received IQ 的 identity160；f_i 是该同一 received IQ 的历史 FFT96。只使用原相位，不增加 view 或编码器前向。该式保留原 D92 256 维结构中的 1:4 块系数；不声称这一系数统计最优，也不根据成绩调整它。

固定参考矩阵 `T=I_d/d`，固定正则强度 `kappa=d`。学习状态为各类位置 `mu_c` 和唯一共享正定矩阵 `Sigma`。定义物理残差 `r_i=x_i-mu_{y_i}`、`delta_i=r_i^T Sigma^-1 r_i`，最小化目标：

`J(mu,Sigma) = (N/2)*logdet(Sigma) + ((d+1)/2)*sum_i log(1+delta_i) + (kappa/2)*D(T,Sigma)`，

其中

`D(T,Sigma)=trace(T*Sigma^-1)-logdet(T*Sigma^-1)-d`。

第一部分是固定自由度 1 的多元 Cauchy 负对数目标，常数省略；D 是两个零均值 Gaussian 的协方差散度的两倍，不是从 source 估得的先验统计。自由度 1 继承已有 Cauchy 结构，不扫描自由度；kappa=d 表示每个特征方向一个固定球形正则单位，不由表现选择，也不宣称为经验 Bayes 最优。

所有类别相同 n，所以每个物理样本等权等价于每类等权。没有 old/new 角色系数，old membership 仅允许用于 OOF 诊断。N 是真实物理样本数，不以 view 数或协方差自由度替代；该目标是正则化似然目标，不是无偏协方差估计。

## 4. 只做一次 MM 更新

这里的 MM 指构造一个与原目标相切的上界，并精确最小化它。只做一次更新，不增加迭代步数超参，不按目标值挑选状态或早停。

初值：

`mu_c^0 = mean_{i:y_i=c}(x_i)`；

`Sigma^0 = [sum_i (x_i-mu^0_{y_i})(x_i-mu^0_{y_i})^T + kappa*T] / (N+kappa)`。

这是固定球形正则下的共同 Gaussian covariance 初值。它在任何有限输入下正定，不需要按矩阵 rank 切换算法。

计算 `delta_i^0` 后，固定权重：

`w_i = (d+1)/(1+delta_i^0)`。

同一次更新中：

`mu_c^1 = sum_{i:y_i=c} w_i*x_i / sum_{i:y_i=c} w_i`；

`Sigma^1 = [sum_i w_i*(x_i-mu^1_{y_i})(x_i-mu^1_{y_i})^T + kappa*T] / (N+kappa)`。

分母是 N+kappa，**不是 sum(w)+kappa**。前者来自原目标 logdet 的系数，后者会改变目标。类中心先更新，covariance 必须使用更新后的中心残差。不得把类内权重归一化后再用于 covariance，从而偷偷改变同一目标。

下降依据来自 `log(1+u)` 对 u 的凹性：用在 `delta_i^0` 处的切线上界后，目标变为带固定权重的共同 Gaussian 二次目标，以上位置和 covariance 是该上界的最小解。因此在精确算术下 `J1<=J0`。这不证明达到全局最优，也不证明分类误差下降。若数值实现出现超出舍入容差的目标上升，应作为技术失败检查公式和求解；不能借机改超参、回退候选或读 query。

两个 covariance 都满足 `lambda_min(Sigma)>=1/(N+d)`，因为 `kappa*T=I_d`，残差项半正定。不存在除以近零 residual trace 的操作；全零 residual、低秩和相消输入都有同一闭式处理。

## 5. K1、K2 与最终判决

K1 时 `mu_c=x_c`，所有训练 residual 为 0，`Sigma=I_d/(C+d)` 是上述目标的闭式最优 covariance。跳过 MM，记录 `NO_WITHIN_CLASS_EVIDENCE`、0 次更新，无 holdout。不得将不同类别当作同类重复观测来估计额外类内信息。

K2 时每类初始两个 residual 互为相反数，因此同类两点的 delta 和 w 相同，`mu_c^1=mu_c^0`。共同 covariance 仍可因各类 residual 的长度不同而改变，但不能宣称已经完成稳健中心纠偏。

K>=2 固定使用更新后的状态一次，不选初值与更新值中的“较好者”。编译：

`W_c=(Sigma^1)^-1*mu_c^1`；`b_c=-0.5*(mu_c^1)^T W_c`；

`score_c(x)=x^T W_c+b_c`。

所有类使用 prior 1/C。这个 argmax 等价于共享 Sigma 的 Cauchy likelihood 排序，因为 `log(1+delta)` 对 delta 单调；但 affine softmax 数值不是 Cauchy 后验。OOF 若报告 affine-softmax NLL，必须标明只是固定判别分数的诊断，不能冒称已校准的 Cauchy likelihood。query 公共二次项对 argmax 可删除，最终不用持久保存 Sigma 或 support。

类分数精确相等时以物理类 ID 稳定裁决。非有限输入/计算、Cholesky 失败一律技术报错。零向量保留零，不构造补充样本；全零样本使 W=b=0。

## 6. 完整 physical CV 与可证伪验证

K>=2 固定 `F=min(K,3)`，每类按物理 ID 排序后 modulo F 分 fold。每折只用 trainfold 的物理样本，从 `mu0`、`Sigma0` 开始，重新计算全部 delta、w、mu1、Sigma1、W/b。held 样本不得进入初始化、权重、参考尺度、covariance 或任何日志中被解释为训练统计的项。

各 fold 类内训练 n 一致，不同 fold 可以不同；N=Cn 按实际训练集合计算。若某折 n=1，它执行上述 K1 闭式，无 MM。OOF 每个真实物理 support 恰一次，报告各类 NLL、macro NLL 与 old/new 分组诊断，不进行选参。最终用全部 support 从相同固定初值重新拟合一次。K1 无 CV；整个方法没有跨 row 状态。

所需合成验证：

1. J 的实现与独立直接公式一致；单步 MM 不增加目标，且更新后的 weighted class residual sum 为 0。
2. 检查 covariance 分母 N+d、初末中心区别、物理样本计数和最小特征值边界；不能只测试代码返回的 metadata。
3. K1 闭式、K2 类中心不动、全零/低秩/重合点都符合公式。
4. 独立二次距离、Cauchy likelihood 排序和编译 affine argmax 一致；不要求不成立的 softmax 概率等价。
5. class/support 顺序等变；query 顺序、batch 分块不改变逐样本结果，state 字节不变。
6. 扰动某 fold 的 held 样本，其 trainfold 的初值、权重、最终 W/b 均不变。
7. 构造一类中心加离群点的已知合成集合，核对中心/协方差影响确实改变；同时构造无离群数据说明不保证提升。不能用合成任务挑自由度或正则强度。

任何真实评价仍需要另外授权并预登记、冻结预测、独立 truth-last scorer。本设计不授权额外真实 run。

## 7. 预算、字节与日志

每个拟合最多一次 MM 更新。K>=2 最多 F+1 次拟合；每次两套 covariance 分解/求解，K1 为一次闭式。主要成本为 `O(N*d^2+d^3+C*d^2)`，内存 `O(N*d+d^2+C*d)`，无自动微分和优化器。query 编译后约 C*d 次乘加；C26 时 6656 次。没有新的多视图成本，冻结编码器每个物理样本仅一次前向，FFT 一次。

float32 原相位 identity160+FFT96 cache 为每物理样本 1024 B，注册/物理 ID 元数据另算。已有四相位 cache 如经 provenance/schema 核对，可只读取相位 0 与 FFT；不能复用 BNNA/OSC 的任何拟合状态。

持久数值状态只保存 float64 W[C,256] 与 b[C]，为 `8*C*257` B，C26 时 53456 B。类表、配置和日志按实际 UTF-8 字节另报。每个 256×256 float64 covariance/Cholesky 为 524288 B；实际峰值须测量，同时存活数组与 BLAS 工作区不能忽略。本设计阶段没有运行测试，不能给出实测秒数或 RSS。

地面新增 payload=0 B；已有固定模型包实际大小单列，已部署时增量模型传输为 0 B。Phase1、原型及摘要均不更新。

日志记录每 fold/final 的真实 n/N、初始/最终 J 及各分量、w 范围/每类有效样本数、中心位移、covariance trace、真实更新次数、分解/权重/中心/covariance/编译耗时、W/b 实际 nbytes 和 RSS。没有 epoch、学习率或梯度；不伪造这些字段。source validation 为 null 并说明无 source 访问。

## 8. 失败模式与否决条件

一个异常 physical support 可能实际上是该类有用的少数模式；重尾权重会损伤这种信息。高维强收缩可能掩盖区分方向；单步更新可能不足以达到好的稳健估计；初始均值和 covariance 仍可能被污染；保留原 1:4 块尺度也保留了其结构风险。本方案只做一次更新，不能在结果不佳时暗中多跑迭代。

最关键的限制是 K1：没有额外观测或可信分布信息时，本机制没有依据重建缺失的类内结构。用户的全 K 全面改善目标需要实验检验，不能从本设计推导。若主任务需要优先突破 K1，而不是验证 D92 的多样本中心/残差估计是否为限制，应保留本文件作为有边界的后备假设，并拒绝仅因需要“下一方案”而启动它。
