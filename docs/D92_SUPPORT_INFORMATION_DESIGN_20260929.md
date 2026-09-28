# K1 信息边界与唯一选择：物理等权多视图判别回归

日期：2026-09-29。状态：§10 的 D92-MVRidge-v1 已实现核心并通过合成验证，本实现任务未启动真实实验、未修改其他冻结方法。未读取历史或当前 target scores、结果解释、评分索引或根交接中的成绩。§2 至 §9 保留对 Gaussian 随机效应候选和摘要先验重复性的否定分析；最终仅选择 §10 的“物理等权多视图共享 ridge 判别头”，不并列实施其他候选。

## 1. 判断

同一 received IQ 的确定性 view 能揭示固定编码器在指定变换下的响应方向和幅度。相较只保留原相位的一条特征，这确实可能提供额外的可用表征；它不增加原始观测的信息，也不增加物理样本数。现有 MVKME、BNNA、OSC 已读取同一四相位响应，不能再次把这些响应称为新信息。

K1 可确定的是当前单个观测的有限扰动轨道，不能由它识别未见物理样本的类内方差、域间差异或类均值估计误差。把轨道拟合为 Gaussian 随机效应仅改变表示和归纳偏置，不能改变这个可辨识性边界。原始轨道多峰时，Gaussian 矩压缩还会丢失结构。

现有合法量化摘要包含额外的跨 source 接收域聚合信息，具有独立的固定先验价值；但该方向已进入 D92/SGJoint 的实现，不能因换写为 covariance prior 就声称发现了未使用的信息。否定这些信息增量主张，不等于否定 K1 标签对共享判别边界的约束。§10 给出利用这一明确结构差异的单一目标，不以“已识别类内 covariance”为前提，也不保证全 K 改善。

## 2. 可辨识对象与不可辨识对象

把第 i 个物理样本的四个固定特征记为 x_iv，v=0,1,2,3。它们来自同一完整 received IQ 的四分之一圈相位旋转；编码器、归一化和 FFT 都冻结。

对这个有限集合，以下量无需独立同分布假设就能精确计算：

`m_i = (1/4)*sum_v x_iv`；

`V_i = (1/4)*sum_v (x_iv-m_i)(x_iv-m_i)^T`。

V_i 是“均匀抽取一个已定义相位时”的有限轨道二阶矩，不是从四个独立物理样本估得的无偏 covariance。分母是定义中的 4，而不是自由度修正后的 3；也不能把 V_i 再除以 4，当作 m_i 对真实类中心的不确定度。

若写一般分层模型 `x_iv=theta_c+u_i+e_iv`，则 K1 时 theta_c 和 u_i 可任意补偿：`theta_c'=theta_c+a`、`u_i'=u_i-a` 给出相同观测。视图差只能消掉这两个项，不能将它们分别识别。即使设定 `sum_v e_iv=0`，得到的也只是当前物理样本轨道中心 `theta_c+u_i`。跨物理随机截距的分布仍需额外假设或独立数据。

因此，只能把 V_i 当作局部变换响应，或者明确把它作为一种工作模型中的扰动 covariance；不能把后者解释成已经估出了真实跨样本噪声。

## 3. 被否决的 Gaussian 候选：完整定义

为严格比较，保持已有 MVKME 全长四相位分支的输入和尺度：

`x_iv = concat(unit(z_iv),unit(f_i))/sqrt(2)`，d=256，`unit(a)=a/max(||a||,1e-12)`。

z_iv 是四相位的冻结 identity160；f_i 是该同一 received IQ 的 FFT96。不同 v 共用 f_i。无新增 view、相位网格、裁剪或模型训练。

把有限分布 `p_i=(1/4)*sum_v delta_{x_iv}` 替换为工作分布 `q_i=N(m_i,V_i)`。这也可写为随机效应 `X_i=m_i+L_i a_i`，其中 `a_i~N(0,I_3)`，`L_i L_i^T=V_i`。rank 至多 3。a_i 是人为引入的连续扰动变量，不是另外观测到的 source 或 target 样本；算法只解析积分，不生成虚拟样本。

使用已有 MVKME 的 bandwidth=1，不新增搜索参数，定义：

`k_G(i,j)=E_{X~q_i,Y~q_j} exp(-||X-Y||^2/2)`。

令 `Delta=m_i-m_j`、`A=I+V_i+V_j`，Gaussian 积分给出：

`log k_G(i,j) = -0.5*logdet(A)-0.5*Delta^T A^-1 Delta`。

对类别 c 的 n 个物理 support：

`score_c(q)=logsumexp_{i:y_i=c}(log k_G(q,i))-log(n)`。

所有注册类使用同一规则，prior 相等；精确并列按物理类 ID 决定。K1 和 K>=2 是同一公式，不把 view 当作四个 support，不除以虚构的有效样本数。query 的 m_q、V_q 只由该 query 自己的四个冻结 view 计算；不更新 support、尺度、类状态或任何其他 query。

以分布之间的内积或积分定义核具有既有统计依据，Gaussian 情况可解析计算；这只支持“公式是一个合理定义的分布相似度”，不支持它适合本数据或能增加 K1 信息。[Jebara、Kondor、Howard，Probability Product Kernels，JMLR 2004](https://www.jmlr.org/papers/volume5/jebara04a/jebara04a.pdf)

上述 expected RBF 公式由 Gaussian 差分分布及完成平方直接推出。奇异 V_i 不影响 A 正定；V_i=0 时退化为均值之间的普通 RBF。四 view 全部重合时没有任何额外可用方向。

## 4. 与 MVKME、OSC 的严格关系

实际核对 `code/cvsrffi/stage2_d92_mv_kme.py::build_fourier_blocks` 和 `configs/d92_mv_kme_frozen_20260928.json`：MVKME 的全长分支是四相位随机 Fourier 特征的平均，固定 128 个频率、bandwidth=1；另有原始分支和 12 个局部 view 分支。它没有把四相位简单压成原空间均值。

对全长分支，若使用精确 Gaussian RBF 特征映射 phi，则：

`<E_{p_i} phi(X),E_{p_j} phi(Y)> = (1/16)*sum_{v,w} exp(-||x_iv-x_jw||^2/2)`。

当前 Fourier 分支是这一核均值的有限维近似。Gaussian 候选只是将 p_i、p_j 换成具有相同一二阶矩的 q_i、q_j，再解析计算近似核；并非引入了新观测。精确 Gaussian 核的分布嵌入具有可区分分布的理论依据，但固定有限频率的实现不能自动继承完整的可辨识保证。[Sriperumbudur 等，Hilbert Space Embeddings and Metrics on Probability Measures，JMLR 2010](https://www.jmlr.org/beta/papers/v11/sriperumbudur10a.html)

两种实现并非数值相等：有限 Fourier 误差与 Gaussian 矩近似误差不同，后续 ridge 分类器与物理 support 混合打分也不同。因此不能断言二者预测必然相同；同样，不能把这些差异当成 Gaussian 候选有更多信息或更高准确率的依据。Gaussian 近似可能消掉对 TX 区分有用的多峰结构。

OSC 保留有序四视图并比较四种循环对应关系；该 Gaussian 候选只保留矩，任意 view 置换都不改变它，因而还会丢失 OSC 使用的循环相对位置。它获得的是不同不变性，不是更丰富的信息。

结论：该候选是已有扰动核思想的另一种近似和分类头选择，缺少足够独立的机制依据，暂不实施。

## 5. 现有合法摘要能提供什么

只读核对 `code/cvsrffi/phase1_center_lowrank_prototype_bundle.py`、`code/cvsrffi/d92_ground_summary.py`，并与入口 Agent 交换代码结构事实：

- schema 为 `int8_domain_class_center_lowrank_residual_radius_v2`，feature schema 为 `ADV3B02:z_id:unit_l2:160:v1`。
- 实际成员为旧类 identity160 的 int8 core、每类 rank=3 的低秩 domain residual basis、domain coefficients、domain×class radius，以及 FP16 scales 和冻结 registry。没有 FFT 分布或交叉 covariance。
- radius 定义为到对应 Phase1 domain-class centroid 的 P90 cosine distance。它不是方差、标准差或样本量；一个分位数无法唯一确定 covariance 或类中心标准误。
- 当前严格 allowlist 没有 phase/time-shift 配对响应、逐样本 ID、源样本数、BatchNorm 状态，也没有可选全局 location/scale 字段。
- 来源代码链为 `post_stage_cli.py` 的 `wisig_domain` 默认 rx_day，经 native SSDG dataset 参数传到 `dataset_wisig.py` 按 RX/day 编号；当前 recipe 没有覆盖该默认。因此低秩方向表示接收机×日期条件下的聚合中心变化。不能仅凭 sampler 的 domain_key 证明该来源。摘要自身 registry 为 opaque 数字字符串，不能独立解出具体接收机和日期；不能将这些域变化改称已测量的纯相位或时间扰动。

本机协议 §5.3.2 允许已绑定冻结摘要作为固定分布锚点或 normalization 参考；§5.3.1 普通类原型不具有在线拟合 covariance/LDA 的权限。本研究不读取实际 source 或重新导出 ground。摘要存在跨 source domain 的信息，不意味着已经识别了 target 卫星物理噪声或新 TX 的 covariance；将这些方向迁移给所有注册类仍是明确的模型假设。

## 6. 固定域先验是否已被现有方法覆盖

实际核对 `stage2_d92_summary_joint.py::build_summary_operator`：它逐域读取合法摘要，对归一化的同类中心计算 radius 加权域均值和类等权 residual geometry G，然后构造固定算子：

`A=(I+160*G/trace(G))^(-1/2)`，退化时 A=I。

`::_transform` 对 target identity 使用该算子并重新单位归一化，随后 `::_statistics` 从当前训练 support 重估物理残差 covariance，`::_fit_from_statistics` 加球形收缩。因此“固定 source 域变化方向 + 当前 row support residual + 所有类同公式”这一主要机制已在 SGJoint 中存在，K1 也已使用固定摘要归一化。其数值配置和归一化细节与其他 prior 写法可能不同，但不能再次作为全新信息来源。

代数上，如果暂时去掉逐样本再归一化，变换 y=A x 后使用 `(1-lambda)*A S A^T+lambda*tau*I`，对应原坐标的 covariance 正是：

`(1-lambda)*S+lambda*tau*A^-1*A^-T`。

所以把这种 whitening 后球形收缩改写成原坐标的固定非球形 prior，可能只是完全等价的写法。若 lambda=0 且 A 可逆，共同 Gaussian LDA 更是精确坐标不变。真实 SGJoint 含逐样本再归一化，因此不能把它与任意 additive-prior 实现声称为严格相等；但去掉或增加这一操作属于归纳偏置变化，仍需独立理由，不能从名字推导信息增量。

D92 的 ground nuisance spectrum 主要参与 support center 的 Cauchy 权重，且类平移保持 within-class residual 不变；它不是上述完整 prior covariance 机制。SFHead 不使用摘要，只有 support 判别损失、旧类 support 上的冻结 teacher 条件蒸馏及参数正则。**非重复性比较的决定性项是 SGJoint 已覆盖固定域先验与 support covariance 的组合，而非声称三个方法都相同。**

在没有新的合法字段或可检验的结构假设前，本研究不把已有摘要再包装成新的层级随机效应方案。

## 7. 时间移位为什么尚不足以立项

代码事实表明 native identity160 不是理论相位不变：Sinc 的同核前端之后存在一般实 1×1 Conv、GroupNorm、ReLU，时间分支仍进入 joint feature。因此相位 view 可以改变模型响应，但这不证明这些变化是独立的物理 nuisance。

对全长 received IQ 作循环时间移位，FFT magnitude 具有已知的离散循环移位不变性；但实际有限片段的首尾通常不连续。循环拼接引入的边界不能在没有信号周期/采样起点协议依据时被称为真实接收机扰动。非循环移位加零填充会裁掉信息并改变边界，已与 MVKME 的局部窗口思想部分重叠。时间反转或共轭还可能改变真正携带 TX 信息的结构。

因此暂不规定时间移位网格。若未来协议或冻结模型实现明确给出某个变换群的标签不变性及边界处理，才有理由讨论它相较现有 phase/crop view 的额外冻结模型响应；当前结构事实不够。

## 8. 若只作数学验证，隔离和成本应如何定义

这不是实施授权。为使被否决候选仍可核查，完整约束如下：

- K1 直接比较 query 与各类唯一物理 support 的分布核，无 holdout、无 CV、无参数拟合。
- K>=2 使用 F=min(K,3)，按每类物理 ID 排序分 fold；同一物理样本全部 view 一起排除。训练 state 只含 trainfold 的 m_i、L_i 和标签。因为这些统计只依赖单个物理样本，预计算缓存不混入 held 信息；仍必须重新组装 trainfold 状态，不能使用任何全 support 共同尺度。
- CV 仅诊断，不选 bandwidth、Gaussian/经验核或其他候选；不把单个 view 留出伪称独立验证。最终状态只取当前 row support，query 不写状态。
- 数学验证包括有限 view kernel 双重求和恒等式、Gaussian 积分与独立数值例子的对应、低秩与稠密线性代数等价、V=0 退化、物理 fold 隔离、所有类同公式及 query batch 独立。不得据合成或目标效果调 bandwidth。

利用 `rank(V_i)<=3`，A 的 inverse/determinant 可经最多 6 列的 Woodbury/determinant-lemma 求得；FFT 在四相位中相同，因此 covariance 因子只需保存 identity 部分。每个物理 support 保存 float64 m_i[256] 和 L_i[160,3]，为 `8*(256+480)=5888 B`；C26/K20 合计 3061760 B，另加标签/物理 ID 元数据。这个 support 状态在卫星本地生成，新增 source 传输仍为 0 B。

每 query 要比较 C*K 个物理分布，主要成本为 `O(C*K*(d*6^2+6^3))`，不能像 OSC 一样只随 C 扩张；只计算轨道均值和 covariance 并不保证整体更快。直接经验核需要每对物理样本 16 个 256 维距离，其预算同样可精确核算；Gaussian 近似并不存在无需测量就能宣称的巨大成本优势。若读取既有四相位特征 cache，核心无需额外编码器运行；从 IQ 重新提取则仍需每物理样本 4 次冻结前向和 1 次 FFT。

以上是算术与存储上界，不是实测时间/RSS。本次只完成设计分析，没有运行这一候选。

## 9. 对新增信息主张的否定结论

有限 view 能产生可用的固定模型响应，但本次候选使用的响应已经由现有方法读取；Gaussian 矩模型没有新增 K1 信息。合法 ground 摘要确有额外域变化信息，但固定 normalization/方向先验与 support covariance 的结合已经被 SGJoint 实现。当前不应以统计名称、矩压缩或坐标重写为理由另开方法。

保留这些分析作为可辨识性边界和非重复性证据；不能由等待时间或方法轮次推动候选数量增长。接下来只考虑一个不同问题：全部真实 support 标签如何共同约束一个对 view 一致的判别头，而不是如何从 K1 估计不存在的物理重复样本。

## 10. 最终唯一机制：物理等权多视图共享 ridge

### 10.1 为什么这个问题在 K1 仍有约束

K1 表示每类一个有标签的物理样本，不表示整个注册问题只有一个标签。C 个不同类的真实标签仍能共同约束一个共享判别边界。四个固定 view 只能为同一标签提供变换一致性约束；它们不增加物理计数，也不产生独立验证。

原 D92 不能整体概括为最近中心：它还含 old-only metric、ground robust center、shared covariance 和组件组合；其少样本某些组件才退化为单位 covariance。SFHead 的实际 `_features` 只取 identity160，其目标是 old/new 组各 0.5 的 CE、旧类 support 的条件 KD，以及向 10 倍 support 均值初值收缩；没有 FFT 或多视图监督项。因此 SFHead 不包含下面这个物理等权目标。

MVKME 在随机 Fourier 映射后先平均 view，再做 ridge。下面目标在同一个线性头上对各 view 的预测分别计算平方损失，其精确分解中多出显式 view 一致性正则；这不是“先求 view 均值再做相同 ridge”的等价重命名。OSC 是循环模板生成式匹配，没有用全部注册标签做共同判别回归。

### 10.2 唯一特征与目标

保持原 D92 的固定尺度，不另搜 identity/FFT 权重：

`x_iv=unit(concat(unit(z_iv),4*unit(f_i)))`，d=256，V=4。

四视图仍为完整 received IQ 的四分之一圈相位，FFT 广播同一份。除数为最终向量自身范数；两个块均非零时等价于除以 sqrt(17)。所有归一化下界固定为 1e-12。不同 v 不生成第二个 LEO 观测。

设 `t_i=onehot(y_i)-1/C`，共享参数 W[d,C]、b[C]。唯一目标：

`L(W,b)=0.5*sum_i [(1/4)*sum_v ||x_iv^T W+b-t_i||^2] + 0.5*||W||_F^2`。

每个物理样本总权重为 1，所有类有相同 K，故每类等权。无 old/new 权重、teacher、ground、样本复制权重、标签平滑搜索或原型初始化。没有单独设定的 view 正则系数：它由上述同标签损失的恒等分解自然确定。

ridge 系数固定为 1。因为 ||x_iv||<=1，一个物理观测对固定 b 时的 W 数据曲率 `(1/4)*sum_v x_iv*x_iv^T` 不超过 I，单位 ridge 以这个预先固定的特征尺度定义一份各向同性曲率正则。该解释不是声称最优，也不是估出了一份真实先验样本。若将数据损失改写为物理平均，对应系数是 1/N；不得再按 V 除一次，或在实际实现时悄悄改成同一个平均损失下的系数 1。

### 10.3 闭式解与 view 信息的实际作用

令 `m_i=mean_v x_iv`、`mbar=mean_i m_i`，V_i 为 §2 定义的有限轨道矩。平方损失的恒等分解为：

`L=0.5*sum_i ||m_i^T W+b-t_i||^2 + 0.5*sum_i trace(W^T V_i W) + 0.5*||W||_F^2`。

第二项惩罚同一物理样本 view 响应造成的 logit 变化；只有 W 在该扰动方向上敏感时才付出损失。它是确定性的判别正则，不是类内物理噪声估计或均值标准误。FFT 输入跨 view 相同；当各 view 的归一化 identity 块范数相同（通常均为 1）时，最终联合归一化因子也相同，V_i 的 FFT 块和交叉块在精确算术下为 0。若部分 identity view 为零或落在归一化下界内，不同 view 的最终归一化因子可能不同，V_i 可以有 FFT 成分；核心继续按唯一 unit 公式处理，不强制把这些成分清零。

由于所有类 K 相同，`sum_i t_i=0`。于是：

`G=I+sum_i (m_i-mbar)(m_i-mbar)^T+sum_i V_i`；

`H=sum_i (m_i-mbar)*t_i^T`；

`W=G^-1 H`，`b=-mbar^T W`。

G 最小特征值至少为 1；中心化只会减小未中心化总二阶矩，而每个物理特征范数至多为 1，故 `condition_number(G)<=N+1`。用 float64 Cholesky 和多右端求解，不显式求 inverse，无优化器或迭代预算。实现可直接由全部中心化 view 的二阶矩构造 G，避免用两项大数相减。

最终 query 独立计算 m_q，返回 `m_q^T W+b`。使用逐样本固定归约保证分块不影响结果；所有类同一条公式，精确并列以物理类 ID 裁决。判别分数不是已校准的概率，不新增温度步骤。

这个解在 K1 通常也不等于最近中心：G 含所有类别的 between-support 几何，并耦合各类标签对应的线性解。中心化后的多类 ridge 有 label-space 耦合；它与只用物理 within-class covariance 的 LDA 并非一般相同。即使所有 V_i=0，该共同判别拟合仍存在；因此立项不依赖尚未测量的 phase 响应是否足够大。V_i=0 时新增的一致性项应精确消失，不能虚报 view 收益。

### 10.4 统一 K、隔离与失败模式

K1 使用完全相同的闭式目标，无 holdout。K>=2 使用 F=min(K,3) 的 per-class physical-ID modulo folds；held physical 的全部 view 一同排除，每折从 trainfold 重算 mbar、G、H、W、b。标签编码维度始终是全部注册类 C，类内训练 K 一致，按实际 N 计算损失。

CV 仅记录 OOF 诊断，不选 lambda、模型分支、尺度或步数。最终用全部当前 row support 再解一次，无跨 row 参数复用。old membership 只供诊断；fit 的公式不接触 query、source 或 ground。

主要失败模式：相位扰动方向可能与 TX 判别方向重叠，一致性项会同时压制有用信息；有限 support 的平方损失可能不如生成式距离适合测试分布；固定 1:4 块尺度可能并不理想；边界变化没有泛化改进保证。该方案不靠拟造更多物理观测规避这些问题，也不承诺全面改善。

### 10.5 最小预算、存储与验证

每个 fit 一次 d×d Cholesky；K1 1 个 fit，K>=2 为 F+1 个 fit。没有额外相位或随机频率；可复用合法既有四相位 identity/FFT cache。无新 source payload。临时二阶矩内存 O(d^2)，主要计算 O(4*N*d^2+d^3+d^2*C)，query 特征提取后仅 O(d*C) 打分。

最终 float64 W+b 为 `8*C*(256+1)` B，C26 为 53456 B；不持久保存 support/view bank，mbar 已编译进 b。cache 仍为每物理样本 float32 identity[4,160]+FFT[96] 的 2944 B；模型、cache 文件容器及元数据大小分列。没有依据现在填实测秒数/RSS。

必须进行的合成检查是：显式四 view 目标与均值/轨道矩分解一致；闭式解与独立 stacked ridge 解一致；目标解析梯度在解处接近 0；ridge 系数及 physical/view 权重正确；K1 与多类最近中心存在非等价例子；V=0、全零、重复 view、低秩退化正确；registered/support 排列等变；held physical 扰动不改变对应 trainfold 全部状态；query batch/顺序与 state 字节不变；真实 W/b nbytes 和各拟合阶段时间日志准确。

**最终选择仅此一个机制。** 它使用了尚未被现有几个目标完整覆盖的“全类等物理权判别监督 + 显式 view 一致性”组合，不声称提供新的物理信息。此前 Gaussian 矩候选继续否决，ground prior 坐标改写继续排除。真实运行由主任务另行安排。

### 10.6 实现与合成验证记录

核心 `code/cvsrffi/stage2_d92_multiview_ridge.py`、测试 `tests/test_d92_multiview_ridge.py` 和冻结配置 `configs/d92_multiview_ridge_frozen_20260929.json` 已落盘。`fit_multiview_ridge` 只接收当前 support 的四相位 identity、FFT、标签、物理 ID 和注册表；`old_classes` 仅用于 OOF 分组诊断。内部按物理类 ID 规范排序，返回 W 的类别列与 `state.classes` 保持调用方注册表顺序。

9 项核心合成测试通过，包括独立增广 least-squares 解、展开目标/解析梯度/有限差分、均值与轨道矩损失分解、全状态 physical fold 扰动隔离、类/物理样本/view 顺序、query 分块与只读状态、K1 非最近中心实例、全零/SPD/存储及部分 identity view 为零的实际联合归一化边界。入口 Agent 另报 13 项入口测试通过；本记录不把这些合成测试解释为真实任务性能证据。

每个 fold/final 审计记录 `loss_data_mean`、`loss_view`、`loss_ridge`、`loss_expanded_data`、`loss_total`、实际解析 W/b 梯度范数、normal-equation residual、N/4view 权重、ridge 系数、目标编码范数和分阶段时间。优化器步数为 0，OOF softmax NLL 明确为固定线性判别分数的诊断，不作后验校准或跨方法自动选择。

本机 `ssr-gpu` Python、2 个 BLAS 线程、`default_rng(1801)` 顺序生成 float32 `[520,4,160]` identity 和 `[520,96]` FFT，C26/K20 一次完整 fit 外层测得 0.0670350 s，核心 0.0483294 s。三个 fold 与 final 的实际 K 为 `[13,13,14,20]`。各 fit 累计 center 0.0016900 s、Gram 0.0109655 s、solve 0.0079915 s、objective 0.0075748 s；OOF score 0.0059734 s。

最终解析梯度范数 6.68798e-14，normal-equation residual 9.03806e-15，W+b 实测 53456 B。通过 Windows `GetProcessMemoryInfo` 读取的进程 working set 为 41537536 B、peak working set 为 54951936 B；这是含 Python/NumPy/输入与 BLAS 的整个测量进程值，不是单独算法增量。测量未包含冻结编码器、cache 文件读写或远端运行，不能保证其他设备时延。
