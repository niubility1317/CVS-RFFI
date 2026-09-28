# D92-SGJoint-v1固定设计

本方法仅使用当前row合法物理support、固定Phase1编码器的identity160、同一received IQ的FFT96，以及已经存在并与checkpoint、类别和特征schema绑定的5.3.2量化摘要。没有source样本、teacher、query拟合、跨row适配状态或新统计生成。本文为实现规范，不包含实验性能结论。

## 权限与接口

`build_summary_operator(summary)`只接收`load_ground_summary`已加载的`GroundSummary`，不打开文件。入口核实checkpoint和联合冻结来源；核心再次校验v2 schema、160维、量化数组形状和值及不可写状态。普通5.3.1原型不能替代此摘要。

`fit_summary_joint(support_features, support_labels, support_ids, classes, old_classes, summary_operator)`仅接收本row的support。`old_classes`为显式物理类别ID集合，不依赖注册表前缀。返回只读state，`score(features)`逐样本返回全部注册类分数；`predict(features)`返回输入`classes`中的列号，精确并列按物理class ID排序决胜。`audit_dict()`提供独立、可JSON序列化的审计副本。

## 冻结摘要归一化

逐域读取单位化中心u_dc与P90余弦半径rho_dc，定义q_dc=1/(1+rho_dc)，a_dc=q_dc/sum_d(q_dc)，m_c=sum_d(a_dc*u_dc)。G按地面类等权累计sum_d a_dc*(u_dc-m_c)(u_dc-m_c)^T；A=(I+160G/trace(G))^(-1/2)。G为零时A=I。P90半径只作固定有界可靠性权重，不解释为方差。域和类按物理ID规范排序后归约，矩阵不受注册表排列影响。

此算子是类无关、固定normalization参考。只进行两次逐域读取，不生成虚拟样本，不给分类头喂地面类中心，不持久化反量化域类bank。摘要保持不可变。

合成退化测试发现：完全相同域中心在加权求均值时也可能产生约1e-32的舍入残差，直接除以trace会放大它。因此固定trace<=1e-24按零处理；这是单位化特征1e-12数值范数下界的平方，同样用于support协方差退化判断，未使用任何真实数据选阈值。原始trace和是否退化均记录。

## 联合特征与等先验判别头

x=[unit(A_beta*z), alpha*unit(f)]/sqrt(1+alpha^2)，其中beta为0时A_beta=I。alpha=0时直接使用160维身份特征，否则使用256维。FFT范数为零时FFTblock置零并记录；身份特征零范数直接失败。历史288维传输中最后32维不参与决策。

每个训练fold有每类n个真实support。每类均值采用同一平均公式；共享协方差S=sum_c,sum_i(x_i-mu_c)(x_i-mu_c)^T/[C(n-1)]。n=1或S=0时使用S=0,v=1/d；其他情况v=trace(S)/d。Sigma=(1-lambda)S+lambda*vI。

每类分数为x^T Sigma^(-1)mu_c-.5 mu_c^T Sigma^(-1)mu_c，先验统一1/C。省略类公共先验常数，去除类公共仿射项后，按本fold训练分数的类中心化RMS缩放；下界1e-12。训练与验证中不使用旧新组权重改变协方差或类别先验。

## 固定选择规则

alpha={0,.5,1,4}、beta={0,1}、lambda={.1,1}，共16项。K>=2时F=min(K,3)，每类按opaque物理support ID排序，再按位置模F分折。每个support恰好留出一次。K不能被F整除时，按各fold实际n训练，不冒用全row的K。

每候选以完整out-of-fold分数求一个全类共用的逆温度gamma，范围[.1,10]，最小化max(L_old,L_new)；单旧类任务只用L_old。组内NLL按真实held support平均，完整OOF内每类均为K个，因此等于逐类平均。温度通过有界一维优化，容差1e-8、最多200次，并显式比较边界及gamma=1。gamma是分数乘数，而非除数。

候选按最坏组NLL、全类macro NLL选择；比较前统一保留10位小数，用作数值并列规则。之后依次优先更强收缩、更小alpha、更小beta。选择后仅用本row全部support重拟合，保留选中的gamma。OOF损失参与选择，不作为独立无偏性能报告。

K1完全不运行holdout或温度优化，固定alpha=1、beta=1、lambda=1、gamma=1，使用尺度10的单位球余弦support原型。澄清零FFT情况：双block拼接向量再单位化，使这个固定K1分支始终为真正余弦判别，而不会把缺失FFTblock造成的范数变化当证据。这是数学退化处理，不根据任何性能选择。

## 数值、通信与计算审计

所有训练采用float64。每个(alpha,beta,fold)只分解一次协方差，不同lambda复用特征向量和特征值。K>=2至多24次fold分解，最终重拟合另一次；K1不分解support协方差。每候选记录参数、温度、OOF任务NLL、目标、每fold实际训练/留出数、训练尺度、谱范围及耗时。不把闭式拟合称为epoch或梯度优化。

新增地面源统计payload为0。既有摘要的逻辑数组及文件字节来自reader的payload_audit；是否已部署由入口显式给定。派生A的float64数组为204800字节，属于本地状态。头为8*C*(d+1)字节。state逻辑数组总字节为头加A；另列support矩阵和协方差逻辑字节，不冒充allocator/BLAS实际峰值，不包含日志JSON或已有编码器。

类与support顺序规范化保证重排等变；query逐行归约保证分块、重排和重复不影响其他query。精确并列用物理class ID，避免数组顺序决定标签。普通原型、非法schema、身份零范数、非finite和不平衡K均失败；不回读source修复、不自动更换算法。

## 验证

测试仅使用合成量化聚合中心与合成特征。覆盖摘要class/domain重排、真实马氏距离公式、K1/2/5/10/20、old-only、26类、零FFT、低秩/相同support、固定K1无holdout、注册表跨旧新重排、物理ID折分、query分块不变、只读状态、输入负测及JSON/字节核算。任何真实性能必须在预测冻结后由独立scorer产生，不能回流本实现。
