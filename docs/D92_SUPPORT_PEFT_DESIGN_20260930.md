# D92 缓存后 support 参数高效微调设计

2026-09-30。状态：历史复盘后暂缓，未冻结配置或启动。工作名 `D92-SupportResidual8-LocalRidge`。直接对照为冻结 `D92-BranchLocalRidge-v1`，不是正在运行的 LocalMargin。该运行和 Phase1 原模型不变。

用户要求吸取既有微调失败经验后，已发现本草案与D11低秩Adapter机制高度重合。下文保留原始提案，**不作为待启动方案**。已有一个未测试的核心代码草稿，尚无入口、配置、数值验证或真实实验结果；先完成D11/D21及完整support证据复盘，再决定后续设计。复盘说明见[D92微调历史复盘](D92_FINETUNING_FAILURE_LESSONS_20260930.md)，其中含历史query回顾，方法研发者应只使用明确标记的support-only原始证据。

## 1. 目标、证据与假设

用户希望先通过目标域旧类 support 改善旧类识别，再注册新类时降低旧类遗忘，使新类准确率接近旧类。旧类适应增加 10 个百分点、注册后旧类下降不超过 1 个百分点、新旧差不超过 3 个百分点是理想方向，不是本候选的硬门槛。H 只作辅助，不替代这三个行为指标。

本设计只依据当前源码、项目协议以及完整 [LocalRidge support 报告](D92_BRANCH_LOCAL_RIDGE_SUPPORT_RESULT_20260929.md)。没有读取 query、ABC 结果、实验索引、交接或 LocalMargin partial 指标。既有 support 证据是：LocalRidge 标准 OOF 改善旧新识别，单锚点 proxy 的旧类未稳定改善；训练准确率近乎饱和但持出误差仍大。旧新差仍明显，不能由训练准确率推断泛化。这支持研究受约束的表示适应，但不证明下面的损失、rank 或顺序策略会有效。

可反驳假设：在冻结 encoder 的 160 维身份表示后，用旧类 support 学一个小幅共享残差，再把它固定用于新类注册，能够改善旧类持出识别，同时不明显损害新类持出识别。反例包括：旧类训练改善但持出不变或下降；映射只适合已见旧类，导致新类恶化；核头重拟合及新增竞争仍造成较大旧类下降。所有这些结果均须保留。

本轮只研究这一候选，不同时搜索其他 rank、损失、温度、训练轮数或网络插入点。参数选择是下文的先验工程选择，不是数据证明的最优值。

## 2. 输入与修改边界

沿用原始 received 单视图缓存：`z_id/t_emb/f_emb/pa_local` 各 160 维，FFT 96 维，共 736 个 float32 数。`z_id` 是原生 `feat_joint`，缓存没有额外分支归一化。实现依据为 [缓存契约](../tools/export_d92_branch_support_features.py#L24)、[冻结提取入口](../tools/export_d92_branch_support_features.py#L173) 与 [身份前向入口](../code/cvsrffi/identity_only_forward.py#L23)。已有外层 `SatAnchorIdentityAdapter` 被这些入口绕过，因此不能只开启其开关而假定缓存已经适应。

只在缓存读取后、LocalRidge 的既有归一化和隐式 interaction 表示构造前，把 `z_id` 替换为下文的适应结果。其他四块特征、FFT 权重 4、各归一化、径向带宽规则、trace 匹配、物理样本权重 1、ridge 系数 1、float64 求解和数值退化规则全部沿用 [冻结配置](../configs/d92_branch_local_ridge_frozen_20260929.json) 和 [实现](../code/cvsrffi/d92_branch_local_ridge.py)。继承的是算法规则；改变输入表示后估计出的带宽、trace 和核当然可能改变。

无需 source 样本、逐样本 source 特征、source 原型初始化的可训练层、teacher 或源域 BN 统计。下文监督锚点完全由当前旧类训练 support 计算，不从地面原型继承。地面原模型只提供不可变 encoder；若另外报告真实 Stage A，使用已合法固定的地面推理状态，不将其状态输入本候选优化。

不得把全矩阵 support 缓存合并成一个训练集。每个当前行、OOF 折或 proxy 锚点必须独立得到其合法 train-support 子集。相同物理样本的视图不增加 K；本候选不增加视图。

## 3. 唯一残差映射

令 d=160、r=8、epsilon=1e-12。对原始身份向量 z，n=||z||2。n<=epsilon 的向量直接保持原值，不参与除法；其训练分类表示按原 norm floor 规则取值，并记录旁路数量。其余向量令 u=z/n，并计算：

\[
h=\operatorname{GELU}(Vu),\quad t=Uh,\quad
\delta(u)=\frac{\rho t}{\sqrt{\rho^2+\|t\|_2^2}},\quad
z'=n\{u+\delta(u)\},\quad \rho=0.25.
\]

U 属于 R^(160×8)，V 属于 R^(8×160)，无 bias；两者可训练。GELU 使用精确 erf 形式，不用可切换近似。残差满足 ||delta||<0.25；有效输入的 ||u+delta||>0.75，限制方向变化并避免后续单位化接近零。最大角度偏移不超过 arcsin(0.25)，约 14.48 度；这限制容量，不保证分类不变。rho=0.25 是保守先验，不经 held/query 搜索。

U 初始化为全零。V0 使用固定 DCT 正交行：j=0,...,159；第 0 行为 1/sqrt(160)，第 k=1,...,7 行为 sqrt(2/160) cos[pi*k*(j+1/2)/160]。V 初始化为 V0。初始化不使用类别编号、随机 seed、地面原型或其他行的 support。DCT 是确定性的坐标基，不声称前 8 个方向具有经验证的语义优势；rank 8 取身份维度的 5%，将新增可训练参数限定为 2,560，而非由结果选择容量。

U=0 时映射在数学上为恒等；初始化恒等检查使用浮点容差。零 U 恒等 shortcut 只允许用于纯推理或无梯度检查；训练首步必须计算完整前向及其正确梯度，不能因 U=0 直接返回输入并跳过残差导数。不能同时把 U、V 初始化为零，否则两层乘积的训练梯度可能全部为零。这里第一步 dL/dV 为零是预期的，但 dL/dU 一般非零；U 更新后 V 可获得梯度。不把“任何数据上都必有非零梯度”当作保证：全同特征、全同监督锚点等本来可能没有判别梯度。

## 4. Stage B 的监督、正则与优化

只使用当前旧类训练集 S_old={(z_i,y_i)}。设 O 为旧类数，N_old 为物理记录数。监督锚点所用 u_i 始终定义为 z_i/max(||z_i||2,1e-12)：先计算每类均值，再按同一 norm floor 单位化为固定锚点 w_c。零均值保留零向量，不用随机向量替换。w_c 在 64 步内 detach 且不更新，不是可训练分类权重。每个训练样本参与自己的类锚点，这是显式的训练集内监督，不是留一监督；held 样本完全不参与。对有效 z 定义 q_theta(z)=unit(u+delta(u))；旁路小范数输入使用原单位化结果。辅助 logits 为 10*w_c^T q_theta(z)。

损失固定为：

\[
L_{CE}=\frac1{N_{old}}\sum_i
\left[\log\sum_{c=1}^{O}e^{10w_c^Tq_i}-10w_{y_i}^Tq_i\right],
\]
\[
L_{res}=\frac1{2N_{old}}\sum_i\|\delta(u_i)\|_2^2,\qquad
L_{param}=\frac{0.01}{2r}\left(\|U\|_F^2+\|V-V_0\|_F^2\right),
\qquad L=L_{CE}+L_{res}+L_{param}.
\]

旁路输入的 delta 定义为零。以 logsumexp 实现 CE。单位化锚点范数不超过 1，logits 位于 [-10,10]，CE 的自然尺度为 nats；残差正则按单位输入的平方位移计量，参数正则按每个低秩方向归一。这里 adapter 训练使用**物理样本平均损失**，因此正则相对强度不随 K 线性缩放；最终 LocalRidge 仍使用原来的物理样本求和目标。两者不能混写成同一个 lambda。

每个物理记录在 CE 和残差项各出现一次。等 K 数据自然类平衡，不添加旧类保护权重、类别编号权重、按错误程度重采样或持出难例挖掘。温度 10、rho、rank、正则和步数都固定；不做 held 早停或温度校准。辅助锚点头训练结束后丢弃，不部署。监督锚点并非最终 LocalRidge 分数，因此存在代理损失与最终分类器不一致的风险，必须通过 held support 比较检验，不能以 CE 降低宣称任务改善。

优化采用 CPU float64、full-batch Adam，恰好 64 次更新，learning_rate=0.01，betas=(0.9,0.999)，eps=1e-8，weight_decay=0；每步在 Adam 更新前对 U/V 联合梯度做 L2 global norm clipping，阈值 1。无 scheduler、AMP、梯度累积、数据 shuffle、最佳训练步选择或多次初始化。64 步是固定计算预算，不是收敛证明。第 64 步状态为唯一输出，即使训练损失不是历次最低也不回滚。

实现不依赖 Torch 或其他自动微分框架：允许并优先使用现有 NumPy float64 与 `scipy.special.erf` 实现等价解析梯度，不加载 encoder，也不为此新增环境。令 a=Vu、s=sqrt(rho²+||t||²)，GELU(a)=a*Phi(a)，其导数为 Phi(a)+a*phi(a)；残差 Jacobian 为 rho*I/s−rho*t*t^T/s³。在有效输入上 v=u+delta 的范数大于 0.75，单位化 Jacobian 为 (I−q*q^T)/||v||。CE 对 q 的梯度为 10*W^T*(softmax(logits)−onehot)/N_old；残差项对 delta 加 delta/N_old；链式反传得到 U/V 梯度后，分别加 (0.01/r)*U 与 (0.01/r)*(V−V0)。所有单位化旁路按前述定义处理，其对 U/V 的数据梯度为零。不得把固定锚点的导数错误地传回训练参数。

Adam 从全零一二阶状态开始，第 t 步先更新矩，再做偏差校正 m_hat=m/(1−beta1^t)、v_hat=v/(1−beta2^t)，参数减去 lr*m_hat/(sqrt(v_hat)+eps)。epsilon 在平方根外，不是添加到 v 后再开根号。解析梯度须由独立中心有限差分检查，覆盖 CE/残差/参数项分别及总和、零 U 首步和非零 U/V 状态；不能用另一份相同手写反传作唯一参考。使用自动微分的未来实现也须满足相同数学与数值契约，不构成另一个科学候选。

如果 O=1，CE 恒零且没有异类判别信号：显式输出恒等 adapter，optimizer_steps=0，记录 single_class_no_discriminative_supervision。若 O>=2 但梯度恰为零，仍按预算执行并记录零梯度；不能改损失、添加噪声或静默切换其他训练法。输入非有限值、损失/梯度/状态非有限、参数 shape 错误是技术失败，保留阶段、train IDs、最后成功步、失败步状态；不回退成 LocalRidge 后冒称候选成功。未在 64 步收敛或持出表现低不是技术失败。

冻结 encoder 参数与 buffers 不变；如果实例化它则始终 eval、requires_grad=False，纯缓存实现无需加载它。缓存数组视为固定输入，仅计算 U/V 梯度；若采用 Torch 等实现，缓存张量 detach，adapter 训练不能置于 inference_mode 内。已有主干使用 GroupNorm 和 Dropout；不调用整网 train()，不更新任何已有 CRRA/NMFDU support 状态。查询采用无梯度推理，不更新归一化、锚点、adapter、核或分类头。

## 5. A/B/C 状态、继承与直接对照

| 状态 | 拟合输入与继承 | 推理竞争集合 |
|---|---|---|
| A：真实地面基线 | 已固定且来源合法的地面推理状态；无本次 target-support 优化 | 与 B 一致的旧类集合 |
| B0：LocalRidge 直接对照 | 原缓存、当前旧类 train-support、冻结原 LocalRidge 规则 | 全部旧类 |
| B：候选旧类适应 | 当前旧类 train-support 从 U=0/V0 独立训练 adapter；变换同一旧类 support 后拟合 LocalRidge | 全部旧类 |
| C0：LocalRidge 注册对照 | 原缓存、同一旧类 train-support 加当前新类 train-support，重新拟合原 LocalRidge | 全部已注册旧+新类 |
| C：候选注册 | **逐值继承并冻结 B 的 U/V**；变换同一旧类及新增 support；重新拟合 LocalRidge | 全部已注册旧+新类 |

真实 A 的推理方法及类别映射使用已有合法定义；本设计不创建新的地面分类器以改善 A/B 数字。若仅有本次五块缓存而没有已确认的地面推理接口/类别映射，A→B 暂记 unavailable，继续报告 B−B0 的直接受控比较，不能将 B0 改名为“无适应地面模型”。A 的分数不进入 adapter 训练。

C 阶段不进行梯度更新，不传入 B 的 Adam 状态；不使用新类标签调节 adapter。继承的是**适应后的表示映射**，不是 B 的核头。C 的带宽、中心化统计量、trace 比例、alpha 和类别映射都由合法旧+新训练 support 重新拟合。保存 B 的只读快照用于配对，C 不能修改它。新类数为零时 C=B、C0=B0，直接复用对应只读状态。

因此本候选不能保证旧类分数不变，也不能自动保证低遗忘；固定表示只能消除“注册梯度改坏表示”这一条路径。新类竞争与核头重拟合依然能改变旧类结果。也不声称新类监督已经改善表示：新类 support 在 C 中只参与核注册。这是为先验证低改动方案作出的能力限制。

所有 held/query 逐样本评分；单个 C scorer 只接收当前记录特征与固定全注册状态，不接收其 truth、old/new 角色、类别数量提示、配额或其他 query。old/new 分组仅由预测固定后的独立评价器连接 truth。Stage B 的训练旧类集合是已知 support 注册范围，不是查询角色特权。

监督损失和锚点对类别重命名等变：双射重命名只置换 w 的行与 logits 列，U/V 及连续分数不变。物理记录顺序固定按 ID 排序，不依赖类别编号的大小。浮点归约差异按容差检查。最终决策沿用冻结 LocalRidge 的 physical_class_id lexicographic tie-break；在**精确分数并列**时，任意重命名可能改变硬标签。这是已有决策规则的显式边界，不能声称硬标签在并列时也完全等变；应审计并列集合的等变性和出现次数，不能趁机改对照的 tie 规则或按旧类优先破平局。

## 6. train-only 诊断与 K1 局限

标准物理 OOF 沿用每类物理 ID 排序后 position mod 3。每一折先分出 train/held，再从 train 的旧类部分构建锚点、训练 B adapter；C 只继承这一折 B，并使用 train 的旧+新部分注册。held 不得参与归一化统计、锚点、优化预算、模型选择、核参数或停止条件。

单锚点 proxy 沿用每个 parent 的全部 K 个排序位置，每类恰好一个训练物理样本，其余 K−1 个持出。每个 anchor 单独训练旧类 adapter，再注册新类；先在 parent 内平均全部 anchor，再让 parent 等权汇总。视图不得增加样本数。不得只取容易或最快收敛的 anchor。

K1 时每类锚点就是该类唯一训练表示，CE 仍可借助不同类之间的排斥产生梯度，但没有任何类内变化证据；这是训练点自参照监督，并非独立正样本学习。正则与小 rank 只能限制容量，不能修复缺失信息。proxy 测到的是较大 parent 中选择单锚点的持出行为，不是实际目标 K1 的泛化保证。真实 K1 pilot 行仅检查输入/数值契约，无独立 held，无拟合准确率晋级；算法本身的 K1 fit 路径由独立合成数据验证，未来正式注册时仍按同一候选执行，不临时关闭训练。

同一旧类 held 物理 ID 集用于 B0/B/C0/C，C 在完整注册集合中竞争。若 A 可用，也在同一旧类 held ID 上评分。每个 parent 先计算旧类适应、注册变化和新旧差，再对 parent 汇总，不能通过不同 episode 均值相减构造配对。

## 7. 小型完整 support pilot

不把历史全 4,800 parent 当作早期门槛。预先固定下面的 160-parent 子矩阵，不读取任何表现选择它：

- model seed：安全冻结模型清单中数值最小的两个；两个 cohort 全部保留。
- 每个 cohort：合法 receiver×scenario 键按原始稳定 ID 排序，取前两个；这只是确定性覆盖抽样，不代表全部场景。
- support seed：取共同可用种子中数值最小的一个。
- 每一组合覆盖 K={1,5,10,20}，新增类数={0,2,5,10,20}。

预计 2×2×2×1×4×5=160 个 parent。其中真实 K1 为 40 个数值诊断；其余 120 个完整执行 3 折 OOF，并遍历全部 proxy anchor。正式预登记前只用 schema/身份元数据解析上述选择，写明实际 ID；如果某个所需组合不存在，明确列出不可构成矩阵的原因并修订设计，不默默换成其他表现更好的行。选择、方法参数与预算在接触该 pilot 指标前固定。这里不要求新增数据重验或额外签名链。

这给出 360 条 OOF 顺序路径和 1,400 条 proxy 顺序路径，共 1,760 条 B→C 路径。最多 112,640 个 adapter 更新；每条最多 B0/B/C0/C 四次 LocalRidge 拟合，头拟合上界 7,040 次。old-only 可合法复用 C=B；若不同新增类数的旧训练 IDs 完全一致，可按 model/cohort、完整旧 train IDs、输入缓存身份与固定配置复用 B，而非跨异 ID 混用；上界不假设命中缓存。跨 OOF 与 proxy 不因 ID 个数相同就复用。

完整执行上述 pilot 后才形成性能结论，不能根据前若干行好坏停掉健康运行或调整超参。技术失败按预登记规则记录，未补齐前结果标为 incomplete，保留全部已执行项；这不是额外性能门槛。

报告至少包括：

- B−B0 的旧类改善；可用时另报 B−A，清楚区分支持对照与真实地面基线。
- C 的旧、新绝对准确率；B旧−C旧 的下降；|C新−C旧|；C−C0 的配对旧/新/H，H 仅辅助。
- 全部 K×新增类数格、old-only、各 model/cohort/receiver×scenario/support seed。真实 K1 不填持出指标，old-only 不填新类/H。
- OOF 与 proxy 分开，proxy 先 parent 内平均；并列、错误、零梯度和退化计数；全部低收益和负向格均展示。
- 原研究中的每 K Δ新>0、ΔH>0、Δ旧>=−1pp 可作为与历史一致的描述性标志，严格旧/新/H 同正另报；两者均不能冒称满足用户新的顺序目标，也不添加全分层必须全正的门槛。

依据完整 pilot 的效果方向、旧类适应是否出现在 held、注册下降、新旧差与实测成本决定是否值得扩展；不设置 10/1/3 的通过线，也不以某个 H 标志自动取代该判断。若需要修改方法，形成新的 support-only 假设与版本，保留本候选完整结果，不按正式 query/ABC 分数选择 loss、rank、K 专属参数或重跑最有利子集。这里的 support-held 顺序诊断与真实 query ABC 评估权限不同，不能互相替代。各模型、anchor 和 support draw 相关，不宣称独立重复或显著性。pilot 的场景覆盖有限，扩展仍需说明这一选择偏差。

## 8. 训练、部署与总成本

U/V 共 2,560 个参数：float64 训练权重 20,480 bytes，梯度与 Adam 一二阶状态另占约 61,440 bytes，合计约 81,920 bytes；不含 V0、锚点、缓存、autograd 激活、分配器和解释器。O 个固定锚点占 8×160×O bytes。两层线性前向每样本约 2,560 次 MAC，另有 GELU、残差约束、单位化、约 160O 次辅助 logit MAC 和反向传播；不能把前向 MAC 当作训练总量或实际延迟。

64 步只对旧类训练子集运行，Stage C 不再反传；已有缓存时新增 encoder 前向/反向均为零。没有缓存的在线物理记录仍须运行原 encoder 及 FFT，不能把它们从星载账目中去掉。内部 LoRA 不在本候选中，以免引入前缀缓存变更和 encoder 后缀反传。

初版部署保持 float64 adapter，以免未经验证的降精度改变方法，增加 20,480 bytes。原 LocalRidge 常驻数值状态仍约为 8N(738+C)+32 bytes：C26K1 为 158,944 bytes，C26K20 为 3,178,272 bytes；加 adapter 后分别为 179,424 与 3,198,752 bytes，不含地面模型、FFT、注册表、Python 对象与临时矩阵。适应输入后状态形状未变。V0、Adam 状态和辅助锚点不需要用于最终推理；B 快照如仅用于离线比较，不计作单个 C 部署的必需驻留状态，但运行时实际保留它就应计入 RSS。

原头仍需 O(N²×736) 距离/核工作和 O(N³) 分解，逐样本评分仍需 O(N×736+NC)，再加当前候选残差计算。本候选只能在结构上证明避开 encoder 反传，**不能宣称总成本低于 LocalRidge**；它新增训练与推理开销。与全网络微调相比的内存/耗时优势也须在同环境测量，不能仅凭参数量推断星载时延、功耗或能耗。

未来验证需分别测量 encoder/FFT、adapter 训练、B/C 核拟合、单条 query 变换和评分的时间，峰值 RSS、临时 Gram、常驻状态、真实缓存/模型文件与新增传输 bytes。批量缓存吞吐不能冒充单条在线时延；CPU 合成测试不能冒充星载实测。先用独立合成形状 C6/C26、K1/K5/K20 做实现与资源验证，不读取目标指标来调整数值预算。

## 9. 实施接口、日志与必要检查

建议接口边界为 `fit_old_support_adapter(old_blocks, old_labels, old_ids)` 返回不可变 B adapter；`fit_registered_local_ridge(adapter, blocks, labels, ids, classes)` 显式接受适应状态并返回固定 head；`score_one(state, feature_record)` 仅评分，不接收 truth/role。B/C 调用都保留物理身份与模型/缓存绑定。监督锚点构造是训练步骤的一部分，不是跨 fold 的预处理。

启动记录全部有效参数、特征契约、train/held IDs、old/new 注册类集合、初始化和固定步数。每一步结构化记录 CE、残差项、参数项、总损失、学习率、U/V 梯度范数、裁剪前后联合范数、U/V 范数、残差均值/最大值、参数位移和耗时；保留完整 JSONL，并提供去大数组的紧凑 CSV 与文本汇总。未测值写 null 并解释。训练准确率如记录，只作训练诊断，不作选步依据。所有技术失败要保存严格 JSON 可序列化的上下文，非有限值显式编码，不因序列化错误丢失失败证据。

实现阶段必要合成检查：首步 U 非零梯度的非退化例子及 V 的预期零梯度；与独立有限差分相符；初始化恒等与 residual bound；class rename 的连续分数/并列集合等变；输入重排按物理 ID 还原；held 不影响锚点或梯度；C 确实逐值继承并冻结 B 而 head 重新估计；小范数、全同、零均值锚点、单类、K1 和非有限失败；逐样本与批量逐行评分一致；冻结 encoder 参数/buffers 不变。测试聚焦这些行为，不扩展为无关全仓测试或新的实验门槛。

原提案完成时仅有此设计文档。随后形成了未验证核心草稿，现随用户要求暂停推进；没有新增实验配置、实际拟合、环境安装、训练、正式评分或远端实验启动。当前健康LocalMargin运行不受影响。此文档不声明实现完成或性能改善。
