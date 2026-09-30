# LocalRidge 与监督谱 adapter 联合设计

状态：IMPLEMENTED_SYNTHETIC_VALIDATION_PASSED_NOT_RUN。这是按用户最新要求实现的单一联合方案：**LocalRidge 是唯一最终分类器，微调只服务其度量，并通过闭式分类头共同优化。** 新core、冻结config和合成tests已落盘；root执行受影响核心及编排43项测试通过，包含解析有限差分、θ0逐元素原值及活跃导数。尚未启动真实support实验；固定 WithinClassMetric 留作组件/消融。

## 1. 为什么改变训练目标

[完整失败复盘](D92_SEQUENTIAL_RESIDUAL_FAILURE_LESSONS_20260930.md)证实 SequentialResidual 各阶段 CE/总目标下降，却没有保留原判决。B 的训练初始准确率100%，OOF中210/360阶段最终训练准确率下降；C_reset为268/288。这个事实不证明微调无效，也没有识别唯一因果，但不支持继续以同一批样本上已能正确判决的原核分数为底，再附加自由 CE logits。

本方案把监督信号改为：合法 outer-train support 内，每个物理内折只用 inner-train 拟合闭式 LocalRidge，再对 inner-held 计算连续预测误差；误差梯度通过闭式解反传到小型 metric adapter。这个内层目标不是独立泛化成绩，不能挑“最好内折”或最佳步；真正效果仍由完全未参与任何估计的 outer-held 评分。

不反传encoder，不引入源样本/逐样本源特征/原型可训练状态，不改变固定Phase1与practical residual。所有输入仍来自已有五块原始接收缓存。10/1/3 pp是理想方向，不能凭该设计声称已达到或变成新硬门槛。

## 2. 跨内折共享的两个实际训练参数

每个内折从它自己的旧类 inner-train，按 [WithinClassMetric 组件](D92_WITHIN_CLASS_METRIC_DESIGN_20260930.md)估计 M、保护空间、类内 R 和 `F=R/||R||F`。不从完整 outer-train 借用方向、类均值、保护空间、tau、核中心或闭式头。

设该折 `F^T F=Σ_j λ_j u_j u_j^T`，λ_j≥0，Σλ_j=1。数值上由F的SVD奇异值平方构造λ，不先取Gram的微小负特征值再静默裁剪；此阶段的方向/λ在全部8次更新中固定，不对SVD求梯度。零信息时直接identity，不伪造λ。唯一训练参数为 `θ=(θ0,θ1)`，可行域 `Δ={θ0≥0,θ1≥0,θ0+θ1≤1}`。定义：

`gθ(λ) = [λ/(1+λ)] (θ0+θ1 λ)`；

`Wθ = I−Σ_j gθ(λ_j) u_j u_j^T`。

θ跨折共享；u_j、λ_j完全是折内状态。没有按第j个方向绑定参数，因此各折的rank、旋转、样本数不需相同。重复特征值的各方向使用相同g，任意正交旋转不改变W；类重命名和行置换也不改变算子。谱接近常数或只有一个非零方向时，两个系数未必分别可辨识，必须记录有效梯度/参数变化，不把“2个参数”写成“两个独立获益机制”。

`0≤gθ(λ)≤1/2`，故 `0.5I≼Wθ≼I`。θ=(0,0)时逐元素走原 LocalRidge identity 分支，初始预测精确等于R0；θ=(1,0)对应固定WithinClassMetric。它是一个真正可训练的线性metric adapter：其特征作用为 `Wθ^(1/2)φ`，只沿旧类内方向改变，梯度直接作用于θ。不是只命名SFT、不实际更新参数，也不是把encoder权重称为已微调。

`∂dθ²/∂θ0=−Σ[λ/(1+λ)](u_j^Tδφ)²`；

`∂dθ²/∂θ1=−Σ[λ²/(1+λ)](u_j^Tδφ)²`。

因此在θ=0仍有可用的单侧数据梯度，不存在两层都零初始化造成零梯度的问题。**训练首步不能因θ=0而跳过距离/核导数；精确R0快捷路径只用于前向值，导数仍按上式计算。** 若没有可辨类内方向或所有下降方向都被可行域投影排除，参数可以保持0；这应如实记为没有适应更新，不能增加步数、学习率或方向数追逐效果。

## 3. 与闭式 LocalRidge 联合的内层目标

outer-train每类物理K为k。内折数 `J=min(k,3)`，每类按物理ID排序、位置mod J分折。每个inner-held物理样本恰参与一次该次目标；不等折按样本数加权，不能先对折均值等权。

对每折f，使用自己的inner-train几何Wθ和全部当前注册inner-train，按原LocalRidge规则重算距离、最近异类中位数tau、Gaussian中心化、迹匹配到**原未变换interaction s0**，并解：

`A_f(θ)=(K_f(θ)+I)^(-1)Y_f`；

`S_f(θ)=K_held,train,f(θ) A_f(θ)`。

Y为onehot−1/C，held cross-kernel仅按inner-train中心化。最终分类就是S的argmax；没有另加 logits、温度或旧类偏置。

训练目标为物理和对应的平均形式：

`L(θ;θanchor) = [Σ_f ||S_f(θ)−Y_held,f||F² + ||θ−θanchor||²] /(2N)`。

N是本阶段全部outer-train物理样本数。每个样本权重1；正则在物理和形式下系数1，因此梯度为 `(θ−θanchor)/N`，不能忘掉N或把每折正则重复累加。所有注册类对称进入误差；类别越多就有更多物理样本，不通过old/new分组额外重加权。

闭式解的导数为 `dA=−(K+I)^(-1)(dK)A`；预测导数为 `dS=(dKcross)A+Kcross dA`。两个参数的梯度用同一Cholesky及三角求解，非有限即技术失败。核心需独立有限差分与显式小矩阵oracle验证；本设计不直接写新训练实现。

tau仍按变换距离的原规则计算，不另固定一个由结果选择的温度。对最近异类距离的并列极小值，导数取全部并列值导数均值；中位数的中间秩若落入相等值组，对该相等组取均值，偶数中位数再取两个中间秩的均值。这是固定的对称广义导数约定，不能按class名打破数值相等来挑梯度。唯一最小/中位值处按通常链式法则；非光滑点的有限差分验证须用方向导数/单侧测试，不伪称处处可微。

tau=0时正定W保证零距离等价关系不变，等价核对θ的导数为0；s0=0或单注册类沿原零分类器规则。原s0不求导，radial中心迹与gamma=s0/sradial按θ正常求导。指数下溢沿原零径向极限处理；不可表示梯度、非法PSD/距离界或不可分辨R记录技术失败，不用隐藏jitter、裁剪核谱或改变bandwidth补救。

## 4. 参数训练、顺序继承与信息边界

固定8次全批量投影梯度更新，学习率0.1；`θ <- ΠΔ(θ−0.1∇L)`。只保留第8次更新后的状态；没有网格、早停、最好步选择或根据outer-held重跑。欧氏投影到二维非负和≤1的单纯形，有确定闭式解；不沿用了SequentialResidual的64步Adam或其训练状态。逐步日志记录2维θ更新前后值、数据/近端/总loss、两个解析梯度、梯度范数、未投影与实际更新范数、活跃约束、各折物理样本数/闭式残差及分项耗时；不虚构梯度收敛或源验证。

B阶段θanchor=θinitial=(0,0)，仅旧outer-train。训练结束后从全部旧outer-train重估一份最终几何，得到B adapter和原LocalRidge形式的B头。B0用同一旧outer-train的原方法，是直接对照；地面A缺失记N/A。

C阶段以学到的θB作为**共享参数初始化和近端锚点**；内层loss覆盖全部旧新注册类。每个C内折重新用该折旧inner-train估计几何，并用该折全部注册inner-train拟合头；绝不把B完整几何代入内折。共享θB来自合法B训练，随后共享θC由内层held标签监督更新，这是允许的SFT参数学习；这些标签不能直接参与该折的类均值、方向、保护空间、核中心或闭式头。不能把内部训练损失/准确率描述为独立验证。

C最终继承B的**完整旧训练几何**（Q、保护空间、谱方向/λ、旧ID绑定）并使用最终θC，再对全部注册outer-train重拟合LocalRidge头。也就是说，几何基底被继承并冻结，但谱系数更新、C核/头重新拟合；W和旧类分数都可能改变。Nnew0直接复用B及计数，不额外训练C。这个定义让C的监督确实面对新增类竞争，但不承诺自动低遗忘。

在本pilot，outer-held只在B/C最终状态固定后逐样本评分；真query也只按冻结最终状态逐样本面对全部注册类。任何outer-held特征/标签、ABC/query结果都不能用于内层估计、损失、选步或参数调整。每个parent/fold/anchor独立fit；不同row不继承任何target状态。

## 5. 小K的具体限制

- 真实K1在support诊断中仍数值检查、零拟合、无持出准确率。实际部署若每类只有一个样本，此联合方法没有合法类内学习信息，adapter保持identity、使用原LocalRidge；不声称改善K1。
- proxy trainK1同样没有内层持出可训练方向，精确复用R0，不制造views或新增物理样本。
- outer-train K2的诚实内层train只有K1，所有内折几何为identity，数据目标对θ没有可辨梯度；预声明为 `NO_IDENTIFIABLE_INNER_GEOMETRY`，不执行8个空更新，不假装进行了有效微调。若C继承非零参数的其他场景存在，须保持锚点并明确无本阶段监督变化；本pilot的B从0开始。
- 已定parent K5的外层train为K3/4，内层train至少K2，才可能出现可训练方向；K10/20同理。少样本方向不稳定和新类身份与旧类内方向重叠仍是实质风险，不能仅凭有界谱消除。

本方案仍不是K1完整答案。未来若需要K1增益，必须有另行说明的合法信息来源；不能为通过此前proxy正增益规则伪造正数，也不把此前screen自动绑定到所有后续机制研究。

## 6. 对照、资源和可反驳判断

建议保持已定160parent身份作为完整小pilot。主对照原LocalRidge；固定W组件θ=(1,0)为解释性消融，说明监督谱训练是否优于未训练收缩。联合路径报告B旧、C旧/新、B−B0、注册旧损失、新旧差、H及完整K×新增类数，保留old-only和模型/场景分层。不得用某个内折或某个stratum筛掉不利结果。

还需同一C计算预算的θanchor=0从头路径，才可区分“联合谱adapter本身”与“B参数继承”的组合收益；与SequentialResidual一样，初始化和近端锚点同时变化时，不能单独归因于warm-start。该对照使用同一C内折几何与标签，重新训练2个参数，不选择较优路径代替预声明顺序路径。

可反驳点是：外层B相对B0、外层C旧新相对原方法是否改善；固定W与联合谱adapter是否存在稳定差异；继承是否只是旧新权衡；谱系数是否实际离开0，更新是否多次被投影取消。内层loss下降本身不算成功，若外层仍损判决，应否定该联合目标在此预算/范围内的收益，而非继续小调步数或rank。

可训练参数只有2，但这不等于总成本低。每阶段8次更新，每次J个闭式头；为真实记录final训练loss再执行1次无梯度内层评估，共9J次Cholesky。J=3时B+C为54次内层head分解，加2个最终候选头和2个直接基线头，合58次；C_reset消融再增加27次内层与1个最终头。初始θ0缓存可减少重复算术但不能少报实际次数。内折几何预计算并缓存，但不同折不得共用标签派生状态。

两个梯度各增加闭式导数三角求解及交叉核导数计算；这个成本可能高于固定W和原LocalRidge很多。要单列B/C训练、内层几何、内层head、梯度、final评估、LOCO诊断、评分、峰值RSS、常驻adapter/头、实际部署包/新增传输字节。复用组件的完整数值状态上界约15.5MB，不能只报16字节θ。现有8步是预设小预算，尚无实测或效果结论；不承诺星载资源降低。

本联合实现与冻结算法JSON已完成合成验证，真实运行配置、登记和启动由root后续统一处理。当前没有读取query或启动真实训练。

## 7. 已实现API、日志与验证边界

实现为 `code/cvsrffi/d92_joint_spectral_local_ridge.py`；冻结算法文件 `configs/d92_joint_spectral_frozen_20260930.json` 顶层为 `{'algorithm': FROZEN_CONFIG}`。原LocalRidge和纯metric组件源码未修改。

```python
prepare_joint_training(
    *, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes, old_classes,
    inherited=None, context=None, log_callback=None,
) -> JointTraining

fit_joint_spectral_local_ridge(
    prepared, *, mode='B', baseline_state=None, log_callback=None,
) -> JointSpectralState

evaluate_joint_objective(prepared, theta, anchor, *, gradient=True)
    -> (loss, gradient, audit)
```

mode为B/C_seq/C_reset/fixed。fixed使用θ=(1,0)，零优化步；即便outer trainK2没有可训练内层信息，只要完整几何非identity，fixed仍应用其固定收缩。C的prepared必须传B state作为inherited；它复用B完整旧几何，C_seq/reset/fixed共用同一prepared，准备费用只计一次。C_seq以B.theta为初始化/锚点，reset以0为锚点。Nnew0由入口分别复用各路径B，不再准备或拟合C。

JointTraining保存不可变的规范类序、物理ID、仅训练原始五块、完整几何、各内折几何和距离/谱系数缓存；`.audit_dict()`返回可修改副本。类序以物理class ID字典序规范化，labels是输入classes所对应的整数索引，经准备后重新映射。C核对旧ID全集、逐ID类标签与b/a特征完全一致，并在提供时核对row_id/split_id，不从另一target row继承状态。

JointSpectralState提供只读`.theta`、`.geometry`、`.classes`及`.score(*,五blocks)`、`.predict(*,五blocks)`、`.audit_dict()`。θ=0或完整几何identity时，前向直接复用原LocalRidge；可传同训练集合的baseline_state避免重复最终拟合。原state没有逐ID原始训练label审计，因此baseline的同label绑定由入口对同一train slice立即拟合并传入保证；core核对其ID/特征/classes，不从预测猜补label。不同输入class列顺序只做规范列重排，不算一次新的拟合。

非identity分数仍逐样本计算。θ0内层score不仅使用原先的距离、tau、中心化与同一次闭式解，还将alpha规范成原state的C-contiguous布局并逐行matvec，防止BLAS路径造成末位差异；没有跳过活跃导数。root测试已对内层θ0 score与原头逐元素比较通过。

| 对象 | 计数、资源与身份字段 |
|---|---|
| prepared | joint_preparation_count, geometry_fit_count, geometry_factorization_count, geometry_seconds, preparation_seconds（prepare_seconds同值别名）, train_k, training_physical_ids, classes, old_classes, inner_folds, no_information, transient_distance_bytes, geometry_state_bytes, inner_geometry_state_bytes |
| 每个inner_fold | inner_fold, training_physical_ids, held_physical_ids, geometry_training_physical_ids, geometry_identity, geometry_audit, train_physical_count, held_physical_count, direct_difference_pair_count |
| state | mode, theta, theta_anchor（anchor同值别名）, optimizer_steps, nonzero_projected_update_count, theta_changed_from_anchor, no_information, no_update_reason, identity_forward, inner_objective_evaluation_count, inner_head_fit_count, inner_factorization_count, derivative_triangular_solve_count, final_head_fit_count, final_factorization_count, final_fit, shared_geometry_state_bytes, head_state_bytes, theta_state_bytes, persistent_state_bytes, fit_seconds |

prepared的identity几何估计仍计一次geometry_fit attempt，但因子分解为0；不是把attempt写成实际学习。训练阶段仅在存在可辨内折几何时执行8步、9次objective评估；执行了8步不等于8次有效适应，应分别报告实际非零投影更新数和最终θ是否偏离锚点。no-information不执行空objective评估，固定路径也不执行优化objective。

prepare回调事件为JOINT_INNER_PREPARED；fit有8条JOINT_SPECTRAL_STEP及一条JOINT_SPECTRAL_FIT。只有STEP含顶层整数step，其他事件没有step。STEP包含pre-update loss/真实gradient和post-update theta，final_objective独立来自第8次更新后的无梯度评估；不得混称最后一条STEP的loss已评价最后一次更新。最终audit保存完整8条标量步骤和各折汇总，入口补齐当前row/split/outer fold或proxy anchor身份，完整保存文本和结构化日志。source_validation=null，原因SOURCE_ACCESS_FORBIDDEN。

NumericalFailure.audit_dict严格JSON可序列化，保留当前θ、已完成步骤、当前失败内折的已执行进度、此前完成内折及准备上下文。因子分解和导数求解按实际尝试累计，即便该objective尚未完整返回；技术失败不能记成零工作量。无jitter、隐藏温度变化或失败后的新候选回退。

对并列导数，合成测试单独验证声明的分支平均及行/类置换对称性；`min(t,-t)`在0的平均广义梯度为0，而两个单侧方向导数均为−1。这些不是同一数量，不要求平均广义梯度等于任一方向有限差分。光滑点另用常规中心有限差分验证完整loss梯度；θ0用可行侧扰动验证仍有实际导数。其余测试覆盖物理held特征不能改变该折几何、不等折样本加权、C继承/reset、K1/K2/no-information、固定θ路径、8/9计数、逐样本批次不变、状态不可写与部分失败计数。

核心与编排43项通过仅说明已测实现行为；不代表support泛化、K1提升、成本降低或真实运行成功。真实耗时/峰值内存/部署字节仍按第6节实测报告。
