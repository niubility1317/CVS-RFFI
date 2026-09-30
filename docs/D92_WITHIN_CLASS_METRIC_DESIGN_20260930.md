# WithinClassMetric support 机制 pilot 设计

状态：COMPONENT_AND_ABLATION_IMPLEMENTED_NOT_RUN。用户最新主线是 BranchLocalRidge 类方法与微调联合、以前者为核心。本文固定 W 仅作为可复用组件和消融，不作为下一完整候选；已写core/config/tests保留。联合方案见 [LocalRidge 联合设计](D92_LOCAL_RIDGE_JOINT_DESIGN_20260930.md)，尚未实现或启动。已有优化授权不变。

## 1. 目的、证据与范围

唯一科学假设是：当前接收特征中，旧类训练 support 的部分类内变化可跨类别复用；对这些方向作有界收缩，同时保留旧类均值方向，可能改善原 LocalRidge 点级邻域与完整注册竞争。下文保留完整的**机制诊断加固定变换对照 pilot 合同**，供组件/消融复用，当前没有正式run；不从该诊断直接晋级。K1 没有类内估计信息，精确退回原 LocalRidge，不声称 K1 收益；10/1/3 pp 保持理想方向。

依据为完整 [SequentialResidual 失败复盘](D92_SEQUENTIAL_RESIDUAL_FAILURE_LESSONS_20260930.md) 和其中列出的完整 support 证据。其 CE 与总目标下降、训练判决也可变坏，提示应先检查能否保持已有多分支几何；这些事实不证明本方向有效。类内变化不等于已分离的真实信道或域扰动，不复用历史权重、源样本、逐样本源特征、query 或 query 反馈。

Phase1、原五块缓存及 practical residual 数据契约不变：route=residual、mode=post_sync、equalization_enabled=false、fs_hz=25000000。已验证 capsule 的场景全集为 practical_high/practical_mid/practical_low_urban；本 pilot 只取既定 practical_high/practical_low_urban、receiver 19-1/20-19、model seeds 2026092701/2026092702、support seed 2026092711。160 parent：4 model/cohort row，各 40 parent；每 row 为两 rx/scene 键×K1/5/10/20×新增0/2/5/10/20。旧类6个。完整 producer matrix 保留，selection 显式登记，不按表现替换身份。

## 2. 唯一度量与 B/C 状态

沿用原 interaction 的归一化和分支权重：

- `b=unit(concat(unit(z_id),4 unit(fft96)))`，维度256。
- `a=concat(unit(t_emb),unit(f_emb),unit(pa_local))/sqrt(3)`，维度480。
- `φ=[b,a,vec(ba^T)]`，隐式维度123616。

所有均值仅取本路径的旧类 train support。设旧类数 c_old、每类物理训练数 k、总数 n=c_old k；μ_c 为类均值，μ 为全部旧训练样本均值，M 的列为 μ_c−μ。定义：

`E=[φ_i−μ_yi]`（列向量表示），`P_M` 为 M 列空间的正交投影；

`R=(I−P_M)E`，`s=||R||_F²`；

`W=(I+RR^T/s)^(-1)`，s>0；结构上 s=0 时 W=I。

没有强度、rank、温度或步数搜索；不是再训练 CE 头。`0.5I ≼ W ≼ I`，所以 `0.5d0² ≤ dW² ≤ d0²`。旧均值差位于保护空间，在理想算术下 W 对其为恒等。这还意味着旧类 nearest-centroid 的排名不因 W 改变；不能以这类距离差证明收益。Gaussian 点级核仍会改变，旧类准确率、新类身份方向均没有自动保护。

B0：原 LocalRidge，仅用旧 train support。B：用相同旧 train support 估计 W，然后拟合下述 metric LocalRidge。C0：原 LocalRidge，全注册 train support。C：继承并冻结 B 的 W，再用全部注册 train support 拟合新的核/头。C 不重新估计 W，不冻结 B 的旧分数，不拼接 B/C score，不设旧类 bonus。新增类为0时，C0复用B0、C复用B。

**控制总能量的明确选择：** W 只改变 Gaussian 距离及其最近异类距离带宽。B/C 的迹匹配目标仍是同一训练集合的**原始** interaction 中心迹 `s0=Σ(i<j)d0_ij²/N`，与对应 B0/C0 相同；另记录 `sW=Σ(i<j)dW_ij²/N`，但不用 sW 改变 ridge 相对强度。此处不是先变换 φ 后把所有能量也改掉的另一候选。

其余分类规则逐项保留原 LocalRidge：tau 为 dW 最近异类平方距离的中位数（含0）；tau=0 用精确特征等价核。W 正定，因此 dW=0 与 d0=0 同义。Gaussian 核按训练参考中心化、迹匹配到 s0，目标 onehot−1/C，物理样本权重1、ridge1；float64 Cholesky、无 jitter/截断/带宽搜索。C=1 或原 s0=0 沿用原零分类器语义。query 逐样本面对全部已注册类，平局按物理 class ID 字典序；没有 query batch 估计状态。

## 3. 稳定的有限坐标实现

### 3.1 压缩为旧 support 张成空间，不作模型近似

按物理ID排序，取第一个旧样本为固定参考 (b0,a0)。对 `[b0,b1−b0,…,b(n−1)−b0]` 和对应 a 矩阵分别做 reduced QR，得到 Qb、Qa；不按学习到的 rank 删除列。列数分别 rb=min(256,n)、ra=min(480,n)，秩不足时 QR 补出的方向只是坐标，不创造样本。验证 Q 正交及参考/差分重建误差。

旧样本及旧类均值/残差都位于此坐标空间，其 interaction 坐标维度 `m=rb+ra+rb*ra`，n≤120 时 m≤14640。对任意 x,y，稳定构造投影后的差分：

`db=Qb^T(bx−by)`，`da=Qa^T(ax−ay)`；

`δψ=[db,da,vec(db (Qa^T ax)^T+(Qb^T by) da^T)]`。

该式只用于与旧残差的内积；**总距离 d0² 始终调用原 LocalRidge rank-two QR 非负平方和算法**，不能用压缩坐标丢弃新样本的正交成分。新类/query 在旧 span 外的变化在 d0² 中保留，不参与收缩。

E 的每个类先以该类最小物理ID为参考构造差分，再减类内差分均值，避免从两个含大公共分量的 φ 中心值直接相减。M 通过全旧参考的类均值差构造。M 的 SVD 在小坐标空间直接完成，不先构造 Gram 矩阵再开平方。R 在相同空间直接投影；不构造123616维协方差。

### 3.2 数值 rank、结构零与不可分辨残差

设 `eta=128*eps64*max(1,n,m,c_old)`。M 奇异值 `σ>eta*σmax` 的方向组成数值保护空间；σmax=0 时为空。记录全部奇异值、实际 rank、阈值和丢弃能量。该阈值是实际数值算法定义，不声称浮点中识别了任意微小的精确 rank；均值保护按 eta 容差核验。

结构 identity 分支优先执行：train K1，或每个类的全部 b/a 均与其类参考逐元素完全相同，均无需 QR/SVD/Cholesky。分别记 `TRAIN_K1_NO_WITHIN_CLASS_INFORMATION`、`EXACT_ZERO_WITHIN_CLASS_VARIATION`。没有绝对 residual-energy floor。

若 E 非零而 `||R||F <= eta ||E||F`（包括投影算出0），报 `NUMERICALLY_UNRESOLVED_ZERO_RESIDUAL` 技术失败，不把舍入噪声归一成明显收缩，也不悄悄改回I。即便真实 E 恰落在 M span，浮点投影也可能无法区分精确零和极小非零；该不可分辨情况明确保留失败状态，不能伪称观测到了可靠方差。近阈值不是新增性能门槛；这是 trace 归一在零点不连续带来的正确性边界。

其余情况令 `F=R/||R||F`。先除以 max(abs(R))，再对缩放值归一化，避免平方下溢；不得因未缩放的 s 下溢而当成 identity。记录 trace 的可表示值和缩放表示（scale、scaled_trace）；不可表示的物理 trace 记 null 加原因，不输出非JSON的Infinity/NaN。

实现中 R/F 采用“每行一个样本残差”的转置约定。令 `L=chol(I+FF^T)`，`T=solve_triangular(L,F)`，则

`dW²(x,y)=d0²(x,y)−||T δψ(x,y)||²`。

归一化后的 Gram 为PSD、trace1；I+Gram条件数≤2。核验 Frobenius norm、Gram对称性、Cholesky残差、`||T||op²≤0.5`、旧均值保护误差和训练距离界，超容差为技术失败，不通过 jitter、负距离截断或增加正则掩盖。

### 3.3 计算复用与近重复样本

每个样本相对于固定旧参考的 `u=T(ψ−ψref)` 可以预计算。常规 pair 用 `u(x)−u(y)`；其差分取消误差按固定界检查：令 `d=max(480,m)`、`gamma=d*eps64/(1−d*eps64)`、`e=8*gamma*(||φx−φref||+||φy−φref||)`、`h=||ux−uy||`。e 必须使用**投影前**原 interaction 参考距离范数，由原稳定距离算法求得；不能用可能几乎为零的压缩范数低估 Q 内积误差。仅当 `2*h*e+e² <= eta*d0²` 时使用缓存差分；否则用上节 δψ 差分式直接算 Tδψ，记录 `direct_difference_pair_count`。两条计算路由是同一个度量的数值求值，不是新参数或候选分支。e 是保守的算术路由量，不是关于真实特征误差的统计置信界。head 常驻保存自己训练support的 u 与参考范数供逐样本评分复用，并计入bytes，不每次预测重算全部训练坐标。

直接差分仍必须满足 `[0.5d0²,d0²]`（相对 eta 容差）；违反即技术失败。完全相同 b/a 的 pair 先精确返回0。不把缓存近重复导致的非零当成新样本，也不从相等距离增加物理 K。每个样本的坐标/预测独立计算，query 批大小和顺序不影响拟合状态；批量只是输出拼接。

## 4. API 与日志合同

新模块 `cvsrffi.d92_within_class_metric`；冻结JSON为 `{'algorithm': FROZEN_CONFIG}`。FROZEN_CONFIG 包括 base_algorithm=原LocalRidge完整常量、channel/allowed_scenarios、上述固定公式/容差/identity规则、共享与诊断规则。root 的 runner 从此JSON导入合同；本设计状态尚不创建该JSON。

```python
fit_within_class_metric(
    *, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes,
    context=None, log_callback=None,
) -> WithinClassMetricState

fit_within_class_local_ridge(
    *, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes, old_classes,
    metric, baseline_state=None,
    context=None, log_callback=None,
) -> WithinClassLocalRidgeState

diagnose_leave_one_class_out(
    *, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes,
    context=None, log_callback=None,
) -> dict
```

labels 与现有核API一致：整数0…C−1，对应传入classes；support IDs唯一、每类相同正物理K。参数为空/维度不符/非法ID属输入错误；浮点/线性代数不符合上述边界抛 `NumericalFailure`，其 `audit_dict()` 严格JSON可序列化。

MetricState不可变，包含 `identity`、`classes`、训练旧IDs/类绑定、Qb/Qa、T及审计；`audit_dict()`返回副本。head API 校验metric旧IDs/类标签及对应b/a与当前旧train完全一致，C只能加新类train，不得替换旧train或用同ID不同特征。metric保存用于绑定的旧b/a（计入bytes）；每个head保存自身原始训练b/a和kernel状态。没有加载encoder或原型派生可训练状态。

head返回 `.score(*,五blocks)`、`.predict(*,五blocks)`、`.audit_dict()`。identity时须传入已完成、精确同集合/同特征/同classes的 `baseline_state`，wrapper引用原state，score直接委托，新增head拟合/因子分解为0；非identity时按第2节拟合，原方法源码/config不修改。原LocalRidge state没有保存逐ID训练label映射，因此core能核对其ID/特征/classes，baseline的同label绑定由入口对同一train slice立即拟合并传入保证，不能声称core独立证明了原state没有保存的事实。MetricState自身保存label，C继承的旧ID/label/特征由core完整检查。Nnew0由入口复用B，不再次调用C拟合。

公共 context 固定包含 row_id、split_id、scope、parent_k、fold、trial、stage；metric/LOCO/head事件分别流式输出一个dict位置参数。审计包含：

| 对象 | 必需字段 |
|---|---|
| metric | identity, identity_reason, training_physical_ids, classes, train_k, train_physical_count, coordinate_dim, protected_rank, protected_singular_values, protected_rank_threshold, within_trace, residual_trace, residual_trace_scale, residual_scaled_trace, normalized_residual_eigenvalues, spectral_concentration, contraction_max, numerical_tolerance, metric_fit_count=1, metric_factorization_count, fit_seconds, state_array_bytes, persistent_state_bytes, optimizer_steps=0 |
| head | final_fit（原LocalRidge兼容的tau/s0/sradial/gamma/normal residual/loss/fit timing字段）, transformed_interaction_centered_trace, metric_head_fit_count, head_factorization_count, metric_identity_reuse, shared_metric_state_bytes, head_state_bytes, persistent_state_bytes, direct_difference_pair_count, optimizer_steps=0 |
| LOCO | status, diagnostic_fit_count, diagnostic_factorization_count, diagnostic_fit_seconds, diagnostic_score_seconds, classes, folds（逐留出类）, optimizer_steps=0 |
| failure | status=TECHNICAL_FAILURE, failure_reason, context, training_physical_ids, classes, completed_stages, 已完成计数/时间/数值界；保留当前阶段部分状态，不伪造未完成结果 |

谱诊断 `spectral_concentration` 固定记录最大特征值/trace、sum(lambda²)、effective_rank=1/sum(lambda²)（F归一trace1）。不根据谱选择rank或强度。身份状态谱为空，trace=0，effective_rank=null，原因明确。bytes为唯一持有的数值数组+显式标量，另记录引用共享状态，不把B/C共享metric重复计为部署总字节；Python/audit/JSON序列化开销另列实测或N/A。

## 5. 留一旧类诊断与完整计数

每个 OOF 路径只在B训练折内做6个留一旧类诊断。对类c，用其他5类的训练物理support拟合独立 W_-c（含它自己的M、P、R），绝不使用c估计投影或收缩。被留出的c也只来自该B训练折；它的类内中心残差 E_c 是评估对象，均值仅用于计算这个报告量，不回流W_-c。

逐类报告 `held_class_within_trace` 和 `held_class_contraction_fraction = sum_i e_i^T(I−W_-c)e_i / sum_i ||e_i||²`；结构上分母0记null+原因。正常范围[0,0.5]。同列报告其余5类的对应训练分数，作描述性比较；不把接近0或较高比例作为自动晋级/拒绝门槛。此量测的是共享方向能解释多少类内能量，不是准确率收益、真实域因果或旧均值距离改善；新类身份也可能与这些方向重叠。

计算 E_c 的能量复用同一稳定pair引擎：den=`Σ(i<j)d0²/K`，num=`Σ(i<j)(d0²−dW²)/K`，避免另行 `u_i−mean(u)` 引入取消误差。报告6类全部结果，不能挑可迁移的类改变算法。OOF held 和 proxy held 仅最终评分，不能进入metric、LOCO拟合或配置选择。LOCO样本内部共享、类别仅6，不宣称独立显著性。

固定覆盖：160 parents、40 true K1 numerical-only（不拟合、不LOCO）、120 OOF parents；360 OOF fold paths、1,400 proxy anchor paths，总1,760路径。metric尝试1,760次，proxy1,400次为identity；非identity最多360。原B0/C0 head拟合3,168次；新增metric head最多648次，identity及Nnew0复用必须反映实际计数。仅360 OOF路径执行6次LOCO，共2,160 diagnostic fit；诊断次数、因子分解、耗时、暂存状态与部署metric分开，不算成实际部署拟合成本。所有optimizer_steps=0。

报告R0与R_metric的B旧、C旧/新、B−B0、B→C旧损失、六种正确性转换、H、完整K×新增类数及模型/receiver/scene层级。原地面A不存在则N/A，不用B0冒充。OOF按物理held样本数合并不等折，proxy先平均parent内全部anchor；new_count0的B复用及跨新增数重复不称独立观测。Nnew0结果、失败和退化strata均透明保留。

## 6. 成本、测试和实现边界

本实现不以参数少或无optimizer宣称低成本。n_old=120时m=14640，T最多1,756,800个float64（14,054,400 bytes），还需Qb/Qa约706,560 bytes、旧b/a约706,560 bytes，合计约15.5MB数值状态；因此先前“约14MB”仅T主项，不是总状态。实际OOF最大n=84时T约4.85MB，加Q与旧b/a约0.99MB，总约5.84MB。类绑定/诊断/解释器额外开销不含于这些公式。每次LOCO只保留小标量报告，释放其临时metric，不同时常驻6份。

拟合的显式坐标工作量约O(n³)，F Gram/矩阵求解乘法约O(n²m)，保存T为O(nm)。每个新样本生成坐标及投影约O(nm)；原核pair距离仍需执行。若所有pair触发直接差分，N个head训练点的修正最坏为O(N²nm)，例如n84、m7224、N364时无对称重复也约4×10^10矩阵乘加，不能以通常缓存路径掩盖这个上界。报告metric fit、head fit、LOCO fit/score、训练与held评分、cache/direct pair计数、峰值RSS、实际常驻bytes和部署包/新增传输bytes；未测项写N/A。与原LocalRidge用同硬件、2CPU lanes×BLAS2比较，不改其他健康实验。

核心合成测试应覆盖：独立显式φ+W小问题oracle；0.5至1距离界/PSD/旧均值保护和nearest-centroid不变；class重命名与列置换、row置换；分批/逐样本相同；C准确继承metric且旧IDs/特征改动被拒；identity精确委托R0；零类内变化、重复/近重复、rank-deficient M、极小可辨残差和不可分辨余量失败；trace目标等于原s0；LOCO彻底排除被评类估计且分母0记null；失败上下文/阶段计数/状态bytes可读；合成N26/364/520对应head规模与旧n上界的资源记录。有限精度比较采用已声明容差，不为通过测试任意放宽。

新core、冻结JSON和核心tests按本合同实施，审阅发现的直接正确性问题在真实运行前解决。测试、实际配置、发布和启动由root统一执行；不修改暂停Residual8草稿或任何原方法。
