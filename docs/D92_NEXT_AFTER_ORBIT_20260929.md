# D92-BranchLocalRidge-v1 设计

2026-09-29；原设计已实现为 `code/cvsrffi/d92_branch_local_ridge.py`，冻结配置见 `configs/d92_branch_local_ridge_frozen_20260929.json`。本文件不含 query 证据。唯一候选为在原始 received 的完整 interaction 表示上采用训练尺度的径向核，并保留物理等权 ridge1。Phase1、现有输入权限和 query 独立推理规则不变。

## 支持证据与归因边界

唯一新增结果来源是 `20260929-phase2-d92-branch-orbit-ce-support-m4-r01/results/support_summary/`。完整覆盖 4,800 个 parent episode，其中真实 K1 数值诊断 1,200 个，标准 OOF/proxy parent 各 3,600 个、proxy anchor 42,000 个。105,600 次 CE 拟合、5,563,959 次更新完成，没有不收敛技术失败。因此不能把性能下降归因于未完成训练或减少迭代。

以下是新增类任务的 parent 等权均值，单位为百分点；三列对应 parent K=5、10、20。

| 改动与诊断 | Δ旧 | Δ新 | ΔH |
|---|---|---|---|
| single CE−single ridge，标准 OOF | −3.816 / −3.009 / −2.363 | −4.611 / −5.405 / −4.897 | −4.804 / −5.155 / −4.277 |
| orbit ridge−single ridge，标准 OOF | −4.424 / −3.821 / −2.853 | −3.615 / −2.953 / −2.434 | −3.817 / −3.364 / −2.604 |
| single CE−single ridge，1-shot proxy | −0.374 / −0.260 / −0.153 | −0.303 / −0.404 / −0.374 | −0.370 / −0.349 / −0.285 |
| orbit ridge−single ridge，1-shot proxy | −4.468 / −4.456 / −4.464 | −3.118 / −3.151 / −3.290 | −3.451 / −3.500 / −3.549 |

single CE 的标准 macro NLL 同时降低 0.271、0.455、0.639；概率损失改善不能替代分类边界改善。父任务另外核对了两个诊断下 3 个 parent K × 4 个新增类数量的全部 12 格：本轮 single CE、orbit ridge、orbit CE 各自相对 single ridge 的旧/新/H 分组均值均为负。这里不是逐 episode 都变差，也不证明所有 CE、所有相位处理普遍无效。

有限结论是：当前固定 CE 目标和完整 C4 池化都没有支持继续推进的持出证据。下一候选不重跑它们、不扫 λ/温度、不增加视图数，也不选择它们中的某个 K 作为拼接部件。此前 BranchMetric 的 K1 退化成 kernel NCM 且 proxy 未通过，也不支持再依赖未经观测的 K1 类内协方差。

## 三条路线与唯一选择

| 路线 | 可获得的新机制 | 主要不足 | 决定 |
|---|---|---|---|
| 冻结骨干之上的支持集可训练 adapter/度量 | 可以重塑特征坐标；允许较大改动 | K1 没有独立同类物理正对；用自身视图训练可能把错误不变性固化。大参数量和更多迭代本身没有证据 | 本次不选 |
| 由支持集学习分支权重或多核权重 | 不同任务调整分支可靠性 | K1 的训练标签几乎就是样本编号，训练内对齐很容易奖励隔离样本；若以外层 held 选权重会破坏诊断 | 本次不选 |
| 原始完整表示上的局部径向 ridge | 保持原始坐标、现有 ridge 目标；加入超出二阶交互的距离非线性。K1 也可使用不同物理注册类的间距 | 局部性是否有用尚无直接证据，窄核可能记忆支持集，尺度会受类数及冲突标签影响 | **唯一候选** |

选择理由是将下一项假设压缩为“固定原始表示上的非线性距离几何是否优于全局二阶关系”，并用严格能量匹配排除隐式改变正则强度。不是因为已有 support 结果证明径向核会更好。本设计不把它包装成新理论；核心贡献若成立，只能来自该合法支持适配机制在完整诊断上的效果。

## 输入与基本表示

只使用当前合法 support 的原始 received 所对应的五块缓存：z_id160、FFT96、t_emb160、f_emb160、pa_local160。不需要四相位缓存、raw IQ、额外网络前向、source 样本或逐样本源 feature，不使用地面类原型或分布摘要训练头。复用同一冻结模型及已核对的来源；不重新加载来源不明权重。每个物理样本一行，K 不变。

与当前 BranchInteraction 完全相同：

\[
B(x)=\operatorname{unit}([\operatorname{unit}(z),4\operatorname{unit}(FFT)]),\qquad
A(x)=[\operatorname{unit}(t),\operatorname{unit}(f),\operatorname{unit}(pa)]/\sqrt3,
\]
\[
\phi(x)=[B(x),A(x),\operatorname{vec}(B(x)A(x)^\top)],\quad
k_0(x,y)=K_B+K_A+K_BK_A.
\]

B 为 256 维，A 为 480 维，完整 φ 为 123,616 维，但不显式物化。沿用现有各块归一化规则和零值语义，不重新归一化 φ。通常自能量为 3，零块例外。以下距离、核和损失均在这些无量纲特征坐标中定义。

## 尺度、径向核和能量匹配

给定训练 support 的 N=CK 个物理样本，令

\[
d_{ij}^2=\|\phi_i-\phi_j\|^2,\qquad
\delta_i=\min_{j:y_j\ne y_i}d_{ij}^2,\qquad
\tau=\operatorname{median}_{i=1}^N\delta_i.
\]

最近异类距离包含零值，不排除冲突标签，不筛除“困难”类。中位数对偶数 N 取中间两项的算术均值。尺度只使用当前训练折的全部注册类；每个物理样本同权。此选择把典型最近异类相似度锚定在 exp(−1)，是预先固定、可解释的几何单位，而非扫描得到的带宽。它不是类内方差估计。

当 τ>0：\(R_{ij}=\exp(-d_{ij}^2/\tau)\)。这与常见 \(\exp[-d^2/(2\sigma^2)]\) 写法的关系是 \(\sigma^2=\tau/2\)，不存在额外系数可调。

令 \(H=I-\mathbf1\mathbf1^\top/N\)、\(G_0=HK_0H\)、\(G_R=HRH\)。定义

\[
s_0=\operatorname{tr}G_0=\frac1N\sum_{i<j}d_{ij}^2,\qquad
s_R=\operatorname{tr}G_R=\frac2N\sum_{i<j}(1-R_{ij}),
\]
\[
\gamma=s_0/s_R,\qquad G=\gamma G_R.
\]

因此每一个非退化训练任务都满足 tr(G)=tr(G0)。γ 是由当前训练特征唯一确定的单位换算，不是待调权重，也不与旧核混合。标签只进入 τ 的异类定义和 ridge 目标；old/new 角色不参与训练。不同新类规模会改变训练竞争集合和 τ，这是同一公式的结果，必须分层披露，不能手动固定某个有利类数的带宽。

PSD 可直接由高斯核的展开理解：\(R(x,y)=e^{-\|\phi_x\|^2/\tau}e^{-\|\phi_y\|^2/\tau}\sum_{m\ge0}(2\langle\phi_x,\phi_y\rangle/\tau)^m/m!\)，每项是正权的张量内积核。中心化和正的 γ 保持 PSD。它包含任意阶距离关系，而非将原有二阶交互重命名。

### 必须固定的退化情形

1. **C=1。**没有异类距离，τ=null，原因 `SINGLE_REGISTERED_CLASS`。唯一分类输出为 0，预测该类；不拟造带宽，也无需因分类执行 Cholesky。仍报告合法物理样本数和算子退化。
2. **s0=0，全同完整特征。**候选的中心化表示和目标最优函数为零，τ=0、γ=null，所有类分数为 0。按物理 class ID 字典序打破平局。不能读 held 来打破该退化。
3. **τ=0 且 s0>0。**采用高斯核的精确 τ↓0 极限：\(R(x,y)=\mathbf1[d^2(x,y)=0]\)。这是相同完整特征所构成的等价类核，PSD；训练存在多个等价类，所以 sR>0，照常 γ=s0/sR。它不是偷偷退回旧头。对所有与训练特征无精确匹配的 held/query，raw 核行均为零，中心化后会给出相同的分数行；这种泛化退化必须记录，按真实 held 表现判失败，不人为添加 epsilon 来“救活”。
4. **τ>0 但极小。**按 float64 的指数极限处理大比值：−∞ 的 exp 为 0。`1−R` 使用 `-expm1(-d²/τ)`，保留小差值；不设经验带宽下限。
5. **不一致数值状态。**理论上 s0>0 时 sR>0。若实际出现 sR≤0、非有限 γ、距离在非同一特征下整体下溢到零、Cholesky 失败或超出冻结数值残差界，报技术失败并保存上下文；不加 jitter、不截谱、不回退其他方法。γ 过大但有限只记录，并校验 G 及残差，不能依据数值大小切换候选。

这些规则对全部 K、全部类一致。C≥3、K1 时径向非线性通常不与原 interaction 成比例，可以改变竞争边界；并不声称每个 K1 episode 都改变。C2K1 只有一个中心化方向，trace 匹配后的训练 Gram 相同，但未见点的径向核延伸仍不必等于原线性延伸，应单独测试，不能声称完整预测严格等价。

## 数值距离与中心化

禁止默认用 \(k_{ii}+k_{jj}-2k_{ij}\) 计算接近重合的距离。固定使用等价低秩分解。设 ΔB=Bi−Bj、ΔA=Ai−Aj，则

\[
B_iA_i^\top-B_jA_j^\top=\Delta B A_i^\top+B_j\Delta A^\top=UV^\top,
\quad U=[\Delta B,B_j],\quad V=[A_i,\Delta A].
\]

对 U 做 reduced QR，U=QT、Q 列正交；交互距离为 \(\|TV^\top\|_F^2\)。总距离为 \(\|\Delta B\|^2+\|\Delta A\|^2+\|TV^\top\|_F^2\)，是非负平方和，不靠相减后截零。B、A 逐元素完全相同的记录直接定义距离为精确 0；其他点不按经验容差合并。按无序训练点对只计算一次并写入两个对称位置，query 用相同公式。与显式完整 φ 的小规模核对必须覆盖近重复、零块和秩亏 U。

R 可减去常数 1 后再采用现有参考样本差分中心化，以减少常量消减；中心化后与 HRH 完全等价。s0 用正距离和，sR 用 `-expm1` 的正项和计算。所有训练均值、参考值、τ、γ 都是 train-only；held/query 不参与 trace 匹配。

## 目标、解与逐样本推理

目标保持原始物理求和：

\[
\min_W\ \tfrac12\sum_i\|\langle\psi_i-\bar\psi,W\rangle-(e_{y_i}-\mathbf1/C)\|^2+\tfrac12\|W\|^2,
\quad \langle\psi_i,\psi_j\rangle=\gamma R_{ij}.
\]

λ=1，物理权重 1，没有 class-specific 系数、温度或 K-specific 超参。非退化解为 \(\alpha=(G+I)^{-1}(Y-\mathbf1\mathbf1_C^\top/C)\)，使用 float64 Cholesky 与两次三角求解，不显式求逆。G+I 最小特征值至少 1；理论条件数上界为 1+s0。

对单独 query q，仅取得自身五块原视图特征，以固定 τ 计算 \(r_j=R(q,x_j)\)。

\[
g(q)_j=\gamma\left[r_j-N^{-1}\sum_l r_l-N^{-1}\sum_l R_{lj}+N^{-2}\sum_{l,m}R_{lm}\right],\quad f(q)=g(q)^\top\alpha.
\]

不重新估计 q 的局部带宽，不读其他 query，不按 query 的预测类调整 τ。批次大小、顺序、其他 query 的存在均不得改变该行分数。

数值残差报告 \(\|A\alpha-Y_c\|_F/(\|A\|_2\|\alpha\|_F+\|Y_c\|_F)\)，A=G+I；可用 1+s0 作为可计算分母上界，需明示使用该上界而非声称测量真实谱范数。冻结验收阈值设为 `128*eps64*max(N,C)`，仅为线性代数检验，不参与统计决策。另报告 tr(G) 对 s0 的相对误差，同一阈值；s0=0 用精确零路径。残差验收失败必须保留训练 ID、scope、parent/train K、fold/trial、已完成 stages。

## 最小对照与可反驳假设

三臂固定为 `branch_ridge`、`interaction_ridge`、`local_ridge`。前两臂精确复用已冻结公式；后一臂为唯一候选。三臂同一物理训练/持出映射，不以 OrbitCE 的低结果作主比较基线。

interaction 还有强机制对照含义：对 τ>0，将 R 替换为一阶径向展开 \(1-d^2/\tau\)，则中心化后为 \(2G_0/\tau\)；同样 trace 匹配后恰好恢复 G0。因此无需额外拟合一个重复的“能量对照”。改进若存在，不能仅归因于整体 kernel 能量或更小的有效 ridge 系数。但 trace 匹配不匹配全谱、每个对角、有效自由度或 query 范数，这些差异正是径向非线性的组成部分，必须量化。τ=0 边界不适用该一阶推导，单列报告。

预定可反驳假设：

- H1：训练竞争尺度上的距离非线性能改善物理 held 的新旧类共同竞争，且不靠任何 source 或新 shot。若只增加训练准确率而 held 下降，则拒绝。
- H2：K1 proxy 也优于两个原基线，不需估计不可识别的单类方差。若只有多样本标准 OOF 改善而 proxy 失败，不能推进全面目标。
- H3：检查收益与退化在 K 和新增类数量上的分布。全部分层透明报告，广泛退化需要科学讨论，不改选有利 strata，也不把单个分层下降新增为硬门槛。

失败机制包括：最近异类尺度过小导致支持记忆；最近异类距离受错标或重复特征控制；高维距离集中使径向核近常量或近单位阵；support 中最近竞争类不是 held 的主要干扰类；trace 匹配保留总能量但放大不利小特征方向；增加注册类导致几何尺度收缩。没有理论保证所有 K 或旧新类均改善。也不能从一次失败反推应当扫多种带宽。

## 完整物理诊断与科学判据

保持原完整 8 模型/cohort、K=1/5/10/20、新增类数=0/2/5/10/20、接收机、场景和 support seed 矩阵。不新增抽取或挑选 episode。

标准 OOF：K≥2 时每类物理 ID 排序，按位置模 min(K,3) 分折。每一折从训练物理 ID 重新估计 τ、γ、参考核及头；held 的标签只由独立 support 诊断器连接。1-shot proxy：parent K≥2 按每类排序位置遍历全部 K 个 anchors，每类选 1 个训练物理 ID，持出其余 K−1 个；保留 parent K、train K=1、完整 train/held ID。true K1 仍只做数值诊断，无独立持出，不把其他 parent 的 proxy 冒称正式 K1 证据。视图和同一物理记录不跨角色。

OOF 和 proxy 分别汇总；proxy 先在 parent 内平均 anchors，不把重复 held 次数作为独立样本量。完整保存旧类、新类、H、macro accuracy、逐类 floor、原始混淆/NLL，并按 parent K×新增类数、模型、cohort、receiver×scene、support seed 展示。

本轮研究筛选保留已有 A/B：在每个 parent K=5/10/20，新类准确率与 H 相对两个原基线均严格改善，旧类退化不超过 1 个百分点，且两种诊断都通过；old-only 任务不能从汇总中消失。同时逐项显示 3×4 新增类数格的旧/新/H 变化，禁止只展示汇总掩盖广泛退化。用户要求各 K 兼顾旧类、新类与 H，并说明新增类数量的影响；没有要求每个 K×new_count 子组都严格提升。本设计不新增全格严格单调门槛，也不因任一分层下降自动判整体失败。全面改善作为单独结果报告；满足旧类容忍但旧类下降时，必须区别研究筛选通过与旧新共同提升，不能宣告全面目标完成。分层广泛退化要讨论其范围和限制，但不得事后改变既定筛选、bandwidth 或拼接旧头。正式 K1 的科学结论最终仍需独立、冻结后的合法 query 确认，设计者不接收其反馈。

预定数值/机制诊断全部只读 train 或实际物理 held：τ 的 min/median/max、δ 分布、零异类距离比例、τ=0 比例、s0/sR/γ、中心化 trace 误差、ridge 残差、拟合成本；记录训练非对角 R 分位数、每行 `sum(R)`、有效自由度 `tr(G(G+I)^−1)`。后者只在已计算的分解上求解计成本，不据此选择头。held 另报最大相似度、正确类与最近错误类最大相似度差、训练/held 正确率差，以及 τ=0 时无精确匹配率；这些只检验局部性假设，不能进入训练阈值或回滚。

## 复杂度、通信和建议测试

不新增通信数据或冻结网络前向。原 support cache 中五块 float32 合计每物理记录 2,944 bytes，N=520 时约 1.53 MB；不需要显式 123,616 维交互张量（其 float64 大小每样本约 0.989 MB）。低秩距离按点对分块计算，QR 只处理 256×2 矩阵，随后乘成 2×480 矩阵；每对基本复杂度是固定常数倍 O(256+480)，不是 O(256×480)。N=520 时有 134,940 个无序训练点对，QR/临时块预算必须在实现阶段用合成数据测量，本设计不虚构秒数。

训练距离与核 O(N²·736)，Cholesky O(N³/3)，C 列三角解 O(N²C)，训练工作内存 O(N²+N·736+NC)，分块临时区上限在实现预登记中固定。单 query O(N·736+NC)，只读取固定 support。N=520 的单个 float64 N×N 矩阵为 2,163,200 bytes。状态需要 B/A、α、训练中心化参考/均值和固定尺度；沿用七数组式可将实测 bytes 逐数组列出，不把临时 QR、Gram 或 Python audit 混入持久状态声明。

完整矩阵每臂标准 OOF 10,800 次头、proxy 42,000 次头，共 52,800；三臂上界 158,400 次 Cholesky（C1/全常量免解时实际更少），无梯度 optimizer steps。父任务如严格复用同 split 和全部训练物理 ID 的既有 baseline artifacts，应分别报告“候选实际计算”和“对照复用”，不能把旧成本消去。完整成本还包括可选自由度诊断的额外三角解，需单列。

实现前冻结测试至少包括：显式 φ 与 rank2 距离等价；极近重复和零块不出现负距离；正常径向 PSD 与 τ=0 等价类 PSD；中位数0但 s0>0；全同、C1、C2K1；τ>0 一阶径向线性化经 trace 匹配精确回到 interaction 的训练 Gram 和合法 query 延伸；held 变动不改变 τ/γ/头；逐 query 批次/排列不变；物理与类名置换等变；旧新角色不影响训练；两个旧臂精确回归；每折和 proxy 物理角色完整；异常保存前序 stages；小规模独立 primal/kernel 解核对；C26K1/C26K20 合成资源检查。只有合成正确性通过后才进入既有实验登记与独立审查流程，不在本设计回合启动拟合。

## 建议 API 与当前边界

建议后续模块 `code/cvsrffi/d92_branch_local_ridge.py`：`fit_branch_local_ridge(...five_blocks, support_labels, support_ids, classes, old_classes=(), arm='local_ridge')`、`probe_branch_local_ridge(...)`，返回不可变 state 与既有物理 OOF/proxy schema。候选 audit 额外保存 `distance_rule,bandwidth_tau,bandwidth_rule,bandwidth_zero,interaction_centered_trace,radial_centered_trace,trace_scale,degeneracy_reason,normal_equation_residual`。

实现固定每块最多 256 个点对，QR 临时输入每块为 256×256×2，乘积临时区为 256×2×480。常驻状态实际保留 B、A、alpha、参考核行、中心化均值五个 float64 数组，以及 reference_self、center_grand 和适用时的 tau、gamma；字节数逐项报告。无需为了“七数组式”的建议额外保留无用途的数组。

两条对照直接调用既有 `fit_branch_ridge` 和 `fit_branch_interaction`。三臂完成完整物理 OOF 与全部锚点 proxy；真实 K1 probe 不拟合。任何候选数值失败通过 `NumericalFailure.audit_dict()` 保留当前 arm、训练 ID、parent/train K、scope、fold/trial 和先前已完成 stages。候选使用 SciPy 的两次三角求解；有效自由度诊断另记两次三角求解的耗时与次数。

本实现过程仅访问协议、算法代码与本设计，没有读取 query、正式 benchmark 或 source 样本。`tests/test_d92_branch_local_ridge.py` 只生成合成数据。数值测试须由已核实的项目 `ssr-gpu` 环境执行；环境恢复和测试证据由本次主任务记录，不能将标准库语法检查写成数值通过。
