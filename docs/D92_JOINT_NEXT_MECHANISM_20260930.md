# JointSpectral 之后：分支通道适配与 LocalRidge 联合训练

状态：IMPLEMENTED_SYNTHETIC_VALIDATION_PASSED_NOT_RUN。本文规定一个候选：**在冻结的五块分支缓存上训练有界通道门控，通过内层物理持出的 LocalRidge 间隔损失求梯度；最终分类器仍只有 LocalRidge。** 核心、算法常量 JSON 及合成测试已落盘；主任务串行完成核心/编排 60 项、入口/汇总 15 项及新增 Decimal 近重复回归 1 项，共 76 个不同检查，均通过。尚无本候选真实 support 拟合或启动。原实现及暂停中的 Residual8 草稿保留。

## 1. 完整支持集证据说明了什么

本次只读取完整 JointSpectral 支持集产物的 [summary.json](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-joint-spectral-support-m2-r01/results/support_summary/summary.json) 与 [training_objectives.jsonl](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-joint-spectral-support-m2-r01/results/support_summary/training_objectives.jsonl)，并完整解析后者 7744 条记录，覆盖 4576 个联合阶段、3168 个固定阶段、936 个实际训练阶段和 7488 步。没有以抽样代替全量结论。完整文本日志的独立汇总仍由主任务处理；本文不声称已扫描全部 stdout 的错误或恢复文本。

OOF、新类存在的 96 个可测 parent 上，R0 的 B 旧类准确率为 71.0417%，C 旧类为 63.7674%，C 新类为 54.8464%，H 为 58.4168%。R_joint、R_reset 对应的汇总指标与 R0 完全相同。R_fixed 的 B 旧类变化为 −0.0347 pp，C 旧类为 +0.0781 pp，C 新类为 −0.0573 pp，H 为 +0.0018 pp。这些是 parent 均值，不是所有物理样本合池的准确率，也不构成独立重复试验或显著性结论。

| 完整 OOF 训练阶段 | 阶段数 | 最终参数不同于锚点 | 最终参数恰为零 | 最终参数和均值 | 最终参数和最大值 | 初始到最终目标平均下降 |
|---|---:|---:|---:|---:|---:|---:|
| B_joint | 360 | 285 | 75 | 0.000287078 | 0.001752022 | 1.9276×10⁻⁷ |
| C_joint | 288 | 266 | 30 | 0.000477750 | 0.003496091 | 7.9868×10⁻⁸ |
| C_reset | 288 | 244 | 44 | 0.000196471 | 0.001744079 | 7.9316×10⁻⁸ |

7488 步中 6340 步投影后更新非零，没有一步触及参数和等于 1 的约束上界。逐阶段初始与最终内层正确样本总数均相同；这不证明逐样本预测均相同，因为正确与错误可以相互抵消。按记录的相邻目标比较，没有超过 10⁻¹⁵ 的上升。目标下降幅度很小，不能把它写成有效适应。

原谱规则满足 `g(λ)≤(θ0+θ1)/2`。由全部最终参数可得，任一最终拟合的平方距离相对收缩至多为 0.174805%。这是数学上界，不是实测 held 分数或间隔变化。它说明训练确实执行，但实际学习的变换接近恒等；不能据此单独归因于学习率。固定强度 θ=(1,0) 的结果同时说明，把同一旧类内方向收缩得更强也没有显示一致收益。

结构上的限制是确定的：谱适配只在从旧类训练残差估计的方向上收缩，并保护旧类均值 span。它不能学习新的通道组合，也不能直接改变该受保护空间中的类间几何。该保护可保持相应最近均值排序，但不保证核岭头排序不变。结合 [SequentialResidual 完整失败复盘](D92_SEQUENTIAL_RESIDUAL_FAILURE_LESSONS_20260930.md)，当前证据反对两种直接延续：继续给同一小谱族加步数；或在训练准确率已很高的核头上增加自由 CE logits。证据没有证明通道适配一定有效，也没有证明 prior CE 失败只有“过拟合”一个原因。

覆盖限制保持不变：2 个 model seed、2 个 cohort 各 2 个 receiver/scenario 键、1 个 support seed、6 个旧类、新增 0/2/5/10/20 类及 K=1/5/10/20，共 160 parent。只含 practical_high 与 practical_low_urban 的已选接收机场景，不覆盖 practical_mid 或其余 RX。真实地面阶段 A 未在该支持集实验中测量，A 与 B−A 均为 N/A。真实 query、ABC 报告、总索引和交接未读取。

## 2. 唯一候选：有界对数通道门控

使用现有五块缓存：z_id 160 维、fft96 96 维、t_emb/f_emb/pa_local 各 160 维。Phase1 encoder、BN 状态、原始 received 特征缓存及 practical residual 契约均固定。实际训练的是缓存后的对角特征 adapter，不称为 encoder 微调，也不使用源样本、逐样本源特征或原型派生的可训练状态。

实现接入点明确放在**原方法已经构造好的 b/a 上**。先对原始五块调用一次原特征构造，得到 `b0∈R^256`、`a0∈R^480`，包括原始 unit_floor、FFT floor、背景权重 4、背景整体归一化及辅助权重 `1/sqrt(3)`。然后直接切分为 `r_z=b0[:160]`、`r_fft=b0[160:]`、`r_t=a0[:160]`、`r_f=a0[160:320]`、`r_pa=a0[320:]`。令 `ρ_l=||r_l||₂`，每块有参数 `u_l∈R^{d_l}`，约束为：

`Σ_j u_lj=0，|u_lj|≤h，h=log(2)/2`。

`D_l(u)=diag(exp(u_l))`。

当 ρ_l>0，定义 `r'_l = ρ_l D_l r_l / ||D_l r_l||₂`；当原子块逐元素为零，输出精确零。范数采用最大绝对值缩放的稳定计算，不能将非零子块的平方和下溢误判为结构零。直接拼接 `b_u=concat(r'_z,r'_fft)`、`a_u=concat(r'_t,r'_f,r'_pa)`，不再调用 unit_floor、背景整体归一化或乘一次既有权重。精确算术下，这保留原每块范数；非零参数的浮点实现检查范数保持误差，不宣称所有归约都逐位相等。全零 u 前向直接返回原 b0/a0 的值，确保 bitwise identity；训练导数仍计算，不能用恒等快捷分支跳过。

LocalRidge 的交互表示为 `φ_u=(b_u,a_u,b_u⊗a_u)`。拟合与评分都必须进入接受 b/a 的核层 helper，不能把 r' 伪装成原始五块重新走原 raw-feature API。否则低于 floor 的原输入会被再次归一化，改变范数、原权重和零参数基线，这是必须排除的 P1 接入风险。范数保持映射满足 `T_u(c r)=c T_u(r)`（c>0），所以在精确算术下它与原固定权重和背景归一化尺度可交换；这只用于解释数学等价，不用它声称两条浮点执行路径完全相同。直接 b/a 路径是唯一实现合同。

物理和 ridge 系数 1、最近异类中位数带宽及全类竞争均不改变。Gaussian 核的中心迹匹配到**该次闭式头 inner-train 在 u=0 时的原 interaction 迹 s0**；s0 不对 u 求导。这样不把改变总核能量相对于 ridge 的强度混进通道学习。

保存 736 个 float64 参数，五个零和约束给出 731 个自由度。参数按固定缓存坐标跨内折共享，不从整个 outer-train 估计方向或均值。每块对角增益条件数不超过 2，允许通道增强和减弱，不限定为旧类内方向收缩，不保护类均值 span，因此有能力改变旧类间和旧新类间的角度与 LocalRidge 间隔。它不能创造 encoder 已丢失的信息，不能表示任意旋转，也不保证有用的判决改变。这是要验证的结构假设，参数数目本身不是贡献或效果证据。

## 3. 监督目标与折边界

outer-train 中每类有 k 个物理 support。k≥2 时使用 `J=min(k,3)` 个内折，每类按物理 ID 排序、位置模 J 分折。每次目标中每个物理样本恰好作为 inner-held 一次；不等折按样本数累计，不先对折均值等权。

给定共享 u，每折只用 inner-train 经 adapter 变换后的特征，重新计算距离、tau、核中心、迹缩放和 LocalRidge 闭式头。该折 inner-held 的标签只进入预声明的监督损失，不进入头、带宽或任何该折统计。共享 u 从全部内折监督学习，因而内层准确率和损失属于训练指标；真正的 outer-held 不参与更新、步数选择、损失设计或可行域设定。

记该折输出 S，目标类 y，`m_i=S_iy−max_{c≠y} S_ic`。使用原 onehot−1/C 目标所对应的单位类别差距，不另拟合温度或按 old/new 调尺度：

`L(u;u_anchor) = [Σ_i (max(0,1−m_i))² + ||u−u_anchor||₂²] /(2N)`。

N 为本阶段全部 outer-train 物理样本数。正则是物理和形式下系数 1，梯度为 `(u−u_anchor)/N`，不除以 736，也不在每折重复加一次。margin=1 的铰链点平方损失导数为 0；多个错误类并列最大时按全部并列类均分导数。所有注册类完全对称，没有旧类偏置或新类惩罚。类重命名仅改变标签列映射，不能改变损失与特征状态。

这个目标集中约束真实闭式分类器的竞争间隔，避免拟合大量与最终 winner 无关的分数坐标；它仍可能提高训练间隔而损害 outer-held。选择它是预声明的新机制，不是从 query 或不同 loss 的试跑中选优。

## 4. 可实施的解析反向与固定预算

不建立 `[N,N,736]` 导数张量，也不对 736 个参数分别解岭方程。每折记 `M=K+I，A=M⁻¹Y，S=L_cross A`。先从间隔损失得到 `G=∂loss/∂S`，再计算：

`bar_L=G Aᵀ，Z=M⁻¹(L_crossᵀG)，bar_K=−sym(Z Aᵀ)`。

Z 复用本折同一个 Cholesky。将 bar_K/bar_L 经过训练中心化、迹缩放、Gaussian 和 tau 的反向传播，得到训练及持出特征的伴随梯度。原 s0 固定，`dγ=−γ ds_radial/s_radial`；tau 继续采用最近异类与中位数的对称并列广义导数规则。并列广义梯度不等于每个方向导数，合成测试必须区分二者。

距离前向保留原稳定 rank-2 差分 QR，不使用大数 Gram 相减恢复近重复距离。反向不对 QR 分解求导，可按交互距离的差分形式累加。例如对样本 i 的 b：

`∂d²(i,j)/∂b_i=2[(1+||a_i||²)(b_i−b_j)+(a_i·(a_i−a_j))b_j]`。

a 的梯度交换 a/b，j 端交换 i/j。按 pair chunk 累加，覆盖差分很小的情况。对一个非零的原 b/a 子块 r，设 `w=Dr/||Dr||`，adapter 雅可比为 `∂r'/∂u=ρ(I−wwᵀ)diag(w)`，u=0 一般非零。梯度直接回到这五个子块，不经过第二次原始特征归一化。单通道非零等退化输入可确实没有可学角度；日志需如实报告零梯度。

单一优化预算预设为 8 次投影 Adam 更新，学习率 0.02，β=(0.9,0.999)，ε=10⁻⁸，全参数梯度范数裁剪上限 1。每阶段 Adam 矩清零，偏差修正按本阶段步数计算。更新后逐块投影到零和盒：`u_j=clip(v_j−λ,−h,h)`，求使总和为零的 λ；在 `[min(v−h),max(v+h)]` 上最多二分 80 次，要求 `abs(sum(u))≤128 eps64 d_l h`，盒约束也须通过 float64 容差检查，未达到则技术失败。先投影更新，不用“减均值后截断”冒充交集投影。固定完成 8 步，最后另算一次目标；不挑最好步、不早停、不改学习率重跑。这里的预设优化器服务新的 731 维双向通道空间，不是对失败的两参数谱候选继续扫学习率。

共 9 次内层目标、最多 27 次内层闭式头和 Cholesky，每个训练阶段再拟合一个最终全 outer-train 头。最后一次目标不反向。每次反向另有一次矩阵右端线性求解，不计成新的 Cholesky。通过反向聚合，主计算为 `O(N³+N²(256+480)+NC·N)` 级的核/解算及 pair 运算；不需要旧谱方法的折内高维几何状态。实际常数和稳定差分费用须测量，不能按两种算法同为 27 次分解便声称耗时相同。

参数盒只限制增益，不能保证原距离的 [1/2,1] 谱界，也不保证原分类不变。不得复用旧谱方案的该界作新实现正确性断言。非有限、未分辨的正带宽/径向迹、Cholesky 或残差失败保留上下文后报技术失败，不添加 jitter 或改损失。原 s0=0、单类、全部可学梯度为零均明确记录；只有确定退化才零更新，不用数值失败静默回退 R0。

零带宽的等价关系以原 b0/a0 的逐元素精确相等定义，不能比较原始 raw 块或另一次归一化结果。原距离引擎已经对“非相同块却得到零距离”报下溢错误，不能移除。精确算术下，范数保持角度映射可逆，且拼接保留分块，因此原 b/a 相等当且仅当变换后相等。浮点实现会确定性地保留相同输入，但不能假设它永不合并极近的不同输入。为此，每个头若原 tau0=0，直接复用原 b0/a0 等价核及其零导数；held/query 的等价判断也在各自一次原构造后的 b0/a0 上逐样本完成。正带宽分支计算 pair 时核对原与变换后块的精确相等标记；若原不同的块被浮点映射合并，报技术失败。若原 tau0>0 而变换计算得到 tau≤0，同样按未分辨正带宽技术失败处理，不把舍入产生的零当成新的类别等价关系。这个 guard 固定了既有退化语义，不改变正常正带宽分支或另加方法回退。

## 5. B/C 实际继承与 K1

B 仅用当前 row 的旧类 outer-train，从 u=0、anchor=0 学习，得到 u_B 及旧类 LocalRidge 头。C_seq 继承 u_B，anchor=u_B；加入新类后全部注册 support 对称进入内层目标，再拟合全注册类 LocalRidge 头。C_reset 用同一物理 support、同一折划分，从零及零 anchor 学习。两者差异检验完整继承策略，不能单独归因为 warm-start，因为近端锚点也不同。C 不继承旧头的分数或旧核中心，旧类分数不保证不变。

Nnew=0 直接复用各路径 B，不额外拟合 C。传入 C 时逐物理 ID 核对旧类标签与五块原始缓存等于 B；只能继承当前 row/相同旧 support 的状态。不同 outer fold、proxy anchor、model 或 cohort 都从各自合法 support 独立构建，不跨任务复用目标适应状态。

k=1 没有物理内持出，B 参数保持零；C 保留合法继承的 anchor，不作监督更新。单注册类没有错误类最大分数，其数据损失定义为零，也不进行监督更新。只有这两种情况在 prepare 阶段标 no-information，目标评估和优化步数均为零。其他阶段即使当前梯度为零也固定完成 8 步，因为 Adam 历史矩仍可能产生更新；结构零核可得到 8 次零更新记录，不以瞬时零梯度提前退出。当前平衡 K1 的 B anchor 为零，故精确等于 R0。support 诊断里的 true K1 仍仅数值检查、不拟合；one-shot proxy 的 trainK1 也不能增加视图充当物理样本。k=2 可做两折 one-shot 内训练头，其 held 标签提供合法训练信号，但统计很弱。新候选不依赖估计非零类内协方差，所以这与旧谱候选的 k=2 无信息条件不同，须明确计数。

部署将固定 adapter 应用到每个 query 的五块特征，再由同一最终 LocalRidge 对全部注册类评分。只逐样本推理，不读取 query 标签/角色/真实类别数，不根据 query 批次重估归一化、门控或核统计。原始 encoder 特征缓存可复用，但旧的距离/核/预测缓存不可当新方法结果。

## 6. 最小实现接口与可反驳检查

以下接口已在新核心中实现并通过相关合成检查：

```python
prepare_channel_training(
    *, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes, old_classes,
    inherited=None, context=None,
) -> ChannelTraining

evaluate_channel_objective(prepared, u, anchor, *, gradient=True)
    -> (loss, gradient, audit)

fit_channel_local_ridge(
    prepared, *, mode='B', baseline_state=None, log_callback=None,
) -> ChannelLocalRidgeState
```

prepared 保存原始不可变五块、一次原构造得到的 b0/a0、规范类序、内折 ID 和每折原 s0/tau0。state 保存不可变 u、最终头和继承身份；零带宽分支保存核判断所需的原 b0/a0。提供逐样本 score/predict/audit_dict。外部基线 state 只能由相同训练切片即时构建并核对后复用。C_seq/reset 共享 prepared，不重复记准备费用。B、C_seq、C_reset 都输出逐步实际梯度、范数、裁剪前后值、参数盒活跃量、投影残差、非零更新数及最终是否偏离 anchor。

实现前的必要合成检查包括：

1. u=0 的原 b/a、全部前向、tau、中心迹、最终及内层分数逐元素等于原 LocalRidge，同时有非零解析梯度的确定样例。用 raw 范数低于 floor、等于 floor、略高于 floor、全零以及不同幅度但原 b/a 精确相等的样例覆盖拟合和评分；断言原特征构造只执行一次，适配后的子块不再通过 raw-feature helper。非零 u 检查块范数保持及零块精确零。原 tau0=0 的 train/cross 等价核逐元素不变；构造或注入近不同块浮点合并情形，验证不把原 tau0>0 静默变成零带宽分支。
2. 光滑小问题的完整损失方向有限差分，与反向矩阵公式一致；近重复 pair 用独立高精度距离/梯度对照；tau/max 并列点测试声明的平均分支梯度及置换对称，不要求它等于所有方向导数。
3. 改某个 inner-held 特征或标签不能改变该折的头拟合输入与 s0；它可改变监督损失和下一步 u。改 outer-held 不能改变任何状态。每个物理 ID 一次监督、不等折样本加权。
4. 五块零和盒投影、参数不可变、类名/输入行重排等变、单样本/不同 batch 分数一致；B/C 旧缓存错配失败；N0/K1、零梯度与失败时已完成费用保留。
5. 构造旧类均值 span 内的不同通道区分样例，确认门控可改变两个旧类的相对 LocalRidge 间隔，而原受保护谱变换不能作同一种改变。这只验证表达能力，不代替真实 support 效果。

## 7. 一轮完整 pilot 与资源口径

建议沿用已确定的同一 160 parent practical residual support pilot，路径仅 R0、R_channel_seq、R_channel_reset。不增设固定门控强度网格或单独 CE 头。所有 outer-held 只在状态固定后评分；完整矩阵完成后同时报告全部 K×新增类数、两个 model/cohort、RX/scenario、old-only 与 new-present。A 缺失写 N/A，B0→B、B→C 旧类下降、C 新旧差距和 H 完整保留。10/1/3 pp 是理想方向，不变成硬门槛。K1 的结构性恒等不能称为正增益，也不能悄悄修改既有晋级规则以“通过”。这一轮首先检验联合机制，不由一张平均 H 表直接晋级。

除准确率外，记录 inner-training 与 outer-held 的分数间隔、winner 改变及正确/错误转换，二者标清用途；inner loss 下降不证明 held 因果。若参数明显变化但错误类竞争间隔没有改善，则该通道空间/监督目标缺乏所需作用；若内层改善而外层普遍变差，则当前训练信号未泛化；若更新长期为零或扰动极小，则如实诊断，而不追加步数碰运气。这些是反驳机制的报告维度，不是隐含选最佳步或额外启动门槛。

已有 JointSpectral pilot 实测总 wall 为 905.381 s；累计 base fit 48.676 s、base score 151.663 s，候选 fit 484.618 s、候选 score 631.357 s；阶段累计值与双 lane wall 不能相加混称单次部署延迟。其最大候选数值状态为 8317392 B，最大 lane RSS 为 455905280 B，环境为 CPU、两 lane、每 lane BLAS 2。新增机制不能直接继承这些数值当自身结果。

新 adapter 参数载荷为 736×8=5888 B；若分别保留 u 与预计算增益则为 11776 B，Adam 两矩额外 11776 B 仅训练使用。还必须加上原 LocalRidge support/alpha/中心状态、五块特征缓存、梯度/分解临时数组及 encoder 的既有部署负担。缓存五块每 N 行为 5888N B；prepared 是否另存归一化副本必须计入实际 bytes。对 N=520，这一份缓存为 3061760 B。不能只报 5888 B 就称星载模型更小。

同一 pilot 的 936 个可训练阶段仍最多 8424 次 objective、25272 次 inner head、936 次额外 final head；加原基线 3168 次，合计最多 29376 次闭式头。proxy 恒等及 N0 复用必须按实际减少。每次推理新增约 736 次通道乘法和五块范数保持变换，但原 LocalRidge 核评分仍在；反向 pair 累加与解算可能远大于 adapter 本身成本。须实测训练/评分分项、峰值 RSS、实际持久状态、部署包与增量传输字节，注明硬件；未测项为 N/A。该方案只确定避免 encoder 反传，尚不能声称总计算低于 LocalRidge。

## 8. 核心实现与审计接口

源码为 `code/cvsrffi/d92_joint_channel_local_ridge.py`，冻结算法常量为 `configs/d92_joint_channel_frozen_20260930.json` 的 algorithm，合成测试为 `tests/test_d92_joint_channel_local_ridge.py`。原 LocalRidge、谱适配和 paused Residual8 均未修改。实现采用 NumPy float64、SciPy 三角求解与全参数伴随，不依赖 Torch。

prepare 返回 `ChannelTraining`，包括不可变原五块、一次构造的 b/a、规范标签/物理 ID、内折问题及可选继承 B。`audit_dict()` 顶层为 `channel_preparation_count`、`prepare_seconds`、`prepared_numeric_state_bytes`、`transient_distance_bytes`、`no_information`、`no_information_reason`、`inherited_state` 和 `inner_folds`。每内折包含训练/持出物理 ID、`original_interaction_centered_trace`、`original_bandwidth_tau` 和数量，折头状态不得取自 inner-held。

fit mode 为 B/C_seq/C_reset；返回 `ChannelLocalRidgeState`，有只读 u、classes、score/predict/audit_dict。最终状态保留原 b/a、raw support 与 labels，用于后续 C 的逐 ID、标签及原始缓存一致性检查；这些 retained arrays 全部计入 `lineage_state_bytes`，并与 `head_state_bytes`、`adapter_state_bytes=5888` 相加得到 `persistent_state_bytes`。`optimizer_state_bytes=11776` 是训练用的两个 Adam 矩，不算部署参数但计入训练资源。现实现没有只保留 5888 B 的低内存部署导出器，不得以参数载荷代替完整 state bytes。

`baseline_state` 签名不变，但不能仅凭 IDs/features/classes 复用。原 LocalRidge state 不存物理标签，因此核心重建原 K，验证实际 alpha 满足当前规范标签的 `(K+I)alpha=Y`，并核对 tau/gamma/核中心。该绑定不做新的 Cholesky，计 `baseline_binding_seconds`、`baseline_binding_distance_evaluation_count=1`、`baseline_binding_factorization_count=0`。单类或零 interaction 核验证精确零头并标 `EXACT_ZERO_HEAD_LABEL_INVARIANT`：此时任何合法标签配置给出同一零预测，不能声称从 alpha 恢复了历史标签。绑定耗时包含在 fit_seconds 内。

每个训练阶段 8 个 `JOINT_CHANNEL_STEP`，另有不含 step 的 `JOINT_CHANNEL_FIT`；prepare 可发 `CHANNEL_INNER_PREPARED`。STEP 保存 u_pre/u_post/anchor、裁剪前真实 gradient、gradient_norm、clipped_gradient_norm、gradient_clip_scale、projection_zero_sum_residuals、active_box_count、实际更新范数及步骤时间。末步后的第 9 次目标单独保存 final_objective，不把第 8 步更新前目标称为最终目标。

总训练目标的 `loss_data/loss_proximal/loss_total` 是 held 间隔平均、近端项及其和，标记 `loss_scope=PHYSICAL_INNER_HELD_MARGIN_PLUS_PROXIMAL`。内折 `held_margin_loss_sum` 才是用于总目标的监督物理和；`head_training_loss_data/ridge/total` 独立记录闭式头自己的训练岭损失。兼容字段 `loss_data_sum` 同 held_margin_loss_sum，而内折原 `loss_data/ridge/total` 同头训练损失，另有明确 loss_scope。汇总不能把头训练损失累计为 adapter 监督目标。

实际工作量通过 `optimizer_steps`、`nonzero_projected_update_count`、`u_changed_from_anchor`、`inner_objective_evaluation_count`、`inner_head_fit_count`、`inner_factorization_count`、`derivative_triangular_solve_count` 和 `final_head_fit_count/final_factorization_count` 分别记录。正常三折阶段为 8/9/27 次更新/目标/内头及 48 次伴随三角求解；最后无梯度评估不做伴随，tau0 退化可进一步减少伴随求解。全 pilot 对应伴随求解实际上界 44928，不能报成每个 objective 都执行反向。

NumericalFailure 保留阶段上下文、完成步骤、当前内折及其已尝试分解、之前完成内折、当前 u/Adam 矩，并把非有限数显式编码为 JSON 字符串。它不静默返回另一个候选。

主任务的核心/编排验证证据为 `E:/type10-7/.codex_tmp/pytest_utf8_1790755858887852000.stdout`。随后只补一项真实近重复回归：从实际 binary64 输入用 Decimal.from_float 构造 100 位显式 interaction 及四端点 Jacobian，验证约 10⁻²⁶ 级平方距离与稳定 VJP，容差 rtol=8×10⁻¹⁵、atol=0；独立运行通过，证据为 `E:/type10-7/.codex_tmp/pytest_utf8_1790756026674225700.stdout`，未修改生产数学或重复已有检查。这些验证说明已测实现行为，不证明 support 泛化或星载资源收益。本候选未真实运行，没有读取任何真实 query，性能收益仍是假设。
