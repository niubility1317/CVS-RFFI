# AJLR 解析自由截距的最小实现映射

日期：2026-10-01。状态：`IMPLEMENTATION_MAP_ONLY_NOT_FROZEN`。

本文将 [D92_AJLR_INTERCEPT_MATH_AUDIT_20261001.md](D92_AJLR_INTERCEPT_MATH_AUDIT_20261001.md) 中已经完成的闭式解和伴随推导映射到当前 [d92_anchor_joint_local_ridge.py](../code/cvsrffi/d92_anchor_joint_local_ridge.py)。**这只是实现边界说明，未冻结新候选，未修改核心、配置、入口或当前 release，未执行测试或实验，也不宣称新方法有效。** 本文只读取上述数学审计、当前核心和已有合成测试。

## 1. 最小路线与不兼容边界

最小路线保留当前旧参考 q、原始旧 support 估计的 τ/γ、DCT8 adapter、函数坐标、RMS CE、近端和 4×12 Armijo。每个 residual head 增加一个由当前合法 train support 解析估计的无惩罚类截距。截距不进入 adapter optimizer，不设置组权重、clip、扫描系数或 query 校准。

使用数学审计式（4）至（7）的**一般 q 合并右端路线**：一次 n×n SPD Cholesky，对 `[E,e]` 同时执行两次 triangular solve，缓存 Schur z/s，再形成截距和 canonical α。不将（n+1）×（n+1）的 saddle 系统当作 SPD 分解；不构造逆矩阵，不加 jitter 或 Schur floor。该路线保持现有中心计算与数据角色，改动范围小于切换到全 train 均值表示。

这里的自由截距不能命名为缓存中的 `b`：当前 `b/hb` 已表示背景几何。建议固定使用 `intercept`，Schur 缓存使用 `schur_z`、`schur_s`，伴随使用 `adjoint_T`、`adjoint_eta`。是否实际采用这些名称、另立模块或新方法 schema，由后续明确的实现任务决定。

当前冻结配置仍声明 `free_intercept=False`、`E=Y-M` 的无截距 solve。未来实现必须使用可区分的新 method/schema 和配置；不能覆盖现有配置、复用旧 archive schema 后静默改变预测公式。本文没有选定新方法名称或发布方式。

## 2. 核心函数映射

| 当前函数或对象 | 最小必要改动 | 对应数学审计 |
|---|---|---|
| `_forward` | 在已形成 K/L/Y/M/E 后，将原 `A^-1 E` 改为合并右端 Schur solve；生成 `intercept`、`schur_z/s` 和 canonical α；train/held scores 加截距 | 式（4）至（7） |
| `_forward` 的正规方程审计 | 检查 `A alpha + e intercept - E`，分母加入 `sqrt(n)*norm(intercept)`；另检查样本维度 `e^T alpha=0`、Schur s 有限且正，并按实际 trace 核验其范围 | 式（2）、（3）、（6）；第 2 节残差口径 |
| `_forward` 的类别和与均值审计 | 保留每行类别和检查，并纳入 `intercept`；继续检查 `q^T K alpha=0`，但完整旧参考 score 均值应为 `q^T M+intercept`，不能继续声称完整函数均值为零 | 式（12）、（13）；第 3 节 |
| `_forward` 的 loss | data loss 使用包含截距的完整 scores；ridge loss 仍只为 `0.5*sum(alpha*(K@alpha))`，不添加截距罚项 | 式（1）、（7） |
| `_backward` | 从 `gscore` 汇总 `g_b=sum_rows(gscore)`；复用 Cholesky 和 Schur z/s，用审计式（23）形成 T；以 T 替换现有 `A^-1 L^T G` 后形成 barK | 式（20）至（24） |
| `_center_vjp` | 最小路线先保留当前完整 Pq VJP，但上游必须是自由截距的完整 barK/barL；验证与审计式（26）的消去形式等价。不能只删参考项而保留旧伴随 | 式（25）、（26） |
| `_backward` 的 Gaussian、interaction、adapter 链 | 保留固定 τ/γ、双距离半半、对称 pair 双向累计、train/held 两端 VJP；最终仍为 `g_U@W+Z`。旧参考 raw train 点仍参加当前 U 的导数 | 式（27）、（28） |
| `_score_head_geometry` | 每个样本在 `kernel_cross@alpha` 后加已冻结 `intercept`；零 residual kernel 分支也必须返回截距，不能保持当前直接返回全零的行为 | 式（7）；第 8 节零核分支 |
| `_make_problem`、`prepare_anchor_joint_training` | prior scores 包含 actual B 的真实截距；继续将其冻结为 M_train/M_held。inner prior 只用同折旧 inner-train，final prior 精确绑定完整 actual B | 第 4、7.2、10 节 |
| `_head_arrays`、`_cache_for_state`、`AnchorJointState.to_arrays` | 将截距及必要 Schur 缓存纳入 lossless archive；C final 的 `prior_B_` 档案必须包含实际 B 的截距或明确的已核实零值表示 | 第 9.3 节 |
| `_resident_values`、`_deployment_values`、fit 的字节计数 | resident 包括真实保留的 Schur/RHS/cache；deployment 包括 residual 截距和实际 B prior 的截距，不能按独立 contrast 数代替数组字节 | 第 9.3 节 |
| `evaluate_anchor_joint_objective` | 继续跨全部 folds 汇总每类 CE，再计算 RMS；只需使用修改后完整 scores 和完整 `_backward`。CE 温度、近端、W/anchor 绑定不变 | 式（20）、（28） |
| `fit_anchor_joint_local_ridge` | INITIAL/TRIAL/GRADIENT/FINAL 缓存必须绑定本次截距头；保持 first-accepted Armijo 和最后接受 cache。Nnew=0 的早返回仍在任何 C solve 前 | 第 8 节 Nnew=0 |

当前 `_forward` 中的 `old_reference_residual_mean_norm` 仍可作为 residual 核函数的中心化检查；它不应改成对完整 score 强制零均值。已有 `fitted_old_reference_mean` 应保存实际完整均值，并增加对应期望值，明确是否包含 actual B prior。

数值检查仍属于直接实现正确性。不得把截距变化转成额外数据重验、固定审批、重复实验审查或性能门槛。具体浮点阈值需在后续实现中使用现有容差体系完成合成验证，本文不额外冻结新的阈值。

## 3. 状态与档案字段

建议至少将以下数值对象与同一 head 的 `K/L/alpha/Y/M/E` 一起保存。它们必须来自同一次 forward，不得由摘要重新拟合或由另一 trial 拼接。

| 字段 | 形状及保存范围 | 必要性 |
|---|---|---|
| `intercept` | C；每个 prior/student/final head，以及推理状态 | 完整预测函数的新增部分 |
| `schur_z` | n；训练 forward/cache/archive | 伴随复用与 Schur 独立复算；不属于必要部署数组 |
| `schur_s` | 标量；训练 forward/cache/archive | 前向与伴随使用同一正 Schur 标量 |
| 合并 RHS/其输出 | n×（C+1）；按实际 buffer 保留情况记录 | 不是必须永久保存的部署状态；若保存应进入真实 resident/cache 字节 |
| `sample_sum_alpha`、其残差审计 | C 或 audit 数值 | 样本维度零和，不能被 `alpha` 每行类别和替代 |
| Schur 与完整等式审计 | audit | 包括 s 的实际 trace 界、正规方程、class sum、`q^T M+intercept` 的参考均值 |
| `prior_B_intercept` | C_B；C final archive | 实际 B 的完整函数继承；旧列映射继续使用 `prior_old_class_indices` |
| 可选 `adjoint_T/eta/g_b` | n×C、C、C；GRADIENT 证据 | 可在必要的无损梯度档案中保存；不应加入部署或伪装为新 head 拟合 |

若已有 INITIAL/GRADIENT/TRIAL/STEP aggregate 只保存 scores，其 head_refs 必须绑定包含截距的新 head 档案。summary 的独立复算要同时核验 Schur solve、完整 scores 和 prior 截距。训练与 inference array schema、state serialization、独立 summarizer 因预测公式改变而需要一同更新；这些是后续实现的集成边界，本文不编辑入口或摘要器。

`intercept` 保存 C 个 float64 值，虽然类间独立自由度为 C−1，实际字节仍为 8C。它是 support 解析学习的决策状态，不能计入 adapter SGD 参数；也不能因“不做 SGD”而将其当作固定或免费状态。

## 4. B 等价与 actual B 继承

当前 public preparation 要求所有注册类相同正 K，按每类物理位置切分的 B inner-train 也保持平衡。B 的 prior 为零、q 为完整 B train 均值。因此数学审计式（11）保证**每个固定 U 上 B 截距为零，原无截距 B 函数和其完整 U 梯度等价**。这是数学等价，不承诺因浮点操作顺序导致的 bitwise 一致。

最小实现仍可统一执行一般 q Schur solve，并记录 B 截距的数值残差；不需要为了证明该结论另外扫描。C 只能继承同一新 run、同一 row/fold 当次合法执行的 actual B，将其已核实的零截距作为完整函数的一部分保存。即使均衡 B 在数学上等价，也不能加载已结束 AJLR 或其他 run 的目标适应 head/adapter；不能凭类型相同或缺失字段就把来源不明头视为零截距头。

不平衡 B 通常有非零截距，原解不等价。已有 `_prepare` 的等量 K 限制不因本映射放宽；不平衡构造只能作为低层公式的确定性合成测试。每行类别和为零不代表样本维度标签均值为零。

C 继续继承同一 row 的实际 U_B 和分类函数，旧列按 class ID 映射、新列补零。actual B prior 的截距与核部分都属于冻结 M，其 C 梯度严格为零。C 不能用“理论上均衡 B 等价”改成另一个 B head，也不能通过新 residual 截距更新 B、源域或 prior。

即使新增截距，C 在 Z=0 仍需完整 residual solve，注册即可改变旧 score；截距没有旧类准确率保持保证。自由截距下 q 的函数等价性只适用于同一 raw R/Q、τ/γ、support、U 和 actual B prior。不能据此重新估计尺度、改 train 集合或删除旧 support raw 几何的导数。

## 5. 退化与复用边界

| 条件 | 未来实现必须保留或修改的行为 |
|---|---|
| 真正 K1 或本阶段合法 train K1 | 不生成 inner-held 风险，不更新 adapter；完整 support head 仍执行解析截距 solve。均衡 B 截距为零，C 只从当前 train support 求截距；proxy 不能填补真正 K1 OOF |
| retained rank0、H 全零 | U 精确保留 anchor、Z 为空；完整 residual head 与截距仍有定义，不能跳过闭式 head 或虚报已训练 adapter 参数 |
| τ0=0、s0>0 | 原始完整 feature 等价核与已固定 γ 不变；截距仍闭式求解，adapter 梯度为零，不合成小 τ。相同输入必须得到同一完整 score |
| τ/γ 缺失，当前零 prior、K=L=0 | 显式零核公式为截距 `mean(Y)`、α=`Y−e*intercept`；当前均衡 support 下截距为零，仍全类并列。0 factor/0 triangular，不新增 kernel 信息 |
| 一般固定非零 prior 下 K=L=0 | 低层数学公式为截距 `mean(Y−M)`、score=`M_h+intercept`。不能用这一一般边界静默改写当前“缺旧尺度时零 B prior”的冻结约定 |
| Nnew=0 | 必须在任何 C preparation/head/截距 solve 前精确复用 actual B；不能另做一次 residual refit。截距为零也不使 refit 等价于复用 |
| C=1 | 类间 contrast 为零；平凡单类预测不是多类注册证据 |
| 同特征异标签 | 保留相同核行和相同完整 score，不用 truth/role 路由、配额或标签修改 |
| 真正非有限值、距离/Cholesky/Schur/等式检验失败 | 技术失败并保留产物；不加入 floor、jitter 或结果驱动 fallback |

零核分支会改变低层 canonical α 的表达：从当前无截距的 E 改为 E−e·mean(E)。均衡且 M=0 时该差为零；不能因此漏掉一般公式的测试。当前 `gamma is None` 的推理早返回必须与截距状态同步修改。

## 6. 资源计量变化

一般 q 合并路线不增加 head 数、factorization 数、Cholesky 阶数、adapter 参数数、optimizer 更新或试探预算。每个有信息前向仍是 2 次 triangular 调用，但 RHS 宽度由 C 增为 C+1；不能用调用数不变声称计算量不变。缓存 z/s 的完整 CE 伴随仍为 2 次、各 C 个 RHS，新增 g_b/eta/T 的 reduction 和秩一修正。

应保留当前 `head_triangular_solve_count`、`prior_triangular_solve_count` 和 `derivative_triangular_solve_count` 的调用语义，并额外记录真实 RHS 宽度/物理规模。可以按每次求解保存 n 与 RHS 列数，或提供明确的累计计数。若把 F 与 z 分开求，原解为 4 次 triangular 调用；若伴随重求 z，额外的两次单 RHS 调用必须计费。不能混用一般 q 的状态与全 train 均值路线的预算。

新增 analytic head 参数数为实际保存的 C；独立 class contrast 为 C−1。新增部署 numeric buffer 通常为 8C bytes，训练缓存 z/s 为 8n+8 bytes；合并 RHS 与 scratch 的实际共享情况决定峰值，不能机械相加理论上界。实际 B 若保存零截距数组，还需计其真实 8C_B bytes。当前 `_resident_values` 的唯一 buffer 计数、cache 字节、records 字节和 `_deployment_values` 必须覆盖这些对象。

每条 residual score 新增 C 次截距加法；prior score也按其实际保存状态计费。原有 distance/kernel/dictionary/adapter 工作并不消失，geometry 的 reference 子集计数继续不能与 raw distance 重复相加。新增包体、上传量、训练/推理时间、峰值 RSS/显存均需由未来实现实际测量；本文全部记 N/A。

EDF 不是求截距或其伴随的必要工作。数学审计第 9.1 节给出的 conditional EDF 如未来另行测量，应独立计额外 solve；本映射不要求追加 EDF、额外拟合或新统计门槛。

## 7. 真正必要的合成验证

未来数值测试应使用有限、确定性构造验证改变的行为，沿用 [现有 AJLR 合成测试](../tests/test_d92_anchor_joint_local_ridge.py) 的数据生成和权限边界。本文未执行这些测试。

1. **解析头与 canonical 条件。** 用独立有限特征带未惩罚常数的 primal oracle，或独立 saddle solve，核验 α/截距/完整 train-held scores；覆盖非均匀 q、非零 actual B prior、不平衡低层标签、重复行和奇异 K。检查完整等式、`e^T alpha=0`、类别零和及 ridge loss 无截距罚项。
2. **完整截距伴随。** 在 C 非零 prior、非均匀 q、`g_b` 确实非零的构造上，对 K/L 和 Z 做方向有限差分。核验审计式（22）至（24），并构造遗漏 g_b 的反例。当前 B/C Z finite-difference 测试需改为完整截距头，不只复测旧无截距 VJP。
3. **参考测度等价。** 保持同一 R/Q/τ/γ/U/prior，比较旧参考 q 与全 train 均值的 canonical α、完整 scores、ridge loss 和 Z 梯度；截距按审计式（16）转换。完整中心 VJP应与式（26）一致，但 raw train 中旧点的实际 U 导数不能被 detach。
4. **均衡 B 等价与不均衡不等价。** 比较每个固定 U 下的新旧 B score、目标和梯度，而非只检查初始 U=0。用低层不平衡构造确认新增截距确实改变函数；不放宽 public equal-K 契约。
5. **actual B 继承与预测复算。** C prior 的核与截距均来自实际 B，同 fold old inner-train/final B 绑定正确；C 不反传 prior。序列化后独立重算 prior+residual+截距，与单样本/任意批切分一致；全类 argmax、class-ID tie、行/类置换与禁止额外输入检查继续成立。
6. **边界与真实费用。** 修改已有 K1/rank0/τ0/缺尺度/Nnew0 测试，验证截距的正确公式、零核 0 solve、Nnew0 精确对象复用。验证一般 q 前向调用数与 C+1 RHS、缓存伴随调用数与 C RHS、实际截距/Schur buffer 字节，拒绝 trial 不覆盖 accepted cache。无需重验不受影响的 IQ 数据或增加实数据查询。

现有 `test_closed_residual_no_intercept_no_second_sample_centering_and_primal_oracle` 不能原样作为新头的 oracle：其“无 intercept”断言专属当前冻结方法。未来 affine 测试应保留 Y−M 的来源，并验证样本均值通过**完整解析截距**表达；不能仅减去 E 的样本均值然后丢掉截距。现有 `test_reference_subset_center_VJP_explicit_reference_perturbation` 仍可测试通用中心化算子，但新方法的完整自由截距梯度还必须满足测度等价与消去公式，不能要求它复现旧头的性能解释。

没有必要再次证明审计中已经给出的公式，也没有必要为截距新增参数网格、独立 seed 搜索、全矩阵早期门槛或 query/outer score 驱动的修改。后续是否实现、冻结并开展新实验，仍由主任务另行决定。
