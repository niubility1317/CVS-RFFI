# 条件 Affine 核解析头与完整伴随

日期：2026-10-01。状态：`SYNTHETIC_VERIFIED / UPPER_METHOD_INTEGRATION_PENDING`。

[独立模块](../code/cvsrffi/d92_conditional_affine_kernel.py)实现[数学设计](D92_POST_AFFINE_JOINT_MATH_DESIGN_20261001.md)式（6）至（9）、（14）至（16）。本模块使用 NumPy/SciPy float64，只处理 `m,p,C>0`、原始旧 Gram A 可稳定求解为 SPD 的情况。它不是完整方法，不构造 Gaussian 核、不更新 adapter、不建立 actual B lineage，也不处理零核、new0、重复旧约束或 tau0 分组。未来上层先按数学契约处理这些分支，再决定是否调用本模块；已证明无损压缩后的 A 若满足输入契约，可作为正常正定矩阵输入。

## API 与形状

`fit_conditional_affine(*, A, B, D, F, E, M_O, M_N, M_H, R_N)` 要求全部参数为有限 NumPy float64 二维数组，不静默转换 float32、列表或非有限值。

| 参数 | 形状 | 含义 |
|---|---|---|
| A | m×m | 真正 raw `k(O,O)`；不能传 `expm1` 或已减常数/中心化的表示 |
| B、D | m×p、p×p | `k(O,N)`、`k(N,N)` |
| F、E | h×m、h×p | 当前 held 的 `k(H,O)`、`k(H,N)`，允许 h=0 |
| M_O、M_N、M_H | m×C、p×C、h×C | 同一路径实际 B 的冻结 padding 分数，由上层负责来源与全部类列绑定 |
| R_N | p×C | 已冻结 `Y_N−M_N`，不再减样本均值；模块不接收 labels 或拟合来源 |

返回 `ConditionalAffineState`：

- `.arrays` 为只读数值 mapping，包含全部九项输入，`J/z/s/c_N/c_H/K_perp/L_perp/alpha/beta/v`，`chol_A/chol_ridge/projection_rhs`，`old_residual/old_scores/train_new_scores/held_scores`。`raw_A/raw_D` 另外无损保存输入，A/D 保存通过对称检查后的半和。`s` 是 0D float64；alpha、beta、v 分别为 p×C、m×C、C。
- `.audit` 包含已执行工作计数、固定容差、实测约束/求解/类别和残差、谱诊断及部署系数字节数。输入和结果均复制为不可写数值 buffer，调用方修改原输入不会改变求解状态。
- `.expansion` 是 `ConditionalAffineExpansion(alpha,beta,v)`。`.residual(k_old=...,k_new=...)` 计算 `k_new@alpha−k_old@beta+v`；`.score(...,M=...)` 加冻结 prior。cross kernel 为 q×m、q×p，M 为 q×C；所有行面对所有 C 列，无 labels、role、配额或重新拟合接口。

前向在 A 上合并求解 `[B,1]`，再构造完整 Schur 补和自由常数 rank-one 项；在 `K_perp+I` 上求 alpha。v 已通过条件核解析消元，部署不再添加第二个自由截距。实际 `B@alpha−A@beta+v` 保留为 `old_residual`，不截断为理论零；`old_scores` 是 M_O 加实测残差。M_N 仅用于完整新 train 分数，旧误差是约束下常数，不影响解析解。

`conditional_affine_adjoint(state,G)` 接收冻结 forward 上游 G[h,C]，返回 `ConditionalAffineAdjoint`。`.arrays` 的 `A/B/D/F/E` 为完整原核上游，另存 `G/T/Lambda/X_alpha/X_T/g_b/projection_adjoint_rhs`；T 为 p×C，后三个 saddle 解量为（m+1）×C，g_b 为 C。使用已有两套 Cholesky：先求 T，再以 `(F.T@G, sum_rows(G))` 求完整 Lambda。A/B/F 的旧端点没有 detach，常数伴随不省略；无 h 列 projection RHS，也不构造或 Cholesky 分解不定 saddle 矩阵。

M_O/M_N/M_H/R_N 在该 API 中全部冻结，没有相应梯度输出。A/D 上游采用完整对称矩阵的 Frobenius 内积约定；上层距离 VJP 需同时处理两端，但不能再次把非对角上游乘二。例如有限特征 O/N/H 的上游为 `(barA+barA.T)O+barB N+barF.T H`、`(barD+barD.T)N+barB.T O+barE.T H`、`barF O+barE N`。

## 固定数值政策与失败

固定 `eps=np.finfo(float64).eps`，`eta(n)=64*eps*max(1,n)`。64 是预先固定的算术安全倍数，覆盖该有限矩阵运算链的舍入余量；n·eps 来自密集内积/求解误差的矩阵规模项。此政策不是任意条件数下的正确性证明，也不由训练效果或 held 分数选择。

令 d=max(m,p,h,C)。求解和对称残差使用 `eta(d)`：求解残差分母为 `max(1, ||matrix||_F||solution||_F+||rhs||_F)`，包含额外常数项时同时加入其范数。输入 A/D 的对称残差先检查，只有舍入量内的不对称才取半和；不改特征值，不加 nugget 或 jitter。

旧 A 的谱倒条件数必须严格大于 `sqrt(eta(m))` 且最小特征值为正。该阈值使一阶条件数放大的舍入包络 `eta(m)*condition(A)` 小于 `sqrt(eta(m))`；无法达到时明确失败，不自动改变核、带宽或约束。A、raw Schur 补、K_perp 各进行一次实际谱诊断。后两者最小特征值允许的负舍入量为 `eta(p)*max(1, construction_scale)`，scale 分别取 `||D||+||B||||J||` 和加上 `||c_N||²/s`；不把负特征值截断，也不以此制造新方向。

旧约束除检查上述 backward residual，还要求实测 `||r(O)||_F <= sqrt(eta(d))*max(1,||M_O||_F,||R_N||_F)`。展开等价、系数总和及应为零的 class sum 使用同一 `sqrt(eta(d))` 尺度包络。无论输入是否 class-centered，都记录实际 alpha/beta/v 和完整分数的 class-sum 残差；仅在全部冻结 M/R_N 原本 class-centered 时要求其延续。归档保存实测值和使用的容差，不写虚构的零。

shape/dtype/非法输入给出 `ValueError`；非 SPD、病态、PSD/等式/约束失败或不可表示运算给出 `NumericalFailure`，携带截至失败时 `.audit` 的 phase、计数及已有残差。没有重试、jitter、伪逆、带宽回退、软约束或参数搜索。

## 实际工作与合成验证

前向 projection/residual 各一次 Cholesky：A 上 `[B,1]` 为 p+1 列，residual 上为 C 列，分别两次 triangular 调用。反向 residual_adjoint/projection_adjoint 各 C 列、两次 triangular 调用，零 upstream 或 h=0 仍按当前实际调用计费，无新增分解。每个 prefix 保存 `_factorization_count`、`_triangular_solve_count`、`_triangular_rhs_count`、`_triangular_rhs_element_count`、`_triangular_dense_work_unit_count`；后两项累计 n×r、n²×r，是工作代理而非实测 FLOPs。`factorization_count` 记录尝试，`completed_factorization_count` 记录成功返回；谱诊断另计。解析部署系数为 `8[(m+p)C+C]` bytes，不包含 prior、geometry、adapter 或 Python 对象开销。

[合成测试](../tests/test_d92_conditional_affine_kernel.py)用显式有限特征的 primal 权重、无罚常数与旧约束乘子组成独立 KKT 系统，不调用实现中的条件 Schur 函数。覆盖完整分数、alpha/beta/v 展开、RKHS 范数、旧 residual、class sum、old/new/held 行及 class 列置换；分别对五个原核块做方向差分，并用独立 primal oracle 验证 old/new/held 全端点导数。另有故意删 rank-one、detach A、删非零 g_b 的反例，逐记录推断、真实费用、冻结状态、病态/非 SPD 和非法输入负测。

margin 测试只用合成点，核验 `margin_C >= margin_padB−sqrt(2)*sigma_perp*norm_r` 的充分界，并展示 Gaussian 远处 cross kernel 为零时 residual 趋于非零 v、可能翻转原 margin 的反例。它不引入运行时 query margin 筛选，也不声称旧 support 保护等于 query 零遗忘。

子任务未执行测试、Conda、SSH、Git 或实验，未读取真实权重、cache、query/outer、support_summary、全局索引、交接或 snapshot。主任务在项目 `ssr-gpu` 环境串行运行上述测试：**20 passed in 2.20s**。完整输出保存在 `E:/type10-7/.codex_tmp/pytest_utf8_1790817454805617800.stdout` 与同前缀 `.stderr`。这是独立 KKT、完整伴随和边界的合成验证，不是实际 support/query 性能验证。

未修改现有 Affine 模块，后续 joint SFT、真实分支距离/adapter VJP、lineage、退化分支、日志与实验集成由独立上层模块负责。三个 eigvalsh 谱诊断与两次 Cholesky 均有实际立方规模计算；上述计数不代表总计算量下降，尚无星载时间/能耗测量。
