# D92 条件 Affine 头的独立完整 KKT 数学证书

日期：2026-10-01。状态：`IMPLEMENTED / ROOT_SYNTHETIC_VERIFIED / NOT_A_PERFORMANCE_RESULT`。

[独立分析 helper](../tools/d92_conditional_analysis_math.py)只接收明确给出的有限 float64 数组，不读取真实权重、cache、query/outer、support_summary、配置、全局索引、交接或实验产物。它不接收 labels、role、配额、筛选条件或优化参数，不调用、导入或复刻 candidate 的条件 Schur 前向与伴随。完整不定 KKT 求解是独立分析 oracle，不能用于训练、调参、选步或代替方法原有正定性和数值失败规则。

本文件对应[联合数学设计](D92_POST_AFFINE_JOINT_MATH_DESIGN_20261001.md)式（5）及其完整常数通道。训练正核模块仍见[原实现说明](D92_CONDITIONAL_AFFINE_KERNEL_IMPLEMENTATION_20261001.md)。本 helper 的独立性来自求解原始一阶系统；并非把原实现的 `J/z/s/K_perp/L_perp` Schur 代码改名。

## 明确输入与完整系统

输入固定为 `A[m,m]`、`B[m,p]`、`D[p,p]`、`F[h,m]`、`E[h,p]`、`M_O[m,C]`、`M_N[p,C]`、`M_H[h,C]`、`R_N[p,C]`。要求 `m,p,C>0`，允许 `h=0`。A/B/D 是未中心化的真正 raw kernel train blocks；F/E 是同一核的完整 held cross blocks。所有 prior 和 `R_N` 冻结，helper 不重新拟合 prior，也不推断 `R_N` 的标签来源。

令未知量和完整系统为

\[
\theta=\begin{bmatrix}\beta\\\alpha\\v\end{bmatrix},\qquad
Q=\begin{bmatrix}
A&-B&-\mathbf1_m\\
-B^\top&D+I_p&\mathbf1_p\\
-\mathbf1_m^\top&\mathbf1_p^\top&0
\end{bmatrix},\qquad
Q\theta=\begin{bmatrix}0\\R_N\\0\end{bmatrix}. \tag{1}
\]

第一组方程是旧锚点全部 C 列的完整 residual 等于零。第二组是新点 ridge stationarity。最后一行保留无罚自由常数的 stationarity，不可根据逐行 class sum 为零而删除。部署展开直接为

\[
r(O)=B\alpha-A\beta+v,\quad
r(N)=D\alpha-B^\top\beta+v,\quad
r(H)=E\alpha-F\beta+v. \tag{2}
\]

三组分数分别加冻结 `M_O/M_N/M_H`。每行返回全部 C 列，不按 old/new role 改变竞争集合。完整系统没有中心化、额外 ridge、第二个截距、jitter、伪逆、带宽回退或自动重试。

在 raw train Gram PSD 且 A SPD 时，式（1）等价于显式有限特征的 equality-constrained ridge：

\[
\min_{w,v}\frac12\|Nw+\mathbf1_pv-R_N\|_F^2+\frac12\|w\|_F^2,
\qquad Ow+\mathbf1_mv=0.
\]

其 primal 未知量是 `[w_feature;v;mu_old]`。stationarity 给出 `w=N.T@alpha−O.T@beta`、`alpha=R_N−N@w−v`、`beta=mu_old`，因此可在不同矩阵大小、不同未知量表示下独立对照完整 KKT 系数与分数。

## 公开 API 和两种证书语义

`rebuild_positive_kernel_head(*, A,B,D,F,E,M_O,M_N,M_H,R_N)` 返回 `PositiveKernelHeadCertificate`。它首先用 `eigvalsh` 分别检查 A 严格正定和完整 raw train Gram PSD，再用一次 `numpy.linalg.solve(Q,rhs)` 独立求 `[beta;alpha;v]`。它从给出的 frozen inputs 重建三组 scores，不借 candidate 打印分数完成自己的重建。

`certify_positive_kernel_head(*, 同九项输入, alpha,beta,v,old_scores,train_new_scores,held_scores)` 返回同类对象。它只核验明确给出的已归档系数和三组分数，不进行任何求解或谱分解。它的 `positive_kernel_checked=False`、`general_solve_count=0`、`spectral_diagnostic_count=0` 必须保留真实值。此模式证明给定数组之间满足式（1）至（2）；它不能单独证明核 PSD/SPD、物理数据角色、actual B lineage 或 frozen prior 的真实性。未来 summary 应结合独立 raw Gaussian geometry 重建、旧点无损压缩对应关系及已归档真实分解残差决定充分证据，不能给每条记录新增完整谱门槛。

只更改 `M_O/M_N/M_H` 而保留原归档分数会使 score certificate 失败；只更改 `R_N` 或系数会使 KKT certificate 失败。若有人一致地替换全部输入和输出，数学恒等式仍可能成立；没有独立可信 reference 时，本 helper 不宣称能够识别这样的来源替换。

`.arrays` 为不可写、独立复制的数值 mapping，保留全部九项输入、`raw_A/raw_D`、`Q/rhs/theta`、`alpha/beta/v`、`kkt_error`、三组实际 residual、重建三组 scores，以及三组 `claimed_*_scores`。这些数组足以由未来 summary 独立重建整个系统。`.audit` 保留固定容差、所有实测残差、实际 work counters、谱证据和常驻数组字节；不把理论零记成实测零。

证书检查完整 KKT、旧约束、新点 stationarity、`sum_rows(beta)=sum_rows(alpha)` 以及三组 score reconstruction。另保存真实 class-sum 最大绝对值、完整 raw expansion 的 RKHS norm

\[
\|g\|_{\mathcal H}^2=
\operatorname{tr}(\beta^\top A\beta)
-2\operatorname{tr}(\beta^\top B\alpha)
+\operatorname{tr}(\alpha^\top D\alpha),
\]

以及新点 fit 加 regularizer 的 objective。它没有访问旧标签来虚构完整训练损失，也不把这些数学残差解释为真实 accuracy。

## 完整独立 kernel VJP

`positive_kernel_head_vjp(certificate,G)` 接收 `G[h,C]`，返回 `PositiveKernelHeadVJP`。其完整 RHS 为

\[
Q\lambda=\begin{bmatrix}-F^\top G\\E^\top G\\\mathbf1_h^\top G\end{bmatrix}.
\tag{3}
\]

最后一行 `g_b=sum_rows(G)` 为逐列行总和。即使 `sum_classes(G_i)=0`，它也通常非零。helper 对整个 Q 做一次独立 general solve，没有删除常数边界，没有使用 candidate 的 T、Lambda 或 saddle Schur 解。

由隐式微分，完整对称矩阵上游为

\[
\overline Q=-\operatorname{sym}(\lambda\theta^\top).
\]

取其 O/O 和 N/N blocks 分别给 A、D；B 上游是对两块 `−B` 和 `−B.T` 的贡献相加：

\[
\overline B=-(\overline Q_{O,N}+\overline Q_{N,O}^\top),\qquad
\overline F=-G\beta^\top,\quad\overline E=G\alpha^\top. \tag{4}
\]

M 和 `R_N` 没有梯度通道。`.arrays` 保存完整 `A/B/D/F/E` 上游、`G/g_b/adjoint_rhs/lambda_full/Q_bar/theta/Q`；`.audit` 保存真正完整伴随残差和 solve 工作。A/D 使用完整对称矩阵 Frobenius 内积约定，不再给非对角项额外乘 2。

显式有限 feature 的 old/new/held 上游分别为

\[
\overline O=(\overline A+\overline A^\top)O+\overline BN+\overline F^\top H,\quad
\overline N=(\overline D+\overline D^\top)N+\overline B^\top O+\overline E^\top H,\quad
\overline H=\overline FO+\overline EN.
\]

这保留旧 raw A、旧/新 B 和 held/旧 F 的所有端点。未来上层仍须自行验证完整 Gaussian 距离、真实分支与 adapter 到 Z 的链路；本 helper 不参与方法设计或实测 query 评分。

## 固定数值政策与真实成本

固定 `eps=finfo(float64).eps`、`eta(n)=64*eps*max(1,n)`；64 是预先声明的有限 dense arithmetic 安全倍数，不依据训练效果或分数调整。证书尺寸取 `n=max(m+p+1,h,C)`。shape、dtype 和有限值严格检查，不静默接收列表、float32、非有限值或缺失列。A/D 的对称残差在该尺度内才取半和，原始输入同时无损保存。

完整 solve 残差以 `max(1,||Q||_F||theta||_F+||rhs||_F)` 归一化。各约束和 score 残差用对应各乘积项、常数项和 frozen prior/claimed score 的范数之和归一化，全部使用固定 `eta(n)`。raw PSD 检查只允许 `eta(m+p)*max(1,||raw_train_Gram||_F)` 的负舍入量；不会截断特征值。RKHS norm 只允许按对应二次项规模声明的负舍入量，不把失败值剪成零。

`MathCertificateFailure` 携带到失败时的 `.audit`。shape/dtype/非法有限值为 `ValueError`。无自动重试、jitter 或静默回退。完整 KKT 成功也不会授权训练 candidate 越过其原有病态 A 失败规则；该 oracle 的成功仅证明其所报告数值尺度内的方程残差。

一般不定 `numpy.linalg.solve` 的矩阵 order 为 `q=m+p+1`，每次 forward 或 VJP 都实际调用一次 dense general solve，主要形式工作为 `O(q^3+q^2 C)`。rebuild 还实际做 A 与完整 raw train Gram 的两个 `eigvalsh`，分别有 `O(m^3)` 和 `O((m+p)^3)` 成本。它不复用训练 Cholesky，不能称为比训练分块算法省计算。audit 分别记录 solve 次数、成功次数、RHS 列和元素数、`q^3` 与 `q^2 C` 工作代理、谱诊断次数和对应立方工作代理；这些是形式工作代理，不是测量 FLOPs、耗时或能耗。

residual-only certificate 不求解、不做谱分解，主要成本是矩阵乘法和范数，仍构造并保留 `Q[q,q]` 与充分重建数组，不能声称零计算、零内存或重新证明 PSD。部署系数字节为 `8[(m+p)C+C]`；它不含 prior、geometry、adapter、Q、全部归档数组或 Python 对象开销。`certificate_array_bytes` 和 `vjp_array_bytes` 是保存数组的逻辑总字节，不是进程峰值 RAM。

## 合成用例与验证状态

[合成测试文件](../tests/test_d92_conditional_analysis_math.py)只生成确定性的随机有限 features 和 synthetic frozen priors/R_N。它把 candidate 作为被测对照对象，helper 本身不依赖 candidate。用例包括：

- finite-feature primal equality oracle 与完整 KKT 的 coefficients、三组 scores 和 RKHS norm 对照。
- 五个原核块 A/B/D/F/E 的方向差分。扰动 raw Gram 后显式 Cholesky 实现有限 features，再用 primal oracle 给出独立数值导数。
- old/new/held 全端点 primal 方向差分；完整非零 `g_b` 伴随，以及故意遗漏该常数项的反例。
- 篡改 alpha、beta、v、任一冻结 M、R_N 或归档 scores 的拒绝；故意省略自由常数 rank-one 方向的旧约束可成立、系数总和不成立反例。
- 旧点全部注册列 residual 为零，以及只保护旧列的篡改负测；old/new/held 行与 class 列置换的一致性。
- 远处 F/E 为零时 residual 仍等于非零 v、可翻转 frozen prior decision 的反例。此例明确不能从旧 support 保护推出全域无遗忘。
- 不可写 buffer、输入 dtype/finite/shape、空 held、不对称、非 SPD/非 PSD 和无 jitter/重试工作计数。

编写 worker 没有执行测试、Conda、SSH、Git、实验或实际 support/query 评分。主任务已在 `E:/type10-7/local_envs/ssr-gpu` 串行执行本文件对应测试：**63 passed，0.58 s**。原始输出为 `E:/type10-7/.codex_tmp/pytest_utf8_1790818786669069400.stdout`，stderr 和 JSON 记录同前缀；已核对完整有限特征 primal、五核块及三个端点的独立差分路径。此结论仅验证本 helper 的合成数学正确性。当前没有新方法 accuracy、星载耗时、峰值 RAM/VRAM 或传输测量；这些项目均为 N/A。
