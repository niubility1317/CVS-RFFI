# D92 组均衡 SupportMetric gate 推导与限定审查

日期：2026-10-02。范围只包括新增组均衡 gate 的数学一致性及其冻结源码的直接 P0/P1 检查。本文不读取 query、历史目标评分、真实输入或产物，不修改 core，也不执行数值、Git、SSH 或实验。root 已授权推进候选源码，本文不增加审批要求。

## 1. 可实施结论与条件

该候选可以沿用现有 finite-barrier gate 的 raw PSD kernel、free intercept、scaled SPD Newton 与完整解析求导。必要条件是：权重严格正、绑定当前 inner-train 的物理样本和注册类；旧、新两组及各注册类在该 train 中均存在；总监督质量保持 N；所有旧对新阈值保留；weighted loss、梯度、曲率、误差模型和 dual audit 同步修改。

原型只定义冻结几何；全部监督权重、head、gate 与校正依旧来自合法目标 support。权重中的类数是 train 注册信息，不是 query 真实类数或配额。B 仍是该候选路径实际拟合的 old-only B；C 必须继承它，new0 必须 exact reuse，不能把其他路径的 B 当作本路径结果。

这些条件使内层监督目标和求导自洽。它们不证明 H 或 query 性能提升，也不证明总训练成本下降。

## 2. 两组质量与 free intercept 尺度

设固定 train 的物理样本数为 N，组 \(r\in\{O,N\}\) 的注册类集合为 \(\mathcal C_r\)，\(C_r=|\mathcal C_r|\)，类 c 的合法 train 数为 \(n_c>0\)。定义

\[
 w_i=\frac{N}{2C_{r(i)}n_{y_i}}.
 \tag{1}
\]

逐类质量为 \(N/(2C_r)\)，因此

\[
 \sum_{i\in O}w_i=\sum_{i\in N}w_i=N/2,
 \qquad\sum_iw_i=N.
 \tag{2}
\]

当前合法训练契约为每类相同正 K，旧类数 6、新类数 q。此时 \(w_O=(6+q)/12\)，\(w_N=(6+q)/(2q)\)。相对于原单位权重，q<6 时新组逐样本权重增加，q>6 时旧组逐样本权重增加；它不是始终加重新组，也不是按哪一组当前 accuracy 较低来适应权重。式 (1) 仍是唯一权重定义；不能根据准确率再选择组质量。K1/no-held 只关闭外层校正更新，不允许省掉 C 的最终新 head 或 gate。

保留 weighted **sum** logistic 与 \(\tfrac12\|g_0\|^2\) 的系数 1。因为总质量仍为 N，这没有把原 RKHS 正则尺度悄悄改成平均 loss 下的另一个系数。free intercept b 不正则，也不能删掉；组权重不等于一次固定 logit offset。

无 kernel、无 barrier 的纯截距 toy model 有：原 gate \(b=\log(n_O/n_N)\)，两组均衡 gate \(b=0\)。这只是解析极简模型。在 finite-barrier 的实际中心点，截距 stationarity 为

\[
 \sum_iw_i(\sigma(g_i)-t_i)=\sum_{i\in O,j\in N}\mu_{ij},
 \quad\mu_{ij}=\zeta/s_{ij}>0.
 \tag{3}
\]

所以 \(\sum_iw_i\sigma(g_i)=N/2+\sum\mu_{ij}\)，不是 N/2。kernel 项、阈值及 barrier 都参与实际校准；不能宣称实际 b 为零，也不能只比较 b 来判断该机制。

权重绑定应在 canonical 类列、实际物理 train 顺序确定后建立。归档需保留权重及其 train 标签/类计数来源。样本排序只允许相应同步重排，不能改变 class-ID 与物理标签关系。

## 3. barrier 目标、预算与 dual audit

旧组 target 为 \(t_i=1\)，新组为 0。设 \(g=K\alpha+b\mathbf1\)，K 为同一 raw PSD kernel。新目标仅替换 logistic 监督权重：

\[
 \Phi=\tfrac12\|g_0\|_{\mathcal H}^2
 +\sum_iw_i[\operatorname{softplus}(g_i)-t_i g_i]
 -\zeta\sum_{i\in O,j\in N}\log s_{ij},
 \quad s_{ij}=g_i-a_{ij}>0.
 \tag{4}
\]

保留 m 个全部旧、新 pair，\(m=n_Oq\)，并令

\[
 \zeta=\frac{N\epsilon_{\mathrm{bar}}}{m},
 \quad\epsilon_{\mathrm{bar}}=10^{-4}.
 \tag{5}
\]

N、m、w 与 \(\zeta\) 在固定 fold 中均为离散 train 元数据，求校正导数时 \(dw=d\zeta=0\)。total weight=N 使原每物理监督质量的 approximation gap 预算保留。精确中心点满足 \(\mu_{ij}s_{ij}=\zeta\)，其对原 weighted hard-objective 的 primal-dual gap 为 \(m\zeta=N\epsilon_{\mathrm{bar}}\)。这是中心路径理论，不是某个浮点状态的实测精确 dual certificate。

候选 dual 使用 weighted logistic conjugate。令 \(p_i=\sigma(g_i)\)，\(\alpha_*=-q^{\mathrm{eff}}\)，则相应候选表达式为

\[
 D_*=-\tfrac12\alpha_*^\top K\alpha_*
 -\sum_iw_i\{p_i\log p_i+(1-p_i)\log(1-p_i)\}
 +\sum_{ij}\mu_{ij}a_{ij}.
 \tag{6}
\]

不能复制未乘 w 的旧 entropy sum。weighted logistic 对偶变量 \(v_i=w_i(p_i-t_i)\) 的 box 为 \([-w_it_i,w_i(1-t_i)]\)，等价于 \(p_i\in[0,1]\)。严格截距可行性需要 \(\sum q^{\mathrm{eff}}=0\)；残差小只能作为近似平衡证据。若未证明精确可行，仍须标注 candidate dual，不宣称严格 lower bound 或 verified exact gap。

free b 给出严格可行初始化：\(\alpha=0\)、\(b>\max a_{ij}\)。两组正 logistic 质量控制两端，RKHS 二次项控制函数尺度。固定阈值时，函数与截距的最优解唯一；PSD 奇异 K 可以有等价系数表示，canonical \(\alpha=-q^{\mathrm{eff}}\) 消除该表示歧义，不需要 K 的伪逆或 jitter。

## 4. weighted 梯度、稳定数值和 JVP

令 \(\beta_{ij}=\zeta/s_{ij}^2\)。effective gradient 与 curvature 是

\[
 q_i^{\mathrm{eff}}=w_i(\sigma(g_i)-t_i)
 -\mathbf1_{i\in O}\sum_j\mu_{ij},
\]

\[
 D_i=w_i\sigma(g_i)\sigma(-g_i)
 +\mathbf1_{i\in O}\sum_j\beta_{ij}>0.
 \tag{7}
\]

数值上应沿原 label-aware logistic residual：旧样本用 \(-w_i\sigma(-g_i)\)，新样本用 \(w_i\sigma(g_i)\)，不能在极端 logit 上用 `p-1` 的消减误差代替。logistic curvature 应用稳定 \(e^{-|g|}/(1+e^{-|g|})^2\) 再乘 w，不能用已舍入概率的 `p*(1-p)`。真实 underflow、非有限或无可解释条件数仍是技术失败，不增加曲率 floor。

objective 与 roundoff 模型也必须同步乘权重：例如 logistic score-input 扰动项是 \(\sum_iw_i\Delta g_i\)，weighted reduction 的绝对项是 \(\sum_i|w_i\ell_i|\)，q 的传播项使用新的 D。若式 (1) 的除法产生了浮点权重，其表示/计算舍入也应按模型计入；不能仅因总质量理论上为 N 就声称实际向量逐 bit 满足式 (2)。这些是数值诊断尺度，不改变科学目标或允许按结果放宽 tolerance。

定义 pair-to-physical operator

\[
 (\mathcal B\,da)_i=\mathbf1_{i\in O}\sum_j\beta_{ij}\,da_{ij}.
\]

canonical 方程为 \(\alpha+q^{\mathrm{eff}}=0\)、\(\mathbf1^\top\alpha=0\)。对固定 w、\(\zeta\)、标签求微分：

\[
 (I+DK)d\alpha+D\mathbf1\,db
 =-D\,dK\,\alpha+\mathcal B\,da,
 \qquad\mathbf1^\top d\alpha=0.
 \tag{8}
\]

用 \(S=\sqrt D\)、\(P=I+SKS\)，设

\[
 R=-S\,dK\,\alpha+S^{-1}\mathcal B\,da,
 \quad z=P^{-1}R,\quad u=P^{-1}S\mathbf1,
 \quad h=(S\mathbf1)^\top u>0.
\]

则

\[
 db=\frac{(S\mathbf1)^\top z}{h},\quad
 d\alpha=S(z-u\,db),\quad
 dg_{\mathrm{eval}}=dL\,\alpha+L\,d\alpha+\mathbf1\,db.
 \tag{9}
\]

实现 \(S^{-1}\mathcal B da\) 时可用 \(S_i\sum_j(\beta_{ij}/D_i)da_{ij}\)。比值非负且每行总和不超过 1；D 必须是新 weighted D。多方向共用同一个 SPD 因子与 stacked RHS，不应把 r 个方向虚计为 r 次 Cholesky。free-intercept RHS 及其 Schur solve 仍计实际工作。

## 5. VJP、阈值和完整 head 链

对于 gate eval covector \(\bar g\)，令 \(c_\alpha=L^\top\bar g\)、\(c_b=\mathbf1^\top\bar g\)。伴随满足

\[
 r+KDr+c\mathbf1=c_\alpha,\qquad
 \mathbf1^\top Dr=c_b.
 \tag{10}
\]

令 \(z=P^{-1}S c_\alpha\)、\(u=P^{-1}S\mathbf1\)、\(h=(S\mathbf1)^\top u\)，则

\[
 c=\frac{(S\mathbf1)^\top z-c_b}{h},\quad
 y=z-uc,\quad v_D=Sy=Dr.
\]

在 K 按对称矩阵变化的口径下

\[
 \bar K=-\operatorname{sym}(v_D\alpha^\top),\qquad
 \bar L=\bar g\alpha^\top,\qquad
 \bar a_{ij}=r_i\beta_{ij}
 = (v_D)_i\frac{\beta_{ij}}{D_i}.
 \tag{11}
\]

直接得到 \(v_D\) 及 bounded ratio，避免先从大数相减恢复 r 再乘 D。新权重没有新增可训练输入；若未来把 w 当连续输入，其额外 VJP 为 \(\bar w_i=-r_i(\sigma(g_i)-t_i)\)，本候选不启用这条路径。

阈值仍为

\[
 a_{ij}=d_i-\log p_B(y_i\mid x_i)+\log p_N(j\mid x_i).
 \tag{12}
\]

旧 inner teacher 在冻结实际 B 校正上只由 old inner-train 拟合；final prior 是实际 full B。对 C 校正微分，\(dd_i=d\log p_B=0\)。负 margin 不 clamp，所有 pair/ties 保留，故

\[
 da_{ij}=dh_{ij}-\sum_k p_N(k\mid x_i)dh_{ik},\quad
 \bar h_{ij}^{\mathrm{bounds}}
 =\bar a_{ij}-p_N(j\mid x_i)\sum_k\bar a_{ik}.
 \tag{13}
\]

新头必须在 new inner-train 上解完整 raw Ridge 与 free intercept。令 \(A=K_{NN}+I\)、中心化固定目标 Y，则

\[
 A\alpha_N+\mathbf1 b_N=Y,\quad\mathbf1^\top\alpha_N=0,
\]

\[
 A\,d\alpha_N+\mathbf1\,db_N=-dK_{NN}\alpha_N,\quad
 \mathbf1^\top d\alpha_N=0,
\]

\[
 dh=dL_N\alpha_N+L_N\,d\alpha_N+\mathbf1\,db_N.
 \tag{14}
\]

old-train 上式 (13) 的 covector 必须与 eval 新条件头的直接 covector 合并再反传 Ridge。旧、新 cross-kernel 和 train/eval 两端均保留；不能仅对 new-new block 求导，也不能省掉截距/随特征变化的训练均值。链的末端是当前 physical basis 校正与完整 raw kernel JVP，不做 encoder backward。

最终全类 score 的导数仍为

\[
 ds_o=\sigma(-g)dg,\qquad
 ds_j=-\sigma(g)dg+d\log p_N(j).
 \tag{15}
\]

对 RMSCE 的样本权重 \(\rho_i=\ell_{y_i}/(C R n_{y_i})\)，直接 eval gate covector 为 \(\rho_i(\sigma(g_i)-t_i)\)；直接 new-head covector 为 \(\rho_i\mathbf1_{t_i=0}(p_N-e_{y_i})\)。旧样本仍通过 gate 阈值的新头依赖反传。构造预测 Fisher 时必须使用完整式 (15) 的各类 JVP，不能用 CE 的一个标量梯度代替。

## 6. 与 H 的关系及成本边界

固定条件头时，weighted logistic 正是旧、新组各一半、组内类均全类 CE 的 gate 部分。这个等价只针对固定条件头的 gate 子问题；解析新 Ridge、移动阈值和外层校正并没有整体改成一个两组均衡 CE 优化器。它也不是 H。H 的旧、新 accuracy 是离散 argmax 统计，且对较弱组更敏感；权重在 train 固定，不从 H 的导数、held 指标或 query 信息更新。外层仍为当前 RMSCE，校正由同一 physical metric、一步与固定 Armijo 预算接受。

冻结旧条件使 C 不能纠正旧类内部排序；新条件排序正确的样本仍可能被跨组竞争挤出。组均衡假说只针对内层监督质量，不解除这些结构边界。CE 改善和经验 accuracy/H 不单调，亦不构成人口风险或泛化保证。校准 surrogate 的 excess-risk 理论不能直接升级为本候选一步 RMSCE/H 的单调性。[Bartlett、Jordan 与 McAuliffe，一手作者稿](https://www.stat.berkeley.edu/~bartlett/papers/bjm-ccrb-05.pdf)

没有新增可训练坐标、源样本或地面统计。可选权重数组 payload 为 \(8N\) 字节；它不等于 wire 或 process memory。kernel \(O(N^2)\)、全部 slack \(O(n_Oq)\)、gate 因子 \(O(N^3)\)、new Ridge \(O(n_N^3)\)、r 方向 stacked RHS、fold/final/trial/失败工作均仍存在。weight 计算及乘法是 \(O(N)\)，不保证 Newton 迭代数、墙钟时间或总内存下降。真实性能、硬件成本与传输字节尚未测量或证实，不填为零。

## 7. 冻结源码限定审查记录

数学阶段冻结之后，root 告知以下四个新路径已 FREEZE，本次只读检查其新增改动：

- [组均衡 gate](../code/cvsrffi/d92_group_balanced_barrier_gate.py)与[gate 合成测试](../tests/test_d92_group_balanced_barrier_gate.py)。
- [组均衡 SupportMetric core](../code/cvsrffi/d92_group_balanced_support_metric_joint_local_ridge.py)与[core 合成测试](../tests/test_d92_group_balanced_support_metric_joint_local_ridge.py)。

另仅有界核对新 `run/preflight/publish_d92_group_balanced_support_metric_joint_probe.py` 的 producer/schema/method 身份与显式 runtime 源闭包；没有复审旧控制模块、读取实际 spec 或验证远端 release。

初次源码检查中，式 (1) 至式 (15) 的新增 weighted 目标、q/D、entropy、free b、完整 K/L/all-bounds JVP、原 pure scaled VJP 复用及新 Ridge 链均相符。core 从实际同方法 B 的 U 坐标出发，校验物理旧 support、标签、上下文与资源，拒绝旧方法 state；new0 原对象返回。原模块只被纯函数别名/调用，没有重绑定其全局。weight 状态只读、成功与失败状态保留相应来源，新 forward/JVP 的实际 SUM 计数进入完整账本。公开原型几何与监督标签角色未改变。

### 7.1 直接 P1：未接受的 trial 下溢直接中断 gate

初版 `fit_group_balanced_barrier_gate` 的 line-search 在 rate=1 上调用 `_evaluate`，其 `LOGISTIC_CURVATURE_UNDERFLOW` 直接进入外层 failure，尚未接受的试探因而没有机会缩步。合法的零 kernel、六旧一新、全部 bounds=−30 例中，初始 b=−29，平衡监督的最优截距接近零；初始曲率小，完整 Newton trial 可以过大。有限目标有解，而试探值超出可求导的浮点范围，不应被视为已经接受的状态，也不能用 jitter/floor 修改目标。

root 告知首次原生合成运行 19 case 中 18 PASS、1 FAIL，唯一失败为该例，原始证据前缀为 `.codex_tmp/pytest_native_activation_1790905144054525900`。本审查未读取或见证该日志，只把 root 告知与直接源码路径分开记录。作者已接收此局部修复；初审其余范围未发现其他直接 P0/P1。

### 7.2 修复后的唯一一次定点 readback

作者再次 FREEZE 后，本审查只读回上述 trial loop 与对应新增合成回归。新代码仅在未接受 candidate 的 `_evaluate` 周围捕获 `ArithmeticError`：只将明确的 `FloatingPointError`、非有限 logit/slack/目标、非严格 slack、曲率下溢/无效曲率，以及 trial 的 slack/目标舍入模型不可解析视为数值域拒绝，记录 `last_invalid_trial_code/step_size`，消耗本次原 trial 并折半。结构错误与未列举的错误继续抛出。初始化、current 和 final 的 `_evaluate` 不在这个局部捕获中，仍严格检查并保留技术失败。

拒绝分支在更新 alpha/b/value 之前 `continue`，没有把 candidate 替换为 current，也没有使用旧 candidate 派生量。所有 `_evaluate` attempt 已进入原 objective/物理记录账，耗尽原 line-search budget 仍产生 `LINE_SEARCH_LIMIT`；失败数组重建自最后实际接受的 alpha/b。没有新增 jitter、曲率 floor、容差、试探数、监督权重或自动重试。SUM/MAX ABI 不变，只增加两个诊断标量。

新增仅 1 trial 的回归检查：初始 b=−29、alpha=0、slack=1 均保留；2 次 objective evaluation、14 次 weighted physical record evaluation、1 次实际 factor、2 次 triangular call 与 budget failure 一致。原零 kernel 成功例另检查确实发生过 invalid trial，然后继续满足原 stationarity、严格 slack 和 free-b 条件。这些是已阅读的测试断言，不是本审查执行结果。

该直接 P1 的状态为 `SOURCE_FIX_READBACK_VERIFIED`。在本次限定新 gate/core delta 与静态控制身份/闭包范围内，没有剩余直接 P0/P1。root 正独立执行受影响 gate 合成测试，本审查不把待执行/未见证的数值结果写成 PASS，也不要求重跑未改旧模块。性能、完整 head/JVP 误差证书、实际硬件成本与远端发布状态继续未证实。
