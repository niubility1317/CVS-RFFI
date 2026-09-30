# MC 之后的联合设计：函数坐标残差与 LocalRidge

状态：`IMPLEMENTATION_APPROVED_NOT_RUN`。本文件只定义一个下一候选 **FunctionCoordinateResidual8–LocalRidge（FCR8）**；核心、冻结算法配置和合成测试已按本设计落盘，数值验证与真实实验状态见[核心实现说明](D92_FCR8_CORE_20260930.md)。没有复用已完成 MC run 的适配权重。它保留 LocalRidge 为唯一分类器，通过物理 support 内的持出监督训练表征残差。改变的是残差的可学结构及其优化度量，不是扫描 rank、学习率或保持权重。

## 1. 完整证据与可作出的判断

证据来自完整 run `20260930-phase2-d92-mc-residual8-support-m2-r01` 的合法 support 结果和训练诊断。本次完整解析了 `stages.jsonl` 的 4576 条、`curves.jsonl` 的 17604 条和 `teachers.jsonl` 的 1944 条；原始文本及 NPZ 的完整性沿用 collector 的已验证扫描，未重新读取全部原始数组。没有读取 query、ABC、历史评分、总索引或交接。

来源：[完整 support 结果](D92_MC_RESIDUAL8_SUPPORT_RESULT_20260930.md)、[完整训练机制发现](D92_MC_RESIDUAL8_TRAINING_FINDINGS_20260930.md)、[派生诊断 summary](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-mc-residual8-support-m2-r01/results/training_diagnostics/summary.json)。

| 已核实事实 | 对下一设计的含义 |
|---|---|
| 936 个信息阶段全部更新，3544 个接受更新；没有零更新信息阶段 | 不能把效果小归因为训练没有执行 |
| guard 激活、球投影改变位移、keep 拒绝均为 0；1283 个拒绝全是 Armijo 和总目标上升 | 没有证据支持扩大保持 slack；keep 的损失和梯度仍然参与优化 |
| 177 个阶段试探耗尽，全在 train K=3、4；末次仍有非零任务梯度 | 只能说明有限预算及当前方向下的优化不足，不能称已收敛，也不能证明多走几步即可改善 |
| B／seq／reset 内层准确率均值只增加 0.3964／0.3729／0.2030 个百分点 | 连续目标下降未充分转化为分类变化；这些是训练监督统计，不是独立验证 |
| B 教师 q=0 占 28.57%，C 教师占 28.21%；q 均值约 0.2893／0.2916 | 教师不能为全部旧样本提供正 margin 下限。q=0 的 deficit 仍约束负 margin；不能称其无约束 |
| 96 个 new-present OOF 主候选相对 R0：ΔB=−0.1736、ΔCold=+0.0955、ΔCnew=−0.0234、ΔH=+0.0111 个百分点 | 尚未实现明显旧类适应和新旧共同改善；旧类下降幅度缩小部分来自 B 本身下降 |
| 参数坐标广泛变化；完整派生记录没有输入角度、隐层激活、适配前后距离或核变化 | 不能从 U/V 范数声称“几何扰动太小”，也不能声称已发现 latent 退化 |

B 信息阶段的 360 个上下文只有 72 个不同物理绑定，另有 280 个 K1 绑定各重复五种新增类上下文；不同绑定也不因此具有统计独立性。诊断仅覆盖两个 model seed、每 cohort 两个 rx/scene 键、一个 support seed、6 个旧类、新增 0/2/5/10/20 类和 K=1/5/10/20；不覆盖 practical_mid 和其他 RX。

初末总目标均值 B 为 0.372340→0.371475，seq 为 0.466429→0.464561，reset 为 0.467149→0.465908。相应 proximal 末值为 0.001223、0.001045、0.000884。近端项抵消了部分任务与保持项下降，但这些数值不证明近端系数错误。

**待检验假设**：两因子残差使用参数空间的长度与近端项，实际残差功能变化却受固定字典能量及因子尺度影响；在四次更新的预算下，这种坐标依赖可能妨碍有效的 margin 优化。这个假设来自源码中的精确数学性质，不是本轮未测量的几何事实。

## 2. 单一结构与明确取舍

冻结原确定性 DCT 输入字典及 GELU，仅训练其输出矩阵；把 support 上的残差输出变化作为优化坐标和近端度量。LocalRidge 的物理和 ridge、原始 interaction 迹匹配、双几何、教师保持和实际接受检查继续保留。

与 MC 的区别有三项，构成同一个函数坐标机制：

1. V 固定为原始确定性 DCT，而不是训练两个相乘的因子；真实 SFT 参数最多为 U 的 736×8=5888 个。
2. 用当前训练 support 的固定隐层 Gram 建立白化优化坐标，使单位坐标位移具有明确的平均残差输出含义。
3. 近端项从 U/V 参数差改为 support 上的残差函数差；四次更新的预算直接限制这种函数位移。没有自由 CE 分类头，也不增加旧类 logit bias。

这降低了可学容量：字典方向不再学习。若主要瓶颈是固定字典表达能力，而非参数化，FCR8 可能更差。本文不把“参数更少”当作效果或总算力优势，也不预设固定字典一定够用。

### 2.1 固定字典与原始分块

从原 LocalRidge 的一次 `_blocks` 构造取得五个块，维度为 160、96、160、160、160。写为 \(r_\ell=\rho_\ell w_\ell\)；非零块 \(\|w_\ell\|=1\)，零块令 \(w_\ell=0\)。不对已归一化块再次施加 `unit_floor`。

\[
x=\operatorname{concat}(w_1,\ldots,w_5)/\sqrt5\in\mathbb R^{736},\qquad
h(x)=\operatorname{GELU}(V_0x)\in\mathbb R^8.
\]

\(V_0\) 为当前 MC 的前 8 行正交 DCT：第 0 行 \(1/\sqrt{736}\)，其余第 j 行第 k 项为 \(\sqrt{2/736}\cos[\pi j(k+1/2)/736]\)。使用精确 erf GELU；固定 V 不接收梯度，不把它计为可训练参数。所有阶段和查询使用同一字典。

给定当前阶段的原坐标 anchor \(U_a\)，唯一可学函数为 \(v(x)=U h(x)\)，\(U\in\mathbb R^{736\times8}\)。B 的 \(U_a=0\)；C_seq 的 \(U_a=U_B\)；C_reset 的 \(U_a=0\)，但其教师仍来自本行实际 B。

### 2.2 函数坐标：一次薄 SVD，不反传 SVD

当前阶段合法 outer-train support 集为 T，包含 N 个不同物理记录。按物理 ID 排序构造 \(H\in\mathbb R^{N\times8}\)，每行是 \(h(x_i)^T\)。不按类别重采样，不增加 view 或复制样本。

\[
H/\sqrt N=P\Sigma R^T,\qquad
W=R_r\Sigma_r^{-1}\in\mathbb R^{8\times r},\qquad
U(Z)=U_a+ZW^T,\quad Z\in\mathbb R^{736\times r}.
\]

只保留数值可分辨的奇异值，具体政策见第 7 节。Z 初始精确为零，r≤8 是数据数值秩，不是搜索出来的模型 rank。存储并部署原坐标 U；W 是训练坐标，不是额外的 query 归一化器。

在精确算术中，令 \(C_h=H^TH/N\)，则 \(W^TC_hW=I_r\)。因此

\[
\frac1N\sum_{i\in T}\|(U(Z)-U_a)h(x_i)\|^2=\|Z\|_F^2.
\]

这是**饱和与切向投影之前的残差输出度量**，不是最终特征距离、核距离、分类分数或泛化误差的等式。它使白化坐标的单位有可复核的功能含义。

秩亏时，仅更新 T 能辨识的 latent 子空间；anchor 在其余方向的作用原样保留，因为采用加法 \(U_a+ZW^T\)，没有将 \(U_a\) 投影或重新拟合。C 的新 W 可以不同，Z=0 时仍是同一个原坐标 U_B，不能把旧 Z 复制进新坐标充当继承。

在满秩且忽略浮点误差时，若固定字典作任意可逆线性换基 \(h'=Ah\)，相应 \(U'=UA^{-1}\)，两套白化坐标只相差正交变换。Frobenius 目标、梯度归一化、保持半空间和接受条件因此对应同一函数轨迹。秩截断、接近秩阈值和查询落在 support 未辨识 latent 方向时，不宣称一般可逆换基不变；只保证上述原坐标 anchor 保留。奇异子空间内的正交基旋转不应改变数学结果。

### 2.3 每个样本统一保范映射

保持 MC 的 \(\kappa=0.25\)：

\[
t_\ell=(I-w_\ell w_\ell^T)v_\ell,\quad
\delta_\ell=\frac{\kappa t_\ell}{\sqrt{\kappa^2+\|t_\ell\|^2}},\quad
r'_\ell=\rho_\ell\frac{w_\ell+\delta_\ell}{\|w_\ell+\delta_\ell\|}.
\]

零块精确为零；不加入标签、预测类别或 old/new 路由。U=0 的前向直接返回原始 b/a 数组以保持 R0 精确值，但仍计算活跃的 U/Z 导数。查询逐样本按固定顺序求和，训练和单条/批量评分使用相同标量归约路线。

原始 LocalRidge 的完整 interaction 是 \(\phi=(b,a,b\otimes a)\)，维度为 256+480+122880=123616；不是单独的 tensor 项。相应 \(\phi'=(b',a',b'\otimes a')\)。最终 head 使用

\[
D_{ij}=\tfrac12\|\phi_i-\phi_j\|^2+\tfrac12\|\phi'_i-\phi'_j\|^2.
\]

这是拼接特征 \((\phi,\phi')/\sqrt2\) 的平方欧氏距离，保证 PSD Gaussian 核和 \(D\ge d_0^2/2\)。适配映射允许非单射；只检查“原始相等必有适配相等”，不错误要求适配相等推出原始相等。稳定 interaction 差分引擎保留，禁止用大 Gram 项相减替代近重复差分。

## 3. LocalRidge 与监督目标

每个物理内折 f 的 train 集 T_f、held 集 V_f 沿用按各类物理 ID 排序后位置 modulo min(K,3)。held 是合法训练监督，不是独立验证。固定 U 时，每折的 bandwidth、中心化、原始迹、闭式头及 C 教师头只由该折 T_f 构造。

令每个 train 点的最近异类距离为 \(m_i=\min_{j:y_j\ne y_i}D_{ij}\)，\(\tau=\operatorname{median}_i m_i\)，包括零。若原始折 \(\tau_0=0\)，沿用原始等价关系核及其零导数；双几何保持等价关系，不用额外 epsilon 制造带宽。正带宽时

\[
R_{ij}=\exp(-D_{ij}/\tau),\quad C=I-\mathbf1\mathbf1^T/n,
K=\gamma CRC,\quad \gamma=s_0/\operatorname{tr}(CRC),
\]
\[
\alpha=(K+I)^{-1}Y_c,\qquad S=K_*\alpha.
\]

\(s_0\) 始终是原始 interaction 的 train 中心化迹，不改 ridge=1 的物理和约定；使用当前 LocalRidge 的稳定中心化表示和标签编码。\(K_*\) 仅以 train 统计中心化。原始 \(s_0=0\)、单类、迹无法分辨等边界沿用明确退化或技术失败，不静默换算法。

定义 held margin \(m_i=S_{i,y_i}-\max_{c\ne y_i}S_{ic}\)，\(\ell_i(q)=\tfrac12[\max(0,q-m_i)]^2\)。每个类的损失先跨全部内折累积物理和，再除以该类真实 held 数，最后做类别 RMS：

\[
t_c=\frac{\sum_{f,i\in V_f:y_i=c}\ell_i(1)}{n_c},\quad
L_{task}=\sqrt{\frac1C\sum_c t_c^2}.
\]

B 教师是原始 R0 的相应旧类内头；C 教师是冻结当前 B 的 U_B 后，仅用相应旧类内训练 support 重拟合的旧类头。令 \(q_i=\operatorname{clip}(m_i^{teacher},0,1)\)，按旧类定义同样跨折的 deficit 均值 \(k_c\)，\(L_{keep}=\sqrt{\sum_{c\in old}k_c^2/C_{old}}\)。没有正 q 时仍保留负 margin deficit。只有一个旧类时 keep 无错误类竞争，明确不可用。

唯一新目标为

\[
L(Z)=L_{task}(U(Z))+L_{keep}(U(Z))+\tfrac12\|Z\|_F^2.
\]

近端等于 \(\frac1{2N}\sum_{i\in T}\|(U-U_a)h_i\|^2\)，没有遗漏或多除 N。它和原来的 \((\|U-U_a\|^2+\|V-V_a\|^2)/(2N)\) **不是同一个正则强度**；这里选择的是每个物理训练输入平均功能位移的系数 1，而非通过目标评分换算系数。这个选择也是可证伪的设计假设，不称天然最优或“只有重参数化、目标未变”。

### 3.1 内层隔离与共享坐标的准确语义

H 的 SVD 使用全部当前 outer-train 的无标签 h，只作为共享 adapter 优化器的坐标和函数近端统计。它不进入任何 head 的带宽、中心化、标签矩阵或教师原型；给定固定原坐标 U，改变 W 不改变任何 forward。正如共享 U 由所有内 held 标签训练一样，共享优化器可以使用这些合法训练输入；不能据此把内层准确率称作独立 held 验证。

这与“先在全部 outer-train 估计类别方向，再把 inner-held 当未见验证”不同：没有用 inner-held 标签估计内折头状态，也没有将 Gram 的无标签估计冒充内训练专属统计。audit 必须分别标明 `optimizer_coordinate_ids=all_outer_train_ids` 与每个 `head_train_ids/head_held_ids`。真正 outer-held 完全不进入 H、W、教师、损失、接受条件或停步。

## 4. 两套伴随和完整解析梯度

每折用已缓存 Cholesky 进行 task 与 keep 两套伴随，不按 5888 个参数重拟合头。对任一通道 \(G_S\)：

\[
G_\alpha=K_*^TG_S,\quad A=(K+I)^{-1}G_\alpha,
\quad G_K=-\operatorname{sym}(A\alpha^T),\quad G_{K_*}=G_S\alpha^T.
\]

每套伴随两次三角求解，正向 alpha 的两次求解单独计费。中心化、trace scale、Gaussian、tau、interaction 和保范映射均反传，不将 K 或 alpha detach。

正带宽处

\[
dR=R\odot(-dD/\tau+D\,d\tau/\tau^2),\quad
d\gamma=-\gamma\,\operatorname{tr}(C\,dR\,C)/\operatorname{tr}(CRC).
\]

最近异类和 median 并列按原精确并列集合均分导数；偶数 median 平均中间两个值及各自并列组。wrong-max 并列也均分。它们是声明的对称广义梯度，不要求等于所有单侧方向导数。

若 \(L_{task}>0\)，属于类 c 的 held 损失梯度权重为 \(t_c/(C L_{task}n_c)\)；keep 同理且只对旧类，RMS=0 时取零梯度。对双距离仅适配部分乘 1/2；原始分量与 s0 的导数为零。

设反传到 \(v_i\) 的梯度为 \(g_{v_i}\)，则

\[
G_U=\sum_i g_{v_i}h_i^T,\quad
g=\nabla_ZL=(G_U^{task}+G_U^{keep})W+Z,\quad
a=\nabla_ZL_{keep}=G_U^{keep}W.
\]

H/W 固定，V0 无梯度。没有 V=0/U=0 双零死区：B 初始 Z=0 时前向是原 R0，若内折监督经 head 与切向映射在可辨识 h 子空间上有非零投影，则 \(G_UW\ne0\)。不能承诺所有合法数据首步都非零；零类信号、核退化、零字典或纯径向被切向投影消掉都可产生真实零梯度。

## 5. 固定优化预算与缓存

只使用四次外迭代，每次最多三个试探，步长固定为 1/8、1/16、1/32；沿用 Armijo 系数 \(10^{-4}\)、旧类物理 slack \(\epsilon_k=1/(2k\sqrt{C_{old}})\)。没有 rank、步数、步长或权重 grid。

阶段起点 keep 为 K_a，保持上限 \(K_{max}=K_a+\epsilon_k\)。每次以当前剩余 \(b=\max(0,K_{max}-L_{keep})/(1/8)\) 构造

\[
d_0=-g/\|g\|_F,\qquad
d=d_0-\frac{[\langle a,d_0\rangle-b]_+}{\|a\|_F^2}a
\]

（a=0 时不投影）。这是在函数坐标上的保持半空间投影。半空间包含原点，因此 \(\|d\|\le1\)。不再施加坐标依赖的 U/V 产品球；无需额外函数球，因为四次接受更新的路径长度至多 4/8=1/2，故整个阶段 \(\|Z\|\le1/2\)，平均 pre-tangent 功能位移平方至多 1/4。这个界不是 query 位移界，也不保证最终旧类分数保持。

试探 \(Z'=Z+\eta d\)，必须同时满足

\[
L(Z')\le\min\{L(Z),L(Z)+10^{-4}\langle g,Z'-Z\rangle\}+tol_L,
\quad L_{keep}(Z')\le K_{max}+tol_K.
\]

容差延续 \(128\epsilon_{64}\max(1,|\text{比较标量}|)\)。实际位移用于 Armijo；不因为理论下降方向而省略总目标非增检查。无 keep 时仅保留目标条件。梯度真零、投影方向真零或三试探全部拒绝结束当前阶段，保留最后接受状态，不重新选择“最好一步”。

初始 forward 缓存一次，每次接受后的完整 forward 供下一反向复用；拒绝 cache 不得覆盖接受 cache。最后状态只复用最后接受的目标和新拟合的 full head。技术失败保存当前试探、已完成的折、全部实际计数和最后接受状态；不静默恢复不同参数。

## 6. B/C、评分与无信息情形

| 路径 | adapter 初始化 | teacher | 最终 head |
|---|---|---|---|
| B | 本次当前旧 support，U=0；新 H/W；Z=0 | 每折 R0 旧头 | B 的旧 support 重拟合 |
| C_seq，主候选 | 当前本行 B 的原坐标 U_B；新 H/W；Z=0 | 冻结同一 U_B，每折旧类头 | 当前旧+新全部 support 重拟合 |
| C_reset，继承消融 | U=0；共享同一 C H/W | 与 seq 同一 B 教师 | 同一全注册 support 重拟合 |
| Nnew=0 | 对应 B 直接复用 | 不新增教师 | 不重复拟合 |

继承检查包含相同旧物理 ID→label→原 b/a，B 不可变、同一 row/receiver/scenario/model/capsule，不复用任何历史 run 的 target 参数。C 的 U_B 前向起点必须逐样本与 B adapter 相同；但 C 的核、带宽及 head 全类重拟合，不能称旧分数不变。

true K1 只做既定数值核查，不训练或伪造 held；one-shot proxy 的 train K1 不训练，当前流程的 B=R0、C_seq/reset=R0。单类无监督竞争时保留合法 anchor；若全 H=0 则不存在可辨识输出更新，跳过 optimizer 并明确原因。若只有部分内折 tau0/s0 退化，则这些折仍计入真实监督统计，不能只选非退化折；全部折结构上无可学几何时零更新且记录原因。任意单步恰好零梯度与结构无信息分别记录。

query 仅逐样本使用最终 U、固定 V0、原/适配两套 support 和最终 full head，在全部已注册类中 argmax。没有 query batch 统计、role、配额、伪标签或二次路由；truth 在 prediction 固定后由独立评分端连接。

## 7. 数值政策与必要合成验证

H 按物理 ID 规范排序，以缩放薄 SVD 计算，避免先形成小量 HᵀH 再平方损失精度。令 \(\eta_{num}=128\epsilon_{64}\max(N,8)\)，固定保留 \((\sigma_j/\sigma_1)^2>\eta_{num}\)，即针对要逆的 Gram 能量设数值分辨阈值；实际比较使用 \(\sigma_j/\sigma_1>\sqrt{\eta_{num}}\)，避免平方下溢。阈值是算法的数值秩定义，非效果门槛；报告全部 8 个奇异值、阈值和 r。全零 H 精确 r=0。逆奇异值或坐标映射不可表示、非有限量时技术失败，不增加自适应 epsilon、不换秩候选。

验白化应使用 \((HW)/\sqrt N\) 的 Gram，避免 C_h 的下溢；记录其谱范数残差 e，要求 \(e\le\eta_{num}\max(1,\sigma_1/\sigma_r)\)，否则技术失败。阈值显式覆盖逆变换的条件数，不要求病态矩阵具有与良态矩阵相同的无条件 eps 误差；保留规则使该上界至多 \(\sqrt{\eta_{num}}\)。函数等式与路径长度界在正文指精确算术，实际平均平方位移满足 \((1-e)\|Z\|^2\le \sum\|\Delta Uh_i\|^2/N\le(1+e)\|Z\|^2\)，另报告重构舍入残差。数值秩与浮点残差分别记录。重复奇异值的任意正交基可用，不用标签挑方向。C 的 Z=0 直接复制 U_B，避免重构产生末位差；评分统一使用原坐标 U，而非在不同批形下切换 Z/W 路由。

必要的独立合成检查：

1. 非零 Z 下 task、keep、函数近端及全目标有限差分，包含 tau/trace/双几何；B 的精确 R0 前向和非零首步导数同时成立。
2. 完全独立地直接累加 \(\sum\|\Delta U h_i\|^2/N\)，核对等于 \(\|Z\|^2\)；检查梯度是 Z 而非 Z/N。满秩字典换基及重复奇异子空间旋转核对函数轨迹等变，秩亏测试只主张规定的有限边界。
3. 固定原坐标 U 时改 W 不改变 head/score；改变 inner-held 标签不改变相应 head、带宽、中心化、教师 train IDs；改变 outer-held 不能影响任何 preparation/fit。内 held 用于监督的梯度允许改变。
4. C 零增量逐元素继承 U_B，rank 增减不删旧函数；C_reset 用零 U 但同一 teacher；N0 复用、物理 K1 零更新、零/重复特征和单类明确分支。
5. 物理行置换、类别双射重命名、单条与不同批形评分一致；保留 original-equal⇒adapted-equal 和联合距离下界。近重复距离/VJP 对 Decimal 显式 interaction oracle，不能只用谱界掩盖消去误差。
6. min/median/max 并列分别核对声明的分支平均与真实方向导数，不能强行把它们判为相等；边界两侧分别作平滑 FD。
7. 接受/拒绝、半空间投影、实际计数、拒绝 cache 隔离、最终接受状态、技术失败进度和严格 JSON/NPZ IO；不复测不变协议以制造额外门槛。

## 8. 资源、实现接口和机制判据

实现核心模块 `cvsrffi.d92_function_coordinate_residual8_local_ridge`，确切接口合同为：

```python
prepare_function_coordinate_training(
    *, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes, old_classes,
    inherited=None, context=None, log_callback=None, state_callback=None)
fit_function_coordinate_local_ridge(
    prepared, *, mode='B', baseline_state=None,
    log_callback=None, state_callback=None)  # modes: B, C_seq, C_reset_init
evaluate_function_coordinate_objective(
    prepared, Z, anchor_U, *, gradient=True, forward_cache=None)
state.score(*, z_id, fft, t_emb, f_emb, pa_local)
state.predict(**features)
state.audit_dict()
prepared.audit_dict()
```

state 提供不可变 U、固定字典标识、原/适配 support、完整 ID/label 绑定和只读 base_state。prepared 提供 H、W、r、谱/白化残差、原始块、内折及教师审计。objective 返回 `(loss, g_Z, keep_g_Z, audit, forward_cache)`；cache 绑定 prepared、anchor_U 和实际 Z，不接受仅 shape 相同的复用。prepared 共享 C 的 H/W、内折几何和教师，但两种 mode 独立 Z=0，不能共享错误的初始头 cache。本文表格中的 C_reset 是消融简称，API/结构化日志统一写 `C_reset_init`。

日志回调为 `log_callback(event_dict)`，事件名 `FCR_INITIAL/GRADIENT/TRIAL/STEP/FINAL`；梯度事件保留 task、keep、total 范数和方向投影证据，trial 保留独立的三个 pass 标记、实际位移与拒绝原因。`state_callback(key, arrays)` 外置完整 Z/U/anchor_U/g_Z/keep_g_Z/d_Z/W，返回不可覆盖状态引用；拒绝 trial 同样保存。JSON 只含严格有限 Python 原生标量和引用，不用 `default=str`。计数保留 MC 的 initial/teacher/inner/final head、两个伴随通道、objective/forward/backward、trial attempts/completed/accepted/rejected；新增 `latent_svd_count` 和 `dictionary_physical_evaluation_count`，不把 cached 展示再次计费。真实数值失败返回已完成内折和当前试探的实际费用。

每个 preparation 一次 N×8 薄 SVD，成本 O(N·8²+8³)，不建 5888² 矩阵。固定 h 按唯一物理样本计算一次并供内折共享；这只缓存无标签、无参数的字典值。每个 trial 的 U 构造约 O(736·8·r)，每样本残差约 O(736·8)。正向和两套反向 LocalRidge 仍是主要费用，不能由参数减半推断总训练成本减半。

每个信息阶段至多 13 个逻辑 forward objective（初始+4×3），每个至多 3 个内头；四次反向、两通道、每通道每内折两次三角求解，即至多 48 次伴随三角求解。最终 full head 至多 1 个。教师和原 R0 另计，accepted/final cached 不重复计费。若使用当前相同 160-parent 描述矩阵，936 个信息阶段的伴随上界仍为 44928；不是实测耗时承诺。

U 原坐标 float64 为 47104 B；固定 V0 若物化也是 47104 B，必须计入实际常驻或说明由公式重建。Z 最多 47104 B，W 最多 512 B，H 为 64N B，另有 anchor、梯度、试探、教师和 head/cache。当前 MC 的对照实测为 wall 48.4153 min、最大 lane RSS 733.2305 MiB、候选 fit 累计 4365.680 s；这些是下一实验的比较基准，不是 FCR8 的估计结果。部署总包、额外传输、查询推理、峰值内存与训练成本均须实测，含 LocalRidge 与冻结 encoder；原始轨迹归档不能混称模型大小。

本候选必须补齐当前缺失的函数机制量，全部来自已使用的训练数据/forward，不另读 query：H 奇异谱与 h RMS；原坐标参数范数和 Z 范数；真实 pre-tangent 位移 RMS；各块角度、\(\|t\|/\kappa\)、适配 interaction 距离变化、双距离变化和核 Frobenius 变化；原始距离零的 pair 单列，非零相对变化不以任意 epsilon 掩盖；内 held margin 分位数与 winner 变化。训练统计与独立 outer-held 评分分别记录。

**可证伪的机制判断，而非新增晋级门槛：**

- 若函数恒等式或满秩换基等变性失败，属于实现错误，不能讨论性能。
- 若坐标确实改变功能步幅，但相同有限预算内任务 margin、内层正确性和拒绝集中现象没有改善，则“不良参数坐标是主要瓶颈”的解释缺乏支持；不因此继续扫 rank/LR。
- 若内层 margin 明显改变但 outer-held 新旧表现未共同改善，则是训练机制与泛化之间的缺口，不能用更低训练损失宣布成功。
- 若固定字典使可用梯度/秩受限，记录限制；不能归咎于未触发的保持硬约束，也不在同一 run 静默改成学习 V。

必要比较保持 R0、主候选 seq 和继承消融 reset。报告完整预声明 pilot 的 B0、B、Cold、Cnew、H、注册下降与绝对新旧差距及 K×新增类数分层。可以复用当前有限 160-parent 确定性 support 范围作下一诊断，不把完整 4800 设为早期门槛。10/1/3 个百分点仍是理想方向，不是本候选的硬条件；机制统计不自动授权 query 晋级。

## 9. 理论来源与未被论文证明的部分

已核对以下原论文全文中的具体公式和适用条件：

- Bertinetto et al., *Meta-learning with Differentiable Closed-form Solvers*, ICLR 2019，§3.1–3.2、式 (2)–(5)：支持通过闭式 ridge 学习表示、训练集头与持出监督分工以及 sample-space 解法。[论文全文](https://arxiv.org/pdf/1805.08136)。本文不复现其跨数据集 meta-training，也不借用其性能结论。
- Martens, *New Insights and Perspectives on the Natural Gradient Method*, JMLR 21(146), 2020，§12、Theorem 1、式 (14)：说明优化度量与重参数化变换的一致性，且非线性有限步一般只有近似不变。[论文全文](https://www.jmlr.org/papers/volume21/17-678/17-678.pdf)。FCR8 使用固定线性字典函数的经验二次度量，不是 softmax Fisher、不是完整 LocalRidge 自然梯度；本文只对明确的满秩线性换基给出精确算术论证。

FCR8 的固定 DCT–GELU 字典、类别 RMS、教师 q、功能近端系数 1、\(\kappa=0.25\)、双几何 1/2、四次三试探和物理 slack 均是本项目的预声明设计选择。论文及有限差分不保证旧类提升、新类改善、≤1 个百分点遗忘或星载省算力。下一实现和实验必须保留这个边界。
