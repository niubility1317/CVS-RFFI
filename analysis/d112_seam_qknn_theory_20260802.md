# D112-SEAM-qKNN轻型球面域适应理论设计

状态：`G1_ANALYZED / HEAD_POSITIVE / SEAM_MOTION_ZERO_DECISION_GAIN / CLOSE_SEAM_DA`

日期：2026-08-02

## 0.结论

D112选择`SEAM-qKNN`：Spherical Equivariant Anchor Motion＋unit-mass qKNN。它不是删除D111安全门后的重跑，而是改变统计映射本身：在单位球切空间估计跨类共享低秩位移，以平行运输和指数映射把Phase1旧类锚沿球面移动；donor不一致只连续降低运输与anchor质量，不再使用`chi/6B/eta`硬gate。分类端保留M0 qKNN的K内多样本结构，用单位质量双专家混合替换一部分support密度，禁止增加旧类票数或logit bias。

第一版设计经WP-DA、WP-HEAD和两轮独立监督审查后被退回：审查发现rank-3内在方差与160维ambient MSE混用，以及无donor时只令`alpha=0`却仍给ground anchor正质量。修订版分离两套方差、加入球面映射Jacobian并把无donor／非旧类改为逐列原样返回M0。两名独立复审者最终均给出`MERGE / P0=0 / P1=0`；该结论只冻结理论设计，不构成实现完成、功能成立或性能收益。

## 1.为什么D111必须永久关闭

D111-r2真实588行archive的K1/K5/K10均为28fold、588query，anchor、score、margin和argmax变化全部为0，三K各自0/168类状态获得正anchor质量。直接原因不是实现故障：`B=0.489572`使`E_t=2.944215–2.990095`，稳定资格要求`||t||≥3.444215–3.490095`，实际`||t||=0.966506–1.014135`。

更强的结构反证是：在单位球几何中，`g`和support均为单位向量，投影残差与几何中位数的范数有界，D111预归一化运输向量的理论上界约为3，而冻结门已经要求至少3.44。因此零激活不是偶然坏fold，不能通过修runner、调`eta/B/gap/rho/rank`解决。

## 2.候选路线比较

|路线|K1信息来源|主要优点|不可接受问题|裁决|
|---|---|---|---|---|
|欧氏anchor＋normalize|其他类support的共享位移|计算小|近零归一化奇点；最坏界导致结构性全回退|永久关闭|
|全局Procrustes／Gram关系保持|5个旧类关系|显式保持类间结构|5点中心化秩最多4；映射欠定，logit校正随注册类数漂移|只作理论诊断|
|逐类support残差直接回写|本类K-shot|最简单|K1把单样本噪声当域漂移，压扁qKNN多峰结构|拒绝|
|共同变换query与support|support估计|形式对称|正交变换在距离中抵消；非等距变换易伤new|拒绝|
|SEAM球面LOO运输|其他5个旧类support|无归一化奇点；K1提供有条件的rank-3 LOO共享位移估计；连续收缩；轻量|依赖共享低秩球面位移模型|主线|

## 3.生成模型与Phase1封存资产

表示位于单位球面`S^{p-1}`，`p=160`。对旧类`c`和域`d`，局部模型为：

\[
\mu_{dc}=\operatorname{Exp}_{g_c}\!\left(
\operatorname{PT}_{q_0\to g_c}(U h_d)+\xi_{dc}
\right),
\]

其中`g_c`是旧类地面锚，`q_0`是共享参考点，`U∈R^{p×r}`是`T_{q_0}S^{p-1}`中的正交共享基，`r=3`，`h_d`是域共享坐标，`ξ_dc`是类域交互。模型只声称多数旧类存在共享低秩球面位移，不声称任意conditional shift可适应。

在任何target访问前，Phase1从多样本域×类聚合中心构造并与checkpoint共同封存。令`t_dc=PT_{g_c→q_0}Log_{g_c}(mu_dc)`；所有方差分成rank-`r`内在坐标和ambient-`p`逐坐标两套，不再复用同名标量：

- `g_c=N(mean_d μ_dc)`；域中心按向量内容的canonical byte order归约；
- `q_0=N(sum_c g_c)`；类锚同样按向量内容而非类ID作canonical归约，若和向量退化则资产无效；
- `U`：每类对`{t_dc}`取top-3投影矩阵，六类投影矩阵等权平均后再取top-3；
- `z_dc=U^Tt_dc`。对Phase1域×类单样本`x_dci`，定义`r_dci=PT_{mu_dc→q_0}Log_{mu_dc}(x_dci)`，先冻结：

\[
\sigma_{0,c}^{2,(r)}=\epsilon_{var}^{(r)}+
\operatorname{Mean}_{d}\operatorname{Mean}_{i\mid d,c}\frac{\lVert U^Tr_{dci}\rVert^2}{r},\qquad
\sigma_{0,c}^{2,amb}=\epsilon_{var}^{amb}+
\operatorname{Mean}_{d}\operatorname{Mean}_{i\mid d,c}\frac{\lVert x_{dci}-\mu_{dc}\rVert^2}{p}.
\]

- Phase1 LOO权重只依赖先行冻结的`sigma_0^(r)`，不依赖后算的`v_g`，因此没有循环：

\[
\omega_{cj}^{P1}=\frac{(\sigma_{0,j}^{2,(r)})^{-1}}
{\sum_{\ell\ne c}(\sigma_{0,\ell}^{2,(r)})^{-1}},\qquad
h_{d,-c}^{P1}=\sum_{j\ne c}\omega_{cj}^{P1}z_{dj}.
\]

- `v_g,c^(r)=epsilon_var^(r)+Mean_d(||z_dc-h^P1_{d,-c}||^2/r)`；令`hat mu_dc=Exp_{g_c}(PT_{q_0→g_c}(Uh^P1_{d,-c}))`，则`v_g,c^amb=epsilon_var^amb+Mean_d(||mu_dc-hat mu_dc||^2/p)+q_c^amb`，其中`q_c^amb`是Phase1封存的int8 endpoint量化误差项；两者衡量类域交互而非把可解释的共享位移算作噪声；
- `tau_h^{2,(r)}=Mean_{d,c}(||h^P1_{d,-c}||^2/r)`：仅用Phase1 LOO坐标形成的内在每坐标共享位移功率；
- checkpoint、feature schema、旧类registry、量化尺度和误差receipt。

所有数值知识保持只读int8载荷及量化尺度，不保存source行、单样本feature、ID、路径、成员清单或可独立替换sidecar。所有上标`amb`的量都是归一化球面endpoint之间的逐坐标chord-MSE；所有上标`(r)`的量都是`q_0`切空间rank坐标的逐坐标MSE，禁止复用符号。令`epsilon_geo=64 epsilon_float32`、`epsilon_var^(r)=epsilon_geo^2/r`、`epsilon_var^amb=epsilon_geo^2/p`；所有variance proxy加入该固定正数底座及对应量化误差，避免`w=∞`或`0/0`，这些常量不允许调参。`g_c`、类projector、`tau/v_g/sigma_0`的跨类归约均按数值内容canonical排序和固定pairwise reduction完成；特征值重根、basis符号、int8解码后的切空间投影、正交化和rank receipt均采用预先固定规则。资产缺失、checkpoint/schema/registry不一致、Phase1任一必需`g_c/mu_dc/q_0` Log/PT chart无效或共享rank退化属于**全局bundle无效**，D112整体解析回M0；不得从target选择`q_0/U/r/tau_h`。

## 4.球面Log、平行运输与Exp

对单位向量`x,y`且不互为对径点，实施时用`atan2`避免`arccos(theta)/sin(theta)`在小角附近失稳：

\[
q=x^Ty,\quad b=y-qx,\quad
\theta=\operatorname{atan2}(\lVert b\rVert,q),\qquad
\log_x(y)=\frac{\theta}{\lVert b\rVert}b.
\]

对`v∈T_xS^{p-1}`，沿最短大圆的平行运输为：

\[
\operatorname{PT}_{x\to y}(v)=
v-\frac{v^Ty}{1+x^Ty}(x+y).
\]

指数映射为：

\[
\operatorname{Exp}_x(v)=
\cos\lVert v\rVert x+
\sin\lVert v\rVert\frac{v}{\lVert v\rVert}.
\]

`||b||=0,theta=0`和Exp零向量均按连续极限处理。封存的数值常量`epsilon_geo=64 epsilon_float32`只判定球面chart或rank是否可定义。Phase1出现`||sum_cg_c||<=epsilon_geo`、任一必需`1+q_0^Tg_c<=epsilon_geo`或构造U/variance所需的Log/PT pair无效时，属于全局bundle无效，所有类原样走M0。

row内另定义`donor_valid(c)`：`||sum_kx_ck||>epsilon_geo`且`Log_{g_c}(s_c)`的固定chart有效。无效类自身仍可作为待运输类使用其他donor，但它不进入任何其他类的`D_c`；若本类support和向量本身退化，无法计算`s_c/e_c`，则本类`I_c=0`并原样走M0。这样全局资产退化与单row类回退不混用。所有判断不读取性能，也不按target结果调整。这些运算始终在球面上，不再出现`N(g+Delta)`的近零分母。

## 5.Phase2严格LOO共享位移

当前row中旧类support原型为：

\[
s_c=N\!\left(\sum_{k=1}^{K}x_{ck}\right).
\]

每个旧类形成共享参考切空间中的rank-3观测：

\[
z_c=U^T\operatorname{PT}_{g_c\to q_0}\log_{g_c}(s_c).
\]

support不确定度分别在rank空间和ambient空间计算：

\[
v_{s,c}^{(r)}=\frac{\sigma_{0,c}^{2,(r)}+\widehat\sigma_c^{2,(r)}}{K},\qquad
v_{s,c}^{amb}=\frac{\sigma_{0,c}^{2,amb}+\widehat\sigma_c^{2,amb}}{K},
\]

其中K1固定两种`hat sigma=0`，不把单样本伪装成协方差；K>1分别使用围绕归一化support原型的rank投影与ambient逐坐标球面dispersion proxy，不声称无偏方差。

定义有效donor集合`D_c={d in Y_old\\{c}:donor_valid(d)}`。非Phase1旧类不进入集合；对待运输旧类`c`严格排除本类：

\[
w_d=(v_{s,d}^{(r)}+v_{g,d}^{(r)})^{-1},\qquad d\in\mathcal D_c,
\]

\[
h_{-c}=\sum_{d\in\mathcal D_c}\widetilde w_dz_d,\qquad
\widetilde w_d=\frac{w_d}{\sum_{j\in\mathcal D_c}w_j}.
\]

加权均值不确定度与donor不一致度分别为：

\[
V_{-c}^{(r)}=\left(\sum_{d\in\mathcal D_c}w_d\right)^{-1},
\]

\[
D_{-c}^{(r)}=
\frac{\sum_{d\in\mathcal D_c}\widetilde w_d\lVert z_d-h_{-c}\rVert^2}
{r\left(1-\sum_{d\in\mathcal D_c}\widetilde w_d^2\right)}.
\]

共享位移的解析收缩为：

\[
\alpha_c=\frac{\tau_h^{2,(r)}}
{\tau_h^{2,(r)}+V_{-c}^{(r)}+D_{-c}^{(r)}}.
\]

这是闭式经验贝叶斯型收缩proxy，不声称精确后验。没有`chi`或行级性能阈值；donor越不一致，`D`越大，运输连续趋近0。若本类support和向量退化、有效donor少于2、权重和无效或加权离散度分母退化，则定义`I_c=0`并直接令`alpha_c=rho_c=0`、返回原M0列且不评估anchor核。这是模型缺信息时的解析回退，不是按表现筛选的gate。

## 6.无硬角阈值的球面运输

先把共享切向量送回类锚切空间：

\[
u_c=\alpha_c\operatorname{PT}_{q_0\to g_c}(Uh_{-c}).
\]

为避免对径点而不引入可调`φ_max`，采用固定光滑径向压缩：

\[
R_\pi(u)=\frac{\pi u}{\sqrt{\pi^2+\lVert u\rVert^2}},
\qquad
a_c=\operatorname{Exp}_{g_c}(R_\pi(u_c)).
\]

`R_pi`把任意有限切向量连续映射到开球`||v||<pi`，没有接受/拒绝边界。令`t=||u_c||`、`beta=pi/sqrt(pi^2+t^2)`、`s=beta t`。`F_c=Exp_{g_c} o R_pi`在`u_c`径向和横向的奇异值分别为：

\[
\kappa_{rad}=\frac{\pi^3}{(\pi^2+t^2)^{3/2}},\qquad
\kappa_{tan}=\beta\frac{\sin s}{s},
\]

零点均按连续极限取1。对`J_c=D F_c(u_c) alpha_c PT_{q_0→g_c}U`，因为PT等距且`U`正交：

\[
\operatorname{tr}(J_cJ_c^T)=\alpha_c^2
\left(\kappa_{rad}^2+(r-1)\kappa_{tan}^2\right).
\]

在`Cov(h_-c)≈(V_-c^(r)+D_-c^(r))I_r`的各向同性proxy下，这给出rank不确定度经非线性球面运输后的**一阶delta-method chord-MSE proxy**，不是有限扰动的精确方差；计算只需标量闭式函数。D112消除了D111硬资格下界造成的必然全回退；共享位移为零、几何无效、数值下溢或anchor与support核完全相同时仍可能零函数差异，因此不预先宣称功能或收益。

## 7.单位质量SEAM-qKNN分类头

M0 support密度保持：

\[
L_c^{sup}(q)=\operatorname{LSE}_{k\le K}\ell(q,x_{ck})-\log K.
\]

SEAM anchor与support使用相同Student-t`ν`、相同类尺度`h_c`和相同logit原点。定义：

\[
e_c=\lVert s_c-a_c\rVert^2/p,\qquad
v_{h,c}^{amb}=\frac{V_{-c}^{(r)}+D_{-c}^{(r)}}{p}
\operatorname{tr}(J_cJ_c^T),
\]

\[
\rho_c=\frac{v_{s,c}^{amb}}
{v_{s,c}^{amb}+v_{g,c}^{amb}+v_{h,c}^{amb}+e_c}.
\]

旧类最终分数：

\[
L_c(q)=\operatorname{logaddexp}\left(
\log(1-\rho_c)+L_c^{sup}(q),
\log\rho_c+\ell(q,a_c)
\right).
\]

令`C_{nu,d_eff}`为M0对所有类共同省略的Student-t归一化常数，完整定义为`logaddexp(log(1-rho)+[Lsup+C],log(rho)+[ell_anchor+C])-C`；实现可代数消去同一个`C`，但必须验证继承的M0 method lock满足`kernel_volume_gamma=1`，D112运行时不得覆盖它，并让anchor与support使用完全相同的`nu/h_c/d_eff`和logit原点。这样每类总质量恒为1，anchor质量从support质量中扣除；禁止裸`K+1`平均、裸max、old bias、role分池和按类ID调权。

所有不在Phase1旧类registry中的注册类，以及`I_c=0`的旧类，直接逐位返回已有M0 score列，不调用`z/Log/PT/Exp`，不通过`v_g=+infinity`模拟回退。运行时只读registry成员关系和组件有效性，不读取query的old/new真实角色。

`p=160`只用于球面几何、chord dispersion和`e_c`；M0冻结的`d_eff=12`只进入Student-t归一化常数、带宽体积项和尾指数，不得混用。

## 8.K1可辨识性与类置换等变

K1不能识别160维任意变换。在Phase1固定rank-3共享位移模型且donor残差满足零均值、有限各向同性proxy的条件下，其他5个旧类各给出一个有噪3维`z_d`观测，`h_-c`是共享位移的LOO条件估计量，不是无条件可辨识的target变换。类`c`的运输不读取本类`z_c`；本类K1仅进入M0密度、两种`v_s,c`和一致性项`e_c`。

任意旧类标签置换会同步置换`g/v_g^(r)/v_g^amb/sigma_0^(r)/sigma_0^amb`行、support bank列和LOO donor集合；`q_0/U/tau_h^(r)`由等类投影聚合保持不变，`h/alpha/a/rho/L`逐类同步置换。basis重根和符号规则只依赖数值几何，不依赖类名；无类ID白名单、专属阈值或专属超参数。

## 9.失败边界

- 多数旧类不存在共享低秩球面位移；
- 类域交互`ξ_dc`主导，donor形成共同但错误的方向；
- target位移主要位于`span(U)`外；
- ground anchor虽改善old，却把new query吸向旧类；
- Phase1参考点、切向基或dispersion与checkpoint坐标不一致；
- support靠近ground anchor对径点，使`Log/PT`失去唯一最短大圆；
- K1 donor噪声使连续收缩仍不足以压制错误方向。

这些情形不能由理论排除。数值/资产退化精确回M0；统计方向是否正确必须由未开封G1的old/new、floor和negative tail决定，不能据G0调`τ_h²/r/ρ`或核参数。

## 10.资源与最小验证

|阶段|上界|说明|
|---|---:|---|
|Phase1聚合|6个小型SVD＋球面Log/PT|一次性，无训练|
|enrollment|`O(CKp+C²r+Crp)`|`C_old=6,r=3,p=160`|
|单query额外|最多`6×160=960MAC`|只评估旧类SEAM anchor核|
|query依赖状态|0B|逐query只读|
|优化器／反向传播|0|纯闭式轻型适配|

下一步只允许一个实现波次：复用D111的source tap读取边界和M0 qKNN接口，但建立新的D112共同封存资产；现有D111 bundle不含两套方差和Phase1 LOO功率，不能直接冒充D112 bundle。随后一个本地验证波次、一次独立P0/P1复审，再运行真实588×K1/K5/K10无真值G0。

G0只审计`positive_rho_count`、`||a_c-g_c||`、logit/margin差异和资源；argmax变化仅作附属诊断。只有三种K在anchor、rho、logit和margin层面全部严格零变化且能给出结构原因，才关闭D112；任何单个K无argmax翻转都不是停止理由。任一K证明合法非零函数后即可进入一次冻结G1，不先跑125，不做参数扫描，也不据G0选择`r/tau/rho`或核参数。

G1采用三个而非机械四个可辨识臂：`M0`为原Student-t qKNN；`M_HEAD_GROUND`以未移动的Phase1地面锚`g_c`作为单位质量anchor expert，其`rho`仍由同一ambient不确定度公式计算，但固定`alpha=0,v_h=0,a_c=g_c`；`M_JOINT_SEAM`为完整D112。三臂分别回答“原基线”“加入Phase1 ground-anchor head”“再加入target support估计的共享球面运动”。不设置`M_DA`：anchor motion本身不产生prediction，必须通过anchor expert进入score；将其伪装成独立臂只能得到机械恒等M0或引入本文未定义的support/query变换。G1只报告`HEAD_GROUND−M0`、`SEAM_MOTION_AT_HEAD=M_JOINT_SEAM−M_HEAD_GROUND`和`JOINT−M0`，不虚构`DA_AT_BASE`。

## 11.理论来源与证据边界

- Pennec，*Probabilities and Statistics on Riemannian Manifolds*：球面Log/Exp、平行运输和流形统计工具，https://www-sop.inria.fr/asclepios/Publications/Xavier.Pennec/Pennec.NSIP99.pdf
- Snell等，*Prototypical Networks for Few-shot Learning*：少样本类条件原型与指数族距离解释，https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html
- Mettes等，*Hyperspherical Prototype Networks*：单位超球面原型分类几何，https://proceedings.neurips.cc/paper/2019/hash/02a32ad2669e6fe298e607fe7cc0e1a0-Abstract.html
- Minsker与Strawn，*The Geometric Median and Applications to Robust Mean Estimation*：稳健位置估计与误差背景，https://arxiv.org/abs/2307.03111
- Ben-David等，*A theory of learning from different domains*：域适应不可识别边界，https://proceedings.mlr.press/v9/david10a.html

这些文献只提供球面统计、原型分类和域适应边界；SEAM的LOO平行运输、光滑`R_π`压缩、连续解析收缩和单位质量qKNN组合是本项目自研设计，不构成性能证据。

## 12.理论复审与数值恒等式

|检查|结果|证据边界|
|---|---|---|
|独立精确公式复审|`P0=0/P1=0 / MERGE`|只读审查，无实验、无truth|
|独立监督复审|`P0=0/P1=0 / DESIGN_FROZEN`|P2仅要求实现保留回退计数与类置换回归|
|`R_pi→Exp` Jacobian迹|闭式与中心有限差分绝对误差`2.82e-11`|随机合成160维切空间，非性能数据|
|球面PT切向与保范|100组最大误差`3.56e-15`|随机合成单位向量，非性能数据|

因此下一步只允许实现本文唯一公式，不得并行保留旧量纲版、调`r/tau/rho`、恢复硬角gate或从G0选择分支。

## 13.真实G1结果与理论裁决

真实source-held r3完整封存63行、三个臂和189个prediction单元后独立打开truth。K1登记42行中，`M_HEAD_GROUND`相对M0的old BA与seen-new均为`+1.3228pp`，H为`+1.9736pp`，old floor为`+4.5855pp`；但`M_JOINT_SEAM`与`M_HEAD_GROUND`在全部63行的prediction逐值相同，`SEAM_MOTION_AT_HEAD`所有指标严格为0。联合状态并非回退：21个唯一package均有6个正ρ旧类，最大`alpha=0.387345`、最大anchor位移`0.014563`，说明球面共享运动在数值层起作用，却不足以改善任何决策。

因此D112的正收益版本是静态Phase1 ground-anchor单位质量head，不是SEAM域运动。保留`M_HEAD_GROUND`作为轻量head组件；关闭SEAM motion域适应路线，不调`r/tau/rho`、不补seed、不运行125。下一条域适应理论必须产生相对该head的独立增益，不能再把head收益记到DA名下。
