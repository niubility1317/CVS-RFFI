# D111-r2轻型快速域适应理论收敛报告

状态：`DESIGN_R2_FROZEN / IMPLEMENTATION_PENDING / CURRENT_R1_G0_REJECTED / NO_PERFORMANCE_RESULT`

日期：2026-08-02

## 0.结论

本轮不发布实验，也不扫描rank、权重、阈值或带宽。D111-r1的真实G0当前被拒绝，原因不是发布流程，而是方法输入与理论定义尚未闭合：现有资产不能同时提供与ADV3B02 checkpoint一致的6类源锚、逐类rank-3域基、28×6域类中心和逐坐标RMS chord radius；理论又混用了表示维数160和M0核有效维数12，并遗漏归一化映射接近零点时的误差放大。

经过三路独立审查与一次source-only假设审计，当前候选空间收敛为：

1.拒绝继续D110类“低方差方向应逆方差放大”的度量路线；真实source-held已证明其K1平均H下降2.7957pp、old floor下降10.0529pp。
2.拒绝把checkpoint CosFace权重直接当源类锚。它们能正确编码类顺序和判别方向，但不是当前ReLU后`z_id`的原始类中心。
3.拒绝support/query共同变换、全坐标CORAL/OT、query伪标签和完整高维映射；前者会在距离中抵消，后几者分别不可识别、协议不允许或资源过重。
4.保留并修订D111：只用不可变Phase1多样本聚合类锚与域类几何；Phase2用其他5个旧类support估计共享rank-3位移；以LOO方式运输当前旧类锚；用单位质量Student-t混合而不是old bias加入评分。

D111-r2的核心价值不是“保证增益”，而是把K1可识别性、适配强度、负迁移回退和计算成本都绑定到一个明确统计模型。任何成立条件失败都严格回退M0，不以Target性能调参。

## 1.问题定义与不可绕过的理论边界

Phase2固定接收观测在表示层写为：

\[
z_{c,k}=N\!\left(g_c+Uh+\xi_c+\varepsilon_{c,k}\right),
\]

其中：

- \(g_c\in\mathbb R^{160}\)是Phase1旧类聚合锚；
- \(U\in\mathbb R^{3\times160}\)是由各旧类域漂移子空间等类聚合得到的共享正交基；
- \(h\in\mathbb R^3\)是当前target row共享的低秩位移；
- \(\xi_c\)是类特异域交互，D111不能消除它；
- \(\varepsilon_{c,k}\)是K-shot物理样本噪声；
- \(N(v)=v/\lVert v\rVert_2\)。

没有任何source-only或support-only方法能在任意\(H_d,R_d\)下保证目标风险下降。D111只在“多数旧类共享同一低秩位移、类特异项受控、Phase1锚与当前checkpoint同坐标”这一局部模型下工作。该边界与domain adaptation不可能性结论一致：降低边缘分布差异本身不能在conditional shift下保证目标风险改善。

## 2.历史证据对统计对象的约束

|路线|真实性能证据|被否定的假设|对D111-r2的约束|
|---|---|---|---|
|D62|完整125；351/375状态回退或零接纳，K1全回退|support安全门可预测query安全|不使用support自正确率或hard Pareto gate决定强度|
|D92|完整125；K10/new20旧类+2.622pp、floor+4.600pp，但new-0.653pp；K1恒等|只靠注册后协方差能同时解决old/new|不得按old/new角色分池或增加旧类先验质量|
|SVRN|完整125；相对D62的H-31.838pp|support方差重整天然稳健|不重入方差缩放和回滚组合|
|D106|source-held G1有弱均值正值但负尾；Target未形成结果|小均值即可晋级|D111必须报告LOO稳定性与负尾，不能只看均值|
|D110|真实G0三K非恒等；source-held K1 H-2.7957pp、old floor-10.0529pp|改变argmax或放大低方差方向等于有效DA|G0只验证功能；性能方向只能由未开封G1决定|

所以，下一方法必须改变真实决策面，但不能把“决策改变”当成收益；必须直接利用类均值的跨域结构，而不是把条件方差误解释为判别不变性。

## 3.候选空间的理论比较

|候选|K1可识别性|old/new对称性|主要缺陷|裁决|
|---|---|---|---|---|
|SCPM逆方差metric|由Phase1方差先验提供|统一作用全部类|方差小不等于身份稳定，真实floor崩落|关闭|
|checkpoint权重直接锚|6个旧类权重可用|仅旧类有锚；可用单位质量控制|权重是判别法向，不是ReLU后原始类中心|拒绝直接采用|
|权重＋全局偏置／单尺度|5个旧类可估计160维偏置和1个尺度|同上|source-only LOO几何仍粗，不能替代真实聚合锚|后备理论诊断，不实现|
|support/query共同正交或仿射变换|K1可拟合共享量|全部类一致|共同等距变换在距离中抵消；非等距变换易伤new|拒绝|
|5点Procrustes／全矩阵映射|形式上可计算|可作用全部类|中心化类空间最多4维，160维映射严重欠定|拒绝|
|CORAL／OT／伪标签TTA|通常需query分布|可能统一|query更新和跨query统计不符合`p2_min_v1`|拒绝|
|D111-r2 LOO-GAT|其他5个旧类估计共享rank-3位移|只有旧类有anchor，但各类总质量恒为1，K增大时anchor质量自动衰减|依赖真实Phase1聚合锚和共享位移模型|主线|

## 4.checkpoint权重锚的必要假设审计

ADV3B02使用CosFace头，`id_backbone.cls_head.head.weight`形状为`[6,160]`。CosFace对feature和weight同时L2归一化，因此weight确实是角度判别方向；neural collapse理论也说明终端训练中分类器权重可能与中心化类均值对齐。但这不等于weight就是当前ReLU后`z_id`的未中心化类中心。

只读审计输入为同SHA checkpoint和D105的8400条Phase1 source-only strict tap，每类1400条；没有Target、query或参数选择。checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`，strict tap SHA256=`6626afbf5d5987b2944b53f9b4bddbb6c9397f4c577accb95cea5e0039b24578`，类映射SHA256=`15b5f2327b8f657558fdbc21b2ede83f644a84853110df2a64b19afcabf29a25`。计算先取`x=N(ReLU(pre_relu))`，再按6类分别求均值；没有按结果选择样本或模型。结果：

|量|结果|解释|
|---|---:|---|
|最近CosFace权重source准确率|86.5119%|权重行顺序与6类身份几何真实对应|
|6个类中心最近权重索引|0,1,2,3,4,5|类置换绑定正确|
|直接weight到对应类中心平均夹角|62.2582°|weight不是可直接混入qKNN的原始类锚|
|全局`center=b+alpha*weight`解释率|43.0676%|偏置＋单尺度只解释不足一半类间能量|
|LOO偏置＋尺度预测平均夹角|51.3597°|虽比直接weight好10.90°，仍是粗锚|
|LOO预测最近真实类中心|6/6正确|可作后备source-hypothesis方向，不能取代聚合原型|

因此D111-r2不读取或复制CosFace weight作为`g_c`。权重路线保留为理论对照，不写实现、不启动G0。

## 5.Phase1聚合几何的唯一构造

D111-r2只接受与checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`同坐标的Phase1 strict tap。当前可用的最小正确来源是D106固定588条source tap；它覆盖7receiver×4day×6class的28×6完整网格，每cell-class至少2个互异物理样本。

生成必须在任何G0 fold或Target访问前一次完成，不能按fold、K或结果重建：

\[
x_i=N(\operatorname{ReLU}(pre\_relu_i)),
\]

\[
\mu_{dc}=N\!\left(n_{dc}^{-1}\sum_{i\in(d,c)}x_i\right),
\qquad
g_c=N\!\left(28^{-1}\sum_d\mu_{dc}\right),
\]

\[
B_c=\operatorname{Top3}_{canonical}\left(
\{\mu_{dc}-g_c\}_{d=1}^{28}
\right),
\]

\[
R_{dc}=\sqrt{
\frac{\sum_{i\in(d,c)}\lVert x_i-\mu_{dc}\rVert_2^2}
{n_{dc}\,p}}
,
\qquad p=160.
\]

输出数值载荷只允许保存int8聚合值及其量化尺度，并绑定registry、feature schema、checkpoint/hash receipt和量化误差界；不得保存FP16/FP32原型、source行、ID、路径、count sidecar或全精度exemplar。缺cell、重复physical ID、半径非正、非有限、basis退化、registry或checkpoint漂移时拒绝生成，不补epsilon、不换rank、不扫描阈值。

现有资产不能直接替代该构造：

- D106-RDCE只有共享rank-3 basis、tau和spectrum，没有6类`g_c/B_c/R_dc`；
- D110只有4个条件方差；
- D105没有formal aggregate asset；
- 旧v2组件绑定不同checkpoint，且radius是P90余弦距离而非逐坐标RMS chord scatter；
- D99开发aggregate缺少正确radius与formal lineage。

G0阶段允许一个显式`NONFORMAL_G0_FUNCTIONAL_ONLY`聚合件，只验证相同数学是否能改变真实决策；不得伪造formal字段或进入G1。正式G1仍要求同checkpoint、同语义的不可变Phase1 deployment bundle。

## 6.共享子空间与LOO运输

共享投影为：

\[
P_{cons}=\frac{1}{6}\sum_{c=1}^{6}B_c^\top B_c,
\qquad
U=\operatorname{Top3}_{canonical}(P_{cons}).
\]

量化后必须满足谱隙证书和`U U^T≈I`。对当前row的旧类support：

\[
m_c=N\!\left(K^{-1}\sum_{k=1}^Kx_{c,k}\right),
\qquad
r_c=U(m_c-g_c).
\]

为避免本类K1噪声自我复制，类\(c\)的位移只由其他5类估计：

\[
\hat h_{-c}=\operatorname{GeoMedian}_{32,\,damping=1/2}
\{r_j:j\ne c\}.
\]

资格首先要求Weiszfeld primal-dual gap不超过运行时确定计算的\(\epsilon_F\)，且至少3/5 donor位于Phase1预封存包络\(B\)内。bundle中的\(B\)必须是把量化舍入误差向上包含后的保守上界，不能使用反量化点估计替代上界。在此条件下，低秩坐标估计先使用条件误差界：

\[
E_h=6B+\epsilon_F,
\qquad
\lVert\hat h_{-c}-h\rVert\le E_h.
\]

3/5共识只能证明donor彼此一致，不能排除“共同但错误”的位移。因此它是资格条件，不是性能保证。

## 7.归一化稳定性修正

D111-r1直接使用：

\[
a_c=N(g_c+U^T\hat h_{-c}),
\]

但当预归一化向量接近0时，\(N(\cdot)\)不是Lipschitz；原来的\(E^2/p\)不能约束角度误差。r2固定：

\[
t_c=g_c^{q}+(U^{q})^T\hat h_{-c},
\qquad
\eta=\frac12,
\]

\[
E_{t,c}=\epsilon_{g,c}+E_h+\delta_U\lVert\hat h_{-c}\rVert_2.
\]

其中，\(g_c^q,U^q\)是Phase2实际读取的int8反量化值，\(\epsilon_{g,c}\)是类锚L2量化误差上界，\(\delta_U\)是共享基的算子范数量化误差上界；二者均由Phase1量化receipt给出，不从support或Target拟合。以真实未量化量\(g_c^*,U^*,h\)为参照，按

\[
t_c-t_c^*=(g_c^q-g_c^*)+(U^q-U^*)^T\hat h_{-c}+(U^*)^T(\hat h_{-c}-h)
\]

得到\(\lVert t_c-t_c^*\rVert_2\le E_{t,c}\)。资格条件固定为：

\[
\chi'_c=\chi_c\mathbf1\{\lVert t_c\rVert_2\ge\eta+E_{t,c}\}.
\]

若条件成立，则实际与未量化运输向量的范数均至少为\(\eta\)，由归一化映射的标准界：

\[
\lVert N(t_c)-N(t_c^*)\rVert_2
\le\frac{2E_{t,c}}{\eta}.
\]

因此anchor不确定度代理修正为：

\[
v_{a,c}=v_{g,c}+\frac{(2E_{t,c}/\eta)^2}{p}.
\]

`eta=1/2`是唯一新增固定数值，只用于证明归一化稳定和fail-to-M0，不由support表现或Target结果选择。条件失败时`rho_c=0`。这里\(v_{g,c}=28^{-1}\sum_dR_{dc}^2\)只是跨域类内扰动的保守不确定度代理，不声称是\(g_c\)均值估计量的严格抽样方差。

## 8.双维定义：`p=160`与`d_eff=12`

两个维数承担不同职责，必须同时冻结：

|符号|冻结值|用途|
|---|---:|---|
|`p`|160|真实embedding坐标数；RMS chord scatter、`v_t`、`v_a`、`e_c`均按它归一化|
|`d_eff`|12|M0锁定Student-t核有效维数；只进入核归一化常数和尾指数|

因此：

\[
S_c^2=\frac{1}{(K-1)p}\sum_k\lVert x_{c,k}-m_c\rVert^2,
\qquad
e_c=\frac{\lVert m_c-a_c\rVert^2}{p},
\]

而Student-t常数保持：

\[
C_{\nu,d_{eff}}=
\log\Gamma\frac{\nu+d_{eff}}2-
\log\Gamma\frac\nu2-
\frac{d_{eff}}2\log(\nu\pi).
\]

不得为了D111把M0的`kernel_effective_dim`从12改成160；也不得把12用于160维RMS chord dispersion。

## 9.无调参适配强度与old/new保护

目标均值的chord-dispersion proxy为：

\[
v_{t,c}=\begin{cases}
v_s,&K=1,\\
\max(v_s/K,S_c^2/K),&K>1.
\end{cases}
\]

最终混合质量：

\[
\rho_c=\chi'_c
\frac{v_{t,c}}{v_{t,c}+v_{a,c}+e_c}.
\]

因为\(m_c=N(K^{-1}\sum_kx_{c,k})\)，\(S_c^2\)不是围绕未归一化样本均值计算的无偏坐标方差；它与\(v_s,v_g\)统一解释为单位球面chord dispersion的保守代理。该量只通过冻结公式控制anchor质量，不作概率校准声明。

它不是手工weight：当support均值不确定时，合格anchor获得更大质量；K增加、target均值变可靠时，`v_t`约按`1/K`下降，anchor影响自动减弱；transport与本类support不一致时，`e_c`使质量连续下降。

新类固定`rho=0`。为避免旧类因多一个anchor天然获得更多证据，全部类保持单位质量：

\[
L_c^{sup}(q)=\operatorname{LSE}_k\ell(q,x_{c,k})-\log K,
\]

\[
L_c(q)=\operatorname{logaddexp}
\left(\log(1-\rho_c)+L_c^{sup}(q),
\log\rho_c+\ell(q,a_c)\right).
\]

未适配旧类和全部新类与M0同原点、同先验、同Student-t核；禁止old bias、裸`K+1`平均、裸max、按role调权和query选择。

## 10.可辨识性、误差传播与失败边界

### 10.1为什么K1可识别

本类K1只用于M0 support密度和`e_c`，不用于构造自己的transported anchor。anchor由其他5个旧类的support和Phase1聚合知识产生，因此不是本类单样本的复制。K1仍可能`rho=0`或不改变argmax；非恒等只能由G0观测，不能从公式保证。

### 10.2何时可能正收益

若至少3个donor满足共享rank-3位移，Phase1锚误差和类特异项小，且运输向量远离归一化奇点，则anchor可降低K1旧类中心估计误差。单位质量和`rho(K)`衰减避免固定旧类logit偏置，理论上比D92的old/new任务分池更少产生new交换。

### 10.3明确失败情形

- target位移主要位于`span(U)`外；
- receiver效应与TX强交互，导致多数类没有共同`h`；
- Phase1聚合锚与当前checkpoint、feature schema或registry不一致；
- 多于2个donor同时异常；
- `||t_c||<eta+E_t`，归一化误差不可控；
- 旧类anchor虽稳定但把new query吸向旧类，表现为seen-new或floor负尾；
- K1的5个donor噪声过大，`3/5`形成错误共识。

这些情形触发回退或后续G1淘汰，不触发参数扫描。

## 11.资源上界

|阶段|复杂度／状态|说明|
|---|---|---|
|Phase1一次性聚合|`O(588×160×3)`加6次小型SVD|无训练、无optimizer|
|Phase2 enrollment|约2880投影MAC＋2880个固定Weiszfeld标量步|6旧类、rank3、32步|
|单query额外成本|最多`6×160=960MAC`|只评估旧类anchor密度|
|7类示例状态|当前实现4711B；r2增加少量稳定性标量|远低于256KiB|
|query依赖状态|0B|逐query只读|

不设置RSS gate，不做硬件微基准作为发布前置；实现后只记录真实数值state bytes和公式MAC。

## 12.最小验证顺序

当前不发布实验。理论冻结后的唯一顺序为：

1.实现r2归一化稳定资格和双维语义测试；
2.实现一次性`NONFORMAL_G0_FUNCTIONAL_ONLY`源聚合器，输出固定小型数值件；
3.真实checkpoint无query smoke只核查checkpoint／feature／registry绑定、有限性和零query更新；
4.独立审查`P0=0/P1=0`；
5.一个真实588条、28fold、K1/K5/K10合并G0，只看anchor／score／margin／argmax变化，不读accuracy；
6.任一K的`argmax_changed_count=0`即关闭r2；只有三K均非零才进入一份固定、未开封source-held G1；
7.G1若方向为负或有不可接受floor/new负尾，关闭机制并研发下一统计对象，不调`rho`、rank、`eta`、包络或核参数。

G0不是性能实验；G1也只回答机制方向。只有G1通过后才讨论固定Target25，完整125继续保持不运行。

## 13.当前裁决

|项目|裁决|
|---|---|
|D111-r1真实G0|`REJECT_CURRENT_RELEASE`|
|D111-r1理论|有正收益路径，但存在归一化P1和维数定义缺口，已由r2设计取代|
|checkpoint权重锚|判别方向有效，原始锚假设不足，拒绝实施|
|D111-r2设计|独立审查初始`P0=0/P1=2`；已吸收`E_t`量化传播与`p/d_eff`修正并冻结；代码仍待实现和独立复审|
|真实性能|`NO_NEW_PERFORMANCE_RESULT`|
|N607实验|未授权、未启动|

## 14.理论来源

- Ben-David等，*A theory of learning from different domains*：域适应风险与不可识别边界，https://proceedings.mlr.press/v9/david10a.html
- Wang等，*CosFace: Large Margin Cosine Loss for Deep Face Recognition*：归一化feature／weight的角度分类几何，https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_CosFace_Large_Margin_CVPR_2018_paper.html
- Papyan等，*Prevalence of neural collapse during the terminal phase of deep learning training*：终端训练中中心化类均值与分类器权重的自对偶现象，https://pubmed.ncbi.nlm.nih.gov/32958680/
- Liang等，*Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation*：冻结source hypothesis的source-free转移思想；其query伪标签训练不适用于本协议，https://proceedings.mlr.press/v119/liang20a.html
- Minsker与Strawn，*The Geometric Median and Applications to Robust Mean Estimation*：几何中位数的稳健位置估计与数值误差背景，https://arxiv.org/abs/2307.03111

以上文献只提供理论工具，不构成本项目性能证据。D111-r2的LOO域锚运输、单位质量混合和协议化回退是本项目自研组合。
