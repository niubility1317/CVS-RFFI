# 轻型DA×D92-Lite联合冻结设计

状态：`DESIGN_FROZEN / IMPLEMENTATION_NOT_STARTED / TARGET_DEVELOPMENT_ONLY / NO_NEW_PERFORMANCE_RESULT`

本文件只冻结下一轮实现和小筛选所需的方法、因果设计、资源口径与停止语义。它不包含新性能结果，也不把source-held代理、代码可运行性或理论可行性解释为Target收益。

## 1.研发问题与已有证据

本轮同时回答两个问题：

1.能否只更新极小的support-conditioned状态，使基础模型的160维身份表示在target receiver上更有判别力；
2.能否把formal D92的288维复杂注册管线替换为全类对称、160维、解析且可量化的轻型分类头，并在DA前后都保持正收益。

现有证据只支持“值得继续”，不支持预设胜者：D106轻型DA在source-held开发面取得小幅同向增益；formal D92改善旧类、floor和H但轻微损伤新类，K1不变；D122表明head可以产生更大开发面增益，但联合尾部修正会抵消floor。D106 Target r7技术退出，严格为`NO_PERFORMANCE_RESULT`。

## 2.共同协议与不变量

- Phase2只读取不可变Phase1资产、当前row全部registered-class合法support及其固定received-IQ数学view；
- query零fit、零update、零selection，每条query独立面对全部注册类；
- DA对全部类使用同一状态，不读取old/new role、class quota、truth、query batch或全局重排；
- 基础checkpoint在Phase2冻结；A/B只更新二维`a`一次，C由support直接生成二维`a`；
- base和adapted表示采用同一L2归一化；`M_DA/M_JOINT`复用同一DA state与adapted feature缓存；
- D92-Lite按各臂自己的合法support重新拟合；`M_JOINT`不得复用`M_L92`的base head；
- K1没有类内残差自由度，严格`M_L92=M0`、`M_JOINT=M_DA`；
- source-held只用于Phase1方向学习和资产审计，不模拟26类formal D92性能。

## 3.三条DA候选

三候选共享rank2、5%相对残差预算、同一support视图规则和同一D92-Lite；不扫描层、rank、步数、K专属参数或view。

|候选|唯一干预点|Phase2状态来源|特点|
|---|---|---|---|
|`DA-A-FSRG-time_fuse`|`id_backbone.time_fuse.1`输出、既有ReLU之前|support损失的一次二维预条件梯度|最早、可能改变后续ReLU mask，计算最高|
|`DA-B-FSRG-t2norm`|`id_backbone.t2.norm`输出、既有ReLU之前|与A相同|中层折中，避免把研究固定在浅层|
|`DA-C-RDHA-joint_proj`|`id_backbone.cls_head.joint_proj.0`的320维输入|5维全类对称support summary经封存hypernetwork直接生成|无Phase2反传，最快但更接近晚层adapter族|

### 3.0一次性Phase1资产构建

Phase1资产构建是方法实现的一部分，不是额外性能gate。三候选共用同一份receiver-held×TX/class-LOCO episode清单、同一物理ID互斥规则和同一确定性训练预算：

1.每个fold只用inner source实体。对class等权去中心后的receiver均值差矩阵做canonical float64 SVD，以前两个右奇异方向初始化`U`列和`V`行；奇异向量符号由最大绝对坐标为正固定。有效秩不足2或前两方向因近重根而不能唯一确定时只关闭该候选，不换seed或初始化；同一冻结episode清单中全部support物理ID与全部outer-query物理ID必须全局互斥，包括跨K和跨episode；
2.模型checkpoint全冻结，资产参数用float32前向，loss、梯度方差、summary均值/标准差和投影范数用float64累积；固定K1与K5 episode等权、receiver等权、class等权并按预冻结词典序全批计算；唯一优化器为确定性full-batch L-BFGS，`max_iter=128`、`line_search_fn=strong_wolfe`，单初始化、无early stop、无学习率/epoch/正则扫描；
3.A/B对每个inner episode按下述`S_src→a¹→Q_src`计算outer query交叉熵，只更新`U/V`；`D_F`由inner receiver的逐样本二维梯度方差累计，数值下限固定为`epsilon64×max(1,mean(D_F_raw))`；`rho=0.05×median_inner||p_l||₂`只从Phase1 inner tap计算，并与`U/V/D_F/a_max`共同封存，Phase2不得由target support重估预算；
4.C的`Q`初始化为前两维summary到二维`a`的单位选择矩阵、`b=0`，不另设随机初始化；对同一inner episode由support summary生成`a`，以独立query的qKNN交叉熵只更新`U/V/Q/b`；`m_P1/d_P1`的均值和标准差仅由inner episode summary累计，标准差只使用`epsilon64×max(1,|m_P1|)`防止除零，Phase2固定使用`(s-m_P1)/d_P1`；
5.最终资产只保留量化后的`U/V/Q/b`、必要FP16尺度、`D_F`、`rho/a_max`或`m_P1/d_P1`，不得保留source样本、样本feature、receiver/TX/class键或FP32 sidecar。量化前后必须做函数parity receipt；
6.outer fold只检查物理隔离、标签置换、状态非零和独立query上的可观测函数变化，不设置性能阈值，也不用于A/B/C排序。唯一候选排序发生在全部S0 prediction封存后的一次truth评分。

所有outer审计闭合后，每个候选以完全相同的初始化、目标和128次确定性预算在全部Phase1 source receiver上重建一次最终资产；不得读取outer分数选择checkpoint、迭代或fold资产。最终资产只绑定全source训练清单及其物理ID根，不携带任何fold专属样本状态。C的Phase1 outer路径必须让独立query loss直接反传到`U/V/Q/b`；Phase2的support summary和query forward仍保持无optimizer、query零梯度。

最终DA资产采用固定的对称INT8布局：`U`按rank列量化，`V/Q`按rank行量化，`b`按整向量量化；每个量化组只保留一个FP16 scale。`D_F/rho`或`m_P1/d_P1/a_max`保存为FP16；正统计若转换后下溢为零或非有限，只关闭该候选，不得静默抬升到另一数值下限。Phase2可一次解码为只读float32运行时视图，但不得序列化或携带FP32 sidecar。若`d`为A/B的tap通道数，则其数值payload为`4d+14B`；C的数值payload为`1328B`。C与C=26的D92-Lite合计`5592B`，相对formal D92的`16492B`减少66.09%；A/B按真实tap维度另行代入并报告。量化parity只在固定Phase1 fixture上核验函数与argmax，不读取Target truth。

训练预算固定不等于必须等待完整source性能报告。只要资产闭合、非零、协议负测通过，就直接进入S0；若某候选在固定预算下资产退化为零/常量或无法改变真实checkpoint功能，只关闭该候选，不追加优化轮次。

### 3.1A/B共享的一阶FSRG规则

对唯一tap`p_ell(x)`定义：

\[
p_\ell^+(x)=p_\ell(x)+U_\ell\left[a\odot\tanh(V_\ell p_\ell(x))\right],
\quad U_\ell\in\mathbb R^{d_\ell\times2},\quad
V_\ell\in\mathbb R^{2\times d_\ell}.
\]

Phase1使用物理ID互斥的source episode`S_src→Q_src`。在`a=0`求全类等权support损失的二维梯度，并用inner source receiver间梯度方差形成冻结对角预条件器：

support梯度损失固定只用同一次模型forward在`joint_proj.0`得到的最终pre-ReLU向量`u`构造两个同IQ数学视图：

\[
z_A(u)=\mathcal N(\operatorname{ReLU}(u)),\qquad
z_B(u)=\mathcal N(u).
\]

若且仅若`ReLU(u)`为零范数，`z_A`确定性总化为`z_B`；不重跑、不更换物理样本或LEO观测。对每类分别用另一个视图的class mean构造stop-gradient单位prototype，并以冻结qKNN lock的`temperature=0.85`计算全注册类cosine CE：

\[
L_S=\tfrac12\left[
CE(z_B,\operatorname{sg}(P_A))+CE(z_A,\operatorname{sg}(P_B))
\right].
\]

每个physical support在两个视图中仍是同一个K样本，类别先等权再求均值。K1允许同一物理support的两个数学视图互为prototype/query，但不得把它们计为两个独立样本。该损失只用于support产生二维状态；正式query仍只输出冻结adapted `z_id160`并按全注册类head评分。

Phase1 outer元目标使用与部署相同的identity-metric Student-t qKNN数学和同一K对应的冻结lock。adapted source support先按正式路径构建INT8向量、FP16 scale/class scale并解码为stop-gradient bank；独立source query保持可微。Student-t class logit以float64计算，在正式部署的logit输出点闭合为float32，再转回float64除以冻结temperature计算逐query全类CE。量化support不回传梯度，但query真实checkpoint下游必须直接向A/B的`U/V`或C的`U/V/Q/b`产生非零梯度；不得以tap空间平方误差、float support proxy或手工raw asset代替。

\[
g_S=\left.\nabla_a\mathcal L_S(a)\right|_{a=0},\qquad
D_F=\operatorname{sg}\left[\epsilon+
\frac1{|R_{inner}|}\sum_r
\operatorname{Var}_{i\in S_{src,r}}\left(\nabla_a\ell_i(0)\right)
\right].
\]

\[
a^1=\operatorname{sg}\left[
\Pi_{\|a\|_2\le\rho,\ |a_j|\le a_{max}}
\left(-D_F^{-1/2}g_S\right)
\right],
\]

\[
\rho=0.05\operatorname{median}_{inner}\|p_\ell\|_2,
\qquad a_{max}=\rho/\sqrt2.
\]

Phase1只优化：

\[
\min_{U_\ell,V_\ell}\mathcal L_Q
\left(Q_{src};
\operatorname{qKNN}(\Phi_\ell(S_{src};a^1)),
\Phi_\ell(Q_{src};a^1)
\right).
\]

`sg`截断`a¹`对`U/V`的高阶依赖，但outer loss仍通过非零`a¹`下的残差分支直接更新`U/V`。这是一阶FOMAML式学习，不保留二阶图，不使用额外Fishr penalty或残差正则权重。`D_F`只是Fishr启发的预条件器，不能宣称复现Fishr方差对齐。

`rho/a_max`是Phase1 inner统计并封存的资产。Phase2只对当前row全部registered-class support求一次`g_S→a¹`，不得由target support重估预算；之后support统一重注册，query只运行冻结后的forward。若固定source fixture上`g_S`、`a¹`或outer`U/V`梯度为零，则实现P0失败；不得回退到`a=0`直接训练。

### 3.2C的无反传RDHA规则

令`h(x)∈R^320`为`joint_proj.0`输入：

\[
h_a(x)=h(x)+U\left[a\odot\tanh(V\mathcal N(h(x)))\right].
\]

对每类support先计算二维响应均值`\bar r_c`，再形成标签置换不变的5维summary：

\[
r_{ck}=\tanh(V\mathcal N(h(x_{ck}))),\quad
s=\left[\bar r;\operatorname{vech}\left(
\frac1C\sum_c(\bar r_c-\bar r)(\bar r_c-\bar r)^T
\right)\right]\in\mathbb R^5,
\]

\[
a=a_{max}\tanh\left(Q\frac{s-m_{P1}}{d_{P1}}+b\right),\qquad
a_{max}=0.05\operatorname{median}_{inner}\|h\|_2/\sqrt2.
\]

`U/V/Q/b/m_P1/d_P1`只由Phase1 receiver-held×TX/class-LOCO学习并封存。Phase2只执行summary、固定标准化和一次adapted forward，不运行optimizer或反传。旧D125文档中的独占路线、588条G0和fresh63流程被本设计替代；保留的只有上述候选C公式与协议边界。

## 4.唯一共享D92-Lite

名称：`D92-Lite-DR-OAS-LDA`。输入固定为L2归一化的`z_id160`，`d=160`。K5/K10启用；K1严格alias到qKNN。

对C个registered class、每类K个support：

\[
\mu_c=\frac1K\sum_{i=1}^Kx_{ci},\qquad
r_{ci}=x_{ci}-\mu_c,
\qquad n_{eff}=C(K-1),
\]

\[
s_j=\frac1{n_{eff}}\sum_{c,i}r_{ci,j}^2,
\quad t=\sum_js_j,
\quad A_2=\sum_js_j^2,
\quad\tau=t/d,
\quad\Delta=A_2-t^2/d.
\]

若`t≤0`或任一统计非有限，P0失败；不回退到单位协方差。若`Δ≤0`，令`lambda=1`，否则：

\[
\lambda=\min\left(1,
\frac{(1-2/d)A_2+t^2}
{(n_{eff}+1-2/d)\Delta}
\right).
\]

数值下限只防止浮点下溢，不作为可调正则：

\[
v_{floor}=\max\left(\operatorname{tiny}_{64},
\epsilon_{64}\max(1,\tau)\right),
\qquad
v_j=\max\left((1-\lambda)s_j+\lambda\tau,v_{floor}\right).
\]

等先验仿射头为：

\[
w_{cj}=\mu_{cj}/v_j,
\qquad
b_c=-\frac12\sum_j\mu_{cj}^2/v_j.
\]

在量化前去除对所有类共同的仿射项：

\[
w_c\leftarrow w_c-\frac1C\sum_{c'}w_{c'},
\qquad
b_c\leftarrow b_c-\frac1C\sum_{c'}b_{c'}.
\]

每类使用单平面对称INT8：

\[
\alpha_c=\operatorname{FP16}^+\left(\max_j|w_{cj}|/127\right),
\quad q_{cj}=\operatorname{clip}_{[-127,127]}\operatorname{round}(w_{cj}/\alpha_c),
\quad b_c^{16}=\operatorname{FP16}(b_c).
\]

部署态只保存`q_int8[C,160]`、`scale_fp16[C]`和`intercept_fp16[C]`，无FP32 sidecar、均值、方差、残差、协方差、old/new角色或query状态。评分为`a_c(q)=alpha_c(q^Tq_c)+b_c^{16}`。

该收缩系数是OAS-form对角版本：它借用标准OAS的解析系数，但以对角矩阵的`A2`和池化类内残差自由度替代完整协方差统计。因此只能称“解析OAS-form收缩”，不能声称继承原始i.i.d.Gaussian全协方差OAS的风险最优保证。标准OAS闭式形式见[Chen等，Shrinkage Algorithms for MMSE Covariance Estimation](https://arxiv.org/abs/0907.4698)。

## 5.因果臂与可解释边界

核心筛选只使用同一160维空间：

|臂|表示|head|
|---|---|---|
|`M0`|base z160|qKNN|
|`M_DA[c]`|候选c的adapted z160|qKNN|
|`M_L92`|base z160|D92-Lite|
|`M_JOINT[c]`|候选c的adapted z160|D92-Lite|

交互为：

\[
I_{DA\times L92}=M_{JOINT}-M_{DA}-M_{L92}+M_0.
\]

`R_D92_FORMAL`使用正式288维D62/D81/D92完整管线，只作同row外部参照。`M_L92-R_D92_FORMAL`只能解释为“formal288全管线被z160 Lite全管线替换后的总差异”，不能解释为纯head效应。

formal代码审计确认，若以后运行`M_DA_D92`，必须把adapted z160与同一received-IQ生成的FFT96/RF32重新通过`registered_feature`组装为formal288，再用adapted support重新拟合历史D92；不得复用base D92 head。该臂仅为S1胜者的可选兼容性诊断，不阻塞S0，因此本轮不为它先开发通用formal288 provider。

## 6.最小实验矩阵

|阶段|数据|运行对象|用途|结论级别|
|---|---|---|---|---|
|Phase1资产审计|source receiver-held×class/TX-LOCO|三候选各自方向|训练/封存资产，证明物理隔离与非零功能|source-only机制证据|
|S0|词典序预冻结3receiver×`{K1/new20,K5/new20}`×3scene=18row|三候选|选择唯一胜者|`TARGET_DEVELOPMENT_SELECTION`|
|S1|剩余2receiver×`{K1/new20,K5/new20,K10/new20}`×3scene=18row|仅S0胜者|receiver外推与K10确认|`TARGET_DEVELOPMENT_CONFIRMATION`|
|Target25|本方法未见且具有完整同键D92 artifact的seed×正式5slice|仅冻结胜者|完整真实性能|`TARGET25_SCREEN`|

S0每row的公共输出为`M0/M_L92/R_D92_FORMAL`，三候选各输出`M_DA/M_JOINT`，共9个逻辑输出；K1别名不重复计算。三候选全部prediction封存后才一次性打开评分。

S0只保留三个方向条件：

1.`M_DA-M0`的池化`H>0`；
2.K5下`M_JOINT-M_DA`的池化`H>0`；
3.`M_JOINT-M0`的池化`H>0`且old+new总正确数增加。

不再设置0.5pp级多指标门。`A_old/N/F_old`、逐receiver、逐scene、四个简单效应和交互全部报告。合格候选依次按`min(DA_Q_H,L92_AFTER_DA_H)`、最差receiver联合增益、联合总正确数、端到端资源排序，只选一个胜者。S1失败不递补S0第二名；失败方法直接关闭并研发新原理。

## 7.资源目标

formal D92的核心INT8/FP16数值数组为：

\[
B_{formal}=4\times288+2C\times288+2\times C\times3\times2+2C
=1152+590C\ \text{B}.
\]

D92-Lite为：

\[
B_{lite}=160C+2C+2C=164C\ \text{B}.
\]

|C|formal D92|D92-Lite|减少|
|---:|---:|---:|---:|
|11|7,642B|1,804B|76.39%|
|16|10,592B|2,624B|75.23%|
|26|16,492B|4,264B|74.15%|

query head按与formal报告一致的点积口径由`288C`降至`160C`；C=26时由7,488降至4,160MAC，减少44.44%。K5 head fit必须相对同row、同预计算feature的formal D92达到MAC-equivalent减少≥90%、同机同线程预热后墙钟中位数减少≥50%。

head轻不等于联合方法轻。每个候选还必须单列Phase2 support更新、base/adapt forward、query adapter额外MAC、端到端注册墙钟、峰值VRAM和联合数值状态；计入DA资产后的联合状态相对formal D92至少减少50%。不同GPU的未配对墙钟不能用于候选排名。

## 8.仅保留的实现硬门

1.聚焦协议负测：query零fit/零update/零selection，无clean/source/query truth/role/quota/global reassignment访问；
2.A/B固定fixture上`g_S`、`a¹`和outer`U/V`梯度非零；C的summary置换不变且系数非bias常量；
3.D92-Lite类标签置换等变、K1严格alias、K5非alias、无FP32 sidecar；
4.同一真实checkpoint无query smoke可改变feature、neighbor、margin或argmax至少一项；
5.不可覆盖run ID、本地Git提交、独立Terra Max代码审查`P0=0/P1=0`；
6.N607只做一次短preflight/resource check，然后立即交给唯一Terra Max runner。

重复数据验证、额外authority/signature、通用平台、重复hash wrapper、588条G0、fresh63、D62/D92/SVRN重复125及报告美化均为非阻塞P2或已删除流程。

## 9.agent与文件所有权

|工作|模型|边界|
|---|---|---|
|目标、协议、设计整合、数据/结果分析、晋级|主agent和WP-DATA，`gpt-5.6-sol/high`|保留最终决定权|
|DA核心、D92-Lite核心、联合接口、科学测试、独立审查|互相独立的`gpt-5.6-terra/max`|非重叠文件；作者不得自证|
|hash、manifest、报告骨架、字段检查、执行冻结的本地命令|`Luna/max`|不得编辑科学代码、SSH、启动实验或解释性能|
|N607唯一runner|`gpt-5.6-terra/max`|不得改方法/矩阵、调参、按性能停止或重跑|

## 10.当前裁决

`PROCEED_TO_NONOVERLAPPING_IMPLEMENTATION / S0_AFTER_MINIMAL_P0_P1 / NO_NEW_PERFORMANCE_RESULT`

下一步不是继续扩写理论或添加gate，而是让不同Terra Max agent分别实现DA核心与D92-Lite核心，主agent整合核心4臂；完成最小测试、真实checkpoint smoke、独立审查和Git提交后立即发布S0。
