# D119 GN-ISF-48/r1 Phase1科学证伪设计

状态：`DESIGN_FROZEN_R2 / REJECT_GN_ISF_UNVERIFIABLE_CONFOUND / STAGE2_CLOSED / NO_NEW_PERFORMANCE_RESULT`

## 1.裁决

本轮只保留一个候选：`GN-ISF-48/r1`。它作用于冻结ADV3B02 checkpoint的`id_backbone.time_fuse.1`，即`Conv1d(100→48)→GroupNorm(16,48)→ReLU`中的GroupNorm。候选不扫描其他23个GN层，不扫描rank、正则、步长或K专属公式。

最终裁决是`REJECT_GN_ISF_UNVERIFIABLE_CONFOUND/STAGE2_CLOSED`。现有D102、D103、D105和JP4归档都没有早期GN激活、raw received-IQ或GN JVP，无法从旧feature文件补算新资产；更关键的是，实际Phase1 source archive schema、真实manifest和真实NPZ均不存在独立数值CFO元数据。第6.4节已预注册“CFO字段不存在则关闭候选”，因此不启动新的Phase1 falsifier，不进入实现、G0、G1或Target25。该结论不是性能失败，而是必要混杂证伪不可执行；GN-ISF永久关闭，不换层、不升rank、不补造CFO标签。

只读事实证据如下：

- Phase1 source archive加载类`code/cvsrffi/rxid_metabias4_source_archive.py`的`POOL_MEMBERS`只含`z_id,z_dom,pre_relu,labels,receiver_ids,day_ids,physical_ids,scenario_names,observation_ids,class_ids`，且明确不接受或持久化clean-IQ/received-IQ；
- 真实`E:\type10-7\automation_reports\CV-SincNet\d104_r1_angq_feasibility_20260725\local_d104_source_split_v2_builder_releasefix1_20260725\L_s\manifest.json`仅含archive/checkpoint/runtime hash、candidate、fraction、physical-ID唯一性、role、schema和访问声明等字段，无CFO/carrier/frequency-offset；
- 同目录真实`features.npz`仅含`z_dom,pre_relu,receiver_ids,day_ids,tx_labels,physical_ids`；
- 从相同IQ临时估计CFO再作为独立truth会构成循环证据，禁止用于补门。

## 2.≤20行可行性卡

1.冻结基座为`best_joint_safe_ssdg.pth`，SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
2.唯一干预点为`id_backbone.time_fuse.1=GroupNorm(16,48,eps=1e-5,affine=true)`。
3.干预位于time-path首个ReLU之前；DOM分支、RCN和`z_dom`完全不读。
4.状态维数固定`r=1`，K1/K5/K10使用同一公式。
5.只调GN affine的一个Phase1学习方向，基座195个checkpoint tensor均冻结。
6.每个3通道GN组的gamma/beta方向和均为0，消除组公共平移/缩放坐标。
7.方向以Phase1输出JVP能量归一，并用最大绝对坐标固定符号。
8.公共状态只由old support拟合；new support不得重估状态。
9.K1的六个old support必须来自六个独立物理样本；same-IQ view不增加K。
10.Stage2闭式fit无ridge、无optimizer、无query fit/update/selection。
11.资产只含两个C48 INT8方向、两项FP16 scale和六个INT8 old-class锚。
12.新增数值payload为`1,084B`；运行时状态只有一个INT8标量。
13.现有tap不能构造该资产，必须Phase1重新forward固定source received-IQ。
14.科学证伪同时做receiver-held、class/TX-LOCO和物理ID互斥query；held receiver不得进入方向或锚。
15.必须证明状态跨类稳定，而不是由某个TX残差主导。
16.必须证明功能切线不属于D102、GRB-JP4、late-FiLM或固定末端PSD族。
17.任一P0失败即`REJECT_GN_ISF_EQUIVALENT_OR_UNIDENTIFIABLE`。
18.本轮已在实现前触发第17项P0，故停止，不消耗Phase1/N607实验资源。

## 3.为什么选择time_fuse.1

真实checkpoint经安全loader严格重建：`DualCVSincNetDisentangle`共有195个state tensor，missing/unexpected/skipped均为0。ID分支含10个GroupNorm；最早的是`id_backbone.time_fuse.1`，C=48、groups=16。其后依次经过ReLU、time_down、t1/t2/t3、pool、projection和联合分类路径。这个位置满足三个必要条件：

- 它早于多层非线性，输入相关JVP可能改变ReLU mask，不能仅凭层名等价为末端metric；
- 它只在ID分支内，不读已出现TX泄漏风险的DOM/RCN路径；
- C48允许以两个INT8方向和一个标量状态保持千字节级资产。

选择它不表示方法已成立。GroupNorm对组内正缩放和常数平移本来近似不变，因此GN-ISF不能声称恢复被GN消除的receiver增益或DC项。它只检验一个更窄的假设：Phase1 receiver变化在GN归一化后的通道相对仿射方向上留下跨类共享、可预测的低维残差。

Wu和He提出GroupNorm的目的正是消除batch-size依赖，GN在每个样本内部按组计算均值和方差，而不是保存可由target batch替换的running statistics。因此AdaBN或Tent式“更新目标batch统计”不适用于本checkpoint；Tent还会使用测试batch熵更新参数，违反本项目query零更新边界。[Group Normalization](https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html)；[Tent](https://openreview.net/forum?id=uXl3bZLkr3c)。

## 4.唯一干预族与函数gauge

设`u∈R^{48×T}`为`time_fuse.0`输出，`g(i)`是通道`i`所属的3通道组。冻结checkpoint参数为`γ0,β0`。候选定义

\[
\operatorname{GN}_a(u)_{i,t}=
\gamma_{0i}\exp(a s_\gamma q_i^\gamma)
\frac{u_{i,t}-\bar u_{g(i)}}{\sqrt{v_{g(i)}+10^{-5}}}
+\beta_{0i}+a s_\beta q_i^\beta,
\quad a\in[-1,1].
\]

其中`qγ,qβ∈Z^48`是INT8方向，`sγ,sβ`是正FP16 scale。三个规范共同固定坐标：

1.组规范：每个GN组内`Σ_i qγ_i=Σ_i qβ_i=0`；
2.函数尺度：在Phase1训练面的输出单位特征`f_a(x)`上，`E||∂f_a/∂a|_0||²=1`；
3.符号规范：拼接解码方向中最大绝对值坐标为正。

`r=1`消除了可逆旋转`M`的自由度。仍可能存在后续网络补偿导致的函数等价，所以参数规范不是充分条件；第7节必须在held函数输出上做等价反证。

## 5.Stage2闭式状态

令`f_a(x)=Norm2(z_id(x;a))∈R^160`，六个Phase1 old-class锚为`μ_c`。对当前row的old support集合`S_O`定义

\[
j(x)=\left.\frac{\partial f_a(x)}{\partial a}\right|_{a=0},\quad
I=\frac1{6K}\sum_{(x,c)\in S_O}\|j(x)\|_2^2,
\]

\[
b=\frac1{6K}\sum_{(x,c)\in S_O}j(x)^\top[\mu_c-f_0(x)],\quad
\hat a=\operatorname{clip}_{[-1,1]}(b/I),
\]

\[
a_Q=\operatorname{round}(127\hat a)/127.
\]

该式是在`a=0`处对完整非线性`exp`-FiLM网络做的一步tangent estimator，不是全局非线性目标的精确最优解。`j(x)`必须来自冻结eval模型的完整forward，并在最终L2归一化160维`z_id`上求JVP；`f_0(x)`与`μ_c`必须在完全相同的坐标中。候选不使用ridge或先验，因此`I`必须由数据本身提供信息。K1的数值运行条件固定为`I>64ε_fp32`；该条件只排除零JVP，不能证明receiver状态可识别。

所有拟合和held评分都使用量化后的`a_Q`，不能用FP32 teacher通过、INT8部署状态失败。`a=0`必须逐值复现基线；`a∈[-1,1]`的固定数值网格必须证明gamma有限、输出非零且无数值异常。`a_Q`在new support出现前冻结；随后以同一个`a_Q`重编码old/new support。每条query独立forward，不更新`a_Q`、support bank、head或任何跨query状态。

## 6.Phase1学习与科学证伪

### 6.1拆分

外层按receiver留一；内层按class/TX留一。support和query物理ID严格互斥，每个clean/raw物理记录只使用预先随机选定的一条允许`leo_*_weak`received-IQ。对每个外层held receiver`r`，方向训练和旧类锚都只能使用其余receiver：

\[
\mu_c^{(-r)}=\operatorname{Aggregate}\{f_0(x):receiver(x)\ne r,label(x)=c\}.
\]

锚按class先等权、class内receiver/day再等权，并且每个cell至少2个物理样本。对`r`和被留TX类`c`，只用`r`中其他类的support拟合`a`，在被留类的独立query上评价。被留类真值只由外部Phase1 scorer读取，不进入fit。每个fold必须记录锚的物理ID集合Merkle root、anchor hash和排除的receiver；通过后才允许用全部Phase1 source重训方向并seal最终锚。

### 6.2无调参方向构造

Phase1冻结全部基座权重，不使用optimizer学习方向。冻结checkpoint只读核查确认`id_backbone.time_fuse.1.weight`为48维float32，最小值`0.916147172451`、最大值`1.17473506927`，负数、零值和非有限值均为0，因此本候选的gamma坐标可唯一写为`log γ`。令原始GN参数坐标为`p=(log γ[48],β[48])∈R^96`；`P_G`为逐3通道组零和投影。对外层receiver`r`、LOCO类`c`、训练receiver/day域`d`和其余类`y≠c`，在物理样本上计算

\[
g_{d,y}^{(-r,-c)}=\operatorname{Mean}_{x\in(d,y)}P_G\nabla_p
\frac12\|f_0(x)-\mu_y^{(-r)}\|_2^2,
\quad
g_d^{(-r,-c)}=\frac1{5}\sum_{y\ne c}g_{d,y}^{(-r,-c)}.
\]

把全部`d≠r`训练域的`g_d^{(-r,-c)}-Mean_d g_d^{(-r,-c)}`堆叠为矩阵`G^{(-r,-c)}`，做canonical float64 SVD；唯一方向是第一右奇异向量。被留类锚`μ_c^{(-r)}`只供方向冻结后的held scorer使用，绝不进入该fold的方向SVD。

对每个fold要求`σ1`高于标准数值rank阈值，且谱隙

\[
\sigma_1-\sigma_2>
\tau_{svd}=\max(10\epsilon_{dir\_quant}\sigma_1,
100\max(m,96)\epsilon_{64}\sigma_1).
\]

canonical方向还必须在同一outer receiver的六个LOCO fold间满足最小余弦`≥1/√2`；任一fold方向退化、量化后翻向或不满足谱隙，候选关闭。该构造提取跨receiver/day变化最强、class-balanced的共同校正梯度，不含学习率、epoch、初始化或early stop。将96维向量拆为gamma/beta方向，执行组零和投影、函数JVP能量归一、canonical sign和INT8量化；量化后重新核验全部规范。

### 6.3唯一外层评分

方向构造完成后，对被留类`c`计算

\[
\Delta_{r,c}=
\frac1{|Q_{r,c}|}\sum_{x\in Q_{r,c}}
\left[
\|f_{a_Q(S_{r,-c})}(x)-\mu_c^{(-r)}\|_2^2-
\|f_0(x)-\mu_c^{(-r)}\|_2^2
\right].
\]

外层receiver-held只用于一次科学证伪，不用于选择层、rank、损失、阈值或方向版本。通过后，使用同一gradient-SVD规则在全部Phase1 source receiver/day上重建一次方向和锚并seal。

### 6.4跨类状态而非TX残差

class就是TX身份，因此不另建重复的leave-one-TX矩阵。对每个receiver，记录各个LOCO support子集得到的`a_Q^(-c)`。公共状态假设要求：同receiver内删除任一TX后，状态符号与主要幅度保持一致；不同receiver之间允许变化。`a_Q=0`按方向稳定性失败处理，不得以“没有翻向”通过。

TX/class leakage probe在同一outer receiver split上使用每个cell的固定三维影响向量`[b_c/I_c,log I_c,clip_c]`，以训练receiver的class中心作nearest-centroid分类器，在held receiver预测TX；不得训练另一神经网络或调正则。置换null使用由checkpoint SHA导出的固定seed执行1,024次class-label置换，若实际balanced accuracy高于null的99%分位数则失败。CFO只允许使用数据中预先存在的独立数值元数据；若该字段不存在，必须记录`CFO_PROBE_UNAVAILABLE`并关闭候选，不能从同一IQ估计后当作独立truth。

每个类必须报告独立的`b_c,I_c`以及量化前后贡献；主结论不能由一个TX、一个receiver或`a=±1`饱和episode驱动。若删除任一类后状态翻向，或任一有效episode依赖clip边界，则判为类条件残差而不是共享状态。TX/CFO/class leakage probe必须采用cross-receiver训练/held划分，不能在同receiver上训练和报告。

本轮不预注册一个任意accuracy百分比作为gate。生命周期固定为：`S_(r,-c)`拟合并冻结`a_Q`→以同一`a_Q`编码并注册held类support→外部scorer评估`Q_(r,c)`。计数规则固定为：每个K的全fold`mean Δ<0`、总qKNN净正确数`>0`、逐receiver净正确数小于0的fold数`=0`、LOCO状态翻向或归零数`=0`、clip饱和episode数`=0`。同时报告逐receiver`Δ<0`的fold数，不用其比例选择方法。任一K不满足任一规则即关闭。即使全部通过，术语也只能是“receiver-held可迁移干预坐标”，不能声称物理分离了纯receiver响应。

## 7.函数级非等价反证

文献表明，仅对齐边缘分布或找到一个可行映射不足以识别正确域映射；分布对称性会产生多个同样对齐但标签不同的spurious maps。GN-ISF必须排除旧函数族，而不能只比较模块名。[Identifiability Conditions for Domain Adaptation](https://proceedings.mlr.press/v162/gulrajani22a.html)；[On Learning Invariant Representations for Domain Adaptation](https://proceedings.mlr.press/v97/zhao19a.html)。

所有对照必须绑定同一checkpoint SHA、同一raw received-IQ输入、同一最终单位`z_id`和同一qKNN。若历史模块不能在该checkpoint上重放，就记录`CONTROL_UNAVAILABLE`，不得声称已排除它，也不得让falsifier通过。对每个旧族，在训练物理样本上构造切线设计矩阵`B_train=[Δf_1,…,Δf_r]`，以canonical Moore-Penrose伪逆拟合系数；在独立held物理样本上用同一系数和对应`B_held`复现GN量化效应。可重放对照为：

- D102：`joint_proj.0`输出的`pre_relu+B a`，随后ReLU和L2；
- GRB-JP4：`joint_proj.0.weight`rank-4扰动，随后完整冻结网络；
- late-FiLM：在`id_backbone.t_proj`、`id_backbone.f_proj`、`id_backbone.pa_proj.0`和`id_backbone.fuse.0`四个Linear输出上分别施加`.film_scale/.film_bias`切线，共8个160维参数tensor；随后计算同一最终单位feature；
- fixed PSD：使全部pairwise squared-distance导数最小的常数对称`ΔM`。

对每个旧函数族`B`，令`c_B=pinv(B_train)vec(Δf_GN,Q,train)`，在独立held样本上定义量化后GN效应的相对最小二乘残差

\[
\rho_B=\frac{\|vec(\Delta f_{GN,Q,held})-B_{held}c_B\|_2}
{\|\Delta f_{GN,Q,held}\|_F}.
\]

fixed PSD使用训练pair拟合常数对称`ΔM`，只在物理ID互斥的held pair评价距离导数残差，禁止在同一pair集合拟合和评分。数值rank阈值采用标准FP64 SVD容差`max(m,n)ε64σ_max`。令teacher和INT8量化函数效应分别为`Δf_T,Δf_Q`，定义同量纲相对parity

\[
\epsilon_{parity}=\frac{\|\Delta f_Q-\Delta f_T\|_F}
{\max(\|\Delta f_Q\|_F,\tau_0)},\quad
\tau_0=100\max(m,n)\epsilon_{64}\|f_0\|_F,
\]

\[
\tau_{eq}=\max(10\epsilon_{parity},100\max(m,n)\epsilon_{64}).
\]

若`||Δf_Q||_F≤τ0`，候选直接按零效应拒绝。qKNN score parity定义为同一query/class上teacher与量化状态score的最大绝对差`ε_score_parity=max|s_Q-s_T|`，单位与qKNN score一致。必须同时观察到：GN后的首个ReLU mask确有变化；对每个旧族`ρ_B>τ_eq`；至少一条held query的qKNN排序差异margin严格大于`2ε_score_parity`；pairwise距离变化对fixed PSD也满足同一残差门。阈值完全由数值/量化parity决定，不从held表现调节。任一旧对照不可重放、残差未过门或排序差异落在parity内，均为`REJECT_GN_ISF_EQUIVALENT_OR_UNIDENTIFIABLE`。

## 8.资产与资源

共同封存数值payload的设计估算为：

|字段|形状/类型|字节|
|---|---:|---:|
|`q_gamma`|`[48]int8`|48|
|`q_beta`|`[48]int8`|48|
|方向scale|`[2]float16`|4|
|old-class anchor codes|`[6,160]int8`|960|
|anchor scales|`[6]float16`|12|
|聚合计数|`[6]uint16`|12|
|合计|—|1,084|

聚合计数必须全部`≥2`。锚必须由target访问前、与checkpoint绑定的Phase1多物理样本聚合得到。资产不得含IQ、单样本feature、成员ID、receiver/day名称或可逆索引。header另含checkpoint hash、层路径、groups、eps、协议和seal，不把固定header字节冒充数值payload。`1,084B`必须在实现波次用实际序列化文件复核，当前不是已验证资源结果。

运行态新增只有一个INT8`a_Q`。fit计算为`2×160×6K=1,920K MAC`，即K1/K5/K10分别`1,920/9,600/19,200 MAC`；JVP数为`6K`。合并动态GN affine后，query使用正常网络forward，不增加独立adapter分支或post-backbone MAC。

## 9.晋级与停止

Phase1 falsifier只有两个终态：

- `PHASE1_IDENTIFIABILITY_FALSIFIER_PASS / STAGE2_IMPLEMENTATION_ALLOWED / NO_TARGET_RESULT`；
- `REJECT_GN_ISF_EQUIVALENT_OR_UNIDENTIFIABLE / STAGE2_CLOSED / NO_TARGET_RESULT`。

本轮在实现前已经得到第二个终态的具体实例：`REJECT_GN_ISF_UNVERIFIABLE_CONFOUND / STAGE2_CLOSED / NO_TARGET_RESULT`。原因是独立CFO truth不存在，预注册P0无法执行。没有Phase1 falsifier运行、没有N607发布、没有性能结论。

最终独立复审：方法作者复核`MERGE，P0=0，P1=0`；反方复核`MERGE，P0=0，P1=0`。两者均确认该裁决是对预注册fail-closed规则的执行，不是CFO泄漏结论，也不是性能失败。

若本轮未触发该关闭条件，通过也不等于性能成功；原设计随后才会允许一次核心实现、协议负测、真实checkpoint无query smoke、独立`P0=0/P1=0`、本地commit和真实588条K1/K5/K10 G0。该反事实路径仅用于说明停止边界，不构成后续执行授权。
