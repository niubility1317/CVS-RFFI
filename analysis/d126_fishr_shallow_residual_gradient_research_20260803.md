# D126浅层Fishr启发残差梯度适配研究

状态：`THEORY_RESEARCH_COMPLETE / PHASE1_FALSIFIER_CANDIDATE / NO_NEW_PERFORMANCE_RESULT / NO_EXPERIMENT_RELEASE`

## 1.直接裁决

用户提出的“浅层＋Fishr＋残差梯度更新”在机理上值得继续，但不能照字面实现为“在Phase2对前几层直接运行Fishr并更新权重”。本项目更可辨识、更轻、更合法的版本应当是：

1.只在Phase1使用多source receiver构造Fishr启发的梯度方差约束，学习并封存一个二维浅层残差子空间；
2.Phase2只用全registered-class support的标签和固定IQ数学视图，对二维系数做一步预条件梯度更新；
3.基础checkpoint、Sinc滤波器、卷积、GroupNorm原参数、PA/DAC身份分支和qKNN规则全部冻结；
4.query只使用冻结后的一次forward，零fit、零update、零selection；
5.唯一优先干预点为`id_backbone.time_fuse.1`的GroupNorm输出、首个ReLU之前，不扫描层位、rank、K专属步长或残差形式。

建议候选名为`D126-FSRG-2`（Fishr-sealed Shallow Residual Gradient，rank-2）。它是Fishr启发方法，不应在论文中表述为原始Fishr的直接复现。

## 2.当前轻型域适应到底取得了什么成效

当前没有新Target125结果。现有可核验数字均来自D104 source-held的63行/252臂开发面：

|路线|同排比较|old|seen-new|H|old floor|all floor|总正确数|证据结论|
|---|---|---:|---:|---:|---:|---:|---:|---|
|D106 RDCE|`M_DA-M0`|+0.2604pp|+0.3632pp|+0.4447pp|+0.2824pp|+0.4438pp|+0.9206|轻型DA五指标同向，小幅正收益|
|D122静态ground head|`M_HEAD-M0`|+1.0788pp|+1.2510pp|+1.4831pp|+0.8452pp|+0.1458pp|+3.7777|轻型head有较强开发面增益，但all floor增量很小|
|D122组合|`M_JOINT-M0`|+1.0673pp|+1.2510pp|+1.5011pp|+0.6246pp|+0.0058pp|+3.7460|均值提升但尾部收益几乎被组合抵消|
|D123 LOO-CRES|相对D122|0|0|0|0|0|0/63行变化|row-local重构无功能增量，已关闭|

因此能支持的结论是：轻型适配在当前模型上确实能产生正收益，且早期/中间表示变换仍有研发价值；但现有增益远低于Target25，组合冲突说明不能继续把若干轻模块相加。D106 Target运行没有完整性能结果，D62、D91、D92和SVRN的完整125历史结果也不能被上述63行开发面替代。

## 3.Fishr原理纠偏

Fishr的核心不是“挑一个浅层更新”。它在多个训练域上计算逐样本梯度，匹配各域内梯度的方差，从而使训练域的局部损失几何更接近。原论文把该量与Fisher信息和局部Hessian联系起来：[Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization](https://proceedings.mlr.press/v162/rame22a.html)。

官方实现还揭示了一个容易误解的细节：Fishr penalty的逐样本梯度取自`classifier.parameters()`，而总任务目标的optimizer仍包含featurizer和classifier全部参数。因此“Fishr证明只对浅层做Fishr有效”不是原论文或官方实现的直接结论：[Fishr官方实现](https://github.com/alexrame/fishr)。

Phase2直接照搬Fishr有三个根本问题：

- K1时每class只有一个独立物理support，无法稳定估计目标域内逐样本梯度方差；
- 当前只有一个target receiver row，缺少多个target domain供方差匹配；
- 对query/test batch做熵或梯度更新会违反query零更新边界。Tent式test-time adaptation正是通过test batch熵更新归一化参数，因此不适用于本协议；当前模型又使用GroupNorm，不存在可替换的BatchNorm running statistics：[Tent](https://openreview.net/forum?id=uXl3bZLkr3c)。

所以正确迁移方式是：Fishr负责在Phase1塑造“更新坐标的跨receiver稳定性”，Phase2只估计二维更新系数。

## 4.为什么优先浅层，但不是越浅越好

理想干预点必须同时满足：域扰动仍可校正、TX身份尚未被破坏、低维更新能改变后续非线性、计算和资产足够小、Phase2可从support得到任务梯度。

|层位|域适应潜力|身份破坏风险|本轮裁决|
|---|---|---|---|
|原始IQ/Sinc滤波器|可直接影响前端频谱|极高；滤波器本身承载RFFI频谱差异，且D120已证明若假定的物理干预未发生会变成虚构逆变换|禁止首轮更新|
|`time_fuse.1`后、ReLU前|接收机/信道尺度仍局部存在；C=48；残差可改变后续ReLU mask和深层路径|可用rank2、trust radius和TX/class-LOCO限制|唯一优先层|
|`t1/t2`内归一化|比首层更语义化，仍可能校正中层统计|与TX判别逐渐纠缠；参数与候选面扩大|首层失败后也不在同revision回退扫描|
|DAC/PA分支浅层|与硬件非线性强相关|直接抹除TX指纹的风险最高|禁止|
|`fuse/joint_proj`前后|稳定、易实现|D102、D106、D124和D125已覆盖大量末端偏置、度量或hyper-adapter函数族|仅作非等价对照|
|分类器/head|参数小|当前Stage2使用support qKNN，source classifier更新未必传递到决策；D123已显示一种局部head修订无argmax变化|不是DA首选|

残差adapter文献说明，小型串联/并联残差模块可以跨域复用共享骨干，并以少量额外参数实现域特化；这支持“冻结骨干＋小残差支路”的总体方向，但不能直接证明K1 RFFI有效：[Residual Adapters](https://openaccess.thecvf.com/content_cvpr_2018/html/Rebuffi_Efficient_Parametrization_of_CVPR_2018_paper.html)。语音适配中也有低于0.5%参数量的residual adapter接近全量微调收益的跨任务证据，但这里只能作为工程可行性旁证：[Residual Adapters for Parameter-Efficient ASR Adaptation](https://research.google/pubs/residual-adapters-for-parameter-efficient-asr-adaptation-to-atypical-and-accented-speech/)。

## 5.冻结的候选函数

令`u(x)∈R^{48×T}`为`time_fuse.1`输出、首个ReLU输入。冻结rank2残差为

\[
u_a(x)_{:,t}=u(x)_{:,t}+U\left[a\odot\tanh\left(Vu(x)_{:,t}\right)\right],
\quad U\in\mathbb R^{48\times2},\;V\in\mathbb R^{2\times48}。
\]

其中`U/V`由Phase1学习、canonical化、INT8封存；`a∈R²`是Phase2唯一可变状态。base网络所有参数保持冻结。该位置在ReLU前，因而同一个二维系数可以通过样本依赖的`tanh(Vu)`改变不同样本的ReLU mask；若held样本上不能产生这种样本依赖变化，就应视为D102常量偏移或固定PSD重入并关闭。

预计数值payload仅`U 96B＋V 96B`及少量FP16 scale/preconditioner/step/trust-radius；精确字节和时延必须实现后实测，当前不作为性能结论。

## 6.Phase1如何使用Fishr

外层必须receiver-held，内层class/TX-LOCO，support/query物理ID互斥。每个inner episode只在source received-IQ上模拟Phase2注册流程。令`L_e(a)`为episode`e`的全类等权support交叉视图qKNN损失，逐样本对二维系数的梯度为

\[
g_i=\left.\nabla_a\ell_i(a)\right|_{a=0},\qquad
v_d=\operatorname{Var}_{i\in d}(g_i)\in\mathbb R^2。
\]

Phase1训练目标为

\[
\min_{U,V}\;\mathbb E_eL_e+\lambda_F\sum_d\|v_d-\bar v\|_2^2+lambda_R\mathbb E\|u_a-u\|_2^2。
\]

这里的Fishr项只对二维适配坐标的梯度方差做跨source receiver对齐，不对48通道原权重估计高维BatchGrad。`λ_F/λ_R`、rank、层位和优化预算必须在inner面冻结，outer receiver只作一次证伪，不能反调。

同时封存对角预条件器

\[
D_F=\epsilon+\mathbb E_d[v_d]。
\]

它的作用不是识别真实CFO或物理receiver参数，而是降低source receiver间最不稳定的更新方向权重。

## 7.Phase2怎样更新最有效

每条fixed received-IQ允许产生两个确定性后接收数学视图，但它们不增加K。对每个registered class逐类等权：视图A形成stop-gradient prototype，视图B作为support训练项；再交换A/B并平均。所有类别参加同一竞争，禁止old-only状态、role、quota或query统计。

\[
L_S(a)=\frac12\left[
CE\big(f_a(B),\operatorname{sg}(P_a(A))\big)+
CE\big(f_a(A),\operatorname{sg}(P_a(B))\big)
\right]。
\]

只在`a=0`求一次梯度：

\[
g=\left.\nabla_aL_S(a)\right|_{a=0},\qquad
\tilde a=-\eta D_F^{-1/2}g,
\]

\[
a_1=\operatorname{Proj}_{\|a\|_2\le\rho,\;|a_j|\le a_{max}}(\tilde a)。
\]

`η/ρ/a_max`随Phase1资产封存，K1/K5/K10使用同一规则。得到`a1`后，以同一状态重新forward所有old/new support并统一注册；随后每条query只forward，不再反传或改变状态。

这比直接更新浅层卷积/GN更合适，原因是：

- Phase2只优化2个数，不是48到数千个参数；
- 预条件器抑制跨source receiver不稳定方向；
- trust region限制旧类遗忘和K1过拟合；
- 同一状态作用于old/new与query，避免注册坐标不一致；
- 一步更新没有epoch、early stop和target truth选择。

## 8.为什么它可能超过既有方法

1.相对D106：D106在输出度量上做rank3非欧氏校正，稳定但效应小；D126在第一处ReLU前改变激活路径，表达力更强。
2.相对D112：D112主要改变head，D126先修正表示再沿用原qKNN，不依赖类别特定head规则。
3.相对D119：D119试图从old support闭式识别一个GN物理状态，并因缺少独立CFO混杂证据关闭；D126不声称识别CFO，只声称support监督梯度在outer receiver上改善分类，因此不需要CFO truth，但仍必须做TX/class-LOCO。
4.相对D125：D125用support摘要经hypernetwork直接预测`a`，Phase2无反传、速度更快；D126的`a`由当前row的任务梯度产生，更贴近分类错误方向，理论上可减少“summary与receiver状态不可辨识”的风险。
5.相对D102/D124：非线性浅层残差可改变样本特定ReLU mask；只有这种变化在outer样本上超过量化误差并不能被固定线性/PSD拟合时，才算新函数族。

## 9.最主要风险

|风险|为什么严重|一次性反证|
|---|---|---|
|K1梯度噪声|同一物理样本的两视图不能替代独立样本|K1 outer receiver的系数方向稳定性、净正确数和held/seen双侧结果|
|support proxy过拟合|交叉视图损失可能只学到视图不变性|support loss下降但outer query不增益即关闭|
|common-transform cancellation|相同变换可能同时移动prototype与query而不改变邻序|必须有超过量化包络的neighbor/argmax变化|
|TX身份擦除|浅层更新可能把RFFI信号当域噪声消掉|class/TX-LOCO、逐类old floor和forgetting同排报告|
|D102/PSD函数重入|低秩adapter可能只是末端固定度量的重参数化|inner拟合旧族、outer物理互斥样本评分残差|
|计算高于D125|support需要反向JVP和第二次forward|实测K1/K5/K10时延、峰值内存和能耗；不以参数量代替|
|量化后更新失活|二维梯度或basis量化后可能归零|INT8-reference parity与真实588条G0功能变化|

## 10.D125与D126的选择

|属性|D125 RDHA-2|D126 FSRG-2|
|---|---|---|
|Phase2状态来源|5维support统计经sealed hypernetwork预测|support监督损失的一步二维梯度|
|Phase2反传|无|只对2维`a`做一次|
|干预位置|`joint_proj.0`的320维输入|`time_fuse.1`后、首个ReLU前|
|优势|最快、实现简单、资源低|对当前row的任务错误有直接响应、函数表达力更强|
|主要风险|support摘要不可辨识或近似常量|K1梯度噪声、计算增加、浅层身份破坏|
|与旧路线距离|较接近late adapter/metric族|若ReLU mask改变则更可能形成新函数族|

当前建议是：暂停D125实现，不废弃其已冻结设计；优先对D126做一个同预算Phase1 falsifier。原因不是D126已有性能，而是D125尚未实现，且D126对用户提出的残差梯度假设给出了更直接、可证伪、与D102/D106不同的机制。两者不得组合，不同时实现，不做125矩阵赛马。

## 11.最小下一步与停止规则

只允许一个source-only Phase1 falsifier，输出D125与D126共同的identity对照，但不调参赛马：

1.唯一层位`time_fuse.1`后、ReLU前，rank2，一步更新；
2.outer receiver-held×class/TX-LOCO，K1/K5为主，K10只作稳定性描述；
3.逐row记录old、held/seen-new proxy、H、old/all floor、总正确数和系数；
4.K1与K5聚合H和总正确数均须严格高于identity，held/seen任一侧不得系统性塌陷；
5.至少一个outer K1 package产生超过INT8误差的neighbor和argmax变化；
6.固定PSD、D102常量偏移和D125 late adapter若能在outer样本解释其函数效应，则关闭；
7.任一核心项失败立即关闭D126，不换层、rank、步长、视图、seed或阈值复活；
8.全部通过只允许最小实现、真实checkpoint无query smoke和真实588条K1/K5/K10 G0，不直接进入Target或125。

最终研究裁决：`PRIORITIZE_D126_PHASE1_FALSIFIER / PAUSE_D125_IMPLEMENTATION / NO_NEW_PERFORMANCE_RESULT`。
