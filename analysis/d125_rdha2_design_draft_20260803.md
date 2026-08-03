# D125 RDHA-2 Phase1证伪冻结设计

> **已被联合冻结设计替代（2026-08-03）：**本文件仅保留候选C的公式来源。D125不再是独占路线，旧`Stage2 implementation forbidden`、source-held性能硬门、588条G0和fresh63流程不再生效；候选C现在与A/B共享同一个D92-Lite，并接受统一S0/S1因果筛选。当前权威设计见`analysis/joint_da_d92_lite_codesign_frozen_20260803.md`。

文档执行状态：`HISTORICAL_SUPERSEDED / DO_NOT_EXECUTE`。下文所有“当前裁决”“必须”“允许”和实验步骤均为旧决策记录，不再授权实现或发布。

## 1.当前裁决

- 名称：`D125-RDHA-2`（Receiver-Held Dynamic Hyper-Adapter，rank-2）。
- 状态：`DESIGN_FROZEN / PHASE1_FALSIFIER_IMPLEMENTATION_ALLOWED / STAGE2_IMPLEMENTATION_FORBIDDEN`。
- 独立冻结审查：`FREEZE_ONE_PHASE1_FALSIFIER / P0=0`；原6项P1已并入本文件的唯一冻结定义，待实现后再做一次代码级`P0/P1`终审。
- 当前不实现Stage2、不发布G0/G1/Target/125；只允许一次source-only嵌套Phase1 falsifier。失败即永久关闭D125，不再调整层位、rank、summary、seed、训练步数或门限。

## 2.为什么可能是新机制

D125不新增Phase2 observable；新增信息来自Phase1的receiver-held反事实响应监督：support set摘要必须预测一个对未参与拟合的receiver、TX/class和物理query仍有效的低维adapter响应。它严格禁止`z_dom`、source receiver/day bank、sample-to-bank matching、D102/D105 precision/codebank和support残差闭式ridge。

候选adapter唯一放在`id_backbone.cls_head.joint_proj.0`的320维输入`h(x)`上；不扫描其他层。归一化算子固定为无仿射单组GroupNorm，即对320维样本向量做LayerNorm等价标准化：

\[
h_a(x)=h(x)+U\left[a\odot\tanh\left(V\mathcal N(h(x))\right)\right],
\]

其中`U∈R^{320×2}`、`V∈R^{2×320}`，`a∈[-a_max,a_max]^2`。规范固定为`U^TU=I_2`、`V`逐行单位范数、列序按Phase1响应能量降序、每列最大绝对坐标为正。后续checkpoint保持冻结。样本依赖的`tanh`残差必须改变后续ReLU mask或qKNN排序；若量化后的函数效应可被固定末端线性、D102常量`B a`或固定PSD metric在独立outer样本上解释到INT8-reference误差包络内，立即关闭。

初始化不是扫描：对每个outer fold只用inner训练实体，先按class等权去中心，再对receiver均值差矩阵做canonical float64 SVD；前两个右奇异向量同时初始化`U`列和`V`行。谱rank不足2直接证伪，不换seed。`a_max`两维相同，唯一规则为

\[
a_{max}=0.05\,\operatorname{median}_{x\in L_s^{inner}}\|h(x)\|_2/\sqrt2,
\]

从而在`U`正交且`tanh`有界时把adapter分支范数限制在inner中位hidden范数的5%以内；该预算只由inner Phase1计算并以FP16封存，不读取outer、Stage2或Target。

## 3.Phase2合法summary修订

作者原先使用old anchor/W或old-only状态的summary均被拒绝。注册完成时只读取全部registered classes的合法support。响应映射不另增`rho`，直接复用adapter的两个规范化坐标：

\[
r_{ck}=\tanh\left(V\mathcal N(h(x_{ck}))\right)\in\mathbb R^2,
\qquad
\bar r_c=\frac1K\sum_{k=1}^{K}r_{ck},
\qquad
\bar r=\frac1{|C_R|}\sum_{c\in C_R}\bar r_c,
\]

\[
s=\left[\bar r;\operatorname{vech}\left(
\frac1{|C_R|}\sum_{c\in C_R}(\bar r_c-\bar r)(\bar r_c-\bar r)^T
\right)\right]\in\mathbb R^5,
\qquad
a=a_{max}\tanh(Q\operatorname{std}_{P1}(s)+b)。
\]

该5维summary逐类等权、对class顺序和标签置换不变；`V/Q/b/std_P1`只由Phase1学习并与checkpoint共同封存。old和new support共同生成唯一state；全类注册完成后只生成一次`a`，同一`a`重新forward全部old/new support并统一注册，随后每个query只做一次同adapter forward，零fit、零update、零selection。不得读取query、truth、held role、quota或query-batch统计。

唯一层位、`r=2`、summary维度5和`a_max`规则已冻结。不得由outer结果选择它们，也不得引入K专属参数。

## 4.Phase1 episodic监督

每个outer fold同时held receiver与held TX/class。held实体必须从`U/V/Q/b`训练、`a_max`和summary标准化统计中完全排除。inner support、inner query和outer eval query的物理ID两两互斥。

不再引入teacher或响应蒸馏。每个inner episode用合法support生成`a=H(S)`，以冻结qKNN在独立inner query上的class-balanced交叉熵直接学习`U/V/Q/b`；每class和每receiver先等权，再对episode等权。优化器固定为确定性full-batch float64 L-BFGS，`max_iter=128`、`line_search_fn=strong_wolfe`，无early-stop、无多初始化、无超参数扫描。outer receiver×TX/class-LOCO只做一次冻结评估，绝不反调训练。Phase2和outer eval均不运行optimizer，也不打开query标签。

## 5.最小Phase1 falsifier

只运行一个预冻结source-only嵌套包，K1为主，并用同一公式覆盖一个常规K；不运行Target或125。

必须同时记录：

1.资产allowlist仅含INT8`U/V/Q/b`、FP16尺度、`std_P1`和`a_max`；不得含source/clean/raw/cache、样本feature、物理ID、receiver/TX/day键、可检索bank、`z_dom`或FP32残留。
2.outer receiver-held×TX/class-LOCO上的同rowheld-class/seen-class BA、H、class floor和总正确数；K1和K5聚合H与总正确数均须严格高于identity，且held或seen任一侧不得下降超过2pp。该判据只用于快速证伪，不构成Stage2性能目标。
3.summary/系数的TX与class probe不得超过由固定1,024次标签置换得到的99% null分位数。
4.K1六个独立物理support形成的amortized code必须有限、非bias常量、跨package方差高于INT8-reference误差；leave-one-class code不得由单一TX主导，且至少一个outer K1 package产生超过量化误差的feature、neighbor或argmax效应。不得把数值rank2表述为已识别物理receiver状态。
5.联合标签置换后summary逐值不变、logit列等变；INT8与reference的argmax逐值等价，score最大误差形成后续非等价检验的唯一数值包络。
6.D102常量偏移、固定末端线性和固定PSD对照只在inner拟合，在outer物理互斥样本/样本对上评分；相对残差必须严格大于INT8-reference误差包络，且outer样本Jacobian、ReLU mask或qKNN排序必须出现超过该包络的样本依赖变化。

任一核心项失败即`CLOSE_D125_RDHA2_PHASE1_FALSIFIER`，不调rank、层、`a_max`、summary、seed、训练步数或门限来复活。全部通过只允许实现Stage2功能路径并运行一次真实588条K1/K5/K10 G0；不构成Stage2性能或真实新类结论。G0任一K无功能变化立即关闭；只有三K均变化才允许一次fresh63行G1。G1若DA简单效应无独立正收益或held/seen任一侧系统性塌陷，则永久关闭且不运行Target或125矩阵。

## 6.资源边界

Phase2 enrollment预计每个support一次base forward生成summary，再一次adapted re-forward注册；每query只一次adapted forward。对320维hidden，adapter额外为`2×320×2=1,280`MAC/样本，另计归一化与完整backbone；无反传、optimizer或query state。INT8数值payload设计值为`U 640B+V 640B+Q 10B+b 2B`，另加FP16 scales、5维标准化和稳定界，实际序列化字节、forward延迟和INT8误差必须由falsifier实测。

## 7.可行性摘要（20行内）

1.D125新增的是Phase1 receiver-held响应监督，不是新的target observable。
2.早层样本依赖非线性adapter有机会区别于D102常量偏移和PSD metric。
3.Phase2禁止z_dom、source bank、闭式support残差solve和query反馈。
4.adapter系数只由all-registered support逐类等权summary生成。
5.old/new support和query使用同一冻结adapter state。
6.outer held receiver/TX/class从训练、标准化和任何选择中完全排除。
7.K1可拟合2维系数不等于可辨识，必须由outer held迁移证明。
8.rank2、joint_proj输入、5维summary和5%分支预算已冻结，不再选择。
9.当前只允许一个Phase1 falsifier，不允许Stage2实现或N607性能矩阵。
10.falsifier失败立即关闭，不回退到D102/D105或PSD路线。
