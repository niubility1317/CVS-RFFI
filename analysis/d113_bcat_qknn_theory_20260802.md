# D113-BCAT-qKNN轻型快速域适应理论冻结

状态：`DESIGN_FROZEN / IMPLEMENTATION_NOT_STARTED / NO_NEW_PERFORMANCE_RESULT`

日期：2026-08-02

## 0.结论

D113选择`BCAT-qKNN`（Bayesian Chordal Additive Translation＋qKNN）：以六个Phase1旧类地面锚和当前row的合法旧类support估计一个跨类共享的加性接收机偏移，再用该偏移对应的球面解析逆映射同时规范化所有旧类、新类support和每条独立query。它不是对D110低方差轴加权或D112旧类锚运动调参，而是改变域适应的作用对象：先恢复共同坐标系，再让原Student-t qKNN和已有正收益的ground-head在同一canonical表示上分类。

该设计经WP-DA与WP-HEAD分别从K1可辨识性、球面逆映射、新类保护、旧类floor和资源上交叉审查。原始`r=Σw(s-g)`与`normalize(x-b)`均被否决；冻结版采用固定ground切投影的贝叶斯矩估计和球面加性模型的唯一正根逆。无噪声下更精确的`P_s`方程因K1中同一support噪声同时污染设计矩阵和右端，作为后续errors-in-variables分支记录，不进入首发。

理论冻结只说明方法自洽、轻型且可证伪，不说明性能成立。D113尚无Target25或source-held性能结果。

## 1.为什么继续研发而不重跑D112

D112的source-held G1证明ground单位质量head是一个正组件，但`M_JOINT_SEAM`与`M_HEAD_GROUND`在63row、189个臂单元上的预测逐项相同，SEAM运动的独立决策收益为0。继续修改其rank、收缩或门限属于在零独立效应机制上调参，因此永久关闭SEAM运动，只复用已经有正证据的ground-head作为D113四臂中的分类头因素。

D113要回答两个新的、彼此可分的问题：

1.仅将target表示恢复到Phase1共同坐标系，是否改善原qKNN；
2.坐标恢复后，ground-head是否仍能增加old/new联合正确数，而不是只把new query吸向old。

## 2.生成模型与适用边界

令canonical单位表征为`u∈S^{p-1}`，`p=160`。对目标接收机域`d`，冻结局部模型为：

\[
x=F_{\delta_d}(u)+\varepsilon,
\qquad
F_{\delta}(u)=\frac{u+\delta}{\lVert u+\delta\rVert}.
\]

`δ_d`是同一target receiver对所有已注册发射机共享的加性表示偏移，`ε`汇总类内采样噪声、类×域交互和模型误差。该假设比任意仿射、全矩阵或类专属transport弱得多，但仍是可证伪假设；类条件旋转、尺度变化或new类采用不同偏移时，BCAT可能负迁移。

Phase1旧类地面锚记为`g_c`。当前row的旧类support原型为：

\[
s_c=N\!\left(\sum_{k=1}^{K}x_{ck}\right),
\qquad c\in Y_{old}.
\]

对小至中等共同偏移，在`g_c`处一阶展开：

\[
s_c-g_c=P_c\delta_d+\eta_c+O(\lVert\delta_d\rVert^2),
\qquad
P_c=I-g_cg_c^T.
\]

因此只有切向分量携带一阶域偏移信息；把未投影的`s_c-g_c`直接放入右端会把球面归一化产生的径向二阶项误当作域偏移，冻结版禁止这样做。

## 3.Phase1封存资产

所有资产必须在任何target访问前由Phase1多样本聚合并与checkpoint共同封存：

- 六个旧类单位锚`g_c`；
- 每类support预测噪声`σ²_0,c`；
- 每类ground／类域交互残差`v_g,c`；
- 每类int8 endpoint量化误差`q_c`；
- 共同偏移先验每坐标功率`τ_b²`；
- checkpoint、feature schema、旧类registry、量化尺度和误差receipt。

一个无循环的Phase1构造如下。先从每个source domain的多样本类中心`μ_dc`得到`g_c=N(Σ_d μ_dc)`，再定义：

\[
\bar u_d=\frac1{|Y_{old}|}\sum_cP_c(\mu_{dc}-g_c),
\]

\[
\tau_b^2=\epsilon_0+
\operatorname{Mean}_d\frac{\lVert\bar u_d\rVert^2}{p},
\]

\[
v_{g,c}=\epsilon_0+
\operatorname{Mean}_d\frac{\lVert P_c(\mu_{dc}-g_c)-\bar u_d\rVert^2}{p}+q_c,
\]

\[
\sigma_{0,c}^2=\epsilon_0+
\operatorname{Mean}_{d,i\mid d,c}\frac{\lVert x_{dci}-\mu_{dc}\rVert^2}{p}.
\]

这里`ε_0`只取固定浮点数值底座，不是性能参数。跨类平均和所有按类构造使用相同公式、canonical数值归约及同一量化规则，不按TX ID设专属参数。payload不得包含source行、单样本feature、成员ID、路径或可逆索引。

## 4.固定设计的贝叶斯共同偏移估计

K1不能从单个target support伪造类内方差。定义：

\[
\widehat\sigma_{c,K}^2=
\begin{cases}
0,&K=1,\\
\dfrac{1}{(K-1)p}\sum_k\lVert x_{ck}-s_c\rVert^2,&K>1,
\end{cases}
\]

\[
v_{c,K}=v_{g,c}+q_c+rac{\sigma_{0,c}^2+\widehat\sigma_{c,K}^2}{K},
\qquad
w_{c,K}=v_{c,K}^{-1}.
\]

冻结矩估计为：

\[
u_c=P_c(s_c-g_c),
\]

\[
A=\tau_b^{-2}I+\sum_{c\in Y_{old}}w_{c,K}P_c,
\qquad
r=\sum_{c\in Y_{old}}w_{c,K}u_c,
\]

\[
\widehat\delta=A^{-1}r.
\]

这相当于各向同性Gaussian先验下的闭式线性MAP proxy。若`Q=ΣwP_c`，则一阶条件均值为：

\[
E[\widehat\delta\mid\delta]
\approx(Q+\tau_b^{-2}I)^{-1}Q\delta.
\]

因此它有明确的收缩偏差，不得宣称无条件恢复真实receiver shift。只要至少两个ground锚不共线，`Q`对所有方向正定；即便锚退化，正先验仍令`A≻0`并给出连续有限解。K1是条件可估，不等于低方差。

### 4.1为什么首发不采用`P_s`精确式

无噪声模型`s_c=N(g_c+δ)`严格蕴含：

\[
(I-s_cs_c^T)\delta=-(I-s_cs_c^T)g_c.
\]

它在无噪声时比固定`P_g`一阶式更精确，但K1时同一个随机`s_c`同时出现在设计矩阵与右端，形成errors-in-variables；普通加权岭不再具有其声称的条件无偏性。D113首发优先固定Phase1设计矩阵，以接受可量化的一阶模型误差换取K1噪声不进入设计矩阵。`P_s`只有在建立并封存方向噪声校正后才可成为新的方法版本，不能与首发并行调参择优。

## 5.有界偏移与球面解析逆

对任意有限`δ_hat`定义固定径向收缩：

\[
b=\frac{\widehat\delta}{1+2\lVert\widehat\delta\rVert},
\qquad \lVert b\rVert<\frac12.
\]

它没有接受／拒绝门，零点一阶导为恒等，并把任意估计连续压入稳定域。它也意味着后续逆映射精确对应的是有效偏移`b`；对未收缩的真实`δ_d`只是一阶校正，不能写成全局精确恢复。

若`x=F_b(u)=N(u+b)`，由`u=ax-b`和`||u||=1`得到唯一正根：

\[
t=x^Tb,
\qquad
a=t+\sqrt{t^2+1-\lVert b\rVert^2},
\]

\[
C_b(x)=ax-b.
\]

于是`C_b(F_b(u))=u`且`F_b(C_b(x))=x`。当`||b||<1/2`时，根号项至少为`√3/2`，`a∈(1/2,3/2)`，不存在除零、对径chart或归一化奇点。实现可在浮点误差下对输出做一次只纠正舍入的单位化，但不得把`N(x-b)`重新作为方法。

BCAT在enrollment时用同一个`b`变换全部old/new support；推理时对每条query独立计算`C_b(q)`。query不参与`b`、带宽、原型、阈值或任何选择。

## 6.严格四臂与分类头

|臂|表示|分类头|回答的问题|
|---|---|---|---|
|`M0`|raw|冻结Student-t qKNN|共同基线|
|`M_DA`|全部support/query统一`C_b`|同一个Student-t qKNN|BCAT在base head上的独立效应|
|`M_HEAD`|raw|D112已有正证据的ground单位质量head|分类头独立效应|
|`M_JOINT`|与`M_DA`逐字节同一`b`和变换bank|canonical ground单位质量head|BCAT与head联合效应|

所有臂继承同一`ν`、`d_eff`、Student-t核、带宽规则和`kernel_volume_gamma=1`。support密度保持：

\[
L_c^{sup}(q)=\operatorname{LSE}_{k\le K}\ell(q,x_{ck})-\log K.
\]

对`M_JOINT`的旧类`c`，令`~tilde x=C_b(x)`、`~tilde q=C_b(q)`、`~tilde s_c=N(Σ_k~tilde x_ck)`，ground锚保持canonical的`g_c`，不能再对`g_c`施加`C_b`。旧类分数为：

\[
L_c^J=\operatorname{logaddexp}\left(
\log(1-\rho_c)+L_c^{DA},
\log\rho_c+\ell(\widetilde q,g_c;\widetilde h_c)
\right),
\]

其中anchor核与该类`M_DA` support核使用同一`~tilde h_c`和logit原点。令：

\[
e_c=\frac{\lVert\widetilde s_c-g_c\rVert^2}{p},
\qquad
\rho_c=\frac{\widetilde v_{s,c}}
{\widetilde v_{s,c}+v_{g,c}+v_{b,c}+e_c}.
\]

`v_b,c`是共同偏移估计误差经`C_b`传播到该类canonical原型的每坐标delta-method proxy。令`Σ_delta=A^{-1}`，`J_b=∂b/∂δ_hat`，`H_c=∂C_b(s_c)/∂b`，则：

\[
v_{b,c}=p^{-1}\operatorname{tr}
\left(H_cJ_b\Sigma_\delta J_b^TH_c^T\right).
\]

它只降低不可靠偏移下的ground质量，禁止放入分子。该项是head可靠度的一部分，不改变`M_DA`表示；G1必须同时报告`DA_AT_BASE=M_DA-M0`、`HEAD_AT_RAW=M_HEAD-M0`、`DA_AT_HEAD=M_JOINT-M_HEAD`和factorial interaction，不能只报联合最好值。

new类没有Phase1 ground expert，必须逐列满足：

\[
L_n^J(q)=L_n^{DA}(q),\qquad n\in Y_{new}.
\]

这保证new类不因旧类拥有更多prototype而失去总质量，但不能保证new query绝不被更好的old anchor击败；后者必须由同排`seen_new_acc`、`H_old_new`、new floor和净正确数检验。

## 7.协议、对称性与跨new规模不变性

- `b`只由Phase1 bundle、当前row的合法old support和K构造；query fit/update/selection均为0。
- 同一old support下，new5/new10/new20不得改变`b`、旧类`rho`或旧类带宽；三种规模应具有相同`b receipt`。
- 对旧类标签的任意同步置换只会置换`(g,s,σ,v,q)`项，求和后的`A/r/b`不变；new类标签置换只置换support bank列。
- 每条query独立面对全部已注册类；不读取真值、old/new query角色、batch类数、quota或跨query重排。
- 对所有new类使用同一变换和同一qKNN公式；禁止用new类数量调节old bias。

## 8.资源上界

因为：

\[
A=\left(\tau_b^{-2}+\sum_cw_c\right)I-GWG^T,
\]

可用Woodbury只解一个`6×6`系统，不显式求`160×160`逆。冻结资源目标为：

|阶段|额外开销|
|---|---:|
|Phase1|六类多域聚合；无训练|
|row enrollment|`O(6²p+6³+CKp)`；一次6×6解|
|每个support/query的`C_b`|约`2p=320MAC`＋1次sqrt|
|joint每query anchor核|最多`6p=960MAC`|
|部署新增持久状态|`b`的160维定点载荷、6个`rho`及receipt；既有ground资产复用|
|query依赖状态|0B|
|反向传播／optimizer|0|

`v_b`的trace必须使用相同低秩解或固定解析trace，不得为了方便持久化全精度`160×160`协方差。

## 9.最强反例与停止规则

以下任一现象都可否定D113，而不是触发参数扫描：

1.真实checkpoint G0中K1、K5或K10的`b`、变换feature、score、margin和argmax全部为零函数差异；
2.source-held四臂中`M_DA=M0`或`M_JOINT=M_HEAD`逐预测恒等，说明DA没有独立决策贡献；
3.共同偏移假设不成立，六旧类切向残差方向互相抵消或由类条件扭曲主导；
4.K1离群support让共同`b`同时损伤old floor与new；
5.new类不服从旧类估得的共同偏移，canonicalization降低seen-new、H或总净正确数；
6.偏移接近收缩上界时，一阶估计误差超过逆映射收益。

G0只检查功能和资源，不读truth、不选参数。任一K产生非零合法argmax变化即可进入一次冻结source-held G1；若三K全部argmax变化为0，则关闭D113并研发下一方法。G1只运行四个冻结臂；若`DA_AT_BASE`和`DA_AT_HEAD`均无独立正收益，立即关闭，不跑seed扩展或125。只有G1同时保护old、new与floor后，才运行目标文档规定的单seed Target25必要矩阵。

## 10.理论来源与证据边界

- Snell等，*Prototypical Networks for Few-shot Learning*：少样本类条件原型与距离分类框架，https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html
- Mettes等，*Hyperspherical Prototype Networks*：单位球表征与原型几何，https://proceedings.neurips.cc/paper/2019/hash/02a32ad2669e6fe298e607fe7cc0e1a0-Abstract.html
- Ben-David等，*A Theory of Learning from Different Domains*：域差异、源风险与不可约联合误差的理论边界，https://link.springer.com/article/10.1007/s10994-009-5152-4

上述文献只提供距离分类、球面表示和域适应边界背景。BCAT的固定投影矩估计、有界有效偏移、球面解析逆、四臂结构与资源实现是本项目自行研发的候选机制；未获得真实Target25结果前，不作性能、论文新颖性或可推广性声明。
