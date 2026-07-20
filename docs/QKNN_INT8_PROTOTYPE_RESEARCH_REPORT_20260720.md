# qKNN量化记忆、目标域适应与新类注册研究报告

- 日期：2026-07-20
- 适用协议：`p2_min_v1`
- 研究对象：CVS-RFFI/CV-SincNet的Stage2-B旧类目标域适应与Stage2-C新类注册
- 证据等级：当前协议、代码实现、development support-only实验与理论分析的综合；不把未联合封存组件或诊断结果写成正式性能结论

## 摘要

本项目中的qKNN是quantized K-nearest neighbors，即量化K近邻；这里的`q`表示quantized，不表示query。它从普通KNN演化而来：KNN保存浮点support特征并按近邻投票，qKNN把support特征或类原型压缩为int8，在推理时通过解量化余弦相似度或int8点积完成分类。量化主要解决存储、带宽和部署计算问题，不直接解决目标域偏移、新旧类碰撞与灾难性遗忘。

Stage2的核心困难是同时满足两个目标：旧类需要从地面/source几何迁移到目标receiver的LEO弱信道几何，新类需要用少量目标域support完成追加注册。注册后，每个query必须在\(\mathcal Y_o\cup\mathcal Y_n\)全部已注册类别中独立竞争。即使旧类模型和旧类原型逐bit不变，只要候选集合加入新类，旧类决策区域就会被重新切分，因此旧类准确率仍可能下降。

当前原型体系包含三种不同角色：Phase1地面旧类int8聚合知识是只读身份先验和域漂移参考；target-old support原型负责目标域校正；target-new support原型负责独立新类注册。三者不能被简单平均。理想状态是：同一旧类的地面原型经正确域变换后与target-old真实类中心一致；target-old和target-new都逼近各自在目标域的真实类中心；任意两类的中心距离大于两类半径、量化误差和安全margin之和。若目标域类分布本身重叠，则任何单原型方法的理论上限都低于100%。

截至2026-07-20，ground旧类v2组件已实现并完成真实生成：以一个全局maximin中心域为core，用rank-3低秩残差表达其余域，再量化中心、基、系数和p90半径；D85实测组件状态为5,816B。target-old与target-new已有统一int8原型bank和append-only旧前缀实现，采用逐向量FP16 scale的对称int8量化。D89的development诊断中，ground组件5,816B、目标INT8 affine head 8,583B、总持久状态14,399B，INT8与FP32 matched预测完全一致；但性能相对D81/D85没有严格提升，而且ground v2组件仍为`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`。因此，当前可以声明“量化与状态效率路线可行”，不能声明“旧类适应和新类注册已达到项目目标”。

## 1. 科学问题与协议边界

项目研究“地面训练、星上部署”条件下的弱标注跨接收机域泛化，以及部署到目标receiver后的少样本旧类适应与新类注册。Phase2只允许读取：

- Phase1联合封存的deployment bundle；
- 目标receiver已经接收到的固定`leo_*_weak` IQ；
- 当前row合法注册support及其标签；
- 不含query真值的split与注册表。

一个物理样本只能对应一个固定LEO弱观测。由该固定IQ计算的`z_id160`、FFT、RF统计或均衡view仍属于同一物理样本，不增加K。support与query物理ID互斥，三个场景的物理ID集合也互斥。query只用于锁定后的测试：不能更新模型、原型、温度、阈值、门控、候选选择或回滚状态。

Stage2权限关系如下。

|阶段|可用目标域标签|任务|不能替代的结果|
|---|---|---|---|
|Stage2-A|无target TX标签|zero-label目标域参考或诊断|不能称为few-shot旧类适应或新类注册|
|Stage2-B|`Y_old`的K-shot support|校正旧类目标域几何|不能据此声明新类注册成功|
|Stage2-C|`Y_old∪Y_new`的K-shot support|旧类适应与新类注册共同评估|缺少任一侧都不是完整Stage2-C成功|

预测规则必须是逐query、全注册类决策：

\[
\widehat y(x)=\arg\max_{c\in\mathcal Y_o\cup\mathcal Y_n}S_c(x).
\]

禁止使用query真值、真实old/new角色、batch真实类别数、类别quota、Hungarian匹配、optimal transport或跨query全局重排。

## 2. 从KNN到qKNN

### 2.1 普通KNN

给定support记忆\(\mathcal M=\{(z_i,y_i)\}_{i=1}^{N_s}\)与query特征\(z_q\)，普通KNN计算距离或相似度，选出\(K_{nn}\)个最近support，再进行多数投票或距离加权投票。\(N_s\)是记忆中的support总数，\(K_{nn}\)是近邻数量，不等于后文每类K-shot的\(K_s\)：

\[
\mathcal N_{K_{nn}}(z_q)=\operatorname{TopK}_{K_{nn},i}\;\cos(z_q,z_i),
\qquad
\widehat y=\arg\max_c\sum_{i\in\mathcal N_{K_{nn}}}\mathbf 1[y_i=c]w_i.
\]

其中，\(\mathcal N_{K_{nn}}(z_q)\)是query的\(K_{nn}\)近邻索引集合，\(\mathbf 1[y_i=c]\)是类别相等时取1、否则取0的指示函数，\(w_i\ge0\)是第\(i\)个近邻的投票权重。若采用多数投票，则\(w_i=1\)；若采用距离加权，则\(w_i\)随距离增大而减小。

它的优点是无需重新训练分类头，天然支持追加样本；代价是需要保存较多浮点support向量，类别和shot数增加时，状态与逐query比较次数同步增加。

### 2.2 qKNN中的`q`

本项目中的`q`明确表示quantized。旧版support-level qKNN先对L2归一化support特征执行固定尺度int8量化：

\[
q(z)=\operatorname{clip}(\operatorname{round}(127z),-127,127).
\]

推理时执行：

\[
\tilde z=\operatorname{normalize}(q(z)/127),
\qquad
S_i(x)=\langle \operatorname{normalize}(x),\tilde z_i\rangle.
\]

然后再做top-K近邻、类内投票或类原型混合。这里使用固定127，是因为归一化向量各坐标位于`[-1,1]`。该旧基线仍保存每个选中support的int8向量、标签和old/new标记；其“量化”对象是support记忆，不等于“每类只保存一个原型”。

### 2.3 qKNN与量化原型头的关系

项目后续路线把qKNN的“量化记忆＋相似度分类”思想进一步压缩为类级原型头：

|形态|持久状态|逐query主要计算|特点|
|---|---|---|---|
|普通KNN|全部FP32 support|query对全部support比较|简单，但状态和MAC随`K×类别数`增长|
|support-level qKNN|全部或筛选后的int8 support|解量化/归一化后近邻比较|状态下降，仍保留样本级记忆|
|单/多原型qKNN|每类1个或少量int8原型、scale、radius|query对全部类原型比较|部署最轻，但更依赖原型中心和半径质量|
|编译INT8 affine head|适配器编译进所有类权重|一次全类int8 dot＋逐样本argmax|query路径不再执行适配器，属于原型思想的线性头实现|

因此，qKNN不是一个固定不变的单一算法，而是一条从样本级量化近邻到类级量化原型/线性头的部署路线。它解决“如何存、如何比”，旧类域适应和新类注册还需要额外的几何估计、半径、margin与稳定性约束。

## 3. 三类原型的职责与生命周期

### 3.1 统一符号表

后文统一使用下列符号。向量均为列向量；除非另行说明，原型和用于评分的特征均已L2归一化。

|符号|数学定义|含义|
|---|---|---|
|\(\mathcal Y_o\)|\(\mathcal Y_o=\{1,\ldots,C_o\}\)|旧类集合；\(C_o=\lvert\mathcal Y_o\rvert\)为旧类数|
|\(\mathcal Y_n\)|\(\mathcal Y_n=\{C_o+1,\ldots,C_o+C_n\}\)|新类集合；\(C_n=\lvert\mathcal Y_n\rvert\)为新类数|
|\(\mathcal Y\)|\(\mathcal Y=\mathcal Y_o\cup\mathcal Y_n\)|Stage2-C全部已注册类别|
|\(\mathcal D_g\)|\(\mathcal D_g=\{1,\ldots,D_g\}\)|Phase1地面聚合知识覆盖的ground域集合|
|\(K_s\)|\(K_s\in\mathbb N^+\)|每个已注册类别的shot数；对应\(K_s\)个不同物理support样本|
|\(K_{nn}\)|\(1\le K_{nn}\le N_s\)|KNN推理选择的近邻数量；与K-shot的\(K_s\)不是同一参数|
|\(x\)|\(x\in\mathcal X_t\)|目标receiver已经接收到的一个固定LEO弱信道IQ样本|
|\(f_\theta\)|\(f_\theta:\mathcal X_t\rightarrow\mathbb R^F\)|冻结backbone与可选轻量support适配器组成的特征映射；\(\theta\)为其参数|
|\(\mathcal N\)|\(\mathcal N(v)=v/\lVert v\rVert_2\)|L2归一化算子|
|\(z\)|\(z=\mathcal N(f_\theta(x))\in\mathbb S^{F-1}\)|用于原型评分的\(F\)维单位特征|
|\(\mathcal S_c^t\)|\(\mathcal S_c^t=\{(x_{c,k},c)\}_{k=1}^{K_s}\)|目标域类别\(c\)的K-shot support集合|
|\(g_{c,d}\)|\(g_{c,d}\in\mathbb S^{F-1}\)|Phase1 ground域\(d\)中旧类\(c\)的多样本聚合原型|
|\(t_c\)|\(t_c\in\mathbb S^{F-1}\)|由\(\mathcal S_c^t\)估计的target-old support原型，\(c\in\mathcal Y_o\)|
|\(n_j\)|\(n_j\in\mathbb S^{F-1}\)|由\(\mathcal S_j^t\)估计的target-new support原型，\(j\in\mathcal Y_n\)|
|\(p_c^-\)|\(p_c^-\in\mathbb S^{F-1}\)|新类注册前，旧类\(c\)实际进入预测器的最终原型|
|\(p_c^+\)|\(p_c^+\in\mathbb S^{F-1}\)|新类注册后，类别\(c\in\mathcal Y\)实际进入预测器的最终原型|
|\(r_{c,d}^g,r_c^t,r_j^n\)|\(r\ge0\)|ground旧类、target-old与target-new的类内半径或不确定度尺度|
|\(S_c(x)\)|\(S_c(x)=\langle z,p_c\rangle\)|样本\(x\)对类别\(c\)的余弦score；更复杂头也可将其替换为统一的类score函数|
|\(Q_8\)|\(Q_8(p)=(q_p,a_p)\)|将浮点原型\(p\)编码为int8向量\(q_p\)和FP16 scale \(a_p\)的量化算子|
|\(\widetilde p\)|\(\widetilde p=\mathcal N(a_pq_p)\)|int8原型临时解码并重新归一化后的部署向量|
|\(d_{\angle}\)|\(d_{\angle}(u,v)=\arccos(\operatorname{clip}(\langle u,v\rangle,-1,1))\)|单位球面上的角距离；用于严格表达中心、半径与分离关系|
|\(\gamma\)|\(\gamma>0\)|不同类别之间要求保留的安全margin|

上标“\(-\)”表示新类注册前，上标“\(+\)”表示新类注册后。它们描述生命周期阶段，不表示数值正负。

单位球面记号的完整定义为

\[
\mathbb S^{F-1}
=
\{v\in\mathbb R^F:\lVert v\rVert_2=1\}.
\]

项目配置和实验表通常把shot数写成`K`；为避免与KNN近邻数混淆，本报告公式统一写成\(K_s\)，而KNN近邻数统一写成\(K_{nn}\)。

### 3.2 ground旧类原型：身份先验和域变化参考

在Phase1中，可将ground域\(d\)内旧类\(c\)的聚合过程抽象为

\[
g_{c,d}=\mathcal N\!\left(
\operatorname{Aggregate}\{f_{\theta_0}(x_i):y_i=c,\ d_i=d\}
\right),
\qquad c\in\mathcal Y_o,\ d\in\mathcal D_g,
\]

其中，\(\theta_0\)是Phase1完成后的冻结模型参数，\(y_i\)和\(d_i\)分别表示Phase1样本的类别与ground域标记，\(\operatorname{Aggregate}(\cdot)\)表示多样本聚合。该式只描述target访问前的生成过程；Phase2不能读取式中的单样本\(x_i\)、\(y_i\)或全精度特征，只能读取与checkpoint联合封存的压缩聚合结果。

ground bank的核心价值不是替target receiver直接作最终分类，而是提供三个先验：旧类身份中心\(g_{c,d}\)、跨ground域漂移方向\(g_{c,d}-\bar g_c\)以及类内半径\(r_{c,d}^g\)，其中

\[
\bar g_c=\mathcal N\!\left(\sum_{d\in\mathcal D_g}\omega_{c,d}g_{c,d}\right),
\qquad
\omega_{c,d}\ge0,
\qquad
\sum_{d\in\mathcal D_g}\omega_{c,d}=1.
\]

\(\omega_{c,d}\)是ground域可靠度权重。它必须在target query不可见的条件下由封存统计或support-only规则确定。记压缩ground bundle的持久状态为\(\mathcal G\)。Phase2生命周期中：

\[
\mathcal G_{Stage2-C}
=
\mathcal G_{Stage2-B}
=
\mathcal G_{Phase1},
\]

即ground bundle在Phase1封存、Stage2-B适配和Stage2-C注册三个阶段内容完全相同，也不接受Phase2更新。

### 3.3 target-old原型：把旧身份搬到目标域

对旧类\(c\in\mathcal Y_o\)，target-old原型只由该类合法K-shot support形成：

\[
t_c=\mathcal N\!\left(
\operatorname{RobustCenter}
\{\mathcal N(f_\theta(x_{c,k}))\}_{k=1}^{K_s}
\right).
\]

\(\operatorname{RobustCenter}(\cdot)\)可以是均值、medoid、Huber中心或预先锁定的稳健中心，但不能使用query选择。\(t_c\)的作用是估计旧类在当前receiver和LEO弱场景下的位置，因此它承担域校正，而不是复制ground中心。

若使用ground弱先验，注册前最终旧类原型可写为

\[
p_c^-=\mathcal N\!\left(
\lambda_c t_c+(1-\lambda_c)\bar g_c^{\,t}
\right),
\qquad 0\le\lambda_c\le1,
\]

其中，\(\bar g_c^{\,t}\)是经合法、类无关target域校正后的ground参考；\(\lambda_c\)是target证据权重。\(K_s\)越大、\(r_c^t\)越小、support内部稳定性越高，\(\lambda_c\)应越接近1。ground不确定度更低且target support更弱时，才允许提高\(1-\lambda_c\)，但ground仍不能直接覆盖\(t_c\)。

### 3.4 target-new原型：建立新身份而不是修正旧身份

对新类\(j\in\mathcal Y_n\)，没有同身份ground原型，因此

\[
n_j=\mathcal N\!\left(
\operatorname{RobustCenter}
\{\mathcal N(f_\theta(x_{j,k}))\}_{k=1}^{K_s}
\right),
\qquad
p_j^+=n_j.
\]

target-new原型必须独立表达新身份。把某个ground旧类原型映射给新类，相当于假设不存在的身份对应关系，违反项目的类身份边界。

三类状态的职责可压缩为下表。

|状态|主要数学对象|职责|生命周期约束|
|---|---|---|---|
|ground旧类|\(\{g_{c,d},r_{c,d}^g\}\)|旧类身份先验、ground域漂移与不确定度参考|Phase2全程只读且注册前后不变|
|target-old|\(\{t_c,r_c^t\}\)|估计旧类在目标receiver中的中心与散布|Stage2-B由旧类support形成并锁定|
|target-new|\(\{n_j,r_j^n\}\)|建立新类在目标receiver中的新决策区域|Stage2-C仅作为新后缀追加|

### 3.5 从Stage2-B到Stage2-C的状态转移

Stage2-B完成后，旧类注册表记为

\[
\mathcal R_B=
\left\{
(c,Q_8(p_c^-),r_c^t,K_s):c\in\mathcal Y_o
\right\}.
\]

Stage2-C追加新类后：

\[
\mathcal R_C=
\mathcal R_B\ \Vert\
\left\{
(j,Q_8(n_j),r_j^n,K_s):j\in\mathcal Y_n
\right\},
\]

其中，\(\Vert\)表示append-only拼接，不是重新拟合或重排旧状态。严格的旧前缀不变条件为

\[
\operatorname{bytes}(\mathcal R_C[1:C_o])
=
\operatorname{bytes}(\mathcal R_B).
\]

对应的旧类原型关系是

\[
p_c^+=p_c^-,
\qquad c\in\mathcal Y_o.
\]

这些等式只保证旧类持久状态没有被改写。注册后的预测集合从\(\mathcal Y_o\)扩大到\(\mathcal Y_o\cup\mathcal Y_n\)，所以它们不能推出旧类准确率不变。

## 4. 当前int8如何压缩

### 4.1 旧版support-level qKNN

对每个归一化support向量使用统一尺度127：

\[
q_i=\operatorname{clip}(\operatorname{round}(127z_i),-127,127),
\qquad q_i\in\mathbb Z^{F}_{int8}.
\]

持久状态包括`quantized_matrix`、support标签、old/new标记，以及可选的FP类原型、半径和计数。由于该路径的类原型仍可能保留FP64/FP32，它是历史qKNN基线，不等同于当前“全部target原型int8”的部署目标。

### 4.2 ground旧类v2组件

设原始地面域×类中心bank为

\[
P^g\in\mathbb R^{D_g\times C_o\times F_g},
\qquad F_g=160.
\]

其中，\(D_g\)是ground域数，\(C_o\)是旧类数，\(F_g\)是ground原型特征维数。当前v2实现不再持久化完整dense bank，而执行：

1. 在Phase1离线阶段用全局maximin规则选一个中心域；其\(C_o\times F_g\)类中心形成core。
2. 对其余\(D_g-1\)个域相对core的残差，按类做rank-3 SVD。
3. 持久化\(Q^{core}\in\mathbb Z_{int8}^{C_o\times F_g}\)、\(Q^{basis}\in\mathbb Z_{int8}^{C_o\times3\times F_g}\)和\(Q^{coeff}\in\mathbb Z_{int8}^{(D_g-1)\times C_o\times3}\)。
4. 持久化p90类内余弦距离半径\(Q^{radius}\in\mathbb Z_{int8}^{D_g\times C_o}\)。
5. 中心、基、系数采用“最后一维每向量一个FP16 scale”的对称int8量化；半径采用“每类一个FP16 scale”的非负`[0,127]`量化。

ground v2向量codec先用FP32 scale计算int8 code，再把scale持久化为FP16：

\[
a_v^{32}=
\begin{cases}
\dfrac{\max_{1\le k\le L}|v_k|}{127},&\max_k|v_k|>0,\\
1,&\max_k|v_k|=0,
\end{cases}
\qquad
a_v^{16}=\operatorname{FP16}(a_v^{32}),
\]

\[
q_{v,k}=\operatorname{clip}
\left(operatorname{round}\left(\frac{v_k}{a_v^{32}}\right),-127,127\right),
\qquad
\widehat v_k=a_v^{16}q_{v,k}.
\]

其中，\(v=(v_1,\ldots,v_L)\)表示待量化的一个长度为\(L\)的向量，\(k\)是坐标索引，\(a_v^{32}\)是计算量化code使用的FP32 scale，\(a_v^{16}\)是实际持久化的FP16 scale，\(q_{v,k}\)是int8 code，\(\widehat v_k\)是临时解码值。

ground半径按类别跨域共享scale。对旧类\(c\)：

\[
a_c^{r,32}=
\begin{cases}
\dfrac{\max_{d\in\mathcal D_g}r_{c,d}^g}{127},
&\max_d r_{c,d}^g>0,\\
1,&\max_d r_{c,d}^g=0,
\end{cases}
\qquad
a_c^{r,16}=\operatorname{FP16}(a_c^{r,32}),
\]

\[
q_{d,c}^{r}=\operatorname{clip}\left(
\operatorname{round}\left(\frac{r_{c,d}^g}{a_c^{r,32}}\right),0,127
\right),
\qquad
\widehat r_{c,d}^g=a_c^{r,16}q_{d,c}^{r}.
\]

其中，\(a_c^{r,32}\)是计算半径code使用的FP32 scale，\(a_c^{r,16}\)是旧类\(c\)实际持久化的FP16半径scale，\(q_{d,c}^r\)是域\(d\)、类\(c\)的非负int8半径code，\(\widehat r_{c,d}^g\)是解码半径。零向量或全零半径使用scale 1，避免除零。

部署时只临时重构ground原型：

\[
\hat g_{c,d}=\hat g^{core}_c+\sum_{r=1}^{3}\hat\alpha_{d,c,r}\hat b_{c,r}.
\]

其中，\(\widehat g_c^{core}\)是旧类\(c\)的解码中心域core，\(\widehat b_{c,r}\)是该类第\(r\)个解码残差基，\(\widehat\alpha_{d,c,r}\)是域\(d\)对应的解码低秩系数，\(r\in\{1,2,3\}\)是固定rank索引。帽号“\(\widehat{\cdot}\)”统一表示由int8 code和FP16 scale临时重构的量。

D85真实组件包含14个ground域×6个旧类cell，逻辑组件状态为5,816B；相对旧表示，ground组件状态下降77.13%。但该组件当前仍等待外部联合封存，不能作为已获正式Phase2资格的bundle。

### 4.3 target-old与target-new原型bank

当前统一target prototype bank对每个归一化类向量采用逐向量scale：

\[
a_c=\operatorname{FP16}\left(\frac{\max_k|p_{c,k}|}{127}\right),
\qquad
q_{c,k}=\operatorname{clip}(\operatorname{round}(p_{c,k}/a_c),-127,127).
\]

其中，\(p_{c,k}\)是类别\(c\)浮点原型的第\(k\)个坐标，\(a_c\)是该类持久化的FP16 scale，\(q_{c,k}\)是对应int8 code。`-128`被明确禁止。每类同时保存FP16`radius`和support`count`。旧类bank先建立，新类注册时只计算新类后缀并拼接；旧类`q/scale/radius/count/class registry`由SHA256前缀约束保持不变。INT8评分为：

\[
S_c^{INT8}(x)=a_c\langle z,q_c\rangle,
\qquad z=\mathcal N(f_\theta(x)).
\]

FP32与FP16只作为matched ablation；活动研究目标要求最终target-old与target-new采用同一量化schema，优先int8。D36、D37及D81-D89系路线已在真实development流程中使用INT8旧/新头，但“代码实现”和“实验使用”仍不等于“性能晋级”。

### 4.4 当前三类状态的准确结论

|对象|当前压缩状态|已验证内容|尚不能声明的内容|
|---|---|---|---|
|ground旧类|int8 core＋rank-3 int8残差＋int8半径＋FP16 scales|真实84个cell生成、重构、量化误差和资源审计；D89中5,816B|尚未完成外部联合封存，不能称为正式Phase2部署组件|
|target-old|统一类向量int8＋逐向量FP16 scale＋FP16 radius/count|bank编码、评分、旧前缀hash；多条development路线真实运行|不能由“prefix不变”推出注册后旧类无遗忘|
|target-new|与target-old相同schema，append-only后缀|新类独立量化并参加全类评分|不能由量化保真推出新类可达或旧类安全|

## 5. 为什么新类注册会影响旧类性能

### 5.1 候选集合扩张是最基本原因

设旧类query \(x\)的真实类别为\(y\in\mathcal Y_o\)。注册前，query只需战胜其他旧类：

\[
\widehat y^-(x)=\arg\max_{c\in\mathcal Y_o}S_c^-(x).
\]

定义注册前旧类margin：

\[
m_o^-(x)=S_y^-(x)-
\max_{c\in\mathcal Y_o\setminus\{y\}}S_c^-(x).
\]

其中，\(S_c^-(x)\)是注册前类别\(c\)的score；\(m_o^-(x)>0\)表示\(x\)在旧类集合内被正确分类。

若旧状态冻结，注册后旧类score仍为\(S_c^-(x)\)，但同一query必须面对全部新类：

\[
\widehat y^+(x)=\arg\max_{c\in\mathcal Y_o\cup\mathcal Y_n}S_c^+(x).
\]

定义旧类相对新类的注册margin：

\[
m_n(x)=S_y^-(x)-\max_{j\in\mathcal Y_n}S_j^+(x).
\]

在旧状态完全不变时，注册后仍保持正确的充分且必要条件是

\[
\min\{m_o^-(x),m_n(x)\}>0.
\]

因此，满足\(m_o^-(x)>0\)但\(m_n(x)\le0\)的样本会纯粹因为候选集合扩张而从正确旧类翻转为新类。这一机制不需要灾难性参数遗忘；增加新类score列本身就会改变决策区域。

### 5.2 新类support具有目标域匹配优势

用局部加性模型表示单位球面归一化前的特征：

\[
h(x)=\mu_y^{id}+\delta_t+\varepsilon_x,
\]

其中，\(\mu_y^{id}\)是旧类\(y\)的身份分量，\(\delta_t\)是目标receiver与当前LEO场景的公共域偏移，\(\varepsilon_x\)是样本噪声。ground旧类原型和target-new原型可近似写为

\[
g_y\approx\mu_y^{id}+\delta_g,
\qquad
n_j\approx\mu_j^{id}+\delta_t+\bar\varepsilon_j,
\]

其中，\(\delta_g\)是ground域偏移，\(\mu_j^{id}\)是新类\(j\)的身份分量，\(\bar\varepsilon_j\)是K-shot中心估计误差。

在欧氏局部近似下，query到两类中心的距离分别包含

\[
\lVert h(x)-g_y\rVert_2
\approx
\lVert\delta_t-\delta_g+\varepsilon_x\rVert_2,
\]

\[
\lVert h(x)-n_j\rVert_2
\approx
\lVert\mu_y^{id}-\mu_j^{id}+\varepsilon_x-\bar\varepsilon_j\rVert_2.
\]

第二式中的公共目标域偏移\(\delta_t\)被抵消，第一式却保留\(\delta_t-\delta_g\)。当旧类身份优势小于这一残余域偏移时，可能出现

\[
d_{\angle}(z,n_j)<d_{\angle}(z,g_y),
\]

即新类凭借共享target域特征而非正确身份获得更高score。target-old原型\(t_y\)的任务正是消除这一不对称；只保留ground旧锚会使新类具有目标域匹配优势。

### 5.3 few-shot中心估计噪声

设新类\(j\)在目标域的真实特征均值与协方差分别为\(\mu_j^t\)和\(\Sigma_j^t\)，K-shot样本均值为

\[
\widehat\mu_j^t=\frac{1}{K_s}\sum_{k=1}^{K_s}z_{j,k}.
\]

在support独立同分布的近似下：

\[
\mathbb E[\widehat\mu_j^t]=\mu_j^t,
\qquad
\operatorname{Cov}(\widehat\mu_j^t)=\frac{\Sigma_j^t}{K_s},
\]

\[
\mathbb E\lVert\widehat\mu_j^t-\mu_j^t\rVert_2^2
=\frac{\operatorname{tr}(\Sigma_j^t)}{K_s}.
\]

其中，\(\mathbb E[\cdot]\)表示期望，\(\operatorname{Cov}(\cdot)\)表示协方差，\(\operatorname{tr}(\Sigma_j^t)\)是协方差矩阵迹，即类内各维方差之和。\(K_s\)越小，原型中心误差越大；如果误差方向指向某个旧类区域，归一化后的\(n_j=\mathcal N(\widehat\mu_j^t)\)就会扩大新类Voronoi区域并侵入旧类。\(K_s=1\)时只有一个support，无法从该类内部估计散布，故半径必须使用预锁定值\(r_0\)，不能把self-distance \(d_{\angle}(z_{j,1},n_j)=0\)解释为“新类方差为零”。

### 5.4 多新类带来的极值效应

对固定旧类query \(x\)，令事件

\[
E_j(x)=\{S_j^+(x)\ge S_y^-(x)\}
\]

表示新类\(j\)的score不低于真实旧类score。\(\Pr(\cdot)\)表示事件概率，\(E_j^c(x)\)表示事件\(E_j(x)\)的补集。若\(\pi_j(x)=\Pr(E_j(x))\)，至少一个新类侵入的概率为

\[
\Pr\!\left(\bigcup_{j\in\mathcal Y_n}E_j(x)\right)
=1-\Pr\!\left(\bigcap_{j\in\mathcal Y_n}E_j^c(x)\right).
\]

在仅用于说明趋势的独立、同概率近似\(\pi_j(x)=\pi(x)\)下：

\[
\Pr(\text{至少一个新类侵入})
=1-[1-\pi(x)]^{C_n}.
\]

当\(\pi(x)>0\)时，该概率随新类数\(C_n\)单调增加。真实新类score并不独立，但\(\max_{j\in\mathcal Y_n}S_j^+(x)\)仍呈极值竞争：新类从2扩展到5、10、20时，旧类必须同时战胜更多竞争者，最差类和尾部margin通常先受损。

### 5.5 联合适配造成表示或校准漂移

若Stage2-C更新adapter、温度、bias或旧类prototype，定义旧类score漂移

\[
\Delta_c(x)=S_c^+(x)-S_c^-(x),
\qquad c\in\mathcal Y_o.
\]

注册后的旧类内部margin变为

\[
m_o^+(x)=S_y^-(x)+\Delta_y(x)
-\max_{c\in\mathcal Y_o\setminus\{y\}}
\left[S_c^-(x)+\Delta_c(x)\right].
\]

即使新类score尚未超过真实旧类，只要正确类漂移\(\Delta_y(x)\)低于某个错误旧类的漂移，旧类内部也会发生翻转。为了区分“新增竞争”和“旧状态漂移”，可定义：

\[
F_{comp}=A_o^- - A_o^{append},
\]

\[
F_{drift}=A_o^{append}-A_o^+,
\]

\[
F_{total}=A_o^- - A_o^+
=F_{comp}+F_{drift}.
\]

其中，\(A_o^-\)是注册前旧类准确率；\(A_o^{append}\)是在旧状态完全冻结、只追加新类时的旧类准确率；\(A_o^+\)是实际Stage2-C后的旧类准确率。\(F_{comp}\)度量纯候选扩张损失，\(F_{drift}\)度量共享适配或校准改写造成的额外损失。D36中ground anchor与公共margin校准未形成Pareto改善，说明全局offset只能重新分配old/new错误，不能自动修复类特异重叠。

### 5.6 量化误差会翻转小margin样本

设浮点原型为\(p_c\)，int8解码原型为\(\widetilde p_c\)，向量误差为

\[
e_c=\widetilde p_c-p_c,
\qquad
\epsilon_c=\lVert e_c\rVert_2.
\]

因为\(\lVert z\rVert_2=1\)，Cauchy-Schwarz不等式给出单类score误差上界：

\[
|S_c^{INT8}(x)-S_c^{FP}(x)|
=|\langle z,e_c\rangle|
\le\epsilon_c.
\]

定义浮点全注册类margin：

\[
M_{FP}(x)=S_y^{FP}(x)-\max_{c\in\mathcal Y\setminus\{y\}}S_c^{FP}(x).
\]

若

\[
M_{FP}(x)>\epsilon_y+
\max_{c\in\mathcal Y\setminus\{y\}}\epsilon_c,
\]

则正确类score即使向下偏移\(\epsilon_y\)，最强竞争类score即使向上偏移其误差上界，量化后仍不会翻转。该条件是逐样本的充分条件，不要求原型逐元素无误差。D37两级residual-int8量化误差约为\(10^{-6}\)量级，D89中INT8/FP32 outer预测与margin sign均零翻转，说明当前development瓶颈是类几何，而不是8bit表示精度。

### 5.7 六类机制如何在实验中区分

|机制|关键数学量|直接诊断信号|
|---|---|---|
|候选集合扩张|\(m_n(x)\)、\(F_{comp}\)|旧类状态和旧类score逐bit不变，但部分旧query的winner变为新类|
|target域匹配优势|\(\lVert\delta_t-\delta_g\rVert_2\)|old→new错误集中在ground-target域差较大的旧类；target-old校正后应下降|
|few-shot中心噪声|\(\operatorname{tr}(\Sigma_j^t)/K_s\)|更换support seed时新类中心和侵入数波动；\(K_s\)增大后应收敛|
|多新类极值效应|\(1-[1-\pi(x)]^{C_n}\)|在相同\(K_s\)和旧类状态下，\(C_n\)增大时尾部margin与最差旧类率先下降|
|联合适配漂移|\(\Delta_c(x)\)、\(F_{drift}\)|不加入新类score、仅比较适配前后旧类score，旧类winner已经改变|
|量化翻转|\(M_{FP}(x)\)、\(\epsilon_c\)|INT8/FP32 top-1不一致、margin sign flip或错误集中在小浮点margin样本|

这六类机制可以同时发生。实验报告应至少保留注册前score、append-only全类score、最终适配后score和FP32 matched score，才能分别估计\(F_{comp}\)、\(F_{drift}\)与量化翻转；只比较最终注册前后准确率无法识别根因。

## 6. 如何同时适应旧类并注册新类

一个合理的统一方法应按以下顺序工作。

### 6.1 Stage2-B先形成可靠target-old状态

对每个旧类，用合法K-shot目标support估计稳健中心与半径：

\[
t_c=\mathcal N\!\left(
\operatorname{RobustCenter}
\{\mathcal N(f_\theta(x_{c,k}))\}_{k=1}^{K_s}
\right).
\]

\(f_\theta\)可以是冻结backbone加极轻adapter，但开发只能使用support内部physical-rank交叉拟合。首要目标是提高注册前旧类总体准确率和最差类floor；若Stage2-B本身不足，Stage2-C再精细的注册门也只是保护一个较弱旧头。

### 6.2 ground知识只作不确定度受控先验

ground旧类不应直接覆盖target-old。可使用不确定度权重融合：

\[
p_c^-=\mathcal N\!\left(
\lambda_c t_c+(1-\lambda_c)\bar g_c^{\,t}
\right),
\]

其中，\(\bar g_c^{\,t}\)是经合法、类无关域校正后的ground先验，\(\lambda_c\)随\(K_s\)增大、target半径减小和support可信度提高而增大。实际部署中ground权重应是弱先验；新类没有同类ground原型，必须保持纯target注册。

### 6.3 Stage2-C追加新类而不改写旧类持久状态

对每个新类独立形成：

\[
p_j^+=n_j=\mathcal N\!\left(
\operatorname{RobustCenter}
\{\mathcal N(f_\theta(x_{j,k}))\}_{k=1}^{K_s}
\right).
\]

量化后只追加到registry后缀。注册前旧状态作为teacher/anchor，用于support侧蒸馏、参数位移约束和old-prefix审计。append-only保证可追踪性，但最终安全门必须直接检查held旧样本的old→new侵入。

### 6.4 用半径和margin约束真实碰撞

对任意两个注册类\(i,j\in\mathcal Y\)，理想分离条件为：

\[
d_{\angle}(p_i,p_j)
>
\rho_i^t+\rho_j^t+\eta_i+\eta_j+\gamma,
\]

其中，\(\rho_i^t,\rho_j^t\)是目标域角半径，\(\eta_i,\eta_j\)是两类原型的量化角误差，\(\gamma\)是部署安全margin。损失应同时覆盖：

- 旧类support分类与最差类风险；
- 新类support分类与新类可达性；
- 注册前后旧类margin保持；
- old/new及new/new的radius-sum分离；
- adapter参数位移和量化感知误差。

所有类必须使用同一公式，不能按历史TX名称设置专属阈值或白名单。

### 6.5 量化感知锁定而非事后压缩

候选选择时应同时计算FP32、FP16与INT8 matched结果，保持相同support、类别顺序、半径和推理规则。选择顺序是：先满足old/new/H/floor/forgetting非劣，再比较状态、延迟、MAC和临时内存。INT8位宽更低不代表端到端一定更快；若FP16在真实batch=1 kernel上形成更好的联合Pareto，可以锁定FP16，而不能预设INT8必胜。

## 7. 三者最理想的关系与理论上限

### 7.1 几何上限

设目标域类别\(c\)的单位特征随机变量为\(Z\mid Y=c\)，其真实球面中心为

\[
\mu_c^t=\mathcal N\!\left(\mathbb E[Z\mid Y=c]\right).
\]

其中，\(Y\)是真实类别随机变量，\(Z\)是目标域单位特征随机变量，上标\(t\)表示target域。对旧类，设\(T_t\)为不依赖query真值、由合法support确定的ground到target域变换。理想同类对齐关系为

\[
T_t(\bar g_c)=t_c=p_c^-=\mu_c^t,
\qquad c\in\mathcal Y_o.
\]

若backbone已经实现完全域不变，则\(T_t\)退化为恒等映射，\(\bar g_c=t_c=\mu_c^t\)。真实部署通常不满足这一条件，所以ground与target-old不应被强制重合；正确目标是经过域校正后指向同一target类中心。

对新类，没有对应ground身份，理想关系只有

\[
n_j=p_j^+=\mu_j^t,
\qquad j\in\mathcal Y_n.
\]

定义类别\(c\)的目标域角半径上界\(\rho_c^t\)：

\[
\Pr\!\left[
d_{\angle}(Z,\mu_c^t)\le\rho_c^t\mid Y=c
\right]\ge1-\alpha,
\]

其中，\(\alpha\in[0,1]\)是允许落在半径之外的尾部概率。若实现使用p90半径，则概念上对应\(\alpha=0.1\)，但代码中的余弦距离半径需先转换为角距离后才能直接使用以下三角不等式。

对任意不同类别\(i,j\in\mathcal Y\)，理想稳健分离条件为

\[
d_{\angle}(\mu_i^t,\mu_j^t)
>
\rho_i^t+\rho_j^t+\gamma.
\]

\(d_{\angle}\)是单位球面上的角距离，\(\rho_i^t+\rho_j^t\)覆盖两类自身散布，\(\gamma\)提供额外安全margin。该条件同时约束old/old、old/new和new/new三种类对。若全部类球面区域满足该式，则新类注册不会切入旧类高概率区域。

### 7.2 有限K-shot条件下最理想的ground-target融合

ground参考\(\bar g_c^{\,t}\)和target-old估计\(t_c\)都可能带误差。设它们在同一局部切空间中是对\(\mu_c^t\)的近似无偏估计，误差方差分别为\(\sigma_{g,c}^2\)和\(\sigma_{t,c}^2\)，对应精度为

\[
\tau_{g,c}=\frac{1}{\sigma_{g,c}^2},
\qquad
\tau_{t,c}=\frac{1}{\sigma_{t,c}^2}.
\]

最小方差线性融合的target权重为

\[
\lambda_c^*=\frac{\tau_{t,c}}
{\tau_{t,c}+\tau_{g,c}},
\]

最终旧类原型为

\[
p_c^{-*}=\mathcal N\!\left(
\lambda_c^*t_c+(1-\lambda_c^*)\bar g_c^{\,t}
\right).
\]

当\(K_s\)增大或target support半径下降时，\(\sigma_{t,c}^2\)减小、\(\tau_{t,c}\)增大，因此\(\lambda_c^*\rightarrow1\)，最终决策更多依赖target-old。ground域与target域差异过大时，ground估计存在偏差，上述无偏融合不再成立，此时ground权重应进一步减小。新类没有\(\bar g_j^{\,t}\)，所以其理想原型仍是\(p_j^+=n_j\)，不能套用旧类融合式。

### 7.3 量化无损条件

对浮点原型\(p_c\)执行量化和解码：

\[
(q_c,a_c)=Q_8(p_c),
\qquad
\widetilde p_c=\mathcal N(a_cq_c).
\]

定义类别\(c\)的角量化误差与L2量化误差：

\[
\eta_c=d_{\angle}(p_c,\widetilde p_c),
\qquad
\epsilon_c=\lVert p_c-\widetilde p_c\rVert_2
=2\sin\frac{\eta_c}{2}.
\]

对任意单位query特征\(z\)，有

\[
|\langle z,p_c\rangle-\langle z,\widetilde p_c\rangle|
\le\epsilon_c.
\]

因此，若每个评估样本\((x,y)\)的浮点margin满足

\[
M_{FP}(x)>\epsilon_y+
\max_{c\in\mathcal Y\setminus\{y\}}\epsilon_c,
\]

则int8与FP32的top-1预测严格一致。数据集级量化无损定义为

\[
\forall x\in\mathcal Q:\quad
\arg\max_{c\in\mathcal Y}S_c^{INT8}(x)
=
\arg\max_{c\in\mathcal Y}S_c^{FP32}(x),
\]

其中，\(\mathcal Q\)是只用于测试的合法query集合。更严格的工程无损还要求margin符号、radius排序和旧类前缀hash不变。D89的development cell实现了INT8/FP32预测等价与零margin sign flip，但这只证明量化没有进一步损伤既有分类器，不证明FP32原型已经达到理想几何。

将量化误差显式加入类间分离后，一个保守的充分条件是

\[
d_{\angle}(p_i,p_j)
>
\rho_i^t+\rho_j^t+\eta_i+\eta_j+\gamma,
\qquad i\ne j.
\]

该式表示：浮点类中心间距必须同时容纳两类散布、两类量化角误差和安全margin。

### 7.4 新类注册不降低旧类性能的理想条件

设\(\mathcal Q_o\)为旧类query集合。若旧状态冻结，则注册后旧类准确率完全不下降的逐样本条件为

\[
\forall(x,y)\in\mathcal Q_o:\quad
S_y^-(x)>
\max\left\{
\max_{c\in\mathcal Y_o\setminus\{y\}}S_c^-(x),
\max_{j\in\mathcal Y_n}S_j^+(x)
\right\}.
\]

在该条件下：

\[
A_o^+=A_o^-,
\qquad
F_o=A_o^- - A_o^+=0,
\]

其中，\(A_o^-\)和\(A_o^+\)分别是注册前后旧类准确率，\(F_o\)是旧类遗忘。若还要求新类全部正确，则需同时满足

\[
\forall(x,j)\in\mathcal Q_n:\quad
S_j^+(x)>
\max_{c\in\mathcal Y\setminus\{j\}}S_c^+(x),
\]

其中，\(\mathcal Q_n\)是已注册新类query集合。两组不等式共同成立时，注册实现“旧类零损失＋新类完全可达”。

### 7.5 不可突破的统计上限

理想原型不能突破特征空间本身的信息上限。给定目标域特征\(Z\)，Bayes最优准确率为

\[
A_{Bayes}
=
\mathbb E_Z\left[
\max_{c\in\mathcal Y}\Pr(Y=c\mid Z)
\right].
\]

设\(\mathcal H_{proto}\)是所有合法单原型分类器的集合，其最优准确率为

\[
A_{proto}^*
=
\sup_{h\in\mathcal H_{proto}}\Pr[h(Z)=Y].
\]

设\(\mathcal H_{int8}\subseteq\mathcal H_{proto}\)是满足当前int8表示和资源约束的原型分类器集合，则

\[
A_{int8}^*
\le
A_{proto}^*
\le
A_{Bayes}
\le1.
\]

若量化无损条件对最优浮点原型的全部query成立，则\(A_{int8}^*=A_{proto}^*\)。若不同TX在目标receiver和LEO弱场景下的条件分布重叠，则\(A_{Bayes}<1\)，任何方法都无法达到100%。若类别呈多模态分布，单原型集合\(\mathcal H_{proto}\)还会受到模型表达能力限制，此时\(A_{proto}^*<A_{Bayes}\)。少量子原型可以扩大假设空间，但必须将新增状态和MAC计入matched Pareto审计。

有限\(K_s\)还引入估计上限。对均值型原型，在独立同分布和有限二阶矩条件下，\(K_s\rightarrow\infty\)时\(\widehat\mu_c^t\)收敛到\(\mu_c^t\)；\(K_s\)有限时仍保留约为\(\operatorname{tr}(\Sigma_c^t)/K_s\)的中心估计误差。因此，“理论上限”应分成三层：Bayes上限描述特征本身可分性，浮点原型上限描述单/多原型模型能力，int8上限描述量化与资源约束。三者不能混为同一个100%目标。

## 8. 当前实验脉络与证据结论

下表只汇总与“旧类适应＋新类注册＋int8原型”直接相关的development证据。它们均不能替代独立多receiver、多seed确认。

|路线|核心问题|同row关键结果|证据结论|
|---|---|---|---|
|D36联合编译int8|联合学习轻adapter、ground弱锚和old/new校准|D36-C：before-old80.56%、after-old66.11%、new52.00%、H56.82%、遗忘14.44pp|注册前旧头已弱于B3，ground权重与公共margin未解决类碰撞；负路线|
|D37 B3-preserving residual-int8|尝试保留旧头并让旧/新使用同一两级int8|D37-A：82.22%/71.11%/58.67%/H62.99%，遗忘11.11pp；量化误差约`10^-6`|量化极保真，但接入的是较弱旧头，且OOF公共offset区间全部不可行；负路线|
|D85 ground radius v2|真实生成ground core＋rank-3残差＋p90半径|92.78%/82.78%/84.67%/H82.94%，遗忘10.00%；INT8/FP32同预测|ground状态显著压缩，但半径加权未改变离散预测；效率正、性能中性|
|D86反事实鲁棒中心|用ground方向和半径重加权target support中心|与D85的15/15 outer预测相同；FP32出现1次负flip|中心变化被int8边界吸收；性能中性且量化不稳定|
|D87 sigma margin head|让ground半径直接改变全类边界|after-old85.00%、new83.33%、H83.58%、遗忘7.78%|旧类改善但新类下降，属于old/new交换；不晋级|
|D88逐类Pareto保护|限制每类clean OOF CE不升|after-old82.22%、new84.67%、H82.62%、遗忘10.56%|保护新类但撤销旧类收益，并相对D85略退化；过约束负路线|
|D89 v2半径Cauchy中心|用5,816B v2 ground谱无损替代D81旧ground谱|92.78%/82.78%/84.67%/H82.94%、遗忘10.00%，与D81/D85 15/15预测相同|总状态14,399B，效率正、性能中性、组件未联合封存；当前最强“性能等价压缩”证据，不是性能最强新版本|

这条实验链给出三个稳定认识：

1. int8量化已经不是主要准确率瓶颈。D37和D89都显示量化可以保持FP32决策。
2. ground原型的绝对中心或全类共享平移难以修正类特异碰撞；其更有价值的信息是域漂移方向、半径和support可靠性。
3. 旧类收益与新类损失容易互换。任何只提高after-old或只提高seen-new的路线都不能晋级，必须形成同row Pareto改善。

## 9. 评价方法与晋级标准

每个候选必须在同一row、同一旧类query和同一推理规则下报告：

\[
F_{old}=A_{old}^{before}-A_{old}^{after},
\]

\[
H_{old,new}=\frac{2A_{old}^{after}A_{new}}
{A_{old}^{after}+A_{new}}.
\]

至少保留以下联合指标：

- 注册前old、注册后old、seen-new与`H_old_new`；
- 每个旧类和新类准确率、最差类floor与row floor；
- old→new、new→old与new→wrong-new混淆；
- forgetting及paired注册前后变化；
- FP32/FP16/INT8重构误差、top-1一致率、margin sign flip；
- 持久状态、临时状态、每query MAC、注册MAC、平均/P95延迟和峰值显存。

当前活动目标要求最终确认覆盖5个receiver、至少5个seed、3个场景、`K∈{1,5,10,20}`和新类规模`{2,5,10,20}`。125-row screen只是局部稳定性筛选。D85-D89仅为一个development receiver、一个seed、K10实际K8、new5、3场景×5fold诊断，不能外推为正式达标。

## 10. 研究判断与下一步

当前最可靠的研究判断是：继续扫描int8 scale、ground融合权重或old/new公共offset，预期收益很低。量化保真和压缩效率已有充分development证据，剩余主要问题是类特异的目标域margin不足，尤其是弱旧类被新类挤压与部分新类不可达同时存在。

下一路线应满足四个约束：

1. 先在Stage2-B建立不弱于当前强比较器的target-old几何，并提高最差旧类；
2. 用类身份无关的support可靠性或局部margin机制处理类特异碰撞，而不是全类共享平移；
3. target-old与target-new使用完全相同的量化与评分公式，新类只追加、旧状态可审计；
4. 只有support内部physical-rank代理通过联合门后才打开一次development query；通过后再做独立seed和完整确认。

可检验的核心假设是：ground v2中的域方向和半径不应直接决定类别分数，而应约束“每个target support样本对本类原型和局部边界的可信贡献”。这一机制必须同时降低old→new侵入和new不可达，并保持INT8/FP32决策等价；否则应停止该路线，而不是增加更多hard gate。

## 11. 结论

qKNN是KNN的量化部署版本，`q`表示quantized。它通过int8 support或int8原型降低状态和计算，但旧类域适应与新类注册的成败取决于目标域类几何，而不是位宽本身。

三类原型的正确关系是：ground旧类作为不可变弱先验，target-old承担域校正，target-new承担独立注册；旧、新target状态统一量化并在一个全注册类空间中竞争。新类注册影响旧类的根因是决策集合扩张和几何碰撞，即使旧状态完全不变也会发生。

目前已实现较完整的int8生命周期：ground v2为中心core＋rank-3域残差＋半径，target-old/new为逐向量scale的统一int8状态并支持append-only。D89证明可用14,399B总持久状态无损复现当前development基线预测，但尚未提高性能，ground组件也未完成联合封存。因此研究已从“int8能否压缩”进入“如何利用ground不确定度形成类特异、old/new共同受益的目标域margin”阶段。

## 本地证据索引

- 科学与数据协议：`E:\type10-7\项目.md`
- 当前研发目标：`docs/STAGE2_METHOD_RESEARCH_GOAL.md`
- 旧版support-level qKNN：`code/scripts/phase2_compressed_proto_knn_sweep.py`
- ground v2 codec：`code/cvsrffi/phase1_center_lowrank_prototype_bundle.py`
- target原型bank：`code/cvsrffi/stage2_target_prototype_bank.py`
- D36联合int8报告：`automation_reports/CV-SincNet/d36_compiled_joint_int8_20260718/report.md`
- D37保真int8报告：`automation_reports/CV-SincNet/d37_b3_preserving_int8_20260718/report.md`
- D85 ground radius v2报告：`automation_reports/CV-SincNet/d85_ground_radius_v2_20260720/report.md`
- D87 sigma margin报告：`automation_reports/CV-SincNet/d87_ground_radius_sigma_margin_20260720/report.md`
- D88 Pareto保护报告：`automation_reports/CV-SincNet/d88_ground_sigma_pareto_guard_20260720/report.md`
- D89 v2半径Cauchy报告：`automation_reports/CV-SincNet/d89_v2_radius_cauchy_center_20260720/report.md`
