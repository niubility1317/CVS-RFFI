# qKNN量化记忆、目标域适应与新类注册研究报告

- 日期：2026-07-20
- 适用协议：`p2_min_v1`
- 研究对象：CVS-RFFI/CV-SincNet的Stage2-B旧类目标域适应与Stage2-C新类注册
- 证据等级：当前协议、代码实现、development support-only实验与理论分析的综合；不把未联合封存组件或诊断结果写成正式性能结论

## 摘要

本项目中的qKNN是quantized K-nearest neighbors，即量化K近邻；这里的`q`表示quantized，不表示query。它从普通KNN演化而来：KNN保存浮点support特征并按近邻投票，qKNN把support特征或类原型压缩为int8，在推理时通过解量化余弦相似度或int8点积完成分类。量化主要解决存储、带宽和部署计算问题，不直接解决目标域偏移、新旧类碰撞与灾难性遗忘。

Stage2的核心困难是同时满足两个目标：旧类需要从地面/source几何迁移到目标receiver的LEO弱信道几何，新类需要用少量目标域support完成追加注册。注册后，每个query必须在`Y_old∪Y_new`全部已注册类别中独立竞争。即使旧类模型和旧类原型逐bit不变，只要候选集合加入新类，旧类决策区域就会被重新切分，因此旧类准确率仍可能下降。

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
\hat y(x)=\arg\max_{c\in Y_{old}\cup Y_{new}}S_c(x).
\]

禁止使用query真值、真实old/new角色、batch真实类别数、类别quota、Hungarian匹配、optimal transport或跨query全局重排。

## 2. 从KNN到qKNN

### 2.1 普通KNN

给定support记忆`M={(z_i,y_i)}`与query特征`z_q`，普通KNN计算距离或相似度，选出K个最近support，再进行多数投票或距离加权投票：

\[
\mathcal N_K(z_q)=\operatorname{TopK}_{i}\;\cos(z_q,z_i),
\qquad
\hat y=\arg\max_c\sum_{i\in\mathcal N_K}\mathbf 1[y_i=c]w_i.
\]

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

为避免概念混淆，记：

- `g_{c,d}`：Phase1地面域`d`中旧类`c`的聚合原型；
- `t_c`：由目标receiver旧类support估计的target-old原型；
- `n_j`：由目标receiver新类support估计的target-new原型。

三者的职责不同。

|状态|来源|主要作用|是否可更新|是否有ground身份对应|
|---|---|---|---|---|
|ground旧类原型`g`|target访问前的Phase1多样本聚合|旧类身份先验、域漂移方向、半径或不确定度参考|Phase2只读|仅旧类有|
|target-old原型`t`|目标receiver的旧类K-shot support|校正receiver与LEO弱信道造成的域偏移|只能由合法support形成|有同类ground先验，但不能被其直接覆盖|
|target-new原型`n`|目标receiver的新类K-shot support|建立新身份决策区域|注册时append-only|没有ground同类原型|

正确的生命周期是：Stage2-B先形成并锁定旧类目标状态；Stage2-C只追加新类状态；ground bundle始终不变；注册后旧类持久字节最好保持append-only前缀不变。需要注意，旧前缀不变只能证明“没有改写旧状态”，不能证明“旧类预测不下降”，因为新增新类仍会参与竞争。

## 4. 当前int8如何压缩

### 4.1 旧版support-level qKNN

对每个归一化support向量使用统一尺度127：

\[
q_i=\operatorname{clip}(\operatorname{round}(127z_i),-127,127),
\qquad q_i\in\mathbb Z^{F}_{int8}.
\]

持久状态包括`quantized_matrix`、support标签、old/new标记，以及可选的FP类原型、半径和计数。由于该路径的类原型仍可能保留FP64/FP32，它是历史qKNN基线，不等同于当前“全部target原型int8”的部署目标。

### 4.2 ground旧类v2组件

设原始地面域×类中心bank为`P∈R^{D×C×160}`。当前v2实现不再持久化完整dense bank，而执行：

1. 在Phase1离线阶段用全局maximin规则选一个中心域；其`C×160`类中心形成`core`。
2. 对其余`D-1`个域相对core的残差，按类做rank-3 SVD。
3. 持久化`core_q[C,160]`、`basis_q[C,3,160]`和`coeff_q[D-1,C,3]`。
4. 持久化p90类内余弦距离半径`radius_q[D,C]`。
5. 中心、基、系数采用“最后一维每向量一个FP16 scale”的对称int8量化；半径采用“每类一个FP16 scale”的非负`[0,127]`量化。

向量量化公式为：

\[
a_v=\frac{\max_k|v_k|}{127},
\qquad q_k=\operatorname{clip}(\operatorname{round}(v_k/a_v),-127,127).
\]

部署时只临时重构：

\[
\hat g_{c,d}=\hat g^{core}_c+sum_{r=1}^{3}\hat\alpha_{d,c,r}\hat b_{c,r}.
\]

D85真实组件包含14个ground域×6个旧类cell，逻辑组件状态为5,816B；相对旧表示，ground组件状态下降77.13%。但该组件当前仍等待外部联合封存，不能作为已获正式Phase2资格的bundle。

### 4.3 target-old与target-new原型bank

当前统一target prototype bank对每个归一化类向量采用逐向量scale：

\[
a_c=\operatorname{FP16}\left(\frac{\max_k|p_{c,k}|}{127}\right),
\qquad
q_{c,k}=\operatorname{clip}(\operatorname{round}(p_{c,k}/a_c),-127,127).
\]

`-128`被明确禁止。每类同时保存FP16`radius`和support`count`。旧类bank先建立，新类注册时只计算新类后缀并拼接；旧类`q/scale/radius/count/class registry`由SHA256前缀约束保持不变。INT8评分为：

\[
S_c(x)=a_c\langle \operatorname{normalize}(x),q_c\rangle.
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

注册前，旧类query只需战胜其他旧类：

\[
\hat y_{before}(x)=\arg\max_{c\in Y_{old}}S_c(x).
\]

注册后，同一个query还必须战胜全部新类：

\[
\hat y_{after}(x)=\arg\max_{c\in Y_{old}\cup Y_{new}}S_c(x).
\]

对真实旧类`y`，注册前正确只要求

\[
S_y(x)>\max_{c\in Y_{old}\setminus\{y\}}S_c(x),
\]

注册后还必须满足

\[
S_y(x)>\max_{j\in Y_{new}}S_j(x).
\]

后一条件更严格。因此，即使encoder、旧类原型和旧类score逐bit不变，新增一列新类score也可能使旧query从正确旧类翻转为新类。

### 5.2 新类support具有目标域匹配优势

ground旧类原型来自Phase1地面/source receiver，target-new原型直接来自当前目标receiver。若旧类仍带有source域残差，而新类原型与query共享目标receiver、LEO场景和接收链特征，新类可能凭借“域相似”而非“身份相似”获得更高分，造成old→new侵入。这也是只靠地面旧类锚难以保护旧类的原因。

### 5.3 few-shot中心估计噪声

K较小时，新类样本均值的方差较大。一个偏向旧类区域的新类原型会错误扩大新类Voronoi区域。K=1时没有可靠类内半径，必须使用预锁定`r0`；把self-distance当成零方差会造成过度自信。

### 5.4 多新类带来的极值效应

即使每个新类单独侵入旧类的概率不高，新类数从2增加到5、10、20时，`max_{j∈Y_new}S_j(x)`上升的机会增加。旧类需要同时战胜更多随机竞争者，因此多新类下遗忘通常比单新类更严重。

### 5.5 联合适配造成表示或校准漂移

若Stage2-C继续更新adapter、温度、bias或prototype，旧类自身score也会变化。新类loss可能推动共享表征向新类support旋转，造成旧类内部混淆；old/new公共offset则可能只是在“旧类安全”和“新类可达”之间搬移错误。D36的ground anchor与连续margin校准未形成Pareto改善，正说明共享校准无法自动解决类级重叠。

### 5.6 量化误差会翻转小margin样本

量化本身通常不是主瓶颈，但当正确类与竞争类score margin很小时，原型重构误差可能改变top-1。设每类量化引起的score误差上界为`ε_c`，若

\[
S_y^{FP}(x)-\max_{c\ne y}S_c^{FP}(x)>
\epsilon_y+\max_{c\ne y}\epsilon_c,
\]

则量化后预测保持不变；不满足该条件的边界样本可能翻转。D37两级residual-int8量化误差极低，D89中INT8/FP32 outer预测与margin sign均零翻转，说明当前主要矛盾是类几何，不是8bit精度。

## 6. 如何同时适应旧类并注册新类

一个合理的统一方法应按以下顺序工作。

### 6.1 Stage2-B先形成可靠target-old状态

对每个旧类，用合法K-shot目标support估计稳健中心与半径：

\[
t_c=\operatorname{normalize}\left(\operatorname{RobustCenter}\{f_\theta(x_i):y_i=c\}\right).
\]

`f_θ`可以是冻结backbone加极轻adapter，但开发只能使用support内部physical-rank交叉拟合。首要目标是提高注册前旧类总体准确率和最差类floor；若Stage2-B本身不足，Stage2-C再精细的注册门也只是保护一个较弱旧头。

### 6.2 ground知识只作不确定度受控先验

ground旧类不应直接覆盖target-old。可使用不确定度权重融合：

\[
p_c^{old}=\operatorname{normalize}\left(
\alpha_c t_c+(1-\alpha_c)\tilde g_c
\right),
\]

其中`\tilde g_c`是经合法、类无关域校正后的ground先验，`α_c`随K增大、target半径减小和support可信度提高而增大。实际部署中ground权重应是弱先验；新类没有同类ground原型，必须保持纯target注册。

### 6.3 Stage2-C追加新类而不改写旧类持久状态

对每个新类独立形成：

\[
p_j^{new}=\operatorname{normalize}\left(
\operatorname{RobustCenter}\{f_\theta(x_i):y_i=j\}
\right).
\]

量化后只追加到registry后缀。注册前旧状态作为teacher/anchor，用于support侧蒸馏、参数位移约束和old-prefix审计。append-only保证可追踪性，但最终安全门必须直接检查held旧样本的old→new侵入。

### 6.4 用半径和margin约束真实碰撞

对任意两个注册类`i,j`，理想分离条件为：

\[
d_{cos}(p_i,p_j)>r_i+r_j+m+2\epsilon_q,
\]

其中`r_i,r_j`是support估计半径，`m`是部署安全margin，`ε_q`是量化几何误差上界。损失应同时覆盖：

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

设目标域真实类条件中心为`μ_c^t`。对旧类，理想域校正应满足：

\[
T_t(g_c)=t_c=\mu_c^t.
\]

若backbone已实现完全域不变，则`T_t`退化为恒等映射，ground旧类与target-old同类原型重合。对新类：

\[
n_j=\mu_j^t.
\]

同时所有不同类满足：

\[
d_{cos}(\mu_i^t,\mu_j^t)>r_i^t+r_j^t+m.
\]

这时ground提供正确旧类身份，target-old只需进行无偏域修正，target-new落在独立区域，新类注册不会切走旧类决策区域。

### 7.2 量化上限

理想int8压缩不是要求重构向量逐元素等于FP32，而是要求所有部署决策与FP32一致：

\[
\arg\max_c S_c^{INT8}(x)=\arg\max_c S_c^{FP32}(x)
\]

对全部合法query成立，且radius排序、margin符号和old-prefix均不变。D89的development cell已经达到INT8/FP32预测等价，但这只证明量化无损，不证明FP32几何本身正确。

### 7.3 不可突破的上限

如果两个TX在目标receiver和当前LEO弱场景下的特征分布物理重叠，或者K-shot support无法代表query分布，则真实Bayes误差大于零。此时即使原型等于真实均值、量化误差为零，也不能达到100%。单原型尤其无法表示多模态类分布；可在资源允许时使用少量子原型，但必须通过matched Pareto证明其收益。

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
