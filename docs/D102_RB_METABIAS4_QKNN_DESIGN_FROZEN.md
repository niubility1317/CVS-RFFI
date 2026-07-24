# D102-RB-MetaBias4-qKNN设计冻结

状态：`DESIGN_FROZEN / TARGET25_BLOCKED_PENDING_PHASE1_ASSET_AND_HELD_FALSIFIER`

日期：2026-07-24

## 1.不超过20行的可行性结论

1.唯一主要delta是Phase1元训练的rank-4类无关receiver MetaBias。
2.基础网络冻结；适配发生在`joint_proj.0`的ReLU前：`z(a)=Norm(ReLU(u+Ba))`。
3.`B∈R^(160×4)`、类无关domain encoder和domain meta bank在Phase1完成receiver-held训练。
4.Phase2只从当前row合法target-old/target-new support解析求解4维`a`。
5.Stage2-B和Stage2-C使用同一loss、同一rank、同一求解器和同一约束。
6.Stage2-C全部注册类逐类等权；不使用old/new角色权重或新类数量权重。
7.coverage只连续缩小precision，不触发hard gate、类别fallback或全坐标transport。
8.K1每类singleton贡献一个4维meta-observation，不估计类内协方差。
9.`Λ0≻0`保证唯一解；同时报告data information rank和先验占比。
10.非零`a`不是成功；必须证明ReLU mask、pairwise geometry、邻居贡献和净纠错变化。
11.统一重编码全部support后只构建一个全注册类typed INT8 Student-t qKNN。
12.query逐样本、只读、全类竞争；不参与fit、选择、更新或回退。
13.D62/D92只作历史或非门控诊断；D91不扩臂；BCRR不进入本revision。
14.Phase2预计0 trainable parameter、0 optimizer step、总state<80KiB。
15.support domain编码、bank matching、两次重编码和4维求解必须完整计MAC。
16.INT8要求top1 agreement≥99.5%、large-margin flip=0、无FP32持久sidecar。
17.先完成Phase1 receiver-held/class-LOCO/TX泄漏/标签置换证伪。
18.再完成真实checkpoint无queryheld falsifier和独立P0/P1复审。
19.门全部通过后才运行seed713102的5receiver×5slice=25个Target job。
20.Target25只声明`DA_COMPONENT_FALSIFIER_NON_PROMOTABLE`，不计算`I_syn`。

## 2.冻结输入与禁止输入

### 2.1 Phase1可生成并联合封存的资产

- 冻结ADV3B02 checkpoint和功能tap runtime；
- MetaBias basis`B`；
- class-free domain encoder`U∈R^(32×160)`；
- class-free domain bank`{g_m,t_m,Λ_m,σ_m}_{m=1..M}`；
- `T,Λ0,a_max,R`；
- 每个聚合项的物理样本数、class-balanced聚合回执、量化回执、receiver-held训练回执；
- method lock、checkpoint SHA、runtime SHA、bundle SHA和外部seal。

Phase1-C封存后，Phase2只能读取INT8/FP16聚合资产。原始source/clean IQ、单样本ground feature、ground标签、receiver/day名称、成员物理ID和训练episode不得进入Phase2。

### 2.2 Phase2可读取的row输入

- 当前row固定received-IQ产生的`pre_relu u`和`z_dom`；
- target-old/target-new support及其opaque class handle；
- query固定received-IQ，但仅在状态封存后逐样本推理；
- typed qKNN method lock、registry和资源上限。

禁止读取query truth/role、true batch class count、类别配额、全局重分配、clean/source数据、receiver/TX名称或跨row可变状态。

## 3.class-free domain表示与Phase1 bank

### 3.1 domain encoder

对冻结checkpoint同次前向得到的`z_dom∈R^160`，使用Phase1训练后封存的

`r(x)=Norm(U z_dom(x))∈R^32`。

Phase2不得直接用raw`z_dom`做bank匹配。`U`必须通过class-LOCO、TX泄漏和标签置换审计；`U`及其INT8量化状态与checkpoint共同封存。

### 3.2 聚合单元

每个bank项对应一个不向Phase2暴露名称的source receiver/day domain cell。对每个source class先在该cell内聚合至少2个互异物理样本，再对可用class等权平均并单位化得到`g_m`。每个bank项至少覆盖2个source class；缺失class不能由其他class的样本数补权。bank中不保存class ID、单样本向量或成员ID。

`t_m∈R^4`是该domain cell的MetaBias code；`Λ_m=diag(λ_m1...λ_m4)`且每个对角项为正；`σ_m>0`为该domain cell在class-free 32维表示中的Phase1内聚尺度。三者只在Phase1学习或估计，Target不得重估。

## 4.Phase1元训练

### 4.1 episode

每个episode留出一个source receiver作为pseudo-target；其support/query物理ID互斥。pseudo-target support按Stage2完全相同的第5节公式解析得到`a`，再对held query使用全类等权Student-t qKNN损失。留出receiver对应的bank项从该episode匹配bank中整体移除。

训练参数仅为`B,U,{t_m,Λ_m}`和正标量的无约束参数化；checkpoint其余参数冻结。`g_m`由多物理样本class-balanced聚合固定。`T,σ,Λ0,a_max,R`只由Phase1 receiver-held开发/确认划分一次性冻结，Target不得选择。

### 4.2 必须通过的Phase1证伪

- receiver-held：每个held receiver均有完整K1/K5/K10同式episode；
- class-LOCO：被留出class不参与`B/U/bank`梯度或聚合，但使用同式推理；
- 标签置换：一致置换class handle后state和预测等价；
- TX泄漏：`r(x)`的source-class线性probe balanced accuracy不得超过25%，且class-LOCO held H不得低于raw qKNN；
- 聚合：每个class-cell至少2个物理样本，support/query物理ID互斥；
- K1：相对M0的old、new、H净正确均非负，至少H或全类floor严格提高；
- K5/K10：old和new净正确均非负，wrong→correct多于correct→wrong；
- 量化：INT8 top1 agreement≥99.5%，large-margin flip=0。

上述门只用于Phase1方法冻结和target前证伪；Target row不能据其performance选择identity或更换超参数。

## 5.Stage2唯一4维解析推断

对support样本`i`：

`r_i=Norm(U z_dom_i)`，

`s_im=clip(r_i^T g_m,-1,1)`，

`π_im=softmax_m(s_im/T)`，

`q_i=Σ_m π_im exp(-(1-s_im)/σ_m^2)`，

`P_i=q_i Σ_m π_im Λ_m`，

`m_i=Σ_m π_im t_m`。

`T>0`、`σ_m>0`、`Λ_m`均来自Phase1封存。`q_i∈(0,1]`只连续缩小信息量，不产生阈值、accept mask或fallback。

对注册类集合`Y`，每类`c`有`K_c`个support：

`A_c=(1/K_c)Σ_(i:y_i=c) P_i`，

`b_c=(1/K_c)Σ_(i:y_i=c) P_i m_i`，

`A_data=(1/|Y|)Σ_c A_c`，

`b_data=(1/|Y|)Σ_c b_c`。

唯一无约束解为：

`a_tilde=(Λ0+A_data)^(-1)b_data`，

其中`Λ0=diag(λ01...λ04)≻0`由Phase1封存。

唯一确定性可行映射为：

1.逐坐标执行`a_box_j=clip(a_tilde_j,-a_max_j,+a_max_j)`；
2.若`a_box^TΛ0a_box≤R^2`，则`a=a_box`；
3.否则`a=R a_box/sqrt(a_box^TΛ0a_box)`。

该顺序固定，不作迭代投影、回溯、目标表现选择或逐row超参数更换。

Stage2-B令`Y=old registry`得到`a_B`；Stage2-C令`Y=all registered old+new classes`得到`a_C`。两者公式完全相同，每类一票；不使用old/new角色、新类数量或类别样本量权重。

每个state必须报告：

- `rank(A_data)`；
- `eigmin/eigmax(Λ0+A_data)`和condition number；
- `prior_fraction=tr(Λ0)/tr(Λ0+A_data)`；
- `||a_tilde||,||a||`及box/ellipsoid约束是否激活；
- `q_i`分布，但不得用这些字段触发target性能fallback。

## 6.表征和qKNN

冻结basis解码后：

`z_i(a)=Norm(ReLU(u_i+B a))`。

S_B用`a_B`统一重编码全部old support；S_C用`a_C`重新编码全部old/new support，不复用S_B旧bank前缀。每个state只构建一个typed INT8 Student-t qKNN。每类score继续使用`logsumexp-log(K_c)`，全注册类一次竞争。

query使用与对应state逐字节相同的`B,a`和ReLU/归一化。query不能更新`a`、bank、prototype、metric、阈值或registry。

## 7.真实检查点held falsifier

Target25前必须在真实checkpoint和无query-truth路径上同时发布M0/M_DA，检查：

- M0与现有typed qKNN基线逐字节一致；
- M_DA的ReLU activation mask、pairwise cosine/angle、neighbor contribution、margin和argmax确实变化；
- K1/K5/K10分别报告wrong→correct、correct→wrong、old/new净正确和全类floor；
- 重复query两次的state hash和prediction完全一致；
- prediction/COMMIT先封存，truth随后由独立scorer打开；
- `U/B/bank/a/qKNN`完整state分别计入，不共享抵扣；
- bank matching、S_B重编码和S_C再次统一重编码全部计MAC；
- INT8 top1 agreement≥99.5%、large-margin flip=0、无FP32持久sidecar；
- 每臂总state≤262,144B，post-backbone MAC/query≤262,144。

任一项失败均关闭D102实例；不得从held或Target结果调rank、T、σ、precision、trust、bank大小或投影顺序。

## 8.Target25

只有held falsifier通过且独立release复审`P0=0、P1=0`后才能运行：

`5 receivers×seed713102×{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}=25 jobs`。

每job覆盖3个互斥scene，共75个scene slice。因果臂仅为：

|臂|表示DA|分类头|
|---|---|---|
|M0|identity|统一typed INT8 Student-t qKNN|
|M_DA|D102 MetaBias4|同一qKNN公式|

闭合产物为100份before/after prediction和150条logical score。D62/D92只进入历史表或非门控诊断；D91不扩臂；BCRR不进入。状态固定为`DA_COMPONENT_FALSIFIER_NON_PROMOTABLE`，不计算`I_syn`，不声称完整主方法或目标完成。

Target性能门沿用活动目标：K10的A-old≥92%、Min-old≥85%、new5/new10/new20≥92%/90%/86%；K5/new20相对matched K10/new20四项衰减≤5pp；K1相对M0的H/A-old/Min-old至少一项严格正且old/new保护项不恶化。

## 9.与历史方法的因果区别

- 相对D62：连续precision和4维解析推断替代hard row gate及351/375注册后fallback。
- 相对D92：全注册类逐类等权替代old/new角色各0.5，不引入第二affine LDA头。
- 相对D93/D94：只推断4维bias code，不做ground→target全坐标transport。
- 相对SCXMAP：改变真实pre-ReLU activation mask和归一化邻域，不是post-feature标量cross-map。
- 相对GRB-JP4-CFM：只更新4维bias而非160×320权重残差，删除strict LOO Jacobian和D92联合state。
- 相对SVRN/BCRR：不重整全局方差，不使用已在375/375注册后slice中identity的OTHER。
