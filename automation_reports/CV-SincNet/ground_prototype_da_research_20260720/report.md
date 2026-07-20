# 地面压缩原型驱动的单观测域适应研发报告

## 研发登记

- 研发ID：`ground_prototype_da_research_20260720`
- 日期：2026-07-20
- 操作者：Codex与并行研究子agent
- 状态：D93本地实现与协议/数值测试已完成，待N607窄矩阵验证；尚无性能结论
- 目标：研究如何把地面clean样本库压缩为与Phase1 checkpoint共同封存的聚合知识，并仅用一次固定LEO接收观测构成的target-old/target-new注册support学习可部署域适应，使性能尽可能逼近历史持有地面clean样本库的qKNN上界。
- matched基线：D62、D81、D92、identity-only single-qKNN、ProtoNet CDA、direct ADV3B02；历史qKNN V42与target多view路线仅作信息上界或失败教训，不作为协议合法基线。

## 现实观测模型与硬边界

合法问题固定为：

1. 地面端在Phase1阶段可以读取地面clean训练样本，但只能把不可更新、不可独立替换、与checkpoint共同封存的int8多样本聚合知识送入Phase2。
2. 卫星端每个target物理IQ只接收一次已经随机叠加某个允许`leo_*_weak`信道后的固定IQ；不能恢复clean IQ，不能对同一个物理IQ重新叠加clear/low-elevation/rain或构造其它信道view。
3. target-old与target-new的注册support可以使用注册清单给出的身份标签；普通query的发射身份、old/new角色和类别计数均不可见。
4. adapter、原型、校准和候选选择只读注册support；query只能在候选锁定后逐样本面对全部已注册类独立判决。
5. 历史qKNN V42若读取地面clean样本库，只能用于估计“未压缩地面知识”的性能上界；历史target多view若从同一clean物理IQ生成多种星地信道，只能用于分析多观测信息价值，不得进入当前候选或baseline。

因此，本轮明确排除：target clean访问、同物理IQ多次信道重放、target多view增强、source样本replay、query伪标签/统计、query角色Oracle、类别配额与全局重分配。

## 当前反证

- D92只在注册后重平衡旧类/新类support协方差；K10/new20旧类提高2.622pp、最低旧类提高4.600pp，但新类下降0.653pp，K1逐值不变。
- 这说明head侧协方差修补可以减轻一部分遗忘，却没有学习ground→LEO域变换，也没有利用地面压缩知识补足单样本统计。
- 下一候选必须显式回答：地面clean样本库中哪些判别信息被当前单中心/半径压缩丢失；目标旧类support如何提供配对的域偏移标定；同一共享变换如何服务没有地面配对的新类注册。

历史机制审计进一步给出以下边界。

|路线|同row证据或失败机制|对下一版的约束|
|---|---|---|
|D19地面中心直接融合|强融合使新类仅6.67%—22%，弱融合退回baseline|ground旧类身份不能直接覆盖target prototype或形成额外旧类logit|
|D65冻结旧行追加新行|旧类86.11%、遗忘6.11%，但新类59.33%|不能把旧类保护与新类注册拆成不共享坐标的两个head|
|D69跨阶段旧/新行拼接|新类相对D62下降10pp|不同阶段的score坐标不能直接交换|
|D77 ground预条件共同下降|15/15预测不变，11/15 fold退化为零更新|仅改变优化预条件不足以学习星地形变|
|D80/D82 ground协方差/残差收缩|出现old/new系统性交换或负迁移|ground统计必须先由target-old配对标定|
|D81/D84/D85/D89/D90中心族|共享平移趋于性能饱和；D85虽加载半径，15/15预测仍与D81相同|停止继续替换中心聚合公式|
|D87/D88 sigma-margin与Pareto guard|旧类收益伴随新类/floor下降；撤销新类退化时也撤销旧类收益|必须让new support进入共享适配目标，而不是事后guard|

## 待完成的并行证据

1. 当前地面int8组件的字段、shape、语义、资格与信息缺口。
2. 历史qKNN V42及target多view路线的输入、性能与非法信息增益来源。
3. 一手文献中只需密封源统计和一次性有标签target support的可迁移机制。
4. ground→LEO共享变换、旧类锚定、新类注册联合目标的可识别性、资源和量化设计。
5. 候选矩阵、support-held开发门、K1/K5/K10行为与冻结后的125验证计划。

## 文献筛选结论

常规UDA不适合直接移植，因为SHOT、CPGA、PCT、Tent、T3A和多数跨接收机RFFI域适应方法需要无标签target/query、query伪标签、test-time更新、source raw或batch类别结构。与本项目最接近的是source-free k-shot adaptation：IJCAI 2022的LCCS只用冻结源模型统计和少量有标签target support，在源统计与support统计张成的低维空间中优化归一化参数，且锁定后推理不再更新。

可合法借鉴的机制如下。

|来源|可借鉴机制|不能照搬的部分|本项目映射|
|---|---|---|---|
|[Few-Shot Adaptation of Pre-Trained Networks for Domain Shift，IJCAI 2022](https://www.ijcai.org/proceedings/2022/0232.pdf)|只优化低维统计组合系数；K1也可工作|原实现依赖网络BN结构|把可学习星地算子限制在ground域扰动子空间|
|[Prototype-Oriented Framework，NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/8edd72158ccd2a879f79cb2538568fdc-Abstract.html)|以原型代替source raw；双向约束防坍缩|原版读取全部无标签target并估计运输|只在注册support上做成对旧类锚定与全类防坍缩|
|[Simple CNAPS，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Bateni_Improved_Few-Shot_Visual_Classification_CVPR_2020_paper.html)|K依赖的类内/任务协方差解析收缩|元训练适应器不可直接获得|K1依赖密封ground协方差，K增大后逐步相信target统计|
|[CORAL，AAAI 2016](https://ojs.aaai.org/index.php/AAAI/article/view/10306)|用一、二阶统计进行白化—着色|运行时source样本不可用|只使用密封ground统计与target support解析对齐|
|[FeTrIL，WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Petit_FeTrIL_Feature_Translation_for_Exemplar-Free_Class-Incremental_Learning_WACV_2023_paper.html)|“中心+共享残差/平移”的几何分解|复制新类特征形成旧类伪样本|只估计解析传输，不生成任何伪feature或伪shot|
|[PASS，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.html)|均值之外保留聚合方差/半径|原型噪声采样与输入增强|Phase1密封低秩协方差，Phase2只解析评分|
|[TOPIC，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Tao_Few-Shot_Class-Incremental_Learning_CVPR_2020_paper.html)|保持旧类拓扑关系|原版增量图结构较重|保持ground旧类Gram或成对距离|

严格排除：target多信道view、同物理IQ重放、伪特征采样、query熵最小化、query伪标签、query图、target类别比例估计、OT全局分配、source raw replay以及receiver/TX/场景专用分支。

## 当前压缩包能表达什么

现有v2地面组件可重构：

\[
G\in\mathbb R^{14\times6\times160},\qquad
R\in\mathbb R^{14\times6}.
\]

它由`core_q[6,160]`、`residual_basis_q[6,3,160]`、`residual_coeff_q[13,6,3]`和`radius_q[14,6]`构成，可表达14个地面域中6个旧类的聚合中心、低秩跨域残差和p90余弦半径。它不包含单样本feature、样本ID、类内局部邻域或一般多峰密度。

组件单独存在时状态为`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`且`formal_phase2_eligible=false`；现有代码已实现`phase1_adv3b02_deployment_bundle.py`外层联合封存加载路径，验证外层签名、checkpoint绑定、runtime parity和内容根后可返回`formal_phase2_eligible=true`。但D85实际实验artifact仍是standalone pending组件。研发阶段按用户授权可继续使用其development证据；正式声明前只需把冻结候选绑定到实际外层bundle，不重复建设数据gate，也不触发`VALIDATED_ONCE`重验。

旧v1组件为`domain_class_q[26,6,160]`，共有84个有效domain×class cell，逻辑状态25,428B；v2把84个有效cell压成14个domain的中心+rank-3残差+p90半径，逻辑状态5,816B。v2相对v1主要是效率提升，并未增加类内局部邻域或多峰统计。

## target注册support与插入位置

实际K1/new20注册包只包含一次固定接收后的`support_leo_weak_iq[26,2,256]`、类索引、类内rank、opaque物理/overlay token、卫星seed与接收IQ哈希；不包含clean/raw/source feature或第二信道观测。同一固定IQ可计算`z160+FFT96+RF32`数学表征，但这些不是独立shot。

共享transport必须放在全局Stage2 fitter，而不是D81逐类component wrapper：

1. 一次性提取全部target-old/new support的288D表征；ground只覆盖其中`z160`，FFT96/RF32先保持identity。
2. 在Stage2-B旧类头拟合前，用全部6组ground-old↔target-old配对拟合共享transport。
3. Stage2-B记录变换前后旧类；Stage2-C读取target-new support后，保持transport冻结或只执行预登记的全类联合微调。
4. old/new最终prototype与统一affine head全部量化；new状态append-only，不改写ground组件。

## 历史qKNN V42与多view上界的拆解

V42高分不是“持有地面clean样本”单一因素的纯效果，而是多个额外信息源叠加：

|历史配置|Old|New|H|现行解释|
|---|---:|---:|---:|---|
|严格单观测轻量V42|75.12%|64.64%|68.56%|最接近单观测，但性能远低于目标|
|5-view+60epoch adapter，关闭角色/quota但保留dense query图|78.09%|73.83%|75.01%|仍使用同物理target多view与跨query图，不可部署|
|legacy完整体|84.07%|93.24%|88.23%|同时包含5-view、60epoch适配、角色Oracle和Hungarian类别配额，只能作违规上界|

旧source-teacher适配器还按sample key读取完整source feature与teacher feature，拟合并保存160×160全精度ridge映射；这既不是压缩聚合原型，也不是当前Phase2可访问状态。D4a审计已把D1/D3及旧V42三场景同源cache标为`PROTOCOL_INVALID_FOR_PHASE2_SINGLE_OBSERVATION`。

因此，需要逼近的是“未压缩地面分布知识带来的判别能力”，而不是复制V42整条流水线。可迁移信息只包括共同封存的int8类×地面域中心、半径、低秩共享漂移、量化方差/噪声和多样本聚合局部中心；不能迁移clean IQ、sample feature/logit/cache、可逆ID、独立ridge/MLP sidecar、多LEO view或dense query结构。

## 底层统计突破：K1仍有域适应监督

K1不能估计单类target协方差，但6个旧类仍提供6组同身份配对：

\[
\{(\bar g_c,t_{c1})\}_{c=1}^{6},\qquad
\bar g_c=\frac1{14}\sum_d g_{dc}.
\]

因此可估计共享平移和最多5秩的类间形变。K1真正不可识别的是类专属变换、target类内方差以及ground身份子空间外的新类形变；它不应像D81/D92那样直接回退identity。

## 主方法设计：配对地面—LEO共享传输度量

暂名`PGT-Metric`。方法不让ground原型直接给query产生类别分数，而是让ground知识学习一个全部target类共享的低维非正交度量。

### 地面域字典分解

旧类身份基：

\[
H_g=[\bar g_c-\bar g]_{c=1}^{6},\qquad
V_i=\operatorname{basis}(H_g),\quad r_i\le5.
\]

地面域扰动基：

\[
N_g=[g_{dc}-\bar g_c]_{d,c},\qquad
U_n=\operatorname{basis}(N_g),\quad r_n\le13.
\]

`V_i`描述旧身份可分方向，`U_n`描述接收机/域变化方向；二者均由84个聚合cell临时解析，不读取地面样本。

### 用旧类配对标定非正交算子

由合法target-old support计算稳健中心`\hat t_c`，拟合：

\[
B_\theta=I+U_nMV_i^\top+U_nSU_n^\top,
\]

其中`M`描述身份方向如何被LEO域扰动，`S`为对角或低秩对称缩放。配对目标为：

\[
L_{pair}=\frac1{C_o}\sum_c w_c\,
\rho\!\left(\|B_\theta\bar g_c+b-\hat t_c\|_2^2\right)
+\lambda\|B_\theta-I\|_F^2.
\]

`w_c`只由统一support不确定度公式产生，不按TX ID设权。全部target feature使用逆映射：

\[
\tilde z=B_\theta^{-1}(z-b).
\]

最终旧类与新类原型都来自变换后的target support：

\[
p_c=\operatorname{RobustMean}\{\tilde z_{ci}\},
\quad c\in Y_{old}\cup Y_{new}.
\]

query逐样本使用同一度量对全部注册类argmax；无query状态、角色路由或类别配额。ground只决定共享几何，不直接给旧类额外分数，从结构上降低D19/D78式“旧类保护压制新类”的风险。

### 为什么正交Procrustes只能作负对照

若同一个正交矩阵`Q`同时作用于query和所有target prototype：

\[
\|Qx-Qp_c\|_2=\|x-p_c\|_2,
\qquad\cos(Qx,Qp_c)=\cos(x,p_c).
\]

预测逐值不变。只有非正交`B`产生的共享Mahalanobis度量

\[
d_c^2(x)=(x-p_c)^\top B^{-\top}B^{-1}(x-p_c)
\]

才可能真正改变分类。正交对齐可以测量ground-target几何可配准程度，但不能冒充性能机制。

## 旧类锚定与新类注册联合训练

闭式配对初始化后，只在合法support上执行不超过20步的低维优化：

\[
L=\lambda_pL_{pair}
+\tfrac12L_{old}^{LOO}
+\tfrac12L_{new}^{LOO}
+\lambda_rL_{rel}
+\lambda_tL_{tail}
+\lambda_sL_{stable}.
\]

- `L_old^LOO/L_new^LOO`：leave-one-physical-sample-out全注册类prototype CE；旧类组与新类组各占0.5，防止20个新类在样本数上淹没6个旧类。预测分母始终包含全部注册类。
- `L_rel`：保持变换后target-old中心与ground旧类中心的Gram/成对距离关系。
- `L_tail=τlog(C^{-1}Σ_c exp(ℓ_c/τ))`：对全部类使用同一公式优化下尾，不建立难类名单。
- `L_stable`：限制`M,S`范数和`cond(B)`，避免6组配对对新类未见方向任意放大。

K1没有LOO正样本时，不生成数学view替代第二物理样本；使用6折leave-one-old-class-out配对误差选择解析收缩强度，并让新类单例只参与全类margin。K≥5后才启用物理样本LOO目标。

## 逼近qKNN V42所需的下一代地面压缩

单一均值或“均值+协方差”只有在单峰椭圆分布假设下才接近充分；样本级qKNN利用的是局部密度、多峰结构和尾部。若当前PGT-Metric证明ground-target配对有效但仍与V42上界差距大，下一代共同封存包应优先增加多聚合中心，而不是恢复clean样本库：

\[
G\in\mathbb Z_8^{14\times6\times M\times160},
\qquad M\in\{4,8\}.
\]

|聚合中心数|int8主体|表达能力|协议要求|
|---:|---:|---|---|
|1|13.4KB|单中心与跨域均值|当前bundle|
|4|53.8KB|粗粒度多峰/局部边界|每个中心必须聚合足够多物理样本|
|8|107.5KB|更接近局部qKNN密度|不得保存成员ID、样本索引或可逆归属|
|16|215KB|更细局部结构|几乎耗尽256KB，不优先|

`M=4/8`仍是多样本聚合知识，不是exemplar；不得出现一中心对应一物理样本、Phase2回源、可替换sidecar或伪样本采样。备选是每类/全局共享低秩协方差，用于K1解析Mahalanobis收缩，但它对多峰结构的逼近弱于多聚合中心。

## 候选递进与必要消融

|候选|唯一主要变化|地面知识如何生效|预期|停止风险|
|---|---|---|---|---|
|D93-PGTMetric|旧类配对学习低秩非正交`B`；final prototype全为target support|学习ground→LEO逆度量|K1首次非identity，旧/新共同受益|6旧类不足以覆盖新类方向|
|D94-PGTMetricCov|D93+经`B`标定的ground共享协方差先验|补足K1/K5类内统计|进一步降遗忘和下尾|可能重现D80 old/new交换|
|D95-PGTJoint|D94+全类LOO、关系蒸馏、下尾和稳定目标|新类support进入适配训练|改善floor与`H_old_new`|support代理与held query失配|
|负对照|仅共享平移或正交Procrustes|理论上target-target距离不变|验证实现和不变性|不应产生性能变化|
|风险消融|ground旧类prototype直接进入分数|强旧类先验|判断旧类锚是否必要|最易压制新类，不作主线|

每版至少保留：D92、`B=I`、平移、正交、非正交低秩、去掉新类LOO、未标定/已标定ground covariance、ground只作训练正则/直接评分、FP32/INT8 matched消融。

## 压缩与传输误差审计

若ground样本由聚合中心近似且`||u-g||≤ε_vq`，则余弦分数误差不超过`ε_vq`；对称int8每维量化步长为`s`时：

\[
\epsilon_q\le\frac{\sqrt p\,s}{2}.
\]

经映射后总误差受

\[
\epsilon_{total}\le
\|B\|_2(\epsilon_{vq}+\epsilon_q)+\epsilon_{fit}
\]

控制。实验必须记录`cond(B)`、旧类配对残差、leave-one-old-class-out误差、ground nuisance解释率、INT8/FP32 argmax与margin flip、top1-top2 margin相对误差上界的比例。

## 开发与完整125验证

先在预登记K10 development support-held代理上只比较D93的2个高信息量形态：`S=0`的身份—域低秩交互、以及对角`S`的域缩放；不做大范围rank/loss扫描。若旧类配对LOCO不优于identity，或support-held旧/新联合目标退化，直接停止该形态。

候选锁定代码SHA、外层bundle、秩规则、正则、优化步数和INT8导出后，必须运行同一完整125 screen：5receiver×5seed×`{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}`，每job内部覆盖三个固定且物理ID互斥的LEO弱场景。

每个matched row必须同时报告注册前/后旧类、新类、`H_old_new`、遗忘、旧/新最低类、逐类、全部混淆、receiver/seed/scenario分层、`cond(B)`、配对/LOCO误差、量化一致率、参数/MAC/平均与P95时延、峰值显存、状态量和前向次数。晋级不能只看旧类：K10/new20必须同时改善`H_old_new`、旧类floor和遗忘，且新类不能出现D78/D80/D92式系统性下降；K1必须不再逐值identity并在5个receiver上保持旧类适配增益非负。

## 版本状态

- Git工作树：`E:\type10-7\code\snapshots\ground_proto_da_rd_wt`
- 分支：`codex/ground-prototype-da-rd`
- 起始提交：`f65f8934`
- 根目录`E:\type10-7`不是Git仓库；完成后只把简要索引镜像到根目录报告面，版本化主报告保存在本工作树。

## D93实现与N607窄验证登记

### 本轮只允许的输入

D93不读取目标域或新类clean样本，不对同一物理IQ重新叠加星地信道，也不把同一received IQ的数学变换计作多个shot。Phase2唯一target输入是密封包内已经生成的一份固定`leo_*_weak received IQ`；其`z160/FFT96/RF32`来自同一次received IQ。地面端只读取多物理样本聚合后的int8 domain×class原型，不读取成员ID、样本feature、exemplar或clean IQ。最终target-old/target-new int8头只由target support产生，ground原型不直接生成query类别分数。

可机检字段固定为：`target_clean_iq_access=false`、`target_new_clean_iq_access=false`、`same_physical_iq_multi_channel_views=false`、`phase2_channel_simulator_calls=0`、`ground_aggregate_prototypes_only=true`、`query_rows_used_for_transport_fit=0`。

### 方法锁

- 候选：`d93_paired_ground_transport_interaction`。
- 主要差异：由6个旧类的ground聚合中心与合法target-old support中心拟合共享非正交低秩算子`B=I+U_n M V_i^T`，随后对target-old、target-new和每个query应用同一逆映射；D62头、target int8量化及全部query独立决策边界保持不变。
- 选择理由：本轮不通过真实query比较两个D93形态。先锁定参数更少、无额外nuisance尺度项的interaction-only形态；`interaction_plus_nuisance_scale`保留为未运行消融，不能根据本轮query结果回头选择。
- 固定超参数：`ridge_ratio=0.10`、`max_update_spectral_norm=0.50`、ground nuisance rank由地面残差participation ratio向上取整、`FFT96/RF32`保持D38/D62锁定的同received-IQ辅助块归一化与权重4。
- 资源：闭式求解、0 optimizer step、无query batch优化、无dense query图；新增持久状态仅为FP16低秩系数/尺度与平移，正式审计须低于256KB。
- 解释审计：逐类ground有效域数`D_eff`、stable rank、余弦距离`1e-4`近重复比例，以及target-old配对偏移在ground nuisance子空间中的覆盖率`rho`。这些值只解释正/负迁移，不参与本轮candidate、rank或阈值选择。

### 压缩格式边界

本轮不使用125或真实target误差选择地面压缩格式。下一代`Direct3-INT8`、`MeanINT8+TangentResidualINT4-G32`或带共享domain nuisance的`Redundancy-Aware G3-T4`，必须先用Phase1地面LODO与固定单次LEO弱伪目标锁定`M/bitwidth/rank/tau`，并确保每个原型聚合多个独立物理记录；禁止保存“每类3个单clean样本特征”。D93 target配对RMSE只用于D93算法诊断，不能进入压缩格式选择损失。

### 本地实现与验证

|文件|用途|
|---|---|
|`code/cvsrffi/stage2_d93_paired_ground_transport.py`|ground几何、配对transport、单样本变换、覆盖与冗余审计|
|`code/cvsrffi/stage2_d93_query_evaluation.py`|复用密封D81 I/O，接入D62 target-support-only int8头与D93 scorer|
|`code/scripts/run_cvs_somph_diag_row_pipeline.py`|D93候选行级分发|
|`code/scripts/run_d93_125_stability.py`|冻结候选后的完整125八分片调度|
|`tests/test_stage2_d93_paired_ground_transport.py`|K1非identity、拟合改善、资源与协议审计|
|`tests/test_stage2_d93_query_evaluation.py`|D42/D62 int8头集成和全注册类评分|
|`tests/test_run_cvs_somph_diag_row_pipeline.py`|行流水线CLI与分发回归|
|`code/tests/test_d93_125_stability_threads.py`|125候选锁和CPU线程上限|

本地`ssr-gpu`验证：`py_compile`通过；D93 core/query/row pipeline测试10/10通过。Pytest结束时Windows临时symlink清理产生已知`PermissionError`噪声，但进程退出码为0且测试结果为PASS；D42只读buffer警告不影响数值结果。

### 预登记窄矩阵

冻结同一candidate后只运行两个development诊断row，不做参数扫描：

|row|receiver|seed|K|seen-new|作用|query后动作|
|---|---|---:|---:|---:|---|---|
|D93-dev-K10-N20|`8-8`|713101|10|20|检验旧/新/H/floor/遗忘的主工作点|只判定该冻结方法是否有正信号，不回调超参数|
|D93-dev-K1-N20|`8-8`|713101|1|20|检验配对旧类监督能否避免D81/D92 K1逐值identity|只作K1机制诊断，不参与candidate选择|

每个row内部仍覆盖三个物理ID互斥的LEO弱场景。matched比较使用同row D62/D81/D92与direct ADV3B02；每行必须报告`old_acc_before_increment`、`old_acc_after_increment`、`seen_new_acc`、`H_old_new`、`average_forgetting`、`min_old_class_acc`、逐类/混淆、`D_eff`、`rho`、配对RMSE、INT8一致性与资源。若K10的H/floor/遗忘没有联合正信号，或K1仍逐值identity/出现任一receiver旧类负增益，则D93不进入125；进入下一机制版本。若满足预登记晋级条件，冻结Git SHA后直接运行同一完整125，不再调整公式。

### N607路径与待落地命令

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`。
- 计划隔离源码：`runs/d93_source_snapshot_20260720`。
- 计划输出根：`runs/d93_paired_ground_transport_dev_20260720`。
- 计划日志根：`logs/d93_paired_ground_transport_dev_20260720`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- checkpoint：`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- sealed runtime/method lock：`runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/`。
- cache：`runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix/rx_8_8/seed_713101/cache_set.json`。
- authority bundle：`runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1/authority_bundle_rx_8_8_seed_713101`。
- ground组件：`runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`，manifest SHA256=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`。

启动前必须重新执行直连preflight、检查每GPU训练进程数、确认上述输入存在并计算`COMMIT.json` SHA256。实际GPU、PID、完整展开命令、日志和输出路径在落地后回填；每个子进程CPU线程上限2，不超过每GPU两项训练实验。
