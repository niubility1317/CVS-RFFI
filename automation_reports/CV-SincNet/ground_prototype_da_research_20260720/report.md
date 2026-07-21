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
|D93-dev-K10-N20|`20-1`|713101|10|20|检验旧/新/H/floor/遗忘的主工作点|只判定该冻结方法是否有正信号，不回调超参数|
|D93-dev-K1-N20|`20-1`|713101|1|20|检验配对旧类监督能否避免D81/D92 K1逐值identity|只作K1机制诊断，不参与candidate选择|

每个row内部仍覆盖三个物理ID互斥的LEO弱场景。matched比较使用同row D62/D81/D92与direct ADV3B02；每行必须报告`old_acc_before_increment`、`old_acc_after_increment`、`seen_new_acc`、`H_old_new`、`average_forgetting`、`min_old_class_acc`、逐类/混淆、`D_eff`、`rho`、配对RMSE、INT8一致性与资源。若K10的H/floor/遗忘没有联合正信号，或K1仍逐值identity/出现任一receiver旧类负增益，则D93不进入125；进入下一机制版本。若满足预登记晋级条件，冻结Git SHA后直接运行同一完整125，不再调整公式。

### N607路径与待落地命令

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`。
- 计划隔离源码：`runs/d93_source_snapshot_20260720`。
- 计划输出根：`runs/d93_paired_ground_transport_dev_20260720`。
- 计划日志根：`logs/d93_paired_ground_transport_dev_20260720`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- checkpoint：`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- sealed runtime/method lock：`runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/`。
- cache：`runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix/rx_20_1/seed_713101/cache_set.json`。
- authority bundle：`runs/somph_stage2bc_leo_weak_cache_20260716/offline_controller/authority_bundle_rx_20_1_seed_713101`。`8-8/713101`在confirmation-only authority根中不存在，因此在访问query前改用已存在且与development seed匹配的`20-1/713101`，没有新建或重验数据。
- ground组件：`runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`，manifest SHA256=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`。

启动前必须重新执行直连preflight、检查每GPU训练进程数、确认上述输入存在并计算`COMMIT.json` SHA256。实际GPU、PID、完整展开命令、日志和输出路径在落地后回填；每个子进程CPU线程上限2，不超过每GPU两项训练实验。

### 远端同步与启动前现场

- D93实现提交：`0e89085b292a95bb03e0756a31492dd225ef6cbd`。
- 2026-07-20 16:04 CST直连preflight通过；8张RTX 3090均为10MiB、0%利用率，无GPU计算进程。
- D93隔离源码由`d92_source_snapshot_retry2_20260720`复制后只覆盖下表4个文件；未修改远端共享源码。

|远端隔离源码文件|SHA256|
|---|---|
|`cvsrffi/stage2_d93_paired_ground_transport.py`|`2ffdc866ee1f2c097b2cedd14982e53e9b9a1a4fe4f752f2516206b6eda4b8ec`|
|`cvsrffi/stage2_d93_query_evaluation.py`|`cc7f02bb65e06f905d13c3b4acbc1fe4b7a7c7d22af1d3a5f2ae5103042ecdb5`|
|`scripts/run_cvs_somph_diag_row_pipeline.py`|`9183c1ed86b7ae777d79de4736b443b01de11d7be2460a9f69e9f97c741b855c`|
|`scripts/run_d93_125_stability.py`|`0694bb9da7730f1ac37b192a00bffcc6a059871c44ac81247e54e638ff2355ff`|

远端4文件SHA逐项匹配且`py_compile`通过。开发authority `COMMIT.json` SHA256=`407a1dba5f666af0101ccd746a598d29f7183f884c5e4596bd54aee5199c7da3`。K10/K1输出目录在启动前均不存在。

两条展开命令除`CUDA_VISIBLE_DEVICES`、`--k-shot`、输出与日志路径外完全一致：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CVSRFFI_CPU_THREADS=2 CVSRFFI_CPU_INTEROP_THREADS=1 \
CUDA_VISIBLE_DEVICES=<0-for-K10|1-for-K1> \
PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d93_source_snapshot_20260720:/home/szu2070436088/2510044040/CV-SincNet \
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u \
/home/szu2070436088/2510044040/CV-SincNet/runs/d93_source_snapshot_20260720/scripts/run_cvs_somph_diag_row_pipeline.py \
--cache-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix/rx_20_1/seed_713101/cache_set.json \
--authority-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/somph_stage2bc_leo_weak_cache_20260716/offline_controller/authority_bundle_rx_20_1_seed_713101 \
--authority-commit-sha256 407a1dba5f666af0101ccd746a598d29f7183f884c5e4596bd54aee5199c7da3 \
--phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth \
--sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt \
--method-lock /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/method_lock.json \
--output-root /home/szu2070436088/2510044040/CV-SincNet/runs/d93_paired_ground_transport_dev_20260720/<k10_new20|k1_new20> \
--receiver 20-1 --seed 713101 --k-shot <10|1> --new-count 20 --device cuda:0 \
--candidate d93_paired_ground_transport_interaction \
--ground-component-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component \
--ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c
```

### 初始启动的技术失败与retry1

初始K10/K1各自产生21行、1845B完整日志，均在`build_somph_offline_row_pair()`验证authority时以`authority commit contract drift` fail closed。两个进程已退出、GPU回到10MiB；输出中只有空的预创建子目录，没有预测、评分、optimizer step或性能指标。根因是误用了旧`cvs.phase2.somph_leo_weak_authority_commit.v1`开发bundle，而当前D92/D93消费者精确要求v2；这不是数据协议或D93算法失败。

同一D18开发run已经存在与该cache共同生成的v2 `signed_authority_bundle`：`runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/signed_authority_bundle`，其`COMMIT.json` SHA256=`fdedd9cfdfbb5db9f8962ba529403042b7de7011570dff514e9a629a44695147`。retry1仅把两条命令的authority路径/SHA替换为该现有v2 bundle，并使用不可覆盖的新输出`retry1_k10_new20`、`retry1_k1_new20`和新日志；candidate、数据、support/query、K、超参数和代码均不变。

retry1成功越过authority与密封包构建，但K10/K1均在首次D93 fit、任何query预测之前以`D93 ground/target-old class binding drift`停止。根因是地面组件类注册表按字典序保存，target-old注册表保持项目顺序；类别集合一致，仅tuple顺序不同。修复按类句柄把ground聚合原型/掩码重排到target-old顺序，继续拒绝缺类、重复类或集合漂移；该修复类置换等变，不增加类别专用规则。更新后D93联合测试12/12通过，其中集成测试显式使用逆序ground registry并验证重排审计。retry2使用新源码哈希和不可覆盖的`retry2_*`输出；数据、方法公式和超参数仍不变。

retry2再次在首次fit前停止并揭示更精确的接口事实：target predictor内部类注册表是opaque class handle，而ground组件注册表是TX标签，二者不能直接按字符串比较。`offline_build_receipt.json`确认双方实际旧TX集合均为`14-10,14-7,20-15,20-19,6-15,8-20`，且顺序与opaque class index 0至5一一对应。最终修复由行流水线把合法注册support清单中的`old_tx_labels`传给D93，执行`registered_target_old_tx_label_order_to_opaque_class_index`绑定；仍精确拒绝缺类、重复类、长度或ground集合漂移，不读取query角色/真值。该修复后联合测试12/12通过。retry3继续使用不可覆盖新目录。

### retry3 K1完整性能与K10技术失败

retry3 K1/new20已完成全行和三个场景的不可变预测后评分，score SHA256=`05557a8fc06f6ce86713035343ebb02cc2f37525ee414a6010caeea9579a1f75`。K10在任何query预测前因变换后D62内部D43 block协方差非正定停止；其28行、3187B日志已完整读取，没有NaN、Inf、OOM或Killed。K10没有性能结果，不能与K1拼接，也不能把技术失败视为算法性能。

K1同一行联合指标如下，单位为百分比：

|candidate|receiver/seed|K/new|B-old|A-old|Min-old before→after|New|H|F|判定|
|---|---|---|---:|---:|---:|---:|---:|---:|---|
|D93 interaction|20-1/713101|1/20|55.556|33.333|20.000→8.333|28.167|30.533|22.222|真实负信号，不晋级125|

场景分层：

|场景|A-old|New|H|old→new|new→old|
|---|---:|---:|---:|---:|---:|
|clear|33.333|37.500|35.294|57.500|9.250|
|low-elev|30.833|19.250|23.702|50.833|11.750|
|rain|35.833|27.750|31.278|56.667|10.250|

旧类逐类结果：

|旧TX|before|after|遗忘pp|
|---|---:|---:|---:|
|14-10|55.000|38.333|16.667|
|14-7|20.000|8.333|11.667|
|20-15|85.000|46.667|38.333|
|20-19|38.333|15.000|23.333|
|6-15|50.000|16.667|33.333|
|8-20|85.000|75.000|10.000|

新类逐类准确率：

|新TX|Acc|新TX|Acc|新TX|Acc|新TX|Acc|
|---|---:|---|---:|---|---:|---|---:|
|1-16|23.333|1-18|23.333|1-8|35.000|10-10|18.333|
|11-19|23.333|13-14|8.333|14-11|23.333|16-19|13.333|
|18-10|11.667|18-8|51.667|19-13|23.333|19-6|50.000|
|19-8|48.333|19-9|18.333|2-16|16.667|2-5|41.667|
|20-12|18.333|3-8|21.667|4-10|35.000|8-3|58.333|

K1的三场景transport都不是identity：更新谱范数为`0.5000/0.5000/0.4909`，条件数为`1.741/1.711/1.702`；配对RMSE相对仅平移从`0.0580/0.0858/0.0636`降到`0.0522/0.0816/0.0591`。然而D42 support训练20epoch全部保持100% support accuracy，loss分别从`0.6516→0.0331`、`0.3287→0.00433`、`0.4876→0.0131`，held query却很差，说明主要缺陷是单shot support过拟合和ground子空间外推不足，而不是没有优化或没有使用ground。

地面84个有效cell对应每类14个名义域，但由ground-only审计得到的`D_eff`仅为`[2.307,3.814,4.302,2.543,2.139,4.207]`，stable rank为`[1.600,2.325,2.629,1.705,1.508,2.949]`；说明84个cell的独立域信息远少于名义数量。余弦距离`1e-4`下近重复pair比例虽仅0至2.20%，但更宽尺度上的低participation ratio已经表明强相关冗余。当前nuisance全局participation ratio=`13.640`、保留rank=14，可能相对每类有效域数过高，是K1外推不稳的重要机制线索。

资源审计：trainable parameters=2260，adaptation epochs/steps=20/20，状态峰值44,419B，ground逻辑状态25,428B，新增transport每query 6,080MAC，总score时延0.0120ms/query，峰值CUDA分配约20.42MiB；无dense query图、无query优化、每query一次backbone/FFT、target clean访问为false、同物理IQ多信道view为false、Phase2信道模拟调用为0。INT8与FP32 support argmax变化为0。

K10修复不调整transport。只在捕获精确`D43 structured covariance is not positive definite`时回退到原D42 auto-shrinkage等先验头；其它异常继续fail closed。该回退只读support、query行数为0，并新增coverage、`D_eff`、stable-rank和近重复审计。修复后本地联合测试13/13通过；K10必须以新输出完整重跑。

为避免用D81完整125的跨seed均值冒充matched单行，预登记并同时运行同一`20-1/713101`、同一cache/authority/query的D81 K1/new20与K10/new20；其候选固定为`d81_ground_nuisance_cauchy_center`，不调整任何D81参数。D81只用于隔离“低秩配对transport相对既有ground中心方法”的同row变化。输出分别为`matched_d81_k1_new20`和`matched_d81_k10_new20`，计划GPU3/GPU2；D93 retry4 K10使用GPU0。三条命令CPU线程上限均为2，使用各自不可覆盖输出和日志。

### D93最终窄矩阵matched结论

D93 retry4 K10、matched D81 K1与K10均完整结束，全部进程退出且8张GPU回到10MiB。三份stdout均已完整读取：D93 K10为3行（warning+最终JSON），D81 K1/K10各1行最终JSON；无NaN、Inf、OOM、Killed或未完成epoch。D93 K10 score SHA256=`48499ef1e6a88c34975d90eae3cf11b78596246ba6b46f84e35de3b883001843`，D81 K1/K10分别为`ae461eb3343ddd0ec0749eee7c7c725fd8053d49e327a64318e9b7a16821cab9`和`7b5ca193654aefec54a32b6757a24e0d4b974ba3130c0d4a313bea5343c3263e`。

|方法|K/new|B-old|A-old|Min-old after|New|H|F|
|---|---|---:|---:|---:|---:|---:|---:|
|D81 matched|1/20|61.667|37.500|13.333|27.583|31.786|24.167|
|D93 interaction|1/20|55.556|33.333|8.333|28.167|30.533|22.222|
|ΔD93−D81|1/20|-6.111|-4.167|-5.000|+0.583|-1.253|-1.944|
|D81 matched|10/20|87.222|69.722|48.333|68.917|69.317|17.500|
|D93 interaction|10/20|83.611|61.111|43.333|66.083|63.500|22.500|
|ΔD93−D81|10/20|-3.611|-8.611|-5.000|-2.833|-5.817|+5.000|

K10三个场景的D93−D81差值全部为负：

|场景|ΔA-old|ΔNew|ΔH|ground coverage ρ|out-of-ground ratio|
|---|---:|---:|---:|---:|---:|
|clear|-11.667|-0.750|-6.250|0.192|0.808|
|low-elev|-8.333|-3.500|-6.130|0.144|0.856|
|rain|-5.833|-4.250|-5.048|0.227|0.773|

K10旧类同row逐类：

|旧TX|D81 before/after|D93 before/after|Δafter|
|---|---:|---:|---:|
|14-10|81.667/73.333|86.667/66.667|-6.667|
|14-7|78.333/48.333|83.333/51.667|+3.333|
|20-15|85.000/71.667|80.000/75.000|+3.333|
|20-19|85.000/71.667|75.000/43.333|-28.333|
|6-15|95.000/58.333|90.000/48.333|-10.000|
|8-20|98.333/95.000|86.667/81.667|-13.333|

K10新类同row逐类：

|新TX|D81|D93|Δ|新TX|D81|D93|Δ|
|---|---:|---:|---:|---|---:|---:|---:|
|1-16|70.000|83.333|+13.333|1-18|55.000|45.000|-10.000|
|1-8|60.000|65.000|+5.000|10-10|68.333|51.667|-16.667|
|11-19|76.667|70.000|-6.667|13-14|38.333|36.667|-1.667|
|14-11|78.333|65.000|-13.333|16-19|50.000|53.333|+3.333|
|18-10|91.667|86.667|-5.000|18-8|93.333|81.667|-11.667|
|19-13|83.333|75.000|-8.333|19-6|91.667|85.000|-6.667|
|19-8|73.333|75.000|+1.667|19-9|83.333|80.000|-3.333|
|2-16|63.333|50.000|-13.333|2-5|63.333|56.667|-6.667|
|20-12|53.333|78.333|+25.000|3-8|40.000|33.333|-6.667|
|4-10|71.667|66.667|-5.000|8-3|73.333|83.333|+10.000|

D93 K10三场景support trace均完整20epoch：loss为`1.035→0.113`,`1.117→0.131`,`1.015→0.0936`，support accuracy从91.67%/91.67%/93.33%升至100%；训练优化正常但held query退化，排除“未收敛”解释。operator更新谱范数为0.387/0.481/0.476，条件数1.544/1.683/1.754；配对RMSE虽比仅平移分别降低0.00423/0.00416/0.00590，但`ρ`只有0.144至0.227，说明小幅support配对拟合改善来自对target主要偏移方向覆盖不足的ground子空间，不能泛化。

最终状态：`D93_COMPLETED_NARROW_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。按预登记停止条件不运行D93的125；完整125只留给下一版通过同row联合门的冻结候选。

## D94 coverage-controlled transport登记

- 研发失败点：D93在ground nuisance仅覆盖14.4%至22.7%target偏移时仍施加0.39至0.48谱范数更新，导致跨场景旧/新/H同时退化。
- 唯一主要变化：interaction-only的原始非正交更新`ΔB`先按原D93的0.5谱范数上限裁剪，再乘`α=ρ`；`ρ=||U_nU_n^Tδ||²/||δ||²`只由ground聚合基与target-old注册support配对偏移计算。共享均值平移仍保留，D62/D42头、target int8量化与query逐样本规则不变。
- 候选ID：`d94_paired_ground_transport_coverage_shrink`；schema=`cvs.phase2.d94.full_query_evaluation.v1`。
- 无阈值、无ρ扫描、无receiver/场景/类别分支；`α=ρ`在读取query前锁定。ρ=0时非正交项严格回到identity，ρ=1时等于D93，连续且类置换等变。
- 预期可观察结果：本row K10有效更新谱范数约从0.387/0.481/0.476降至0.074/0.069/0.108；相对D93恢复旧类、floor和H，同时保持非identity。K1也应避免满幅更新。
- 停止条件：K10相对matched D81若`H`不升、最低旧类不升或new下降超过1pp，则不进入125；K1任何old/floor进一步下降同样停止。不能因单个新类或单场景上涨晋级。
- 最小矩阵：同一合法`20-1/713101`的K10/new20与K1/new20各一次；复用已完成的matched D81，不重跑D81。每行三个物理ID互斥LEO弱场景。
- 本地验证：D94公式/协议/row/125 launcher联合14/14通过；coverage测试确认`0≤ρ≤1`且D94更新谱范数不大于D93。已知D42只读buffer warning和pytest临时symlink cleanup warning不影响exit code 0。
- Git提交：`9e93aca6ff0d559d527e6cbe025c6c0f4f3c7541`。

### D94完整窄验证结果

D94 K1/K10均完整结束，两个进程已退出，8张GPU均回到10MiB。两份stdout各3行并已完整读取，均为一个只读buffer warning和最终JSON；未发现NaN、Inf、OOM、Killed、缺失epoch或query前失败。K1/K10 score SHA256分别为`12dd33dabf201309dda4e76b3cd0e944d41f5d5a6b6270b8f116555866bf4c77`和`14c89a5d12ba4d6f9e0719cf6a62dffbd95848e9fd6819cff2c66175995d20df`。

|方法|K/new|B-old|A-old|Min-old after|New|H|F|
|---|---|---:|---:|---:|---:|---:|---:|
|D81 matched|1/20|61.667|37.500|13.333|27.583|31.786|24.167|
|D93 interaction|1/20|55.556|33.333|8.333|28.167|30.533|22.222|
|D94 coverage shrink|1/20|56.389|33.333|8.333|28.167|30.533|23.056|
|ΔD94−D81|1/20|-5.278|-4.167|-5.000|+0.583|-1.253|-1.111|
|D81 matched|10/20|87.222|69.722|48.333|68.917|69.317|17.500|
|D93 interaction|10/20|83.611|61.111|43.333|66.083|63.500|22.500|
|D94 coverage shrink|10/20|82.500|61.667|46.667|65.333|63.447|20.833|
|ΔD94−D81|10/20|-4.722|-8.056|-1.667|-3.583|-5.870|+3.333|
|ΔD94−D93|10/20|-1.111|+0.556|+3.334|-0.750|-0.053|-1.667|

K1的遗忘值比D81小1.111pp，原因是D94注册前旧类起点已经低5.278pp，而注册后旧类仍低4.167pp；这不是更好的保留能力。K10相对D93确实恢复0.556pp旧类和3.334pp最低类，并减少1.667pp遗忘，但新类下降0.750pp、H仍下降0.053pp；相对真正晋级基线D81则旧类、新类、H和遗忘全部更差。

K10逐场景同row结果：

|场景|D94 A-old|D94 New|D94 H|ΔA-old vs D81|ΔNew vs D81|ΔH vs D81|ρ|更新谱范数|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|69.167|73.500|71.268|-10.833|-1.500|-6.152|0.192|0.0744|
|low-elev|57.500|64.000|60.576|-8.333|-4.750|-6.684|0.144|0.0692|
|rain|58.333|58.500|58.417|-5.000|-4.500|-4.750|0.227|0.1084|

K1逐场景同row结果：

|场景|D94 A-old|D94 New|D94 H|ΔA-old vs D81|ΔNew vs D81|ΔH vs D81|ρ|更新谱范数|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|32.500|37.000|34.604|-8.333|+1.000|-3.660|0.204|0.1020|
|low-elev|30.000|20.000|24.000|-2.500|+0.500|-0.375|0.105|0.0523|
|rain|37.500|27.500|31.731|-1.667|+0.250|-0.409|0.136|0.0668|

K10旧类逐类结果：

|旧TX|D81 before/after|D94 before/after|Δafter|
|---|---:|---:|---:|
|14-10|81.667/73.333|88.333/65.000|-8.333|
|14-7|78.333/48.333|78.333/51.667|+3.333|
|20-15|85.000/71.667|80.000/75.000|+3.333|
|20-19|85.000/71.667|71.667/48.333|-23.333|
|6-15|95.000/58.333|88.333/46.667|-11.667|
|8-20|98.333/95.000|88.333/83.333|-11.667|

K10新类逐类准确率：

|新TX|D81|D94|Δ|新TX|D81|D94|Δ|
|---|---:|---:|---:|---|---:|---:|---:|
|1-16|70.000|81.667|+11.667|1-18|55.000|45.000|-10.000|
|1-8|60.000|65.000|+5.000|10-10|68.333|55.000|-13.333|
|11-19|76.667|68.333|-8.334|13-14|38.333|33.333|-5.000|
|14-11|78.333|65.000|-13.333|16-19|50.000|51.667|+1.667|
|18-10|91.667|81.667|-10.000|18-8|93.333|81.667|-11.666|
|19-13|83.333|73.333|-10.000|19-6|91.667|85.000|-6.667|
|19-8|73.333|75.000|+1.667|19-9|83.333|78.333|-5.000|
|2-16|63.333|48.333|-15.000|2-5|63.333|56.667|-6.666|
|20-12|53.333|81.667|+28.334|3-8|40.000|33.333|-6.667|
|4-10|71.667|63.333|-8.334|8-3|73.333|83.333|+10.000|

K10三场景support训练均完成20epoch，loss分别为`1.026→0.111`,`1.109→0.129`,`1.003→0.0906`，support accuracy分别由91.67%、91.67%、93.33%升至100%。coverage shrink把更新谱范数压到0.069至0.108，配对RMSE仅比纯平移改善0.0010至0.0021；它消除了D93的过强变换，却没有恢复D81的判别结构。K1同样完整20epoch且始终100% support accuracy，loss分别为`0.651→0.0334`,`0.331→0.00437`,`0.485→0.0131`，仍表现为support拟合充分而query泛化不足。

资源审计：trainable parameters=2260，adaptation epochs/steps=20/20，K10状态峰值44,414B，ground逻辑状态25,428B，新增transport每query 6,080MAC，score矩阵时延0.01369ms/query，峰值CUDA分配约20.95MiB；K1对应44,419B和0.01318ms/query。无dense query图、无query优化或角色/配额访问、每物理IQ仅一个接收观测、target clean/source访问均为false、Phase2信道模拟调用为0。K10 INT8与FP32 support argmax变化1次，K1为0次。

最终状态：`D94_COMPLETED_NARROW_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D94不运行125。失败机制不是没有使用地面原型，而是低覆盖ground transport替代了D81已有的收缩判别结构；下一版D95保留完整D81为base，仅把coverage-controlled非正交项作为小残差应用于其输入空间。

## D95 D81-base coverage residual预登记

- 候选ID：`d95_d81_coverage_residual`；schema=`cvs.phase2.d95.full_query_evaluation.v1`。
- 核心假设：D81已经提供当前最可靠的ground-spectrum收缩中心与D62判别头；D93/D94失败来自整体替换该结构。D95先对同一固定LEO弱support/query应用D94的coverage-controlled小残差，再在变换后的support上完整执行D81，不删除D81的ground nuisance basis、Cauchy中心平移、full/block/held闭包或正式INT8线性头。
- 地面知识职责分离：D94残差只表达ground-old与target-old support之间、且落在ground nuisance span内的共享偏移；D81只把不可变ground聚合域谱作为target support判别的收缩先验。ground均不直接给旧类query加分，最终old/new权重全部由合法target support编译。
- 无新增可调量：沿用D94唯一公式`α=ρ`、既有0.5谱范数安全上限和冻结D81；无ρ阈值、残差权重扫描、receiver/场景/类别分支或125反向选择。
- 协议边界：只读sealed Phase1 checkpoint、现有84个INT8多样本聚合ground cell与当前row的单LEO弱support；不读clean/source、query真值/角色/配额，不做query拟合、同物理IQ多信道view、Phase2信道模拟或全局重分配。
- 主要风险：对support/query共同施加小非正交变换后，D81的中心平移与D62协方差可能近似吸收该变化，产生零增益；也可能改变D81稳健权重并退化。实现必须审计D81在before/after全部full/block/held fit中实际执行，不能把D95误跑成D94。
- 最小矩阵：同一`20-1/713101`、K10/new20与K1/new20各一次，复用既有matched D81。K10相对D81要求`H`和最低旧类均提高且New下降不超过1pp；K1不得降低old/floor。任一失败则不进入125。
- 资源预期：在D94的2260参数、20step、约44KB target状态、6,080额外MAC/query基础上恢复D81 ground transform；总状态仍远低于256KB，optimizer step不超过20，query仍单次逐样本线性评分。
- 本地变更范围：`stage2_d93_query_evaluation.py`新增D95候选、D81原始loader/builder装配与闭包审计；row pipeline和125 launcher通过共享候选表自动获得该ID；测试新增候选与CLI路由断言。数据构建、authority、checkpoint、method lock和ground artifact均不变。
- 本地验证：在`ssr-gpu`环境串行运行D93/D94/D95公式、query包装、row CLI和125 launcher联合测试，15/15通过。唯一PyTorch只读buffer warning为既有D42路径且测试内部立即clone；pytest退出后的临时symlink清理PermissionError发生在exit code 0之后，不影响测试结论。

### D95远端同步与窄实验命令

- Git提交：`63bbc652`；本地与远端`cvsrffi/stage2_d93_query_evaluation.py` SHA256均为`cf939b772bf6b2c206bad61cbf3ac31bc45b264f8980bd431401f2a6bbfcc92a`，远端`py_compile`通过。
- 2026-07-20 17:02 CST直连preflight通过；启动前8张RTX 3090均为10MiB、0%利用率，未发现D93/D94/D95行进程；`/home`可用7.5TB。同步只覆盖隔离源码快照中的上述单文件，共享源码、数据、checkpoint和既有输出均未修改。
- 工作目录：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；K10使用GPU0，K1使用GPU1；每个子进程CPU线程上限2。
- K10完整子进程命令：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CVSRFFI_CPU_THREADS=2 CVSRFFI_CPU_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d93_source_snapshot_20260720:/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/runs/d93_source_snapshot_20260720/scripts/run_cvs_somph_diag_row_pipeline.py --cache-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix/rx_20_1/seed_713101/cache_set.json --authority-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/signed_authority_bundle --authority-commit-sha256 fdedd9cfdfbb5db9f8962ba529403042b7de7011570dff514e9a629a44695147 --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --method-lock /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/method_lock.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/d93_paired_ground_transport_dev_20260720/d95_k10_new20 --receiver 20-1 --seed 713101 --k-shot 10 --new-count 20 --device cuda:0 --candidate d95_d81_coverage_residual --ground-component-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c
```

- K1命令仅将`CUDA_VISIBLE_DEVICES=0`改为`1`、`--k-shot 10`改为`1`、输出改为`d95_k1_new20`；其余字节级参数相同。日志分别固定为`logs/d93_paired_ground_transport_dev_20260720/d95_k10_new20.stdout.log`和`d95_k1_new20.stdout.log`，启动器使用`start_new_session=True`，SSH仅负责短时落地并立即断开。
- 预期输出：每行`pipeline_receipt.json`、before/after immutable prediction、fit/resource audit、scorer `diag_cosine_score.json`和COMMIT；成功标准与上一节完全一致。任何D81-base闭包失败、协方差数值失败或联合指标不达门均停止，不用125补证。
- 17:09 CST两条子进程已落地：K10 PID=`1173317`/GPU0，K1 PID=`1173318`/GPU1。落地SSH正常退出；后续PID/GPU核验、短轮询、完整日志与artifact回收已移交实验子agent`repo_protocol_audit`，主线不重复启动或并行监控同一run ID。

## 实验发布双轨工作流复盘

前序低效主要来自主线同时承担方法设计、文件同步、远端启动、轮询、日志回收和指标解析，导致研发在等待GPU时停顿；另外，初始D93曾因authority版本、类顺序与opaque handle接口连续产生技术失败，说明发布交接缺少机器可核对的输入合同。自D95起固定为以下五段：

|阶段|唯一责任方|最小输入|完成证据|禁止事项|
|---|---|---|---|---|
|1.方法研发|主线|目标、协议、历史负证据|公式、唯一主要变化、停止门|target/query反向选参|
|2.本地冻结|主线|候选代码与测试|Git commit、diff、测试、文件SHA、预登记报告|未版本化即同步|
|3.实验发布|单一runner子agent|冻结交接包|preflight、远端SHA/compile、PID/GPU、不可覆盖路径|主线重复启动、runner改方法|
|4.证据回收|同一runner子agent|run ID/PID/预期artifact|完整stdout、receipt、fit/resource、score、SHA、异常扫描|只读tail冒充完整日志、失败后擅自重启|
|5.分析决策|主线|matched artifacts与runner handoff|逐row/场景/类/资源表、缺陷、晋级或停止|用单项最大值或跨row拼接晋级|

每个run ID使用状态机`LOCAL_VERIFIED→LANDED→RUNNING→ARTIFACTS_COMPLETE→ANALYZED`；只允许runner子agent改变服务器侧前三个状态，主线只在完整证据后写`ANALYZED`。交接包固定包含Git提交、文件SHA、候选/矩阵、完整子进程命令、Conda/Python/CWD、输入路径与SHA、GPU、日志/输出、预期artifact、停止门和重试权限。由此可以并行推进“服务器执行上一版”和“主线研发下一版/复盘”，同时避免双重启动与口径漂移。

该协作规则已写入根目录`AGENTS.md`；根目录不是Git仓库，因此同步镜像到本Git工作树`AGENTS.md`并随本报告提交。规则仅改变工作流，不改变`项目.md`的数据协议或科学场景。

## D93—D95三轮正式技术复盘

复盘前已重新完整读取active goal与2026-07-20版`项目.md`，刷新项目conversation index至1001条并检索`D81/D92/D93/D94/D95/ground prototype/coverage/qKNN/SRDA/forgetting`。主线再次完整读取本报告、D81完整125、D92完整125、D95 runner handoff，以及本目录全部9份D81/D93/D94/D95 stdout；D95 K10的32行Traceback与D93首次K10的失败栈均确认是同一D43非正定族，D93 retry4采用精确D42回退后才形成可比较结果，D95没有回退且未产生query预测。

### 同row与完整125因果表

|方法/证据面|K/new|B-old|A-old|Min-old|New|H|F|相对可信基线的结论|
|---|---|---:|---:|---:|---:|---:|---:|---|
|D81完整125|10/20|86.111|68.711|38.067|68.803|68.591|17.400|原合法轻量ground基线；绝对门全失败|
|D92完整125|10/20|86.111|71.333|42.667|68.150|69.555|14.778|当前完整125联合最强：old/floor/H改善，但New下降0.653pp|
|D81 matched dev|10/20|87.222|69.722|48.333|68.917|69.317|17.500|D93—D95唯一同row比较基线|
|D93 full interaction|10/20|83.611|61.111|43.333|66.083|63.500|22.500|相对D81：A−8.611、New−2.833、H−5.817、F+5.000pp|
|D94 coverage shrink|10/20|82.500|61.667|46.667|65.333|63.447|20.833|相对D93只恢复A0.556/floor3.334，仍相对D81全面为负|
|D81 matched dev|1/20|61.667|37.500|13.333|27.583|31.786|24.167|K1同row基线|
|D93 interaction|1/20|55.556|33.333|8.333|28.167|30.533|22.222|遗忘下降是假象：before先低6.111pp|
|D94 coverage shrink|1/20|56.389|33.333|8.333|28.167|30.533|23.056|A/floor/H仍负；New仅+0.583pp|
|D95 D81-base residual|1/20|56.389|33.333|8.333|28.167|30.533|23.056|与D94性能相同；D81 K1分支identity，无法恢复base|
|D95 D81-base residual|10/20|—|—|—|—|—|—|D43在query前失败，无性能结果|

当前最强的完整125方法仍是D92，而不是D93—D95；但D92距离K10/new20目标仍差A-old20.667pp、Min-old45.333pp、New17.850pp，且K1逐值无作用，因此也只是`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D93—D95只属于单receiver/seed development窄诊断，且ground组件`formal_launch_authority=false`，不得替代完整125或正式确认结论。

### 三轮共同成功经验与已证伪路线

|结论|证据|下一轮约束|
|---|---|---|
|显式读取地面原型并不等于有效适应|D93/D94/D95均真实读84个cell、非identity、资源与协议闭包通过，但held query退化|每个ground机制必须证明任务margin/纠错收益，不能以重构、RMSE或非identity晋级|
|域覆盖是首要可辨识瓶颈|D93/D94 K10 `rho=0.144—0.227`；D95 K1 `rho=0.105—0.204`，77.3%—89.5%偏移在ground span外|coverage低时ground只能进入弱协方差/可靠度先验，不能搬动整个坐标系|
|84个名义cell高度冗余|每类`D_eff=2.139—4.302`、stable rank1.508—2.949，而全局保留rank14|下一代bundle按冗余密度加权与自适应rank；当前bundle不得强行解释14个独立域|
|support训练目标不是held泛化证据|D93/D94的20epoch最终support accuracy均100%，loss充分下降，query却全面负迁移|下一候选优先闭式/解析头；support-CV只能决定可靠度，不能当性能晋级|
|共同非正交变换会破坏可靠base|D93替换D81后退化；D95叠加小残差后K1仍复现D94，K10触发数值失败|停止全坐标重写；保留D81/D92分支，用并行证据头或只作用协方差先验的残差|
|不能以较低before换取“较低遗忘”|D93—D95 K1的F少1.1—1.9pp，但before低5.3—6.1pp且after低4.2pp|反遗忘门同时锁定B、A、floor、New和H；F只作同row联合指标|
|注册竞争是真实可修复但有交换代价|D92 K10/new20稳定提高A2.622/floor4.600/H0.964并降低F2.622pp，New下降0.653pp|保留task-balanced covariance正信号，但新类必须有独立局部证据或统一校准防止被old保护挤压|

已否决第四轮继续尝试：更强ground→target全矩阵/更高rank transport、按receiver/场景调`rho`阈值、以D43回退后再把D95视为同一候选、ground旧类直接logit/原型融合、同物理IQ多信道view、query伪标签/图/OT/quota，以及只通过support loss选择残差强度。D95 K10的数值修复只能作为实现消融，不能在K1已违反性能门后获得重试晋级资格。

### 协议、Stage2-B/C与资源复核

- D93—D95均使用固定单LEO弱received IQ，support/query view count=1；FFT96/RF32只来自同一接收IQ数学表征；clean/source、query truth/role/count/quota、query fit、dense query graph和Phase2信道模拟均为0/false。
- final target-old/new均为INT8统一头，无FP32 sidecar；ground只读且不更新，不直接给旧类query打分。D93—D95科学访问面未发现越界，但当前ground artifact缺formal launch authority，所以性能只能写development diagnostic。
- 三轮均同时报告注册前/后旧类、New、H、F、全部旧/新逐类与场景；没有把单一old改善、New改善或边际最大值作为晋级。
- 参数2260、20epoch/20step、峰值状态约44KB、额外6,080MAC/query、无query图均满足资源门；性能而非资源是停止原因。

### 下一轮决策边界

在并行研发线交叉审查完成前不实现D96。候选必须保留当前最强D81/D92几何为base，不再统一搬动support/query；优先比较两个正交方向：①coverage-gated shrinkage RDA，仅让ground共享域谱进入类无关协方差先验，所有old/new均值来自target support；②合法single-view qKNN局部头与SRDA全局头并行，使用Phase1预锁/K≥2 support-CV可靠度做整row全局融合。若两头没有互补rescue或K1只能依赖未获批ground统计，则在本轮直接否决，转向新的Phase1 redundancy-aware/domain-factorized checkpoint与共同封存bundle，不用target125反向选择格式。

并行协作规则也已同步写入根目录与Git镜像`AGENTS.md`：综合方法轮默认设置域适应、分类头、反遗忘监督三条研究线，前两线必须向监督线提交中间方案；监督线检查协议、K-shot可识别性、共同变换不变性、support过拟合、old/new联合门、类置换、资源和matched证据。设计阶段只读，主线裁决后才分配不重叠代码面；实验runner与方法作者分离。

## D96/D97并行核心研发与交叉监督

本轮不再由主线串行设计全部候选。域适应代理独立实现D96，分类头代理独立实现D97；两名作者互审对方文件，反遗忘监督代理同时做第三方准入，主线只负责协议裁决、阻断修复、测试与Git集成。四个文件的职责互不重叠：

|候选|文件|主要机制|ground职责|target职责|K1行为|
|---|---|---|---|---|---|
|D96 RA-CGSRDA|`code/cvsrffi/stage2_d96_ra_cgsrda.py`|冗余密度反权、`D_eff`自适应rank≤4、coverage-gated共享SRDA精度、D81残差融合|只提供类无关低秩nuisance covariance与coverage基准，不直接给旧类logit|所有old/new均值只由当前row注册support构造，同一公式评分|融合权重强制0，精确回退base|
|D97 QK-D81-LGF|`code/cvsrffi/stage2_qk_d81_lgf.py`|三block归一化、逐support/逐block INT8 qKNN、按`K_c`归一化log-mean-exp、row-global概率融合|不读取ground分类原型|old/new均保存同式target support INT8 bank，query逐样本全类argmax|默认eta=0；仅允许Phase1锁定K1 prior|

首次交叉审查发现并阻断了以下实质问题：D96把实际FP32状态误报为INT8且低报约7.73倍、build/fit/fusion配置未绑定、密度权重未回算class mean、融合温度公式不标准、base/SRDA列顺序未绑定；D97未强制均衡K-shot、非零eta只靠字符串自述、base/qKNN列顺序未绑定、资源统计与INT8 margin审计不完整。主线已全部按fail-closed方向修复：

- D96要求64位Phase1 receipt并计算逐字段config digest，ground build→fit→fusion必须同一digest；`max_rank`必须是真正整数且≤4；非零support-CV可靠度必须绑定64位receipt；融合改为`(1-w)base/T_base+w·aux/T_aux`并显式核对注册类顺序。
- D96状态审计改为真实`coefficient_fp32+intercept_fp32+target_means_fp32`字节，不再宣称未实现的INT8；ground class mean与残差在density权重后重算。
- D97要求均衡K-shot；配置绑定Phase1 receipt和margin-audit SHA；非零eta必须绑定support-CV receipt；base classes必须与bank registry精确相同。
- D97资源面计入quantization audit序列化字节并区分点积MAC与归一化、解码、LSE、softmax等标量操作上界；新增Phase1-only FP32/INT8 logit误差、top1一致率、teacher-margin sign flip审计。

本地验证使用`ssr-gpu`串行执行：

```text
conda run -n ssr-gpu python -m pytest tests/test_stage2_d96_ra_cgsrda.py tests/test_stage2_qk_d81_lgf.py
15 passed in 0.79s
```

测试覆盖partial-domain硬失败、密度/rank闭包、all-class target-only均值、Woodbury/PSD、config漂移、整数rank、receipt绑定、K1/w0对象级精确回退、三block单位化、support顺序不变、`1/K_c`消除数量奖励、均衡K、query逐样本batch等价、类顺序闭包、持久状态和INT8 margin审计。`git diff --check`与两个核心`py_compile`均通过。当前只完成纯核心和最小验证，尚未接D81 pipeline、尚未生成真实Phase1锁、尚未访问target或N607。

### Phase1 LODO输入勘察

当前84-cell组件为`phase1_int8_domain_class_centroids_v1`，本地路径`E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component`；manifest SHA=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`，NPZ SHA=`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`。NPZ只含`[26,6,160]`INT8中心、FP16 scale、mask、domain/class registry与feature schema；实际14个完整domain×6类=84个cell。

|锁定项|仅当前84-cell是否足够|裁决|
|---|---|---|
|D96 `tau/max_rank`|是|可做14折leave-one-domain-out，按held残差重构、margin flip、`D_eff`和basis稳定性选择|
|D96 `ridge/temp_base/temp_aux`|否|缺物理样本级类内残差、完整288D D81 logits与held support/query|
|D97 `beta/temp/eta/K1 prior`|否|缺合法Phase1 sample-level 288D feature、独立source-validation query与D81 logits，不能用target补选|

此外该组件仍标记`formal_phase2_eligible=false`和`UNVERIFIED_UNDER_CURRENT_PROTOCOL`；所以84-cell几何LODO最多产生development partial lock，不能直接授权target/N607。完整锁必须由Phase1地面独立物理样本生成receiver/day LODO预测面，并把规范JSON receipt、margin audit及其SHA与配置闭合。下一步先定位或导出该Phase1-only预测面；在此之前不接target pipeline、不发布窄实验、更不会跑125。

## 并行研发—监督—实验发布工作流v2

从本轮起不再采用“主agent设计→实现→实验→分析”的单链串行方式。并行单元按证据职责拆分，任何一条方法线都不能自行宣布晋级：

|并行单元|当前对象|独立交付|必须接受的监督|停止/交接点|
|---|---|---|---|---|
|域适应线|D96冗余感知ground nuisance/SRDA、Phase1单观测exporter|数学机制、纯核心、geometry/coverage证据、exporter与测试|分类头作者做接口互审；反遗忘监督检查低coverage负迁移、old/new对称和协议闭包|本地验证与审查通过后交主线集成；不自行启动N607|
|分类头线|D97 D81+INT8 qK局部—全局头、receiver-LODO锁|部署等价量化、support-only融合、真实D81 episode scorer闭包|域适应作者检查特征几何；反遗忘监督检查融合是否损害floor/forgetting|形成完整Phase1 lock receipt后交实验runner|
|反遗忘线|D98 STRIMS连续尾部风险收缩|class-symmetric support-OOF可靠度、CVaR floor/侵入/retention目标、K1精确回退|同时监督D96/D97；主线检查与D97是否重复融合|纯核心通过后只作为独立matched列，不偷换D97结果|
|主线|协议裁决、接口集成、Git与同row晋级|阻断越界、合并代码、全套本地测试、版本提交、结果联合判决|不得绕过独立监督结论|释放一个不可变Git提交和runner handoff包|
|实验runner|N607 Phase1 export/LODO、target窄实验、通过后125|preflight、同步SHA、命令/PID/GPU/log/artifact、完整指标|只执行已预登记提交；不改算法/参数；主线不重复启动|实验落地后继续监控和回收；研发线同时进入下一候选|

协作节拍固定为：作者A提交中间设计给作者B和监督线→作者B检查部署接口与可识别性→监督线按协议、old/new联合门、floor、forgetting和资源给出P0/P1→主线只合并无P0版本→runner在服务器运行上一版时三条研发线并行研究下一版。远端执行与本地研发因此重叠，但同一run ID只有一个runner拥有启动权；完整125始终是晋级后的确认，不是选参工具。

## D96地面几何LODO诊断结果

主线修复了独立监督指出的raw/normalized中心不一致：basis训练与held residual现在都相对density-weighted raw class center计算，只有重构后的余弦分类中心做归一化。专项测试`6 passed`，`git diff --check`通过；Windows pytest退出阶段仍出现既有临时symlink `WinError 5`噪声，但主体exit code为0。

|候选|tau分位数|实际tau|请求/有效rank|平均投影误差|最差折投影误差|最差折解释率|平均basis稳定度|BA/floor|harmful flip|裁决|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D96 geometry diagnostic|0.25|0.937973|4/4|0.757930|0.926472|7.353%|0.958735|1.000/1.000|0|`PARTIAL_PHASE1_GEOMETRY_SELECTION_DIAGNOSTIC`|

分类指标饱和且所有候选均无flip，不能区分transport有效性；最差折只解释7.353%域残差，反而再次证明现有地面域子空间覆盖不足。产物明确保持`full_phase1_lock=false`、`geometry_effectiveness_pass=false`、`target_admission_authorized=false`。因此`tau=0.937973/rank4`只记录为84-cell几何诊断结果，不能写入正式target method lock，也不能触发125。

## 当前交叉监督阻断与修复队列

|对象|独立审查状态|主要缺陷|当前动作|
|---|---|---|---|
|D96 geometry LODO|`YES_AS_DIAGNOSTIC_ONLY`|无任务效果门；覆盖解释率低|保留诊断，不晋级；raw-center P1已修复|
|Phase1 single-observation exporter|`NOT_READY_FOR_REMOTE_PHASE1_EXPORT`|初版输出几何与D97不等价；测试callback可绕过sealed runtime；缺ADV3B02 checkpoint→runtime lineage|域适应线修复精确consumer schema、正式路径禁止注入、runtime manifest闭包和Phase2排除声明|
|D81 Phase1 episode scorer|交叉审查中|初版未绑定实现依赖SHA、sklearn/runtime/checkpoint且ground audit浅可变|域适应线加深不可变状态与完整依赖receipt，分类头锁定器拒绝普通callback冒名|
|D97 receiver-LODO selector|交叉审查中|初版量化/三block/`/beta`不等价、K5/K10直接用`eta_max`、任意callback可冒充D81|分类头线改为部署INT8选参、support内部OOF可靠度和真实scorer receipt闭包|
|D98 STRIMS|设计通过，纯核心实现中|尚无本地/真实性能证据|反遗忘线独立实现；只消费D81与qK原始logit，禁止二次融合|

当前没有N607发布或target预测。只有exporter、D81 scorer、D97 LODO三者完成同一schema/receipt闭包并通过独立复审后，才由新的实验runner执行Phase1离线export和receiver-LODO；方法作者与主线均不抢占其启动权。

## Phase1单观测流水线第二轮独立审查与修复

第一次端到端审查虽然跑通26项测试，但仍发现4个可导致错误正式声明的缺陷：自描述runtime manifest可伪造ADV3B02 lineage；selector可把generic/diagnostic manifest写成`full_phase1_lock=true`；任意余弦callback可复制字段冒充D81；D81 scorer直接把qK使用的raw concat288当作历史D81 registered feature。主线据此否决初版远端发布，修订后由同一审查线实现、主线再独立运行45项组合回归。

|对象|第二轮修复|当前状态|
|---|---|---|
|Phase1 exporter|formal模式复用既有`phase1_adv3b02_deployment_bundle.py`外部联合bundle verifier；development模式只允许预登记的真实ADV3B02 runtime SHA与parity receipt，TinyRuntime仅能进入test diagnostic；输出8成员精确allowlist和formal/development生命周期|`READY_FOR_DEVELOPMENT_N607_ASSET_DISCOVERY`；本机无真实runtime，尚未端到端导出|
|D81 episode scorer|要求精确`D81Phase1EpisodeScorer`类型与完整依赖receipt；fit/query都把raw concat288恢复为`normalize([normalize(z160),4·normalize(FFT96‖RF32)])`，qK仍保留三block等能量几何|本地真实D22 K1 smoke输出`[12,6]`有限FP32，scorer receipt稳定|
|D97 receiver-LODO selector|只接受exporter v2 manifest文件及预期SHA；核对精确成员、formal/development状态、特征语义、runtime lineage和Phase2禁入生命周期；删除generic/in-memory正式旁路|development产物固定`full_phase1_lock=false`、`development_lock_frozen=true`、`formal_target_claim_allowed=false`|
|support-only融合|K1只使用Phase1先验；K5/K10每个shot index留出每类1个独立physical support，D81与INT8 qK都只在其余support上拟合，calibration/evaluation标签不参与eta|selector专项12/12通过；候选网格不重复拟合D81|

主线组合验证命令使用`ssr-gpu`解释器串行执行，结果`45 passed`；`py_compile`和`git diff --check`均通过。警告仅包括TorchScript弃用提示与pytest退出阶段既有Windows临时symlink `PermissionError`，命令exit code为0。

### 本次N607 development release候选文件

|本地文件|SHA256|用途|计划远端相对路径|
|---|---|---|---|
|`code/scripts/export_phase1_singleobs_feature_archive.py`|`fe995a960a9cf56752a9ec010331b222f8cc81fb1a6969b0c19a5a28cbbec962`|单物理样本单LEO Phase1临时特征导出|`code/scripts/export_phase1_singleobs_feature_archive.py`|
|`code/cvsrffi/stage2_d81_phase1_episode_scorer.py`|`05dc600ea169ce9deb629ff4c764179cffda7eded16280ae6914bf4d950c0ef4`|真实D81-before episode head|`code/cvsrffi/stage2_d81_phase1_episode_scorer.py`|
|`code/cvsrffi/stage2_d96_d97_phase1_lodo.py`|`b636f9d3a5270b7cbc0f81b63204dfad9899394a5b043aa347012bd18785b913`|receiver-LODO与development lock|`code/cvsrffi/stage2_d96_d97_phase1_lodo.py`|
|`code/scripts/run_d97_phase1_lodo_selection.py`|`293ae9987bf70e8647152416894c549518a97b8661fbaf6d57d5677f0fc2beee`|SHA绑定配置驱动的不可覆盖D97 LODO命令行runner|`code/scripts/run_d97_phase1_lodo_selection.py`|
|`code/scripts/run_d96_ground_geometry_lodo.py`|`cf89a0e27091a4bfb7618099be9b31b8583d7685f67b17d2b5a2d09e573bc4d2`|D96 diagnostic-only geometry复现|本次不必同步|

第一段N607委派仅做`READ_ONLY_ASSET_DISCOVERY`：执行规定preflight，核对GPU/进程，确认source-validation cache、已知ADV3B02 base runtime、parity receipt、checkpoint SHA、Python环境与目标输出不存在；不修改、同步或启动。资产路径和SHA返回主线后，主线生成并提交development runtime manifest、selection-salt receipt、候选网格和完整exact command，再由同一唯一runner执行`LOCAL_VERIFIED→LANDED→RUNNING`。这种两段交接避免在缺失真实runtime receipt时用猜测命令启动，同时不把development研发阻塞在尚不存在的外部formal authority上。

为避免第二阶段临时拼接远端多行Python，主线新增`run_d97_phase1_lodo_selection.py`：它只接受附带SHA256的精确JSON配置，绑定D81 scorer/LODO模块SHA、ADV3B02 checkpoint、archive/ground manifest、候选网格、seed、device和不可覆盖输出目录；内部构造真实`D81Phase1EpisodeScorer`，输出完整receipt与小型release summary，并对已有输出fail-closed。新增专项3项与D81 scorer/selector组合回归共`23 passed`，脚本`py_compile`与`git diff --check`通过；pytest退出后的Windows临时symlink `PermissionError`仍是已知清理噪声，主体exit code为0。

上述Phase1流水线、D96 geometry diagnostic、专项测试与本节报告已由Git提交`e1135e5c`承载；本次N607资产发现和后续release均以该提交为唯一代码基线，D98未纳入该提交或本次同步范围。

## D98反遗忘线当前状态

D98-STRIMS第二轮作者修复后，新的独立复审结论为`ACCEPT_LOCAL_CORE_ONLY`。专项`11 passed`，D81+D97+D98联合`39 passed`；K1强制`alpha=0`且不调用OOF producer，合法K5/K10路径在模块内按physical ID形成exact complement，并实际调用typed D81 scorer和D97 INT8 raw head；重复/空physical ID、非均衡K、class permutation、logit gauge、温度/receipt漂移均有fail-closed测试。当前公开API没有generic probability/logit fuse或deploy inference入口，状态保持`LOCAL_CORE_PENDING_TYPED_D81_INTEGRATION`。

但它有三项实验阻断。第一，仍无typed target D81 inference，私有数学fuse不能绑定真实target D81 state/logits、class registry和列顺序。第二，每个support OOF fold都会调用包含20个optimizer steps的D81 scorer，端到端K5至少100步、K10至少200步，超过探索75步硬门；D98局部`resource_audit.optimizer_steps=0`不能代表组合方法总资源。第三，Python私有capability只是API约定，不是认证边界；独立审查可读取私有token、注入任意logit/fold record并重算receipt后被fit接受。因此D98只作为本地数学核心与tail/intrusion/retention诊断工具进入Git，不得进入当前D97 release、N607 target窄实验或125，也不能声称已有性能收益。下一次复审必须同时补typed target D81闭环、消除任意数组推理旁路并把完整OOF资源压回硬门。

## N607第一阶段只读资产发现

唯一实验runner已完成`READ_ONLY_ASSET_DISCOVERY`，未同步、未启动、未写远端。规定的`N607`直连preflight通过；主机为`dell-DSS8440`，8张RTX3090均为0%利用率且无训练计算进程。每次短连接后均确认本地`ssh.exe`退出、到N607和bridge的TCP22连接为0。

|资产|远端绝对路径|SHA256|核验|
|---|---|---|---|
|Phase1 source-validation cache set|`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json`|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|匹配；三份single-LEO NPZ均存在，实际IQ布局为`[N,2,256]`|
|ADV3B02 base runtime|`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ci_strict_matrix_20260716_v1/runtime_artifacts_v2/adv3b02_base_runtime.ts`|`b2021ca1ac97848a8cfda353a4070530bfa41bc08a711f746f329bd2d8d870d9`|匹配；CPU只读加载成功，输入经256-row padding后返回`(z160,logits)`|
|ADV3B02 checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|匹配；历史checkpoint audit记录`input_len=256`|
|历史parity receipt|`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ci_strict_matrix_20260716_v1/runtime_artifacts_v2/runtime_parity_receipt.json`|`db8635b986bcaea6cbe6f954e90e5ed37b9fb6042876628392db96fe82be42f4`|历史PASS，但schema为`cvs.adv3b02_effective8_torchscript_parity.v1`，不能冒充新exporter所需receipt|

P0阻塞是远端不存在`cvs.phase1.runtime_checkpoint_parity_receipt.v1`兼容receipt和runtime manifest。历史receipt缺checkpoint lineage、parity vector root、TorchScript archive/state structure roots等字段，不能通过改名或人工补字段进入D97。主线因此新增development-only验证器：从真实ADV3B02 checkpoint重建eager model，对既有allowlisted base runtime执行显式`input_len=256/parity_seed=20260720/parity_rows=8/tolerance=1e-5/device=cuda:0`的确定性数值比较；同一receipt固定覆盖batch`[1,8,256]`，逐批验证z160/logits的shape、finite和delta，最终delta取全矩阵最大值。请求CUDA不可用或索引非法时立即失败，禁止静默回退CPU；stdout必须报告`resolved_device=cuda:0`。checkpoint/runtime从同一稳定bytes完成hash、load和runtime structure计算，落盘前再次核对路径SHA以阻断TOCTOU。完整但不含原始IQ的vector摘要另存non-authority audit并可独立重算receipt root。

独立复审裁决为`ACCEPT_DEVELOPMENT_ONLY`：相邻20/20测试、`py_compile`和`git diff --check`通过；脚本SHA=`e611297d04fa4cb98aebcc75d95bde198325942fc4d1e240b90f271c56f7048b`，测试SHA=`604dc4e4402fec1a83b63aa679fac9e569c5bc0e0cea9c3d2422a1913d7b0499`。允许唯一runner在Git提交后只运行development parity并封存stdout/vector/receipt SHA；该self-generated数值receipt不是external signature或formal authority，不能单独授权formal bundle、target晋级或性能结论。极端情况下第二个文件写失败只可能留下明确non-authority vector，不会留下PASS receipt。

远端项目根不是Git仓库，且多个D81依赖与本地提交不同，因此第二阶段不得覆盖远端`code`。发布将从最终Git提交生成只含`code/`与`paper_reproduction/`的commit-bound隔离源码包，落入`runs/d97_phase1_singleobs_lodo_20260720_v1/source_<commit>/`；D98虽然作为本地研究核心进入Git，但显式排除在D97源码包之外。run/log v1目标路径均已确认不存在。当前状态为`READ_ONLY_ASSET_DISCOVERY_COMPLETE / NOT_LANDED / BLOCKED_PENDING_REVIEWED_DEVELOPMENT_PARITY`。

### Development parity发布预登记

独立复审通过后，冻结代码提交为`798dedfd12f2db067e10c6c280be1fe62d982841`。本地使用`git archive`从该提交生成隔离源码包`E:\type10-7\code\snapshots\d97_phase1_singleobs_lodo_20260720_v1\source_798dedfd.zip`，大小7,398,449B，SHA256=`700b4318597d7d21c3e7936a388713964bc6ec4c0461973e16d18b10ca4b164b`，共1,558个成员；必需的parity/export/LODO/D81/bundle文件全部存在，`stage2_d98_strims.py`明确排除。远端同步映射为：

|本地|远端|用途|
|---|---|---|
|`E:\type10-7\code\snapshots\d97_phase1_singleobs_lodo_20260720_v1\source_798dedfd.zip`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d97_phase1_singleobs_lodo_20260720_v1/source_798dedfd.zip`|不可变commit-bound源码包|
|源码包内`code/scripts/verify_adv3b02_runtime_checkpoint_parity.py`，commit/archive字节SHA=`bfc55243a33fd956087e1ff352932064ef0291ebd6a5026be77995c95c8e3169`|解包后`source_798dedfd/code/scripts/verify_adv3b02_runtime_checkpoint_parity.py`|本阶段唯一执行脚本|

首次LANDED哈希门正确停止：ZIP整体SHA、大小、成员数、D98排除和`py_compile`均通过，但报告最初误把Windows工作树CRLF脚本SHA`e611297d...`当成commit-bound发布SHA；`git archive`内LF字节及远端解包脚本均为`bfc55243...`。没有parity进程被启动，locks为空，SSH/TCP22已清零。新发布权威改为ZIP成员字节而非工作树字节，完整关键成员如下：

|ZIP成员|commit/archive字节SHA256|
|---|---|
|`code/scripts/verify_adv3b02_runtime_checkpoint_parity.py`|`bfc55243a33fd956087e1ff352932064ef0291ebd6a5026be77995c95c8e3169`|
|`code/scripts/export_phase1_singleobs_feature_archive.py`|`b23496b72019257febeab1beae30692498b43d92f16be5bb3d4bb8e8a3ab25bd`|
|`code/scripts/run_d97_phase1_lodo_selection.py`|`1daa990d00cc33e916758030bc8d94b0ac57ebad8a50887289f0b191c583e945`|
|`code/cvsrffi/stage2_d81_phase1_episode_scorer.py`|`54ee742c81b60e00b6c1c36d2d6bf1f0409ad10f72a25e01c2dcd589093be55d`|
|`code/cvsrffi/stage2_d96_d97_phase1_lodo.py`|`8ea89935a18a3852a8fa388c531ec18495fe87b3edf7597870ecdeb9776ae18a`|
|`code/cvsrffi/phase1_adv3b02_deployment_bundle.py`|`075a575f0dfe85e1201960b089fd7c33deee1d5ad247613ed0b35e0c5ffa2c84`|

该阻断属于发布预登记字节口径修复，不是实验失败或参数重试。下一次继续前只接受已landed且整体SHA匹配的同一ZIP，并要求远端关键成员逐项匹配上表；不得重打包、覆盖或放宽哈希门。

本阶段只授权唯一runner执行parity，不授权feature export、LODO、runtime manifest生成、exporter binding或target访问。重新preflight并确认GPU0可用、已landed run/log状态与报告一致且locks为空后，核对源码包与脚本SHA。固定CWD为`/home/szu2070436088/2510044040/CV-SincNet/runs/d97_phase1_singleobs_lodo_20260720_v1/source_798dedfd`，使用远端`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`从隔离源码运行：

```bash
env PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d97_phase1_singleobs_lodo_20260720_v1/source_798dedfd/code \
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
/home/szu2070436088/2510044040/CV-SincNet/runs/d97_phase1_singleobs_lodo_20260720_v1/source_798dedfd/code/scripts/verify_adv3b02_runtime_checkpoint_parity.py \
  --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth \
  --runtime /home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ci_strict_matrix_20260716_v1/runtime_artifacts_v2/adv3b02_base_runtime.ts \
  --receipt-out /home/szu2070436088/2510044040/CV-SincNet/runs/d97_phase1_singleobs_lodo_20260720_v1/locks/runtime_checkpoint_parity_receipt.json \
  --vector-audit-out /home/szu2070436088/2510044040/CV-SincNet/runs/d97_phase1_singleobs_lodo_20260720_v1/locks/runtime_checkpoint_parity_vector_audit.json \
  --input-len 256 --parity-seed 20260720 --parity-rows 8 \
  --device cuda:0 --max-abs-tolerance 1e-5
```

detached wrapper固定使用上述CWD和child command；wrapper PID写入`/home/szu2070436088/2510044040/CV-SincNet/runs/d97_phase1_singleobs_lodo_20260720_v1/parity_validation.pid`，child退出码以不可覆盖方式写入同目录`parity_validation.exit`。runner必须记录launch PID、`nvidia-smi`中的GPU0进程、完整cmdline、CWD、启动时间和完成时间；每次短SSH后验证本地`ssh.exe`与N607/bridge TCP22均为0。

本阶段成功条件仅是进程exit0、receipt/vector/stdout/exit-status均完整且SHA已回收、`resolved_device=cuda:0`、`batch_sizes=[1,8,256]`、最大delta≤`1e-5`、checkpoint/runtime SHA匹配。不得在远端临时生成runtime manifest或调用exporter binding。主线回收真实receipt SHA后，才在本地用`apply_patch`生成runtime manifest、selection-salt、grid与后续config，完成本地exact binding、测试和Git提交，再交由同一runner执行下一阶段。失败时不得自动重试、不得修改参数或runtime/checkpoint；返回完整日志、PID/exit和GPU/断连证据给主线。状态转换只允许`LOCAL_VERIFIED→LANDED→RUNNING→ARTIFACTS_COMPLETE`，本阶段不能进入`ANALYZED`性能结论。

### Development parity执行结果

本轮最终状态为`PARITY_NUMERICAL_GATE_FAIL / NOT_ARTIFACTS_COMPLETE`，不是性能结果。第一次landing因CRLF/LF预登记SHA口径错误在脚本执行前正确停止；修订报告提交`131815384686fda6cd9e7dd72b5cdb877644bb7b`锁定archive成员SHA后，唯一runner使用同一已landed包启动一次parity，未重试、未改seed/threshold/runtime，也未运行export、LODO、runtime manifest、binding或target访问。

|状态|证据|
|---|---|
|`LOCAL_VERIFIED`|Git提交、ZIP整体和关键成员SHA通过|
|`LANDED`|隔离源码落地，D98/D99均不在包，`py_compile`通过|
|`RUNNING`|wrapper PID=`1241943`，GPU0；child使用绝对脚本/资产路径和固定`PYTHONPATH`|
|`ARTIFACTS_COMPLETE`|未达到；exit=`1`，locks成员数0，无receipt/vector|

实际child CWD记录为`/home/szu2070436088`，而非后来补充预登记的隔离source CWD；但脚本、checkpoint、runtime、output均使用绝对路径，且脚本按`__file__`重建`REPO_ROOT/CODE_ROOT`，所以该偏差需要记录但没有证据表明它造成数值差异。实际Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，设备为`cuda:0`，参数为`input_len=256/parity_seed=20260720/parity_rows=8/tolerance=1e-5`。完成后GPU0恢复0%/10MiB且无计算进程。

|batch|feature最大绝对差|logit最大绝对差|门槛|裁决|
|---:|---:|---:|---:|---|
|1|0|0|`1e-5`|通过|
|8|`1.0484457015991211e-4`|`2.4023056030273438e-3`|`1e-5`|失败|
|256|`1.3887882232666016e-4`|`2.3717880249023438e-3`|`1e-5`|失败|

失败证据回收到`E:\type10-7\code\snapshots\d97_phase1_singleobs_lodo_20260720_v1\remote_artifacts`：

|文件|字节|SHA256|
|---|---:|---|
|`parity_validation.out`|1,139|`763e35354d67aae203ce217b7717662ce7b6136b30a69de1208efb686c138a66`|
|`parity_validation.pid`|8|`bf7102f72c23778c6033e5183d35f9dc3585d3a1d0166f613077c3148d0def91`|
|`parity_validation.exitcode`|2|`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`|

历史b202生成日志曾报告同一checkpoint重建`missing/unexpected/skipped=0`、`input_len=256`，且其历史单batch探针中`base_eager_vs_torchscript_feature/logit=0/0`。历史PASS只覆盖旧seed、单batch8和旧脚本门，不能覆盖本次`[1,8,256]`新固定探针。当前形态最像batch>1时eager与traced TorchScript使用不同CUDA数值路径，约`1e-4`的z误差经分类头放大到约`2.4e-3`；仍不能排除当前eager重建代码/依赖与历史导出环境漂移。不得通过换seed、放宽阈值或CPU运行把它改成PASS。

source-validation三份NPZ一致证明旧类原始label顺序`[0,1,2,3,4,5]`映射为`[14-10,14-7,20-15,20-19,6-15,8-20]`；独立mapping artifact SHA=`97af6115b51a6a3252e22315e40183c4c3efd7ccfeb1f16a61710028f72fda7f`和strict plan使用相同顺序。后续runtime manifest必须同时绑定checkpoint SHA、mapping SHA和该顺序，不能只依赖NPZ首次出现顺序。

第二个已在Git allowlist中的历史runtime为`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`，远端4,613,201B、CPU TorchScript加载成功、forward同为`(Tensor,Tensor)`，D18 method lock与row log把它和同一checkpoint SHA配对；但未找到独立生成命令或checkpoint→runtime parity receipt。它只能进入新的、事先提交的v2软件实现验证，不能因b202失败自动切换或继承历史PASS。

协议合法的下一步先是non-authority数值诊断：固定现有seed/batch/threshold，比较eager↔eager、runtime↔runtime、fresh trace↔eager/旧runtime，以及CPU/CUDA、deterministic、TF32、cuDNN设置和PyTorch/CUDA/cuDNN版本；不写PASS receipt。若旧runtime存在稳定语义差异，则拒绝绑定并从冻结checkpoint+代码生成新的不可覆盖runtime，再重新通过原`[1,8,256]/1e-5`门。正式D97 Phase1 export继续阻断。

### non-authority runtime诊断第二次独立复审：REVISE

诊断修订版专项`12 passed`，与parity/exporter联合`22 passed`，但独立二审仍为`REVISE`，不得提交、发布N607或据此选择runtime。已确认双arm SHA限制、固定`[1,8,256]` probe、canonical exclusive写、eager/existing/fresh tensor根、CPU不可授权及无PASS receipt均成立；剩余P0如下：

|阻断|证据|修复要求|
|---|---|---|
|arm lineage不完整|`RUNTIME_ARMS`只有SHA和自描述字符串；相同bytes复制到任意path仍可继承b202/f119身份，f119也未绑定D18 method-lock/row-log SHA|预锁canonical origin path与真实lineage artifact SHA，隔离副本必须保留origin receipt|
|进程隔离不足|deterministic子进程虽在启动前收到CUBLAS，但父orchestrator顶层仍import Torch/相关依赖；fresh trace在每个comparison worker中内联生成|父进程不得import Torch并证明CUDA未初始化；独立trace-builder先生成immutable fresh runtime，comparison worker只读|
|dependency/Git根不完整|漏`model_dual_cvsincnet.py`、两个paper reproduction模块、runtime trust/prototype/predictor bundle及`train_ssdg.py`加载模块；只报HEAD不报dirty/untracked，ZIP无`.git`时接受caller声明|闭合实际import依赖，绑定source archive；记录HEAD/status/dirty/untracked/diff roots，不能由caller自授权commit|
|worker→final非fail-closed|父进程先parse文件再重读hash，存在TOCTOU；stdout summary未与artifact SHA/bytes闭合；fake launcher无launch audit仍能生成final|同一stable bytes同时parse+hash；强制校验stdout、launch audit、mode/scope/device/CUBLAS/source/env/fresh roots|

P1还包括subprocess无timeout、`nvidia-smi`失败被静默降为`driver=None`、缺总orchestrator耗时/host peak以及per-channel quantization registry不完整。第三轮修复继续保持development diagnostic only，不访问数据、不输出runtime晋级结论。

第三轮作者补充canonical origin、独立trace-builder、既有SOMP-H Ed25519公钥、Git dirty根和worker证据后，专项与联合`30 passed`，但第三次独立复审仍为`REVISE`。实测在worker/trace-builder的JSON parse与后续文件hash之间替换文件，内存payload和记录SHA来自不同bytes仍被接受；删除`asset_lineage.runtime_origin`后validator也继续接受。完整origin对象必须由固定arm contract重算并核对canonical/copy路径、scope和origin receipt，所有文件必须单次`read_bytes()`后用同一bytes完成parse/hash/length/stdout绑定。

签名source archive也尚未绑定实际执行源码：父进程验证的是archive SHA/commit，worker实际import当前工作树；dirty/untracked只记录不阻断，无`.git`也可继续，实际依赖未逐成员对照signed archive manifest。下一版必须从已验证archive隔离执行，或逐一闭合实际import dependency的path/SHA/size并拒绝路径逃逸、symlink、缺失和额外执行依赖。既有SOMP-H公钥是真实信任根，f119/b202 lineage SHA也有历史实物，但仓库当前没有任何真实`cvs.development.source_archive_commit_receipt.v1`签名receipt，只有源码和测试fixture。因此工具必须显式`BLOCKED_MISSING_SIGNED_SOURCE_RECEIPT`；单元测试的签名fixture不能变成N607发布授权。

第四轮独立复审裁决为`ACCEPT_DEVELOPMENT_TOOL_BLOCKED_MISSING_RECEIPT`。worker/trace均改为单一bytes snapshot完成parse/hash/length/stdout绑定；runtime origin按固定arm与resolved path重算完整对象；signed source manifest逐path/size/SHA与ZIP、实际loaded modules及Git策略闭合，父子execution contract也绑定相同canonical bytes。测试fixture只能通过私有Python参数进入且明确`UNIT_TEST_SIGNATURE_FIXTURE_NOT_AUTHORIZED`，生产CLI没有该开关。没有真实archive/receipt时实测exit2、`BLOCKED_MISSING_SIGNED_SOURCE_RECEIPT`且不生成final artifact。独立复审报告51项通过；主线以`ssr-gpu`环境复跑当前三个稳定测试文件得到`49 passed`、exit0，差异来自测试收集口径而非失败。工具可进入Git作为development-only阻断诊断，但真实signed source receipt尚未生成，仍禁止N607发布。

### offline signed source release producer

为闭合上述真实receipt缺口，新增独立offline producer，而不是让诊断consumer自行签名。producer固定复用既有SOMP-H issuer/key/public verifier和pinned OpenSSL；只允许clean HEAD、版本化member lock和commit blob；从排序`.py`成员直接构造deterministic ZIP_STORED，签名正文与consumer canonical schema逐字段一致。发布使用独立staging、O_EXCL/O_NOFOLLOW、fsync、0444和no-replace目录事务，错误key、dirty Git、member/path/schema漂移、并发目标及中途失败均零输出或回滚。

独立复审裁决为`ACCEPT_LOCAL_PRODUCER_BLOCKED_EXTERNAL_ARTIFACTS`：producer与consumer联合`60 passed`，固定身份、Git blob、ZIP、签名正文、exact loaded-module binding及回滚闭合。当前仍缺真实reviewed 51-member lock、外部有效private key/签名receipt，以及从签名ZIP安全展开到无Git隔离根再启动诊断的runner；因此只能合入local core，不能生成authority、发布N607或作formal launch/指标声明。

### signed-source隔离runner收口与实验优先级调整

隔离runner第三轮独立复审裁决为`MERGE`，范围仅限可选local diagnostic tooling。此前同inode等长改写、child读取修改内容、再恢复原bytes与mtime的窗口，现由初始snapshot、held FD和current path三侧`ctime_ns`一致性检查关闭；Linux定向负例已复现并确认正确拒绝。主线在`ssr-gpu`环境复跑member-lock、isolated-runner、source-release producer和runtime diagnostic四组测试，得到`88 passed,1 skipped`、exit0；skip为Windows上不执行Linux能力负例，告警为TorchScript弃用、故意构造的duplicate ZIP fixture和pytest退出后临时目录清理噪声。

根据用户最新目标，以上源码签名、review authority envelope和额外代码对齐不再作为无线信号模型研发或N607 development experiment的前置门。后续只保留Git commit、文件SHA、不可覆盖run ID和现有实验报告作为代码版本证据；模型实验的硬约束回到`p2_min_v1`：复用匹配`VALIDATED_ONCE`数据、support-only适应、query逐样本只前向、无clean/source runtime访问、无query truth/role/quota。正式论文/部署claim仍需完整产物闭合，但不得因此阻断ground-only LODO、target narrow diagnostic或matched性能验证。

### D99/D100实验优先发布预注册

N607于2026-07-21 00:07 CST完成只读preflight：8张GPU均为0%利用率、10MiB显存占用且无compute process；未发现D81/D99/D100或Phase2目标进程。远端环境固定为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Torch2.1.0+cu121；每次短SSH结束后本地`ssh.exe`及N607/bridge TCP22连接均为0。历史完整125位于`runs/d81_comprehensive_125_v2auth_retry9_20260720`；下表两条精确matched窄基线来自`runs/d93_paired_ground_transport_dev_20260720/matched_d81_k{1,10}_new20`，同数据、同checkpoint/runtime/method-lock，无需重复计算。

|matched D81 cell|B-old|A-old|min-old|seen-new|H|forgetting|
|---|---:|---:|---:|---:|---:|---:|
|receiver20-1/seed713101/K1/new20|61.667|37.500|13.333|27.583|31.786|24.167|
|receiver20-1/seed713101/K10/new20|87.222|69.722|48.333|68.917|69.317|17.500|

第一阶段先运行固定Phase1 encoder/ground basis上的receiver-level pseudo-target LODO，分别冻结K1/K5/K10/K20的D99与D100参数；D99的held receiver ground row会删除，但固定D81 global basis不随fold删除，因此不得将其表述为“全方法严格ground-domain外留一”。无额外authority时允许输出`NONFORMAL_LODO_DIAGNOSTIC`，但不得读取target/query或反向用target性能选择参数。

LODO预注册64候选：`eta=[0.25,0.5]`、`shared_h0=[0.35,0.5]`、`T99=[0.85,1.0]`、`lambda=[0.08,0.2]`、`Tridge=[0.85,1.0]`、`alpha=[0.2,0.35]`；`nu=3`、`gamma=1`、`scale_prior=2`、`scale_min/max=0.5/2`固定。非搜索research prior固定为`density_tau=0.2`、ground/target rank=`2/2`、coverage floor=`0.01`、energy scale=`0.01/0.01`、shrinkage prior=`2`、ground/target weight cap=`0.8/0.6`、kernel effective dim=`12`、三block权重=`0.7/0.2/0.1`。这些值是target访问前的development预注册，不是LODO选出的最优值；base lock中的非正式占位SHA不代表已完成formal margin、validation或D81 lock。

地面bundle由D19的14条有效day-domain×class INT8聚合中心在Phase1离线侧压缩为7个receiver×class中心：固定pair解量化、每条L2归一化、`0.5/0.5`球面均值、再L2归一化和per-vector INT8重编码。使用等权而非另一个cache的样本数，避免把非成员计数冒充prototype membership；`physical_sample_count_floor=2`只表示每个输出由两条独立day-domain聚合项合成，不能声明精确物理样本数。bundle保存TX registry，不保存成员ID、sample feature、raw/clean IQ或target/query row。

第二阶段发布2个真实narrow job：K1/new20与K10/new20，均为receiver20-1、seed713101、三场景；每个job内部同时产生D81、D99、D100三候选的严格同row预测与指标。GPU0运行K1，GPU1运行K10，每job CPU thread=2，GPU2-7留空；不再把D99/D100拆成4个重复job。run ID和完整命令在query adapter及LODO独立复审、Git提交后写入本节再交sole launch subagent。每个job必须输出同row B/A/min-old/min-new/min-all/New/H/F、全注册balanced accuracy、逐场景、逐类、三类混淆、INT8量化、资源和prediction artifact；无论正负都完整分析，不以单指标极值替代同row判断。

### D99/D100发布实现终审

最终实现由三条独立复核链收口。LODO要求D99相对D81同时满足worst floor不降、balanced NLL严格改善超过`1e-6`、D81↔Student-t双向rescue非零；K1还必须至少改变一个D81预测。D100相对D99要求每个receiver×pseudo-new pair的floor、pseudo-old、pseudo-new、H均不降、balanced NLL严格改善超过`1e-6`且D99↔ridge双向rescue非零。D99失败则该K不产锁；D100失败但D99通过时保留D99、alpha强制为0并标D100不晋级。最终LODO独立裁决为`MERGE`，专项`51/51`通过。

query adapter使用唯一canonical D81→D99→D100公式，完整校验LODO fixed point、7域ground bundle、base prior、D81组件与checkpoint；兼容N607真实D19 v1 class binding，确定性升级为typed v2并同时记录raw SHA。逐query batch size固定1，全部prediction artifact完成后才读取truth；输出min-old、min-new、min-all、全注册balanced accuracy、三场景、逐类、混淆、量化和资源。线程参数在NumPy等导入前解析，同时支持`--cpu-threads N`与`--cpu-threads=N`，并显式限制Torch/threadpool。独立裁决为`MERGE`，最终联合`89/89`通过。

输入链新增两项明确nonformal工具：`f119` runtime的SHA-only Phase1 exporter，以及D19 14条day-domain聚合中心压缩为7条receiver中心的builder。formal exporter仍强制合法parity SHA；development exporter固定parity为null。ground builder调用完整D19 validator、固定6个TX与14条active row、检查FP16 scale和INT8范围，输出development ground schema/status及nonformal prior wrapper；同一wrapper可被Phase1 runner和narrow runner严格解析，formal入口拒绝。最终集成独立裁决为`MERGE`，builder→narrow lock digest一致。

主线在`ssr-gpu`环境对8个关键测试文件复跑得到`86 passed`、exit0；随后wrapper集成与formal parity定向复跑`14 passed`、exit0。唯一附带信息仍为TorchScript弃用告警和pytest退出后的Windows临时目录`WinError 5`清理噪声，未影响测试退出码。数据无关selection salt已写入`preregistered_inputs/d99_d100_phase1_selection_salt.json`；其bundle ID绑定f119 runtime SHA、ADV3B02 checkpoint、6个TX顺序和288D特征维度，`target_access=false`。

## 下一轮唯一集成候选D99

跨方法监督否决继续增加第三个全局头或D98式二次融合权，建议下一轮只实现`D99 RA-CGTMK-D81`：保留D81一次拟合，将D96的密度反权、`D_eff`、共享ground nuisance basis和support-only coverage certificate用于构造严格PSD低秩Mahalanobis度量，再把D97的各向同性qK分支替换为类数归一化Student-t metric-kernel。ground只改变可观测距离，不直接给旧类加分；old/new仍由同式target support注册。

其核心收缩为：ground权重同时受`rho`、target偏移能量、`D_eff`和旧类support数量控制；target within-class basis按全部注册类平衡估计且rank≤4，K1时严格为0；最终非正交`P_z=Sigma_z^{-1}`只降低已知nuisance方向权重，不执行D93式全坐标transport。Student-t局部头保留`1/K_c`归一化，并用Phase1冻结的共享/类内尺度抑制异常support与过宽类别侵入。D81与metric-kernel只使用Phase1 receiver-LODO冻结的`eta_K`做一次row-global融合，target阶段不做K折D81重训，也不学习classwise alpha。

D99作者已形成纯核心初版，专项15/15、D81/D96/D97/D98/D99联合61/61通过，但独立复审裁决为`REVISE`，所以文件保持未提交且无性能结果。复审确认密度反权、加权几何、`D_eff`自适应rank≤4、PSD precision、class/support permutation、`1/K_c`、三block INT8 bank、K1共享尺度及逐query无状态成立；同时发现6类阻断：Student-t缺`-d·log h`尺度体积项；裸FP32中心可冒充聚合bundle；ground类可被调用方映射到new类参与coverage；margin audit接受任意features却自述Phase1-only；K10资源至少因遗漏`residual.T@residual`而低报28.07倍且resource未进入receipt；低ground coverage错误地同时关闭合法K≥2 all-class target residual metric。仓库仍无typed target D81 state，因此无public deploy fuse/predict是正确边界。

D99作者已收到上述P0并在原两个独立文件中修复；修订版仍需新的独立复审。最终准入要求不变：Phase1每个K均不得降低worst-class floor，NLL必须严格改善且D81/metric-kernel双向rescue均非零；K1还必须产生非identity预测。target窄实验只允许matched K1/new20与K10/new20，同row`B-old/A-old/New`均不得下降，`H/floor`至少一项严格上升且forgetting不得增加；通过后才运行完整125，125不反向选择`rank/eta/nu/h/rho`公式。D96 standalone SRDA保留为`REVISE`，D97为`MERGE+REVISE`，D98只保留typed provenance和tail指标工具，不进入D99推理链。

## D99第二次独立复审：仍为REVISE

修订版专项`17 passed`，D81/D96/D97/D99联合`56 passed`，但新的独立复审仍裁决为`REVISE`，不得提交为晋级候选、发布N607或运行target窄实验。Student-t尺度体积项、typed INT8 ground local bundle、sealed `Y_old`映射、低coverage保留合法K≥2 target residual metric这4项已经修复；剩余阻断不属于一般单元测试覆盖不足，而是权限真实性和资源上界被对抗样例直接证伪。

|阻断|独立攻击/实测|裁决|
|---|---|---|
|Phase1 validation自授权|任意raw feature、任意physical ID和任意64hex SHA可由`produce_phase1_validation_artifact`自行声明为`PHASE1_SOURCE_VALIDATION_ONLY`；名称为`target-query-*`的输入也成功通过|必须改为绑定producer、manifest、checkpoint、实际feature archive SHA和生命周期的external typed receipt；margin support bank同样需要Phase1 episode lifecycle|
|query MAC低报|合法`C=2,K1,rank1`报告1,856 MAC，解析下界2,016 MAC，至少漏160维query侧precision transform/norm|重算完整逐query路径并加入固定对抗测试|
|ground peak低报|合法`D=14,C=6`报告723,488B，保守同时存活数组下界1,133,088B，低报409,600B|必须覆盖全部`D×C×160`临时float64数组和真实生命周期|
|serialized bytes低报|报告18,078B使用`resource=None`的metadata；receipt-bearing完整artifact为18,985B，低报907B|实际序列化对象必须包含resource并以完整artifact计数|
|typed D81未接入|虽然commit`907bd620`新增typed state，D99仍是pending schema；随后独立审查又证实该typed state本身不等价于真实D81|D99第三轮只闭合自身provenance/resource；在corrected typed D81独立通过前保持fail-closed pending|

因此本轮没有target性能指标，也没有新的old/new/H/floor/forgetting结果。测试通过只证明当前local core内部一致，不能覆盖source/target权限真实性、资源上界或历史D81语义等价。

### D99第三次独立复审：仍为REVISE

第三轮作者把正常资源公式修正为`C=2,K1,rank1` pair kernel 1,856 MAC、query precision/norm 160 MAC、query kernel合计2,016 MAC、完整prediction call 6,336 MAC；`D=14,C=6` ground peak上界1,756,736B；K10完整receipt-bearing wire 19,182B。专项`20 passed`、D81/D96/D97/D99联合`75 passed`，typed D81也正确保持`LOCAL_CORE_BLOCKED_CORRECTED_TYPED_D81_P0`，无base-logit/probability/fuse/predict绕过。然而独立攻击仍发现两个P0：

1. `ExternalPhase1ValidationReceipt`仍是无密钥自签摘要。攻击者用修改后的validation feature、`target-query-physical-*`、任意producer bytes和错误checkpoint SHA=`f`×64，重算archive/manifest/receipt后仍能load并进入margin audit。名称过滤只检查episode名称，physical ID被哈希后无法证明source语义；bank support receipt也能一并重签。
2. resource只防止“改字段不重签”，不防内部重签。将`query_mac_upper_bound=0`、`actual_serialized_runtime_artifact_bytes=1`后重签bank receipt，score仍成功，实际serializer输出13,248B却接受声明1B。

第四轮修复因此必须把external expected receipt、allowlisted producer/checkpoint/archive/manifest SHA及source-validation lifecycle写入不可由D99调用链生成的Phase1 method/deployment lock，并对实际bytes逐项核验；若真实外部authority尚无，local core必须明确阻断margin audit。resource则必须从numeric state、维度和serializer独立重算，在构造、score和serialize三个边界验证；receipt-bearing fixed point必须满足reported bytes等于实际wire，修改resource并重签仍须失败。正常资源数字通过本轮复算不等于authority和tamper门已经通过。

第四轮作者新增`Phase1ValidationMethodLock`并实hash四份bytes，正常资源固定点更新为K10 wire 19,311B；专项`20 passed`、稳定相邻联合`47 passed`。第四次独立复审仍为`REVISE`。第一，真实Phase1 authority尚未provision，调用侧仍可同步构造method lock、external receipt和D99 lock并让margin audit得到`matches_phase1_lock=True`；copy/deepcopy/pickle后即使loader token identity丢失也可继续audit，说明token只能防进程内误用，不能授予权限。缺真实immutable authority envelope时，正式audit必须`formal_phase1_eligible=false`或fail-closed，不能把development自建链写成正式match。第二，bank自身resource可独立复算，但score/serialize只核metric receipt，未从dimensions重算metric resource；将`residual_covariance_mac_upper_bound=0`后同时重签metric与bank receipt，score和19,311B序列化仍成功。下一版在bank构造、score和serialize三处统一调用metric dimension公式验证，不能信任重签receipt。

第五轮独立复审最终裁决为`ACCEPT_LOCAL_CORE_BLOCKED_AUTHORITIES`。仓库trusted authority envelope SHA保持`None`；任何调用侧自建method lock/receipt/D99 lock/envelope及其copy/deepcopy/pickle都无法通过formal precompute/audit，development输出强制`authority_status=BLOCKED`、`formal_phase1_eligible=false`、`matches_phase1_lock=false`、`formal_result_claimed=false`。统一metric资源验证也已在metric构造、bank构造、score和serialize四个边界从dimensions独立重算；把residual covariance MAC改为0并同时重签metric/bank fixed point仍全部拒绝。正常固定点保持query kernel 2,016 MAC、完整调用6,336 MAC、ground peak 1,756,736B、numeric logical state 15,068B、wire 19,311B。独立复审D99专项`20 passed`、稳定联合`47 passed`；主线用`ssr-gpu`环境Python复跑同一稳定联合也得到`47 passed`。该裁决只接受本地核心闭包，不授予正式Phase1 authority、corrected typed D81、N607或性能实验权限。

### K20完整扩展

主动目标包含K20，但此前D99与typed D81只支持K={1,5,10}，这是完整矩阵前的P0。扩展后typed D81支持K20并继续保持external ordered row；非排序K20对历史D81的`log_diag/before/final`均bit-exact。D99新增无默认值的独立`eta_k20`及K20 LODO artifact字段，禁止复制或回退`eta_k10`；仓库trusted K20 LODO SHA仍为`None`，所以任意caller SHA都只能得到`BLOCKED_PHASE1_K20_LOCK`、`formal_k20_eligible=false`。

|K20固定点|数值|
|---|---:|
|typed D81 component fits|168（D46 84＋D62 84）|
|typed D81 fit MAC|25,096,476,544|
|typed D81 fit peak|352,748,491B|
|typed D81 query MAC|40,474|
|typed D81 head logical/wire|20,032B / 35,746B|
|D99 bank logical/wire|159,124B / 163,810B|
|D99 query MAC / peak|2,397,760 / 4,496,640B|
|D81 ground＋D99 ground＋D81 head＋D99 bank persistent|198,832B（194.17KiB）|
|距256KiB上限余量|63,312B|

终审裁决为`ACCEPT_LOCAL_K20_BLOCKED_LODO_AUTHORITY`。typed专项`32 passed`、D99专项`23 passed`、稳定联合`111 passed`；主线复跑相同7文件联合亦得到`111 passed`。K20资源和状态预算通过，但真实Phase1 receiver-LODO K20 eta artifact未封存，故仍不能N607、target或确认矩阵。

### 确认矩阵口径修正

历史“125实验”是5个receiver×5个seed×5个cell：K10/new5、K10/new10、K10/new20、K5/new20、K1/new20；每job含3个场景。它不含K20和new2。当前active objective显式要求K={1,5,10,20}与seen-new={2,5,10,20}全组合，因此正式成功矩阵应为400个job、1,200个scenario row。后续保留历史125作为可比screen，但不得再把它称完整目标矩阵；除非用户明确改变因子集合，正式成功声明必须覆盖400/1,200。

## D100 RA-CGSPR-LGF本地核心

D100是在并行方法监督中唯一保留的下一候选：coverage-gated D99广义余弦metric负责ground域方向，D99 Student-t头保留局部多峰/异常support，新增全类simplex dual ridge用全部注册类提供显式负证据。SRDA因与D81/D96线性协方差几何重复被拒，Bayesian vMF因与D99正样本密度高度重复且缺真实P3/P4多原型bundle而暂缓。simplex ridge对old/new每个实际类严格等权，不使用D92式任务组权重。

D100从exact typed D99 INT8 bank解码support，使用D99 precision的解析平方根构造同一三block feature，以centered simplex标签闭式求解dual ridge；K1/K5/K10/K20各自冻结lambda、temperature和alpha，不允许K10外推K20。target-class weight为INT8，scale/bias为FP16；共享target-conditioned metric basis/attenuation仍是FP32，因此只能表述为“无FP32 class prototype/weight sidecar”。formal Phase1/LODO authority缺失时保持硬阻断。

第一次复审发现公开融合入口接受任意caller probability，且query MAC漏算INT8权重解码。修复后唯一融合入口接收并验证exact typed D99 bank，内部调用D99 canonical raw scorer与K-specific锁温度softmax；`alpha=0`也内部生成D99概率并确认完全跳过ridge。C26/K20 query上界为：rank4 D99 1,732,160＋D100增量18,014=1,750,174 MAC；rank8 D99 2,397,760＋D100增量19,298=2,417,058 MAC。

终审裁决为`ACCEPT_LOCAL_D100_BLOCKED_AUTHORITIES`，专项`15 passed`、D99+D100联合`38 passed`。rank8已知bytes小计为D100 16,268B＋D99 163,810B＋typed D81 exact head 35,746B＋ground裸数组13,860B=229,684B；但ground完整serializer、registry/resource receipt及D99/D100 LODO和external authorities未固定，所以`complete_combined_state_upper_bound_available=false`、`formal_combined_resource_claim=false`，不得用该小计宣称正式低于256KiB，也不得运行N607/target/确认矩阵。

## typed D81提交907bd620独立复审：REVISE

commit`907bd620`的专项与相关联合测试共`76 passed`，但独立复审发现其核心fit改变了历史D81定义：当前实现把全部registered old/new support送入20-step metric，而真实D81只用`Y_old` support拟合一次metric，冻结后再用全部old/new support拟合最终D62/D81 head。四类独立oracle攻击结果如下，差异远大于`2e-6`，属于方法改变而非浮点噪声。

|K-shot|`log_diag`最大差|query logit最大差|判定|
|---:|---:|---:|---|
|1|0.197202|0.272931|FAIL|
|5|0.270690|0.059278|FAIL|
|10|0.304465|0.170045|FAIL|

此外当前support receipt仅对调用者提供的raw arrays/labels/physical ID字符串自签，未绑定`VALIDATED_ONCE`、`capsule_id`、`split_id`、sealed row/support artifact、ordered registration state和feature-runtime/checkpoint receipt；`from_scorer()`也可接受development伪component。源码自hash只能发现漂移，不能授予外部数据权限或method-lock真实性。

资源审查同样不闭合：没有真实wire serializer；logical/serialized未覆盖完整audit和receipt；peak遗漏support副本、registered feature、Torch optimizer/梯度状态及LDA内部临时量；逐query MAC遗漏raw288注册几何、归一化和INT8 decode。更严重的是每次score前后都重新读取并hash依赖源码、numeric arrays和完整audit/resource，这些文件I/O、SHA和序列化成本完全未计入query路径。

修复链已冻结为：typed lifecycle artifact同时绑定old support、all-registered support、old/final registry、capsule/split/row receipt和Phase1 method-lock；只用old support拟合20-step metric，冻结后分别形成before old head与final all-class head；实际wire save/load报告完整bytes；external authority/dependency只在load-time验证一次，query只执行`typed immutable state + raw288`纯前向。至少覆盖2old+2new的K1/K5/K10独立oracle，以及6old+5/10/20new registry形状、固定old改变new不改metric、K1 fallback、伪authority/内部重签攻击、serializer往返和load后无文件I/O。修复前D97/D99均不得接入该state，更不得运行N607 target narrow或125。

### typed D81修复版独立复审：仍为REVISE

修复作者已将fit改为old-only metric与before-old/all-final head，并在字典序fixture上得到K1/K5/K10 oracle差异全0；专项`24 passed`、D42+D81 scorer+typed+D99联合`76 passed`。但独立审查用非排序sealed payload证伪严格等价：typed内部按class+physical ID重排，而历史D81保持capsule payload order；固定噪声由此绑定到不同物理行。

|K-shot|`log_diag`最大差|before logit最大差|final logit最大差|
|---:|---:|---:|---:|
|1|0|0|0|
|5|0.171410412|0.002393246|0.001184523|
|10|0.117926039|0.002886534|0.005068243|

历史D81等价优先，external row receipt必须显式绑定ordered support row/physical/feature root；fit保持该顺序，old/all中的old行逐项同序同bytes。输入重排应因row receipt不匹配被拒绝，不能在typed内部另行canonical排序后仍声称D81。

资源也被独立下界证伪。2old+2new K1报告fit 49,180,800 MAC，而真实D46/D62 inventory下界为97,731,960 MAC，约低报1.99倍；6old+20new K10报告101,358,592 MAC，而88个LDA component fit、D62 OOF/fisher/reliability/gate、rank21 translation、metric/geometry合计下界为11,835,007,168 MAC，约低报116.76倍。peak同样未覆盖最大D62/sklearn临时生命周期。正常wire fixed point与query MAC 8,772/40,474本身通过复算。

最后，将query/fit MAC与peak改为0或将metric seed从713101改为713102，再重算wire和artifact SHA，load/verify/query仍可成功，说明resource/config仍只受内部自签保护。下一版必须从numeric state、维度、真实D46/D62 call inventory及外部Phase1/scorer lock独立推导，任何内部重签不得绕过。正式capsule producer依然不存在，所以即便local exact-fit修复通过，状态也只能保持`LOCAL_CORE_PENDING_EXTERNAL_CAPSULE_PRODUCER_AND_REVIEW`，formal query入口必须拒绝local-fit state。

第三轮独立复审最终裁决为`ACCEPT_LOCAL_EXACT_FIT_PENDING_EXTERNAL_PRODUCER`。实现删除内部support排序，严格保持external sealed payload order；非排序K1/K5/K10的`log_diag/before/final`与独立历史D81 oracle最大差全部为0，输入重排由ordered row receipt在fit前拒绝。D46/D62完整资源固定点为：2old+2new K1共4个component fit、总fit 97,872,856 MAC、peak 24,272,503B、query 8,772 MAC；6old+20new K10共88个component fit、总fit 11,835,222,784 MAC、peak 142,162,891B、query 40,474 MAC。resource归零重签和seed重签均被外部lock/dimension inventory拒绝。

外部formal producer仍不存在，因此两个formal query API在函数入口无条件fail-closed，不读取任何state字段；4个authority/config/state dataclass均为frozen+slots且无`__dict__`，copy/replace/字段替换不能打开正式路径。wire实际固定点锁为K1 17,743B与6old+20new K10 35,706B，旧35,709B是作者预报误差，没有人为padding。专项`30 passed`，13文件直接依赖联合实际收集`141 passed`。该裁决证明local exact-fit、wire和资源闭合，不授予external capsule authority，仍禁止N607、target narrow、125或正式性能声明。

### method-free Phase2 data authority producer

扩展元数据搜索没有找到真实同时包含`VALIDATED_ONCE+p2_min_v1+capsule_id+split_id`的receipt。现有D18 cache、D81 matrix和SOMP-H envelope可证明单LEO弱观测、固定received-IQ、registry prefix及若干物理不复用事实，但其schema和formal范围不足，不能直接冒充typed D81 row authority。因此新增metadata-only producer，只读取predictor manifest、detached seal、offline audit和data-validation COMMIT四个控制JSON，不打开support/query/cache/truth payload，不重建IQ或split。

第一次复审发现总support数不能证明逐类exact-K、Stage2-B registry root逻辑错误及capsule/split职责混叠。修复后每个scenario按final registry闭合逐类count和physical root，并逐类强制`count==K`；Stage2-B要求old/final registry与root相同，Stage2-C要求strict prefix且root不同。`capsule_id`只绑定receiver、scenario、role-agnostic physical/received-IQ内容根和单观测事实，不含seed、K或support/query角色；`split_id`绑定capsule、K、method-free registry identity和角色化physical split，不含seed、token或模型/package句柄。实际`Path.open`守卫证明writer仅读取四个控制JSON。

第二轮复审裁决为`ACCEPT_LOCAL_PRODUCER_BLOCKED_SIGNED_UPSTREAM`；主线专项`10 passed`、相邻bundle/builder联合`46 passed`。输出固定为`UPSTREAM_COMMIT_BLOCKED_UNSIGNED_NOT_FORMAL`与`UNVERIFIED_UNSIGNED`，上游VALIDATED_ONCE只记录为未验证声明，formal data/launch/metric全部为false。local producer可进入Git，但真实signed upstream COMMIT/envelope出现前不得接typed formal consumer、N607 target或性能声明。

## D99/D100实验优先发布交接（2026-07-21）

本节取代上文把额外签名、authority envelope或数据握手作为无线信号模型development实验前置门的旧结论。用户已明确该工作是无线信号域适应、分类头和类增量学习研发，与网络安全任务无关；本轮只执行现有`p2_min_v1`科学协议和实验约束。Phase1 LODO不访问target；后续target narrow复用匹配`VALIDATED_ONCE`数据，query保持逐样本只前向。development结果必须标为nonformal，但不得再因缺少额外代码签名或重复数据握手而阻断算法实验。

### 不可变发布身份

|字段|值|
|---|---|
|实验ID|`d99_d100_phase1_inputs_aa3a0266_20260721_r1`|
|登记时间|`2026-07-21 01:34:31 +08:00`|
|operator|主线Codex；N607唯一launch owner为`n607_d99_d100_release_prep`子agent|
|目标|导出固定单LEO弱观测Phase1特征archive并构建7 receiver×6 class的D99地面聚合bundle，为K1/K5/K10/K20 receiver-level pseudo-target LODO冻结D99/D100参数|
|比较目标|D81 exact Phase1 episode scorer；D99相对D81；D100相对D99|
|Git提交|`aa3a02662b5609d633fdf2ce1bcfde4c3bab0efb`（`Add D99 D100 Phase2 experiment pipeline`）|
|本地Git状态|登记时clean，分支`codex/ground-prototype-da-rd`|
|源码归档|`E:\type10-7\code\snapshots\d99_d100_phase1_inputs_aa3a0266_20260721_r1\source_aa3a0266.zip`|
|源码归档SHA256|`5185e2847b09191419ea58c010c214e6954faea8c6ebde31880dc39e1bc4640c`|
|源码归档规模|31,121,736B；4,314个Git成员|
|selection salt|`preregistered_inputs/d99_d100_phase1_selection_salt.json`|
|selection salt SHA256|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|
|状态|`LOCAL_VERIFIED`；尚未声称`LANDED`、`RUNNING`或性能收益|

本提交涉及D99/D100 LODO、query adapter、ground builder、development exporter、canonical D100融合及对应测试共18个文件。主线合并回归为`86 passed`，wrapper/formal定向回归为`14 passed`；LODO独立审查51/51、query adapter联合89/89、builder/exporter独立13/13，均为exit0。测试只证明实现一致性，不代替真实Phase1或target性能证据。

### N607固定输入

|输入|远端路径|预期SHA256|
|---|---|---|
|source-validation cache set|`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json`|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|
|ADV3B02 runtime|`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt`|`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`|
|ADV3B02 checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；由selection salt和LODO config逐字绑定|
|D19 ground component|`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`|manifest `15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`；NPZ `3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|Torch`2.1.0+cu121`|

远端不可覆盖根为`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1`，日志根为`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_inputs_aa3a0266_20260721_r1`，隔离源码根为前者下的`source_aa3a0266`。runner必须先执行规定的`tools/n607_ssh_preflight.ps1`，复核GPU/进程和所有输入SHA，确认三个远端根均不存在后才可创建；若任一路径已存在则停止，不覆盖、不续写、不自动换ID。

### 预登记child command

Phase1 exporter固定GPU4、batch size256：

```bash
env PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/source_aa3a0266/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/source_aa3a0266/code/scripts/export_phase1_singleobs_feature_archive.py --mode development --cache-set /home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json --cache-set-sha256 125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74 --runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --expected-runtime-sha256 f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a --class-ids 14-10,14-7,20-15,20-19,6-15,8-20 --selection-salt-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/input/d99_d100_phase1_selection_salt.json --selection-salt-receipt-sha256 38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/phase1_feature_archive --device cuda:4 --batch-size 256
```

ground builder固定读取D19聚合组件，不读取clean/raw IQ：

```bash
env PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/source_aa3a0266/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/source_aa3a0266/code/scripts/build_d99_receiver_ground_bundle.py --source-component-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component --source-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle
```

两个child均由runner用不可覆盖的`.pid/.exit/.log`包装为detached job；实际wrapper命令、PID、CWD、GPU、开始/结束时间必须回填本报告。预期exporter产物为`phase1_singleobs_feature_archive.npz`与`phase1_singleobs_feature_archive.manifest.json`；builder产物为`d99_ground_bundle_dev.npz`、`d99_ground_bundle_dev.manifest.json`、`d99_ground_aggregation_spec.json`、`d99_base_method_lock_dev.json`和`build_result.json`。所有产物完成且内部复核通过后才生成绑定真实产物SHA的LODO config；不得提前猜写archive/bundle SHA。

### 成功、停止和后续准入

- 第一阶段成功：源码、salt和固定输入SHA一致；远端隔离源码`py_compile`通过；两个child exit0；输出目录完整且manifest/NPZ互相校验；短连接结束后本地SSH连接清零。
- 第一阶段停止：输入SHA漂移、目标路径已存在、GPU占用超过每GPU两训练任务、脚本异常、产物不完整或内部校验失败。不得覆盖、重启或修改候选参数。
- LODO固定比较K={1,5,10,20}。D99必须相对D81同时满足worst floor不降、balanced NLL改善`>1e-6`、双向rescue非零，K1还必须产生非identity预测；D100必须相对D99在每个receiver×pseudo-new pair的floor/old/new/H均不降、balanced NLL改善`>1e-6`且双向rescue非零。
- LODO不合格的K不得用target补选。合格后只发布receiver20-1、seed713101的K1/new20与K10/new20 matched narrow；每job内部同row输出D81/D99/D100。窄实验通过后才进入历史125 screen，再根据完整目标要求进入400 job/1,200 scenario row确认矩阵。
- 这一阶段没有target性能指标。任何启动、完成或测试通过都不能写成性能成功；每个完成版本必须随后补齐B/A/min-old/min-new/min-all/New/H/F、逐场景、逐receiver、逐类、混淆、量化和资源表，并详细解释缺陷。

### 第一阶段实际结果：exporter schema不兼容，builder完成

唯一runner已完成一次且仅一次落地，状态链为`LOCAL_VERIFIED→LANDED→RUNNING→STOPPED_EXPORTER_INPUT_SCHEMA_MISMATCH`，未达到`ARTIFACTS_COMPLETE`。2026-07-21T01:38:18+08:00规定preflight通过；8张RTX3090均0%利用率、10MiB且无compute app，所有固定输入SHA匹配。源码4,314成员安全解压、目标脚本`py_compile/import`通过。实际PID为exporter`1390812`/GPU4、builder`1390814`/CPU；两者均在隔离源码CWD运行，结束后PID消失、GPU4恢复0%/10MiB，SSH和N607/bridge TCP22连接均为0。

|child|exit|完成内容|缺陷/结论|
|---|---:|---|---|
|Phase1 exporter|1|在cache loader入口完成外层manifest读取|固定source-validation cache为`cvs_leo_weak_iq_cache_set_v1`，代码只接受`..._v2`，报`LEO cache-set manifest contract failed: ['schema']`；未进入runtime前向，未创建archive|
|D99 ground builder|0|7 receiver×6 class bundle、aggregation spec、base lock、manifest、result五文件齐全，typed load和lock digest复核通过|development/nonformal；mean/min requantization cosine为0.9999880195/0.9999678135，说明量化重构稳定，但尚无分类或transport性能|

builder实际SHA为：NPZ`e69409268ad1215d440fd0555a4f8e2903214e95062f22a0c5f436fb2b799bd4`，manifest`f92a1bd6e936c4e661517da73ee39cc5e94f77b796f50587dd33f6c0d2da8f0d`，base lock`7481c351a3114432df0c06c8527b1f625c15c05b1e3aed89000098b26f022a9e`，aggregation spec`f4db8091aeb7204bb4a641d02810c19acbb9fc002ec8e82c82fd9f4fe2820efe`，result`b9c0213f9dbbd2f438ab31541ac4404d852388bd42fbe439b9a768fbf87714c2`。typed bundle SHA为`78904ed3c569f79815021ef76cc9e46bb42fd0a06c7bb1e91483237e2606ce78`，narrow loader lock digest为`9d12c638176e0bc7dfa2d27664f737e606ec2064eb8d280a9a059d91c2122063`。

完整wrapper、child command、PID/GPU/CWD、输入/输出SHA和异常证据见`artifacts/d99_d100_phase1_inputs_aa3a0266_20260721_r1/runner_handoff.md`，handoff SHA256为`594cba875bd47dc491d557732c00f4ebd27442f629e5a3f4dfdc14281cdef184`。本run没有target性能指标，不能给出old/new/H/floor/forgetting，也不能启动LODO；四个feature archive字段明确为`UNAVAILABLE_EXPORTER_EXIT_1`。runner没有重启、转换schema、改参数、换run ID或访问target。

修复范围冻结为：默认/formal cache loader继续只接受v2；仅development SHA-only exporter显式接受实际v1 outer+inner schema，其他manifest字段、NPZ成员、role、physical ID、IQ hash、overlay和cache SHA验证不放宽。修复必须本地测试、提交、重新预登记并使用全新不可覆盖run ID，禁止向本run补写。

#### v1精确兼容修复

修复只修改`leo_weak_cache.py`、Phase1 exporter及其测试。底层loader新增显式且受限的outer/inner schema集合参数，默认值仍是v2-only；空集、字符串冒充集合和未知schema全部拒绝。formal与runtime-manifest development入口继续使用默认v2-only。只有SHA-only development入口使用内部固定的v1-only loader，并在任何cache文件读取和loader调用前要求cache-set SHA精确等于已冻结历史值`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`。因此第二份自洽v1、v2重标v1或重算全部inner/outer SHA均不能到达loader。

通过SHA门后仍逐项执行原有NPZ成员allowlist、forbidden member、role、IQ digest、physical ID、overlay ID、inner SHA、场景顺序和cross-scenario检查。输出manifest保持`DEVELOPMENT_PHASE1_TEMPORARY_ASSET`/nonformal，并记录outer/inner observed schema、legacy compatibility、original SHA和authority SHA，后两者必须相等。

独立首审发现“仅锁schema但未锁历史lineage”的P1并以另一份完整有效v1cache攻击证实；加入固定SHA allowlist后终审为`MERGE`，P0=0、P1=0，聚焦攻击2/2通过。作者聚焦及相邻25/25通过；主线在`ssr-gpu`直接Python环境复跑exporter与cache matrix得到31/31通过，`py_compile`和`git diff --check`均exit0。附带信息仅为TorchScript弃用告警和pytest退出后的Windows临时目录`WinError5`清理噪声，主体exit0。

最终文件SHA256：`leo_weak_cache.py`=`851ceeaacf8146e4c7e480d22278df9914db6051eb63c50032f9471f66d28b86`；exporter=`ab4d3c40251f2bd147e7948ced392d185d0ef7b3f45c18924e7ab1bd457dac6d`；测试=`d988183fd7a53febbf9c89663d3228c5e413321d2b77b12e24f5d07513d06499`。该修复只消除实际固定v1缓存与v2-only consumer之间的兼容缺陷，不改变缓存内容、selection salt、runtime、ground bundle、D99/D100公式或候选网格，也没有访问target。

#### r2 exporter重新发布预登记

|字段|值|
|---|---|
|run ID|`d99_d100_phase1_export_391f51ed_20260721_r2`|
|代码提交|`391f51ed`（`Support frozen v1 Phase1 cache export`）|
|本地源码包|`E:\type10-7\code\snapshots\d99_d100_phase1_export_391f51ed_20260721_r2\source_391f51ed.zip`|
|源码包SHA/规模|`faad85ee50b83353015dbba51653b33975985296d0930e6ee0b2635897ff236e`；31,150,916B；4,331成员|
|远端run根|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_391f51ed_20260721_r2`|
|远端log根|`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_export_391f51ed_20260721_r2`|
|隔离源码根|`<run>/source_391f51ed`|
|GPU/CPU|exporter固定GPU4、batch256、CPU thread2；不重跑builder|
|复用产物|只读复用r1 ground NPZ/manifest/base lock及其已回收SHA，不向r1写入|
|状态|`LOCAL_VERIFIED`，尚未LANDED/RUNNING/完成|

r2只重跑上轮在loader入口失败的exporter，selection salt、source-validation cache、runtime和所有SHA不变。runner必须重新执行规定preflight，确认r2 run/log/source均不存在、GPU占用合规，复核源码包、salt、cache、runtime、checkpoint及r1 ground产物SHA；任一漂移即停止。child command冻结为：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_391f51ed_20260721_r2/source_391f51ed/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_391f51ed_20260721_r2/source_391f51ed/code/scripts/export_phase1_singleobs_feature_archive.py --mode development --cache-set /home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json --cache-set-sha256 125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74 --runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --expected-runtime-sha256 f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a --class-ids 14-10,14-7,20-15,20-19,6-15,8-20 --selection-salt-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_391f51ed_20260721_r2/input/d99_d100_phase1_selection_salt.json --selection-salt-receipt-sha256 38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_391f51ed_20260721_r2/phase1_feature_archive --device cuda:4 --batch-size 256
```

采用不可覆盖`.pid/.exit/.log`detached wrapper；首个health check约90秒，若进程提前退出则先读完整日志，不自动重启。成功要求exit0、两个archive文件齐全、内部verify通过、manifest明确记录outer/inner v1、legacy=true、original/authority cache SHA均为`125bb312...d74`、runtime SHA为`f119...e2a`，并回收文件SHA与行数/receiver/day/class/scenario分布。r2完成后才生成LODO config；不得用r2 target性能选择任何参数，因为r2不读取target。

#### r2实际结果与真实v1成员合同修复

r2状态链为`LOCAL_VERIFIED→LANDED→RUNNING→FAILED_DIAGNOSTIC`。preflight、GPU空闲、固定输入、r1 ground五文件、ZIP整体SHA及本地ZIP成员↔远端解压成员均通过；Windows worktree CRLF SHA与Git archive LF成员SHA的差异已按两种口径记录，不是执行代码漂移。exporter PID`1414309`于02:38:12启动、02:38:14结束，exit1；GPU4恢复0%/10MiB。没有archive、重启、builder、LODO或target访问，SSH/连接终态均为0。

失败原因是首轮兼容测试把v1构造成“v2成员＋v1 schema”。真实v1 NPZ具有17个历史成员，不含v2新增的`source_dataset_sha256`和`source_record_indices`；当前loader在schema分支前仍共用v2 required members，因而报缺少这两项。完整handoff见`artifacts/d99_d100_phase1_export_391f51ed_20260721_r2/runner_handoff.md`，SHA`69d9973979a85635881ba5a9e676242688f8208a2b0c8d732ccac80738b39b55`；结构审计见同目录`v1_cache_structure_audit.md`，SHA`a34f295fee180baba6d748182d35aec082b5684187a7276076abcd8537bb68bf`。

只读结构审计证明三个NPZ均为8400行、精确相同17-member顺序；v1 physical ID可由现有`dataset_role|tx|rx|day|eq|sig`逐行复算，每场景8400个ID均唯一，三场景顺序相同，root=`d2def2acf96a9338f94b4626f77ca9b7b106a65f41615dd5c703b1b76461e1a3`，与inner manifest、outer manifest和cache audits一致。IQ内容未下载；每个NPZ的冻结SHA保护内容，原存`post_channel_iq_sha256`和overlay root可继续由生产loader逐行复算验证。

第二次本地修复直接对照升级前Git实现`454c1a61^:code/cvsrffi/leo_weak_cache.py`，恢复schema-specific合同：v1精确17-member、旧physical ID和5字段provenance；v2继续19-member、dataset SHA/index身份和7字段provenance。loader先只读取`manifest_json`确定已允许schema，再检查全部forbidden/required成员，之后才物化IQ；v1/v2都逐行重算sample ID、IQ digest、overlay ID、唯一性和outer root。audit按schema记录root policy，v1不再错误声称v2`immutable_preoverlay_lineage_token`。SHA-only固定cache allowlist、formal/普通development v2-only均保持不变。

作者聚焦13/13＋相邻36/36共49/49通过；独立终审`MERGE`，P0=0、P1=0，聚焦攻击6/6通过。主线复跑exporter、cache matrix和D96/D97相邻回归得到48/48通过，`py_compile`与`git diff --check`exit0；仅有既有TorchScript弃用告警和pytest退出后临时目录`WinError5`噪声。最终loader/test工作树SHA为`3fc35aeea182560fc67cd468a7615ca110b528ca210327c0620370d1b68606fb`/`9b3dd452b195b409b922442d4ec7fcadd7ba1de2bbce104fc8c3453f7f2e2be8`；exporter继续为`ab4d3c40251f2bd147e7948ced392d185d0ef7b3f45c18924e7ab1bd457dac6d`。

#### r3真实v1 exporter发布预登记

run ID固定为`d99_d100_phase1_export_7932dbf9_20260721_r3`，代码提交`7932dbf9`，源码包为`E:\type10-7\code\snapshots\d99_d100_phase1_export_7932dbf9_20260721_r3\source_7932dbf9.zip`，SHA256=`40a2ecc9d01a12759d9a67693d7eaa974751bae02c05faab0dabfc580efdbd72`，31,168,653B、4,342成员。远端run/log根分别为`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_20260721_r3`和`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_export_7932dbf9_20260721_r3`，隔离源码为`<run>/source_7932dbf9`。三者必须创建前不存在；不得补写r1/r2。

固定输入、salt、runtime、cache、checkpoint、class顺序、GPU4、batch256和CPU thread2与r2完全相同；继续只读复用r1 ground五文件。唯一child command为：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_20260721_r3/source_7932dbf9/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_20260721_r3/source_7932dbf9/code/scripts/export_phase1_singleobs_feature_archive.py --mode development --cache-set /home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json --cache-set-sha256 125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74 --runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --expected-runtime-sha256 f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a --class-ids 14-10,14-7,20-15,20-19,6-15,8-20 --selection-salt-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_20260721_r3/input/d99_d100_phase1_selection_salt.json --selection-salt-receipt-sha256 38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_20260721_r3/phase1_feature_archive --device cuda:4 --batch-size 256
```

成功门与r2相同，并额外要求真实v1每scenario 17-member合同通过、每scenario 8400行、selection后每个physical ID只保留一个固定scenario。若出现新异常仍不自动重启；回收完整日志和产物后再决定LODO。当前状态`LOCAL_VERIFIED`，没有新性能结果。

#### r3实际结果与r4 GPU0重发预登记

r3状态`LOCAL_VERIFIED→LANDED→RUNNING→FAILED_DIAGNOSTIC`。真实v1三个NPZ的17-member、每场景8400行、sample ID、IQ digest、overlay ID和outer root全部通过生产loader，证明`7932dbf9`的v1修复在真实N607资产上成立。失败推进到首批TorchScript forward：历史f119 runtime内部卷积weight固着`cuda:0`，冻结命令把input放在`cuda:4`，PyTorch报设备不一致。PID`1427919`，03:07:05→03:07:08，exit1；没有archive、重启、参数修改或LODO，GPU4与SSH终态均已释放/清零。完整handoff SHA为`1cef738a544d311a7dfbebe9f1f846f6f3b78456c80887b9dc709afbac7feee4`。

该异常不是模型算法、数据协议、v1内容或GPU不可用问题。`torch.jit.load(...,map_location=cuda:4)`未改写runtime内部固着的device常量；最小运行修复是在其原生`cuda:0`上同时加载weight和input，不修改runtime bytes、代码、batch、selection或特征公式。

r4 run ID冻结为`d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4`，继续使用提交`7932dbf9`和同一源码包SHA`40a2ecc9d01a12759d9a67693d7eaa974751bae02c05faab0dabfc580efdbd72`。远端run/log/source分别为`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4`、`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4`和`<run>/source_7932dbf9`；创建前必须不存在。所有固定输入及SHA与r3相同，唯一预登记差异是`--device cuda:0`并分配物理GPU0。

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code/scripts/export_phase1_singleobs_feature_archive.py --mode development --cache-set /home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json --cache-set-sha256 125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74 --runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --expected-runtime-sha256 f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a --class-ids 14-10,14-7,20-15,20-19,6-15,8-20 --selection-salt-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json --selection-salt-receipt-sha256 38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/phase1_feature_archive --device cuda:0 --batch-size 256
```

r4成功门沿用r3，并新增日志中model/input同为cuda0、至少完成全部33个batch或等价8400行前向、输出288D有限值和reference logits有限值。失败仍不自动重启。当前没有target性能结果。

### r4 Phase1特征归档完成与r5 LODO预登记

r4状态链为`LOCAL_VERIFIED→LANDED→RUNNING→ARTIFACTS_COMPLETE`。固定child在GPU0以PID`1436954`运行，时间`2026-07-21 03:25:52→03:26:04 +08:00`，exit0。执行使用提交`7932dbf9`的隔离源码、原始f119 runtime和唯一变化后的`--device cuda:0`；没有修改runtime、cache、selection、batch或特征公式，也没有访问target。

|检查项|r4实际证据|结论|
|---|---|---|
|物理样本/单观测|8400个唯一physical ID、每ID严格1行；scenario分布2852/2820/2728|通过；未用同一IQ多信道重放|
|特征|`features_fp32=[8400,288]`、全有限|通过|
|参考logit|`reference_logits_fp32=[8400,6]`、全有限|通过|
|archive成员|严格8项；生产internal verify通过|通过|
|cache lineage|outer v1、3个inner v1、legacy=true；original=authority=`125bb312…b8d74`|通过|
|runtime|requested/resolved device均为cuda0；SHA=`f119e8cb…e2a`|通过|
|archive NPZ|`phase1_singleobs_feature_archive.npz`，SHA=`cdd8747d267336b48e8c555ce7e010206f042ff07c695af351541a97187fad03`|完整|
|archive manifest|`phase1_singleobs_feature_archive.manifest.json`，SHA=`5f363bc09503f882c66aa92805657199ca57484f627c5d2805254cad07bffa15`|完整|

r4只完成合法Phase1特征导出，没有D81/D99/D100分类结果，因此没有old/new/H/floor/forgetting性能可报告，不能写成方法收益。完整wrapper、输入输出SHA、分布和SSH清零证据由唯一runner回收到`artifacts/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/runner_handoff.md`，handoff SHA256=`b62257569baba75e537627156e7eeae13bbcc236ab2d4fae6b743037261c4816`。最终状态为`ARTIFACTS_COMPLETE→ANALYZED`，SSH连接0/0。

r5不可变run ID为`d99_d100_phase1_lodo_7932dbf9_20260721_r5`；冻结配置位于`preregistered_inputs/d99_d100_phase1_lodo_7932dbf9_20260721_r5.json`，SHA256=`6df14d7d17fc4fe9d3eb786001eeeedcf92d640aa190d6974a60793bf6b8e30e`。该配置逐字绑定r4 archive、r1 ground bundle、D19 D81 ground component、ADV3B02 checkpoint、5个模块SHA、seed991、metric seed713101和64候选网格。执行模式固定为`development_diagnostic`，不得产生formal声明。

候选网格只在7个地面receiver之间执行nested receiver LODO，覆盖K={1,5,10,20}，不读取target。参数规模为2×1×1×2×1×1×1×2×2×2×2=64。准入规则保持不变：D99相对D81要求floor不降、balanced NLL严格改善、双向rescue非零且K1非identity；D100相对D99要求每receiver×pseudo-new pair的old/new/H/floor均不降、balanced NLL严格改善和双向rescue非零。失败K档不得进入target。

r5固定使用物理GPU5，通过`CUDA_VISIBLE_DEVICES=5`向child暴露单卡并以配置中的`d81_device=cuda`运行；CPU线程均为2。唯一命令为：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code/scripts/run_d99_d100_phase1_lodo.py --config /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_7932dbf9_20260721_r5/input/d99_d100_phase1_lodo_7932dbf9_20260721_r5.json --config-sha256 6df14d7d17fc4fe9d3eb786001eeeedcf92d640aa190d6974a60793bf6b8e30e
```

runner必须先做规定preflight，确认r5 run/log/output均不存在、GPU5占用合规，复核配置、r4 archive、r1 ground和D19 manifest SHA，验证隔离源码的5模块SHA与配置一致；随后同步配置到不可覆盖input路径，单次detached启动并回收完整receipt。不得改候选、自动重启、使用target调参或同时启动同run ID。

#### r5启动前SHA口径诊断与r6重发预登记

r5在任何远端状态创建前由代码SHA硬门安全停止。r5配置中的5项SHA来自Windows工作树CRLF字节，而规定复用的r4源码来自Git archive的LF字节；模块逻辑版本相同，但原始字节SHA必然不同。r5 run/log/output均未创建，配置未上传，child未启动，GPU5及r4/r1/D19资产门均通过，SSH连接0/0。完整prelaunch handoff位于`artifacts/d99_d100_phase1_lodo_7932dbf9_20260721_r5/runner_handoff.md`，SHA256=`d65dd2617bb0d138c328bbc6a9fae3228aa5c9dcc10fa0909210044c2e28cfc8`。r5没有分类、LODO或性能结果。

不修改已提交r5配置，另行冻结新run ID`d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6`。r6配置位于`preregistered_inputs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6.json`，SHA256=`8a63aaab10dceb05811af8f0aa82bfc76e45468e2603beffd419092eebe949f6`。除run ID、output路径和5个模块SHA口径外，seed、候选网格、archive、ground、checkpoint、GPU与所有方法参数均与r5逐字相同。5个LF SHA已直接从原始`source_7932dbf9.zip`成员复算，并与N607 r4解压源码一致：

|模块|r6冻结LF SHA256|
|---|---|
|`run_d99_d100_phase1_lodo.py`|`110295caa83ab0d7717e26b17b1d4ac33423337afaa8877067f64649d06c7ea1`|
|`stage2_d100_ra_cgspr_lgf.py`|`86c185ee13222bc0c97c4576984b9cd07f981201da4f0b62f8d4bc66970b4714`|
|`stage2_d81_phase1_episode_scorer.py`|`54ee742c81b60e00b6c1c36d2d6bf1f0409ad10f72a25e01c2dcd589093be55d`|
|`stage2_d99_d100_phase1_lodo.py`|`aa99b3d726338481ed7f22f4acc5cdf2cfe4b2ef420e44da6f2ff2f674841e0e`|
|`stage2_d99_ra_cgtmk_d81.py`|`c166a5e375b0b8be5c95e678e63a6f04526474cd1a01544616829106af52f56f`|

r6远端run/log根分别为`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6`和`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6`。唯一child命令为：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code/scripts/run_d99_d100_phase1_lodo.py --config /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/input/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6.json --config-sha256 8a63aaab10dceb05811af8f0aa82bfc76e45468e2603beffd419092eebe949f6
```

r6仍由同一唯一runner发布，不授权自动重试。成功、停止、LODO准入和target禁止条件与r5完全相同。

#### r6实际结果：D81无索引CUDA设备失败

r6状态链为`LOCAL_VERIFIED→LANDED→RUNNING→ARTIFACTS_COMPLETE_DIAGNOSTIC_FAILURE_NOT_ANALYZABLE`。所有preflight、GPU5、配置、r4 archive、r1 ground、D19和5个LF模块SHA门均通过；唯一child PID`1447845`，exit1，未重试。失败发生于首个LODO episode进入D81 metric fit时：release schema允许并冻结`d81_device="cuda"`，`torch.device("cuda")`的index为None，而历史D42调用`torch.cuda.set_device(device)`要求显式索引，报`ValueError: Expected a torch.device with a specified index or an integer, but got:cuda`。

r6没有完成任何K1/K5/K10/K20候选，也没有D81/D99/D100的old/new/H/floor/NLL/rescue或资源指标，因此不能判定方法正负。output目录未生成；GPU5恢复0%/10MiB，远端PID退出，SSH进程/连接均为0。完整handoff位于`artifacts/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/runner_handoff.md`，SHA256=`b711dcf7689a21a1fb7f22f65cd9bb3ca9eb913821924df16a2f3c2cf43bd827`。

本地最小修复位于`stage2_d81_phase1_episode_scorer.py`：新增纯正规化函数，仅当CUDA device缺少逻辑索引时映射为`cuda:0`，CPU和显式索引设备保持不变；外部release schema、D81 receipt的配置字符串、metric seed、方法公式和候选均不改变。fake-metric回归证明冻结`cuda`实际传给D42前变为`cuda:0`且不需要本机GPU；D81 scorer与LODO runner相邻回归`12 passed`，`py_compile`和`git diff --check`均exit0。pytest退出后的Windows临时目录`PermissionError`为既有清理噪声，主体exit0。

#### r7 CUDA正规化源码重发预登记

|字段|冻结值|
|---|---|
|run ID|`d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7`|
|源码提交|`d6efa5ad`（`Normalize D81 CUDA device for LODO`）|
|本地源码包|`E:\type10-7\code\snapshots\d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7\source_d6efa5ad.zip`|
|源码包SHA/规模|`701e124fbb53046c9361995f2a6141841853153eabaef2d6865ce3187bd5b82a`；31,216,534B；4380成员|
|冻结配置|`preregistered_inputs/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7.json`|
|配置SHA|`3241eb36d4f774f6e3751af7f7682060ce0a0e8204de18227870c133cebdb4e2`|
|GPU/CPU|物理GPU5；`CUDA_VISIBLE_DEVICES=5`；CPU thread2|
|状态|`LOCAL_VERIFIED`；尚未LANDED/RUNNING|

源码包已在独立本地目录完整解压，并在`ssr-gpu`环境中从该LF源码直接运行release config validator：精确schema通过、64候选通过、5个模块SHA与配置逐项一致。r7不改变r6的archive、ground、checkpoint、seed、候选网格、准入规则或target禁止条件；唯一代码变化是无索引CUDA设备在D81调用D42前正规化为逻辑`cuda:0`。

r7远端run/log/source分别为`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7`、`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7`和`<run>/source_d6efa5ad`。唯一child命令为：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7/source_d6efa5ad/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7/source_d6efa5ad/code/scripts/run_d99_d100_phase1_lodo.py --config /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7/input/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7.json --config-sha256 3241eb36d4f774f6e3751af7f7682060ce0a0e8204de18227870c133cebdb4e2
```

唯一runner必须重新direct preflight，确认r7 run/log/source/output均不存在、GPU5占用合规，上传源码ZIP和配置并复核整体/成员SHA，安全解压后再运行`py_compile/import/config validator`。只允许一次不可覆盖detached启动，不授权自动重试；若成功则读取完整LODO receipt并逐K报告所有指标与缺陷，若失败则保留完整诊断且不能写成方法负结果。

### D101直接shrinkage RDA交叉审查

D101定位为D99上的alternative global head，与D100二选一，不叠成第三个融合头。当前裁决为`REVISE`：只允许后续实现Phase1 nested-LODO诊断臂，尚不允许target或N607。原因不是协议违法，而是其相对D99 metric和D100 simplex ridge的独立纠错能力尚未被证实。

冻结结构使用D99 metric-sqrt后的288D三block特征。所有old/new类均值只来自同row support，使用同一共享协方差、等先验、温度和判别式。K1不估计target协方差方向，只用sealed三block各向同性先验和coverage-gated ground共享谱；K>1只估计3个block residual scalar及最多2个target residual方向，Woodbury小逆rank≤6，0 optimizer、0 query update。C=26时D101增量INT8核心payload约7.75KB，已复用D99特征时增量query约7,514MAC；实际wire、共享D99状态、完整fit peak与INT8 margin审计仍须实现后实测，不能把payload估算写成正式资源结论。

共同可逆变换后若完整重估均值和协方差，LDA margin严格不变；因此D101不得把“公共对齐”声称为收益。唯一可能产生不同决策的机制是固定ground prior、三block投影、rank≤2非可逆截断、shrinkage/ridge与量化。每个K在Phase1必须相对D100有非零disagreement、双向rescue各≥`max(5,0.1% held queries)`且各覆盖至少2个held receiver，`oracle-union(D99,D101)`比`oracle-union(D99,D100)`至少高0.25pp；每个receiver×pseudo-new pair的old/new/H/floor不得下降，balanced NLL严格改善，K1还须非identity。任一条件失败即`REJECT`，不得用target补选。

只有D101完整LODO通过后，才允许一次K1/new20与K10/new20 matched窄测；相对D100同row的B-old、A-old、New、H、floor均不得下降、forgetting不得增加，且H或floor至少一项严格上升，否则不运行历史125。

#### D101本地核心实现与独立终审

D101现已新增`code/cvsrffi/stage2_d101_shrinkage_rda.py`及`tests/test_stage2_d101_shrinkage_rda.py`，仅实现Phase1 nested-LODO可调用的解析核心，不含target runner、N607发布或125入口。唯一support入口为exact typed D99 INT8 bank；所有注册类均值从其decoded target/pseudo-target support统一计算。ground对象只抽取共享nuisance basis/spectrum和receipt，不读取或持久化ground class mean，也不直接产生old类logit/bias。K1 target covariance自由度和rank均为0；K>1 target residual rank≤2，ground rank≤4，总Woodbury rank≤6。部署状态只保存INT8线性权重、FP16分块scale/bias和闭合receipt；formal预测入口硬阻断。

canonical查询路径固定为D81＋D99 base与D101 RDA的单一alpha融合，D101替换D100而不是叠成第三头。TypedD81 batch在消费前重算完整receipt，任何logit/classes/K/query receipt内存篡改均fail-closed；所有公开概率矩阵统一校验有限、shape、row sum及元素范围`[0,1]`。ground nuisance basis在D99 metric-sqrt中的变换明确标注为逐样本归一化之前的linear/first-order proxy，不声称是完整归一化映射的精确push-forward，也不把共同可逆变换称为性能来源。

初审发现并修复两项P1：一是TypedD81 logit可写篡改后旧batch receipt未重算；二是公开结果对象会接受`[1.2,-0.2]`这类和为1但越界的伪概率。修复后独立聚焦攻击5/5通过，主线在`ssr-gpu`复跑D101专项与D99/D100相邻回归共70/70通过，`py_compile`和`git diff --check`均exit0。独立最终裁决为`MERGE_LOCAL_CORE_PHASE1_LODO_ONLY`，P0=0、P1=0。模块SHA=`b15702f3ca313e34d82925645a45b64e5df8c47a094bc4b64058d90106d68e3d`，测试SHA=`8aa4cdb9a136cb04c982db69899290098f06c6b99ff23a4439808dde6c45601c`。

D101仍无性能结果。独立D101 LODO complementarity/逐receiver×pseudo-new门、held-LODO量化margin authority、D81持久head与完整ground wire、combined MAC和完整`≤256KiB`系统资源尚未闭合，所以当前不得进入target、N607窄测或125，也不能宣称优于D100或满足完整资源上限。下一步只允许在不读取r7/target结果选参的前提下，把D101作为D100的替代臂接入独立Phase1 nested-LODO。

### r7完整Phase1 LODO结果与晋级裁决

r7状态链为`LOCAL_VERIFIED→LANDED→RUNNING→ARTIFACTS_COMPLETE→ANALYZED_NONFORMAL_LODO_DIAGNOSTIC`。唯一wrapper PID`1457448`、Python PID`1457450`于`2026-07-21 04:11:49→04:23:10 CST`在物理GPU5运行，exit0，未重试。主结果`d99_d100_phase1_lodo_blocked_diagnostic.json`为20,592,814B，SHA256=`6a7b6cb0ab9b0201fe99a7290067925ae7138490cd0b86e1255749a0eb7d46bf`；`result.json` SHA256=`2787514b890970886e8b4410d33fc2708028f6541a4f0298a722672773d3045e`；receipt SHA256=`8af595bb3984a525472dd33232872c5b19e678ea4bbef74214a82a9c6ebff826`。完整runner handoff、84个pair、14个receiver汇总和12个pseudo-new类汇总已回收到`artifacts/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7/`。

本轮最终裁决为：D99仅K5、K10通过development LODO准入，K1、K20因worst-class floor退化被拒；D100在K1/K5/K10/K20均为`0/64`候选通过，所有有效融合权重强制回退为`alpha=0`。K5/K10冻结D99参数一致：`eta=.25`、`nu=3`、`gamma=1`、`h0=.35`、`scale prior=2`、`scale ratio=.5–2`、`T99=.85`、`lambda0=.2`、`Tridge=.85`。

|K|结果作用域|D81 BA→D99 BA|balanced NLL|old|new|H|全局worst floor|裁决|
|---:|---|---:|---:|---:|---:|---:|---:|---|
|1|selection top blocked诊断|76.15%→78.94%|1.0673→0.7417|76.15%→78.94%|76.15%→78.94%|71.97%→75.31%|4.49%→2.13%|均值、H和NLL改善，但floor下降2.37pp，拒绝|
|5|final outer-LODO|81.54%→88.21%|1.0341→0.7583|81.54%→88.21%|81.54%→88.23%|80.59%→87.63%|31.96%→38.46%|D99准入|
|10|final outer-LODO|87.32%→88.75%|0.9493→0.7090|87.32%→88.75%|87.32%→88.76%|86.45%→88.10%|19.15%→36.26%|D99准入；当前最强LODO分支|
|20|selection top blocked诊断|90.00%→85.82%|0.9041→0.5448|90.00%→85.82%|90.00%→85.82%|89.49%→84.42%|36.17%→12.22%|NLL改善但识别、H和floor全面退化，拒绝|

K5相对D81的同row增量为BA`+6.68pp`、old`+6.67pp`、new`+6.69pp`、H`+7.04pp`、floor`+6.50pp`、NLL改善`0.2758`；K10增量为BA`+1.43pp`、old`+1.43pp`、new`+1.44pp`、H`+1.65pp`、floor`+17.11pp`、NLL改善`0.2402`。K5有22,740条final query、4,029条预测改变；D81救回Student-t错误1,716条，Student-t救回D81错误2,713条。K10同为22,740条final query、3,181条预测改变，双向救回为1,925/1,638条，证明D99不是数值identity。

#### Receiver级完整表现

每行汇总同一held receiver下6个pseudo-new轮换fold；floor为6个fold的最低值。虽然aggregate gate通过，K5与K10各有`24/42`个receiver×pseudo-new pair出现floor下降，说明现有aggregate floor门不能证明局部反遗忘稳定。

|K|receiver|mean rho|BA D81→D99|NLL D81→D99|H D81→D99|min floor D81→D99|floor回退pair|
|---:|---|---:|---:|---:|---:|---:|---:|
|5|1-1|.0722|68.69→88.19|1.119→.790|66.94→87.83|43.59→73.33|0/6|
|5|1-19|.0632|92.11→88.69|.895→.668|91.68→87.86|75.00→64.29|6/6|
|5|14-7|.0770|92.35→92.29|.829→.611|92.18→91.95|81.91→73.91|6/6|
|5|18-2|.0118|70.30→75.79|1.197→.987|69.26→73.97|45.74→38.46|6/6|
|5|19-2|.0768|94.15→90.01|.905→.674|94.07→89.67|85.42→77.08|6/6|
|5|2-1|.1300|66.28→91.65|1.326→.852|63.36→91.35|31.96→76.84|0/6|
|5|2-19|.1602|86.86→90.87|.968→.727|86.62→90.74|78.41→80.68|0/6|
|10|1-1|.0898|92.46→86.26|.870→.663|92.37→85.70|85.90→66.28|6/6|
|10|1-19|.0941|93.91→89.06|.887→.658|93.70→88.29|81.82→66.67|6/6|
|10|14-7|.0780|88.20→94.13|.753→.560|87.38→93.93|63.83→80.43|0/6|
|10|18-2|.0138|63.86→76.23|1.154→.946|59.60→74.08|19.15→36.26|0/6|
|10|19-2|.1292|92.78→91.10|.922→.676|92.54→90.87|77.08→80.21|0/6|
|10|2-1|.1538|91.81→92.50|1.038→.721|91.60→92.28|82.11→81.05|6/6|
|10|2-19|.1978|88.25→92.00|1.021→.740|87.95→91.58|75.64→70.45|6/6|

receiver异质性很强。K5在`1-1`与`2-1`分别提高BA`19.50pp`和`25.37pp`，但在`1-19`与`19-2`分别下降`3.42pp`和`4.14pp`。K10在`18-2`与`14-7`分别提高`12.37pp`和`5.94pp`，却在`1-1`与`1-19`下降`6.20pp`和`4.84pp`。pair级rho与BA/floor增量相关方向在K5约为`+.22/+.34`，K10却约为`-.30/-.33`，因此coverage不能单独作为安全门。

#### Pseudo-new类别轮换表现

|K|pseudo-new|new D81→D99|old D81→D99|H D81→D99|
|---:|---|---:|---:|---:|
|5|14-10|68.87→87.86|84.07→88.27|74.22→87.95|
|5|14-7|77.77→84.89|82.29→88.86|79.82→86.58|
|5|20-15|92.17→94.24|79.41→87.02|84.60→90.27|
|5|20-19|69.45→70.38|83.95→91.80|75.60→79.00|
|5|6-15|87.77→95.66|80.29→86.74|83.80→90.89|
|5|8-20|93.17→96.33|79.21→86.57|85.48→91.07|
|10|14-10|76.03→88.81|89.58→88.75|80.31→88.63|
|10|14-7|78.71→85.90|89.05→89.33|83.32→87.40|
|10|20-15|95.73→96.48|85.64→87.21|90.20→91.40|
|10|20-19|80.05→68.77|88.78→92.76|83.90→78.27|
|10|6-15|94.87→96.25|85.81→87.23|90.04→91.47|
|10|8-20|98.54→96.35|85.08→87.24|90.92→91.44|

K5最差局部new变化是receiver`18-2`、pseudo-new`20-19`的`63.74%→38.46%`，下降`25.27pp`；K10最差是receiver`1-1`、pseudo-new`20-19`的`88.37%→66.28%`，下降`22.09pp`。K10跨receiver平均时，`20-19`的new由`80.05%`降到`68.77%`，H由`83.90%`降到`78.27%`。这不是按ID设计专属修复的依据，而是要求下一版采用标签置换等价的support margin、cross-fit分歧和下尾风险联合门。

#### D100负结果与RDA替代动机

D100所有K的64候选均未让balanced NLL严格优于D99。K5 selection中请求D100使BA`87.31%→87.05%`、NLL`0.7666→0.8342`、H`86.47%→86.28%`，24/42 pair退化；K10使BA`87.97%→87.87%`、NLL`0.7162→0.7903`、H`87.09%→87.02%`，19/42 pair退化。final outer诊断中K5/K10各有28/42 pair至少一个old/new/H/floor下降。

另一方面，D100 ridge头确有互补错误：K5中D99救回ridge错误516条、ridge救回D99错误393条，oracle union为89.95%；K10分别为620/306条，oracle union为90.10%。负结论是“固定凸融合不能稳定利用互补性”，而不是“全局判别头没有任何补充证据”。因此D101 Shrinkage RDA继续作为D100的替代全局头做独立LODO，不与D100叠成第三头。

#### 资源、协议与下一实验门

|资源项|r7已知值|正式结论|
|---|---:|---|
|D99/D100解析状态构建|10,752次；0 epoch、0 optimizer step|满足解析适配性质|
|D81 Phase1 selection fit|28次×20步=560步|Phase1选参成本，不是Phase2适配步数|
|最大D99+D100已知持久wire|33,070B|只是已知组件，非完整系统上界|
|最大query MAC|207,754/sample|只是D99/D100已知组件|
|最大参数等价|1,734|D99/D100解析头|
|最大D99 fit瞬态|1,243,520B|未含D81/D100完整peak|
|D81 ground basis|18,032B|Phase1聚合知识|
|D99 ground bundle|6,930B|Phase1聚合知识|

完整combined fit peak、参数、persistent upper bound与query MAC均未闭合，资源状态保持`NONFORMAL_PARTIAL_KNOWN_COMPONENTS_ONLY`，不能据局部数字宣称正式满足`≤256KiB/≤80k`。协议审计为`phase1_only=true`、`single_leo_observation_archive=true`、`clean_or_raw_iq_used=false`、`target_rows_used=0`、`query_rows_used_for_selection=0`、`class_specific_hyperparameters=false`；但固定D81全局ground basis可能含held receiver，故只能声明support adaptation与D99局部ground消融的pseudo-target LODO，不能声明whole-method严格receiver外留一。

下一步执行门更新为：①D99仅允许K10/new20匹配target窄验证，K1因LODO失败不发布；②D101必须先完成独立nested LODO并通过逐receiver×pseudo-new、量化margin、互补性与资源门，才可进入target；③任何target窄结果都不能反向改r7参数；④D99窄测若没有同时守住B-old、A-old、New、H、floor和forgetting，则不运行其125；⑤若窄测通过，再运行固定历史125，并保留逐receiver/逐类局部退化审计。

### D99 K10/new20匹配target窄实验预登记

|字段|冻结值|
|---|---|
|run ID|`d99_d100_narrow_rx20_1_seed713101_k10_new20_d6efa5ad_20260721_r8`|
|目标|在唯一预登记development row上比较D81、D99与effective D100，验证Phase1 LODO正信号能否迁移到真实target support/query|
|Git代码提交|`d6efa5ad`；D99/D100窄runner及候选代码与r7隔离源码一致|
|报告/证据提交|`68796722`；包含完整r7结果、receipt及逐pair/receiver/class表|
|row|receiver`20-1`、seed`713101`、K10、seen-new20、三个物理ID互斥LEO弱场景|
|matched比较|同一job内部的D81/D99/D100；历史同row D81只作预期核对，不替代本次同job列|
|GPU/CPU|冻结物理GPU1，`CUDA_VISIBLE_DEVICES=1`、内部`cuda:0`；CPU thread2、interop1|
|状态|`LOCAL_VERIFIED`；15项窄runner/query-evaluation回归通过；尚未LANDED/RUNNING|

唯一主要变化是把D81各向同性/ground-spectrum距离替换为r7锁定的D99 coverage-gated低秩Mahalanobis Student-t局部头，并用`eta=.25`进行整row融合；D100在r7被强制`alpha=0`，只作为验证其确实回退为D99的同row诊断列。没有target超参数、类专属分支、query门或新数据权限。

冻结输入如下：

|输入|远端路径/SHA|
|---|---|
|cache|`runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix/rx_20_1/seed_713101/cache_set.json`|
|v2 authority|`runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/signed_authority_bundle`；COMMIT SHA=`fdedd9cfdfbb5db9f8962ba529403042b7de7011570dff514e9a629a44695147`|
|checkpoint|`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`；SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|runtime/method lock|`runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt`与`method_lock.json`|
|D81 ground|`runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`；manifest SHA=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`|
|D99 ground|r1 `d99_receiver_ground_bundle/d99_ground_bundle_dev.npz`、manifest和base lock；文件SHA分别为`e6940926…9bd4`、`f92a1bd6…8f0d`、`7481c351…22a9e`|
|LODO receipt|r7 `output/d99_d100_phase1_lodo_blocked_diagnostic.json`；文件SHA=`6a7b6cb0ab9b0201fe99a7290067925ae7138490cd0b86e1255749a0eb7d46bf`、receipt SHA=`8af595bb3984a525472dd33232872c5b19e678ea4bbef74214a82a9c6ebff826`|
|class binding|`runs/d20_int8_maxold_fftrf_20260717/input/class_binding.json`；SHA=`bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f`|

唯一child命令冻结为：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 BLIS_NUM_THREADS=2 CVSRFFI_CPU_THREADS=2 CVSRFFI_CPU_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7/source_d6efa5ad/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7/source_d6efa5ad/code/scripts/run_d99_d100_narrow.py --cache-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix/rx_20_1/seed_713101/cache_set.json --authority-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/signed_authority_bundle --authority-commit-sha256 fdedd9cfdfbb5db9f8962ba529403042b7de7011570dff514e9a629a44695147 --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --method-lock /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/method_lock.json --d81-ground-component-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component --d81-ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c --d99-ground-bundle-npz /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_ground_bundle_dev.npz --d99-ground-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_ground_bundle_dev.manifest.json --base-d99-lock /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_base_method_lock_dev.json --phase1-lodo-json /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7/output/d99_d100_phase1_lodo_blocked_diagnostic.json --class-binding-json /home/szu2070436088/2510044040/CV-SincNet/runs/d20_int8_maxold_fftrf_20260717/input/class_binding.json --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_d6efa5ad_20260721_r8 --receiver 20-1 --seed 713101 --k-shot 10 --new-count 20 --device cuda:0 --cpu-threads 2
```

sole runner必须先执行直连preflight，确认精确输入存在且SHA匹配、run/log/output均不存在、GPU1不超过每卡两项训练任务；随后只允许一次不可覆盖detached启动。远端runner LF SHA256必须为`4d8ebf9750a0b4a4f8a3b2643cc20fb8e35733f1c9b1d0a6386fd74fa1b87a2b`，不能使用Windows CRLF工作树SHA。预期产物包括`narrow_receipt.json`、D81/D99/D100的before/after不可变prediction、三份score与detailed score、offline build/registration pair、INT8和资源audit。所有prediction完成后才允许truth join。初版`d99_k10_new20_narrow_d6efa5ad_20260721_r8`只存在于报告提交`37a7cd93`且从未创建远端状态；本次修订只统一既有K10发布槽、线程上限和不可覆盖run ID，不改变方法或数据。

晋级门按同row冻结：D99相对D81的`old_acc_before_increment`、`old_acc_after_increment`、`seen_new_acc`均不得下降，`H_old_new`与全部注册类floor至少一项严格提高，forgetting不得增加；同时逐场景、逐类和old→new/new→old混淆不能出现未解释的集中崩溃。任何一项失败则标记`D99_COMPLETED_NARROW_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不运行D99历史125；全部通过才发布固定125，且不回调`eta/nu/h/rank/rho`。

#### r8实际结果：当前row opaque handle绑定技术失败

r8唯一启动一次，状态链为`LOCAL_VERIFIED→LANDED→RUNNING(瞬时)→FAILED_PRE_PREDICTION_BINDING_DRIFT`。wrapper PID`1588355`、Python PID`1588357`均已退出，exit1；物理GPU1恢复`0%/10MiB`，SSH进程和N607/bridge TCP22连接均为0。完整日志1,344B，SHA256=`92cbfef67fcffb6209da60511bcae1c66b30dfe32e766b82e5f5ea8cc54b203c`。

唯一异常为`D99D100QueryEvaluationError: class binding TX/handle bijection drift`。当前row before manifest的6个opaque handles与历史D20 binding中的handles不同；但6个Phase1 TX、TX顺序、class/direct-logit index、checkpoint和类数均匹配。opaque handle本来就是row-specific，旧实现错误地把历史handle当作跨row永久身份。该失败发生在D81/D99/D100任何query prediction之前：candidate prediction数0、score/detailed数0、`narrow_receipt.json`不存在。因此r8不是性能结果，不能报告B/A/New/H/floor/forgetting，也不能标记算法负结果。

已完成的partial仅包括offline row build、before/after enrollment/apply seal和registration pair。`offline_build_receipt.json` SHA256=`60d9936355eff135ac513599b25ef122b959a11d43b9cb7b9e49ffd663f6fed3`，`registration_pair.final.json` SHA256=`4048db7ce58d830c0d33d9642d229e84dd26058dcd1cb4497b27dce970146525`，row manifest仍为`8f83db32662da701f36099536bd775a4d33fc5a7f1a01c613b6291d9163aac56`。完整53文件、29,194,421B证据已保存到`artifacts/d99_d100_narrow_rx20_1_seed713101_k10_new20_d6efa5ad_20260721_r8/`；原远端run/log保留且不得覆盖或重试。

#### r9最小修复、双重验证与重发预登记

修改前review确认D93已经采用正确语义：稳定身份是Phase1 TX顺序，当前row的opaque handle由当前合法注册表提供。r9只修改D99 query evaluator、narrow runner和两份相邻测试：D20 binding继续锁定checkpoint、6个Phase1 TX、class/direct-logit index和TX顺序，但不再要求历史opaque handle等于当前row handle；当前row的`old_tx_labels`与当前6个opaque handles按同一类索引一一绑定。仍精确拒绝缺类、重复TX/handle、checkpoint漂移、TX顺序漂移和raw/typed语义漂移；不读取query真值、old/new query角色或类别配额。

|字段|r9冻结值|
|---|---|
|run ID|`d99_d100_narrow_rx20_1_seed713101_k10_new20_88db56d3_20260721_r9`|
|代码提交|`88db56d3`（`Bind D99 target TXs to current row handles`）|
|修改文件|`stage2_d99_d100_query_evaluation.py`、`run_d99_d100_narrow.py`及两份相邻测试|
|工作树验证|`ssr-gpu`专项15/15通过；`git diff --check`通过|
|精确发布源码验证|从提交`88db56d3`生成Git archive，解压后的LF源码再次15/15通过|
|源码ZIP|`E:\type10-7\code\snapshots\d99_d100_narrow_88db56d3_20260721_r9\source_88db56d3.zip`|
|ZIP SHA/规模|`b57c879dd8e0c38faf9ab63c2c463e83a16e48dc96a0cf7a1ee25bbc42650de4`；32,809,184B；4,402成员|
|GPU/CPU|物理GPU1、内部`cuda:0`；CPU thread2、interop1|
|状态|`LOCAL_VERIFIED_REVIEW_PASSED`；尚未同步或启动|

远端release根冻结为`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_88db56d3_20260721_r9`，源码ZIP同步到该根并解压为`source_88db56d3`；不可覆盖output为`<run>/output`，log根为`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_narrow_rx20_1_seed713101_k10_new20_88db56d3_20260721_r9`。r9除源码、run/log/output路径外，全部数据、checkpoint、runtime、method lock、ground、LODO、receiver、seed、K、new-count、候选和参数与r8逐字相同。

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 BLIS_NUM_THREADS=2 CVSRFFI_CPU_THREADS=2 CVSRFFI_CPU_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_88db56d3_20260721_r9/source_88db56d3/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_88db56d3_20260721_r9/source_88db56d3/code/scripts/run_d99_d100_narrow.py --cache-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix/rx_20_1/seed_713101/cache_set.json --authority-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/signed_authority_bundle --authority-commit-sha256 fdedd9cfdfbb5db9f8962ba529403042b7de7011570dff514e9a629a44695147 --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --method-lock /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/method_lock.json --d81-ground-component-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component --d81-ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c --d99-ground-bundle-npz /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_ground_bundle_dev.npz --d99-ground-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_ground_bundle_dev.manifest.json --base-d99-lock /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_base_method_lock_dev.json --phase1-lodo-json /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7/output/d99_d100_phase1_lodo_blocked_diagnostic.json --class-binding-json /home/szu2070436088/2510044040/CV-SincNet/runs/d20_int8_maxold_fftrf_20260717/input/class_binding.json --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_88db56d3_20260721_r9/output --receiver 20-1 --seed 713101 --k-shot 10 --new-count 20 --device cuda:0 --cpu-threads 2
```

独立review已裁决`P0=0、P1=0、MERGE`。专项15/15通过，独立负向攻击5/5通过，`git diff --check`通过。审查确认稳定身份是checkpoint绑定的6个Phase1 TX及`class_index/direct_logit_index`顺序，当前opaque handle只在当前sealed row内有效；修复没有读取query真值/角色，评分仍在预测密封后进行。sole runner现在可执行direct preflight、源码ZIP同步/整体SHA与关键文件SHA核验、隔离源码`py_compile/import`、不可覆盖单次detached启动。r9不授权自动重试。成功与性能晋级门完全沿用r8，不因技术修复放宽。

#### 固定代码修改链

后续每次修改采用以下v2链：

1. 修改前先冻结“一个根因、一个主要机制差异、一个预期可观察结果、一个停止条件”，并核对真实artifact、历史正确实现和输入/输出语义；不在同一提交混入算法、数据、runner和报告重构。
2. 先写能复现当前失败的最小反例，再写不变量攻击：顺序置换、重复项、缺项、checkpoint漂移、row-specific handle变化、K/类数边界和truth/role不可达。
3. 作者只提交最小非重叠diff；适配层只负责语义转换，核心方法不感知历史路径、opaque handle或scorer truth。
4. 本地按“专项单测→相邻集成→协议负例→`git diff --check`”四层验证；失败只修最早失败边界，不顺手改超参数或候选机制。
5. Git提交后，从精确commit生成LF archive并在解压源码复跑同一验证，防止工作树、CRLF、未跟踪文件或远端源码漂移。
6. 独立review只读裁决P0/P1/P2，作者不得自证；P0/P1未清零不发布。
7. 报告预登记冻结run ID、commit/SHA、单一变化、matched baseline、矩阵、GPU/CPU、输出路径和晋级门；唯一runner只负责N607落地与证据回收。
8. 结果按三态分流：无prediction是技术集成失败，只修直接失败项；有完整prediction但门失败是算法负结果，回到机制设计；窄测全部通过才进入125，125通过后仍需完整400-job/1200-scenario确认。
9. 每个完成版本必须同row报告B-old、A-old、seen-new、H、forgetting、全部注册类floor、逐类/逐receiver/逐场景混淆及资源；总体均值不能掩盖局部退化。

技术失败不得冒充算法负结果；作者不得自证晋级；不得用多轮源码签名、authority或数据握手替代无线信号算法实验。
