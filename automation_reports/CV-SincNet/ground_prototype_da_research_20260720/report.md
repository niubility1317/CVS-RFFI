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

本阶段只授权唯一runner执行parity，不授权feature export、LODO或target访问。重新preflight并确认GPU0可用、run/log根仍不存在后，建立独立run/log目录，核对源码包与脚本SHA，使用远端`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`从隔离源码运行：

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

stdout/stderr固定到`/home/szu2070436088/2510044040/CV-SincNet/logs/d97_phase1_singleobs_lodo_20260720_v1/parity_validation.out`。成功条件是进程exit0、receipt/vector/stdout均完整且SHA已回收、`resolved_device=cuda:0`、`batch_sizes=[1,8,256]`、最大delta≤`1e-5`、checkpoint/runtime SHA匹配，并用同一隔离源码再次通过exporter exact development binding。失败时不得自动重试、不得修改参数或runtime/checkpoint；返回完整日志与GPU/进程证据给主线。状态转换只允许`LOCAL_VERIFIED→LANDED→RUNNING→ARTIFACTS_COMPLETE`，本阶段不能进入`ANALYZED`性能结论。

## 下一轮唯一集成候选D99

跨方法监督否决继续增加第三个全局头或D98式二次融合权，建议下一轮只实现`D99 RA-CGTMK-D81`：保留D81一次拟合，将D96的密度反权、`D_eff`、共享ground nuisance basis和support-only coverage certificate用于构造严格PSD低秩Mahalanobis度量，再把D97的各向同性qK分支替换为类数归一化Student-t metric-kernel。ground只改变可观测距离，不直接给旧类加分；old/new仍由同式target support注册。

其核心收缩为：ground权重同时受`rho`、target偏移能量、`D_eff`和旧类support数量控制；target within-class basis按全部注册类平衡估计且rank≤4，K1时严格为0；最终非正交`P_z=Sigma_z^{-1}`只降低已知nuisance方向权重，不执行D93式全坐标transport。Student-t局部头保留`1/K_c`归一化，并用Phase1冻结的共享/类内尺度抑制异常support与过宽类别侵入。D81与metric-kernel只使用Phase1 receiver-LODO冻结的`eta_K`做一次row-global融合，target阶段不做K折D81重训，也不学习classwise alpha。

D99尚无实现或性能结果。准入要求比D97更严格：Phase1每个K均不得降低worst-class floor，NLL必须严格改善且D81/metric-kernel双向rescue均非零；K1还必须产生非identity预测。target窄实验只允许matched K1/new20与K10/new20，同row`B-old/A-old/New`均不得下降，`H/floor`至少一项严格上升且forgetting不得增加；通过后才运行完整125，125不反向选择`rank/eta/nu/h/rho`公式。D96 standalone SRDA保留为`REVISE`，D97为`MERGE+REVISE`，D98只保留typed provenance和tail指标工具，不进入D99推理链。
