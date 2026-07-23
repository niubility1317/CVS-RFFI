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
|精确发布源码验证|从提交`88db56d3`生成Git archive，解压后的原始blob字节源码再次15/15通过；EOL只作信息记录，不作为源码身份|
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
5. Git提交后，从精确commit生成保留原始blob字节的archive并在解压源码复跑同一验证；源码身份由archive整体SHA和关键成员原始字节SHA共同绑定，EOL只作信息记录，不另造规范性门。
6. 独立review只读裁决P0/P1/P2，作者不得自证；P0/P1未清零不发布。
7. 报告预登记冻结run ID、commit/SHA、单一变化、matched baseline、矩阵、GPU/CPU、输出路径和晋级门；唯一runner只负责N607落地与证据回收。
8. 结果按三态分流：无prediction是技术集成失败，只修直接失败项；有完整prediction但门失败是算法负结果，回到机制设计；窄测全部通过才进入125，125通过后仍需完整400-job/1200-scenario确认。
9. 每个完成版本必须同row报告B-old、A-old、seen-new、H、forgetting、全部注册类floor、逐类/逐receiver/逐场景混淆及资源；总体均值不能掩盖局部退化。

技术失败不得冒充算法负结果；作者不得自证晋级；不得用多轮源码签名、authority或数据握手替代无线信号算法实验。

#### r9发布前停止与r10原始字节链修正

r9没有启动child。状态固定为`LOCAL_VERIFIED→SOURCE_LANDED→STOPPED_PRE_CHILD_RELEASE_PACKAGE_LINE_ENDING_ASSERTION`。direct preflight、11项冻结输入SHA、GPU1空闲、run/output/log初始不存在均已通过；本地与远端ZIP均为32,809,184B、4,402成员，SHA256=`b57c879dd8e0c38faf9ab63c2c463e83a16e48dc96a0cf7a1ee25bbc42650de4`，archive成员与远端解压成员逐项一致。r9远端只保留ZIP和解压源码，`output`与log目录不存在、匹配进程0、GPU1=`0%/10MiB`、SSH/TCP22连接0。r9路径不得复用。

停止原因是本报告此前错误声明“Git archive必须LF-only”。项目规则只要求通过stdin发送给远端bash的多行脚本使用LF，不要求Python Git blob/archive统一LF。冻结ZIP中的四个关键文件与已审提交原始blob一致，均为纯CRLF；Python支持该EOL。真实源码权威链应为`commit 88db56d3→原始Git blob→b57c…ZIP成员→远端解压成员`，每层核原始字节SHA，而不是把EOL当身份。该停止发生在`py_compile/import`与任何prediction之前，不是N607运行失败或性能结果。

|关键成员|bytes|原始字节SHA256|EOL信息|
|---|---:|---|---|
|`code/cvsrffi/stage2_d99_d100_query_evaluation.py`|41,476|`c9e3a25e72c01484e36aed85e4c40bd53e94523e2fa193d0964f832ca094c1a2`|923个CRLF，0个lone CR|
|`code/scripts/run_d99_d100_narrow.py`|22,571|`8e05f189336bfe327b24b3a4108a3cf84949dd789e992220175364f09a4435dc`|512个CRLF，0个lone CR|
|`tests/test_stage2_d99_d100_query_evaluation.py`|10,936|`076ede6c3913d7a9bcfe5fcb328777724348f46a75167df7e6fe86a802c1373f`|291个CRLF，0个lone CR|
|`tests/test_run_d99_d100_narrow.py`|8,914|`7cbe3f02a3eb0e80fcf477a9c0c11c98a0e0131e2f087b1efe322be439f5e1fc`|246个CRLF，0个lone CR|

r10冻结如下：

|字段|冻结值|
|---|---|
|run ID|`d99_d100_narrow_rx20_1_seed713101_k10_new20_88db56d3_20260721_r10`|
|方法代码|`88db56d36be3e7fc7b67e34145eab69a51fda5df`，与r9完全相同|
|源码ZIP|沿用本地同一未修改、未重打包的`source_88db56d3.zip`；SHA/bytes/成员数仍为`b57c…50de4`/32,809,184/4,402|
|唯一变化|删除无项目依据的LF-only前置断言，改为整体ZIP＋四关键成员原始字节SHA＋`py_compile/import`闭合|
|独立review|P0=0、P1=0、`MERGE`|
|远端run/output|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_88db56d3_20260721_r10`及其`output`|
|远端log|`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_narrow_rx20_1_seed713101_k10_new20_88db56d3_20260721_r10`|
|启动策略|重新direct preflight；新路径同步同一ZIP；核整体与成员原始SHA；`py_compile/import`通过后只启动一次；不自动重试|

r10的child命令只把r9命令中的run/source/output路径替换为r10路径。数据、checkpoint、runtime、method lock、ground、LODO、receiver=`20-1`、seed=`713101`、K=`10`、new-count=`20`、GPU/CPU、D81/D99/D100候选和全部超参数逐字不变。若任一ZIP/member SHA、成员数、compile或import偏差，继续fail closed；不因删除错误EOL门而放宽真实源码漂移检查。

#### r10实际结果：active-K配置错误读取未封存K1锁

r10唯一启动一次且没有重试。状态链为`LOCAL_VERIFIED_REVIEW_PASSED→SOURCE_LANDED_VERIFIED→PY_COMPILE_IMPORT_PASSED→LANDED→RUNNING(瞬时)→FAILED_PRE_PREDICTION_LODO_K1_CONFIG_DRIFT`。源码ZIP、四关键成员原始SHA、远端`py_compile`和两模块import均通过，证明r9的EOL修正有效，r10失败与源码传输无关。

wrapper PID=`1608261`、Python PID=`1608263`，均已退出，exit1。完整日志2,155B，SHA256=`e89bc0f9619ef8daf698c362317c8a6911899ab8a6eedddcc1d32d1badfff591`。partial output为49文件、29,193,188B；candidate prediction=0、score/detailed=0、`narrow_receipt.json`不存在。`offline_build_receipt.json` SHA256=`33174dd7d4055f743de674370b5f60a7fb6143be1488dab806eedd049f0008b0`，`registration_pair.final.json` SHA256=`118a61e1ad4c2c1ee08db40a60ee6737868f2d2a956b2dafde76d77a40b5e8ee`。完整回收证据保存在根报告artifact目录`artifacts/d99_d100_narrow_rx20_1_seed713101_k10_new20_88db56d3_20260721_r10/`。远端run/source/output/log全部保留且不得覆盖或复用；GPU1、匹配进程、SSH和TCP22连接均已清零。

唯一异常是K10 row已在入口成功取得K10锁后，`_d99_config`仍遍历全部`ALLOWED_K`并首先调用`locked_parameters_from_lodo(..., k_shot=1)`。r7的K1因LODO floor失败而按设计没有封存可发布锁，因此抛出`D99D100QueryEvaluationError: K-specific LODO parameters are missing`。这不是缺数据、N607故障或D99性能负结果，而是active-K runtime错误预构造inactive-K配置。

修复边界冻结为：K10只消费已验证K10参数，不能复制K10参数到K1，也不能读取被拒K1记录；K1作为active row时仍必须fail closed。任何修复先经过active-K语义pre-review，再以最小query evaluator＋相邻测试提交；方法超参数、数据、checkpoint、LODO、候选和target row保持不变。r11须使用新commit、新不可覆盖run ID和新报告预登记。

#### r11 active-K最小修复与第三次目标窄实验预登记

r11方法提交为`dac89ae5`（`Bind D99 query config to active K only`），只修改`stage2_d99_d100_query_evaluation.py`和其相邻测试。入口对当前row只调用一次`locked_parameters_from_lodo(active_k)`；D99只覆盖active K的eta与该row公共参数，inactive eta保持base原值且不宣称LODO锁；D100仅active K使用同一锁参数，inactive K固定数据无关占位`lambda=1,temperature=1,d99_temperature=1,alpha=0`并标记为本row未使用。每次构建D100 state前同时验证`bank.metric.k_shot==active_k`和active参数逐项相等。K1/K20作为active row但缺锁时仍fail closed；K20原有artifact门未改变。

独立pre/post review均通过，post裁决`P0=0、P1=0、P2=2、MERGE`。两个非阻断P2为active-K audit缺直接序列化断言、K20缺artifact的显式相邻回归可继续补强；静态路径确认均未绕过，且不影响本次K10窄测。工作树`py_compile`通过，query evaluator＋narrow runner＋D100 core为46/46；从精确提交生成archive并在解压源码复跑相同编译与46/46，exit0。pytest结束后的Windows Temp `PermissionError`是已知atexit清理噪声，pytest进程exit0。

|字段|r11冻结值|
|---|---|
|run ID|`d99_d100_narrow_rx20_1_seed713101_k10_new20_dac89ae5_20260721_r11`|
|方法提交|`dac89ae5`|
|唯一机制变化|runtime只组装并消费当前row的K10锁；不读取K1/K5/K20 inactive锁|
|源码ZIP|`E:\type10-7\code\snapshots\d99_d100_narrow_dac89ae5_20260721_r11\source_dac89ae5.zip`|
|ZIP SHA/规模|`613d03a71f14f859ec1feba3647defab7b1816cc6862f75ddab3d329a1219847`；32,821,939B；4,408成员|
|精确archive复测|`py_compile`通过；46/46通过|
|远端run/output|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_dac89ae5_20260721_r11`及其`output`|
|远端log|`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_narrow_rx20_1_seed713101_k10_new20_dac89ae5_20260721_r11`|
|GPU/CPU|物理GPU1、内部`cuda:0`；CPU thread2、interop1|
|重试|不授权；只启动一次|

|关键ZIP成员|bytes|原始字节SHA256|
|---|---:|---|
|`code/cvsrffi/stage2_d99_d100_query_evaluation.py`|43,884|`1fd87f77ccabab418062be7874a2bbe1779e2922a17194c3f314b4b32ba3fa09`|
|`code/scripts/run_d99_d100_narrow.py`|22,571|`8e05f189336bfe327b24b3a4108a3cf84949dd789e992220175364f09a4435dc`|
|`tests/test_stage2_d99_d100_query_evaluation.py`|12,877|`d2599183c6d32655f1a37ec82e9e875defe2945db17d8bbd338e2bc80667dcec`|
|`tests/test_run_d99_d100_narrow.py`|8,914|`7cbe3f02a3eb0e80fcf477a9c0c11c98a0e0131e2f087b1efe322be439f5e1fc`|

唯一child命令冻结为：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 BLIS_NUM_THREADS=2 CVSRFFI_CPU_THREADS=2 CVSRFFI_CPU_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_dac89ae5_20260721_r11/source_dac89ae5/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_dac89ae5_20260721_r11/source_dac89ae5/code/scripts/run_d99_d100_narrow.py --cache-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix/rx_20_1/seed_713101/cache_set.json --authority-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/signed_authority_bundle --authority-commit-sha256 fdedd9cfdfbb5db9f8962ba529403042b7de7011570dff514e9a629a44695147 --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --method-lock /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/method_lock.json --d81-ground-component-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component --d81-ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c --d99-ground-bundle-npz /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_ground_bundle_dev.npz --d99-ground-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_ground_bundle_dev.manifest.json --base-d99-lock /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_base_method_lock_dev.json --phase1-lodo-json /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7/output/d99_d100_phase1_lodo_blocked_diagnostic.json --class-binding-json /home/szu2070436088/2510044040/CV-SincNet/runs/d20_int8_maxold_fftrf_20260717/input/class_binding.json --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_dac89ae5_20260721_r11/output --receiver 20-1 --seed 713101 --k-shot 10 --new-count 20 --device cuda:0 --cpu-threads 2
```

r11除active-K配置组装和新不可覆盖路径外，与r10的数据、checkpoint、runtime、method lock、ground、LODO、receiver、seed、K、new-count、候选和超参数完全相同。晋级门仍为D99相对同row D81的B-old/A-old/New均不下降，H或全部注册类floor至少一项严格提高，forgetting不增加，并检查逐场景、逐类和双向混淆；失败则不运行125，通过才进入固定历史125。

#### r11实际结果：重复GPU前向被误用作support身份断言

r11唯一启动一次、无重试。状态链为`LOCAL_VERIFIED_REVIEW_PASSED→SOURCE_LANDED_VERIFIED→PY_COMPILE_IMPORT_PASSED→LANDED→RUNNING(瞬时)→FAILED_PRE_PREDICTION_GPU_FORWARD_NUMERIC_IDENTITY`。wrapper PID=`1626048`、Python PID=`1626050`，均已退出，exit1。完整日志1,155B，SHA256=`c95b08e92fb241bd58f388de33ef6992cacd7fa75bd9fbd0f569a0be4d38b7fe`。partial output为49文件、29,193,186B；prediction=0、score/detailed=0、无`narrow_receipt.json`。offline receipt SHA=`1a458181c7136055480393692cf4fbb1f215aefdc5ef9dc3a0c855d62084a464`，registration pair SHA=`ce715fccab94aeef0abe7b769cbf5b14f2c0079a22a3d313bdaf331910866c4d`。完整证据已回收至根报告artifact目录，r11远端路径保留且不得复用；GPU1、进程、SSH和TCP22已清零。

三层只读诊断证明数据完全一致：每场景before/after旧support均60条、token无重复，ID序列与集合完全相同；class index、opaque label序列完全相同，每类10条；按token对齐后原始IQ逐字节相同且max_abs=0。clear/low/rain的IQ SHA分别为`e9644e19…2469`、`3e7d0c93…0ddf`、`ad199840…6320`，两侧一致。真实差异仅来自重复GPU forward：`leo_clear_weak`相同IQ的z160 max_abs=`5.14984130859375e-05`、registered feature max_abs=`2.56318598985672e-05`，超过旧硬编码`atol=1e-6`；low/rain本次为0。loader和forward均为eval/no-grad，但runtime未承诺bit-level重复确定性。

因此r11不是support集合漂移或D99性能负结果。r12修复不能根据target观测选择更宽容差，而应删除重复计算：先用token/class/index/raw IQ原始字节完成before/after旧support精确闭合，再只对after全注册support做一次GPU forward，before旧特征从同一次结果按稳定token映射取得。这样同时减少旧support重复前向、数值非确定性和GPU开销；任何token、label、index或IQ字节漂移仍fail closed。

#### r12单次support前向修复与第四次目标窄实验预登记

r12只修复r11已证实的重复GPU前向集成缺陷，不改变D81/D99/D100算法、数据、checkpoint、bundle、LODO锁、receiver、seed、K、new-count、候选或超参数。before/after旧support先按稳定token、class index、rank、label和raw IQ C-order原始字节完成精确双射；随后仅对after全注册support执行一次GPU forward，before-old特征从同一次`all_x`按token位置映射复用。任何active-K缺项、重复项、非有限IQ或身份/字节漂移仍在forward前fail closed；inactive高rank数据不进入当前K的计算或有限性门。

|字段|r12冻结值|
|---|---|
|run ID|`d99_d100_narrow_rx20_1_seed713101_k10_new20_3e4c54f6_20260721_r12`|
|方法提交|`3e4c54f6effe7ce8e21f5f34307b66dbd878a3ec`（`Reuse one support feature realization`）|
|修改文件|`code/cvsrffi/stage2_d99_d100_query_evaluation.py`、`tests/test_stage2_d99_d100_query_evaluation.py`|
|独立终审|`P0=0`、`P1=0`、`MERGE`|
|工作树验证|`ssr-gpu`中`py_compile`通过；query evaluator＋narrow runner＋D100 core为56/56，exit0|
|源码ZIP|`E:\type10-7\code\snapshots\d99_d100_narrow_3e4c54f6_20260721_r12\source_3e4c54f6.zip`|
|ZIP SHA/规模|`d230e6c547d34cf261b79410239d542cdc46d8df5755648b2d0ca2b9d6dcccda`；32,849,576B；4,410成员|
|精确archive复测|解压源码`py_compile`通过；相同56/56通过，exit0；pytest结束后的Windows Temp `PermissionError`仍为已知atexit清理噪声|
|远端run/output|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_3e4c54f6_20260721_r12`及其`output`|
|远端log|`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_narrow_rx20_1_seed713101_k10_new20_3e4c54f6_20260721_r12`|
|GPU/CPU|物理GPU1、内部`cuda:0`；CPU thread2、interop1|
|重试|不授权；唯一release subagent只允许一次不可覆盖启动|

|关键ZIP成员|bytes|原始字节SHA256|
|---|---:|---|
|`code/cvsrffi/stage2_d99_d100_query_evaluation.py`|46,789|`00ce54324c155e22d0c688e9e432289edf792a1a6cdd796ad30d1726a2ceac03`|
|`code/scripts/run_d99_d100_narrow.py`|22,571|`8e05f189336bfe327b24b3a4108a3cf84949dd789e992220175364f09a4435dc`|
|`tests/test_stage2_d99_d100_query_evaluation.py`|19,460|`b935eab84f7d50d0e3fb08057e3c077bee2687c5a34449699829833d37d00eb8`|
|`tests/test_run_d99_d100_narrow.py`|8,914|`7cbe3f02a3eb0e80fcf477a9c0c11c98a0e0131e2f087b1efe322be439f5e1fc`|

唯一child命令冻结为：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 BLIS_NUM_THREADS=2 CVSRFFI_CPU_THREADS=2 CVSRFFI_CPU_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_3e4c54f6_20260721_r12/source_3e4c54f6/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_3e4c54f6_20260721_r12/source_3e4c54f6/code/scripts/run_d99_d100_narrow.py --cache-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix/rx_20_1/seed_713101/cache_set.json --authority-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/signed_authority_bundle --authority-commit-sha256 fdedd9cfdfbb5db9f8962ba529403042b7de7011570dff514e9a629a44695147 --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --method-lock /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/method_lock.json --d81-ground-component-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component --d81-ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c --d99-ground-bundle-npz /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_ground_bundle_dev.npz --d99-ground-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_ground_bundle_dev.manifest.json --base-d99-lock /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_base_method_lock_dev.json --phase1-lodo-json /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_d6efa5ad_cudafix_20260721_r7/output/d99_d100_phase1_lodo_blocked_diagnostic.json --class-binding-json /home/szu2070436088/2510044040/CV-SincNet/runs/d20_int8_maxold_fftrf_20260717/input/class_binding.json --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_3e4c54f6_20260721_r12/output --receiver 20-1 --seed 713101 --k-shot 10 --new-count 20 --device cuda:0 --cpu-threads 2
```

r12的同row保留门按更新目标收紧：D99相对D81的`old_acc_before_increment`、`old_acc_after_increment`、`seen_new_acc`、`H_old_new`、全部注册类floor、最低旧类和最低新类均不得下降，`H_old_new`或全部注册类floor至少一项严格提高，forgetting不得增加，并检查逐场景、逐类和old→new/new→old集中崩溃。无完整prediction仍只算技术集成失败；完整prediction未过门则标记`D99_COMPLETED_NARROW_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。即使全部通过，也只把D99保留为后续联合设计的legacy候选证据，不能由r12单独授权当前目标的125，且不能回调任何参数。

更新后的总目标要求每轮另有头固定的`C-id/C-dom/C-joint`显式域适应消融。D99同时改变z_id metric、Student-t局部头及与D81的融合，因此r12只用于完成既有冻结候选的真实target迁移证据，不能单独充当新目标中的纯`C-id`因果臂。D101依赖D99 bank/mapped feature后构造RDA，也不能单独充当identity/no-DA的纯B臂。新一轮将另行冻结A/B/C-id/C-dom/C-joint/D，并由该完整窄矩阵统一决定是否授权125；125通过后完整确认按5 receivers×5 seeds×3 scenes×4K×4 new-count＝1,200评价单元执行，不能把125写成完整矩阵。

#### D101 Shrinkage RDA nested LODO实现状态

D101 Phase1 nested receiver LODO实现与测试已独立提交为`fd38b861`。最终独立终审为`P0=0、P1=0、MERGE`；主线在`ssr-gpu`复跑D101专项、RDA core、D99/D100 LODO和D100相邻测试共70/70通过。它精确绑定D99/D100/D101顶层冻结候选网格，支持全失败及mixed partial的可验证`REJECT` receipt，真实执行K1 alpha=0 fallback数值路径，并拒绝删除、重排、重复候选后重签。该提交当前只有Phase1 LODO核心证据，尚未创建release wrapper、尚未运行N607或target，不构成性能结果。

#### D101 Phase1 LODO wrapper收口

D101 development-only release wrapper及其测试已独立提交为`47053bc50d96a0223c6814220091cf2ff4d50f7e`。wrapper SHA256为`ae0c8d9fcf2dda6f2218914a7d83ba8e6e1d8167cae56b1c26532dbf2bd2d932`，测试SHA256为`eb8b562e88584511c6452d2a95e3e047a8fa40171437528bc54ea2efb32ed894`。独立终审为`P0=0`、`P1=0`、`MERGE`，非阻断P2是`result.json`未内嵌wrapper自身SHA；外部Git提交、配置中的D101 code registry和报告仍负责入口身份闭合。

主线在`ssr-gpu`中对wrapper、D101 LODO/RDA core、D99/D100 LODO runner和D100相邻模块复跑80/80，`py_compile`通过；仅有既有pytest Temp atexit清理噪声且进程exit0。当前仍未生成D101冻结release config、未运行N607、未访问target，也没有old/new/H/floor/forgetting性能。由于D101强制在D99 mapped bank后构造RDA，它只能作为D99联合分类头的Phase1诊断，不能充当更新目标中的identity/no-DA纯B臂，也不能授权target或125。

#### r12实际结果：D99/D100完整目标窄实验显著负收益，拒绝125

r12唯一真实child完成并返回`exit.code=0`。状态链为`LOCAL_VERIFIED→LANDED→START_SSH_SELF_MATCH_PRE_CHILD→RUNNING→ARTIFACTS_COMPLETE→ANALYZED`。真实child前，一次只读gate误写evaluator检查路径而失败，另一次启动SSH中的裸`pgrep`把自身长shell命令误识别为既有进程；两次均未创建日志、output或child，未改变冻结child命令，真实child启动数仍为0。经只读确认后，在同一run ID、同一源码和同一冻结命令下执行唯一一次真实child。wrapper PID=`1659913`、child PID=`1659921`，物理GPU1、`CUDA_VISIBLE_DEVICES=1`、内部`cuda:0`；曾观测GPU计算进程占用552MiB。真实child退出较快，发布侧未及时捕获live CWD/cmdline，因此只保留启动命令和PID/GPU证据，不伪造live观测。实验结束后GPU1、本地SSH进程和TCP22连接均清零。

远端run为`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_narrow_rx20_1_seed713101_k10_new20_3e4c54f6_20260721_r12`，远端log为`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_narrow_rx20_1_seed713101_k10_new20_3e4c54f6_20260721_r12`。`runner.log`完整1行、3,337B，SHA256=`da21e128…`；`narrow_receipt.json` SHA256=`5f29f81d…`。3个候选均生成before/after prediction，共6个NPZ；每个before为360条旧类query，每个after为1,560条全部注册类query，score、detailed和evaluation audit齐全。结果级42个文件、4,557,133B已回收到`E:\type10-7\automation_reports\CV-SincNet\ground_prototype_da_research_20260720\artifacts\d99_d100_narrow_rx20_1_seed713101_k10_new20_3e4c54f6_20260721_r12`，未回收数据、truth sidecar、checkpoint或sealed runtime。

|候选|机制/类别|receiver|K/new|old-before|old-after|seen-new|H|registered BA|all-floor|min-old|min-new|forgetting|结论|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D81|matched强基线|20-1|10/20|87.2222%|69.7222%|68.9167%|69.3171%|69.1026%|15.0000%|48.3333%|15.0000%|17.5000pp|保留同row比较基线|
|D99|ground-guided metric＋Student-t＋D81融合|20-1|10/20|80.8333%|45.8333%|44.1667%|44.9846%|44.5513%|0.0000%|10.0000%|0.0000%|35.0000pp|显著负收益，拒绝|
|D100|D99＋RDA/ridge候选，但`alpha_K10=0`|20-1|10/20|80.8333%|45.8333%|44.1667%|44.9846%|44.5513%|0.0000%|10.0000%|0.0000%|35.0000pp|与D99预测完全相同，不是独立纠错|

D99相对同row D81：old-before=`-6.3889pp`、old-after=`-23.8889pp`、seen-new=`-24.7500pp`、H=`-24.3325pp`、registered BA=`-24.5513pp`、all-floor=`-15.0000pp`、min-old=`-38.3333pp`、min-new=`-15.0000pp`、forgetting=`+17.5000pp`。损害在注册前已经出现，因此不能仅归因于新类加入后的边界挤压。D100的`alpha_K10=0`，before/after prediction SHA均与D99完全相同，未产生第二头纠错证据。

|场景|ground coverage rho|ground weight|D99 old-after|D99 seen-new|D99 H|
|---|---:|---:|---:|---:|---:|
|`leo_clear_weak`|0.06243|0.04736|45.0000%|40.5000%|42.6316%|
|`leo_low_elev_weak`|0.01645|0.01266|49.1667%|49.2500%|49.2083%|
|`leo_rain_weak`|0.06535|0.05003|43.3333%|42.7500%|43.0397%|

三场景old-support配对残差能量中约1.6%–6.5%投影到当前ground nuisance子空间，其余support-derived residual能量位于该子空间的正交补。`rho`只是由合法old support得到的coverage proxy，不代表全部target query偏移。该结果不能证明“合规地面聚合知识无用”，但明确否定了D99这种低coverage proxy下同时改变metric、Student-t局部头和D81融合的使用方式。后续ground知识只能提供共享方向、协方差先验和coverage证书；`rho`低时必须精确回退identity，不能直接提高旧类logit。

协议审计为`support_only_fit=true`、`single_leo_weak_observation_only=true`、`query_batch_dependency=false`、`query_state_updates=0`，truth只在全部不可变prediction形成后由独立scorer连接。每场景after的D99已知persistent numeric state=`130,444B`，D100组合已知值=`140,716B`；D99 incremental query MAC estimate=`866,880`，D99＋D81 total query MAC upper bound=`882,144/query`。虽然已知numeric state低于256KiB，但`complete_serialized_state_total_available=false`，不能宣称完整部署资源闭合；`d99_deployment_status=LOCAL_CORE_BLOCKED_EXTERNAL_PHASE1_AND_D81_CAPSULE_AUTHORITIES`，formal eligible仍为false。

最终裁决：`ARTIFACTS_COMPLETE/ANALYZED/DEVELOPMENT_DIAGNOSTIC_NEGATIVE_REJECT_125`。D99/D100停止target调参，不进入125；D101不得冒充纯B/RDA头。下一轮必须先冻结`A/B/C-id/C-dom/C-joint/D`六个因果臂，再做一次K1/new20与K10/new20、3场景的matched窄实验。125仅是冻结方法的稳定性screen，不能选参；通过后仍需5 receivers×5 seeds×3 scenes×4K×4 new-count＝1,200评价单元的完整确认。

#### 三轮失败复盘与下一代码链冻结

本轮r10、r11、r12形成的工程教训分别是：active-K运行时不得预构造被Phase1拒绝的inactive-K锁；同一support身份应先以token/label/index/raw IQ字节闭合并只前向一次；发布进程探针不得用会匹配自身长命令的裸`pgrep`。方法教训是：Phase1 LODO晋级不能替代真实target窄证据；support-fit达到100%不能证明query泛化；重构地面原型或domain×class中心的高余弦保真不能替代`D_eff`、coverage、margin和最终old/new/floor证据。

下一代码链冻结为`typed dual-feature input→A(identity z_id＋single-qKNN)→B(identity z_id＋RDA/SRDA head)→C-id→C-dom→C-joint→D(best C＋B head)`。每个patch只能改变一个因果变量，修改前写唯一差异、禁止输入、identity等式和资源上界；修改后由非作者独立review，要求P0=0、P1=0，再运行专项/相邻回归、真实checkpoint smoke并形成独立Git commit。机器回执必须满足`C.classifier_hash==A.classifier_hash`、`B.DA_hash==identity_hash`、`D.DA_hash==best_C.DA_hash`和`D.head_hash==B.head_hash`。六臂冻结前不再访问target query或发布N607目标实验。

#### Patch 0：同一received-IQ双特征runtime本地实现

Patch 0新增`dual_feature_forward.py`、dual TorchScript exporter/parity verifier及3个专项测试，不修改既有identity runtime、D99/D101或target runner。一次固定received-IQ前向只执行`id_backbone`、`dom_backbone`和`dom_enhancer`，输出`z_id160`、`z_dom160`和`tx_logits`；不序列化、不执行、不读取`dom_head/adv_head/tx_adv_head`。runtime固定`capacity=256`、输入长度和TX宽度，checkpoint/adapter各只读取一次为bytes，同一bytes同时用于SHA和`BytesIO`加载；candidate verifier绑定checkpoint、adapter、runtime、export receipt四类SHA，并对batch1/mid/256执行三输出parity。

作者修复前的交叉review发现3个P1：trace后T未锁定、checkpoint/adapter存在路径级swap-restore窗口、verifier不能证明禁用head不存在。修复后非作者复审为`P0=0、P1=0、P2=2、MERGE_NONFORMAL_CORE`；32/32最小复审通过。两个保留P2为：正式authority前应让runtime bytes快照贯穿export parity与SHA，且verifier自身应执行T±1和batch257负探针。当前代码明确`formal_phase2_eligible=false`、`bundle_created=false`，不授权target或N607。

主线在`ssr-gpu`中将Patch 0、Patch A候选与旧identity/export/parity/bundle、D81/D99/D101相邻链合并复跑，`py_compile`通过、154/154测试通过，进程exit0。TorchScript弃用/TracerWarning和pytest Temp atexit权限告警均未改变exit0；动态输入边界由trace外层script验证，不依赖trace内被常量化的Python条件。

|文件|SHA256|
|---|---|
|`code/cvsrffi/dual_feature_forward.py`|`eeaca06f84f5771c90dfb92e6bbbc4980f2772e9fcdf80d54e06fee387afd815`|
|`code/scripts/export_adv3b02_dual_feature_torchscript.py`|`339a4c11320cd78a137030103eb4ebfe0095ca9f97e4291e19e385fabf6ebabc`|
|`code/scripts/verify_adv3b02_dual_runtime_checkpoint_parity.py`|`92eafc2e93c525e9d9c05592f6e31aeab4bb9816c1c61a69bb1b3d0187cbfed3`|
|`tests/test_dual_feature_forward.py`|`4683a9aa7e163d48a3946d23102add54eca5b81506f9c1a511ca672d906980f8`|
|`tests/test_export_adv3b02_dual_feature_torchscript.py`|`9afa96ea72945a0ad500aee0666e967ac4da6a37ed9b3e0c5343dd157735826c`|
|`tests/test_verify_adv3b02_dual_runtime_checkpoint_parity.py`|`2589fb0046a0cdc71260445378f3f2b7d113bf926f5bb43dc97042ee89a1e9cb`|

#### Patch A：纯z_id160 identity Student-t single-qKNN基线

Patch A新增独立`stage2_zid_student_t_qknn.py`及其测试，不导入或融合D81/D99/D100/D101，不读取`z_dom`、FFT/RF、ground class mean、query label或old/new角色。所有注册类共享同一Phase1锁、INT8 support bank、FP16逐向量scale和normalized Student-t `logsumexp-log(K_c)`公式；rank0 metric严格是identity，作为后续C-id固定分类头。非identity metric只为未来C-id保留共享PSD接口，必须携带typed provenance，fit scope仅允许`phase1_lodo`或`target_support_only`且`query_rows_used_for_fit=0`。

首轮作者自测22/22通过后，非作者review发现3个P1和1个P2：FP32 teacher错误复用INT8 bank的FP16带宽、MAC被误称为upper bound、metric来源声明缺少typed provenance、wire缺少严格loader。修复后第二轮非作者复审为`P0=0、P1=0、P2=0、MERGE`；专项30/30、D81/D99/D101相邻回归71/71、`py_compile`均通过。量化审计现在从full-precision support独立重算teacher带宽；wire严格拒绝截断、尾随字节、header/数组顺序、dtype/shape和payload篡改，并对identity/rank2执行byte-exact roundtrip。

最大`C=26,K=20,rank=2`探针包含520条support：numeric arrays=`85,660B`；两次独立provenance payload探针的实际wire为`88,718–88,719B`，该范围是实测值而非全局上界；persistent decoded cache=`0B`。matmul-only审计为每次score固定`S*r*d=166,400MAC`，每query可变部分`S*d+r*d+S*r=84,560MAC`；明确不包含hash、decode、normalize、reduction、elementwise、exp/log和serialization，也不宣称端到端MAC上界或实测latency。上层sealed capsule/admission仍须解析并核验typed provenance所绑定的真实source receipt；本模块只封存来源SHA，不自行读取外部文件。

|文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_zid_student_t_qknn.py`|`f7bc2ab7e6f9457085973099431db934edfa840ba37e904288ff4720726101e2`|
|`tests/test_stage2_zid_student_t_qknn.py`|`a7c349b99917f2dc388ace3d99ae9f5a4cf2346566f7c1405b68e513baef099f`|

Patch A当前仅完成本地组件和审计，不构成target性能、Phase1 LODO晋级、bundle或N607发布。下一步Patch B必须只消费该公开typed z_id bank，在identity metric不变的条件下增加Shrinkage RDA全局头，并保证`alpha=0`时概率、预测、dtype和A输出SHA逐元素相同。

#### 失败经验总复盘与L0–L10代码修改链

前述失败不能笼统归因于“地面原型无效”。D93在ground coverage仅约0.144–0.227时对全坐标施加transport并替换强base，matched K10相对D81的old、new、H和floor均下降；D94只缩小错误方向幅度，没有修正方向。D99同时改变ground metric、Student-t局部头和D81融合，真实target r12中old-after、seen-new、H和floor分别相对D81下降23.8889pp、24.7500pp、24.3325pp和15.0000pp，forgetting增加17.5000pp，无法定位单一失败机制。D100在K10的融合系数为0，预测与D99完全相同，证明“再加一个全局头”不自动形成互补纠错。r10/r11等prediction=0的运行属于接口或重复GPU前向技术失败，不得混写成算法负结果。

Patch B首轮又暴露出另一类错误：公式和19项正常测试均正确，但Phase1锁未绑定A config、identity metric及ground prior/source/rank；state、ground prior和resource只在构造时验SHA，内存篡改或联合重签可绕过。第二轮修复入口receipt后，监督review继续复现`bool/string`数值类型别名；第三轮数学review进一步指出fit/quant指标即使类型正确也只是builder自述，普通self-hash无法证明其真实性。最终代码明确把三类诊断降级为`builder_reported_non_authoritative_not_for_promotion`，正式LODO必须从绑定输入现场重算并由外层authority封存。

|层|唯一输入/输出|必须通过的门|禁止项|
|---|---|---|---|
|L0因果臂合同|项目协议、A基线→A/B/C-id/C-dom/C-joint/D唯一delta|每臂只改一个机制，冻结identity等式、允许依赖和资源上界|把metric、kernel、transport和fusion重新混成D99式候选|
|L1 typed输入闭合|A bank、active-K support、sealed ground prior和Phase1 lock→validated inputs|classes/K/registry/receipt精确绑定；A config/identity和ground prior/source/rank预锁|inactive-K依赖、clean/source/query、跨row opaque handle复用|
|L2纯数学核|decoded support和class-agnostic prior→FP64 teacher state|K1 target residual DOF/rank精确为0；target rank≤2、ground rank≤4；Woodbury=dense|在数学核混入量化、query、角色分支和I/O|
|L3编译量化|teacher state→INT8 weight＋FP16 scale/bias|独立held-LODO量化top1、margin sign flip和large-margin flip门|把support-fit诊断当Phase1 authority或保留FP32 sidecar|
|L4 sealed state/wire|精确A/ground/lock绑定＋compiled arrays→byte-exact B wire|每个public consumer重验receipt；exact JSON/Python类型；恶意wire、内存篡改和错配fail closed|“有SHA就可信”、可替换先验、caller自报bytes/MAC|
|L5逐query融合|同一A bank＋B state＋单条z_id→A/RDA/fused probability|alpha仅Phase1冻结；alpha0跳过RDA且bit-exact A；query permutation/chunk等价|query置信度调alpha、伪标签、图、quota和batch统计|
|L6 Phase1 nested LODO|合法Phase1 single-observation archive→每K独立lock和外部receipt|独立runner现场重算fit/quant/resource；双向rescue、NLL、old/new/H/floor/forgetting联合门|用builder诊断、真实target或125选择alpha/rank/formula|
|L7本地集成与发布前review|精确Git commit和冻结matrix→LOCAL_VERIFIED handoff|修改前合同review；修改后diff、专项、相邻回归、独立P0/P1清零|远端直接改、多个agent并改同文件、技术失败冒充性能失败|
|L8 matched target窄验证|冻结六臂、同row K1/K10/new20/3场景→完整prediction与truth后连接|B-old/A-old/New/H/floor/min-old/min-new均不退，forgetting不增；逐类逐场景审计|根据窄结果回调任何科学参数|
|L9历史125 screen|唯一冻结候选→125完整稳定性结果|125只验证泛化稳定性，保持所有同row指标和资源审计|用125筛候选、rank、bitwidth、alpha或门|
|L10完整确认|通过125的同一commit→1,200评价单元|5 receiver×5 seed×3 scene×4K×4 new-count，无代码/lock漂移|把125称为完整目标矩阵|

#### Patch B：纯identity z_id160 qKNN＋Shrinkage RDA本地核心

Patch B新增`stage2_zid_srda_fusion.py`及其专项测试，只消费Patch A公开typed bank；不导入D81/D99/D100/D101，不读取z_dom、FFT/RF、receiver/TX、old/new角色、query truth或query-fit输入。RDA均值全部来自当前row全部注册类target support；ground仅提供Phase1 LODO封存的class-agnostic共享协方差basis/spectrum，不提供class mean、bias或旧类logit。K1时target scatter、nres和target rank均精确为0；K≥2使用class-balanced scatter，target positive rank≤2、ground rank≤4、总Woodbury rank≤6。融合系数`alpha_phase1`只由Phase1冻结；未实现support-CV或K−1 bank，因为Patch A合法active K仅为1/5/10/20，构造K−1 bank会破坏既有typed合同。

三轮独立review依次发现并修复：①state/ground内存可写后receipt未重验、Phase1锁未绑定A/ground、combined resource可拼接错row；②strict wire接受同步重签的state alpha bool、temperature string、lock sigma string和audit shrinkage string；③fit/quant合理数值可联合重签，self-hash不能充当性能真实性证明，B state中的A wire bytes也不可独立验证。最终所有serialize/fuse/resource入口重验state，decode/build重验ground prior；A config、identity metric和ground prior/source/rank全部进入Phase1锁；wire拒绝duplicate key、NaN、shape overflow、payload bitflip、截断/尾随、非canonical JSON和超过16MB总长；A wire字段从B state删除，combined resource从实际绑定A/B对象现场重算。

最终独立结论为：数学审计`P0=0、P1=0、P2=0、MERGE_NONFORMAL_ONLY`；监督审计`P0=0、P1=0、P2=3、MERGE_NONFORMAL_CORE`。保留P2是尚未逐项参数化所有wire拒绝分支、nested eigenvalue list仅浅冻结但public consumer会拒绝篡改、A component/Phase1/p2_min_v1真实authority仍由上层sealed admission核验。Patch B明确`formal_phase2_eligible=false`、`bundle_created=false`，当前提交不授权target、N607、125或正式性能结论。

|C/K/ground|B numeric arrays|B实际wire|A实际wire|component numeric sum|B linear-head MAC/query|build-support量化一致率|诊断authority|
|---|---:|---:|---:|---:|---:|---:|---|
|26/20/off|4,264B|8,704B|88,046B|89,596B|4,160|99.6154%|不可用于晋级的builder diagnostic|

上表来自固定随机本地资源探针，只是实现/资源表现，不是分类性能；其中B MAC只覆盖compiled RDA linear matmul，不含bias、softmax、fusion、hash、decode、normalize、序列化或端到端latency。完整combined wire container仍不存在，因此只能称component resource evidence，不能称deployment resource closed。最终文件SHA256：core=`8aef42e14518911284c781493a9c1b1331735ef63264a8d650bdda3176112439`，test=`b2667134896f8b19d5b831bb96d4e3ff92ad75fb25512112b11a54f451f8a8df`；专项28/28与`py_compile`通过。本节所在Git commit即Patch B版本authority。

Patch B下一步不是直接发布target，而是先实现独立Phase1 LODO runner：冻结每K候选和Git/config/A/ground receipts；只读合法Phase1 single-observation archive；从原始绑定输入现场重算fit、held-pseudo-query量化margin、A/RDA/B预测、双向rescue、NLL、old/new/H/floor/forgetting和资源；产出immutable外部receipt。只有matched LODO性能门、独立量化门、K1因果门和现场资源门全部通过，才允许生成六臂target窄验证release config。

提交前最终回归在`ssr-gpu`中覆盖Patch A/B、D81 typed/episode/ground、D99/D100 query/LODO和D101 RDA/LODO，共183/183通过，进程exit0；`git diff --check`通过。pytest结束后访问`pytest-current`的Windows Temp `PermissionError`仍为已知atexit清理噪声，不改变测试结论。

#### 分享设计吸收裁决与Patch C-id v0本地诊断核心

2026-07-22针对用户提供的ChatGPT分享对话完成域适配、分类头和遗忘/floor监督三路交叉审查。分享设计中可吸收的主干是：将receiver公共偏移、sample-level LEO/channel扰动和TX身份残差分开建模；以类内散度和类间保护构造低秩nuisance subspace；只对所有类共享的低秩方向做软抑制；身份memory与decision head分离。该方向与当前`A(identity z_id＋single-qKNN)→B(identity z_id＋SRDA)`因果链兼容。首轮明确不吸收vMF替换、local scaling、hubness、PoE、类特异QDA、碰撞重编译、partial equalization、`U_H`或`24 rx＋72 id`的FFT96改造，因为这些机制会同时改变kernel、表示、校准或融合，重演D99无法归因的混合失败。

监督最终裁决为`REVISE`：C-id v0只按`MERGE_LOCAL_DIAGNOSTIC`落地；主方案仍是C-dom receiver-context correction，但必须等待新的Phase1共同封存bundle，禁止复用D99/ground bundle或换名进入target。新bundle至少需要在合法Phase1 single-observation archive上经nested receiver/day LODO封存`U_R^dom`、rank≤4的`z_dom→z_id`交叉映射、中心/尺度、`D_eff`、coverage、LOCO一致性和Phase1锁定的收缩规则。Phase2届时只允许从当前row全部注册类support按类均衡估计共享receiver context；K1只能估该共享context，不能估类内scatter或sample-channel basis。首版C-dom仍不引入`U_H`、partial equalization和FFT96重构。

C-id v0实现冻结为：

\[
S_W=\frac1C\sum_c\frac1{K-1}\sum_k(z_{c,k}-\bar z_c)(z_{c,k}-\bar z_c)^\top,
\quad
G=S_W-\beta S_B,
\quad r\le2,
\]

从`G`的正特征方向中，仅保留类内能量达到Phase1锁阈值且`within/(within+between)`达到预锁nuisance比例的方向`U`，再通过Patch A已有typed metric闭合`M=I-U^T diag(a)U`。当前实现不另行变换或重建support bank，Patch A的INT8量化、class bandwidth、Student-t核、`logsumexp-log K_c`、temperature和逐query全类打分公式全部保持不变。`attenuation`、rank上限、between guard和选择阈值只能来自Phase1 nested LODO lock；该锁同时绑定精确Patch A config lock digest与identity metric receipt，不能只凭同一K替换temperature/bandwidth等A参数。target侧为零优化步。支持行顺序、注册表顺序和类名重命名不改变几何；输入表面不含query、receiver、role、scenario、ground或source。

K1执行严格可辨识回退：target类内scatter、rank和update均为0，并返回Patch A逐值相同的`identity_rank0` metric。该回退是协议正确性，不代表C-id通过P1-3的K1非identity目标；K1非identity只能由未来C-dom的Phase1冻结cross-map实现。K≥2若没有同时满足类内能量和类间保护的方向，同样精确回退identity。任何coverage、condition或receipt不满足上层封存要求时不得发布target。

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_zid_support_nuisance_metric.py`|C-id Phase1锁、class-balanced解析解、rank≤2 PSD软抑制、K1/无方向identity回退及typed audit|`d859cba8e1affa0f663157c81bb5c5d4c923a72eec707ad0e8ad329a9db9b636`|
|`tests/test_stage2_zid_support_nuisance_metric.py`|K1回退、低秩方向、类/行置换、类名重命名、Patch A bank不变、无合法方向回退、禁用输入表面和fail-closed测试|`c4b9a7e0b5c91c084d4d85f79ad49fd60e48909b2df518635bf1a43d73c8601c`|

本地验证在`ssr-gpu`中串行执行：

```text
conda run -n ssr-gpu python -m pytest -q tests/test_stage2_zid_support_nuisance_metric.py tests/test_stage2_zid_student_t_qknn.py tests/test_stage2_zid_srda_fusion.py
.................................................................. [100%]
67 passed, exit 0
```

`python -m compileall`通过。独立代码审查首轮发现并修复2项Important：①C-id锁原先只比较active K，现已共同绑定Patch A config lock digest与identity metric receipt；②SHA验证原先会把64位整数转为字符串，现已要求全部receipt/digest为精确`str`。相应的同K不同temperature、错误identity receipt以及非字符串SHA测试均已加入，修复后无Critical或未解决Important。`ssr-gpu`未安装`ruff`，因此未获得ruff证据；这不是测试失败，后续以`git diff --check`、专项/相邻回归和人工diff审查闭合。本节没有运行target/N607实验，没有生成bundle，也没有授权K1、D、125或正式性能晋级。

后续顺序冻结为：①实现独立Phase1 nested LODO runner，并对C-id加入`eta=0`、support标签置乱、随机/置乱`U`负对照；②只有C-id在held receiver/pseudo-new联合门上相对A无old/new/H/floor/min-class退化、forgetting不增且balanced NLL改善，才允许一次预登记`K10/new20×3 scenes`机制诊断；③并行研发新的C-dom Phase1 bundle；④C-id与C-dom分别对A取得独立正收益前，C-joint与D保持`HOLD`。D未来只能采用`best-C qKNN expert＋固定B-RDA(raw z_id) expert`，不得让B在C变换后重新拟合却仍声称`D.head_hash==B.head_hash`。

#### 开放方法设计波次与`JOINT-RCHM-BPP/r1f`冻结

2026-07-22在完整复核本报告、当前治理文档和用户给出的[Phase2设计对话](https://chatgpt.com/share/6a60a592-a60c-83ec-bfe6-eecd9780dac4)后，分别完成域适应、统一分类头和联合可行性监督。该设计波次保持只读，没有重验`VALIDATED_ONCE`数据，没有修改数据builder、Patch0、Patch A/B/C-id，也没有启动本地性能实验或N607任务。分享方案中吸收receiver公共因素、sample-level不确定性、类均衡support估计和身份/决策分离；不直接吸收多视图、partial equalization、hubness、PoE或整套联合包。若共享平移、正交变换、等比缩放或完整协方差重估可抵消decision geometry，则不把它计为有效域适应。

##### 方法卡1：`JOINT-RCHM-BPP/r1f`

- 状态：`DESIGN_DRAFT→FEASIBILITY_REVIEW(MERGE)→DESIGN_FROZEN`，是本波次唯一优先候选；没有target或promotable性能含义。
- 机制：Phase1通过nested receiver/day LODO共同封存TX低可解码的receiver basis、`z_dom→metric`cross-map、class-agnostic BPP先验`a0/b0/T_KC`及全部门限。Phase2从当前row全部注册类support类均衡估计共享receiver context，生成`M(c)=I+U diag(a(c))U^T`的非等值低秩PSD metric；统一分类头以类均值、RSS和Phase1 inverse-Gamma先验给所有old/new类计算同式Bayesian posterior-predictive Student-t似然。
- 协议与可辨识性：只读共同封存bundle、当前row support标签及逐query的`z_id/z_dom`，query不更新状态；无clean/source、query truth、role、quota或全局重分配。`D_eff≥6`且`r≤min(4,floor((D_eff-2)/2))`；K1只靠跨类公共context启用最多rank2，BPP的`RSS=0`并完全收缩到Phase1先验；K5/K10只降低context方差并启用类内RSS，不增加DA自由度。
- 几何与互补性：只允许能改变方向残差、类内半径和margin相对关系的非等值各向异性metric；receiver公共轴向失真由DA处理，类密度、support噪声及old/new竞争由BPP处理。类名、注册表和support行置换必须等价，所有类使用uniform prior和同一公式。
- 回退与风险：coverage、manifold distance、leave-one-class-out稳定性、condition或int8任一不过门，必须在评分前整行bit-exact回到identity scorer；不得以角色、类别或query置信度选择回退。主要风险是context混入TX残差、BPP先验误校准、K1有效样本不足和int8 margin翻转。
- 资源冻结：Phase2新增参数和optimizer step均为0；combined wire≤128KiB，support build≤0.34MMAC，联合后处理≤8kMAC/query；每query只允许一次冻结dual forward和一次score，无第三次前向、图或batch优化。
- falsifier：正确receiver context不优于zero/permuted context；`M≠I`但margin/prediction不变；`I_syn(H_old_new)≤0`；联合臂弱于任一单组件；old/new/min-old/min-new/floor任一下降、forgetting增加；top1一致率<99.5%、large-margin flip>0或资源超帽，均立即拒绝该revision。

##### 方法卡2：`JOINT-RCHM-SKR/r0`

- 状态：`FEASIBILITY_REVIEW(REJECT)`，不生成冻结revision，不允许改代码或发布实验。
- 机制：沿用RCHM metric，以全类centroid Gram矩阵和ridge逆形成simplex kernel/ridge统一头；能改变全类decision boundary，但每次注册新类会重算全局逆矩阵并漂移全部旧类score。
- 协议/可辨识性：输入表面可在`p2_min_v1`内闭合，K1也可由RCHM跨类context产生非identity几何；但扩类稳定性、类置换等价和old-score保持尚未成立。
- 资源/风险/falsifier：C=26时理论上可运行，但`O(C²)`state、`O(C³)`fit及完整wire/时延未闭合。任何扩类后旧类score漂移、floor下降、联合不优于单组件或`I_syn≤0`即停止。

##### 方法卡3：`JOINT-FNP-BPP/r0`

- 状态：`FEASIBILITY_REVIEW(REVISE)`，保留方法族但当前revision不落地。
- 机制：新Phase1 encoder为每份received IQ输出rank≤4的sample-level nuisance variance，Phase2用`Σ_q+Σ_s`的Woodbury precision边缘化query-support异方差，再以BPP统一分类。
- 协议/可辨识性：只读冻结variance head/basis、support embedding/variance和query自身临时variance，不用query batch或状态更新；K1因variance来自冻结单样本预测器而可非identity，K5/K10仅增加独立证据。
- 风险/资源/falsifier：pairwise异方差可能与BPP类半径重复解释同一噪声，新encoder总MAC/时延和TX身份泄漏门未闭合。必须先补zero/permuted variance、BPP radius masking、标签不可解码和完整int8生命周期；若variance可解码TX、联合只改善NLL而不产生argmax双向救援、`I_syn≤0`或资源超门即拒绝。

##### 方法卡4：`JOINT-FNP-SKR/r0`

- 状态：`FEASIBILITY_REVIEW(REJECT)`，不生成冻结revision。
- 原因：同时引入新encoder、pairwise likelihood和扩类全局逆矩阵，不能维持单一主要delta；sample-level DA可辨识性尚未证明时又叠加old-score漂移、floor和`O(C³)`fit风险。只有FNP可归因性与SKR扩类稳定性分别形成新revision并独立过门后，才可重新讨论该组合。

##### 冻结四态、证据包和下一步

`JOINT-RCHM-BPP/r1f`使用同一support/raw bank冻结四态：`M0=(identity metric,Patch A head)`、`M_DA=(RCHM metric,Patch A head)`、`M_HEAD=(identity metric,BPP head)`、`M_JOINT=(RCHM metric,BPP head)`。唯一协同主指标为：

\[
I_{syn}(H)=H_{JOINT}-H_{DA}-H_{HEAD}+H_{M0}>0.
\]

同时要求DA与head各自至少有独立正收益，联合臂产生old与new两个方向的wrong→correct救援，且各方向wrong→correct均超过correct→wrong；不得用平均值掩盖min-old、min-new或全类floor退化。首个实现范围冻结为新增RCHM纯核、BPP纯头、joint receipt和相应协议/bit-exact/int8/resource测试；第二步才实现只读合法Phase1 single-observation archive的nested receiver/day LODO/LOCO runner。未取得完整Phase1 held四态证据前，不生成target bundle、不发布N607，也不以本地fit、量化重构或代码测试声称性能成功。

#### `JOINT-RCHM-BPP/r1f`本地非正式核心闭合

2026-07-22在`DESIGN_FROZEN`后进入`IMPLEMENTING`，只新增下列6个文件；未修改Patch0、Patch A、Patch B、C-id、数据builder、protocol schema或任何既有科学规则，未触发数据重验：

- `code/cvsrffi/stage2_receiver_context_hypermetric.py`：只读Phase1 RCHM锁、类均衡receiver context、预算感知rank收缩、整行identity回退和实际build MAC账本。
- `code/cvsrffi/stage2_bayesian_predictive_head.py`：全部注册类共式Bayesian posterior-predictive Student-t头，K1完全收缩到Phase1先验，K>1使用metric-specific RSS。
- `code/cvsrffi/stage2_joint_rchm_bpp.py`：同一support bank的`M0/M_DA/M_HEAD/M_JOINT`四臂、typed wire、外部expected SHA、递归receipt、锁绑定和资源审计。
- `tests/test_stage2_receiver_context_hypermetric.py`、`tests/test_stage2_bayesian_predictive_head.py`、`tests/test_stage2_joint_rchm_bpp.py`：协议负例、置换/分块等价、回退、wire攻击、int8/FP16生命周期和资源边界。

实现保持`formal_phase2_eligible=false`和`bundle_created=false`。Phase2新增可训练参数为0、optimizer step为0；support bank、target context、metric basis、BPP类均值和充分统计量只以INT8/FP16进入持久化wire，无可评分FP32 sidecar。每条query只读冻结state并独立在全部注册类上评分，不读query truth、role、真实batch类数、quota、clean/source或其他query。

##### 关键实现闭包

- RCHM仅用当前row全部注册类support的类均衡`z_dom`均值形成公共context；coverage、有效类数、manifold、LOCO、quantization或condition不过门时，评分前整行回到identity。`attempted_rank`和`execution_stage`进入RCHM receipt，使condition回退也保留回退前已执行的Gram成本。
- 冻结规则只要求`r≤4`。资源选择在执行超帽rank前完成；最大压力探针`C26/K20/p8`自动收缩到rank3。这里的K20仅用于本地资源上界压力测试，不是`p2_min_v1`下K1/K5/K10的方法选择或性能证据。
- BPP保留完整multivariate Student-t类条件式；所有old/new类使用uniform prior和同一公式。Phase2不再执行target support×all-class teacher/compiled评分；只保存6个不同量纲、分别限幅的non-authoritative compiled-stat误差。Phase1 held top1、margin和quantization receipt仍是独立权威绑定，不能由该support诊断替代。
- typed wire反序列化必须提供外部expected wire SHA、qKNN锁、RCHM锁和BPP锁，才能重建完整四臂可评分state。fixed record顺序、未知/缺失/重复字段、shape溢出、NaN、截断、尾随、bitflip、共同重签和内存篡改均fail-closed。

##### 独立审查与裁决

实现审查没有一次性放行。首轮修复了Student-t公式、typed wire和INT8持久化；第二轮发现support全类teacher/compiled诊断使真实build下界达到`9,690,912MAC`，裁决`REVISE`。删除该target诊断并完整计入3次Gram后，rank4压力行仍需`341,792MAC`而超过冻结帽，因此改为预算感知rank3。下一轮又修复condition identity回退的历史Gram账本及“单teacher阈值混合不同量纲”的语义错误。最终独立监督结果为：`P0=0，P1=0，P2=1(non-blocking)→MERGE_NONFORMAL_CORE`；唯一P2是六分量测试复用已篡改state，随后已改为每分量fresh合法nonidentity state并精确匹配具名gate错误，专项回归通过。

##### 本地验证与资源证据

验证环境：`ssr-gpu`。主流程串行执行3个新增core的`python -m py_compile`，再执行新增3组和Patch A/C-id相邻2组测试，最终结果为`52 passed`。首条PowerShell包装命令因外层提前展开`$env`/`$LASTEXITCODE`产生包装噪声；按本项目规则串行重跑后命令干净退出0，未把包装噪声当成项目成败。

|本地压力对象|effective rank|完整wire|support build|BPP后处理/query|结论|
|---|---:|---:|---:|---:|---|
|C26/K20/p8|3|110,951B|255,232MAC|4,718MAC|三项均过冻结帽；仅资源压力证据|
|condition identity探针|0，attempted rank=1|已完成typed roundtrip|1,632MAC|identity BPP路径|保留context、cross和2次已执行Gram|
|pre-build effective-class identity探针|0|已完成typed roundtrip|1,280MAC|identity BPP路径|只执行context projection|

最大压力对象的实际matmul ledger为：`1×160×8`receiver projection、`1×8×4`cross-map、两次`3×3×160`metric Gram、`520×3×160`共享support projection和一次`3×3×160`BPP logdet Gram。build期把`_metric_d2`替换为强制抛错后仍能成功构建，证明不再隐藏执行`520×160×26`全类评分。MAC口径是具名matmul执行账本，不包含add/reduction/elementwise/log/hash/serialization，也不等同端到端latency；本轮没有GPU latency、峰值显存或target性能测量，因此不能作资源部署或性能晋级结论。

新增文件SHA256：

|文件|SHA256|
|---|---|
|`stage2_receiver_context_hypermetric.py`|`7c08a53bf0c38be45475f04e1e5ddb99b2bcdc8501bacfec17e489d94d13b58a`|
|`stage2_bayesian_predictive_head.py`|`5ba59f9fcf6b6f8177c51b73950670b2c35cc88721ea0e0b42f1b86309457b07`|
|`stage2_joint_rchm_bpp.py`|`05dbb835a77f0dddd28ee8e2a2141d4ee3eaab1ca6d619ca54900b4fa9e108d3`|
|`test_stage2_receiver_context_hypermetric.py`|`e822490d20cbe9b351412366166e581df49865b2605b66f63f07f71c29efeb34`|
|`test_stage2_bayesian_predictive_head.py`|`6761ac55d8667d566376901e67a32bf1aec1e8af42bc104b981783183ef6d11f`|
|`test_stage2_joint_rchm_bpp.py`|`23a40ad14b98c9f2031edfaed8703f468ae4e12f198c932406b5f042b4081760`|

##### 证据边界与下一artifact

本节状态为`IMPLEMENTING→LOCAL_VERIFIED_NONFORMAL_CORE`，不是`LANDED`、`ARTIFACTS_COMPLETE`、target performance或promotable success。没有生成prediction，因此不存在old/new/H/BA/floor/forgetting或协同收益数值；代码测试、support fit、量化误差和资源探针均不能替代性能结果。未创建target bundle、未访问N607、未发布run，也未建立125 screen。

下一唯一artifact是只读合法Phase1 single-observation archive的nested receiver/day LODO/LOCO runner及其held receipt：先在K1/K5/K10分别冻结RCHM basis/cross-map/gates和BPP先验/六分量量化阈值，再产出同一held row的`M0/M_DA/M_HEAD/M_JOINT`四臂。只有该artifact同时证明DA、head各自独立正收益，`I_syn(H_old_new)>0`，old/new双向wrong→correct救援超过correct→wrong，min-old/min-new/floor不降、forgetting不增、top1一致率≥99.5%、large-margin flip=0且资源仍过门，才允许生成target bundle并登记N607窄实验；否则按对应falsifier停止该revision。

#### `JOINT-RCHM-BPP/r1f` Phase1 held入口审计与双表征归档spike

2026-07-22在本地非正式核心闭合后，主线先审计下一artifact的真实输入，而未直接编写held runner。既有`cvs.phase1.single_leo_feature_archive.v2`历史运行声明8400行、6类、7个receiver、4天和3场景，但当前本地只有manifest，没有5.17MB NPZ；其`features[8400,288]`仅为`z_id160+FFT96+RF32`，没有冻结RCHM必需的`z_dom160`，也未保存被选观测的权威`overlay_id`。此外，本地快照中的manifest、selection receipt和cache-set因换行转换出现字节SHA漂移，只能用于语义审计，不能送入严格loader。上述缺口不是数据协议失败，也没有触发数据重验：received IQ、物理ID、receiver/TX、场景、K、support/query划分和`p2_min_v1`均未改变。

三路审查中，runner架构草案曾假定旧archive可直接复用；数据实物审计否定该假设。监督对“立即实现nested receiver/day held runner”的唯一裁决为`REJECT`：缺少真实NPZ、`z_dom160`和可验证的receiver×day×TX覆盖时，不得生成held prediction或性能结论。随后把唯一允许动作缩小为development-only双表征单观测归档`FEASIBILITY_SPIKE`；监督裁决`MERGE`，不改变冻结RCHM/BPP/joint核心、旧exporter、cache、allowlist或authority。

##### 冻结归档合同

- 每个由既有selection salt选中的source-validation received-IQ只进入一次Patch0 dual runtime调用，同次返回`z_id160`、`z_dom160`和`tx_logits`；不持久化IQ。
- NPZ精确成员顺序为`z_id,z_dom,tx_logits,labels,receiver_ids,day_ids,physical_ids,scenario_names,class_ids,observation_ids`；`observation_ids`逐行原样复制所选场景中已验证的`overlay_ids`，不得重算或用physical ID替代。
- 输入闭包绑定cache-set及三场景NPZ SHA、selection-salt receipt、runtime role/runtime SHA、export receipt、parity receipt、checkpoint/adapter SHA、input length和三输出schema。
- 生产路径只复用既有v1-only source-validation loader和既有known-SHA allowlist，不放宽为v1/v2通吃；输出保持`formal_phase2_eligible=false`、`bundle_created=false`。
- 有序`class_ids`作为显式缓存标签注册表进入composite manifest。现有runtime receipts不能证明`tx_logits`列到类别的映射，因此`tx_logits`只保留为raw checkpoint column-index审计，`held_runner_tx_logits_allowed=false`；后续held方法不得按类解释或消费它。
- 输出先在同父唯一staging目录完整写入并验收，再单次rename发布；任一写入、SHA、schema、shape、dtype、有限性、唯一性或语义检查失败时，只清理本次staging，最终目录保持不存在且不可覆盖。

##### 实现、独立review与验证

本轮只新增`code/scripts/export_phase1_singleobs_dual_feature_archive.py`和`tests/test_export_phase1_singleobs_dual_feature_archive.py`。首轮review为`P0=0,P1=3→REVISE`，修复了未冻结class registry、未显式封存三场景SHA和非原子双文件发布；第二轮为`P0=0,P1=2→REVISE`，进一步修复真实r4的v1 cache不能通过默认v2 loader、以及独立verifier未检查三场景SHA mapping。最终非作者复审为`P0=0,P1=0,P2=0→MERGE`。

主线首次尝试`conda activate ssr-gpu`时非交互PowerShell未加载Conda hook，实际落到base Python并报缺少pytest；该命令没有运行项目测试，记为环境激活噪声。显式加载`F:\App\miniconda3\shell\condabin\conda-hook.ps1`后，确认解释器为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，随后`py_compile`通过，并串行复跑新归档、旧单表征归档、Patch0 dual exporter和dual parity四组测试，结果`44 passed`、exit0。TorchScript弃用/trace警告及pytest临时目录atexit权限提示均未改变exit0。

|文件|SHA256|
|---|---|
|`code/scripts/export_phase1_singleobs_dual_feature_archive.py`|`264ec80d68c4468d086fc96ba4e134ab429a007f457a0c57ec365aca2efa3b32`|
|`tests/test_export_phase1_singleobs_dual_feature_archive.py`|`cce3ece0dc8f97737c0a9a7b8b5101baca8abb6f560a17a6e29d57daba7ca31f`|

本节状态为`FEASIBILITY_SPIKE_LOCAL_VERIFIED`，不是`LANDED`、真实archive、held evaluation或性能结果；prediction仍为0，状态保持`NO_PERFORMANCE_RESULT`。下一唯一artifact是在新的不可覆盖run报告中，由单一`gpt-5.6-terra high` runner在N607生成并回收：strict ADV3B02 base dual runtime/export/parity receipts、真实8400行双表征archive/manifest，以及不读取特征值的receiver×day×TX计数与fold可行性receipt。只有这些实物逐项闭合后，才重新审查held runner；当前不生成target bundle、不访问target/query、不运行125。

##### 双表征归档run本地发布冻结

run ID冻结为`rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1`，方法源提交为`9ca1a59a7522393c43ee09c7f95dde6588cd8f4a`，Git归档SHA256为`95127701d2c9f9989fcc6409b1e069e232f2d3e3654611d92e9ac2abe26937a0`。强制run报告与wrapper已同时保存在根目录report面和本Git镜像；wrapper SHA256为`eb4e591f875434bb7e7f4c90b6a020435f3d7f356b4e05a33091231438210ffd`。

首轮独立发布审查发现coverage只记录不阻断、清单路径漂移、PID/目录顺序不闭合和`set -u` GPU变量错误路径，裁决`P0=1,P1=4→REVISE`。最小修订后，`bash -n`、embedded Python compile、synthetic coverage正例1个与row-count/zero-cell/K10-min负例3个、unset CUDA退出70均通过；复审为`P0=0,P1=0,P2=0→MERGE`。当前仅达到`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`，尚未访问N607；下一步由该run ID的唯一Terra runner执行preflight、落地、运行、短连接监控和artifact回收。

##### 双表征归档r1远端技术失败与GEOFF/r2冻结

唯一runner经direct preflight在GPU0启动`rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1`一次；远端ZIP、wrapper、6个源文件、checkpoint、adapter、v1 cache和selection salt SHA、解包、`py_compile`及`bash -n`全部闭合。child exit=1：export receipt为PASS，但独立base runtime/checkpoint parity在batch1/8/256呈0→非零，256行maxabs为`z_id=1.9640e-4`、`z_dom=5.0431e-4`、`tx_logits=3.2592e-3`，超过冻结`1e-5`，因此archive、coverage和prediction均未生成。状态永久为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，原run ID禁止复用。

回收runtime的本地无数据探针把根因定位为TorchScript CUDA graph executor冷/热执行计划切换：相同256行fresh第1→2次漂移、2→3次全0，首图含843个`prim::profile`、后续图为0；CPU稳定，仅关闭graph executor optimization后CUDA连续三次全0。`P1-DUAL-ARCHIVE-GEOFF/r2`经独立监督`MERGE`并`DESIGN_FROZEN`：唯一delta是在export、verify和archive consumer的首次JIT边界前fail-closed封存并回读`graph_executor_optimize=false`，升级v2 contract并绑定精确Torch/CUDA版本和SHA；`1e-5`不放宽，batch1/8/256的eager-vs-runtime及runtime连续3次均需通过。任何API/readback/version/hash/数值/语义门失败都停止且不得生成archive。

##### r1f held-runner静态监督裁决

真实archive缺失期间只读草案发现当前6类合同与冻结`D_eff≥6`冲突：pseudo-new LOCO后的before只有5类，所有K下必然`effective_class_identity`，故结构上`M_DA=M0`、`M_JOINT=M_HEAD`；after到6类才可能启用RCHM，registration forgetting会同时混入类别竞争与DA开关变化。三个core还只提供Phase2 consumer，没有从archive确定性拟合Patch A bank/qKNN锁、RCHM basis/cross-map/gates和BPP先验/量化门的Phase1算法。

独立监督对“直接实现r1f held runner”的裁决为`REVISE`。这不是协议违法或性能负结果，而是因果与接口不可辨识；必须创建下一candidate revision并完整重走`DESIGN_DRAFT→FEASIBILITY_REVIEW→DESIGN_FROZEN→IMPLEMENTING`。当前r1f core只能作为只读消费组件复用；before须明确登记为identity结构null，`I_syn`只在after计算，forgetting只声明端到端注册转移，且Phase1 lock-fitting算法、`(K,C=5/6)`锁语义、zero/permuted context负对照和量化teacher必须在任何held prediction前冻结。唯一运行侧下一artifact仍是GEOFF/r2成功生成的真实archive/manifest/coverage receipt。

##### GEOFF/r2实现、复审与发布源边界

`P1-DUAL-ARCHIVE-GEOFF/r2`已在正式Git工作树实现，范围严格为dual runtime export、独立checkpoint parity、dual archive consumer及三份对应测试。实现把export/parity/archive schema升级到v2，首次JIT边界前fail-closed设置并严格回读`graph_executor_optimize=false`，把Torch版本、CUDA版本、设备、`max_abs=1e-5`和canonical contract SHA封入并交叉核验；fresh batch1/8/256均对同一eager输出执行runtime第1/2/3次比较，容差未放宽。

主线在显式加载Conda hook后的`ssr-gpu`环境完成`py_compile`、35项GEOFF专项与dual-forward/joint core相邻回归，最终`48 passed`、exit0；TorchScript弃用/trace警告及pytest临时目录清理权限提示均未改变exit0。对r1回收candidate runtime的本地无数据CUDA探针严格回读`False`，batch1/8/256的第1↔2和第1↔3三输出最大差全部为0。首轮独立Terra审查为`P0=0,P1=1,P2=0→REVISE`：v2 receipt实际执行3次却同时声明`runtime_invocations_per_parity_batch=1`。最小4行修复把verifier、archive consumer、fixture和正式断言统一为3；复审为`P0=0,P1=0,P2=0→MERGE`。这只证明本地技术实现闭合，不是archive、prediction或性能结果。

旧r1本地解包树`E:/type10-7/code/snapshots/rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1/source_9ca1a59a/`在一次被主线及时阻断的错误实现落点中有6个GEOFF源码/测试文件与不可变ZIP成员SHA不同，现标记为`CONTAMINATED_LOCAL_EXTRACTION / DO_NOT_RELEASE`。原ZIP SHA256仍为`95127701d2c9f9989fcc6409b1e069e232f2d3e3654611d92e9ac2abe26937a0`，远端r1源未变；后续发布只能从新Git提交重新生成archive，不能复用该解包树或原run ID。

##### JOINT-RCHM-BPP/r2a一次可行性监督

同一设计波次的DA草案使用Phase1 dual archive拟合robust receiver context、`z_dom→z_id`低秩cross-map与27个预注册family，HEAD草案使用统一全类Patch A/BPP、nested receiver/day/class/pseudo-new隔离、K1/K5/K10锁和FP32/FP64量化teacher。独立Sol-max联合监督裁决为`REVISE`，未生成`DESIGN_FROZEN`且禁止实现。首个静态falsifier已触发：inner C4→C5在core的`D_eff≥6`合同下before和after均为identity，无法选择任何nonidentity DA family或证明correct context优于zero/permuted；使用outer C6选择又会污染held证据。

同次监督还要求下一revision闭合从split receipt到Q/R/B typed locks的确定性总函数、`b0`统计与正根存在条件、C5/C6充分统计/未归一化logit/posterior口径、整row回退、量化state销毁以及真实wire/MAC/时延/显存。当前唯一获准的下一artifact不变：先由GEOFF/r2新run生成真实dual archive、manifest和只读coverage receipt；没有这些实物前不进入第二设计波次、不写method fitter、不产生prediction。

##### GEOFF/r2新run发布冻结

全新run ID为`rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2`，方法提交为`ca5d0c4bcf8fb295cdfb70e067f9009617bb3a5f`，release-control提交为`d45f4cc22ac379c287ad09baed53fe07cdb791d2`。commit-bound Git archive SHA256=`5adbef8a1ebf2f0846132226f702e95648c99334a0ba5296b7487e45095e4778`，wrapper SHA256=`e1f497a757d54cef95a9559ac3de910a26cf2d9a3d0407d3cc865b628847afcf`；根报告与Git镜像逐字节一致。

独立最终发布审查为`P0=0,P1=0,P2=0→MERGE`。冻结run只执行GEOFF/r2 base parity→base dual archive→元数据coverage，export/parity/archive使用v2 contract，coverage保持既有v1合同；`1e-5`、batch1/8/256、每batch三次调用、8400行/168cell/K10余量和全部资产SHA均不变。不重复数据验证、不访问target/query/held/125，retry=`NO`；技术完成也只标记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。

##### `R2A-FIXED-XCOV-BPP-K5/v1.1`实现追踪

本候选保持既有`DESIGN_FROZEN`机制、K5、18个held slice及`M0/M_DA/M_HEAD/M_JOINT`四臂不变。独立代码审查发现truth封存、prediction行完整性、真实dual archive/coverage绑定及BLAS执行收据尚未闭合，裁决`P0=2，P1=2→REVISE`；以下是提交前唯一允许的修复范围，不增加候选结构、阈值或数据权限。

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|R2A-T1|独立review P0-1|校验truth schema、`truth_sha256`、packet绑定及18行逐row/query标签闭包|`code/cvsrffi/r2a_fixed_held_four_arm.py`、专项测试|verified|篡改truth后必须fail-closed|scorer才可解封truth|
|R2A-T2|独立review P0-2|prediction必须与packet的18个row一一对应、唯一、顺序及四臂结构完整，并绑定logits argmax|同上|verified|复制单row或篡改prediction并重签COMMIT必须拒绝|不得以72项数量代替row闭包|
|R2A-T3|独立review P1-1|CLI复用既有dual archive verifier，核验manifest schema、成员顺序、array/NPZ SHA|同上|verified|最小三字段manifest必须拒绝|不重复received-IQ数据验证|
|R2A-T4|独立review P1-1|显式读取并验封coverage receipt，绑定receipt SHA、archive SHA、manifest SHA及冻结coverage schema|同上|verified|伪造coverage SHA/schema/绑定必须拒绝|coverage只作held选择盐和准入收据|
|R2A-T5|独立review P1-2|在SVD执行点实际限制并核验BLAS线程为1，记录NumPy和BLAS/LAPACK实现/版本|同上|verified|receipt必须来自实时执行面|不改变SVD/rank机制|
|R2A-T6|提交门|补齐上述正负例并运行专项、相邻core回归、真实checkpoint无query smoke|专项测试及后续run report|verified|`ssr-gpu`合计19项通过；N607真实archive build→无标签predict→score exit0|不得用测试宣称性能|
|R2A-T7|冻结证据包与当前性能优先指令|按`H_JOINT-H_DA-H_HEAD+H_M0`计算同row`I_syn`，并输出old adaptation gain、逐类、receiver、scene、K及双向混淆|scorer及专项测试|verified|四臂同row公式精确断言|只修正证据输出，不改变decision geometry|

独立复审确认prediction已严格绑定同一logits的逐行argmax，重签COMMIT不能掩盖prediction篡改；最终裁决为`P0=0，P1=0→MERGE`。当前仅达到本地实现门，不构成archive、coverage、prediction或性能结果。

##### `R2A-FIXED-XCOV-BPP-K5/v1.1`真实held结果与停止裁决

冻结实现commit为`8b163af1c3f43d94ef1f546da2306b43533c5046`，release-control commit为`e0707fb90c932067517210dadc74843818e7d9e5`。唯一N607 run ID为`r2a_fixed_xcov_bpp_k5_held_r1_8b163af1_20260723`；direct/GPU0/PID420304自然exit0，prediction=18 slices，score=72/72 rows。prediction SHA=`eb9593769100dba20451b5aa1b7d49999a2754cdaf6c2337dd6f6e854da2e7df`，score SHA=`bf6e5f55c8c8d33e754184b30af3fb6a36b142a173156d8ff3e9cc3b0d201222`；query NPZ只含1105个唯一`query_ids`和`z_id160 float32`，truth/COMMIT/argmax及全部artifact SHA独立复核通过。

|arm|old-before|old-after|seen-new|H|BA|floor|min-old|min-new|forgetting|old→new|new→old|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`M0`|0.817038|0.785392|0.781784|0.747766|0.781784|0.419908|0.446988|0.781784|0.031647|0.042596|0.218216|
|`M_DA`|0.817038|0.785392|0.781784|0.747766|0.781784|0.419908|0.446988|0.781784|0.031647|0.042596|0.218216|
|`M_HEAD`|0.799448|0.779418|0.776255|0.752412|0.776255|0.462804|0.486391|0.776255|0.020031|0.043620|0.223745|
|`M_JOINT`|0.799448|0.779418|0.776255|0.752412|0.776255|0.462804|0.486391|0.776255|0.020031|0.043620|0.223745|

RCHM在C6的18/18个slice均建立rank1非标量metric，但`M_DA-M0`的6630次after argmax变化为0；`M_JOINT-M_HEAD`仅1次wrong→wrong标签互换，所有准确率指标仍完全相同。18/18个slice的`I_syn(H)`均精确为0。BPP相对M0虽然把mean H提高0.004646、floor提高0.042896并把forgetting降低0.011616，但old-after、seen-new、BA和min-new同时下降，且old/new wrong→correct分别为225/45，低于correct→wrong的265/53。

最终裁决为`ANALYZED / COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`：该revision因DA决策退化、联合不优于HEAD及协同量为0而停止，不发布target窄实验，不运行125。完整18-row、逐场景/逐类、量化、MAC/state及缺失latency/VRAM边界见该run报告。

独立`gpt-5.6-sol high`结果审计重算18个slice、72个arm-row和6630次query，与score最大差`2.22e-16`，确认全部算术、同row绑定和停止裁决无误；审计结论为`P0=0，P1=0→MERGE_REPORT`。

runner期间完成的下一候选只读spike把`JOINT-CID-BPP/r0`裁为`MERGE_SPIKE`，但尚非`DESIGN_FROZEN`：它用K5 support-only C-id替换RCHM并保留BPP，必须先由独立监督闭合outer-train/nested LODO锁及C-id/BPP残差重复收缩，MERGE前不修改代码或发布实验。

##### `JOINT-CID-BPP/r0-spike`可行性冻结

独立联合监督裁决为`MERGE`，仅批准`DESIGN_FROZEN_FOR_FEASIBILITY_SPIKE→IMPLEMENTING`，不批准target、125或性能晋级。四臂固定为`M0=qKNN(identity)`、`M_DA=qKNN(C-id)`、`M_HEAD=BPP(identity)`、`M_JOINT=BPP(C-id)`；四臂共享同一K5 support bank、query、注册表和Patch A锁，`M_DA`与`M_JOINT`必须复用同一metric receipt，BPP只允许按该metric重编译RSS/projection/logdet，`a0/b0/T`、算法及量化门与`M_HEAD`相同。

每个outer pseudo-new lock先排除coverage确定的完整held receiver和当前pseudo-new class；该并集的全部physical ID、18个outer row ID及support/query ID以显式列表和SHA进入receipt。候选参数只可在剩余receiver×pseudo-new×scene的Phase1 nested LODO内选择，禁止读取outer held特征、标签或结果。预注册family为`(rank,attenuation,beta,min_fraction,min_energy)`：`F00=(1,.25,.5,.60,1e-7)`、`F01=(1,.50,.5,.60,1e-7)`、`F02=(1,.50,1.0,.75,1e-7)`、`F03=(2,.25,.5,.60,1e-7)`、`F04=(2,.50,.5,.60,1e-7)`、`F05=(2,.50,1.0,.60,1e-7)`、`F06=(2,.50,1.0,.75,1e-7)`、`F07=(2,.65,1.0,.75,1e-7)`；不得追加或删除family。

选择顺序冻结为：最大化`min(mean(H_joint-H_M0),mean(H_joint-H_DA),mean(H_joint-H_HEAD))`，再依次最大化mean`I_syn`、mean joint floor、最小化mean joint forgetting、降低rank、降低attenuation、按family ID稳定决胜。每个inner episode和outer row均执行5个同步leave-one-shot方向jackknife；若任一子拟合无方向，或最小归一化projector overlap低于预锁`0.50`，整row、query无关地精确回退identity。Phase1 receipt另报告jackknife query预测一致率、INT8 teacher top1一致率和large-margin flip；低于`99.5%`或large-margin flip非0则本revision停止。

冻结改动仅允许新增`code/cvsrffi/cid_bpp_phase1_nested_lodo.py`、`code/cvsrffi/cid_bpp_fixed_held_spike.py`和`tests/test_cid_bpp_fixed_held_spike.py`；既有C-id、BPP、R2A、archive、coverage和数据代码不得修改。最低held证据仍为同一18 slice、72个同row arm结果及完整prediction；若nonidentity C-id的`M_DA-M0`为零argmax变化、`M_JOINT=M_HEAD`、mean`I_syn≤0`，或联合臂损害old/new/min/floor并增加forgetting，则立即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不运行125。

##### `JOINT-CID-BPP/r0-spike-tech1`真实held结果与停止裁决

原r1在prediction前因verifier没有接受预注册的`jackknife_overlap→identity`fallback而exit1，prediction=0，严格记为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。修订commit=`e3aa2da5af520e493d40ec343b913ce24e7629dd`只修复fallback验证分支，没有改变fit、family、metric、BPP、数据或四臂；N607同环境packet verifier回执SHA=`b153049167629c4ccd1932934d5149d1868183dd4dcfb530a6d0df98f70113b3`并在launch前PASS。

新run ID=`cid_bpp_k5_held_r2_e3aa2da5_20260723`，direct/GPU0/PID475079/exit0，prediction=18、score=72。独立复算canonical seal、truth/prediction绑定、144个logit argmax和72行全部score，最大差=`0.0`。archive/coverage继续复用r8，未改变任何需要重验的数据协议输入。

|arm|old-before|old-after|old gain|seen-new|H|BA|floor|min-old|min-new|forgetting|old→new|new→old|I_syn|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|M0|0.817038|0.785392|-0.031647|0.781784|0.747766|0.781784|0.419908|0.446988|0.781784|0.031647|0.042596|0.218216|0.003168|
|M_DA|0.817570|0.785570|-0.031999|0.781784|0.747863|0.781962|0.418967|0.446046|0.781784|0.031999|0.042774|0.218216|0.003168|
|M_HEAD|0.824655|0.804355|-0.020299|0.799331|0.776810|0.799034|0.531354|0.543206|0.799331|0.020299|0.038612|0.200669|0.003168|
|M_JOINT|0.826082|0.806121|-0.019961|0.801428|0.780075|0.801013|0.533770|0.545614|0.801428|0.019961|0.038095|0.198572|0.003168|

M_JOINT相对M_HEAD的H为+0.003265、old-after为+0.001766、seen-new为+0.002096、floor为+0.002416、forgetting为-0.000339；相对M_DA的H为+0.032212。可是M_DA相对M0仅净增加1个正确old决策，floor/min-old各下降0.000942且forgetting增加0.000353。`I_syn`为正4/18、零13/18、负1/18；clear均值为负、rain为0，只有low-elev为正。逐类收益主要由`20-19`救援贡献，相对M0另5个旧类after accuracy下降。

因此技术闭环通过但性能预注册门失败，最终裁决=`ANALYZED / COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；不得事后放宽“DA、HEAD各自正收益且安全”的门，不运行125。本轮runner期间只读准备的下一候选`SVRN-BCR/r0`被监督裁为REVISE：必须把M_HEAD gate严格绑定raw support、M_JOINT gate严格绑定SVRN support后再冻结，以避免两臂互读破坏因果隔离。

##### `SVRN-BCR/r1` DESIGN_FROZEN

独立监督结果为`P0=0，P1=0→MERGE`。r0唯一P0为共享head-enable同时读取raw与SVRN两支；r1将其拆为branch-local `g_raw/g_svrn`。监督同时把不同量纲logit margin比较冻结为尺度不变score margin，并删除不必要的Phase1 selector。当前状态=`DESIGN_FROZEN→IMPLEMENTING`。

|ID|冻结要求|实现归属|验证|
|---|---|---|---|
|S01|只读immutable bundle、合法support及冻结常数；不读query truth/role/quota/clean/source|held模块|predict输入负例与truth密封|
|S02|SVRN固定`κ=2.5,η∈{0,.25,.5}`，并列取小|state模块|常数/receipt篡改拒绝|
|S03|mask固定`j mod5∈{0,1}`、保留率0.8；同物理ID全部view同步LOO|state模块|physical-ID排除测试|
|S04|双向选择一致且逐类方向不弱于η0才启用，否则identity|state模块|一致/回退构造例|
|S05|K1固定`η=0,g_raw=g_svrn=0`；K5/K10分别4/9个同类LOO中心|state模块|K1/K5专项测试|
|S06|BCR固定`D_ii=1/K`、centered Y、`λ0=1`和统一全类ridge|state模块|闭式/置换/退化测试|
|S07|raw/SVRN各自拟合qKNN、BCR和gate，不共享enable、统计或对方SHA|两模块|`g_raw≠g_svrn`隔离测试|
|S08|gate前固定`N(s)=(s-mean(s))/(||s-mean(s)||₂+1e-12)`|state模块|零范数/非有限回退|
|S09|`δ=m(N(BCR),y)-m(N(qKNN),y)`；逐类方向median≥0且全体median>0才开gate|state模块|边界/严格不等式测试|
|S10|四臂固定M0/Qraw、M_DA/Qsvrn、M_HEAD/raw switch、M_JOINT/svrn switch|held模块|四臂精确logit测试|
|S11|Switch为hard switch；两gate失败互不影响|state模块|独立false-path测试|
|S12|receipt封存mask、η-grid/tie-break、λ0、gate公式、branch SHA、逐类δ和fallback|两模块|wire篡改fail-closed|
|S13|M_DA/M_JOINT共享η receipt；HEAD只绑raw gate；JOINT只绑svrn gate|held模块|跨支SHA负例|
|S14|support bank和BCR权重int8封存；top1≥99.5%、large-margin flip=0|state模块|FP32/int8一致性与sidecar拒绝|
|S15|`d=160,C≤40`、参数0、optimizer step0、state≤256KB|两模块|resource ledger测试|
|S16|BCR LOO复用一次分解/rank-one downdate或等价闭式，禁止逐support完整`d³`|state模块|MAC ledger/调用计数|
|S17|同r8句柄、rx1-1/K5、6 pseudo-new×3 scene=18 slice/72 row；SVRN专属封包|held模块|build/predict/score闭环|
|S18|η全identity或DA零变化、两gate全关或HEAD无救援、单组件伤害保护指标即判负|scorer/report|同row停止门|
|S19|JOINT不弱于DA/HEAD、mean`I_syn>0`且收益不只来自单class/scene才可进125|scorer/report|18-slice/scene分布|
|S20|代码范围仅新增2个模块＋1个专项测试|Git diff|路径allowlist|
|S21|不新增Phase1 selector，不改数据、encoder、qKNN、CID、r8或runner控制面|Git diff|路径/依赖审计|

SVRN公式固定为`LN(z)=(z-mean(z))/sqrt(mean((z-mean(z))²)+1e-6)`，`Rκ(z)=||z||₂·clip(LN(z),-κ,κ)/(||clip(LN(z),-κ,κ)||₂+1e-6)`，`Tη=(1-η)z+ηRκ(z)`。BCR固定为`W=(HᵀDH+λI)^-1HᵀDY`，`λ=tr(HᵀDH)/d`。四臂实验只用于机制证伪；mask只能解释为稳定性探针，不声明物理频带、接收机维度或真实域因子发现。

冻结文件范围：

- `code/cvsrffi/stage2_svrn_bcr.py`
- `code/cvsrffi/svrn_bcr_fixed_held_spike.py`
- `tests/test_svrn_bcr_fixed_held_spike.py`

最小held falsifier保持rx1-1/K5/18 slice/72 row及完整old-before、old-after、seen-new、H、BA、floor、min-old、min-new、forgetting、双向混淆、逐类/scene、gate/fallback、int8、state、MAC、mean/P95和VRAM；未达到S18/S19即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不运行125。

##### 主路线纠偏与`SVRN-qKNN-BCRR/r2` DESIGN_FROZEN

用户随后冻结主路线为`domain adaptation＋qKNN＋other`：qKNN必须始终承担全部注册旧类/新类的统一逐样本决策，other只能解决剩余误差，不得用第二head替代qKNN。因此`SVRN-BCR/r1`在Git提交和N607发布前停止，状态=`SUPERSEDED_BEFORE_COMMIT / NO_PERFORMANCE_RESULT`；其未提交SVRN、BCR、int8和held封包实现仅作为r2工程复用基础，不形成方法证据。

本波次比较两张卡：`SVRN-qKNN-BCRR/r2`复用SVRN与BCR但改为连续qKNN残差；`CSRDA-qKNN-LDC/r0`复用D92/RDA经验。独立监督裁决A=`MERGE(P0=0,P1=0)`，B=`REVISE(P0=0,P1=4)`；B的未闭合项为条件数符号/退化回退、LDC幅度`m`未定义、重新引入hard gate、K5奇偶fold缺类。唯一冻结候选为A。

|ID|`SVRN-qKNN-BCRR/r2`冻结要求|
|---|---|
|R01|DA固定为既有SVRN `Tη`，`η∈{0,.25,.5}`，双向masked-view一致且逐类不劣才启用|
|R02|K1固定`η=0,ω_raw=ω_svrn=0`；K5首个falsifier；K10只确认不改公式|
|R03|qKNN始终使用INT8 support bank、全注册类同式、逐query独立竞争|
|R04|对logit固定`N(s)=sqrt(C)(s-mean(s))/||s-mean(s)||₂`；退化时仅当前query回退qKNN|
|R05|OTHER固定`F=N(Q)+ω[N(B)-N(Q)]`，`0≤ω≤0.5`，qKNN权重始终不低于BCR|
|R06|raw/SVRN各自在同物理ID LOO双向cross-view逐类非增损失安全集中求`ω*`；branch间不互读|
|R07|部署权重固定`ω_q=floor(254ω*)/254`；量化后复核安全集，失败仅该branch回退0|
|R08|M0=`Q_raw`，M_DA=`Q_svrn`，M_OTHER=`F_raw`，M_JOINT=`F_svrn`|
|R09|M_DA/M_JOINT共享同一SVRN receipt；M0/M_OTHER共享同一raw qKNN receipt|
|R10|BCRR不得改变qKNN bank、邻居ID、顺序或kernel；只允许融合后的argmax变化|
|R11|DA必须报告邻居、argmax、wrong→correct、correct→wrong和净正确变化|
|R12|M_DA总净正确>0且old/new净变化各自不负；M_OTHER必须有独立正收益|
|R13|单组件和JOINT必须保护old-before/after/gain、seen-new、H、BA、floor、min-old/min-new、forgetting及双混淆|
|R14|mean`I_syn(H)>0`、正协同slice≥9/18、正scene均值≥2/3，否则判负|
|R15|任一性能门失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不得运行125|
|R16|state≤256KB、optimizer step≤50、无query graph/批优化、正式state无FP32 sidecar|
|R17|int8 top1≥99.5%、large-margin flip=0，并报告state、MAC、mean/P95、VRAM和前向次数|
|R18|最小held固定rx1-1/K5/6 pseudo-new×3 scene=18 prediction/72 score|
|R19|唯一代码delta为把未提交r1 hard-switch改成branch-local连续BCRR并完成四臂接线|
|R20|只允许现有3个候选文件，不改数据、encoder、qKNN、CID、r8、coverage或runner控制面|

冻结文件仍为`code/cvsrffi/stage2_svrn_bcr.py`、`code/cvsrffi/svrn_bcr_fixed_held_spike.py`、`tests/test_svrn_bcr_fixed_held_spike.py`。r2复用r1已完成的SVRN、BCR fit/score、INT8 wire和专属sealed held骨架；必须删除hard switch及`g_raw/g_svrn`，新增尺度不变residual、连续`ω`安全集和量化向下取整。核心输入或适应规则再变化必须创建r3并重新审查。

##### `SVRN-qKNN-BCRR/r2`本地闭合

状态=`LOCAL_VERIFIED`。唯一实现delta仍在上述3个候选文件：SVRN保留为DA，qKNN始终先完成全注册类逐query竞争，BCR改为`0≤ω≤0.5`的连续残差；四臂固定为M0/M_DA/M_OTHER/M_JOINT。cross-view安全拟合对destination support和BCR权重使用正式qint8＋fp16 scale解码路径，`ω_q=floor(254ω*)/254`后逐方向逐类复核CE；state封存canonical bank row到support physical ID的映射，truth-free prediction封存邻居顺序并证明M0=M_OTHER、M_DA=M_JOINT及BCRR邻居变化为0。

本地`ssr-gpu`证据：3文件`py_compile`通过；专项`8 passed`；相邻qKNN/R2A/CID回归`40 passed`，仅有pytest临时目录清理PermissionError的P2环境噪声。真实GEOFF/r8 support-only smoke在receiver=`1-1`、scene=`leo_clear_weak`、C5=25条真实support上通过：query/truth fit rows均为0，canonical ID映射25/25，wire state=30,060B，optimizer step=0；该row的`η=ω_raw=ω_svrn=0`只证明identity回退可运行，不是性能结论。

第一次独立review发现receiver别名、量化安全和邻居证据P1；均在原3文件内修复。第二次独立review裁决=`MERGE / P0=0 / P1=0`，允许Git提交及N607发布；K10专项和显式前向次数为P2，不阻塞。下一步仅为提交当前实现、建立全新不可覆盖run ID/report并交唯一Terra runner复用同一r8 archive/coverage执行18 prediction/72 score，未过性能门不得运行125。

##### `SVRN-qKNN-BCRR/r2`首次N607技术失败与`r3`冻结修订

方法提交=`922293b1cc2e15a2f595fc124074bae217ae427e`，发布合同提交=`8894912b1b54897bb1386df44df6ad3d7977574c`。run=`svrn_qknn_bcrr_k5_held_r1_922293b1_20260723`经direct N607/GPU0/PID520750单次启动后自然exit1；wrapper、source和GEOFF/r8四项绑定均通过，但build阶段触发`BCR INT8 teacher gate failed`，prediction=0、score=0，裁决=`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，无retry和125。

完整72-branch只读复算将根因锁定为：仅`14-10/leo_low_elev_weak/C5`的raw/SVRN两支失败，二者`eta=0、omega_q=0`；所有active BCR支均通过原INT8门。`r3`唯一delta是在`omega_q=0`时部署零qint8 BCR codes＋正fp16 scale，并在反序列化强制该不变量；active路径、0.995门、margin门、receipt、融合和资源不变。真实r8无query复测72/72通过，44 active/28 inactive、failure0；原两支agreement=1、flip=0、error=0。独立设计监督和代码review均为`MERGE / P0=0 / P1=0`；方法提交=`165ca03133a8fc724ecccd37e4a55e09a0596dff`。

下一run已预注册为`svrn_qknn_bcrr_k5_held_r2_165ca031_20260723`，继续复用同一GEOFF/r8 archive/coverage，目标仍是18 prediction/72 score的M0/M_DA/M_OTHER/M_JOINT同row性能矩阵。support-only复测的18行`eta`均为0是强负信号但不是性能结果；必须由新run产生完整prediction后按预登记门裁决，失败不得进入125。

##### `SVRN-qKNN-BCRR/r3`第二次N607技术失败

run=`svrn_qknn_bcrr_k5_held_r2_165ca031_20260723`经direct N607/GPU0/PID538739单次启动后自然exit1。wrapper、源码ZIP、方法文件及GEOFF/r8 parity/archive/manifest/coverage SHA全部通过；build和predict完成并生成18个prediction slice，但score阶段报`prediction four-arm order drift`，score=0，因此仍严格裁决为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，未运行125。

回收的packet/truth/query/prediction外部SHA分别为`ef15a8488d40ac70d129db9ac15c796418b4afe5fa64624883eab0f66fd4e95b`、`9745068bc5961ebe90f6305c672cacc8ce338d745579e1c97d4ea503cbd06d8`、`be089f42be790a73cd7a95d68cb13956a64735019b10f6cd4ba32199c33c56c9`、`0f9313e632884e9987caaa262e2e7d261338bfe9b7f84beae85753571b72e06e`。完整只读审计确认canonical JSON将mapping键排序为`M0/M_DA/M_JOINT/M_OTHER`，而冻结`ARMS`为`M0/M_DA/M_OTHER/M_JOINT`；36个before/after mapping均精确包含四臂，prediction COMMIT、row/query绑定、144个logit、argmax及neighbor receipt均通过。唯一根因是评分器错误地把JSON mapping迭代顺序当成语义顺序。

##### `SVRN-qKNN-BCRR/r3-scorefix1`发布冻结

唯一修复把四臂校验改为精确键集合相等，并继续按冻结`ARMS`顺序评分；缺臂或多臂即使重签COMMIT仍拒绝。`ssr-gpu`下py_compile及专项`9 passed`，父prediction直接产生72行score；独立review裁决=`MERGE / P0=0 / P1=0`。方法、prediction、truth、query、qKNN、SVRN、BCRR、状态、参数、资源公式和停止门均未改变。修复提交=`b0baa0dc328ec7fe7a8d5870f35bdee256c9b686`。

全新score-only run=`svrn_qknn_bcrr_k5_scorefix1_b0baa0dc_20260723_070006`已预注册。它在启动前对父packet/truth/query/prediction四个绝对路径逐一校验上述固定SHA，另验truth内部SHA=`637e845fec201627118181a5eb256861b86e76880c101d1b6a5452563cce64b4`和prediction COMMIT=`2524a1aa291cb05ed055625c496f8abc12fc692b57736070334b65ce1c68211a`，随后只读生成72行正式score；禁止重新build、predict、调参、数据重验、retry或125。源码ZIP SHA=`21538751f8e1cdc53d0cb127588f0a239ed9250890eba89ea1b49b93d96ed3ef`，wrapper Git提交=`7ac9805d58860da0b98512f695c14e357c0182cb`，wrapper SHA=`f1fd6a0381b89b9f2c38c84d4db4637db846e72dbf804e8e091bec39f9268892`，`bash -n`通过。正式score、完整log和匹配SHA闭合前，本地只读评分仅为诊断，不作性能裁决。

scorefix1在direct preflight后、任何远端写入前因truth外部SHA录入少一个`3`而被阻断；N607与本地回收文件均确认真实SHA=`9745068bc5961ebe90f6305c672cacc8ce338d745579e1c97d4ea503cb3d06d8`。该run未创建remote root、未同步、未启动，无PID/exit/log/score/marker，状态=`BLOCKED_PRE_LAUNCH / NO_PERFORMANCE_RESULT`。唯一修复为更正此64位SHA；不修改方法、父artifact、prediction或评分。

全新不可覆盖run=`svrn_qknn_bcrr_k5_scorefix2_b0baa0dc_20260723_071226`已预注册，wrapper Git提交=`56e746ea7f7ed406336c4c3f2264e3c132d80ea6`，wrapper SHA=`c72510be802254a969494dc4fb7c99a750f748b94eb49a33a81b8999ad0c097b`，`bash -n`通过，独立review=`MERGE / P0=0 / P1=0`。Git冻结后立即交唯一Terra runner；仍只生成父18个prediction对应的72行正式score，不重建数据或prediction。

##### `SVRN-qKNN-BCRR/r3`正式K5 held性能裁决

scorefix2经direct N607/GPU0/PID559507单次启动后自然exit0；父prediction=18、正式score=72、marker与父四artifact/内部truth/prediction COMMIT/源码/wrapper SHA全部闭合。score SHA=`c3ac8b462009675e316929e82df58d6c53dd47ec4bf51ef426c2f96da8b738fe`。独立分析解码144个logit块和144份neighbor receipt，从prediction＋truth逐样本复算72行全部标量、逐类和transition，与score最大绝对差=`2.22e-16`。

|arm|old-before|old-after|old gain|seen-new|H|BA|floor|min-old|min-new|forgetting|old→new|new→old|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|M0|0.817038|0.785392|-0.031647|0.781784|0.747766|0.781784|0.419908|0.446988|0.781784|0.031647|0.042596|0.218216|
|M_DA|0.817038|0.785392|-0.031647|0.781784|0.747766|0.781784|0.419908|0.446988|0.781784|0.031647|0.042596|0.218216|
|M_OTHER|0.820042|0.797489|-0.022553|0.793192|0.764833|0.793192|0.476318|0.493128|0.793192|0.022553|0.040127|0.206808|
|M_JOINT|0.820042|0.797489|-0.022553|0.793192|0.764833|0.793192|0.476318|0.493128|0.793192|0.022553|0.040127|0.206808|

OTHER相对M0取得old-after`+0.012098`、seen-new`+0.011408`、H`+0.017067`、floor`+0.056410`并把forgetting降低`0.009093`；96次wrong→correct、18次correct→wrong，净`+78`，其中old净`+65`、new净`+13`。但SVRN的18/18行η均为0，DA邻居和prediction变化均为0，故`M_DA=M0`、`M_JOINT=M_OTHER`，mean`I_syn=0`、正slice=`0/18`、正scene=`0/3`。量化、state、MAC、时延、显存和coverage全部通过。

最终状态=`ARTIFACTS_COMPLETE -> ANALYZED / COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不运行125。失败是DA全identity和联合退化为OTHER的机制负结果，不是artifact、scorer或BCRR失败；BCRR的真实独立正收益保留为下一联合候选的可复用资产。

##### 下一联合候选接口结论

`RBSC-qKNN-BPDC/r1`一次监督裁决=`REVISE / P0=3 / P1=3 / NOT_DESIGN_FROZEN`：RBSC尚未闭合为qKNN可消费的确定性metric/bandwidth state，新BPDC的RSS归一化与query积分尺度也不一致，禁止实现。随后两次只读直接复用审计均STOP：D92只封存LDA coefficient/intercept，无法恢复qKNN的rank≤8 typed PSD metric；D101与既有SRDA同样只封存线性分类头而非qKNN metric。因此不能把历史方法头直接拼接后声称`DA＋qKNN＋OTHER`。

当前唯一下一设计artifact为`RBSC-TM-qKNN-BPP/r0`方法卡：保留D92的old/new各0.5类内协方差思想，但显式生成现有qKNN schema可消费的rank≤8 typed metric；qKNN bank/bandwidth与CID-BPP算法、先验、selector和量化门全部复用。它仍处于只读`DESIGN_DRAFT`编写阶段，必须经一次独立FEASIBILITY_REVIEW达到MERGE才允许实现；未冻结前不修改方法代码或发布实验。

##### `RBSC-TM-qKNN-BPP/r0`可行性监督与停止点

独立监督裁决=`REVISE / P0=1 / NOT_DESIGN_FROZEN`，因此本revision不实现、不发布N607。唯一P0是作者卡让`M_OTHER/M_JOINT`直接以BPP predictive logits完成argmax，qKNN只提供bank/config/metric；这重复违反已冻结的“qKNN始终承担全部注册类统一逐query决策，OTHER不得以第二head替代qKNN”主路线。共享bank或typed metric不能把BPP替代头重新定义为`qKNN＋OTHER`，修复必须创建新revision并重新审查。

可复用结论仅限RBSC-TM的数学与接口可行性：非标量rank≤8 SPD metric会改变M-cosine、邻居排序及可能的argmax，identity bandwidth不会抵消；对最终qint8 basis和fp16 attenuation复算谱界可把condition约束在4以内。尚需在新revision闭合的P1包括：完整9/18 slice与2/3 scene协同门及全指标保护门；把非identity qKNN每次调用的support projection `CKDr`计入实际MAC；候选专属teacher-vs-deployment INT8证据；覆盖D=160 eigensolver误差的tie容差；把before隔离改写为builder只接收old support的可验证接口契约。

下一只读候选改为`RBSC-TM-qKNN-BCRR/r1`：RBSC-TM仅替换正式结果中18/18行退化identity的SVRN DA，OTHER完整复用已在同row取得净正确`+78`的连续qKNN残差BCRR，禁止BPP或其他第二head接管最终分类。新卡仍须一次独立FEASIBILITY_REVIEW达到MERGE后才可进入实现。

##### `RBSC-TM-qKNN-BCRR/r1` DESIGN_FROZEN

独立监督裁决=`MERGE / P0=0 / P1=0`。作者提出的principal-sqrt等价性和联合INT8生命周期均已由不可变合同闭合，无需第二设计波次。当前状态=`DESIGN_FROZEN -> IMPLEMENTING / NO_PERFORMANCE_RESULT`；历史BCRR净正确`+78`只证明OTHER值得复用，不构成本候选DA或联合性能证据。

|ID|冻结合同|
|---|---|
|RB01|candidate固定为`RBSC-TM-qKNN-BCRR/r1`；before四臂=`QI,QI,FI,FI`，after=`QI,QM,FI,FM`。|
|RB02|M0/M_OTHER共享identity qKNN receipt；M_DA/M_JOINT共享同一RBSC qKNN metric receipt。|
|RB03|qKNN bank、registry、Phase1 K lock及`class_scales_fp16`四臂共享，不重建bank或bandwidth；BCRR仅以`omega<=0.5`加入连续残差。|
|RB04|RBSC只读当前row support、support标签、typed old/new registry和既有Phase1 bundle；query不得进入fit、audit、fallback或state。|
|RB05|before builder只接收old registry/support，要求`new_classes=[]`并封存独立physical-ID SHA；before RBSC恒identity。|
|RB06|after old/new非空、互斥且并集等于registry；先组内按类等权平均类内scatter，再令old/new各权重0.5。|
|RB07|K1不计算协方差，metric rank0且`omega_I=omega_M=0`；K5首验；K10只按同式确认。|
|RB08|`delta_eig=64*[160*eps64/(1-160*eps64)]*max(1,||S/tau||2)`；跨`lambda=1`或rank8边界的cluster整体丢弃，可保留cluster必须整体进入。|
|RB09|保留cluster按spectral projector、坐标轴升序和两遍MGS生成规范基；cluster内用block mean eigenvalue统一算attenuation；孤立向量按最大绝对坐标最小索引定号。|
|RB10|basis qint8后以实际`B_q`和预量化fp16 attenuation算penalty谱半径`p`；只允许一次公共缩放`gamma=min(1,nextafter_fp16(0.75,0)/p)`。最终fp16重算condition不严格小于4则identity，禁止强度扫描。|
|RB11|metric SHA须在本地/N607一致；tie、半量化边界或SHA不稳定只允许identity，不得更换solver结果。|
|RB12|principal sqrt必须由实际序列化`M_q=I-B_q^Tdiag(a_q)B_q`构造，并验证对称、SPD及`L_q L_q≈M_q`；禁止用未量化正交basis近似。|
|RB13|RBSC BCR只用`H_M=row_norm(X_q L_q)`拟合；部署先计算`L_q W`再按列qint8/fp16封存。由于`N(B)`消去query共有正尺度，该折叠与在`H_M`评分严格等价。|
|RB14|BCRR conditional support-only LOO固定该branch实际metric和formal bank `class_scales_fp16`；每折只删除held physical ID的邻居与BCR训练行，禁止重拟合metric/bandwidth或读取另一branch。|
|RB15|共同bank audit失败为技术失败；metric qKNN audit失败只令RBSC精确回identity并绑定raw state；对应BCR/fusion audit失败只令该branch `omega=0`；query退化仅该query回对应qKNN。|
|RB16|teacher固定为FP64 support、未量化metric/weights和连续`omega*`；deployment必须从序列化bytes反解。所有audit只用support masked/LOO，qKNN与fusion均要求top1≥99.5%、large-margin flip=0。|
|RB17|共享计算保守per-query MAC=`2Nd+Ndr+dr+Nr+2dC`，`Ndr`逐query计费；K5/C6/r8=`51,440`、K10=`99,680`。参数0、optimizer step0、无持久FP32 sidecar、总wire≤256KiB、numpy CPU、VRAM0，并实测build/predict mean/P95。|
|RB18|K5固定复用GEOFF/r8、rx1-1、18 prediction/72同row score。DA净正确≤0、old/new任一净负、OTHER不独立正、JOINT mean H不严格胜两单臂、mean`I_syn≤0`、正slice<9/18、正scene<2/3，或任一old-before/after/gain、seen-new、BA、floor、min-old/min-new、forgetting、双混淆、INT8、condition、state、MAC、时延、协议门失败，均判负且不运行125。|

冻结代码范围仅为`code/cvsrffi/stage2_rbsc_tm_bcrr.py`、`code/cvsrffi/rbsc_tm_bcrr_fixed_held_spike.py`、`tests/test_rbsc_tm_bcrr_fixed_held_spike.py`。不得修改既有qKNN、SVRN-BCRR、数据、GEOFF/r8、coverage或scorer；run报告只承载实验元数据，不扩大方法delta。

##### 用户完整125发布策略覆盖

用户后续明确要求每次正式性能发布均直接覆盖完整125，以避免只在有利K、receiver、scene或seed上取得成绩。因此，本节从现在起以该要求覆盖RB18及更早“先发K5窄实验、通过后再运行125”的发布顺序；RB01–RB17的方法、状态、量化、资源和因果合同不变，不重新打开设计波次。K5/rx1-1/18-slice模块只保留为本地实现、协议负例和无query smoke入口，不再形成独立N607性能run。

当前正式首发矩阵固定为`5 target receivers×5 independent seeds×5 registration slices=125 jobs`，切片为`(K10,new5)`、`(K10,new10)`、`(K10,new20)`、`(K5,new20)`、`(K1,new20)`；每个job覆盖3个LEO weak场景。模型DA目标生效后输出同row的`M0/M_DA_NG/M_DA/M_OTHER/M_JOINT`，预期闭合`375 prediction slices/1875 score rows`。必须一次性报告aggregate、逐类、receiver、scene、K、seed、new-count、transition、coverage、量化和资源；不得以旧的K={1,2,5,10,20} matched-history bundle或任一有利子集冒充本目标完整125。

性能停止门按完整矩阵执行：M_DA净正确决策必须为正且old/new净变化均非负，M_OTHER必须独立为正，JOINT mean H必须严格胜两个单臂，mean`I_syn>0`，正协同至少188/375个scene slice且至少2/3个scene均值为正；old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new不得下降，forgetting及old→new/new→old不得增加。任一失败均形成完整prediction和同row诊断后裁为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；没有完整prediction只能裁为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。125只作冻结候选的全面性能验证，不得用于回调结构、rank、omega、阈值、量化或fallback。

##### 快速模型DA主线纠偏与`DSSC_ZDOM_JG_QKNN_R4_BCRR/design-r1f` DESIGN_FROZEN

实时目标已改为地面压缩知识驱动的快速模型域适应＋qKNN＋唯一OTHER。`RBSC-TM-qKNN-BCRR/r1`因此在提交和N607发布前停止，状态=`SUPERSEDED_BEFORE_COMMIT / NO_PERFORMANCE_RESULT`；其RBSC数学和完整125骨架只保留为reference，不再计作模型DA。目标文档及traceability已由commit=`e19190eb7634d9d8b055e39a7e9219e455aba5bb`建立基线；用户随后进一步冻结为：每个候选、每个revision的任何正式N607性能发布都直接运行完整125，局部行仅可作为本地专项、协议负例和真实checkpoint无query smoke，不得形成独立性能run或方法裁决。

并行方法波次比较了`DSSC_ZDOM_JG_QKNN_R4`与receiver/LEO双因子RDA，同时审查qKNN/OTHER复用。联合监督最终裁决=`MERGE / P0=0 / P1=0`；唯一冻结候选为`DSSC_ZDOM_JG_QKNN_R4_BCRR/design-r1f`，状态=`DESIGN_FROZEN -> IMPLEMENTING`。BCRR是唯一OTHER；双因子RDA、RBSC、BPP和SRDA均不进入本revision。

|ID|冻结合同|
|---|---|
|D01|五臂固定为`M0/M_DA_NG/M_DA/M_OTHER/M_JOINT`，不得缺臂、换序或增加第六臂。|
|D02|`M_DA_NG`与`M_DA`使用相同rank-4 adapter、dual-view、optimizer、step、S_B/S_C与量化路径；唯一差异是ground prior mask。两臂都必须产生非零merged delta。|
|D03|adapter仅为写入实际`id_gate.0→joint_proj.0`的共享四系数rank-4 weight delta；merge后query只走identity backbone的`feat_joint`路径，不调用domain backbone，`ΔMAC_query(adapter)=0`。|
|D04|S_B只读old support；S_C从S_B继续并对全部registered classes逐类等权。query不进入fit、early stop、fallback、BCRR权重或候选选择。|
|D05|每scene使用两两不交的物理ID、独立K-shot、独立adapter和BCRR state；历史三LEO重复support接口永久删除。|
|D06|dual-view只允许同一fixed received IQ的原始view与固定RMS归一化数学view；它不增加K，qKNN bank只存原始view adapted feature。|
|D07|唯一训练目标为对称class-balanced cross-view prototype CE，加`M_DA`的ground-center ridge；删除`M_T/tail/sep/mem`及其他loss。|
|D08|K1只估计4个row-global系数，S_B/S_C固定2/3步；K5/K10固定25/25步。SGD固定`lr=0.02`、`weight_decay=1e-4`、无momentum。|
|D09|ground old multiprototype每类1至3个、每prototype至少2个互不重复Phase1物理样本；`z_id/z_dom`主体INT8＋FP16 scale，无成员ID或FP32 sidecar。|
|D10|class-shared domain basis独立封存、rank≤4、INT8；不能沿用或改名旧ground组件。bundle与lock都声明`ground_old_multiprototype_enabled=true`。|
|D11|ground只初始化/约束共享adapter，不直接投票、增加old logit或生成new/unknown状态；新bundle只改变`bundle_id`，不触发Phase2数据重验。|
|D12|qKNN始终对全部注册类逐query统一竞争；`M0/M_OTHER`共享raw qKNN receipt，`M_DA/M_JOINT`逐字节共享adapter及adapted qKNN receipt。|
|D13|BCRR固定`F=N(Q)+omega[N(B)-N(Q)]`、`0≤omega≤0.5`并branch-local support-only拟合；K1固定`omega=0`，不允许全局协同门。|
|D14|正式state总wire≤256KiB；adapter按rank`[0,1]/[2,3]`两组保存INT8 code＋组内共享FP16 scale且无FP32 sidecar；报告ground、adapter、qKNN、BCRR、INT8一致性、build/predict mean/P95、VRAM、完整forward数及MAC。|
|D15|held artifact必须证明feature、neighbor、argmax和wrong→correct/correct→wrong变化；只有数值微变不构成DA成功。|
|D16|科学代码范围固定为一个方法模块、一个Phase1 bundle builder、一个完整125 runner和一个专项测试；不修改现有模型、qKNN、BCRR、数据builder或scorer。|
|D17|每个冻结revision的每次正式N607性能发布均固定完整125：`125 jobs×3 scenes×5 arms=375 prediction slices/1875 score rows`，8GPU动态队列，不先发窄性能run。|
|D18|125只验证冻结候选；缺prediction为`TECHNICAL_FAILURE`，完成但任一独立DA、ground、OTHER、JOINT、协同、性能、资源或协议门失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。|

实现第一步为本地Phase1新bundle与无query`FEASIBILITY_SPIKE`：复用GEOFF/r8 Phase1 archive生成joint-sealed bundle，验证prototype物理计数、INT8 round-trip、basis rank、checkpoint/bundle/lock绑定；随后用合法support-only smoke证明`M_DA_NG`与`M_DA`均产生非零merged delta、identity-backbone-only`feat_joint`变化和qKNN邻居变化，且fit query rows=0。任一项失败即停止，不发布N607，也不重验Phase2数据；全部通过后只完成专项、协议负例、真实checkpoint smoke、独立P0/P1 review、Git提交和新run报告，随后直接发布完整125。

##### `DSSC_ZDOM_JG_QKNN_R4_BCRR/design-r1f`本地技术闭合

本revision只修改冻结的4个科学文件，没有修改既有模型、qKNN、BCRR、数据builder、GEOFF/r8 coverage或scorer。完整125 runner对每个row的10个prediction artifact、2个prediction receipt、11个score artifact和2个launcher log逐项重算SHA，并闭合row identity、old count、token唯一性、三scene计数、五臂matched token/order、score→prediction绑定、same-row summary/full metrics、forgetting和15行精确计数；专项测试实际生成、删除和篡改artifact验证拒绝路径。base model全部冻结，只有4个共享adapter系数可训练；五臂实际只执行3次qKNN和2次BCRR，不以重复计算伪增资源。

复用的GEOFF/r8输入均为既有`VALIDATED_ONCE`资产，没有重复数据验证：coverage SHA=`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`，Phase1 archive SHA=`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`，manifest SHA=`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`，checkpoint SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`，parity receipt SHA=`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`。最终method lock SHA=`7663bbc4b7b199d98caa85b7736547a6927a2c7eb8e6a4de636967edca1e9c10`，新ground bundle SHA=`109724913cac4f82ff58359b927a7f1e7f7e7d233c0bfd0d05d323f94b1b12da`；bundle含18个old multiprototype、rank4共享domain basis、INT8主体＋FP16 scale，builder记录`query_rows_used=0`。

第一版adapter单共享scale在真实ground S_C的110条support上产生1个低margin top1差异，部署一致率仅`0.99090904`，因此没有发布。改变的唯一部署schema是按rank`[0,1]/[2,3]`分为两个连续组，各保存INT8 code与FP16 scale；真实复测ground S_C一致率升为`0.99999994`、large-margin flip=0、最大logit误差=`0.000752494`。独立首轮代码review随后发现全零组的原scale floor转换FP16后会下溢为0；修复为最小正FP16子正规，并增加零组正scale＋load/round-trip专项负例。该修复不改变输入、loss、adapter方向、训练步数、qKNN或fallback，不构成新revision。

最终真实checkpoint support-only无query smoke根目录为`E:\type10-7\automation_reports\CV-SincNet\dssc_r1f_local_smoke_20260723_140134`，receipt SHA=`0d84219d5c325a0695a73225d880295fdfe99334034971daab4ed57f16008cab`。诊断包只加载before/after enrollment support，`query_packages_loaded=false`、`query_rows_used_for_fit=0`；注册类由6增至11，S_B/S_C分别使用60/110条support。

|branch|S_B delta norm|S_C delta norm|S_B/S_C steps|S_B INT8 top1|S_C INT8 top1|large-margin flip|merged|可训练参数|
|---|---:|---:|---:|---:|---:|---:|---|---:|
|M_DA_NG/no-ground|0.0716443|0.1405842|25/25|1.0|0.99999994|0|true|4|
|M_DA/ground|3.8354450|3.9354210|25/25|1.0|0.99999994|0|true|4|

`ssr-gpu`最终`py_compile`通过，专项与协议负例`21/21 passed`。独立终审确认artifact/hash/row闭合门未被INT8修订绕过，零系数组产生正FP16 scale并可load/round-trip，canonical lock精确封存两组格式；裁决=`MERGE / P0=0 / P1=0`。当前状态=`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`。以上均为技术证据，不是目标域性能；下一步只允许Git提交、建立全新不可覆盖完整125 run报告并交唯一Terra runner发布，不再追加静态设计、控制面或数据验证。

##### `DSSC_ZDOM_JG_QKNN_R4_BCRR/design-r1f`首次完整125技术失败与techfix1

方法提交=`849fa342cd46cb8294b5d9b4f5358cea630d0643`，首发run=`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_849fa342_20260723_141937`。direct N607 preflight、源码/input/checkpoint/runtime SHA、4文件编译、coverage/archive/parity绑定和GPU安全slot均通过；PID=`742449`在GPU0–7启动后自然退出，未retry。125份launcher receipt全部return1，row receipt/prediction/score分别为`0/0/0`，所以状态严格为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

完整回收的125份stderr显示两个共同运行时边界：119份非GPU0任务在sealed TorchScript前向时将内部默认CUDA0张量与row GPU混用；6份GPU0任务越过该点后，在runner自身`torch.as_tensor(ndarray)`处触发N607的PyTorch2.1＋NumPy2桥接失败。techfix1只在row打开sealed runtime前执行并读回`torch.cuda.set_device(row GPU)`，同时把runner全部ndarray→Tensor边界改为contiguous FP32 buffer→clone→device、Tensor→NumPy改为`tolist`→FP32；候选、五臂、矩阵、输入、loss、adapter、qKNN、BCRR、INT8和decision geometry均不变。新增CUDA设备负例与NumPy2桥接攻击后，本地`py_compile`通过，专项`23/23 passed`；真实ADV3B02 checkpoint＋sealed enrollment support无query smoke输出`[2,160]`有限FP32、两行norm均为1、query rows=0。独立终审=`MERGE / P0=0 / P1=0`。新run启动前还须在N607 GPU1执行零IQ/noquery兼容smoke；该技术smoke不产生prediction或性能结果。

##### techfix1 pkgfix1预启动失败与techfix2闭合

新run=`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix1_pkgfix1_f02dd8b0_20260723_150624`使用不可覆盖远端root。direct preflight、源码与5项输入landing、全部冻结SHA、ZIP条目SHA及`py_compile`均通过；但GPU1零IQ/noquery smoke在`current_device=1`且`map_location=cuda:1`时仍发现TorchScript convolution filters位于`cuda:0`。因此完整125命令未执行，无PID、无新增GPU占用、无archive/coverage/prediction/score产物，最终状态=`BLOCKED_PRE_LAUNCH / NO_PERFORMANCE_RESULT`，本run不得复用。

techfix2不把任务收缩到物理GPU0，而是保持GPU0–7共8个并行worker：父launcher为每个row设置唯一`CUDA_VISIBLE_DEVICES=<physical_gpu>`，该子进程只看见一张物理卡并统一以逻辑`cuda:0`加载TorchScript和运行row。LPT动态队列、每物理GPU同时1个本run worker、125项矩阵、五臂、输入、参数、loss、adapter、qKNN、BCRR、INT8和全部科学哈希均不变。

首轮独立review发现P1：launcher收据记录了预期映射，但row不可变收据未回绑实际子进程命名空间。修订后，row在CUDA激活时从实际环境与PyTorch读取`CUDA_VISIBLE_DEVICES`、physical GPU、requested logical device、`device_count`及`current_device`；正式row严格要求单卡命名空间、`cuda:0`、`device_count=1/current_device=0`，并把固定schema evidence写入create-once row receipt。matrix最终验收将launcher的physical/visible/logical映射与row evidence逐字段交叉绑定，缺失或任一漂移均失败关闭。

`ssr-gpu`下runner与专项测试`py_compile`通过，专项及设备命名空间负例`29/29 passed`，`git diff --check`通过；pytest退出后的Windows临时符号链接清理告警不影响退出码0。独立复审最终裁决=`MERGE / P0=0 / P1=0`。当前状态=`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`；下一步仅为Git提交、生成raw Git blob源码包、建立全新不可覆盖完整125 run报告并交唯一Terra runner发布。

##### 公共receiver切向低秩基路线可行性审查

用户提出的精确链路此前没有完整实现：历史方法没有同时执行“类专属球面Log→公共切空间平行移动→按source receiver跨类RobustMean→SVD低秩基→target-old闭式系数→Exp先验”。但其核心假设已有相邻反证。D78/D79的ground切向基改善old与min-old时分别损害new、min-new并增加new→old；D93把ground公共低秩方向经target-old闭式拟合后统一作用于old/new/query，K10相对matched D81的old-after/new/H分别下降`8.611/2.833/5.817pp`、forgetting增加`5pp`，coverage仅`0.144～0.227`；D94只加入coverage连续缩放后，K10 old-after/new/H仍下降`8.056/3.583/5.870pp`。因此，不能把“球面低秩分解”本身当作已证明有效，也不能重复直接transport或old-prior投票。

原式存在三个硬缺陷。第一，`Log_{p_c}(p_{c,r})`分别位于不同`T_{p_c}S^{d-1}`，必须先沿唯一最短测地线平行移动到共同Fréchet切空间，闭合cut-locus、近对跖点、再投影、等类权RobustMean及`B→BR`子空间旋转不变性。第二，共同平移或正交变换不会改变统一qKNN距离与排序，只有DSSC adapter产生的非等距表示变化并实际改变neighbor、margin或argmax才算DA。第三，只把Exp结果用作old原型、old logit或qKNN bank会重演D78/D79的old/new交换；ground先验不得投票。

唯一保留的新假设暂记为`DSSC_TANGENT_PRIOR_QKNN_BCRR/design-draft-r0`：只用“公共Fréchet切空间rank≤4先验＋中心化且尺度归一的Gram关系loss＋连续收缩/fallback”替换当前DSSC ground-center ridge。DSSC rank-4真实模型adapter、dual-view、optimizer、S_B/S_C、全部注册类逐类等权task loss、INT8 merge、qKNN和BCRR均保持不变；`M_DA_NG/M_DA`唯一差异仍是ground块，`M_JOINT`逐字节复用`M_DA` adapter/qKNN state后才加入BCRR。

target-old闭式系数只允许读取当前row合法support。K1下6个旧类可代数估计至多4个row-global系数，但单类不确定度只能来自Phase1 LODO/LOCO证书与跨类残差；K5/K10才可加入support散度。连续收缩固定为`alpha=coverage×confidence×LODO_reliability∈[0,1]`，且confidence随不确定度单调下降；秩亏、病态、Fréchet不唯一、近对跖点或不确定度不可得时精确ground-off。Phase1持久状态只能是共同封存的INT8 basis、公共点/类中心、半径与FP16 scale；不得保存FP32 Log/Exp、系数、先验或receiver成员sidecar。query额外adapter MAC必须保持0，总wire仍须≤256KiB。

独立监督裁决=`REVISE / P0=2 / P1=3`，不授权实现或N607。下一步仅允许一个本地无query`FEASIBILITY_SPIKE`：Phase1-only验证类/receiver置换、全局正交和`B→BR`等价及Fréchet fallback；在一个合法support-only capsule上固定K1/K5/K10记录rank、条件数、alpha来源和fallback；ground-on必须相对ground-off产生非零DSSC delta及至少一个support-LOO neighbor或margin变化；最后以实际INT8/FP16序列化重算等价性、state、训练MAC/时延和query增量MAC。任一K伪造不确定度、三个K全部identity、只有公共正交漂移或真basis不优于置乱basis即停止该draft。当前完整125 run与其source package不因本节改变。

##### `DSSC_ZDOM_JG_QKNN_R4_BCRR/design-r1f`techfix2终局与registry techfix3闭合

techfix2完整125 run=`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix2_b77cc6c4_20260723_155058`经direct N607单次启动，wrapper PID=`796973`、matrix PID=`796975`，最终自然exit=`1`；未kill、restart或retry。GPU0–7均实际参与，子进程各自通过`CUDA_VISIBLE_DEVICES=<physical_gpu>`只看见一张物理卡并使用逻辑`cuda:0`，最终GPU0/1/2各执行15个job、GPU3–7各执行16个job，因此不是只使用物理GPU0。

终局artifact为125/125份launcher receipt、0份row receipt、0 prediction、0 score；125个returncode全为1，stderr SHA256全部为`30b024f3b0191d09fc68f88c1ec0e4106ec446900a808dd387f13de621555b1c`。统一根因是sealed opaque registry采用old-prefix/new-append顺序，而共享SVRN state builder要求全局字典序，在任何query预测前抛出`registered class registry drift`。因此裁决固定为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不可填写性能或协同结论。387个回收文件已逐项验SHA，`missing=0/mismatch=0/extra=0`；archive、manifest、parity和coverage仍只复用冻结SHA，没有重复数据验证。

techfix3只在DSSC→legacy SVRN接口内部把类轴投影到字典序，qKNN、BCRR及五臂出口再精确逆置换回原sealed registry。类集合、support、距离、BCRR公式、参数、五臂、runner和method lock均不变；`BCRRState`保留旧三参数位置兼容，缺少显式sealed classes的旧对象只能从canonical bank恢复，否则fail-closed。`ssr-gpu`下实现、测试和正式runner的`py_compile`通过，专项`30/30 passed`，`git diff --check`通过；pytest退出时只有已知Windows临时目录清理权限提示，主体exit0。独立Terra终审=`MERGE / P0=0 / P1=0`。

真实checkpoint opaque-registry无query smoke使用SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`及`20-1/leo_clear_weak/K10`的11类110条enrollment support：strict重建`missing=0/unexpected=0`，`z_id`为有限FP32`[110,160]`；sealed registry非字典序，内部bank为字典序，三个qKNN state对外均保持sealed顺序，五臂state构建完成；`query_packages_loaded=false`、`query_rows_used=0`。下一步只允许提交techfix3、建立新不可覆盖完整125 run报告并交唯一Terra runner发布。

##### 下一实验路线`ADV3B02-TS-DRQKNN-BCRR/r1` DESIGN_FROZEN

用户指定下一实验必须保留`z_id/z_dom`双qKNN核心，并只选择性吸收`ADV3B02_双分支双注册qKNN快速适应设计报告_20260723.md`。独立Sol-max监督最终裁决=`MERGE / P0=0 / P1=0`；候选状态=`DESIGN_FROZEN`，当前DSSC完整125不因下一设计改变。

冻结机制为：Phase1同SHA ADV3B02 checkpoint通过head-bypass路径输出`z_id/z_dom`；Stage2-B用target-old support在`z_dom`中拟合TX抑制的rank≤2类内域邻域，只重加权每个候选类内部的`z_id`Student-t qKNN证据，最终跨类竞争仍由`z_id`qKNN完成；Stage2-C冻结旧`Q/A/alpha/mu_c/bank`并只append新类。support与query都按候选类减同一`mu_c`，固定2槽`rho`使弱方向连续衰减；K1或数值异常时逐值identity。

BCRR是唯一OTHER，raw/dual branch分别以自身同步physical-ID support-LOO logits拟合`omega`，但共享同一`z_id`BCR状态和预锁规则；不得读取query或直接读取`z_dom`。四臂固定为`M0/M_DA/M_OTHER/M_JOINT`，完整125闭合`125 jobs/375 scene slices/1500 score rows/1000 arm-state prediction artifacts`。DSSC保留为matched reference，不与本revision的DA臂混塞。

选择性吸收结果：保留双分支、双注册、target-old冻结/new append、INT8 support bank、类内归一化和BCRR；拒绝直接双余弦、hard membership gate、ground投票、domain→ID transport、第二分类头和Stage2 optimizer；选择性rescue与Phase1双episodic重训延期。冻结代码范围仅新增method module、完整125 runner和专项test；不修改模型、既有qKNN/BCRR、数据、coverage、authority或scorer。完整公式、runtime合同、资源门和falsifier记录于`docs/ADV3B02_TS_DRQKNN_BCRR_DESIGN_FROZEN.md`。
