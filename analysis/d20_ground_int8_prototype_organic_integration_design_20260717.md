# D20地面int8原型与单观测高性能路线有机结合设计

日期：2026-07-17  
状态：机制分析与路线锁定前设计；尚未启动D20实验  
权威边界：根目录`项目.md`正式Stage2-B/Stage2-C、`LEO_weak-only`、单物理样本单LEO观测、逐样本全注册类决策、无query标签拟合、无角色Oracle、无类别配额

## 1. 结论先行

地面int8域×类原型不应再与target support原型直接平均，也不应作为额外qKNN样本或旧类统一加分项。现有D19已经证明：强融合能保护旧类却系统性压制真实seen-new；弱融合又几乎退化为无地面原型的Z0，继续细扫CIAF的`alpha/beta`没有研究价值。

推荐路线是两层解耦：

1. **合法单一received-IQ表征层**负责真正的域适应与新类注册。输入只能是已经叠加一次且仅一次LEO_weak信道的固定IQ；可从这份IQ计算`z_id160+FFT96+RF32`，用不超过20epoch的极小对角度量/分类头改善全注册类可分性。
2. **地面int8旧类身份层**只负责旧类内部纠错。它与同一次ADV3B02前向产生的旧类direct logits形成弱辅助排序，随后执行`max_old`保持重标定。该结构在数学上不改变旧/新组边界、不改变任何新类分数，也不改变seen-new样本的预测结果，只可能改变旧类内部的类别选择，因此适合专门修复旧类floor。

这不是把多个旧方法堆叠，而是让每种信息只做其擅长且可验证的工作：received-IQ表征解决域偏移，target support负责在轨身份和新类注册，int8原型恢复Phase1旧类身份方向，direct logits提供冻结判别面信息，`max_old`保持算子隔离旧类身份修复与旧/新检测。

## 2. 协议与证据边界

- Phase2的support和query均必须已经是唯一的`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`观测；不得访问clean/raw IQ，也不得从同一clean样本产生多个LEO场景观测参与适配。
- FFT、RF统计、均衡或其他计算view只能来自当前已密封的同一份received IQ，不增加K，不生成第二个LEO状态。
- 正式Phase2只能读取与ADV3B02 checkpoint共同封存的Phase1 int8域×类聚合原型组件；历史组件当前只能用于一次性`PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`，不能打开query或产生正式性能声明。
- direct logits若使用，只能是当前LEO_weak IQ在同一次封存ADV3B02前向中产生的输出；它不是source样本、source logit cache或独立source artifact。
- predictor逐样本面对全部注册类，输入schema不得含query真标签、old/new/unknown角色、真实批次类别计数、类别quota、query块顺序或全局分配信号。
- 本文引用的D1高性能结果已因同一物理样本跨三个LEO场景平行复用而失去当前协议效力，只能作为机制线索，不能作为当前性能、超参数或Pareto证据。

## 3. int8组件到底包含什么

当前历史组件的逻辑张量为`int8[26,6,160]`，只有14个训练域对6个旧类有效，共84个有效域×类cell。6个旧类依次为`14-10、14-7、20-15、20-19、6-15、8-20`。

|口径|字节数|解释|
|---|---:|---|
|84个有效int8质心|13,440B|`84×160×1B`|
|84个有效FP16 scale|168B|`84×2B`|
|有效payload小计|13,608B，约13.29KiB|此前所称“13.6KB”|
|当前稠密26域int8张量|24,960B|包含12个无效域槽位|
|当前稠密scale+mask|468B|312B+156B|
|当前逻辑组件状态|25,428B|D19资源审计口径|
|压缩NPZ文件|5,363B|文件压缩大小，不等于运行时逻辑状态|

因此当前组件不是1.3KB。若重建正式bundle时只登记14个实际有效域，仍保持`int8[D,C,P]`正式schema，payload可无损压到约13.6KB。由于本次几何分析发现跨域原型近乎重复，若未来把每类压成一个int8共识方向，则纯质心和scale约为`6×160+6×2=972B`，加registry/schema后才接近1.0–1.3KB；但这不再是当前`domain_class_centroids_v1`的域×类payload，正式采用前必须先得到用户授权并修订`项目.md`。当前路线不偷换该口径。

## 4. 原型几何揭示的真实作用

对历史int8组件仅在内存中临时解量化并归一化，不持久化任何解量化原型。结果如下。

|旧类|跨14域两两余弦均值|最小值|到类共识方向均值|最差纯度margin|
|---|---:|---:|---:|---:|
|14-10|0.9987|0.9951|0.9994|0.9713|
|14-7|0.9985|0.9948|0.9993|0.9674|
|20-15|0.9987|0.9963|0.9994|0.9875|
|20-19|0.9972|0.9901|0.9987|0.9423|
|6-15|0.9989|0.9949|0.9995|0.9731|
|8-20|0.9988|0.9966|0.9994|0.9797|

六个类共识方向之间的最大非对角余弦只有约0.0169，说明旧类身份方向高度纯净；同一类跨域余弦却高于0.997，说明“域×类”中的域差异极弱。由此得到三点：

1. int8组件是很好的旧类身份字典，但不是强域估计器。
2. support驱动top-M域选择主要是在极小数值差异中选域，容易放大support噪声，不能承担D20核心门控。
3. 计算上可预登记一个固定medoid域索引，或在新正式bundle中采用经批准的压缩方案；无需每个query遍历84个原型。当前26域组件的全局max-min中心域为候选固定medoid，但具体索引必须在方法锁中由离线规则固定，不能按query选择。

因此，当前未提交的DALI草案中“按support为每类选择top-M域anchor”的设计不应直接启动，需要先改为固定medoid/弱身份证据并加入`max_old`保持算子。

## 5. 之前高性能与负结果分别告诉了什么

### 5.1 D19/CIAF：原型直接融合轴已经探索充分

|候选|after-old|seen-new|H|遗忘|结论|
|---|---:|---:|---:|---:|---|
|Z0 support-only|48.33%|52.67%|48.97%|22.78pp|无地面anchor基线|
|强anchor示例A025_095_B012|63.33%|12.00%|17.59%|5.56pp|保护旧类但新类崩塌|
|更强旧类偏置A010_080_B018|64.44%|6.67%|很低|降低|不可用|
|弱anchor A095/A090/A080|约48.33–48.89%|约52.67–53.33%|约48.85–49.47%|局部变化|几乎复制Z0，floor仍为0|

强anchor的失败是结构性的：只有旧类有地面anchor，新类没有对称证据；把anchor直接混入原型或直接抬高旧类score，会改变`max_old-max_new`，随着新类数增加还会放大组间极值竞争。继续微调融合权重只会在“旧类保真”和“seen-new压制”之间移动。

### 5.2 D1：表征修复有价值，但历史数值不能继承

D1采用`z_id160+FFT96+RF32`及对角余弦分类头，20epoch、约3,467参数。在协议无效的历史开发行上曾得到Stage2-C old 96.94%、old floor 95.00%、seen-new5 90.67%、H 93.70%、遗忘1.67pp。这说明固定received-IQ的多表征和轻对角判别头具有潜力，但这些数值来自当前已禁止的跨场景同物理样本复用，不能用于当前候选锁参或成功声明。

D1在K10/new20历史125结果中的旧类before→after为：14-10 89%→84%、14-7 86%→78%、20-15 98%→97%、20-19 86%→72%、6-15 98%→76%、8-20 99%→99%。遗忘集中在20-19和6-15等floor类，并非所有旧类一起下降。因此需要类对称的最差类优化和旧类内部纠错，不能施加统一旧类bias。

### 5.3 direct logits：可作弱判别证据，不能再作强旧类偏置

历史K30中0.25权重的同IQ ADV3B02 logits曾把new5/new20的old floor分别提高到80.00%/77.50%，但new10 floor反而下降3.33pp，跨规模不稳定。它和int8几何方向具有信息互补性，但必须中心化、小幅、support-only锁定，并通过`max_old`保持算子阻断其对旧/新组边界的影响。

### 5.4 不再重复的路线

- full-precision source anchor/source prototype：当前协议禁止，且与int8身份先验重复。
- int8+ProtoNet或继续target centroid blend：D19已经等价覆盖。
- 把int8原型复制成qKNN exemplar：既不是真实support，又重复放大旧类证据。
- CIAF强anchor+D13/D15–D17类new-score惩罚：双重旧类偏置，会继续压制真实新类。
- 继续细扫D19b弱`alpha/beta`：已贴近Z0且floor仍为0。
- 首版叠加target多原型：历史证据只显示遗忘局部缓解，未改善floor，并增加状态和多新类极值竞争。
- 继承D1旧高分超参数或三场景平行view：协议无效。

## 6. 推荐架构：received-IQ适配+最大值保持旧类身份重排

### 6.1 全注册类基础分数

对唯一固定LEO_weak IQ提取

\[
\phi(x)=[z_{id}^{160}(x),\operatorname{FFT}^{96}(x),\operatorname{RF}^{32}(x)].
\]

基础分数为

\[
b_c(x)=\tau\cos(\exp(\gamma)\odot\phi(x),w_c)+\beta_c,
\]

其中`gamma`是共享对角scale，`w_c、beta_c`是注册类轻量头。所有support均来自同一目标接收机的合法LEO_weak单观测；训练最多20epoch，无query参与。K1采用强收缩或关闭可训练对角scale，K5/K10再逐步释放，避免单样本过拟合。

Stage2-B用旧类target support形成基础头；Stage2-C以相同规则注册新类并用全部已注册support进行类平衡轻量拟合。旧类target support仍是合法在轨数据，可以用于旧状态保持；不需要任何source样本。

### 6.2 地面int8与direct logits只生成旧类内部辅助排序

对六个旧类，使用预登记固定medoid域的int8方向`p_c`，计算

\[
g_c(x)=\cos(z_{id}(x),p_c).
\]

同一次ADV3B02前向得到旧类direct logits`l_c(x)`。二者均按support锁定的尺度进行中心化、截断和K-shot收缩：

\[
a_c(x)=h(K)\left[\eta_g\tanh\frac{g_c-\bar g}{s_g}+\eta_l\tanh\frac{l_c-\bar l}{s_l}\right],
\quad h(K)=\frac{K}{K+k_0}.
\]

`s_g、s_l、eta_g、eta_l、k_0`必须在开发support-only阶段统一锁定，不能按query标签或场景重选。首轮候选必须包含`eta_l=0`和`eta_g=0`消融。

### 6.3 `max_old`保持算子

先对辅助证据做旧类内零均值与有界截断：

\[
d_c=\operatorname{clip}(a_c-\operatorname{mean}_{j\in O}a_j,-\delta,\delta),
\quad u_c=b_c+d_c.
\]

再对全部旧类加同一个常数：

\[
\tilde b_c=u_c+\max_{j\in O}b_j-\max_{j\in O}u_j,\quad c\in O.
\]

新类分数保持`\tilde b_n=b_n`。于是严格有

\[
\max_{c\in O}\tilde b_c=\max_{c\in O}b_c,
\quad \max_{n\in N}\tilde b_n=\max_{n\in N}b_n.
\]

因此在确定性tie规则下：

- old-vs-new组判定逐样本不变；
- 所有新类分数逐位不变；
- 被预测为新类的样本，其具体新类预测不变；
- seen-new准确率不会因int8/direct-logit身份重排而下降；
- 只有样本已落入旧类组时，旧类内部argmax可能变化，从而修复或损害旧类身份。

这把“新旧分辨”和“旧类内部floor修复”从结构上解耦。其局限也明确：若旧样本基础模型已误判到新类组，重排无法救回；这部分必须由received-IQ表征层解决。

### 6.4 floor优化与防遗忘

表征层拟合采用类平衡目标，不按类别样本数形成quota：

\[
\mathcal L=\frac{1}{|C|}\sum_c\mathcal L_c+
\lambda_f\operatorname{LSE}_{\tau_f}(\{\mathcal L_c\})+
\lambda_p\mathcal L_{old-state}.
\]

第一项保证每类同权，第二项近似最差类损失并直接关注floor，第三项仅使用Stage2-B旧support和旧头状态约束注册后的遗忘。它不读取query角色或类别quota。方法锁必须同时审核：总体old、逐旧类floor、seen-new、H、before/after遗忘；任何只提高均值却损伤20-19/6-15等floor类的候选不能晋升。

## 7. 分阶段验证，避免盲跑

### 阶段A：分析门，已完成

- 确认84个有效int8原型的尺寸、schema与真实字节口径。
- 审核跨域/跨类几何，否定把其作为强domain gate。
- 复核D19/D19b、D1、direct logits、source anchor、qKNN、ProtoNet、多原型、D13的同row证据。
- 明确不再扫CIAF强弱融合轴，不直接启动现有top-M DALI草案。

### 阶段B：support-only方法锁

只打开合法开发LEO_weak注册support，使用固定leave-sample-out划分；历史int8组件不打开query。候选保持很小：

|候选|received-IQ表征|int8|direct logits|`max_old`保持|用途|
|---|---|---|---|---|---|
|B0|z_id target prototype|无|无|不需要|严格Z0|
|B1|同B0|固定medoid弱证据|无|是|验证纯int8旧类内部纠错|
|B2|同B0|固定medoid弱证据|弱中心化|是|验证两种旧类身份信息互补|
|B3|合法单IQ的z_id+FFT+RF轻头|无|无|不需要|隔离表征修复|
|B4|同B3|固定medoid弱证据|弱中心化|是|完整分层组合|

不再展开7×多权重笛卡尔网格。只允许一个统一K-shot工作点和一组预登记强度，K1/K5通过固定`h(K)`收缩。support-only晋升要求：每个场景×fold的逐旧类非劣，floor改善，注册后H与遗忘非劣；失败即回退B0/B3。该阶段只有方法筛选证据，不产生正式accuracy声明。

### 阶段C：正式bundle与开发query

只有阶段B为正，才重新生成共同封存的ADV3B02+int8 bundle和新method lock。随后在开发seed确定统一超参数，严格采用预测artifact与truth scorer隔离，先报告K1/K5/K10，并核验K5相对K10每种指标下降不超过3pp。K1必须单列适配增益、每receiver非负约束和相对direct ADV3B02的提升。

### 阶段D：独立确认矩阵

开发锁参后才生成5个目标receiver×至少5个独立确认seed×3个互斥LEO场景，覆盖2/5/10/20个seen-new TX的正式矩阵。正式结果必须同时提供注册前Stage2-B和注册后Stage2-C，逐类、逐receiver、完整训练日志、预测—评分隔离证据、资源审计、合法TX/receiver/support-query清单和Git提交。K1也必须纳入用户要求的重跑矩阵；矩阵任务数以最终生成器和`项目.md`定义为准，不能拿旧125 artifact替代。

## 8. 资源预期与Pareto目标

以20个新类、共26个注册类估算：

- 共享对角scale+26类288维权重+bias约7,802个参数，显著低于50k上限；训练最多20epoch。
- FP16轻头约15.6KB；target support不必像qKNN一样逐样本常驻为query邻居状态。
- 正式v1有效int8 payload约13.6KB；当前历史稠密实现为25.4KB，重建时可删除无效域槽位。
- 基础26类×288维约7,488个点积MAC/query，固定medoid int8旧类证据约960个点积MAC/query；同IQ direct logits复用同一次backbone前向，不增加第二次LEO观测或第二次backbone前向。
- identity-only single qKNN在K10、26类时仅160维邻居点积约41,600MAC/query，K20约83,200MAC/query，FP16样本状态分别约83.2KB和166.4KB。D20头部的状态和MAC不随K线性增长，理论上具备明显Pareto优势；FFT/RF的实际MAC、平均/P95时延和峰值显存仍必须实测，不能用估算代替。

正式审计需把checkpoint、int8 payload、scale/registry、target头/原型、一次性enrollment开销、每query额外MAC、平均/P95时延和峰值显存分项报告。若B4表征层没有明显收益，优先保留0参数/0epoch的B1/B2安全重排，而不是为追分增加复杂adapter。

## 9. 实施前必须修复的具体问题

1. 放弃当前DALI草案的support top-M域门控；改为预登记固定medoid索引和`max_old`保持旧类内部重排。
2. class binding除feature列外必须显式登记ADV3B02 direct-logit列索引，运行前逐列hash/映射验证。
3. 特征和direct logits必须在同一IQ、同一次backbone前向中提取并记录forward count；不得分别走可能漂移的loader。
4. runner先输出不可变prediction artifact，再由隔离scorer连接truth；support-only筛选阶段不得打开query。
5. 资源审计同时报告当前历史稠密25,428B和正式重建有效payload约13,608B，不能把5,363B压缩文件大小写成运行时状态，也不能写成1.3KB。
6. 任何候选都必须保留B0/B3回退、逐类原子门、K1数值稳定检查、NaN/单样本/重复注册检查以及旧/新角色和类别quota不可达检查。

## 10. 当前决策

当前不启动N607实验。先按本文修改D20模块和support-only runner，完成单元测试与资源/协议静态审计；再只运行B0–B4五个有明确机制分工的support-only候选。只有B1/B2确实修复旧类内部错误且B3/B4显示合法单观测表征收益，才进入正式bundle重建与开发query评估。若support-only证据为负，直接淘汰相应分支，不用125矩阵为失败机制支付计算成本。
