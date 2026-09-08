# ADV3B02全批次综合报告：A1系列、RX-V2与PAIR三seed实验

分析日期：2026-09-08。状态：ANALYZED；36行ARTIFACTS_COMPLETE，2行FAILED。结果定位：已知六类TX的Phase1闭集目标域诊断，NO_PROMOTION_TO_DEFAULT。

## 技术摘要

A1是当前冻结结果中值得保留的clean优先候选：clean=85.346%，比同批A4高3.314个百分点；7个目标接收机中6个改善，clean最低接收机也提高。但其LEO均值只有71.165%，source训练经历接近六类机会水平的崩塌后恢复，域分类损失最终占总损失约83.43%。高clean分数与优化异常同时成立，不能相互抵消。

后续RX-V2和PAIR改革未形成明显的大幅提升。V2的P3为clean=83.167%、LEO=72.693%；PAIR中TANGENT三seed的LEO均值72.694%，比同seedMATCHED_ZERO平均高0.301个百分点，POINT_MEMORY三seed的clean均值82.777%且LEO差值均为正。三seed和小幅平均增益仍不足以证明稳定的机制收益，尤其SAFE采用修复后的EMA版本，其余21行保留旧版本，不能作严格单因素SAFE对POINT比较。

本报告重新解析38行、7318个epoch、1191个联合字段、85个标准损失字段，扫描117份日志共330578行，交付完整数值、失败记录、模型与损失说明、36行最终测试以及逐接收机/日期/类别等已存在细分数据。缺失指标保持N/A，不补造。没有重训、补测或依据target结果改动训练。

## 范围、数据口径与可比性

纳入V1的A1—A7七行、V2的P1—P5及E1/R1七行、PAIR的八种配置×三seed共24个有效行。36行完成200epoch；P5只记录E1—E9，在E10失败；B_SAFE_S392005记录至E109后停止。故7318=36×200+9+109。历史技术恢复中被替代的15个PAIR失败尝试不计入有效24行；其运行历史另附原报告，不能把重试当新seed。

数据为ManySig地面OTA代理IQ，equalized=true。source RX=[1,3,4,6,8]、day=[1,2,3]；source池90000，L_s=6300、U_s=56700、只读V=27000。训练池标注比例为6300/(6300+56700)=10%；全source池标注占7%。目标RX=[0,2,5,7,9,10,11]、day=[0,1,2,3]、TX六类；目标RX未参与训练，但部分日期共享，不能将全部target称为未见日期。

共同从ADV3B02 CORE90的checkpoint初始化，再训练200epoch=130+70；不是随机初始化从零训练。最终统一使用E200，不按target挑epoch。clean保留真实地面接收效应，只是不再叠加LEO模拟扰动；LEO三种weak场景是模拟压力条件，不是实测卫星链路。

**两个LEO评测口径必须分开。**早期13个完成行每个场景168000条，三种LEO共504000次观测，模拟seed=392005。PAIR23行固定同一168000个physical样本，每physical只分配一种LEO场景：clear=56164、low-elev=56095、rain=55741，总计168000，模拟seed=2027；clean另有168000。PAIR输入被持久化并跨23模型复用，同批差值有配对基础。跨批LEO分数只作背景描述，不能拿0.00几百分点的差异论证升级。图中family保留这一区别。

source验证曲线约98%不等于目标接收机约82%的测试准确率。所有target分数只在固定预测后由独立评分器连接truth；本次报告不会把目标标签用于训练、超参选择或重跑。

### E200目标clean与LEO均值：按批次区分

图表与可排序数据见report.html。

## 模型、训练主链与版本

学生模型保留CV-SincNet身份支路z_id、域支路z_dom、身份分类头及域分类/对抗头；clean和单个LEO学生前向承担监督梯度。教师为EMA参数副本，额外教师前向使用no_grad。CosFace头使用单位特征与单位类别权重的余弦logit；SAFE只对实际bias-free CosFace几何成立，不能把任意非线性分类头末层矩阵替代进去。

早期完成行总参数1130809；PAIR完成行1116471。教师数、辅助头、采样和优化路径不同，参数量接近不表示计算量相同。早期名义L/U batch=128/256；V2结构化L sampler实际batch=90，PAIR实际L观测按每epoch重复取样可达28416，并非28416个不同的标注样本。每epoch U观测56700。配置和日志保留了实际观测次数。

基础学习率早期为0.0002；采用既有MUSE/fasttrust日程，具体每epoch学习率见全字段数据。PAIR完整命令参数在configurations.json。标注/无标签日程、source几何辅助项、EMA和梯度控制均需区分名义开关与实际非零贡献。

PAIR有效release分三组：r1负责POINT/ASYMMETRIC/MEMORY/ZERO共12行；r2修复探针sample-rate参数缺失等技术问题后完成TANGENT/ROUTE/TANGENT_ROUTE共9行；r3修复EMA权重更新未使CosFace缓存失效的问题后运行SAFE三行。旧21行健康训练按授权保留，没有为实现版本一致性擅自重跑。r3修复会改变教师logit，SAFE与旧21行比较存在教师版本混杂。

本次取得PAIR三个实际release源码、V2两个现存release源码。A1原始release路径未定位；其公式核对采用历史Git参考72360e49ceb6a4f2a9e2e1235e5eea6784d07685和运行日志。该参考包明确标记并非已验证的A1原始release，不能声称源码身份完全闭合。

## 损失函数：基础装配与字段含义

记CE为交叉熵，Norm为L2单位化，sg为停止梯度，所有epoch损失是batch统计聚合。标注闭集主目标为：



```text
L_closed,L = L_TX + L_MUSE-local + w_dom L_dom +
w_adv L_adv + w_orth L_orth + w_cons L_cons +
w_group L_groupCE + w_Fishr L_Fishr + w_sat
L_satCE + w_satcons L_satcons
```



TX和satCE分别监督clean/LEO的真实source身份；L_dom训练域支路分类，L_adv通过身份支路的梯度反转实施域对抗，两者不能混为一谈；orth约束支路相关性，cons是既有一致性项，groupCE/Fishr分别处理source分组风险和组间梯度统计差异。关闭的CRRA及独立teacher-KL项也保留日志字段，但本批不构成额外有效目标。

标注几何辅助组为

```text
L_open,L = w_proto L_proto + L_DAOT,L + w_OW
L_OW + w_compact L_compact + w_proxy L_proxy +
w_mix L_mix + w_episode L_episode
```

，各项再乘各自warmup/stage系数。基础prototype项与新增orbit prototype蒸馏是不同目标：前者可非零，不能因后者全0而说所有原型学习失效。OW边界、紧致性、代理unknown、mixup及source episode均使用source构造，不表示已经测过真实unknown拒识。

MUSE分支存在时，

```text
L_closed = L_closed,L + L_MUSE,U + L_DAOT,U
```

，没有在外层再乘一次lambda_u。MUSE内部包括base域/自监督/nuisance项、日程加权的hard CE/soft CE/candidate-set CE、被接受样本的卫星身份CE以及跨RX辅助。identity子项形式为

```text
lambda_u(t) × (lambda_high L_hard + lambda_mid
L_soft + lambda_low L_candidate)
```

；U的TX真值不进入该函数。无MUSE的备用路径才在外层以lambda_u与lambda_ent装配identity和entropy。本批解释必须跟实际路径，不能把两个分支同时相加。

进入反向前的总标量是

```text
L = tail_closed_scale × L_closed +
dg_health_open_scale × (L_open,L + L_open,U)
```

后续预算/梯度冲突控制可以改变辅助梯度，所以单个raw loss大小既不等于加权贡献，也不等于最终参数位移。有效参数、stage和缩放的逐epoch值都在全字段CSV；本报告不强行从已聚合标量还原不存在的逐batch梯度。

`train_loss_*`多为原始子损失；`train_w_loss_*`是已乘有效权重的子项；`train_loss_daot_total`和`train_loss_daot_unlabeled`已经是DAOT加权合计，`train_loss_labeled/unlabeled`与`closed/open`是不同分组视角，不能全部再加一次。标准85字段表只是便捷入口，MUSE及其他名称含loss的全部原始字段仍在1191字段文件中。

### E200记录的基础损失名义权重（另受stage/控制系数影响）

|目标|A1|A4|P3|POINT005|SAFE006|
|---|---|---|---|---|---|
|dom|1.0000|1.0000|1.0000|1.0000|1.0000|
|adv|0.3500|0.3500|0.3500|0.3500|0.3500|
|orth|0.0500|0.0500|0.0500|0.0500|0.0500|
|cons|0.0800|0.0800|0.0800|0.0800|0.0800|
|group_ce|0.1600|0.1600|0.1600|0.1600|0.1600|
|fishr|0.0400|0.0400|0.0400|0.0400|0.0400|
|sat_cls|0.6800|0.6800|0.6800|0.6800|0.6800|
|sat_cons|0.0000|0.0000|0.0000|0.0000|0.0000|
|proto|0.0032|0.0032|0.0032|0.0032|0.0032|
|open_world_feat|0.0024|0.0024|0.0024|0.0024|0.0024|
|zid_compact|0.0320|0.0320|0.0320|0.0320|0.0320|
|proxy_unknown|0.0045|0.0045|0.0045|0.0045|0.0045|
|soft_unknown_mixup|0.0045|0.0045|0.0045|0.0045|0.0045|
|source_episode|0.0035|0.0035|0.0035|0.0035|0.0035|

## V1与RX-V2的DAOT目标和实验矩阵

V1教师中心为

```text
z_orbit=Norm(Σ a_k b_k Norm(z_teacher,k))
```

A1使用clean+medium两fresh均值；A2改三fresh（增加hard）；A3采用部署代理权重、物理可靠度、coverage floor和Huber残差的鲁棒球面聚合；A4增加单参数tangent；A5改协方差随机方向；A6改分支选择性并启用nuisance；A7再增加fingerprint保真。额外学生监督主链持续存在，不能把这些行叫纯蒸馏模型。

联合形式为

```text
L_DAOT = Σ_j stage_j(t) × lambda_j × raw_j /
EMA_scale_j
```

feature对齐单位化教师中心，logit项为温度化软目标蒸馏；temperature=3。V1 feature/logit/prototype权重0.50/0.20/0.20，relation关闭；prototype蒸馏虽配置0.20，14行全部epoch均为0。V1 E1—20预热，E21—60引入orbit，E61—140引入tangent，E141—200保持后期日程。

V1局部灵敏度`S=2(1-cos(z_delta,z_base))/delta²`，惩罚超出方向budget的部分；delta=0.05、loss mask比例25%。旧实现仍对扰动batch做前向，不能由25%mask推出75%前向节省。A4—A7 tangent权重0.05；A6/A7异方差nuisance权重0.10，预测9维状态均值与不确定度；A7 fingerprint权重0.10，身份灵敏度下限0.50。A1等行可记录raw nuisance但权重0，不应列为优化贡献。

V2共同改为两fresh+Tensor Memory（容量131072、动量0.85、TTL64epoch）、clean锚定、多因素连续U权重、source Receiver Style Bank与实际batch90。feature/logit/prototype权重0.40/0.075/0.125，clean-anchor=0.025，旧nuisance/fingerprint关闭。P1是共同新基底；P2加tangent=0.035；P3加route=0.05；P4加RX对齐=0.075；P5加tail=0.10，但在tail开始前失败；E1为P5配置加批量identity-only教师；R1再配置subspace=0.05。

V2方向差分采用规范化弦长及各方向自己的步长/预算。route由nuisance和fingerprint在身份/域支路的相对敏感度构成hinge，margin均0.05；RX项只比较source同TX/激励桶的跨RX特征；tail项对RX×severity分组风险加权。各项分别线性ramp：orbit E10—30、soft/RX E25—50、tangent E40—70、route E55—80、tail E70—200。raw提前非零不能取代有效stage证据。

V2的raw tangent末值约5万，但其EMA尺度也约5万，归一化后再乘0.035；不能仅按raw量级判定爆炸。R1最终basis/eigenvalues为null、200轮subspace loss全0，子空间机制没有实际生效证据。E1同时变更tail和教师前向，P5失败使严格效率消融缺失。

## PAIR改革的精确目标、权重与安全边界

记学生clean/LEO单位特征为s_c、s_l，两fresh教师为t_c、t_l；r为物理可靠度，q为分类可信度，二者分开。未知物理质量按配置赋中性r=0.5，这不是质量测量。分类可信度是最高概率超机会水平、top-two margin和两教师JS一致性的启发式乘积，范围[0,1]；不代表经过校准的正确概率。

**点目标。**`t=Norm((1-m)t_c+m t_l)`，`m=0.25r`；定义

```text
D(s,t;epsilon)=mean_B[r ×
max(||Norm(s)-sg(t)||²-epsilon²,0)]
```

POINT使用`0.5×(D(s_c,t;0)+D(s_l,t;0))`，ASYMMETRIC只用`D(s_l,t;0)`，外乘pair_weight=0.5。两者均不是让学生双向互相追逐，教师与权重停止梯度。分母固定batch B，不能改成有效权重之和。

**无标签软分类。**从E21起增加

```text
0.2 × mean_B[r q × CE((p_c+r p_l)/(1+r),
logits_student,LEO)]
```

只使用两次fresh教师概率；缓存feature不充当第三份logit投票。有标签L没有该额外伪标签CE。PAIR feature从E11开启，两条路径分别记录，不应把E21说成所有机制的共同起点。

**SAFE。**固定真实CosFace类别单位权重w，clean教师正确且几何有效时，

```text
rho=alpha × min_{j≠y}
((w_y-w_j)·t_c)/||w_y-w_j||
```

，alpha=0.5；L损失是

```text
mean_B[r × valid × max(||s_l-t_c||²-rho²,0)]
```

再乘0.5。它是当前固定分类平面内的局部充分条件，不是未知域泛化保证；无效锚点不能算作安全样本。U没有真值，使用clean教师点目标和tolerance=0.1，不计算真类安全球。

**MEMORY。**仅POINT_MEMORY开启特征历史：

```text
q_cache=reliability_history × found ×
clamp(0.5+0.5 cos(t,history),0,1)
```

，目标再以`0.25 q_cache`混入历史feature。历史可用性与当前物理质量不是同一个门控。写缓存在optimizer step实际应用后提交，跳过步骤不提交；不从缓存伪造额外教师概率。

**TANGENT/ROUTE。**只在L上抽取floor(B×0.25)样本，以确定性eval前向比较扰动前后，仍保留梯度。定义`d(a,b)=0.5||Norm(a)-Norm(b)||²`；TANGENT为

```text
0.035 ×
mean[max(2d_nui,id/(delta/reference_scale)²-0.1,0)]
```

，delta=reference_scale=0.05。实际nuisance探针为STO，fingerprint从物理硬件干预中采样。ROUTE为

```text
0.05 ×
mean[max(0.05+d_nui,id-d_nui,dom,0)+max(0.05+d_fp,dom-d_fp,id,0)]
```

只TANGENT多2次子batch前向；带ROUTE多4次；U不跑方向目标。

TANGENT、ROUTE、TANGENT_ROUTE均叠加POINT基底。MATCHED_ZERO将pair与pseudo及方向权重置0，不执行两fresh教师PAIR路径。持久梯度投影在持续冲突达到条件后改变辅助梯度，不能以末epoch某次投影为0就断言从未触发。

旧

```text
train_loss_daot_orbit_z/logit/tangent/route
```

在PAIR路径保持0，因为新结果走weighted_components而未同步旧component telemetry；新路径的DAOT总损失、teacher views、feature/pseudo权重、探针前向、梯度cosine和实际投影记录均存在。故PAIR不能据旧零字段判为未运行；反过来，现有epoch日志也不足以精确分解各PAIR子loss的历史数值贡献。完整源码提供公式，但报告不从均值或计数捏造缺失分量。

### PAIR八种配置的实际命令权重

|配置|目标|feature|U soft|tangent|route|memory|
|---|---|---|---|---|---|---|
|A_POINT|point|0.5000|0.2000|0.0000|0.0000|false|
|ASYMMETRIC|asymmetric|0.5000|0.2000|0.0000|0.0000|false|
|B_SAFE|safe|0.5000|0.2000|0.0000|0.0000|false|
|POINT_MEMORY|point|0.5000|0.2000|0.0000|0.0000|true|
|TANGENT|point|0.5000|0.2000|0.0350|0.0000|false|
|ROUTE|point|0.5000|0.2000|0.0000|0.0500|false|
|TANGENT_ROUTE|point|0.5000|0.2000|0.0350|0.0500|false|
|MATCHED_ZERO|point|0.0000|0.0000|0.0000|0.0000|false|

## 实际激活证据与未解决的机制问题

全epoch非零轮数、首次非零epoch、末值与最大值见pair_activation_evidence.csv，不能只读配置。以E200为例：POINT005的两fresh教师观测比为2，L平均feature权重0.6494、U平均pseudo权重0.5251；MEMORY005的L/U缓存质量均值0.6463/0.5314，证明历史feature实际进入目标。TANGENT_ROUTE005抽取7104个L探针样本，额外前向28416样本，route投影比例0.004505；这些是方向与梯度路径的运行证据。

但运行不等于达到设计效果：TANGENT_ROUTE005的fingerprint身份/域弦长均值约0.002391/0.041380，仍表现为域支路对该干预更敏感，与希望的路由方向并不一致；不能由非零route loss宣布分离成功。SAFE006有效锚点率98.997%，实际落在安全半径内比例仅约2.197%（条件分母为有效锚点），分类翻转率约6.630%；这说明安全约束被计算，但大部分样本未进入所定义的充分安全区域。

SAFE006的U平均物理权重0.8376，而POINT005为0.5357，除版本混杂外也应保留实际训练权重分布差异；不能假定SAFE与POINT具有完全相同的目标强度。现有数据不足以把这一差异唯一归因于某个修复。

旧V1/V2的orbit prototype全部0、R1 subspace全部0有专门raw字段与终态状态支持，属于未激活结论；PAIR旧component字段全0则是记录路径缺口，两者证据含义不同。

## 全部最终测试结果

准确率单位%。失败行保持N/A；LEO_mean是三个场景等权。先展示早期同口径13个完成行及P5，再展示PAIR24行，保留方法、seed和版本边界。表格可排序；全精度值与计数附件用于复算。

### A1系列与RX-V2完整结果

|行|clean%|clear%|low%|rain%|LEO均值%|状态|
|---|---|---|---|---|---|---|
|A1|85.3464|72.8143|70.4155|70.2661|71.1653|COMPLETE|
|A2|82.0339|73.2488|71.8762|72.1560|72.4270|COMPLETE|
|A3|81.7429|73.3518|71.9696|72.2750|72.5321|COMPLETE|
|A4|82.0321|73.4923|72.0863|72.4054|72.6613|COMPLETE|
|A5|81.0518|72.9619|71.6190|71.8833|72.1548|COMPLETE|
|A6|80.7464|73.1667|71.8202|72.0548|72.3472|COMPLETE|
|A7|81.6161|73.5214|72.0881|72.3946|72.6681|COMPLETE|
|E1|82.6643|73.5107|72.0262|72.2881|72.6083|COMPLETE|
|R1|82.7429|73.5113|72.0500|72.1655|72.5756|COMPLETE|
|P1|82.5208|73.2155|71.8119|71.9655|72.3310|COMPLETE|
|P2|81.7155|73.1673|71.7423|71.9048|72.2714|COMPLETE|
|P3|83.1667|73.5643|72.1625|72.3530|72.6933|COMPLETE|
|P4|82.5101|72.5857|71.2726|71.4244|71.7609|COMPLETE|
|P5|N/A|N/A|N/A|N/A|N/A|FAILED|

### PAIR三seed完整结果

|行|clean%|clear%|low%|rain%|LEO均值%|状态|
|---|---|---|---|---|---|---|
|A_POINT_S392005|82.4548|73.1875|71.9975|72.0170|72.4007|COMPLETE|
|A_POINT_S392006|82.3708|73.6896|72.2346|72.2484|72.7242|COMPLETE|
|A_POINT_S392007|83.0131|73.5916|72.4967|72.4727|72.8537|COMPLETE|
|ASYMMETRIC_S392005|82.8476|73.1323|72.0242|71.9973|72.3846|COMPLETE|
|ASYMMETRIC_S392006|82.1631|73.4688|72.2221|72.1677|72.6195|COMPLETE|
|ASYMMETRIC_S392007|82.8506|73.5507|72.4164|72.4135|72.7935|COMPLETE|
|B_SAFE_S392005|N/A|N/A|N/A|N/A|N/A|FAILED|
|B_SAFE_S392006|82.4786|73.4901|72.2435|72.2789|72.6709|COMPLETE|
|B_SAFE_S392007|82.4577|73.2302|72.0385|72.0726|72.4471|COMPLETE|
|MATCHED_ZERO_S392005|82.5560|73.4082|72.1169|72.1767|72.5673|COMPLETE|
|MATCHED_ZERO_S392006|82.7363|73.2605|71.8674|71.7874|72.3051|COMPLETE|
|MATCHED_ZERO_S392007|82.3887|73.2409|71.8335|71.8412|72.3052|COMPLETE|
|POINT_MEMORY_S392005|83.3048|73.4065|72.3166|72.3471|72.6901|COMPLETE|
|POINT_MEMORY_S392006|82.4958|73.3815|71.9547|71.9937|72.4433|COMPLETE|
|POINT_MEMORY_S392007|82.5298|73.3406|72.2150|72.1677|72.5744|COMPLETE|
|ROUTE_S392005|82.9381|73.1946|72.0349|72.0744|72.4346|COMPLETE|
|ROUTE_S392006|82.7006|73.7501|72.5198|72.4135|72.8945|COMPLETE|
|ROUTE_S392007|82.5173|73.4385|72.2631|72.2610|72.6542|COMPLETE|
|TANGENT_ROUTE_S392005|81.8458|73.1536|72.1169|72.0134|72.4280|COMPLETE|
|TANGENT_ROUTE_S392006|82.5601|73.4777|72.2703|72.2592|72.6690|COMPLETE|
|TANGENT_ROUTE_S392007|82.1893|73.4438|72.3576|72.2143|72.6719|COMPLETE|
|TANGENT_S392005|83.1863|73.3156|72.1722|72.1677|72.5519|COMPLETE|
|TANGENT_S392006|82.1506|73.6041|72.2542|72.3417|72.7333|COMPLETE|
|TANGENT_S392007|82.6631|73.5934|72.4057|72.3902|72.7964|COMPLETE|

### PAIR均值、样本标准差与同seed差值

|方法|成功seed|clean均值%|LEO均值%|LEO seed样本SD|Δclean/pp|ΔLEO/pp|LEO正差seed数|平均小时|
|---|---|---|---|---|---|---|---|---|
|MATCHED_ZERO|3|82.5603|72.3925|0.1514|0.0000|0.0000|0|21.7769|
|A_POINT|3|82.6129|72.6595|0.2333|0.0526|0.2670|2|40.4849|
|ASYMMETRIC|3|82.6204|72.5992|0.2052|0.0601|0.2067|2|45.9361|
|POINT_MEMORY|3|82.7768|72.5693|0.1235|0.2165|0.1768|3|42.3232|
|TANGENT|3|82.6667|72.6939|0.1270|0.1063|0.3014|2|47.1315|
|ROUTE|3|82.7187|72.6611|0.2300|0.1583|0.2686|2|49.3027|
|TANGENT_ROUTE|3|82.1984|72.5897|0.1400|-0.3619|0.1971|2|46.4566|
|B_SAFE|2|82.4682|72.5590|0.1582|-0.0943|0.2539|2|44.6983|

### PAIR相对同seedMATCHED_ZERO的LEO平均差值

图表与可排序数据见report.html。

## A1优势与各批对照如何解释

A1比A4多5568条clean正确识别，clean错误数减少约18.45%；RX2贡献3041条，占净增54.62%，但排除RX2后其余六RX仍平均提高约1.755个百分点。RX10下降3.608点，不能写成所有RX全面提升。A1的clean最低RX=74.871%，A4为70.133%；LEO则在RX7/RX10相对A4下降约7.232/7.382点，RX×LEO最差单元54.862%明显低于A4的62.008%。

解释性效用`S=w×clean+(1-w)×LEO_mean`下，A1超过A4所需clean权重约31.10%，超过P3约41.21%。若clean与LEO两大类各50%，A1=78.256%、A4=77.347%、P3=77.930%；若四场景各25%，A1=74.710%、A4=75.004%、P3=75.312%。这说明偏好决定取舍，不是新增预登记指标或依据target选模。当前用户重视clean，因此应保留A1价值，同时清楚列出LEO尾部和训练稳定性代价。

同批冻结对照：A2−A1为clean−3.3125pp、LEO+1.2617pp；A4−A3为clean+0.2893pp、LEO+0.1292pp；A7−A6为clean+0.8696pp、LEO+0.3208pp。V2 P3−P2为clean+1.4512pp、LEO+0.4218pp；P4−P3为clean−0.6565pp、LEO−0.9323pp。前者仅支持该次配置对照方向，后者说明当前RX对齐实现与权重没有解决域偏移；这些均为单seed，不能证明普适效应。

PAIR的POINT_MEMORY三seed均比对应ZERO有正LEO差值，但均值仅+0.1768pp；TANGENT均值+0.3014pp，seed005却为负。TANGENT_ROUTE平均clean下降0.3619pp，不能说机制叠加全面更好。SAFE只完成2/3且版本不同，成功seed均值带幸存者偏差，不能当完整三seed成功结果。

大测试样本量不等于大量独立训练重复。接收机/日期/TX内样本相关，三seed共享同一目标集合；本文不报告缺乏设计支持的显著性或置信区间。早期没有同配置A0，A1又有异常优化轨迹，不能唯一归因于两视图均值；PAIR的ZERO也不能跨版本补成早期A0。

## 接收机、日期、类别与混淆数据

早期交付364个RX×场景单元，包含正确数/总数；已有评估没有逐日期、逐TX混淆或逐样本预测，本次不能恢复未记录维度。PAIR交付3588条receiver/day等分组结果和552条逐TX场景结果，另有23×4场景全部混淆矩阵与计数JSON。PAIR分组LEO汇总有按样本池计数加权的口径，不应无提示替换正文三场景等权均值。

PAIR多数行的最弱LEO接收机是RX7，但TANGENT_ROUTE005的最弱RX为RX2；不能写23行全部同一个最弱RX。早期A1 RX7三场景均值55.306%，A4为62.538%；PAIR结果大致仍在低60%区间，没有消除接收机尾部问题。TX类名与RX硬件名可能重复，逐类别表和逐接收机表需按字段区分。

全量附件可按row_id连接，但不能把早期缺少的逐日期/逐TX字段补成0，也不能用最弱接收机准确率冒充最弱类别。此报告没有K-shot适配、新类注册、unknown拒识、old/new harmonic、校准或统一部署延迟，均为未测而不是性能0。

## 完整训练过程、损失曲线与失败

以下曲线覆盖每行全部已记录epoch，既不截尾也不以最后一轮代表全程。为保证可读性，按V1/V2及PAIR三个seed分组；每组同时展示source clean、source LEO和总loss。PAIR图例简写为P=POINT、A=ASYMMETRIC、S=SAFE、Z=MATCHED_ZERO、M=POINT_MEMORY、T=TANGENT、R=ROUTE、TR=TANGENT_ROUTE；seed写在图标题中，完整row_id见结果表。全部子loss仍可在85字段×7318行表与1191字段完整数据中逐轮核对。

A1在E55 source clean最低16.670%、E64 source LEO最低16.735%；E56教师共识率99.296%，说明共识高仍可能一致地错，但没有该时刻预测直方图，不能断言全部预测为同一类。之后身份识别恢复，最后20轮clean约98.381%—98.411%、LEO约88.830%—88.972%；仅证明该次末期稳定。

A1加权域分类loss在E93首次超过10，E127超过100，E200=186.222，总loss=223.211，source域准确率6.667%接近15域机会水平。域分类支路loss巨大与身份对抗目标不同，不能解释为理想去域成功。其余完成行总loss约5—6.5；raw tangent的5万量级需除EMA尺度，不能与A1实际加权域loss混判。

P5在E10/B122触发RC4_SYSTEMIC_NONFINITE_BATCH_GUARD，8个异常批次、比例6.5574%；其tail要E70才启用，因此没有证据将失败归因于tail/CVaR。只有9个完整epoch和约0.490小时累计epoch时间下界，没有E200性能。

B_SAFE_S392005在E109后触发连续两轮zero_optimizer_step_streak保护；E94、97、102、108、109的train_loss没有有效记录，source clean降到16.667%。这些空值保留，不能将无有效optimizer step的损失写成0并宣称收敛。故障证据只能证明非有限/零更新症状，缺少首个异常算子的历史张量，不能唯一锁定根因或推断调低阈值即可修复。

完成行也出现过少量非有限梯度batch跳过，A7还记录过非有限loss跳过；COMPLETED不等于全程无数值问题。旧恢复失败记录另附历史报告，本次未中断、重启任何训练。

### v1：source clean准确率%（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### v1：source LEO准确率%（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### v1：训练总loss（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### v2：source clean准确率%（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### v2：source LEO准确率%（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### v2：训练总loss（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### pair392005：source clean准确率%（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### pair392005：source LEO准确率%（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### pair392005：训练总loss（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### pair392006：source clean准确率%（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### pair392006：source LEO准确率%（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### pair392006：训练总loss（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### pair392007：source clean准确率%（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### pair392007：source LEO准确率%（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### pair392007：训练总loss（全部epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### A1总loss与加权域分类loss（全部200epoch）

完整交互曲线见同目录report.html；原始数值见training_all_fields.csv.gz及training_curves.csv。

### 38行训练健康与资源汇总

|行|epoch|末轮source clean%|末轮source LEO%|末轮总loss|梯度跳过涉及epoch|loss跳过涉及epoch|小时|峰值allocated GiB|
|---|---|---|---|---|---|---|---|---|
|A1|200|98.3852|88.9543|223.2110|30|0|25.0297|2.7864|
|A2|200|98.6111|95.7049|5.0437|24|0|25.9484|2.7898|
|A3|200|98.6000|95.6679|5.0184|17|0|27.6726|2.8295|
|A4|200|98.6333|95.7099|5.2155|14|0|24.6475|2.9121|
|A5|200|98.6185|95.6716|5.3196|29|0|26.7565|2.9525|
|A6|200|98.5926|95.6346|5.1154|16|0|29.7556|2.9146|
|A7|200|98.6407|95.7309|5.1084|23|1|26.8765|3.0851|
|E1|200|98.6111|95.0704|5.4071|14|0|39.1591|5.0252|
|R1|200|98.5889|95.0136|5.4028|16|0|39.0135|4.9877|
|P1|200|98.5519|94.9395|5.0860|21|0|37.6055|4.0607|
|P2|200|98.5407|94.9272|5.1188|16|0|37.3003|4.2329|
|P3|200|98.5519|94.9222|5.2406|14|0|39.0096|5.1358|
|P4|200|98.5778|94.8444|5.3850|17|0|40.2813|5.1323|
|P5|9|97.7963|79.2778|3.6708|4|0|0.4904|N/A|
|A_POINT_S392005|200|98.6704|96.0049|6.4518|30|0|41.1089|3.6601|
|A_POINT_S392006|200|98.6741|95.9877|6.4105|31|0|34.5329|3.6693|
|A_POINT_S392007|200|98.6556|96.0593|6.4076|25|0|45.8128|3.6717|
|ASYMMETRIC_S392005|200|98.6556|96.1247|6.4410|28|0|45.9114|3.8043|
|ASYMMETRIC_S392006|200|98.6852|96.1432|6.4076|29|0|45.5465|3.7990|
|ASYMMETRIC_S392007|200|98.6519|96.1185|6.4107|25|0|46.3505|3.7850|
|B_SAFE_S392005|109|16.6667|16.6667|N/A|12|22|22.5850|N/A|
|B_SAFE_S392006|200|98.6741|96.1099|6.4032|30|0|44.4052|3.7853|
|B_SAFE_S392007|200|98.6370|96.0580|6.3988|26|0|44.9914|3.7723|
|MATCHED_ZERO_S392005|200|98.6481|95.5494|6.3644|28|0|21.0599|1.7754|
|MATCHED_ZERO_S392006|200|98.6593|95.5790|6.3495|30|0|22.4817|1.7601|
|MATCHED_ZERO_S392007|200|98.6556|95.6333|6.2861|28|0|21.7890|1.7579|
|POINT_MEMORY_S392005|200|98.6519|95.9802|6.4462|30|0|47.6900|3.7485|
|POINT_MEMORY_S392006|200|98.7074|95.9951|6.3919|29|0|38.0899|3.7560|
|POINT_MEMORY_S392007|200|98.6407|95.9753|6.3903|26|0|41.1897|3.7500|
|ROUTE_S392005|200|98.6778|96.0395|6.4407|26|0|48.3667|4.2691|
|ROUTE_S392006|200|98.6556|95.9975|6.4374|26|0|49.9572|4.2719|
|ROUTE_S392007|200|98.6556|96.0037|6.4119|28|0|49.5843|4.2531|
|TANGENT_ROUTE_S392005|200|98.6667|96.0284|6.4462|26|0|50.0988|4.2765|
|TANGENT_ROUTE_S392006|200|98.6926|95.9667|6.4191|26|0|49.5229|4.2588|
|TANGENT_ROUTE_S392007|200|98.6556|96.0444|6.4286|25|0|39.7481|4.2648|
|TANGENT_S392005|200|98.6963|95.9815|6.4332|29|0|46.2316|3.9700|
|TANGENT_S392006|200|98.7074|95.9741|6.4229|27|0|47.0885|3.9253|
|TANGENT_S392007|200|98.6556|96.0247|6.4062|28|0|48.0743|3.9560|

## 资源成本与效率边界

A1本次训练墙钟25.030小时，A4为24.647小时（差约23分钟），P3为39.010小时。PAIR ZERO三seed平均21.777小时，POINT40.485小时、TANGENT47.131小时，方向/教师与缓存目标没有自动带来更短训练。早期参数1130809，PAIR1116471；峰值allocated约1.76—5.14GiB视配置而异，reserved与外部nvidia-smi占用并不是同一个口径。

这些时间不含继承CORE90预训练、测试准备及历史失败重试；并发状态、batch与版本也不同，不能称隔离条件的算法加速比。失败行仅列已完成epoch累积时间下界，成功行用resource summary的wall time。未测能耗与统一batch1部署延迟，不能由训练资源推断推理更快。

PAIR每行模型预测覆盖clean168000+LEO168000=336000样本前向，既有执行记录中的约17秒指预测阶段，不包括输入准备、checkpoint加载、评分或完整测试墙钟；不会将它报告为端到端耗时。

## 稳健性、证据缺口与后续问题

当前结论分四层：36行训练与固定E200评分产物闭合；部分机制确实运行；部分预期效果未实现或记录不足；科学默认晋级没有获得充分证据。特别是A1的clean价值、训练异常、LEO弱接收机代价应一起保留，不能靠单一排名抹掉其中一项。

如果后续另行授权研究，应从source证据预登记验证方案：解释A1域分类失衡与身份恢复是否可复现；统一教师缓存修复后的版本再做公平对照；补齐PAIR子损失和有效梯度贡献日志；检验SAFE安全区域实际覆盖与fingerprint路由方向。现有target分数不得回流调参、选择或触发重跑，需要与当前事后分析隔离的确认安排。

仍待回答：A1高clean是否依赖独特优化轨迹？修复域支路能否保留clean优势而改善LEO尾部？PAIR小幅差值是否跨独立seed稳定？这些都没有被本报告证明。技术失败、缺失日志或未激活目标均不能凭设计意图补成成功结果。

## 数据附件、来源与复现

建议先读本报告，再用all_test_results.csv查看36行最终结果及2个失败空值；用combined_pair_method_summary.csv核对三seed统计；用training_health.csv定位异常epoch，再打开training_losses.csv与training_all_fields.csv.gz追踪相应分量。

- training_all_fields.csv.gz：7318行、1191个联合字段，保留嵌套原始值与缺失；是所有已记录训练数值的主表。
- training_losses.csv：7318行、85个标准原始/加权loss字段；training_curves.csv为常用诊断字段；all_numeric_statistics.csv与loss_statistics.csv给出全部数值/标准loss的极值、均值、非零轮数。
- configurations.json：早期运行配置及PAIR实际发布命令；pair_activation_evidence.csv记录新路径激活统计。
- pair_test_all_counts_confusions.json、pair_receiver_day_breakdown.csv、pair_per_class.csv、early_receiver_scenario.csv：既有完整计数、混淆及可用分组数据。
- raw_early_training_test_logs.zip：14行早期完整文本/JSON/CSV证据；raw_pair_training_logs.tar.gz：PAIR24行完整对应日志、JSON/CSV及三个实际release核心源码/配置。
- early_release_sources.tar.gz：取回的V2源码；early_historical_git_reference.zip：A1历史参考源码及明确身份限制。
- log_scan.json、structured_file_inventory.json、raw_*_inventory.json、validation.json和文件清单：完整解析、失败位置及归档完整性证据。
- prior_two_batch_report.md、prior_a1_report.md与prior_pair23_report.md：已核查历史说明；prior_pair_recovery_report.md保留被替代技术尝试及EMA修复历史。

归档明确不含ManySig.pkl、训练checkpoint的.pth张量、逐样本预测大数组或received-IQ输入NPZ；不假称所有大产物已上传。它包含本报告使用的全部已有训练日志、配置、评分汇总和细分计数；早期未记录的逐TX/日期结果不存在。raw文件与完整训练主表可离线复核，重跑原实验仍需原数据、checkpoint及相应环境。

复现入口为仓库tools/build_adv3b02_combined_report.py（collect/sources为读取原环境，analyze为本地解析）及tools/render_adv3b02_combined_report.py（报告数据/正文）；HTML通过Data Analytics标准portable reader生成。Git交付提交与远端一致性以最终独立读回为准，不在文件内自引用未知提交。
