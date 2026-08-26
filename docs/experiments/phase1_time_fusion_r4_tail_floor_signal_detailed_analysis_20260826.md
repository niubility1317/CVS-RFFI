# Time+Fusion Rank-4 Phase1尾类改善信号详细数据与Target5反证报告

日期：2026-08-26

Phase1 run：`phase1_adv3b02_meta_adapter_time_fusion_r4_p4_s392002_20260825_r1`

Target5 run：`stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1`

冻结基线：`ADV3B02_CORE90_SOFT_E200`

## 一、结论

Time+Fusion Rank-4在Phase1 source held-out测试上产生了明确但高度集中的尾类改善：`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`的旧类floor分别提高`+6.7857pp`、`+6.3571pp`和`+5.8071pp`。三场景平均floor由47.6595%提高到53.9762%，平均变化为`+6.3167pp`。

这不是全局鲁棒性提升。三场景平均准确率由76.4079%下降到76.0893%，变化为`-0.3187pp`。逐类结果显示，source弱类class3在三场景共增加2653个正确样本，class4增加1029个，但class1减少3377个；其余class0、class2和class5合计减少1108个。三场景最终净减少803个正确样本。

对应Target5形成了直接反证：15／15个row、6300条opaque query prediction和1800条truth-last old-class评分全部闭合。每个row执行3次真实support反向传播，余弦分数最大变化为0.003380～0.030114，但最终类别变化数为0；`DA1_REG0-DA0_REG0`聚合旧类均值和floor均为0pp。该方法不满足`+1.0pp/+0.5pp`双门槛，结论为`SCIENTIFIC_FAILURE_NO_PROMOTION`。

最关键的机制错位是弱类身份发生了变化：Phase1 source LEO的floor始终由class3决定，Time+Fusion重点抬高的正是class3；Target5中，low-elev和rain的floor由class1决定，clear中class1也与class0、class3同为floor。Phase1训练同时使source class1下降7.61～8.26pp。这种“改善source class3、损伤class1”的重分配没有对准目标接收机的主要瓶颈，因此不能称为目标域适应收益。

## 二、方法与协议

### 2.1 可训练范围

候选只在原双分支编码器中启用time和fusion adapter：

|项目|实际值|
|---|---:|
|adapter rank|4|
|Phase2正式更新步数|3|
|deployment可训练张量|20|
|可训练参数|5780／1055449|
|参数占比|0.547634%|
|快速状态大小|23120字节|

20个deployment可训练张量只属于`id/dom_backbone.meta_adapter_time`和`id/dom_backbone.meta_adapter_fusion`。候选不含freq adapter、分类头、LDA、协方差估计或持久新头。Phase2判决始终使用`frozen_prototype_cosine_v1`。

### 2.2 Phase1训练数据与episode

Phase1只读取WiSig source receiver0～6、source days0～1，角色比例为`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。正式测试使用source days2～3，每个场景84000条样本、6个旧类、每类14000条，与训练和选择角色的物理样本不相交。

训练完成200个outer step，每步4个episode，共800个episode；每个episode执行3步inner update。完整日志中的任务与K-shot分布为：

|维度|类别|数量|占比|
|---|---|---:|---:|
|episode|`Q_SAME_DOMAIN`|320|40%|
|episode|`Q_RX_HOLDOUT`|160|20%|
|episode|`Q_DAY_CHANNEL_HOLDOUT`|120|15%|
|episode|`Q_CLEAN_TO_LEO`|120|15%|
|episode|`Q_LEO_CROSS`|80|10%|
|K-shot|K1|280|35%|
|K-shot|K2|200|25%|
|K-shot|K5|160|20%|
|K-shot|K10|160|20%|

800条episode记录的`loss_adapt`、`loss_guard`和`loss_floor`全部有限；NaN／Inf计数为0。`metrics.csv`完整覆盖step0～199，每步均记录12个实际发生状态变化的参数张量。严格bundle审计记录20个deployment可训练张量；前者是outer step中实际变化的状态数，后者是Phase2允许更新的完整参数集合，二者口径不同。当前日志只保存变化数量，没有保存12个张量的具体名字，因此本报告不进一步推断未变化张量的精确归属。

## 三、完整训练日志分析

### 3.1 outer loss

200个outer step的loss统计如下：

|统计量|数值|
|---|---:|
|均值|18.3760|
|中位数|18.7037|
|P90|24.3746|
|最小值|6.5544（step60）|
|最大值|31.4060（step34）|
|前20步均值|19.1356|
|后20步均值|18.2362|
|后20步−前20步|-0.8993|
|线性斜率|-0.004993／step|

outer loss整体缓慢下降，但不同episode任务没有同步收敛。由于每个meta batch混合不同episode和K-shot，单步loss波动较大；最小值与最大值都出现在训练前半段，不能用某个“最低step”代替最终checkpoint选择。

### 3.2 分任务前后期变化

下表比较step0～19与step180～199的同类episode均值，格式为“前20步→后20步”。

|episode|`loss_adapt`|`loss_guard`|`loss_floor`|
|---|---:|---:|---:|
|`Q_SAME_DOMAIN`|3.4588→2.3879|2.3669→2.8720|10.9935→10.1583|
|`Q_RX_HOLDOUT`|2.6143→3.1013|1.7030→1.6594|9.8840→8.7017|
|`Q_DAY_CHANNEL_HOLDOUT`|5.5782→4.9668|2.3827→2.7389|10.4767→11.0177|
|`Q_CLEAN_TO_LEO`|3.0646→3.1818|4.6624→4.5059|12.6652→12.4849|
|`Q_LEO_CROSS`|0.8641→0.8994|1.9891→1.2218|5.0879→5.5806|

`Q_SAME_DOMAIN`的adapt和floor loss下降，`Q_RX_HOLDOUT`的floor loss下降，但其adapt loss上升；`Q_DAY_CHANNEL_HOLDOUT`的adapt loss下降，同时guard和floor loss上升；两个LEO任务也呈现不同方向。训练没有形成“所有任务、所有损失共同下降”的统一解。最终checkpoint的尾类信号应解释为多任务折中后的类间重分配，而不是普遍收敛。

### 3.3 梯度诊断限制

800条`grad_cos_support_query`全部为`null`。代码在support/query梯度余弦无法生成时保留`null`，因此该字段不能证明梯度同向、反向或恰好为0。本报告只依据实际参数变化、适配曲线、最终分类结果和Target5 prediction解释机制，不使用缺失的梯度余弦作因果证据。

## 四、source-only选择曲线

`source_adaptation_curve.json`完整评价A0、A1、A3、A5和A10。两个`V_cal` episode在所有步数均保持原值：同域K2为100% mean／100% floor；`Q_LEO_CROSS` K1为75% mean／0% floor。`V_select`结果如下：

|episode|A0|A1|A3|A5|A10|floor变化|
|---|---:|---:|---:|---:|---:|---:|
|`Q_SAME_DOMAIN` K2|100.00%|100.00%|100.00%|100.00%|100.00%|0pp|
|`Q_RX_HOLDOUT` K1／`leo_clear_weak`|83.33%|91.67%|91.67%|91.67%|91.67%|0pp（始终50%）|

第二个`V_select` episode在第1步已经获得全部`+8.3333pp`均值收益，继续到3、5或10步没有新增正确分类；guard accuracy始终100%，floor始终50%。source选择规则因此给出`SOURCE_SELECTION_ELIGIBLE`，但它证明的是一个source RX-holdout小episode的均值无遗忘，不是最终大样本LEO floor收益，也不是target receiver收益。

该episode的实际时延为A0 28.49ms、A1 75.62ms、A3 168.11ms、A5 260.82ms和A10 491.92ms。正式部署使用A3；快速状态大小在所有curve row中均为23120字节。

最终Meta-SGD模块步长为：

|模块|步长|
|---|---:|
|`id_backbone.meta_adapter_time`|0.0010003233|
|`id_backbone.meta_adapter_fusion`|0.0009969906|
|`dom_backbone.meta_adapter_time`|0.0009995003|
|`dom_backbone.meta_adapter_fusion`|0.0009995003|

四个步长仍接近0.001。Meta-SGD没有学出数量级上的快慢分离；其作用主要来自adapter权重本身及time／fusion位置，而不是极端放大某个模块的更新率。

## 五、Phase1最终大样本测试

### 5.1 场景级结果

|场景|P0均值|Final均值|均值变化|P0 floor|Final floor|floor变化|正确数变化|
|---|---:|---:|---:|---:|---:|---:|---:|
|clean|92.0036%|91.9298%|-0.0738pp|87.9286%|87.9214%|-0.0071pp|-62|
|`leo_clear_weak`|79.1750%|78.9595%|-0.2155pp|52.8357%|59.6214%|+6.7857pp|-181|
|`leo_low_elev_weak`|75.1333%|74.7810%|-0.3524pp|45.7929%|52.1500%|+6.3571pp|-296|
|`leo_rain_weak`|74.9155%|74.5274%|-0.3881pp|44.3500%|50.1571%|+5.8071pp|-326|

三类LEO平均均值为76.4079%→76.0893%，变化`-0.3187pp`；平均floor为47.6595%→53.9762%，变化`+6.3167pp`。这是一条清晰的均值—尾类权衡曲线。

### 5.2 逐类完整结果

每格为“P0→Final（变化）”。

|场景|class0|class1|class2|class3|class4|class5|
|---|---:|---:|---:|---:|---:|---:|
|clean|95.0857→95.0143（-0.0714）|91.0286→89.3143（-1.7143）|89.3786→87.9214（-1.4571）|87.9286→90.0000（+2.0714）|95.8786→96.8929（+1.0143）|92.7214→92.4357（-0.2857）|
|`leo_clear_weak`|80.5500→80.5429（-0.0071）|76.7643→69.1571（-7.6071）|82.5857→80.9357（-1.6500）|52.8357→59.6214（+6.7857）|92.4714→94.1857（+1.7143）|89.8429→89.3143（-0.5286）|
|`leo_low_elev_weak`|74.8500→74.5286（-0.3214）|72.1429→63.8857（-8.2571）|80.1214→78.4000（-1.7214）|45.7929→52.1500（+6.3571）|89.4857→92.1071（+2.6214）|88.4071→87.6143（-0.7929）|
|`leo_rain_weak`|76.1786→75.8929（-0.2857）|70.9857→62.7286（-8.2571）|80.6143→78.8286（-1.7857）|44.3500→50.1571（+5.8071）|88.4286→91.4429（+3.0143）|88.9357→88.1143（-0.8214）|

### 5.3 正确样本重分配

每个source测试场景中每类有14000条样本。三类LEO合计的正确样本变化为：

|类别|正确样本变化|
|---|---:|
|class0|-86|
|class1|-3377|
|class2|-722|
|class3|+2653|
|class4|+1029|
|class5|-300|
|合计|-803|

floor提升完全来自class3。class3在clear、low-elev和rain分别增加950、890和813个正确样本；class1分别减少1065、1156和1156个。class4的增长部分抵消了class1损失，但不足以保持总体均值。

### 5.4 类间离散度

|场景|P0类准确率标准差|Final类准确率标准差|P0极差|Final极差|
|---|---:|---:|---:|---:|
|`leo_clear_weak`|12.9337pp|11.6585pp|39.6357pp|34.5643pp|
|`leo_low_elev_weak`|14.5885pp|13.5948pp|43.6929pp|39.9571pp|
|`leo_rain_weak`|15.0717pp|14.2842pp|44.5857pp|41.2857pp|

三个场景的标准差和极差都下降，说明类别分布变得更均衡。但均衡是通过“抬高最弱class3、明显压低class1、抬高class4”实现的，不是所有类别共同改善。

## 六、Target5测试设计与完整结果

### 6.1 测试矩阵与协议检查

Target5使用seed392002、receiver20-1，覆盖`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`和`K1/new20`，每个operating point包含三类LEO weak，共15个row。

|证据|结果|
|---|---|
|truth-free工厂|15／15 row，`TARGET_INPUTS_COMPLETE`|
|真实checkpoint无query smoke|PASS，3次backward，参数占比0.547634%|
|prediction|15／15 row，`PREDICTIONS_COMPLETE`|
|prediction总量|6300条opaque query prediction|
|truth-last old-class评分|1800条，15 row×6类×20条|
|query边界|`query_opened_before_adaptation=false`、`query_role_opened=false`、`query_truth_opened=false`|
|状态更新|`query_state_update_count=0`|
|source访问|`source_opened=false`|
|checkpoint|15／15严格加载|
|same-row|15／15为true|
|最终决策变化|0|

6300条prediction覆盖每个row的全部opaque query；REG0的新类指标按协议为N/A。独立scorer只在prediction闭合后连接1800条old-class truth记录计算本报告的旧类均值和floor。

### 6.2 15行数值

每行的`DA0_REG0`与`DA1_REG0`旧类均值和floor完全相同。表中同时列出全部6类余弦score的最大绝对变化与平均绝对变化。

|operating point|场景|support数|prediction数|均值DA0→DA1|floor DA0→DA1|max score变化|mean score变化|
|---|---|---:|---:|---:|---:|---:|---:|
|K10/new5|clear|60|220|63.3333→63.3333|40→40|0.003379732|0.000083673|
|K10/new5|low-elev|60|220|60.8333→60.8333|30→30|0.007703066|0.000203280|
|K10/new5|rain|60|220|63.3333→63.3333|40→40|0.017900199|0.000202995|
|K10/new10|clear|60|320|63.3333→63.3333|40→40|0.005232871|0.000081624|
|K10/new10|low-elev|60|320|60.8333→60.8333|30→30|0.007703066|0.000189892|
|K10/new10|rain|60|320|63.3333→63.3333|40→40|0.017900199|0.000181361|
|K10/new20|clear|60|520|63.3333→63.3333|40→40|0.010603607|0.000081272|
|K10/new20|low-elev|60|520|60.8333→60.8333|30→30|0.012220517|0.000179660|
|K10/new20|rain|60|520|63.3333→63.3333|40→40|0.017899849|0.000143518|
|K5/new20|clear|30|520|63.3333→63.3333|40→40|0.012261957|0.000074095|
|K5/new20|low-elev|30|520|60.8333→60.8333|30→30|0.016915227|0.000190558|
|K5/new20|rain|30|520|63.3333→63.3333|40→40|0.030114323|0.000216321|
|K1/new20|clear|6|520|63.3333→63.3333|40→40|0.018874556|0.000215518|
|K1/new20|low-elev|6|520|60.8333→60.8333|30→30|0.020349502|0.000271282|
|K1/new20|rain|6|520|63.3333→63.3333|40→40|0.013279051|0.000064413|

score确实发生变化，而且K1与K5的部分row比K10产生更大最大扰动；但任何row都没有越过argmax边界。增加support数量也没有形成决策级收益。

### 6.3 Target5逐类准确率

五个operating point在同一场景上得到相同的old-class逐类结果，DA0与DA1也严格相同：

|场景|class0|class1|class2|class3|class4|class5|floor类|
|---|---:|---:|---:|---:|---:|---:|---|
|clear|40%|40%|70%|40%|100%|90%|class0／1／3并列|
|low-elev|40%|30%|70%|45%|90%|90%|class1|
|rain|55%|40%|75%|45%|95%|70%|class1|

source侧最弱类是class3，Target5 low-elev和rain的最弱类却是class1。Time+Fusion在Phase1显著损伤的class1，恰好成为目标接收机的主要floor瓶颈。

### 6.4 决策margin

在1800条实际参与old-class评分的query上：

|统计量|DA0 top1−top2 margin|DA1 top1−top2 margin|
|---|---:|---:|
|最小值|0|0|
|P1|0.014309|0.014165|
|P5|0.084596|0.081789|
|中位数|0.913871|0.913738|
|均值|0.759869|0.759969|

DA1相对DA0的margin变化均值仅`+0.000100`；740条margin下降，1055条上升，5条不变。25条DA0 margin低于0.02，仍没有任何类别变化。更新不是全零，但它主要对不同类别分数产生共同小幅位移，没有形成稳定的跨边界纠错。

## 七、测试与交付证据

### 7.1 Phase1实现测试

|测试项|结果|
|---|---|
|未注册profile RED|Phase1 config、单分支bundle、真实双分支严格回读3项按预期失败|
|新增profile GREEN|只允许`rank4+time,fusion`；rank8同组合继续拒绝|
|配置差异|相对fusion-only rank4只变化`run_id`与`adapter.sites`|
|邻近回归|259项通过|
|生产入口编译|9项通过|
|独立P0/P1审查|P0无、P1无|
|release SHA|本地／远端均为`33b43785255d582e8c49ae654ad27bc50c2d8ab943af284d0a451431ec8b73a7`|
|真实checkpoint smoke|PASS；20个time／fusion张量、3步、0.547634%、无query／target读取|

### 7.2 Phase1运行测试

|测试项|结果|
|---|---|
|outer step|200／200|
|episode|800／800|
|每步episode|固定4|
|inner update|800／800均为3步|
|有限性|2400个loss字段全部有限|
|实际变化状态数|200／200 step均为12|
|stdout错误扫描|Traceback、RuntimeError、ValueError、OOM、Killed、NaN、Inf均为0|
|正式artifact|9／9非空|
|最终状态|`ARTIFACTS_COMPLETE / SOURCE_SELECTION_ELIGIBLE`|

### 7.3 Target5测试

|测试项|结果|
|---|---|
|Stage2聚焦回归|69项通过|
|生产入口编译|本地11项、远端11项通过|
|独立P0/P1审查|P0无、P1无|
|release SHA|本地／远端均为`3a238aeabf155155eb165a7fe6bc63c240b433e0c1fa1e1b755ae179b8caef02`|
|factory|15／15|
|真实no-query smoke|PASS|
|prediction artifacts|15组DA0、15组DA1、15个receipt全部非空|
|score artifacts|15个`score.json`和矩阵summary全部非空|
|协议审计|15／15严格checkpoint、3 backward、query只读、source关闭、same-row|
|结果|`ANALYZED / SCIENTIFIC_FAILURE_NO_PROMOTION`|

## 八、为什么不能称为目标域DA收益

第一，Phase1表格评价的是source receiver0～6、held-out days2～3。即使叠加了LEO weak信道，这些样本仍属于source接收机链路；它们没有覆盖receiver20-1的接收机响应偏移。

第二，Phase1比较的是P0控制checkpoint与meta训练后的Final checkpoint，测量地面训练改变了什么。正式域适应效应必须在同一Target5 row内比较`DA1_REG0-DA0_REG0`，只改变合法target support引起的快速更新状态。该对比结果为0pp／0pp。

第三，source floor改善集中在class3，而Target5 floor主要由class1决定。source弱类排序没有跨接收机保持稳定，按source类尾部优化不能替代目标域support驱动的类风险估计。

第四，Phase1结果只有单seed，且尾类提升伴随均值下降。即使没有Target5反证，它也只能写成“source-side tail signal”，不能写成稳健的跨域提升。

## 九、科学判断

Time+Fusion Rank-4证明了少层meta训练可以重新分配类别性能，并显著抬高source场景中最弱的class3；它也减少了三类LEO场景的类间标准差和极差。这是可重复核对的尾类机制信号。

同一证据同时否定了更强的表述。该机制损伤class1，降低LEO平均准确率；Target5中的真正弱类转为class1，3步support更新只改变score、不改变任何类别决策。Time+Fusion Rank-4不能晋级，也不应作为当前正向DA方法。

这轮实验的有效收获不是“time+fusion已经解决域适应”，而是定位了下一类方法必须满足的条件：弱类目标不能由source class ID或source排序预先固定，必须从每个target support类用同一公式估计风险；更新必须同时约束平均损失和最差类损失，并在冻结判决margin上产生可测的跨边界纠错。后续若恢复实验，应检验这种class-permutation-invariant support目标，而不是继续增加fusion容量。

## 十、证据位置

- Phase1预登记与完成报告：[phase1_adv3b02_meta_adapter_time_fusion_r4_p4_s392002_20260825_r1_report.md](phase1_adv3b02_meta_adapter_time_fusion_r4_p4_s392002_20260825_r1_report.md)
- Target5预登记与完成报告：[stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1_report.md](stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1_report.md)
- Phase1本地完整回读：`E:\type10-7\local_artifacts\meta_adapter_recovery\phase1_time_fusion_r4_r1_complete_20260825`
- Target5本地完整回读：`E:\type10-7\local_artifacts\meta_adapter_recovery\target5_time_fusion_r4_p4_r1_complete_20260825`

本报告读取了Phase1全部800条`logs.jsonl`、全部200条`metrics.csv`、完整source adaptation curve、P0／Final逐类评价、run summary、config snapshot和完整stdout；Target5读取了15个row receipt、15个score、30份prediction数组、factory／smoke／matrix receipt和矩阵summary。未使用日志尾部抽样代替完整解析。
