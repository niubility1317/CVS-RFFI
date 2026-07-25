# Stage2自研方法全面对比（更新至2026-07-25）

## 1.比较口径

本表只纳入符合`项目.md`中`p2_min_v1`输入、查询隔离、逐样本全注册类竞争和声明边界的方法。性能必须来自同一候选、同一row绑定的注册前旧类、注册后旧类、seen-new、`H_old_new`、floor和遗忘；不同矩阵的数值只分层展示，不作直接冠军排序。

当前目标为：K10时注册后旧类≥92%、最低旧类≥85%、new5/new10/new20分别≥92%/90%/86%；K5相对matched K10/new20的注册后旧类、最低旧类、seen-new和H衰减均≤5pp；K1相对同row M0必须产生严格正收益，且旧/新保护项不恶化。

2026-07-25复核说明：D62、D81、D92、SVRN-qKNN-BCRR/r4.2和ADV3B02-r8均重新读取完整报告及结构化汇总；D91重新读取完整105行训练日志汇总、15行outer性能和D62配对哈希；D103-R2重新读取当前正式预注册报告。下文继续严格区分完整125、开发15、Phase1-held和无性能结果，不把不同证据面混排为一个冠军榜。

证据层级：

|层级|矩阵|可作何种结论|
|---|---|---|
|完整125|5receiver×5seed×5slice；375个scene slice|稳定性和matched总体比较|
|Target25|5receiver×1seed×5slice；75个scene slice|当前小步研发的首次目标域证伪|
|开发15|1receiver×1seed×K10/new5×3scene×5fold|机制开发诊断，不代表receiver/seed/K泛化|
|Phase1-held54|1个held receiver×6个pseudo-new×3scene×3K|表示机制伪证，不代表正式Stage2目标性能|
|本地release/smoke|合成或真实检查点无query小样本|实现、协议和资源证据，不是性能|

## 2.拥有完整125证据的方法

所有数值均为百分数或百分点。

|方法|主要机制|B-old|A-old|Min-old|seen-new|H|遗忘|结果|
|---|---|---:|---:|---:|---:|---:|---:|---|
|D81|地面扰动谱Cauchy稳健support中心|81.55|64.40|35.20|59.11|61.09|17.15|完整125阴性|
|D62|cross-fitted Fisher匿名类行Pareto拼接|81.51|64.39|35.15|59.11|61.09|17.11|完整125阴性，几乎等于D81|
|D92|old/new任务均衡共享协方差头|81.55|65.56|36.81|58.93|61.57|15.99|当前合法125中联合指标最强，但新类交换且K1无效|
|SVRN-qKNN-BCRR/r4.2|支持方差重整、qKNN和BCRR回滚|73.10|43.03|11.21|23.46|29.25|30.07|完整125显著劣于D62|
|ADV3B02 TS-DRQKNN-BCRR/r8 M0|双qKNN/BCRR基础臂|72.60|43.02|11.17旧类最小值；全类floor 2.29|23.49|28.95|29.57|完整125阴性|
|ADV3B02 TS-DRQKNN-BCRR/r8 M_DA|domain-conditioned qKNN分支|72.66|43.06|11.20旧类最小值；全类floor 2.27|23.44|28.92|29.60|相对M0的H为−0.027pp，协同为0|

### 2.1 D62逐slice

|slice|B-old|A-old|A-old floor|seen-new|H|遗忘|
|---|---:|---:|---:|---:|---:|---:|
|K10/new5|86.02|76.33|50.47|73.57|74.60|9.69|
|K10/new10|86.02|71.53|42.27|66.75|68.84|14.49|
|K10/new20|86.02|68.68|37.93|68.78|68.56|17.34|
|K5/new20|81.32|61.39|30.87|59.28|60.03|19.93|
|K1/new20|68.14|44.03|14.20|27.15|33.41|24.11|

D62注册后只有24/375个scene状态真正激活，351/375回退或未接纳任何Fisher行；K1为75/75精确fallback。因此其125结果与D81近似逐值一致，不是稳定的Fisher增益。D62未使用地面压缩原型，不能据此否定ground知识。

### 2.2 D92逐slice及相对D81变化

|slice|A-old|Min-old|seen-new|H|遗忘|相对D81的关键变化|
|---|---:|---:|---:|---:|---:|---|
|K1/new20|44.03|14.20|27.15|33.41|24.11|全部逐值一致|
|K5/new20|63.71|33.20|58.88|60.96|17.56|A-old+2.311、Min-old+2.400、new−0.410、H+0.920|
|K10/new5|76.19|49.80|74.13|74.80|9.92|A-old−0.133、Min-old−0.867、new+0.520|
|K10/new10|72.53|44.20|66.35|69.11|13.58|A-old+1.000、Min-old+1.933、new−0.340|
|K10/new20|71.33|42.67|68.15|69.56|14.78|A-old+2.622、Min-old+4.600、new−0.653、H+0.964|

D92证明“改变注册头的共享协方差”能稳定减轻大规模注册遗忘，但每个旧类正向slice都伴随新类下降；其`0.5Σ_old+0.5Σ_new`还依赖old/new任务角色，不适合作为下一DA的训练权重模板。K1因无类内协方差而严格identity。

### 2.3 SVRN和ADV3B02的否定性证据

- SVRN相对D62的125行paired差为：A-old−21.36pp、A-old floor−23.93pp、seen-new−35.64pp、H−31.84pp、遗忘+12.96pp；125/125行的seen-new和H全部更差。
- ADV3B02-r8的M_DA相对M0为：A-old+0.0378pp、seen-new−0.0460pp、H−0.0272pp、floor−0.0267pp、遗忘+0.0222pp。
- ADV3B02注册后`M_OTHER=M0`覆盖375/375个slice和157,500/157,500条query，`M_JOINT=M_DA`同样全覆盖；`I_syn=0`为375/375。因此原BCRR不能作为下一轮无需验证的OTHER。

## 3.开发单元和held证据

|方法|证据面|B-old|A-old|Min-old或floor|seen-new|H|遗忘|关键裁决|
|---|---|---:|---:|---:|---:|---:|---:|---|
|D62开发单元|15个outer性能row|92.78|82.22|Min-old 53.33|84.67|82.62|10.56|当时最强开发点，后续125未保持该绝对水平|
|D91|15个outer性能row；105个候选训练row|92.78|82.22|Min-old 53.33；joint floor 26.67|84.67|82.62|10.56|15/15预测与D62相同，无独立125|
|SCXMAP M0|Phase1-held54|84.69|82.21|floor 58.64|82.31|80.82|2.48|仅held proxy|
|SCXMAP M_DA|Phase1-held54|84.64|82.18|floor 58.64|82.28|80.79|2.46|old/new/H均下降，淘汰|

D91虽然名称和support优化过程不同，但最终15个outer预测哈希与D62完全相同，不能把其82.62% H解释成D62之外的新泛化结果。D91资源为2,159个适配参数、20epoch、40 optimizer step、14,399B持久态、6,624 MAC/query和约25.427B adaptation MAC；它只证明跨折方向共识把D87残差缩小到几乎不改变决策。

SCXMAP的48/54行拟合出非零beta，17,580/19,782条query margin改变，但K5/K10的argmax变化为0；K1仅8次argmax变化，其中0次纠错、7次破坏。它证明“domain→identity低秩残差能改变分数”不等于“能产生有益邻居或决策变化”。

## 4.主要自研路线的机制结论

|路线|代表方法|已验证结论|后续边界|
|---|---|---|---|
|匿名类行与head安全门|D62、D63、D69、D70|support Pareto或jackknife安全不能预测outer old/new联合安全；hard gate导致大面积fallback|不继续扫描门限、角色mask或场景mask|
|局部pair/head重构|D64及其后续|提高某些旧类floor会显著损失A-old、new和H|不得用局部最弱类修补替代全类联合评价|
|共享metric与不可逆投影|D73、D74、D75|可逆metric会被后续LDA吸收；盲删rank-1 nuisance伤害new；安全门最终0接纳|下一DA必须改变真实网络决策几何，并审计邻居/argmax|
|ground切向head更新|D77、D78、D79|可获得旧类增益，但持续以新类和min-new下降为代价|ground只能定义类无关低维先验，不能直接偏置旧类logit|
|ground稳健中心|D81|开发单元小幅联合提升；完整125绝对性能低，K1严格identity|仅作为ground确实被读取的历史基线|
|任务均衡协方差|D92|大new-count时稳定减遗忘和旧类floor，但损害new，K1无效|保留“全类尺度平衡”教训，删除角色0.5/0.5模板|
|ground→target全坐标transport|D93/D94|coverage不足时出现全面负迁移|不提高rank或变换强度，不重入全坐标搬运|
|方差重整和回滚|SVRN-qKNN-BCRR|完整125全面劣于D62|关闭该实例|
|domain→identity cross-map|SCXMAP|大量margin变化、几乎无有益决策变化|关闭该实例|
|rank-4 joint-projection更新|GRB-JP4-CFM|本地release审查P0=0、P1=7；K10联合臂284,775B超state门，LOO MAC、正式runner和D92闭包不完整|`NO_PERFORMANCE_RESULT`，不得直接发布|

## 5.资源对比

|方法|训练/适配参数|epoch/step|持久态|query MAC或延迟|support适配成本|说明|
|---|---:|---:|---:|---:|---:|---|
|D62|2,016|20/20|8,583–18,503B|6,624–15,264 MAC；0.00857ms均值|0.107B–42.152B MAC；26.056B均值|部署轻，但support闭式拟合很重|
|D91|2,159|20epoch/40step|14,399B|6,624 MAC|25.427B MAC|只有开发15行，预测等于D62|
|SVRN-r4.2|0|0/0|≤256KiB|评分矩阵延迟0.07868ms均值|冻结闭式状态构建|性能显著阴性|
|ADV3B02-r8|0|0/0|max 206,394B|总延迟含fit/score，不能当纯query latency|无optimizer路径|资源通过，DA/OTHER/协同失活|
|SCXMAP|0|闭式beta|4,907B smoke|1,296 MAC/query|6,480–64,800 MAC/held行|资源小但决策净伤害|
|GRB-JP4-CFM|4维update|闭式/LOO|M_DA92 284,775B|设计目标≤262,144 MAC/query|严格LOO漏算|release门未闭合|
|D102解析初始化器|0|闭式4维solve|numeric 7,248B；bundle目录12,691B|未获Target授权|真实source-held 8,400行|资源通过；TX泄漏和LOCO门拒绝|

## 6.D102真实Phase1-held结果

`D102-RB-MetaBias4-qKNN`的解析SVD初始化实例已完成真实checkpoint、8,400条source weak-IQ的Phase1-held诊断。该run为`ARTIFACTS_COMPLETE / PHASE1_HELD_FALSIFIER_REJECT / TARGET25_BLOCKED`，不是Target性能。

|K|mean base BA|mean adapted BA|ΔBA(pp)|mean base floor|mean adapted floor|Δfloor(pp)|净纠正|BA退化receiver|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|86.2383%|86.2974%|+0.0591|67.3244%|67.3958%|+0.0714|+5|1/7|
|5|85.2488%|85.2846%|+0.0358|60.8350%|61.0500%|+0.2150|+3|0/7|
|10|85.1000%|85.1540%|+0.0540|62.2650%|62.7312%|+0.4662|+4|1/7|

结果证明pre-ReLU MetaBias确实改变决策：K1/K5/K10分别有9/3/17条argmax变化和467/382/381个ReLU mask flip，平均BA、floor及净纠正均为正。但联合门仍失败：

- TX泄漏probe mean BA=35.1190%，max BA=50.3199%，显著高于25%上限；
- class-LOCO 42个K1 fold中9个BA退化，且9个fold的net correct均为负；
- receiver-held K1和K10各有1/7个receiver退化，不能用边际均值覆盖；
- 当前Phase1只是解析初始化器，不是设计冻结的episodic gradient trainer；
- bundle虽仅7,248B numeric state、28个bank cell，但`formal_phase2_eligible=false`。

因此D102解析实例关闭，固定Target25不运行。下一轮不得只缩小trust radius或放宽门限；应实现带receiver-held/class-LOCO监督和显式TX不变约束的Phase1训练版本，并重新从source-held门开始。D62/D92仍只作历史或非门控matched诊断，BCRR不重新进入。

## 7.D62—D92连续研发链的同口径对比

这一段必须把“开发15个outer row”和“完整125个job”分开。D62和D91在开发单元上数值相同，不表示D91已经复现D62的125稳定性；D62从开发单元扩到完整125后，A-old由82.22%降至64.39%，Min-old由53.33%降至35.15%，说明单receiver、单seed、K10/new5开发点明显高估了跨receiver、跨seed、跨K稳定性。

### 7.1 D62—D81开发单元

|方法|主要变化|证据面|B-old|A-old|Min-old|seen-new|H|遗忘|相对D62裁决|
|---|---|---|---:|---:|---:|---:|---:|---:|---|
|D62|cross-fitted Fisher匿名类行Pareto拼接|15个outer row|92.78|82.22|53.33|84.67|82.62|10.56|开发比较基线|
|D63|再加全部leave-one折稳定门|15个outer row|93.33|82.78|53.33|82.00|81.65|10.56|旧类微增，但new−2.67pp、H−0.97pp|
|D64|全pair局部3-block LDA tournament|15个outer row|92.78|74.44|60.00|77.33|75.39|18.33|floor局部提高，但A/new/H全面下降|
|D65|冻结Stage2-B旧行、仅追加新行|15个outer row|92.22|86.11|70.00|59.33|67.12|6.11|旧类保护最强之一，但new−25.33pp|
|D66|ground域可靠性共享尺度|15个outer row|93.33|83.33|53.33|83.33|82.59|10.00|A+1.11pp但new−1.33pp、min-new−6.67pp|
|D67|D62/D65连续行堆叠|15个outer row|92.78|82.78|53.33|83.33|82.16|10.00|A+0.56pp但new−1.33pp|
|D68|跨折有向冻结registry标定|15个outer row|58.89|51.67|43.33|14.00|18.66|7.22|跨类绝对尺度被破坏，灾难性失败|
|D69|冻结D62旧行、追加D62新行|15个outer row|92.78|81.67|53.33|74.67|77.39|11.11|new−10.00pp，直接拼接关闭|
|D70|原子安全的少量旧行替换|15个outer row|92.78|82.22|53.33|84.67|82.62|10.56|汇总等于D62，仅1/15折接受一行|
|D71|cross-fitted top-2双类重排|15个outer row|91.11|82.22|53.33|84.00|82.33|8.89|遗忘下降来自B先下降，不是A改善|
|D72|physical-rank leave-one完整头bagging|15个outer row|93.33|82.78|53.33|82.67|81.59|10.56|A+0.56pp但new−2.00pp|
|D73|冲突投影对角联合metric|15个outer row|92.78|82.22|53.33|84.67|82.62|10.56|15/15预测等于D62，可逆metric被refit吸收|
|D74|非可逆rank-1 nuisance删除|15个outer row|—|D62−1.67pp|—|D62−5.33pp|D62−3.81pp|D62+1.67pp|盲删nuisance方向伤害新类|
|D75|D74加nested support-held安全门|15个outer row|92.78|82.22|53.33|84.67|82.62|10.56|0/15接受，精确回退D62|
|D77|ground预条件全类共同下降|15个outer row|92.78|82.22|53.33|84.67|82.62|10.56|15/15预测等于D62|
|D78|ground切向最差类margin|15个outer row|92.78|84.44|63.33|82.00|82.14|8.33|A+2.22pp、Min-old+10pp，但new−2.67pp、min-new−10pp|
|D79|中心化ground切向旋转|15个outer row|92.78|84.44|—|82.67|—|8.33|仍以new−2.00pp换旧类收益|
|D80|ground跨域漂移协方差去噪|15个outer row|—|D62+0.56pp|—|D62−0.67pp|D62−0.18pp|D62+0.56pp|旧/新交换，关闭协方差直注路线|
|D81|ground扰动谱Cauchy稳健中心|15个outer row|92.78|82.78|53.33|84.67|82.94|10.00|A+0.56pp、H+0.31pp，无可见交换；进入确认|

其中D65、D78和D79展示了最明显的旧类保护信号，但都通过牺牲seen-new获得；D81是这一段唯一在开发单元通过严格无交换门的方法。完整125随后证明D81的绝对性能仍远低于目标。

### 7.2 D82—D91开发单元

|方法|主要变化|B-old|A-old|Min-old|seen-new|H|遗忘|joint floor|预测变化与裁决|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|D82|D81中心＋Wiener残差收缩|83.33|72.22|56.67|75.33|72.76|11.11|23.33|15/15相对D81变化但系统性负迁移|
|D83|ground nuisance precision loading|92.78|82.78|53.33|84.67|82.94|10.00|26.67|15/15预测等于D81；连续头变化未改argmax|
|D84|跨类一致域模板稳健中心|92.78|82.78|53.33|84.67|82.94|10.00|26.67|15/15预测等于D81，适配MAC下降80.4%|
|D85|半径校准共识中心|92.78|82.78|53.33|84.67|82.94|10.00|26.67|性能等价D81，作为后续压缩基线|
|D86|p90半径反事实中心|92.78|82.78|53.33|84.67|82.94|10.00|26.67|15/15预测不变，且出现量化不稳定|
|D87|半径sigma-point margin head|92.78|85.00|60.00|83.33|83.58|7.78|30.00|A+2.22pp、遗忘−2.22pp，但new−1.33pp|
|D88|逐类Pareto保护D87残差|92.78|82.22|53.33|84.67|82.62|10.56|26.67|撤销旧类收益并退回D62水平|
|D89|v2半径可靠度Cauchy中心|92.78|82.78|53.33|84.67|82.94|10.00|26.67|15/15等于D81，状态由34,011B降至14,399B|
|D90|v2逐方向Cauchy中心|92.78|82.78|53.33|84.67|82.94|10.00|26.67|机制激活但0/15决策变化|
|D91|跨折共识sigma margin|92.78|82.22|53.33|84.67|82.62|10.56|26.67|15/15预测和全部指标精确等于D62；无独立125|

D91最容易被误读。它完成的是105个训练候选row和15个outer性能row，不是125个正式job；其最终15个prediction hash与D62逐行相同。因此，D91既不能继承D62的完整125证据，也不能把开发单元H=82.62%写成D91在跨receiver、跨seed、K1/K5/K10上的稳定性能。

### 7.3 D93—D103与完整125的关系

|方法|证据面|关键结果|裁决|
|---|---|---|---|
|D93 ground→target低秩interaction transport|单receiver/seed，K1/new20与K10/new20|K1 B/A/Min/N/H/F=`55.56/33.33/8.33/28.17/30.53/22.22`；K10=`83.61/61.11/43.33/66.08/63.50/22.50`|相对matched D81两种K均负，不跑125|
|D94 coverage-shrink transport|同上|K1=`56.39/33.33/8.33/28.17/30.53/23.06`；K10=`82.50/61.67/46.67/65.33/63.45/20.83`|收缩D93更新仍未恢复D81，不跑125|
|D95 D81-base coverage residual|K1窄诊断；K10技术失败|K1与D94逐值相同；K10在query前D43非正定，无性能|不把技术失败或K1负结果升级为125|
|D96 RA-CGSRDA|Phase1几何LODO|rank4、最差折解释率7.353%，`target_admission_authorized=false`|只有几何诊断，无Target性能|
|D97 QK-D81-LGF|Phase1/runtime闭包|batch8/256 parity超过`1e-5`，未开放正式Target|无Target性能|
|D98 STRIMS|本地研究core|仅实现/监督证据|无Target性能|
|D99—D101|Phase1 LODO、bundle与后续模型DA设计链|存在局部实现、技术失败或非正式诊断，但未形成可与D62/D92同口径的完整Target125结果|不进入性能排名|
|D102 RB-MetaBias4-qKNN解析实例|真实Phase1-held，非Target|K1/K5/K10平均ΔBA=`+0.0591/+0.0358/+0.0540pp`；TX probe与class-LOCO门失败|Target25阻断，解析实例关闭|
|D103-R2 RXID-CROSSRECEIVER-MB4|Phase1-held正式release，本地验证完成|source-only 8400→588/5292/2520；246fit、98,400step；计划63性能行和49稳定性行；67项定向测试、36文件编译、真实tap/dual 400step无query-truth smoke通过；独立复审P0/P1/P2=`0/0/0`|N607内核驱动535.309.01与用户态NVML580.173.02不匹配，尚未sync或启动；`N607_GPU_STACK_BLOCKED / TARGET25_NO_GO / NO_PERFORMANCE_RESULT`|

## 8.最终排名、答案与下一步

### 8.1只按完整125排名

|顺位|方法|B-old|A-old|Min-old|seen-new|H|遗忘|结论|
|---:|---|---:|---:|---:|---:|---:|---:|---|
|1|D92|81.55|65.56|36.81|58.93|61.57|15.99|合法完整125联合最强，但仍全面低于目标且损害new|
|2|D81|81.55|64.40|35.20|59.11|61.09|17.15|ground确实进入适配，但覆盖极低、K1无效|
|3|D62|81.51|64.39|35.15|59.11|61.09|17.11|与D81统计上近似，351/375状态fallback或零接纳|
|4|SVRN-qKNN-BCRR/r4.2|73.10|43.03|11.21|23.46|29.25|30.07|125/125行seen-new和H均低于D62|

ADV3B02-r8的M0/M_DA完整125也低于前三名，且M_DA相对M0的H为−0.027pp；它保留为外部/基础对照，不作为当前自研最优方法。

### 8.1.1 K5衰减和K1要求

K5表中的数值是matched K10/new20减K5/new20，单位为pp；当前要求四项均≤5pp。负值表示K5反而更高，但不能抵消其他指标或绝对性能失败。

|方法|A-old衰减|Min-old衰减|seen-new衰减|H衰减|K5裁决|
|---|---:|---:|---:|---:|---|
|D62|7.29|7.07|9.50|8.54|FAIL|
|D81|7.31|7.27|9.51|8.56|FAIL|
|D92|7.62|9.47|9.27|8.60|FAIL|
|SVRN-qKNN-BCRR/r4.2|1.19|−1.67|1.88|2.25|仅衰减数值通过；绝对A-old=41.59%、Min-old=10.80%、seen-new=15.42%、H=22.15%，整体拒绝|

K1/new20同口径结果：

|方法|A-old|Min-old|seen-new|H|旧类注册内变化|K1机制裁决|
|---|---:|---:|---:|---:|---:|---|
|D62|44.03|14.20|27.15|33.41|−24.11pp|75/75场景状态精确fallback，无正收益|
|D81|44.03|14.20|27.15|33.41|−24.11pp|历史基线，不能满足K1正收益|
|D92|44.03|14.20|27.15|33.41|−24.11pp|与D81逐值一致，K1协方差头严格identity|
|SVRN-qKNN-BCRR/r4.2|32.41|8.93|14.67|20.07|−33.67pp|大幅阴性|
|ADV3B02-r8 M_DA对M0|同M0|同M0|同M0|同M0|DA增益0|K1为0/48,000个预测变化，严格identity|

因此，SVRN“低K5衰减”只是低基线下的稳定低性能；D62/D81/D92又同时违反K5和K1门。当前完整125中没有方法满足活动目标。

### 8.2结论

1. 当前没有任何方法达到活动性能目标。D92只是“现有合法完整125中相对最好”，不是成功版本。
2. D62的Fisher row splice没有跨矩阵稳定生效。其完整125中仅24/375个scene状态在注册后真正激活，K1为75/75精确fallback。
3. D91不是D62的升级版。它的开发单元结果与D62逐预测相同，且没有完整125，因此不能与D92、D81、D62、SVRN放在同一稳定性排名中。
4. SVRN-qKNN-BCRR的125已经完成，不应重复运行。其修复了技术零向量问题，但性能相对D62是明确且成规模的负结果。
5. 历史最可靠的机制信号只有两类：D92的任务均衡协方差能提高注册后旧类和floor；D81/D89的ground可靠度中心能以极低覆盖率产生少量无交换纠错。前者伤害new，后者覆盖不足，二者都不能直接作为已成功方案合并。
6. 当前下一候选已经冻结为D103-R2。它必须先在63个Phase1-held性能行和49个K1稳定性行上证明TX不变性、receiver/class外推、实际160维shift与K1正收益，才可能开放固定Target25；目前N607 GPU栈阻塞，尚无任何D103性能值。
7. D103-R2即使通过held门，也只获得`Target25_GATE_ELIGIBLE`，不等于性能晋级；只有后续Target25同row联合通过，才值得重新消耗完整125。

### 8.3主要证据入口

|证据|路径|
|---|---|
|D62完整125|`automation_reports/CV-SincNet/d62_comprehensive_125_20260720/report.md`|
|D81完整125|`automation_reports/CV-SincNet/d81_comprehensive_125_20260720/report.md`|
|D91开发结果|`automation_reports/CV-SincNet/d91_crossfit_consensus_sigma_margin_20260720/d91_full_performance_summary.json`|
|D92完整125|`automation_reports/CV-SincNet/d92_registration_balanced_125_20260720/report.md`|
|SVRN完整125|`automation_reports/CV-SincNet/svrn_qknn_bcrr_125_r4_retry2_20260724/report.md`|
|D93—D101研发链|`automation_reports/CV-SincNet/ground_prototype_da_research_20260720/report.md`|
|D102真实Phase1-held|`automation_reports/CV-SincNet/d102_rb_metabias4_phase1held_target25_20260724/report.md`|
|D103-R2当前状态|`automation_reports/CV-SincNet/d103_r2_rxid_phase1held_20260725_r1/report.md`|

本次对比更新修改本文件和D103-R2版本化预注册交接，并对根目录正式报告的一处过时状态文字作同步校正；未访问N607、未启动或重复运行实验。Markdown差异已通过`git diff --check`，D62、D81、D91、D92和SVRN的主指标均从上述现有完成产物复核。
