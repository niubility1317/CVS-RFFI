# Stage2自研方法全面对比（2026-07-24）

## 1.比较口径

本表只纳入符合`项目.md`中`p2_min_v1`输入、查询隔离、逐样本全注册类竞争和声明边界的方法。性能必须来自同一候选、同一row绑定的注册前旧类、注册后旧类、seen-new、`H_old_new`、floor和遗忘；不同矩阵的数值只分层展示，不作直接冠军排序。

当前目标为：K10时注册后旧类≥92%、最低旧类≥85%、new5/new10/new20分别≥92%/90%/86%；K5相对matched K10/new20的注册后旧类、最低旧类、seen-new和H衰减均≤5pp；K1相对同row M0必须产生严格正收益，且旧/新保护项不恶化。

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

## 6.当前唯一研发候选

`D102-RB-MetaBias4-qKNN`当前为`DESIGN_DRAFT`，尚无性能数据。

1. Phase1以receiver-held episodic方式训练rank-4类无关receiver-bias basis`B∈R^(160×4)`和类无关domain meta bank。
2. 冻结backbone；在`joint_proj.0`的ReLU前执行`z_a=Norm(ReLU(u+Ba))`，使不同样本的ReLU mask和归一化几何发生输入相关变化。
3. 每个target-old/target-new support以同一公式产生4维meta-observation和precision。
4. Stage2-B对old类每类等权闭式求`a_B`；Stage2-C以完全相同loss、rank、solve和trust约束，对全部old/new类每类等权闭式求`a_C`。
5. 用同一`a_C`统一重编码全部support并重建一个typed INT8 Student-t qKNN；query只读、逐样本、全注册类竞争。
6. K1不估计类内散度；每个类singleton提供一个Phase1预定义的4维meta-observation，因此6/11/16/26个类票可约束4维系数。若Phase1-held不能证明可辨识和净纠错，则直接证伪。
7. 预计Phase2为0 trainable parameter、0 optimizer step、总state<80KB、support额外<4M MAC；`Ba`可合入偏置，query额外MAC可为0。
8. 必须先通过Phase1-held：K1严格正、K5/K10 old/new净正确均非负、非共同变换、INT8 top1≥99.5%、large-margin flip=0。
9. 通过后才运行固定`5receiver×seed713102×5slice=25`的DA-only矩阵；M0/M_DA为因果主臂，D62/D92只作非门控历史或matched诊断，不把已证伪BCRR包装成OTHER。

该候选针对三项已知失败逐一改变假设：用连续precision替代D62 hard fallback；用全类每类等权替代D92角色0.5/0.5；在真实网络ReLU前改变输入相关几何，而不是重复SCXMAP的post-feature标量修正。
