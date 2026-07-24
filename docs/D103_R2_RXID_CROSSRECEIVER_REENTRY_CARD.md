# D103-R2-RXID-CROSSRECEIVER-MB4重入卡

状态：`DESIGN_FROZEN_REV3 / IMPLEMENTING_LOCAL_ONLY / N607_NO_GO / TARGET25_NO_GO`

日期：2026-07-24

## 1.重入原因

`D103-R1-RXID-DUALSPLIT-MB4`在正式0.07/0.63/0.30切分的真实8400行特征smoke中，于第一个optimizer step前失败：`L_s`的receiver×TX计数为10–13，而R1错误实现要求同receiver内同时取得K10 support和16条query，即至少26条。该失败不产生性能结果，R1封口为`LOCAL_TECHNICAL_INFEASIBLE / NO_PERFORMANCE_RESULT / N607_NOT_RUN`。

R2只修复Phase1元任务的可达性，不改动TX零空间、MMD、receiver/day自监督、MetaBias4、INT8 ABI、Stage2闭式求解、typed qKNN、held覆盖、资源上限或Target门。

## 2.冻结的跨receiver元任务

- `query_per_class=4`；每step仍同时执行K1/K5/K10三个episode。
- 在outer训练允许的receiver有序表中，support receiver按step轮转，query receiver固定为下一个receiver；两者必须不同。
- 每类support只从support receiver取K条，每类query只从query receiver取4条；均按`SHA256(candidate|purpose|step|physical_id)`稳定排序。
- support/query物理ID不相交；元query仅用于Phase1训练loss，不是source-val、Target或正式query。
- receiver outer继续把真实outer-held receiver从`L_s/U_s`全部排除；receiver×class outer继续排除held receiver和held class，并按R1规则禁止`U_s`通过隐藏TX参与class-LOCO。
- singleton其余常量保持R1：Adam 1e-3、20×20=400step、K={1,5,10}、每cell取2、四项loss权重1、MMD gamma={0.5,1,2}、qKNN训练温度0.2。

## 3.正式source split与权限

新建`source_train`全源池LEO-weak缓存；固定seed103713沿用项目按TX×receiver×day分组、组内按物理ID排序后随机置换的语义，但禁止对168个小单元分别取整后接受比例漂移。builder执行冻结的全局最大缺口配额：

1.全局目标必须精确为`N×0.07/N×0.63/N×0.30`；N=8400时必须为588/5292/2520，否则拒绝。
2.42个receiver×TX组各自精确分配14条`L_s`，合计588；组内4天每day必须为2–4条。先每day预留2条，再按`0.07×cell_size-current_count`最大者补到14，平手按day字典序，且单day cap=4。由此删除任一天后仍至少保留10条K10 support。
3.每个TX×receiver×day单元同时预留`U_s≥1、source-val≥1`。
4.按`0.63×cell_size-current_count`最大者优先、TX/receiver/day字典序tie，把`U_s`补到全局5292；source-val取剩余2520。任何单元容量、42组×任一leave-day K10可达性、总量、互斥或union失败即拒绝。

`L_s`可含TX和pre_relu；`U_s`归档结构禁止TX和pre_relu；source-val真实数组只在`scorer_only`目录，fit仅得到`{row_count,content_sha256}`seal。禁止把历史`source_validation development only`归档改名为正式训练输入。上述配额修订发生在任何性能结果和正式归档前；其目的仅是服从固定比例并保持K10可达。

## 4.matched held比较

- source-val每个held receiver×class按physical ID升序；前K条为support，其余为query，M0、D102、D103逐行共用。
- M0为未适配z_id typed qKNN；D103使用对应outer fit的INT8 bundle。
- D102 comparator不得复用单一full-source bundle。held runner在打开source-val scorer数组前，只从同一`L_s`为7个receiver outer和42个receiver×class outer分别构建49个fold-specific解析bundle；D102与D103使用完全相同的held receiver/class排除面。每个D102 bundle必须封存content root、outer spec、`L_s`物理ID root、构建代码SHA和以下父证据：method lock SHA256=`9640267c2913e452a89be39e1b41e8b19d3371499afbed1efe8c9e3b7ad0e52f`、原拒绝receipt SHA256=`01a45e11fe519389071cf1eb279d293c958fc4fa48e0ed4c51bea9ff20c536b2`。生成这些诊断bundle不读取`U_s`、source-val、Target或正式query。
- D102 scorer只允许调用已审计的`infer_metabias4_coefficient/apply_metabias4`数值路径；不得生成、修改或伪造promotion lock。每行输出必须携带`DIAGNOSTIC_REJECTED_D102_COMPARATOR_NON_PROMOTABLE`、fold content root和原拒绝receipt SHA；D102原拒绝状态不改变。
- `joint_score=(balanced_accuracy+per_class_floor)/2`。D103每行BA、floor、net-correct均不得低于M0，且63行平均joint必须严格高于matched D102。
- 4个leave-one-day fit在同一K1 support上各自产生实际160维pre-ReLU位移`s_day=B_day a_day`，outer fit产生`s_outer=B_outer a_outer`。直接比较4维`a`被禁止；`cos(s_day,s_outer)`的中位数必须≥0.80，任一位移范数`<1e-4`均fail closed。INT8/FP32 held-query top1 agreement≥99.5%，large-margin flip=0。
- D102拒绝状态不改变；它只作为同row诊断比较器，不获得Target资格。

## 5.可行性摘要（冻结前，16行）

1.真实源池8400行，覆盖6TX×7receiver×4day全部168个单元。
2.固定split得到每个receiver×TX的`L_s`最少10、最多13。
3.K10 support需要10条，因此单receiver support可达。
4.跨receiver query固定4条，因此query可达。
5.support/query receiver不同，物理ID天然不交叠并仍显式复核。
6.R1的同receiver `K10+16`需求不可达，已用真实特征失败证明。
7.R2不读取source-val完成训练，不以性能调参。
8.R2不改变246fit与98,400step总量。
9.每fit计算量因query16→4只下降，不提高冻结资源上限。
10.GPUh上限仍30、显存4GiB、run root20GiB。
11.D102按49个outer spec从同一`L_s`预先构建fold-specific非晋级诊断bundle。
12.Stage2保持0 optimizer step、query fit rows=0。
13.U_s类型仍无TX/pre_relu字段。
14.任何K1 inactive、160维shift cosine、TX probe、量化或资源失败均整实例reject。
15.当前仅允许独立设计复审与本地实现。
16.未获独立`P0=0/P1=0`前禁止N607与Target25。

## 6.冻结与发布条件

独立审查必须确认跨receiver episode不改变receiver-held因果边界、query4不是结果驱动选择、160维shift余弦消除latent gauge、49个D102 fold-specific诊断bundle与D103具有相同排除面且不冒充合法asset。审查通过后状态才可进入`DESIGN_FROZEN / IMPLEMENTING_LOCAL_ONLY`；完整真实checkpoint无正式query smoke、全部测试、Git commit和release复审通过前，仍为`N607_NO_GO / TARGET25_NO_GO`。

commit-bound Revision复审针对`7136605f`得到`P0=0、P1=0 / GO: DESIGN_FROZEN→IMPLEMENTING_LOCAL_ONLY`。后续真实split smoke发现逐小单元取整不满足全局精确比例；Rev2审查又发现leave-one-day后K10未保证，因此本卡进入Rev3短复审。上一裁决不自动覆盖新增配额算法。未提交文件不属于已验证证据，也不授权N607或Target25。

Rev3 commit-bound复审针对`84e87b98`得到`P0=0、P1=0 / GO: 恢复DESIGN_FROZEN→IMPLEMENTING_LOCAL_ONLY`。实现必须新增精确588/5292/2520、cell下限、42组×4 leave-day K10、容量不足、互斥/union和确定性tie测试；未通过前不得宣称实现完成。
