# D103-R2-RXID-CROSSRECEIVER-MB4重入卡

状态：`DESIGN_DRAFT / FEASIBILITY_REVIEW / N607_NO_GO / TARGET25_NO_GO`

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

新建`source_train`全源池LEO-weak缓存；固定seed103713复用项目`dataset_wisig._meta_ssl_partition_group`，按TX×receiver×day单元一次性切分0.07/0.63/0.30。`L_s`可含TX和pre_relu；`U_s`归档结构禁止TX和pre_relu；source-val真实数组只在`scorer_only`目录，fit仅得到`{row_count,content_sha256}`seal。禁止把历史`source_validation development only`归档改名为正式训练输入。

## 4.matched held比较

- source-val每个held receiver×class按physical ID升序；前K条为support，其余为query，M0、D102、D103逐行共用。
- M0为未适配z_id typed qKNN；D102仅以已封存且明确`PHASE1_HELD_FALSIFIER_REJECT`的数值bundle执行`infer/apply`诊断路径，不伪造promotion lock；D103使用对应outer fit的INT8 bundle。
- `joint_score=(balanced_accuracy+per_class_floor)/2`。D103每行BA、floor、net-correct均不得低于M0，且63行平均joint必须严格高于matched D102。
- 4个leave-one-day fit在同一K1 support上各自产生系数；其与outer-fit系数的方向余弦中位数必须≥0.80。INT8/FP32 held-query top1 agreement≥99.5%，large-margin flip=0。
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
11.D102 comparator已有精确数值bundle，可作非晋级matched诊断。
12.Stage2保持0 optimizer step、query fit rows=0。
13.U_s类型仍无TX/pre_relu字段。
14.任何K1 inactive、day cosine、TX probe、量化或资源失败均整实例reject。
15.当前仅允许独立设计复审与本地实现。
16.未获独立`P0=0/P1=0`前禁止N607与Target25。

## 6.冻结与发布条件

独立审查必须确认跨receiver episode不改变receiver-held因果边界、query4不是结果驱动选择、D102拒绝bundle仅作诊断且不冒充合法asset。审查通过后状态才可进入`DESIGN_FROZEN / IMPLEMENTING_LOCAL_ONLY`；完整真实checkpoint无正式query smoke、全部测试、Git commit和release复审通过前，仍为`N607_NO_GO / TARGET25_NO_GO`。
