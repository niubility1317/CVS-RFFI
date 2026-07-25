# D104-R1-ANGQ-RXID-MB4重入卡

状态：`DESIGN_DRAFT / FEASIBILITY_REVIEW / N607_NO_GO / TARGET25_NO_GO`

日期：2026-07-25

## 1.重入原因

`D103-R2-RXID-CROSSRECEIVER-MB4`在7个development-only真实receiver outer×K1/K5/K10无真值复核中21/21行均实际激活，但K10的`1-1`和`2-1`分别只有298/300、309/310的INT8/FP32 top1一致，合计3次teacher-winner翻转，必定触发其冻结技术门。分量诊断证明单尺度支持向量INT8编码是唯一根因，FP16类带宽不是根因。R2未登陆N607、未读取query标签、无性能结果，不得原地改动。

新candidate为`D104-R1-ANGQ-RXID-MB4`。D104只替换typed qKNN的逐支持向量量化尺度选择；D103-R2的Phase1跨receiver教师、TX零空间、MMD、自监督、MetaBias4闭式系数、全类统一Student-t评分、246fit、资源门和query隔离全部保持不变。

## 2.冻结的ANGQ公式

对每个由合法support产生的160维单位向量\(x\)，固定候选集合：

\[
C=\{0.75+0.005j\mid j=0,\ldots,100\}.
\]

对每个\(c\in C\)：

\[
s_c=c\frac{\max_i|x_i|}{127},\quad
q_c=\operatorname{clip}(\operatorname{round}(x/s_c),-127,127),\quad
\hat{x}_c=\frac{q_cs_c}{\|q_cs_c\|_2}.
\]

选择\(c^\star=\arg\max_{c\in C}x^\top\hat{x}_c\)，完全相同时取较小\(c\)。部署只保存现有形状的`q_c:int8[160]`和`s_c:float16`；类带宽继续从量化后support按现有统一公式计算并保存为FP16。集合包含`c=1.0`，因此support自身重构余弦不低于原量化器。禁止按receiver、K、场景、类别、old/new角色或query改变集合、步长、tie-break和目标。

ANGQ选择只读单条support向量，不读query特征、query真值、batch类计数、角色、clean/source成员样本或地面成员级状态；不产生optimizer step，不更新Phase1资产。query仍逐条独立对全部注册类使用同一Student-t公式。

## 3.四臂归因

source-held与后续Target25均固定同row四臂：

|臂|表示/域适应|qKNN支持量化|用途|
|---|---|---|---|
|M0|基础表示，无D103位移|原单尺度INT8|matched base|
|M_DA|D103跨receiver MetaBias4|原单尺度INT8|冻结R2诊断，不具晋级资格|
|M_HEAD|基础表示，无D103位移|ANGQ|分类头主效应|
|M_JOINT|D103跨receiver MetaBias4|ANGQ|唯一联合候选|

四臂共享checkpoint、split、support/query物理ID、注册表、K、seed和评分器。M_DA即使技术门失败也只保留诊断预测，不允许按行回退、删除或冒充晋级臂。

## 4.新held证据与数据边界

旧development probe已读取旧source-val的query特征和预测但从未读取其标签。D104不得用这些query物理ID做接受证据。新source split固定`split_id=d104_source_seed104713_v1`，仍精确为8400→588/5292/2520并满足42个receiver×TX组、4day、leave-day K10、互斥和union约束；新2520行source-val必须与D103开发探针全部query物理ID不相交。若现有8400行无法同时满足精确比例与不相交约束，停止并等待新source物理观测，不放宽条件。

本次物理ID角色变化按`AGENTS.md`触发一次builder验证；它不改变或重验Target的`p2_min_v1` capsule。D104公式、网格和四臂必须在新split ID及物理ID明细打开前进入Git并通过独立设计复审。

source-held继续执行49个outer、196个leave-day和1个final fit，共246fit/98,400step；63个性能行与49个K1稳定性行不变。truth-side scorer首次打开标签前必须封存四臂prediction manifest。

## 5.冻结门

- 全部K1/K5/K10行的ANGQ/FP32 top1一致≥99.5%，teacher-winner翻转=0；
- K1 rank=4、min singular value≥0.05、condition≤10、prior fraction≤0.80、coefficient norm≥1e-4、4个实际160维shift余弦中位数≥0.80；
- 每行`M_HEAD−M0`与`M_JOINT−M_HEAD`的BA、floor、net-correct均不得为负；
- 63行平均joint要求`M_HEAD>M0`且`M_JOINT>M_HEAD`；
- TX probe≤25%，全部覆盖、访问、序列化、资源和异常门通过；
- 量化网格的101×support向量×160维候选计算完整计入适配MAC和临时内存；持久state形状与query MAC不得增加；
- 任一门失败即拒绝整个D104，不扫描网格、阈值、角色mask、场景mask或选择性回退。

held接受只产生`TARGET25_GATE_ELIGIBLE`，不自动启动Target。Target25固定5receiver×1seed×5slice=25行，按用户目标同row报告K10/K5/K1的old-before、old-after、per-old-class floor、seen-new、H、forgetting和资源；不得从25行选择有利子集。

## 6.可行性摘要（冻结前，16行）

1.现有单尺度INT8的K10失败已在两个receiver精确复现。
2.两行合计3次翻转，未读取query标签。
3.FP32向量配部署FP16带宽均为100%一致。
4.现有INT8向量配teacher带宽复现全部翻转。
5.根因因此定位为支持向量方向量化，不是类带宽。
6.固定101点角度网格只读单条support。
7.网格包含原尺度，support重构余弦保证不降低。
8.两条失败行均恢复100%一致、0翻转。
9.重新计算FP16带宽后结果不变。
10.两行最差重构余弦分别为0.999988981和0.999988849。
11.量化state数组dtype、shape和成员数不变。
12.query评分公式、全类竞争和逐条决策不变。
13.新增开销只在support适配期，query MAC不增加。
14.D103-R2的Phase1训练和MetaBias4求解不改变。
15.旧探针结果只能支持可行性，不能支持D104晋级。
16.独立复审和新held split完成前禁止实现正式release、N607和Target25。

## 7.进入实现的条件

独立复审必须确认：ANGQ没有用query选择尺度；四臂归因不混淆DA与head；新held query与旧诊断query物理ID不相交；FP16尺度在归一化解码中的数值语义闭合；序列化和资源审计能证明state/query MAC不增加；D103-R2旧run命令永久撤回。只有`P0=0/P1=0`后才能从`DESIGN_DRAFT`进入`DESIGN_FROZEN / IMPLEMENTING_LOCAL_ONLY`。
