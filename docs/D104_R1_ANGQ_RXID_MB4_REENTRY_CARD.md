# D104-R1-ANGQ-RXID-MB4重入卡

状态：`DESIGN_FROZEN / IMPLEMENTING_LOCAL_ONLY / N607_NO_GO / TARGET25_NO_GO`

日期：2026-07-25

## 1.重入原因

`D103-R2-RXID-CROSSRECEIVER-MB4`在7个development-only真实receiver outer×K1/K5/K10无真值复核中21/21行均实际激活，但K10的`1-1`和`2-1`分别只有298/300、309/310的INT8/FP32 top1一致，合计3次teacher-winner翻转，必定触发其冻结技术门。在已观察的21行、特别是两个失败K10行范围内，分量交换把这3次翻转定位到支持向量方向量化；这不是对所有receiver/K/未来held根因的泛化证明。R2未登陆N607、未读取query标签、无性能结果，不得原地改动。

新candidate为`D104-R1-ANGQ-RXID-MB4`。D104只替换typed qKNN的逐支持向量量化尺度选择；D103-R2的Phase1跨receiver教师、TX零空间、MMD、自监督、MetaBias4闭式系数、全类统一Student-t评分、246fit、资源门和query隔离全部保持不变。

## 2.冻结的ANGQ公式

正式runtime对合法support批量执行一次`normalize64→32`，随后每个候选不得再次归一化输入；只对候选解码向量执行归一化。对每个160维单位向量\(x\)，固定候选集合：

\[
C=\{0.75+0.005j\mid j=0,\ldots,100\}.
\]

对每个\(c\in C\)：

\[
s_{16,c}=\operatorname{float16}\left(\max\left(c\frac{\max_i|x_i|}{127},\operatorname{tiny}_{16}\right)\right),
\]
\[
q_c=\operatorname{clip}\left(\operatorname{rint}_{\mathrm{even}}\left(\operatorname{float32}(x)/\operatorname{float32}(s_{16,c})\right),-127,127\right),
\]
\[
\hat{x}_c=\operatorname{normalize}_{64\rightarrow32}\left(\operatorname{float32}(q_c)\operatorname{float32}(s_{16,c})\right).
\]

`normalize64→32`沿用现有实现：以float64计算L2范数，再输出float32单位向量。余弦以两个float32单位向量提升到float64后点积。按\(c\)升序枚举，选择最大余弦；完全同分保留首个。任何输入/scale/重构非有限、scale下溢、code=`-128`或重构零范数均fail closed。`c=1.0`由完全相同dtype和舍入顺序逐位复现原量化器，因此support自身重构余弦不低于原量化器。部署只保存现有形状的`q_c:int8[160]`和`s_c:float16`；类带宽继续从量化后support按现有统一公式计算并保存为FP16。禁止按receiver、K、场景、类别、old/new角色或query改变集合、步长、tie-break和目标。

ANGQ选择只读单条support向量，不读query特征、query真值、batch类计数、角色、clean/source成员样本或地面成员级状态；不产生optimizer step，不更新Phase1资产。builder API不提供query参数，receipt固定记录`query_features_used_for_scale=0`、`query_truth_read=false`、`query_state_updates=0`。query仍逐条独立对全部注册类使用同一Student-t公式。

正式端到端量化门以“FP32 support向量+由FP32 support计算的FP32类带宽”为teacher，以“ANGQ向量+由ANGQ解码support计算并保存的FP16类带宽”为deployed。另发布方向归因审计：FP32与ANGQ向量共用ANGQ部署带宽；该审计只解释误差，不参与晋级。K1继续使用Phase1锁定共享`h0`，K5/K10从各自解码后support按同一类对称公式重算带宽；这些语义进入method lock。

## 3.四臂归因

source-held与后续Target25均固定同row四臂：

|臂|表示/域适应|qKNN支持量化|用途|
|---|---|---|---|
|M0|基础表示，无D103位移|原单尺度INT8|matched base|
|M_DA|D103跨receiver MetaBias4|原单尺度INT8|冻结R2诊断，不具晋级资格|
|M_HEAD|基础表示，无D103位移|ANGQ|分类头主效应|
|M_JOINT|D103跨receiver MetaBias4|ANGQ|唯一联合候选|

四臂共享checkpoint、split、support/query物理ID、注册表、K、seed和评分器。M_DA即使技术门失败也只保留诊断预测，不允许按行回退、删除或冒充晋级臂。

每个matched row预注册并报告：

- `H0=M_HEAD−M0`、`H1=M_JOINT−M_DA`；
- `D0=M_DA−M0`、`D1=M_JOINT−M_HEAD`；
- head主效应`(H0+H1)/2`；
- DA主效应`(D0+D1)/2`；
- 交互项`M_JOINT−M_HEAD−M_DA+M0`。

M_DA不具晋级资格不等于从主效应与交互分析中删除。晋级仍以`M_HEAD`相对M0和`M_JOINT`相对M_HEAD的两个预注册简单效应为直接安全门；只有四个简单效应和两个平均主效应均完整报告，才能使用“2×2factorial”表述。

## 4.新held证据与数据边界

旧development probe已读取旧source-val的query特征和预测但从未读取其标签。D104不得用这些query物理ID做接受证据。旧query排除清单固定count=2478；活动canonical list root SHA256=`7870604d8ddba8268ba127065d4eaf1142931660d95411c9633c2ffa59d6b558`。规范编码为对排序ID列表执行`json.dumps(list,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)`后取UTF-8 SHA256，不含尾换行。旧值`036456779eea6594f2330f2e9a96cceda580088b0d451982198e3056f762854d`标记为`WITHDRAWN_UNREPRODUCIBLE_LEGACY_ROOT`，不得参与builder或gate。活动自描述manifest为`d104_historical_query_exclusion_manifest_v2_20260725_r2.json`，文件SHA256=`cf7cad3f6e62a300241a7ad1871e509717b0cafaef7400f46067b72eee31a31f`。新source split固定`split_id=d104_source_seed104713_v2`；为保持预注册成员选择不因身份修复而改变，salt仍为`D104-R1-ANGQ-RXID-MB4|source-split|104713|v1`：

1.每个receiver×TX×day cell排除旧query清单后，按`SHA256(salt|held|receiver|TX|day|physical_id)`升序取前15条进入新source-val；168×15=2520。
2.剩余5880行构成新训练池；历史诊断暴露ID允许进入L_s/U_s并标记`historical_diagnostic_exposed=true`，但永久禁止再次作为独立held证据。
3.每个receiver×TX组先按day各取2条L_s，再按`0.07×cell_size-current_count`最大缺口补到14；平手按day字典序，day cap=4；组内按`SHA256(salt|L|receiver|TX|day|physical_id)`排序。
4.其余全部进入U_s；必须得到L=588/U=5292/source-val=2520、4day L计数2–4、任一leave-day L=10–12、互斥和union闭合，否则拒绝。

只读容量审计已确认：排除2478个旧query后有5922个候选，168个cell的候选min/max=22/46；每cell取15在数量上可构造。该结果只证明容量，不是split builder正确性或held性能证据。

manifest必须同时固定2478个排序query ID、42个排序support ID、support/query交集0、tap/dual SHA、旧source split schema与count、7个receiver、6类、K=1、每receiver package root、派生代码路径和SHA。builder实际读取ID集合并复算root；缺失、重复、增删单ID、错误root、输入SHA或派生代码SHA漂移均fail closed。

本次物理ID角色变化按`AGENTS.md`触发一次builder验证；它不改变或重验Target的`p2_min_v1` capsule。D104公式、网格和四臂必须在新split ID及物理ID明细打开前进入Git并通过独立设计复审。

source-held继续执行49个outer、196个leave-day和1个final fit，共246fit/98,400step；63个matched row key与49个K1稳定性行不变。每个row必须同时封存4个命名臂，形成252个arm-row prediction单元；零缺失、零重复、禁止按臂或row回退。truth-side scorer首次打开标签前统一封存252个prediction SHA、method lock、split、support/query ID root、registry顺序和scorer输入。

## 5.冻结门

- 63个row逐行通过端到端ANGQ/FP32 top1一致≥99.5%且teacher-winner/teacher-runner-up量化margin≤0的计数为0；禁止用合并分母掩盖失败。top1采用冻结registry顺序的stable-first argmax；agreement分母为该row全部query。K1另保留support-only活动receipt，held-query审计不得反向改变状态；
- K1 rank=4、min singular value≥0.05、condition≤10、prior fraction≤0.80、coefficient norm≥1e-4、4个实际160维shift余弦中位数≥0.80；
- 每行`M_HEAD−M0`与`M_JOINT−M_HEAD`的BA、floor、net-correct均不得为负；正确数和net-correct用精确整数，BA/floor由整数分子分母重算，序列化浮点只允许`abs_tol=1e-12`；
- `joint_score=(balanced_accuracy+per_class_floor)/2`，不得从BA、floor或net-correct中事后选择标量；63-row mean joint_score要求`M_HEAD>M0`且`M_JOINT>M_HEAD`。net-correct只作为逐行精确整数非退化门；四个简单效应、两个平均主效应和交互均分别报告BA、per-class floor、joint_score和net-correct。该门命名为“逐行非劣+任意严格增益”，不声称已证明最小实际效果量；
- TX probe≤25%，全部覆盖、访问、序列化、资源和异常门通过；
- 资源按流式单候选实现：每候选160个norm MAC+160个cosine MAC，`adaptation_mac_per_support=101×320=32320`；总量公式为`32320×registered_class_count×K`。vector-elementwise口径只计每元素量化除法、round/clip、decode乘法和归一化除法，`adaptation_vector_elementwise_ops_per_support=101×640=64640`，总量为`64640×registered_class_count×K`。另逐support报告一次maxabs reduction和base-scale除法，逐候选报告factor乘法、FP16 cast、scale下限比较、norm sqrt、有限/零范数检查和best-candidate比较；这些scalar/reduction ops不混入640，必须作为分类计数receipt发布。`adaptation_mac_total≤1,939,200`与`adaptation_vector_elementwise_ops_total≤3,878,400`只对应C=6、K10实例；其他注册类数按公式重算。`peak_temporary_bytes≤16KiB`（不含调用方输入与最终bank）；
- `numeric_bank_array_bytes_before/after/delta`、`actual_serialized_state_bytes_before/after/delta`、`metadata_bytes_before/after/delta`与`query_mac_before/after/delta`逐row发布；ANGQ要求`numeric_bank_array_bytes_delta=0`、`query_mac_delta=0`，数值bank数组成员、shape和dtype不变；可变JSON quantization audit/header导致的metadata与actual serialized字节差必须单列，并继续通过既有总wire上限，不要求其delta为0；
- 任一门失败即拒绝整个D104，不扫描网格、阈值、角色mask、场景mask或选择性回退。

held接受只产生`TARGET25_GATE_ELIGIBLE`，不自动启动Target。Target25固定5receiver×1seed×5slice=25行，按用户目标同row报告K10/K5/K1的old-before、old-after、per-old-class floor、seen-new、H、forgetting和资源；不得从25行选择有利子集。

## 6.可行性摘要（冻结前，16行）

1.现有单尺度INT8的K10失败已在两个receiver精确复现。
2.两行合计3次翻转，未读取query标签。
3.FP32向量配部署FP16带宽均为100%一致。
4.现有INT8向量配teacher带宽复现全部翻转。
5.在两条已观察失败行中，翻转定位到支持向量方向量化而非类带宽。
6.固定101点角度网格只读单条support。
7.部署同构的`c=1.0`逐位复现原量化器。
8.绑定tap全池8400行逐向量量化性质检查为7575行严格改善、825行相同、0行退化；其中2478行为历史诊断query暴露行。
9.部署同构修订后两条失败行分别恢复300/300、310/310一致且0翻转。
10.8400行ANGQ重构余弦min/mean=0.999979969/0.999996080。
11.量化state数组dtype、shape和成员数不变。
12.query评分公式、全类竞争和逐条决策不变。
13.新增开销只在support适配期，query MAC不增加。
14.D103-R2的Phase1训练和MetaBias4求解不改变。
15.旧探针结果只能支持可行性，不能支持D104晋级。
16.独立复审和新held split完成前禁止实现正式release、N607和Target25。

## 7.进入实现的条件

第三轮独立复审对HEAD`3419ac20`的主体裁决为`P0=0/P1=0`，曾允许进入`DESIGN_FROZEN / IMPLEMENTING_LOCAL_ONLY`；N607与Target始终未授权。实现阶段发现旧query root无法从真实2478-ID集合按项目canonical编码复算，独立监督裁决`P0=0/P1=1/P2=1`，状态暂退为`DESIGN_REENTRY_REQUIRED / IMPLEMENTATION_PAUSED / N607_NO_GO / TARGET25_NO_GO`。本卡完成活动root、manifest和split ID身份修复后，必须对修订commit再次独立复审达到`P0=0/P1=0`，才恢复实现。ANGQ公式、四臂、资源门和性能门没有改变。

## 8.当前development-only证据

- 原8400行审计读取了绑定tap全池，其中包含2478条历史诊断query物理行；旧artifact的`query_features_used=0`声明已撤回。`local_d104_r1_support_geometry_audit_20260725_r6.json`中的root值已标记为不可复算旧值，该artifact不得再证明排除集合身份；其7575/825/0几何数值仍仅属development诊断。身份修复后的不覆盖artifact为`local_d104_r1_support_geometry_audit_20260725_r7.json`，SHA256=`93016ec162ab3b8edc16b3e24bb3d1fb5615895eb4621e70291e1ba6a5639f3e`，记录活动canonical root、规范编码和撤回旧值。它仍不是新held或正式性能证据。
- 两条既有失败K10行重放：`local_d104_r1_k10_deploy_isomorphic_angq_probe_20260725_r6.json`，SHA256=`560ae5b8cae3723153d54e654847a4454ca8b42e21664abde92e43a4d83df68c`；`1-1`的ANGQ端到端及共享ANGQ FP16带宽方向审计均为300/300、0翻转，`2-1`均为310/310、0翻转。
- 上述两条query特征已参与机制可行性诊断，因此不得作为D104接受证据。旧未部署同构的r4网格结果已撤回，不用于公式、保证、复审或晋级。
- 当前仍无BA、floor、H、旧类/新类准确率或Target证据；这些数值不能从重构余弦或量化一致性推断。
