# D106-RDCE/GTSM-r3研发与实验报告

状态：`DESIGN_FROZEN / DATA_LOCAL_CODE_G0_GO / DA_LOCAL_CODE_G0_GO / REAL_INTEGRATION_ENTRY_LOCAL_GO / CANONICAL_METHOD_RUNTIME_GIT_PENDING / REAL_INTEGRATION_NOT_EXECUTED / N607_READ_ONLY_DISCOVERY_ONLY / SOURCE_HELD_NOT_OPENED / TARGET25_NO_GO / NO_TARGET_PERFORMANCE_RESULT`

## 1.实验身份

- 实验ID：`d106_rdce_gtsm_20260801_r1`
- 日期：2026-08-01
- 主agent：`gpt-5.6-sol/high`
- 子agent：数据契约、DA方法、矩阵与HEAD分别由`gpt-5.6-terra/max`承担
- 目标：研发合法、K1非identity的共享低秩域适应，并与纯support-only qKNN头组成固定四臂

## 2.当前决策

D105在R8观察到receiver/class/TX source-held门拒绝且没有formal组件。FTU4已在本地commit`9e80849b`修复合法TX负结果的无wire持久化；这不改变D105科学资格。为集中功能研发，本轮不补跑D105 R9，不释放D105 Target25。

D106 DA冻结为`D106-RDCE/GTSM-r3-SCATTER02`。详细公式和证据边界见`analysis/d106_rdce_gtsm_design_freeze_20260801.md`。

## 3.假设与比较目标

假设：Phase1 `L_s`中跨TX一致的receiver-day类中心残差方向是可共享的身份空间nuisance子空间；用INT8低秩SPD度量连续衰减这些方向，可在K1保持非零作用，并由K≥2 support类内scatter小幅调制，而不访问query或ground exemplar。

固定四臂：

|臂|表示|头|
|---|---|---|
|`M0`|旧`z_id`|旧Student-t qKNN|
|`M_DA`|D106 RDCE|旧Student-t qKNN|
|`M_HEAD`|旧`z_id`|D106纯support-only头|
|`M_JOINT`|同一D106 RDCE state|同一D106头|

D62、D91、D92和SVRN只作matched外部基线分析，不替代`M_HEAD`，不污染2×2简单效应。

历史矩阵、逐slice指标、paired差值、资源和证据等级已统一整理到`analysis/d106_external_baseline_comparison_20260801.md`。其中D62、D92、SVRN已有完整125，不重复运行同一冻结revision；D91只有15个development row，不冒充125证据。

## 4.数据与协议

- `protocol_schema=p2_min_v1`
- Phase1 split：588/5292/2520，对应`L_s/U_s/source validation`
- `rho_label=0.1`
- `L_s`SHA256：`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`
- checkpoint SHA256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- source-held truth尚未用于性能计算
- Target capsule保持`VALIDATED_ONCE`，方法变化不得触发数据重验

禁止当前D105 8400行特征tap进入D106训练。builder-only存储验证器必须读取D104 SHA绑定的8400行上游`source_validation` received-IQ池并验证完整存储语义；正式D106 extract/export只允许精确选择的588个`L_s`physical ID及其day/scenario/observation绑定进入方法面，其余7812行不可见。

## 5.已完成的训练面探针

|机制|K1净正确|K5净正确|K10净正确|证据边界|
|---|---:|---:|---:|---|
|cross-cov公共平移|−3|0|0|`L_s`机械探针，拒绝倾向|
|RDCE-r3，`γ=0.20`|+4|+4|+2|`L_s`机械探针，冻结公式|

RDCE绝对正确数为490/513/511，旧表示为486/509/509，总数均为588。不得把这些数字写作source-held、Target性能或相对D62/D92的正式增益。

## 6.冻结公式

```text
rank = 3
a0(K) = min(0.95, 1.5*K/(K+4))
gamma = 0.2
K1: a[j] = 0.3
K>=2:
  e[j] = class-balanced support within-class scatter
  a[j] = clip(a0(K)+0.2*tanh(log((e[j]+1e-8)/(tau[j]+1e-8))),0.05,0.95)
M_S = I-B^T diag(a)B
```

`M_DA/M_JOINT`必须复用同一state SHA。query fit/update/selection均为0。

## 7.正式Target25矩阵

```text
5 receivers × seed713102 ×
{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}
=25 jobs
```

每job包含3个物理ID互斥LEO弱场景和4臂，共300个scenario-arm pair、600个before/after prediction surface。完整预测封存前不得打开truth；不得挑receiver、scene、class或partial。

## 8.性能门

- K10：`A_old≥92%`，`F_old≥85%`；new5/new10/new20的`N≥92%/90%/86%`
- K5/new20相对matched K10/new20：`A_old/F_old/N/H`下降均≤5pp
- K1/new20相对同rowD92：`ΔH≥2pp`、`ΔF_old≥2pp`、`ΔA_old≥0`、`ΔN≥0`，总正确数严格增加
- 四臂G1/G2按同row重算DA、HEAD简单效应和交互，不使用跨run边际极值

## 9.待实现文件面

|工作包|建议文件|状态|
|---|---|---|
|DATA|`stage2_d106_phase1_tap.py`、CLI及测试|588/588纯ID归属已闭合；真实scope/API/receipt修正完成，81通过、1跳过，独立复审`P0=0/P1=0/P2=0 / LOCAL DATA GO`|
|DA|`stage2_d106_rdce_asset.py`、`stage2_d106_rdce_runtime.py`及测试|本地代码20/20、独立复审`P0=0/P1=0/P2=0`；但跨模块release随DATA保持NO-GO|
|HEAD|候选文件待新revision冻结|`SG-LC-CL-OOF/r1`与`SBCM-BR/r2`均在训练面拒绝|
|四臂/held|`stage2_d106_four_arm.py`、source-held predictor/scorer|待HEAD冻结|
|Target25|基于D105骨架的新runner/launcher|G1前禁止实现release|

## 10.N607信息

2026-08-01由唯一Terra Max runner完成direct N607只读preflight、资产发现及588个L_s纯物理ID归属核验；bridge未使用。8张RTX3090两次均为0%/1MiB，所有短SSH后均确认本地`ssh.exe=0`且N607/bridge无`ESTABLISHED TCP22`。本轮未SCP、未创建远端目录、未运行项目脚本、未启动进程或占用GPU。详见§15。

## 11.风险与下一步

当前release硬门为：独立复审修正后的DATA契约，然后执行真实extract→export→validate。纯ID证据已证明D104的588个L_s来自SHA`125bb312…`的8400行`source_validation`池；现有split、物理ID、received-IQ bytes和场景均未变化，因此按`VALIDATED_ONCE`规则不重建split、不重验数据。D106仅修正历史误名字段的解释、公开API、scope验证和validator receipt。HEAD、四臂held/scorer和Target25 runner仍未实现。

`D106-SG-LC-CL-OOF-qKNN/r1`已完成三轮共136组`L_s` train-only预锁检查。没有任何配置同时满足K1/K5/K10总正确数与floor均非退化；最后的最小非零clearance方案仍在三个K各减少1个正确样本。因此该HEAD状态为`DESIGN_REVISION_REQUIRED / IMPLEMENTATION_FORBIDDEN`，不能因margin非零就写成有性能功能。

`D106-SBCM-BR-qKNN/r2`随后冻结为单一无grid lock：`rho=tr(B)/(2tr(B)+w)`、`eta=0.5`、归一化logit残差cap为`ln(4/3)`，无hard gate或fallback。主agent使用真实588条`L_s`、28个receiver-day held folds和同一INT8 Student-t bank完成机械探针；每个K均覆盖588条query。

|K|base正确|HEAD正确|净正确|argmax改变|distance改变覆盖率|最低类floor变化|high-margin flip|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|509|509|0|0|100%|0pp|0|
|5|503|503|0|0|100%|0pp|0|
|10|500|500|0|0|100%|0pp|0|

该结果证明共享metric改变了距离，但没有产生任何prediction功能，违反预登记的K1 argmax改变和三K总净正确严格增加门。`SBCM-BR/r2`记为`REJECT_REVISION_NO_FUNCTION / IMPLEMENTATION_FORBIDDEN`；不得通过调大`rho/eta/cap`继续扫描。HEAD转入机制不同的`ANVC-qKNN`单lock研究。

`D106-ANVC-qKNN/r1`冻结为单个15°全类邻域排斥virtual-core、全部类间距离中位温度、全部类切向范数中位尺度和每类一个等权virtual support，不改class bandwidth且无fallback。同一28-fold探针得到：

|K|base正确|ANVC正确|净正确|argmax改变|active class覆盖率|virtual贡献覆盖率|最低类floor变化|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|509|509|0|0|100%|100%|0pp|
|5|503|502|−1|1|100%|100%|0pp|
|10|500|500|0|0|100%|100%|0pp|

K1/K5/K10最小virtual偏转分别约为0.14075、0.13519和0.15036rad。K5唯一flip为correct→wrong，`20-19`类聚合少1个正确；K1仍无prediction改变。因此`ANVC/r1`同时违反K1功能门、逐K正确数非退化和三K总净正确严格增加门，记为`REJECT_REVISION_NO_FUNCTION / IMPLEMENTATION_FORBIDDEN`；不得扫描virtual角度、温度、尺度或权重。

`D106-PRT-BR-qKNN/r1`进一步使用support-only pair-specific temperature、共同temperature reference消除和零和有界残差。同一28-fold探针中，K1/K5/K10的pair-temperature state覆盖率均为100%，最小temperature spread分别约为0.00350、0.13943和0.14179；pair residual query覆盖率分别为94.898%、100%和100%。但三K的argmax改变均为0，正确数仍为509/503/500，floor逐值不变。它违反K1 residual覆盖、argmax功能和总净正确严格增加门，记为`REJECT_REVISION_NO_FUNCTION / IMPLEMENTATION_FORBIDDEN`；不得扫描temperature范围、`eta`或cap。

## 12.三轮HEAD探索回顾

2026-08-01在启动第四个HEAD候选前，主agent按探索回顾规则重新读取`项目.md`、活动目标、D62/D91/D92/SVRN及本报告，并重建项目对话索引；索引闭合1186条`E:\type10-7`相关记录。三轮指`SBCM-BR/r2`、`ANVC/r1`和`PRT-BR/r1`三个机制，不把首版32+72+32配置误写为136个数据group。

### 12.1已学到的机制事实

|路线|状态有无作用|decision有无作用|淘汰原因|
|---|---|---|---|
|SBCM共享PSD metric＋有界残差|三Kdistance覆盖100%|三Kargmax均0|只改变距离，未改变prediction|
|ANVC全类邻域virtual core|全类active且virtual贡献覆盖100%|K1/K10为0；K5仅1个错误flip|局部虚拟证据过弱，且唯一翻转方向错误|
|PRT pair-specific tournament|pair state三K全active|三Kargmax均0|有界pair残差不足以改变冻结Student-t排序|

共同根因不是K1完全不可辨识，而是三条路线都把“保护base排序”置于“形成新decision surface”之前：冻结Student-t在`d_eff=160`和原class bandwidth下产生很强的原始margin，4:3 odds cap或单个低质量virtual atom只能改变内部几何/分数，难以改变top1。这重复了D91“内部目标变化但outer prediction不变”和D62/D92“K1透传/逐值不变”的历史问题。

### 12.2协议与任务复核

- Phase2仍只读不可变Phase1 bundle、当前row合法support和固定接收IQ；query零fit、零selection、零update并独立面对全部注册类；
- 继续禁止clean/source运行时访问、query truth/role、真实类计数、quota、全局重排和按TX/class ID规则；
- 旧类适应与新类注册保持同等优先；后续四臂必须从同一row同时报告`B_old/A_old/N/H/F_old/forgetting`、逐类和negative tail，不能只看K1 train-only正确数；
- D106 DA未因HEAD三轮失败而降级：RDCE训练面已有K1/K5/K10净正确`+4/+4/+2`的独立机械证据，先完成其实现与G0验收；
- HEAD不得读取ground bank，不能把DA资产直接用于HEAD计分，也不能以D62、D91、D92或SVRN替代`M_HEAD`。

### 12.3第四轮决策

第四轮不再采用“base logits＋小幅有界残差”或“一个低权重virtual atom”。新的Terra Max研究必须提出一个直接替换ranking的纯support-only qKNN decision surface，K1自由度来自全类support图，全部query无条件active且无fallback。优先审查两类机制：support-simplex共享白化/等距化qKNN，以及按全部类对几何归一的直接max-min margin qKNN。候选必须先证明不是原排序的单调变换，再冻结一个无grid lock进入同一28-fold探针；失败后不得回调同一lock。

首个直接ranking候选`D106-SSW-EQ-qKNN/r1`使用`lambda0=tr(B)/(C−1)`的共享ZCA式support-simplex白化，对support和query同变换、重建INT8 bank并直接用共享`h0`的Student-t qKNN排序。它没有base residual、cap或fallback。同一28-fold结果为：

|K|base正确|SSW正确|净正确|argmax改变|wrong→correct/correct→wrong|最低类floor变化|最差fold|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|509|509|0|0|0/0|0pp|0|
|5|503|494|−9|12|1/10|0pp|−3|
|10|500|498|−2|8|2/4|+1.020pp|−2|

三K的state、distance和support-code改变覆盖率均为100%，最大condition number约2.84。该候选已形成新decision surface，但K1仍无功能，K5的`14-10`类减少10个正确，K10的`14-10/14-7`分别减少3/1，违反逐K、逐类和negative-tail门。记为`REJECT_REVISION_NEGATIVE / IMPLEMENTATION_FORBIDDEN`；不得缩小白化、恢复部分identity轴或改`h0`补丁。下一步只验证预先登记、机制独立的全类上下文pair-margin候选。

独立备选`D106-CAG-MM-qKNN/r1`使用其余全部类定义每个类对的上下文SPD几何，直接以最差pair affine margin排序；自由数值超参数为0，不读取base Student-t分数。teacher探针结果为：

|K|base正确|CAG正确|净正确|argmax改变|wrong→correct/correct→wrong|最低类floor变化|`20-19`类变化|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|509|509|0|0|0/0|0pp|0|
|5|503|508|+5|29|13/8|−1.020pp|−2|
|10|500|509|+9|32|13/4|−1.020pp|−1|

三K active state均为100%；三类cycle非零query覆盖率分别为89.796%、100%和100%。CAG已经明显改变K5/K10决策且三K合计净增14个正确，但K1仍无功能，K5/K10的最低类floor均下降，且K5单类损失超过预登记上限。因此仍记为`REJECT_REVISION_NEGATIVE / IMPLEMENTATION_FORBIDDEN`，不能只根据总体正确数挑选它，也不得修改ridge或增加Student-t混合补丁。

第三个直接机制`D106-SPR-CV-qKNN/r1`把同一固定received IQ的一次冻结前向中的`z_id=ReLU(pre_relu)`与signed`pre_relu`等权拼接为320维表征，重建INT8 bank并直接Student-t排序；它不增加K、不读取ground/source或query状态。真实探针中零范数行为0、负坐标active为100%、三K余弦变化覆盖率均为100%：

|K|base正确|SPR正确|净正确|argmax改变|wrong→correct/correct→wrong|最低类floor变化|主要逐类变化|
|---:|---:|---:|---:|---:|---:|---:|
|1|509|504|−5|11|2/7|−1.020pp|`14-7`−5，`20-19`+2|
|5|503|503|0|17|6/6|+1.020pp|`14-10`−2，`20-19`+2|
|10|500|504|+4|20|9/5|+2.041pp|`20-19`+2，其余小幅变化|

SPR证明同IQ的signed第二表征能够让K1产生真实prediction变化，也能改善K10总正确数与floor；但K1变化方向错误且单类损失严重，违反预登记门。记为`REJECT_REVISION_NEGATIVE / IMPLEMENTATION_FORBIDDEN`；不得扫描视图权重、`h0`或kernel。

## 13.第二次三轮HEAD回顾

第二组三轮为`SSW-EQ/r1`、`CAG-MM/r1`和`SPR-CV/r1`。主agent再次读取活动目标、`项目.md`、本报告和项目对话索引；source-held truth和Target仍未打开。

|事实|含义|
|---|---|
|SSW能直接换ranking但K5净−9|单纯等距化会压缩已有强判别轴|
|CAG在K5/K10净增+5/+9但floor各−1.02pp|总正确数改善会在拥挤类之间重新分配错误，不能替代floor|
|SPR在K1产生11个flip但净−5，K10净+4且floor+2.04pp|第二表征有信息，但shot越少越容易把负半轴噪声当身份|

本次不放宽已看到结果上的门，也不把CAG的总体增益或SPR的K10增益拼成“联合候选”。下一HEAD revision必须在任何新探针前预登记一种类置换对称的support可靠度/拥挤度机制，明确处理“整体纠错但最弱类下降”和“K1视图噪声”两个问题；不能按`14-7`或`20-19`写规则，不能给K1 identity/fallback，也不能扫描混合权重。该研究在DATA/DA可信实现闭环之后恢复。

并行的D106 DA实现已经由Terra Max子agent交付；主agent定向与协议负测9/9通过。真实588行`L_s`的asset/runtime no-query smoke也已通过：

|项目|结果|
|---|---:|
|输入行/类|588/6|
|部署wire总字节|2358B|
|解码basis Gram最大绝对误差|`1.22e−15`|
|解码spectrum|0.004943/0.004632/0.003075|
|解码tau|0.033177/0.039619/0.040907|
|K1 attenuation/min eig|0.300/0.7000|
|K5 attenuation/min eig|0.8085/0.7532/0.7259；0.1915|
|K10 attenuation/min eig|0.8961/0.9454/0.9065；0.05460|
|K1/K5/K10 query feature改变|22/22、22/22、22/22|
|每query估算MAC/状态更新|960/0|

这些数据只证明真实输入、量化重放、SPD和query只读功能闭合，不是source-held或Target性能。随后DATA↔DA交叉独立复审推翻了初版GO：

|复审面|结论|主要问题|
|---|---|---|
|DATA|`NO-GO / P0=1,P1=2,P2=1`|完整8400缓存路径在588 ID join前实体化U_s TX标签/IQ；实际构造代码闭包未绑定；关键安全测试被mock|
|DA|`NO-GO / P0=0,P1=7,P2=2`|D104精确计数未全锁；裸数组/SHA可伪绑定；raw INT8 Gram未fail-closed；support row未typed绑定；稠密query路径与960MAC收据不符；wire trust/canonical和原子保存不闭合|

因此此前20/20、9/9、组合29/29和真实smoke均降级为“可执行证据”，不能作为release readiness。DATA与DA原作者已在互不重叠的授权文件面修复；修复完成前不commit、不进入source-held、更不进入N607。

## 14.DATA与DA最终本地功能闭环

2026-08-01主agent对修复后的最终文件面串行验收。当前结论仅覆盖本地代码和合成/结构化夹具，不打开source-held truth或Target query truth。

### 14.1 DATA闭包

- 完整存储验证器逐场景验证3×8400行的role、scenario、view、applied、receiver/day、seed、`[8400,2,256]`有限IQ、每条IQ摘要和overlay provenance；
- 方法提取产物仅包含精确588条`L_s`，并与完整存储validator、selected IQ archive/receipt/content root、D104 split和checkpoint执行收据分层绑定；
- export入口不持有8400缓存、split、disjoint或salt能力；关键嵌套callable、同句柄读取、checkpoint/model/forward收据和completion marker均fail-closed；
- 真实authority修正后的独立终审结论为`P0=0/P1=0/P2=0 / LOCAL DATA GO`；
- 当前DATA、CLI、DA资产/runtime及真实集成入口联合测试：101通过、1跳过；跳过项为缺少环境变量`D106_REAL_INTEGRATION_FIXTURE`的既有真实资产闭环。

### 14.2 DA闭包

- 资产锁定D104精确588行、6TX、每TX98行、7×4 receiver-day、每cell 2–4行；INT8原始Gram在polar闭包前先拒绝非法输入，`tau`与spectrum从解码后的闭合basis重算；
- formal asset只能通过DATA loader、外部SHA绑定的tap archive/receipt构建；裸数学构造保持`NON_DEPLOYABLE_MATH_ONLY`；
- runtime row authority绑定capsule、split、validator、row、seed、K、注册类顺序、support数组收据、有序physical ID、qKNN bank及support/query互斥根；
- query变换使用rank3低秩路径，每行精确960 MAC；basis只在一次临时评分上下文中解码，context为私有loader-origin对象、构造后不可赋值，basis/attenuation使用immutable bytes-backed视图，并由独立弱引用铸造表绑定runtime、asset、basis内容和FP16 attenuation；
- DA资产与runtime测试：20/20通过，其中包含split handle逐字段漂移、有序physical ID漂移、active K、注册类顺序、qKNN bank、feature/label/row/seed漂移、canonical wire，以及“持有合法context后同时篡改basis与自签摘要”仍拒绝。
- 独立终审结论：`P0=0/P1=0/P2=0 / LOCAL G0 DA GO`。

### 14.3 联合与依赖回归

|验证面|结果|结论|
|---|---:|---|
|D106 DATA+CLI+DA及真实集成入口五个测试文件|101通过、1跳过|含portable D104路径、D106 runtime语义和两份模型代码SHA漂移负测；真实夹具仍是release硬门|
|D104 split、D105 tap、Student-t qKNN、VALIDATED_ONCE句柄|51/51通过|authority修正后当前依赖回归未破坏|
|Python编译与`git diff --check`|通过|无语法或空白错误|

本地放行生产文件SHA256：

|文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_d106_phase1_tap.py`|`5a63a5935748f17a1efcbf4069d5c80c1d99a8e813330a2c3a15895483c53e9b`|
|`code/cvsrffi/stage2_d106_rdce_asset.py`|`e9d57245a80cdf31ae4ea5fd76cd521022399d0be512e2647a92cc2a0671da1f`|
|`code/cvsrffi/stage2_d106_rdce_runtime.py`|`9d78b83134bfb668c3b9c32053eaa86b5c9fd4d970e87aa99dc30ac2df8df946`|
|`code/scripts/export_d106_phase1_ls_tap.py`|`1664684de351199a0a825b04bde17308dba1dc46a566ee5826772b4ccfe91c83`|
|`code/scripts/run_d106_real_integration.py`|`3bb8acb3c48ad371c6c0b51f20fbefb0821445f2b7ecfaecd54de71e8a39de27`|
|`code/baseline_origin_sat_view.py`|`fa7221ae505a51a2afc2a51b857675ac4a5384b004d5a4f36e10dafc9d4f8ace`|
|`code/model_dual_cvsincnet.py`|`11b56f2a763eb49de21d0a566d8e4420538cb7af310c9338f4e8d4a21c42c235`|
|`code/model.py`|`afc6e6266a09fd5f5be967fed85254c6c92fa0241a0336fd5ffa3eb12aa1c417`|
|`configs/d106_rdce_method_lock_20260801.json`|`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`|
|`configs/d106_candidate_runtime_manifest_20260801.json`|`0e8bc733ce9650aea3463da90242f97e969210ca8a95983fee032f1474f87cb2`|

当前尚无D106 source-held、Target25、D62/D91/D92/SVRN matched性能结果。不得把机械训练面`+4/+4/+2`、真实588行no-query smoke或本地测试通过数写成性能提升。

## 15.N607只读真实authority闭合

### 15.1 可用资产

|资产|远端路径|SHA256|
|---|---|---|
|上游`source_validation`cache set|`runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json`|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|
|上游`source_train`cache set|同根`phase1_caches/source_train/cache_set.json`|`d719808ceaed07c13f6c8d8053acf910a61904243ec1d47c28cc4e4b679cffd2`|
|selection salt|`runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json`|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|
|checkpoint|`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|历史D105 runtime manifest，`NON_AUTHORITATIVE_FOR_D106`|`runs/d105_phase1_sourceheld_230c6cbc_20260801_r8/source/configs/d105_candidate_runtime_manifest_20260731.json`|`5de5926bbb2e9fd78b2f3315ec6e109964ddd6216ebe4f75e428b6b9f6bf11bc`|

D104正式远端split root、588条`L_s`和D106 disjoint receipt均不存在；四个D106生产文件也均未同步。预登记run root`runs/d106_real_integration_72850073_20260801_r1`实时为`ABSENT`。

### 15.2 纯ID归属证据

本地不可变D104 manifest SHA为`4a1e23cc999b7e7b6d5b53e44a6c02f142625c10dd0c9caf7ac0cee6dd2ada21`，其中输入字段名为`source_train_cache_set_sha256`且值为`125bb312…`。实时远端文件证明该SHA对应`cache_scope=source_validation`；真正`cache_scope=source_train`的SHA是`d719808c…`。runner只在内存中比较physical ID，不读取IQ、TX标签或性能。

|池|池规模|L_s命中|缺失|额外|结论|
|---|---:|---:|---:|---:|---|
|`source_validation`，SHA`125bb312…`|8400|588|0|7812|三个场景的matched ordered/set root均与本地L_s一致|
|`source_train`，SHA`d719808c…`|6000|22|566|5978|不是D104 L_s权威池|

本地L_s ordered root为`2798044663c1c727346c1142002e9b2dfc3e282e8b58d9752b420171c28ea12e`，set root为`ee81fd5c5efdbe171eaff9601990594b61b79adf8c3622a8cda8dbc2fd228d4b`。`source_validation`三个场景均得到相同根；`source_train`仅22个交集。

### 15.3 修正决策

1.保留D104 manifest及588/5292/2520切分；旧字段名作为不可变provenance存在，但D106显式解释为`D104_LEGACY_SOURCE_POOL_HASH_FIELD`，不再把它冒充真正source_train；
2.D106公开入口改为`upstream_source_pool_cache_set`/`--upstream-source-pool-cache-set`，并硬验`cache_scope=source_validation`、8400行、三个场景同序物理ID根及完整存储语义；
3.validator升级为v2，显式记录`upstream_source_pool_cache_set_sha256`、实际scope和D104旧字段名；选中588行的method artifact仍不携带全池SHA或全池能力；
4.由于received-IQ bytes、physical IDs、receiver/TX集合、场景、K、support/query split和`p2_min_v1`均未变化，本次只修实现契约，不触发数据重验；
5.本地独立复审已达`P0=0/P1=0/P2=0 / LOCAL DATA GO`，authority修正已提交为`bd9f1944`；在真实fixture闭环之前，DATA跨模块release、正式DA asset、source-held和Target仍为NO-GO。

## 16.真实fixture入口追踪表

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|`RI-01`|设计冻结§数据入口|严格fixture字段、输入正规文件与外部SHA绑定|`code/scripts/run_d106_real_integration.py`、测试|`verified`|extra query、输入篡改、非导入construction code负测|真实fixture执行仍`blocked`|
|`RI-02`|DATA闭包|调用builder-only 8400×3 extract并只封存588条L_s IQ|同上|`verified`|调用链和能力边界断言|真实8400×3执行仍`blocked`|
|`RI-03`|same-IQ dual forward|用冻结checkpoint/runtime从同一选中IQ导出strict tap|同上|`verified`|selected→export调用链断言|真实checkpoint forward仍`blocked`|
|`RI-04`|formal tap门|使用外部SHA严格重载tap并验证z_id=ReLU(pre_relu)|同上|`verified`|loader-origin调用链断言|真实tap仍`blocked`|
|`RI-05`|DA资产门|以冻结method lock SHA和构造代码SHA构建正式RDCE资产|同上|`verified`|typed lock、实际导入路径和SHA断言|真实asset仍`blocked`|
|`RI-06`|部署wire|原子保存并以精确lineage和wire SHA重载|同上|`verified`|save/load、receipt/binding roundtrip|真实wire仍`blocked`|
|`RI-07`|证据边界|发布canonical结果收据和完成marker，明确query/target/source-held访问均为false|同上|`verified`|exact flags、canonical result/marker及result SHA|无性能字段|
|`RI-08`|runner交接|单一CLI、精确退出语义、可由N607专属runner执行|同上|`implemented`|专项测试、`--help`、py_compile、release source commit`b268364b`|专属runner交接仍`blocked`|
|`RI-09`|Windows→N607交接|不可变D104 manifest中的Windows反斜杠相对路径必须在Linux安全解析，同时拒绝绝对路径、盘符和上跳|`stage2_d106_phase1_tap.py`、测试|`verified`|portable path正负测、execution closure、独立复审`P0=0/P1=0/P2=0`|未改写D104 manifest或SHA|
|`RI-10`|D106 runtime权威|不得借用历史D102/D105 runtime；新manifest须绑定设计、DATA、DA、入口、checkpoint、split和上游池，入口须解析canonical语义|`configs/d106_candidate_runtime_manifest_20260801.json`、真实集成入口及测试|`verified`|repo hash正测、runtime query漂移负测、独立复审`P0=0/P1=0/P2=0`|不引入方法参数或性能门|
|`RI-11`|真实checkpoint重建|release archive必须携带并由runtime绑定`model.py`与`model_dual_cvsincnet.py`，否则N607 exact model factory不可达|两模型文件、D106 runtime、入口及测试|`verified_local`|repo正向SHA、两字段参数化SHA漂移负测、独立依赖闭包复审|正式archive仍须核对entry/hash；不得退回项目根目录未封存模型|

独立审查：`P0=0/P1=0/P2=2 / LOCAL REAL-INTEGRATION ENTRY GO`。P2为`release_commit`需在正式交接时由fixture SHA、Git commit和同步文件SHA外部闭合，以及当前核心链专项测试使用替身；二者均不冒充真实N607证据。真实fixture、真实checkpoint forward、正式asset/wire/receipt和runner交接仍为`blocked`。

RI-09独立代码复审：`P0=0/P1=0/P2=0 / IMPLEMENTATION GO`。Windows反斜杠先规范为portable组件；绝对路径、UNC、盘符、colon、空组件、`.`和`..`均在文件访问前拒绝，随后仍执行root escape、symlink、正规文件和archive SHA边界。

RI-10独立代码复审：`P0=0/P1=0/P2=0 / IMPLEMENTATION GO`。新D106 runtime为单向绑定，无自哈希循环；历史D102/D105 runtime因schema、candidate、字段和内容绑定不符而拒绝。为消除Markdown在Windows/Git archive之间的CRLF/LF差异，正式方法锁另封为canonical JSON`e7a1982b…`，两份JSON由`.gitattributes`强制`eol=lf`；正式fixture只能指向SHA为`09d7b350…`的新D106 runtime。

RI-11独立窄审查：实现机制`P0=0/P1=0/P2=0`。入口从自身`CODE_ROOT`定位并按实际SHA绑定`model_dual_cvsincnet.py`与`model.py`；exact builder同时校验模块`__file__`和factory来源。`model_dual_cvsincnet.py`唯一额外本地依赖为`model.py`，后者只依赖标准库、NumPy和PyTorch。补齐两模型SHA漂移负测后本地release门闭合；正式archive必须按原布局包含`source/code/model_dual_cvsincnet.py`和`source/code/model.py`，并在交接前逐entry复核。

已生成本地ID-only disjoint receipt：`artifacts/d106_train_held_disjoint_receipt.json`，SHA256=`ee7005fcc99d703dac2f3e529e39426587ffa8967d19c15cf848c98f5295d961`，`train_held_intersection_count=0`、`tx_labels_read=false`、`formal_query_access=false`。

## 17.真实集成release预登记

release source commit为`deefd57c4185a5343f87772be78b5038c37e6217`。本地专项8/8、D106主链101通过1跳过、依赖51/51，RI-11复核为`P0=0/P1=0/P2=0 / LOCAL IMPLEMENTATION GO`。N607 run ID冻结为`d106_real_integration_deefd57c_20260801_r4`，唯一启动owner为专属Terra Max runner，不授权自动retry。

|Release资产|SHA256|状态|
|---|---|---|
|`artifacts/d106_real_integration_source_deefd57c.zip`|`6d30d85b624ca2a94d8b5fcde4be0ba4d32d36e87c0e21d07fc793fee65e21a2`|Git commit导出；关键五entry及SHA已本地复核|
|`artifacts/d104_split_4a1e23cc.zip`|`b1884cf1a7e287aa489a2b591fc5688a7e655c6b541f6f90eafcf71cf476372e`|D104冻结split|
|`artifacts/d106_real_integration_fixture_deefd57c.json`|`74b2367f82a682a41f46447b089ec85bb21433b39ec8205167356909e3cd0ff1`|canonical无换行；字段集合与release commit已本地复核|
|`artifacts/d106_train_held_disjoint_receipt.json`|`ee7005fcc99d703dac2f3e529e39426587ffa8967d19c15cf848c98f5295d961`|ID-only且train/held交集为0|

详细路径、启动命令、GPU、健康门、预期文件和SSH清理要求见`d106_real_integration_runner_handoff_deefd57c.md`。当前状态仍为`LOCAL_RELEASE_READY / NOT_LANDED`，没有真实checkpoint forward、N607 asset或性能结果。

## 18.RCMR-2V分类头冻结

第四个HEAD候选冻结为`D106-RCMR-2V-qKNN/r1.1`。它用同一IQ一次前向的`ReLU(pre_relu)`与signed`pre_relu`两视图，在量化support图上计算跨视图秩可靠度\(R_i\)，再按query全局秩、support局部profile秩和query可靠度\(R_q\)形成双侧拥挤度分数。最终类分数为类内等K平均，不读取旧Student-t分数，也没有阈值、温度、残差、hard gate、identity fallback或可扫描权重。

设计和资源上界已独立冻结在`analysis/d106_rcmr_2v_qknn_design_freeze_20260801.md`。\(N=260,D=160\)时固定二进制payload为86,060B，arrays-only临时峰值为2,285,920B，含固定state resident峰值为2,371,980B；一次`prepare`为10,774,400 MAC，单query为83,200 MAC。这些是解析上界，不是实测吞吐或性能。

当前结论仅为`DESIGN_FROZEN / IMPLEMENTATION_AUTHORIZED`。实现必须由不同Terra Max子agent完成formal state、strict loader、一次性context、query scorer、wire/receipt和协议负测；再由非作者独立复审。真实特征G0若与旧qKNN逐query同序，则直接`REJECT_NO_FUNCTION`，不得据此扫描参数。

## 19.r4技术失败与r5修复

`d106_real_integration_deefd57c_20260801_r4`已完成落地门禁后执行唯一launch，但在3秒内因`D106RealIntegrationError: fixture must be an absolute regular file`退出。原handoff使用相对`--fixture ../input/...`，与入口绝对正规文件合同不一致。`output/result/completion`均不存在，run-owned进程为0，GPU0已释放；异常指纹为`a4aacc5c…`，log SHA为`043ef9b1…`。该run永久封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得重启或覆盖。完整报告见`automation_reports/CV-SincNet/d106_real_integration_deefd57c_20260801_r4/report.md`，9份小型证据见`artifacts/remote_deefd57c/`。

本地修复使用全新run ID`d106_real_integration_deefd57c_20260801_r5`。r5 fixture把全部run-local路径绑定到新root，SHA为`931fc133…`；handoff的`--fixture`和`--output-dir`均冻结为绝对路径，并新增机械测试拒绝`../`及路径越界。r5在独立复审和新commit前保持`NOT_LANDED`，不会把r4失败写成方法或性能结果。

## 20.r5依赖闭包失败

r5的绝对路径修复经独立复审`P0=0/P1=0/P2=0`并以`0dc2484b`发布。完整落地门和绝对命令复核通过后，唯一launch在checkpoint loader导入阶段因source archive遗漏`baseline_origin_sat_view.py`退出。异常指纹为`c2c635b2…`，log SHA为`53d4b8a6…`。partial `selected_ls_iq`完整保留但IQ内容未读取、未拉回；正式result/completion均不存在，run-owned进程为0，GPU和SSH已清理。r5永久为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

本地r6修复不再手写猜测依赖：以D105声明的`D105_CANDIDATE_RUNTIME_MODEL_FILES=(baseline_origin_sat_view.py,model.py,model_dual_cvsincnet.py)`为权威，D106 construction closure和runtime同时绑定三份实际SHA，并新增依赖集合漂移及三字段SHA负测。新archive必须包含并逐entry核验三文件；r4/r5均不得复用。
