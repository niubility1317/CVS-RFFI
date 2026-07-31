# D106-RDCE/GTSM-r3研发与实验报告

状态：`DESIGN_FROZEN / DATA_LOCAL_G0_GO / DA_LOCAL_G0_GO / REAL_INTEGRATION_HARD_GATE_PENDING / N607_NOT_ACCESSED / SOURCE_HELD_NOT_OPENED / TARGET25_NO_GO / NO_TARGET_PERFORMANCE_RESULT`

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

## 4.数据与协议

- `protocol_schema=p2_min_v1`
- Phase1 split：588/5292/2520，对应`L_s/U_s/source validation`
- `rho_label=0.1`
- `L_s`SHA256：`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`
- checkpoint SHA256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- source-held truth尚未用于性能计算
- Target capsule保持`VALIDATED_ONCE`，方法变化不得触发数据重验

禁止当前D105 8400行source-validation tap进入D106训练。正式D106 tap只能在冻结D104 split后精确选择588个`L_s`physical ID，并保留day/scenario/observation绑定。

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
|DATA|`stage2_d106_phase1_tap.py`、CLI及测试|本地实现完成；独立复审`P0=0/P1=0/P2=0`、`LOCAL G0 GO`|
|DA|`stage2_d106_rdce_asset.py`、`stage2_d106_rdce_runtime.py`及测试|实现完成；DA测试20/20、DATA+DA联合80/80通过；独立复审`P0=0/P1=0/P2=0`、`LOCAL G0 GO`|
|HEAD|候选文件待新revision冻结|`SG-LC-CL-OOF/r1`与`SBCM-BR/r2`均在训练面拒绝|
|四臂/held|`stage2_d106_four_arm.py`、source-held predictor/scorer|待HEAD冻结|
|Target25|基于D105骨架的新runner/launcher|G1前禁止实现release|

## 10.N607信息

本轮尚未执行N607 preflight、SSH/SCP、远端目录创建、同步、编译、启动或监控。无server command、PID、GPU分配、log path或远端output。只有完成本地实现、定向/协议负测、真实checkpoint no-query smoke、独立审查`P0=0/P1=0`、Git commit和报告预登记后，才允许唯一runner进入N607 Phase1。

## 11.风险与下一步

当前release硬门为：真实8400×3缓存、selection salt、588条`L_s`、真实checkpoint的extract→export→validate闭环尚未在本机执行；该真实夹具仅在测试中明确跳过，必须在N607 release前由唯一runner完成。HEAD、四臂held/scorer和Target25 runner仍未实现。DATA与DA代码门不得替代真实资产门或性能证据。

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
- 独立复审结论：`P0=0/P1=0/P2=0 / LOCAL G0 DATA GO`；
- DATA测试：60通过、1跳过；跳过项为缺少环境变量`D106_REAL_INTEGRATION_FIXTURE`的真实资产闭环。

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
|D106 DATA+DA四个测试文件|80通过、1跳过|最终receipt接口与DA runtime闭合；真实夹具仍是release硬门|
|D104 split、D105 tap、Student-t qKNN、VALIDATED_ONCE句柄|59/59通过|未破坏直接依赖|
|Python编译与`git diff --check`|通过|无语法或空白错误|

本地放行生产文件SHA256：

|文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_d106_phase1_tap.py`|`a70f1f280c750332e52951007e8184a1e4c4b4e49f9dfca649316c7b176db782`|
|`code/cvsrffi/stage2_d106_rdce_asset.py`|`e9d57245a80cdf31ae4ea5fd76cd521022399d0be512e2647a92cc2a0671da1f`|
|`code/cvsrffi/stage2_d106_rdce_runtime.py`|`9d78b83134bfb668c3b9c32053eaa86b5c9fd4d970e87aa99dc30ac2df8df946`|
|`code/scripts/export_d106_phase1_ls_tap.py`|`744616c0f07b5c232ad4695555b1aa9764e835fd50b2d4000a306199b56bb608`|

当前尚无D106 source-held、Target25、D62/D91/D92/SVRN matched性能结果。不得把机械训练面`+4/+4/+2`、真实588行no-query smoke或本地测试通过数写成性能提升。
