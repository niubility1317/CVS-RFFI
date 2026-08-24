# Phase1伪标签方法近两周进展与星地信道收益综合报告

报告周期：2026-08-11—2026-08-24

报告状态：`ANALYZED / SINGLE_SEED_METHOD_PROGRESS_SUMMARY`

## 一、结论

近两周的实质进展集中在2026-08-19至2026-08-24。对当前Git历史和正式实验报告的检索没有发现8月11日至18日新增的伪标签性能矩阵；因此，本报告的定量证据来自8月19日至24日完成或终止的七组主线实验。七组矩阵共形成43条具有最终性能artifact的候选，另有11条因确定性技术失败没有性能结果，MUSE的完整M1/M3因顺序调度被前序失败阻断而未启动。

研究结论发生了三次关键转变。

第一，项目已经否定“覆盖越高越好”。MUSE从E1开始接收几乎全部无标签样本，最佳LEO均值只有34.09%，比历史ADV3B02同口径70.56%低36.47个百分点。FastTrust随后把身份选择固定为50%，基础伪标签仍比无U身份控制低0.258个百分点LEO均值。RC4把H/P/N有效加权coverage提高到16%—22%后，P5/P6仍分别比严格H低0.283和0.229个百分点LEO均值。

第二，项目找到了稳定有效的信息类型：严格可信身份锚点、可信身份向LEO弱信道视图的迁移、类别均衡，以及受控的部分标签条件监督。FastTrust同row实验中，完整U256相对去掉U星地身份分支，LEO均值提高0.812个百分点、receiver-cell floor提高1.575个百分点；类别cap额外贡献0.339个百分点LEO均值和0.683个百分点floor。这是近两周最强的伪标签星地收益因果证据。

第三，8月24日完成的QB3首次在完整同row E200矩阵中，让优化伪标签候选相对无U身份控制同时提高Clean、三个LEO场景、LEO均值、场景floor和receiver-cell floor。完整C3相对C0的Clean、LEO均值、LEO场景floor和receiver-cell floor分别提高0.802、0.308、0.337和0.375个百分点。该结果证明风险校准P-set/P-conditional可以产生净正收益，但增益仍小、只有单seed，不能直接替换默认方法。

当前科学判断是：伪标签路线已从“高覆盖但不可信”推进到“中等有效coverage、预算受控、同row小幅正收益”。它已经证明可行，还没有证明稳定领先。历史FastTrust R4 U256仍以74.463%的LEO均值和60.383%的floor高于QB3 C3的73.867%和57.617%；差值分别为0.596和2.766个百分点。

## 二、统一协议与证据口径

主线实验固定为Phase1 source-domain weak-label/semi-supervised DG：

```text
L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15
样本数=5,880/52,920/12,600/12,600
source receivers=RX0–RX6
target receivers=RX7–RX11
source/target receiver overlap=0
seed=392002
正式预算=200epoch
U batch=256（FastTrust批量消融另含U128/U384）
```

训练期不读取`U_s`的TX真值。报告中的H/P/N/R数量、校准概率和coverage表示无标签数据的使用规模与加权强度，不是伪标签precision。target Clean与LEO结果只来自冻结final checkpoint的训练后测试，不进入训练、校准、候选重排或选择性重跑。

主线保留Core90的LEO_WEAK拼接增强：

```text
E1–E40:  leo_clear_weak,p=0.30
E41–E90: leo_low_elev_weak+leo_rain_weak,p=0.60
E91–E200:三场景,p=0.80
lambda_sat_cls=0.68
lambda_sat_cons=0
```

每条正式Phase1性能结果必须由同一final checkpoint分别完成Clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`评测。FastTrust、SAT-Anchor、RC4及QB3主线的每个场景均为60000条`test_unseen_day_unseen_rx`样本，5个target receivers各12000条。

本报告使用两个floor：

- LEO场景floor：clear、low-elev、rain三个聚合场景中的最低准确率。
- receiver-cell floor：Clean与三种LEO场景×5个接收机共20个单元中的最低准确率。

跨报告绝对值只用于路线背景；因果结论只来自同一矩阵内相同seed、split、初始化、预算和训练步数的对照。

## 三、证据总览

| 路线 | 日期 | 完整性能行 | 无性能结果 | 主要问题 | 终态判断 |
|---|---|---:|---:|---|---|
| MUSE SSDG | 8月19—21日 | 8 | 0；另有2条完整臂未启动 | 从头训练、E1全量伪标签、校准缺类、对抗尾段崩溃 | `NEGATIVE_EVIDENCE` |
| FastTrust 16-row r2 | 8月21—22日 | 16 | 0 | 固定50%低可靠回填、若干复杂组件无益 | 完整机制有LEO收益，但归因集中在严格U星地身份和class cap |
| SAT-Anchor-SSL | 8月22—23日 | 7 | 1 | 增益小、指标最优点分散、adapter-tail数值失败 | 稳定机制库，不晋级 |
| RC4 E50/E100 | 8月22—23日 | 3 | 5 | 互补概率非有限、E91尾段崩溃 | `SHORT_RUN_REFERENCE_ONLY` |
| RC4 LogitQ E200 | 8月23日 | 3 | 3 | P/N信息密度不足、共享U对抗失稳与OOM | P/N未超过严格H |
| RC4质量预算QB | 8月23—24日 | 1 | 2 | 控制组E104—106对抗崩溃 | 高原始利用率和固定有效预算达成，科学增益未闭合 |
| QB3有界域混淆 | 8月24日 | 5 | 0 | 低频非有限梯度、梯度遥测失效 | 首次完整同row小幅全指标正收益 |

## 四、方法演进

### 4.1 MUSE：把“充分使用U_s”误解成“从第一轮全量自举”

MUSE组合global/local/prototype三头、H/M/L路由、prior、temporal、satellite、cross-receiver、nuisance和prototype更新。M2从E1就接收52864/52864个U样本，初始平均置信度只有0.588；训练又从scratch开始，没有继承Core90强checkpoint或教师约束。

| 候选 | Clean | Clear | Low-elev | Rain | LEO均值 |
|---|---:|---:|---:|---:|---:|
| M2 | 73.33 | 33.91 | 33.19 | 33.48 | 33.53 |
| M3_NO_SATELLITE | 75.85 | 34.57 | 33.85 | 33.86 | 34.09 |
| 历史ADV3B02同口径 | 86.09 | 72.56 | 69.87 | 69.27 | 70.56 |

M2的receiver/day/channel泄漏excess分别达到0.64897/0.26746/0.42472，均超过设定上限；M0则在E173后发生对抗损失爆炸，最终退化到6类随机水平16.67%。MUSE留下的有效结论不是某个组件有效，而是三条设计原则：必须继承稳定底座；伪标签必须允许拒绝；校准、类别覆盖和域泄漏必须可观察。

### 4.2 FastTrust：从全量自举收缩到可信子集，但固定50%配额仍然过强

FastTrust使用high reliability、temporal stability、三头一致和class cap构造hard子集。问题是hard不足时继续从mid、未入hard的high和low reliability样本补齐，最终每epoch固定选择26460/52920，即50%的U样本。

| 候选 | Clean | LEO均值 | receiver-cell floor |
|---|---:|---:|---:|
| R1无MUSE控制U256 | 85.152 | 73.656 | 58.525 |
| R2基础FastTrust H/M/L | 84.225 | 73.398 | 58.808 |
| R3增加U prototype | 84.973 | 73.578 | 58.583 |
| R4 Full U256 | 84.540 | 74.463 | 60.383 |
| R4 No U Sat ID | 84.750 | 73.651 | 58.808 |
| R4 No Class Cap | 84.827 | 74.124 | 59.700 |
| R4 Full U128 | 83.958 | 75.645 | 62.883 |

基础伪标签R2相对R1的LEO均值下降0.258个百分点；R3也下降0.078个百分点。完整U256相对R1提高0.807个百分点LEO均值和1.858个百分点floor，但Clean下降0.612个百分点。

最有解释力的是同机制消融：

- Full U256−No U Sat ID：Clean-0.210、LEO均值+0.812、floor+1.575个百分点。
- Full U256−No Class Cap：Clean-0.287、LEO均值+0.339、floor+0.683个百分点。
- No Temporal−Full：LEO均值+0.253、floor+0.525个百分点。
- No CrossRX−Full：LEO均值+0.189、floor+0.275个百分点。
- No Nuisance−Full：Clean+0.535、LEO均值+0.184、floor+0.492个百分点。
- No Proto Evidence−Full：LEO均值+0.160、floor+0.317个百分点。

这组结果把复杂MUSE包拆开了：严格U星地身份分支和class cap提供正收益，source prior只有小幅帮助；temporal、cross-RX、nuisance和prototype evidence没有独立正贡献。U128单点达到75.645%LEO均值和62.883%floor，但训练时长22.58小时，且每epoch更新步数改变，不能把全部增益归因于方法。

### 4.3 SAT-Anchor：把身份监督、教师锚定和无标签星地配对分开

SAT-Anchor取消通用身份回填，分别测试严格U星地身份CE、冻结Core90 clean教师锚定、全U clean-satellite SimSiam配对、pair降频和class×receiver cap。

| 候选 | Clean | LEO均值 | floor | 相对A0的Clean/LEO/floor |
|---|---:|---:|---:|---|
| A0控制 | 84.847 | 73.486 | 58.467 | 0/0/0 |
| A1严格U星地身份 | 85.075 | 73.702 | 58.408 | +0.228/+0.216/-0.058 |
| A2加入clean锚定 | 84.673 | 73.651 | 59.408 | -0.173/+0.166/+0.942 |
| A3 adaptive全U pair | 84.567 | 73.873 | 58.208 | -0.280/+0.388/-0.258 |
| A3 fixed50 fill | 84.670 | 73.918 | 58.550 | -0.177/+0.432/+0.083 |
| A3 pair interval2 | 84.590 | 73.644 | 58.900 | -0.257/+0.158/+0.433 |
| A4 class×receiver cap | 84.345 | 73.615 | 58.975 | -0.502/+0.129/+0.508 |

7条有效行的LEO增益范围只有0.129—0.432个百分点。clean锚定主要抬升floor，全U pair主要抬升LEO均值，class×receiver cap继续改善floor，但没有一条同时保护Clean、提高LEO均值并取得最大floor。SAT-Anchor证明“不是所有U样本都需要伪身份，也可以参与星地表征学习”，但单独的无标签配对没有形成大幅收益。

### 4.4 RC4：从唯一硬标签扩展到H/P/N/R四态信息

RC4把U样本分为：唯一高可信H、2—3类候选集合P、安全排除N和无身份监督R。P/N不强行指定唯一类别；损失按完整U batch归一化，并使用class×receiver均衡和source-only交叉校准。

短训练首次把约99.1%的U样本分入H/P/N，路由目标已经达到，但旧概率实现使用`log1p(-p)`；float32中的`1-1e-8`仍可能等于1，导致`log1p(-1)=-inf`。四条P6/P7在身份启用后的两个完整epoch全部零更新，属于确定性工程失败。

LogitQ把集合损失改成稳定的logit空间形式：

```text
logsumexp(全部类别logit)-logsumexp(候选集合logit)
```

修复后P3、P5、P6均完成E200，平均optimizer step应用率约99.91%。工程问题被解决，科学结果仍为负：

| 候选 | 有效coverage | Clean | LEO均值 | floor |
|---|---:|---:|---:|---:|
| P3严格H | 9.08% | 85.500 | 73.979 | 58.558 |
| P5 H+P-set+P-cond | 16.04% | 84.822 | 73.696 | 58.050 |
| P6 H+P+N | 21.86% | 84.783 | 73.750 | 58.175 |

P5−P3为Clean-0.678、LEO均值-0.283、floor-0.508个百分点；P6−P3为-0.717/-0.229/-0.383个百分点。覆盖率提高没有改善最困难的RX8，receiver probe准确率反而从P3的75.976%升至P5的78.690%，说明新增伪监督仍沿接收机/信道捷径强化分类。

### 4.5 质量预算QB：把原始样本利用率与有效梯度预算解耦

QB2不再给P固定样本比例，而是先让严格H占用预算，再用V_cal安全P填充剩余的总有效身份预算0.15。E200尾段每batch平均H/P/R为2.372/175.589/77.691，H+P原始利用率约69.6%，但总有效加权coverage始终锁定15%。

QB2完成E200，Clean/LEO均值/floor为85.168/73.592/58.017。历史R1无U身份控制为85.152/73.656/58.525，二者背景差值为+0.016/-0.064/-0.508个百分点，不能证明性能收益。更关键的是QB0和QB1在E104—106因共享对抗分支爆炸终止，QB2缺少本run同row反事实。

QB的贡献是工程性的：它证明“使用约70%的U样本”与“只施加15%的有效身份预算”可以同时成立，并把完整E200墙钟从旧P3约11.94小时降到9小时8分42秒，缩短约23.4%。它没有证明P填充优于严格H。

### 4.6 QB3：有界域混淆恢复完整因果矩阵

QB3统一修复有标签与U两条`z_id→domain`路径。域判别器在detached表示上学习域分类；身份编码器使用有界的均匀分布混淆目标；域头内部固定float32。矩阵同时拆分严格H、P-set、P-conditional和无伪标签U特征锚定。

| 候选 | Clean | Clear | Low-elev | Rain | LEO均值 | LEO场景floor | receiver-cell floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0无U身份 | 84.4000 | 75.4883 | 72.9800 | 72.2083 | 73.5589 | 72.2083 | 57.2417 |
| C1严格H | 84.6417 | 75.3667 | 72.7867 | 72.1450 | 73.4328 | 72.1450 | 57.4333 |
| C2 H+P-set | 84.8867 | 75.6667 | 73.0567 | 72.4333 | 73.7189 | 72.4333 | 57.0750 |
| C3 H+P-set+P-cond | **85.2017** | **75.8517** | **73.2033** | **72.5450** | **73.8667** | **72.5450** | 57.6167 |
| C4 U feature anchor | 84.3150 | 75.3967 | 72.9050 | 72.1483 | 73.4833 | 72.1483 | **57.9167** |

单因素分解为：

- C1−C0：Clean+0.242、LEO均值-0.126、receiver floor+0.192个百分点。严格H单独没有改善LEO平均值。
- C2−C1：Clean+0.245、LEO均值+0.286、LEO场景floor+0.288、receiver floor-0.358个百分点。P-set提高均值但损伤局部floor。
- C3−C2：Clean+0.315、LEO均值+0.148、LEO场景floor+0.112、receiver floor+0.542个百分点。P-conditional补回P-set的局部损失。
- C3−C0：Clean+0.802、LEO均值+0.308、LEO场景floor+0.337、receiver floor+0.375个百分点。
- C4−C0：Clean-0.085、LEO均值-0.076、receiver floor+0.675个百分点。一般U特征锚定改善最坏单元，却不能提高平均性能。

C3相对C0改善5/5个Clean receiver和15个receiver×LEO单元中的12个。唯一一致退化的是RX7在三种LEO场景分别下降0.108、0.575和0.300个百分点；最差单元仍是RX8-rain，仅从57.242提高到57.617%。

## 五、星地信道性能收益的分层判断

### 5.1 已得到强同row支持的收益

| 机制比较 | Clean变化 | LEO均值变化 | receiver-cell floor变化 | 证据强度 |
|---|---:|---:|---:|---|
| FastTrust Full U256−No U Sat ID | -0.210 | **+0.812** | **+1.575** | 强同row单因素证据 |
| FastTrust Full U256−No Class Cap | -0.287 | +0.339 | +0.683 | 同row正证据 |
| QB3 C3−C0 | **+0.802** | **+0.308** | **+0.375** | 完整同row全指标正向 |
| QB3 C3−C2 | +0.315 | +0.148 | +0.542 | P-conditional增量正向 |

严格U星地身份监督的0.812/1.575个百分点收益，是近两周最清晰的星地机制归因。QB3 C3的绝对增益较小，却更均衡：它没有以Clean下降换取LEO提高。

### 5.2 有方向但不足以晋级的收益

SAT-Anchor各臂相对A0仅提高0.129—0.432个百分点LEO均值；最佳floor增益为A2的0.942个百分点，但Clean下降0.173个百分点，LEO均值只提高0.166个百分点。E50 P3相对P0提高1.873个百分点LEO均值和2.700个百分点floor，但E50被预先限定为短训练筛选，不能进入正式性能结论。

### 5.3 没有形成收益的扩展

FastTrust基础H/M/L和U prototype没有超过R1。RC4 LogitQ的P5/P6虽然把有效coverage从9.08%提高到16.04%/21.86%，但LEO均值和floor都低于严格H。QB2把原始身份利用率提高到69.6%，性能却与历史无U身份控制近似持平。星地鲁棒性不是由伪标签数量直接决定，而是由伪标签是否提供跨接收机稳定的身份信息决定。

## 六、无标签利用率取得的真实进展

| 阶段 | 原始身份利用 | 有效加权coverage | 结果 |
|---|---:|---:|---|
| MUSE M2 | 接近100% | 未形成可信预算 | 严重确认偏差，LEO均值33.53% |
| FastTrust固定回填 | 50% | 固定身份配额 | 基础伪标签无收益，严格U星地子集有效 |
| SAT-Anchor adaptive | 约63.91/256=25.0% | 严格no-fill | 稳定但增益小 |
| RC4 P3严格H | 全程原始约9%量级 | 9.08% | E200有效行最佳 |
| RC4 P5/P6 | 更多H/P/N | 16.04%/21.86% | 覆盖增加，性能下降 |
| QB2质量预算 | E200尾段约69.6% | 固定15% | 利用率和预算解耦成功，缺少因果控制 |
| QB3 C2 | 尾段88.29% | 12.9238% | P-set提高均值但降低receiver floor |
| QB3 C3 | 尾段72.85% | 12.8723% | 原始覆盖更低，却取得全指标最佳 |

QB3 C2与C3是最能说明问题的一对。两者尾段有效coverage几乎相同，C3却主动减少约15.4个百分点原始H+P路由，并通过P-conditional提高监督的信息结构，最终Clean、LEO均值和receiver floor都优于C2。项目已经从“多少样本进入身份损失”转向“每单位有效权重能提供多少可靠身份信息”。

## 七、接收机与场景层面的收获

近两周的最低单元几乎始终落在RX8-rain。FastTrust R4 U256将总体floor从R1的58.525%提高到60.383%；RC4 LogitQ P3为58.558%；QB2为58.017%；QB3 C3为57.617%。不同伪标签设计没有稳定消除RX8瓶颈。

QB3 C3相对C0改善了RX8的clear/low-elev/rain约0.275/0.358/0.375个百分点，但RX7三个LEO场景全部轻微下降。C4一般特征锚定把最低单元提高0.675个百分点，却使LEO均值下降0.076个百分点。这说明receiver floor和LEO宏平均受到不同机制控制：表征平滑偏向最差域，类别条件伪标签偏向总体身份可分性。下一轮不能用一个聚合指标替代另一个。

场景难度保持稳定排序：clear最高，low-elev次之，rain最低。QB3 C3相对C0在clear/low-elev/rain分别提高0.363/0.223/0.337个百分点，三个场景方向一致。它没有出现只在最弱或最容易场景单点获益的情况。

## 八、工程稳定性进展

近两周修复了四条会直接破坏科学实验的数值链：

1. FastTrust local probability在AMP下回到float16，造成有限forward、NaN backward和长期零更新。固定float32后16/16候选完成E200。
2. RC4概率空间`log1p(-p)`在饱和概率上产生`-inf/NaN`。LogitQ改为logsumexp差后，P3/P5/P6完整跑到E200。
3. 非有限批次跳过backward时，图连接遥测张量没有detach，显存累积后造成OOM。QB统一detach并加入批次级保护，避免把OOM误认为模型静态显存不足。
4. 共享U domain/adversarial路径在E91或E104以后发生无界放大。QB3改为有界均匀分布混淆并强制域头float32，5行全部完成E200，1000个epoch中的RC4域分量始终有限。

QB3仍保留低频非有限梯度：每行约35—40个batch被跳过，占约41400个训练step的0.0845%—0.0966%。C0和C4也会发生，因此不能归因于P-set/P-cond。异常包没有记录首个非有限参数名，当前只能定位到共享反向路径。

另一个未闭合问题是梯度分项遥测。配置要求E1、E41、E91、E161、E181、E200记录`g_L/g_H/g_Pset/g_Pcond/g_adv`，但训练循环从1开始编号，代码条件检查`batch_idx==0`，1000个epoch全部没有真正启用遥测。该问题不影响最终准确率，却限制了对“P-conditional为什么有效”的梯度级解释。

## 九、训练加速与资源效率

| 路线/候选 | E200训练墙钟 | 关键设置 | 结论 |
|---|---:|---|---|
| FastTrust Full U128 | 22.58h | U batch128，U全覆盖 | 准确率高，但步数约翻倍 |
| FastTrust Full U256 | 15.11h | 效率主点 | 后续默认批量选择 |
| FastTrust Full U384 | 10.37h | 更高吞吐 | LEO均值-0.791、floor-1.317，相对U256 |
| RC4 LogitQ P3 | 11.94h | 重型source评估较频繁 | 数值稳定，仍偏慢 |
| QB2 | 9小时8分42秒 | source-heavy每5轮、尾20轮逐轮 | 比旧P3缩短23.4% |
| QB3 C3 | 9小时5分58秒 | eval batch1024、source-heavy每10轮 | 比QB2只快0.50%，无实质新增加速 |

E50和E100分别约2.9和6.7GPU小时/行，适合数值筛选与尾段诊断，但E200 P3相对E100 P3仍提高5.052个百分点LEO均值和4.366个百分点floor。正式默认预算继续为200epoch。

QB3把峰值allocated从QB2约3.20GiB提高到约6.08GiB，墙钟却没有明显下降。更大的eval batch主要增加瞬时显存，未形成端到端收益。下一轮加速应隔离测试checkpoint I/O、teacher前向缓存和评测batch，不再把多项优化一起改变后根据总墙钟推断原因。

技术失败也消耗了大量资源。RC4 E50/E100矩阵总计19.847GPU小时，其中5条失败行消耗7.368GPU小时；E100 P0单行失败就消耗6.693GPU小时。预注册非有限停止规则把四条RC4早期失败限制在约0.676GPU小时，说明最快的加速仍是尽早发现确定性错误，避免长程坏轨迹。

## 十、当前最佳结果如何理解

如果只看近两周的最高绝对星地指标，FastTrust R4 Full U128达到75.645%LEO均值和62.883%floor；但它用U128使训练步数增加，总时长22.58小时，Clean又比R1低1.193个百分点。效率主点R4 Full U256为74.463%LEO均值和60.383%floor。

如果看最完整的伪标签因果链，QB3 C3更有价值。它相对同run C0在Clean、三个LEO场景、LEO均值和floor上全部为正，并且5行全部技术闭合。代价是绝对LEO均值73.867%、floor57.617%，仍低于历史R4 U256。

因此，“性能冠军”和“方法证据最干净的候选”不是同一个对象：

- R4 U128/U256说明严格U星地身份监督可以取得较大的星地收益。
- QB3 C3说明风险校准P-set/P-conditional在完整因果控制下可以产生小幅净收益。
- 现阶段最合理的研究主线是把两者结合为新的预注册假设，而不是根据target结果直接拼接组件或调参。

## 十一、两周内已经取得的进展

1. 从“全量伪标签”转向“允许拒绝、分层表达不确定性”，确认MUSE全量自举不可行。
2. 通过16行FastTrust消融识别出真正有效的星地机制：严格U星地身份和class cap。
3. 证明无标签样本可以在没有唯一伪身份时，通过P集合、N排除或一般表征路线参与训练。
4. 将原始利用率与有效梯度预算解耦，避免高覆盖自动变成高损失权重。
5. 修复AMP概率下溢、互补概率非有限、遥测图保留和无界域对抗四类系统性故障。
6. 把U256 E200单行时间从约15.11小时压到约9.1小时，同时保留最终四场景完整评测。
7. 首次得到C3相对C0全主要指标同向为正的完整E200矩阵。

## 十二、仍未解决的问题

1. 全部主结果只有seed392002，缺少训练随机性的置信区间。
2. 没有保存候选间逐样本配对预测，无法进行McNemar等配对显著性检验。
3. RX8-rain仍是最低单元；QB3又出现RX7三个LEO场景轻微退化。
4. source satellite proxy与target LEO存在约20个百分点差距，且候选排序并不稳定一致。
5. 风险分数曾长期饱和在0.989附近，变量`risk`实际表示`p_correct`，可解释性不足。
6. P-set/P-conditional的梯度分项没有在正式QB3训练中落盘。
7. 低频共享非有限梯度仍未定位到具体参数。
8. 当前C3的+0.308个百分点LEO均值小于单seed下应谨慎处理的0.5个百分点量级，不能直接晋级。
9. WiSig/ManySig和LEO弱信道是地面代理及物理启发压力测试，不是真实在轨验证。

## 十三、下一步建议

下一轮应先冻结方法，再复验，而不是继续扩大机制包。

1. 冻结QB3 C3和C0进行多seed E200确认；若资源允许，保留C2作为P-conditional单因素对照。
2. 在复验前只修复梯度遥测条件和首次非有限参数定位，不改变伪标签路由、预算或损失，以保持机制可复现。
3. 将严格U星地身份分支作为一个预注册单因素引入C3体系，必须同时保留无该分支的同row控制；不能根据历史target收益直接默认开启。
4. 晋级条件同时约束Clean、clear、low-elev、rain、LEO均值和receiver-cell floor。建议至少要求多seed平均LEO均值≥+0.30个百分点、floor≥+0.30个百分点、Clean不下降，并且三个LEO场景方向一致。
5. 对RX7和RX8做source-only可靠性分桶、receiver条件预算和梯度贡献诊断；target truth只用于最终冻结评分。
6. 单独开展不改变优化数学定义的加速A/B：checkpoint频率、teacher前向缓存、eval batch512/1024和分段计时。正式预算继续E200，E50/E100只作开发筛选。

## 十四、证据定位

- Core90复现背景：`automation_reports/CV-SincNet/phase1_adv3b02_core90_triplet_20260819/report.md`
- MUSE：`automation_reports/CV-SincNet/phase1_adv3b02_muse_ssdg_20260819/report.md`
- FastTrust16：`automation_reports/CV-SincNet/phase1_adv3b02_fasttrust16_s392002_20260821_r2/report.md`
- SAT-Anchor：`automation_reports/CV-SincNet/phase1_adv3b02_sat_anchor_ssl8_s392002_20260822/report.md`
- RC4 E50/E100：`automation_reports/CV-SincNet/phase1_adv3b02_fasttrust_rc4_e50e100_s392002_20260822/report.md`
- RC4 LogitQ：`automation_reports/CV-SincNet/phase1_adv3b02_fasttrust_rc4_logitq_e200_s392002_20260823/report.md`
- RC4质量预算QB：`automation_reports/CV-SincNet/phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1/report.md`
- QB3有界域混淆：`automation_reports/CV-SincNet/phase1_adv3b02_fasttrust_qb3_bc_hps_e200_s392002_20260824_r1/report.md`
- 8月17—23日原周报：`docs/weekly_reports/CVS_RFFI_Phase1伪标签优化周报_20260817_20260823.md`

## 十五、最终裁决

近两周伪标签研究取得了明确的方法论进展和有限的性能进展。

方法论进展已经闭合：全量/固定配额伪标签不是正确方向；严格可信子集、集合监督、有效质量预算和有界域混淆构成了更可靠的技术路线。星地性能方面，严格U星地身份监督已显示0.812个百分点LEO均值和1.575个百分点floor的强同row收益；QB3完整风险校准机制也首次取得Clean+0.802、LEO均值+0.308、receiver floor+0.375个百分点的全指标正向结果。

性能晋级尚未闭合。当前收益仍来自单seed，QB3绝对鲁棒性未超过历史R4，最差接收机问题仍在。最高科学状态应写为：`MECHANISM_PROGRESS_VERIFIED / SINGLE_SEED_GAIN_OBSERVED / MULTI_SEED_PROMOTION_PENDING`。
