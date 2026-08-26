# Phase1 FastTrust方法设计、实现与实验结果报告

> 报告日期：2026-08-27
>
> 研究对象：`phase1_adv3b02_fasttrust16_s392002_20260821_r2`
>
> 证据状态：`ANALYZED`（16/16条E200训练与64/64份最终评测完整）

## 设计—实现—证据追溯表

|ID|来源|要求|实现/配置目标|状态|验证证据|备注|
|---|---|---|---|---|---|---|
|FT-01|FastTrust设计§2|E1–E16禁用U身份监督，E17后启用分层路由|`muse_schedule_for_epoch`、`train_ssdg.py`|已验证|正式报告完整200epoch记录|S1边界保持不变|
|FT-02|FastTrust设计§2|global/local/prototype三头、prior、可靠性、时序稳定性和类别上限共同路由|`route_fasttrust`及训练主链|已验证|代码映射+消融矩阵|hard与后续身份补位须分开理解|
|FT-03|FastTrust设计§3|仅严格hard子集进入U星地身份CE|`train_ssdg.py`的`accepted`掩码|已验证|Full U256对No U Sat ID|最强同batch正机制|
|FT-04|FastTrust设计§3|沿用Core90的clean+LEO拼接、权重和三阶段日程|矩阵` satellite_training`与launcher|已验证|配置读回+四场景终评|属于星地特化，不等于通用泛化|
|FT-05|FastTrust设计§4|U batch独立可调、完整覆盖、拼接前向、AMP和资源遥测|矩阵、训练主链、资源摘要|已验证|U128/U256/U384完整运行|batch改变每epoch更新步数|
|FT-06|FastTrust设计§5|warmup、cosine、backbone尾段缩放、梯度裁剪和技术停止规则|`_fasttrust_lr_scales`及训练循环|已验证|测试+完整训练健康记录|r2修复AMP本地概率梯度故障|
|FT-07|发布矩阵|16条单seed同协议对照与单因素消融|r2矩阵JSON|已验证|16/16条E200、64/64份评测|机制结论限定为单seed|
|FT-08|用户问题|解释U128与U256差异及选择|batch配置+资源/指标对照|已验证|U128/U256同矩阵结果|U128是高预算冠军，U256是效率主配置|
|FT-09|科学裁决|区分伪标签、Core90初始化、星地增强和泛化结论|同seed差值与边界分析|已验证|R0/R1/R2/R3/R4及消融|禁止把完整收益笼统归因于伪标签|

## 1.执行摘要

FastTrust不是一个单一的“高置信度伪标签阈值”，而是一条逐步演进的方法族：原始FastTrust先用global/local/prototype三路身份证据、source prior、时序稳定性和类别均衡上限筛出可信样本；RC4把唯一硬标签扩展为H/P/N/R四态；QB把原始样本利用率与有效梯度预算解耦；QB3再用有界域混淆替换不稳定的GRL对抗，并把H、P-set和P-conditional拆成可归因组件。

截至本报告日期，最重要的结论有五条：

1. 原始FastTrust的普通身份伪标签没有增强通用泛化。R2相对同初始化、无U身份监督的R1，Clean下降0.927个百分点，LEO均值下降0.258个百分点。
2. 原始FastTrust中最强的可归因正机制是严格hard子集上的U星地身份监督。Full U256相对关闭该分支，LEO均值提高0.812个百分点，receiver-cell floor提高1.575个百分点；代价是Clean下降0.210个百分点。
3. U128是原始16行矩阵的绝对星地指标冠军，LEO均值75.645%、四场景receiver-cell floor62.883%；U256是效率主点，LEO均值74.463%、floor60.383%，训练时间15.11小时，而U128需要22.58小时。
4. QB3在三seed上证明H+P-set能稳定提高Clean、LEO均值和LEO场景floor，但最坏receiver×LEO单元仍不稳定。C2相对C0三seed平均为Clean+0.4839、LEO均值+0.1980、场景floor+0.2456、receiver×LEO floor−0.1167个百分点。
5. 当前不应把FastTrust或QB3晋级为Phase1默认方法。证据支持“星地特化与平均指标小幅改善”，不支持“通用跨接收机泛化能力已增强”；最终判定仍为`NO_PROMOTION_TO_DEFAULT`。

## 2.问题定义与实验边界

FastTrust工作在Phase1 source-domain weak-label/semi-supervised DG场景：

```text
L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15
样本数=5,880/52,920/12,600/12,600
source receivers=RX0–RX6
target receivers=RX7–RX11
source/target receiver overlap=0
正式训练预算=200epoch
```

训练期间不读取`U_s`的TX真值，也不读取target/query真值。`V_cal`只承担source-only校准，`V_select`用于独立truth-last质量审计时，伪标签生成与真值提取在不同进程中完成。最终性能只由冻结`final_ssdg.pth`在Clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`四个场景上的独立评测产生。

本报告区分三类结论：

- 通用泛化：至少同时保护Clean、LEO平均和跨receiver最差单元，并具有多seed一致性。
- 星地信道能力：Clean可轻微退化，但LEO均值、场景floor或receiver-cell floor得到明确提升。
- 工程可行性：数值稳定、训练可闭合、质量审计或速度优化成立，但不等于性能晋级。

## 3.原始FastTrust设计

### 3.1三头证据与可靠性

EMA teacher给出global分布，训练期local head给出局部分布，分类prototype给出第三路分布。三路分布在source prior alignment后融合，并由以下因素形成可靠性分数：

- 融合置信度；
- top-1与top-2 margin；
- 三头JS分歧；
- 样本到预测类别prototype的距离；
- temporal memory中的跨epoch稳定性。

三头的argmax完全一致形成`agreement`。严格hard候选定义为：

```text
hard_eligible=high_reliability∩temporal_stable∩three_head_agreement
```

在此基础上施加`hard_max_fraction=0.25`和class-balanced cap，得到`U_H`。原始设计详见[FastTrust设计稿](superpowers/specs/2026-08-21-adv3b02-fasttrust-pseudo-design.md)。

### 3.2H/M/candidate/no-identity四级路由

实际实现并没有在hard不足时停止选择，而是继续执行确定性补位：

1. 先按可靠性从mid与未进入hard的high样本中选择soft；
2. 配额仍不足时，再从low reliability中选择candidate；
3. 直到达到`identity_max_fraction=0.50`或样本耗尽；
4. 剩余样本为no-identity。

这段行为位于[muse_ssdg.py](../code/cvsrffi/muse_ssdg.py)的`route_fasttrust`。hard承担hard CE；soft承担可靠性加权soft CE；candidate承担候选集合CE；no-identity不承担TX身份监督。

形式化地，身份损失为：

```text
L_U_id=λ_U·(λ_H·CE_H+λ_M·SoftCE_M+λ_C·CandidateCE_C)
```

所有teacher分布、路由掩码和伪标签在进入student损失前detach，避免student通过路由反向修改teacher证据。

### 3.3严格U星地身份监督

FastTrust最关键的分支不是一般H/M/C身份损失，而是仅对严格hard子集生成配对LEO弱信道视图，并施加相同伪身份的卫星TX CE：

```text
accepted=fasttrust_route.hard∩identity_satellite_mask
L_U_sat=λ_U·λ_satellite·CE(f_sat[accepted],y_pseudo[accepted])
```

该分支位于[train_ssdg.py](../code/SSDG/train_ssdg.py)的FastTrust损失主链。soft和candidate不会进入U星地身份CE，因此它比普通身份分支更严格。

### 3.4Core90星地增强底座

所有主要候选除scratch控制外，都从`ADV3B02_CORE90_SOFT_E200`初始化，并沿用：

```text
E1–E40:  p=0.30,leo_clear_weak
E41–E90: p=0.60,leo_low_elev_weak+leo_rain_weak
E91–E200:p=0.80,三种LEO_WEAK
lambda_sat_cls=0.68
lambda_sat_cons=0
```

R0与R1的差值证明Core90初始化本身是主要星地能力来源：R0相对R1的LEO均值低4.862个百分点、floor低7.850个百分点。因此，任何FastTrust收益都必须在同一Core90初始化内部做消融，不能把初始化收益归到伪标签上。

### 3.5训练日程、稳定性与资源设计

- E1–E16关闭U身份监督；E17后逐步启用。
- AdamW，E1–E5线性warmup，E6–E160 cosine decay，E161–E180 backbone额外乘0.2，E181–E200乘0.05。
- `max_grad_norm=5`；连续两个完整epoch零optimizer step视为系统性技术失败。
- labeled batch固定128；U batch独立设置128/256/384，并完整覆盖52,920条`U_s`。
- 记录每epoch耗时、U吞吐、前向样本量、峰值显存和optimizer step。

原始run曾因local概率在AMP后回到float16，使非目标类概率下溢为0，出现“forward loss有限、backward梯度NaN、优化器持续跳步”。r2把local概率固定保留float32，其他方法变量不变。修复后16条候选平均optimizer step应用率为99.915%–99.940%，系统性崩溃未复现。

## 4.RC4、QB与QB3演进

### 4.1RC4：从唯一标签转向H/P/N/R四态

RC4不再要求每个可用U样本都产生唯一类别，而是定义：

- H：唯一高可信身份，使用hard CE；
- P：2–3类候选集合，使用集合质量与可选的集合内条件分布监督；
- N：可安全排除的类别集合；
- R：身份信息不足，拒绝身份监督。

`V_cal`上通过out-of-fold source-only校准估计`p_correct`、`p_set_safe`和`p_exclusion_safe`，并设置总体、类别和receiver预算。核心实现位于[muse_ssdg.py](../code/cvsrffi/muse_ssdg.py)的`build_rc4_calibration`、`route_fasttrust_rc4`和`rc4_identity_losses`。

早期RC4的集合损失使用概率空间`log1p(-p)`，在饱和概率处产生`-inf/NaN`。LogitQ改为稳定的logit空间差：

```text
L_Pset=logsumexp(全部类别logit)-logsumexp(候选集合logit)
```

这解决了工程稳定性，但P5/P6仍未超过严格H，说明“集合损失可算”不等于“集合信息有益”。

### 4.2QB：有效质量预算

QB把原始路由覆盖率与有效梯度预算分离。即使约70%的U样本进入H/P原始路由，总有效加权coverage仍可锁定在15%左右。这样可避免“使用样本多”自动等价为“身份梯度大”，但QB2缺少同run完整控制，最终只证明预算机制和速度可行，没有证明性能提升。

### 4.3QB3：有界域混淆

旧共享GRL对抗在E91或E104后出现无界放大。QB3把域头和身份编码器目标拆开：

- 域判别器在detached表示上学习真实域分类；
- 身份编码器通过冻结域头最小化到均匀域分布的KL；
- 每行混淆目标有界于`[0,log(num_domains)]`；
- 域头内部强制float32。

实现位于[bounded_domain_confusion.py](../code/cvsrffi/bounded_domain_confusion.py)，训练接入位于[train_ssdg.py](../code/SSDG/train_ssdg.py)。QB3用五行矩阵隔离：C0无U身份、C1严格H、C2 H+P-set、C3 H+P-set+P-conditional、C4仅U feature anchor。

### 4.4truth-last质量审计与训练速度优化

[phase1_pseudolabel_quality.py](../code/cvsrffi/phase1_pseudolabel_quality.py)和[质量审计脚本](../code/scripts/phase1_fasttrust_vselect_quality.py)把流程拆成`generate→extract-truth→score`三个阶段。真实历史C2 checkpoint在12,600条`V_select-as-U`记录上的结果为：

|质量指标|结果|
|---|---:|
|H precision|99.7574%|
|H coverage|22.9048%|
|H AURC|0.0001604|
|P-set正确类别覆盖率|99.0291%|
|P-set平均集合大小|2.098|
|P-set P95集合大小|3|
|set-safe条件下P-cond排序准确率|96.1676%|
|最弱receiver×day H precision|97.6744%|
|最弱class×receiver P-set覆盖率|50.0%|

总体质量很高，但class×receiver长尾明显。真实共享参数梯度遥测进一步显示H/P-set确实进入计算图，但身份梯度常只有约`1e-8`至`2.5e-5`，与labeled梯度的余弦方向在正负之间波动。因此瓶颈已从“总体伪标签不准”转向“长尾风险与单位伪标签梯度利用不足”。

冻结anchor logits缓存保持truth-blind，只读取确定性clean view和稳定`base_index`，并保存/恢复Python、NumPy、CPU及CUDA RNG。修复AMP dtype一致性后，r3+r4交叉E6结果为稳定epoch训练时间下降6.014%、U吞吐提高5.756%、计入构建后E6净耗时下降4.708%；它是工程加速证据，不是E200性能证据。

## 5.原始FastTrust16行完整实验

正式矩阵配置见[FastTrust16 r2配置](../configs/phase1_adv3b02_fasttrust16_s392002_20260821_r2.json)，终态原始证据见[FastTrust16正式实验报告](../automation_reports/CV-SincNet/phase1_adv3b02_fasttrust16_s392002_20260821_r2/report.md)。16/16条均完成E200，16/16条均有最终checkpoint，64/64份Clean/三LEO评测严格加载epoch200 checkpoint并覆盖60,000个样本，无fallback、missing key、unexpected key或shape mismatch。

|候选|Clean|Clear|Low-elev|Rain|LEO均值|四场景floor|
|---|---:|---:|---:|---:|---:|---:|
|R0 Scratch Control U256|84.662|70.757|68.312|67.315|68.794|50.675|
|R1 Core90 Control U256|85.152|75.345|73.380|72.243|73.656|58.525|
|R2 Fast H/M/L U256|84.225|75.022|73.098|72.073|73.398|58.808|
|R3 Fast H/M/L+U Proto U256|84.973|75.177|73.340|72.218|73.578|58.583|
|R4 Full U128|83.958|76.930|75.370|74.635|**75.645**|**62.883**|
|R4 Full U256|84.540|75.908|74.192|73.290|74.463|60.383|
|R4 Full U384|84.955|75.168|73.408|72.440|73.672|59.067|
|R4 No Class Cap U256|84.827|75.622|73.887|72.865|74.124|59.700|
|R4 No CrossRX U256|84.710|76.082|74.397|73.480|74.653|60.658|
|R4 No Nuisance U256|85.075|76.140|74.347|73.455|74.647|60.875|
|R4 No Prior U256|84.375|75.873|74.050|73.225|74.383|60.333|
|R4 No Proto Evidence U256|84.557|76.072|74.323|73.475|74.623|60.700|
|R4 No Temporal U256|84.475|76.213|74.395|73.542|74.717|60.908|
|R4 No U Proto Update U256|84.163|75.675|73.973|73.178|74.276|60.600|
|R4 No U Sat ID U256|84.750|75.293|73.433|72.227|73.651|58.808|
|R4 Nuisance Detached U256|84.630|76.083|74.413|73.433|74.643|60.717|

### 5.1机制消融

差值均为前者减后者，单位为百分点。

|比较|ΔClean|ΔLEO均值|Δfloor|判定|
|---|---:|---:|---:|---|
|R0−R1|-0.490|-4.862|-7.850|Core90初始化是主要星地能力来源|
|R2−R1|-0.927|-0.258|+0.283|基础身份伪标签无平均收益|
|R3−R1|-0.178|-0.078|+0.058|U prototype接近无效|
|Full U256−R1|-0.612|+0.807|+1.858|完整包提高星地鲁棒性但损伤Clean|
|Full U256−No U Sat ID|-0.210|**+0.812**|**+1.575**|最强且方向一致的正机制|
|Full U256−No Class Cap|-0.287|+0.339|+0.683|class-balanced cap有效|
|No Temporal−Full|-0.065|+0.253|+0.525|temporal gate无正贡献|
|No CrossRX−Full|+0.170|+0.189|+0.275|cross-receiver项无正贡献|
|No Nuisance−Full|+0.535|+0.184|+0.492|nuisance项整体有害|
|No Proto Evidence−Full|+0.017|+0.160|+0.317|prototype evidence无正贡献|
|Full−No Prior|+0.165|+0.081|+0.050|source prior仅小幅正向|
|Full−No U Proto Update|+0.377|+0.188|-0.217|U prototype update作用混合|

### 5.2为什么固定50%选择没有成功

从E17开始，FastTrust候选每epoch都选择26,460/52,920条U样本，即固定50%。即使E17的`temporal_pass=0`，后续soft/candidate补位仍会填满身份配额；训练后期三头一致率和置信度又接近99.7%–99.9%，门控进一步丧失区分力。结果是：普通身份分支覆盖很大，却未超过R1；只消费strict hard的U星地分支反而形成明确收益。

因此，原始FastTrust的核心问题不是hard定义错误，而是把`identity_max_fraction`实现成了“尽量填满的目标配额”，使low reliability样本也持续承担身份损失，确认偏差抵消了高质量子集的收益。

## 6.U128与U256的详细区别

两者网络、初始化、epoch数、数据角色、损失、阈值和星地日程相同，唯一实验变量是`U_s`的mini-batch大小。由于每个epoch必须完整覆盖52,920条U样本，batch越小，U loader step越多，也会同步增加labeled batch和增强前向次数。

|指标|U128|U256|U128−U256|
|---|---:|---:|---:|
|Clean|83.958|84.540|-0.582pp|
|LEO均值|75.645|74.463|+1.182pp|
|四场景floor|62.883|60.383|+2.500pp|
|平均每epoch|406.3秒|271.9秒|+49.4%|
|总训练时长|22.58小时|15.11小时|+7.47小时|
|U吞吐|138.3样本/秒|206.6样本/秒|-33.1%|
|峰值保留显存|2.06GiB|2.40GiB|-0.34GiB|

U128更好并不是因为使用了更多U样本；两者每epoch都完整覆盖同一批U数据。差异来自更小batch带来的更多optimizer update、更高梯度噪声和更细的参数更新节奏。它同时改变了有效计算预算，因此是“训练预算/优化动力学+方法”的联合变化，不是纯FastTrust机制增益。

选择建议：

- 追求单点星地性能且可接受约50%更长epoch、Clean下降时，用U128作高预算参考。
- 做正式机制对照、批量实验或后续多seed时，用U256作为效率主配置。
- U384虽只需10.37小时，但相对U256的LEO均值下降0.791、floor下降1.317个百分点，不建议作为精度主配置。

## 7.FastTrust家族后续实验结果

### 7.1RC4与QB阶段

|阶段/候选|有效coverage|Clean|LEO均值|floor|结论|
|---|---:|---:|---:|---:|---|
|RC4 LogitQ P3严格H|9.08%|85.500|73.979|58.558|该矩阵内最佳|
|RC4 LogitQ P5 H+P-set+P-cond|16.04%|84.822|73.696|58.050|比P3低0.283pp LEO|
|RC4 LogitQ P6 H+P+N|21.86%|84.783|73.750|58.175|比P3低0.229pp LEO|
|QB2|15%|85.168|73.592|58.017|预算工程闭合，缺同run反事实|

这组结果证明：把coverage从9.08%扩大到16%–22%没有提升星地性能；信息类型和梯度预算比覆盖规模更重要。

### 7.2QB3单seed五行矩阵

|候选|Clean|Clear|Low-elev|Rain|LEO均值|场景floor|receiver×LEO floor|
|---|---:|---:|---:|---:|---:|---:|---:|
|C0无U身份|84.4000|75.4883|72.9800|72.2083|73.5589|72.2083|57.2417|
|C1严格H|84.6417|75.3667|72.7867|72.1450|73.4328|72.1450|57.4333|
|C2 H+P-set|84.8867|75.6667|73.0567|72.4333|73.7189|72.4333|57.0750|
|C3 H+P-set+P-cond|**85.2017**|**75.8517**|**73.2033**|**72.5450**|**73.8667**|**72.5450**|57.6167|
|C4 U feature anchor|84.3150|75.3967|72.9050|72.1483|73.4833|72.1483|**57.9167**|

C3相对C0为Clean+0.802、LEO均值+0.308、场景floor+0.337、receiver×LEO floor+0.375个百分点。它是首次在同row内让Clean与三种LEO场景同时提高的优化伪标签候选，但单seed增益量级小，且绝对floor仍显著低于原始FastTrust Full U256。

### 7.3C0↔C2↔C3三seed验证

|seed|候选|Clean|LEO均值|场景floor|receiver×LEO floor|
|---:|---|---:|---:|---:|---:|
|392002|C0|84.4000|73.5589|72.2083|57.2417|
|392002|C2|84.8867|73.7189|72.4333|57.0750|
|392002|C3|85.2017|73.8667|72.5450|57.6167|
|713101|C0|84.7050|73.6006|72.7333|58.4000|
|713101|C2|85.5433|73.8494|72.9883|58.5583|
|713101|C3|85.7067|73.8411|72.9267|58.2750|
|713102|C0|84.0067|73.1267|72.2100|57.7833|
|713102|C2|84.1333|73.3117|72.4667|57.4417|
|713102|C3|84.6383|73.5389|72.6717|58.0250|

三seed配对平均贡献为：

|因果增量|ΔClean|ΔLEO均值|Δ场景floor|Δreceiver×LEO floor|一致性|
|---|---:|---:|---:|---:|---|
|H+P-set：C2−C0|+0.4839|+0.1980|+0.2456|-0.1167|前三项3/3为正；最差单元仅1/3为正|
|P-conditional：C3−C2|+0.3278|+0.1222|+0.0850|+0.2806|seed713101的后三项为负|
|合计：C3−C0|+0.8117|+0.3202|+0.3306|+0.1639|Clean与LEO均值3/3为正；最差单元2/3为正|

三seed候选均值为：C0的LEO均值/floor=`73.4287/57.8083`，C2=`73.6267/57.6917`，C3=`73.7489/57.9722`。这说明QB3提高了平均性能，但没有稳定修复最困难的receiver×LEO单元。完整报告见[C0/C3多seed报告](../automation_reports/CV-SincNet/phase1_adv3b02_fasttrust_qb3_c0c3_ms_e200_20260826_r1/report.md)和[C2多seed报告](../automation_reports/CV-SincNet/phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826_r1/report.md)。

## 8.有效、无效与尚未证实的方法组件

### 8.1已经获得正收益证据

|组件|最直接证据|结论边界|
|---|---|---|
|Core90初始化与LEO_WEAK底座|R1相对R0的LEO均值+4.862、floor+7.850pp|主要星地能力来源，不是伪标签收益|
|严格U星地身份CE|Full U256相对No U Sat ID的LEO+0.812、floor+1.575pp|星地特化强证据，Clean−0.210pp|
|class-balanced cap|Full相对No Class Cap的LEO+0.339、floor+0.683pp|单seed同row正证据|
|source prior|Full相对No Prior的Clean/LEO/floor均小幅为正|作用小，需更多seed|
|QB3 H+P-set|三seedClean/LEO/场景floor均3/3为正|最差receiver floor均值下降0.1167pp|
|QB3 P-conditional|三seed平均继续提高，但非逐seed一致|不能作为稳定floor修复晋级|
|anchor cache|交叉E6训练段约6%加速|只属工程优化，尚无缓存E200实测|

### 8.2已被证实无用或当前实现有害

|方向|证据|判定|
|---|---|---|
|基础FastTrust H/M/L身份伪标签|R2相对R1的Clean−0.927、LEO−0.258pp|当前实现无用|
|固定50%低可靠回填|E17起恒定26,460/52,920，基础分支不增益|应取消|
|单独U prototype更新|R3相对R1近似持平；Full对No U Proto Update作用混合|无稳定正贡献|
|temporal gate|No Temporal比Full的LEO+0.253、floor+0.525pp|当前组合中有害/冗余|
|cross-receiver loss|No CrossRX比Full更好|当前组合中无用|
|nuisance分支|No Nuisance的Clean/LEO/floor均更高|当前实现有害|
|prototype fusion evidence|No Proto Evidence比Full更好|当前组合中无用|
|RC4简单扩大P/N coverage|P5/P6均不如严格H P3|覆盖扩张无效|
|QB3一般U feature anchor|C4只提高最差单元，Clean和LEO均值下降|不能代替身份信息|
|U384精度主配置|速度提高但LEO与floor显著下降|不建议用于精度主线|

### 8.3尚未完全证实

- U128的绝对优势混入更多optimizer update，不能视为纯batch统计或纯方法收益。
- C3的P-conditional三seed平均为正，但seed713101的LEO和floor增量为负。
- QB3的平均增益约0.2–0.3个百分点，缺逐样本配对预测，不能做严格McNemar检验。
- 当前高质量伪标签的梯度强度偏弱、方向波动，尚未证明预算与共享表征更新已经匹配。

## 9.训练成本、异常与交付完整性

原始FastTrust16矩阵中，U256主体每行约15小时，U128约22.58小时，U384约10.37小时。QB3通过更稀疏的source-heavy评测等优化把E200压缩到约9小时，但峰值allocated从约3.20GiB升至约6.08GiB，端到端加速并未随eval batch增大继续改善。

QB3仍存在低频非有限梯度跳步，每行约占41,400个step的0.0845%–0.0966%；C0与C4同样出现，首个异常定位到共享Sinc前端`low_hz_`梯度，不是P-set/P-conditional特有。后续`torch.sinc`与FP32滤波器合成修复已经完成，但没有热补丁或改写既有正式结果。

交付完整性：

- 原始FastTrust r2：16/16条E200，64/64份最终评测，状态`ANALYZED`。
- QB3单seed：5/5条E200与四场景终评，状态`ANALYZED`。
- C0/C3多seed：新增4条正式E200完整闭合。
- C2多seed：新增2条正式E200完整闭合，三seed同row分析完成。
- 所有方法结论均来自完整结构化epoch记录、完整训练/评测日志和冻结checkpoint终评；未用日志tail替代终态分析。

## 10.最终科学裁决

### 10.1关于“FastTrust是否增强泛化”

没有足够证据支持。原始FastTrust主要以Clean下降换取LEO提升；QB3虽然三seed平均同时提高Clean和LEO，但receiver×LEO最差单元仍不稳定，且绝对floor低于原始FastTrust强星地配置。因此，当前只能说FastTrust家族学到了有价值的source-only身份信息，并能小幅提高平均星地性能，不能说模型的通用跨接收机泛化能力已经增强。

### 10.2星地信道提升最大的选择

- 绝对单点冠军：`R4_FAST_FULL_U128`，LEO均值75.645%、floor62.883%。适合高预算星地鲁棒性参考。
- 同batch可归因机制：`U星地身份CE`，Full U256相对No U Sat ID的LEO+0.812、floor+1.575个百分点，是最强单因素证据。
- 工程默认选择：`R4_FAST_FULL_U256`或其简化版，保留严格U星地身份、class cap和source prior，取消固定50%低可靠回填及无效复杂组件。
- 更均衡但收益较小的研究方向：QB3 C2/C3。它们改善平均指标，却尚未恢复原始FastTrust U256的极端receiver floor。

### 10.3下一候选建议

下一候选应以“提高单位U身份梯度质量，而不是增加覆盖率”为原则：保留H+P-set和严格U星地身份；关闭P-conditional作为默认、关闭temporal/cross-RX/nuisance/prototype evidence；对receiver×class小样本单元做source-only风险退回R；取消low reliability补满50%；对实际共享参数做轻量梯度归一化/上限约束；工程上启用已验证的FP16 anchor cache和Sinc FP32修复。

晋级门应同时要求：多seed LEO均值和场景floor为正，receiver×LEO floor相对同seed控制不下降，Clean退化受控。达不到该门槛时，FastTrust保持“星地增强研究分支”，不替代Phase1默认模型。

## 11.实现与证据索引

- 原始路由、RC4校准与损失：[code/cvsrffi/muse_ssdg.py](../code/cvsrffi/muse_ssdg.py)
- 训练主链、FastTrust损失、日程、缓存与遥测：[code/SSDG/train_ssdg.py](../code/SSDG/train_ssdg.py)
- 有界域混淆：[code/cvsrffi/bounded_domain_confusion.py](../code/cvsrffi/bounded_domain_confusion.py)
- truth-last质量分析：[code/cvsrffi/phase1_pseudolabel_quality.py](../code/cvsrffi/phase1_pseudolabel_quality.py)
- 质量审计入口：[code/scripts/phase1_fasttrust_vselect_quality.py](../code/scripts/phase1_fasttrust_vselect_quality.py)
- 原始FastTrust设计：[docs/superpowers/specs/2026-08-21-adv3b02-fasttrust-pseudo-design.md](superpowers/specs/2026-08-21-adv3b02-fasttrust-pseudo-design.md)
- 原始FastTrust16完整结果：[automation_reports/.../fasttrust16.../report.md](../automation_reports/CV-SincNet/phase1_adv3b02_fasttrust16_s392002_20260821_r2/report.md)
- QB3单seed结果：[automation_reports/.../qb3_bc_hps.../report.md](../automation_reports/CV-SincNet/phase1_adv3b02_fasttrust_qb3_bc_hps_e200_s392002_20260824_r1/report.md)
- QB3 C0/C3多seed结果：[automation_reports/.../qb3_c0c3_ms.../report.md](../automation_reports/CV-SincNet/phase1_adv3b02_fasttrust_qb3_c0c3_ms_e200_20260826_r1/report.md)
- QB3 C2多seed与质量审计：[automation_reports/.../qb3_c2_ms.../report.md](../automation_reports/CV-SincNet/phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826_r1/report.md)

## 12.追溯闭合与设计一致性

本报告追溯项共9项：已验证9项，待处理0项，部分实现0项，拒绝0项。原始FastTrust的设计条款已经进入实际训练主链并由16行矩阵验证；RC4/QB/QB3属于在原设计基础上的后续扩展，不是原始FastTrust设计的隐式替代。

需要特别说明的设计一致性边界是：原始设计允许soft/candidate身份分支并设置50%身份上限，实际代码将上限实现为尽量补满的配额。这在实现层面与配置一致，却在科学效果上被证实过强。因此，这不是“代码没有实现设计”，而是“设计—实现一致，但实验否定了该策略”。
