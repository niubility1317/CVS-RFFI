# CVS-RFFI全量消融实验设计：Phase1与Phase2

日期：2026-07-28
状态：`PREREGISTRATION_DRAFT / NO_NEW_RESULT / NOT_YET_RUN`

## 0.本文件解决什么问题

本文件给出CVS-RFFI从地面训练到部署注册的完整消融体系。其目标不是罗列“去掉某个模块”的零散实验，而是回答四类可证伪问题：

1. Phase1的跨接收机表征、弱标注学习、尾部风险和星地外推机制是否各自提供独立作用；
2. Phase2的旧类接收机适配和新类注册是否都得到验证，而不是只优化其中一项；
3. Phase1改进是否能传递到Phase2，Phase2改进是否只是在弥补较弱的Phase1；
4. 所谓轻量部署是否同时满足精度、状态大小、注册成本、逐query成本和数值一致性，而不是只报告线性头MAC。

“全量”在本文件中指覆盖全部产生梯度或改变部署状态的主张相关机制、关键超参数、回退路径、资源路径和两阶段交互，不表示把所有因素做不可承受的全笛卡尔积。实验采用“模块主效应→模块内部机制→敏感性→独立确认”的分层漏斗。

## 1.消融等级与证据边界

|等级|含义|能否进入论文主张|
|---|---|---|
|M：mandatory|主方法、同权限基线和每个核心模块的整体消融|完成独立确认后可以|
|I：internal|核心模块内部的单机制拆解|完成预登记确认后可以|
|S：sensitivity|标签率、K、新类数、权重、阈值和秩等敏感性|用于说明稳定区间，不用于事后选最优|
|D：diagnostic|无安全门、Role-Oracle、不受约束增益等机理诊断|只能解释缺口，不能作为正式部署结果|
|X：prohibited|读取query truth、query role、类别配额、clean target或重复信道观测|禁止运行或禁止进入正式证据|

每个arm必须绑定唯一的`ablation_id`、Git commit、Phase1 bundle hash、capsule ID、split ID、支持集hash、query ID hash、场景分配hash和随机种子。未绑定这些字段的结果只能作为历史线索。除明确写成累加链或因子实验的arm外，所有消融均采用one-factor-at-a-time：只改变表中声明的因素，其他模块、数据、训练预算和选择规则保持冻结。

## 2.不可改变的协议

### 2.1阶段输入与权限

|阶段|允许读取|禁止读取|
|---|---|---|
|Phase1|source labeled、source unlabeled、source receiver/day metadata、source侧物理启发信道增强|任何target receiver样本、target support、target query、target性能|
|Stage2-A|冻结Phase1 deployment bundle、固定target received IQ|target标签、target support、source/clean样本|
|Stage2-B|冻结bundle、旧类K-shot support及标签、固定received IQ|新类support、query truth、source/clean样本|
|Stage2-C|冻结bundle、旧类与新类K-shot support及标签、固定received IQ|query truth、source/clean样本、query批次类别统计|

Phase2遵守`protocol_schema=p2_min_v1`：

- 每条物理记录在Phase2前只对应一个固定允许的LEO received-IQ观测；
- K-shot中的K是K条独立物理support，不是同一物理记录的K个增强、FFT视图或信道重采样；
- FFT、RF统计、均衡或其他确定性数学视图只能读取同一份固定received IQ，不增加K；
- 每条query独立在全部已注册类别上执行argmax；
- query及其任何视图都不能更新模型、原型、协方差、阈值、归一化统计或缓存；
- 禁止query role、真实批次类别数、类别配额、Hungarian、OT全局重分配和跨query图；
- Phase2运行时不能读取source/clean样本。唯一例外是与checkpoint共同封存的INT8多样本聚合Phase1域×类知识；
- old/new成员关系可以在support状态构造中用于固定任务均衡，但不能用于query路由或使用两套角色专属判决器。

### 2.2数据复用与重新验证

同一`capsule_id`、`split_id`和`p2_min_v1`在builder给出`phase2_data_status=VALIDATED_ONCE`后，所有方法arm直接复用。方法、权重、秩、分类头、量化形式和资源预算变化不得触发数据重验证。

只有以下变化需要新capsule或重新验证对应切片：

- received-IQ字节改变；
- physical ID、receiver/TX集合改变；
- 场景分配改变；
- K、support/query划分或协议schema改变。

Phase2信道组件拆解若需要改变received-IQ字节，必须建独立capsule，不能把同一物理样本的多个信道版本当作多份正式观测。

### 2.3论文显示名

论文表格统一使用`clear-sky`、`low-elevation`和`rainy-link`。内部artifact可保留实现枚举，但论文正文不出现“weak”“弱化”或“弱信道”等命名。

## 3.冻结参考方法

### 3.1Phase1完整参考

Phase1完整arm记为`P1-FULL`，包含：

1. 共享Sinc/HF前端；
2. 160维identity branch与160维domain branch；
3. identity的time/frequency/PA路径；
4. domain的time/frequency/PA/DAC/RCN路径；
5. TX监督、domain监督、GRL、协方差正交和same-TX跨域中心一致性；
6. receiver-day分位阈值EMA伪标签闭环；
7. prototype、角几何、`L_zid`、`L_coretail`和三类soft-mix边界；
8. same-TX cross-domain MixStyle、source-domain episode和source侧LEO压力CE；
9. GroupCE与FishR训练稳定器；
10. 200轮冻结调度及不可变deployment bundle。

历史`ADV3B02_CORE90_SOFT_E200`使用过不同的数据比例。正式消融不能直接把该历史checkpoint当作当前协议的`P1-FULL`；完整arm及其所有对照必须在当前`0.07/0.63/0.30`划分上重新训练。

权重为0的satellite consistency、对模型无梯度的prototype domain-align/push或仅存在于旧配置中的字段不属于当前有效方法，不设置“关闭它”的虚假消融。

### 3.2Phase2完整参考

Phase2完整arm记为`P2-FULL`，对应当前RTB-IDR/D92闭环：

1. 160维identity、96维FFT和32维RF联合特征；
2. identity块与128维辅助块分别归一化，辅助块固定权重4，再做288维整体归一化；
3. 从封存INT8域×类中心构造类无关扰动谱并校正量化噪声底；
4. K>2时使用一步Cauchy权重稳健化support中心，K≤2时精确回退；
5. 逐类标准化Ledoit–Wolf协方差，旧类任务与新类任务固定0.5/0.5等权；
6. full covariance和block3 covariance双几何；
7. support留一交叉拟合、logit RMS尺度归一化和类级可靠性融合；
8. 有界Fisher残差、逐类Pareto门与联合原子检查；
9. 单一等先验仿射头；
10. 双层残差INT8权重、FP16块尺度/截距和FP32对角metric。

K≤2时稳健中心、任务均衡协方差和Fisher残差按当前设计回退。K1/K2的零差值用于验证回退正确性，不能被解释为这些模块在其激活区间无效。

## 4.公共数据因子与随机性

### 4.1Phase1

正式标签比例\(\rho_{\mathrm{label}}=0.10\)时：

\[
f_L=0.07,\qquad f_U=0.63,\qquad f_V=0.30.
\]

标签率敏感性使用：

\[
\rho_{\mathrm{label}}\in\{0.005,0.01,0.02,0.05,0.10\},
\]

\[
f_L=0.70\rho_{\mathrm{label}},\qquad
f_U=0.70(1-\rho_{\mathrm{label}}),\qquad
f_V=0.30.
\]

每个正式Phase1 arm至少使用5个独立训练seed。所有paired arm共享相同split、label mask、初始化seed和数据顺序seed；checkpoint只能由source validation选择。任何target receiver指标都不得进入checkpoint选择、早停或超参数选择。

### 4.2Phase2

最低确认因素为：

|因素|正式取值|
|---|---|
|target receiver|`20-1`,`3-19`,`7-14`,`7-7`,`8-8`|
|K-shot|\(\{1,2,5,10\}\)|
|新类数\(C_n\)|\(\{5,10,20\}\)|
|LEO场景|clear-sky、low-elevation、rainy-link|
|support/query seed|至少5个fresh seeds|
|new-class set draw|至少3个独立draw|
|旧类数|6|

Stage2-C的完整确认量为：

\[
5\ \text{receivers}\times
4\ K\times
3\ C_n\times
5\ \text{seeds}\times
3\ \text{class draws}
=900
\]

个注册row/arm；每个row含3个场景分层单元，因此是2700个场景单元/arm。

既有`713102–713106`结果已经被观察，只能作为历史开发或代码回归证据，不能再次命名为fresh confirmation。正式确认seed必须在历史索引中检查为未使用后再封存。`new_class_draw_seed`必须与`support_seed`分开记录，避免把类别难度和支持抽样波动混为一谈。

### 4.3分层筛选矩阵

用于机制筛选的代表slice固定为：

\[
(K,C_n)\in
\{(1,20),(2,20),(5,20),(10,5),(10,20)\}.
\]

筛选阶段使用5个receiver、3个development seeds和1个预登记class draw，共75个注册row/arm、225个场景单元/arm。筛选seed与确认seed不得重合。筛选结果只决定是否进入确认，不直接进入论文最终因果表。

## 5.指标与统计

### 5.1Phase1主指标

|类别|必须报告|
|---|---|
|身份识别|overall、strict UDU、per-receiver、receiver floor、min-class|
|表征泄漏|`z_id→receiver/day`线性probe、`z_dom→receiver/day`probe、`z_dom→TX`probe|
|SSL|pseudo precision、coverage、per-receiver/day precision和coverage、coverage离散度、标签翻转率|
|角几何|类内角半径Q90/Q95、类均衡tail CVaR、最小类间角、overflow率|
|信道鲁棒性|clean与三个LEO profile的同row准确率、mean、floor、clean-to-stress drop|
|训练稳定性|nonfinite batch、失败run、最佳epoch、最终epoch、seed方差|
|资源|参数量、FLOPs/MAC、训练时长、峰值VRAM、deployment bundle字节数|

Phase1模块的主终点按假设预先固定：

- A双表征：strict UDU、receiver floor和`z_id→receiver/day`泄漏；
- B伪标签：等coverage下pseudo precision、strict UDU和receiver floor；
- C尾风险：min-class、receiver floor和Q95角半径；
- D反事实外推：strict UDU、LEO stress floor和clean-to-stress drop。

### 5.2Phase2主指标

|阶段|必须报告|
|---|---|
|Stage2-A|无target标签旧类准确率、per-class old、receiver/scenario floor|
|Stage2-B|注册前旧类准确率\(A_o^{pre}\)、min-old、per-class old、相对Stage2-A增益|
|Stage2-C|注册后旧类准确率\(A_o^{post}\)、新类准确率\(A_n\)、\(H\)、forgetting\(F=A_o^{pre}-A_o^{post}\)、min-old、min-new|
|竞争结构|old→new、new→old混淆、per-class confusion、receiver/scenario floor|
|算法行为|fallback counts、full/block权重、Fisher接受数、原子回滚数、失败闭合数|
|量化|最大/均值logit误差、argmax flip rate、FP32—量化预测一致率|
|资源|bundle/state字节、注册时间、峰值RSS/VRAM、闭式拟合数、MAC等价上界、单query分类头MAC、端到端query时延|

Phase2任何表都必须把\(A_o^{pre}\)、\(A_o^{post}\)、\(A_n\)、\(H\)、\(F\)、min-old和min-new放在同一row。不能从不同arm分别摘取最好的旧类、新类和H。

### 5.3统计单位与检验

1. 配对单位是相同receiver、seed、K、\(C_n\)、class draw和scenario下的完整row，不是单个packet；
2. 主结果报告paired差值、95%置信区间和原始绝对值；
3. 使用receiver为第一层、seed/class draw为第二层的分层bootstrap，建议10000次重采样；
4. 每个模块家族内部的多个子消融使用Holm校正，不跨无关家族制造一个总p值；
5. 同时报告均值、标准差、中位数、最差receiver和最差class，显著性不能替代效应量；
6. 失败row、零预测、fallback和回滚必须作为结果保留，不能静默删除；
7. 不预设一个对所有模块通用的“成功阈值”。每个模块按其预登记主终点解释，并完整披露其他指标上的代价。

## 6.Phase1消融

### 6.1论文必须完成的第一层

|ID|改动|因果问题|等级|
|---|---|---|---|
|P1-FULL|完整Phase1|参考|M|
|P1-SUP|相同backbone预算、只用labeled source与TX监督/CosFace|完整CVS是否超越纯监督训练|M|
|P1-A0|参数量匹配单embedding，移除显式domain表征与解耦|双表征收益是否只是容量增加|M|
|P1-B0|关闭伪标签CE和熵项|无标签闭环的整体作用|M|
|P1-C0|关闭prototype、`L_geo`、`L_zid`、`L_coretail`和soft-mix|角几何与尾部风险组的整体作用|M|
|P1-D0|关闭MixStyle、source episode和LEO压力CE|身份保持外推课程的整体作用|M|

第一层正式预算为6个arm×5个seed=30次完整训练。`P1-A0`必须参数量匹配；若只删除一条分支而不补偿容量，只能回答“完整模型与小模型的差异”。

### 6.2模块A：物理分解式身份—域双表征

|ID|唯一变化|隔离对象|主要指标|等级|
|---|---|---|---|---|
|P1-A1|identity和domain均去掉PA path|PA非线性视图|UDU、hard-TX、参数量|I|
|P1-A2|去掉frequency path|频谱不对称视图|UDU、频谱扰动敏感性|I|
|P1-A3|domain branch仅去掉DAC|接收链I/Q失衡承接|domain probe、identity leakage|I|
|P1-A4|domain branch仅去掉RCN|raw-IQ接收统计承接|同上|I|
|P1-A5|关闭domain监督，保留结构|显式域可辨识约束|domain probe、UDU|I|
|P1-A6|仅关闭GRL|对抗去域|identity leakage、UDU|I|
|P1-A7|仅关闭协方差正交|二阶解耦|交叉协方差、UDU|I|
|P1-A8|仅关闭same-TX跨域中心一致性|类条件域一致性|类内跨域距离、floor|I|
|P1-A9|同时关闭GRL、正交和中心一致性|双结构本身与解耦训练的差异|全部A指标|I|
|P1-A10|identity/domain使用完全对称的物理输入|非对称保留/抑制策略|泄漏与识别联合变化|I|
|P1-A11|Sinc/HF前端替换为参数量匹配普通Conv前端|物理前端本身|UDU、FLOPs|I|
|P1-A12|共享前端改为两套独立前端并匹配总参数|共享低层表征的作用|UDU、泄漏、参数量|S|
|P1-A13|time-only、time+frequency、time+PA、full四级嵌套|identity多视图增量价值|UDU、per-class、FLOPs|S|

需要新增：参数量匹配单embedding、DAC与RCN独立开关、对称输入构型和普通Conv匹配构型。没有独立开关前，不得用不同历史模型替代严格消融。

### 6.3模块B：receiver-day可信伪标签闭环

完整参考使用EMA teacher、receiver-day内0.86分位阈值、0.92–0.97截断、domain gate、temporal gate和strong-view agreement。

|ID|唯一变化|隔离对象|主要指标|等级|
|---|---|---|---|---|
|P1-B1|全局阈值，阈值调到与FULL相同总体coverage|分域阈值而非样本量|precision、域间coverage、UDU|M|
|P1-B2|只按receiver分位，不分day|day条件|per-day precision、UDU|I|
|P1-B3|只按day分位，不分receiver|receiver条件|per-receiver precision、floor|I|
|P1-B4|关闭domain gate|域一致性门|污染率、identity leakage|I|
|P1-B5|关闭temporal gate|时间稳定门|翻转率、precision|I|
|P1-B6|关闭strong-view agreement|增强一致性门|增强后错误率、UDU|I|
|P1-B7|EMA teacher替换为当前student|teacher平滑|阈值抖动、precision/coverage|I|
|P1-B8|关闭正熵项，仅保留pseudo CE|熵项贡献|类分布、collapse、UDU|I|
|P1-B9|关闭pseudo CE，仅保留熵项|pseudo监督与置信锐化差异|collapse和类偏置|D|
|P1-B10|分位数\(\{0.80,0.86,0.90\}\)|接收强度敏感性|precision—coverage曲线|S|
|P1-B11|取消0.92–0.97截断|极易/极难域保护|per-domain precision/floor|S|
|P1-B12|SSL起始epoch\(\{101,131,151\}\)|warm-up敏感性|收敛、pseudo精度|S|
|P1-B13|EMA decay\(\{0.99,0.999,0.9999\}\)|teacher时间尺度|抖动、最终UDU|S|
|P1-B14|teacher低扰动视图与student强扰动视图改为同一视图|视图不对称的必要性|precision、UDU|D|

`P1-B1`是该模块最关键的公平对照。若没有matched coverage，只能说明分域策略与全局策略不同，不能把收益归因于校准本身。

### 6.4模块C：尾部风险约束的角原型几何

|ID|唯一变化|隔离对象|主要指标|等级|
|---|---|---|---|---|
|P1-C1|TX CE/CosFace+prototype，其他角风险关闭|均值几何基线|overall、Q90/Q95、floor|M|
|P1-C2|在C1上加入`L_zid`和`L_coretail`|尾部风险相对均值几何|tail CVaR、min-class|M|
|P1-C3|FULL中仅关闭prototype pull|跨epoch中心记忆|中心漂移、UDU|I|
|P1-C4|FULL中仅关闭`L_geo`|闭集角margin|最小类间角、混淆|I|
|P1-C5|FULL中仅关闭`L_zid`|身份空间紧致和尾部项|Q95、floor|I|
|P1-C6|FULL中仅关闭`L_coretail`|源域留一类边界风险|overflow、min-class|I|
|P1-C7|FULL中仅关闭soft-mix|类间虚拟边界|边界混淆、类均衡边界错误率|I|
|P1-C8|`L_zid`内关闭跨域SupCon|跨域正对约束|跨域类内距离|I|
|P1-C9|`L_zid`内关闭角半径项|显式半径上界|Q90/Q95|I|
|P1-C10|`L_zid`内关闭类均衡CVaR|尾部而非均值|tail CVaR、floor|I|
|P1-C11|soft-mix分别去掉CE、energy或vacuum项|边界整形内部机制|边界能量、min-class|I|
|P1-C12|core quantile\(\{0.80,0.90\}\)|核心集合强度|core recall、overflow|S|
|P1-C13|coretail CVaR fraction\(\{0.20,0.30\}\)|风险敏感度|floor—overall权衡|S|
|P1-C14|所有角风险从epoch 1满权重启用|阶段调度的必要性|早期collapse、最终floor|D|

`P1-C1→P1-C2→P1-FULL`形成预登记累加链；它比从多个历史候选中挑单项最优更能解释均值几何、尾部约束和边界整形的增量作用。

### 6.5模块D：身份保持的源域反事实外推

|ID|唯一变化|隔离对象|主要指标|等级|
|---|---|---|---|---|
|P1-D1|只开same-TX cross-domain MixStyle|风格外推|UDU、receiver floor|M|
|P1-D2|只开source-domain episode|显式留域外推|UDU、episode loss|M|
|P1-D3|只开LEO压力CE|物理启发压力|stress mean/floor、clean drop|M|
|P1-D4|receiver挑战开/关×LEO压力开/关的2×2|互补或冲突|clean与stress同row|M|
|P1-D5|取消MixStyle后期退火|课程调度|后期收敛、floor|I|
|P1-D6|same-TX cross-domain改为same-TX same-domain|跨域混合本身|UDU、类内距离|I|
|P1-D7|改为普通随机配对MixStyle|身份保持约束|识别退化、混淆|D|
|P1-D8|只在一个插入点启用MixStyle|多层注入|UDU、计算量|S|
|P1-D9|source episode去掉held-domain query项|episode外推监督|UDU、floor|I|
|P1-D10|source episode去掉episode soft-mix项|episode边界项|边界混淆|I|
|P1-D11|episode radius cap\(\{25^\circ,33^\circ,40^\circ\}\)|episode半径敏感性|Q95、floor|S|
|P1-D12|LEO CE权重\(\{0.34,0.68,1.00\}\)|压力强度|clean—stress Pareto|S|
|P1-D13|LEO CE起始epoch\(\{40,80,120\}\)|压力课程时机|收敛与Pareto|S|

#### 6.5.1源侧信道机制拆解

|ID|训练压力组成|目的|等级|
|---|---|---|---|
|P1-H0|不施加LEO压力|物理压力基线|M|
|P1-H1|仅AWGN/SNR变化|噪声贡献|I|
|P1-H2|H1+CFO|频偏增量贡献|I|
|P1-H3|H2+相位噪声|振荡器扰动贡献|I|
|P1-H4|H3+衰落/多径|传播结构贡献|I|
|P1-H5|完整已锁定overlay|参考|M|
|P1-H6|分别移除clear-sky、low-elevation或rainy-link训练profile|场景覆盖贡献|S|

该链用于source侧训练压力。若在Phase2 target IQ上做相同物理分量拆解，必须使用独立、重新验证的capsule，并标为外部有效性实验；不得让同一物理记录产生多个正式target观测。

### 6.6全局训练稳定器与调度

|ID|唯一变化|主要问题|等级|
|---|---|---|---|
|P1-E0|同时关闭GroupCE和FishR|全局稳定器整体作用|I|
|P1-E1|仅关闭GroupCE|困难域均衡|I|
|P1-E2|仅关闭FishR|跨域梯度方差|I|
|P1-E3|所有复合损失从epoch 1启用且不warm-up|阶段课程是否必要|D|
|P1-E4|保留起始点但取消线性warm-up|warm-up形状|S|
|P1-E5|关闭AMP并保持其他设置|数值一致性，不是方法创新|D|
|P1-E6|对已证明有效的损失组做0.5×/1×/2×单因素敏感性|局部稳定区间|S|

### 6.7标签受限性

`P1-LABEL`使用第4.1节的5个标签率。每个点至少3个筛选seed，\(\rho=0.10\)和论文声称的低标签关键点使用5个独立确认seed。报告：

- strict UDU—标签成本曲线；
- receiver floor—标签成本曲线；
- pseudo precision/coverage—标签成本曲线；
- 相对`P1-SUP`的增益；
- 固定总source pool下的labeled/unlabeled绝对样本数。

若只测试10%标签点，论文只能写“在10%训练池标注比例下”，不能泛化为普遍的label-efficient方法。

### 6.8Phase1同权限基线

|基线|控制对象|
|---|---|
|相同backbone supervised-only CosFace|无SSL/域约束的训练基线|
|参数量匹配单embedding DANN/GRL|双表征相对单路域对抗|
|MixStyle-only|反事实课程相对通用风格混合|
|全局阈值Mean Teacher/FixMatch式SSL|receiver-day校准相对通用SSL|
|source-only同split跨接收机RFFI方法|外部方法比较|

历史`ADV2 avg.`若不是同split、同seed和同参数预算，不进入严格主表。

## 7.Phase2消融

### 7.1Stage2-A与Stage2-B必须独立成表

|ID|状态构造|作用|等级|
|---|---|---|---|
|P2-S2A|冻结Phase1直接部署，不读target support|零标签跨接收机基线|M|
|P2-S2B-PROTO|旧类support余弦prototype|最小旧类适配基线|M|
|P2-S2B-DIAGOFF|完整旧类状态但关闭20步对角metric|旧类metric贡献|M|
|P2-S2B-FULL|完整Stage2-B旧类状态|参考|M|
|P2-S2B-STEPS|metric步数\(\{0,5,10,20\}\)|适配成本—收益|S|
|P2-S2B-BLOCK|metric作用于identity-only、三块对角或完整288维|metric作用范围|S|

Stage2-B只评估旧类，不借用Stage2-C的H或新类指标证明其有效。Stage2-C表必须保留同一row的`S2B-old（注册前）`，以度量注册造成的遗忘。

### 7.2同权限主基线

全部基线共享同一Phase1 bundle、capsule、physical IDs、support/query、class set、seed和全类逐样本argmax。

|基线|主要控制对象|等级|
|---|---|---|
|Cosine nearest centroid/ProtoNet|最小参数化注册|M|
|Euclidean ProtoNet|归一化余弦先验|M|
|Single qKNN|局部support记忆|M|
|Diagonal LDA|逐维尺度|M|
|Ledoit–Wolf pooled LDA|自动收缩但无task balancing|M|
|Full/block shrinkage LDA without robust center|D92几何主干|M|
|冻结轻量adapter+统一head|support训练相对闭式注册|M|

CSIL、MoPC-HR、Orthogonal Incremental SEI及SDA复现若数据权限或生命周期不同，放在“不同权限外部比较”表，不能与上述基线混成单一排名。

### 7.3模块A：联合特征

|ID|特征或归一化变化|隔离对象|等级|
|---|---|---|---|
|P2-A0|identity160 only|辅助特征整体贡献|M|
|P2-A1|identity+FFT96|FFT单独贡献|M|
|P2-A2|identity+RF32|RF统计单独贡献|M|
|P2-A3|identity+FFT96+RF32|完整参考|M|
|P2-A4|辅助权重\(\beta_{\mathrm{aux}}\in\{0,1,2,4,8\}\)|固定权重4敏感性|S|
|P2-A5|不做块级归一化|尺度控制|D|
|P2-A6|FFT和RF分别归一化后再拼接|辅助块内部尺度|I|
|P2-A7|取消最终288维归一化|统一球面几何|D|
|P2-A8|打乱FFT/RF坐标但保持边缘分布|辅助结构负对照|D|

`P2-A4`只能在development matrix选择一次，并用fresh confirmation确认。不能按receiver、K、场景或新类数动态选择\(\beta_{\mathrm{aux}}\)。

### 7.4模块B：地面扰动谱与稳健中心

|ID|唯一变化|隔离对象|等级|
|---|---|---|---|
|P2-B0|不读地面扰动谱，普通support均值|地面聚合先验+稳健中心整体贡献|M|
|P2-B1|不读地面扰动谱，按完整identity残差能量做Cauchy加权|地面谱相对通用support稳健中心|M|
|P2-B2|地面扰动谱+Cauchy权重|完整参考|M|
|P2-B3|地面扰动谱内将Cauchy改为Huber、Tukey或trimmed mean|稳健权重形式|I/S|
|P2-B4|不减INT8 quantization noise floor|量化噪声校正|M|
|P2-B5|使用全部正特征方向|当前有效秩|I|
|P2-B6|按90%/95%/99%谱能量截断|扰动谱秩敏感性|S|
|P2-B7|用相同秩随机正交基替代地面扰动基|真实地面结构负对照|D|
|P2-B8|稳健平移从identity块扩展到全部288维|identity-primary设计|I|
|P2-B9|K激活阈值改为\(K>1\)、\(K>2\)、\(K>3\)|小K稳定区间|S|
|P2-B10|封存FP32聚合中心与INT8聚合中心对照|量化先验损失|D|
|P2-B11|读取地面谱但强制所有support等权|实现null control，应与普通均值逐logit闭合|D|

`P2-B10`只有在两套bundle都于target访问前独立封存、各自有hash和schema时才合法。Phase2不得临时访问未封存FP32 source状态。

### 7.5模块C：协方差与旧/新任务均衡

|ID|协方差变化|隔离对象|等级|
|---|---|---|---|
|P2-C0|\(\Sigma=I\)|协方差建模整体贡献|M|
|P2-C1|经验对角协方差+固定ridge|逐维尺度基线|M|
|P2-C2|全部类pool后一次Ledoit–Wolf|逐类估计与类均衡|M|
|P2-C3|逐类Ledoit–Wolf后按全部类等权，即D81型|固定旧/新任务均衡|M|
|P2-C4|旧任务/新任务固定0.5/0.5，即D92|完整参考|M|
|P2-C5|旧任务权重\(\{0.25,0.50,0.75\}\)|任务权重敏感性|S|
|P2-C6|逐类估计前不标准化|尺度标准化|I|
|P2-C7|OAS替代Ledoit–Wolf|收缩估计器形式|S|
|P2-C8|无收缩经验full covariance|高维小样本不适定负对照|D|
|P2-C9|仅旧类或仅新类协方差|两个任务的方差贡献|D|
|P2-C10|固定收缩强度\(\{0.25,0.50,0.75,1.0\}\)|自动收缩相对手工ridge|S|

`P2-C5`只用于展示敏感性。不得根据query结果为每个receiver、K或新类数挑选不同task weight。

### 7.6模块D：full/block双几何与support交叉拟合融合

|ID|唯一变化|隔离对象|等级|
|---|---|---|---|
|P2-D0|full-only|完整协方差几何|M|
|P2-D1|block3-only|块对角几何|M|
|P2-D2|full/block固定0.5/0.5|数据驱动可靠性|M|
|P2-D3|全类共享一个LOO可靠性权重|类级可靠性|I|
|P2-D4|类别级in-sample可靠性，不留一|cross-fitting必要性|D|
|P2-D5|类别级support-LOO可靠性|完整参考|M|
|P2-D6|取消logit RMS尺度归一化|分支标尺对齐|I|
|P2-D7|可靠性映射用\(\exp(-\ell)\)而非\(\exp(-K\ell)\)|证据温度|S|
|P2-D8|K≥5时比较LOO与固定2-fold|交叉拟合结构|S|

in-sample arm预期乐观，只用于说明为何需要交叉拟合，不能作为可部署优胜arm。

### 7.7模块E：有界Fisher残差与安全门

|ID|Fisher/门控变化|隔离对象|等级|
|---|---|---|---|
|P2-E0|关闭Fisher residual|残差整体贡献|M|
|P2-E1|Fisher作用于全部行，不设安全门|安全门整体作用|D|
|P2-E2|只做逐类Pareto门，不做联合原子检查|原子竞争安全|D|
|P2-E3|逐类Pareto门+联合原子检查|完整参考|M|
|P2-E4|仅要求TP不降|FP约束作用|D|
|P2-E5|仅要求FP不增|TP约束作用|D|
|P2-E6|允许无严格改善的tie通过|strict-improvement条件|S|
|P2-E7|有界\(\gamma=\beta/(\beta+\nu)\)改为未封顶\(\beta/\nu\)|有界增益安全|D|
|P2-E8|Fisher秩用90%/95%类间能量或全部有效秩|低秩敏感性|S|
|P2-E9|固定增益\(\gamma\in\{0.25,0.5,1.0\}\)|自适应增益|S|

`P2-E1/E2/E4/E5/E7`必须显式标记`NON_DEPLOYABLE_DIAGNOSTIC`。它们不能因query准确率更高而取代安全版本。

### 7.8模块F：量化编译与部署资源

|ID|状态格式|隔离对象|等级|
|---|---|---|---|
|P2-F0|FP32权重+FP32截距|数值参考|M|
|P2-F1|FP16权重+FP16截距|半精度折中|M|
|P2-F2|单层INT8+FP16尺度|第一层量化|M|
|P2-F3|双层残差INT8+FP16尺度/截距|完整参考|M|
|P2-F4|每tensor一个scale|粗粒度量化|I|
|P2-F5|每class一个scale|类级量化|I|
|P2-F6|每class×block scale|当前粒度|M|
|P2-F7|截距FP32与FP16|截距压缩|S|
|P2-F8|当前“解码FP32点积”与真实整数融合kernel|存储压缩与计算加速区别|S|

每个量化arm同时报告state bytes、logit误差、argmax flip、数值闭合、解码成本和真实端到端时延。没有目标硬件整数kernel时，只能声称存储压缩，不能声称INT8推理加速。

### 7.9小K回退

|ID|K|预期激活|验证内容|等级|
|---|---:|---|---|---|
|P2-K1|1|稳健中心、task covariance、Fisher均回退|输出存在、hash闭合、与基线精确一致|M|
|P2-K2|2|同上|证明边界条件，而非只测K1|M|
|P2-K5|5|全部模块激活|低K有效性|M|
|P2-K10|10|全部模块激活|稳定区间与资源|M|

如果K1/K2未与fallback基线逐logit或逐prediction闭合，先修复实现，不进入性能解释。

### 7.10连续注册、顺序和持久状态

该组属于顶级期刊所需的生命周期实验。它扩展当前一次性注册证据，但不改变主协议。

|ID|注册方式|必须报告|等级|
|---|---|---|---|
|P2-G0|一次性加入15类|最终旧/新、状态和成本|M|
|P2-G1|5+5+5三session|每session旧类、历史新类、最新新类|M|
|P2-G2|3种预登记到达顺序|顺序敏感性|M|
|P2-G3|保存全部历史target support|性能上界与内存|D|
|P2-G4|只保存声明的持久状态|真实部署路径|M|
|P2-G5|每次从持久状态+当前support增量更新|持续可用性|M|
|P2-G6|每次用全部历史support重编译|重编译参考|D|
|P2-G7|模拟注册构造失败并原子rollback|状态安全|M|
|P2-G8|跨target receiver切换后沿用旧状态|错误状态负对照|D|
|P2-G9|receiver切换时重建合法新状态|跨接收机生命周期|M|

到达顺序在查看query结果前锁定；不得用“困难类最后到达”等query派生排序。

### 7.11鲁棒性与安全补充

|ID|压力|说明|等级|
|---|---|---|---|
|P2-R0|support样本中0/1/2条预登记物理异常值|稳健中心抗异常|D|
|P2-R1|预登记的5%/10%support标签噪声|支持污染敏感性|D|
|P2-R2|support shot不均衡|当前等K假设外推，需单独协议|D|
|P2-R3|LEO profile分层结果|场景稳健性|M|
|P2-R4|独立capsule中的信道参数失配|模拟器外推|S|
|P2-R5|未注册发射机/拒识|当前闭集任务之外，需独立威胁模型|D|
|P2-R6|replay/forgery|不能由当前识别准确率替代，需要FAR/FRR/EER|D|

`P2-R5/R6`没有完成前，论文不得声称已完成攻击检测、恶意判定或开放集安全认证。

`P2-R0/R1/R2/R4`均改变正式support或received-IQ条件，必须使用独立诊断manifest或独立capsule并重新完成相应验证；其结果不混入标准`p2_min_v1`主排名。

## 8.Phase1×Phase2联合消融

### 8.1最小2×2因子

本节的`P2-PROTO`专指旧类与新类共享同一个余弦nearest-centroid头、全部注册类共同argmax的Stage2-C同权限基线，不是只评估旧类的`P2-S2B-PROTO`。

|Phase1|Phase2|回答的问题|
|---|---|---|
|P1-SUP|P2-PROTO|最小端到端基线|
|P1-FULL|P2-PROTO|Phase1单独贡献|
|P1-SUP|P2-FULL|Phase2能否补偿较弱表征|
|P1-FULL|P2-FULL|完整CVS|

四个cell必须使用对应Phase1模型自身封存的bundle。不得把`P1-FULL`的地面聚合中心嫁接给`P1-SUP`checkpoint。

### 8.2Phase1模块的下游传递

对`P1-A0/P1-B0/P1-C0/P1-D0`分别生成合法deployment bundle，并在同一Phase2 screening matrix上运行固定`P2-FULL`。该实验回答Phase1模块是否改善最终旧类适配和新类注册，但不能取代Phase1 source-only主消融。

### 8.3Phase2模块的固定上游

所有`P2-A0`至`P2-F8`核心消融统一使用同一fresh-confirmed `P1-FULL` bundle。不得给不同Phase2 arm选择不同Phase1 checkpoint。

### 8.4交互解释

对最小2×2报告：

\[
\Delta_{P1}=Y_{\mathrm{P1FULL,P2base}}-Y_{\mathrm{P1SUP,P2base}},
\]

\[
\Delta_{P2}=Y_{\mathrm{P1base,P2FULL}}-Y_{\mathrm{P1base,P2base}},
\]

\[
\Delta_{\mathrm{int}}
=
Y_{\mathrm{FULL,FULL}}
-Y_{\mathrm{FULL,base}}
-Y_{\mathrm{base,FULL}}
+Y_{\mathrm{base,base}}.
\]

对\(A_o^{pre}\)、\(A_o^{post}\)、\(A_n\)、\(H\)、forgetting、min-old和min-new分别计算，不把它们压成一个不透明总分。

## 9.执行顺序与算力预算

### 9.1T0：本地实现与协议检查

在任何N607发布前完成：

- 每个arm只改变声明因素的配置diff测试；
- 参数量匹配测试；
- Phase2 query不可达负测试；
- K1/K2精确fallback测试；
- 全类逐样本argmax测试；
- 量化state无FP32 sidecar测试；
- prediction与truth scorer分离测试；
- 同capsule、support和seed的paired manifest测试。

T0不产生论文性能结果。

### 9.2T1：论文主消融

Phase1：

- `P1-FULL/P1-SUP/P1-A0/P1-B0/P1-C0/P1-D0`；
- 30次完整训练；
- 只使用source validation选checkpoint。

Phase2：

- 7个同权限基线；
- `P2-FULL`；
- 六个主模块家族的整体消融：`P2-A0/P2-B0/P2-C3/P2-D0/P2-D1/P2-D2/P2-E0/P2-F0–F3`；
- 先完成75-row screening；
- 所有拟写入核心贡献的arm必须进入fresh confirmation。

### 9.3T2：模块内部

只有T1证明模块整体具有稳定作用时，才运行对应I级子arm。否则记录“整体作用未成立”，不通过继续扫内部参数寻找局部正结果来挽救主张。

### 9.4T3：敏感性、生命周期与资源

包括标签率、\(\beta_{\mathrm{aux}}\)、任务权重、谱秩、K、\(C_n\)、class draw、连续注册、量化粒度和目标硬件资源。敏感性曲线使用预登记网格，不做query驱动贝叶斯优化或按slice选不同配置。

### 9.5T4：诊断

Role-Oracle、无安全门Fisher、随机扰动基、in-sample融合、FP32 source聚合上界和保存全部历史support都标为`NON_PROMOTABLE_DIAGNOSTIC`。这些结果只能定位差距，不能进入正式主排名或替换协议合法方法。

## 10.论文表格与图

### 10.1主文建议

|编号|内容|
|---|---|
|Table I|数据、receiver/TX、标签率、K、新类数、场景和权限|
|Table II|Phase1同权限基线及A0/B0/C0/D0整体消融|
|Table III|Stage2-A和Stage2-B旧类适配|
|Table IV|Stage2-C同权限基线和六个核心模块消融|
|Table V|参数、状态字节、注册成本、query成本和数值一致性|
|Fig. 1|两阶段流程与数据权限|
|Fig. 2|Phase1模块逐步累加waterfall|
|Fig. 3|标签率—UDU/receiver floor曲线|
|Fig. 4|K×新类数的H、forgetting、min-old/min-new热图|
|Fig. 5|旧类—新类Pareto图，点大小表示状态字节|
|Fig. 6|连续注册session曲线|

### 10.2补充材料

- Phase1 A/B/C/D/E全部内部消融；
- Phase2 feature、center、covariance、fusion、Fisher、quantization全部内部消融；
- per-receiver、per-class和per-scenario表；
- 全部失败、fallback和rollback记录；
- bootstrap分布与多重比较校正；
- 完整manifest、配置diff和artifact schema。

## 11.每个run必须保存的artifact

```text
run_id
ablation_id
evidence_level
git_commit
config_hash
phase1_bundle_hash
protocol_schema
capsule_id
split_id
phase2_data_status
receiver_id
train_seed
support_seed
query_seed
new_class_draw_seed
channel_assignment_hash
k_shot
old_class_ids_hash
new_class_ids_hash
support_physical_ids_hash
query_physical_ids_hash
predictions_hash
score_artifact_hash
scorer_receipt
all_primary_metrics
per_class_metrics
fallback_counts
fisher_gate_accept_counts
atomic_rollback_counts
quantization_error
state_bytes
registration_time
peak_memory
query_latency
exit_status
```

旧类与新类的明文类别ID可保存在受控manifest中；正式结果表使用稳定匿名ID或hash，避免后验挑选容易类。

## 12.禁止的“伪消融”

以下做法不产生合法证据：

1. 用历史不同split、不同seed或不同checkpoint的结果代替同run paired消融；
2. 去掉模块后同时改变参数量、训练轮数、数据量或checkpoint规则；
3. 用target query选择Phase1 checkpoint或Phase2超参数；
4. 从同一物理样本生成多个Phase2信道版本并把它们计作更多K-shot；
5. 根据query old/new role选择不同头、阈值或温度；
6. 根据真实batch类别数进行配额、Hungarian或全局重排；
7. 只报告overall而隐藏receiver floor、min-class或新类代价；
8. 把K1/K2精确fallback的相同结果描述为核心模块无效；
9. 把权重为0或无梯度的历史项包装成创新并做“关闭”实验；
10. 把FP32上界、Role-Oracle、clean-target或保留全部历史support的诊断结果混入正式排名；
11. 把一次运行的最好epoch、最好receiver、最好class draw或最好场景拼成一条不存在的“最优结果”；
12. 只报告线性分类头MAC并据此声称完整星载系统轻量或实时。

## 13.当前需要补的实现开关

正式执行前至少需要：

1. Phase1统一`ablation_id`与单因素配置生成器；
2. 参数量匹配单embedding和普通Conv对照；
3. DAC、RCN、GRL、orth、center consistency及各复合损失内部项的独立开关；
4. matched-coverage全局伪标签阈值构造器；
5. Phase2 feature/center/covariance/fusion/Fisher/quantization arm factory；
6. K1/K2逐logit fallback闭合器；
7. fresh seed和new-class draw注册表；
8. 连续session持久状态与原子rollback接口；
9. 独立resource profiler和目标硬件整数kernel；
10. 统一same-row scorer与论文表格生成器。

这些实现项完成、经独立审查达到`P0=0,P1=0`并提交Git后，才能为具体run ID冻结矩阵并交由唯一N607 runner发布。

## 14.最小可投稿闭环

若算力有限，不能删掉核心因果问题。最低闭环是：

1. Phase1 30次第一层训练；
2. Phase1标签率至少覆盖0.01、0.05、0.10；
3. Stage2-A零标签旧类表；
4. Stage2-B旧类适配表，覆盖K1/K2/K5/K10；
5. Stage2-C同权限基线、`P2-FULL`和六个主模块整体消融；
6. 5个target receivers、3个场景、5个fresh seeds和至少3个new-class draws；
7. Phase1×Phase2最小2×2；
8. FP32/FP16/单层INT8/双层INT8资源—精度表；
9. 5+5+5连续注册与至少3种到达顺序；
10. 所有主表给出paired置信区间、per-receiver、per-class、失败/fallback和完整资源口径。

在此闭环完成前，最稳妥的论文状态仍是“方法框架与诊断证据已形成，但核心消融、独立确认和星载资源证据尚不完整”。
