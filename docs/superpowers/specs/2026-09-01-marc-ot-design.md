# MARC-OT元学习式权重空间接收机校准设计

日期：2026-09-01

## 1. 目标与结论

本设计实现`MARC-OT`（Meta-learned Adaptive Receiver Calibration with Orthogonal Transport），用于CVS Phase2 Stage2-B旧类目标域适配。方法不是Source-Free版MRIOR，也不训练或保留临时目标分类头，而是把Phase1多接收机经验转化为可部署的分块权重校准先验：

$$
\theta_t=\theta_0+Bq_t+r_t.
$$

- `theta_0`：冻结Phase1基础模型；
- `Bq_t`：由Phase1接收机episode学习的分块权重校准初始化；
- `r_t`：只由合法target-old support驱动的残差更新；
- `theta_t`：Stage2-B输出并在query前冻结的域适应模型。

第一轮交付为完整软件路径和最小可证伪pilot，不将Target25、完整125或Stage2-C作为前置条件。只有最小pilot达到预登记科学门槛后才放量。

## 2. 协议与边界

1. Phase2只读取`p2_min_v1/VALIDATED_ONCE`目标received IQ、合法old-class support标签、冻结checkpoint、冻结类原型和允许的量化聚合Phase1摘要。
2. Phase2不得读取source/clean IQ、样本级source embedding、source BatchNorm状态、query truth/role、query类别数量或scorer输出。
3. 所有OT只发生在target support与冻结Phase1聚合摘要或权重基库之间。禁止跨query OT、query配额或全局重排。
4. Stage2-B只完成`phi_D`。Stage2-C默认冻结`phi_D`，另行训练注册状态`phi_R`。
5. query只能在模型、权重组合、残差、gate、学习率、归一化状态、插值系数和D92状态全部冻结后逐样本打开。
6. 预测先闭合，独立scorer之后按opaque token连接truth。
7. 类标签置换必须保持同一公式；不得按具体TX ID设置分支、权重或阈值。

## 3. 历史方法的继承与拒绝

|方法|继承|拒绝或降级为控制|
|---|---|---|
|MRIOR-SDA|全主干梯度必须足够强；新增`MRIOR-H/B/HB`控制接口|不在主方法中读取raw source、不使用全局DV-KL、不使用伪标签；历史HB结果不得倒填H/B|
|DADDA-SDA|类条件统计与低K收缩|不做无条件整体对齐，不以低K经验协方差直接驱动全空间更新|
|ProtoNet-CDA|temporary LOO/cross-fitted prototype监督|prototype不成为最终分类器，不将support记忆当域校准|
|SF-TAPFT|norm与后段block作为消融起点|不保留target head；D92重拟合后该head收益不可计入主方法|
|WISER-P3|包加载、query隔离、可微old-only D92、渐进解冻、`alpha=0`回退、truth-last scorer|替换全局dot辅助梯度投影；不复用已失败的int8中心流形作为唯一域状态|
|旧Meta-adapter|receiver/day/scene episode、FOMAML outer loop、functional fast state、低秩adapter与bundle设施|现有activation adapter和模块级Meta-SGD不等同于权重delta bank或任务条件化学习率|

## 4. Phase1权重校准基库

### 4.1 分块定义

首版允许下列identity路径block：

```text
t1,t2,t3,f1,f2,f3,time_projection,frequency_projection,fusion,identity_mapping
```

Sinc、domain branch、source classifier和所有非浮点buffer默认不进入基库。每个block由稳定参数名、shape、dtype和基础checkpoint绑定。

### 4.2 域专家delta

每个source receiver×scene任务从同一`theta_0`出发，冻结源分类头，在episode support上适配并由同域隐藏query选择：

$$
\Delta\theta_{d,b}=\theta_{d,b}^{*}-\theta_{0,b}.
$$

delta提取必须保证：未允许参数与buffer bitwise不变；所有值有限；任务键包含receiver、day/capture block、scene和K；support/query物理ID不交。

### 4.3 低秩权重基

每个block将任务delta堆叠并做确定性SVD：

$$
D_b\approx B_bQ_b,
$$

rank默认不超过`min(task_count-1,16)`，同时记录实际rank、重构误差和有效mask。第一轮不以bundle体积为优化目标；若重构误差超过配置上限，允许保留全有效秩而不是静默截断。

### 4.4 Bundle绑定

`WeightDeltaBank`必须序列化并校验：

- schema版本；
- base checkpoint identity；
- block参数名、shape、dtype和顺序；
- 任务键与有效mask；
- basis、task coefficients、尺度和rank；
- support encoder、gate/LR predictor状态；
- 禁止成员检查结果。

Phase2加载后整个bank保持冻结。

## 5. Support set域状态推断

`SupportSetEncoder`只读取合法old-class support。每个物理样本只计一次，增强view不得增加`n_eff`。

输入特征包括：

- 冻结`theta_0`各选定层的按类均值、对角方差和范数；
- time/frequency分支差异；
- 多view稳定性；
- 低维CFO/SFO、PSD包络和质量统计；
- K和有效样本mask。

采用permutation-invariant的`DeepSets`式编码：

$$
h=\rho\left(\frac1N\sum_i\phi(s_i)\right).
$$

输出：

```text
q_t: bank coefficient query
u_t: [0,1] uncertainty
gamma_b: [0,1] block gate
eta_b: bounded positive block learning rate
```

encoder对support行重排保持不变；对类标签置换只允许同步置换类条件统计，不改变共享公式。

## 6. Phase2初始权重与残差适配

### 6.1 预测初始化

每个block：

$$
\theta_{t,b}^{(0)}=\theta_{0,b}+(1-u_t)\gamma_{t,b}B_bq_t.
$$

组合前验证block geometry、有限性、相对漂移上限和基础checkpoint绑定。任何失败都回退到`theta_0`，不得部分应用未知状态。

### 6.2 主任务

源分类头`W_0`永久冻结。主损失至少包含：

$$
L_{task}=L_{frozen-head}+\lambda_{cf}L_{cross-fit}+\lambda_{loo}L_{LOO}+\lambda_{sup}L_{SupCon}.
$$

`L_cross-fit`与`L_LOO`只使用support held-out折；temporary prototype参与梯度但不持久化为最终分类器。

### 6.3 合法校准目标

首版校准目标只使用冻结Phase1聚合摘要和support：

- empirical-Bayes normalization融合；
- receiver/channel nuisance子空间的类条件均值、对角或低秩Bures-Wasserstein；
- support与权重基任务原型的熵正则OT匹配；
- Fisher或归一化L2-SP信任域；
- identity健康度和最差类风险。

K自适应统计：K=1只允许均值/尺度；K=2允许对角方差；K=5允许极低秩；K>=10允许配置rank的低秩统计。低K不得假装拥有满秩协方差。

### 6.4 分块主任务优先投影

现有WISER的全局dot投影不满足本设计。必须按block独立处理：

$$
\widetilde g_{cal}^{b}=g_{cal}^{b}-\frac{\min(0,(g_{cal}^{b})^Tg_{task}^{b})}{\|g_{task}^{b}\|^2+\epsilon}g_{task}^{b}.
$$

并限制：

$$
\|\widetilde g_{cal}^{b}\|\le r_b\|g_{task}^{b}\|.
$$

一块冲突不得改变另一块同向梯度。缺失梯度按零处理；非有限梯度立即失败，不使用`nan_to_num`、静默跳步或清零掩盖。

### 6.5 渐进开放与安全回退

阶段顺序：

1. norm affine、fusion、projection；
2. `t3/f3`及identity mapping；
3. `t2/f2`；
4. `t1/f1`；
5. Sinc仅保留为后续独立候选，首轮不开放。

阶段进入、早停、分支选择和插值只使用support cross-fit。插值网格必须含`alpha=0`；没有安全候选时精确恢复base model、dual和非浮点buffer。

## 7. Phase1元训练

复用现有`MetaEpisodeBatch`与FOMAML outer loop，但新增显式bank fast state：

1. receiver/day/scene作为伪目标任务；
2. inner loop只读episode support、冻结bank和基础模型；
3. 由encoder产生`q/u/gamma/eta`并组合初始delta；
4. 执行少步support残差更新；
5. 独立`query_adapt/query_guard`计算outer mean、receiver CVaR和worst-class目标；
6. outer梯度到达encoder、gate/LR predictor和允许学习的bank参数；
7. 保持一阶路径，不意外构建二阶图。

元episode至少覆盖receiver holdout、day/capture holdout、clean到三类LEO弱场景、LEO跨场景和K=`1/2/5/10/20`。首轮真实训练可先使用已有可闭合的K10路径；其他K的软件支持不得冒充真实结果。

## 8. 因果矩阵与晋级

第一轮只运行同一outer、seed、K10和三个LEO场景：

|arm|内容|
|---|---|
|R0|冻结`theta_0`直接部署|
|R1|冻结head＋target监督全主干残差适配，无bank|
|R2|R1＋LOO/cross-fit/SupCon|
|R4|R2＋support encoder预测bank初始化|
|R6|R4＋nuisance条件统计与support-to-bank OT|
|R8|R6＋分块梯度投影和Fisher/L2-SP信任域|

MRIOR-H/B/HB是宽权限或机制控制，必须显式标注数据权限；历史MRIOR-SDA结果不能拆解为这三个控制项。

主评价保持同一冻结`W_0`、冻结source prototype和target support prototype，并使用old-only D92作为P3 probe。报告`DA0_REG0/DA1_REG0`绝对Accuracy、BA、floor、macro-F1、NLL、per-class、help/harm、receiver/scene、effective rank、zero norm、block drift、训练/推理资源。

最小pilot晋级门槛：

- 三场景P3 BA中位提升至少3pp；
- 最差场景P3 BA不低于-0.5pp；
- P3 floor中位不下降，low-elev floor不下降；
- P1/P2任一场景下降不超过2pp；
- 至少两个场景help大于harm；
- zero-id为0，所有状态和梯度有限；
- support CV分数与query收益方向一致。

未过门槛即`ANALYZED/NO_PROMOTION_TO_TARGET25`，不得因低性能视为技术失败，也不得扩大step、seed或Target125。

## 9. 文件边界

新增：

- `code/cvsrffi/meta_weight_bank.py`
- `code/cvsrffi/meta_support_set_encoder.py`
- `code/cvsrffi/meta_weight_calibrator.py`
- `code/cvsrffi/meta_bank_inner_loop.py`
- `code/cvsrffi/meta_bank_trainer.py`
- `code/cvsrffi/meta_weight_bank_checkpoint.py`
- `code/cvsrffi/stage2_marc_ot.py`
- `code/cvsrffi/stage2_marc_ot_runner.py`
- `code/cvsrffi/stage2_marc_ot_pilot.py`
- `code/cvsrffi/stage2_marc_ot_scoring.py`
- `code/scripts/run_stage2_marc_ot_pilot.py`
- `configs/marc_ot_k10_pilot_20260901.json`
- 对应`tests/test_*.py`。

复用但不改变历史语义：

- `stage2_wiser_pilot.py`的包加载器；
- `stage2_wiser_p3.py`的cross-fit、风险和诊断纯函数；
- `stage2_wiser_runner.py`的support-safe插值语义；
- `stage2_binova_d92.py`的D92几何与精确拟合；
- 现有meta episode和FOMAML训练设施。

## 10. 验证要求

所有生产行为先写失败测试并观察预期失败。至少覆盖：

1. delta提取、block geometry、bitwise冻结和低秩重构；
2. support encoder的行置换不变与标签置换一致性；
3. gate/LR范围、有限性和无TX专属分支；
4. outer梯度到达encoder、gate/LR和允许的bank参数；
5. 分块冲突只投影冲突block；
6. support/package重排不改变fold、bank状态或prediction；
7. query批大小、顺序和其他query存在与否不改变逐样本预测；
8. `alpha=0`精确恢复浮点参数、dual和非浮点buffer；
9. query/truth/role/quota在训练和选择期间不可达；
10. prediction闭合和独立truth-last评分；
11. MRIOR-H/B/HB控制项不倒填历史结果；
12. config、CLI、bundle round-trip和真实checkpoint无query smoke。

## 11. 设计同构边界

首轮严格同构包括：权重delta bank、support set encoder、任务条件化gate/LR、bank初始化、support-only残差适配、分块梯度投影、渐进开放、`alpha=0`、最小R矩阵、query隔离和truth-last。

以下不作为首轮完成声明：Sinc解冻、满秩source协方差、跨query OT、Stage2-C注册适应、Target25/125真实结果、真实在轨验证。它们分别是后续候选、协议禁止项或需晋级后执行的实验。
