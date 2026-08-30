# Phase1 HCF-DG架构与实验设计

## 1.目标与适用范围

本设计将Phase1从ADV3B02/ADV3B03的双分支、边缘域对抗和多辅助损失训练，重构为HCF-DG（Hierarchical Counterfactual Factorized Domain Generalization，分层反事实因子化域泛化）。实现严格覆盖用户提供的《Phase1深层域泛化重构方案：从ADV3B02转向HCF-DG》，不把HCF-DG简化为ADV3B02上的残差插件。

HCF-DG只处理Phase1地面source-domain DG。训练、校准、模型选择和候选晋级仅使用`R_s`中的`L_s/U_s/V_cal/V_select`，不读取Phase2 capsule、support、query、truth、prototype或split。冻结候选后的目标接收机评估是一次性零适配确认，不能反馈选种、调参、重训或选择性重跑。

首轮数据配置沿用当前新划分：

- 数据：`Dataset_WigSig/ManySig.pkl`；
- source receivers：`1,3,4,6,8`；
- 训练日期：day1、day2、day3；
- source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；
- 输入：`2×256`原始IQ；
- source与target receiver严格不交。

## 2.设计原则

HCF-DG用三个可直接检验的约束替代边缘域混淆：

1.由其他source receiver形成的TX类别结构必须识别被留出的source receiver；
2.替换receiver、day或channel环境因素后，TX身份预测必须保持；
3.训练目标必须约束最差receiver、day、TX×receiver及TX×day风险，而不只优化平均风险。

V1、V2和V3按报告规定递进。V1验证单主干、矩形batch、LODO分类和公共—特定分解；V2加入反事实环境传输、分层HDRO、内容条件原型和单前向星地增强；V3只在V2通过后加入小型PhaseDelta/DSQ残差模块。

## 3.网络结构

### 3.1单共享身份主干

HCF-DG复用当前`lite_d` identity backbone一次，输出160维`z_id`和公共TX logits。正式推理仅保留：

```text
IQ -> F_shared -> P_id -> z_id -> W0
```

HCF-DG不实例化第二个完整domain backbone，不保留domain enhancer，不把训练期环境头或低秩特定头导出为推理依赖。A1保留`single_parameter_matched`作为参数量控制；A2以后使用真正的HCF单主干，不以容量补偿层冒充环境因子化。

### 3.2轻量环境编码器

环境编码器读取共享主干中间特征的stop-gradient副本与低成本物理统计：

```text
Q_env(stop_gradient(h_early), q_phys(x)) -> z_rx, z_day, z_channel
```

三个子表示各16维，总环境表示48维。`q_phys(x)`只进入环境分支，初始包含RMS/AGC、IQ协方差、平均相位斜率、平滑频谱包络和相位噪声方差。环境分支分别预测receiver、day和channel，不把`rx_day`当成唯一不可分解域标签。

微环境`z_micro`不进入首轮正式矩阵。只有source-only TX probe证明候选微环境的TX泄漏不高于预登记上限后，才能在后续独立候选中启用。

### 3.3公共—特定低秩头

训练期分类头为：

```text
W_e = W0 + U diag(a_rx + a_day + a_channel) V^T
```

首轮固定rank=4、specific dropout=0.5。每个有标签样本同时计算公共头CE和特定头CE；source选模、最终checkpoint评估和部署只使用`W0`。报告记录`Delta_spec=A(W_e)-A(W0)`，若该差距超过3pp，则判定特定头形成域捷径。

## 4.TX×环境矩形batch

正式HCF batch固定为：

```text
6 TX × 4 domains × 4 samples = 96 samples
```

四个domain至少覆盖三个不同receiver，每个被采样TX在每个domain中均有样本。episode按receiver/day/channel轮换，不在同一batch同时执行三个leave-out任务：

- leave-one-receiver-out：0.65；
- leave-one-day-out：0.225；
- leave-one-channel-out：0.125。

不完整cell先按同一TX重采样，仍不足时使用mask；少于两个support domain的TX不计算LODO，不从其他TX填补。采样器必须输出query domain、support mask、有效TX mask和可复现episode seed。

## 5.跨域LODO身份分类

对每个episode选定一个全局query domain。构造类别原型时严格排除该domain的全部样本：

```text
mu_k^{-e_q}=normalize(mean(z_j | y_j=k,e_j!=e_q))
```

query样本只与这些support-domain原型计算温度缩放余弦CE。单个query的真实TX只能用于有监督loss，不得参与其原型。实现必须显式返回参与原型的domain集合和有效query数，测试要证明query domain被排除。

V1使用普通LODO原型。V2加入内容条件原型：内容键由归一化幅度包络、自相关、去绝对增益粗频谱、去均值低阶时序统计和局部能量变化组成，不进入身份分类器。若某类没有足够近的内容支持，则回退到普通LODO原型。

## 6.反事实环境传输

V2在单个时频融合特征层实施有界低秩FiLM式环境传输：

```text
delta_e = z_env(j) - z_env(i)
[gamma,beta] = G_phi(delta_e)
h_cf = (1+clip(gamma,-a,a))*Norm(h_i)+clip(beta,-b,b)
```

初始只执行same-TX receiver swap；课程随后启用day swap，最后启用channel或联合swap。反事实loss包含：

- `L_CF_ID`：公共头保持TX身份；
- `L_CF_INV`：反事实`z_id`与原`z_id`保持余弦一致；
- `L_CF_ENV`：环境预测切换到目标环境；
- `L_style`：反事实特征均值和方差接近目标环境样本。

调制器只作用于一个融合层，gamma/beta均有界；V2不得增加波形decoder或多层调制器。

## 7.分层尾部风险

V2按`receiver`、`day`、`channel`、`TX×receiver`、`TX×day`和`TX×channel`聚合逐样本公共头风险。小组风险向其父组收缩：

```text
R_tilde_g = n_g/(n_g+kappa)*R_g + kappa/(n_g+kappa)*R_parent(g)
L_HDRO = tau*logsumexp(R_tilde_g/tau)
```

首轮固定`kappa=8`、`tau=0.25`、最小有效组样本数=4。HCF-DG启用HDRO时关闭旧Group CE、FISHR proxy、REx和open-world CVaR，防止重复尾部目标。

## 8.损失与关闭项

正式V2总损失为：

```text
L = L_ID
  + 0.40*L_LODO
  + 0.15*L_CF
  + 0.10*L_HDRO
  + 0.15*L_CSD
  + 0.05*L_FAC
```

`L_FAC`由receiver/day/channel环境监督、条件receiver对抗和`z_env`上的TX对抗组成。条件receiver对抗输入`[z_id,onehot(y)]`，最终GRL强度限制在0.05以内。

HCF-DG首轮关闭FastTrust、身份伪标签、entropy minimization、prototype memory、open-world feature、proxy unknown、soft unknown mixup、source episode three-sigma、Group CE、FISHR proxy、旧全局GRL大权重、训练期Phase2 prototype导出和历史残差分支。

## 9.HCF-DG专用星地增强

用户已明确要求严格采用报告方案。HCF-DG不继承ADV3B02/ADV3B03的`clean+satellite`拼接双前向训练；每个主训练batch在模型前向前完成单视图替换：

```text
70%样本：clean
30%样本：mixed_orbit
```

被增强样本仍只占一个batch位置，所有96个样本执行一次共享主干前向。增强器同时输出channel/scenario、CFO、phase noise、SNR、multipath和elevation bin，供`z_channel`监督使用。clean样本使用独立clean channel标签。

clean/satellite成对一致性不是A0–A9默认组成。若后续独立消融启用，只允许每4个optimizer step在25%的batch样本上增加一次配对计算，并单独报告增量时间。

最终checkpoint仍必须分别评估clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。训练使用`mixed_orbit`不改变正式三场景测试口径。

## 10.优化预算与阶段

正式比较以optimizer updates为预算，不以epoch为主预算。

### 10.1 V1快速筛选

- 总更新：4000；
- batch：96；
- AdamW；
- backbone LR=`1e-4`；
- 新增head LR=`3e-4`；
- 前5%更新线性warm-up，随后cosine decay至`1e-6`；
- CosFace margin在前20%更新从0线性增加至0.30；
- AMP开启。

### 10.2 V2正式候选

- Stage0：700个环境编码器预训练update，可读取`L_s+U_s`的receiver/day元数据；
- Stage1：1200个公共头和低秩特定头update；
- Stage2：2100个矩形batch、LODO和条件域对抗update；
- Stage3：1700个反事实传输和HDRO update；
- Stage4：600个低对抗权重收尾update；
- 总计6300个update，其中主干只在Stage1–4更新。

Stage0离线计算`q_phys`，不让`U_s`完整身份前向定义主训练长度。主训练期间环境分支最多每4个主update额外更新一次。V2在总进度50%后冻结Sinc和第一个时域卷积块；冻结点固定，不读取target结果，也不按候选表现动态移动。

## 11.实验矩阵

### 11.1 快速筛选A0–A5

固定seed为`392001/392002/392003`。从五个source receiver中，仅用source环境统计选择一个风格中心receiver和一个风格最远receiver作为两个LODO fold。具体做法是：在`L_s+U_s`上按训练集均值和标准差标准化`q_phys`，计算每个receiver的环境质心；中心receiver取对其他四个质心平均欧氏距离最小者，最远receiver取对source全局质心欧氏距离最大者，若并列则receiver ID升序。选择结果在任何训练启动前写入矩阵，不读取目标receiver。

|ID|结构|新增机制|
|---|---|---|
|A0|ADV3B02闭集精简双分支控制|关闭FastTrust/open/unknown及旧辅助loss，4000-update同预算控制|
|A1|单主干parameter-matched|验证完整domain backbone是否只提供容量|
|A2|HCF单主干+48D环境编码器|验证轻量因子化|
|A3|A2+矩形batch|验证有效跨域监督几何|
|A4|A3+普通LODO分类|首个功能性DG候选|
|A5|A4+rank-4公共—特定头|验证域捷径吸收|

矩阵规模为`6 candidates×2 folds×3 seeds=36 rows`，每行4000 update。N607每张GPU最多并发2个训练进程；调度只决定并发，不改变行配置。

### 11.2 深层外推A6–A9

A6–A9按报告顺序在A5结构上逐项增加机制。A0–A5的source-only结果用于解释各机制贡献和确定后续确认候选，但低性能不停止既定A6–A9方法验证。晋级判断仍要求相对A0的LODO mean提高、LODO floor不下降且source clean下降不超过0.5pp；GPU-hours目标按历史ADV3B02 E200的35%–45%单独报告，不与科学门槛混成一个分数。

|ID|新增机制|
|---|---|
|A6|same-TX receiver counterfactual swap|
|A7|receiver/day/channel课程式组合swap|
|A8|分层HDRO|
|A9|内容条件LODO原型|

A6–A9先沿用2 folds×3 seeds筛选。冻结前两名后，执行5个source LORO fold×3 seeds×6300 update确认。

### 11.3 小型残差A10–A12

只有A8或A9通过上述门槛后才运行：

|ID|新增机制|
|---|---|
|A10|8–16通道PhaseDelta，零初始化gate|
|A11|8–16通道多尺度DSQ，零初始化gate|
|A12|PhaseDelta+DSQ gated fusion及raw identity bypass|

A10–A12不得增加第二频域骨干，必须单独报告相对通过的HCF-DG主体的边际收益和资源增量。

### 11.4 最终冻结

结构、loss和超参数全部由source-only矩阵冻结后，使用8个预登记seed在全部source receiver和day1/2/3上训练。最终seed只按完整source`V_select`的clean、三种LEO场景、LODO mean/floor及预登记调和指标冻结。目标接收机只允许一次零适配、prediction-first确认。

## 12.诊断、评估与资源记录

每行至少保存：

- clean及`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`；
- 每个source LODO receiver/day的mean、floor和逐TX结果；
- 条件receiver泄漏`A_RX|TX`；
- 环境表示TX泄漏`A_TX(z_env)`；
- 公共头与特定头差距`Delta_spec`；
- 反事实身份保持率和环境切换率；
- 跨域漂移比`R_drift`与最小类间margin；
- step time、samples/s、dataloader wait、峰值显存、GPU利用率、forward/backward占比、总update和总GPU-hours。

正式性能选择必须使用同一候选、fold、seed和checkpoint的same-row指标。不得跨行拼接clean、LEO mean、floor或资源最优值。

## 13.代码边界

HCF-DG采用独立包，避免继续膨胀`train_ssdg.py`和`model_dual_cvsincnet.py`：

```text
code/cvsrffi/phase1_hcfdg/config.py
code/cvsrffi/phase1_hcfdg/model.py
code/cvsrffi/phase1_hcfdg/sampler.py
code/cvsrffi/phase1_hcfdg/losses.py
code/cvsrffi/phase1_hcfdg/satellite.py
code/cvsrffi/phase1_hcfdg/trainer.py
code/cvsrffi/phase1_hcfdg/metrics.py
code/scripts/launch_phase1_hcfdg_matrix_20260830.py
code/tests/phase1_hcfdg/
```

现有`lite_d` identity backbone和WiSig source-role builder作为依赖复用。旧ADV3B02/ADV3B03入口保持行为不变；HCF-DG通过独立入口调用新trainer，不用大量布尔开关侵入旧训练循环。

## 14.测试与完成定义

实现采用TDD。聚焦测试必须覆盖：

1.矩形batch形状、receiver覆盖、缺失cell mask和确定性；
2.LODO原型严格排除query domain；
3.内容键和环境键不进入公共身份输入；
4.低秩头只在训练期生效，推理仅使用`W0`；
5.反事实gamma/beta有界并保留batch形状；
6.HDRO父级收缩、最小组mask和有限梯度；
7.单前向batch严格为70%clean、30%mixed_orbit且只调用一次身份主干；
8.4000/6300 update预算、warm-up、cosine和margin日程；
9.A0–A12矩阵依赖和晋级顺序；
10.Phase1入口拒绝target receiver、Phase2路径或query输入；
11.真实checkpoint无query smoke能完成严格重建；
12.最终clean及三种LEO场景artifact闭合。

只有逐项追踪表中所有强制条目达到`verified`，且正式N607行达到`ARTIFACTS_COMPLETE`后，才能声明对应HCF-DG版本实现并完成实验。低性能是科学结果，不是技术停止条件。

## 15.声明边界

HCF-DG结果只能声明为WiSig/ManySig地面代理数据上的source-domain跨接收机、跨日期及LEO压力域泛化结果。它不是Phase2 few-shot适应，不是真实卫星数据，不是真实在轨验证，也不构成Phase3协同、unknown确权或新类注册结论。
