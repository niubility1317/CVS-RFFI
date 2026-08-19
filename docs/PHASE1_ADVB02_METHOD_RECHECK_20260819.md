# Phase1 ADVB02方法核对版：网络、损失、训练配置与星地信道增强

> 核对日期：2026-08-19
> 适用范围：Phase1、ADVB02方法族及代码/历史报告中对应的ADV3B02候选实现
> 核对重点：损失函数、神经网络、训练配置、星地信道增强、评测场景，以及默认训练场景与评测场景的区别

## 1. 先给结论

本文把用户所称的ADVB02作为方法族简称。代码和历史报告中常见的候选标识是<code>ADV3B02</code>，历史冻结候选的完整标识是<code>ADV3B02_CORE90_SOFT_E200</code>。这两个名称不能脱离具体代码提交、配置和checkpoint单独比较。

最重要的场景口径如下：

| 环节 | 正确场景 | 作用 | 是否参与梯度 |
|---|---|---|---|
| Phase1训练时的星地增强 | <code>mixed_orbit</code> | 对源域物理IQ生成星地扰动视图，做辅助监督/一致性训练 | 是 |
| Phase1正式卫星评测 | <code>leo_clear_weak</code>、<code>leo_low_elev_weak</code>、<code>leo_rain_weak</code> | 在固定弱LEO代理场景上评估鲁棒性 | 否 |
| 训练期间的目标域访问 | 不允许 | 不能使用目标接收机、目标日期或正式评测truth做训练/选择 | 不适用 |

用户口语中的<code>mix_orbit</code>，代码注册表中的规范字符串是<code>mixed_orbit</code>。命令、配置和报告应写成<code>mixed_orbit</code>，否则可能找不到注册表项。

因此，本文采用下面这句作为ADVB02的正式简述：

> ADVB02在Phase1使用源域样本，并以<code>mixed_orbit</code>作为默认星地信道增强；训练完成后，在不回传梯度、不更新模型状态的条件下，分别使用<code>leo_clear_weak</code>、<code>leo_low_elev_weak</code>和<code>leo_rain_weak</code>进行卫星代理评测。

## 2. 本次复核发现并修正的错误

| 需要复核的说法 | 代码/配置核对结果 | 本文修正 |
|---|---|---|
| SSDG训练使用<code>CosineAnnealingLR</code> | <code>code/SSDG/train_ssdg.py</code>中创建的是<code>AdamW</code>，没有创建LR scheduler；通用<code>code/train.py</code>另有余弦调度，不能自动归因给SSDG/ADVB02 | ADVB02的SSDG训练写成“AdamW，无scheduler”；只有明确走通用<code>code/train.py</code>时才单独说明余弦调度 |
| 历史<code>resolved_config.json</code>能代表默认卫星训练场景 | 历史<code>ADV3B02_CORE90_SOFT_E200</code>配置包含显式的<code>leo_*_weak</code>训练schedule覆盖 | 将其标为“历史覆盖变体”，不能当作默认<code>mixed_orbit</code>方法口径 |
| 总损失可以写成<code>closed_scale * L_closed + open_scale * L_open</code> | 当前SSDG实现实际是<code>L_closed + L_open</code>，没有这两个总尺度项 | 删除未在代码中确认的<code>closed_scale/open_scale</code> |
| FISHR直接计算网络参数梯度方差 | 当前实现使用logit梯度代理<code>p-one_hot</code>，按域比较方差 | 改写为“logit-gradient proxy”，不夸大为参数梯度方差 |
| <code>leo_rain_weak</code>一定开启了大气雨衰 | 当前弱LEO残差配置中<code>enable_atmospheric_fading=false</code>、<code>apply_path_loss_to_iq=false</code> | 明确说明雨天是场景标签和参数组合；在该弱残差路径中不会通过大气表额外施加雨衰 |
| 历史集中式配置中的<code>wisig_train_ratio=0.2</code>就是当前Phase1协议比例 | 当前项目协议使用角色比例<code>.07/.63/.15/.15</code>；历史集中式配置的<code>.2</code>是旧实验配置，不能覆盖当前协议 | 将历史数据比例和当前协议分开写 |

## 3. Phase1任务与数据边界

### 3.1任务定义

Phase1是源域弱标注/半监督域泛化阶段。模型在源域WiSig/ManySig物理IQ上学习身份判别特征，同时学习域不变表示、身份/域因素解耦、卫星代理扰动鲁棒性，以及面向开放世界的特征空间约束。

当前项目协议的角色比例为：

- <code>0.07</code>：有标签源域样本；
- <code>0.63</code>：无标签源域样本；
- <code>0.15</code>：源域校准验证样本<code>V_cal</code>；
- <code>0.15</code>：源域选模验证样本<code>V_select</code>。

其中有标签比例满足<code>rho_label≤0.1</code>。这组比例是当前协议口径；历史<code>ADV3B02</code>集中式运行中出现的<code>labeled_ratio=0.1</code>、<code>unlabeled_ratio=0.7</code>、<code>source_val_ratio=0.2</code>只能作为历史运行记录，不能反写当前协议。

### 3.2允许与禁止

允许：

- 对源域干净IQ施加普通IQ增强；
- 对源域IQ生成<code>mixed_orbit</code>星地信道视图；
- 使用源域标签计算身份监督；
- 使用源域无标签样本进行teacher-student伪标签/一致性训练；
- 将干净视图与卫星视图送入同一模型，计算卫星辅助分类或一致性损失。

禁止：

- 在训练或超参数选择阶段读取<code>leo_*_weak</code>评测truth；
- 用目标接收机、目标日期、目标场景样本更新模型；
- 把卫星评测视图反向传播到训练过程；
- 把源域代理未知样本当成真实部署未知类；
- 用评测结果选择checkpoint后再回到同一评测集继续调参。

<code>leo_*_weak</code>是可复现的弱LEO代理，不等于真实在轨采集数据。因此，评测结果可以支持“对规定代理信道的鲁棒性”结论，不能单独支持“真实卫星部署性能”结论。

## 4. 训练增强与评测场景必须分开

### 4.1正式方法口径

ADVB02的默认星地训练增强应显式写成：

    sat_train_scenario=mixed_orbit
    sat_train_scenarios=
    sat_view_schedule=
    sat_view_prob=1.0

这里的空<code>sat_train_scenarios</code>和空<code>sat_view_schedule</code>很重要：它们表示不通过多场景schedule覆盖默认场景。只要没有显式schedule，卫星视图生成器的默认阶段场景就是<code>mixed_orbit</code>。

代码中的几个开关含义不同，不能把“场景名”和“分支是否启用”混为一谈：

| 配置项 | 含义 | 代码默认/历史值 |
|---|---|---|
| <code>sat_train_scenario</code> | 默认卫星训练场景名 | <code>mixed_orbit</code> |
| <code>sat_view_prob</code> | 一个batch生成卫星视图的概率 | <code>1.0</code> |
| <code>use_concat_sat_channel_aug</code> | 是否启用拼接式卫星视图增强路径 | parser默认<code>false</code>；历史候选为<code>true</code> |
| <code>concat_sat_ce_only</code> | 是否只让卫星视图承担辅助CE，而不替换干净主batch | parser默认<code>false</code>；历史候选为<code>true</code> |
| <code>use_sat_consistency</code> | 是否允许卫星一致性分支 | SSDG parser默认<code>true</code> |
| <code>lambda_sat_cls</code> | 卫星视图分类损失权重 | 历史候选为<code>0.68</code> |
| <code>lambda_sat_cons</code> | 卫星视图一致性损失权重 | 历史候选为<code>0</code> |

因此，应分别报告：

1. 场景口径：训练增强是<code>mixed_orbit</code>；
2. 分支开关：卫星增强是否启用、采用full-batch还是CE-only；
3. 损失权重：卫星CE和卫星一致性分别是多少。

### 4.2正式评测口径

正式卫星评测场景固定为：

    leo_clear_weak
    leo_low_elev_weak
    leo_rain_weak

三者都是独立的评测场景。评测过程只做前向推理和指标统计，不调用optimizer、不更新EMA teacher、不更新prototype memory、不更新阈值，也不把query truth传回模型。

通用<code>code/train.py</code>和<code>code/post_stage_cli.py</code>的兼容性默认列表包含五个场景：

    clear_leo
    low_elev_leo
    rain_leo
    storm_mp
    mixed_orbit

这只是通用parser的兼容默认，不能覆盖ADVB02正式方法的三场景评测口径。ADVB02主表应使用<code>leo_*_weak</code>三项；若额外报告<code>mixed_orbit</code>，必须标为补充诊断，不能与三项弱LEO主评测混成一个指标。

### 4.3历史配置覆盖的审计说明

历史文件：

    E:/type10-7/automation_reports/CV-SincNet/adv3b02_direct_old_strict_20260714_181100/artifacts/resolved_config.json

该历史配置的评测场景确实是：

    leo_clear_weak,leo_low_elev_weak,leo_rain_weak

但它还包含如下卫星训练覆盖：

    sat_train_scenario=leo_clear_weak
    sat_train_scenario_list=leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    sat_view_schedule=1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak

这说明该文件对应的是一个历史schedule变体，不是本文所定义的“默认<code>mixed_orbit</code>训练增强”配置。复现该历史checkpoint时，应忠实复现它的旧schedule；介绍ADVB02默认方法时，必须把<code>mixed_orbit</code>写在训练增强位置，并把<code>leo_*_weak</code>写在评测位置。

## 5. <code>mixed_orbit</code>训练增强的实现细节

### 5.1注册表级配置

<code>mixed_orbit</code>使用的是混合轨道、城市环境和完整链路仿真口径：

| 参数 | <code>mixed_orbit</code>配置 |
|---|---|
| <code>scenario</code> | <code>urban</code> |
| <code>weather</code> | <code>cloudy</code> |
| <code>look</code> | <code>mid</code> |
| 轨道概率 | LEO<code>0.60</code>、MEO<code>0.30</code>、GEO<code>0.10</code> |
| 仰角范围 | <code>10°–90°</code> |
| SNR范围 | <code>12–30</code> |
| CFO标准差 | <code>300</code> |
| 相位噪声范围 | <code>0–3e-3</code> |
| 多径 | 开启 |
| taps | <code>2–4</code> |
| 最大时延 | <code>5</code> |
| 功率衰减 | <code>0.75</code> |
| channel model | 继承<code>SatSimConfig</code>默认的<code>legacy_full</code> |

<code>mixed_orbit</code>不是“只在LEO上随机几个参数”。它首先按概率抽取LEO/MEO/GEO轨道，再根据轨道和仰角生成传播状态，随后经过路径损耗、大气、衰落、多普勒、相位噪声、多径、AGC、AWGN和IQ不平衡等链路步骤。

### 5.2轨道与传播状态

轨道高度范围为：

- LEO：<code>500–2000 km</code>；
- MEO：<code>8000–20000 km</code>；
- GEO：约<code>35786 km</code>。

城市状态混合权重由仰角<code>theta</code>控制。代码中的核心形式为：

    w1 = clip(1 - (90 - theta)^2 / 7000, 0, 1)
    w3 = (1 - w1) * 4 / 5
    w2 = (1 - w1) * 1 / 5

随后按<code>(LOS, LOO, Rayleigh)</code>解释，因此：

- <code>P_LOS=w1</code>；
- <code>P_LOO=w2</code>；
- <code>P_Rayleigh=w3</code>。

这里的LOO是遮挡/低可见度状态，不应与low-elevation场景名混为一谈。

<code>look=mid</code>的LOO参数为：

    mu=0.8914
    d0=1.3799
    b0=0.126

多径和衰落由当前轨道、状态和配置共同决定。<code>mixed_orbit</code>开启完整链路时，Rician、Rayleigh和LOO状态会使用不同的增益/衰落行为。

### 5.3完整链路处理顺序

在<code>code/sat_channel.py</code>的完整路径中，视图大致经过以下处理：

1. 抽取轨道类型、轨道高度和仰角；
2. 根据斜距计算自由空间路径损耗，参考角度为<code>60°</code>，参考距离为<code>1000 km</code>；
3. 根据城市状态权重抽取LOS、LOO或Rayleigh状态；
4. 根据天气计算大气系数；
5. 根据仰角插值Rician<code>K</code>因子，雨天/风暴状态会降低相应增益；
6. 叠加轨道多普勒与随机CFO；
7. 叠加Wiener相位噪声；
8. 施加多径；
9. 施加路径损耗和大气衰减；
10. 进行目标RMS约为<code>1</code>的AGC，并保留配置的AGC残差；
11. 加入AWGN；
12. 按配置施加IQ不平衡。

<code>mixed_orbit</code>继承的完整配置默认开启：

    apply_path_loss_to_iq=true
    enable_atmospheric_fading=true
    enable_iq_imbalance=true
    use_residual_doppler=false

因此，<code>mixed_orbit</code>训练增强比弱LEO残差评测更重、更宽，承担的是训练时的域随机化和鲁棒性学习，而不是对某一个固定LEO评测条件的精确拟合。

### 5.4训练视图生成与batch行为

<code>baseline_origin_sat_view.py</code>使用按epoch和batch索引派生的随机种子：

    view_seed = seed + epoch * 1009 + batch_idx

每个batch先按<code>sat_view_prob</code>决定是否生成卫星视图。生成后：

- 若未启用拼接路径，模型可直接在生成视图上计算辅助损失；
- <code>concat_sat_full_batch</code>会把干净batch和卫星batch拼接，标签和域标签也相应复制；
- <code>concat_sat_ce_only</code>会单独前向卫星视图，只计算卫星辅助CE，干净主batch保持不变；
- 历史候选采用<code>concat_sat_channel_aug=true</code>、<code>concat_sat_ce_only=true</code>、<code>lambda_sat_cls=0.68</code>、<code>lambda_sat_cons=0</code>，所以其卫星分支主要是“卫星视图辅助分类”，不是clean-to-satellite KL一致性。

## 6. <code>leo_*_weak</code>评测信道的实现细节

三种弱LEO场景使用<code>leo_residual</code>路径。它们都只抽取LEO，且将完整物理链路中的若干强效项关闭：

    apply_path_loss_to_iq=false
    enable_atmospheric_fading=false
    enable_iq_imbalance=false
    use_residual_doppler=true

其中<code>use_residual_doppler=true</code>意味着不叠加轨道多普勒，只保留配置的随机CFO/残差项。这样做的目的是形成可复现、相对受控的弱LEO代理评测，而不是模拟完整的真实在轨传播。

### 6.1三种场景的参数

| 场景 | 仰角 | SNR | CFO标准差 | 相位噪声 | 衰落/K因子 | 多径 | AGC残差 |
|---|---:|---:|---:|---:|---|---|---|
| <code>leo_clear_weak</code> | <code>35°–90°</code> | <code>22–32</code> | <code>50</code> | <code>0–5e-4</code> | Rician，<code>K=16–24</code> | taps<code>2</code>，最大时延<code>2</code>，衰减<code>0.08</code> | <code>±0.2</code> |
| <code>leo_low_elev_weak</code> | <code>10°–35°</code> | <code>16–28</code> | <code>90</code> | <code>1e-4–8e-4</code> | shadowed Rician，<code>K=8–18</code> | taps<code>2</code>，最大时延<code>3</code>，衰减<code>0.12</code> | <code>±0.3</code> |
| <code>leo_rain_weak</code> | <code>20°–80°</code> | <code>14–26</code> | <code>70</code> | <code>1e-4–7e-4</code> | Rician，<code>K=10–20</code> | taps<code>2</code>，最大时延<code>3</code>，衰减<code>0.10</code> | <code>±0.3</code> |

<code>leo_rain_weak</code>虽然带有<code>rain</code>场景标签，但在已核对的弱残差配置中<code>enable_atmospheric_fading=false</code>，因此不会通过完整大气表额外施加雨衰。雨场景的差异主要来自SNR、相位噪声、CFO、仰角、衰落和多径参数组合。不能把它写成“完整雨衰信道”。

## 7. 神经网络结构

### 7.1输入、主干和输出

历史ADVB02/ADV3B02工程配置使用：

| 项目 | 配置 |
|---|---|
| 输入 | 复数IQ拆成I/Q两通道，长度<code>256</code> |
| model variant | <code>lite_d</code> |
| identity/domain骨干 | 双分支CVSincNet式结构 |
| identity表示维度 | <code>160</code> |
| domain表示维度 | <code>160</code> |
| domain enhancer | <code>rcn_stats</code> |
| branch ablation | <code>no_dac</code> |
| domain branch ablation | <code>no_stats</code>（只对相应可选内部stats路径生效） |
| CosFace scale | <code>s=30</code> |
| CosFace margin | <code>m=0.35</code> |
| TX adversarial on <code>z_dom</code> | <code>false</code> |

模型前向的主要输出可以抽象为：

    z_id       = identity backbone feature
    z_dom_raw  = domain backbone feature
    z_dom      = domain enhancer(z_dom_raw, x)
    dom_logits = domain_head(z_dom)
    adv_logits = adversarial_domain_head(GRL(z_id))
    feat_joint = identity classification feature
    feat_imp   = identity-imperfection/domain-related feature

其中GRL是gradient reversal layer：前向传递特征，反向时反转梯度，使<code>z_id</code>尽量不携带可识别域信息。

### 7.2<code>lite_d</code>的确切容量

<code>lite_d</code>相对基础<code>M</code>配置的主要缩放为：

| 模块 | <code>lite_d</code>配置 |
|---|---|
| Sinc输出参数 | <code>sinc_out=24</code> |
| Sinc kernel | <code>79</code> |
| Sinc层实现通道 | <code>2*sinc_out=48</code>，按I/Q配对使用 |
| time bottleneck | <code>48</code> |
| embedding | <code>160</code> |
| frequency bands | <code>32</code> |
| time channels | <code>72/96/96</code> |
| frequency channels | <code>16/32/32</code> |
| PA channels | <code>48/64/64</code> |
| dropout | <code>0.45</code> |

不要把<code>Sinc_out=24</code>误写成整个Sinc层只有24个输出通道；实现层的paired-IQ通道数是<code>48</code>。

### 7.3身份/时域分支

时域分支包括：

1. SincConv1d可学习带通滤波器；
2. 高频率强调模块<code>HighFreqEmphasis</code>；
3. 非线性基函数；
4. <code>1×1</code>卷积、GroupNorm和ReLU；
5. 平均池化；
6. 三个深度可分离时域卷积块；
7. 自适应全局池化；
8. 线性层映射到<code>160</code>维。

对应的核心设置为：

    time_in = 2 * sinc_out + 4
    nonlinear basis enabled -> additional 2 * sinc_out channels
    t1: channels=72, kernel=5, pool=2
    t2: channels=96, kernel=5, pool=2
    t3: channels=96, kernel=3, pool=1

实际<code>lite_d</code>实现会共享identity/domain分支的早期Sinc和高频stem；这来自代码对<code>lite_d</code>的共享条件，不应只根据旧注释中“Lite-B”字样判断。

### 7.4频域分支

频域分支的输入维度基线为<code>4</code>，可按开关加入稳定性特征。主要结构为：

    f1: channels=16, kernel=5, pool=2
    f2: channels=32, kernel=5, pool=2
    f3: channels=32, kernel=3, pool=1
    adaptive pooling -> Linear(32 -> 160)

该分支包含频带门控<code>FreqBandGate</code>。历史配置启用频域统计特征；是否启用raw FFT、稳定性特征等，应以最终resolved config为准，不能仅凭方法名推断。

### 7.5PA分支

PA分支先使用Memory Polynomial Lift：

    depth=4
    orders=(1,3,5)
    clip=2

随后通过EnvelopeGate和三层一维卷积：

    b1: channels=48, kernel=7, dilation=1, pool=2
    b2: channels=64, kernel=7, dilation=2, pool=2
    b3: channels=64, kernel=5, dilation=4, pool=1
    adaptive pooling -> Linear(64 -> 160)

历史候选使用<code>no_dac</code>，因此DAC分支被禁用；PA分支仍然存在。不能把<code>no_dac</code>理解为整个物理感知分支都被删除。

### 7.6域统计增强与分类头

<code>RCNStatEncoder</code>从I/Q序列提取18维统计量，包括：

- I/Q、幅度和log-power的矩统计；
- I/Q相关性与IQ不平衡；
- <code>delta I</code>、<code>delta Q</code>、<code>delta amp</code>均值；
- 相位差均值和相位差summary。

在<code>emb_dim=160</code>时，统计编码器为：

    Linear(18 -> 80)
    LayerNorm
    SiLU
    Dropout
    Linear(80 -> 160)

<code>DomainFeatureEnhancer</code>用门控把原始域特征和RCN统计特征融合：

    gate = sigmoid(Linear(2*emb_dim -> emb_dim))
    z_dom = LayerNorm(z_dom_raw + strength * gate * rcn_stats)

历史配置中的<code>domain_enhancer=rcn_stats</code>和<code>domain_branch_ablation=no_stats</code>必须结合代码开关解读：<code>no_stats</code>关闭的是相应的可选内部stats路径，不能简化为“模型完全没有任何统计特征”。

最终的<code>PhysicalAwareClassifier</code>把身份、频域/PA缺陷相关特征融合为<code>feat_joint</code>，并从缺陷特征得到<code>feat_imp</code>。分类头使用CosFace：

    logits_k = s * (cos(theta_yk + m) for the target class; cos(theta_k) otherwise)

历史配置为<code>s=30</code>、<code>m=0.35</code>。

## 8. 损失函数总览

### 8.1总损失的真实组合

当前<code>code/SSDG/train_ssdg.py</code>的组合逻辑应概括为：

    L_total = L_closed + L_open

闭集部分为：

    L_closed =
        L_tx
        + w_dom * L_dom
        + w_adv * L_adv
        + w_orth * L_orth
        + w_cons * L_cons
        + w_group * L_group_ce
        + w_fishr * L_fishr
        + w_sat_cls * L_sat_cls
        + w_sat_cons * L_sat_cons
        + teacher/pseudo terms

开放世界/特征空间部分为：

    L_open =
        w_proto * L_proto
        + w_ow * L_open_world_feat
        + w_zid * L_zid_compact
        + w_proxy * L_proxy_unknown
        + w_softmix * L_soft_unknown_mixup
        + w_source * L_source_episode
        + direct-metric terms

伪标签阶段还会把<code>lambda_u * L_u</code>、<code>lambda_ent * L_ent</code>、无标签域/对抗/卫星一致性等项加入闭集或开放项。代码没有已确认的<code>closed_scale</code>和<code>open_scale</code>总尺度，因此不应写成额外的两级缩放公式。

### 8.2监督分类与域解耦

#### <code>L_tx</code>

对有标签源域样本使用CosFace身份分类损失，优化<code>feat_joint</code>对应的类别判别边界。

#### <code>L_dom</code>

使用<code>z_dom</code>预测接收机/域信息，使domain branch显式承载域因素。

#### <code>L_adv</code>

使用<code>z_id</code>经过GRL后预测域信息。分类器希望预测正确，而GRL作用于identity encoder的梯度，使identity表示减少接收机/域信息。

#### <code>L_orth</code>

对身份因素和域/缺陷因素施加解耦约束，减少<code>z_id</code>与<code>z_dom</code>的冗余耦合。实现细节以<code>code/cvsrffi/losses.py</code>中的正交/去相关函数为准，不能替换成未经核对的固定矩阵公式。

#### <code>L_cons</code>

用于干净/增强视图或teacher/student之间的预测一致性。它与卫星专用的<code>L_sat_cons</code>不同，不能把所有一致性项合并成一个不分来源的“consistency loss”。

### 8.3域泛化与梯度稳定

#### <code>L_group_ce</code>

按域/组组织分类监督，减少模型只记住某个接收机或日期的风险。

#### <code>L_fishr</code>

当前实现不是昂贵的参数梯度二阶计算，而是使用logit梯度代理：

    g_d ≈ p_d - one_hot(y)

然后按域比较该代理的方差，使不同源域的分类更新方向更接近。正式表述应是“FISHR-style logit-gradient-variance proxy”。

### 8.4卫星视图损失

#### <code>L_sat_cls</code>

对<code>mixed_orbit</code>生成的卫星视图计算身份交叉熵辅助监督。它要求星地增强后仍保留身份判别信息；不要把这个辅助项和主分类头的CosFace margin直接写成同一个loss实现。

#### <code>L_sat_cons</code>

代码采用clean预测分布作为停止梯度的teacher target，对卫星视图预测计算KL：

    L_sat_cons = KL(p_clean || p_sat)

实现上是对<code>sat_logits</code>取log-softmax、对<code>clean_prob.detach()</code>计算<code>F.kl_div</code>。历史候选的<code>lambda_sat_cons=0</code>，所以历史结果不应被描述成“卫星一致性主导”。

### 8.5开放世界特征空间损失

#### <code>L_open_world_feat</code>

<code>open_world_feature_space_loss</code>先归一化特征，再结合类别中心和样本半径约束：

- 类内紧致：样本不要远离自身类别中心；
- 样本margin：离正确中心过远时产生惩罚；
- 中心间隔：不同类别中心保持最小角度/距离；
- 可选域对齐：减少不同域的中心偏移；
- robust tail/CVaR：优先处理困难尾部样本；
- hard-k：只聚焦最困难的一小部分样本。

这是“特征空间开放世界约束”，不是直接把未知类标签当成一个普通softmax类别。

#### <code>L_zid_compact</code>

<code>zid_compactness_loss</code>由三部分组成：

1. supervised contrastive loss；
2. 类内角半径约束；
3. 尾部CVaR惩罚。

代码函数默认CVaR<code>alpha=.90</code>，但历史<code>ADV3B02_CORE90_SOFT_E200</code>resolved config显式使用<code>zid_compact_cvar_alpha=.95</code>。报告中应以最终resolved config为准，不能把函数默认值误写成历史checkpoint实际值。

历史<code>z_id</code>紧致项还使用：<code>start_epoch=8</code>、<code>warmup=25</code>、radius<code>=40°</code>、radius weight<code>=.35</code>、SupCon weight<code>=.30</code>、CVaR weight<code>=.35</code>，并开启domain-aware模式。

#### <code>L_proto</code>

<code>PrototypeMemoryBank</code>维护动量类别原型，并可施加：

- 原型最小计数；
- 动量更新；
- 域对齐；
- 类间margin；
- 原型push。

历史候选的关键值为：<code>proto_min_count=2</code>、momentum<code>=.95</code>、domain align<code>=.1</code>、margin<code>=.15</code>、push<code>=.1</code>。

#### <code>L_soft_unknown_mixup</code>

从不同TX类别构造虚拟混合样本，作为软未知/边界样本训练。历史配置为：

    count=24
    order=3
    alpha=0.5
    CE weight=0.6
    energy weight=1.0
    vacuum weight=0.35
    vacuum width=6
    hard-k=3
    detach=false
    start_epoch=25
    warmup=25

它是源域内部构造的未知边界正则，不是真实未知类数据。

#### <code>L_proxy_unknown</code>

通过proxy unknown的能量/拒识约束，抑制模型对低可信样本的过度自信。历史候选的关键值为：

    lambda_proxy_unknown=0.0045
    proxy_unknown_virtual_count=48
    proxy_unknown_virtual_mode=hard
    proxy_unknown_virtual_detach=false
    proxy_unknown_start_epoch=45
    proxy_unknown_warmup_epochs=25
    proxy_unknown_energy_margin=0.0
    proxy_unknown_known_margin=0.05
    proxy_unknown_unknown_margin=0.08
    proxy_unknown_tail_quantile=0.92
    proxy_unknown_vacuum_radius_deg=40.0
    proxy_unknown_vacuum_width_deg=5.0
    proxy_unknown_vacuum_hard_k=3
    proxy_unknown_vacuum_weight=0.55
    proxy_unknown_vaccept_cvar_alpha=0.3

### 8.6源域episode与三倍标准差约束

<code>source_episode_three_sigma_loss</code>把不同源域/日期组织成leave-one-domain-out episode，在留出的域上约束类别半径和困难尾部：

    start_epoch=20
    warmup=25
    min_domains=2
    radius_cap=33
    mixup_weight=0.75
    hard-k=3

半径模式的代码默认是<code>min_three_sigma_core</code>，核心分位数默认<code>.80</code>，最小sigma默认<code>3</code>。实际运行仍应以resolved config为准。

## 9. 历史工程配置

下面是历史<code>ADV3B02_CORE90_SOFT_E200</code>中已核对到的工程配置。它用于解释历史checkpoint的训练行为，不覆盖当前Phase1协议，也不自动覆盖本文要求的默认<code>mixed_orbit</code>训练口径。

### 9.1训练资源与优化器

| 项目 | 历史配置 |
|---|---|
| from scratch | <code>true</code> |
| device | <code>cuda:0</code> |
| epochs | <code>200</code> |
| label epochs | <code>130</code> |
| pseudo epochs | <code>70</code> |
| train batch size | <code>128</code> |
| eval batch size | <code>256</code> |
| direct eval batch size | <code>512</code> |
| workers | <code>4</code> |
| seed | <code>392002</code> |
| learning rate | <code>2e-4</code> |
| weight decay | <code>1e-4</code> |
| AMP | <code>true</code> |
| optimizer | <code>AdamW</code> |
| LR scheduler | **没有在SSDG路径创建** |
| EMA teacher | 开启 |
| EMA decay | <code>.999</code> |

这里的“没有scheduler”是代码核对结论：<code>code/SSDG/train_ssdg.py</code>创建<code>torch.optim.AdamW</code>和AMP scaler，但没有创建<code>CosineAnnealingLR</code>。通用<code>code/train.py</code>中的余弦调度属于另一条训练路径。

### 9.2历史数据与拆分

历史<code>ADV3B02_CORE90_SOFT_E200</code>解析配置中的数据口径为：

| 项目 | 历史配置 |
|---|---|
| dataset | <code>wisig</code> |
| 数据文件 | <code>ManySig.pkl</code> |
| equalized | <code>1</code> |
| domain key | <code>rx_day</code> |
| IQ长度 | <code>256</code> |
| 类别数 | <code>6</code> |
| split mode | <code>tx_rx_day_1_7_2</code> |
| train days | <code>0,1</code> |
| train receivers | <code>0–6</code> |
| test days | <code>2,3</code> |
| test receivers | <code>7–11</code> |
| historical labeled/unlabeled/source-val | <code>.1/.7/.2</code> |
| historical <code>wisig_train_ratio</code> | <code>.2</code> |

该历史运行还把卫星评测放在三个测试角色上：<code>test_unseen_day_seen_rx</code>、<code>test_seen_day_unseen_rx</code>和<code>test_unseen_day_unseen_rx</code>，场景仍是<code>leo_clear_weak</code>、<code>leo_low_elev_weak</code>和<code>leo_rain_weak</code>。这些是该历史运行的拆分细节；当前项目统一协议仍以<code>.07/.63/.15/.15</code>的<code>L_s/U_s/V_cal/V_select</code>角色定义为准。

### 9.3主要损失权重

历史候选的非零主要权重为：

| 损失项 | 权重 |
|---|---:|
| <code>lambda_adv</code> | <code>.35</code> |
| <code>lambda_domain</code> | <code>1.0</code> |
| <code>lambda_cons</code> | <code>.08</code> |
| <code>lambda_orth</code> | <code>.05</code> |
| <code>lambda_group_ce</code> | <code>.16</code> |
| <code>lambda_fishr</code> | <code>.04</code> |
| <code>lambda_sat_cls</code> | <code>.68</code> |
| <code>lambda_sat_cons</code> | <code>0</code> |
| <code>lambda_proto</code> | <code>.0032</code> |
| <code>lambda_open_world_feat</code> | <code>.0024</code> |
| <code>lambda_zid_compact</code> | <code>.032</code> |
| <code>lambda_proxy_unknown</code> | <code>.0045</code> |
| <code>lambda_soft_unknown_mixup</code> | <code>.0045</code> |
| <code>lambda_source_episode</code> | <code>.0035</code> |
| <code>lambda_u</code> | <code>.16</code> |
| <code>lambda_ent</code> | <code>.01</code> |

旧配置中还有若干PA、DAC、直接度量和未知拒识相关开关。只有在resolved config明确给出非零值时，才能把它们写进该运行的有效总损失；不能把代码中存在的每个loss函数都说成该checkpoint实际启用。

### 9.4伪标签与teacher

历史配置为：

| 项目 | 配置 |
|---|---|
| threshold mode | <code>rx_day_quantile</code> |
| pseudo quantile | <code>.86</code> |
| threshold range | <code>tau_min=.92</code>、<code>tau_max=.97</code> |
| pseudo domain gate | 开启 |
| temporal gate | 开启 |
| temporal window | <code>2</code> |
| minimum confidence | <code>.8</code> |
| strong agreement | 开启 |

teacher通过EMA更新，伪标签只应作用于符合门控条件的无标签样本。

### 9.5普通IQ增强

历史候选的普通IQ增强参数为：

| 增强 | 配置 |
|---|---|
| scale ramp | min<code>.1</code>、max<code>.35</code>、warmup<code>3</code>、ramp<code>15</code>、curve<code>1.25</code> |
| time shift | 概率<code>.35</code>，最大位移<code>32</code> |
| amplitude scale | 概率<code>.45</code>，范围<code>.9–1.1</code> |
| phase rotation | 概率<code>.45</code> |
| CFO | 概率<code>.35</code>，最大<code>.0004</code> |
| phase noise | 概率<code>.30</code>，最大<code>.006</code> |
| AWGN | 概率<code>.4</code>，SNR<code>20–36</code> |
| multipath | 概率<code>.18</code>，taps<code>2–4</code>，最大时延<code>4</code> |
| DC offset | 概率<code>.3</code>，最大<code>.02</code> |
| band-edge | 概率<code>.25</code>，alpha<code>.02–.1</code> |
| PA | 概率<code>.14</code> |
| DAC | 概率<code>0</code> |

普通IQ增强与<code>mixed_orbit</code>星地增强是两层不同的变换：前者是信号级常规扰动，后者是按卫星传播模型生成的星地视图。

## 10. 三阶段训练调度

<code>code/cvsrffi/schedule.py</code>中的阶段控制大致为：

| 阶段 | epoch范围 | 主要行为 |
|---|---|---|
| S1 core | <code>E≤16</code> | 保持基础身份/domain学习；<code>dom=1</code>、<code>adv=.70</code>、<code>orth=.50</code>、<code>cons=0</code>、<code>group=.50</code> |
| S2 stabilize aux | <code>17≤E≤68</code> | ramp引入一致性、辅助分类、正则、joint invariance和KL |
| S3 refine aux | <code>E&gt;68</code> | 逐步把consistency、辅助分类、regularization、joint invariance和KL推向晚期值 |

更具体地，S2使用<code>ramp curve=1.75</code>：

    adv=.70+.30*t
    orth=1
    cons=.20+.55*t
    cls_aux=.15+.55*t
    reg=.35+.45*t
    joint_inv=.15+.20*t
    KL=.15+.35*t
    group=.70+.30*t

S3使用晚期ramp：

    adv=1
    dom=1
    orth=1
    cons=.85+.15*late
    cls_aux=.80+.20*late
    reg=.85+.15*late
    joint_inv=.25+.05*late
    KL=.50+.10*late
    group=1

这些stage系数主要控制辅助路径。历史配置使用<code>no_dac</code>，部分DAC/PA辅助loss的lambda为零，因此不能把每个stage字段都解读成当前运行一定产生了独立梯度。

MixStyle还会在epoch<code>110–150</code>之间退火，概率从<code>.18</code>向<code>.05</code>降低，强度从<code>.70</code>向<code>.32</code>降低。

## 11. 评测与报告应如何写

建议每个checkpoint至少分开报告：

1. clean/source control；
2. <code>leo_clear_weak</code>；
3. <code>leo_low_elev_weak</code>；
4. <code>leo_rain_weak</code>；
5. 三个弱LEO场景的同row汇总。

评测行必须同时保留receiver/TX split、K-shot（若适用）、seed、场景、checkpoint和指标定义。不要把不同场景、不同seed或不同split的单项最大值拼成“最佳结果”。

推荐的最小方法描述是：

> 训练阶段在源域物理IQ上启用<code>mixed_orbit</code>星地信道增强，默认不使用显式卫星多场景schedule；模型采用双分支CVSincNet式身份/域解耦网络，并联合身份分类、域对抗、域特征监督、视图一致性、FISHR-style logit梯度方差代理、原型/开放世界特征空间和源域episode尾部约束。评测阶段冻结checkpoint，仅在<code>leo_clear_weak</code>、<code>leo_low_elev_weak</code>和<code>leo_rain_weak</code>三个弱LEO代理场景上前向评测。

不要写成：

- “训练用<code>leo_*_weak</code>，测试也用<code>leo_*_weak</code>”，除非明确是在复现历史schedule覆盖变体；
- “训练和评测都用<code>mixed_orbit</code>”，这会丢失正式弱LEO评测要求；
- “SSDG使用CosineAnnealingLR”，除非实验明确走通用<code>code/train.py</code>路径；
- “雨天弱LEO开启完整大气雨衰”，当前弱残差配置不支持该说法。

## 12. 可追溯源文件

本核对版主要依据以下项目文件和历史resolved config：

- <code>E:/type10-7/项目.md</code>：Phase1科学边界、数据协议、代理信道和claim语义；
- <code>E:/type10-7/code/SSDG/train_ssdg.py</code>：SSDG训练循环、optimizer、总损失组合、卫星CE/consistency分支；
- <code>E:/type10-7/code/training_controls.py</code>：<code>mixed_orbit</code>和<code>leo_*_weak</code>场景注册参数；
- <code>E:/type10-7/code/sat_channel.py</code>：完整轨道链路与弱LEO residual链路；
- <code>E:/type10-7/code/baseline_origin_sat_view.py</code>：卫星视图生成、随机种子和batch拼接；
- <code>E:/type10-7/code/model.py</code>：<code>lite_d</code>容量、时域/频域/PA分支和分类头；
- <code>E:/type10-7/code/model_dual_cvsincnet.py</code>：identity/domain双分支、RCN统计增强和GRL域对抗；
- <code>E:/type10-7/code/cvsrffi/losses.py</code>：开放世界、原型、z-id紧致、soft unknown mixup、proxy unknown、source episode和FISHR proxy；
- <code>E:/type10-7/code/cvsrffi/schedule.py</code>：S1/S2/S3阶段调度和MixStyle退火；
- <code>E:/type10-7/automation_reports/CV-SincNet/adv3b02_direct_old_strict_20260714_181100/artifacts/resolved_config.json</code>：历史<code>ADV3B02_CORE90_SOFT_E200</code>最终解析配置及其schedule覆盖。

## 13. 最终核对结论

没有发现会改变ADVB02方法主体的结构性错误，但上一版需要按本文修正以下三点：

1. 星地训练增强的默认场景必须写<code>mixed_orbit</code>，不能写成泛化的“LEO弱场景训练”；
2. 正式评测必须写<code>leo_clear_weak</code>、<code>leo_low_elev_weak</code>、<code>leo_rain_weak</code>，与训练增强分开；
3. SSDG优化器必须写“AdamW、无scheduler”；历史配置中出现的LEO训练schedule只能作为旧覆盖变体说明。

这三点是本文的优先级最高的口径，后续README、实验报告、launcher和论文方法段落都应保持一致。
