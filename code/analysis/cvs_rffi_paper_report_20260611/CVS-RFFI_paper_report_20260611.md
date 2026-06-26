# CVS-RFFI: 面向天基地面异构链路的少样本域泛化射频指纹识别论文版报告

生成日期：2026-06-11  
报告范围：基于本地 PPT、CEN51 数据流说明、少样本损失函数说明、星地信道增强说明，以及在线 RFFI 代表文献核验。  
报告性质：论文版技术报告草稿，可作为后续论文 Introduction、Related Work、Method、Experiment、Discussion 的完整骨架。

## 摘要

射频指纹识别（Radio Frequency Fingerprint Identification, RFFI）利用发射机模拟前端不可避免的硬件非理想特性识别设备身份。与依赖协议字段或密钥协商的认证方式不同，RFFI 在接收端直接分析物理层 IQ 信号，可用于 IoT、LoRa、WiFi、无人机、车联网、频谱监管和非合作目标识别等场景。天基 RFFI 进一步放大了传统 RFFI 的泛化难题：训练样本主要来自地面接收链路，部署场景却面向卫星链路；星上训练与标注成本高；星地传播会叠加长距离路径损耗、多普勒、CFO、相位噪声、多径、雨衰、低 SNR 和接收端 I/Q 失衡；模型还必须在跨接收机、跨日期、跨信道和少样本条件下保持发射机可分性。

CVS-RFFI 的核心思想不是单纯加大网络容量，而是把“发射机身份”和“接收链路/信道域”拆开建模。当前 CEN51 主线以 `DualCVSincNetDisentangle` 为实现载体，使用两个 `CVSincNet` 子骨干分别学习 identity 表征 `z_id` 与 domain 表征 `z_dom`。身份骨干强调 Sinc 时域、镜像频谱、PA 非线性与记忆效应；域骨干吸收 receiver/day/channel 相关变化，并通过 Domain Freq DSQ、Domain DAC、RCN raw-IQ 统计增强域表示。训练目标由 TX 分类、Domain 分类、GRL 对抗、正交/协方差解耦、same-TX cross-domain consistency、GroupCE/SmoothGroupDRO、satellite view CE、prototype、domain-aware SupCon、Fishr proxy 和 feature-norm guard 组成。低样本时，这些损失不能无差别叠加；K5/K10 的首要目标是保护 TX identity，K20+ 才逐步恢复 GroupCE、prototype、SupCon、satellite view 和轻量 Fishr。

本报告给出 CVS-RFFI 的完整科研叙事：从 RFFI 物理指纹形成机制出发，解释为何天基场景需要域泛化和少样本学习的统一任务定义；在相关工作中对齐 ORACLE、WiSig、channel-robust LoRa RFFI、receiver-agnostic collaborative RFFI、federated RFFI、RIEI/FedRIEI、single-source DG 和卫星物理层认证等代表性文献；在方法部分系统展开 CVS-RFFI 的模型结构、损失函数、星地信道增强和评估协议；在讨论部分给出与 RIEI/DRIFT 的机制差异、satellite robustness 与 clean OOD 的权衡、低样本 checkpoint 选择和后续消融路线。

关键词：射频指纹识别；天基物理层认证；域泛化；少样本学习；跨接收机泛化；星地信道增强；CV-SincNet；GRL；RIEI；WiSig

## 1. 研究问题与科学动机

### 1.1 RFFI 的物理基础

RFFI 的可行性来自发射机射频前端的制造偏差和动态非理想。即使两台设备使用相同芯片、相同调制方式和相同协议栈，DAC 量化噪声、采样时钟抖动、振荡器相位噪声、CFO、混频器 I/Q 幅相不平衡、镜像泄漏、本振泄漏、滤波器群时延失真、功放 AM/AM 与 AM/PM 非线性、PA 记忆效应等因素都会在 IQ 波形中留下细微但可学习的差异。深度 RFFI 的目标可写为：

$$
\hat{y} = h_{\phi}(g_{\theta}(x)),
$$

其中 \(x \in \mathbb{R}^{2\times T}\) 是接收端 IQ 片段，\(g_{\theta}\) 是特征提取器，\(h_{\phi}\) 是 TX 分类头，\(y\) 是发射机身份。理想情况下，\(g_{\theta}(x)\) 应主要捕捉发射机硬件痕迹；实际接收信号还混入无线信道 \(h_c\)、接收机硬件响应 \(r\) 与噪声 \(n\)。因此，RFFI 的难点不是分类器能不能记住训练集，而是能否在接收链路和传播环境变化后仍然识别同一台发射机。

### 1.2 天基 RFFI 的任务压力

天基 RFFI 可以扩大接收视野，覆盖蜂窝终端、IoT 设备、广播信号、ADS-B 辐射源、船载通信终端和无人机链路。但它同时带来四类部署压力。

第一，训练与部署域不一致。地面接收通常距离近、链路局部、信道较可控；卫星接收距离远、覆盖广、视角变化大，信号要经历多普勒、时延、低仰角路径损耗和天气衰落。地面训练、天上部署构成典型 domain generalization 问题。

第二，非合作条件下标注稀缺。真实目标不会主动提供大量带标签 IQ 样本。少样本学习不是锦上添花，而是 RFFI 系统从实验室走向非合作部署的必要能力。

第三，星上训练不可依赖。星上算力、功耗、存储、热设计和通信链路限制使得全模型在线训练不可作为默认假设。可行路径是地面训练稳定表征，部署时仅做轻量校准、少量样本适配或协作推理。

第四，鲁棒性和可分性存在冲突。强信道增强能提升 satellite robustness，但过早或过强的扰动会抹掉 PA、I/Q、频谱细节等发射机指纹，使 clean OOD 或 strict UDU 准确率下降。好的方法必须在“去域”和“保身份”之间建立可控边界。

### 1.3 本文任务定义

本文把 CVS-RFFI 定义为：在有限标注 WiSig IQ 样本上训练一个物理先验驱动的双表示模型，使其在未见接收机、未见日期和星地信道扰动下仍能识别发射机身份。

训练样本记为：

$$
\mathcal{D}_{train}=\{(x_i,y_i,d_i)\}_{i=1}^{N},
$$

其中 \(x_i\in\mathbb{R}^{2\times T}\) 是 raw IQ，\(y_i\) 是 TX label，\(d_i\) 是 receiver/day 或 compact domain label。模型学习：

$$
f_{\theta}(x)=\{s_{tx},s_{dom},s_{adv},z_{id},z_{dom}\},
$$

其中 \(z_{id}\) 应保留 TX identity，\(z_{dom}\) 应吸收 receiver/day/channel nuisance，\(s_{adv}\) 由 \(\operatorname{GRL}(z_{id})\) 预测 domain，用反向梯度压制 \(z_{id}\) 中的域信息。

## 2. 相关工作与定位

### 2.1 从手工 RF-DNA 到深度 IQ 表征

早期 RFFI 工作常从调制误差、瞬态、频偏、I/Q offset、幅相误差和统计分布中构造人工特征。PARADIS 将 radiometric signatures 用于无线设备识别，RF-DNA 系列将统计特征与 MDA/SVM 等分类器结合。这类方法可解释性强，但需要针对协议、设备和接收条件做特征工程，面对跨接收机和跨信道迁移时稳定性有限。

ORACLE 和 “No Radio Left Behind” 代表深度 IQ RFFI 的转折点。该路线直接从物理层 IQ 样本学习硬件中心的唯一签名，并在 bit-similar 设备识别中展示 CNN 的表达能力。CVS-RFFI 继承了“直接从 IQ 学发射机硬件失真”的思想，但没有把 CNN 当作黑盒，而是把 Sinc 时域滤波、镜像频谱、PA 记忆多项式和域解耦显式写入结构。

### 2.2 从同分布准确率到跨接收机/跨日泛化

WiSig 使 RFFI 的真实部署问题变得可量化。该数据集包含 174 个 WiFi 发射机、41 个 USRP 接收机、4 次跨月采集与 1000 万级 packet，明确服务于 receiver/channel agnostic RF fingerprinting。CVS-RFFI 的 WiSig 训练协议也围绕 receiver/day shift 组织：训练使用 DAY1、DAY2 和 RX0-RX6，测试使用 DAY3、DAY4 和 RX7-RX11。这个设置要求模型同时处理 unseen day 与 unseen receiver，而不是只在同接收机同日期上追求高准确率。

### 2.3 信道鲁棒、接收机无关与协作 RFFI

IEEE TIFS/TMC 近年 RFFI 工作已经从“能否识别”转向“能否部署”。LoRa channel-robust RFFI 证明信道变化会显著削弱深度指纹，receiver-agnostic collaborative RFFI 引入接收机无关训练与多接收机协作推理，GAN-RXA 通过生成式校准处理 receiver mismatch。RIEI/FedRIEI 则把跨接收机问题写成 emitter-related feature 与 receiver-related feature 的解耦问题，并给出联邦版本以减少原始数据集中化。

CVS-RFFI 与这些工作一致地承认 receiver shift 是核心风险，但其技术组织更强调物理分支和训练门控。它不是只在 ResNet-18 上施加一个解耦损失，而是让 Time/Freq/PA/DAC/RCN/DSQ 等模块分别承担可解释角色：身份路径保留发射链特征，域路径吸收接收机、日期和信道风格，GRL 与正交约束限制二者泄漏。

### 2.4 联邦、隐私与少样本适配

Federated RFFI powered by unsupervised contrastive learning 表明，多接收机 RFFI 不能简单假设所有 IQ 数据集中上传。该工作采用联邦预训练、客户端微调和本地识别三阶段流程，在 LoRa 多 SDR 接收机场景中展示了联邦 RFFI 的可行性。CVS-RFFI 当前报告以集中式 CEN51 为主体，但其方法边界与联邦路线兼容：如果未来迁移到联邦，`receiver` 粒度客户端、0.1 WiSig train ratio、200 轮默认设置和 receiver-domain 风格统计都应作为硬约束保留。

少样本方面，RFFI 文献通常把目标类少样本、新设备快速注册或接收机适配作为单独问题处理。CVS-RFFI 的贡献点在于把少样本与域泛化合并成同一个训练目标：在每类样本有限时，模型不仅要拟合 TX identity，还要避免把 receiver/day/channel shortcut 当作 identity。

PPT 中也给出了跨领域 few-shot DG 的参照线，如医学影像识别/分割中的 Domain Generalizer、FAMNet、DSM 和 RobustEMD。它们说明 few-shot 与 domain generalization 可以共同建模，但不能直接替代 RFFI 方法：医学影像的域偏移多来自成像设备、医院协议或组织外观，RFFI 的域偏移来自接收机硬件、采集日期、无线传播和星地链路。CVS-RFFI 的缺口表述应保持明确：RFFI 中已有域泛化，也已有少样本或适配工作，但“跨接收机 x 跨日期 x 星地信道 x K-shot”的统一任务定义和 benchmark 仍不足。

### 2.5 CVS-RFFI 的论文定位

CVS-RFFI 可定位为“物理先验驱动的少样本跨域 RFFI”。与传统手工特征相比，它从 raw IQ 端到端学习；与普通 CNN/ResNet RFFI 相比，它把时域、频域、PA 非线性和域统计结构化；与 RIEI/DRIFT 相比，它不只依赖特征解耦目标，而是用双骨干、RCN、DSQ、GRL、orth、GroupCE、prototype、SupCon、Fishr proxy 和 staged scheduling 共同约束表示；与普通数据增强相比，它把卫星链路建模为物理约束的同标签信道变换。

术语关系建议在论文中固定如下：

| 术语 | 建议含义 | 避免误写 |
|---|---|---|
| CVS-RFFI | 本文方法/任务叙事名，指面向少样本、跨域、星地增强的 RFFI 框架 | 不写成某个单独脚本名 |
| CVS 或 CVS/CV-SincNet | 当前代码和报告中对模型家族的简称 | 不与 CEN51 并列成不同方法 |
| CV-SincNet | 物理先验 backbone 家族，含 Sinc/Time/Freq/PA/DAC 等路径 | 不写成外部 spectrogram 模型 |
| CEN51_R04 | 当前主线实验配置，包含 lite_d、no_dac、Domain DSQ、RCN 等设置 | 不代表所有 CVS-RFFI 变体 |

## 3. 数据协议与符号约定

### 3.1 输入与模型口径

当前 CEN51 主线的输入是 WiSig raw IQ：

$$
x \in \mathbb{R}^{B\times 2\times 256}.
$$

进入模型前，IQ 片段通过 `pad_crop_iq` 裁剪或补齐到 256 点。两个通道分别为 I 和 Q。频谱特征不是外部 spectrogram 输入，而是模型内部从 raw IQ 计算 mirror FFT 特征。

需要明确两个 TX 类别口径：

- 模型配置口径：`DualCVSincNetDisentangle` 当前 CEN51 dataflow 文档中写明 `tx_logits [B,16]`，即模型 head 可按 16 类配置。
- 仿真实验口径：PPT 实验页写 WiSig 子集含 6 个发射机、12 个接收机、4 天，每类样本数取 10、20、30、50、100。

论文中不能把这两个数字混成一个实验设置。方法公式用 \(C\) 表示 TX 类数；实现说明可写当前 head 为 16；实验协议若沿用 PPT，则写选取 6 TX 子集。

### 3.2 Domain 口径

CEN51 的 compact domain label 使用 `rx_day`。在 PPT 协议中，训练接收机为 RX0-RX6，训练天数为 DAY1、DAY2，因此训练域数量为：

$$
G_{train}=7\times2=14.
$$

这解释了 dataflow 中 `dom_logits [B,14]` 与 `adv_dom_logits [B,14]` 的形状。全数据元信息是 12 个 receiver 和 4 个 day，但训练域数 14 不是全局所有 receiver-day 组合。

### 3.3 少样本协议边界

当前材料中存在两种少样本协议，结论必须分开。

Per-combo cap 协议使用：

```text
--wisig_train_ratio 0.1
--wisig_max_train_per_combo K
--wisig_cap_strategy random/front
```

其训练集为：

$$
\mathcal{D}_{train}^{combo}(K)
=
\bigcup_{c,r,d,e}
\operatorname{Cap}_{K}\{(x_i,y_i,d_i):y_i=c,rx_i=r,day_i=d,eq_i=e\}.
$$

这个协议仍保留 CVS 的 train/val/test 结构。历史 CEN51_R04 sweep 中，K=5、10、20、30、50 对应训练规模约为 \(K\times84\)。

严格 per-class K-shot 协议使用：

```text
--wisig_train_shots_per_class K
--wisig_train_shot_strategy rx_day_balanced/domain_balanced/random/front
--no_train_drop_last
--train_steps_per_epoch N
```

其训练集为：

$$
\mathcal{D}_{train}^{class}(K)
=
\bigcup_{c=1}^{C}
\operatorname{Select}_{K}\{(x_i,y_i,d_i):y_i=c\}.
$$

若实验子集有 6 个 TX，K=5 表示总训练样本约 30 条。这才是严格意义上的极少样本 RFFI。

## 4. CVS-RFFI 方法

### 4.1 设计原则

CVS-RFFI 的结构遵循三条原则。

第一，发射机身份特征必须与接收链路域特征分工。若单一 backbone 同时承担 TX 分类和 domain 泛化，模型很容易利用 receiver/day shortcut 获得训练集高准确率，却在 RX7-RX11 或 DAY3/DAY4 上失效。

第二，物理先验要进入表示学习。RFFI 的身份线索来自 PA 非线性、I/Q 不平衡、镜像泄漏、频谱不对称、相位/包络细节和滤波响应。CVS-RFFI 不把这些全部交给普通卷积自己发现，而是构造 Time、Freq、PA、DAC、RCN 等有物理含义的路径。

第三，域泛化损失必须受样本量和 domain 覆盖门控。低 K 时强行开启 GroupCE、SupCon、Fishr 或 full-DG satellite view 会让模型先学域扰动而不是 TX identity。

### 4.2 双骨干结构

当前 CEN51 主线使用 `DualCVSincNetDisentangle`。输入 IQ 同时进入 ID Backbone 和 Domain Backbone：

$$
z_{id},s_{tx}=\operatorname{IDBackbone}(x),
$$

$$
z_{dom},s_{dom}=\operatorname{DomainBackbone}(x).
$$

ID Backbone 负责 identity representation，输出 `z_id [B,160]` 与 `tx_logits [B,C]`。Domain Backbone 负责 domain representation，输出 `z_dom [B,160]` 与 `dom_logits [B,G]`。GRL 分支从 `z_id` 产生 `adv_dom_logits [B,G]`：

$$
s_{adv}=\operatorname{MLPHead}(\operatorname{GRL}(z_{id})).
$$

GRL 前向不改变输入：

$$
\operatorname{GRL}_{forward}(z_{id})=z_{id},
$$

反向时翻转梯度：

$$
\frac{\partial L}{\partial z_{id}}
\leftarrow
-\gamma_{grl}\frac{\partial L}{\partial z_{id}}.
$$

因此，GRL 的目的不是单纯“让域分类器失败”，而是迫使 ID Backbone 学不到可被 domain head 利用的 receiver/day/channel 信息。

### 4.3 ID Backbone: 身份路径

CEN51 R04 的 ID Backbone 保留 Time、Freq、PA，关闭 ID-side DAC：

- Time 分支：SincConv IQ filterbank、非线性基、高频差分、DSConv，用于捕捉瞬态波形、相位/包络变化和短时局部纹理。
- Freq 分支：从 raw IQ 内部计算 mirror FFT 四通道特征 `[logP_pos, logP_neg, logR, asym]`，再经 FreqBandGate 和 DSConv 捕捉镜像不对称、高频能量占比、频谱平坦度、边缘能量和谱再生。
- PA 分支：使用记忆多项式 \(x[n-m]|x[n-m]|^{p-1}\)，其中 \(p=1,3,5\)，并以 dilated convolution 建模 PA 非线性和记忆效应。
- ID-side DAC：在 CEN51 R04 中关闭，`dac_local`、`feat_dac`、`dac_pred` 为零占位，避免身份分类过度依赖易随接收域变化的 DAC/接收链细节。

身份融合后得到：

$$
z_{id}=z_{\mathrm{joint}}\in\mathbb{R}^{B\times160}.
$$

TX 分类头采用 CosFace 思想，先归一化 embedding 和类别权重，再通过角度 margin 拉近同类、拉开异类。这比普通线性分类头更适合身份识别任务。

### 4.4 Domain Backbone: 域路径

Domain Backbone 的目标不是复制 identity branch，而是吸收接收机、日期、信道和噪声风格。

CEN51 R04 中 Domain Backbone 保留 Time、Freq、DAC、PA，但关闭内部手工统计注入，并在 Domain Freq 分支启用 DSQ：

$$
\operatorname{DSQ}(\mathrm{feat}_{f}):\mathbb{R}^{B\times4\times32}\rightarrow\mathbb{R}^{B\times2\times32}.
$$

DSQ 可理解为局部频谱稳定残差：先估计局部平滑背景，再取原始频谱对背景的偏离。它只放在 Domain Freq 分支，不放在 CEN51 R04 的 ID Freq 分支。这样，频域局部不稳定性更倾向进入 `z_dom`，而不是污染 `z_id`。

Domain DAC 分支在域路径中启用，用于吸收非圆复信号效应、I/Q 失衡、镜像泄漏、DAC/接收链非理想。该设计形成清晰分工：ID 去域，Domain 吸域。

### 4.5 RCN Domain Enhancer

RCN 从 raw IQ 直接计算统计量，包括 I/Q mean/std/abs、幅度、功率、I-Q 相关、I/Q 失衡、差分和相位增量等。其投影形式可写为：

$$
z_{rcn}
=
\operatorname{MLP}(\operatorname{stats}(x))
\in\mathbb{R}^{B\times160}.
$$

Domain enhancer 通过 gate 注入：

$$
g_{dom}=\sigma(\operatorname{Linear}([z_{dom,raw},z_{rcn}])),
$$

$$
z_{dom}
=
\operatorname{LayerNorm}
\left(
z_{dom,raw}
+\lambda_{rcn}g_{dom}\odot z_{rcn}
\right).
$$

该模块把可解释 raw-IQ 统计显式送入 domain branch，并通过 gate 控制注入强度，避免统计量覆盖卷积分支。

## 5. 星地信道增强

### 5.1 物理信道算子

星地信道增强的输入是 clean IQ，输出是同标签 satellite-view IQ。写成复基带：

$$
s[n]=I[n]+jQ[n].
$$

增强算子为：

$$
x_{sat}=\mathcal{T}_{sat}(x;\Omega),
$$

其中 \(\Omega\) 包含轨道、仰角、天气、SNR、CFO、多径、相位噪声和 I/Q 失衡等随机参数。完整链路可概括为：

$$
\tilde{s}[n]
=
\sum_{\ell=0}^{L-1}h_{\ell}s[n-\tau_{\ell}],
$$

$$
u[n]
=
G_{pl}a_{atm}\tilde{s}[n]\exp(j\psi[n]),
$$

$$
v[n]
=
\Gamma_{AGC}(u[n])+w[n],
$$

$$
r[n]
=
\alpha v[n]+\beta v^{*}[n],
$$

$$
x_{sat}[n]
=
\begin{bmatrix}
\Re\{r[n]\}\\
\Im\{r[n]\}
\end{bmatrix}.
$$

### 5.2 增强项与 IQ 作用

星地增强由以下物理项组成：

| 模块 | 建模对象 | 对 IQ 的作用 | 论文表述 |
|---|---|---|---|
| 几何路径与 FSPL | 斜距、载频、仰角 | I/Q 同比例缩放 | 形成幅度压力，但 AGC 后不应被当作身份特征 |
| 大气衰落 | 晴空、云、雨、风暴 | 幅度缩放和整体相位偏移 | 形成天气相关 nuisance |
| Rician/Rayleigh/LOO | 主径与散射 | 复增益缩放、旋转和随机起伏 | 区分 LOS、遮挡和无主径状态 |
| 多径 | 延迟路径叠加 | 不同时间位置 IQ 复权重混合 | 改变局部时序与频谱纹理 |
| 多普勒/CFO | 卫星运动与本振偏差 | 采样点累积相位旋转 | 破坏短时相位稳定性 |
| 相位噪声 | Wiener 随机游走 | 局部相位抖动 | 影响相位微结构 |
| AGC/AWGN | 自动增益与噪声 | 幅度归一化、点云发散 | 降低可用 SNR，压缩绝对幅度差异 |
| I/Q 失衡 | 接收端幅相不匹配 | 椭圆化与共轭镜像混合 | 模拟接收端硬件 nuisance |

### 5.3 星地场景参数

当前材料中的场景来自 `code/training_controls.py`，应在论文实验部分作为可复现实验参数写清楚：

| 场景 | 天气 | 轨道概率 | 仰角范围 | SNR | CFO 标准差 | 相位噪声增量 | 多径 |
|---|---|---|---:|---:|---:|---:|---|
| `clear_leo` | clear | LEO 1.0 | 30-90 deg | 20-30 dB | 200 Hz | 0-2e-3 | 否 |
| `low_elev_leo` | clear | LEO 1.0 | 10-30 deg | 15-28 dB | 350 Hz | 5e-4-3e-3 | 否 |
| `rain_leo` | rain | LEO 1.0 | 20-80 deg | 10-25 dB | 250 Hz | 5e-4-3e-3 | 否 |
| `storm_mp` | storm | LEO 0.8, MEO 0.2 | 10-35 deg | 8-20 dB | 400 Hz | 1e-3-4e-3 | 2-5 taps, max 6 samples |
| `geo_clear` | clear | GEO 1.0 | 25-80 deg | 18-30 dB | 100 Hz | 0-1.5e-3 | 否 |
| `mixed_orbit` | cloudy | LEO 0.6, MEO 0.3, GEO 0.1 | 10-90 deg | 12-30 dB | 300 Hz | 0-3e-3 | 2-4 taps, max 5 samples |

这些场景不是不同算法，而是不同物理压力组合。`clear_leo` 是温和基线；`low_elev_leo` 强化低仰角和 Doppler/CFO 压力；`rain_leo` 强化雨衰与低 SNR；`storm_mp` 强化风暴、多径和遮挡；`mixed_orbit` 用于扩大训练分布。

### 5.4 训练角色

Satellite augmentation 是数据层同标签视图增强，不是模型内部新增 satellite branch。其监督逻辑为：

$$
\mathcal{L}_{sat\_cls}
=
\operatorname{CE}(s_{tx}^{sat},y).
$$

当启用 clean-to-sat consistency 时，可写为：

$$
\mathcal{L}_{sat\_cons}
=
1-\cos(z_{id}^{sat},\operatorname{sg}(z_{id}^{clean})).
$$

低 K 下，强 satellite consistency 可能提升 satellite floor，却压低 clean strict UDU。当前更稳妥的策略是：K5/K10 使用弱 CE-only 或延后 satellite view；K20+ 逐步恢复更完整的 satellite 路径；K50/K80 再探索更强的 full-DG satellite 设置。

## 6. 损失函数与训练调度

### 6.1 总目标

CVS-RFFI 的集中式训练目标可写为：

$$
\begin{aligned}
\mathcal{L}_{CVS}
=&
\lambda_{cls}\mathcal{L}_{cls}
+\lambda_{dom}\mathcal{L}_{dom}
+\lambda_{adv}\mathcal{L}_{adv}
+\lambda_{orth}\mathcal{L}_{orth} \\
&+\lambda_{cons}\mathcal{L}_{cons}
+\lambda_{group}\mathcal{L}_{groupCE}
+\lambda_{sat}\mathcal{L}_{sat}
+\lambda_{proto}\mathcal{L}_{proto} \\
&+\lambda_{supcon}\mathcal{L}_{supcon}
+\lambda_{fishr}\mathcal{L}_{fishr}
+\lambda_{norm}\mathcal{L}_{norm}
+\mathcal{L}_{aux}.
\end{aligned}
$$

各项含义如下：

- \(\mathcal{L}_{cls}\)：TX 主分类损失，是所有低样本阶段的核心。
- \(\mathcal{L}_{dom}\)：用 `z_dom` 预测 domain，鼓励域因素进入 domain branch。
- \(\mathcal{L}_{adv}\)：用 `GRL(z_id)` 预测 domain，反向压制 `z_id` 中的域信息。
- \(\mathcal{L}_{orth}\)：降低 `z_id` 与 `z_dom` 的协方差耦合。
- \(\mathcal{L}_{cons}\)：same-TX cross-domain centroid consistency，让同一 TX 跨 domain 的 identity center 稳定。
- \(\mathcal{L}_{groupCE}\)：hard-domain CE 或 SmoothGroupDRO，关注困难域。
- \(\mathcal{L}_{sat}\)：satellite view 的 TX CE 与可选 clean-to-sat consistency。
- \(\mathcal{L}_{proto}\)：类别原型和类-域原型的 pull、align、push。
- \(\mathcal{L}_{supcon}\)：same-TX cross-domain supervised contrastive loss。
- \(\mathcal{L}_{fishr}\)：domain-level logit-gradient variance matching 的轻量 proxy，不是真正 full per-parameter Fishr。
- \(\mathcal{L}_{norm}\)：feature-norm guard，防止 `z_id` 通过范数膨胀绕过去域约束。
- \(\mathcal{L}_{aux}\)：PA/DAC 辅助预测、consistency、KL 和 strength regularization。

实际训练不应把上式所有 \(\lambda\) 固定成常数。来源材料中的调度更适合写为：

$$
w_k(t)=\lambda_k\cdot \operatorname{scale}_k(t),
$$

其中 \(t\) 是 epoch 或 step，\(\operatorname{scale}_k(t)\) 来自三阶段 schedule：`S1_core -> S2_stabilize_aux -> S3_refine_aux`。S1 优先建立 TX identity；S2 逐步恢复 domain/orth/consistency 与轻量辅助项；S3 才让 prototype、SupCon、satellite 和 Fishr proxy 进入更完整的泛化约束。Satellite 也有两条训练路径：`concat_sat_ce_only=true` 时 satellite view 只贡献 TX CE；full-DG satellite path 则把 satellite 样本拼入主 batch，并参与 domain/GRL/GroupCE/prototype/SupCon/Fishr 等完整路径。

### 6.2 Domain loss gate

域相关损失不是每个 batch 都启用。若 batch 中 domain 数不足、每域样本不足或没有 same-TX cross-domain pair，domain/adv/group/supcon/fishr 项必须自动降为 0 或极小权重。该 gate 防止低 K 设置中用伪统计压坏 identity fitting。

### 6.3 少样本权重策略

| 样本区间 | 训练目标 | 推荐打开项 | 需克制项 |
|---|---|---|---|
| K5/K10 | 先建立 TX identity 边界 | TX CE、轻量 GRL、feature-norm guard | GroupCE、Proto、SupCon、Fishr、强 satellite |
| K20/K30 | 恢复跨域稳健性 | 轻量 GroupCE、Proto、consistency、弱 satellite | 强 Fishr、强 full-DG satellite |
| K50/K80 | 接近低样本可用区 | GroupCE、prototype、SupCon、MixStyle、satellite schedule | final rollback、过强 domain 对齐 |
| K>=100 或 ratio=0.1 | 回到 CEN51 主线 | 原始 CEN51_R04 ratio path | 不继续套 pure-shot cap |

### 6.4 Checkpoint 选择

少样本 CVS 不能只用 final epoch。K5/K10 常见现象是早期 strict/val 出现可用峰值，后期 train/val/strict 共同下降或出现 rollback。选择器应综合：

- best primary；
- strict UDU；
- worst-rx；
- satellite floor；
- rollback 幅度；
- SWAD 或滑动平均稳定性；
- final epoch 与 best epoch 的差距。

这也是 CVS-RFFI 与只报告 final accuracy 的普通实验不同的地方：模型的“稳定泛化”比单点最高训练准确率更重要。

## 7. 实验设计与评估协议

### 7.1 WiSig 子集协议

PPT 中的仿真实验协议可写为：

| 项 | 设置 |
|---|---|
| 数据集 | WiSig RFFI 子集 |
| 发射机数量 | 6 |
| 接收机数量 | 12 |
| 采集天数 | 4 |
| 单信号长度 | 256 |
| 训练日期 | DAY1、DAY2 |
| 训练接收机 | RX0-RX6 |
| 测试日期 | DAY3、DAY4 |
| 测试接收机 | RX7-RX11 |
| 每类样本数 | 10、20、30、50、100 |
| 对比方法 | RIEI、DRIFT、RIEI+Sat、DRIFT+Sat |

在论文中，建议同时报告 clean OOD 和 satellite OOD。只报告 satellite accepted accuracy 会掩盖 clean strict UDU 下降；只报告 clean strict UDU 又会低估天基部署压力。

### 7.2 PPT 已有 low-shot 结果

PPT slide 13 的三张图可解析为以下数值。单位均为准确率百分比。

Strict UDU: unseen-day unseen-RX，是最能体现跨日跨接收机泛化的指标。

| Samples/TX | CVS | RIEI | RIEI+Sat | DRIFT | DRIFT+Sat |
|---:|---:|---:|---:|---:|---:|
| 10 | 76.25 | 67.64 | 66.98 | 63.95 | 64.81 |
| 20 | 77.56 | 67.08 | 65.94 | 65.35 | 69.34 |
| 30 | 79.91 | 64.29 | 60.64 | 70.53 | 70.60 |
| 50 | 84.11 | 66.39 | 57.86 | 74.40 | 65.38 |
| 100 | 84.05 | 59.83 | 46.32 | 77.02 | 57.30 |

SDU: seen-day unseen-RX，主要考察接收机迁移。

| Samples/TX | CVS | RIEI | RIEI+Sat | DRIFT | DRIFT+Sat |
|---:|---:|---:|---:|---:|---:|
| 10 | 79.06 | 71.19 | 70.85 | 66.40 | 64.58 |
| 20 | 86.04 | 71.36 | 67.95 | 69.86 | 69.97 |
| 30 | 87.95 | 67.86 | 66.51 | 72.05 | 72.75 |
| 50 | 89.22 | 68.80 | 59.63 | 77.45 | 69.05 |
| 100 | 87.10 | 61.45 | 48.64 | 77.25 | 64.80 |

UDS: unseen-day seen-RX，主要考察日期迁移。

| Samples/TX | CVS | RIEI | RIEI+Sat | DRIFT | DRIFT+Sat |
|---:|---:|---:|---:|---:|---:|
| 10 | 87.77 | 83.33 | 86.71 | 81.55 | 73.49 |
| 20 | 87.85 | 87.78 | 86.21 | 82.19 | 82.13 |
| 30 | 91.20 | 91.18 | 86.98 | 86.62 | 87.40 |
| 50 | 92.07 | 90.78 | 85.96 | 91.09 | 89.29 |
| 100 | 92.55 | 92.42 | 84.57 | 91.25 | 86.93 |

这些结果给出三条结论。第一，CVS 在 strict UDU 上对所有 K 均领先，说明双骨干物理先验与解耦训练主要改善最困难的跨日跨接收机场景。第二，UDS 上各方法差距明显缩小，尤其 K20/K30/K100 时 RIEI 接近 CVS，说明单纯日期迁移比 receiver+day 联合迁移更容易。第三，RIEI+Sat 和 DRIFT+Sat 并不稳定：DRIFT+Sat 在 strict UDU 的 K20/K30 有局部收益，但 K50/K100 明显下降；RIEI+Sat 随 K 增大持续低于 RIEI。这支持本文的 satellite trade-off 判断：星地扰动必须受调度和身份保护约束，否则会破坏 clean OOD identity geometry。

### 7.3 指标建议

核心指标至少包括：

- SDU：seen-day unseen-rx；
- UDS：unseen-day seen-rx；
- UDU：unseen-day unseen-rx；
- strict UDU：最严格跨日跨接收机测试；
- satellite floor：不同星地场景中的最低准确率；
- worst-rx：最弱接收机准确率；
- rollback：best epoch 与 final epoch 差距；
- latency：单样本或 batch 推理延迟；
- accepted/coverage：若使用拒识或协作筛选，必须同时报告覆盖率。

### 7.4 消融矩阵

建议按四组消融组织论文结果。

第一组是结构消融：

- ID Time/Freq/PA；
- ID-side DAC 关闭与打开；
- Domain DAC 打开与关闭；
- Domain DSQ 打开与关闭；
- RCN enhancer 打开与关闭；
- CosFace vs linear head。

第二组是解耦损失消融：

- no GRL；
- no orth；
- no same-TX consistency；
- no GroupCE/SmoothDRO；
- no feature-norm guard。

第三组是少样本策略消融：

- per-combo cap vs per-class K-shot；
- K5/K10/K20/K30/K50/K100；
- fixed final vs best-primary/SWAD；
- low-K weak DG vs full DG。

第四组是星地增强消融：

- no satellite；
- CE-only satellite；
- full-DG satellite；
- satellite consistency；
- clear LEO、low-elev LEO、rain LEO、storm_mp、mixed_orbit。

## 8. 与 RIEI/DRIFT 的机制比较

### 8.1 RIEI

RIEI 的关键是把 received IQ 的表示拆成 emitter-related feature 和 receiver-related feature，并通过互信息、熵或独立性约束强化 receiver-independent emitter representation。它与 CVS-RFFI 都承认 receiver shift 是跨接收机 RFFI 的根本问题。

差别在于，CVS-RFFI 的解耦由模型结构和训练目标共同承担。`z_id` 和 `z_dom` 来自两个物理分工不同的骨干；Domain DAC 与 DSQ 有意吸收域扰动；ID Backbone 通过关闭 ID-side DAC、保留 Time/Freq/PA 和 CosFace 保护身份边界。RIEI 更像“解耦目标驱动”，CVS-RFFI 更像“物理结构加解耦目标共同驱动”。

### 8.2 DRIFT

DRIFT 类方法强调对抗训练、中心约束、负 MSE 或 feature separation，以使 TX 表征更难携带 receiver-specific 信息。其优势是方法清晰，适合用标准 backbone 实现；风险是小样本下对抗梯度和中心估计噪声大，K 很小时可能先破坏 TX 可分性。

CVS-RFFI 对 DRIFT 的核心补充是 shot-aware 调度。K5/K10 不默认堆强对抗和强统计正则，而先通过 TX CE、轻量 GRL 和 norm guard 稳住 identity；K20+ 再恢复 GroupCE、prototype、SupCon 和 satellite view。

### 8.3 Satellite trade-off

RIEI/DRIFT 若把星地信道只当作外部扰动，容易形成两种极端：不增强时 satellite floor 低；强增强时 clean OOD 降。CVS-RFFI 的目标是在 identity-domain 解耦框架内吸收扰动：satellite view 给模型提供同标签传播变化，Domain Backbone 和 `z_dom` 吸收 nuisance，`z_id` 保持 TX identity。

该机制并不保证 satellite robustness 单调提升。多普勒、CFO、相位噪声、低 SNR、天气衰落和多径都会改变 PA、I/Q、频谱细节。若增强过早或过强，模型会先学习抗信道扰动，而不是稳定发射机边界。论文报告应把 satellite robustness 与 clean strict UDU 作为双目标，而不是只选择其中一个。

### 8.4 数值证据绑定

从 PPT 结果看，CVS 在 strict UDU 上的优势最稳定：K10 到 K100 分别达到 76.25、77.56、79.91、84.11、84.05。对应最强 baseline 分别约为 67.64、69.34、70.60、74.40、77.02，CVS 的领先幅度约为 7.03 到 9.71 个百分点。这个结果与方法假设一致：strict UDU 同时改变 day 和 receiver，最容易触发 receiver/day shortcut，CVS 的 `z_id/z_dom` 解耦和 domain-aware 正则在这里收益最大。

SDU 上 CVS 也保持 79.06 到 89.22 的优势区间，说明未见接收机是 CVS 的主要改进对象。UDS 上 CVS 与 RIEI 在 K20、K30、K100 接近，说明只改变 day 时，RIEI 的解耦约束也能保留较强 emitter representation。论文讨论不应把所有 OOD 统一描述为同一难度，而应强调 strict UDU 是区分方法的关键指标。

Satellite 增强的数值结果更能说明 trade-off。RIEI+Sat 在 strict UDU 上从 K10 的 66.98 降到 K100 的 46.32，DRIFT+Sat 在 K20/K30 短暂优于 DRIFT，但 K50/K100 低于 DRIFT。这个趋势说明把 satellite view 作为外部强扰动直接叠加到 baseline，并不自动带来跨域收益。CVS-RFFI 的优势来自“物理增强 + 表示解耦 + shot-aware 调度”的组合，而不是 satellite augmentation 单独起作用。

## 9. 论文贡献写法

建议将 CVS-RFFI 的贡献写成四点。

第一，提出面向天基 RFFI 的少样本域泛化任务定义。该任务同时包含跨接收机、跨日期、星地信道扰动和有限标注，不再把 few-shot 与 domain generalization 分开评估。

第二，提出物理先验驱动的双骨干解耦网络。ID Backbone 通过 Sinc 时域、mirror FFT 频域和 PA 记忆多项式学习发射机身份；Domain Backbone 通过 Domain DAC、DSQ 和 RCN 统计吸收 receiver/day/channel nuisance。

第三，提出 shot-aware 的多目标训练体系。训练目标组合 TX CE、Domain CE、GRL、orth、same-TX consistency、GroupCE、prototype、SupCon、Fishr proxy、satellite CE 和 norm guard，并通过 domain gate 与阶段调度避免低 K 过正则。

第四，构造物理约束的星地信道增强。增强算子将 clean IQ 转为同标签 satellite-view IQ，覆盖路径损耗、大气衰落、小尺度衰落、多径、多普勒/CFO、相位噪声、AGC/AWGN 和 I/Q 失衡，用于评估和提升天基部署鲁棒性。

## 10. 风险边界与后续工作

### 10.1 当前报告边界

本报告不把 StyleBank 写入当前 CVS-RFFI 四份材料的方法模块。StyleBank、FederatedStyleBank 或 VirtualDomainSampler 属于此前 federated DG 设计线，若未来要纳入论文，需要单独提供代码路径、实验结果和消融证据。

本报告也不把 Fishr 写成完整 per-parameter Fishr。当前材料中的 Fishr 是 logit-gradient variance 的轻量 proxy，适合中高 K 或 domain 统计稳定时轻量打开。

### 10.2 模型复杂度

PPT 中提到当前 CVS 模型统计量较多，参数量虽不大，但推理延迟约 12 ms。后续论文应补充：

- 参数量；
- FLOPs；
- 单样本 latency；
- batch latency；
- CPU/GPU/边缘设备测试；
- 剪枝或蒸馏前后准确率与延迟对比。

这部分应写成方法边界，而不是简单缺点。CVS-RFFI 的统计分支、PA 记忆项、RCN 和多目标评估提高了物理可解释性，但它们会带来额外预处理和推理开销。若目标是星上实时部署，论文需要给出“准确率-延迟”Pareto 曲线，并比较三种压缩路线：删除低贡献统计项、剪枝轻量卷积层、用 CEN51 作为 teacher 蒸馏小 student。只有在 strict UDU 和 satellite floor 不明显下降时，延迟优化才是有效部署优化。

### 10.3 实验完整性

当前对比方法仍偏少。建议扩展：

- ORACLE-style CNN；
- ResNet-18 RFFI；
- RIEI；
- DRIFT；
- receiver-agnostic adversarial training；
- single-source DG augmentation；
- no-physics generic augmentation；
- CVS-RFFI without satellite；
- CVS-RFFI with CE-only satellite；
- CVS-RFFI with full-DG satellite。

### 10.4 天基评估

星地增强只是物理约束仿真。后续若能获得真实卫星链路 IQ，应做两类验证：

- 仿真到真实的 transfer gap；
- satellite floor 是否对应真实低仰角、雨衰、遮挡、多径和低 SNR 难例。

## 11. 质量控制与遗漏检查

本报告按子代理审计清单做了以下约束：

- 没有把 satellite augmentation 写成模型内部新增 branch。
- 没有把 6 TX 实验子集与 16-class head 混成同一口径。
- 没有把 per-combo cap 当作严格 per-class K-shot。
- 没有把 CEN51、CVS、CV-SincNet、CVS-RFFI 写成并列四个模型。
- 没有把 StyleBank 写成当前四份材料中的模块。
- 将 Fishr 写成轻量 proxy，而非完整 Fishr。
- 将 ID-side DAC 关闭与 Domain-side DAC 启用分开说明。
- 将 GRL 解释为 `z_id` 去域化，而不是简单让 domain classifier 失败。

## 12. 可直接用于论文的精简方法描述

CVS-RFFI receives raw IQ samples and learns a disentangled representation for transmitter identification under receiver, date, and satellite-channel shifts. The model contains two physics-aware CV-SincNet backbones. The identity backbone extracts transmitter-related cues from Sinc time filters, mirror-spectrum features, and PA memory-polynomial traces, while suppressing domain leakage through adversarial receiver-day prediction and covariance disentanglement. The domain backbone absorbs receiver and channel nuisance using receiver-day supervision, Domain DAC features, spectral stability residuals, and raw-IQ statistical enhancement. For few-shot training, CVS-RFFI uses a shot-aware objective that prioritizes transmitter classification and feature-norm control at extremely low shots, then gradually restores group-aware CE, prototype memory, supervised contrastive learning, Fishr-style gradient variance matching, and physically constrained satellite-view augmentation as domain statistics become reliable.

中文版可写为：

CVS-RFFI 以 raw IQ 为输入，面向跨接收机、跨日期和星地信道扰动学习发射机身份表征。模型由两个具有物理先验的 CV-SincNet 骨干组成：身份骨干从 Sinc 时域滤波、镜像频谱和 PA 记忆多项式中提取 TX 相关线索，并通过 GRL 与协方差解耦压制接收域泄漏；域骨干通过 receiver-day 监督、Domain DAC、DSQ 频谱稳定残差和 raw-IQ 统计增强吸收接收机、日期和信道 nuisance。在少样本训练中，CVS-RFFI 采用 shot-aware 多目标损失：极低 K 时优先保护 TX identity 和特征范数稳定；当跨域样本对与 domain 统计足够可靠后，再逐步恢复 GroupCE、prototype、SupCon、Fishr proxy 和物理约束 satellite-view 增强。

## 13. 参考资料

### 13.1 本地材料

[S1] `C:/Users/lh594/Desktop/CVS-RFFI_model_design_notes - 副本.pptx`，16 页 PPT，已抽取文本至 `E:/type10-7/code/analysis/cvs_rffi_paper_report_20260611/pptx_text_extract.md`。  
[S2] `C:/Users/lh594/Desktop/CVS报告/cvs_cen51_branch_dataflow.md`。  
[S3] `C:/Users/lh594/Desktop/CVS报告/cvs_fewshot_loss_functions_20260610.md`。  
[S4] `C:/Users/lh594/Desktop/CVS报告/satellite_ground_channel_augmentation_principles.md`。

### 13.2 外部 RFFI 代表文献

[1] Brik et al., “Wireless Device Identification with Radiometric Signatures,” ACM MobiCom, 2008. URL: https://oamonitor.ireland.openaire.eu/rpo/rcsi/search/publication?pid=10.1145%2F1409944.1409959  
[2] Bihl, Bauer, and Temple, “Feature Selection for RF Fingerprinting With Multiple Discriminant Analysis and Using ZigBee Device Emissions,” IEEE TIFS, 2016. URL: https://www.researchgate.net/publication/303048268_Feature_Selection_for_RF_Fingerprinting_With_Multiple_Discriminant_Analysis_and_Using_ZigBee_Device_Emissions  
[3] Sankhe et al., “No Radio Left Behind: Radio Fingerprinting Through Deep Learning of Physical-Layer Hardware Impairments,” IEEE TCCN, 2019. URL: https://www.genesys-lab.org/oracle  
[4] Al-Shawabka et al., “Exposing the Fingerprint: Dissecting the Impact of the Wireless Channel on Radio Fingerprinting,” IEEE INFOCOM, 2020. URL: https://dl.acm.org/doi/10.1109/INFOCOM41043.2020.9155259  
[5] Hanna, Karunaratne, and Cabric, “WiSig: A Large-Scale WiFi Signal Dataset for Receiver and Channel Agnostic RF Fingerprinting,” IEEE Access, 2022. URL: https://cores.ee.ucla.edu/downloads/datasets/wisig/  
[6] Shen et al., “Towards Scalable and Channel-Robust Radio Frequency Fingerprint Identification for LoRa,” IEEE TIFS, 2022. URL: https://junqing-zhang.github.io/dataset-code/  
[7] Shen et al., “Towards Length-Versatile and Noise-Robust Radio Frequency Fingerprint Identification,” IEEE TIFS, 2023. URL: https://junqing-zhang.github.io/dataset-code/  
[8] Shen et al., “Towards Receiver-Agnostic and Collaborative Radio Frequency Fingerprint Identification,” IEEE TMC, 2024. URL: https://arxiv.org/abs/2207.02999  
[9] Zhao et al., “GAN-RXA: A Practical Scalable Solution to Receiver-Agnostic Transmitter Fingerprinting,” IEEE TCCN, 2024. URL: https://cores.ee.ucla.edu/research/rf-transmitter-fingerprinting-using-deep-learning/  
[10] Shen et al., “Federated Radio Frequency Fingerprint Identification Powered by Unsupervised Contrastive Learning,” IEEE TIFS, 2024. URL: https://www.eng.auburn.edu/~szm0001/papers/tifs24.pdf  
[11] Zhang et al., “Domain Generalization for Cross-Receiver Radio Frequency Fingerprint Identification,” arXiv, 2024. URL: https://arxiv.org/abs/2411.03636  
[12] Wang et al., “Avoiding Shortcuts: Enhancing Channel-Robust Specific Emitter Identification via Single-Source Domain Generalization,” IEEE TWC, 2025. URL: https://keio.elsevierpure.com/en/publications/avoiding-shortcuts-enhancing-channel-robust-specific-emitter-iden-2/  
[13] Oligeri et al., “PAST-AI: Physical-Layer Authentication of Satellite Transmitters via Deep Learning,” IEEE TIFS, 2023. URL: https://colab.ws/articles/10.1109/tifs.2022.3219287  
[14] Soltani et al., “More Is Better: Data Augmentation for Channel-Resilient RF Fingerprinting,” IEEE Communications Magazine, 2020. URL: https://www.semanticscholar.org/paper/More-Is-Better%3A-Data-Augmentation-for-RF-Soltani-Sankhe/ea7bf9142e0122d6ebefd5dad0ddf87e630a7e33
