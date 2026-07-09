# CVS阶段性成果技术报告：`ADV3B02_CORE90_SOFT_E200`与`qKNNV42`

日期：2026-07-09
修订说明：本版按用户要求重写模型分支、损失公式、创新模块和`qKNNV42`章节。所有符号和公式均使用LaTeX格式；`qKNNV42`只作为Phase2 Stage2-C轻量注册/适应头描述，类别集合限定为旧类和seen-new新类；ADV3B02创新模块C改写为源域core/tail几何稳健性模块。

## 1.结论边界

本文将CVS当前成果拆成两个阶段：

|阶段|对象|可声明内容|不可声明内容|
|---|---|---|---|
|Phase1|`ADV3B02_CORE90_SOFT_E200`|source-only弱标注/半监督跨接收机DG表征基座；输出可用于后续目标域注册的身份表征`z_id`和域表征`z_dom`|不能单独声明Phase2 target-old/target-new完成|
|Phase2 Stage2-C|`qKNNV42`|冻结ADV3B02表征后的目标域K-shot support-memory注册/适应头；目标是target-old保留和target-new seen-new注册|不是新backbone，不是端到端训练模型，不覆盖Phase2以外目标|

当前组合成果来自N20 HP08L5注册新类包的同row候选：

|seed|K old|K new|old acc|min old|seen-new acc|min new|H old-new|verdict|
|---:|---:|---:|---:|---:|---:|---:|---:|---|
|421070|5|5|94.52%|85.71%|90.14%|81.43%|92.28%|Phase2 K5旧类适应+seen-new注册候选|

其中

$$
H_{\mathrm{old,new}}=
\frac{2\,A_{\mathrm{old}}A_{\mathrm{new}}}{A_{\mathrm{old}}+A_{\mathrm{new}}}
=\frac{2\times0.945238\times0.901429}{0.945238+0.901429}
=0.922814.
$$

## 2.符号与任务定义

地面训练阶段使用源域集合$\mathcal R_s$和旧类集合$\mathcal Y_{\mathrm{old}}$。目标域注册阶段使用目标接收机域$\mathcal R_t$，并满足

$$
\mathcal R_t\cap\mathcal R_s=\varnothing,\qquad
\mathcal Y_{\mathrm{new}}\cap\mathcal Y_{\mathrm{old}}=\varnothing.
$$

样本和标签记为

$$
x\in\mathbb R^{2\times L},\qquad
y\in\mathcal Y_{\mathrm{old}},\qquad
d\in\mathcal D_s,
$$

其中$x$是IQ片段，$y$是TX身份，$d$是receiver/day/rx_day/channel view等域标签。CVS的观测模型写为

$$
x=R_d\!\left(H_d*T_y(s)\right)+n,
$$

其中$T_y$是发射机硬件非理想性，$H_d$是传播/星地信道扰动，$R_d$是接收机链路响应，$n$是噪声。

Phase1训练集为

$$
\mathcal L_s=\{(x_i,y_i,d_i)\}_{i=1}^{N_l},\qquad
\mathcal U_s=\{(u_j,d_j)\}_{j=1}^{N_u},
$$

标注比例为

$$
\rho_{\mathrm{label}}=
\frac{|\mathcal L_s|}{|\mathcal L_s|+|\mathcal U_s|}
=0.1.
$$

模型输出定义为

$$
f_\theta(x)=
\left(
\boldsymbol\ell_y,\boldsymbol\ell_d,\boldsymbol\ell_{\mathrm{adv}},
\mathbf z_{\mathrm{id}},\mathbf z_{\mathrm{dom}}
\right),
$$

其中$\boldsymbol\ell_y$是TX分类logits，$\boldsymbol\ell_d$是域分类logits，$\boldsymbol\ell_{\mathrm{adv}}$是经GRL后的域对抗logits，$\mathbf z_{\mathrm{id}}$用于身份分类、prototype和qKNN，$\mathbf z_{\mathrm{dom}}$用于接收机/信道扰动建模。

## 3.ADV3B02实际模型配置

`ADV3B02_CORE90_SOFT_E200`由`code/SSDG/train_ssdg.py`构建，实际使用默认结构参数：

|参数|值|
|---|---|
|`model_size`|`M`|
|`model_variant`|`lite_d`|
|`arch_family`|`cvsincnet`|
|`input_len`|`L=256`，由WiSig/ManySig数据上下文传入|
|`num_classes`|`C=6`，即`Y_old`旧类数量|
|`num_domains`|`D`为源域receiver/day映射得到的domain数量|
|identity branch ablation|`no_dac`|
|domain branch ablation|`no_stats`|
|shared stem|`lite_d`触发`sinc`和`hf`在identity/domain分支间共享|
|identity feature key|`feat_joint`|
|domain feature key|`feat_imp`|
|domain enhancer|`rcn_stats`，`strength=0.35`|

`lite_d`将CVSincNet主干缩窄为：

|宽度参数|值|
|---|---:|
|`sinc_out`|24|
|`sinc_kernel`|79|
|`time_bottleneck`|48|
|`emb_dim`|160|
|`freq_bands`|32|
|`time_ch1,time_ch2,time_ch3`|72,96,96|
|`dac_ch`|32|
|`freq_ch1,freq_ch2,freq_ch3`|16,32,32|
|`pa_ch1,pa_ch2,pa_ch3`|48,64,64|
|`pa_memory_depth`|4|
|`pa_orders`|`(1,3,5)`|
|`drop`|0.45|

由于identity branch使用`no_dac`，身份分支启用time/frequency/PA路径，不启用DAC路径。domain branch使用`no_stats`，保留time/DAC/frequency/PA路径，但禁用频谱统计投影、DAC subband聚合和PA统计增量。

## 4.ADV3B02分支逐层结构

### 4.1共享Sinc/IQ前端

共享前端接收$x\in\mathbb R^{B\times2\times256}$。

|层|参数|输出|
|---|---|---|
|`SincConv1d.forward_iq_pair`|`out_channels=24`，`kernel=79`，`padding=39`，`stride=1`，每个IQ通道共享同一24个可学习带通滤波器|`s: Bx48x256`|
|`HighFreqEmphasis`|固定一阶差分核`[-1,1]`和二阶差分核`[1,-2,1]`，`groups=2`|`h: Bx4x256`|

Sinc滤波器参数为每个带通滤波器的低频和带宽：

$$
\mathbf w_k(t)=
\frac{\sin(2\pi f_{2,k}t)-\sin(2\pi f_{1,k}t)}{\pi t}\cdot
\left(0.54-0.46\cos\frac{2\pi n}{78}\right),
$$

其中$f_{1,k}$和$f_{2,k}$由`low_hz_`和`band_hz_`学习得到。

### 4.2identity branch：time path

identity branch的time path把Sinc IQ、三阶非线性基和高频差分拼接：

$$
\mathbf t_0=
\operatorname{concat}\left[
\mathbf s,\;
\mathbf s|\mathbf s|^2,\;
\mathbf h
\right]\in\mathbb R^{B\times100\times256}.
$$

|层|参数|输出通道|
|---|---|---:|
|`time_fuse`|`Conv1d 100->48`，`kernel=1`，`bias=0`；`GroupNorm g=16`；ReLU|48|
|`time_down`|AvgPool1d，`kernel=2`|48|
|`MixStyle`|仅identity branch；`layers=time_down,t1`；`p=0.18`，`alpha=0.10`，`mix=same_tx_crossdomain`，`strength=0.70`，`fallback=skip`，`late_start=110`|48|
|`t1`|`Depthwise Conv1d 48->48`，`kernel=5`，`groups=48`；`Pointwise Conv1d 48->72`；`GroupNorm g=8`；ReLU；`MaxPool1d 2`；`Dropout 0.10`|72|
|`t2`|`Depthwise Conv1d 72->72`，`kernel=5`，`groups=72`；`Pointwise Conv1d 72->96`；`GroupNorm g=16`；ReLU；`MaxPool1d 2`；`Dropout 0.10`|96|
|`t3`|`Depthwise Conv1d 96->96`，`kernel=3`，`groups=96`；`Pointwise Conv1d 96->96`；`GroupNorm g=16`；ReLU；`pool=Identity`；`Dropout 0.10`|96|
|`t_pool`|`AdaptiveAvgPool1d 1`|96|
|`t_proj`|`Linear 96->160`|160|

输出为$\mathbf e_t\in\mathbb R^{B\times160}$。

### 4.3identity branch：frequency path

frequency path从原始IQ的镜像FFT统计构造

$$
\mathbf f_0=
\left[
\log(1+P_+),\;
\log(1+P_-),\;
\log\frac{P_++\epsilon}{P_-+\epsilon},\;
\frac{|P_+-P_-|}{P_++P_-+\epsilon}
\right]\in\mathbb R^{B\times4\times32}.
$$

|层|参数|输出通道|
|---|---|---:|
|`freq_gate`|`Conv1d 4->1`，`kernel=5`，`padding=2`；gate scale见下方公式说明|4|
|`f1`|`Depthwise Conv1d 4->4`，`kernel=5`，`groups=4`；`Pointwise Conv1d 4->16`；`GroupNorm g=16`；ReLU；`MaxPool1d 2`；`Dropout 0.05`|16|
|`f2`|`Depthwise Conv1d 16->16`，`kernel=5`，`groups=16`；`Pointwise Conv1d 16->32`；`GroupNorm g=16`；ReLU；`MaxPool1d 2`；`Dropout 0.05`|32|
|`f3`|`Depthwise Conv1d 32->32`，`kernel=3`，`groups=32`；`Pointwise Conv1d 32->32`；`GroupNorm g=16`；ReLU；`pool=Identity`；`Dropout 0.05`|32|
|`f_pool`|`AdaptiveAvgPool1d 1`|32|
|`f_proj`|`Linear 32->160`|160|
|`freq_stats_proj`|`Linear 3->160`；ReLU；`Dropout 0.1125`|160|

输出为

frequency gate的缩放形式为

$$
g_{\mathrm{freq}}=1+0.6\left(2\sigma(\cdot)-1\right).
$$

$$
\mathbf e_f=\operatorname{Linear}_{32\to160}(\operatorname{pool}(\mathbf f_3))
+\operatorname{MLP}_{3\to160}(\mathbf r_{\mathrm{freq}}).
$$

其中$\mathbf r_{\mathrm{freq}}$包含高频能量比例、镜像不对称均值和谱平坦度。

### 4.4identity branch：PA path

PA path使用memory polynomial lift。对$m\in\{0,1,2,3\}$和$p\in\{1,3,5\}$：

$$
\phi_{m,p}(x)[n]=x[n-m]\left|x[n-m]\right|^{p-1}.
$$

因此输入通道数为$2\times4\times3=24$。

|层|参数|输出通道|
|---|---|---:|
|`pa_lift`|`memory_depth=4`，`orders=(1,3,5)`，`clip=2.0`|24|
|`pa_gate`|`EnvelopeGate Conv1d 1->24`，`kernel=5`，`padding=2`，`alpha=0.5`|24|
|`pa_b1`|`Conv1d 24->48`，`kernel=7`，`dilation=1`，`padding=3`；`GroupNorm g=16`；SiLU；`AvgPool1d 2`；`Dropout 0.08`|48|
|`pa_b2`|`Conv1d 48->64`，`kernel=7`，`dilation=2`，`padding=6`；`GroupNorm g=16`；SiLU；`AvgPool1d 2`；`Dropout 0.08`|64|
|`pa_b3`|`Conv1d 64->64`，`kernel=5`，`dilation=4`，`padding=8`；`GroupNorm g=16`；SiLU；`pool=Identity`；`Dropout 0.08`|64|
|`pa_pool`|`AdaptiveAvgPool1d 1`|64|
|`pa_proj`|`Linear 64->160`；ReLU；`Dropout 0.1125`|160|
|`pa_stats_proj`|`Linear 3->160`；ReLU；`Dropout 0.1125`|160|

输出为

$$
\mathbf e_{\mathrm{pa}}=
\operatorname{MLP}_{64\to160}(\operatorname{pool}(\mathbf p_3))
+0.25\,\operatorname{MLP}_{3\to160}(\mathbf r_{\mathrm{pa}}),
$$

其中$\mathbf r_{\mathrm{pa}}$包含edge ratio、regrowth ratio和谱峰度。

### 4.5identity branch：融合和分类头

由于identity branch关闭DAC路径，base输入为

$$
\mathbf b_{\mathrm{in}}=
\operatorname{concat}(\mathbf e_t,\mathbf e_f,\rho)
\in\mathbb R^{B\times321},
$$

其中$\rho=|\mathbb E[z^2]|/(\mathbb E[|z|^2]+\epsilon)$是circularity统计。

|层|参数|输出|
|---|---|---|
|`fuse`|`Linear 321->160`；ReLU；`Dropout 0.45`|`b: Bx160`|
|`con_proj`|`Linear 160->160`；ReLU；`Dropout 0.1125`|`feat_con`|
|`id_proj`|`Linear 160->160`；ReLU；`Dropout 0.225`|`feat_cls`|
|`pa_proj` in classifier|`Linear 320->160`；ReLU；`Dropout 0.225`|`feat_pa`|
|`id_gate`|`Linear 160->160`；Sigmoid；`gate_alpha=0.35`|identity gate|
|`joint_proj`|`Linear 320->160`；ReLU；`Dropout 0.225`|`feat_joint=z_id`|
|`imp_merge`|`Linear 160->160`；ReLU；`Dropout 0.1125`|`feat_imp`|
|`CosFaceHead`|`weight: 6x160`，`scale=30`，`margin=0.35`|`tx logits`|
|`pa_head`|`Linear 160->80`；ReLU；`Linear 80->1`；Sigmoid|PA辅助强度预测|

CosFace logits为

$$
\ell_{y,c}
=s\left(
\frac{\mathbf z_{\mathrm{id}}^\top\mathbf w_c}
{\|\mathbf z_{\mathrm{id}}\|_2\|\mathbf w_c\|_2}
-m\cdot\mathbb 1[c=y]
\right).
$$

### 4.6domain branch：域表征路径

domain branch使用同样的`lite_d`主干，但配置为`domain_branch_ablation=no_stats`和`mixstyle_on=False`。它保留time、DAC、frequency、PA路径，禁用统计投影增量。time、frequency、PA路径的卷积层宽与identity branch相同；不同点如下：

|组件|domain branch实际参数|
|---|---|
|MixStyle|关闭|
|stats path|关闭，`freq_stats_proj`、`pa_stats_proj`、`dac_subband_agg`不参与|
|DAC HF projection|`Conv1d 4->48`，`kernel=1`，`bias=0`；`GroupNorm g=16`；SiLU|
|`dac_b1`|WLComplexBlock，`complex Conv 24->24`，`kernel=5`，`dilation=1`，`pool=Identity`，`Dropout 0.05`，`residual=Identity`|
|`dac_b2`|WLComplexBlock，`complex Conv 24->32`，`kernel=3`，`dilation=1`，`pool=Identity`，`Dropout 0.05`，`residual Conv1d 48->64`|
|`dac_b3`|WLComplexBlock，`complex Conv 32->32`，`kernel=3`，`dilation=2`，`AvgPool1d 2`，`Dropout 0.05`，`residual Conv1d 64->64 + AvgPool1d 2`|
|`dac_proj`|`Linear 64->160`；ReLU；`Dropout 0.1125`|

WL complex convolution使用四个实卷积：

$$
\begin{aligned}
\mathbf y_r&=W_r\mathbf x_r-W_i\mathbf x_i+V_r\mathbf x_r+V_i\mathbf x_i,\\
\mathbf y_i&=W_r\mathbf x_i+W_i\mathbf x_r-V_r\mathbf x_i+V_i\mathbf x_r.
\end{aligned}
$$

domain backbone输出`feat_imp`作为$\mathbf z_{\mathrm{dom,raw}}$。随后`DomainFeatureEnhancer`把原始IQ的receiver/channel/noise统计注入域表征：

|层|参数|输出|
|---|---|---|
|`RCNStatEncoder`|18维IQ统计；`Linear 18->80`；`LayerNorm 80`；SiLU；`Dropout 0.05`；`Linear 80->160`|`z_rcn`|
|gate|`Linear 320->160`；Sigmoid|`g_rcn`|
|enhance|LayerNorm残差增强，公式见下文|`z_dom`|

域分类头和对抗头为

域增强的显式形式为

$$
\mathbf z_{\mathrm{dom}}
=\operatorname{LayerNorm}
\left(
\mathbf z_{\mathrm{dom,raw}}
+0.35\,\mathbf g_{\mathrm{rcn}}\odot\mathbf z_{\mathrm{rcn}}
\right).
$$

|层|参数|输出|
|---|---|---|
|`dom_head`|`Linear 160->80`；ReLU；`Dropout 0.10`；`Linear 80->D`|`domain logits`|
|`adv_head`|`GRL(z_id)`；`Linear 160->80`；ReLU；`Dropout 0.10`；`Linear 80->D`|`adversarial domain logits`|
|`tx_adv_head`|默认关闭|无|

GRL定义为

$$
\operatorname{GRL}_{\lambda}(\mathbf z)=\mathbf z,\qquad
\frac{\partial\operatorname{GRL}_{\lambda}}{\partial\mathbf z}=-\lambda\mathbf I.
$$

## 5.训练参数

|类别|参数|值|
|---|---|---|
|数据|`wisig_pkl`|`Dataset_WigSig/ManySig.pkl`|
|数据|`split_mode`|`tx_rx_day_1_7_2`|
|数据|`labeled_ratio,unlabeled_ratio,source_val_ratio`|`0.10,0.70,0.20`|
|优化器|AdamW|`learning_rate=2e-4`，`weight_decay=1e-4`|
|训练轮数|`epochs`|200|
|阶段|`label_epochs,pseudo_epochs`|130,70|
|label smoothing|`epsilon_ls`|0.01|
|伪标签|`tau_min,tau_max,pseudo_quantile`|`0.92,0.97,0.86`|
|伪标签|`pseudo_threshold_mode`|`rx_day_quantile`|
|伪标签|`use_ema_teacher`|true|
|prototype|`lambda_proto`|0.0032|
|prototype|`proto_domain_align_weight,proto_margin,proto_push_weight`|`0.10,0.15,0.10`|
|身份几何|`lambda_zid_compact`|0.032|
|身份几何|SupCon/radius/CVaR权重|`0.30,0.35,0.35`|
|身份几何|radius,CVaR alpha|`40 deg,0.95`|
|源域边界|历史CLI名`lambda_proxy_unknown`|0.0045|
|源域边界|core/accept/tail/overflow quantile|`0.90,0.85,0.92,0.97`|
|源域边界|core/component/tail/source权重|`0.45,0.65,0.20,0.20`|
|源域边界|CVaR alpha|0.30|
|类间软混合|历史CLI名`lambda_soft_unknown_mixup`|0.0045|
|类间软混合|count/order/alpha|`24,3,0.5`|
|类间软混合|CE/energy/vacuum权重|`0.60,1.0,0.35`|
|source episode|`lambda_source_episode`|0.0035|
|source episode|start/warmup/min domains/radius cap|`20,25,2,33 deg`|
|源域LEO压力|`sat_train_scenarios`|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|源域LEO压力|`lambda_sat_cls,lambda_sat_cons`|`0.68,0`|
|域损失|`lambda_domain,lambda_adv`|`1,0.35`|
|Group/FishR|`lambda_group_ce,lambda_fishr`|`0.16,0.04`|
|无标签|`lambda_u,lambda_ent`|`0.16,0.01`|
|checkpoint|`best_metric`|`joint_safe`|

## 6.训练损失函数

### 6.1监督TX分类

带label smoothing的目标分布为

$$
\tilde{\mathbf y}_{i,c}=
(1-\varepsilon_{\mathrm{ls}})\mathbb 1[c=y_i]
+\frac{\varepsilon_{\mathrm{ls}}}{C}.
$$

监督TX分类损失为

$$
\mathcal L_{\mathrm{tx}}
=-\frac{1}{|\mathcal B_l|}
\sum_{i\in\mathcal B_l}
\sum_{c=1}^{C}
\tilde{\mathbf y}_{i,c}\log p_\theta(c|x_i).
$$

### 6.2域监督与域对抗

域监督损失为

$$
\mathcal L_{\mathrm{dom}}
=-\frac{1}{|\mathcal B_l|}
\sum_{i\in\mathcal B_l}
\log p_\phi(d_i|\mathbf z_{\mathrm{dom},i}).
$$

域对抗头从$\operatorname{GRL}(\mathbf z_{\mathrm{id}})$预测域标签：

$$
\mathcal L_{\mathrm{adv}}
=-\frac{1}{|\mathcal B_l|}
\sum_{i\in\mathcal B_l}
\log p_\psi(d_i|\operatorname{GRL}_{1.0}(\mathbf z_{\mathrm{id},i})).
$$

反向传播时$\mathcal L_{\mathrm{adv}}$对$\mathbf z_{\mathrm{id}}$的梯度符号被反转，从而压低身份表征中的域可预测性。

### 6.3GroupCE与FishR

设$\mathcal G$为源域group集合，$\mathcal B_g=\{i\in\mathcal B_l:g_i=g\}$。GroupCE写为

$$
\mathcal L_{\mathrm{group}}
=\sum_{g\in\mathcal G}
\alpha_g
\left(
-\frac{1}{|\mathcal B_g|}
\sum_{i\in\mathcal B_g}
\log p_\theta(y_i|x_i)
\right),
$$

其中$\alpha_g$由实现中的弱域/困难域策略确定。FishR约束各group梯度方差一致：

$$
\mathcal L_{\mathrm{fishr}}
=
\sum_{g\in\mathcal G}
\left\|
\operatorname{Var}_{i\in\mathcal B_g}
\left(\nabla_{\mathbf z_{\mathrm{id}}}\ell_i\right)
-
\frac{1}{|\mathcal G|}
\sum_{g'\in\mathcal G}
\operatorname{Var}_{i\in\mathcal B_{g'}}
\left(\nabla_{\mathbf z_{\mathrm{id}}}\ell_i\right)
\right\|_2^2.
$$

### 6.4prototype几何损失

类原型为

$$
\boldsymbol\mu_c=
\frac{\sum_{i:y_i=c}\mathbf z_{\mathrm{id},i}}
{\left\|\sum_{i:y_i=c}\mathbf z_{\mathrm{id},i}\right\|_2+\epsilon}.
$$

prototype pull/push可写为

$$
\mathcal L_{\mathrm{proto}}
=
\frac{1}{|\mathcal B_l|}
\sum_{i\in\mathcal B_l}
\left(1-\cos(\mathbf z_{\mathrm{id},i},\boldsymbol\mu_{y_i})\right)
+
\frac{1}{C(C-1)}
\sum_{c\ne c'}
\max\left(0,m_{\mathrm{proto}}-\left(1-\cos(\boldsymbol\mu_c,\boldsymbol\mu_{c'})\right)\right).
$$

### 6.5身份表征紧致性损失

Supervised contrastive部分为

$$
\mathcal L_{\mathrm{supcon}}
=
\frac{1}{|\mathcal B_l|}
\sum_{i\in\mathcal B_l}
\frac{-1}{|\mathcal P(i)|}
\sum_{p\in\mathcal P(i)}
\log
\frac{\exp(\cos(\mathbf z_i,\mathbf z_p)/\tau)}
{\sum_{a\in\mathcal B_l\setminus\{i\}}\exp(\cos(\mathbf z_i,\mathbf z_a)/\tau)}.
$$

角半径项为

$$
\mathcal L_{\mathrm{rad}}
=
\frac{1}{|\mathcal B_l|}
\sum_{i\in\mathcal B_l}
\max\left(0,\angle(\mathbf z_i,\boldsymbol\mu_{y_i})-r_{\mathrm{id}}\right)^2,
\qquad r_{\mathrm{id}}=40^\circ.
$$

tail CVaR项取每类角距离尾部均值：

$$
\mathcal L_{\mathrm{cvar}}
=
\frac{1}{C}\sum_{c=1}^{C}
\operatorname{CVaR}_{0.95}
\left(
\{\angle(\mathbf z_i,\boldsymbol\mu_c):y_i=c\}
\right).
$$

因此

$$
\mathcal L_{\mathrm{zid}}
=0.30\,\mathcal L_{\mathrm{supcon}}
+0.35\,\mathcal L_{\mathrm{rad}}
+0.35\,\mathcal L_{\mathrm{cvar}}.
$$

### 6.6known-core保真与尾部风险抑制

这一项在本文创新模块中按known-core保真和源域尾部风险抑制解释，代码参数名沿用`proxy_unknown_*`。给定一批源域类$\mathcal C_B$，每次留出一个源域TX类$c^-\in\mathcal C_B$构造边界压力样本，其他类作为known core。

类$c$的core半径和accept半径为

$$
r^{\mathrm{core}}_c=Q_{0.90}\left(\{\angle(\mathbf z_i,\boldsymbol\mu_c):y_i=c\}\right),
\qquad
r^{\mathrm{acc}}_c=Q_{0.85}\left(\{\angle(\mathbf z_i,\boldsymbol\mu_c):y_i=c\}\right).
$$

known-core保真损失：

$$
\mathcal L_{\mathrm{core}}
=
\frac{1}{|\mathcal B_l|}
\sum_{i\in\mathcal B_l}
\max\left(0,\angle(\mathbf z_i,\boldsymbol\mu_{y_i})-r^{\mathrm{core}}_{y_i}\right)^2.
$$

边界接收风险使用最小类角距离

$$
a(\mathbf z)=\min_{c\in\mathcal C_B\setminus\{c^-\}}
\angle(\mathbf z,\boldsymbol\mu_c),
$$

并定义尾部CVaR：

$$
\mathcal L_{\mathrm{tail}}
=
\operatorname{CVaR}_{0.30}
\left(
\{\max(0,r^{\mathrm{acc}}_{\hat c_j}-a(\mathbf v_j)):
\mathbf v_j\in\mathcal V_B\}
\right),
$$

其中$\mathcal V_B$是由源域留出类和类间扰动构造的边界压力集合，$\hat c_j$是最近known core类。component gate和source safe项可写为

$$
\mathcal L_{\mathrm{gate}}
=
\frac{1}{|\mathcal V_B|}
\sum_{\mathbf v\in\mathcal V_B}
\sigma\left(\frac{r^{\mathrm{acc}}_{\hat c}-a(\mathbf v)}{T_{\mathrm{comp}}}\right),
$$

$$
\mathcal L_{\mathrm{safe}}
=
\frac{1}{|\mathcal B_l|}
\sum_{i\in\mathcal B_l}
\max(0,a(\mathbf z_i)-r^{\mathrm{acc}}_{y_i})^2.
$$

合并为

$$
\mathcal L_{\mathrm{coretail}}
=
0.45\,\mathcal L_{\mathrm{core}}
+1.00\,\mathcal L_{\mathrm{tail}}
+0.65\,\mathcal L_{\mathrm{gate}}
+0.20\,\mathcal L_{\mathrm{safe}}.
$$

### 6.7类间软标签mixup

从不同TX类$a\ne b$抽取源域特征，构造

$$
\tilde{\mathbf z}
=\lambda\mathbf z_a+(1-\lambda)\mathbf z_b,\qquad
\tilde{\mathbf y}
=\lambda\mathbf e_a+(1-\lambda)\mathbf e_b,\qquad
\lambda\sim\operatorname{Beta}(0.5,0.5).
$$

软标签CE为

$$
\mathcal L_{\mathrm{mixCE}}
=-\sum_{c=1}^{C}\tilde y_c\log p_\theta(c|\tilde{\mathbf z}).
$$

能量项使用

$$
E(\tilde{\mathbf z})
=-\log\sum_{c=1}^{C}\exp(\ell_c(\tilde{\mathbf z})),
$$

并写为

$$
\mathcal L_{\mathrm{energy}}
=\max(0,m_E-E(\tilde{\mathbf z})).
$$

vacuum项把混合样本推离任何单一类的窄角锥：

$$
\mathcal L_{\mathrm{vac}}
=
\min_{c\in\{1,\ldots,C\}}
\max\left(0,r_{\mathrm{vac}}-\angle(\tilde{\mathbf z},\boldsymbol\mu_c)\right)^2.
$$

组合为

$$
\mathcal L_{\mathrm{softmix}}
=0.60\,\mathcal L_{\mathrm{mixCE}}
+1.00\,\mathcal L_{\mathrm{energy}}
+0.35\,\mathcal L_{\mathrm{vac}}.
$$

### 6.8source episode三sigma损失

对类$c$和源域$d$，用其他源域构造prototype：

$$
\boldsymbol\mu_{c}^{(-d)}
=
\frac{\sum_{i:y_i=c,d_i\ne d}\mathbf z_i}
{\left\|\sum_{i:y_i=c,d_i\ne d}\mathbf z_i\right\|_2+\epsilon}.
$$

令角距离的均值和标准差为$\bar r_c$和$\sigma_c$，episode半径为

$$
r_c^{\mathrm{epi}}
=
\min\left(33^\circ,\bar r_c+3\sigma_c\right).
$$

损失为

$$
\mathcal L_{\mathrm{epi}}
=
\frac{1}{|\mathcal B_l|}
\sum_{i\in\mathcal B_l}
\max\left(0,\angle(\mathbf z_i,\boldsymbol\mu_{y_i}^{(-d_i)})-r_{y_i}^{\mathrm{epi}}\right)^2.
$$

### 6.9源域伪标签SSL

EMA teacher参数为$\bar\theta$。对$u_j\in\mathcal U_s$，weak view预测为

$$
\mathbf q_j
=p_{\bar\theta}(y|a_w(u_j)),\qquad
\hat y_j=\arg\max_c q_{j,c},\qquad
\kappa_j=\max_c q_{j,c}.
$$

按域自适应阈值为

$$
\tau_d=
\operatorname{clip}
\left(
Q_{0.86}\left(\{\kappa_j:d_j=d\}\right),0.92,0.97
\right).
$$

门控变量为

$$
m_j=
\mathbb 1[\kappa_j\ge\tau_{d_j}]
\cdot\mathbb 1[\mathrm{DomainGate}(u_j)=1]
\cdot\mathbb 1[\mathrm{TemporalGate}(u_j)=1]
\cdot\mathbb 1[
\arg\max p_\theta(y|a_s(u_j))=\hat y_j
].
$$

伪标签CE为

$$
\mathcal L_u
=
-\frac{1}{\sum_j m_j+\epsilon}
\sum_{j\in\mathcal B_u}
m_j\log p_\theta(\hat y_j|a_s(u_j)).
$$

熵项为

$$
\mathcal L_{\mathrm{ent}}
=
\frac{1}{|\mathcal B_u|}
\sum_{j\in\mathcal B_u}
\sum_{c=1}^{C}
p_\theta(c|u_j)\log p_\theta(c|u_j).
$$

### 6.10源域LEO压力视图

对源域样本$x_i$构造LEO压力视图$a_{\mathrm{leo}}(x_i)$，其监督CE为

$$
\mathcal L_{\mathrm{satCE}}
=
-\frac{1}{|\mathcal B_l|}
\sum_{i\in\mathcal B_l}
\log p_\theta(y_i|a_{\mathrm{leo}}(x_i)).
$$

一致性项定义为

$$
\mathcal L_{\mathrm{satCon}}
=
\frac{1}{|\mathcal B_l|}
\sum_{i\in\mathcal B_l}
\left(
1-\cos(\mathbf z_{\mathrm{id}}(x_i),\mathbf z_{\mathrm{id}}(a_{\mathrm{leo}}(x_i)))
\right),
$$

但B02中$\lambda_{\mathrm{satCon}}=0$，实际贡献为0。

### 6.11总目标

B02实际优化目标写为

$$
\begin{aligned}
\mathcal L_{\mathrm{ADV3B02}}
=&
\mathcal L_{\mathrm{tx}}
+1.00\,\mathcal L_{\mathrm{dom}}
+0.35\,\mathcal L_{\mathrm{adv}}
+0.16\,\mathcal L_{\mathrm{group}}
+0.04\,\mathcal L_{\mathrm{fishr}}\\
&+0.0032\,\mathcal L_{\mathrm{proto}}
+0.032\,\mathcal L_{\mathrm{zid}}
+0.0045\,\mathcal L_{\mathrm{coretail}}
+0.0045\,\mathcal L_{\mathrm{softmix}}\\
&+0.0035\,\mathcal L_{\mathrm{epi}}
+0.16\,\mathcal L_u
+0.01\,\mathcal L_{\mathrm{ent}}
+0.68\,\mathcal L_{\mathrm{satCE}}
+0\cdot\mathcal L_{\mathrm{satCon}}.
\end{aligned}
$$

## 7.创新模块划分与消融设计

### 模块A：物理先验双表征主干

模块A包括Sinc滤波前端、time path、frequency path、PA path、domain branch、GRL和$\mathbf z_{\mathrm{id}}/\mathbf z_{\mathrm{dom}}$解耦。它回答的问题是：硬件指纹和接收链路扰动能否在表征空间分离。

|消融|改法|观察指标|
|---|---|---|
|`single_backbone`|去掉domain backbone，只保留单CVSincNet|strict UDU、receiver floor、`z_id->d`泄漏probe|
|`no_pa_path`|关闭PA path|strict UDU、sat_floor、hard TX类|
|`no_freq_path`|关闭frequency path|receiver floor、LEO压力视图鲁棒性|
|`no_grl`|`lambda_adv=0`|域泄漏和目标域old acc|
|`no_mixstyle`|关闭identity branch MixStyle|跨receiver泛化和弱receiver floor|

### 模块B：源域半监督伪标签与一致性学习

模块B只处理$\mathcal U_s$中的源域无TX标签样本，不参与Phase2 support注册。它的创新点是receiver/day-aware阈值、EMA teacher、domain gate、temporal gate和strong-view一致性同时约束伪标签。

|消融|改法|观察指标|
|---|---|---|
|`no_ssl`|`lambda_u=0`，不使用`U_s`伪标签|source val、strict UDU、prototype质量|
|`no_ema`|用student替代EMA teacher|伪标签稳定性、pseudo precision|
|`global_tau`|`tau_d`改为全局阈值|弱receiver伪标签覆盖率和错误率|
|`no_temporal_gate`|移除TemporalGate|连续窗口伪标签一致性|
|`no_strong_agreement`|移除strong-view一致性|增强扰动下伪标签污染率|

### 模块C：known-core保真与类内尾部风险抑制

模块C的目标是在Phase1源域内保护每个旧类的高置信核心，同时抑制类内尾部和类间边界样本把prototype半径拉大。它不改变类别集合，也不参与Phase2类别注册；消融时只观察known core半径、tail CVaR、receiver floor和后续qKNN支持集稳定性。

|消融|改法|观察指标|
|---|---|---|
|`no_coretail`|关闭`L_coretail`|known core半径、tail CVaR、receiver floor|
|`core80_vs_core90`|`Q_core=0.80`与`0.90`对比|core保真与tail覆盖权衡|
|`accept80_vs_accept85`|`Q_acc=0.80`与`0.85`对比|accept半径和旧类召回|
|`alpha20_vs_alpha30`|`CVaR alpha=0.20`与`0.30`对比|尾部风险抑制强度|
|`no_component_gate`|关闭component gate|局部簇半径、component数量、source overflow|

### 模块D：prototype几何、类间软混合和LEO压力视图

模块D把$\mathbf z_{\mathrm{id}}$变成可导出、可检索、可注册的Phase2资产。它包括prototype memory、$\mathbf z_{\mathrm{id}}$紧致性、类间软标签mixup、source episode和源域LEO压力视图。

|消融|改法|观察指标|
|---|---|---|
|`no_proto`|`lambda_proto=0`|Phase2 prototype导出质量和qKNN支持集检索|
|`no_zid_compact`|`lambda_zid=0`|类内角半径、min class acc|
|`no_softmix`|`lambda_softmix=0`|类间边界混淆和tail风险|
|`no_source_episode`|`lambda_epi=0`|跨源域episode外推|
|`no_sat_ce`|`lambda_satCE=0`|LEO压力视图下sat_floor|

## 8.qKNNV42：Phase2 Stage2-C轻量注册头

### 8.1任务定义

qKNNV42只处理Phase2 Stage2-C：

$$
\mathcal S_t=
\mathcal S_{\mathrm{old}}\cup\mathcal S_{\mathrm{new}},
\qquad
\mathcal Q_t=
\mathcal Q_{\mathrm{old}}\cup\mathcal Q_{\mathrm{new}}.
$$

其中

$$
\mathcal S_{\mathrm{old}}=\{(x_i,y_i):y_i\in\mathcal Y_{\mathrm{old}},x_i\in\mathcal R_t\},
\qquad
\mathcal S_{\mathrm{new}}=\{(x_i,y_i):y_i\in\mathcal Y_{\mathrm{new}},x_i\in\mathcal R_t\}.
$$

qKNNV42不更新ADV3B02参数$\theta$，只在冻结特征

$$
\mathbf z_i=
\frac{g_\theta(x_i)}{\|g_\theta(x_i)\|_2}
$$

上建立support memory。

### 8.2int8量化support memory

每个support向量量化为

$$
\mathbf q_i=
\operatorname{clip}\left(
\operatorname{round}(127\,\mathbf z_i),-127,127
\right)\in\mathbb Z^{d}_{8}.
$$

推理时近似恢复为

$$
\hat{\mathbf z}_i=
\frac{\mathbf q_i/127}{\|\mathbf q_i/127\|_2+\epsilon}.
$$

当前K5主结果中

$$
|\mathcal Y_{\mathrm{old}}|=6,\qquad
|\mathcal Y_{\mathrm{new}}|=20,\qquad
K_{\mathrm{old}}=K_{\mathrm{new}}=5,
$$

因此support code数量为

$$
N_{\mathrm{code}}=(6+20)\times5=130.
$$

### 8.3qKNNV42打分

query特征为

$$
\mathbf z_q=\frac{g_\theta(x_q)}{\|g_\theta(x_q)\|_2}.
$$

与support code的余弦相似度为

$$
s_{qi}=\mathbf z_q^\top\hat{\mathbf z}_i.
$$

对类别$c$取类内top-$m$均值：

$$
\operatorname{KNN}_c(q)
=
\frac{1}{m}
\sum_{i\in\operatorname{TopM}(\{s_{qj}:y_j=c\},m)}
s_{qi}.
$$

类别prototype为

$$
\boldsymbol\mu_c=
\frac{\sum_{i:y_i=c}\hat{\mathbf z}_i}
{\left\|\sum_{i:y_i=c}\hat{\mathbf z}_i\right\|_2+\epsilon},
\qquad
\operatorname{Proto}_c(q)=\mathbf z_q^\top\boldsymbol\mu_c.
$$

V42线路的主分数写为

$$
S_c(q)
=
(1-\lambda_p)\operatorname{KNN}_c(q)
+\lambda_p\operatorname{Proto}_c(q)
+b_{\mathrm{old}}\mathbb 1[c\in\mathcal Y_{\mathrm{old}}]
+\Delta_{\mathrm{scen}}(q,c)
+\Delta_{\mathrm{lp}}(q,c).
$$

其中当前K5 high-floor行使用

$$
\lambda_p=0.45,\qquad
b_{\mathrm{old}}=0.001,\qquad
m=1.
$$

$\Delta_{\mathrm{scen}}(q,c)$是scenario-aware residual补全项，$\Delta_{\mathrm{lp}}(q,c)$是support-query图传播项。当前行参数为

$$
\lambda_{\mathrm{lp}}=0.025,\quad
k_{\mathrm{lp}}=10,\quad
\alpha_{\mathrm{lp}}=0.76,\quad
T_{\mathrm{lp}}=0.05,\quad
R_{\mathrm{lp}}=8,
$$

以及

$$
\lambda_{\mathrm{scen}}=0.5,\qquad
\mathrm{scope}_{\mathrm{scen}}=\mathcal Y_{\mathrm{new}}.
$$

预测为

$$
\hat y_q=\arg\max_{c\in\mathcal Y_{\mathrm{old}}\cup\mathcal Y_{\mathrm{new}}}S_c(q).
$$

### 8.4当前同row结果

证据文件：

```text
E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v53_fftlogmag_20260706\local_v55_diagnostics_20260706\k5_strict_seed421070_floor_param_best_predictions_20260707.json
```

|参数|值|
|---|---|
|`transform_mode`|`diag_whiten_fisher`|
|`transform_strength`|0.1|
|`topm`|1|
|`proto_mix`|0.45|
|`old_bias`|0.001|
|`labelprop_weight`|0.025|
|`labelprop_alpha`|0.76|
|`scenario_residual_weight`|0.5|
|`stored_quantized_support_code_count`|130|
|`stored_raw_support_count`|0|
|`stored_class_prototype_count`|26|
|`support_index_sha16`|`a84b66e28e565c52`|
|`query_index_sha16`|`75c99f6361810ca9`|

结果：

|metric|value|
|---|---:|
|A old|94.52%|
|min old class acc|85.71%|
|A new|90.14%|
|min new class acc|81.43%|
|H old-new|92.28%|

### 8.5qKNNV42创新点

qKNNV42的贡献在Phase2部署方式，而不是新神经网络结构：

|创新点|具体表现|
|---|---|
|冻结表征上的注册头|不更新ADV3B02参数，只写入目标域K-shot support memory|
|int8 support code|用`q_i in Z8^d`保存support，不保存原始IQ|
|旧类/新类同一评分空间|`Y_old`和`Y_new`共享`z_id`检索空间|
|类内top-m+prototype混合|兼顾局部近邻和类中心稳定性|
|old-class anchor|`b_old`保护旧类适应，不让新类注册吞掉旧类|
|scenario residual|用目标LEO场景support结构补足同场景缺失|
|轻量图传播|`Delta_lp`只在冻结特征和support/query图上做分数平滑，不训练backbone|

相对现有RFFI，qKNNV42的差异是：普通RFFI闭集分类通常固定$\mathcal Y$并训练一个softmax分类器；qKNNV42在目标接收机域到达后，用少量support即时扩展类别集合$\mathcal Y_{\mathrm{old}}\cup\mathcal Y_{\mathrm{new}}$，并把部署状态限制为support code、prototype和少量标量。

### 8.6与机器学习方法的关系

|类似方法|相似点|qKNNV42差异|
|---|---|---|
|KNN|按embedding相似度分类|使用量化support code，并区分old/new角色|
|Nearest Class Mean|使用类中心|同时使用top-m局部近邻和prototype|
|Prototypical Networks|K-shot support形成prototype|backbone不做episodic训练，部署期只更新memory|
|Matching Networks|query-support相似度|qKNNV42不用端到端attention训练|
|iCaRL/增量prototype|新类注册和旧类保持|qKNNV42不保存原始样本，不训练分类器权重|
|量化检索|int8 embedding降低存储|用于RFFI目标接收机K-shot注册|

### 8.7qKNNV42消融

|组件|消融|观察指标|
|---|---|---|
|int8量化|float support vs int8 support|old/new均值、min class、存储码数|
|top-m|`m in {1,2,4}`|新类最低类和旧类地板|
|prototype mix|`lambda_p in {0,0.25,0.45}`|局部近邻/类中心权衡|
|old bias|`b_old in {0,0.001}`|旧类遗忘和new-over-old混淆|
|scenario residual|`lambda_scen in {0,0.5}`|LEO场景缺失下的新类地板|
|labelprop|`lambda_lp in {0,0.025}`|query-free图平滑对弱类的影响|
|support selection|不同seed/support策略|是否能把`seed=421070`式强support转成oracle-free注册机制|

## 9.K10非压缩/压缩更新

当前同一Phase2口径下，K10 40seed结果为。该表只覆盖target-old旧类目标域适应和target-new/seen-new注册识别；不含unknown互斥、unknown拒识或open-set优化。

|候选|stored codes|old mean|old p10|min old mean|min old p10|seen-new mean|seen-new p10|min new mean|min new p10|min new >=75|min new >=80|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|不压缩性能上限|260|94.39%|93.57%|85.25%|82.86%|92.22%|90.78%|78.46%|70.00%|32/40|20/40|
|V59统一hard-diverse budget8|208|94.43%|93.57%|85.46%|82.86%|91.95%|90.29%|77.57%|69.86%|29/40|16/40|
|V62旧类budget5、新类全量|230|94.58%|93.31%|85.82%|82.71%|92.17%|90.83%|78.29%|70.00%|31/40|19/40|
|V63旧类budget5+local competition|230|94.60%|93.31%|85.86%|82.71%|92.16%|90.83%|78.36%|70.00%|32/40|20/40|
|V64旧类budget5+新类budget9|210|94.60%|93.31%|85.89%|82.71%|92.13%|90.86%|78.07%|70.00%|30/40|15/40|
|V66弱类保护式新类budget9|218|94.60%|93.31%|85.89%|82.71%|92.17%|90.79%|78.14%|70.00%|32/40|22/40|
|V67弱类保护式新类budget8|206|94.59%|93.10%|85.86%|82.71%|92.10%|90.57%|78.18%|70.00%|32/40|21/40|
|V68分级预算诊断top16->9|214|94.60%|93.10%|85.89%|82.71%|92.15%|90.64%|78.18%|70.00%|32/40|21/40|
|V69轻量labelprop+new残差|218|94.68%|93.31%|86.07%|82.71%|92.44%|91.14%|78.68%|70.00%|32/40|22/40|
|V70轻量labelprop+new残差+新类budget8 protect10|210|94.67%|93.31%|86.04%|82.71%|92.45%|90.85%|78.68%|70.00%|32/40|23/40|

解释：

- K10不压缩行是性能上限，存储260个support code。
- V62把旧类support压缩到每类5码，seen-new support保留全量；旧类均值和地板更强，seen-new均值近似持平，但`min_new>=75`少1个seed。
- V63在V62上加入`local_competition_weight=0.02,k=5,scope=role`，以同样230个support code恢复`min_new>=75`到32/40，并把旧类均值提高到94.60%。它是当前K10最稳高效压缩候选，但`min_new p10`仍为70.00%，不能写成最低类彻底解决。
- V64继续压缩seen-new support到9码/类后，总码数降到210，但`min_new>=75`从32/40降到30/40，`min_new>=80`也从20/40降到15/40。该结果证明当前不能简单降低seen-new预算；更高压缩率需要弱类保护式多原型/残差码，而不是删掉seen-new K-shot覆盖。
- V65原型锚点恢复为负诊断：210个support code之外增加4160个锚点标量，`anchor_weight=0.01`也把`min_new>=75`降到29/40，不晋升。
- V66在V64基础上做support-only弱类保护：旧类5码/类，seen-new默认9码/类，按半径保护top8 seen-new类全量K10 support，并沿用V63局部竞争。它用218个support code保持`min_new>=75=32/40`，把`min_new>=80`从V63的20/40提升到22/40，是上一版K10最强压缩候选；但`min_new p10`仍为70.00%，不能声称最低类问题彻底解决。
- V67把非保护seen-new类继续压到8码/类，总support code降到206，仍保持`min_new>=75=32/40`，但`min_new>=80`为21/40，低于V66的22/40。因此V67是当前高压缩候选，V66仍是均衡最优压缩候选。
- V68新增默认关闭的分级预算诊断参数，尝试让中风险seen-new类从8码升到9码。当前最好同口径行仍只有`min_new>=80=21/40`，没有恢复V66的22/40，也不比V67更省码，因此不注册为稳定策略。
- V69在V66压缩状态不变的前提下加入轻量`labelprop_weight=0.015`和`scenario_residual_weight=0.5,scope=new`。它仍为218个support code，但把旧类均值提升到94.68%、min old均值提升到86.07%、seen-new均值提升到92.44%、seen-new p10提升到91.14%、min new均值提升到78.68%，因此取代V66成为当前K10均衡最优压缩候选。边界不变：`min_new p10=70.00%`，`min_new>=75/80`仍为32/40和22/40，最低类坍塌尚未彻底解决。
- V70把V69机制移到更高效压缩：旧类5码/类，seen-new默认8码/类，按半径保护top10 seen-new类全量K10 support。它用210个support code保持old和min old基本不变，seen-new均值为92.45%，并把`min_new>=80`提升到23/40。因此V70取代V69成为当前K10均衡高效压缩候选；但`min_new p10=70.00%`仍未改善，`1-12`和`1-1`仍是低端瓶颈。
- K5 `seed=421070`仍是单split强support证据，后续必须转成注册期可执行的support选择机制。

## 10.证据索引

|证据|路径|用途|
|---|---|---|
|项目协议|`E:\type10-7\项目.md`|CVS科学场景、数据协议、Stage2边界|
|AGENTS规则|`E:\type10-7\AGENTS.md`|Git、N607、报告和中文排版规则|
|ADV3B02主报告|`E:\type10-7\automation_reports\CV-SincNet\phase1_adv3_mechanism32_queue_20260701\report.md`|B02身份、数据、候选和prototype导出|
|ADV3B02分析|`E:\type10-7\automation_reports\CV-SincNet\phase1_adv3_mechanism32_queue_20260701\full_analysis_20260702.md`|Phase1边界和B02结论|
|ADV3B02候选表|`E:\type10-7\automation_reports\CV-SincNet\phase1_adv3_mechanism32_queue_20260701\adv3_m32_candidate_summary.csv`|B02同row指标|
|ADV3B02训练快照|`E:\type10-7\code\snapshots\phase1_adv3_mechanism32_queue_20260701\launch_phase1_adv3_mechanism32_queue_20260701.sh`|训练参数与候选variant|
|CVS模型|`E:\type10-7\code\model.py`、`E:\type10-7\code\model_dual_cvsincnet.py`|CV-SincNet、双分支解耦和层参数|
|SSDG训练|`E:\type10-7\code\SSDG\train_ssdg.py`|伪标签、loss、guard和日志字段|
|qKNNV42策略实现|`E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_support_metric_qknn_probe.py`|V42策略、top-m、prototype、labelprop、scenario residual|
|qKNNV42主报告|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\report.md`|V42矩阵和high-floor行解释|
|qKNNV42最佳JSON|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v53_fftlogmag_20260706\local_v55_diagnostics_20260706\k5_strict_seed421070_floor_param_best_predictions_20260707.json`|当前同row指标和support/query指纹|

## 11.下一步

1.按模块A-D跑ADV3B02消融，主表保留strict UDU、receiver floor、sat_floor、prototype半径和Phase2 qKNN后续指标。

2.按qKNNV42组件跑Phase2消融，主表保留`K`、stored codes、old mean、min old、seen-new mean、min new、`H_old,new`和support/query指纹；当前主线不得把unknown互斥/拒识作为优化目标。

3.把K5 strong support从`seed=421070`证据转成oracle-free support selection，例如支持集覆盖度、类内多原型和scenario coverage gate。

4.继续复核更多`R_t`目标接收机域，避免单receiver或单support split过拟合。
