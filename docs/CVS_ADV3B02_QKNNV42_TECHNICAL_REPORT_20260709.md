# CVS阶段性成果技术报告：`ADV3B02_CORE90_SOFT_E200`与`qKNNV42`

日期：2026-07-10
修订说明：本版彻底移除正文行内数学和表格内LaTeX，所有正式符号与公式均使用独立LaTeX公式块；补全ADV3B02逐层参数、张量尺寸、分支作用、收益与代价；依据2026-07-01历史运行快照重建完整损失、符号表、局部符号表、阶段权重和优化方向；从严厉审稿视角把Phase1重组为物理双表征、可信伪标签、尾部角几何和源域反事实外推四个可消融模块。`qKNNV42`仍只作为Phase2 Stage2-C轻量注册/适应头，不包含unknown拒识主线。

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

为适配当前Markdown渲染器，本文只在独立公式块中写LaTeX。正文和表格使用`x_i`、`z_id`等稳定文本别名；别名与公式块中的标准数学符号一一对应。全文禁止正文行内数学和表格内LaTeX。

地面训练阶段的源域、旧类集合与Phase2目标域、新类集合分别定义为

$$
\mathcal{R}_{s},\qquad
\mathcal{Y}_{\mathrm{old}},\qquad
\mathcal{R}_{t},\qquad
\mathcal{Y}_{\mathrm{new}}.
$$

两个阶段必须满足接收机域不重叠和身份集合不重叠：

$$
\mathcal{R}_{t}\cap\mathcal{R}_{s}=\varnothing,
\qquad
\mathcal{Y}_{\mathrm{new}}\cap\mathcal{Y}_{\mathrm{old}}=\varnothing.
$$

单个Phase1样本、TX标签和域标签为

$$
\mathbf{x}_{i}\in\mathbb{R}^{2\times L},
\qquad
y_i\in\mathcal{Y}_{\mathrm{old}},
\qquad
d_i\in\mathcal{D}_{s}.
$$

其中`x_i`是两通道IQ片段，`y_i`是TX身份，`d_i`是由receiver、day、rx_day或channel view构成的源域标签。CVS观测模型写为

$$
\mathbf{x}_{i}
=\mathcal{R}_{d_i}\!\left(
\mathcal{H}_{d_i}\circledast\mathcal{T}_{y_i}(\mathbf{s}_{i})
\right)+\mathbf{n}_{i}.
$$

`T_y`表示TX硬件非理想映射，`H_d`表示传播或星地信道作用，`R_d`表示接收机链路响应，`n_i`表示噪声。该分解解释了CVS为何同时需要身份表征和域表征：同一观测中，TX特征、传播效应和接收机响应是耦合的。

Phase1标注集和无TX标签集为

$$
\mathcal{L}_{s}
=\left\{(\mathbf{x}_{i},y_i,d_i)\right\}_{i=1}^{N_l},
\qquad
\mathcal{U}_{s}
=\left\{(\mathbf{u}_{j},d_j)\right\}_{j=1}^{N_u}.
$$

标注预算固定为

$$
\rho_{\mathrm{label}}
=\frac{|\mathcal{L}_{s}|}
{|\mathcal{L}_{s}|+|\mathcal{U}_{s}|}
=0.1.
$$

模型的主要输出定义为

$$
f_{\boldsymbol{\theta}}(\mathbf{x})
=\left(
\boldsymbol{\ell}^{\mathrm{tx}},
\boldsymbol{\ell}^{\mathrm{dom}},
\boldsymbol{\ell}^{\mathrm{adv}},
\mathbf{z}^{\mathrm{id}},
\mathbf{z}^{\mathrm{dom}}
\right).
$$

|文本别名|对象|形状或集合|用途|
|---|---|---|---|
|`x_i`|IQ输入|2xL，B02中L=256|模型输入|
|`y_i`|TX标签|`Y_old`|监督分类、原型和类条件几何|
|`d_i`|源域标签|`D_s`|域监督、GRL、跨域一致性和分组风险|
|`ell_tx`|TX logits|C维|CosFace分类|
|`ell_dom`|域 logits|D维|训练domain branch保留接收机/日期信息|
|`ell_adv`|对抗域 logits|D维|经GRL压低`z_id`中的域可预测性|
|`z_id`|身份表征|160维|TX分类、原型、角几何和Phase2 qKNN|
|`z_dom`|域表征|160维|吸收接收机、信道和噪声统计|

## 3.ADV3B02实际模型配置

`ADV3B02_CORE90_SOFT_E200`由`code/SSDG/train_ssdg.py`构建，实际使用默认结构参数。损失和调度公式以`code/snapshots/phase1_adv3_mechanism32_queue_20260701/train_ssdg.py`与同目录`losses.py`为历史运行权威；当前`code/`已经继续演化，不能反向替代B02训练语义。2026-07-01快照没有独立保存`model.py`与`model_dual_cvsincnet.py`的远端哈希；当前Git谱系显示两文件在该运行前后未变化，且checkpoint加载无missing/unexpected key。因此本章层结构属于谱系一致证据，但不声称拥有独立的远端模型文件哈希证明。

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

共享前端输入为

$$
\mathbf{X}\in\mathbb{R}^{B\times2\times256}.
$$

|层|完整参数|输入到输出|作用|
|---|---|---|---|
|`SincConv1d.forward_iq_pair`|24个可学习带通滤波器，`kernel=79`，`stride=1`，`padding=39`；I/Q共享滤波器参数|`Bx2x256 -> Bx48x256`|用受约束的带通核代替自由卷积，先按频带分解IQ，减少前端学习任意接收机纹理的自由度|
|`HighFreqEmphasis`|每个IQ通道分别施加一阶差分`[-1,1]`和二阶差分`[1,-2,1]`，`groups=2`|`Bx2x256 -> Bx4x256`|显式放大瞬态边沿、相位变化和高频残差，为time/DAC路径提供局部变化线索|

第k个Sinc带通核由可学习下截止频率和带宽决定。先定义离散时间和未归一化滤波器：

$$
 t_n=\frac{n}{f_s},
\qquad
\widetilde{w}_{k}[n]
=\frac{
\sin(2\pi f_{2,k}t_n)-\sin(2\pi f_{1,k}t_n)
}{\pi t_n}
\left(0.54-0.46\cos\frac{2\pi n}{78}\right),
\qquad n\in\{-39,\ldots,39\}.
$$

中心点使用连续极限，最终每个滤波器按最大绝对值归一化：

$$
\widetilde{w}_{k}[0]=2(f_{2,k}-f_{1,k}),
\qquad
w_k[n]=
\frac{\widetilde{w}_{k}[n]}
{\max_{n'}|\widetilde{w}_{k}[n']|+\varepsilon}.
$$

其中`f_1k`和`f_2k`由`low_hz_`与`band_hz_`学习得到，并满足上截止频率高于下截止频率。WiSig上下文未显式提供采样率时，当前实现回退到25 MHz。

作用与收益：该前端让identity和domain两条backbone在相同底层频带坐标系中比较特征，避免两条分支从第一层就产生不可对齐的尺度。它仍保留24个带通滤波器的自适应能力，因此不是固定滤波器组。代价是长度79的核计算量高于小卷积核，并且若采样率或频谱占用显著改变，需要重新校准滤波器频率范围。

### 4.2identity branch：time path

identity branch的time path把Sinc IQ、三阶非线性基和高频差分拼接：

$$
\mathbf{t}_0=
\operatorname{concat}\left[
\mathbf{s},\;
\mathbf{s}|\mathbf{s}|^2,\;
\mathbf{h}
\right]\in\mathbb{R}^{B\times100\times256}.
$$

|层|完整参数|输入到输出|作用|
|---|---|---|---|
|`time_fuse`|`Conv1d 100->48`，`kernel=1`，`stride=1`，`padding=0`，`bias=False`；`GroupNorm groups=16`；ReLU|`Bx100x256 -> Bx48x256`|压缩Sinc、三阶非线性基和差分特征，学习三类低层证据的逐时刻组合|
|`time_down`|`AvgPool1d kernel=2,stride=2`|`Bx48x256 -> Bx48x128`|抑制高频采样噪声并降低后续卷积计算量|
|`MixStyle@time_down`|仅identity branch；同TX跨域配对；基础`p=0.18,alpha=0.10,strength=0.70`，失败时跳过|`Bx48x128 -> Bx48x128`|交换同一TX在不同receiver/day中的通道统计，削弱身份分支对接收机风格的依赖|
|`t1`|depthwise `Conv1d 48->48,k=5,groups=48`；pointwise `Conv1d 48->72,k=1`；`GroupNorm groups=8`；ReLU；`MaxPool1d 2`；`Dropout 0.10`|`Bx48x128 -> Bx72x64`|提取短时瞬态和局部包络变化；depthwise-separable结构控制参数量|
|`MixStyle@t1`|参数同上|`Bx72x64 -> Bx72x64`|在更高层局部模式上继续做同身份跨域风格扰动|
|`t2`|depthwise `72->72,k=5,groups=72`；pointwise `72->96,k=1`；`GroupNorm groups=16`；ReLU；`MaxPool1d 2`；`Dropout 0.10`|`Bx72x64 -> Bx96x32`|扩大有效感受野，组合较长时间尺度的调制与瞬态模式|
|`t3`|depthwise `96->96,k=3,groups=96`；pointwise `96->96,k=1`；`GroupNorm groups=16`；ReLU；无池化；`Dropout 0.10`|`Bx96x32 -> Bx96x32`|在不继续丢失时间分辨率的条件下细化高层时域模式|
|`t_pool`|`AdaptiveAvgPool1d 1`|`Bx96x32 -> Bx96`|把变长局部响应聚合为固定长度向量|
|`t_proj`|`Linear 96->160`|`Bx96 -> Bx160`|投影到与frequency/PA路径一致的融合维度|

time path输出定义为

$$
\mathbf{e}^{\mathrm{time}}\in\mathbb{R}^{B\times160}.
$$

作用与收益：time path主要保留Sinc子带内的瞬态、局部相位/幅度变化和三阶非线性响应。这些线索比单纯功率谱更能区分硬件链路中的短时差异。两处同TX跨域MixStyle只作用于identity branch，使模型看到“身份不变、接收机风格变化”的反事实样本。其配对要求同时存在TX标签和域标签；无标签前向没有TX标签，因此MixStyle在无标签批次跳过。代价是过强MixStyle可能抹除与TX相关的幅度统计，因此第110轮后将概率和强度退火，给分类边界留出收敛阶段。

### 4.3identity branch：frequency path

frequency path把原始IQ压缩为32个频带，并构造正负频率镜像统计。正频率功率、负频率功率及两个派生比值定义为

$$
\mathbf{F}_{0}=
\left[
\log(1+\mathbf{P}^{+}),\;
\log(1+\mathbf{P}^{-}),\;
\log\frac{\mathbf{P}^{+}+\varepsilon}{\mathbf{P}^{-}+\varepsilon},\;
\frac{|\mathbf{P}^{+}-\mathbf{P}^{-}|}{\mathbf{P}^{+}+\mathbf{P}^{-}+\varepsilon}
\right]\in\mathbb{R}^{B\times4\times32}.
$$

|层|完整参数|输入到输出|作用|
|---|---|---|---|
|`freq_gate`|`Conv1d 4->1,k=5,padding=2`；Sigmoid；缩放幅度0.6|`Bx4x32 -> Bx4x32`|按频带自适应增强或抑制镜像统计，避免所有频带等权|
|`f1`|depthwise `4->4,k=5,groups=4`；pointwise `4->16,k=1`；`GroupNorm groups=16`；ReLU；`MaxPool1d 2`；`Dropout 0.05`|`Bx4x32 -> Bx16x16`|学习局部频带组合和镜像不对称模式|
|`f2`|depthwise `16->16,k=5,groups=16`；pointwise `16->32,k=1`；`GroupNorm groups=16`；ReLU；`MaxPool1d 2`；`Dropout 0.05`|`Bx16x16 -> Bx32x8`|扩大频带感受野并提高通道容量|
|`f3`|depthwise `32->32,k=3,groups=32`；pointwise `32->32,k=1`；`GroupNorm groups=16`；ReLU；无池化；`Dropout 0.05`|`Bx32x8 -> Bx32x8`|细化高层频谱模式，不再压缩频带轴|
|`f_pool`|`AdaptiveAvgPool1d 1`|`Bx32x8 -> Bx32`|聚合频带响应|
|`f_proj`|`Linear 32->160`|`Bx32 -> Bx160`|投影为频域嵌入|
|`freq_stats_proj`|`Linear 3->160`；ReLU；`Dropout 0.1125`|`Bx3 -> Bx160`|注入高频能量比例、镜像不对称均值和谱平坦度|

frequency gate和频域嵌入为

$$
\mathbf{G}^{\mathrm{freq}}
=1+0.6\left[
2\sigma\!\left(\operatorname{Conv1D}_{4\to1}(\mathbf{F}_{0})\right)-1
\right].
$$

$$
\mathbf{e}^{\mathrm{freq}}
=\operatorname{Linear}_{32\to160}\!\left(\operatorname{Pool}(\mathbf{F}_{3})\right)
+\operatorname{MLP}_{3\to160}(\mathbf{r}^{\mathrm{freq}}).
$$

`r_freq`包含高频能量比例、镜像不对称均值和谱平坦度。

作用与收益：频域路径补充time path难以稳定表达的长期频谱形状、镜像不对称和带外能量分布。32频带压缩使其计算量远小于直接处理256点频谱。其风险是频谱线索也会携带接收机滤波响应，因此该路径不能单独承担身份判别，必须与域解耦损失和time/PA证据联合使用。

### 4.4identity branch：PA path

PA path使用memory polynomial lift。记延迟索引为r、非线性阶数为o：

$$
\phi_{r,o}(\mathbf{x})[n]
=\mathbf{x}[n-r]\left|\mathbf{x}[n-r]\right|^{o-1},
\qquad
r\in\{0,1,2,3\},
\quad
o\in\{1,3,5\}.
$$

I/Q两通道、4个记忆延迟和3个奇数阶共同产生24个通道：

$$
C_{\mathrm{PA}}=2\times4\times3=24.
$$

|层|完整参数|输入到输出|作用|
|---|---|---|---|
|`pa_lift`|`memory_depth=4`，`orders=(1,3,5)`，幅度截断2.0|`Bx2x256 -> Bx24x256`|显式展开PA奇数阶非线性和短记忆效应|
|`pa_gate`|包络经`Conv1d 1->24,k=5,padding=2`；Sigmoid；`alpha=0.5`|`Bx24x256 -> Bx24x256`|让高包络、强非线性区间获得更高权重|
|`pa_b1`|`Conv1d 24->48,k=7,dilation=1,padding=3`；`GroupNorm groups=16`；SiLU；`AvgPool1d 2`；`Dropout 0.08`|`Bx24x256 -> Bx48x128`|提取局部PA响应|
|`pa_b2`|`Conv1d 48->64,k=7,dilation=2,padding=6`；`GroupNorm groups=16`；SiLU；`AvgPool1d 2`；`Dropout 0.08`|`Bx48x128 -> Bx64x64`|用膨胀卷积覆盖更长记忆范围|
|`pa_b3`|`Conv1d 64->64,k=5,dilation=4,padding=8`；`GroupNorm groups=16`；SiLU；无池化；`Dropout 0.08`|`Bx64x64 -> Bx64x64`|进一步扩大有效感受野并保持64步分辨率|
|`pa_pool`|`AdaptiveAvgPool1d 1`|`Bx64x64 -> Bx64`|聚合非线性响应|
|`pa_proj`|`Linear 64->160`；ReLU；`Dropout 0.1125`|`Bx64 -> Bx160`|形成PA局部嵌入|
|`pa_stats_proj`|`Linear 3->160`；ReLU；`Dropout 0.1125`|`Bx3 -> Bx160`|注入edge ratio、regrowth ratio和谱峰度|

输出为

$$
\mathbf{e}^{\mathrm{pa}}
=\operatorname{MLP}_{64\to160}\!\left(\operatorname{Pool}(\mathbf{P}_{3})\right)
+0.25\,\operatorname{MLP}_{3\to160}(\mathbf{r}^{\mathrm{pa}}).
$$

`r_pa`包含edge ratio、regrowth ratio和谱峰度。

作用与收益：PA path直接编码AM/AM、AM/PM相关非线性及短记忆效应，目标是提取更接近发射机硬件来源的证据。相比让普通卷积自己发现高阶乘积，memory polynomial提供了明确归纳偏置。代价是它仍可能受AGC、接收机非线性和信道幅度变化影响，因此需要包络门控、域分支和角几何共同约束，不能把PA path单独解释为纯TX特征。

### 4.5identity branch：融合和分类头

由于identity branch关闭DAC路径，base输入为

$$
\mathbf{b}^{\mathrm{in}}
=\operatorname{Concat}
\left(\mathbf{e}^{\mathrm{time}},\mathbf{e}^{\mathrm{freq}},\chi\right)
\in\mathbb{R}^{B\times321}.
$$

复IQ信号及circularity统计定义为

$$
z[n]=I[n]+\mathrm{j}Q[n],
\qquad
\chi=
\frac{|\mathbb{E}\{z^2\}|}
{\mathbb{E}\{|z|^2\}+\varepsilon}.
$$

|层|完整参数|输入到输出|作用|
|---|---|---|---|
|`fuse`|`Linear 321->160`；ReLU；`Dropout 0.45`|`Bx321 -> Bx160`|融合时域、频域和circularity证据，形成共享base|
|`con_proj`|`Linear 160->160`；ReLU；`Dropout 0.1125`|`Bx160 -> Bx160`|产生`feat_con`；不是最终分类表征|
|`id_proj`|`Linear 160->160`；ReLU；`Dropout 0.225`|`Bx160 -> Bx160`|形成基础身份向量`feat_cls`|
|classifier内`pa_proj`|`Linear 320->160`；ReLU；`Dropout 0.225`|`Concat(base,pa_local): Bx320 -> Bx160`|把PA局部嵌入与base联合校准为`feat_pa`|
|`id_gate`|`Linear 160->160`；Sigmoid；`gate_alpha=0.35`|`feat_pa: Bx160 -> Bx160`|由PA证据逐维调制基础身份向量，调制范围受0.35限制|
|`joint_proj`|`Linear 320->160`；ReLU；`Dropout 0.225`|`Concat(feat_cls,feat_pa): Bx320 -> Bx160`|生成最终身份表征`z_id`|
|`imp_merge`|`Linear 160->160`；ReLU；`Dropout 0.1125`|`feat_pa: Bx160 -> Bx160`|生成identity backbone的`feat_imp`；Phase1主身份输出不选该键|
|`CosFaceHead`|权重矩阵`6x160`；`scale=30`；`margin=0.35`|`Bx160 -> Bx6`|在单位超球面上扩大真实类角间隔|
|`pa_head`|`Linear 160->80`；ReLU；`Linear 80->1`；Sigmoid|`Bx160 -> Bx1`|输出PA强度辅助量；B02没有给该辅助头单独的非零监督权重|

CosFace logits为

$$
\ell^{\mathrm{tx}}_{i,c}
=s_{\mathrm{cf}}
\left[
\frac{(\mathbf{z}^{\mathrm{id}}_i)^{\top}\mathbf{w}_c}
{\|\mathbf{z}^{\mathrm{id}}_i\|_2\|\mathbf{w}_c\|_2}
-m_{\mathrm{cf}}\mathbf{1}[c=y_i]
\right],
\qquad
s_{\mathrm{cf}}=30,
\quad
m_{\mathrm{cf}}=0.35.
$$

作用与收益：融合头不是简单拼接后分类。它先以time/frequency/circularity构造基础身份，再由PA证据做受限门控，最后投影到160维CosFace空间。该结构允许PA线索提高身份判别，但限制PA分支直接主导分类。代价是多级投影增加参数和过拟合风险，因此配合较高Dropout、角度margin和后述跨域几何损失。

### 4.6domain branch：域表征路径

domain branch使用同样的`lite_d`宽度，但配置为`domain_branch_ablation=no_stats`和`mixstyle_on=False`。它保留time、DAC、frequency、PA路径，禁用`freq_stats_proj`、`pa_stats_proj`和`dac_subband_agg`。底层Sinc与HF对象同identity branch共享，后续卷积和投影参数独立。time、frequency和PA卷积层的通道宽度与identity branch相同；关键差异如下：

|组件|完整参数|输入到输出|作用|
|---|---|---|---|
|MixStyle|关闭|形状不变|domain branch需要保留真实域统计，不能主动混洗接收机风格|
|stats path|关闭`freq_stats_proj`、`pa_stats_proj`、`dac_subband_agg`|对应增量为0|避免手工谱统计重复注入；原始RCN统计由后置enhancer统一处理|
|domain time path|`time_fuse,time_down,t1,t2,t3,t_pool,t_proj`参数与4.2完全相同，但参数独立且无MixStyle|`Bx100x256 -> Bx160`|保留与域相关的瞬态和局部变化|
|domain frequency path|`freq_gate,f1,f2,f3,f_pool,f_proj`参数与4.3相同；`freq_stats_proj`关闭；circularity在`no_stats`下置0|`Bx4x32 -> Bx160`|保留学习得到的谱形，避免手工频谱统计重复注入|
|domain PA path|`pa_lift,pa_gate,pa_b1,pa_b2,pa_b3,pa_pool,pa_proj`参数与4.4相同；`pa_stats_proj`增量为0|`Bx2x256 -> Bx160`|允许domain branch吸收接收链路相关的非线性变化|
|domain `fuse`|`Concat(time160,freq160,circularity_zero1)`；`Linear 321->160`；ReLU；`Dropout 0.45`|`Bx321 -> Bx160`|形成同时供DAC和PA投影使用的domain base|
|`dac_hf_proj`|`Conv1d 4->48,k=1,bias=False`；`GroupNorm groups=16`；SiLU；输出乘0.5后与Sinc IQ相加|`Bx4x256 -> Bx48x256`|把高频差分注入复滤波器组，突出接收机/DAC相关局部失真|
|`dac_b1`|widely-linear complex `24->24,k=5,dilation=1`；无池化；`Dropout 0.05`；恒等残差|`Bx48x256 -> Bx48x256`|建模I/Q共轭耦合和一阶复失真|
|`dac_b2`|widely-linear complex `24->32,k=3,dilation=1`；无池化；`Dropout 0.05`；残差`Conv1d 48->64`|`Bx48x256 -> Bx64x256`|提升复通道容量并保留残差信息|
|`dac_b3`|widely-linear complex `32->32,k=3,dilation=2`；`AvgPool1d 2`；`Dropout 0.05`；残差`Conv1d 64->64 + AvgPool1d 2`|`Bx64x256 -> Bx64x128`|扩大复域感受野并降采样|
|`dac_pool`|`AdaptiveAvgPool1d 1`|`Bx64x128 -> Bx64`|聚合DAC/IQ失衡证据|
|`dac_proj`|`Linear 64->160`；ReLU；`Dropout 0.1125`|`Bx64 -> Bx160`|形成DAC局部域嵌入|
|domain classifier内`dac_proj`|`Linear 320->160`；ReLU；`Dropout 0.225`|`Concat(base,dac_local): Bx320 -> Bx160`|生成`feat_dac`|
|domain classifier内`pa_proj`|`Linear 320->160`；ReLU；`Dropout 0.225`|`Concat(base,pa_local): Bx320 -> Bx160`|生成`feat_pa`|
|`imp_merge`|`Linear 320->160`；ReLU；`Dropout 0.1125`|`Concat(feat_dac,feat_pa): Bx320 -> Bx160`|形成`z_dom_raw`，同时吸收base、DAC和PA中的域相关成分|

WL complex convolution使用四个实卷积：

$$
\begin{aligned}
\mathbf{y}_{r}
&=W_r\mathbf{x}_{r}-W_i\mathbf{x}_{i}
+V_r\mathbf{x}_{r}+V_i\mathbf{x}_{i},\\
\mathbf{y}_{i}
&=W_r\mathbf{x}_{i}+W_i\mathbf{x}_{r}
-V_r\mathbf{x}_{i}+V_i\mathbf{x}_{r}.
\end{aligned}
$$

domain backbone输出`feat_imp`作为`z_dom_raw`。随后`DomainFeatureEnhancer`从原始IQ计算18维receiver/channel/noise统计：I/Q/幅度的一阶与二阶统计、对数功率统计、I/Q相关与不平衡、差分幅度和相位增量摘要。

|层|完整参数|输入到输出|作用|
|---|---|---|---|
|`RCNStatEncoder`|18维统计；`Linear 18->80`；`LayerNorm 80`；SiLU；`Dropout 0.05`；`Linear 80->160`|`Bx18 -> Bx160`|把显式接收机/信道/噪声统计映射为`z_rcn`|
|RCN gate|`Linear 320->160`；Sigmoid|`Concat(z_dom_raw,z_rcn): Bx320 -> Bx160`|逐维决定需要注入多少显式域统计|
|enhance|残差系数0.35；`LayerNorm 160`|`Bx160 -> Bx160`|以受限幅度增强域表征，防止手工统计覆盖学习特征|

域增强的显式形式为

$$
\mathbf{z}^{\mathrm{rcn}}
=\operatorname{RCNEncoder}(\mathbf{X}),
\qquad
\mathbf{g}^{\mathrm{rcn}}
=\sigma\!\left(
W_g[\mathbf{z}^{\mathrm{dom,raw}};\mathbf{z}^{\mathrm{rcn}}]+\mathbf{b}_g
\right).
$$

$$
\mathbf{z}^{\mathrm{dom}}
=\operatorname{LayerNorm}
\left(
\mathbf{z}^{\mathrm{dom,raw}}
+0.35\,\mathbf{g}^{\mathrm{rcn}}\odot\mathbf{z}^{\mathrm{rcn}}
\right).
$$

|层|完整参数|输入到输出|作用|
|---|---|---|---|
|`dom_head`|`Linear 160->80`；ReLU；`Dropout 0.10`；`Linear 80->D`|`z_dom: Bx160 -> BxD`|要求domain branch保留可解释的域信息|
|`adv_head`|GRL；`Linear 160->80`；ReLU；`Dropout 0.10`；`Linear 80->D`|`z_id: Bx160 -> BxD`|分类头学习域标签，GRL迫使identity backbone消除可预测域信息|
|`tx_adv_head`|默认关闭|无|B02不对`z_dom`做TX对抗，不据此声称域表征完全无身份信息|

GRL定义为

$$
\operatorname{GRL}_{\lambda}(\mathbf{z})=\mathbf{z},
\qquad
\frac{\partial\operatorname{GRL}_{\lambda}}
{\partial\mathbf{z}}
=-\lambda I.
$$

作用与收益：双分支采用非对称设计。identity branch关闭DAC并混洗同TX跨域风格，目标是保留TX稳定证据；domain branch保留DAC、关闭MixStyle并注入RCN统计，目标是主动吸收接收机和信道扰动。域监督、GRL、协方差正交和同TX跨域一致性共同训练这两个输出。该设计比单纯DANN更可解释，因为域信息不是只被擦除，而是由第二条分支显式承接。代价是双backbone增加训练计算量，且共享Sinc/HF意味着两种目标仍在底层耦合；因此报告只能声称“降低域泄漏”，不能声称严格统计独立。

训练时`return_aux=True`，两条backbone都运行；快速推理时`return_aux=False`，模型只执行identity backbone，domain branch被绕过。domain backbone内部的`con_proj`、`id_proj`、`id_gate`、`joint_proj`、内部CosFace、`dac_head`和`pa_head`虽然会在训练辅助前向中执行，但当前损失不读取这些输出，因此没有有效梯度；真正参与`z_dom`优化的是base、`dac_proj`、`pa_proj`、`imp_merge`、RCN enhancer和域分类头。这是当前实现的计算冗余，不应包装成创新点。

### 4.7分支协同总览

|分支或机制|主要捕获对象|直接收益|主要风险|由什么机制约束|
|---|---|---|---|---|
|共享Sinc/HF|频带结构、瞬态差分|统一底层坐标并降低前端自由度|采样率变化敏感|可学习截止频率、后续多路径融合|
|identity time|瞬态、局部相位/幅度、三阶非线性|补充短时硬件线索|接收机风格捷径|同TX跨域MixStyle、GRL、跨域一致性|
|identity frequency|镜像不对称、频谱形状、带外能量|提供长期谱域证据|接收机滤波器泄漏|双分支解耦、GroupCE/FishR|
|identity PA|PA奇数阶和记忆效应|增强TX硬件归因|AGC/接收机非线性混入|受限门控、角几何和域约束|
|domain DAC/RCN|I/Q耦合、接收机、噪声和信道统计|为nuisance提供显式承接空间|可能残留TX信息|仅作域建模；不声称完全身份无关|
|CosFace与角几何|单位超球面上的类内/类间结构|为跨域分类和Phase2检索提供统一距离|尾类可能仍扩张|prototype、CVaR、source episode|

## 5.训练参数

|类别|参数|值|
|---|---|---|
|数据|`wisig_pkl`|`Dataset_WigSig/ManySig.pkl`|
|数据|`split_mode`|`tx_rx_day_1_7_2`|
|数据|`labeled_ratio,unlabeled_ratio,source_val_ratio`|`0.10,0.70,0.20`|
|数据|实际L/U/V样本数|8400/58800/16800|
|数据|`wisig_out_len`|256|
|数据|TX类别/源域数|C=6，D=14；D由7个源receiver与2个源day的rx_day组合得到|
|批处理|`batch_size,eval_batch_size`|128,256|
|加载|`num_workers,prefetch_factor`|4,2|
|优化器|AdamW|`learning_rate=2e-4`，`weight_decay=1e-4`|
|AdamW内部参数|`betas,eps,amsgrad,maximize`|`(0.9,0.999),1e-8,false,false`|
|数值|AMP|开启|
|数值|学习率调度/梯度裁剪|均未启用|
|模型|可训练参数量|1,049,665|
|训练轮数|`epochs`|200|
|阶段|`label_epochs,pseudo_epochs`|130,70|
|阶段调度|`stage1_epochs,stage2_epochs,stage3_ramp_epochs`|16,68,17|
|label smoothing|`epsilon_ls`|0.01|
|伪标签|`tau_min,tau_max,pseudo_quantile`|`0.92,0.97,0.86`|
|伪标签|`pseudo_threshold_mode`|`rx_day_quantile`|
|伪标签|EMA/门控|teacher=true，decay=0.999；domain/temporal/strong agreement均开启|
|MixStyle|基础配置|`p=0.18,alpha=0.10,strength=0.70,same_tx_crossdomain`|
|MixStyle|后期退火|第110轮起40轮退火至`p=0.05,strength=0.32`|
|prototype|`lambda_proto`|0.0032|
|prototype|`proto_domain_align_weight,proto_margin,proto_push_weight`|`0.10,0.15,0.10`|
|prototype|memory momentum/min count|0.95/2|
|角几何|历史CLI名`lambda_open_world_feat`|0.0024；start=12，warmup=25|
|角几何|radius/inter/sample/tail/vacuum|12 deg/55 deg/5 deg/0.14/0.40|
|身份几何|`lambda_zid_compact`|0.032|
|身份几何|start/warmup|8/25|
|身份几何|跨域SupCon/radius/CVaR权重|`0.30,0.35,0.35`|
|身份几何|radius,CVaR alpha|`40 deg,0.95`|
|源域边界|历史CLI名`lambda_proxy_unknown`|0.0045|
|源域边界|start/warmup/virtual mode/count|45/25/hard/48|
|源域边界|core/accept/tail/overflow quantile|`0.90,0.85,0.92,0.97`|
|源域边界|vacuum/vaccept/core/component/tail/source权重|`0.55,1.00,0.45,0.65,0.20,0.20`|
|源域边界|CVaR alpha|0.30|
|类间软混合|历史CLI名`lambda_soft_unknown_mixup`|0.0045|
|类间软混合|start/warmup/count/order/alpha|`25,25,24,3,0.5`|
|类间软混合|CE/energy/vacuum权重|`0.60,1.0,0.35`|
|source episode|`lambda_source_episode`|0.0035|
|source episode|start/warmup/min domains/radius cap/mix weight|`20,25,2,33 deg,0.75`|
|源域LEO压力|`sat_train_scenarios`|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|源域LEO压力|start/`lambda_sat_cls,lambda_sat_cons`|`80/0.68/0`|
|域解耦|`lambda_domain,lambda_adv,lambda_orth,lambda_cons`|`1,0.35,0.05,0.08`；还要乘训练stage scale|
|Group/FishR|`lambda_group_ce,group_top_frac,lambda_fishr`|`0.16,0.35,0.04`|
|Group/FishR|`group_ce_min_domains,fishr_min_domains`|4,4|
|无标签|`lambda_u,lambda_ent`|`0.16,0.01`|
|checkpoint|`best_metric`|`joint_safe`|

ADV3B02的实际优化不是“全部损失从第1轮以固定权重同时开启”。训练有两个数据阶段和三个权重阶段：

|epoch范围|数据阶段|权重阶段|主要变化|
|---:|---|---|---|
|1-16|label|S1 core|域对抗乘0.70，正交乘0.50，跨域一致性为0，GroupCE乘0.50|
|17-68|label|S2 stabilize|域对抗、跨域一致性和GroupCE按幂次曲线逐步升高|
|69-130|label|S3 refine|域对抗和GroupCE达到全权重；一致性从0.85倍升到1倍|
|131-200|pseudo|S3 refine + SSL|在标注损失之外加入通过门控的源域无标签伪标签CE和熵最小化|

配置存在但对B02总梯度贡献为0的项必须与有效损失分开：

|零权重项|B02值|结论|
|---|---:|---|
|`lambda_sat_cons`|0|没有LEO clean/satellite一致性优化|
|`lambda_tx_proto,lambda_rx_proto,lambda_mask_aux`|0|仅做Phase1分布审计，不参与训练|
|`lambda_tx_supcon_masked,lambda_rx_supcon_masked,lambda_txrx_rect`|0|不进入总目标|
|`lambda_teacher_clean_kl,lambda_teacher_sat_kl,lambda_teacher_zid_mse`|0|EMA用于伪标签，不是蒸馏teacher|
|`lambda_u_domain,lambda_u_adv,lambda_u_sat_cons`|0|无标签样本不训练域头、对抗头或LEO一致性|
|`lambda_u_direct_metric_accept,lambda_u_quarantine_accept`|0|无标签几何接收项不参与B02|
|PA/DAC辅助分类与回归权重|0|PA辅助输出存在，但没有单独监督项|

## 6.训练损失函数

### 6.1总符号表

表格中的符号采用稳定文本别名，标准数学形式全部放在独立公式块中。

$$
\begin{gathered}
\mathcal{B}_{l}=\{(\mathbf{x}_i,y_i,d_i)\}_{i=1}^{N_l},
\qquad
\mathcal{B}_{u}=\{(\mathbf{u}_j,d_j)\}_{j=1}^{N_u},\\
C=6,
\qquad
D=14,
\qquad
\overline{\mathbf{z}}=\frac{\mathbf{z}}{\|\mathbf{z}\|_2+\varepsilon},\\
\vartheta(\mathbf{a},\mathbf{b})
=\arccos\!\left(
\overline{\mathbf{a}}^{\top}\overline{\mathbf{b}}
\right),
\qquad
[v]_{+}=\max(0,v),\\
\operatorname{sp}(v)=\log(1+\exp v),
\qquad
Q_q(\mathcal{A})=\mathcal{A}\text{的}q\text{分位数}.
\end{gathered}
$$

|文本别名|标准对象|形状或范围|统一含义|
|---|---|---|---|
|`B_l`,`B_u`|标注批、无标签批|集合|当前mini-batch中的源域样本|
|`N_l`,`N_u`|批内样本数|正整数|对应批大小|
|`C`,`D`|TX类数、域数|6、14|分类输出维数|
|`x_i`,`u_j`|标注/无标签IQ|2x256|模型输入|
|`y_i`,`d_i`|TX/域标签|离散整数|类条件与域条件监督|
|`z_i`,`z_id_i`|身份表征|160维|未归一化身份向量；两种写法同义|
|`z_dom_i`|域表征|160维|接收机/日期/信道表征|
|`bar_z`|L2归一化向量|单位球面|所有余弦和角距离的输入|
|`vartheta(a,b)`|角距离|0到pi|归一化向量夹角|
|`mu_c`|TX类原型|160维|类c的单位中心|
|`mu_cd`|类域原型|160维|TX类c在域d中的单位中心|
|`ell_tx`,`ell_dom`,`ell_adv`|三类logits|C维或D维|TX、域和对抗域未归一化分数|
|`p_ic`,`q_jc`|预测概率|0到1|student或EMA teacher的softmax输出|
|`e_c`|one-hot向量|C维|类别c的单位标签向量|
|`Q_q`|分位数算子|q在0到1之间|构造域阈值、核心半径和尾部集合|
|`TopK_k`|最大k项均值|k为正整数|强调最难样本或最危险类别|
|`TopFrac_alpha`|最大alpha比例均值|alpha在0到1之间|实现经验CVaR尾部均值|
|`sg`|stop-gradient|算子|前向保留数值，反向阻断梯度|
|`sp`|softplus|非负平滑函数|替代硬hinge并保留梯度|
|`1[condition]`|指示函数|0或1|控制样本选择或阶段启用|
|`e`|训练epoch|1到200|不得与one-hot别名混用|
|`r(e;s,w)`|线性warm-up门|0到1|从start轮起用w轮升到1|
|`w_k(e)`|stage权重|非负实数|基础lambda与S1/S2/S3倍率的乘积|

### 6.2阶段权重与完整总目标

线性warm-up门定义为

$$
r(e;s,w)=
\begin{cases}
0, & e<s,\\
\min\!\left(1,\dfrac{e-s+1}{w}\right), & e\ge s.
\end{cases}
$$

S2与S3的幂次进度为

$$
t_2(e)=\left(\frac{e-16}{52}\right)^{1.75},
\qquad
t_3(e)=\min\!\left[
1,\left(\frac{e-68}{17}\right)^{1.75}
\right].
$$

有效域相关权重为

$$
\begin{array}{c|ccc}
&e\le16&17\le e\le68&e\ge69\\
\hline
w_{\mathrm{dom}}(e)&1&1&1\\
w_{\mathrm{adv}}(e)&0.245&0.35(0.70+0.30t_2)&0.35\\
w_{\mathrm{orth}}(e)&0.025&0.05&0.05\\
w_{\mathrm{cons}}(e)&0&0.08(0.20+0.55t_2)&0.08(0.85+0.15t_3)\\
w_{\mathrm{group}}(e)&0.08&0.16(0.70+0.30t_2)&0.16
\end{array}
$$

ADV3B02历史运行的完整目标为

$$
\begin{aligned}
\mathcal{L}_{\mathrm{ADV3B02}}(e)
={}&\mathcal{L}_{\mathrm{tx}}
+w_{\mathrm{dom}}(e)\mathcal{L}_{\mathrm{dom}}
+w_{\mathrm{adv}}(e)\mathcal{L}_{\mathrm{adv}}
+w_{\mathrm{orth}}(e)\mathcal{L}_{\mathrm{orth}}
+w_{\mathrm{cons}}(e)\mathcal{L}_{\mathrm{cons}}\\
&+w_{\mathrm{group}}(e)\mathcal{L}_{\mathrm{group}}
+0.04\mathcal{L}_{\mathrm{fishr}}
+0.0032\mathcal{L}_{\mathrm{proto}}\\
&+0.0024r(e;12,25)\mathcal{L}_{\mathrm{geo}}
+0.032r(e;8,25)\mathcal{L}_{\mathrm{zid}}\\
&+0.0045r(e;45,25)\mathcal{L}_{\mathrm{coretail}}
+0.0045r(e;25,25)\mathcal{L}_{\mathrm{softmix}}\\
&+0.0035r(e;20,25)\mathcal{L}_{\mathrm{epi}}
+0.68\mathbf{1}[e\ge80]\mathcal{L}_{\mathrm{satCE}}\\
&+\mathbf{1}[e\ge131]
\left(0.16\mathcal{L}_{u}+0.01\mathcal{L}_{\mathrm{ent}}\right).
\end{aligned}
$$

这里把历史代码名`open_world_feat`记为`L_geo`，把历史代码名`proxy_unknown`记为`L_coretail`。两项在本报告中只解释为Phase1源域闭集角几何和尾部风险正则，不构成unknown拒识、FAR或open-set成功证据。

|符号别名|意义|怎么优化|
|---|---|---|
|`r(e;s,w)`|延迟启用并线性升权|避免训练早期几何项压倒基础分类|
|`w_adv,w_orth,w_cons,w_group`|S1/S2/S3有效权重|先学习可分类表征，再逐步加强域解耦与困难域约束|
|`1[e>=131]`|SSL阶段门|前130轮只用标注源域学习teacher可依赖的基础表征|

### 6.3监督TX分类

带label smoothing的目标分布为

$$
\widetilde{y}_{i,c}=
(1-\varepsilon_{\mathrm{ls}})\mathbf{1}[c=y_i]
+\frac{\varepsilon_{\mathrm{ls}}}{C}.
$$

监督TX分类损失为

$$
\mathcal{L}_{\mathrm{tx}}
=-\frac{1}{|\mathcal{B}_l|}
\sum_{i\in\mathcal{B}_l}
\sum_{c=1}^{C}
\widetilde{y}_{i,c}
\log p_{\boldsymbol{\theta}}(c|\mathbf{x}_i),
\qquad
\varepsilon_{\mathrm{ls}}=0.01.
$$

|符号别名|意义|
|---|---|
|`y_tilde_ic`|label smoothing后的类c目标概率|
|`epsilon_ls`|0.01的平滑系数|
|`p_theta(c given x_i)`|CosFace logits经softmax后的TX概率|
|`C`|6个旧TX类别|

优化方式：最小化该项提高真实TX的CosFace概率，同时用1%的均匀质量限制过度自信。梯度更新identity backbone、CosFace权重以及共享Sinc/HF；domain branch不直接接收该项梯度。

### 6.4域监督、域对抗、正交与类条件一致性

域监督损失为

$$
\mathcal{L}_{\mathrm{dom}}
=-\frac{1}{|\mathcal{B}_l|}
\sum_{i\in\mathcal{B}_l}
\log p_{\boldsymbol{\phi}}
\left(d_i|\mathbf{z}^{\mathrm{dom}}_i\right).
$$

域对抗损失为

$$
\mathcal{L}_{\mathrm{adv}}
=-\frac{1}{|\mathcal{B}_l|}
\sum_{i\in\mathcal{B}_l}
\log p_{\boldsymbol{\psi}}
\left[
d_i|\operatorname{GRL}_{1}
\left(\mathbf{z}^{\mathrm{id}}_i\right)
\right].
$$

对中心化后的批特征矩阵，协方差正交损失为

$$
\mathcal{L}_{\mathrm{orth}}
=\frac{1}{D_z^2}
\left\|
\frac{
(Z_{\mathrm{id}}-\overline{Z}_{\mathrm{id}})^{\top}
(Z_{\mathrm{dom}}-\overline{Z}_{\mathrm{dom}})
}{|\mathcal{B}_l|-1}
\right\|_F^2,
\qquad
D_z=160.
$$

同一TX在源域d中的批内中心为

$$
\boldsymbol{\mu}_{c,d}
=\frac{
\sum_{i:y_i=c,d_i=d}\overline{\mathbf{z}}^{\mathrm{id}}_i
}{
\left\|
\sum_{i:y_i=c,d_i=d}\overline{\mathbf{z}}^{\mathrm{id}}_i
\right\|_2+\varepsilon
}.
$$

跨域类条件一致性为

$$
\mathcal{L}_{\mathrm{cons}}
=\frac{1}{|\mathcal{P}|}
\sum_{(c,d,d')\in\mathcal{P}}
\left[
1-\cos(\boldsymbol{\mu}_{c,d},\boldsymbol{\mu}_{c,d'})
\right],
$$

$$
\mathcal{P}
=\{(c,d,d'):d\ne d',\;n_{c,d}>0,\;n_{c,d'}>0\}.
$$

|符号别名|意义|
|---|---|
|`phi`,`psi`|domain head和adversarial domain head参数|
|`Z_id`,`Z_dom`|批内身份/域特征矩阵|
|`bar_Z`|逐维批均值广播矩阵|
|`D_z`|两种表征的维度160|
|`mu_cd`|同TX同源域的单位中心|
|`P`|批内可用的同TX跨域中心对集合|

优化方式：`L_dom`提高`z_dom`的域可预测性；`L_adv`正常训练对抗域头，但GRL把传向identity backbone的梯度乘以负1，从而降低`z_id`的域可预测性；`L_orth`同时更新两条分支，使批内交叉协方差趋近0；`L_cons`直接拉近同一TX在不同源域中的身份中心。四项分别提供“域信息承接、域信息擦除、统计去相关、类条件对齐”，任一项都不能单独证明完全解耦。

### 6.5困难域GroupCE与FishR梯度方差

每个有效源域group的label-smoothed CE为

$$
\ell_g
=-\frac{1}{|\mathcal{B}_g|}
\sum_{i\in\mathcal{B}_g}
\sum_{c=1}^{C}
\widetilde{y}_{i,c}\log p_{i,c},
\qquad
\mathcal{B}_g=\{i\in\mathcal{B}_l:d_i=g\}.
$$

B02实际使用hard-domain模式，不是一般形式的可学习GroupDRO权重。它选择域损失最大的前35%：

$$
K_g=\left\lceil0.35|\mathcal{G}|\right\rceil,
\qquad
\mathcal{L}_{\mathrm{group}}
=\frac{1}{K_g}
\sum_{g\in\operatorname{TopK}_{K_g}
(\{\ell_h:h\in\mathcal{G}\})}
\ell_g.
$$

FishR使用logit交叉熵梯度代理，而不是对身份特征求逐样本真实梯度：

$$
\mathbf{g}_i
=\mathbf{p}_i-\mathbf{e}_{y_i},
\qquad
\mathbf{V}_g
=\operatorname{Var}_{i\in\mathcal{B}_g}(\mathbf{g}_i),
\qquad
\overline{\mathbf{V}}
=\operatorname{sg}\!\left(
\frac{1}{|\mathcal{G}|}\sum_{g\in\mathcal{G}}\mathbf{V}_g
\right).
$$

$$
\mathcal{L}_{\mathrm{fishr}}
=\frac{1}{|\mathcal{G}|C}
\sum_{g\in\mathcal{G}}
\|\mathbf{V}_g-\overline{\mathbf{V}}\|_2^2.
$$

|符号别名|意义|
|---|---|
|`G`|当前批中样本数足够的源域集合|
|`ell_g`|域g的平均TX交叉熵|
|`K_g`|被选入GroupCE的困难域数量|
|`p_i`|样本i的TX softmax概率向量|
|`e_yi`|真实TX one-hot向量|
|`g_i`|对分类logits的CE梯度代理|
|`V_g`|域g中梯度代理的逐类方差|

优化方式：`L_group`把更新重点放到当前批最难的receiver/day组，避免均值被容易域主导；`L_fishr`让不同源域的分类梯度方差接近公共目标，降低域特定优化方向。两者是稳健训练器，不应单独包装成RFFI原创模块。若有效域少于配置下限，相关项返回0。

### 6.6跨epoch prototype memory

prototype bank不是当前批均值，而是在成功optimizer step后用动量更新的跨epoch memory：

$$
\boldsymbol{\mu}_c^{(e)}
=\operatorname{norm}\!\left[
0.95\boldsymbol{\mu}_c^{(e-1)}
+0.05\widehat{\boldsymbol{\mu}}_c^{(e)}
\right],
$$

$$
\widehat{\boldsymbol{\mu}}_c^{(e)}
=\operatorname{norm}\!\left(
\sum_{i\in\mathcal{B}_l:y_i=c}
\overline{\mathbf{z}}^{\mathrm{id}}_i
\right).
$$

设`I_act`为累计样本数不少于2的活跃类样本集合，原型损失为

$$
\begin{aligned}
\mathcal{L}_{\mathrm{proto}}
={}&\frac{1}{|\mathcal{I}_{\mathrm{act}}|}
\sum_{i\in\mathcal{I}_{\mathrm{act}}}
\left[
1-\cos\!\left(
\overline{\mathbf{z}}^{\mathrm{id}}_i,
\operatorname{sg}(\boldsymbol{\mu}_{y_i})
\right)
\right]\\
&+0.10\operatorname{mean}_{c,d}
\left[
1-\cos\!\left(
\boldsymbol{\mu}_{c,d},
\operatorname{sg}(\boldsymbol{\mu}_c)
\right)
\right]\\
&+0.10\operatorname{mean}_{c\ne c'}
\left[
\cos(\boldsymbol{\mu}_c,\boldsymbol{\mu}_{c'})-0.15
\right]_{+}^{2}.
\end{aligned}
$$

|符号别名|意义|
|---|---|
|`mu_c^(e)`|第e轮后类c的动量原型|
|`mu_hat_c^(e)`|当前成功step中类c的批均值|
|`mu_cd`|跨epoch类域原型memory|
|`I_act`|原型计数达到2的样本集合|
|`0.15`|prototype余弦相似度上界margin|

优化方式：第一项把当前`z_id`拉向历史类中心，提供跨batch稳定锚点。当前实现中的class/domain prototype都是无梯度memory tensor，因此第二项和第三项会改变loss数值，但不会把梯度传入模型；它们不能按现实现解释为有效的domain-prototype对齐或prototype推远。报告必须保留这一实现边界，消融时应把主要可训练作用理解为prototype pull。

### 6.7闭集角几何间隔

该项在历史CLI中名为`open_world_feat`，但B02只使用源域旧类样本。本文将其记为闭集角几何损失。记样本到本类中心角为`theta_i+`，到最近异类中心角为`theta_i-`，类间中心角为`theta_cc'`：

$$
\theta_i^{+}
=\vartheta(\mathbf{z}^{\mathrm{id}}_i,\boldsymbol{\mu}_{y_i}),
\qquad
\cos\theta_i^{-}
=\max_{c\ne y_i}
\cos\!\left(
\overline{\mathbf{z}}^{\mathrm{id}}_i,
\boldsymbol{\mu}_{c}
\right).
$$

每类robust-three-sigma半径定义为

$$
r_c^{3\sigma}
=\min\!\left[
\max\Theta_c,
\operatorname{median}(\Theta_c)+3\sigma_{\mathrm{rob}}(\Theta_c)
\right],
\qquad
\Theta_c=\{\theta_i^{+}:y_i=c\}.
$$

完整几何损失为

$$
\begin{aligned}
\mathcal{L}_{\mathrm{geo}}
={}&\operatorname{mean}_i
\left[
\cos(12^\circ)-\cos\theta_i^{+}
\right]_{+}^{2}\\
&+\operatorname{mean}_{c<c'}
\left[
\cos\vartheta(\boldsymbol{\mu}_c,\boldsymbol{\mu}_{c'})
-\cos(55^\circ)
\right]_{+}^{2}\\
&+\operatorname{mean}_i
\left[
\cos\theta_i^{-}+1-\cos(5^\circ)-\cos\theta_i^{+}
\right]_{+}^{2}\\
&+0.14\operatorname{mean}_i
\left[
\theta_i^{+}-r_{y_i}^{3\sigma}
\right]_{+}^{2}\\
&+0.40\operatorname{mean}_i
\operatorname{TopK}_{3}
\left\{
\left[r_c^{3\sigma}+6^\circ
-\vartheta(\mathbf{z}^{\mathrm{id}}_i,\boldsymbol{\mu}_c)
\right]_{+}^{2}:c\ne y_i
\right\}.
\end{aligned}
$$

|符号别名|意义|
|---|---|
|`theta_i+`|样本i到真实类中心的角距离|
|`theta_i-`|样本i到最近异类中心的角距离|
|`Theta_c`|类c全部类内角距离集合|
|`sigma_rob`|依次用MAD、IQR或标准差回退的robust尺度|
|`r_c_3sigma`|不超过类内最大角的robust-three-sigma半径|
|`12,55,5 deg`|类内半径、类间中心和样本级角margin|
|`6 deg`|异类vacuum宽度|

优化方式：第一项压缩普通类内样本，第二项推远过近的类中心，第三项增加样本对真实类与最近异类的角差，第四项只惩罚超过robust半径的类内尾部，第五项把样本推出异类接收锥。全部中心由当前批特征计算，因此梯度回到identity branch。该项不使用目标域、新类或unknown样本。

### 6.8身份表征紧致性与尾部CVaR

跨域supervised contrastive的正样本集合只包含同TX、不同源域样本：

$$
\mathcal{P}(i)
=\{p:y_p=y_i,\;d_p\ne d_i,\;p\ne i\}.
$$

$$
\mathcal{L}_{\mathrm{supcon}}
=-\frac{1}{|\mathcal{I}_{+}|}
\sum_{i\in\mathcal{I}_{+}}
\frac{1}{|\mathcal{P}(i)|}
\sum_{p\in\mathcal{P}(i)}
\log
\frac{
\exp\!\left(
\cos(\overline{\mathbf{z}}_i,\overline{\mathbf{z}}_p)/0.12
\right)
}{
\sum_{a\in\mathcal{B}_l\setminus\{i\}}
\exp\!\left(
\cos(\overline{\mathbf{z}}_i,\overline{\mathbf{z}}_a)/0.12
\right)
}.
$$

角半径与类均衡尾部CVaR为

$$
\mathcal{L}_{\mathrm{rad}}
=\operatorname{mean}_{i}
\left[
\vartheta(\mathbf{z}^{\mathrm{id}}_i,\boldsymbol{\mu}_{y_i})-40^\circ
\right]_{+}^{2},
$$

$$
\mathcal{L}_{\mathrm{cvar}}
=\frac{1}{C}
\sum_{c=1}^{C}
\operatorname{TopFrac}_{0.05}
\left\{
\vartheta(\mathbf{z}^{\mathrm{id}}_i,\boldsymbol{\mu}_c):y_i=c
\right\}.
$$

$$
\mathcal{L}_{\mathrm{zid}}
=0.30\mathcal{L}_{\mathrm{supcon}}
+0.35\mathcal{L}_{\mathrm{rad}}
+0.35\mathcal{L}_{\mathrm{cvar}}.
$$

|符号别名|意义|
|---|---|
|`P(i)`|样本i的同TX跨域正样本集合|
|`I_+`|至少有一个有效正样本的anchor集合|
|`0.12`|SupCon温度|
|`40 deg`|允许的身份角半径|
|`TopFrac_0.05`|每类角距离最差5%的均值|

优化方式：`L_supcon`拉近同TX跨receiver/day样本并相对推远其余批样本；`L_rad`只压缩超过40度的样本；`L_cvar`持续压低每类最差5%尾部。三项均更新identity branch，核心目标不是提高均值，而是缩小最差类和最差接收机的长尾角半径。

### 6.9源域留一类边界与尾部风险

该项的历史代码名为`proxy_unknown_energy_loss`，但训练数据仍全部来自Phase1源域旧类。每个batch按epoch与batch索引循环留出一个TX类h，剩余类构成当前核心类集合：

$$
\mathcal{C}_{K}=\mathcal{C}_{B}\setminus\{h\}.
$$

留出类样本与48个源域特征派生样本合并为边界压力集合。hard模式把48个派生样本均分为component-shell、interclass-bridge和tail-outward三类：

$$
\mathcal{V}
=\mathcal{Z}_{h}
\cup\mathcal{V}_{\mathrm{shell}}
\cup\mathcal{V}_{\mathrm{bridge}}
\cup\mathcal{V}_{\mathrm{outward}}.
$$

相对核心类中心的能量为

$$
E(\mathbf{v})
=-\log\sum_{c\in\mathcal{C}_{K}}
\exp\!\left[
\cos(\overline{\mathbf{v}},\boldsymbol{\mu}_c)
\right].
$$

类内角距离的0.90分位定义核心样本；0.90到0.92分位定义tail；大于0.97分位定义overflow。核心能量阈值使用核心样本能量的0.85分位：

$$
\mathcal{I}_{\mathrm{core}}
=\{i:\theta_i^{+}\le Q_{0.90}(\Theta_{y_i})\},
\qquad
t_E=Q_{0.85}\!\left(
\{E(\mathbf{z}_i):i\in\mathcal{I}_{\mathrm{core}}\}
\right).
$$

$$
\mathcal{I}_{\mathrm{tail}}
=\{i:Q_{0.90}(\Theta_{y_i})<\theta_i^{+}
\le Q_{0.92}(\Theta_{y_i})\},
\qquad
\mathcal{I}_{\mathrm{overflow}}
=\{i:\theta_i^{+}>Q_{0.97}(\Theta_{y_i})\}.
$$

component gate把类半径、能量、最近两类角间隔和核心密度四个门相乘，并在类别维取最大值：

$$
A(\mathbf{v})
=\max_{c\in\mathcal{C}_{K}}
G^{\mathrm{rad}}_c(\mathbf{v})
G^{\mathrm{energy}}(\mathbf{v})
G^{\mathrm{margin}}(\mathbf{v})
G^{\mathrm{density}}(\mathbf{v}).
$$

各有效子损失定义为

$$
\mathcal{L}_{\mathrm{margin}}
=\left[
0-\left(
\operatorname{mean}_{\mathbf{v}\in\mathcal{V}}E(\mathbf{v})
-\operatorname{mean}_{i\notin\mathcal{Z}_h}E(\mathbf{z}_i)
\right)
\right]_{+}^{2},
$$

$$
\mathcal{L}_{\mathrm{vac}}
=\operatorname{mean}_{\mathbf{v}\in\mathcal{V}}
\operatorname{TopK}_{3}
\left\{
\left[
r_c^{\mathrm{gate}}+5^\circ
-\vartheta(\mathbf{v},\boldsymbol{\mu}_c)
\right]_{+}^{2}:c\in\mathcal{C}_{K}
\right\},
$$

$$
\mathcal{L}_{\mathrm{vaccept}}
=\operatorname{TopFrac}_{0.30}
\left\{
\operatorname{sp}\!\left(
\frac{t_E+0.08-E(\mathbf{v})}{0.04}
\right):\mathbf{v}\in\mathcal{V}
\right\},
$$

$$
\mathcal{L}_{\mathrm{core}}
=\operatorname{mean}_{i\in\mathcal{I}_{\mathrm{core}}}
\operatorname{sp}\!\left(
\frac{E(\mathbf{z}_i)-(t_E-0.05)}{0.04}
\right),
$$

$$
\mathcal{L}_{\mathrm{gate}}
=\operatorname{TopFrac}_{0.30}
\{A(\mathbf{v}):\mathbf{v}\in\mathcal{V}\},
$$

$$
\begin{aligned}
\mathcal{L}_{\mathrm{tail}}
={}&\operatorname{TopFrac}_{0.30}
\left\{
\operatorname{sp}\!\left(
\frac{t_E+0.08-E(\mathbf{z}_i)}{0.04}
\right):i\in\mathcal{I}_{\mathrm{tail}}
\right\}\\
&+\operatorname{TopFrac}_{0.30}
\left\{
\operatorname{sp}\!\left(
\frac{A(\mathbf{z}_i)-0.45}{0.04}
\right):i\in\mathcal{I}_{\mathrm{tail}}
\right\},
\end{aligned}
$$

$$
\begin{aligned}
\mathcal{L}_{\mathrm{safe}}
={}&\operatorname{TopFrac}_{0.30}
\left\{
\operatorname{sp}\!\left(
\frac{t_E+0.08-E(\mathbf{z}_i)}{0.04}
\right):i\in\mathcal{I}_{\mathrm{overflow}}
\right\}\\
&+\operatorname{TopFrac}_{0.30}
\left\{
\operatorname{sp}\!\left(
\frac{A(\mathbf{z}_i)-0.25}{0.04}
\right):i\in\mathcal{I}_{\mathrm{overflow}}
\right\}.
\end{aligned}
$$

最终有效组合为

$$
\mathcal{L}_{\mathrm{coretail}}
=\mathcal{L}_{\mathrm{margin}}
+0.55\mathcal{L}_{\mathrm{vac}}
+1.00\mathcal{L}_{\mathrm{vaccept}}
+0.45\mathcal{L}_{\mathrm{core}}
+0.65\mathcal{L}_{\mathrm{gate}}
+0.20\mathcal{L}_{\mathrm{tail}}
+0.20\mathcal{L}_{\mathrm{safe}}.
$$

|符号别名|意义|
|---|---|
|`h`|当前batch轮流留出的一个源域TX类|
|`C_K`|除h外的核心类集合|
|`V`|留出类样本与48个源域派生边界样本|
|`E(v)`|相对核心类中心的负log-sum-exp能量|
|`t_E`|核心样本能量的0.85分位阈值|
|`r_c_gate`|类c角距离的0.80分位component半径|
|`A(v)`|半径、能量、margin和密度门的联合软接收概率|
|`0.30`|最危险30%样本的经验CVaR比例|
|`0.04`|energy与accept softplus温度|

优化方式：最小化`L_core`降低高置信核心能量；其余项提高留出类、派生边界、tail和overflow的能量或降低其软接收概率，并把边界样本推出各类角锥。`virtual_detach=false`，派生样本的梯度会回流到构造它们的源样本特征。该项只构造源域闭集边界压力，不能以其代理指标声明unknown拒识能力。

### 6.10三类软混合边界整形

B02实际不是两类Beta mixup。每个虚拟样本从三个不同TX来源采样均匀随机数，再以二次幂归一化为权重：

$$
a_r\sim\mathcal{U}(0,1),
\qquad
\lambda_r=\frac{a_r^2}{\sum_{s=1}^{3}a_s^2},
\qquad
\sum_{r=1}^{3}\lambda_r=1.
$$

混合特征、软标签和混合logits为

$$
\widetilde{\mathbf{z}}
=\operatorname{norm}\!\left(
\sum_{r=1}^{3}\lambda_r\mathbf{z}_{i_r}
\right),
\qquad
\widetilde{\mathbf{y}}
=\sum_{r=1}^{3}\lambda_r\mathbf{e}_{y_{i_r}},
$$

$$
\widetilde{\boldsymbol{\ell}}
=\sum_{r=1}^{3}\lambda_r
\boldsymbol{\ell}^{\mathrm{tx}}_{i_r}.
$$

注意：CE使用原样本logits的加权和，不是把混合特征重新送入CosFace head。

$$
\mathcal{L}_{\mathrm{mixCE}}
=-\operatorname{mean}
\sum_{c=1}^{C}
\widetilde{y}_{c}
\log\operatorname{softmax}
(\widetilde{\boldsymbol{\ell}})_c.
$$

相对当前批类中心的能量差hinge为

$$
\mathcal{L}_{\mathrm{mixE}}
=\left[
1-\left(
\operatorname{mean}E(\widetilde{\mathbf{z}})
-\operatorname{mean}E(\mathbf{z})
\right)
\right]_{+}^{2}.
$$

vacuum项使用每类robust-three-sigma半径和6度边界宽度：

$$
\mathcal{L}_{\mathrm{mixVac}}
=\operatorname{mean}_{\widetilde{\mathbf{z}}}
\operatorname{TopK}_{3}
\left\{
\left[
r_c^{3\sigma}+6^\circ
-\vartheta(\widetilde{\mathbf{z}},\boldsymbol{\mu}_c)
\right]_{+}^{2}:c=1,\ldots,C
\right\}.
$$

$$
\mathcal{L}_{\mathrm{softmix}}
=0.60\mathcal{L}_{\mathrm{mixCE}}
+1.00\mathcal{L}_{\mathrm{mixE}}
+0.35\mathcal{L}_{\mathrm{mixVac}}.
$$

|符号别名|意义|
|---|---|
|`i_1,i_2,i_3`|三个不同TX来源样本索引|
|`a_r`|0到1均匀随机数|
|`lambda_r`|归一化二次幂混合权重|
|`z_tilde`|归一化三类混合特征|
|`y_tilde`|三类soft target|
|`ell_tilde`|三个源样本TX logits的加权和|
|`mixup_count`|每批构造24个虚拟混合样本|

优化方式：`L_mixCE`让分类logits沿类间插值平滑；`L_mixE`要求三类混合特征比真实源样本具有更高能量；`L_mixVac`把混合特征推出任何单类的窄角锥。`detach_mixup=false`，三项梯度均可回流到来源特征，其中CE还更新TX head。

### 6.11leave-one-source-domain episode

对TX类c和被留出的源域d，使用其他源域样本构造单位中心：

$$
\boldsymbol{\mu}_{c,-d}
=\operatorname{norm}\!\left(
\sum_{i:y_i=c,d_i\ne d}
\mathbf{z}^{\mathrm{id}}_i
\right).
$$

其他域样本到该中心的角距离集合为

$$
\Theta_{c,-d}
=\left\{
\vartheta(\mathbf{z}^{\mathrm{id}}_i,\boldsymbol{\mu}_{c,-d})
:y_i=c,d_i\ne d
\right\}.
$$

robust尺度依次用MAD、IQR和标准差回退：

$$
\sigma_{\mathrm{rob}}(\Theta)
=\begin{cases}
1.4826\operatorname{MAD}(\Theta),&\operatorname{MAD}(\Theta)>0,\\
0.7413\operatorname{IQR}(\Theta),&\operatorname{IQR}(\Theta)>0,\\
\operatorname{Std}(\Theta),&\text{其他情况}.
\end{cases}
$$

episode安全半径为

$$
r_{c,-d}^{\mathrm{epi}}
=\min\!\left[
33^\circ,
\operatorname{median}(\Theta_{c,-d})
+3\sigma_{\mathrm{rob}}(\Theta_{c,-d})
\right].
$$

被留出域的同类样本构成query，三类soft-mix特征同时作为边界压力样本：

$$
\mathcal{L}_{\mathrm{query}}
=\operatorname{mean}_{c,d}
\operatorname{mean}_{i:y_i=c,d_i=d}
\left[
\vartheta(\mathbf{z}^{\mathrm{id}}_i,\boldsymbol{\mu}_{c,-d})
-r_{c,-d}^{\mathrm{epi}}
\right]_{+}^{2},
$$

$$
\mathcal{L}_{\mathrm{epiMix}}
=\operatorname{mean}_{\widetilde{\mathbf{z}}}
\operatorname{TopK}_{3}
\left\{
\left[
r_{c,-d}^{\mathrm{epi}}
-\vartheta(\widetilde{\mathbf{z}},\boldsymbol{\mu}_{c,-d})
\right]_{+}^{2}
\right\}.
$$

$$
\mathcal{L}_{\mathrm{epi}}
=\mathcal{L}_{\mathrm{query}}
+0.75\mathcal{L}_{\mathrm{epiMix}}.
$$

|符号别名|意义|
|---|---|
|`mu_c,-d`|由除d外的源域构造的类c中心|
|`Theta_c,-d`|其他域类内角距离集合|
|`sigma_rob`|robust尺度估计|
|`r_epi_c,-d`|不超过33度的episode安全半径|
|`L_query`|留出源域同类query超半径损失|
|`L_epiMix`|三类混合特征进入任一episode球的惩罚|

优化方式：`L_query`把留出域中的同TX样本拉入由其他域定义的类球，直接模拟未见源域外推；`L_epiMix`把类间混合样本推出这些球，防止单纯扩大半径获得低query损失。只有批内同一TX覆盖至少2个源域时该项有效。

### 6.12receiver-day条件的源域伪标签SSL

EMA teacher只由student参数的指数滑动平均更新：

$$
\overline{\boldsymbol{\theta}}
\leftarrow
0.999\overline{\boldsymbol{\theta}}
+0.001\boldsymbol{\theta}.
$$

对源域无TX标签样本，teacher在weak/clean view上输出

$$
\mathbf{q}_j
=p_{\overline{\boldsymbol{\theta}}}
(y|a_w(\mathbf{u}_j)),
\qquad
\widehat{y}_j
=\operatorname*{arg\,max}_{c}q_{j,c},
\qquad
\kappa_j=\max_c q_{j,c}.
$$

按域自适应阈值为

$$
\tau_d
=\operatorname{clip}
\left(
Q_{0.86}\!\left(\{\kappa_j:d_j=d\}\right),
0.92,
0.97
\right).
$$

domain gate要求teacher的域头预测与已知源域标签一致；temporal gate要求相邻时间窗口的伪标签和置信度一致；strong gate要求student在高斯噪声strong view上的类别与teacher一致。最终门控为

$$
m_j=
\mathbf{1}[\kappa_j\ge\tau_{d_j}]
\mathbf{1}[\widehat d_j=d_j]
\mathbf{1}[T_j=1]
\mathbf{1}
\left[
\operatorname*{arg\,max}_{c}
p_{\boldsymbol{\theta}}
(c|a_s(\mathbf{u}_j))
=\widehat y_j
\right].
$$

伪标签CE为

$$
\mathcal{L}_{u}
=-\frac{1}{\sum_{j\in\mathcal{B}_u}m_j+\varepsilon}
\sum_{j\in\mathcal{B}_u}
m_j\log p_{\boldsymbol{\theta}}
\left(\widehat y_j|a_s(\mathbf{u}_j)\right).
$$

代码实际最小化正的信息熵：

$$
\mathcal{L}_{\mathrm{ent}}
=-\frac{1}{|\mathcal{B}_u|}
\sum_{j\in\mathcal{B}_u}
\sum_{c=1}^{C}
p_{j,c}^{s}\log p_{j,c}^{s},
\qquad
\mathbf{p}_j^{s}
=p_{\boldsymbol{\theta}}(y|a_s(\mathbf{u}_j)).
$$

|符号别名|意义|
|---|---|
|`theta_bar`|EMA teacher参数，不接收梯度|
|`q_j`|teacher weak-view TX概率|
|`y_hat_j`|teacher伪TX标签|
|`kappa_j`|teacher最大置信度|
|`tau_d`|域d内置信度0.86分位截断到0.92至0.97|
|`d_hat_j`|teacher domain head预测|
|`T_j`|时间邻域一致性指示量|
|`m_j`|四个门全部通过时为1|
|`p_j^s`|student strong-view TX概率|

优化方式：`L_u`只拉近通过全部门控的strong view和EMA伪标签；`L_ent`对所有无标签strong view做熵最小化，使预测更尖锐。两项只在第131至200轮启用。门控本身不反向传播；EMA teacher通过成功optimizer step后的参数滑动平均更新。分域阈值的目的不是提高总体接收率，而是避免容易receiver/day垄断伪标签并保护困难域覆盖。

### 6.13源域LEO压力视图

对标注源域样本构造简化LEO残余信道视图，另做一次模型前向。实际启用的监督CE为

$$
\mathcal{L}_{\mathrm{satCE}}
=
-\frac{1}{|\mathcal{B}_l|}
\sum_{i\in\mathcal{B}_l}
\log p_{\boldsymbol{\theta}}
\left(y_i|a_{\mathrm{leo}}(\mathbf{x}_i)\right).
$$

实现中若启用一致性，使用clean概率作为停止梯度teacher，对LEO logits计算单向KL：

$$
\mathcal{L}_{\mathrm{satCon}}
=\frac{1}{|\mathcal{B}_l|}
\sum_{i\in\mathcal{B}_l}
\operatorname{KL}
\left[
\operatorname{sg}\!\left(
p_{\boldsymbol{\theta}}(y|\mathbf{x}_i)
\right)
\;\middle\|\;
p_{\boldsymbol{\theta}}
(y|a_{\mathrm{leo}}(\mathbf{x}_i))
\right].
$$

B02中该一致性项权重为0，只有`L_satCE`从第80轮起产生梯度。历史训练入口中的scenario schedule概率没有控制该快照的前向采样概率；实际生效的是按epoch切换场景列表。因此报告不能声称B02做了LEO表征一致性学习。

|符号别名|意义|
|---|---|
|`a_leo`|简化LEO残余信道增强|
|`p_clean`|clean IQ的TX概率，KL中停止梯度|
|`p_leo`|LEO压力IQ的TX概率|
|`lambda_satCE`|0.68，从第80轮启用|
|`lambda_satCon`|0，不进入B02总目标|

优化方式：`L_satCE`要求在LEO压力视图下保持原TX标签，更新identity backbone、TX head和共享前端。它只能支持“物理启发的源域压力鲁棒性”，不能替代真实卫星数据验证。

### 6.14优化器、梯度流和实际数值行为

AdamW对全部可训练参数执行最小化。忽略偏置修正记号后，更新可概括为

$$
\begin{aligned}
\mathbf{m}_t&=0.9\mathbf{m}_{t-1}+0.1\mathbf{g}_t,\\
\mathbf{v}_t&=0.999\mathbf{v}_{t-1}+0.001\mathbf{g}_t^{\odot2},\\
\boldsymbol{\theta}_{t+1}
&=\boldsymbol{\theta}_t
-2\times10^{-4}
\frac{\widehat{\mathbf{m}}_t}
{\sqrt{\widehat{\mathbf{v}}_t}+10^{-8}}
-2\times10^{-4}\times10^{-4}\boldsymbol{\theta}_t.
\end{aligned}
$$

|损失|主要更新对象|优化方向|
|---|---|---|
|`L_tx`|identity backbone、CosFace、共享前端|提高真实TX概率|
|`L_dom`|domain backbone、RCN enhancer、domain head、共享前端|提高域标签可预测性|
|`L_adv`|adv head、identity backbone、共享前端|adv head识别域；GRL让identity路径混淆域|
|`L_orth`|两条backbone及共享前端|减小身份/域交叉协方差|
|`L_cons`|identity backbone|拉近同TX跨源域中心|
|`L_group`,`L_fishr`|identity backbone、TX head|强调困难域并匹配域间梯度方差|
|`L_proto`|主要是identity backbone|把当前特征拉向跨epoch类memory|
|`L_geo`,`L_zid`|identity backbone|收紧类内/尾部角半径并扩大类间间隔|
|`L_coretail`|identity backbone|保护源域核心并抬高边界/tail能量|
|`L_softmix`|identity backbone、TX head|平滑类间logits并清理类间角空间|
|`L_epi`|identity backbone|提升leave-one-source-domain外推|
|`L_u`,`L_ent`|identity backbone、TX head|学习可信伪标签并降低无标签预测熵|
|`L_satCE`|identity backbone、TX head、共享前端|在源域LEO压力视图保持TX身份|

数值执行边界：AMP开启，没有学习率scheduler和梯度裁剪；prototype memory只在成功optimizer step后更新；EMA同样只在成功step后更新。历史B02共200轮，约13000个batch中有12个batch因non-finite gradient跳过step，比例约0.092%；没有non-finite total loss。PAIC guard在该候选中未触发，因此没有实际把LEO权重乘0.75。上述事实属于训练执行审计，不改变最终损失定义。

## 7.严厉审稿视角下的Phase1创新重组

### 7.1对原A-D划分的主要批评

原划分不宜直接写进论文方法贡献，原因不是模块数量不对，而是抽象层级和因果边界不成立。

|审稿问题|原划分表现|为什么会被质疑|修订原则|
|---|---|---|---|
|粒度失衡|A是完整架构，C近似单个复合损失，D容纳五类机制|四个模块无法做同层级比较|按科学问题而非代码目录分组|
|边界重叠|GRL、MixStyle、source episode、LEO增强都被解释为跨域不变性|性能变化无法归因|把表征分解、SSL、角风险和反事实外推分开|
|改名代替论证|历史`proxy_unknown`直接改称known-core|代码名改变不构成方法创新|只依据实际源域闭集优化对象命名|
|Phase2污染|用qKNN、target-old或support稳定性证明Phase1模块|违反Phase1 source-only边界|Phase1主消融只报告strict UDU、receiver floor和源域压力指标|
|组件新颖性夸大|SincNet、GRL、MixStyle、EMA、SupCon、CVaR分别列为创新|这些机制均有通用先例|贡献放在RFFI任务特定耦合和可验证假设上|
|无效项混入|把权重为0的sat consistency写成有效机制|公式与实际梯度不一致|只包装真实产生梯度的项|

重组后的四个模块互相正交到“可消融”层面：A回答如何表示，B回答如何利用无标签源数据，C回答如何控制最差角几何，D回答如何构造身份保持的域外挑战。

### 7.2模块A：物理分解式身份-域双表征

核心假设：TX的PA非线性、频谱不对称和瞬态结构，与接收机I/Q耦合、日期漂移、信道和噪声统计具有部分可分性。单一embedding同时承担分类与域不变约束容易在identity-style conflict下走捷径；显式的`z_id`与`z_dom`分工更可控。

组成机制：共享Sinc/HF前端；identity的time/frequency/PA路径；domain的time/frequency/PA/DAC/RCN路径；`z_id/z_dom`双输出；域监督、GRL、协方差正交和同TX跨域中心一致性。MixStyle、source episode和LEO压力不归入本模块。

重要性：这是后续SSL和角几何成立的表征基础。若`z_id`仍高度编码receiver，伪标签会放大接收机偏差，prototype也会按域而不是按TX聚类。

相对现有方法的可辩护差异：不能声称SincNet、PA建模或GRL本身新颖。可检验差异是“RFFI物理多视图identity encoder + 显式receiver-nuisance encoder + 非对称分支保留/抑制策略”的联合设计，而不是单路DANN只擦除域信息。

|最小消融|改法|必须观察的Phase1指标|可证伪结论|
|---|---|---|---|
|`A0_parameter_matched_single`|构造参数量匹配的单embedding模型，去掉`z_dom`、域头、GRL和正交|strict UDU、receiver floor、`z_id->receiver` probe、source val|若泄漏不降且UDU不升，双表征贡献不成立|
|`A1_no_pa`|identity与domain均关闭PA path|strict UDU、min class、hard TX类|检验PA物理视图是否提供独立身份证据|
|`A2_no_freq`|关闭frequency path|strict UDU、receiver floor、频谱扰动敏感度|检验镜像频谱视图价值|
|`A3_no_dac_rcn`|domain关闭DAC和RCN enhancer|域分类准确率、`z_id`泄漏、receiver floor|检验显式nuisance承接路径|
|`A4_no_grl_orth_cons`|保留双backbone但关闭三类解耦约束|同上|区分结构容量与解耦训练作用|

声明边界：只有参数量匹配消融同时显示域泄漏下降和strict UDU/receiver floor提高，才能声称任务特定双表征有效。不能声称两种表征统计独立，也不能把domain branch内部无梯度层列为贡献。

### 7.3模块B：receiver-day条件的可信伪标签闭环

核心假设：不同receiver/day的置信度分布不可直接比较。全局阈值会过度接收容易域、排斥困难域，并把接收机偏差写入伪标签。在标注比例0.1下，伪标签质量必须按域校准并经过时间和增强一致性验证。

组成机制：EMA teacher weak view；rx_day内0.86分位阈值及0.92至0.97截断；domain gate；temporal gate；strong-view类别一致性；伪标签CE；正熵最小化。

重要性：无标签数据量是标注数据的7倍。错误伪标签不仅损伤平均准确率，还会扩大困难receiver的类内角尾部，因此需要把precision、coverage和域均衡同时作为优化对象。

相对现有方法的可辩护差异：不能声称Mean Teacher或strong-view一致性新颖。候选差异是利用RFFI已有的receiver/day metadata定义分域可信集合，并将域正确性、时间邻接和增强一致性串成闭环。

|最小消融|改法|必须观察的Phase1指标|可证伪结论|
|---|---|---|---|
|`B0_no_ssl`|`lambda_u=lambda_ent=0`|strict UDU、receiver floor、pseudo oracle audit|无标签闭环的整体增益|
|`B1_global_equal_coverage`|全局阈值，但调到与分域阈值相同总体coverage|pseudo precision、per-receiver coverage、困难域精度|排除“只因接收更多样本”|
|`B2_no_domain_gate`|去掉域预测一致性|伪标签污染率、域泄漏|检验receiver metadata门控|
|`B3_no_temporal`|去掉时间邻域门|相邻窗口标签翻转率、pseudo precision|检验时间稳定性|
|`B4_no_strong`|去掉strong-view agreement|增强后伪标签错误率、strict UDU|检验扰动一致性|
|`B5_student_teacher`|EMA teacher替换为当前student|伪标签抖动、precision和coverage|检验teacher平滑作用|

声明边界：若没有等coverage全局阈值对照，只能称为domain-aware SSL实现，不能声称阈值策略本身优于现有SSL。该模块只使用源域无TX标签样本，不是Phase2 support学习。

### 7.4模块C：尾部风险约束的角原型判别几何

核心假设：CosFace和平均prototype pull可以改善均值，但无法直接控制最差类别、最差receiver和类内角距离长尾。跨接收机失败往往集中在少数尾部样本；因此优化目标必须从均值几何扩展到类均衡CVaR、核心保真和类间边界。

组成机制：跨epoch prototype pull；闭集角几何`L_geo`；跨域SupCon、40度半径和类均衡CVaR；源域留一类`L_coretail`；三类soft-mix边界整形。source episode归入模块D。

重要性：该模块直接对应Phase1的worst receiver、receiver floor和min-class风险，而不是以平均accuracy掩盖少数类坍塌。

相对现有方法的可辩护差异：不能声称prototype、SupCon、mixup或CVaR单项原创。候选差异是把中心、样本margin、类内尾部和类间混合边界放在同一160维角空间，并按类均衡尾部而非全局均值优化。

|最小消融|改法|必须观察的Phase1指标|可证伪结论|
|---|---|---|---|
|`C0_no_angular_risk`|同时关闭`L_proto,L_geo,L_zid,L_coretail,L_softmix`|strict UDU、min class、receiver floor、Q90/Q95角半径|整套角风险目标的必要性|
|`C1_mean_geometry_only`|只保留TX CE、CosFace和prototype pull|同上|建立普通均值度量学习基线|
|`C2_plus_tail`|在C1上加入`L_zid`与`L_coretail`|tail CVaR、min class、receiver floor|检验尾部风险是否超越均值几何|
|`C3_no_softmix`|关闭三类soft-mix|最小类间角、边界混淆、tail CVaR|检验类间虚拟边界|
|`C4_core_q`|core 0.80/0.90对照，其余不变|核心召回、tail overflow、receiver floor|检验核心保真强度|
|`C5_cvar_alpha`|0.20/0.30对照|尾部稳定性和平均准确率代价|检验风险敏感度|

声明边界：该模块只用Phase1闭集DG指标证明成立，不跨用任何Phase2目标域support/query指标或qKNN结果。历史实现命名只作为代码兼容信息，不进入论文方法命名。prototype domain-align和push当前没有模型梯度，也不能列为有效创新。

### 7.5模块D：身份保持的源域反事实外推课程

核心假设：只要求表征域不变不足以覆盖未见receiver和LEO残余信道。训练中应构造“TX身份不变、观测域系统变化”的源域反事实挑战，并在模型已经具备基础分类能力后逐步增强。

组成机制：标注批上的same-TX cross-domain MixStyle；leave-one-source-domain episode及其soft-mix边界项；从第80轮启用的简化LEO压力视图TX CE。GroupCE和FishR作为全局训练稳定器报告，不包装成独立创新；权重为0的sat consistency不属于模块D。

重要性：MixStyle扰动receiver/day统计，source episode模拟留一域外推，LEO CE加入物理启发信道压力。三者分别覆盖风格、域划分和部署信道，均保持TX标签不变且严格source-only。

相对现有方法的可辩护差异：普通MixStyle不要求同TX，通用episodic DG不必利用RFFI身份结构，常规信号增强也不对应残余CFO、相位噪声、Rician/shadowed-Rician、弱多径和SNR变化。创新候选是三类挑战围绕身份保持约束的课程化组合。

|最小消融|改法|必须观察的Phase1指标|可证伪结论|
|---|---|---|---|
|`D0_no_counterfactual`|关闭MixStyle、`L_epi`和`L_satCE`|strict UDU、worst receiver、sat mean/floor|整体反事实课程价值|
|`D1_mixstyle_only`|只开启MixStyle|receiver floor、clean准确率|风格外推贡献|
|`D2_episode_only`|只开启source episode|leave-domain loss、strict UDU|显式域外推贡献|
|`D3_leo_only`|只开启LEO CE|clean-to-stress drop、sat floor|物理压力训练贡献|
|`D4_rx_by_leo_2x2`|receiver挑战开/关与LEO压力开/关二因素|strict UDU与sat floor同row|检验二者互补或冲突|
|`D5_no_late_anneal`|取消MixStyle后期退火|后期收敛、source val和receiver floor|检验课程而非固定强扰动|

声明边界：LEO增强只支持物理启发压力鲁棒性，不是真实在轨验证；MixStyle在无标签批次因缺少TX标签而跳过；sat consistency权重为0，不能写成有效机制。

### 7.6贡献优先级与主消融顺序

|优先级|模块|建议论文定位|先验风险|
|---:|---|---|---|
|1|A物理分解式双表征|核心架构贡献|参数量与计算量混杂，必须容量匹配|
|2|B receiver-day可信伪标签|核心弱标注贡献|必须做等coverage阈值对照|
|3|C尾部角原型几何|条件性方法贡献|组件多，必须先整体再逐层消融|
|4|D源域反事实外推课程|部署导向训练策略|LEO增强不能替代真实部署证据|

推荐先跑`A0/B0/C0/D0`四个模块级消融，再对通过模块级检验的模块做内部拆分。主表只使用Phase1指标：overall、strict UDU、min class、worst receiver、receiver floor、pseudo precision/coverage、`z_id->receiver` leakage、类内角尾部和satellite stress mean/floor。Phase2的target-old、seen-new、qKNN和support指标单独放在Phase2章节。

## 8.qKNNV42：Phase2 Stage2-C轻量注册头

### 8.1任务定义

qKNNV42只处理Phase2 Stage2-C：

$$
\mathcal{S}_t
=\mathcal{S}_{\mathrm{old}}\cup\mathcal{S}_{\mathrm{new}},
\qquad
\mathcal{Q}_t
=\mathcal{Q}_{\mathrm{old}}\cup\mathcal{Q}_{\mathrm{new}}.
$$

其中

$$
\mathcal{S}_{\mathrm{old}}
=\{(\mathbf{x}_i,y_i):y_i\in\mathcal{Y}_{\mathrm{old}},
\;r(\mathbf{x}_i)\in\mathcal{R}_t\},
\qquad
\mathcal{S}_{\mathrm{new}}
=\{(\mathbf{x}_i,y_i):y_i\in\mathcal{Y}_{\mathrm{new}},
\;r(\mathbf{x}_i)\in\mathcal{R}_t\}.
$$

其中`r(x_i)`返回样本对应的receiver domain；support与query来自同一目标接收机域，但样本索引互不重叠。query集合按相同方式定义。

qKNNV42不更新ADV3B02参数。冻结参数记为

$$
\boldsymbol{\theta}^{\star}.
$$

support memory使用冻结且L2归一化的身份特征：

$$
\mathbf{z}_i
=\frac{g_{\boldsymbol{\theta}^{\star}}(\mathbf{x}_i)}
{\|g_{\boldsymbol{\theta}^{\star}}(\mathbf{x}_i)\|_2+\varepsilon}
$$

### 8.2int8量化support memory

每个support向量量化为

$$
\mathbf{q}_i=
\operatorname{clip}\left(
\operatorname{round}(127\,\mathbf{z}_i),-127,127
\right)
\in\{-127,\ldots,127\}^{D_z},
\qquad D_z=160.
$$

推理时近似恢复为

$$
\widehat{\mathbf{z}}_i=
\frac{\mathbf{q}_i/127}{\|\mathbf{q}_i/127\|_2+\varepsilon}.
$$

当前K5主结果中

$$
|\mathcal{Y}_{\mathrm{old}}|=6,\qquad
|\mathcal{Y}_{\mathrm{new}}|=20,\qquad
K_{\mathrm{old}}=K_{\mathrm{new}}=5,
$$

因此support code数量为

$$
N_{\mathrm{code}}=(6+20)\times5=130.
$$

### 8.3qKNNV42打分

query特征为

$$
\mathbf{z}_q
=\frac{g_{\boldsymbol{\theta}^{\star}}(\mathbf{x}_q)}
{\|g_{\boldsymbol{\theta}^{\star}}(\mathbf{x}_q)\|_2+\varepsilon}.
$$

与support code的余弦相似度为

$$
s_{qi}=\mathbf{z}_q^{\top}\widehat{\mathbf{z}}_i.
$$

对每个类别取类内top-m均值。邻居数记为

$$
M_{\mathrm{nn}}.
$$

$$
\operatorname{KNN}_c(q)
=\frac{1}{M_{\mathrm{nn}}}
\sum_{i\in\operatorname{TopM}
(\{s_{qj}:y_j=c\},M_{\mathrm{nn}})}
s_{qi}.
$$

类别prototype为

$$
\boldsymbol{\mu}_c
=\frac{\sum_{i:y_i=c}\widehat{\mathbf{z}}_i}
{\left\|\sum_{i:y_i=c}\widehat{\mathbf{z}}_i\right\|_2+\varepsilon},
\qquad
\operatorname{Proto}_c(q)=\mathbf{z}_q^{\top}\boldsymbol{\mu}_c.
$$

V42线路的主分数写为

$$
S_c(q)
=
(1-\lambda_p)\operatorname{KNN}_c(q)
+\lambda_p\operatorname{Proto}_c(q)
+b_{\mathrm{old}}\mathbf{1}[c\in\mathcal{Y}_{\mathrm{old}}]
+\Delta_{\mathrm{scen}}(q,c)
+\Delta_{\mathrm{lp}}(q,c).
$$

其中当前K5 high-floor行使用

$$
\lambda_p=0.45,
\qquad
b_{\mathrm{old}}=0.001,
\qquad
M_{\mathrm{nn}}=1.
$$

scenario-aware residual只用support估计场景偏移。设`C_s`为场景s中至少有support的类别集合：

$$
\boldsymbol{\mu}_{c,s}
=\operatorname{norm}\!\left(
\sum_{i:y_i=c,\,s_i=s}\widehat{\mathbf{z}}_i
\right),
\qquad
\boldsymbol{\delta}_s
=\frac{1}{|\mathcal{C}_s|}
\sum_{c\in\mathcal{C}_s}
(\boldsymbol{\mu}_{c,s}-\boldsymbol{\mu}_c).
$$

当新类c在场景s中没有support、但该场景至少有2个其他类时，合成场景prototype：

$$
\widetilde{\boldsymbol{\mu}}_{c,s}
=\operatorname{norm}(\boldsymbol{\mu}_c+\boldsymbol{\delta}_s).
$$

对场景属于s的query，残差只补充正向分数差并截断到0.5：

$$
\Delta_{\mathrm{scen}}(q,c)
=0.5\,\mathbf{1}[c\in\mathcal{Y}_{\mathrm{new}}]
\mathbf{1}[c\notin\mathcal{C}_s]
\operatorname{clip}
\left(
\left[
\mathbf{z}_q^{\top}\widetilde{\boldsymbol{\mu}}_{c,s}
-S_c^{\mathrm{pre}}(q)
\right]_{+},
0,
0.5
\right).
$$

label propagation在全部support和query特征上建立10近邻图，但只用support标签初始化。邻接权重为

$$
W_{ij}
=\frac{
\mathbf{1}[j\in\mathcal{N}_{10}(i)]
\exp\!\left(
\mathbf{z}_i^{\top}\mathbf{z}_j/0.05
\right)
}{
\sum_{k\in\mathcal{N}_{10}(i)}
\exp\!\left(
\mathbf{z}_i^{\top}\mathbf{z}_k/0.05
\right)
}.
$$

设support行的Y为one-hot、query行为0，传播8轮并在每轮把support行钳回真实标签：

$$
F^{(0)}=Y,
\qquad
F^{(r+1)}=0.76WF^{(r)}+0.24Y,
\qquad
F_{\mathrm{sup}}^{(r+1)}=Y_{\mathrm{sup}}.
$$

query传播分数逐行标准化并截断：

$$
\Delta_{\mathrm{lp}}(q,c)
=0.025\,
\operatorname{clip}
\left(
\frac{F_{q,c}^{(8)}-\operatorname{mean}_{c'}F_{q,c'}^{(8)}}
{\operatorname{std}_{c'}F_{q,c'}^{(8)}+10^{-6}},
-2,
2
\right).
$$

当前行参数为

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
\operatorname{scope}_{\mathrm{scen}}=\mathcal{Y}_{\mathrm{new}},
\qquad
\operatorname{clip}_{\mathrm{scen}}=0.5.
$$

|符号别名|意义|
|---|---|
|`M_nn`|类内top-m邻居数，当前为1|
|`lambda_p`|prototype混合权重0.45|
|`b_old`|旧类统一加分0.001|
|`mu_c,s`|类c在场景s中的support prototype|
|`delta_s`|由场景s中已有类估计的平均场景残差|
|`S_pre`|加入scenario residual前的当前类别分数|
|`W`|support-query 10近邻行归一化图|
|`Y`,`F`|初始标签矩阵与传播状态|
|`lambda_lp`|label propagation分数权重0.025|

预测为

$$
\widehat y_q
=\operatorname*{arg\,max}_{c\in
\mathcal{Y}_{\mathrm{old}}\cup\mathcal{Y}_{\mathrm{new}}}
S_c(q).
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

相对常见闭集RFFI，qKNNV42不把训练期类别集合永久固定。部署期可识别集合由旧类和已注册新类组成：

$$
\mathcal{Y}_{\mathrm{deploy}}
=\mathcal{Y}_{\mathrm{old}}\cup\mathcal{Y}_{\mathrm{new}}.
$$

目标接收机域到达后，系统用少量support即时扩展该集合，并把持久化部署状态限制为int8 support code、prototype和少量标量。该差异是部署期注册机制，不等于提出了新的神经网络backbone。

### 8.6与机器学习方法的关系

|类似方法|相似点|qKNNV42差异|
|---|---|---|
|KNN|按embedding相似度分类|使用量化support code，并区分old/new角色|
|Nearest Class Mean|使用类中心|同时使用top-m局部近邻和prototype|
|Prototypical Networks|K-shot support形成prototype|backbone未按目标域K-shot分类episode端到端训练，部署期只更新memory|
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
|labelprop|`lambda_lp in {0,0.025}`|无query标签的转导图平滑对弱类的影响|
|support selection|不同seed/support策略|是否能把`seed=421070`式强support转成oracle-free注册机制|

## 9.K10非压缩/压缩更新

当前同一Phase2口径下，K10 40seed结果为。该表只覆盖target-old旧类目标域适应和target-new/seen-new注册识别。

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
|V71A旧类budget4诊断|204|94.43%|93.07%|85.46%|81.43%|92.46%|90.79%|78.61%|70.00%|32/40|21/40|
|V71B额外新类预算top14->10诊断|218|94.65%|93.31%|85.96%|82.71%|92.54%|91.27%|78.79%|70.00%|32/40|23/40|
|V73 radius_proto_sim protect8压缩分支|206|94.69%|93.10%|86.07%|81.43%|92.37%|91.00%|78.54%|70.00%|33/40|20/40|
|V73C radius_proto_sim protect14诊断|218|94.70%|93.10%|86.11%|81.43%|92.49%|91.07%|78.82%|70.00%|33/40|22/40|
|V74 radius_proto_sim extra top12->9诊断|210|94.68%|93.10%|86.04%|81.43%|92.44%|91.00%|78.79%|70.00%|33/40|21/40|
|V74 radius_proto_sim extra top14->10诊断|218|94.70%|93.10%|86.11%|81.43%|92.49%|91.07%|78.82%|70.00%|33/40|22/40|
|V75 query-pair cluster诊断|210|94.67%|93.10%|86.04%|81.43%|92.17%|90.36%|77.04%|70.00%|25/40|17/40|
|V76 budget7 radius_proto_sim protect12|206|94.67%|93.10%|86.04%|81.43%|92.45%|91.14%|78.64%|70.00%|33/40|23/40|
|V76 budget7 radius_proto_sim protect14诊断|212|94.67%|93.10%|86.04%|81.43%|92.48%|91.00%|78.82%|70.00%|33/40|23/40|
|V78 K5 compact-diverse fill诊断|130|93.38%|92.83%|82.82%|81.43%|87.58%|85.59%|66.93%|61.14%|3/40|1/40|
|V78 K5 global max-min诊断|130|94.95%|94.05%|86.82%|84.29%|90.59%|86.94%|72.64%|62.00%|17/40|10/40|
|V78 K5 scenario-edge诊断|130|94.95%|94.05%|86.82%|84.29%|91.65%|90.16%|75.43%|68.43%|23/40|16/40|
|V78 K10 budget6 radius_proto_sim protect12高压缩分支|198|94.69%|93.31%|86.14%|82.71%|92.38%|91.06%|78.39%|70.00%|32/40|21/40|
|V79 K10轻labelprop高压缩分支|198|94.67%|93.31%|86.14%|82.71%|92.51%|91.13%|78.32%|70.00%|33/40|21/40|
|V81 K10风险类小额support预算|202|94.67%|93.31%|86.14%|82.71%|92.60%|91.32%|78.61%|70.00%|33/40|22/40|
|V82 K10完整support邻域对比诊断|202|94.67%|93.31%|86.14%|82.71%|92.35%|91.14%|78.39%|71.29%|32/40|17/40|
|V83 K10窄support邻域对比诊断|202|94.67%|93.31%|86.14%|82.71%|92.54%|91.27%|78.71%|70.00%|33/40|21/40|
|V84 K10微support邻域对比诊断|202|94.67%|93.31%|86.14%|82.71%|92.59%|91.32%|78.68%|70.00%|33/40|22/40|
|V86 K10辅视图增强正式策略|202|94.90%|93.52%|86.57%|82.86%|93.31%|91.84%|80.14%|71.29%|34/40|25/40|
|V87 K10辅视图floor稳定正式策略|202|96.06%|94.29%|89.43%|84.29%|95.77%|94.41%|85.07%|77.00%|37/40|34/40|
|V88 K10 180码高效压缩正式策略|180|96.40%|94.76%|90.07%|87.00%|95.92%|94.42%|85.50%|77.14%|39/40|34/40|
|V77 K5 scenario_diverse support选择|130|95.02%|94.05%|86.79%|84.29%|91.67%|89.29%|76.64%|68.57%|24/40|14/40|
|V77 K5+V56 support-LOO重链诊断|130|93.96%|93.10%|83.93%|81.43%|85.47%|84.00%|68.96%|61.43%|3/40|0/40|

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
- V70把V69机制移到更高效压缩：旧类5码/类，seen-new默认8码/类，按半径保护top10 seen-new类全量K10 support。它用210个support code保持old和min old基本不变，seen-new均值为92.45%，并把`min_new>=80`提升到23/40。因此V70在当时取代V69成为K10均衡高效压缩候选；但`min_new p10=70.00%`仍未改善，`1-12`和`1-1`仍是低端瓶颈。
- V71A继续压缩旧类support到4码/类，stored codes降到204，但旧类均值下降到94.43%、min old均值下降到85.46%、`min_new>=80`下降到21/40；旧类域适应损失超过压缩收益，不晋升。
- V71B在V70基础上给额外高风险seen-new类更多support码，最好均值行为top14->10，seen-new均值升至92.54%，但stored codes回到218，旧类均值略降，`min_new p10`、`min_new>=75`和`min_new>=80`均不改善。因此最低类坍塌不是单纯增加seen-new support码可以解决，V71B只作为负诊断保留。
- V72A support质量加权、V72C同场景pair refine和V73A slot release均未超过V70；V72B审计把低类失败定位到`1-1/1-12/8-3`雨弱簇和`19-3/1-15`晴弱簇。
- V73把保护指标从单半径换成`radius_proto_sim`并保留top8 seen-new全量K10 support。它以206个support code把`min_new>=75`从32/40升到33/40，但`min_new>=80`降到20/40、seen-new均值略低。因此V73是高压缩/floor75分支，不替代V70。
- V74修正当前实验语义：命令和输出主字段使用`new_role=target_new`，旧N20 HP08L5包的抽样来源单独记录为`new_selection_role=target_unknown`和`used_legacy_target_new_role=true`。这只是旧导出role名兼容，不改变当前任务边界；当前主线只评价target-old旧类目标域适应和target-new/seen-new注册识别。
- V74继续扫描`radius_proto_sim`下的额外新类预算。最佳均值行top14->10达到218码、seen-new均值92.49%、min-new均值78.82%、`min_new>=75=33/40`，但`min_new>=80=22/40`仍低于V70；同码数top12->9也只有`min_new>=80=21/40`。因此V74不晋升，说明继续按同一风险排序补support码不能解决最低类坍塌。
- V75在V70上扫描query-pair cluster局部配额重排。最佳行仍把`min_new>=75`从32/40降到25/40，`min_new>=80`从23/40降到17/40，说明batch-local pair重排会放大当前低类簇不稳定性，不晋升。
- V76把seen-new默认预算从8码压到7码，并用`radius_proto_sim protect12`保护高风险新类。该行仅保存206个support code，比V70少4码，同时保持`old=94.67%`、`min_old=86.04%`、`seen_new=92.45%`、`min_new>=80=23/40`，并把`seen_new p10`升到91.14%、`min_new>=75`升到33/40。因此`stable_dualview_v76`取代V70成为当前K10高效压缩最佳优化版本。
- 当前边界仍未变：`min_new p10=70.00%`，最低类失败仍集中在`1-1/1-12/8-3`和`19-3/1-15`。V76解决的是更高效压缩和floor75覆盖，不是彻底解决最低类坍塌。
- V77把K5从单split强support证据推进到注册期可执行的support选择机制：在`K=5,pool=10`下使用已有`scenario_diverse`策略，每个旧类和seen-new类仍只保存5个support code，总码数130，不增加存储；40seed均值达到`old=95.02%`、`min_old=86.79%`、`seen_new=91.67%`、`min_new=76.64%`，相对K5 `stable_first`的`old=94.11%`、`seen_new=87.88%`、`min_new=68.57%`明显提升。因此当前K5推荐执行口径为`--policies scenario_diverse`加K5严格qKNN打分，而不是继续依赖`seed=421070`单点。
- V77的K5边界也必须保留：`min_new p10=68.57%`，最低类仍集中在`1-1/1-12/8-3`以及局部`19-3/1-15`。它解决的是K5 support选择稳定性和旧类域适应，不是彻底解决最低类坍塌。
- K5+`stable_dualview_v56` support-LOO重链为负诊断：它额外引入平均4186个ridge标量、4160个old-residual标量、1288个pair-linear标量和20个query-cluster临时prototype，但seen-new均值降到85.47%、`min_new>=75`降到3/40，违背高效压缩方向，不晋升。
- V78首先检查K5更轻量的support-only覆盖选择。紧凑补位把`min_new>=75`降到3/40，全局max-min把`min_new p10`降到62.00%，scenario-edge虽把`min_new>=80`升到16/40但在`seed=421050`出现`2-13=34.29%`极端坍塌。因此这些K5实验策略均为负诊断，未保留生产入口；K5默认仍是V77 `scenario_diverse`。
- V78同时注册K10高压缩分支`stable_dualview_v78`：旧类5码/类，seen-new默认6码/类，按`radius_proto_sim`保护top12 seen-new类全量K10 support，并沿用V76轻量`local_competition`、`labelprop`和`scenario_residual`。正式40seed复验使用198个support code，相对当前worktree的V76复核少8码，保持`min_new>=75=32/40`和`min_new>=80=21/40`不降，旧类均值和min old略升，但seen-new均值从92.44%小降到92.38%、min new均值从78.54%小降到78.39%。因此V78是高压缩分支，不替代旧V76历史均衡最佳。
- V78边界仍是`min_new p10=70.00%`，低类失败仍有`421069(1-1=64.29%)`、`421059(1-1=67.14%)`、`421066(1-12=68.57%)`、`421047/421045(19-3=70.00%)`、`421076(1-12=71.43%)`、`421068(2-13=74.29%)`等seed。它提升的是样本压缩效率，不是最低类坍塌的最终解。
- V79在V78的198码结构上把`labelprop_weight`从0.015降到0.01，并保留`local_competition_weight=0.02`和`scenario_residual_weight=0.5`。正式40seed复验达到`old=94.67%`、`min_old=86.14%`、`seen_new=92.51%`、`min_new=78.32%`、`min_new>=75=33/40`、`min_new>=80=21/40`。因此V79成为当前推荐K10高压缩分支：同198码下比V78恢复floor75并提升seen-new均值；但若强调floor80，V76 historical仍是强基线。
- V79边界仍是`min_new p10=70.00%`，最低seed为`421069(1-1=64.29%)`、`421059(1-1/1-12=67.14%)`、`421066(1-12=68.57%)`、`421047/421045(19-3=70.00%)`。它不是低类坍塌终解，下一步应面向`1-1/1-12/8-3`和`19-3/1-15`设计support-only弱类簇或轻量多原型机制。
- V80验证了两个不应继续加重的方向：`support_proto_anchor`会新增4160个标量且把`min_new>=80`降到19/40；`core_proto`会新增52-78个core prototypes且最好也只到19/40。因此简单全量support原型回拉和通用类内多原型不是当前低类修复方向。
- V81在V79基础上只给support-only风险排序第13-14的新类小额补码：保持旧类5码/类、seen-new默认6码/类、`radius_proto_sim protect12`，并设置`extra_top=14,extra_budget=8`。正式40seed复验达到202个support code、`old=94.67%`、`min_old=86.14%`、`seen_new=92.60%`、`min_new=78.61%`、`min_new>=75=33/40`、`min_new>=80=22/40`。因此V81成为当前K10“高效压缩+低类floor80折中”推荐分支：比V79多4码但恢复1个floor80 seed，仍比V76 historical少4码；若只看最低码数，V79仍保留为198码分支。
- V81仍未改变`min_new p10=70.00%`和worst seed 64.29%的硬边界，最低类继续集中在`1-1/1-12`、`19-3/1-15`和局部`2-13`。后续不能再靠单纯增加support预算，应转向更细的support-only弱类簇判据。
- V82-V84在V81的202码结构上尝试support-only邻域对比，不使用query分布门控，也不改变当前target-old和target-new/seen-new评估边界。V82完整对比新增平均889.02个标量，虽把`min_new p10`抬到71.29%，但`min_new>=80`从22/40坍塌到17/40，是负诊断。V83窄对比新增318.85个标量，`min_new mean`升到78.71%，但`min_new>=80`仍降到21/40，也不晋升。
- V84微对比只覆盖1个support-only最高风险seen-new类，新增平均164.93个标量，在同202个support code下保持V81的`min_new>=75=33/40`和`min_new>=80=22/40`，并把`min_new mean`从78.61%小幅升到78.68%。但V84没有改善`min_new p10=70.00%`、worst seed 64.29%或floor80数量，因此只能作为可选微对比分支；默认最佳仍是无额外标量的V81。
- V85在V81的202码结构上扫描`labelprop_weight`、`scenario_residual_weight`和`scenario_residual_clip`。较轻传播能提高old/new均值，但会损失floor75或floor80；保留`min_new>=75=33/40`且`min_new>=80=22/40`的最好组合仍是V81原参数。因此V85是负诊断，不注册稳定策略。
- V86固定V81的202码压缩结构，只把已有辅视图融合权重从0.34提高到0.38，并把旧类偏置设为0.002；不增加support code，不增加邻域对比标量。正式40seed复验达到`old=94.90%`、`min_old=86.57%`、`seen_new=93.31%`、`min_new=80.14%`、`min_new p10=71.29%`、`min_new>=75=34/40`、`min_new>=80=25/40`、worst seed 65.71%。因此`stable_dualview_v86`取代V81/V84/V76，成为当前K10默认最佳优化版本。
- V87继续固定V86的202码压缩结构，只把已有辅视图融合权重提高到0.58，并保持`old_bias=0.002`；不增加support code，不增加邻域对比标量。正式40seed复验达到`old=96.06%`、`min_old=89.43%`、`seen_new=95.77%`、`min_new=85.07%`、`min_new p10=77.00%`、`min_new>=75=37/40`、`min_new>=80=34/40`、worst seed 70.00%。相对V86，old均值+1.15pp、min old均值+2.86pp、seen-new均值+2.46pp、min new均值+4.93pp；40个seed里seen-new全提升，min-new为36升、3平、1降。因此`stable_dualview_v87`取代V86，成为当前K10默认最佳优化版本。0.64高权重诊断均值更高，但`min_new p10=75.71%`、`min_new>=80=33/40`低于0.58，故不作为默认入口。
- V88在V87基础上同时提高辅视图融合权重和压缩效率：旧类5码/类、seen-new默认5码/类、`radius_proto_sim protect10`，不再使用额外新类support预算，最终保存180个support code。正式40seed复验达到`old=96.40%`、`min_old=90.07%`、`seen_new=95.92%`、`min_new=85.50%`、`min_new p10=77.14%`、`min_new>=75=39/40`、`min_new>=80=34/40`、worst seed 72.86%。相对V87，support code减少22个，old均值+0.34pp、min old均值+0.64pp、seen-new均值+0.14pp、min new均值+0.43pp，并保持floor80。因此`stable_dualview_v88`取代V87，成为当前K10默认最佳优化版本。
- V89把K5从V77的support选择最佳推进到正式打分策略：在`K=5,pool=10`下仍使用`--policies scenario_diverse`，不再压缩support code，保持130个support code；`stable_dualview_v89`只注入`aux_score_weight=0.64`、`local_competition_weight=0.02`和`scenario_residual_weight=0.25,scope=new`，并显式关闭K5 labelprop。正式40seed复验达到`old=96.35%`、`min_old=89.89%`、`seen_new=94.96%`、`min_new=83.25%`、`min_new p10=75.71%`、`min_new>=75=37/40`、`min_new>=80=30/40`、worst seed 72.86%。相对V77 K5 `scenario_diverse`同40seed，old均值+1.32pp、min old均值+3.11pp、seen-new均值+3.29pp、min new均值+6.61pp，floor75从24/40升到37/40，floor80从14/40升到30/40。因此当前推荐口径为K5使用`scenario_diverse + stable_dualview_v89`，K10仍使用V88的180码结构；V89的K10入口仅作为继承V88机制的兼容策略，不替代V88作为K10命名基线。

### 9.8星上轻量head：移除dense query graph

2026-07-14在当前正式8类Stage2-C协议上追加了独立轻量化确认。原publication runner的label propagation会为每个query batch构造`(S+Q)×(S+Q)`float64相似度矩阵和转移矩阵；启用160维主特征+96维FFT辅助时，该dense链执行两次，既要求整批query同时到达，也使内存随batch平方增长。

轻量路径固定单视图FFT96、逐样本argmax和原support-memory qKNN，只移除dense label propagation，并把旧类常数偏置从`+0.001`调整为`-0.001`。偏置在seed 713101-713105诊断集选择，最终证据来自未参与选择的seed 713106-713110，覆盖5个target receiver×5个seed×5档K-shot，共125个paired run。

|head|old_acc|seen-new acc|H_old,new|head MAC|dense graph下界|延迟/query|
|---|---:|---:|---:|---:|---:|---:|
|FFT96+dense LP|74.4133%|65.2133%|68.6486%|22.725 M|1,658,880 B|0.10824 ms|
|FFT96+无LP轻量head|74.3711%|65.7867%|68.9109%|2.818 M|0 B|0.06053 ms|

轻量head的估算MAC下降87.60%，实测head延迟下降44.08%；`old_acc`变化`-0.042pp`，`seen_new_acc`变化`+0.573pp`，`H_old_new`变化`+0.262pp`，三个矩阵均值均满足相对原head下降不超过3pp。它不依赖query-query图、query batch状态、old/new角色Oracle或类别配额，可执行逐样本流式推理。逐run层面仍有7行seen-new和2行H下降超过3pp，因此该版本证明的是矩阵均值资源-性能门槛，不是每一行的最坏情况保证。

该head压缩不能与历史5-view完整栈混为一谈。完整栈同时包含60 epoch feature adapter、5-view、FFT96和非部署Hungarian Oracle；其高准确率无法单独归因于TTA，而且Oracle使用普通在线部署不可获得的query角色与类别配额。下一轮端到端压缩必须固定adapter与逐样本决策后单独比较1/2/3/5-view。

## 10.证据索引

|证据|路径|用途|
|---|---|---|
|项目协议|`E:\type10-7\项目.md`|CVS科学场景、数据协议、Stage2边界|
|AGENTS规则|`E:\type10-7\AGENTS.md`|Git、N607、报告和中文排版规则|
|ADV3B02主报告|`E:\type10-7\automation_reports\CV-SincNet\phase1_adv3_mechanism32_queue_20260701\report.md`|B02身份、数据、候选和prototype导出|
|ADV3B02分析|`E:\type10-7\automation_reports\CV-SincNet\phase1_adv3_mechanism32_queue_20260701\full_analysis_20260702.md`|Phase1边界和B02结论|
|ADV3B02候选表|`E:\type10-7\automation_reports\CV-SincNet\phase1_adv3_mechanism32_queue_20260701\adv3_m32_candidate_summary.csv`|B02同row指标|
|ADV3B02训练入口快照|`E:\type10-7\code\snapshots\phase1_adv3_mechanism32_queue_20260701\launch_phase1_adv3_mechanism32_queue_20260701.sh`|B02参数、候选variant和启动语义|
|ADV3B02训练逻辑快照|`E:\type10-7\code\snapshots\phase1_adv3_mechanism32_queue_20260701\train_ssdg.py`|B02实际损失装配、阶段调度、伪标签和优化器；本报告公式权威|
|ADV3B02损失实现快照|`E:\type10-7\code\snapshots\phase1_adv3_mechanism32_queue_20260701\losses.py`|B02各子损失精确定义；本报告公式权威|
|CVS模型谱系|`E:\type10-7\code\model.py`、`E:\type10-7\code\model_dual_cvsincnet.py`|CV-SincNet和双分支层参数；Git谱系一致，但缺少2026-07-01远端独立哈希|
|当前SSDG演化版本|`E:\type10-7\code\SSDG\train_ssdg.py`|后续演化参考，不用于覆盖B02历史训练公式|
|qKNNV42策略实现|`E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_support_metric_qknn_probe.py`|V42策略、top-m、prototype、labelprop、scenario residual|
|qKNNV42主报告|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\report.md`|V42矩阵和high-floor行解释|
|qKNNV42最佳JSON|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v53_fftlogmag_20260706\local_v55_diagnostics_20260706\k5_strict_seed421070_floor_param_best_predictions_20260707.json`|当前同row指标和support/query指纹|
|qKNNV77 K5 support选择证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v77_k5_support_policy_20260709\k5_v77_support_policy_seed421038_40.csv`|K5 40seed support选择策略网格|
|qKNNV77 K5负诊断证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v77_k5_v56_scenariodiverse_20260709\k5_v77_v56_scenariodiverse_seed421038_40.csv`|K5 support-LOO重链负诊断|
|qKNNV78 K5覆盖选择负诊断|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v78_k5_coverage_policy_20260709\k5_v78_coverage_policy_seed421038_40.csv`|K5 compact/max-min/scenario-edge support选择负诊断|
|qKNNV78 K10预算扫描证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v78_k10_budget5_6_protect_20260709\k10_v78_budget5_6_protect_seed421038_40.csv`|K10 V76机制下5/6/7码预算和保护范围扫描|
|qKNNV78正式策略证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v78_policy_20260709\k10_v78_policy_seed421038_40.csv`|`stable_dualview_v78` 198码高压缩分支40seed复验|
|qKNNV79轻labelprop诊断证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v79_k10_v78_light_knobs_20260709\k10_v79_v78_light_knobs_seed421038_40.csv`|V78 198码结构下`local_competition`、`labelprop`和`scenario_residual`轻量扫描|
|qKNNV79正式策略证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v79_policy_20260709\k10_v79_policy_seed421038_40.csv`|`stable_dualview_v79` 198码当前推荐高压缩分支40seed复验|
|qKNNV80轻量原型负诊断|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v80_k10_support_anchor_diag_20260709\k10_v80_support_anchor_diag_seed421038_40.csv`、`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v80_k10_core_proto_diag_20260709\k10_v80_core_proto_diag_seed421038_40.csv`|support anchor和core proto负诊断|
|qKNNV81风险类小额预算诊断|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v81_k10_extra_budget_diag_20260709\k10_v81_extra_budget_diag_seed421038_40.csv`|V79 198码结构上的额外风险类support预算扫描|
|qKNNV81正式策略证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v81_policy_20260709\k10_v81_policy_seed421038_40.csv`|`stable_dualview_v81` 202码当前K10效率-floor80折中分支40seed复验|
|qKNNV82完整support邻域对比诊断|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v82_k10_neighbor_contrast_20260709\k10_v82_neighbor_contrast_seed421038_40.csv`|`stable_dualview_v82` 202码完整support-only邻域对比负诊断|
|qKNNV83窄support邻域对比诊断|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v83_k10_narrow_neighbor_contrast_20260709\k10_v83_narrow_neighbor_contrast_seed421038_40.csv`|`stable_dualview_v83` 202码窄support-only邻域对比负诊断|
|qKNNV84微support邻域对比诊断|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v84_k10_micro_neighbor_contrast_20260709\k10_v84_micro_neighbor_contrast_seed421038_40.csv`|`stable_dualview_v84` 202码微support-only邻域对比非负诊断|
|qKNNV85轻传播/残差负诊断|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v85_k10_v81_light_knobs_20260709\k10_v85_v81_light_knobs_seed421038_40.csv`、`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v85b_k10_v81_clip_knobs_20260709\k10_v85b_v81_clip_knobs_seed421038_40.csv`|V81 202码结构下labelprop和scenario residual扫描，未超过V81 floor约束|
|qKNNV86辅视图增强正式策略证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v86_policy_20260709\k10_v86_policy_seed421038_40.csv`|`stable_dualview_v86` 202码当前K10默认最佳分支40seed复验|
|qKNNV87辅视图高权重诊断|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v87_auxhigh_diag_20260709\k10_v87_auxhigh_seed421038_40.csv`、`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v87_auxhigher_diag_20260709\k10_v87_auxhigher_seed421038_40.csv`、`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v87_auxmax_diag_20260709\k10_v87_auxmax_seed421038_40.csv`、`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v87_auxupper_diag_20260709\k10_v87_auxupper_seed421038_40.csv`|V86 202码结构下`aux_score_weight=0.40..0.64`扫描，用于选择floor稳定权重0.58|
|qKNNV87辅视图floor稳定正式策略证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v87_policy_20260709\k10_v87_policy_seed421038_40.csv`|`stable_dualview_v87` 202码当前K10默认最佳分支40seed复验|
|qKNNV88压缩扫描证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v88_k10_compression_aux_diag_20260709\k10_v88_compression_aux_seed421038_40.csv`、`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v88_k10_deeper_compression_aux_diag_20260709\k10_v88_deeper_compression_aux_seed421038_40.csv`|V87之后的K10 support预算压缩扫描，用于选择180码正式结构|
|qKNNV88 180码正式策略证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v88_policy_20260709\k10_v88_policy_seed421038_40.csv`|`stable_dualview_v88` 180码当前K10默认最佳分支40seed复验|
|qKNNV89 K5轻残差正式策略证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v89_policy_20260709\k5_v89_policy_seed421038_40.csv`|`scenario_diverse + stable_dualview_v89` 130码当前K5默认最佳分支40seed复验|
|qKNNV89 K10继承复验证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v89_policy_20260709\k10_v89_policy_seed421038_40.csv`|`stable_dualview_v89` K10继承V88 180码结构的兼容复验|
|qKNNV42星上轻量head报告|`E:\type10-7\automation_reports\CV-SincNet\qknnv42_lightweight_head_20260714_173701\report.md`|FFT96独立125-run的dense LP移除、MAC/内存/延迟和≤3pp门槛证据|

## 11.下一步

1.按新模块A-D跑ADV3B02模块级消融，主表只保留Phase1指标：overall、strict UDU、min class、worst receiver、receiver floor、pseudo precision/coverage、`z_id->receiver` leakage、类内角尾部和satellite stress mean/floor。

2.按qKNNV42组件跑Phase2消融，主表保留`K`、stored codes、old mean、min old、seen-new mean、min new、`H_old,new`和support/query指纹；当前主线只评价target-old旧类目标域适应和target-new/seen-new注册识别。

3.K5当前默认最佳为`scenario_diverse + stable_dualview_v89`，保持130个support code，并显著提升旧类域适应、seen-new注册和低类地板；下一步应继续围绕`1-1/1-12`、`19-3/1-15`和局部`2-13`做更细的support-only弱类簇机制。

4.K10当前默认最佳为V88 180码高效压缩分支，V89在K10上只提供兼容继承入口，V87保留为202码强基线，V79保留为198码历史高压缩对照；下一步应针对`19-3/1-15`和局部`1-1/1-12`做低类专门的support-only类簇机制，而不是继续单纯增加support预算或加重邻域对比。

5.继续复核更多`R_t`目标接收机域，避免单receiver或单support split过拟合。
