# CV-SincNet type1-type9 模型迭代完整报告

## 0. 报告范围与结论概览

本报告分析 `D:\codefile\CV-SincNet` 中 `type1` 到 `type9` 的 `model.py` 演进过程。这里的“type10 之前”按目录主线理解为 `type1/type2/.../type9`。分析重点是模型结构，而不是完整训练脚本；但由于很多结构变化是为了配合对比学习、原型约束、DAC/PA 辅助监督、WiSig 迁移和跨域泛化，本报告会在必要处说明训练目标对模型设计的牵引作用。

总体结论是：`type1` 到 `type9` 不是简单堆叠网络层数，而是一条逐步清晰的技术路线：

```text
原始 IQ 分类 baseline
  -> SincConv 物理前端
  -> 时域 + 频域双分支
  -> DAC 缺陷辅助监督
  -> 身份特征与缺陷特征解耦
  -> 镜像频谱 / 圆度 / 高频统计等复 IQ 专用特征
  -> Lite 化与 CosFace 判别头
  -> 频域贡献强化
  -> WiSig 长度适配 + PA 扩展 + 星地/跨域任务前置准备
```

这条路线背后的核心思想是：射频指纹识别不能只追求“分类准确”，还要尽量区分三类信息：

1. 发射机身份相关的稳定硬件缺陷；
2. DAC/PA 等物理链路造成的可解释非理想特征；
3. 接收机、信道、采集日期、星地链路等非身份域因素。

早期模型主要证明 SincConv 可以处理 IQ 序列；中期模型开始显式建模 DAC 非线性和频谱镜像；后期模型则把模型设计推向“物理感知、分支解耦、跨域可迁移”的方向。

## 1. 版本结构证据

从文件体量和哈希可以看到模型结构的增长过程：

| 版本 | 行数 | 文件大小 | 与前一关键版本关系 | 主要结构变化 |
|---|---:|---:|---|---|
| `type1/model.py` | 116 | 4046 B | 起点 | 简单 SincConv + CNN 分类器 |
| `type2/model.py` | 305 | 10613 B | 大幅重写 | Hz 参数化 SincConv、时频双分支、DAC 辅助头 |
| `type3/model.py` | 305 | 10613 B | 与 type2 完全相同 | 模型不变，训练目标增强 |
| `type4/model.py` | 370 | 13223 B | 结构升级 | Volterra 多项式、身份/DAC 解耦、full FFT |
| `type5/model.py` | 450 | 16634 B | 频域优化 | 镜像压缩频谱、圆度特征 |
| `type6/model.py` | 641 | 22691 B | Lite 重构 | 深度可分离卷积、Sinc 后非线性基、CosFace、DAC-aware classifier |
| `type7/model.py` | 728 | 26748 B | 频域贡献增强 | 频带门控、频域统计、时/频辅助头、branch drop |
| `type8/model.py` | 728 | 26748 B | 与 type7 完全相同 | 模型不变，训练/增强策略变化 |
| `type9/model.py` | 907 | 32615 B | WiSig 迁移前完整版本 | 输入长度适配、PA 分支、缺陷子带聚合、稳定 CosFace、模型工厂 |

需要特别说明两点：

- `type2` 和 `type3` 的 `model.py` 完全一致，所以 `type3` 的主要贡献不在模型结构，而在训练目标中加入 SupCon、Prototype 和 DAC loss。
- `type7` 和 `type8` 的 `model.py` 完全一致，所以 `type8` 的主要贡献也不在模型结构，而在 domain generalization 或增强策略实验上。

## 2. type1: 原始 SincNet 分类 baseline

### 2.1 模型结构

`type1/model.py` 中只有两个核心类：

- `SincConv_Fast`
- `CVSincNet`

输入为 `(B, 2, L)` 的 IQ 序列，其中通道 0 是 I，通道 1 是 Q。`SincConv_Fast` 不直接学习普通卷积核，而是学习每个滤波器的低截止频率 `f1` 和带宽 `band`，再用 sinc 公式生成带通滤波器：

```text
g[n] = 2*f2*sinc(2*pi*f2*n) - 2*f1*sinc(2*pi*f1*n)
```

I 路和 Q 路分别通过同一组 Sinc 滤波器，随后拼接为 `2*out_channels` 个通道。之后进入常规 CNN：

```text
IQ
 -> SincConv_Fast
 -> BatchNorm + LeakyReLU + MaxPool
 -> Conv1d + BN + LeakyReLU + MaxPool
 -> Conv1d + BN + LeakyReLU + MaxPool
 -> Global Average Pooling
 -> FC classifier
```

### 2.2 设计意义

这个版本的意义是建立最小可行基线。射频指纹识别的原始输入是 IQ 时序，而不是图像或语音特征。普通卷积可以学习局部波形模式，但它没有频带解释性。SincConv 的价值在于把第一层约束成可学习带通滤波器组，既保留神经网络的自适应能力，又引入信号处理先验。

对 RFFI 来说，设备缺陷往往表现为频偏、调制误差、IQ 不平衡、非线性失真、高频纹波等。这些特征和频率结构高度相关。因此，type1 使用 SincConv 是合理的起点：它让模型从一开始就按照“频带响应”去观察原始 IQ。

### 2.3 局限

type1 仍是纯分类网络，所有可分辨信息都被压进同一个 embedding。它无法区分：

- 哪些特征来自发射机身份；
- 哪些特征来自信道或接收机；
- 哪些特征来自 DAC/PA 等物理缺陷；
- 哪些只是数据集同分布捷径。

因此它适合作为 baseline，但不足以支撑跨 day、跨 rx、星地信道等更复杂场景。

## 3. type2: Hz 参数化 SincConv 与时频双分支

### 3.1 模型结构变化

`type2/model.py` 是第一次大重写。核心变化包括：

1. `SincConv_Fast` 改为 `SincConv1d`；
2. 频率参数从归一化数值改为真实 Hz；
3. 用 mel 间隔初始化频带；
4. 加入 `HighFreqEmphasis`；
5. 加入频域 FFT 分支；
6. 加入 DAC 强度预测头。

整体数据流变为：

```text
IQ
 ├─ time branch:
 │    I -> SincConv1d
 │    Q -> SincConv1d
 │    IQ -> HighFreqEmphasis
 │    concat -> Conv blocks -> t_emb
 │
 ├─ freq branch:
 │    I + jQ -> complex FFT -> log magnitude
 │    -> Conv blocks -> f_emb
 │
 └─ fuse:
      concat(t_emb, f_emb)
      -> shared embedding
      -> class logits
      -> dac_pred
```

### 3.2 Hz 参数化 SincConv 的意义

type1 的频率参数是归一化的，适合快速实验，但不够物理化。type2 使用 `sample_rate=5e6`，把滤波器参数放到真实 Hz 语境中。这带来两个好处：

1. 模型参数更容易和采样率、信号带宽联系起来；
2. 后续迁移到不同数据集或不同采样率时，频带设计更可控。

mel 初始化虽然源自语音，但这里的作用主要是让低频区域分布更细，高频区域分布更宽，避免所有滤波器线性挤在频谱上。对 RF 信号来说，这不是严格物理定律，但作为初始化策略可以提高滤波器组覆盖的稳定性。

### 3.3 HighFreqEmphasis 的意义

`HighFreqEmphasis` 用固定一阶差分和二阶差分处理 I/Q：

```text
d1[n] = x[n] - x[n-1]
d2[n] = x[n] - 2x[n-1] + x[n-2]
```

这类差分对突变、量化台阶、高频纹波和细小非平滑变化敏感。DAC 量化误差、slew-rate 限制、采样保持效应、硬件非线性都可能在局部波形中留下高频痕迹。把差分特征显式喂给模型，等于提醒网络：不要只看低频包络和调制主体，也要看细小硬件缺陷。

### 3.4 频域分支的意义

type2 的频域分支把复 IQ 做 FFT，并取正频幅度谱。这是模型从纯时域走向“时频联合”的第一步。

时域分支擅长捕捉局部波形形状，频域分支擅长捕捉频谱结构。RFFI 中很多硬件缺陷会表现为：

- 谱旁瓣变化；
- 频谱泄漏；
- 镜像频率增强；
- 非线性导致的频谱再生；
- 高频能量比例变化。

如果只用时域 CNN，这些结构需要模型自己间接学习；加入频域分支后，频谱证据被显式暴露出来。

### 3.5 DAC 辅助头的意义

type2 增加 `dac_pred`，让模型不仅分类设备，还预测 DAC 扰动强度。这个设计非常关键，因为它把“物理扰动”变成一个可监督对象。

它的意义不是最终要预测 DAC 强度，而是通过辅助任务迫使 backbone 组织出与 DAC 相关的特征。这样可以让模型内部特征更可解释，也为后续身份/缺陷解耦打基础。

## 4. type3: 模型不变，训练目标增强

### 4.1 与 type2 的关系

`type3/model.py` 与 `type2/model.py` 完全一致。因此 type3 不是结构迭代，而是训练范式迭代。

训练脚本中加入：

- supervised contrastive loss；
- prototype memory；
- prototype contrastive loss；
- DAC loss；
- warmup 与 ramp。

### 4.2 设计意义

type2 已经提供了 `feat` 和 `dac_pred`，type3 则开始真正利用这些输出。其核心目标是让同一设备的不同增强视图在特征空间中更接近，不同设备更远离。

这对 RFFI 很重要。因为一个设备的信号经过相位旋转、噪声、轻微频偏、DAC 扰动后，标签仍然是同一个发射机。如果模型只用交叉熵，它可能依赖某些偶然域特征；加入 SupCon 和 prototype 后，训练目标变成：

```text
同一设备的多视图 -> embedding 聚合
不同设备的样本 -> embedding 分离
每个设备类别 -> 靠近稳定原型
```

这一步意味着项目开始从“分类器”变成“表示学习系统”。

## 5. type4: DAC 专门化、Volterra 基与特征解耦

### 5.1 模型结构变化

type4 的注释直接说明了三点设计：

1. 使用 Volterra-like polynomial basis；
2. 分离身份 embedding `feat_id` 和 DAC embedding `feat_dac`；
3. 使用 full-spectrum complex FFT 捕捉非线性造成的 spectral regrowth。

模型中新增 `_volterra_stack`：

```text
IQ -> [I, Q, I^3, Q^3, I^5, Q^5]
```

如果启用 Volterra，则 6 个通道分别通过 SincConv，然后与高频差分特征拼接。

融合后不再直接得到一个 `feat`，而是：

```text
base -> id_proj  -> feat_id  -> classifier
base -> dac_proj -> feat_dac -> dac_head
```

### 5.2 Volterra 基的意义

DAC、PA、IQ 链路中的很多非理想效应可以近似看作非线性系统。经典通信建模中，非线性常用多项式或 Volterra 系列近似。例如：

```text
y = a1*x + a3*x^3 + a5*x^5 + ...
```

type4 把 `I^3/Q^3/I^5/Q^5` 显式作为输入基函数，意义是把模型可能需要学习的非线性变换提前暴露出来。这样做有两个好处：

1. 降低模型从零学习非线性模式的难度；
2. 让 DAC 非线性相关特征更容易进入辅助分支。

不过，这种设计也带来计算开销，因为 SincConv 要作用于 6 路输入，而不是 2 路输入。这一点后来在 type6 被优化。

### 5.3 身份/DAC 特征解耦的意义

type2/type3 中，分类和 DAC 预测共享同一个 `feat`。这会产生目标冲突：分类希望特征对增强和扰动稳定，而 DAC 预测希望特征对 DAC 强度敏感。

type4 将其拆成：

- `feat_id`：用于设备身份分类、对比学习、原型约束；
- `feat_dac`：用于 DAC 强度预测。

这一步是模型思想上的重要转折。它承认 RF 信号中同时存在“身份信息”和“扰动信息”，而且二者不应该完全混在一起。

对于后续星地信道问题，这个思想同样重要：星地信道可以被视为另一类强 nuisance factor。要提升抗信道能力，不能只增强训练数据，还要让模型学会把身份因素和信道因素拆开。

### 5.4 Full FFT 的意义

type2 只取正频幅度，type4 改成 full FFT，可选 `fftshift`。对复 IQ 来说，正频和负频不是冗余关系。IQ 不平衡、镜像泄漏、非圆性、硬件缺陷都可能破坏正负频的结构关系。

因此 full-spectrum FFT 能捕捉更多复基带特有的信息，尤其是后续 type5/type7/type9 中强调的镜像频谱不对称。

## 6. type5: 镜像感知频谱压缩

### 6.1 模型结构变化

type5 保留 type4 的身份/DAC 解耦思想，但重构频域分支。它不再直接把完整 FFT 幅度送入 CNN，而是构造压缩后的镜像频谱特征：

```text
pos power -> logP_pos
neg power -> logP_neg
ratio     -> log(pos / neg)
circularity rho = |E[z^2]| / E[|z|^2]
```

正频和负频先按频率绝对值对齐，再池化成 K 个子带。

### 6.2 为什么关注正负频镜像

理想复基带信号与实际硬件采集信号之间，经常存在镜像相关问题。例如：

- IQ 增益不平衡；
- IQ 相位不平衡；
- DC offset；
- 混频器镜像泄漏；
- 非线性失真导致频谱再生；
- DAC/PA 缺陷在正负频上的不对称表现。

这些现象不一定能通过单边幅度谱充分表达。type5 把正频、负频、正负频比值显式组织起来，让模型更容易利用复 IQ 的物理结构。

### 6.3 圆度系数的意义

`rho = |E[z^2]| / E[|z|^2]` 描述复信号的非圆性或 improperness。理想圆对称复信号的 `E[z^2]` 接近 0；如果存在 IQ imbalance 或镜像泄漏，`E[z^2]` 可能增大。

因此 `rho` 是一个很轻量但物理意义明确的统计量。把它拼入融合层，相当于给模型一个全局 IQ 质量指标。

### 6.4 频谱压缩的工程意义

type4 的 full FFT 输入长度较长，频域 CNN 计算较重。type5 把频谱池化成 K 个子带，大幅缩短频域序列长度。这既提升效率，又减少模型在单个频点上过拟合噪声的可能。

因此 type5 的设计可以概括为：保留复 IQ 频域物理信息，但用更紧凑、更可解释的方式表达。

## 7. type6: Lite 重构、Sinc 后非线性基与 CosFace

### 7.1 模型结构变化

type6 是又一次大重构。核心变化包括：

1. SincConv 只作用于原始 I/Q，不再作用于 6 路 Volterra 输入；
2. 非线性基改为在 Sinc 滤波器组输出后构造；
3. 加入 `time_fuse` bottleneck；
4. 早期下采样；
5. 普通 ConvBlock 改为 depthwise-separable ConvBlock；
6. 引入 `CosFaceHead`；
7. 引入 `DACAwareClassifier`；
8. 引入专门给 SupCon/Prototype 使用的 `con_proj`。

### 7.2 Sinc 后非线性基的意义

type4 的做法是：

```text
[I,Q,I^3,Q^3,I^5,Q^5] -> SincConv
```

type6 改成：

```text
[I,Q] -> SincConv -> z|z|^2 / z|z|^4
```

这种改变很聪明。它保留了非线性缺陷建模能力，但避免对 6 路输入分别做 SincConv，计算量明显下降。同时，先滤波再构造非线性项，也更像“在不同频带上观察非线性响应”。

### 7.3 Lite 化的意义

type6 使用：

- bottleneck 1x1 卷积；
- early AvgPool；
- depthwise-separable Conv1d。

这些设计不是单纯为了加速。RFFI 数据量通常有限，模型过大容易记住采集条件、接收机或信道特征。Lite 化有助于控制模型容量，让模型更偏向学习稳定、可迁移的指纹特征。

### 7.4 CosFace 的意义

普通线性分类头学习的是 `W*x + b`。CosFace 先归一化特征和类别权重，再用余弦相似度分类，并对正确类别加入 margin：

```text
logits = s * (cos(theta) - m)
```

它的效果是让同类特征更紧，不同类之间角度间隔更大。对于射频指纹识别，这很有价值，因为不同设备之间的差异可能非常细微。CosFace 能促使 embedding 空间形成更稳定的类别边界。

### 7.5 DAC-aware classifier 的意义

type6 的分类器不再简单用 `feat_id` 分类，而是：

```text
base -> feat_cls
base -> feat_dac
feat_dac -> gate feat_cls
concat(feat_cls, feat_dac) -> joint_proj -> CosFace
feat_dac -> dac_pred
```

这说明设计者意识到：DAC 缺陷既可能是身份的一部分，也可能是扰动的一部分。完全丢弃 DAC 信息会损失分类线索；完全混入身份特征又会破坏泛化。因此 type6 采用折中方式：让 DAC 分支参与分类，但通过单独投影和门控控制其作用。

### 7.6 con_proj 的意义

`feat_con` 专门供 SupCon/Prototype 使用，而分类使用 `feat_cls/joint`。这是为了避免不同训练目标抢同一个 embedding。

交叉熵、CosFace、SupCon、Prototype 的几何偏好不完全一致。单一 embedding 同时承载所有损失，容易出现优化冲突。`con_proj` 相当于给对比学习一个专用投影空间，这是现代表示学习里很常见也很合理的设计。

## 8. type7: 频域贡献强化

### 8.1 模型结构变化

type7 在 type6 基础上增加了频域增强机制：

- `FreqBandGate1d`；
- `freq_stats_proj`；
- `aux_head_t`；
- `aux_head_f`；
- `branch_drop_p`；
- 更丰富的 forward 输出字典。

### 8.2 为什么要强迫频域分支贡献

在双分支模型中，经常出现一个问题：融合后模型可能主要依赖更强、更容易优化的分支，另一个分支变成摆设。对 IQ 序列来说，时域分支通常更直接，频域分支如果没有额外约束，很可能贡献不足。

但频域信息对 RFFI 很重要，尤其是：

- 非线性频谱再生；
- 正负频镜像不对称；
- 高频能量比例；
- 频谱平坦度；
- IQ 非圆性。

因此 type7 加入机制，强迫频域分支参与。

### 8.3 FreqBandGate1d 的意义

`FreqBandGate1d` 对频域 K 个子带生成门控系数：

```text
scale = 1 + alpha * (2*sigmoid(g)-1)
y = x * scale
```

这让模型能够动态强调某些子带。例如某些设备的 DAC/PA 缺陷可能主要表现在高频边缘或某些镜像子带；固定平均池化可能稀释这些证据，而频带门控可以把关键频带放大。

### 8.4 频域统计的意义

type7 中的 `f_stats` 包含：

- `hf_ratio`：高频能量比例；
- `asym_hf_mean`：高频镜像不对称均值；
- `flatness`：频谱平坦度。

这些统计量非常适合做 RF 缺陷提示。它们不依赖某一个具体频点，而描述整体频谱形态，因此比原始频谱更稳健。

### 8.5 分支辅助头的意义

`aux_head_t` 和 `aux_head_f` 分别要求时域 embedding、频域 embedding 自己也具备分类能力。这样做有两个作用：

1. 防止某个分支被融合层忽略；
2. 便于训练时诊断时域和频域到底哪个更有效。

这在后续消融中很重要，因为可以判断模型性能来自时域、频域、统计特征还是物理缺陷分支。

### 8.6 branch_drop_p 的意义

训练时随机丢掉时域 embedding，可以迫使模型在部分 batch 中依赖频域分支。这类似多分支 dropout，目的是降低模型对单一强分支的依赖。

这种设计对跨域泛化有意义。因为信道变化可能严重扰动某一类特征，如果模型只依赖单一路径，遇到新域时容易崩溃。多分支都具备判别力，鲁棒性会更好。

## 9. type8: 模型冻结，训练策略迁移

`type8/model.py` 与 `type7/model.py` 完全一致。因此 type8 的意义不在结构，而在训练和实验目标。结合目录命名和训练脚本可知，type8 更偏向 domain generalization 方向。

这说明到 type7 为止，单模型结构已经比较成熟，后续开始更多探索：

- 如何构造训练视图；
- 如何安排增强强度；
- 如何加权损失；
- 如何评估跨域泛化。

从研究路线看，这是合理的。模型结构不可能无限增加复杂度；当结构已经具备时域、频域、缺陷分支和辅助输出后，真正影响泛化的往往变成训练策略。

## 10. type9: WiSig 迁移与 PA/星地任务前置准备

### 10.1 模型结构变化

type9 是 type10 前最完整的单模型版本。相对 type7/type8，主要变化包括：

1. 加入 `pad_crop_iq`；
2. `CVSincNet` 增加 `dataset/input_len/pad_crop_mode`；
3. `DACAwareClassifier` 扩展为 DAC/PA impairment classifier；
4. 新增 `SubbandGatedAggregator`；
5. `CosFaceHead` 改为 FP32 稳定实现；
6. 新增 `build_model` 工厂函数；
7. forward 返回 `pa_pred` 和 `feat_imp`。

### 10.2 输入长度适配的意义

早期 ORACLE 数据常见长度为 1024，而 WiSig IdSig 常用 256。type9 的 `pad_crop_iq` 负责把任意输入适配到目标长度。

这一步看起来只是工程细节，但意义很大。它标志着模型从单一数据集实验，走向多数据集迁移。没有这个适配，很多结构中固定的池化、频带数、FFT 分辨率都会被输入长度影响。

### 10.3 从 DAC 到 DAC/PA 的扩展

type9 中 `feat_dac` 更名或泛化为 `feat_imp`，同时输出：

- `dac_pred`
- `pa_pred`

这说明物理缺陷建模从单一 DAC 扩展到更完整的发射链路。PA 非线性、记忆效应、饱和压缩等也是 RF 指纹的重要来源。将 PA 加入辅助监督，能让模型更好地区分“发射机固有缺陷”和“外部域扰动”。

不过这也带来风险：如果 PA/DAC 增强参数不是设备稳定的，而是随机强扰动，那么它可能变成非身份因素，反而破坏分类。因此后续训练中需要谨慎设计 PA/DAC 标签和增强方式。

### 10.4 SubbandGatedAggregator 的意义

`SubbandGatedAggregator` 对频域子带进行 softmax 加权聚合，生成 `imp_delta`，再注入到缺陷 embedding：

```text
feat_f -> band weights -> weighted aggregation -> imp_delta
feat_imp = feat_imp + imp_delta
```

这个设计非常贴合 RF 指纹。硬件缺陷不一定均匀分布在全频带，而可能集中在某些子带、边缘频带、镜像频带或高频部分。子带聚合器相当于让模型自动学习“哪些频带更像缺陷证据”。

它比全局平均更灵活，也比直接堆 CNN 更可解释。

### 10.5 FP32 稳定 CosFace 的意义

type9 的 `CosFaceHead` 在 AMP 下强制用 FP32 做归一化和余弦相似度计算，避免 float16 下 `normalize` 产生 0/0 或 NaN。

这属于训练稳定性设计。随着模型引入更多辅助分支、对比损失、物理增强和混合精度，数值稳定会变得非常关键。type9 的这个修改说明项目已经从“能跑”进入“长训练可稳定复现”的阶段。

### 10.6 build_model 的意义

`build_model` 支持：

- `model_size = S/M/L`
- `dataset = oralce/wisig`
- `input_len`
- `sample_rate_hz`

这让模型构造从硬编码变成可配置。它是后续大量实验、消融和迁移的基础。对科研项目来说，这一步的意义是提升复现性和实验组织能力。

## 11. type9 的完整前向传播解释

type9 forward 可以拆成以下步骤：

### 11.1 输入适配

```text
x -> pad_crop_iq(x, input_len)
```

保证 ORACLE/WiSig 等不同长度输入能进入同一模型结构。

### 11.2 时域 Sinc 滤波

```text
I -> SincConv
Q -> SincConv
concat -> sinc_iq
```

这里提取的是可学习频带响应，保留 SincNet 的信号处理先验。

### 11.3 Sinc 后非线性基

```text
sinc_iq -> z|z|^2
optional -> z|z|^4
```

用于捕捉 DAC/PA 非线性痕迹。

### 11.4 高频差分

```text
IQ -> first difference + second difference
```

用于捕捉高频纹波、突变、量化台阶等局部硬件缺陷。

### 11.5 时域卷积编码

```text
concat(sinc_iq, nonlinear_basis, high_freq)
 -> time_fuse
 -> time_down
 -> DSConv blocks
 -> t_emb
```

得到时域身份/缺陷混合表示。

### 11.6 频域镜像特征

```text
IQ -> FFT
 -> pos/neg power
 -> logP_pos, logP_neg, logR, asym
 -> freq_gate
 -> f_emb
```

得到频域表示，同时计算 `rho`、`f_stats`。

### 11.7 缺陷子带聚合

```text
feat_f -> SubbandGatedAggregator -> imp_delta
```

从频域子带中提取更像 DAC/PA 缺陷的增量。

### 11.8 双分支辅助分类

```text
t_emb -> logits_t
f_emb -> logits_f
```

保证时域和频域分支各自具有判别能力。

### 11.9 融合与对比学习投影

```text
concat(t_emb, f_emb, rho)
 -> fuse
 -> base
 -> con_proj -> feat_con
```

`feat_con` 用于 SupCon/Prototype，避免对比学习和分类头直接争抢同一表示。

### 11.10 缺陷感知分类

```text
base -> feat_id
base -> feat_imp
feat_imp += imp_delta
feat_imp -> gate feat_id
concat(feat_id, feat_imp) -> feat_joint
feat_joint -> CosFace -> logits
feat_imp -> dac_pred / pa_pred
```

最终输出不仅有分类 logits，还有 DAC/PA 预测、身份特征、缺陷特征和对比学习特征。

## 12. 关键设计思想总结

### 12.1 SincConv: 把第一层变成可解释滤波器组

SincConv 是整个项目的底座。它把普通卷积核约束为带通滤波器，让第一层天然具备频率解释性。对于 RF 信号，频率结构和硬件缺陷密切相关，因此这个选择比纯 CNN 更符合任务性质。

### 12.2 时频双分支: 同时观察波形细节和频谱形态

时域分支关注局部波形和瞬态变化，频域分支关注频谱镜像、高频能量和非线性再生。二者互补，尤其适合 RFFI。

### 12.3 高频差分: 显式突出细粒度硬件缺陷

差分算子不需要学习，直接把量化、slew、纹波等高频变化暴露给模型。这是低成本但很有效的物理先验。

### 12.4 Volterra / 非线性基: 把 DAC/PA 缺陷显式参数化

多项式非线性是通信系统建模中的常见近似。模型显式加入 `x^3/x^5` 或 `z|z|^2/z|z|^4`，可以降低神经网络学习非线性缺陷的难度。

### 12.5 镜像频谱: 针对复 IQ 的专用建模

正负频不是普通实信号频谱中的简单冗余。IQ imbalance、镜像泄漏、非圆性都体现在正负频关系上。type5 后的模型把这一点作为频域分支核心，是很有 RF 任务特色的设计。

### 12.6 特征解耦: 让身份和物理扰动各有空间

从 `feat_id/feat_dac` 到 `feat_id/feat_imp/feat_joint`，模型逐步承认不同信息源需要不同表示。这个思想对跨域泛化和星地信道鲁棒性尤其重要。

### 12.7 CosFace: 提升细粒度设备类别间隔

RFFI 的类别差异很细，普通 softmax 可能形成松散边界。CosFace 通过角度 margin 提升类间分离和类内紧凑性，适合设备身份识别。

### 12.8 分支辅助头: 防止融合网络偷懒

`logits_t/logits_f` 让时域和频域都必须具备分类能力，避免某个分支在融合后被忽略。这有利于可解释评估，也有利于跨域鲁棒性。

### 12.9 频带门控与子带聚合: 让模型找关键缺陷频带

硬件缺陷往往集中在某些频段。频带门控和子带聚合让模型从“全频平均”变成“选择性关注关键频段”，这比盲目增加卷积层更有意义。

### 12.10 输入长度和模型工厂: 从单实验走向可复现实验体系

type9 的 `pad_crop_iq` 和 `build_model` 表示项目开始面向不同数据集、不同输入长度和不同模型规模。科研项目后期需要大量消融和复现，这些工程设计不可少。

## 13. 与星地信道鲁棒性的关系

虽然 `type1` 到 `type9` 的模型迭代主要围绕 DAC/PA、WiSig 和跨 day/rx 泛化，但其中很多思想可以直接服务于星地信道鲁棒性。

星地信道会带来：

- 大尺度路径损耗；
- 小尺度 Rice/Loo/Rayleigh 衰落；
- 多普勒频移；
- 相位噪声；
- 多径；
- SNR 变化；
- 大气附加衰落；
- 遮挡状态变化。

这些因素会改变 IQ 信号的幅度、相位、频谱和时序结构。若模型把这些变化误当作发射机身份，就会在星地测试中性能灾难下降。

从已有迭代看，最有价值的思想包括：

1. 继续保留 `feat_id` 与 `feat_imp` 的分离；
2. 增加信道相关分支，例如 `feat_ch` 或 `channel_pred`；
3. 把星地信道参数作为辅助监督或对抗监督；
4. 对同一发射机不同星地信道视图加入一致性约束；
5. 在频域分支中显式处理多普勒/CFO/相位旋转造成的谱偏移；
6. 避免把随机信道扰动注入到“发射机固有缺陷”分支。

也就是说，现有 type9 的结构已经具备“物理因素拆分”的雏形。下一步不应只是把星地信道作为普通增强加入训练，而应把它纳入解耦学习框架：身份分支保留发射机不变信息，信道分支吸收星地变化，分类头尽量基于信道不变身份表示。

## 14. 版本迭代的研究叙事

如果用于论文或组会汇报，可以把模型迭代讲成以下逻辑：

### 14.1 问题起点

原始 IQ 信号中包含发射机硬件缺陷，但也混有信道、接收机、噪声和采集条件。简单 CNN 可以在同分布数据上分类，却容易学习数据集捷径。

### 14.2 第一层改造

引入 SincConv，用可学习带通滤波器替代完全自由的一维卷积核，使模型前端更符合 RF 信号频带分析直觉。

### 14.3 双域观察

增加时域和频域双分支，让模型同时利用波形局部细节和频谱全局结构。

### 14.4 物理缺陷建模

加入高频差分、DAC 强度预测、Volterra/非线性基，把硬件非理想性从隐式特征变成显式可学习对象。

### 14.5 表示解耦

将身份特征和 DAC/PA 缺陷特征拆开，减少辅助任务与身份分类之间的冲突。

### 14.6 判别空间优化

加入 SupCon、Prototype、CosFace 和专用 projection head，使同一发射机的多视图聚合，不同发射机之间保持更大间隔。

### 14.7 频域贡献强化

通过镜像频谱、圆度、频带门控、频域统计、频域辅助头，使模型真正利用复 IQ 的频域物理结构。

### 14.8 迁移与泛化准备

通过输入长度适配、PA 分支、模型工厂和更稳定的数值实现，支持 WiSig、day/rx 跨域和后续星地信道测试。

## 15. 当前模型路线的优点与风险

### 15.1 优点

1. 物理先验明确：SincConv、差分、非线性基、镜像频谱都与 RF 信号特性相关。
2. 表示更可解释：模型输出不只有 logits，还有 `feat_cls/feat_imp/feat_con/dac_pred/pa_pred`。
3. 支持多目标训练：分类、对比、原型、DAC/PA 辅助、分支一致性都能接入。
4. 有跨域扩展潜力：身份/扰动拆分思想可以扩展到信道解耦。
5. 工程可复现性增强：type9 开始提供模型工厂和输入长度适配。

### 15.2 风险

1. 辅助分支过多时，损失权重容易冲突。
2. DAC/PA 如果随机增强过强，可能从身份特征变成域扰动，破坏标签一致性。
3. 频域分支可能受 CFO/多普勒影响，需要配合补偿或不变性约束。
4. CosFace、SupCon、Prototype 同时使用时，embedding 空间约束较强，需要稳定 warmup。
5. 星地信道测试中，信道因素可能压过发射机缺陷，现有模型还缺少显式信道分支。

## 16. 面向后续 type10+ 或星地信道的建议

基于 type1-type9 的演进，后续建议不是盲目加深网络，而是沿着“解耦与不变性”继续推进：

1. 增加 `feat_ch` 信道分支，用于预测或吸收星地信道参数。
2. 对 `feat_id` 加 domain adversarial loss，使其不含 day/rx/channel 信息。
3. 对同一发射机不同星地信道视图加入 supervised contrastive 或 consistency loss。
4. 明确区分发射机固有缺陷增强和信道扰动增强，避免标签语义冲突。
5. 在输入前加入 CFO/多普勒补偿模块，降低模型学习负担。
6. 做通道因素消融，确认灾难下降主要来自多普勒、相噪、多径还是遮挡衰落。
7. 保留 type9 的频域镜像和高频统计，但增加对频移的鲁棒处理。

## 17. 最终总结

`type1` 到 `type9` 的模型演进可以概括为一次从“神经网络分类器”到“物理感知射频指纹表示学习模型”的转变。

`type1` 证明了 SincConv + CNN 可以处理原始 IQ；`type2/type3` 引入时频双分支和 DAC 辅助监督；`type4/type5` 开始显式建模非线性和镜像频谱，并拆分身份与 DAC 特征；`type6` 通过 Lite 化、CosFace 和 DAC-aware classifier 提升效率和判别性；`type7/type8` 强化频域分支与泛化训练；`type9` 则完成 WiSig 迁移前的关键工程和物理扩展。

这条路线最有价值的地方在于：它没有把射频指纹识别当作普通时间序列分类，而是不断把通信物理、硬件缺陷、复 IQ 频谱结构和表示学习结合起来。对于后续天基射频指纹识别，最应该继承的不是某一个具体模块，而是这种设计原则：

```text
让模型看见物理机制，
让特征空间区分身份与扰动，
让同一发射机在不同信道下保持表示一致。
```

