# ADV3B02物理因子化交叉重构设计规格

状态：待用户书面规格复核

设计来源：`E:\codex\home\attachments\1d377d1c-fe81-4e09-ac71-dc857e445413\pasted-text.txt`

项目协议：`E:\type10-7\项目.md`，版本2026-08-30，`protocol_schema=p2_min_v1`

## 1.目标与范围

本设计把报告中的因子化交叉重构完整接入ADV3B02 Phase1，候选暂命名为`ADV3B02-FCR`。它保留现有ADV3B02身份主干、域分支和PA分支，在显式开关下增加可组合的内容、发射机响应和nuisance因子模型。关闭FCR时，旧模型参数、`state_dict`键、forward输出、checkpoint加载和训练行为必须保持不变。

FCR是Phase1地面辅助训练机制。它不得读取Phase2 support/query，不改变`p2_min_v1`数据权限，也不把source proxy结果写成真实target unknown或Phase3协同结果。

## 2.科学语义

本设计采用报告第十二节的方案A：clean是参考观测，LEO视图由同一已裁剪clean IQ施加一次允许的LEO弱信道增强得到。因此`z_n^leo`解释相对于clean的新增复合nuisance，不声明恢复纯星地信道；`z_f`也不声明恢复绝对TX CFO。

clean/LEO配对只直接识别共享因素`[z_s,z_f]`与变化因素`z_n`。内容与发射机的进一步分离必须依赖三轴干预batch、身份监督、公共preamble或激励状态匹配，不能仅凭`L_swap`作因子可辨识声明。

## 3.候选集成方式

采用并行、可开关分支，不原位替换ADV3B02：

- `use_fcr=false`：不实例化FCR参数，保持旧ADV3B02逐项兼容。
- `use_fcr=true`：保留`z_id_raw`和原分类输出作为匹配对照；新增`z_f_id`、`z_tx_state`、`z_s`、`z_n`及FCR辅助输出。
- 报告规定的身份损失只作用于`z_f_id`。第一轮实验不擅自把`z_f_id`与`z_id_raw`残差融合，也不继承另一份ECRS设计中的`K=28`、64维和`rho_max=0.25`。
- 正式候选通过显式feature schema选择`z_f_id`；`z_id_raw`只承担兼容性和matched control。

现有ECRS-V1中的固定响应基、可微加权岭回归、锚点编码和可辨识性统计可以作为`FingerprintResponseOperator`的内部实现参考，但不得替代三因子编码器、物理顺序Decoder、latent cross-cycle、定向指纹移植和三轴干预。

## 4.整体架构

FCR采用以下生成顺序：

```text
x
├─C_omega(x) -> [x_tilde, eta_hat, r_can]
├─E_s(x_tilde) -> z_s -> G_s(z_s)=s_hat
├─E_f(x_tilde,r_can,e(s_hat)) -> z_f=[z_f_id,z_tx_state]
├─E_n(x,eta_hat) -> z_n=[z_ch,z_rx,z_sync,z_gain]
└─D(z_s,z_f,z_n)=C_zn(T_zf(G_s(z_s))) -> [mu_hat,sigma_hat]
```

统一Decoder满足：

```text
D = C_zn o T_zf o G_s
```

禁止把三个latent简单concat后交给自由MLP/卷积Decoder，禁止从输入或`E_n`到Decoder的逐采样skip connection，禁止目标波形绕过低容量nuisance bottleneck。

## 5.新增模块与接口

### 5.1配对与三轴干预

`InterventionCubeBatchBuilder`接收Phase1 batch和现有卫星增强器，输出：

```text
physical_sample_id
pair_id
view_type
label_mask
receiver_id
day_id
crop_offset
sat_meta
nuisance_pair_index
content_pair_index
fingerprint_pair_index
pair_valid_mask
```

Nuisance Pair固定发射机和内容，只改变clean/LEO、信道、接收机或同步条件。Content Pair优先在同一物理包内选择不同内容窗口，保持发射机和链路条件。Fingerprint Pair只在公共preamble或激励状态、receiver/day、视图和nuisance参数匹配时连接不同TX。不能证明配对条件时，`pair_valid_mask=false`，不得用普通随机负样本冒充严格干预。

### 5.2保守Canonicalizer

`ConservativeCanonicalizer`输出规范化IQ`~x`、nuisance参数`\hat eta`和规范化残差`r_can`。初始实现只解析处理公共CFO、公共相位、标量增益和报告允许的粗同步变量。它不得自由消除细粒度相位噪声、振荡器残差、TX IQ不平衡或PA失真。

### 5.3内容因子

`ContentSequenceEncoder`输出低采样率时序token`z_s in R^(T' x d_s)`，承载符号/局部波形、调制状态、幅度访问区、相位跳变和激励历史。`ContentGenerator`从`z_s`恢复`\hat s`，并支持masked content prediction。TX CE默认不能更新内容编码器。

### 5.4发射机响应因子

`FingerprintResponseEncoder`输出：

```text
z_f = [z_f_id,z_tx_state]
```

`z_f_id`表示长期稳定身份，`z_tx_state`表示温度、功率和慢时变PA状态。跨天约束只强制`z_f_id`稳定，不强制整个`z_f`完全相同。

`ExcitationConditionedFingerprintOperator`计算：

```text
delta_f[n] = G_f(e[n],z_f)
u_hat[n] = s_hat[n] + delta_f[n]
```

`e[n]`来自`\hat s`的幅度、相位、slew、PAPR和历史窗口。`G_f`由固定或受约束物理基与小型学习残差组成，并限制残差能量、通道数、感受野、rank、输出带宽和参数规模，不能重新生成全部内容。

### 5.5结构化nuisance因子

`StructuredNuisanceEncoder`输出：

```text
z_n = [z_ch,z_rx,z_sync,z_gain]
```

四个分量分别表示信道频响、接收机残差、CFO/Doppler/STO/SFO和AGC/公共相位/幅度尺度。其维度和带宽必须显著低于输入波形，不允许与输入长度相同的自由latent，不允许携带TX类别。

### 5.6物理顺序概率Decoder

`PhysicsOrderedDecoder`先生成`\hat s`，再施加`delta_f`，最后执行结构化信道、接收机、同步和增益变换。输出复数条件均值`\hat mu_x`和有界方差`\hat sigma_x^2`。训练使用异方差复高斯负对数似然，不要求`z_n`记忆独立随机噪声realization。

### 5.7物理特征和可辨识性门控

`FrozenFingerprintFeatureBank`由固定、可微或停止梯度的物理特征组成：IQ椭圆/非圆性、AM/AM、AM/PM、memory-polynomial residual、谱肩/带外再生、局部相位噪声PSD、幅度条件残差和循环平稳统计。

`FisherIdentifiabilityGate`依据激励覆盖、PAPR、Gram谱、effective rank、SNR和噪声地板产生分块权重。PA未被充分激励时，PA相关物理损失接近零，防止Decoder制造不存在的指纹伪迹。物理特征提取器不能与Decoder自由协同训练。

### 5.8定向指纹移植

`CounterfactualFingerprintTransplant`构造：

```text
x_tilde_(i<-j) = D(z_s_i,z_f_j,z_n_i)
```

并同时要求独立冻结身份分类器识别为目标TX`j`、`E_s`恢复`z_s_i`、`E_n`恢复`z_n_i`、`E_f`恢复`z_f_j`。同TX交换必须保持身份和波形有效；删除或平均`z_f`必须增加指纹残差误差。只增加shuffle gap不算通过。

## 6.损失函数

总损失为：

```text
L_total = L_id
        + lambda_self * L_self
        + lambda_swap * L_swap
        + lambda_share * L_shared
        + lambda_cycle * L_latent_cycle
        + lambda_eta * L_eta
        + lambda_factor * L_factor
        + lambda_need * L_need_star
        + lambda_phys * L_phys
```

- `L_id=CE+lambda_supcon*SupCon+lambda_proto*L_proto`，只作用于`z_f_id`。
- `L_self`比较同组合概率重构。
- `L_swap`执行clean到LEO和LEO到clean双向交叉重构。
- `L_shared`对clean/LEO的`z_s,z_f`进行双向stop-gradient一致性，同时加入batch variance、covariance去冗余和必要负样本，防止常数塌缩。
- `L_latent_cycle`重新编码交叉输出，分别恢复源内容、源发射机和目标nuisance代码。
- `L_eta`监督已知模拟Doppler、SNR、delay、Doppler rate、taps、SFO、STO等参数；不直接无限增大`z_n^clean`与`z_n^leo`距离。
- `L_factor`要求`z_f`高身份/低域、`z_n`高域/低身份、`z_s`高内容/低身份。默认使用cross-covariance、条件域混淆和独立probe，不完全依赖全局DANN。
- `L_need_star=L_target_id+L_preserve_s+L_preserve_n+L_same_f+L_drop_f`。
- `L_phys`包含指纹残差能量、响应平滑、参数边界和Fisher门控后的固定物理特征损失。

重构误差`\mathcal E`组合有界概率时域误差、多尺度STFT幅度、幅度门控的共轭乘积相位增量和物理特征误差。STFT低能量区域设置噪声地板；相位损失不直接对wrapped phase做L1；允许的对齐只处理粗粒度公共nuisance，不能消除细粒度TX振荡器指纹。

## 7.Phase1数据权限

- `L_s`参与全部损失，包括身份、同TX/异TX配对和定向移植。
- `U_s`只参与`L_self/L_swap/L_shared/L_latent_cycle/L_eta/L_phys`，训练张量和采样器不得读取隐藏TX真值。
- 第一版不使用硬伪标签锚定`U_s`的`z_f`。EMA Teacher只作为后续独立消融，不与核心因子化机制混写。
- `V`只用于source侧选模、阈值冻结和诊断，不反向传播或更新任何持久状态。
- clean/LEO配对使用同一物理片段和同步crop，不产生第二个Phase2观测；本方法不访问任何target query。

## 8.训练日程

默认训练预算保持200epoch，候选初始日程为：

1. E1-40：`L_id+L_self+L_eta`，建立内容、链路和基本重构；关闭强shuffle、跨TX移植和强latent对抗。
2. E41-90：线性引入`L_swap+L_shared+L_latent_cycle`。
3. E91-150：启用三轴干预和定向移植；交替更新真实组合Decoder、冻结或弱更新Decoder的`E_f`必要性路径，再联合微调。
4. E151-200：降低raw waveform reconstruction对共享主干的梯度，重点优化`L_id+SupCon+prototype+shared+transplant`。

保留项目默认LEO_WEAK日程和E80开始的卫星辅助CE。普通ADV3B02继续保持`lambda_sat_cons=0`；FCR一致性损失只在显式`phase1_method=adv3b02_fcr`时启用，不修改全局默认。

## 9.递进消融与评测

正式消融顺序固定为：

```text
R0 CE baseline
R1 +self reconstruction
R2 +swap
R3 +shared
R4 +latent-cycle
R5 +basic L_need diagnostic
R6 +targeted transplant
R7 +physics-ordered decoder
R8 +three-axis intervention
```

先运行单seed、source-only最小可证伪矩阵。训练完成的每个Phase1候选必须使用选定最终checkpoint分别评测clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`，保存checkpoint身份、配置和逐场景日志。低性能是科学负结果，不是技术停止条件。

## 10.诊断与验收

必须保存以下诊断：

- `z_f`TX分类和receiver/domain独立线性probe；
- `z_n`receiver/channel预测和TX独立线性probe；
- `z_s`内容/调制恢复和TX独立线性probe；
- clean/LEO的`z_f`距离；
- 同TX不同内容的`z_f`距离；
- 同内容不同TX的`z_s`距离；
- 删除`z_f`后的fingerprint residual error；
- 同TX交换的身份保持与波形有效性；
- 跨TX移植的目标身份、源内容和源nuisance保持；
- 已知IQ/PA参数与恢复值的单调关系；
- reconstruction NLL、MRSTFT、相位增量误差、物理特征误差；
- Gram条件数、effective rank、激励覆盖、Fisher gate和fallback率；
- 训练时间、峰值显存、参数量和单样本推理延迟。

实现正确性要求关闭态兼容、所有新增模块训练路径可达、所有CLI参数真正接入损失/日程、无标签真值不可达、`z_n`不存在高容量复制路径、checkpoint能够单条LEO IQ推理。科学晋级要求响应可辨识性诊断与clean/三种LEO弱场景身份指标在同row共同成立，不能从不同候选拼接最佳值。

## 11.目标文件

新增：

```text
code/cvsrffi/phase1_fcr_types.py
code/cvsrffi/phase1_fcr_interventions.py
code/cvsrffi/phase1_fcr_canonicalizer.py
code/cvsrffi/phase1_fcr_factors.py
code/cvsrffi/phase1_fcr_fingerprint.py
code/cvsrffi/phase1_fcr_nuisance.py
code/cvsrffi/phase1_fcr_decoder.py
code/cvsrffi/phase1_fcr_physics.py
code/cvsrffi/phase1_fcr_transplant.py
code/cvsrffi/phase1_fcr_losses.py
code/cvsrffi/phase1_fcr_schedule.py
code/cvsrffi/phase1_fcr_diagnostics.py
```

修改：

```text
code/model_dual_cvsincnet.py
code/dataset_wisig.py
code/baseline_origin_sat_view.py
code/cvsrffi/tensors.py
code/cvsrffi/checkpoint.py
code/train.py
```

测试按模块建立`code/tests/test_phase1_fcr_*.py`，正式入口单独建立`code/scripts/launch_phase1_adv3b02_fcr_*.sh`，不改写现有ADV3B02 launcher。

## 12.风险和禁止近似

最高风险是WiSig元数据能否严格构造“同内容、同链路、不同TX”的Fingerprint Pair。公共preamble、窗口定位或receiver/day匹配不足时，该项必须在追踪表标为`blocked`，不能用随机异TX配对后声称设计一致。

以下实现不算设计一致：

- 只实现三个latent和普通concat Decoder；
- 只加入`L_swap`或随机shuffle gap；
- `z_n`与输入等长或存在U-Net skip；
- 逐点拟合随机噪声；
- 用可学习`R_fp`与Decoder共同作弊；
- CLI有开关但训练循环不调用；
- 模块已实现但正式launcher不可达；
- 把ECRS-V1局部响应辨识近似写成完整FCR；
- 把source proxy或模拟LEO结果写成真实在轨、Phase2或Phase3结论。

## 13.设计自审

- 占位扫描：本文没有占位符或未决实现语句。
- 一致性：数据权限、相对链路语义、三因子接口、Decoder顺序、损失、训练日程、消融和评测相互一致。
- 范围：本规格只覆盖ADV3B02 Phase1 FCR，不扩展Phase2/Phase3。
- 严格性：目标是完整设计一致，不是ECRS近似版。不能严格构造的干预pair必须显式阻断该条声明。
