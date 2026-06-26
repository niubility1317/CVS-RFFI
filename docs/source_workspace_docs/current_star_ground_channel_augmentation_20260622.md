# 当前 CVS 星地信道增强完整讲解

生成时间：2026-06-22  
工作区：`E:\type10-7`  
依据文件：`AGENTS.md`、`项目.md`、当前本地代码、当前本地分析文档与历史索引  
适用对象：CVS-RFFI / CV-SincNet 的科研解释、实验设计、自动化矩阵、论文/汇报表述与 Stage2 部署协议说明

## 1. 一句话结论

当前 CVS 的“星地信道增强”不是简单随机加噪，也不是已经完成真实在轨验证。它是一个物理启发的 satellite/LEO 信道视图系统，用来把 WiSig/ManySig 这类地面可接入代理数据转换成星地链路压力视图，并在训练、评估、联邦、本地部署少样本适配四个层面使用。

更准确地说，它现在由五个层次组成：

| 层次 | 当前作用 | 关键文件 |
|---|---|---|
| 科研协议层 | 定义 satellite/LEO view 是 deployment-primary，clean view 只是 control/reference | `项目.md` |
| 物理信道层 | 对 raw IQ 施加轨道、仰角、路径损耗、天气、CFO/Doppler、相位噪声、多径、AWGN、IQ imbalance 等扰动 | `code/sat_channel.py`、`code/training_controls.py` |
| 训练视图层 | 把 clean 样本和 satellite 样本组织成辅助 CE、clean-sat consistency、拼接 2B batch 或 CE-only satellite view | `code/train.py`、`code/baseline_origin_sat_view.py`、`code/concat_sat_channel_aug.py` |
| 评估层 | 默认按五类 satellite scenario 对主要 OOD split 做星地压力测试 | `code/cvsrffi/eval.py`、`code/train.py` |
| Stage2 部署层 | 在 target receiver domain 上把 target-old、target-new、unknown 的 support/query 都放在 satellite/LEO target view 下解释 | `code/cvsrffi/spaceborne_fewshot.py`、`code/eval_spaceborne_fewshot.py`、`tools/spaceborne_fewshot_da_matrix.py` |

当前最稳妥的表述是：

> CVS 使用物理启发的星地信道增强与压力测试来模拟部署视图，在地面训练中作为受控 satellite view，在 Stage2 中作为 target receiver domain 的 deployment-primary view。该机制可以提升部分星地鲁棒性，但仍存在 clean strict UDU 与 satellite robustness 的 tradeoff；因此不能把 clean 成绩或 satellite augmentation 成绩写成真实在轨验证。

## 2. 协议边界

`项目.md` 明确 CVS 的主场景是：

```text
天基射频指纹识别中的弱标注跨接收机域泛化与在轨跨域少样本适应
```

这个场景里，星地信道增强的地位由三条规则决定。

第一，星地信道是破坏 TX 指纹的部署扰动，不是普通数据增强装饰。项目抽象为：

```text
x = R_d( H_d * T_y(s) ) + n
```

其中：

- `T_y` 是发射机硬件非理想性，是应该保留的身份来源；
- `H_d` 是传播/星地信道，是域扰动来源；
- `R_d` 是接收机链路响应，是跨接收机偏移来源；
- `n` 是噪声。

CVS 要学的是稳定的 TX 身份表征，而不是把 `H_d` 或 `R_d` 当成身份捷径。

第二，WiSig/ManySig 不是卫星真实数据。它们是 terrestrial proxy benchmark / ground-accessible source domain family。当前 satellite-channel augmentation 和 satellite stress 是 physics-informed deployment stress，不能写成真实在轨 IQ 验证。

第三，clean view 只能作为对照。项目文件要求 deployment support/query 必须按 satellite/LEO 视图报告，推荐 target channel view：

```text
clear_leo
low_elev_leo
rain_leo
storm_mp
mixed_orbit
```

因此任何报告或论文都不能把 clean view 成功直接提升为 satellite/LEO deployment success。

## 3. 当前信道模拟器做了什么

核心实现是 `code/sat_channel.py` 的 `apply_sat_gnd_channel_batch()`。输入是形状 `[B, 2, T]` 的 IQ tensor，其中第 2 维是 I/Q 两路；输出仍是 `[B, 2, T]`，可以直接送入现有 CV-SincNet / CVS 主干。

### 3.1 物理常量和基础函数

信道模拟器显式使用：

```text
C = 299792458.0              # 光速
MU_EARTH = 3.986004418e14    # 地球引力常数
R_EARTH = 6371000.0          # 地球半径
```

它实现了：

- `fspl_db(d_m, fc_hz)`：自由空间路径损耗；
- `slant_range_from_elevation(theta_deg, h_m)`：由仰角和轨道高度计算斜距；
- `orbital_speed_circular(h_m)`：圆轨道速度；
- `steady_probs_from_elevation(theta_deg)`：按仰角生成 LOS / LOO / Rayleigh 状态概率；
- `complex_awgn_like()`：按样本功率和 SNR 添加复高斯噪声；
- `wiener_phase_noise()`：Wiener 过程相位噪声；
- `apply_iq_imbalance()`：I/Q 幅相不平衡；
- `apply_mild_agc()`：轻量 AGC，把 RMS 拉回目标幅度并保留小残差。

这说明当前星地增强不是“只加 AWGN”。它覆盖传播、轨道、接收链路、频偏、相位、噪声和 I/Q 失衡多个扰动源。

### 3.2 单个 batch 的处理顺序

一次 satellite channel forward 的实际顺序可以概括为：

1. 把 `[I, Q]` 组合成 complex IQ。
2. 按场景采样轨道类型：LEO / MEO / GEO。
3. 按轨道类型采样高度 `h_km`。
4. 采样仰角 `theta_deg`。
5. 由高度和仰角计算 slant range，再计算路径损耗。
6. 根据仰角采样链路状态：LOS、LOO、Rayleigh。
7. 根据天气表采样大气复衰落。
8. 根据链路状态生成主信道系数：
   - LOS 使用 Rician-like 结构；
   - Rayleigh 使用复高斯散射；
   - LOO 使用 LOO 表中的 lognormal-like 阴影/遮挡结构。
9. 由轨道速度和仰角计算 Doppler，再叠加随机 CFO。
10. 对 IQ 施加频率旋转。
11. 施加 Wiener 相位噪声。
12. 如果场景启用 multipath，则构造多个 delay tap；否则使用 flat fading。
13. 叠加路径损耗和大气衰落。
14. 施加 mild AGC。
15. 按 SNR 施加复 AWGN。
16. 施加 IQ amplitude/phase imbalance。
17. 输出回 `[B, 2, T]`。

### 3.3 为什么幅度扰动不是主信号

`sat_channel.py` 确实计算了自由空间路径损耗和大气衰落，但后面又有 mild AGC，常见训练/评估管线也会有 IQ 归一化。因此路径损耗这种“强幅度差异”不会直接变成模型可轻松利用的特征。

当前真正更容易残留到模型输入中的扰动是：

- Doppler / CFO 引起的频率旋转；
- 相位噪声；
- 低 SNR 下的噪声结构；
- IQ imbalance；
- 多径延迟；
- LOS / LOO / Rayleigh 状态变化；
- 天气引起的统计变化。

这正是星地增强效果不总是显著的原因之一：它主要增加 nuisance stress，不直接增加 TX identity 信息。

### 3.4 当前完整星地信道建模如何实现

当前实现可以理解为一个可插拔的 batch-level channel operator：

```text
H_sat_scenario:
  x_iq [B, 2, T] -> x_sat [B, 2, T]
```

它不改变标签、不改变 batch 中样本顺序，也不引入新的 TX identity。它只把 clean IQ 变成某个 satellite scenario 下的受扰动视图。代码入口是：

```text
code/cvsrffi/eval.py
  make_sat_config(scenario, args)
  apply_sat_channel_for_scenario(x, scenario, args)

code/sat_channel.py
  apply_sat_gnd_channel_batch(x_iq, cfg, gen, return_meta)
```

`make_sat_config()` 先从 `training_controls.py` 读取 scenario 参数，再覆盖运行参数中的采样率和载频：

```text
fs_hz = --sat_fs_hz, default 25e6
fc_hz = --sat_fc_hz, default 2.462e9
```

然后 `apply_sat_gnd_channel_batch()` 在 `torch.no_grad()` 下对 IQ 做物理启发变换。这意味着信道增强本身不是可学习模块，训练时不会反向更新信道参数；模型学习的是如何在这些扰动下保持 TX 判别。

### 3.5 张量接口和复数 IQ 表示

输入张量约定为：

```text
x_iq.shape = [B, 2, T]
x_iq[:, 0, :] = I
x_iq[:, 1, :] = Q
```

函数内部先组合为复数基带：

```text
x_c[n] = I[n] + j Q[n]
```

后续所有传播、频偏、相位噪声、多径和 IQ imbalance 都在 complex IQ 上完成。最后再拆回：

```text
y_iq[:, 0, :] = real(y_c)
y_iq[:, 1, :] = imag(y_c)
```

因此它可以直接接入现有 CV-SincNet 输入，不需要改 dataloader、模型第一层或标签结构。

### 3.6 scenario 参数如何控制信道分布

每个 satellite scenario 本质上是一组采样分布，而不是一个固定通道。以 `mixed_orbit` 为例，它定义：

```text
weather = cloudy
scenario = urban
loo_level = mid
orbit_probs = {LEO: 0.6, MEO: 0.3, GEO: 0.1}
theta_deg = 10-90
snr_db = 12-30
cfo_std_hz = 300
phase_noise_inc_std = 0-3e-3
enable_multipath = true
num_taps = 2-4
max_delay_samp = 5
pwr_decay = 0.75
```

一次 batch forward 中，每个样本会独立采样轨道、仰角、链路状态、SNR、CFO、相位噪声、AGC 残差和 IQ imbalance。因此同一个 scenario 表示一个信道族，而不是一个确定性滤波器。

### 3.7 轨道、高度、仰角、斜距和自由空间损耗

第一步是采样轨道类型：

```text
orbit ~ Categorical(orbit_probs)
```

轨道类型决定高度范围：

```text
LEO: 500-2000 km
MEO: 8000-20000 km
GEO: 35786 km
```

随后从 scenario 的仰角区间采样：

```text
theta ~ Uniform(theta_min, theta_max)
```

代码使用地球半径 `R_EARTH` 和卫星轨道半径 `R_EARTH + h` 计算斜距：

```text
rho = -R_e sin(theta)
      + sqrt((R_e + h)^2 - (R_e cos(theta))^2)
```

再计算自由空间路径损耗：

```text
lambda = C / fc
FSPL(dB) = 20 log10(4 pi rho / lambda)
```

实现中不是把绝对 FSPL 直接压到极小幅度，而是相对一个参考链路计算增益：

```text
reference: theta = 60 deg, h = 1000 km
g_pl = 10 ^ (-(FSPL - FSPL_ref) / 20)
```

这个设计很关键。真实星地链路绝对路径损耗很大，但数据管线和接收端通常存在增益控制/归一化。当前模拟器保留的是不同仰角和轨道导致的相对幅度变化，而不是让所有样本因绝对路径损耗数值崩掉。

### 3.8 链路状态：LOS、LOO、Rayleigh

第二步是根据仰角采样链路状态：

```text
state in {LOS, LOO, Rayleigh}
```

代码使用 `steady_probs_from_elevation(theta, scenario)` 计算概率。高仰角时 LOS 概率更高；低仰角时非 LOS 成分增加。对当前 `urban` 场景：

```text
w_los = 1 - ((90 - theta)^2) / 7000
w_loo = (1 - w_los) * 1/5
w_ray = (1 - w_los) * 4/5
```

三类状态分别对应不同主信道系数 `h0`。

LOS 状态使用 Rician-like 系数：

```text
h0 = sqrt(K / (K + 1)) * exp(j phi0)
     + sqrt(1 / (K + 1)) * CN(0, 1)
```

其中 `K_db` 会随仰角从 `K_db_range` 内插。仰角越高，LOS 分量越强。雨天或风暴场景会把 `K_db` 下调 3 dB，表示直达径可靠性下降。

Rayleigh 状态使用纯散射：

```text
h0 = CN(0, 1)
```

LOO 状态使用 `LOO_TABLE` 中的遮挡参数，构造一个 lognormal-like 直达项加散射项：

```text
z = exp(mu + sqrt(d0) * N(0, 1))
h0 = z * exp(j phi0) + sqrt(b0) * CN(0, 1)
```

`loo_level` 有 `light`、`mid`、`severe`。`storm_mp` 使用 `severe`，所以它比普通 rain/clear 场景更容易产生强遮挡和深衰落。

### 3.9 天气和大气复衰落

天气由 `ATM_TABLE` 给出：

```text
clear, cloudy, storm, rain
```

每种天气都有幅度均值/方差和相位均值/方差：

```text
r_a   ~ Normal(mu_a, sqrt(sigma2_a))
phi_a ~ Normal(m_a, sqrt(eta2_a))
a_atm = r_a * exp(j phi_a)
```

其中 `rain` 和 `storm` 的幅度方差更大，`rain` 的相位方差也显著更大。这使雨衰不只是降低 SNR，而是同时改变幅度统计和相位统计。

### 3.10 Doppler、随机 CFO 和相位噪声

轨道速度按圆轨道近似：

```text
v = sqrt(MU_EARTH / (R_EARTH + h))
```

径向速度用仰角近似：

```text
v_r = +/- v cos(theta)
```

于是 Doppler 为：

```text
f_D = (v_r / C) * fc
```

代码再叠加设备/同步残差形式的随机 CFO：

```text
cfo ~ Normal(0, cfo_std_hz)
f_off = f_D + cfo
```

对第 `n` 个采样点施加频率旋转：

```text
freq_rot[n] = exp(j 2 pi f_off n / fs)
```

相位噪声使用 Wiener process。先从 scenario 的范围内为每个样本采样增量标准差：

```text
sigma_phi ~ Uniform(phase_noise_min, phase_noise_max)
```

再累加高斯增量：

```text
phi[n] = sum_{k<=n} Normal(0, sigma_phi)
phase_noise[n] = exp(j phi[n])
```

这两项共同模拟星地链路中的高速相对运动、振荡器/同步误差和短 IQ 片段中的相位漂移。

### 3.11 多径实现

如果 `enable_multipath=false`，信道是 flat fading：

```text
y_c = h0 * x_c
```

如果 `enable_multipath=true`，代码会采样 tap 数：

```text
L ~ Integer[num_taps_min, num_taps_max]
```

再采样每个 tap 的延迟：

```text
delay_k ~ Integer[0, max_delay_samp]
delay_0 = 0
```

tap 功率按指数衰减：

```text
p_k = pwr_decay^k
p_k = p_k / sum_k p_k
```

第一个 tap 使用前面得到的主信道 `h0`，后续 tap 使用复高斯散射。实现上对 `x_c` 做 `torch.roll`，并把 roll 后前端越界部分置 0：

```text
y_c = sum_k tap_k * delay(x_c, delay_k)
```

当前多径是短片段离散 delay-tap 模型，适合 WiSig 这类短 IQ snippet。它不是完整几何射线追踪，也不包含地形/建筑物地图。

### 3.12 总信道组合顺序

当前实现的总体组合可以写成：

```text
x_c = I + jQ

y_c = flat_or_multipath_channel(x_c, h0, taps)
y_c = g_pl * a_atm * y_c
y_c = y_c * exp(j 2 pi f_off n / fs)
y_c = y_c * exp(j phi_wiener[n])
y_c = mild_agc(y_c)
y_c = awgn(y_c, snr_db)
y_c = iq_imbalance(y_c)

x_sat = [real(y_c), imag(y_c)]
```

更紧凑地说：

```text
x_sat = IQImbalance(
          AWGN(
            AGC(
              g_pl * a_atm * Multipath_h(x)
              * DopplerCFO(f_D + cfo)
              * PhaseNoise(phi)
            )
          )
        )
```

这里的 `Multipath_h(x)` 在无多径场景下退化为 `h0 * x`。

### 3.13 AGC、AWGN 和 IQ imbalance

路径损耗、天气衰落和多径完成后，代码先做 mild AGC：

```text
rms = sqrt(mean(|y_c|^2))
y_c = y_c * target_rms / rms
y_c = y_c * residual_gain
```

其中 `target_rms=1.0`，`residual_gain` 由 `agc_resid_db` 采样，默认范围为 `-1` 到 `1` dB。这样可以模拟接收机增益控制后仍存在的小幅残差。

随后按采样 SNR 加复 AWGN：

```text
signal_power = mean(|y_c|^2)
noise_var = signal_power / 10^(snr_db/10)
w ~ CN(0, noise_var)
y_c = y_c + w
```

最后施加 IQ imbalance。代码使用 widely-linear 形式：

```text
y_out = alpha * y_c + beta * conj(y_c)
```

其中 `alpha` 和 `beta` 由 I/Q 幅度不平衡 `amp_db` 与相位不平衡 `phase_deg` 决定：

```text
amp_db ~ Uniform(iq_amp_db_min, iq_amp_db_max)
phase_deg ~ Uniform(iq_phase_deg_min, iq_phase_deg_max)
```

这个步骤模拟接收链路中的 I/Q 分支幅相不匹配。它对 RFFI 很敏感，因为 TX 指纹和接收机 I/Q 非理想性在频谱镜像、相位结构和短时波形中容易纠缠。

### 3.14 随机性、可复现性和元数据

信道变换接受 `torch.Generator`。评估路径中，`evaluate_sat_scenarios()` 会按 scenario index 和 loader index 构造确定性 seed：

```text
seed = sat_seed + scenario_index * 1009 + loader_index * 97
```

训练路径中的 `BaselineOriginSatViewAugment` 用：

```text
seed = sat_view_seed + epoch * 1009 + batch_idx
```

因此同一配置、同一 epoch/batch、同一 seed 下，satellite view 是可复现实验对象。

`apply_sat_gnd_channel_batch(..., return_meta=True)` 还可以返回采样元数据：

```text
orbit
h_km
theta_deg
d_km
state
pl_db
fD_hz
cfo_hz
snr_db
K_db
```

这些字段目前主要用于诊断和审计。若后续要证明某个 satellite curriculum 的强度变化，应该把这些 meta 聚合写入 report，而不是只写 scenario name。

### 3.15 当前实现的边界

当前建模已经覆盖 RFFI 训练最相关的星地 nuisance：

```text
orbit / elevation / slant range / FSPL
LOS-LOO-Rayleigh state
weather-dependent complex atmospheric fading
Doppler + random CFO
Wiener phase noise
short-tap multipath
mild AGC
AWGN
IQ amplitude/phase imbalance
```

但它仍不是完整通信系统仿真。当前代码没有实现：

- TLE/SGP4 真实轨道过境轨迹；
- 地面站经纬度、卫星星历和时间连续 pass；
- 天线方向图、极化、指向误差；
- 地形、建筑物、雨区实况或 ITU-R 逐时气象场；
- 调制/同步/解调链路；
- 真实星上接收机 AGC/PLL/ADC 非线性；
- 实测卫星 IQ 数据校准。

所以当前应称为“物理启发的星地信道建模与部署压力视图”，而不是“真实星地链路数字孪生”或“在轨链路复现”。它的科研价值在于把 clean terrestrial proxy IQ 系统性地映射到 satellite/LEO stress view，让模型和 Stage2 协议接受受控部署扰动检验。

## 4. 当前内置 satellite scenario

`code/training_controls.py` 定义了当前可用的 satellite scenarios。

| scenario | 物理含义 | 轨道/天气 | 仰角 | SNR | CFO std | 相位噪声 | 多径 |
|---|---|---|---:|---:|---:|---:|---|
| `clear_leo` | 高仰角、清晰 LEO 对照 | LEO=1.0, clear | 30-90 deg | 20-30 dB | 200 Hz | 0-2e-3 | 否 |
| `low_elev_leo` | 低仰角 LEO，斜距更长、频相扰动更强 | LEO=1.0, clear | 10-30 deg | 15-28 dB | 350 Hz | 5e-4-3e-3 | 否 |
| `rain_leo` | 雨衰 LEO | LEO=1.0, rain | 20-80 deg | 10-25 dB | 250 Hz | 5e-4-3e-3 | 否 |
| `storm_mp` | 风暴/遮挡/多径压力场景 | LEO=0.8, MEO=0.2, storm, severe LOO | 10-35 deg | 8-20 dB | 400 Hz | 1e-3-4e-3 | 2-5 taps |
| `geo_clear` | 清晰 GEO 对照 | GEO=1.0, clear | 25-80 deg | 18-30 dB | 100 Hz | 0-1.5e-3 | 否 |
| `mixed_orbit` | 默认混合轨道压力视图 | LEO=0.6, MEO=0.3, GEO=0.1, cloudy | 10-90 deg | 12-30 dB | 300 Hz | 0-3e-3 | 2-4 taps |

项目协议推荐主报告视图是：

```text
clear_leo, low_elev_leo, rain_leo, storm_mp, mixed_orbit
```

`geo_clear` 在代码中存在，但不属于 `项目.md` 推荐主视图集合；如果使用，应明确写成 control 或 sensitivity。

## 5. 评估路径

### 5.1 默认评估是打开的

在 `code/train.py` 中，`--eval_sat_channel` 默认启用。默认评估场景为：

```text
clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
```

默认评估 split 是：

```text
test_unseen_day_seen_rx
test_seen_day_unseen_rx
test_unseen_day_unseen_rx
```

其中 `test_unseen_day_unseen_rx` 通常作为 strict UDU 的核心 split。

### 5.2 评估调用栈

评估调用栈是：

```text
train.py
  -> evaluate_sat_scenarios()
     -> evaluate_loader_sat_channel()
        -> apply_sat_channel_for_scenario()
           -> make_sat_config()
              -> sat_channel_config_for_scenario()
           -> apply_sat_gnd_channel_batch()
```

`evaluate_sat_scenarios()` 会对每个 scenario、每个 selected named loader 都施加 satellite transform，再计算 TX accuracy，并输出：

- 每个 scenario 的 aggregate；
- 每个 scenario 的 `strict_udu`；
- 每个 selected split 的 named stats；
- `selected_names`。

因此 satellite evaluation 是真正把测试 IQ 变成 satellite-stressed IQ 后再跑模型，不是只换标签或只改日志。

### 5.3 指标解释

当前评估里常见几类值：

- `tx_acc`：卫星扰动后的发射机分类准确率；
- `strict_udu`：通常取 `test_unseen_day_unseen_rx` 在该 scenario 下的 TX accuracy；
- `aggregate`：主 OOD splits 的聚合准确率；
- `sat_mean` / `sat avg`：多个 scenario 的平均；
- `sat min` / `sat worst`：最坏 scenario 指标。

报告时必须同时列 clean strict UDU 和 satellite strict/mean/worst。只给 satellite 平均值会掩盖 storm / low elevation 的风险；只给 clean strict UDU 又不能说明星地部署鲁棒性。

## 6. 训练路径一：CVS satellite consistency / auxiliary satellite loss

第一类训练路径由 `--use_sat_consistency` 控制。

### 6.1 基本机制

在 centralized training loop 中，如果满足：

```text
use_sat_consistency = true
concat_sat_aug is None
epoch >= sat_cons_start_epoch
lambda_sat_cls > 0 or lambda_sat_cons > 0
```

训练循环会：

1. 从 `sat_train_scenario_list` 中按 epoch/batch 选择 scenario；
2. 对 clean batch `x` 施加 `apply_sat_channel_for_scenario()`；
3. 得到 `x_sat_train`；
4. 对 `x_sat_train` 做一次 forward；
5. 计算 satellite TX CE：

```text
loss_sat_cls = CE(model(x_sat).tx_logits, y)
```

6. 计算 clean/sat 的 `z_id` cosine consistency：

```text
loss_sat_cons = cosine_consistency_loss(z_id_sat, stopgrad(z_id_clean))
```

7. 加入总损失：

```text
loss += lambda_sat_cls * loss_sat_cls
loss += lambda_sat_cons * loss_sat_cons
```

### 6.2 典型默认语义

通用 parser 中：

```text
use_sat_consistency = false
sat_train_scenario = mixed_orbit
lambda_sat_cls = 0
lambda_sat_cons = 0
sat_cons_start_epoch = 1
```

也就是说，普通手动训练不会自动启用 satellite training；只有显式打开开关或使用特定 profile / launcher 时才进入训练增强。

在一些 CVS/FedCVS profile 中，代码会自动设置类似：

```text
use_sat_consistency = true
sat_train_scenario = mixed_orbit
sat_cons_start_epoch = 20
lambda_sat_cls = 0.10
lambda_sat_cons = 0.00
fl_sat_aug_mode = cvs_consistency
```

这个路径的核心是“后期弱 satellite CE”，而不是强 clean-sat 不变性。因为 `lambda_sat_cons` 常见为 0，很多历史 run 实际上主要是 satellite CE 辅助项。

### 6.3 优点和问题

优点：

- 不改变主 clean batch 的完整 CVS 损失；
- satellite view 不会直接混入主 batch 的 domain labels；
- 可以晚启动，降低早期表征被强扰动破坏的风险。

问题：

- 如果 `lambda_sat_cons=0`，它只是告诉模型 satellite view 也要分对 TX，并没有强制 `z_id(clean)` 和 `z_id(sat)` 对齐；
- 如果只用 `mixed_orbit`，覆盖比 all-five 小；
- 如果权重太大，仍可能牺牲 clean strict UDU；
- 如果启动太早，模型还没学稳 TX identity，就先学强 nuisance invariance，容易不稳定。

## 7. 训练路径二：拼接星地信道增强

第二类训练路径由 `--use_concat_sat_channel_aug` 控制。当前代码把它命名和日志写成：

```text
拼接星地信道增强
```

实现文件：

- `code/baseline_origin_sat_view.py`
- `code/concat_sat_channel_aug.py`
- `code/train.py`

### 7.1 BaselineOriginSatViewAugment

`BaselineOriginSatViewAugment` 有两个 API：

```text
transform(x) -> satellite view only
expand(x, y, d_raw) -> clean+sat concatenated batch
```

它支持：

- scenario 列表；
- `sat_view_prob` 概率；
- `sat_view_seed` 可复现实验；
- `sat_view_schedule` 分阶段 schedule；
- scenario token repeat，例如 `mixed_orbit*2`；
- stage start epoch。

示例 schedule：

```text
1@1.0:mixed_orbit;
61@0.75:mixed_orbit*2,low_elev_leo,rain_leo;
121:mixed_orbit,rain_leo,storm_mp
```

含义：

- 第 1 轮开始只用 `mixed_orbit`，概率 1.0；
- 第 61 轮开始以 0.75 概率使用 `mixed_orbit,mixed_orbit,low_elev_leo,rain_leo`；
- 第 121 轮开始使用 `mixed_orbit,rain_leo,storm_mp`，概率沿用默认值。

### 7.2 full concat：2B 主 batch

如果启用：

```bash
--use_concat_sat_channel_aug
```

但不启用：

```bash
--concat_sat_ce_only
```

则训练会执行：

```text
x_cat = concat([x_clean, x_sat])
y_cat = concat([y, y])
d_cat = concat([d_raw, d_raw])
```

然后把 2B batch 送入完整 CVS 主损失，包括：

- TX CE；
- domain CE；
- GRL / adversarial domain；
- group CE / GroupDRO；
- Fishr；
- MixStyle 相关路径；
- prototype / SupCon 等主损失。

这个路径最接近“baseline-style clean+sat 监督扩张”，但对 CVS 来说有明显风险：`d_raw` 被复制给 satellite view，而 satellite view 已经叠加了新传播风格。这样 domain head / GRL / Fishr 看到的是“同一个 receiver/day label 下混入 satellite style”，可能污染域解耦目标。

因此 full concat 适合做 baseline 对照，不适合作为默认最稳主线。

### 7.3 CE-only concat：主路径 clean，satellite 只做 TX CE

如果启用：

```bash
--use_concat_sat_channel_aug
--concat_sat_ce_only
```

训练会保持：

```text
clean x -> 完整 CVS 主损失
sat x   -> 单独 forward，只加 TX CE
```

总损失中 satellite 只贡献：

```text
concat_sat_ce_weight * CE(model(x_sat).tx_logits, y)
```

这是当前更干净的“拼接星地信道增强”解释，因为：

- clean 样本继续负责 `z_id/z_dom`、GRL、domain、Fishr 等完整目标；
- satellite 样本只告诉模型“同一个 TX 在星地扰动下仍应分类正确”；
- satellite style 不被误当作原始 receiver/day 域标签；
- 更适合与 domain DSQ、RCN stats、receiver-agnostic BEX02 结合。

### 7.4 当前关键实验语义

`code/scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh` 把几类路径放在同一个对比队列：

| 分支 | 语义 |
|---|---|
| `SA01_cvs_loss_mixed` | 当前 CVS satellite loss，mixed_orbit，晚期弱 satellite CE |
| `SA02_concat_sat_mixed` | full concat，mixed_orbit |
| `SA03_cvs_loss_all5` | 当前 CVS satellite loss，五场景循环 |
| `SA04_concat_sat_all5` | full concat，五场景循环 |
| `SA09_concat_sat_ceonly_mixed` | CE-only concat，mixed_orbit |
| `SA10_concat_sat_ceonly_all5` | CE-only concat，五场景循环 |
| `SA11` onward | CE-only all5 作为 anchor，再叠加 backbone/stability 消融 |

当前更值得在报告中解释的路线是 CE-only concat，而不是早期 full concat。

## 8. 训练路径三：联邦训练中的 satellite view

联邦训练由 `code/federated/fed_trainer.py` 执行。星地增强在联邦里有额外语义，因为每个 client 本身就是 receiver 或 receiver_day 粒度。

### 8.1 联邦 satellite evaluation 是强制主路径

`train.py` 中 `enforce_federated_sat_eval_args()` 要求联邦训练不能关闭 satellite-channel evaluation。也就是说 `fedavg/fedprox` 类训练必须在每轮或计划评估中保留 satellite eval。

默认联邦 satellite eval split 被扩成：

```text
test_unseen_day_seen_rx
test_seen_day_unseen_rx
test_unseen_day_unseen_rx
```

默认 scenario 是 all-five：

```text
clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
```

### 8.2 `fl_sat_aug_mode`

联邦路径的关键参数是：

```bash
--fl_sat_aug_mode baseline_view
```

或：

```bash
--fl_sat_aug_mode cvs_consistency
```

`baseline_view` 会使用 `BaselineOriginSatViewAugment` 生成 satellite view；`cvs_consistency` 更接近 centralized 的 satellite auxiliary loss。

### 8.3 `fl_baseline_view_ce_only`

联邦中最重要的安全开关是：

```bash
--fl_baseline_view_ce_only
```

如果它为 false，baseline satellite view 可能被拼进本地完整目标，类似 centralized full concat。风险同样存在：satellite view 会影响本地 DG、domain、Fishr、GRL 等损失。

如果它为 true，则 satellite view 只贡献：

```text
fl_baseline_view_ce_weight * satellite TX CE
```

这个语义最接近 centralized `SA16` 的成功因素：

```text
clean 主路径保持完整 CVS/DG；
satellite view 只作为同 TX 监督；
domain backbone 使用 DSQ；
client = receiver；
domain label = rx_day；
train ratio = 0.1；
rounds = 200。
```

### 8.4 当前联邦经验边界

历史分析里，`FSDG49` 是稳定联邦锚点：

```text
receiver-client FedProx
receiver-agnostic BEX02
CVS sat consistency
best strict UDU 约 76.295
final strict UDU 约 75.917
```

旧 `FSDG50` 使用 baseline satellite view，但 final strict UDU 约 70.517，弱于 `FSDG49`。这说明“baseline view 粗暴搬进完整联邦本地目标”不稳。

后续更合理的候选是类似：

```text
FL82_16_fedprox_rx_ra_bex02_baselineview_ceonly_domain_dsq_r010
```

其思想是复制 centralized `SA16` 的更硬成功因素：

```text
known physical satellite view
+ clean/sat supervised CE-only view
+ domain backbone DSQ
+ receiver-client FedProx
+ rx_day domain labels inside each receiver client
```

## 9. 模型结构如何承接星地增强

当前 CVS 模型不是单一路径分类器。`code/model_dual_cvsincnet.py` 中的核心语义是：

```text
raw IQ -> id_backbone  -> z_id  -> tx_logits
       -> dom_backbone -> z_dom -> dom_logits
```

### 9.1 `z_id`

`z_id` 是身份表征，负责保留 TX hardware fingerprint。星地增强对 `z_id` 的理想作用是：

```text
同一 TX:
  z_id(clean)
  z_id(clear_leo)
  z_id(low_elev_leo)
  z_id(rain_leo)
  z_id(storm_mp)
  z_id(mixed_orbit)
应该尽量保持身份一致
```

但这只能在不抹掉 TX 指纹的前提下做。过强的 satellite CE、过强 consistency 或过早启用，可能把硬件指纹也当成 nuisance 洗掉。

### 9.2 `z_dom`

`z_dom` 是域表征，应该吸收 receiver、day、rx_day、channel、satellite-style nuisance。当前 `DomainFeatureEnhancer` 可以把 RCN statistics 融入 `z_dom`，作为 receiver/channel/noise sidecar。

这也是为什么专家特征或统计特征不应替代主分类器。它们更适合做：

- gate；
- diagnostic sidecar；
- domain branch 增强；
- adapter/gate 的输入；
- leakage audit。

### 9.3 GRL 与泄漏控制

模型中有 `adv_head(grad_reverse(z_id))`，用于抑制 `z_id` 中的域信息泄漏。星地增强加入后，正确目标不是“让模型不知道任何变化”，而是：

```text
z_id 不携带可直接预测 receiver/day/satellite style 的捷径；
z_dom/sidecar 可以记录这些 nuisance；
最终 TX decision 主要依赖稳定硬件指纹。
```

这也是 full concat 污染风险的来源：如果 satellite view 使用原始 `d_raw`，域分支可能把星地扰动错误纳入 receiver/day 域结构。

## 10. Stage2 部署协议中的星地视图

Stage2 不是普通训练增强，而是在 target receiver domain `R_t` 上做在轨部署代理评估和少样本适配。

### 10.1 target receiver domain

`项目.md` 要求：

```text
intersection(R_t, R_s) = empty
|R_t| >= 1
```

也就是说目标 receiver domain 必须和地面训练 receiver 域不相交。`R_t` 可以是一个 receiver，也可以是多个 receiver 构成的 deployment proxy domain。

### 10.2 target-old / target-new / unknown

Stage2 必须区分：

```text
Y_old     = 地面训练已知 TX
Y_new     = 地面训练未见、部署时可少样本注册的 seen-new TX
Y_unknown = 地面训练未见、也不能作为 seen-new support 的未知 TX
```

所有 target-old、target-new、unknown 的 support/query 都必须来自 `R_t`，且处于相同定义的 satellite/LEO target view 下。

### 10.3 Stage2-A/B/C 与星地视图

| 阶段 | 支持集 | 查询集 | 可声明内容 |
|---|---|---|---|
| Stage2-A | empty | target-old query + non-old/unknown query | old-class target recognition、non-old rejection、unknown FAR/FPR95/AUROC |
| Stage2-B | target-old K-shot support | target-old query + unknown query | old-class calibration、old_acc_delta、unknown FAR 不恶化 |
| Stage2-C | target-old K-shot + seen-new K-shot support | old query + seen-new query + unknown query | old performance、seen-new accuracy、H_old_new、unknown FAR |

关键边界：

- Stage2-A/B 不能声明 seen-new identity accuracy；
- unknown query 不能用于阈值拟合；
- clean target support 不能被写成 satellite deployment support；
- 缺 target-old 或 target-new 覆盖时不能声明完整 Stage2-C。

### 10.4 Stage2 中的 OA-MSE / adapter / gate

`code/cvsrffi/spaceborne_fewshot.py` 中已经存在：

- `OrbitAdaptiveMSEHead`：Orbit-Adaptive Masked Subspace Energy head；
- `fit_weibull_tail()`：EVT/Weibull tail，用于开集/拒识阈值；
- `LowRankTargetAdapter`：低秩 feature-level target adapter；
- `SiameseAnchorVerifier`：anchor verifier；
- open-set result 中的 energy、subspace residual、Mahalanobis、OpenMax 等 score；
- new-class lifecycle：quarantine、active_local、ground_confirmed 等状态。

这些不是简单“星地增强变换”，而是部署后对 satellite/LEO target view 的轻量适配、拒识和安全更新机制。它们的正确位置在 Stage2-B/C 或 onboard adaptation bundle 中，不应和 Phase1 source-only DG 混写。

## 11. 当前效果证据

当前证据支持一个谨慎结论：

> 星地信道增强确实生效，但主要表现为弱到中等强度的鲁棒正则项；它尚未稳定解决 clean strict UDU 与 satellite robustness 同时上升的问题。

### 11.1 centralized CE-only + backbone 稳定性结果

`code/analysis/satellite_channel_augmentation_effect_analysis_20260527.md` 记录了完成结果：

| branch | best primary | strict UDU | final-primary SAT avg/min | 解释 |
|---|---:|---:|---:|---|
| `SA11` anchor | 82.97 | 80.73 | 43.49 / 38.62 | CE-only 星地增强 anchor |
| `SA16` domain DSQ | 84.45 | 82.78 | 43.66 / 39.56 | clean/UDU 最强，卫星小幅改善 |
| `SA14` ID phase+DSQ | 82.58 | 79.80 | 47.17 / 40.99 | 卫星更强，但 clean/UDU 下滑 |
| `SA17` all phase+DSQ | 82.42 | 79.80 | 46.55 / 40.67 | 鲁棒分支，不适合作默认主线 |

解读：

- `SA16` 更像 clean/UDU 主线；
- `SA14/SA17` 更像 satellite robustness 专用分支；
- 卫星鲁棒性提升和 clean strict UDU 之间存在 tradeoff。

### 11.2 DSQ follow-up 结果边界

同一分析文档还记录：

| branch | latest parsed | best primary | strict UDU at best primary | latest/curve-best SAT avg | 解释 |
|---|---:|---:|---:|---:|---|
| `SA18` domain DSQ ch2 | E109/170 | 85.39 | 83.86 | 41.93 / 43.57 | clean 更强，卫星没有同步升高 |
| `SA20` domain phase+DSQ | E110/170 | 83.70 | 81.78 | 44.39 / 44.85 | 卫星稍强，primary 低于 SA16 |
| `SA23` DSQ CE weight 1.5 | E102/170 | 83.24 | 81.30 | 43.01 / 45.13 | 加大卫星 CE 有鲁棒收益，但伤主指标 |
| `SA24` ID phase+DSQ CE 0.7 | E130/170 | 83.00 | 80.51 | 45.35 / 46.29 | 鲁棒路线，主指标不够 |

这进一步说明“不是没有注入”，而是目标冲突真实存在。

### 11.3 LEO-only 分支边界

LEO-only 分支也没有自动解决问题：

| branch | best primary | strict UDU | latest/curve-best LEO SAT avg | 解释 |
|---|---:|---:|---:|---|
| `SA26` LEO3 CE1.0 | 82.23 | 80.53 | 42.01 / 43.97 | LEO-only 不自动突破 |
| `SA27` ch2 LEO3 CE1.0 | 83.60 | 82.15 | 44.12 / 44.99 | 比 SA26 好，但还不够 |
| `SA29` LEO3 CE0.7 | 84.28 | 82.59 | 44.40 / 44.97 | 较平衡 |
| `SA30` clear-only train | 80.62 | 78.92 | 42.43 / 43.78 | 只训 clear_leo 泛化不足 |

所以“只改成 LEO”不是充分条件。场景覆盖需要 schedule/curriculum，而不是简单 all-five 或 single-clear。

## 12. 为什么效果不总是明显

### 12.1 增强增加的是 nuisance，不是新身份信息

星地增强不会产生新的 TX 硬件指纹。它只是让同一个 TX 样本经历更强传播链路扰动。RFFI 的核心信息来自 PA、DAC、oscillator、IQ chain 等硬件非理想性；星地信道主要改变传播和接收 nuisance。

### 12.2 AGC 与归一化削弱了幅度物理差异

路径损耗、雨衰、大气衰落本来会造成很强幅度变化，但模拟器中的 mild AGC 和常见 IQ normalize 会削弱幅度直接作用。因此模型更需要处理相位/频偏/噪声/多径这类难扰动。

### 12.3 full concat 会污染 domain loss

full concat 复制 `d_raw` 给 satellite view。对普通 baseline 分类器，这只是 label-preserving augmentation；对 CVS 的 domain/disentangle 目标，它可能让 domain branch 语义混乱。

### 12.4 CE-only 更干净，但约束不够强

CE-only 只要求 satellite sample 分对 TX，没有显式要求 clean/sat `z_id` 对齐，也没有分场景做几何约束。它比 full concat 干净，但更像额外带噪监督。

### 12.5 clean 与 satellite 有真实 tradeoff

提高 satellite robustness 往往需要模型忽略更多相位/频谱变化；但这些变化里可能包含部分 TX 指纹线索。压得太强就会牺牲 clean strict UDU。

### 12.6 联邦下更难

联邦 receiver-client 本地 batch 天然缺少 centralized step 内的跨 receiver/day 全局多域对照。卫星增强如果又混进完整本地 DG loss，冲突更大。因此联邦更需要 CE-only、receiver client、rx_day domain label 和明确诊断。

## 13. 当前最准确的使用建议

### 13.1 写论文/报告时

推荐写法：

> We use physics-informed satellite-channel augmentation as a deployment-oriented stress view over WiSig/ManySig proxy data. The stress scenarios include clear LEO, low-elevation LEO, rain LEO, storm multipath, and mixed orbit. Clean-view performance is reported as a control; satellite/LEO stress is reported separately and does not constitute real in-orbit validation.

中文可写：

> 本工作将 WiSig/ManySig 作为地面可接入代理数据，并通过物理启发的星地信道视图构造部署压力测试。星地视图覆盖 clear LEO、低仰角 LEO、雨衰 LEO、风暴多径和混合轨道等场景。clean view 只作为对照，satellite/LEO stress 用于评估部署鲁棒性，不能等同于真实在轨 IQ 验证。

### 13.2 设计实验时

优先保留三组：

```text
clean-only / no satellite train
CE-only satellite view
CE-only satellite view + domain DSQ
```

再单独比较 robustness route：

```text
ID phase/DSQ + satellite CE
stronger satellite CE weight
late clean-sat z_id consistency
scenario schedule/curriculum
```

不要把 full concat 当成默认主线。它应作为 baseline-style 对照。

### 13.3 联邦实验时

推荐候选语义：

```bash
--train_mode fedprox
--fl_client_key receiver
--wisig_domain rx_day
--wisig_train_ratio 0.1
--fl_rounds 200
--fl_local_objective receiver_agnostic_bex02
--fl_sat_aug_mode baseline_view
--fl_baseline_view_ce_only
--fl_baseline_view_ce_weight 1.0
--domain_freq_stability_mode dsq
--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
--sat_view_prob 1.0
```

并检查日志：

```text
[FED-CONFIG-SAT]
diag_baseline_sat_view_active
diag_sat_cls_active
diag_fishr_domain_count
diag_rx_adv_active
```

### 13.4 Stage2 时

必须把每个 row 的以下字段写清楚：

```text
target_receiver_ids
source_receiver_ids
target_old_tx_ids
target_new_tx_ids
target_unknown_tx_ids
target_channel_view
target_channel_scenarios
support/query split
K
threshold calibration scope
```

如果 `target_channel_view=clean`，只能说 clean control。若要写 deployment-primary，必须是 satellite/LEO target view。

## 14. 与最新文献启发的关系

最近文献映射把 Bi-InterML / KDM 对 CVS 的启发归纳为：

```text
结构化物理扰动
+ 身份保持约束
+ 动态扰动强度
+ 无标签/自监督预训练
+ expert-feature sidecar / gate
+ Stage2 support-query 边界
```

这对当前系统的含义是：

- 不能把星地增强简化成“整段 IQ 随机加噪”；
- 可考虑小波/频带分解，使低频包络/路径损耗和高频局部扰动分开消融；
- satellite CE 最好配合软标签、特征几何约束或 clean-sat assignment consistency；
- 扰动强度应该 dynamic/curriculum，而不是 epoch 1 起固定全强度；
- RCN/statistical expert features 适合作 gate/diagnostic sidecar，不适合替代主干分类器；
- Stage2-A/B/C 的 target-old / target-new / unknown 边界不能因为无标签或自监督预训练被破坏。

这些属于下一步优化方向；不能反向写成当前已经完整实现。

## 15. 当前 claim 边界

可以声明：

- 已有物理启发 satellite channel simulator；
- 已有 clear LEO / low-elevation LEO / rain LEO / storm multipath / mixed orbit 压力视图；
- satellite evaluation 默认覆盖主要 OOD split；
- centralized training 支持 satellite auxiliary CE/consistency；
- centralized training 支持 baseline-style clean+sat concat；
- centralized training 支持 CE-only satellite view；
- federated training 支持 baseline_view / cvs_consistency 两种 satellite mode；
- Stage2 few-shot/open-set 路径可以把 target receiver domain 置于 satellite/LEO target view；
- 当前证据显示 satellite robustness 与 clean strict UDU 有 tradeoff。

不能声明：

- WiSig/ManySig 就是真实卫星训练集；
- satellite augmentation 等价真实在轨验证；
- clean strict UDU 成功就是 deployment success；
- Stage2-A/B 的 unknown rejection 等于 seen-new identity recognition；
- full concat 是当前最稳默认主线；
- satellite CE-only 已经稳定同时提升 clean strict UDU 和 satellite worst；
- StyleBank 或小波结构化增强已经在当前主线完整落地。

## 16. 文件索引

本说明主要依据以下本地文件：

```text
AGENTS.md
项目.md
code/sat_channel.py
code/training_controls.py
code/cvsrffi/eval.py
code/train.py
code/baseline_origin_sat_view.py
code/concat_sat_channel_aug.py
code/federated/fed_trainer.py
code/model_dual_cvsincnet.py
code/target_domain_adaptation.py
code/cvsrffi/spaceborne_fewshot.py
code/eval_spaceborne_fewshot.py
tools/spaceborne_fewshot_da_matrix.py
tools/optimizer_validate_matrix.py
code/scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh
code/analysis/satellite_channel_augmentation_effect_analysis_20260527.md
code/analysis/fl_sat16_replication_from_fsdg49_20260527.md
docs/spaceborne_fed_dg_fsl_prototype_synthesis.md
docs/fed_pvs_cprffi_cvs_integration_analysis.md
```

## 17. 最短复述版

当前 CVS 的星地信道增强是“物理启发 satellite view + 部署压力评估 + 少样本目标域协议”的组合系统。`sat_channel.py` 负责把 clean IQ 转成带轨道、仰角、天气、Doppler/CFO、相位噪声、低 SNR、多径和 IQ imbalance 的 satellite-stressed IQ；`train.py` 决定它是作为 late weak satellite CE、clean-sat consistency、full 2B concat，还是更安全的 CE-only satellite view 进入训练；`cvsrffi/eval.py` 默认用 five-scenario stress 对 OOD splits 评估；Stage2 则要求 target-old、target-new、unknown 全部在 `R_t` 的 satellite/LEO target view 下组织 support/query。

当前最稳的技术判断是：星地增强有效但不是单独解决方案。full concat 容易污染 CVS 的 domain/DG 语义；CE-only 更干净，但仍主要是带噪监督。已有结果显示 domain DSQ 更能保 clean/UDU，ID phase/DSQ 更能抬 satellite robustness，但二者尚未统一。因此论文和报告应把它写成 deployment-oriented stress / auxiliary view，而不是真实在轨验证或唯一核心创新点。
