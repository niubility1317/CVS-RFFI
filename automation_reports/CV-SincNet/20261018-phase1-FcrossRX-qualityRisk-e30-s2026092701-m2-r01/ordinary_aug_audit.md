# 当前 L 普通增强的有效配置审计

本记录只用于报告澄清，不是发布或候选晋级门槛。当前 I01/S01 健康训练不改动、不停止、不重启。审计只读生产代码和既有 own20 参数；未运行增强、IQ、checkpoint、NN 或远端操作。

## 结论

NoExplicitCFO 只将参考响应 29 维中的第27坐标置零，保留必要的内部 CFO 估计与补偿。它不关闭普通 IQ CFO 随机扰动，也不等于全模型对隐式 CFO 的严格不变性。

实际 L 普通路径启用 PA，DAC 关闭，class signature 关闭。这些是旧共有普通增强，不是 qualityRisk 新增机制；启用事实、旧源健康或历史性能不能充当 PA 标签安全性已被独立验证的证据。

进入公共 Sinc 前没有新增逐包 RMS 或 LayerNorm：实际 identity wrapper只传递输入；backbone在256点时pad_crop直接返回，随后Sinc做卷积。统一Sinc修正只改变频率参数坐标，仍调用原forward_iq_pair。因此不能把普通幅度变化认定为在公共入口被完全归一化消除。反过来，也不能声称全模型保留该增益：Sinc后time_fuse/DSConv有GroupNorm，默认融合头有LayerNorm；参考响应分支自身有窗口能量归一化及功率比。该静态事实不量化全网络对增益的敏感度。

## 有效配置的来源与时钟

生产链为native build_aug_base_cfg(args) → make_augmentor(base_cfg) → DataAugmentation.build_augmentor(**kwargs) → RFFIAugmentor；factory不额外覆盖CFO/PA参数。当前 native L 调用显式传 no_pa=(not aug_enable_pa_normal)，own20参数true使其为False，不能用__call__默认no_pa=True推断PA关闭。

尾10轮 Observer.native_scope 使用strict own20参数，将原 configure_augmentor_for_epoch 固定在epoch13；不是用新10轮预算重算ramp。父参数为warmup3、ramp15、min.1、max.35、curve1.25，late_stable_start0。有效scale：

    .1 + .25 * ((13 - 3) / 15)^1.25 = .25060033393497416

下表为有效配置概率，不是实测触发频次。除PA逐包选择外，表中各操作先用一次整批共享Bernoulli gate，再生成相应参数。

| 实际顺序 | 有效概率 | 参数与作用 |
|---|---:|---|
| DC offset | 0.07518010018049225 | 每包I/Q偏移各U[−.005012006679,+.005012006679] |
| band-edge taper | 0.06265008348374354 | 本次调用共享Tukey-like alpha，U[.005012006679,.025060033393] |
| 循环移位 | 0.08771011687724095 | 命中后每包整数−8至+8点，gather模256循环 |
| 幅度缩放 | 0.11277015027073838 | 命中后每包U[.9,1.1]；上下界本身不随ramp缩放 |
| 相位旋转 | 0.11277015027073838 | 命中后每包U[0,2π] |
| CFO | 0.08771011687724095 | 命中后每包f均匀±.000100240133574 cycles/sample，25MHz下±2506.003339Hz，乘exp(j2πfn) |
| 相位噪声 | 0.07518010018049225 | 每包sigma均匀[0,.001503602004]，原相位随机游走实现 |
| multipath | 0.04510806010829535 | Python随机tap数2至4，最大delay1点 |
| AWGN | 0.10024013357398967 | 每包SNR均匀20至36dB，噪声功率按此时波形功率计算 |
| DAC | 0 | p_dac=0，因此普通DAC链关闭 |
| PA | 0.03508404675089639 | 每包独立选入PA子批；no_pa=False |
| 最后处理 | 必经 | nan_to_num及恢复dtype，没有逐包RMS归一化 |

class_signature=False。虽然普通调用传入真实L标签，但相应标签签名分支不会启用；不能把class signature关闭等同于PA关闭。

PA的四个随ramp覆盖的幅度参数分别为mp_sigma=.012530016697、mem_sigma=.010024013357、ampm_max=.050120066787、iq_img_max=.005012006679。内部组件顺序为memory polynomial → FIR memory → Rapp AM/AM → AM/PM → IQ imbalance；内部调用概率依次.9/.4/.9/.6/.05，由Python random逐PA子批决定。strength仍是原sqrt(U)采样；Rapp a_sat仍U[.6,1.1]、p仍U[2,5]，有效饱和阈值含(1−.55s)。所以不能将整个PA链说成统一缩小到四分之一。memory polynomial末尾除以(1+.5s)不是RMS归一化。

## 归一化与固定窗口边界

1. dataset先中心裁剪为256点并按原策略RMS归一化；普通增强之后无整体IQ RMS步骤。
2. 外层DualCVSincNetDisentangle.forward_identity_only与正常identity路径通过backbone_forward_compat原样传x，兼容函数也不做归一化。Phase1IdentityAdapter将x.float()交给SequenceReference/ReferenceCVS；IdentityOnlyCVS.features直接调用真实backbone。
3. backbone只pad/crop；长度已经256时无变换。公共Sinc原forward_iq_pair为reshape→conv1d→reshape，没有输入RMS/LN。CENTER_25K复用该同一forward。
4. Sinc后time_fuse/DSConv的GroupNorm属于特征层，默认头LayerNorm也不是输入波形RMS。线性/非线性basis、噪声和PA的组合使得不能仅凭这些归一化断言整个增益扰动被精确抵消。
5. 参考响应单独固定使用x[:,80:160]，其内部重复平均后按能量归一化；relative transfer和部分功率比可抑制理想全局增益。它不将归一化后的IQ反向送入公共Sinc。

循环移位是对现有256点切片做模256拼接，没有更长真实上下文。它可能产生人为首尾接缝；不能称为已验证的真实采样窗口变化。±8点移位也改变固定80:160参考窗口内的内容与时序对齐，可能影响参考质量/响应。这里不声称每个接缝直接落入参考窗口，也不据此新增筛选或热修改。

## 与As和RDrop的关系

qualityRisk共享bank从native普通增强前raw L生成；原Practical只做其既有normalize_input和完整ResidualChannel。bank As没有先叠加普通CFO、PA、multipath或AWGN，不能将本轮描述为修复As重复损伤。

ordinary两次RDrop前向复用同一次已生成ordinary IQ；mixed两次复用同一个已替换bank IQ/标签/metadata的视图。Dropout仍按原训练随机子网络执行。已通过的真实hook烟测核对同IQ；本报告澄清不新增NN。

## 精确代码依据

- 父E20实际参数：E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/runtime_metadata_r01/B01-s2026092701/stage20/actual_native_args.json。
- factory/base/ramp：E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/adv3b02_xuc/code/cvsrffi/schedule.py:150、:189、:193；DataAugmentation.py:1050。
- native显式no_pa：同冻结依赖SSDG/train_ssdg.py:9020；aug_base构造:8571。
- 当前tail冻结13：E:/type10-7/code/worktrees/cvs_iq_timing_20261016/experiments/cvs_quality_risk30/observer.py:42。
- 顺序/概率gate：冻结DataAugmentation.py:839、:856、:1005；CFO公式:110；PA:780；RMS缺省与末尾处理:1033。
- NoExplicitCFO：E:/type10-7/code/worktrees/cvs_iq_timing_20261016/experiments/cvs_source_identity_readout/input_policy.py:5。
- entry：E:/type10-7/code/worktrees/cvs_iq_timing_20261016/experiments/cvs_phase1_overlay/model.py:88；cvs_identity_ce/model.py:30；冻结model.py:20、:1672、:1682、:302。
- 统一Sinc复用：E:/type10-7/code/worktrees/cvs_iq_timing_20261016/experiments/cvs_rdrop_sinc_units/coordinate.py:29。
- 参考窗口及自身归一化：E:/type10-7/code/worktrees/cvs_iq_timing_20261016/experiments/cvs_reference_identity/model.py:50、:61、:73。

补充外层入口依据：冻结 model_dual_cvsincnet.py:71、:894、:1039 原样转发x；无该入口逐包RMS/LN。所读实际F来自已保存normalization_structure_audit.json标识的DualCVSincNetDisentangle及Phase1替换identity链，不把原unused单网络构造分支当部署路径。
