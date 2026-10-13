# 首种子源特征导数：完整结果

伪标签信号进入了真实浅层、中层和 joint160，但量级较小。11 个 fθ 参数路径的加权 U/L 梯度范数比中位数约 1.44% 至 2.51%。时序中层 L/U cosine 中位数为 0.0490、−0.1091、0.0680，方向不稳定。它是源 eval 代理诊断，不能当作原生 train/dropout 梯度历史或目标性能因果解释。

状态 ANALYZED / VERIFIED。run 20261014-phase1-source-feature-gradient-manysig-s01-m3-r01，实际代码/config d4c1e9280854a3bfc988626f98224e737347260b。三行全部终止、无失败或缺项。独立核对 75 批、825 条参数路径、4125 条激活/RX 记录、18900 个跨模型源观测和 56700 包次前向。模型 seed 只有 2026092701 一个，三组不是三模型 seed；未产生新的目标结果，84% 主目标不变。全部值见 [results.json](results.json)，复现与权限见 [预登记](report.md)。

## 梯度对象与不变项

OLD_GATE_COS50、RANDOM_ORDER、PAIRED_ORDER 各使用已固定 final50/11100 学生及其真实 EMA，来源/角色/祖先/终轮经 source_proof 核验。默认 ResidualPhysicalHead cosine6×160/scale30 保留，没有拟合、替换或优化头。autograd.grad 只请求实际 fθ176065（仅排除 class W960，保留全部 norm/gain/reference）及真实浅中层/joint160 激活。没有 backward、optimizer、.grad 累积、EMA 或参数/buffer 更新；三个实际模型状态均逐元素未变。

完整 L_s6300 恰一次，允许物理 PAIRS 排序，batch256、尾156、25批；合法 L 标签在原完整 mask 固定后进入监督损失/正确率观察。既有 V cache materialize 后丢弃，无 V 前向/求导；U IQ/真值、target IQ/真值读取均 0。NoExplicitCFO 和内部补偿保留。这些 RX 都曾参加原训练，不能叫源留一 RX 泛化。

学生/teacher 均 eval、无 dropout，固定 strong 噪声 0.015。比较 clean 真标签 CE/smoothing.01 与原权重 .16 pseudo CE＋.01 entropy 合并梯度。L/U 代理共用 source batch，并非原训练 L128/U256 组合。Sinc Hz 单列，不将不同路径的原始 norm 作为重要性排名。

## 实际特征参数路径

U/L 与 cosine 为 25 批中位数；图中误差条是批次四分位区间，不是模型 seed 不确定性。

| 路径 | OLD U/L（%） | RANDOM U/L（%） | PAIR U/L（%） | OLD cosine | RANDOM cosine | PAIR cosine |
|---|---:|---:|---:|---:|---:|---:|
| reference | 1.6523 | 1.6291 | 1.6393 | -0.0045 | 0.0195 | -0.0203 |
| Sinc_Hz | 2.0315 | 1.4667 | 1.9624 | 0.2193 | 0.2128 | 0.2687 |
| time_shallow | 1.4941 | 1.4964 | 1.7273 | 0.2950 | 0.1249 | 0.2247 |
| time_middle | 1.4450 | 1.6295 | 1.8853 | 0.0490 | -0.1091 | 0.0680 |
| time_projection | 1.5376 | 1.7088 | 1.9493 | 0.1301 | 0.1577 | 0.0696 |
| frequency | 1.6626 | 1.7485 | 2.0176 | 0.3128 | 0.2341 | 0.1858 |
| PA_shallow | 2.0980 | 2.3713 | 1.9979 | 0.4972 | 0.4597 | 0.4228 |
| PA_middle | 2.4567 | 2.4554 | 2.1710 | 0.6671 | 0.6990 | 0.6372 |
| PA_projection | 2.1065 | 2.3488 | 2.0484 | 0.2916 | 0.4005 | 0.2794 |
| fusion | 1.6421 | 2.0033 | 2.0019 | 0.1408 | 0.1128 | 0.0541 |
| feature_norm_gain | 2.3183 | 2.5052 | 2.4742 | 0.1629 | 0.1821 | 0.1380 |

[梯度路径图](gradient_paths.png) 来自真实记录。time_middle 的负 cosine 批次数为 11/25、13/25、11/25；reference/fusion 也有正负变化。不能据此直接宣称训练梯度冲突或实施强对齐。所有路径都有非零源导数；Sinc 的非零导数也不能代替数值更新证据，原 raw-Hz 学生核保存态仍几乎不变。

## 源门控与 RX 敏感度

| 组 | 接收/源样本 | 源代理覆盖（%） | 接收正确 | teacher 弱视图正确 | 导数时间（秒） |
|---|---|---:|---:|---|---:|
| OLD_GATE_COS50 | 5873/6300 | 93.2222 | 5873 | 6265/6300 | 31.45 |
| RANDOM_ORDER | 5865/6300 | 93.0952 | 5865 | 6261/6300 | 30.31 |
| PAIRED_ORDER | 5882/6300 | 93.3651 | 5882 | 6263/6300 | 30.26 |

本次接收样本均正确，是合法 L_s 代理观察，不是 U_s 准确率。eval 无 dropout 与旧 train-mode 门控诊断不同，未混算为同配方复现。

下表是同一 joint160 路径的逐样本 L CE 导数 norm 均值，移除了 batch mean 的 1/N 因子，每 RX1260 样本；不能与其他层原始 norm 直接比较。全部层/RX 值在 results.json。

| 源 RX | OLD | RANDOM | PAIR |
|---|---:|---:|---:|
| 1 | 0.033583 | 0.034058 | 0.033823 |
| 3 | 0.104995 | 0.106667 | 0.105170 |
| 4 | 0.044983 | 0.046416 | 0.044423 |
| 6 | 0.024596 | 0.024623 | 0.025097 |
| 8 | 0.072444 | 0.071710 | 0.069799 |

PAIR 的 RX3 约为 RX6 的 4.19 倍，RX8 约为 2.78 倍。源 RX 敏感度仍有差异；不证明 RX 信息都是捷径或应删除域信息。

## 执行与成本

一归档本地/远端 SHA 比较、remote compile、真实 CP 零 IQ/no-query 导数 smoke 均 PASS。独立 PID/CWD/argv/CPU 和日志 0→185 字节读回 VERIFIED；最终日志 277 字节、三 completion 齐全、进程终止、failure 不存在。一次直接 P0/P1 审查在发布前发现 OLD 参数入口误走禁止历史重训的 make_args；改为读核实 saved native args，三 row 预检覆盖同一路径，唯一 targeted 复核 PASS。没有真实失败、重启、热修改或额外审查。

新训练、optimizer 和单独分类头更新均 0；三个来源各已训练 50 轮/11100，没有增加累计预算。56700 包次前向，导数测量合计 92.03 秒，另有 CP 加载/预检/cache IO/传输成本，不是全流程墙钟时间。原 IQ/权重/release 日志留在 N607，定位见 experiment.json。

## 下一轮最多两个动作

1. 检查真实 t1/t2 统计与原生身份保护调用，准备一个独立的源 L 身份保护跨 RX MixStyle 因素。保持默认头、旧门控＋EMA＋cos、原 input/data、scratch≤50；扰动进入浅/中层、推理关闭。不叠加 PAIRS/Sinc/TIME，不因目标分数改变 λ 或层位。MixStyle 原作者在较低连续层混合特征统计；射频跨 RX 适用性是待检验推断。[官方实现](https://github.com/KaiyangZhou/mixstyle-release)、[论文](https://arxiv.org/abs/2104.02008)。
2. 源接线/机制支持后，固定单 seed 匹配控制做真实训练，自动默认 clean 预测/独立 truth-last；否则保留负证据后选另一独立机制，不扫描参数追 84%。Fishr 当前缺本网络逐样本分域梯度方差证据。官方 DomainBed 实现用 classifier 参数为梯度对象，fθ 浅/中层变体须另定义对象与域分组并披露差异，不能把本次 norm/cosine 当作 Fishr 验证。[官方方法](https://github.com/alexrame/fishr)、[对象代码](https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py)。
