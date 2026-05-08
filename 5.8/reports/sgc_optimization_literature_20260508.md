# SGC 模块优化与星上目标域泛化方案

日期：2026-05-08  
代码基线：当前 root `CVS-RFFI`，结合 `5.8/logs` 与已解析指标。

## 1. 当前结论

5.8 的结果说明，SGC 不是完全无效，而是“显式全链路补偿”方向不稳：

| 组别 | 结论 |
|---|---|
| `E2_residual_only_std_res001` | 当前最强 SGC 候选。相对 `E0_no_adapter_continue`，primary +0.40、strict UDU +0.45、worst-RX +0.48、SAT Avg +0.73。 |
| `E4_full_sgc_mild_res001` | full SGC 不建议作为主线。幅度归一化、CFO 补偿、频谱抑制叠加后，primary -0.23、worst-RX -0.56。 |
| `E3_no_res_control` | no-res 控制组 primary 高，但 SAT Avg 与 worst-RX 弱于 E2，说明残差分支仍有价值。 |

核心判断：RFFI 不是通信解调任务，不能把 SGC 做成尽可能强的均衡/滤波器。星地链路干扰和发射机硬件指纹混在同一个 IQ 片段里，过强的显式补偿会把 PA/LO/IQ 非理想等可识别痕迹一起抹掉。SGC 应改成“保守残差去扰动器”，只削弱星地链路带来的域偏移，并把分类证据保留给主干。

## 2. 文献依据与可迁移思想

1. Radio Transformer Networks 证明无线前端可以学习同步/归一化变换，但其目标是调制识别，学习到的同步变换会服务分类目标；迁移到 RFFI 时必须额外约束“不要去掉发射机指纹”。参考：O'Shea 等，Radio Transformer Networks，arXiv:1605.00716。https://arxiv.org/abs/1605.00716

2. DeepRx 说明全卷积接收机可以联合学习信道估计、均衡与检测，但它服务的是通信 bit 检测。RFFI 的目标相反：不应把所有非理想都恢复成“干净通信信号”，而应只去除接收/信道域偏移。参考：Honkala 等，DeepRx，arXiv:2005.01494。https://arxiv.org/abs/2005.01494

3. Land-mobile satellite channel 文献强调 LMS/MSS 链路包含 LOS、shadowed LOS、Rayleigh multipath、遮挡和慢/快衰落。当前 `sat_channel.py` 的 LOO/Rician/Rayleigh、低仰角、多径、Doppler/CFO/IQ imbalance 设定与这类模型一致。参考 NASA NTRS《Propagation Modeling for Land Mobile Satellite Systems》。https://ntrs.nasa.gov/api/citations/19880016310/downloads/19880016310.pdf

4. RFFI 接收机无关方向已经明确指出 receiver/channel 会干扰发射机指纹提取，需要对接收机域特征做对抗、解耦或源无关适应。参考 Shen 等 receiver-agnostic RFFI。https://arxiv.org/abs/2207.02999 ；RIEI/domain generalization。https://arxiv.org/abs/2411.03636 ；feature disentanglement + adversarial training。https://arxiv.org/abs/2510.09405

5. 源无关目标域适应文献与“星上部署后利用目标域样本”高度相关。CSCNet 冻结分类器，用目标接收机无标签样本做 entropy、pseudo-label 和 contrastive adaptation。SGC 可以采用更保守版本：只训练 SGC 残差/少量 BN，不反向改动最终分类器。参考 Sensors 2025 CSCNet。https://www.mdpi.com/1424-8220/25/14/4451

6. MixStyle 与 Fishr 仍应保留为主干域泛化工具。MixStyle 通过混合特征统计隐式合成新域；Fishr 通过匹配域级梯度方差让 loss landscape 在域间更一致。参考 MixStyle OpenReview。https://openreview.net/forum?id=6xHJ37MVxxp ；Fishr ICML 2022。https://proceedings.mlr.press/v162/rame22a.html

## 3. 新 SGC 目标定义

SGC 的目标不应只写成：

`maximize accuracy on satellite-overlaid test_unseen_day_unseen_rx`

应改成三个同时满足的目标：

1. 星地干扰削弱：降低 clean vs satellite view 的 ID feature drift、domain probe 可分性、按信道场景的性能掉点。
2. 指纹保真：保持 clean/OOD、worst-RX 和 per-class accuracy，残差能量受控，不能把输入改成另一个“通信均衡后”的分布。
3. 星上目标域适应：部署后允许使用无标签目标域样本做 source-free adaptation，但只能使用 calibration/adaptation split，最终评估 split 不参与训练。

## 4. 已落地代码优化

### 4.1 SAT 评估口径修正

已将 `train.py` 的 `--eval_sat_on` 默认值改为 `all`，不再只评估 `test_unseen_day_unseen_rx`。同时默认 SAT 测试场景改为 `simple_leo`，即基于汇报图公式的简化星地信道：

`h(t)=L(t) * xi(t) * exp(j phi(t))`

其中 `L(t)` 为距离路径损耗，`xi(t)` 为对数正态阴影衰落，`phi(t)` 为 LEO 多普勒相位，最终 `y(t)=h(t)x(t)+n(t)`。该测试不再默认叠加 Rician/Rayleigh 状态、多径、IQ imbalance、phase noise 等复杂项。

新的 SAT 输出包括：

- `[SAT-TEST]`：保留原有 primary OOD 聚合 `overall_tx`，新增 `all_named_tx`。
- `[SAT-TEST-SPLIT]`：对每个 scenario 输出每个 named split，包括 `test_unseen_day_seen_rx`、`test_seen_day_unseen_rx`、`test_unseen_day_unseen_rx`、`test_day_*`、`test_rx_*`。

这样后续星地测试会和普通测试一样覆盖所有划分。

### 4.2 残差分支改造

`sgc_adapter.py` 新增两类保守残差能力：

| 模式 | 配置 | 作用 |
|---|---|---|
| bounded plain residual | `residual_mode="plain"`, `residual_max_gamma` | 保留原残差结构，但用 `tanh(gamma) * max_gamma` 限制残差幅度，避免过修正。 |
| multiscale residual | `residual_mode="multiscale"` | 用 3/5/9 kernel 与 1/2/4 dilation 捕捉局部相位噪声、轻微频偏、多径拖尾。 |
| gated multiscale residual | `residual_mode="gated_multiscale"` | 根据 RMS、DC、phase-step std、spectrum entropy 生成样本级 gate，只在疑似强信道扰动样本上放大残差。 |

同时 `sgc_aux` 新增：

- `adapter_input_rms`
- `residual_delta_rms`
- `residual_effective_gamma`
- `residual_gate_mean`

这些指标用于判断残差是否真的在工作，而不是只看最终准确率。

### 4.3 残差正则改造

`sgc_losses.py` 的 `residual_regularization` 现在同时约束：

- effective gamma：限制残差通道整体强度；
- normalized residual delta：限制实际输入改变量，避免 gamma 小但卷积分支输出过大的情况。

## 5. 创新 idea：SGC-TADA

建议把下一代模块命名为：

**SGC-TADA: Satellite-Ground Channel Target-Aware Domain Adaptation**

核心思想：SGC 是一个可部署的、轻量的、源无关目标域残差适配器，而不是普通数据增强模块。

### 5.1 训练阶段

主训练仍然使用 Lite-B no-DAC + conservative MixStyle + Fishr。SGC 只在后段打开：

`L = L_ce(clean) + lambda_sat_cls L_ce(sat) + lambda_sat_cons D(z_clean, z_sat) + lambda_fishr L_fishr + lambda_res L_res`

其中：

- `sat` 是 synthetic satellite view；
- `D(z_clean, z_sat)` 用 cosine/MSE feature consistency；
- `L_res` 约束残差幅度和输入改变量；
- SAT consistency 不应过早开启，建议 `sat_cons_start_epoch=20` 或更晚。

### 5.2 部署/星上目标域适应阶段

星上收到目标域无标签样本后，使用 adaptation buffer 更新 SGC，但必须冻结主分类器：

`L_target = lambda_ent H(p_t) + lambda_pl CE(p_t, pseudo_y_t) + lambda_con L_contrast + lambda_res L_res`

建议规则：

- 只训练 SGC residual 或 SGC + 少量 BN affine；
- classifier/head 冻结，防止目标域伪标签错误导致类别漂移；
- pseudo-label threshold 从 0.95 降到 0.85，前 5-10 epoch 只做 entropy/ECC 弱约束；
- 每次星上 adaptation 必须划分 calibration buffer 与 holdout buffer，holdout 不参与更新。

### 5.3 论文级创新点表述

可写成：

> Different from conventional receiver equalization, the proposed SGC-TADA learns a fingerprint-preserving residual satellite-channel suppressor. It attenuates nuisance channel shifts under a bounded residual constraint and adapts to unlabeled on-orbit target-domain samples in a source-free manner, while freezing the classifier to preserve transmitter identity boundaries.

中文表述：

> 与传统通信均衡追求恢复符号不同，SGC-TADA 追求“指纹保真下的星地信道去扰动”。模块通过有界残差与样本级信道门控削弱低仰角、多径、遮挡和频偏导致的域偏移；部署阶段仅利用星上无标签目标域样本更新轻量残差适配器，使模型在不接触最终测试标签的情况下完成源无关域泛化。

## 6. 下一轮实验矩阵

### 6.1 先跑的 SGC 残差结构组

已写入 `run_final_best_sgc_queue.sh` 的 SGC suite：

| 实验 | 目的 |
|---|---|
| `E0_no_adapter_continue` | source checkpoint 继续训练对照。 |
| `E1_residual_only_std` | 复现无正则 residual-only。 |
| `E2_residual_only_std_res001` | 复现 5.8 最优 residual-only。 |
| `E3_no_res_control` | 显式前端无残差控制组。 |
| `E4_full_sgc_mild_res001` | full SGC 负例复核。 |
| `E5_residual_bounded_res001` | 验证 bounded gamma 是否降低过修正。 |
| `E6_residual_multiscale_res001` | 验证多尺度残差是否提升 storm/multipath/low-elev。 |
| `E7_residual_msgated_res001` | 验证 channel-stat gate 是否只在强扰动样本上工作。 |

### 6.2 目标域适应验证组

下一步需要单独实现/启用 target adaptation loader，严禁直接用最终 test 做训练。建议 split：

- `target_calib_unlabeled`：目标域无标签适应样本；
- `target_holdout_eval`：目标域最终评估样本；
- 如果数据量小，用按时间连续切分，calibration 在前、holdout 在后，避免随机穿插造成泄露。

实验：

| 实验 | 更新参数 | 目标损失 | 判断标准 |
|---|---|---|---|
| `TA0_no_adapt` | none | none | 星上无适应基线。 |
| `TA1_sgc_entropy` | SGC only | entropy + residual | 看是否削弱域偏移但不坍塌。 |
| `TA2_sgc_pl` | SGC only | high-conf pseudo CE + residual | 看目标域 accuracy 与 per-class 是否提升。 |
| `TA3_sgc_pl_contrast` | SGC only | pseudo CE + contrastive + residual | 看类间边界是否更稳。 |
| `TA4_sgc_bn_affine` | SGC + BN affine | TA3 | 评估少量主干自适应是否有收益。 |

## 7. 机制验证指标

不能只看 SAT Avg。每个 SGC 实验至少输出和分析：

1. clean OOD：overall、strict UDU、unseen_day_seen_rx、seen_day_unseen_rx。
2. all split SAT：每个 scenario x 每个 named split。
3. worst-RX：普通与 SAT overlay 下都要看。
4. feature drift：`1 - cosine(z_clean, z_sat)`，SGC 后应下降。
5. domain probe：用 z_id 预测 day/rx/channel scenario 的准确率，SGC 后应下降。
6. residual diagnostics：`residual_effective_gamma`、`residual_delta_rms/input_rms`、`residual_gate_mean`。
7. class-wise table：确认提升不是只来自优势类别，差类别不能继续恶化。

## 8. 判断成功的门槛

SGC-TADA 进入主线必须满足：

- 对 `E0_no_adapter_continue`，primary OOD 不下降超过 0.10；
- strict UDU 或 worst-RX 至少提升 0.30；
- SAT Avg 至少提升 0.80，且 storm_leo_mp/low_elev_leo 至少一个提升明显；
- all split SAT 没有出现单个核心 split 大幅下跌；
- residual ratio 保持小于 5%-8%，证明是去扰动而不是重写输入；
- 目标域 adaptation 只用 calibration buffer，holdout 上提升，且没有使用测试标签。
