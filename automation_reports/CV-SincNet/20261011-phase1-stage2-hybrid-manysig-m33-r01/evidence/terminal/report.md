# 混合机制三种子 E40：固定 33 实验预登记

PLANNED。当前用户扩大方法范围，允许合规混合机制。每卡最多 3 个进程，包含其他任务；显存或进程容量不足时排队。所有 8 张卡参与调度。

|方案|固定机制|
|---|---|
|H0|CE|
|H1|CE + cross-RX supervised contrastive|
|H2|CE + same-TX cross-RX feature mix|
|H3|CE + between-TX manifold mix|
|H4|CE + cross-RX contrastive + cross-RX feature mix|
|H5|SAM CE|
|H6|CE + cyclic-tail weight averaging|
|H7|SAM + cyclic-tail weight averaging|
|H8|H4 + coverage finite L/T|
|H9|H4 + SAM + cyclic-tail averaging + coverage finite L/T|
|H10|H4 + balanced conditional finite L/T|

全部 11 方案 × 3 种子（2026092701、2026092702、2026092703），每行从对应原始 E200 纯 CE 权重开始，40 epoch ×222=8880 次 AdamW 身份更新，batch128；原身份网络、旧门控、cosine 分类头保持。H0 新跑为同批对照，不复用旧 E40 作为初始化。训练只读 L_s6300；V27000 只验证，不参与梯度、选模、排序；U、EMA、伪标签、星地信道生成均不执行。数据划分、物理 ID、继承来源运行前严格核对。

普通 permutation 和每 8 步 relation32 替换流与已有 B0 相同（普通96+两个same-TX/different-RX的8+8袋）；所有方案匹配。对比项用最终 normalized joint160，temperature0.1、权重0.05，只将不同物理包的同TX不同RX作为正样本、不同TX作为负样本，同TX同RX不充当负样本。重复物理ID排除。无目标域聚类或标签。
跨RX同TX特征混合每8步对关系袋配对，lambda~U[.25,.75]、权重0.25；仅复用本步特征，以原cosine头预测同身份。H3 每4步在全128特征内随机配对，Beta(.4,.4)，混合标签CE权重0.25。全部附加项前444步线性ramp。
SAM rho0.02、非adaptive，两次前后向用同一批及相同全局随机状态，临时参数扰动精确恢复后只执行一次AdamW更新。多出的前后向与耗时单列，不能称等算力。原网络不允许BatchNorm静默重复统计。
H6/H7/H9 前30轮用原cosine日程，31–40轮五个2epoch余弦周期（peak为原组peak的0.5，floor1e-6）；固定平均E32/34/36/38/40的全部身份可训练参数，最终buffer。原始E40另存但不选用、不测试。该方案是AdamW+循环尾部参数平均，不冒称原始SGD-SWA。
H8/H9 用固定coverage方向的有限L/T风险；H10 用独立新版本平衡坐标条件策略，时间四坐标全tanh，noise取abs，消除旧sigmoid约0.5偏置；仍沿用原L∞半径U[0,1]、0.5coverage/0.5conditional、旧物理边界与真实IQ算子。H10两个独立L/T网络在原E200固定身份网络上各完整两遍L_s（第一遍raw margin，第二遍有限risk），每seed200辅助更新，三seed600；fit后冻结，无在线优化。每4个身份step取最后32个样本执行，LLTT循环，finite风险权重1、keep权重0.1。物理扰动属于已授权轻微L/T源域动作，不是星地增强。
每行预定最终/平均权重冻结后立即预测clean全168000个独立物理样本，再由独立CPU scorer连接truth。7RX、6TX、4days、逐seed、混淆矩阵与配对差全部报告。主方案H9单列；全部预登记方案如有完整三seed均值>=84%，同时记录目标达到。不能借测试结果改方案、选参数、重跑、重排或隐去负结果。多重比较不自动宣称显著性。
SAM、循环平均、SupCon与Manifold Mixup分别参考原始论文，但本任务有效性尚待验证：[SAM](https://arxiv.org/abs/2010.01412)、[SupCon](https://papers.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)、[SWA](https://arxiv.org/abs/1803.05407)、[Manifold Mixup](https://arxiv.org/abs/1806.05236)。
历史反思：R统计可拟合不等于身份判别受益；此前有限风险搜索589/600步为零梯度、temporal条件方向被noise偏置占据，不能据此认为只需加大损失。此次单独改坐标及raw→risk热身，H10-H8只作固定对比，完整源域方向/风险诊断记录。解耦作为一个待证机制，不继续以其单独承担性能目标。
成本：记录SAM额外样本、主训练样本、动作执行2220次、relation1110次、共享准备与新增辅助拟合、每epoch时间/损失/权重/梯度、GPU峰值显存、推理时间/参数。资源竞争下wall time只描述实测，不预承诺加速倍数。
审查限定P0/P1直接正确性。技术失败保留产物并停止新pending，已健康运行的实验继续；不以低性能停止，不覆盖重启。唯一launch owner codex/root/stage2-hybrid-20261011。

|要求|实现|核验|
|---|---|---|
|匹配CE父来源/源角色|design/common.verify_parent/source|原E200来源与物理角色|
|多个混合方案和消融|design H0-H10|33精确配置|
|每方案40轮|source|8880更新/逐步日志|
|SAM/固定平均|optim/source/evaluate|精确恢复/五快照及冻结来源|
|合规独立L/T|actions|L_s拟合/边界与源诊断|
|八卡每卡最多3进程|dispatch|共享capacity锁与实时读回|
|逐行自动clean|dispatch/evaluate|完整prediction后truth-last|

本地验证：11方案真实同构网络前后向、SAM恢复与RNG、固定五快照均值及来源、源诊断接口、pure CE分支、每卡3上限、truth-last负测均PASS。独立P0/P1审查PASS。真实父权重与源物理角色、辅助拟合和远端图由不可变发布preflight继续验证。


# Fixed-parent hybrid mechanisms: H0–H10 E40 clean test

Each row uses 168000 fixed physical queries across 7 receivers and 4 days. Predictions precede independent truth access.
P0 predictions are reused when registered. Results do not feed model, hyperparameter, or rerun selection.

| Arm | Seeds | Mean accuracy (%) | SD (pp) | Minimum (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|---:|
| P0 | 3 | 80.526 | 1.320 | 79.052 | 79.950 |
| H0 | 3 | 80.886 | 1.517 | 79.135 | 80.371 |
| H1 | 3 | 80.688 | 1.349 | 79.173 | 80.226 |
| H2 | 3 | 80.872 | 1.522 | 79.115 | 80.355 |
| H3 | 3 | 79.888 | 0.963 | 78.780 | 79.416 |
| H4 | 3 | 80.698 | 1.350 | 79.181 | 80.236 |
| H5 | 3 | 80.469 | 1.807 | 78.383 | 79.982 |
| H6 | 3 | 80.791 | 1.485 | 79.078 | 80.266 |
| H7 | 3 | 80.433 | 1.847 | 78.302 | 79.943 |
| H8 | 3 | 80.721 | 1.153 | 79.435 | 80.302 |
| H9 | 3 | 80.644 | 1.334 | 79.112 | 80.216 |
| H10 | 3 | 80.666 | 1.047 | 79.465 | 80.238 |

## Every seed

| Arm | Model seed | Accuracy (%) |
|---|---:|---:|
| H0 | 2026092701 | 81.733 |
| H1 | 2026092701 | 81.133 |
| H2 | 2026092701 | 81.737 |
| H3 | 2026092701 | 80.362 |
| H4 | 2026092701 | 81.145 |
| H5 | 2026092701 | 81.454 |
| H6 | 2026092701 | 81.595 |
| H7 | 2026092701 | 81.446 |
| H8 | 2026092701 | 81.070 |
| H9 | 2026092701 | 81.261 |
| H10 | 2026092701 | 81.149 |
| H0 | 2026092702 | 79.135 |
| H1 | 2026092702 | 79.173 |
| H2 | 2026092702 | 79.115 |
| H3 | 2026092702 | 78.780 |
| H4 | 2026092702 | 79.181 |
| H5 | 2026092702 | 78.383 |
| H6 | 2026092702 | 79.078 |
| H7 | 2026092702 | 78.302 |
| H8 | 2026092702 | 79.435 |
| H9 | 2026092702 | 79.112 |
| H10 | 2026092702 | 79.465 |
| H0 | 2026092703 | 81.792 |
| H1 | 2026092703 | 81.760 |
| H2 | 2026092703 | 81.764 |
| H3 | 2026092703 | 80.520 |
| H4 | 2026092703 | 81.768 |
| H5 | 2026092703 | 81.570 |
| H6 | 2026092703 | 81.701 |
| H7 | 2026092703 | 81.551 |
| H8 | 2026092703 | 81.659 |
| H9 | 2026092703 | 81.558 |
| H10 | 2026092703 | 81.385 |
| P0 | 2026092701 | 80.928 |
| P0 | 2026092702 | 79.052 |
| P0 | 2026092703 | 81.599 |

## Paired differences

| Comparison | Seeds | Mean (pp) | Positive seeds |
|---|---:|---:|---:|
| H0 − P0 | 3 | 0.360 | 3 |
| H1 − H0 | 3 | -0.198 | 1 |
| H2 − H0 | 3 | -0.014 | 1 |
| H3 − H0 | 3 | -0.999 | 0 |
| H4 − H0 | 3 | -0.188 | 1 |
| H5 − H0 | 3 | -0.418 | 0 |
| H6 − H0 | 3 | -0.095 | 0 |
| H7 − H0 | 3 | -0.453 | 0 |
| H8 − H0 | 3 | -0.165 | 1 |
| H9 − H0 | 3 | -0.243 | 0 |
| H10 − H0 | 3 | -0.220 | 1 |
| H4 − H1 | 3 | 0.009 | 3 |
| H4 − H2 | 3 | -0.175 | 2 |
| H7 − H5 | 3 | -0.036 | 0 |
| H8 − H4 | 3 | 0.023 | 1 |
| H9 − H8 | 3 | -0.078 | 1 |
| H10 − H8 | 3 | -0.055 | 2 |

Primary H9 success requires all three seeds and clean mean >=84%.
All 11 preregistered arms are reported. A complete arm meeting 84% is recognized separately; target scores never select or change training.
H0-P0 includes the optimizer, schedule, sampling, and additional training. Each treatment-H0 estimates the added mechanisms under equal optimizer-update budget.
SAM uses extra forward/backward passes; equal update budget is not equal compute. Averaging arms also change the fixed tail schedule.
Three seed differences are descriptive evidence, not an automatic statistical significance claim.
Parent, shared preparation, incremental training, and test costs are referenced separately in resources.json.
