# ADVB02 CRRA-S LEO弱信道增强设计

## 已确认目标

在当前Phase1 source-only数据协议内，实现并发布`ADVB02_CRRA_S_LEO_WEAK_E200`。它以`ADV3B02_CORE90_SOFT_E200`为训练主干，只在身份路径加入CRRA-S，以提高弱LEO残余信道和接收机链路扰动下的星地识别性能。

本设计取代旧CRRA草案中“训练使用`mixed_orbit`”的场景选择。旧`mixed_orbit`记录仅作历史对照，不能被用于本次训练、选模或测试结果。

## 冻结的协议和实验配置

- Phase1角色固定为`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，TX、receiver和物理样本ID保持既有互斥约束。
- 训练与最终测试只使用弱LEO族：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。不使用`mixed_orbit`。
- 星地增强日程严格复用`ADV3B02_CORE90_SOFT_E200`：E1–40为`leo_clear_weak,p=0.30`；E41–90为`leo_low_elev_weak,leo_rain_weak,p=0.60`；E91–200为三场景并集`p=0.80`。
- 训练超参数继承Core90：200 epochs、label/pseudo=`130/70`、AdamW、`lr=2e-4`、`weight_decay=1e-4`、无学习率调度、`lite_d`、`branch_ablation=no_dac`、RCN、EMA和原Core90损失项。
- 本次随机种子为`392034`。最终选择的checkpoint必须独立完成clean对照和三种LEO弱场景逐场景测试。
- Phase1不访问target receiver、target query、target calibration或任何Phase2适配状态。CRRA-C不启用。

## CRRA-S结构

CRRA-S只接入身份分支的共享Sinc/IQ后时间特征。域分支继续读取原始共享特征；PA分支继续读取原始IQ并完全旁路CRRA。不得在频率或PA分支额外叠加一个等价的通用CRRA层。

1. 对每个I/Q滤波器对独立计算2×2协方差收缩白化。白化输出必须保留原均值，且强度由每个I/Q对各自的`alpha_j∈[0,0.25]`控制。
2. 残差为`DWConv(k=5)→rank=8→FiLM(gamma,beta)→up`。`gamma,beta`由条件向量产生；上投影零初始化，保证早期近似恒等。
3. 条件向量为`q=Pq([RCN(raw_IQ),GAP(F_s)])`，其输入和用于门控的`q`均stop-gradient。不得使用`z_dom`。
4. 支持门保存源域多中心的对角Mahalanobis统计；支持度为`exp(-d²/tau)`，只允许由训练源样本更新。
5. 时间、频率和PA特征以`q`条件的残差可靠度进行凸融合；PA本身不白化、不重构。
6. 输出诊断至少包括逐对alpha、总gate、修正能量、支持距离、条件TX对抗准确率及三支路可靠度。

## 损失和日程

- E1–16：CRRA严格恒等，不启用CRRA损失。
- E17–46：线性ramp CRRA及其损失。
- E47–200：固定强度，CRRA参数组学习率固定为主学习率的0.25。
- 有效卫星一致性KL权重只有一个来源：`lambda_sat_cons=0.05`。`lambda_crra_sat_kl`只可作为兼容别名，不能与前者叠加。
- 其余CRRA权重：pair=`0.05`、energy=`0.001`、gate L1=`0.001`、nuisance=`0.02`、q TX adversarial=`0.02`、shell=`0.0`。
- 同一清洁/卫星对来自一次既有星地信道生成，不生成第二个卫星视图。

## 最终评估与结果边界

每次完整训练后，自动运行独立checkpoint评估：clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`必须分别保留指标和日志，不能用平均值替代逐场景结果。

除准确率外，评估输出以下诊断：clean-satellite视图距离、跨域同类半径、修正能量、逐对alpha和gate、支持距离、q的TX泄漏准确率、时间/频率/PA可靠度权重。诊断仅用于解释同一行结果。

结果只能表述为当前source-only Phase1协议下、三种弱LEO残余信道代理的鲁棒性证据；不能表述为真实卫星全链路性能或已经消除所有接收机干扰。
