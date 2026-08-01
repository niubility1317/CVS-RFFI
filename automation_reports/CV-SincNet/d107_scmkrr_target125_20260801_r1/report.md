# D107-SCMKRR/r1完整125研发与实验报告

状态：`DESIGN_FROZEN / IMPLEMENTING`

## 目标与终止背景

|字段|值|
|---|---|
|experiment ID|`d107_scmkrr_target125_20260801_r1`|
|日期/operator|2026-08-01；主agent负责集成、数据与结果分析；Terra Max子agent分工实现核心和执行面|
|目标|以机制独立的新方法完成完整125，并与D62、D91、D92、SVRN-qKNN-BCRR全面比较|
|claim scope|`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`|
|Git|仅本地提交；不push、不上传GitHub|

D106在r1-r7始终没有形成完整performance manifest，连续暴露执行接口缺陷。按用户要求，该路线已经终止，不创建r8。D107不得导入D106 K路由、RCMR qKNN或ReLU plus-view链。

## 冻结方法

候选：`D107-SCMKRR/r1`，即Support-Centered Mean-Embedding Kernel Ridge。

1. 只使用当前row的signed z_id support；所有行先L2归一化。
2. before的6个旧类support构成冻结domain anchor A；after新增类不能反向改变A。
3. 每个注册类构造归一化support均值原型p_c；K1直接使用唯一support，不估协方差、不fallback。
4. 带宽b为全部注册类原型两两平方距离的中位数，只允许机器精度下界totalization。
5. RBF核`k(x,y)=exp(-||x-y||²/b)`；以冻结A作RKHS均值中心化，消除当前receiver/scene公共分量。
6. 从密封Phase1 D106资产只读取tau/spectrum摘要生成无量纲ridge比率；ground原型不得直接参与query打分。
7. simplex目标`Y=I-11ᵀ/C`，`B=(K+λI)⁻¹Y`；每条query独立计算`k(q,P)ᵀB`并在全部注册类竞争。
8. 四臂固定：`M0=未中心化kernel prototype`、`M_DA=中心化kernel prototype`、`M_HEAD=未中心化simplex-KRR`、`M_JOINT=中心化simplex-KRR`。无ROUTED臂、无K路由。
9. 无query truth、role、quota、fit、update、selection、batch-count或global reassignment；old/new公式完全相同。

该方法不是D92的old/new协方差LDA，不是D62的Fisher行拼接，不是D91的OOF梯度残差，也不是D106/SVRN的qKNN/BCRR变体。K1仍产生新的非线性决策面。

## 资源预算

最大`C=26,K=10,|A|=60,D=160`时，建态核计算约0.934M MAC，单query约14.4k量级；持久数值状态预计约15.4KB，无梯度、epoch或新checkpoint。只复用已密封Phase1摘要和已验证target received-IQ输入。

## 冻结完整125

|维度|取值|
|---|---|
|receiver|`20-1,3-19,7-14,7-7,8-8`|
|seed|`713102,713103,713104,713105,713106`|
|slice|`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|arm/state|4臂；before/after|
|闭包|125 outer、375 scene、1500 arm pairs、3000 prediction surfaces、500 arm级聚合行|

K5必须复用matched K10物理池并只取前5shot；`receiver=3-19/rain/K1-new20`只作预登记压力切片，不用于选参或分支。

## 精简研发与发布流程

实现只分两个不重叠面：SCMKRR typed核心/数学负例；Target125矩阵、真实input、不可覆盖prediction和truth-side评分。主agent只做diff集成和最小聚焦验证。通过一个真实no-truth state smoke后直接发布完整125，不设置source-held性能gate、候选扫描或额外签名流程。只有协议/确定性执行故障可以停止，性能弱则完整结束后淘汰并进入下一方法。

## 性能目标与分析

沿用当前目标：K10三slice要求`A_old≥92%`、`F_old≥85%`、`N≥92/90/86%`；K5/new20相对matched K10/new20的`A/F/N/H`下降均≤5pp；K1/new20相对同row D92要求`ΔH≥2pp、ΔF≥2pp、ΔA≥0、ΔN≥0`且总正确严格增加。

完成后必须保留同rowreceiver、seed、slice、scene、before/after old、old floor、seen-new、H、forgetting和correct count。D62、D92、SVRN用完整125比较；D91单列15行development证据。
