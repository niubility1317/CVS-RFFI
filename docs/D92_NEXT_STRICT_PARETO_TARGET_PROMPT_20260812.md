# D92下一轮严格Pareto高效研发目标Prompt

你是`E:\type10-7`下CVS-RFFI/CV-SincNet Phase2方法研发主代理。请基于已完成实验直接研发并验证一个新候选，目标不是“某一项变好”，而是在最难同排矩阵上相对`E0_FULL_ONLY`实现八项严格Pareto改进，同时保持轻量部署。全程遵守`AGENTS.md`、`项目.md`和`p2_min_v1`；不要重复已通过的Phase2数据验证，不要增加与本实验正确运行无关的审计或发布框架。

## 一、必须吸收的实验事实

1. `E0_FULL_ONLY`是唯一正式基线。当前Hard10均值：`H=73.3472%`、`old BA=74.8611%`、`c_old_acc=74.8611%`、`old floor=44.8333%`、`seen_new_acc=72.0333%`、`forgetting=12.9167%`、`new_to_old=15.0417%`、`old_to_new=15.3333%`。
2. FloorBoost证明单向抬旧类可以改善old floor和遗忘，但代价过大：相对E0，old floor`+10.3333pp`、forgetting`-2.9444pp`，同时seen-new`-12.1583pp`、H`-6.0634pp`。因此禁止再次采用纯旧类bias、纯old-contrast增强或“保旧优先、再看新类”的串行方案。
3. NewGuard v3在10/10难例均成功激活且无fallback，但八项相对E0全部为`0.0000pp`；终态30个same-outer/same-scene资源配对的注册wall p90为`179.172ms`、配对中位比为`1.74784×`、peak最大增量为`5,951,488`字节。因此禁止继续微调NewGuard尺度、放宽保护容差或使用多轮codec回缩；“头字节改变但决策不变”视为无效路线。
4. 同一Hard10上的历史D92 FULL五项方向优于E0：H约`+0.8201pp`、old acc约`+1.2778pp`、old floor约`+3.6667pp`、seen-new约`+0.3417pp`、forgetting约`-1.2778pp`。这只作为机制方向证据，不能代替八项正式结果；优先研究如何以更低注册成本恢复D92中FULL/BLOCK互补几何的有效部分。
5. TCRA safe-v2的Hard9只通过3/8方向，8/9 outer预测完全不变；registration wall P90为`336.968ms`。因此禁止继续support-tail逐原子、逐prefix真实重评分和TPCE/TCRA同轴改名路线。
6. CSOAS的Hard9中old floor`+10.3704pp`、forgetting`-4.7222pp`，但H`-0.4233pp`、seen-new`-4.6667pp`、new→old`+3.3241pp`。因此禁止继续旧类偏置式协方差重估、Cauchy/OAS扫描或把新类损失解释为可接受交换。

## 二、冻结科学假设

唯一候选为`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`（CCOC），完整公式见`docs/superpowers/specs/2026-08-13-d92-ccoc-strict-pareto-design.md`。不得并行试跑第二候选。

候选必须满足：

1. 只读取同一注册support和联合封存的Phase1聚合知识；query及其任何view均不得参与fit、选择、调参、停止或回退。
2. 保留现有D92类均值和old/new group auto-shrinkage FULL协方差。raw逐类scatter只估计off-block跨类一致性，不得作为协方差端点直接平均或求逆。
3. 固定`160/96/32`三块。每类`Q_c=offblock(S_c)`、`u_c=Q_c/||Q_c||_F`；每组`rho_g`严格等于所有不同类`u_c`的平均pairwise Frobenius cosine并clip到`[0,1]`，无阈值、温度或扫描。
4. 最终`Sigma_g*=rho_g Sigma_g^auto+(1-rho_g)blockdiag(Sigma_g^auto)`，再固定old/new`0.5/0.5`，执行一次FULL solve和一次D42发布。禁止第二FULL/BLOCK fit、LOO、Fisher、rank-one task contrast、逐边/逐prefix搜索和多codec回缩。
5. 所有类必须具有finite且非零的off-block范数；任一失败、SPD/solve/codec失败或结构漂移时exact E0 fallback。不得丢弃弱类、加epsilon、jitter、伪逆或放宽容差。
6. old/new组内label permutation、support row permutation和old/new任务交换必须等变。不得使用类别顺序、receiver、scene、seed、K或new count特判。
7. G0发布前必须证明最终D42 state非E0、至少一个`rho∈(0,1)`，且以隔离E0/CCOC support-only执行的真实解码头计算，`max_j|Delta cross-group margin_j|`达到由两份state实际D42 block scale与同一support块幅度定义的一个量化量子；否则本地无效，不发布Hard9。

## 三、硬性能目标

正式性能先看与G0不重叠的9个冻结performance outer，K1仅作liveness。相对同排`E0_FULL_ONLY`，八项均值必须同时满足：

- `ΔH_old_new>0`
- `Δold_balanced_accuracy>0`
- `Δc_old_acc>0`
- `Δold_floor>0`
- `Δseen_new_acc>0`
- `Δaverage_forgetting<0`
- `Δnew_to_old_rate<0`
- `Δold_to_new_rate<0`

目标幅度：H`≥+1.00pp`、old BA`≥+1.50pp`、c_old_acc`≥+1.00pp`、old floor`≥+4.00pp`、seen-new`≥+0.50pp`、forgetting`≤-1.50pp`、两向混淆各`≤-0.50pp`。

稳定性最低要求：H、old floor、seen-new至少7/9 outer不下降；9个outer×6个旧类中不得出现系统性旧类退化；receiver、K/new_count和三种LEO weak场景的H与seen-new组均值不得下降。禁止用加权总分补偿任一反方向指标。

## 四、资源目标

- 最终query仍为单一F0线性头；query MAC和永久state必须与E0精确相同。
- K>2注册严格为FULL一次、dense solve一次、D42正式发布一次；BLOCK/LOO/Fisher fit均为0。
- 注册wall p90目标`≤120ms`且相对E0`≤1.25×`；硬上限仍为`≤150ms`且`≤1.50×`。
- 增量peak working set`≤512KiB`。
- 若性能八项全优但只超过资源目标、不超过硬上限，裁决`REVISE_ONCE`；超过硬上限直接拒绝。

## 五、最小高效实验流程

1. 已完成TCRA/CSOAS复盘和三方数学/新颖性/监督审查；CCOC已由用户批准冻结，不再等待同类流程性确认。
2. 本地只做新数学、真实D42部署、K1/K2 alias、query零访问、fit/resource收据和必要负测；不重跑数据验证，不扩通用框架。
3. 做一次独立P0/P1审查；P0=0、P1=0后立即Git提交、登记不可覆盖run ID并交给唯一N607 runner。P2只记录，不阻塞。
4. 第一轮只跑固定`rx_7_7/seed713106/K10/new5`三场景truth-free G0，不运行scorer。三场景机制、D42量子和资源全过后，自动进入与G0不重叠的Hard9+K1：9个performance outer、1个K1 liveness、每个3scene、8shard。
5. runner只按系统健康停止，绝不读取性能决定停机；Hard9完整取回后由独立analyzer一次性连接truth并裁决。
6. 最多允许两轮纯发布工程修复；不得把技术失败计作方法失败，也不得在同一run ID重启或覆盖。
7. Hard9任一八项均值方向不优于E0，或候选再次出现决策近零变化，立即`REJECT_ROUTE`，不跑125。八项方向全对但幅度、稳定性或资源目标未齐且硬上限未破，才允许`REVISE_ONCE`。全部通过后按本轮用户授权自动创建新的不可覆盖Target125发布；公式、协议、矩阵或阈值发生实质变化时仍须暂停说明。

## 六、报告与交付

交付必须包含：冻结公式和候选ID、实际fit图、真实D42部署行为、query/state/MAC、G0机制与资源表、Hard9逐outer同排表、八项均值、逐旧类floor/准确率、receiver/K/scene稳定性、注册wall/peak、最终裁决和下一步。涉及DA与注册的状态统一写成`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`；REG0的新类与H写`N/A`。

不要声称“设计上保证八项提升”。八项全优是实验晋级条件；唯一可信结论来自冻结候选在完整Hard10上的同排结果。
