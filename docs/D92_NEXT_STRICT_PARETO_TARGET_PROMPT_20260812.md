# D92下一轮严格Pareto高效研发目标Prompt

你是`E:\type10-7`下CVS-RFFI/CV-SincNet Phase2方法研发主代理。请基于已完成实验直接研发并验证一个新候选，目标不是“某一项变好”，而是在最难同排矩阵上相对`E0_FULL_ONLY`实现八项严格Pareto改进，同时保持轻量部署。全程遵守`AGENTS.md`、`项目.md`和`p2_min_v1`；不要重复已通过的Phase2数据验证，不要增加与本实验正确运行无关的审计或发布框架。

## 一、必须吸收的实验事实

1. `E0_FULL_ONLY`是唯一正式基线。当前Hard10均值：`H=73.3472%`、`old BA=74.8611%`、`c_old_acc=74.8611%`、`old floor=44.8333%`、`seen_new_acc=72.0333%`、`forgetting=12.9167%`、`new_to_old=15.0417%`、`old_to_new=15.3333%`。
2. FloorBoost证明单向抬旧类可以改善old floor和遗忘，但代价过大：相对E0，old floor`+10.3333pp`、forgetting`-2.9444pp`，同时seen-new`-12.1583pp`、H`-6.0634pp`。因此禁止再次采用纯旧类bias、纯old-contrast增强或“保旧优先、再看新类”的串行方案。
3. NewGuard v3在10/10难例均成功激活且无fallback，但八项相对E0全部为`0.0000pp`；终态30个same-outer/same-scene资源配对的注册wall p90为`179.172ms`、配对中位比为`1.74784×`、peak最大增量为`5,951,488`字节。因此禁止继续微调NewGuard尺度、放宽保护容差或使用多轮codec回缩；“头字节改变但决策不变”视为无效路线。
4. 同一Hard10上的历史D92 FULL五项方向优于E0：H约`+0.8201pp`、old acc约`+1.2778pp`、old floor约`+3.6667pp`、seen-new约`+0.3417pp`、forgetting约`-1.2778pp`。这只作为机制方向证据，不能代替八项正式结果；优先研究如何以更低注册成本恢复D92中FULL/BLOCK互补几何的有效部分。

## 二、首选科学假设

首选候选暂命名为`E0_FULL_BLOCK_PARETO_DISTILL`。如你能在不超过20行的可行性摘要中证明另一条路线更可能同时改善八项且更轻量，可以替换；否则直接执行本路线，不做多路线并行试跑。

候选必须满足：

1. 只读取同一注册support和联合封存的Phase1聚合知识；query及其任何view均不得参与fit、选择、调参、停止或回退。
2. 以E0 FULL头为主干，从同一support充分统计量中提取一个低成本BLOCK互补方向；最多允许一次额外的块求解，禁止K折重拟合、LOO重拟合、Fisher、Pareto枚举和多候选query择优。
3. 将FULL与BLOCK互补信息一次性蒸馏为一个最终线性头。允许旧类行和新类行共同变化；必须保持old组内、new组内类别置换等变，不得使用类别顺序、receiver、scene、seed或query角色特判。
4. 推荐使用无人工权重扫描的词典序支持目标：先最大化“六个旧类CVaR20 margin增益”和“新类q20 margin增益”二者的最小值，再最小化new→old与old→new双向hinge代理，最后最小化相对E0头的变化范数。支持交叉评估可以做，但不得重新拟合FULL/BLOCK。
5. 直接面向D42部署格点求解或投影，最多一次最终codec回环加一次确定性局部格点修正；禁止NewGuard式20级scale搜索。最终部署头若与E0 byte-exact，或所有support跨组margin变化均小于一个真实D42量化步，则判定本地无效，不发布N607实验。
6. 数值失败、结构漂移或部署保护失败时exact E0 fallback；不得降低既有query协议门，也不得用放宽容差换取激活。

## 三、硬性能目标

正式性能只看10个冻结performance outer，K1仅作liveness。相对同排`E0_FULL_ONLY`，八项均值必须同时满足：

- `ΔH_old_new>0`
- `Δold_balanced_accuracy>0`
- `Δc_old_acc>0`
- `Δold_floor>0`
- `Δseen_new_acc>0`
- `Δaverage_forgetting<0`
- `Δnew_to_old_rate<0`
- `Δold_to_new_rate<0`

目标幅度：H`≥+1.00pp`、old BA`≥+1.50pp`、c_old_acc`≥+1.00pp`、old floor`≥+4.00pp`、seen-new`≥+0.50pp`、forgetting`≤-1.50pp`、两向混淆各`≤-0.50pp`。

稳定性最低要求：H、old floor、seen-new至少8/10 outer不下降；10个outer×6个旧类中不得出现系统性旧类退化；receiver、K/new_count和三种LEO weak场景的H与seen-new组均值不得下降。禁止用加权总分补偿任一反方向指标。

## 四、资源目标

- 最终query仍为单一F0线性头；query MAC和永久state必须与E0精确相同。
- K>2注册最多`FULL一次+低成本BLOCK一次`，不得出现随K线性增长的组件fit。
- 注册wall p90目标`≤120ms`且相对E0`≤1.25×`；硬上限仍为`≤150ms`且`≤1.50×`。
- 增量peak working set`≤512KiB`。
- 若性能八项全优但只超过资源目标、不超过硬上限，裁决`REVISE_ONCE`；超过硬上限直接拒绝。

## 五、最小高效实验流程

1. 先用不超过20分钟完成一次短复盘，只读取E0、FloorBoost、NewGuard v3报告和对应analysis；输出不超过20行的`FEASIBILITY_REVIEW`，随后立即冻结一个候选。
2. 本地只做新数学、真实D42部署、K1/K2 alias、query零访问、fit/resource收据和必要负测；不重跑数据验证，不扩通用框架。
3. 做一次独立P0/P1审查；P0=0、P1=0后立即Git提交、登记不可覆盖run ID并交给唯一N607 runner。P2只记录，不阻塞。
4. 第一轮只跑现有冻结Hard10+K1：10个performance outer、1个K1 liveness、每个3scene、8shard。复用E0历史paired/raw/per-old基线，不重跑E0或D92。
5. runner只按系统健康停止，绝不读取性能决定停机；完整取回后由独立analyzer一次性连接truth并裁决。
6. 最多允许两轮纯发布工程修复；不得把技术失败计作方法失败，也不得在同一run ID重启或覆盖。
7. Hard10任一八项均值方向不优于E0，或候选再次出现八项全零，立即`REJECT_ROUTE`，不跑125。八项方向全对但幅度、稳定性或资源目标未齐且硬上限未破，才允许`REVISE_ONCE`。全部通过才`ADVANCE_TO_TARGET125_CANDIDATE`，仍不得自动启动125。

## 六、报告与交付

交付必须包含：冻结公式和候选ID、实际fit图、真实D42部署行为、query/state/MAC、Hard10逐outer同排表、八项均值、逐旧类floor/准确率、receiver/K/scene稳定性、注册wall/peak、最终裁决和下一步。涉及DA与注册的状态统一写成`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`；REG0的新类与H写`N/A`。

不要声称“设计上保证八项提升”。八项全优是实验晋级条件；唯一可信结论来自冻结候选在完整Hard10上的同排结果。
