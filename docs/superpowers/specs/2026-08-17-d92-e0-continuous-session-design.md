# D92 E0连续类注册session实验设计

## 状态

`DESIGN_FROZEN / DEVELOPMENT_ONLY_CONTINUOUS_SESSION_SCREEN`

用户已授权直接执行实验。本设计只验证D92 E0能否按一个或少数新类分批到达并持续注册，以及这种部署方式相对一次性注册的精度、遗忘、时间和资源影响；不改变现有一次性E0基线，不构成Target125正式性能声明。

## 1. 目标与核心判断

现有D92 E0只有一次`before/after`生命周期，部署状态不保存类均值或协方差，因此不能从终态量化头中精确追加新类。连续版本采用“冻结DA锚点＋累计support重放”：旧类DA状态只生成一次；每个session只开放本次到达的新类support，并与此前已开放support合并，重新执行一次E0 FULL闭式注册和一次D42发布。

这不是低成本的单行追加。它换取的是：

- 每个session立即得到全已注册类竞争的合法模型；
- 第5个新类到齐时，使用同一support并集、同一规范类顺序和同一公式，终态必须与一次性`[5]`注册逐字节等价；
- 不把query、truth、role、quota或全局重分配写入注册状态；
- 代价是保留累计support并重复注册，累计注册时间预计高于一次性注册。

## 2. 冻结方法

方法ID：`D92_E0_CUMULATIVE_REPLAY_SESSION_V1`

共同规则：

1. `DA1_REG0`只生成一次旧类`log_diag`、旧类注册表和旧类量化头；后续session不得重新做域适应。
2. 每个新类的K10 support包在其session到达前不可打开；到达后记录不可变类handle、物理support token和包身份，token不得重复。
3. session `s`只用旧类support和截至`s`已到达的新类support；所有行按类handle和support字节规范排序，禁止用到达顺序、receiver、scene或truth分支。
4. 每个session重新计算累计support的D81 support-only变换、D92 task-balanced FULL统计量、一个288维solve，并执行一次D42 codec；不得做BLOCK、LOO、扫描、回缩或第二次codec。
5. 新类数`n=2..4`时，使用原始D92公式的直接前缀扩展：分别对6个旧类和`n`个新类做相同`sklearn lsqr/auto`组内协方差估计，再固定`0.5/0.5`合并。
6. 新类数`n=1`时，sklearn LDA不能拟合单标签组，因此使用唯一临时桥接：对该类K10中心化残差调用与`shrinkage=auto`相同的StandardScaler＋Ledoit-Wolf协方差原语，作为`Sigma_new_auto`；仍固定`Sigma=0.5*Sigma_old_auto+0.5*Sigma_new_auto`。无参数、无扫描。
7. 新类数`n=5`时必须调用原始D92 E0统计构造器；连续终态的state SHA、query预测和指标必须与一次性`[5]`完全相同，否则实验拒绝。
8. 任一非有限、协方差非正定、token重复、未来support访问、类/K/场景漂移或不可覆盖输出，均作为技术失败停止该run；不得改规则后续跑同一run ID。

## 3. session状态与数据边界

允许跨session保留：

- `DA1_REG0`冻结状态、旧类注册表、checkpoint/capsule/seal身份；
- 已到达support的类handle、K、物理token、不可变包身份和累计token账本；
- 当前D42量化头、session序号、资源和零访问收据。

禁止跨session保留：

- query特征、query预测、query truth；
- receiver/scene角色Oracle、真实batch类数、class quota、全局重分配信息；
- 未到达类的support内容；
- scorer输出或用性能结果选择顺序/桥接。

## 4. 冻结实验矩阵

数据：既有`p2_min_v1 / VALIDATED_ONCE`封存行；K10；old=6；new=5；三种`leo_*_weak`场景；seed=713106；receiver outer为`20-1`、`3-19`、`7-14`、`7-7`、`8-8`。

每个outer和scene执行四种预登记调度：

| schedule | session增量 | 用途 |
|---|---:|---|
| `batch_5` | `[5]` | 原始一次性E0参考 |
| `singleton_forward` | `[1,1,1,1,1]` | 一个类一次的连续注册 |
| `singleton_reverse` | `[1,1,1,1,1]`反序 | 检查到达顺序不变性 |
| `chunk_2_2_1` | `[2,2,1]` | 少数类一批的连续注册 |

每个session在同一固定query token集合上独立预测所有样本，但预测阶段不读truth。离线truth-last scorer只计当前已注册真类；尚未注册真类标记为`UNREGISTERED_NOT_SCORED`，不得作为unknown或错误样本混入。

## 5. 指标与裁决

生命周期标签为`DA1_REG0`、`DA1_REG1_S1`…`DA1_REG1_S5`；本实验固定DA，不生成DA0对照，因此`DA0_REG0/DA0_REG1=N/A`。

每个session报告：

- old BA、old class floor、forgetting；
- 当前已注册new accuracy、H_old_new；
- old→new和new→old；
- 本session注册wall、实测增量peak、累计wall、累计support字节；
- query state bytes、logical MAC和head latency。

主裁决：

1. `singleton_forward`、`singleton_reverse`和`chunk_2_2_1`的S5必须与`batch_5`终态state/prediction/八项指标完全相同。
2. 单session注册wall目标`<=150ms`，增量peak硬门按用户放宽为`<=2MiB`；资源失败保留完整产物并判`REJECT_RESOURCE`，不得通过改阈值重跑同一run。
3. query state bytes和logical MAC必须与相同注册类数的一次性E0完全相同；推理端不允许新增持久状态或MAC。
4. S1单类桥接单独报告，不用它替代S5原始E0性能结论。

## 6. 预期与风险

终态性能理论上应与一次性E0完全相同；连续化本身不承诺提升精度。它的价值是降低单次必须凑齐全部新类的业务延迟，并给出每次到达后的可用模型。代价是累计重放：`[1x5]`需5次FULL注册，累计wall预计约为一次性注册的3至5倍；但每次query计算与状态格式不变。

最大风险是S1单类协方差估计方差较大，可能短期伤害旧类floor或new→old；这正是轨迹实验要测量的对象。S2以后回到多类原始公式，S5必须消除所有顺序差异。
