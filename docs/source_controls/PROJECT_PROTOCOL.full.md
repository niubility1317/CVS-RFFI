# CVS 项目场景与数据协议

版本：2026-06-24  
适用范围：`E:\type10-7` 中 CVS-RFFI / CV-SincNet 的科研叙事、算法优化、实验矩阵、数据协议、自动化控制面、N607 运行设计、论文/汇报写作与结果解释。

## 1. 文件地位

本文件是 CVS 项目的场景与数据协议源文件。后续任何涉及 CVS 研究方向、实验设计、候选矩阵、数据划分、Stage2-A/B/C 解释、论文叙事、优化路线、自动化决策或指标口径的工作，都必须先读取并遵守本文件。

`AGENTS.md` 仍然拥有工程安全、N607 访问、环境、报告、版本管理和远端同步规则的最高优先级。本文件拥有 CVS 科研场景、数据协议、任务边界和声明口径的优先级。若本文件与旧报告、旧记忆、旧 prompt、旧矩阵或历史 launcher 冲突，以本文件为准；若本文件与 `AGENTS.md` 冲突，以 `AGENTS.md` 为准，并在报告中记录冲突。

## 2. 项目主场景

CVS 面向的主场景是：

> 天基射频指纹识别中的弱标注跨接收机域泛化与在轨跨域少样本适应。

这个场景由四个约束共同定义：

1. **星上算力受限**：卫星端难以开展完整模型训练。项目采用“地面训练、天上部署”架构。地面端完成主干训练、模型选择、部署包、原型库和初始阈值；星上端只允许推理、原型更新、轻量校准、阈值微调或小 adapter 更新。
2. **身份标签极少但域标签丰富**：广域接收能产生大量 raw IQ，但发射机身份标签难以可靠获得。地面训练阶段 TX 身份标注比例低于 `0.1`；无 TX 标签样本可以大量存在，并携带 receiver、day、rx_day、观测窗口、信道场景或其他 domain label。
3. **星地信道破坏发射机特征**：星地链路中的残余 Doppler/CFO、相位噪声、低 SNR、低仰角、弱多径和弱 Rician/shadowed-Rician fading 会破坏 raw IQ 中的发射机相关细节。路径损耗和绝对 slant range 默认只用于 metadata/link budget，不得作为强幅度捷径直接主导 IQ；IQ imbalance 和接收链路差异归入 `R_d` 接收机响应，不再作为默认传播信道 `H_d` 扰动。clean view 不能代表部署成功。
4. **在轨部署遇到旧类和新类少样本**：部署后，卫星接收机可能获得少量旧类带标签样本，也可能获得少量新辐射源带标签样本。旧类是地面训练已知 TX；新类必须是地面训练未见 TX。二者都必须来自同一个部署卫星接收机域，并叠加星地信道扰动。

项目不应被表述为普通 WiSig 少样本分类、普通全监督域泛化、纯 few-shot learning，或真实卫星部署已完成验证。

## 3. 核心科学问题

CVS 要回答的问题是：

> 在身份标签极少、无标签信号大量存在、接收机域变化强、星地信道破坏严重、部署后还会出现新辐射源的条件下，如何先在地面学习稳定的发射机身份表征，再在星上用少量样本完成旧类校准、新类注册和未知类拒识。

该问题的核心矛盾是 identity-style conflict：发射机硬件指纹、接收机响应、日期漂移、采集域差异和星地信道扰动在 raw IQ 中纠缠。CVS 的价值不在于堆叠多个 DG trick，而在于用物理先验和身份-域双表征控制这种纠缠。

## 4. 符号与集合定义

设接收 IQ 样本为 `x`，发射机身份为 `y`，域标签为 `d`。天基 RFFI 观测可抽象为：

```text
x = R_d( H_d * T_y(s) ) + n
```

其中：

- `T_y`：发射机硬件非理想性，是应保留的身份来源。
- `H_d`：传播/星地信道，是域扰动来源。
- `R_d`：接收机链路响应，是跨接收机偏移来源。
- `n`：噪声。

接收机集合：

```text
R_s = {r_1, ..., r_m}        # 地面/source training receivers
R_t = {r_a, ..., r_b}         # target receiver domain / deployment proxy domain
intersection(R_t, R_s) = empty
|R_t| >= 1
```

`R_t` 是第二阶段的 target receiver domain `R_t`。`R_t` and `R_s` must be disjoint；single `r_sat` is allowed but not mandatory。关键不是强制单接收机，而是保证 target receiver domain 不泄漏到地面训练 receiver 域，并且后续 target-old / target-new / unknown 的 support/query 权限都按 `R_t` 定义。

发射机集合：

```text
Y_old = ground-training transmitter set
intersection(Y_new, Y_old) = empty
intersection(Y_unknown, union(Y_old, Y_new)) = empty
```

严禁把与地面训练 TX 一致的发射机写成“新类”。如果一个 TX 在地面训练中出现过，它在部署阶段只能是旧类 target-old；它的问题是目标域校准或少样本域适应，不是新类注册。

## 5. 地面训练阶段协议

地面训练阶段是 weakly labeled / semi-supervised source-domain DG，不是部署意义上的 few-shot learning。

训练数据分两类：

```text
L_s = {(x_i, y_i, d_i): receiver(x_i) in R_s}
U_s = {(x_j, d_j): receiver(x_j) in R_s, y_j hidden or unavailable}
```

身份标注比例：

```text
rho_label = |L_s| / (|L_s| + |U_s|) <= 0.1
```

推荐实验网格：

```text
rho_label in {0.005, 0.01, 0.02, 0.05, 0.1}
```

地面阶段不得使用 `R_t` 的任何样本、统计、BN 信息、阈值、prototype、adapter、伪标签、验证结果或 early stopping 信息。只要目标接收机域数据参与训练或模型选择，该结果就不再是 source-only DG，必须单独标为 DA / TTA / few-shot adaptation。

无 TX 标签样本的使用边界：

- 若 `U_s` 的类别空间确认属于 `Y_old`，可使用闭集 semi-supervised learning，例如 Mean Teacher、FreeMatch/UPS、prototype agreement、class quota、receiver quota。
- 若 `U_s` 可能混入 `Y_old` 外发射机，必须使用 open-set SSL / reject 机制；不能把所有无标签样本强制伪标成旧类。
- 无标签样本的 domain label 可用于 `z_dom` 域监督、`z_id` receiver leakage 抑制、采样平衡、伪标签覆盖审计和 episode 组织。

## 6. CVS 地面训练架构

CVS 地面阶段目标是学习跨域稳定的发射机身份空间：

```text
raw IQ -> CV-SincNet/CVS -> z_id, z_dom
```

- `z_id`：保留发射机硬件指纹，用于分类、原型、少样本注册和旧类校准。
- `z_dom`：吸收 receiver、day、rx_day、channel、satellite-style nuisance，用于域诊断、域监督、adapter gate 和泄漏审计。

推荐机制：

| 模块 | 用途 | 边界 |
|---|---|---|
| 物理先验 CV-SincNet | 保留 raw IQ 中的硬件非理想线索 | 不应被替换成无物理解释的纯黑箱 backbone，除非作为 baseline |
| `z_id/z_dom` 解耦 | 分离身份和域因素 | 不能让 `z_dom` 参与 TX prototype 距离 |
| domain-supervised `z_dom` | 利用无 TX 标签样本的域标签 | 域标签不能替代 TX 标签 |
| GRL / leakage probe | 抑制 `z_id` 中的接收机信息 | 过强会抹掉 TX identity，需报告 probe |
| Mean Teacher / FreeMatch / UPS | 利用无 TX 标签样本 | 必须有不确定性、原型一致性和 quota |
| Prototype agreement | 防止 logit shortcut | 原型更新需 class/receiver balance |
| MLDG / episodic source split | 模拟未见接收机外推 | support/query 只能来自源域 |
| satellite strong-view consistency | 提升星地压力鲁棒性 | 只能使用源域派生视图，不能使用目标卫星接收机 |

## 7. 在轨部署阶段协议

在轨阶段面对目标接收机域 `R_t`。`R_t` 可以是一个接收机，也可以是多个接收机组成的 deployment proxy domain；所有 target-old、target-new、unknown 的 support/query 都必须来自 `R_t`，并处于相同定义的 satellite/LEO target view 下。

Phase2 样本选择的核心约束：target receiver domain 必须与地面训练接收机域不同，同时 must include target-old samples from `Y_old` and target-new samples from `Y_new`。其中 target-old 是与地面训练发射机类别相同的旧类，target-new 是与训练发射机类别不同且不在 `Y_old` 中的新类。若 `R_t` 中缺少旧类或新类样本，不能声明完整 Stage2-C；相关 row 必须降级为 Stage2-A/B、`LOCAL_DATASET_EXTENSION_REQUIRED`、`LOCAL_PROTOCOL_REPAIR_REQUIRED` 或 `NON_LAUNCH_DIAGNOSTIC`。

K-shot 设置不再是硬枚举。每个 Stage2-B/C run 必须显式记录 `K`，且 `K` 必须为正整数。

```text
K >= 1
recommended anchor K in {1, 2, 5, 10, 15, 20, 50}
```

报告口径：

- `{1,2,5,10,15,20,50}` 是推荐锚点，用于跨 run 对齐和画主曲线，不是唯一允许取值。
- 自动化可选择 `3/4/8/12/16/25/30` 等中间值或任务驱动值，但必须记录 `k_shot`、选择理由和报告分层。
- `K<=20` 可作为 few-shot / low-shot 区间报告。
- `K>20` 应写作 higher-shot、medium-shot 或 saturation point，不能把所有结论都笼统称为 strict few-shot。

部署阶段默认冻结：

```text
freeze:
  CV-SincNet backbone
  z_id extractor
  source classifier or source prototype bank

update:
  target-old prototype shrinkage
  target-new prototype
  temperature / bias / threshold
  small adapter or BN affine only when explicitly allowed
```

星上禁止默认采用 full-model fine-tuning、完整 MAML inner loop、无门控伪标签重放、强在线 GRL/DANN、大型生成模型增强、无回滚持续自训练。

## 8. WiSig / ManySig 数据协议

### 8.1 数据集角色

- 地面训练使用 WiSig/ManySig 中的源接收机域数据。
- 在轨推理少样本阶段使用同一数据体系下的目标卫星接收机代理域，或明确保留的其他 TX 子集。
- WiSig/ManySig 是 terrestrial proxy benchmark / ground-accessible source domain family，不是真实卫星训练集。
- satellite-channel augmentation 和 satellite stress 是 physics-informed deployment stress，不是真实在轨 IQ 验证。

### 8.2 接收机划分

主协议必须使用与 source training receivers 不相交的目标接收机域：

```text
source receivers: R_s
target receiver domain: R_t
intersection(R_t, R_s) = empty
|R_t| >= 1
```

示例：

```text
source receivers: rx0-rx6
target receiver domain: rx7-rx11
```

`R_t` 可以只包含一个 receiver，例如 `{rx7}`，也可以包含多个 receiver，例如 `{rx7, rx8, rx9, rx10, rx11}`。自动化不得再以 exactly-one `r_sat` 作为 launchability gate；它必须检查 `R_t` 是否与 `R_s` disjoint，并检查 `R_t` 中是否存在目标阶段需要的 old/new/unknown 样本覆盖。多接收机目标域的结果必须报告为 target receiver domain / deployment proxy domain，不再写成“单星接收机”证据。

### 8.3 发射机划分

旧类：

```text
Y_old:
  used in ground training
  used for target-old support/query on R_t
```

新类：

```text
Y_new:
  not used in ground training
  used for Stage2-C seen-new support/query on R_t
```

未知类：

```text
Y_unknown:
  not used in ground training
  not used as Stage2-C seen-new support
  used only for rejection query on R_t
```

若当前 ManySig 仅提供六个旧类 TX，可采用：

```text
Y_old = {0,1,2,3,4,5}
Y_new = held-out non-ManySig TX set from another WiSig subset
Y_unknown = held-out non-ManySig TX set disjoint from Y_new
```

如果本地数据暂时不能提供非旧类 TX，Stage2-C 不能声明新类识别，只能做 Stage2-A/B，或标为 `LOCAL_DATASET_EXTENSION_REQUIRED`。

### 8.4 已确认的 Phase2 WiSig 候选样本池

当前 N607 已确认存在以下原始 WiSig compact subsets：

```text
/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl
/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyTx.pkl
/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyRx.pkl
/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/SingleDay.pkl
```

本地若只存在 synthetic smoke features，不能作为部署证据；真实 Phase2 样本选择应以 N607 上的 pkl 为准，或先把真实 pkl 显式同步/登记到本地。

当前主候选池定义如下：

```text
Y_old = {14-10, 14-7, 20-15, 20-19, 6-15, 8-20}  # ManySig old/source TX
R_s = {1-1, 1-19, 14-7, 18-2, 19-2, 2-1, 2-19}  # ManySig source receivers
```

Phase2 matrix generator should first choose target receiver domains from this confirmed pool before proposing new dataset routes:

| target receiver label in `R_t` | ManySig target index | ManyTx receiver index | target-old source | target-new / unknown source | confirmed non-old TX count in ManyTx | confirmed non-old eq1 samples |
|---|---:|---:|---|---|---:|---:|
| `20-1` | 7 | 10 | ManySig / `Y_old` | ManyTx / non-`Y_old` TX | 144 | 27,638 |
| `3-19` | 8 | 12 | ManySig / `Y_old` | ManyTx / non-`Y_old` TX | 143 | 26,887 |
| `7-14` | 9 | 13 | ManySig / `Y_old` | ManyTx / non-`Y_old` TX | 137 | 26,445 |
| `7-7` | 10 | 14 | ManySig / `Y_old` | ManyTx / non-`Y_old` TX | 142 | 26,868 |
| `8-8` | 11 | 17 | ManySig / `Y_old` | ManyTx / non-`Y_old` TX | 140 | 26,474 |

这些候选满足 Phase2 样本选择核心条件：`R_t` 与 `R_s` 不相交，`R_t` 中存在与训练 TX 相同的 target-old 样本，也存在与训练 TX 不同的 target-new / unknown 样本。矩阵生成时必须按 receiver label 对齐不同 pkl 的 `rx_list`，不能把 ManySig 的 index 直接当成 ManyTx 的 index。

`ManyTx` 是当前 Stage2-C target-new / unknown 的主来源。`ManyRx` 可作为 receiver-rich 或低新类数量 control；`SingleDay` 可作为 single-day smoke/control，但它缺少 `3-19` target receiver，且只有单日样本。若使用 `ManyRx` 或 `SingleDay` 替代 `ManyTx`，矩阵必须显式标明这是 control/sensitivity 设计，而不是默认主线 Stage2-C 候选。

生成 Stage2-C 时，`ManyTx` 中 non-`Y_old` TX 必须再拆分成互斥的 `Y_new` 与 `Y_unknown`；不得把同一个 TX 同时用于 seen-new enrollment 和 unknown rejection query。每个 row 还必须记录 `target_receiver_ids`、`source_receiver_ids`、`target_old_tx_ids`、`target_new_tx_ids`、`target_unknown_tx_ids`、support/query 划分、`K`、satellite/LEO view 和阈值选择 label scope。

`ManyTx` 行的 `target_new_tx_ids`、`unknown_tx_ids` 以及对应 `target_new_tx_labels`、`unknown_tx_labels` 必须是从 `ManyTx.pkl` 的 `tx_list` 解析出来的真实 TX label（例如 `1-16`、`10-1`），不得使用 synthetic numeric IDs、rank 占位或“稍后解析”的说明文字作为 launchable 字段。矩阵生成器在写入 launchable row 前必须按目标 receiver label 预筛每个 `Y_new` / `Y_unknown` TX 的可用样本数，确认满足该 row 的 support/query 需求；只记录 aggregate non-old TX count 或 aggregate samples 不足以证明 launchable。若任何 TX 与 `Y_old` 重叠、无法按 `ManyTx.tx_list` 解析、在目标 receiver 下样本不足或过滤后会产生空 tensor，该 row 必须标为 `LOCAL_PROTOCOL_REPAIR_REQUIRED` / `LOCAL_DATASET_EXTENSION_REQUIRED` / `NON_LAUNCH_DIAGNOSTIC`，不得发往 N607 runner。

### 8.5 星地信道视图

deployment support/query 必须按 satellite/LEO 视图报告。自 2026-06-24 起，主协议采用简化 LEO 残余信道版本：LEO-only 几何，高度、仰角和 slant range 只用于 metadata/link budget；完整 Doppler 视为已由星历/同步补偿，IQ 上只叠加 residual Doppler/CFO；默认使用 mild phase noise、flat/Rician fading、AWGN/SNR、RMS/AGC normalization，并加入弱多径。低仰角或边界场景可以使用弱 shadowed-Rician，但不得默认启用 severe LOO、强风暴多径、MEO/GEO 混合或未补偿完整 Doppler。

推荐主报告视图：

```text
target_channel_view in {
  leo_clear_weak,
  leo_low_elev_weak,
  leo_rain_weak
}
```

旧场景 `clear_leo`、`low_elev_leo`、`rain_leo`、`storm_mp`、`mixed_orbit` 保留为 legacy stress/control。`storm_mp` 和 `mixed_orbit` 只能作为 diagnostic/sensitivity，不再作为默认 deployment-primary 成功门槛。

clean view 是 control/reference。不得把 clean view 成功提升为 satellite/LEO deployment success。

## 9. Stage2-A/B/C 协议

### 9.1 Stage2-A：Zero-label deploy

目的：无目标标签时测试旧类识别和非旧类拒识。

```text
support:
  empty

query on R_t:
  target-old query from Y_old
  target-new/unknown query from Y_new or Y_unknown
```

允许声明：

- old-class target recognition。
- non-old rejection。
- unknown FAR / FPR95 / AUROC。

禁止声明：

- new identity recognition。
- target-label threshold fitting。
- seen-new accuracy。

Stage2-A 是部署安全底线，不是新类注册。

### 9.2 Stage2-B：Old-class few-shot calibration

目的：在同一个卫星接收机域下，用少量旧类标注样本提升旧类目标域识别，并保持 unknown FAR。

```text
support on R_t:
  K shots per old TX from Y_old

query on R_t:
  held-out target-old query from Y_old
  target-new/unknown rejection query from Y_new/Y_unknown
```

允许声明：

- target-old full accuracy / accepted accuracy。
- old_acc_delta_pp。
- old retention。
- rescue / harm / net_gain。
- unknown FAR / FPR95 / AUROC 不恶化。

禁止声明：

- seen-new identity accuracy。
- target-new support 使用。
- unknown query 参与阈值拟合。

### 9.3 Stage2-C：Old + seen-new enrollment

目的：在同一个卫星接收机域下，同时利用旧类 support 校准和新类 support 注册。

```text
support on R_t:
  K shots per old TX from Y_old
  K shots per seen-new TX from Y_new

query on R_t:
  held-out target-old query from Y_old
  held-out seen-new query from Y_new
  unseen-new/unknown query from Y_unknown
```

允许声明：

- target-old performance。
- seen-new identity accuracy。
- `H_old_new`。
- unknown FAR under constraint。
- output semantics: old label, seen-new label, reject, uncertain, defer。

禁止声明：

- 把 `Y_unknown` 当 seen-new 识别。
- 用 unknown query 调阈值。
- 把 clean-view success 写成 deployment success。

Stage2-C 是主线目标，但只有在 `Y_new` 与 `Y_old` 不相交、`R_t` 与 `R_s` 不相交、并且 target-old 与 target-new support/query 都来自 `R_t` 时才成立。

## 10. 指标与成功判据

### 10.1 地面训练阶段

- strict UDU / unseen receiver-day accuracy。
- worst receiver / receiver floor。
- pseudo-label audit precision。
- pseudo-label coverage by class and receiver。
- `z_id -> receiver` leakage probe accuracy。
- satellite stress mean/floor，作为 deployment-oriented validation-control。

### 10.2 Stage2-B

- target-old full accuracy。
- target-old accepted accuracy + coverage。
- old_acc_delta_pp。
- rescue / harm / net_gain。
- unknown_FAR <= 0.05。
- FPR95 / AUROC。
- rollback trigger rate。

### 10.3 Stage2-C

- seen_new_acc。
- old_acc。
- `H_old_new`。
- unknown_FAR <= 0.05。
- new_acc_drop_pp <= 2 pp，或明确标为 exploratory。
- old->new, new->old, unknown->new confusion。
- latency / memory / prototype storage。

## 11. 论文与报告声明边界

可以写：

- CVS 面向天基 RFFI 的弱标注跨接收机 DG 与在轨跨域 few-shot 适应。
- WiSig/ManySig 是地面可接入源域代理。
- satellite stress 是物理启发部署压力测试。
- Stage2-B 是旧类目标域校准。
- Stage2-C 是 seen-new enrollment，前提是 `Y_new` 与 `Y_old` 不相交。

不能写：

- WiSig/ManySig 是真实卫星训练集。
- satellite augmentation 等价于真实在轨验证。
- source-only DG 等价于 few-shot learning。
- 旧类 target support 带来的提升是新类识别。
- Stage2-A/B 的 rejection 结果是 seen-new identity accuracy。
- target receiver domain 与 source receiver domain 重叠后仍称为部署泛化。
- 缺少 target-old 或 target-new 样本覆盖时仍声称完整 Stage2-C。

## 12. 自动化控制面约束

CVS 自动化不只是工程调度层，也承担实验语义落地责任。任何 monitor、optimizer、runner、launcher、matrix generator、validator、registry、state ledger、report generator 或 N607 launch plan 都必须遵循本文件中的科研场景和数据协议。

自动化生成或放行实验时，必须检查以下语义条件：

- 训练阶段是否仍是 source-domain weak-label / semi-supervised DG，而不是误写成 target few-shot。
- target receiver domain `R_t` 是否与 source receiver domain `R_s` 不相交。
- `R_t` 是否同时包含 target-old `Y_old` 样本和 target-new `Y_new` 样本；缺任一类时不得声称完整 Stage2-C。
- target-old、target-new、target-unknown 是否满足 `Y_old/Y_new/Y_unknown` 的互斥定义。
- Stage2-A/B/C 是否使用对应 support/query 权限，尤其禁止 Stage2-A/B 声称 seen-new identity accuracy。
- Stage2-B/C 是否显式记录正整数 `K`；推荐锚点 `{1, 2, 5, 10, 15, 20, 50}` 可用于主曲线，但中间值允许进入 launchable row。`K>20` 必须解释为 higher-shot / medium-shot / saturation point，而不是 strict few-shot。
- satellite/LEO view 是否按 deployment-primary 处理，clean view 只作为 control。
- 指标、阈值和成功声明是否符合第 10 节和第 11 节。
- Phase2 row 是否满足 target receiver domain、support/query、TX split、satellite/LEO view 和本地字段要求；只要某个 Phase2 row 已满足这些要求，自动化就应继续 Runner gates，而不能用旧的整 lane local-patch 状态掩盖该 launchable row。

若自动化候选与本文件冲突，默认处理为：

- 不能进入 launchable row。
- 不能写成 deployment success。
- 不能作为论文主结论。
- 只能标为 `NON_LAUNCH_DIAGNOSTIC`、`LOCAL_PROTOCOL_REPAIR_REQUIRED` 或先提交本文件修订。

自动化可以提出新路线，但新路线必须先被翻译为本文件允许的场景、数据集合、support/query 权限、指标和声明边界；翻译不清楚时，不得由 runner 或 launcher 通过默认参数补齐。

## 13. 修改流程

任何后续优化、修改或实验设计如需改变以下内容，必须先更新本文件，再改 prompt、contract、runner、matrix 或报告模板：

- 项目主场景。
- `L_s/U_s` 标注与无标注定义。
- `rho_label` 网格。
- `R_s/R_t` 接收机划分。
- `Y_old/Y_new/Y_unknown` TX 划分。
- K-shot 策略、推荐锚点或报告分层。
- Stage2-A/B/C 边界。
- satellite/LEO 视图是否为 deployment-primary。
- 指标、成功判据或可声明结论。

若某个新想法违反本文件，但仍有探索价值，必须标为 `NON_LAUNCH_DIAGNOSTIC` 或先提交协议修订；不得直接作为 launchable row、论文主结论或部署成功证据。

## 14. Git与Markdown同步纪律

任何CVS项目相关改动都必须进入Git可追踪流程。改动前必须确认目标文件所在目录是否为Git仓库，并记录`git status -sb`或明确说明该目录尚未纳入Git。改动后必须检查`git diff`/`git status -sb`，完成必要验证，并把变更文件、验证结果和未提交风险写入对应报告或Markdown交接记录。

协作输出规则：DO NOT send optional commentary。

如果目标目录已经是Git仓库，完成验证后应提交本次意图明确的改动，除非用户明确要求不要提交。若目标目录不是Git仓库，不得把改动描述为“已版本化”；必须先选择或初始化Git承载目录，或同步到已经约定的Git-backed发布工作区/分支。

任何代码、配置、脚本、矩阵、prompt、报告模板或协议改动，都必须同步检查项目相关Markdown是否需要更新：

- 工作流、安全、环境、N607、Git或协作规则改动，更新`AGENTS.md`。
- CVS科学场景、数据协议、receiver/TX划分、`rho_label`、Stage2-A/B/C边界、K-shot、satellite/LEO视图、指标或声明口径改动，先更新本文件。
- README、docs、实验报告、发布说明或交接文档中涉及的用法、结果解释、发布范围和复现边界发生变化时，必须同步更新对应Markdown。
