# CVS 项目场景与数据协议

版本：2026-07-07  
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
4. **Phase2主线是在轨目标域少样本适应与新类学习**：部署后，卫星接收机可能获得少量旧类带标签目标域样本，也可能获得少量新辐射源带标签目标域样本。旧类是地面训练已知 TX；新类必须是地面训练未见 TX。Phase2主线必须使用同一个目标卫星接收机域 `R_t` 中叠加简化LEO星地信道后的旧类样本和新类样本，完成目标域适应、旧类校准和新类学习。

项目不应被表述为普通 WiSig 少样本分类、普通全监督域泛化、纯 few-shot learning，或真实卫星部署已完成验证。

## 3. 核心科学问题

CVS 要回答的问题是：

> 在身份标签极少、无标签信号大量存在、接收机域变化强、星地信道破坏严重、部署后还会出现新辐射源的条件下，如何先在地面学习稳定的发射机身份表征，再在星上利用叠加LEO星地信道的少量目标域旧类样本和新类样本完成跨域适应、旧类校准和新类学习。

该问题的核心矛盾是 identity-style conflict：发射机硬件指纹、接收机响应、日期漂移、采集域差异和星地信道扰动在 raw IQ 中纠缠。CVS 的价值不在于堆叠多个 DG trick，而在于用物理先验和身份-域双表征控制这种纠缠。

未知类拒识、open-set / open-world 学习和 unknown FAR 优化自 2026-07-07 起下沉为 Phase3 备用项。它们可以作为安全扩展或诊断项保留，但不得作为 Phase2 主线成功门槛、Phase2 主线优化目标或论文主线结论。

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

在轨阶段面对目标接收机域 `R_t`。`R_t` 可以是一个接收机，也可以是多个接收机组成的 deployment proxy domain；Phase2 target-old 和 target-new 的 support/query 都必须来自 `R_t`，并处于相同定义的 satellite/LEO target view 下。若启用 Phase3 open-set 备用项，unknown query 也必须来自同一个 `R_t`，但它不得反向改写 Phase2 主线门槛。

Phase2 主线样本选择的核心约束：target receiver domain 必须与地面训练接收机域不同，同时 must include target-old samples from `Y_old` and target-new samples from `Y_new`；这些 support/query 样本必须按简化LEO星地信道目标视图构造和报告。其中 target-old 是与地面训练发射机类别相同的旧类，target-new 是与训练发射机类别不同且不在 `Y_old` 中的新类。若 `R_t` 中缺少旧类或新类样本，不能声明完整 Phase2 主线/Stage2-C；相关 row 必须降级为 Stage2-A/B、`LOCAL_DATASET_EXTENSION_REQUIRED`、`LOCAL_PROTOCOL_REPAIR_REQUIRED` 或 `NON_LAUNCH_DIAGNOSTIC`。

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

Phase2 主线的优化目标是 target-old 域适应、target-new 新类学习和二者的同 row 平衡。open-set rejection、unknown FAR、FPR95、AUROC、Weibull/OpenMax类拒识门控和 unknown query 压力测试属于 Phase3 备用项；Phase2 可以保留相关字段作为 diagnostic/safety metadata，但不得把它们列为主线必达指标或用低 unknown FAR 掩盖旧类/新类学习不足。

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

`Y_unknown` 是 Phase3 open-set 备用项使用的集合。Phase2 主线只要求 `Y_old` 与 `Y_new` 的合法 target-domain support/query；若某个 Phase2 row 同时携带 `Y_unknown`，必须标明 unknown query 为 evaluation-only / Phase3-backup metadata，不能参与 Phase2 阈值拟合、adapter 更新、主排序或成功声明。

若当前 ManySig 仅提供六个旧类 TX，可采用：

```text
Y_old = {0,1,2,3,4,5}
Y_new = held-out non-ManySig TX set from another WiSig subset
Y_unknown = held-out non-ManySig TX set disjoint from Y_new
```

如果本地数据暂时不能提供非旧类 TX，Stage2-C 不能声明新类识别，也不能作为 Phase2 主线完成；只能做 Stage2-A/B，或标为 `LOCAL_DATASET_EXTENSION_REQUIRED`。

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

Phase2 matrix generator should first choose target receiver domains from this confirmed pool before proposing new dataset routes. 当前 Phase2 主线优先构造 target-old + target-new 的简化LEO目标域少样本适应/新类学习 row：

| target receiver label in `R_t` | ManySig target index | ManyTx receiver index | target-old source | target-new / unknown source | confirmed non-old TX count in ManyTx | confirmed non-old eq1 samples |
|---|---:|---:|---|---|---:|---:|
| `20-1` | 7 | 10 | ManySig / `Y_old` | ManyTx / non-`Y_old` TX | 144 | 27,638 |
| `3-19` | 8 | 12 | ManySig / `Y_old` | ManyTx / non-`Y_old` TX | 143 | 26,887 |
| `7-14` | 9 | 13 | ManySig / `Y_old` | ManyTx / non-`Y_old` TX | 137 | 26,445 |
| `7-7` | 10 | 14 | ManySig / `Y_old` | ManyTx / non-`Y_old` TX | 142 | 26,868 |
| `8-8` | 11 | 17 | ManySig / `Y_old` | ManyTx / non-`Y_old` TX | 140 | 26,474 |

这些候选满足 Phase2 主线样本选择核心条件：`R_t` 与 `R_s` 不相交，`R_t` 中存在与训练 TX 相同的 target-old 样本，也存在与训练 TX 不同的 target-new 样本。矩阵生成时必须按 receiver label 对齐不同 pkl 的 `rx_list`，不能把 ManySig 的 index 直接当成 ManyTx 的 index。

`ManyTx` 是当前 Stage2-C target-new 的主来源；若进入 Phase3，`ManyTx` 也可为 `Y_unknown` 提供备用 open-set 样本。`ManyRx` 可作为 receiver-rich 或低新类数量 control；`SingleDay` 可作为 single-day smoke/control，但它缺少 `3-19` target receiver，且只有单日样本。若使用 `ManyRx` 或 `SingleDay` 替代 `ManyTx`，矩阵必须显式标明这是 control/sensitivity 设计，而不是默认主线 Stage2-C 候选。

生成 Stage2-C 时，`ManyTx` 中 non-`Y_old` TX 必须提供合法的 `Y_new` target-new support/query。若同一 row 额外携带 Phase3 open-set 备用信息，则 `Y_new` 与 `Y_unknown` 必须互斥；不得把同一个 TX 同时用于 seen-new enrollment 和 unknown rejection query。每个 Phase2 主线 row 必须记录 `target_receiver_ids`、`source_receiver_ids`、`target_old_tx_ids`、`target_new_tx_ids`、support/query 划分、`K`、satellite/LEO view 和阈值选择 label scope；Phase3 备用 row 还必须记录 `target_unknown_tx_ids` 和 unknown query evaluation-only 范围。

`ManyTx` 行的 `target_new_tx_ids` 以及对应 `target_new_tx_labels` 必须是从 `ManyTx.pkl` 的 `tx_list` 解析出来的真实 TX label（例如 `1-16`、`10-1`），不得使用 synthetic numeric IDs、rank 占位或“稍后解析”的说明文字作为 launchable 字段。Phase3 备用 row 的 `unknown_tx_ids` / `unknown_tx_labels` 也必须遵守同一真实标签规则。矩阵生成器在写入 launchable row 前必须按目标 receiver label 预筛每个 `Y_new` TX 的可用样本数，确认满足该 row 的 support/query 需求；若启用 Phase3，则也必须预筛 `Y_unknown`。只记录 aggregate non-old TX count 或 aggregate samples 不足以证明 launchable。若任何 TX 与 `Y_old` 重叠、无法按 `ManyTx.tx_list` 解析、在目标 receiver 下样本不足或过滤后会产生空 tensor，该 row 必须标为 `LOCAL_PROTOCOL_REPAIR_REQUIRED` / `LOCAL_DATASET_EXTENSION_REQUIRED` / `NON_LAUNCH_DIAGNOSTIC`，不得发往 N607 runner。

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

自2026-07-13起，CVS后续Phase1/Phase2训练验证、候选比较、checkpoint终局评估和主报告中的默认测试增强统一使用上述三个`leo_*_weak`视图。训练和测试可以使用不同随机种子、不同样本扰动和独立固定评估缓存，但测试场景族不得再默认切换到`legacy_full`。任何launcher、runner或评估脚本若未显式指定测试场景，也必须解析为：

```text
eval_sat_scenarios = leo_clear_weak,leo_low_elev_weak,leo_rain_weak
```

自2026-07-14起，星上轻量化中的TTA视图数量属于同一简化LEO物理观测之后的接收侧推理机制，不构成新的信道场景，也不要求正式协议固定为5-view。允许在完全相同的物理样本、scenario、satellite seed、support/query划分、checkpoint和adapter下比较`none`、`rx_shift3`、`rx_cfo3`与`rx_light5`；其中各策略分别执行1、3、3、5次backbone前向及同数量的FFT辅助提取。不同TTA策略不得重新训练不同adapter、重新采样不同LEO扰动或混入clean view后再归因于view数量。正式晋升必须使用逐样本可部署决策，显式报告view count、backbone forward count、FFT count以及相对5-view的`old_acc`、`seen_new_acc`、`H_old_new`变化；legacy角色/类别配额Oracle只能保留为non-deployment diagnostic。

自2026-07-13起，CVS与外部方法的正式对比实验中，所有进入论文主表、主图、统计检验或方法排序的测试样本都必须实际叠加上述简化LEO星地信道之一；不得把未叠加星地信道的clean测试混入正式主结果。clean只允许作为单独control/reference，必须与deployment-primary结果分表。若测试入口没有记录scenario、satellite seed或增强是否实际启用，该测试结果视为artifact-incomplete，不得形成论文结论。

每个正式实验的最终测试artifact必须同时保留sample-level score table和分组详细统计。分组至少包含逐receiver、逐transmitter、receiver x transmitter以及receiver x transmitter x day四个层级；每组必须记录sample count、correct count、accuracy和稀疏confusion明细。Phase2还必须保留target-old/target-new角色、support/query sample ID和support/query overlap检查。只有overall accuracy而没有逐接收机/逐发射机详细结果的实验，不满足CVS发表证据要求。

该约束用于统一项目主指标口径并避免把不同星地信道实现的结果直接横向比较。由于训练与测试使用同一简化LEO场景族，主报告必须明确写作`leo_weak`族内独立随机压力鲁棒性，不得把它扩大解释为跨信道模型、跨实现或真实在轨泛化。需要检验跨增强族能力时，旧场景必须以显式`legacy stress/control`附加评估运行，单列结果，不得参与默认checkpoint选择、候选promotion或deployment-primary成功门槛。

旧场景 `clear_leo`、`low_elev_leo`、`rain_leo`、`storm_mp`、`mixed_orbit` 保留为 legacy stress/control。`storm_mp` 和 `mixed_orbit` 只能作为 diagnostic/sensitivity，不再作为默认 deployment-primary 成功门槛。

clean view 是 control/reference。不得把 clean view 成功提升为 satellite/LEO deployment success。

## 9. Stage2-A/B/C 协议

Stage2-A/B/C 自 2026-07-07 起共同归入 Phase2 deployment adaptation 主线。Phase2 主线的中心是：在叠加简化LEO星地信道的目标接收机域 `R_t` 上，利用少量 target-old 和 target-new 样本完成跨域适应、旧类校准和新类学习。unknown/open-set 拒识从 Phase2 主线中移出，作为 Phase3 备用项管理。

### 9.1 Stage2-A：Zero-label target-domain LEO reference

目的：无目标标签时建立目标接收机域简化LEO视图下的旧类识别和非旧类观测参考。Stage2-A 是 Phase2 前置参考/安全基线，不是新类学习完成项，也不是 open-set 主线。

```text
support:
  empty

query on R_t:
  target-old query from Y_old
  target-new query from Y_new for non-old reference only
  optional Phase3-backup unknown query from Y_unknown, evaluation-only
```

允许声明：

- old-class target recognition。
- target-new is non-old reference / not-yet-enrolled reference。
- optional Phase3-backup unknown diagnostic，必须标为非Phase2主线。

禁止声明：

- new identity recognition。
- target-label threshold fitting。
- seen-new accuracy。
- Phase2 open-set success。

Stage2-A 是目标域LEO参考底线，不是新类注册，也不是 Phase3 open-set 结果。

### 9.2 Stage2-B：Target-old few-shot domain adaptation

目的：在同一个卫星接收机域和简化LEO目标视图下，用少量旧类标注样本进行目标域适应和旧类校准。

```text
support on R_t:
  K shots per old TX from Y_old

query on R_t:
  held-out target-old query from Y_old
  target-new query from Y_new as not-yet-enrolled reference
  optional Phase3-backup unknown query from Y_unknown, evaluation-only
```

允许声明：

- target-old full accuracy / accepted accuracy。
- old_acc_delta_pp。
- old retention。
- rescue / harm / net_gain。
- target-old adaptation under LEO target view。
- optional Phase3-backup unknown diagnostic，不作为 Phase2 主线成败。

禁止声明：

- seen-new identity accuracy。
- target-new support 使用。
- unknown query 参与阈值拟合。
- 把 low unknown FAR 或 high AUROC 写成 Phase2 主线成功。

### 9.3 Stage2-C：Old + seen-new enrollment

目的：在同一个卫星接收机域和简化LEO目标视图下，同时利用旧类 support 校准和新类 support 注册。这是 Phase2 当前主线目标。

```text
support on R_t:
  K shots per old TX from Y_old
  K shots per seen-new TX from Y_new

query on R_t:
  held-out target-old query from Y_old
  held-out seen-new query from Y_new
  optional Phase3-backup unknown query from Y_unknown, evaluation-only
```

允许声明：

- target-old performance。
- seen-new identity accuracy。
- `H_old_new`。
- output semantics: old label, seen-new label, uncertain, defer；若启用 Phase3 备用项，可额外报告 reject。

禁止声明：

- 把 `Y_unknown` 当 seen-new 识别。
- 用 unknown query 调阈值。
- 把 clean-view success 写成 deployment success。
- 把 Phase3 open-set 指标写成 Phase2 主线门槛。

Stage2-C 是 Phase2 主线目标，但只有在 `Y_new` 与 `Y_old` 不相交、`R_t` 与 `R_s` 不相交、target-old 与 target-new support/query 都来自 `R_t`，并且这些样本按 satellite/LEO target view 构造时才成立。

### 9.4 Phase3：Open-set backup

Phase3 是备用项，不是当前 Phase2 主线。只有在 Phase2 已完成或用户明确要求 open-set 安全扩展时，才把 unknown/open-set 学习作为正式优化对象。

Phase3 support/query 边界：

```text
support on R_t:
  optional target-old support from Y_old
  optional target-new support from Y_new

query on R_t:
  held-out target-old query from Y_old
  held-out seen-new query from Y_new when Y_new is enrolled
  unknown query from Y_unknown, evaluation-only unless a Phase3 protocol explicitly allows a separate calibration split
```

Phase3 可以报告：

- unknown FAR / unknown rejection。
- FPR95 / AUROC。
- open-set confusion。
- reject / uncertain / defer 行为。
- Phase2 old/new 性能在加入拒识门控后的保留率。

Phase3 禁止：

- 用 unknown query 反向调 Phase2 阈值、adapter、prototype 或主排序。
- 用 open-set 指标替代 Phase2 的 target-old / seen-new 同 row 学习指标。
- 把 Phase3 备用项写成当前主线。

### 9.5 当前阶段化优化优先级：PHASE2_ADAPT_NEWCLASS_FIRST

当前H06/Phase2修复路线采用阶段化优化顺序：

1. 先提升旧类目标域准确率，阶段门槛为`old_acc>=0.80`。该门槛用于判定是否进入下一阶段优化，不等同于部署成功、论文主结论或Stage2-C成功。
2. 在同一协议边界下达到`old_acc>=0.80`后，Phase2下一主目标是Stage2-C的`seen_new_acc`、`H_old_new`和旧类/新类同 row 平衡。
3. 若当前row仍是Stage2-B old-only / old + target-new reference，则不得报告`seen_new_acc`，此时只能把下一阶段写成“OLD80达成后进入Stage2-C seen-new enrollment优化”。
4. unknown拒识性能（`unknown_FAR`/`unknown_rejection`）下沉为Phase3备用项。Phase2阶段可同步记录unknown diagnostic，但不得为了unknown FAR牺牲旧类或新类学习后仍称为完成Phase2主线。
5. PHASE2_ADAPT_NEWCLASS_FIRST是当前工程/实验优先级，不会放宽`R_t`/`R_s`不相交、`Y_old`/`Y_new`互斥、target-new query不参与阈值拟合、unknown query不参与Phase2阈值拟合、clean view不代表部署成功等协议约束。

### 9.6 目标域旧类微调上限诊断：TARGET_OLD_ONLY_FT_DIAG

当PHASE2_ADAPT_NEWCLASS_FIRST路线的旧类阶段未达到`old_acc>=0.80`时，可以追加一个诊断性实验：只使用目标接收机域`R_t`中的旧类`Y_old`带标签样本，在目标域内部划分support/query，评估target-only微调或线性头/小adapter对旧类query准确率的上限提升。

该诊断的边界如下：

1. 该诊断只回答“如果暂时不考虑FAR、新类和未知拒识，目标域旧类样本本身能把旧类性能提升到什么程度”。
2. support和query必须都来自`R_t`中的`Y_old`，且query不能与support重叠。可以使用`K`或train-per-TX网格记录目标域样本量。
3. 该诊断不使用`Y_new`support，不使用`Y_unknown`query调阈值，也不报告`seen_new_acc`、unknown FAR、Phase2完成或部署成功。
4. 该诊断可使用feature-level classifier/head/adapter微调或full-model target-only fine-tune，但必须标为`NON_DEPLOYMENT_DIAGNOSTIC`或`TARGET_OLD_ONLY_UPPER_BOUND_DIAGNOSTIC`。如果使用full-model fine-tune，还必须单独报告其星上不可部署或需离线重训练的边界。
5. 若该诊断达到`old_acc>=0.80`，只能说明旧类目标域样本中存在可利用的判别信号；后续仍需回到Stage2-B/Stage2-C协议中重新加入target-new支持、新类学习和同 row 平衡门槛。unknown拒识若继续优化，应作为Phase3备用项单独进入。

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
- PHASE2_ADAPT_NEWCLASS_FIRST第一阶段门槛：`old_acc>=0.80`，仅表示旧类校准阶段可继续进入新类学习优化。
- rollback trigger rate。
- TARGET_OLD_ONLY_FT_DIAG诊断指标：target-old query accuracy、相对source-only baseline的delta、support/query划分、train-per-TX，不报告FAR或seen-new。
- optional Phase3-backup unknown diagnostic：unknown_FAR、FPR95、AUROC，必须标为备用项，不参与Phase2主线排序。

### 10.3 Stage2-C

- seen_new_acc。
- old_acc。
- `H_old_new`。
- new_acc_drop_pp <= 2 pp，或明确标为 exploratory。
- old->new, new->old confusion。
- latency / memory / prototype storage。
- 当前优化顺序必须先满足旧类`old_acc>=0.80`阶段门槛，再优化`seen_new_acc`和`H_old_new`；未达OLD80时不得把seen-new单点作为主线成功。
- optional Phase3-backup unknown diagnostic：unknown_FAR、unknown_rejection、FPR95、AUROC、unknown->old / unknown->new confusion，必须与Phase2主线指标分表或分栏报告。

### 10.4 Phase3 open-set backup

- unknown_FAR <= 0.05。
- unknown_rejection。
- FPR95 / AUROC。
- old/new性能保留率。
- reject / uncertain / defer coverage。
- unknown->old、unknown->seen-new、old/new->reject confusion。
- Phase3结果必须绑定同一candidate/run，且不得替代Phase2的`old_acc`、`seen_new_acc`和`H_old_new`主线判断。

## 11. 论文与报告声明边界

可以写：

- CVS 面向天基 RFFI 的弱标注跨接收机 DG 与在轨跨域 few-shot 适应。
- WiSig/ManySig 是地面可接入源域代理。
- satellite stress 是物理启发部署压力测试。
- Stage2-B 是旧类目标域校准。
- Stage2-C 是 Phase2 主线的 target-old adaptation + seen-new enrollment，前提是 `Y_new` 与 `Y_old` 不相交，且target-old / target-new support/query来自同一个LEO目标域。
- Phase3 是 open-set / unknown rejection 备用项，不是当前主线。

不能写：

- WiSig/ManySig 是真实卫星训练集。
- satellite augmentation 等价于真实在轨验证。
- source-only DG 等价于 few-shot learning。
- 旧类 target support 带来的提升是新类识别。
- Stage2-A/B 的 rejection 结果是 seen-new identity accuracy。
- open-set / unknown FAR 结果是 Phase2 主线成功。
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
- Phase2主线是否以 target-old适应、target-new新类学习、`old_acc`、`seen_new_acc`和`H_old_new`为核心；不得把open-set / unknown FAR设为Phase2主线必达门槛。
- 若携带unknown query，是否被标为Phase3-backup / evaluation-only metadata，且未参与Phase2阈值拟合、adapter更新、主排序或成功声明。
- 指标、阈值和成功声明是否符合第 10 节和第 11 节。
- Phase2 row 是否满足 target receiver domain、support/query、TX split、satellite/LEO view 和本地字段要求；只要某个 Phase2 row 已满足这些要求，自动化就应继续 Runner gates，而不能用旧的整 lane local-patch 状态掩盖该 launchable row。

若自动化候选与本文件冲突，默认处理为：

- 不能进入 launchable row。
- 不能写成 deployment success。
- 不能作为论文主结论。
- 只能标为 `NON_LAUNCH_DIAGNOSTIC`、`LOCAL_PROTOCOL_REPAIR_REQUIRED` 或先提交本文件修订。

自动化可以提出新路线，但新路线必须先被翻译为本文件允许的场景、数据集合、support/query 权限、指标和声明边界；翻译不清楚时，不得由 runner 或 launcher 通过默认参数补齐。

Phase3 open-set 备用路线可以作为单独实验矩阵、后续安全扩展或诊断分支提出；它不能阻塞已经满足 target-old + target-new LEO样本协议的 Phase2 主线 row 进入Runner gates。

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

协作输出规则：对于使用工具或长时间运行的任务，首次工具调用前、关键阶段切换时、重连或上下文压缩恢复后、出现阻塞时，以及持续工作期间至少每60秒，必须发送简洁、基于证据的进度更新；只报告可观察操作、发现和下一步，不披露私有思维链，不倾倒原始日志。仅无工具的简短问答可以省略过程更新。

如果目标目录已经是Git仓库，完成验证后应提交本次意图明确的改动，除非用户明确要求不要提交。若目标目录不是Git仓库，不得把改动描述为“已版本化”；必须先选择或初始化Git承载目录，或同步到已经约定的Git-backed发布工作区/分支。

任何代码、配置、脚本、矩阵、prompt、报告模板或协议改动，都必须同步检查项目相关Markdown是否需要更新：

- 工作流、安全、环境、N607、Git或协作规则改动，更新`AGENTS.md`。
- CVS科学场景、数据协议、receiver/TX划分、`rho_label`、Stage2-A/B/C边界、K-shot、satellite/LEO视图、指标或声明口径改动，先更新本文件。
- README、docs、实验报告、发布说明或交接文档中涉及的用法、结果解释、发布范围和复现边界发生变化时，必须同步更新对应Markdown。
