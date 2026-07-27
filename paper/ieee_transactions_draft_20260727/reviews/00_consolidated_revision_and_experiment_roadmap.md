# CVS-RFFI综合审稿结论与补实验路线图

日期：2026-07-28

状态：`REJECT_AND_RESUBMIT / EXPERIMENT_BACKLOG_OPEN`

用途：把文献审计、Q1匿名审稿、方法与编辑审计转成可执行的改稿和实验清单。本文件不把待做实验写成已完成结果，也不替代正式实验报告。

## 1.综合结论

CVS-RFFI提出的问题具有明确价值：地面训练阶段需要在目标星载接收机不可见、发射机标签受限的条件下学习跨接收机表征；部署阶段需要只依赖固定的目标域support完成旧类适应和新类注册，同时禁止source replay、query反馈、role oracle和类别配额。把这两个阶段放入同一访问合同，是当前稿件最有辨识度的贡献。

当前版本仍不适合直接投稿IEEE一区。三路独立审查给出的共同原因不是英文表达，而是证据链尚未闭合：

1. Phase1数字来自历史`0.10/0.70/0.20`全池划分，训练池内标签比例实际为12.5%；当前正式`0.07/0.63/0.30`划分下尚无冻结确认。
2. Phase2的125-row结果只隔离了task-balanced covariance相对D81的组件效应；D92整体仍是负诊断，不能代表最终可晋级方法。
3. Phase1和Phase2均缺少足以归因各模块贡献的matched ablation，也没有Phase1×Phase2的联合因子实验。
4. CSIL和MoPC-HR使用不同训练权限、base representation和old-before起点，不能作为严格同权限排行榜。
5. WiSig/ManySig和当前LEO算子都是代理。没有真实上行卫星数据、hardware-in-the-loop、完整链路预算或目标处理器测量。
6. 当前实验是一次性联合注册，不是连续多session的class-incremental生命周期。
7. 若以TIFS为目标，现有闭集识别指标不足以支撑authentication或security主张；还缺威胁模型、未知/非法发射机和攻击实验。

按现有方向，IEEE Internet of Things Journal的主题适配度最高，但仍需完成本文件P0闭环以及至少一部分P1星上证据。IEEE Transactions on Information Forensics and Security需要额外安全闭环；IEEE Transactions on Mobile Computing需要明显更强的移动系统、任务调度和端侧运行证据。

## 2.唯一场景定义

后续正文、图、代码配置和实验报告应始终使用以下唯一方向：

```text
地面发射机/终端
    → 地面到卫星上行及星载接收链
    → 星上RFFI推理
    → 物理身份辅助证据
```

- 被识别对象：地面终端、网关或其他已授权发射机。
- 部署接收机：卫星上的目标接收机；它在地面Phase1训练期间不可见。
- Phase1：只读source receivers；发射机标签受限；不读target receiver。
- Phase2 support：授权标注的目标接收机记录，用于旧类适应和新类注册。
- Phase2 query：只测试；每条query独立在全部已注册类别上竞争。
- 输出作用：辅助凭据—设备一致性检查、跨接触期设备溯源、干扰归因和注册状态维护。
- 安全边界：闭集RFFI分数不能单独证明恶意、拒绝未知设备或替代密码认证。

Starlink Direct to Cell可说明“地面终端向星载接收机发射”已是现实系统方向，并可说明多星、运动、Doppler、短接触和星上资源约束为何重要。本文没有Starlink IQ、终端、波形、卫星前端、接口或飞行处理器，因此Starlink只能是systems motivation，不能是实验对象或有效性证明。

## 3.当前证据能支持到哪里

|主张|当前证据|当前等级|允许写法|尚缺什么|
|---|---|---|---|---|
|两阶段访问合同|协议、capsule、predictor/scorer边界|协议与实现设计证据|protocol-governed two-stage lifecycle|公开复现和独立审计|
|Phase1历史性能|32候选审计；`89.18/84.89/75.55/68.77%`|历史内部审计|historical Phase1 audit reports…|当前划分、多seed、未触碰确认|
|Phase1各机制有效|只有完整候选与ADV2 aggregate|不足|只能描述机制，不能归因效果|模块消融、容量匹配、leakage probe|
|task-balanced covariance有效|D81/D92同row paired矩阵|严格组件效应|在特定slice减轻旧类遗忘，并同时报告新类退化|独立确认和多重比较控制|
|完整RTB-IDR优于简单头|尚无同权限matched主表|不足|不能写优于ProtoNet/qKNN/LDA|统一capsule基线和完整消融|
|few-shot新类注册|一次性联合注册结果|部分支持|one sealed class-set expansion event|连续session、顺序和持久状态|
|星上轻量|最终分类核心数组和head MAC|局部资源证据|compact compiled head|encoder、特征、注册RAM/时延/能耗/WCET|
|LEO适用性|残余基带压力代理|代理级|simulated LEO residual-channel proxy|统计校准、HIL或真实上行外部验证|
|Starlink相关性|官方系统资料和公开研究|动机级|motivated by large LEO systems such as Starlink|Starlink-compatible数据和硬件才可作系统主张|
|authentication/security|辅助身份论证，无攻击实验|不足|auxiliary physical identity cue|威胁模型、FAR/FRR/EER、重放/伪造/投毒|

## 4.P0：任何一区投稿前必须完成

### P0-E1：Phase1当前协议冻结确认

- 数据：`0.07/0.63/0.30`全池划分，训练池内`\rho_label=0.10`。
- 选择：预注册checkpoint选择规则，只读source validation。
- 重复：至少5个独立training seeds；确认receiver/seed不能参与32候选探索。
- 输出：overall、strict UDU、receiver floor、min-class、stress floor、每receiver结果、95%receiver-clustered interval。
- 解锁主张：当前协议下的Phase1性能，而非历史内部审计。

### P0-E2：Phase1第一层模块消融

|Arm|改动|回答的问题|必须报告|
|---|---|---|---|
|P1-FULL|完整Phase1|参考|全部Phase1指标、参数量、FLOPs|
|P1-A0|参数量匹配单embedding|双表征是否超越容量增加|UDU、floor、identity/domain leakage probes|
|P1-B0|移除SSL和entropy项|receiver-day门控半监督是否有效|pseudo precision、coverage、UDU、floor|
|P1-C0|移除角度和尾部风险组|尾风险是否改善最差类|overall、min-class、floor、Q90/Q95角半径|
|P1-D0|移除MixStyle、source episode和LEO CE|身份保持外推课程是否有效|UDU、receiver floor、stress floor|

只有第一层模块显示稳定贡献后，才拆分内部子模块。不得从32个历史候选中选择最有利的差值作为正式消融。

### P0-E3：Phase1同预算强基线

至少包括CE-only、DANN、MixStyle、receiver-disentanglement和channel-robust baseline。所有方法共享：

- 相同labeled/unlabeled/validation physical IDs；
- 相同epoch预算、输入长度和checkpoint选择规则；
- 参数量或计算量匹配；
- 相同source-only权限；
- 相同seed集合和统计单位。

### P0-E4：Phase2同权限主基线

至少包括：

1. frozen cosine head；
2. nearest-class mean/ProtoNet；
3. identity-only qKNN；
4. joint-feature qKNN；
5. pooled shrinkage LDA；
6. class/task-balanced LDA；
7. ridge或multinomial logistic head；
8. D81 matched control；
9. 最终冻结CVS候选。

每个方法必须共享capsule、physical IDs、receiver、seed、scenario、`K`和新类规模。所有类别使用同一预测接口，禁止query truth、old/new role、类别配额和query-query关系。

### P0-E5：Phase2模块消融

|Arm|改动|主要归因|
|---|---|---|
|P2-A0|identity160 only|FFT96和RF32联合表示的整体贡献|
|P2-A1|identity+FFT、identity+RF、identity+FFT+RF|两个辅助视图的独立贡献|
|P2-B0|关闭perturbation basis并使用普通中心|ground aggregate与robust center整体贡献|
|P2-B1|Cauchy换为均值或Huber|稳健权重形式|
|P2-B2|不减quantization noise floor|量化噪声校正|
|P2-C0|D81，关闭0.5/0.5 task balancing|已有严格paired组件效应|
|P2-C1|改变old/new covariance权重|task-balance敏感性|
|P2-D0|full-only、block-only、cross-fitted fusion|双几何融合|
|P2-E0|关闭Fisher residual|判别残差贡献|
|P2-E1|关闭per-class gate或atomic gate|安全门贡献与失败模式|
|P2-F0|FP32、single-residual INT8、dual-residual INT8|量化精度—状态大小权衡|

每一行同时报告`old-pre`、`old-post`、`min-old`、`new`、`min-new`、`H_old_new`、forgetting、fallback/accept counts、核心状态大小、注册峰值RAM和注册成本。只报告某一侧的最优值不构成消融。

### P0-E6：两阶段2×2因子实验

|Phase1|Phase2|用途|
|---|---|---|
|基础表征|基础头|完整基础线|
|最终Phase1|基础头|隔离Phase1贡献|
|基础表征|最终Phase2|隔离Phase2贡献|
|最终Phase1|最终Phase2|检验联合效果与交互|

四组共享确认capsule、support/query、seeds和统计方案。只有该实验能把“两阶段同等重要”从结构性陈述提升为实验证据。

### P0-E7：最终未触碰确认

开发结束后冻结Git commit、Phase1 bundle、配置、primary endpoints、统计方法和promotion gate。使用未参与方法选择的新receiver/seed/capsule执行一次完整确认。失败行必须保留，不能按性能选择重跑或删行。

### P0-E8：数据与复现闭合

必须公开或内部封存可审计的：

- 数据集版本和许可；
- receiver、transmitter、day/capture划分；
- 每split物理记录数和physical ID hash；
- channel profile、builder版本、ordered RNG和batch-shared delay说明；
- capsule ID、split ID和文件hash；
- 环境、精确命令、配置和预期artifact；
- immutable prediction与truth-isolated scorer；
- 论文表格到artifact的逐行映射。

### P0-E9：可晋级的Phase2结果

D92当前状态应保持`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。最终主候选必须在预注册的主要端点上达到门槛，并同时控制old、new、`H`、floor和资源；不能仅用旧类正增益掩盖新类退化。

## 5.P1：使LEO和星上主张可信

### 5.1信道机制与敏感性

- 机制消融：no overlay、SNR-only、CFO-only、phase-noise-only、multipath-only、fading-only、full profile。
- 参数曲线：elevation、SNR、residual CFO、Rician factor、tap delay、phase increment的低/中/高档。
- 输出审计：EVM、PSD、CFO估计、phase drift、tap statistics、输入输出相关性。
- 物理边界：当前`2.462 GHz`和`500–2000 km`只进入FSPL metadata，不改变IQ；absolute FSPL、explicit atmospheric loss和extra IQ imbalance未应用。
- 实现边界：当前capsule使用role-seeded ordered RNG，batch内delay共享；可复现性依赖sealed bytes，而不是每条physical ID独立seed。

### 5.2上行方向匹配验证

推荐证据层级：

1. 当前代理输出的统计审计；
2. 与3GPP或授权channel-emulator trace对齐；
3. SDR hardware-in-the-loop的地面发射机→接收端实验；
4. 合法获得的真实卫星上行外部测试。

如果只能完成前两级，标题、摘要和结论必须继续使用`spaceborne deployment proxy`，不得使用`validated on Starlink`、`in-orbit validation`或`complete satellite channel`。

### 5.3跨接收机切换和时间漂移

- 同一发射机从receiver/satellite chain A切换到B；
- 报告切换前后身份稳定性、校准漂移和重新注册代价；
- receiver-day、温度、长时间间隔和接触窗口变化；
- 比较无更新、仅旧类support更新和完整重新编译。

### 5.4连续注册状态机

至少执行三个独立增量session，比较不同新类到达顺序。每次只读取上一时刻持久状态和当前合法support，记录：

- 每session old/new/H/floor；
- 累积forgetting；
- 状态增长、更新时间和峰值RAM；
- rollback和原子切换；
- 同一类重复注册、标签冲突和support撤销。

### 5.5目标处理器测量

在代表性的ARM、嵌入式GPU、DSP或FPGA平台上分层测量：

1. encoder inference；
2. FFT/RF特征；
3. support registration；
4. query head；
5. 持久状态；
6. 临时workspace；
7. 端到端时延、peak RSS、energy、thermal和WCET；
8. FP32/FP16/INT8数值闭合与失败回退。

16.11 KiB只能描述当前26类核心数组，不能代表整套系统内存。

## 6.若目标为TIFS，需要增加的安全闭环

若不做以下实验，应把论文定位为identification/IoT方法，并避免把closed-set accuracy等同于authentication：

- 明确合法终端、冒充者、重放者、波形生成者、relay和恶意enrollment者的能力；
- unknown/illegal transmitter rejection；
- 低采样与高采样重放；
- waveform regeneration或fingerprint imitation；
- support poisoning、label flip和bundle rollback；
- FAR、FRR、EER、ROC/DET、固定FAR下TPR、calibration error；
- RFFI与密码凭据融合前后的增益；
- 误封、未经授权设备跟踪和数据许可的负责任使用说明。

## 7.统计设计

- 独立单位优先使用receiver；packet不能当作独立重复。
- 报告receiver-level和receiver×seed完整分布，不只报告全局平均。
- 主比较使用paired effect和receiver-clustered bootstrap interval。
- 多arm、多slice使用Holm或FDR校正。
- 探索seed与确认seed分离；探索结果明确标为exploratory。
- 同一表格行保持old-pre、old-post、new、`H`、floor和资源完整上下文。
- 明确primary endpoints和promotion gate，防止从大量query结果中事后挑选。

## 8.正文仍需补充的图表

### 主图

1. 应用与数据流图：地面训练、上行、星载receiver、old/new support、query、truth-side scorer。
2. 两阶段模型图：Phase1双表征及其输出bundle；Phase2多视图support到统一仿射头。
3. 访问权限图：source、support、query、truth在各阶段的可见性。
4. Channel proxy图：captured terrestrial IQ→residual overlay，并标出applied、metadata-only和disabled项。
5. 连续注册状态图：Stage2-A/B/C及后续session。

### 主表

1. 相关工作权限对照表；
2. 数据集与receiver/TX/split/count表；
3. Phase1主结果、强基线与四模块消融；
4. Phase2同权限主表和组件消融；
5. 2×2两阶段交互表；
6. channel/HIL与敏感性表；
7. 目标处理器端到端资源表；
8. validity、threat和responsible-use边界。

## 9.建议执行顺序

### Gate A：定义闭合

- 固定唯一上行方向；
- 修正标签比例分母；
- 明确post-capture residual overlay；
- 冻结channel参数和RNG实现语义；
- 完成主张—证据矩阵。

### Gate B：冻结Phase1

- 当前split下完成P1-FULL和A0/B0/C0/D0；
- 通过后做强matched baselines；
- 生成新的immutable Phase1 bundle。

### Gate C：冻结Phase2

- 在新bundle上完成同权限基线；
- 完成Phase2第一层组件消融；
- 选定可晋级候选并执行未触碰确认；
- 再做Phase1×Phase2的2×2闭环。

### Gate D：补星上证据

- channel机制和参数敏感性；
- 至少HIL或外部上行验证；
- 连续session和切换；
- 目标处理器profile。

### Gate E：按期刊补专属证据

- IoTJ：IoT/IoE运维流程、端侧资源、规模和部署闭环；
- TIFS：威胁模型、攻击和认证指标；
- TMC：移动性、contact-window、任务调度、通信—计算权衡和运行系统。

## 10.主张解锁表

|完成项|可以新增的主张|
|---|---|
|P0-E1至E3|当前协议下Phase1具有可重复的跨接收机收益|
|P0-E4至E5|完整Phase2优于同权限简单头，且各模块贡献可归因|
|P0-E6|Phase1和Phase2具有独立且互补的联合贡献|
|P0-E7至E9|最终方法达到预注册性能并具有未触碰确认|
|P1信道/HIL|方法对方向匹配的LEO上行条件具有外部有效性|
|连续session|方法支持严格的持续class-incremental生命周期|
|目标硬件profile|方法满足具体星上处理平台的资源约束|
|TIFS安全闭环|RFFI可在明确威胁模型下提供authentication/forensics价值|

在这些证据完成之前，应继续使用：

- `terrestrial proxy`
- `simulated LEO residual-channel overlay`
- `historical Phase 1 audit`
- `matched component diagnostic`
- `one sealed registration event`
- `compact compiled head`
- `not in-orbit or flight-software validation`

## 11.配套独立审查文件

- `01_starlink_spaceborne_rffi_literature_audit.md`：Starlink、星上RFFI意义、来源和表述边界。
- `02_q1_reviewer_novelty_evidence_audit.md`：以IEEE一区匿名审稿视角检查创新、证据、公平比较和期刊适配。
- `03_method_experiment_ablation_editorial_audit.md`：核对公式、实现、标签比例、LEO模型、消融、资源和IEEE写作。

后续每完成一组实验，应同时更新正文、`claim_evidence_matrix.md`和对应实验报告；不能只替换摘要数字。
