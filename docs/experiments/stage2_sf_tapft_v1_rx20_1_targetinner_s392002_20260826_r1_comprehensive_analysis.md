# SF-TAPFT V1设计、实现与实验全面分析报告

## 结论摘要

SF-TAPFT V1是一条“只用目标域support、渐进式解冻小部分身份编码器、配合持久目标分类头”的快速适配路线。它以ADV3B02 CORE90 checkpoint为起点，在`rx20-1`目标接收机上使用6个旧类、每类`K=10`、共60条support，完成4-fold target-inner OOF筛选。该轮没有注册新类，也没有读取query、query truth、query role、源域样本或源域cache。

数值上，适配后的OOF balanced accuracy由60.4167%升至89.5833%，提高29.1667个百分点；NLL由4.746183降至0.410615，下降91.3485%；4/4个fold准确率均未下降，fold方差下降73.8461%。按照预登记规则，选择结果为`adapted`。

但本轮不能晋级为正式Phase2性能，科学判定为`DIAGNOSTIC_POSITIVE_BUT_INVALID_FOR_PROMOTION`。原因不是数值不足，而是三个实现与归因边界同时存在：

1. top-3 checkpoint averaging对完整`model.state_dict()`中的floating tensor求均值，导致167个许可集合外tensor发生数值漂移，最大绝对偏移达到0.5；
2. 方法训练并保存一个持久目标分类头，不符合当前正式Phase2冻结原型边界，因此代码明确标记为`DIAGNOSTIC_NON_FORMAL`；
3. 最终bundle直接保存fold0模型，只用44条inner-train support拟合，不是4-fold模型平均，也不是全部60条support上的最终重拟合模型。

因此，89.5833%只能表述为“存在冻结漂移的4-fold target-inner OOF诊断准确率”，不能等同于最终bundle的query性能，更不能表述为truth-last正式结果。当前没有SF-TAPFT query prediction和独立评分数据。

## 1.实验身份与证据范围

- run ID：`stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1`
- 候选：`SF_TAPFT_V1_REPORT_DEFAULT`
- 方法：`sf_tapft_v1`
- 权限：`DIAGNOSTIC_NON_FORMAL`
- 运行提交：`1023d70b37bccc7f5144e018b9045aad68ebd013`
- 随机seed：`392002`
- 目标接收机：`rx20-1`
- 基础checkpoint：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 数据协议：`protocol_schema=p2_min_v1`
- 数据状态：`phase2_data_status=VALIDATED_ONCE`
- capsule：`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`
- split：`stage2b-rx20-1-seed713101-before-support-prefix`
- support received-IQ SHA256：`f5591fa081b197c90969095faba1ff88a3360c4fab1c719cc2014316d13e9c9f`
- support形状：`[60,2,256]`
- 类别：旧类0–5，共6类，每类10条，即`K=10`、总计60条
- 新类注册：无，即全程为`REG0`
- query：未打开
- truth-last scorer：未运行

完整stdout日志、`selection.json`和`sf_tapft_bundle.pt`均已完整读取，而不是只分析日志尾部。stdout日志只有结束时写出的最终JSON，没有逐step或逐fold事件；完整4500点loss轨迹保存在最终bundle的audit中。

## 2.方法设计

### 2.1核心思想

SF-TAPFT将目标域适配拆成三个连续阶段。每个阶段都保持模型为`eval()`，避免Dropout随机性并冻结BN running statistics，只允许指定参数接收梯度。适配只读取目标域support，函数签名没有source、query或target-eval入口。

方法在基础checkpoint上复制teacher和student。teacher完全冻结；student新增一个rank=16的残差时间适配器。新增适配器采用恒等初始化：`up.weight`和`up.bias`清零，因此插入时不改变基础模型输出。

初始目标分类头由两部分混合得到：基础checkpoint的归一化源分类器权重，以及目标support初始embedding的类均值原型。混合系数`rho=0.5`，随后每行再次归一化；分类时对embedding和head权重均做L2归一化，并乘`prototype_scale=8.0`。

### 2.2三阶段渐进解冻

|阶段|每fold步数|允许训练的模型部分|含head参数量|占模型与head总参数比例|主要学习率|
|---|---:|---|---:|---:|---|
|A|500|`t1/t2/t3.norm`及`time_fuse.1`|1,584|0.1500%|head `1e-3`，norm `1e-4`|
|B|1,500|A阶段参数+`meta_adapter_time`|6,882|0.6518%|head/adapter `3e-4`，norm `1e-4`|
|C|2,500|B阶段参数+最后时间块`t3.dw/t3.pw`|16,386|1.5518%|head/adapter `1e-4`，norm `1e-4`，last block `3e-5`|

A阶段先校准归一化仿射参数和时间融合归一化；B阶段再打开低秩时间适配器；C阶段最后允许身份backbone的末级时间块卷积发生小幅变化。三个阶段始终训练960参数的6×160持久分类头。

4个fold各执行4500步，合计18,000个optimizer step。按每fold的inner-train/validation规模估算，训练行呈现约810,000次，逐step验证行呈现约270,000次，总前向行规模约1,080,000次。

### 2.3优化目标

每一步使用full-batch target-inner train support。总损失为：

`CE(head logits)+0.5×CE(leave-one-out prototype logits)+1e-4×L2-SP+0×selective KD`

其中：

- 主CE使用类别平衡权重和`label_smoothing=0.05`；
- prototype项对当前样本构造leave-one-out类原型，减少样本对自身原型的直接自匹配；
- L2-SP只锚定norm和最后时间块到输入checkpoint，不锚定新增adapter；
- selective KD的实现存在，但本轮权重为0，teacher不参与损失；
- 优化器为AdamW，`weight_decay=1e-4`；
- 使用5%warmup和余弦衰减、mixed precision、梯度裁剪1.0。

### 2.4target-inner选择与回退

60条support被seed固定地分为4个stratified fold。由于NPZ不含真实采集group或真实physical ID，runner生成`validated-support-row-*`不透明行ID，只能保证行级train/validation互斥，不能证明session或采集段级互斥。

每个optimizer step之后，都在该fold的inner-validation上计算一个字典序score：

1. balanced accuracy越高越优先；
2. NLL越低越优先；
3. true-class margin越高越优先；
4. 与基础checkpoint的state-distance越小越优先。

每fold保留top-3 snapshot并求平均。完成4个fold后，域级选择规则要求：超过半数fold准确率不下降、平均NLL改善，并且平均accuracy或margin至少一项改善。否则回退`zero_adapt`。本轮4/4个fold准确率不下降，NLL和accuracy均改善，因此选择`adapted`。

## 3.实现内容与产物结构

### 3.1主要代码

- `code/cvsrffi/target_only_progressive_adapt.py`：数据载体、持久目标分类头、leave-one-out prototype、L2-SP、A/B/C解冻策略、4-fold切分、逐步选择、checkpoint averaging和域级回退。
- `code/cvsrffi/target_only_progressive_runner.py`：配置解析、support NPZ加载、不可覆盖输出、bundle写入、selection receipt和严格bundle回读。
- `code/scripts/run_target_only_progressive_adapt.py`：无query smoke入口。
- `code/scripts/run_target_only_progressive_nested.py`：4-fold target-inner性能筛选入口。
- `configs/stage2_sf_tapft_v1_report_default_s392002_20260826.json`：本轮正式诊断配置。
- `configs/stage2_sf_tapft_v1_rx20_1_clear_smoke_s392002_20260826.json`：A/B/C各1步的真实checkpoint smoke配置。

### 3.2安全与协议实现

runner要求`p2_min_v1`、`VALIDATED_ONCE`、非空capsule/split绑定，并拒绝复用已存在的output root。适配函数不接受source或query参数；audit明确记录`source_loader_opened=false`、`source_samples_opened=false`、`source_cache_opened=false`、`target_eval_opened=false`、`query_opened=false`。严格消费者只接受固定顶层allowlist和`cvs.sf_tapft.v1`schema，加载后把模型与head全部设为只读，并声明`query_input_capability=false`。

### 3.3最终bundle

最终bundle大小4,336,478字节，包含：

- 基础checkpoint路径和Phase2数据绑定；
- 完整适配配置；
- 201个model state tensor，共1,055,337个state标量；
- 6×160的head权重，共960个参数；
- 类别ID0–5；
- 4500点support-loss轨迹；
- A/B/C可训练参数名、实际变化tensor名及协议访问审计。

严格回读得到模型参数1,054,963个、head参数960个。state标量数比parameter数多出的部分来自buffer等非parameter state。

## 4.实验执行数据

### 4.1真实checkpoint smoke

smoke使用同一60条support，但A/B/C各执行1步，共3步：

- loss：`1.41589725→1.39902306→1.39283490`
- 总下降：0.02306235，相对下降1.6288%
- 实际更新15个参数tensor
- BN running statistics未变化
- source/query/truth/role/target-eval均未打开
- 严格bundle消费者回读成功

smoke只证明代码路径可执行、参数许可和数据访问边界具备基础运行能力，不构成性能证据。

### 4.2逐fold数据

|fold|train/validation|frozen BA|adapted BA|BA增益|frozen NLL|adapted NLL|frozen margin|adapted margin|source distance|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|44/16|61.1111%|94.4445%|+33.3333pp|4.905730|0.230126|4.834957|2.758255|0.0285520|
|1|46/14|69.4445%|86.1111%|+16.6667pp|4.504064|0.410973|7.725545|2.888400|0.0294290|
|2|46/14|44.4444%|83.3333%|+38.8889pp|6.015609|0.795072|2.092148|2.576888|0.0297441|
|3|44/16|66.6667%|94.4445%|+27.7778pp|3.559328|0.206288|6.958073|2.958114|0.0283762|

fold0–3的validation大小分别为16/14/14/16，合计覆盖全部60行且每行恰好进入一次validation。各fold的类别计数如下：

|fold|train每类计数|validation每类计数|
|---:|---|---|
|0|7/7/7/7/8/8|3/3/3/3/2/2|
|1|8/8/7/8/8/7|2/2/3/2/2/3|
|2|8/8/8/8/7/7|2/2/2/2/3/3|
|3|7/7/8/7/7/8|3/3/2/3/3/2|

### 4.3聚合结果

|指标|DA0_REG0 frozen|DA1_REG0 adapted|变化|
|---|---:|---:|---:|
|target-inner OOF balanced accuracy|60.4167%|89.5833%|+29.1667个百分点，relative +48.2759%|
|NLL|4.746183|0.410615|-4.335568，下降91.3485%|
|true-class margin|5.402681|2.795414|-2.607266，下降48.2588%|
|BA fold variance|0.00940394|0.00245949|下降73.8461%|
|BA fold标准差|9.6975个百分点|4.9593个百分点|-4.7382个百分点|
|non-degrading fold fraction|N/A|100%|4/4个fold准确率不下降|
|checkpoint source distance|0|0.0290253|state-distance，不是精度指标|

这里的新类准确率、old/new harmonic和注册收益均为`N/A`，因为本轮没有注册新类，状态始终为`REG0`。

### 4.4如何理解accuracy、NLL与margin的分歧

balanced accuracy和NLL出现一致的大幅改善，说明适配后的模型在这60条support的行级OOF切分上修正了大量错误分类，同时显著降低了真实标签的负对数似然。fold2从44.4444%升至83.3333%，是最明显的困难fold修复。

但平均true-class margin下降48.2588%，4个fold中只有fold2的margin改善。这意味着模型虽然更经常把真实类排到第一，并给出更好的概率损失，却没有普遍扩大真实类相对最强竞争类的logit间隔。一个合理解释是归一化持久head和label smoothing共同把极端logit压得更温和；另一个不能排除的解释是冻结tensor漂移改变了表征尺度。由于当前实现归因不干净，不能把margin下降简单解释为“更好校准”。

## 5.完整loss轨迹与效率

最终bundle保存的fold0 loss共4500点：

|阶段|步区间|起点loss|终点loss|相对下降|阶段内上升步数|
|---|---:|---:|---:|---:|---:|
|A|0–499|1.625207|0.456088|71.9366%|0|
|B|500–1999|0.455782|0.244023|46.4606%|111|
|C|2000–4499|0.244029|0.243382|0.2651%|1,203|

全局最低loss为0.243380，出现在step4468。最后100步均值0.243385、标准差0.00000344、极差0.00001168；最后500步仅下降0.00001448。A阶段承担主要快速收敛，B阶段继续显著改善；C阶段2500步几乎全部处于平台区，边际收益极低。若冻结归因问题修正后重新验证，C阶段应首先缩短，并用OOF选择而不是support loss平台决定是否保留末块解冻。

运行从2026-08-26 15:58:23持续到17:10:22，墙钟约4319秒，即1小时11分59秒。18,000步对应约4.17 optimizer step/s。运行期间GPU0抽样利用率18%–23%、显存692–702MiB，而CPU瞬时约40核占用。结合代码每一步都执行完整inner-validation、计算checkpoint state-distance并维护完整state snapshot，瓶颈主要来自逐step验证、CPU参数距离和同步/复制开销，而不是GPU显存或纯前向计算。

stdout在结束前一直为0字节，最终只写入一行完整JSON。该设计无法在运行中判断fold、phase、step、loss或ETA，是明显的可观测性缺陷，但不影响本报告对最终artifact的完整读取。

## 6.独立审计发现

### 6.1许可集合外漂移

A/B/C模型可训练参数名的并集共有16个tensor，最终其中13个发生精确变化。然而bundle audit共报告180个变化tensor，其中167个不在许可并集内：

|模块|许可集合外变化tensor数|
|---|---:|
|`dom_backbone`|91|
|`id_backbone`|58|
|`dom_enhancer`|10|
|`adv_head`|4|
|`dom_head`|4|
|合计|167|

185个非许可floating tensor中有167个变化；最大绝对偏移0.5，平方L2和0.627503。突出项包括两个backbone的`sinc.low_hz_`各偏移0.5、`sinc.band_hz_`各偏移0.25。作为对照，许可参数最大绝对偏移0.0415013，平方L2和0.192286。

根因在`CheckpointAverager.average()`：每个top-k snapshot保存完整model与head state，随后所有floating tensor统一执行`stack().mean()`。即使冻结tensor在三个snapshot中理论相同，浮点求和再除法也可能产生舍入差异；这些差异随后被严格加载回student。整数buffer才会原样复制。因此，梯度许可本身没有越界，但checkpoint averaging破坏了最终state的冻结不变性。

### 6.2持久分类头的协议边界

SF-TAPFT训练6×160目标分类头并将其持久化进bundle。这正是报告方法的一部分，但不满足当前正式Phase2冻结原型边界。因此实现选择了明确暴露而非伪装：方法和bundle都标记`DIAGNOSTIC_NON_FORMAL`。本轮只能用于机制筛选，不能直接作为正式Phase2 checkpoint发布。

### 6.3最终bundle不是全support模型

4-fold OOF指标来自4个分别训练的模型，但域级选择为`adapted`后，代码直接返回`fitted_folds[0]`。该模型的训练/验证规模为44/16，top-3 snapshot也由fold0的16条inner-validation选择。它没有使用全部60条support重新拟合，也没有平均4个fold模型。

因此存在两个不同对象：89.5833%是4个模型的OOF均值；bundle是fold0单模型。把OOF均值当作bundle的query预期准确率没有证据支持。

### 6.4行级切分不是物理group切分

support NPZ没有真实physical ID或session/group字段。`validated-support-row-*`只是在既有`VALIDATED_ONCE`句柄上生成的稳定行标识，用于避免同一行同时进入inner train和validation。它不能证明同一采集段、同一物理事件或强相关样本没有跨fold。当前OOF结果可能高估跨session泛化能力，正式设计应在不改变K-shot物理样本定义的前提下使用数据构建阶段已有的真实group元数据。

## 7.能够与不能够声明的结论

### 7.1能够声明

- 实际使用`K=10`，6个旧类共60条目标域support；
- source、query、query truth、query role和target-eval均未打开；
- 4-fold行级OOF中，适配数值相对frozen显著改善，4/4个fold准确率不下降；
- A/B阶段贡献了几乎全部support-loss下降，C阶段明显平台化；
- 当前checkpoint averaging会改变冻结floating tensor；
- 当前bundle是fold0模型，不是全support最终模型；
- 该轮是强阳性机制诊断，但不具备正式晋级资格。

### 7.2不能声明

- 不能声明89.5833%是query准确率；
- 不能声明SF-TAPFT已经通过truth-last独立评分；
- 不能声明最终bundle达到89.5833%；
- 不能把29.1667个百分点提升完全归因于许可小参数子集；
- 不能声明session或物理样本group级泛化；
- 不能声明与ERBT-IDR叠加有效；该级联实验已按用户要求停止，尚未产生正式prediction；
- 不能声明新类注册、新类准确率或old/new harmonic收益。

## 8.后续优化建议（仅分析，不执行）

优先级P0是修复归因，而不是继续叠加新方法：top-k averaging只允许平均head和当前阶段许可更新tensor，其余model state必须从基础checkpoint逐tensor原样复制并用`torch.equal`验证。修复后的新run应首先复跑同一`K=10`、4-fold最小矩阵；只有OOF门槛仍成立，才能进入下一步。

优先级P1是明确最终模型生成策略。建议在OOF只负责选择超参数与是否适配，选择通过后用全部60条support按预登记步数重新拟合一个最终模型；不能继续把fold0模型当作部署模型。若正式协议仍禁止持久可训练head，则需要把head改为冻结原型或将其限定为非持久、非学习的support统计量。

优先级P1还包括计算优化：把逐step inner-validation改为固定间隔验证，只在候选step计算state-distance和保存快照；C阶段可先从2500步缩短到500步或采用早停。这样可显著减少CPU同步与完整state复制，并改善GPU利用率。

在上述问题修正前，不应继续SF-TAPFT+ERBT-IDR级联。当前最可靠的研究结论是：SF-TAPFT的目标域support适配信号很强，但现有实现尚不能提供干净、可晋级的checkpoint归因。

## 9.最终判定

- 计算状态：`SELECTION_COMPLETE`
- 证据状态：`ANALYZED`
- 数值选择：`adapted`
- 科学判定：`DIAGNOSTIC_POSITIVE_BUT_INVALID_FOR_PROMOTION`
- 正式query性能：`UNKNOWN/NOT_RUN`
- 新类注册性能：`N/A`
- 推荐动作：保持停止状态，先修复冻结state averaging与最终全support拟合策略，再用新run ID复验同一最小矩阵。
