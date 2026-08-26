# SF-TAPFT V1目标域内部性能筛选预登记

## 实验身份

- run ID：`stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1`
- 候选：`SF_TAPFT_V1_REPORT_DEFAULT`
- 权限：`DIAGNOSTIC_NON_FORMAL`
- Git commit：`1023d70b37bccc7f5144e018b9045aad68ebd013`
- 数据绑定：`protocol_schema=p2_min_v1`、`phase2_data_status=VALIDATED_ONCE`、`capsule_id=d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`、`split_id=stage2b-rx20-1-seed713101-before-support-prefix`

## 可证伪矩阵与停止规则

- 单seed：`392002`。
- 单目标接收机：`rx20-1`。
- `K=10`旧类target support，4-fold target-inner选择。
- frozen与SF-TAPFT在相同OOF fold上比较balanced accuracy、NLL和true-class margin。
- 只有多数fold不下降、平均NLL改善且accuracy或margin改善时选择`adapted`；否则选择`zero_adapt`并停止该候选，不进入query性能验证。
- 技术停止仅限协议/query越界、错误checkpoint/split/GPU、输出碰撞、错误checkout、确定性重复异常、无法产生`selection.json`或进程归属不清；不得因中途指标低而停止。

## 版本与命令

- 本地Git工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\meta-adapter-tri-r4-v1-20260824`
- N607 release目标：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1_fix3`
- N607 CWD：上述release目录。
- smoke命令：`CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -X utf8 code/scripts/run_target_only_progressive_adapt.py --config configs/stage2_sf_tapft_v1_rx20_1_clear_smoke_s392002_20260826.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1_smoke --device cuda:0`
- 性能筛选命令：`CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -X utf8 code/scripts/run_target_only_progressive_nested.py --config configs/stage2_sf_tapft_v1_report_default_s392002_20260826.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1 --device cuda:0 --folds 4`

## 环境、输入与输出

- N607 GPU：物理GPU0；preflight时利用率0%、显存1MiB。
- 服务器解释器：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，PyTorch`2.1.0+cu121`，CUDA可用。服务器不存在`ssr-gpu`环境，本次不伪造环境名。
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- support：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2b_sclba_a_t5t25_s713101_20260824_v1/input/support_rx20_1_k10_clear_smoke.npz`
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1_smoke`
- 性能筛选输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1.log`
- 预期artifact：smoke的`sf_tapft_bundle.pt/smoke.json`；性能筛选的`selection.json`，仅当选择`adapted`时另有`sf_tapft_bundle.pt`。

## 声明边界

该实验只产生target-inner OOF筛选证据，不读取query，不产生正式Phase2 prediction，也不连接truth。`RUNNING`、smoke通过或OOF选择均不得表述为正式最终性能；只有后续独立query prediction和truth-last scorer闭合后才可声明query性能。

## 启动记录

- 首次真实checkpoint smoke在模型更新前停止：既有已验证support NPZ仅含`received_iq/support_labels`，runner错误要求内嵌`support_physical_ids`，因此报`target support NPZ allowlist mismatch`；smoke/run输出均未创建，无性能结果。
- 定点修复提交为`9f8bf87bc590ad7c53240e6ae052c79562e8755e`：对于匹配`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`的既有两字段support，runner生成仅用于inner-fold行隔离的稳定opaque row ID，并在receipt中记录`physical_id_origin=validated_support_row_index`；不把它声明为新数据验证或真实物理ID证据。
- 修复后19项SF-TAPFT聚焦测试通过；等待新release落地并重新执行同一run ID的首次有效smoke。原失败发生在输出创建和参数更新前，因此不会覆盖或混合artifact。
- 原release目录保持不变作为失败证据；修复版使用新的`..._fix1`release目录，不覆盖旧release。
- `fix1` smoke在support tensor构造时因N607的PyTorch2.1/NumPy桥接异常停止：`torch.from_numpy`拒绝实际`numpy.ndarray`；仍未创建smoke/run输出。
- `fix2`提交`1aad719c2f513fc8b9d85f1fe0586d4184fc2df7`保留正常`from_numpy`快路径，仅在`TypeError`时对小型support数组使用列表构造tensor；故障注入聚焦测试通过。新的`..._fix2`release不覆盖前两版。
- `fix2`随后进入适配器，但N607 PyTorch2.1缺少`torch.amp.GradScaler`，仍在首步更新前停止且未创建输出。`fix3`提交`1023d70b37bccc7f5144e018b9045aad68ebd013`在新API存在时使用新入口，否则回退`torch.cuda.amp.GradScaler`；20项聚焦测试通过。
- `fix3`真实checkpoint smoke为`SMOKE_PASS`：60条support、3步更新、15个参数张量更新，BN running statistics未变，source/query/truth/role均未打开；严格消费者回读为`cvs.sf_tapft.v1/3 steps/6 classes`。
- 性能筛选已启动：wrapper PID`3660022`，Python PID`3660023`，PPID/CWD/cmdline/run-root均匹配预登记；GPU0连续两次利用率22%、显存约688MiB，未影响GPU4/5既有任务。
- 当前状态：`RUNNING`。runner只在结束时输出最终JSON，启动检查时日志仍为0字节，不能声明“日志增长通过”；`selection.json`尚未出现，不得声明性能或artifact完成。后续只读检查进程、GPU、最终日志和`selection.json`，不重复启动。

## 2026-08-26 17:10全面运行中分析

### 完成状态

- Python PID`3660023`仍存活，状态在`Rl/Sl`之间，累计运行约1小时12分；CWD、cmdline和run-root继续匹配`fix3`release。
- GPU0利用率连续观测为18%–23%，显存692–702MiB。进程累计CPU时间约2天，瞬时CPU约4077%，即约40核并行占用。
- 完整stdout日志已经逐字节读取：文件大小0字节，最后修改时间仍为启动时刻。runner在结束前不输出step/fold事件，因此没有可解析的loss曲线、fold完成标记或当前step。
- 性能run目录仍不存在；runner只在全部4-fold结束并完成最终选择后创建目录。`selection.json`、性能bundle和fold指标均未生成。
- 最高交付状态保持`RUNNING`，不是`ARTIFACTS_COMPLETE`或`ANALYZED`。

### 已验证的输入与smoke数据

- support形状为`[60,2,256]`，6类各10条，类别计数严格为`[10,10,10,10,10,10]`。
- 既有support NPZ没有真实group或内嵌physical ID；runner在匹配既有`VALIDATED_ONCE`数据句柄后生成稳定opaque row ID，仅用于inner-fold行互斥，`physical_id_origin=validated_support_row_index`。
- smoke bundle大小4,289,502字节，分类头形状`[6,160]`，类别ID为`[0,1,2,3,4,5]`。
- 3步smoke support loss依次为`1.41589725→1.39902306→1.39283490`，总下降`0.02306235`，相对下降`1.6288%`。这只证明优化路径能降低support目标，不是OOF或query性能。
- smoke中A/B/C阶段均执行1步，共更新15个参数张量；BN running statistics未更新，source loader/sample/cache、target eval和query均未打开。

### 4-fold精确切分

|fold|inner train行数|inner validation行数|train每类计数|validation每类计数|
|---:|---:|---:|---|---|
|0|44|16|7/7/7/7/8/8|3/3/3/3/2/2|
|1|46|14|8/8/7/8/8/7|2/2/3/2/2/3|
|2|46|14|8/8/8/8/7/7|2/2/2/2/3/3|
|3|44|16|7/7/8/7/7/8|3/3/2/3/3/2|

四个validation fold合计覆盖60条support且每类总计10条；没有真实group时采用seed`392002`固定的label-stratified fallback。该证据说明类别层面分层成立，但不能替代采集段/session级group证据。

### 参数与计算预算

- 适配后模型参数1,054,963个，诊断分类头960个，总参数1,055,923个。
- A阶段每fold 500步：训练1,584个参数，占总参数`0.1500%`。
- B阶段每fold 1,500步：训练6,882个参数，占`0.6518%`。
- C阶段每fold 2,500步：训练16,386个参数，占`1.5518%`。
- 4-fold总优化步数18,000步，其中A/B/C分别2,000/6,000/10,000步。
- 按full-batch统计，累计训练行呈现810,000次，逐步inner-validation行呈现270,000次，总前向行规模约1,080,000次。

### 性能数据边界与瓶颈判断

当前没有frozen/adapted balanced accuracy、NLL、margin、fold variance、non-degrading fold fraction或`adapted/zero_adapt`判定。任何数值推断都将违反当前证据边界。

资源证据显示CPU约40核满载而GPU利用率仅约20%、显存不足1GiB。结合冻结代码每一步都执行inner-validation，并在checkpoint排序中计算模型到源checkpoint的距离且把参数拉回CPU，本轮主要瓶颈很可能是逐步验证、GPU同步和CPU参数距离计算，而非显存容量或GPU计算能力。由于runner不记录step/fold进度，不能从这些数据可靠换算完成百分比或ETA。本轮保持只读监控，不因运行慢或未知中途指标停止进程。

## 2026-08-26最终完成与深度分析

### 最终状态

- 计算状态：`SELECTION_COMPLETE`；进程已退出，GPU0无该run残留计算进程。
- 交付状态：`ANALYZED`。完整stdout日志、`selection.json`和`sf_tapft_bundle.pt`均已读取，bundle已由严格消费者重新加载。
- 科学判定：`DIAGNOSTIC_POSITIVE_BUT_INVALID_FOR_PROMOTION`。target-inner数值筛选为强阳性并选择`adapted`，但发现冻结参数被checkpoint averaging数值改写；同时该方法本身使用持久可训练分类头并被代码标记为`DIAGNOSTIC_NON_FORMAL`。因此不得把本轮结果表述为正式Phase2性能，也不得直接进入query晋级。
- 时间：日志创建于2026-08-26 15:58:23，最终产物写入于17:10:22，墙钟时间约1小时11分59秒。
- 产物：完整日志10,344字节，`selection.json`12,736字节，bundle 4,336,478字节。

### 聚合性能

|指标|DA0_REG0 frozen|DA1_REG0 adapted|差值/变化|
|---|---:|---:|---:|
|target-inner OOF balanced accuracy|60.4167%|89.5833%|+29.1667个百分点，relative +48.2759%|
|NLL|4.746183|0.410615|-4.335568，下降91.3485%|
|true-class margin|5.402681|2.795414|-2.607266，下降48.2588%|
|balanced accuracy fold variance|0.00940394|0.00245949|下降73.8461%|
|balanced accuracy fold标准差|9.6974个百分点|4.9593个百分点|-4.7381个百分点|
|non-degrading fold fraction|N/A|100%|4/4个fold准确率不下降|
|checkpoint source distance|0|0.0290253|仅为代码定义的state-distance，不是精度指标|

选择规则逐项核对：多数fold不下降为4/4，满足；平均NLL改善，满足；平均balanced accuracy改善，满足；平均margin改善，不满足。规则要求“NLL改善且accuracy或margin至少一项改善”，因此代码选择`adapted`与预登记规则一致。margin在4个fold中仅1个改善，说明适配主要修复错误分类和概率校准，而没有普遍扩大真实类对最强竞争类的平均间隔。

### 逐fold完整结果

|fold|train/validation|frozen BA|adapted BA|BA增益|frozen NLL|adapted NLL|NLL下降|frozen margin|adapted margin|margin变化|source distance|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|44/16|61.1111%|94.4445%|+33.3333pp|4.905730|0.230126|4.675604|4.834957|2.758255|-2.076701|0.0285520|
|1|46/14|69.4445%|86.1111%|+16.6667pp|4.504064|0.410973|4.093091|7.725545|2.888400|-4.837145|0.0294290|
|2|46/14|44.4444%|83.3333%|+38.8889pp|6.015609|0.795072|5.220537|2.092148|2.576888|+0.484741|0.0297441|
|3|44/16|66.6667%|94.4445%|+27.7778pp|3.559328|0.206288|3.353040|6.958073|2.958114|-3.999959|0.0283762|

4个validation fold大小为16/14/14/16，总计60行且每行恰好进入一次validation；每fold的train/validation opaque row ID均不相交。由于原已验证NPZ没有真实physical/session group，切分证据只能证明行级互斥和类别分层，不能证明采集段或session级互斥。

### 优化轨迹与资源效率

最终bundle保存的是被选fold0模型，其审计中包含4,500个support-loss点：

|阶段|步区间|起点loss|终点loss|相对下降|阶段内上升步数|
|---|---:|---:|---:|---:|---:|
|A|0–499|1.625207|0.456088|71.9366%|0|
|B|500–1999|0.455782|0.244023|46.4606%|111|
|C|2000–4499|0.244029|0.243382|0.2651%|1,203|

全程最低loss为0.243380，出现在step 4468；最后100步均值0.243385、标准差0.00000344、范围0.00001168，最后500步仅下降0.00001448。C阶段后半段已经高度平台化，2,500步预算的边际收益极低。4-fold共18,000优化步，墙钟约4,319秒，对应约4.17 optimizer step/s；按先前估算的约1,080,000次train/validation行前向呈现计，约250行/s。运行中GPU0抽样利用率18%–23%、显存692–702MiB，而CPU约40核占用，支持“逐step验证、CPU state-distance与同步开销主导”的瓶颈判断。

### bundle独立回读与冻结边界异常

严格消费者在N607上成功重建并加载bundle：`schema=cvs.sf_tapft.v1`、6类、head形状`[6,160]`、模型参数1,054,963、head参数960；加载后所有参数均为只读，`query_input_capability=false`。bundle绑定`p2_min_v1/VALIDATED_ONCE`，source/query/truth/role/target_eval均未打开。

但对bundle与原ADV3B02 CORE90 checkpoint进行逐tensor比较后，发现冻结边界不成立：

- A/B/C可训练参数名并集只有16个tensor，最终其中13个发生精确变化。
- bundle审计共报告180个变化tensor，其中167个不在可训练并集内；分布为`dom_backbone`91个、`id_backbone`58个、`dom_enhancer`10个、`adv_head`4个、`dom_head`4个。
- 185个非许可floating tensor中有167个发生变化；非许可部分最大绝对偏移为0.5，平方L2和为0.627503。最大项包括两个backbone的`sinc.low_hz_`各0.5、`sinc.band_hz_`各0.25。
- 许可部分最大绝对偏移0.0415013，平方L2和0.192286。

代码路径表明根因是checkpoint averaging把top-k snapshot中的全部`model.state_dict()`一起求均值，而不是只平均可训练tensor并原样复制冻结tensor。即使三个冻结snapshot理论值相同，浮点求和再除法也会造成舍入变化；在Sinc频率参数上偏移达到0.5。逐fold adapted指标是在该平均后模型上计算，因此29.1667个百分点增益不能严格归因于预登记的小子集梯度更新。

### 最终bundle与OOF指标的对应边界

OOF聚合指标来自4个分别拟合的fold模型，但代码在选择`adapted`后直接把`fitted_folds[0]`作为最终bundle；它只用44行训练、16行validation完成选择，没有在全部60行support上按冻结规则重新拟合。因此当前bundle不是“4-fold平均模型”，也不是“全support最终模型”，其实际query行为不能由89.5833%的OOF均值直接代表。

### 结论与后续决策

本轮已经完成计算和诊断分析，证明SF-TAPFT机制在target-inner行级OOF上具有显著潜力：4/4 fold准确率与NLL同时改善，平均准确率提升29.1667个百分点，波动性下降73.8461%。但是三个边界阻止晋级：持久可训练head不符合当前正式Phase2冻结原型边界；checkpoint averaging改写167个非许可tensor；最终bundle仅代表fold0而非全部support。修复时应原样保留冻结tensor、只平均许可更新tensor，并明确全support最终拟合策略；随后使用新run ID重跑同一最小target-inner矩阵。只有修复后仍满足选择门槛，才可产生query prediction并由独立truth-last scorer闭合正式性能。
