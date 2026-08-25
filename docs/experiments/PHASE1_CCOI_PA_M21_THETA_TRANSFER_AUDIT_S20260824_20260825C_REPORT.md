# PA-M2.1独立theta迁移审计实验报告

## 1.当前状态

- run ID：`PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C`
- 当前最高状态：`ANALYZED`
- 阶段A：`A_FAIL`
- 阶段B：`NOT_RUN_A_GATE`
- 最终路线：`STOP_CURRENT_PA_THETA_TRANSFER`
- 性能边界：本轮得到的是source-only机制审计结果，不是Phase2目标域性能，也不是Core90分类精度提升
- 旧实验边界：既有A/B实验及其artifact全程只读，不重启、不覆盖、不复现
- 唯一launch owner：本任务主runner
- 冻结代码提交：`fdab76362c1288301f2b3b26a5cb5350535fd4f2`
- 分支：`codex/phase1-ccoi-pa-v1-20260824`
- 本地与远端分支OID：已核对一致

## 2.科学问题和声明边界

阶段A`M2.1A_THETA_TRANSFER_AUDIT`验证：在当前已知受TX、receiver、day和位置捷径污染的冻结q条件下，新C4′的support theta是否仍具有超出q-only的跨packet、跨receiver和跨day TX增量，以及该增量是否超过同容量非条件C1′。

`V_audit_retro`只对本轮新C1′/C4′权重独立。旧C4架构和超参数已经受历史完整V_select结果影响，因此本实验不是研究历史完全未见确认集，也不能证明“正确challenge-conditioned系统辨识”。即使阶段A通过，下一路线也只能是重新设计连续challenge，不能直接晋级当前q。

阶段B`M2.1B_TRUTH_BLIND_EXPERT_GATE`只在`A_PASS`后运行；`A_PARTIAL`或`A_FAIL`必须闭合为合法科学结果`NOT_RUN_A_GATE`，不是系统技术失败。

## 3.候选与冻结矩阵

### 3.1 C1′/C4′

|候选|q条件|挑战匹配|DiD|holdout|用途|
|---|---:|---:|---:|---:|---|
|C1′|常量|否|否|否|同容量非条件控制|
|C4′|冻结当前q|是|是|是|当前完整条件sidecar|

Core90和旧C4 challenge encoder完全冻结。C1′与C4′从同一随机模板初始化response、pool、operator classifier和holdout predictor，参数量、训练step、训练seed和L_s训练输入一致；只用`V_select_fit`的operator source accuracy选epoch，`V_audit_retro`不参与训练或选模。

### 3.2 数据角色

保持`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。仅在方法内部把V_select按`(TX,RX,day,eq,capture_block)`拆成约65%`V_select_fit`、约35%`V_audit_retro`和guard block。`capture_block=floor(sig_i/B)`，B只从`10/20/25`按metadata覆盖选择，不读取q、theta或性能。

### 3.3 F0—F9

- F0：audit q+零theta；
- F1：同packet raw-disjoint support theta；
- F2：同TX/RX/day、不同packet；
- F3：同TX、跨RX、同day；
- F4：同TX/RX、跨day；
- F5：异TX、同RX/day；
- F6：同TX、跨RX，并按RMS、PAPR、四/六阶矩、包络差分、幅度自相关和正则化memory-polynomial条件数匹配；
- F7：`UNAVAILABLE`，因为sig_i不是已验证的跨receiver同步物理事件ID；
- F8：正确theta配异TX/异packet q；
- F9：训练目标均值。

主比较只在F2∩F3∩F5共同anchor上进行，同时保留all-valid结果。审计覆盖全部4个raw-disjoint fold；support theta只能来自`V_select_fit`独立bank，关系先按metadata硬约束，再按固定seed稳定选择，不读取q且禁止fallback。

## 4.预登记判据

阶段A通过要求：C4′ F3相对F0改善至少5%且分组bootstrap 95%CI下界大于0；相对F5改善至少5%且CI下界大于0；相对F2退化不超过10%；F3覆盖至少80%；每个TX至少两个可用跨receiver关系；主要TX×RX×day cell不少于10个样本；至少2/3 head seed和2/3 mapping seed方向一致；两个satellite seed不发生结论反转。C4′相对C1′ F3还需改善至少3%且CI下界大于0，否则为`A_PARTIAL`。

阶段B采用：

```text
logits_final=logits_base+g(x)*eta*Clip(logits_operator-logits_base)
eta in {0.05,0.10,0.20}
```

gate只读部署时可得的margin、entropy、JS divergence、top1分歧、RMS、PAPR、PA条件数、spectral null ratio、clipping ratio、SNR proxy、残余CFO、相位不稳定度和challenge coverage。禁止true TX、true receiver、day和审计标签作为输入。V_cal按TX×RX×day×eq×capture block做group-CV，拟合Rescue/Harm双多项logistic并固定`eta/tau/lambda_h/clip`；`V_audit_retro`只评估一次。

阶段B通过要求：三个source synthetic LEO场景平均提升至少0.20pp且分组bootstrap CI下界大于0；clean下降不超过0.10pp；最差receiver下降不超过0.05pp；selected weighted utility为正；gate coverage至少5%；leave-one-source-receiver CV多数receiver效用为正。

## 5.落地实现

- `code/cvsrffi/ccoi_pa_m21.py`：V3 sidecar契约、retro split、近重复审计、四fold记录、F关系与同容量head、阶段A判定、q条件probe、M0、LOTO residual、truth-blind gate与阶段B判定；
- `code/audit_phase1_ccoi_pa_m21.py`：真实checkpoint source-only runner、C1′/C4′独立replay、14个聚合artifact和两阶段状态机；
- `code/scripts/launch_phase1_ccoi_pa_m21_20260825.sh`：不可覆盖smoke和正式run；
- 三份新增/扩展测试：核心模块、runner和launcher；
- `docs/CVS_PHASE1_CCOI_PA_M21_TRACE_20260825.md`：42项逐项追踪。

唯一一次P0/P1正确性复核定点修复了四项：四fold分别使用raw遮罩后的Core90 PA map；共同anchor为空时输出科学不可用并进入A_FAIL而不崩溃；LOTO只使用共同有效anchor；补齐码本聚合诊断和between-TX/same-TX cross-RX residual距离。定点复核后未发现仍会导致下一次真实实验跑错、越权、覆盖输出、误杀进程、不能启动或不能产生合法prediction/artifact的问题。

## 6.本地验证

- `ssr-gpu`环境相关测试：93项通过；
- Python编译检查：通过；
- launcher语法：经`C:\Program Files\Git\bin\bash.exe`确认`MSYSTEM=MINGW64`后，`bash -n`通过；
- 已知非阻断告警：`torch.cuda.amp.autocast` FutureWarning；
- 追踪状态：39项`verified`、1项`implemented`、1项`rejected`、1项`deferred`、0项`pending`、0项`blocked`。

## 7.发布环境和命令

- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\ccoi-pa-v1`
- N607 Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- N607 canonical root：`/home/szu2070436088/2510044040/CV-SincNet`
- GPU：预登记`GPU=0`，启动前按资源preflight核对；默认最多两项训练进程/GPU，不干预无关进程
- checkpoint：`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- WiSig：`Dataset_WigSig/ManySig.pkl`
- 旧C4 sidecar：`runs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A/C4/sidecar.pth`
- output root：`runs/phase1_ccoi_pa_m21_20260825/PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C`
- log：`logs/phase1_ccoi_pa_m21_20260825/PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C.out`
- 启动入口：`code/scripts/launch_phase1_ccoi_pa_m21_20260825.sh`

正式发布使用一个`git archive`release归档，只比较一次本地/远端归档SHA，并在全新release目录执行一次远端编译。真实checkpoint无query smoke通过后立即进入同一launcher的正式矩阵，不创建smoke授权artifact。

## 8.预期artifact

正式聚合artifact固定为14个：`split_manifest.json`、`sidecar_architecture_c1p.json`、`sidecar_architecture_c4p.json`、`sidecar_training_summary.json`、`duplicate_audit.json`、`q_conditional_probe.json`、`m0_exact_pair_retrieval.json`、`factor_matrix_c1p.json`、`factor_matrix_c4p.json`、`loto_residual_audit.json`、`gate_calibration_summary.json`、`gate_audit_summary.json`、`decision_manifest.json`和`final_report.md`。另外保留两个非样本级V3技术模型文件，但不计入14个聚合artifact。

不得提交或发布样本级q、theta、embedding、IQ或逐样本prediction stream。全流程`target_or_query_access=false`。

## 9.系统技术失败停止规则

只在以下情况停止：目标/query访问或审计标签泄漏；错误checkpoint、checkout、角色、fold或split；output/log冲突；同一确定性pre-artifact异常至少重复两次；NaN/Inf、OOM/Killed或无artifact进展；无法闭合14个聚合artifact；进程归属不清或可能影响无关任务。只终止该run明确绑定的进程树并保留全部partial artifact。

低性能、共同anchor覆盖不足、A_FAIL、A_PARTIAL或B_FAIL均是科学结果，不是技术停止理由。

## 10.明确延期和否定

- `REJECTED_EXTRA_GATE`：逐文件/逐成员SHA、seal、signature、receipt链、环境锁和额外审批不进入发布门；
- 延期：Soft-DTW、OT、强制码本均衡、多机制混合、Core90解冻和完整多seed确认；
- 当前q泄漏不作为M2.1自动技术停止条件，但禁止将其解释为正确物理challenge。

## 11.最终实验结果

### 11.1阶段判定

`decision_manifest.json`给出`ANALYZED/A_FAIL/NOT_RUN_A_GATE`。阶段A只有一项预登记条件失败：C4′的F3相对F2退化超过10%。其余增益、覆盖、条件化对照和敏感性条件均通过。阶段B按预登记状态机未拟合、未校准、未评估；两个gate artifact均明确记录`technical_failure=false`，因此这不是技术失败或缺失结果。

|阶段A条件|C4′结果|门槛|判定|
|---|---:|---:|---|
|F3相对F0平均改善|22.6303%|至少5%，CI下界>0|通过；3个mapping seed的CI下界最低16.2385%|
|F3相对F5平均改善|7.8748%|至少5%，CI下界>0|通过；CI下界最低1.3509%|
|F3相对F2平均退化|13.6870%|不超过10%|**失败**；3个mapping seed分别14.7101%、12.7918%、13.5591%|
|F3覆盖|100%|至少80%|通过|
|每TX至少2个跨RX关系|是|是|通过|
|主要cell最小样本数|10|至少10|通过|
|C4′相对C1′F3改善|20.5948%|至少3%，CI下界>0|通过；CI下界18.1951%|
|head/mapping方向|3/3、3/3|至少2/3|通过|
|2个satellite seed结论反转|否|不得反转|通过|

### 11.2F0—F9结果

下表为3个candidate mapping seed的macro NMSE均值，越低越好。每个mapping seed内部又平均3个同容量head seed和4个raw-disjoint fold；F2、F3、F5核心比较使用同一批16704个common anchors。

|Row|C1′|C4′|解释|
|---|---:|---:|---|
|F0 q-only/零theta|0.410335|0.253557|C4′的q分支本身已携带大量可预测信息|
|F1同packet raw-disjoint|0.210470|0.165774|同包上限最好，但包含共享packet状态|
|F2同TX/RX/day跨packet|0.212027|0.169189|theta在同receiver内较稳定|
|F3同TX/跨RX/同day|0.243731|0.193096|相对F0和F5有效，但相对F2退化超门槛|
|F4同TX/RX跨day|0.216111|0.172863|跨day损失小于跨receiver损失|
|F5异TX/同RX/day|0.274631|0.209778|正确TX支持优于错误TX支持，说明theta含TX信息|
|F6固定PA统计匹配|0.230523|0.183630|优于随机关系匹配F3，物理激励控制有帮助但未闭合跨RX差距|
|F7真实同步challenge|不可用|不可用|没有已验证的跨receiver同步物理事件ID|
|F8正确theta+异TX/异packet q|0.206770|0.173579|错误q仍接近F1/F2，暴露q–theta对应关系不够选择性|
|F9训练目标均值|0.410083|0.410083|无条件基线|

C1′同样在F3相对F0和F5上表现为正，但相对F2平均退化14.8730%。C4′降低了绝对F3误差，却没有把跨receiver退化压到10%以内。这一结果支持“PA集合表示含可迁移TX信息”，但不支持“当前theta已经形成足够稳定的receiver不变量”。

### 11.3训练曲线和选模

C1′和C4′均完整训练60轮，每轮45个step，全部120条epoch记录已解析，无非有限值。C1′在第50轮达到最高`V_select_fit`准确率92.5634%，第60轮为92.4384%；C4′在第53轮达到92.9384%，第60轮为92.7509%。末轮相对最佳轮仅下降0.1250pp和0.1875pp，没有崩塌或发散。

C1′总损失从2.3290降至1.0032，分类损失从1.7416降至0.4246；C4′总损失从2.4516降至1.0288，分类损失从1.7387降至0.4201。C4′的holdout损失从0.4374降至0.0676，DiD诊断从0.4020降至0.1818，证明优化过程正常。pair损失只从3.9008降至3.8467，下降很小，说明跨样本配对约束没有成为强识别信号。C1′不启用条件holdout目标，其holdout记录从0.1839升至0.3976只能视为只读诊断，不构成训练异常。

### 11.4独立性和近重复审计

源角色保持`L_s/U_s/V_cal/V_select=5880/52920/12600/12600`。V_select按84个TX×RX×day×eq cell和`block_size=10`拆为`V_select_fit=8001`、`V_audit_retro=4176`、guard=423；最小每cell有29个block，fit/audit的`base_index`交集为0。新C1′/C4′只用L_s训练、只用V_select_fit选模，审计集未进入训练或选模。

精确重复为0，guard block生效。最近邻规范化IQ相似度大于0.999的比例为15.3606%，大于0.995的比例为55.4556%；相似度中位数0.995938，99%分位数0.999740。最近`sig_i`间隔中位数157、99%分位数280。高相似度提示WiSig波形本身存在强公共结构，但较大的采集间隔、零精确重复和有效guard不支持“角色间只是相邻重复包”的解释。`V_audit_retro`仍只对本轮权重独立，不是研究历史完全未见集。

### 11.5q、码本和M0

q的ordered sequence probe平衡准确率为TX57.1011%、receiver52.5044%、day58.7261%；token shuffle后分别降至36.6891%、32.7278%、51.8842%，DeepSets分别为44.3614%、39.9720%、55.3318%。ordered相对shuffle的TX增量为20.4120pp，说明token顺序是捷径的重要来源，但shuffle和集合化后仍保留明显TX/RX信息。

在固定其他因素的条件probe中，固定RX/day预测TX为90.2758%，固定TX/day预测receiver为89.6083%，固定TX/RX预测day为90.3405%。当前q不能解释为纯激励条件；更准确的名称仍是受TX、receiver、day和位置共同污染的received-waveform excitation proxy。

硬码本48个code中只有5个在token层出现，packet dominant只有4个；每包平均使用3.4907个硬code。soft effective codes为35.2287，soft entropy为3.5619，表明soft分布并未数值塌缩，但argmax分配高度集中。该现象只作诊断，未参与优化或判定。

M0在同TX/RX/day/fold候选池上的Recall@1为17.2593%、Recall@5为36.9792%、median rank为10、MRR为0.2822、exact-pair distance AUC为0.7170。平均exact-pair margin为-0.001102，说明真实clean–satellite配对通常不是最近theta。当前theta有弱到中等的配对信息，但不足以支持精确物理challenge识别。

### 11.6leave-one-TX-out residual

6个held-out TX全部闭合，训练TX集合均不含被留出TX，16704个结果无NaN/Inf。LOTO residual macro NMSE为0.282016，4个fold分别为0.275509、0.235651、0.298927、0.317976。residual probe准确率为TX50.5328%、receiver28.2268%、day52.0295%；day接近二分类随机水平，但TX和receiver仍可预测。between-TX平均距离1.1721，same-TX cross-RX平均距离0.9790，后者仍为前者的83.53%。LOTO避免了旧HR的同TX泄漏，却没有得到干净的TX保留、RX去除分解。

### 11.7阶段B

阶段A未通过，故`gate_calibration_summary.json`和`gate_audit_summary.json`均为`NOT_RUN_A_GATE`。没有产生gate coverage、rescue、harm、clean/LEO增益或worst-receiver指标；这些值必须记为`N/A`，不能填0，也不能借用旧B实验结果。

## 12.N607发布记录

- 只读直连preflight：`VERIFIED`；服务器时间2026-08-25 17:16:45 CST，项目根可见；8张RTX 3090在检查时均为0%利用率、1MiB显存；
- 相关训练进程：未发现；预登记GPU 0可用；
- checkpoint、WiSig和旧C4 sidecar：目标文件存在；
- 新run、smoke和两份日志路径：均确认不存在；
- release归档：`E:\type10-7\local_artifacts\PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C_7d2a9d41.tar.gz`；
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C_7d2a9d41.tar.gz`；
- release目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C_7d2a9d41`；
- 唯一归档SHA256：`fa8d914960611534ddfdd2c994aede16db3ae76639ef77303a554e33c3d3afad`，本地/远端一致；
- 远端编译：`REMOTE_COMPILE_PASS`；
- 发布状态：release归档、远端SHA和远端编译均为`VERIFIED`；正式run随后已自然闭合为`ANALYZED`。

## 13.运行健康与完整日志分析

- 启动时间：2026-08-25 17:19 CST；
- launch owner PID：`2990576`；正式runner PID：`2991016`；
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C_7d2a9d41`；
- cmdline、run root、release和GPU 0绑定：一致；
- 真实checkpoint无query smoke：`PASS`；随后由同一launcher进入正式矩阵；
- 初始GPU占用：runner独占GPU 0，约266MiB，随后增长到约712MiB；
- 日志与artifact：smoke日志增长并闭合；正式run已生成`split_manifest.json`和`duplicate_audit.json`；
- Traceback/OOM/Killed/NaN/Inf：初始扫描未发现；
- 正式运行约从17:19持续到19:00，launch owner和runner均自然退出；只读复核时原PID已不存在，未发现该run仍在执行；
- 完整正式stdout、launcher stdout和smoke stdout均已读取，共9行，未出现Traceback、OOM、Killed、NaN或Inf；
- 13个JSON全部成功解析，14个预登记聚合artifact和2个V3模型文件均存在；
- 完整训练历史为C1′60轮+C4′60轮，未只读tail或抽样epoch；
- 最终状态：`ANALYZED`。系统运行健康，科学结论为A_FAIL。

## 14.设计正确性复核

### 14.1设计中被实验支持的部分

1. 两阶段门控正确。A_FAIL后不训练gate，避免用更复杂selector掩盖theta迁移不足。
2. C1′/C4′同模板、同参数量、同step对照有效地证明C4′整体F3误差更低。
3. 独立support bank、metadata硬关系、无fallback和3个mapping seed消除了旧审计的transductive support与q内生选邻问题。
4. 四fold轮转消除了固定位置单fold结论；所有fold和head seed方向一致。
5. M0、F6、F8和LOTO把“有TX信息”与“正确物理对应、跨RX稳定”分开，避免只凭单一NMSE晋级。

### 14.2实验暴露的设计局限

第一，预登记的“C4′相对C1′F3绝对改善”不能单独归因于challenge–theta交互。C4′在F0上已显著优于C1′；C1′从F0到F3的平均相对改善为42.9830%，反而高于C4′的22.6303%。按绝对NMSE下降计算，C1′为0.166604，C4′为0.060461。原对照证明的是“带q的C4′整体预测更好”，不是q与theta正确配对产生了更大的增量。

第二，F8中错误TX/错误packet q仍取得0.173579，接近F2的0.169189且优于F3的0.193096。这与M0负margin一致：当前q–theta对应关系缺少选择性。后续若重做challenge，必须把交互效应`(F0-F3)`、正确配对与错误配对差值、以及精确事件检索共同列为主判据，不能继续只比较C4′/C1′的绝对F3。

第三，`V_audit_retro`解决了权重直接泄漏，但没有解决架构和超参数的研究历史适配。该结果适合停止当前路线，不能作为最终泛化估计。若未来启动新机制，应在设计前冻结真正一次性`V_final`。

第四，F7保持不可用是正确处理，不是遗漏。`sig_i`没有跨receiver同步事件语义，强行构造F7会制造伪物理配对。真正验证条件系统辨识需要payload/前导或采集事件级同步键。

### 14.3实现与效率问题

本轮没有发现会使结果失效的P0/P1实现错误。协议角色、fold遮罩、common-anchor、LOTO训练TX排除、阶段状态机、不可覆盖输出和无query访问均由测试与artifact交叉验证。

工程上仍有三项可优化但不影响本轮数值：物理特征在不同mapping seed间重复计算；metadata关系候选被重复扫描；M0对16704个单位执行大规模全候选距离检查。这使单次审计耗时约1小时41分。后续实现应缓存与seed无关的PA特征和关系索引，并把M0改成分组矩阵化或分块最近邻；不得为优化耗时重跑或覆盖本run。

## 15.对修订计划的逐项闭合

- P0协议和独立性：已完成。A/B和旧C4只读，retro split、guard、重复审计和V3契约闭合。
- P1独立replay：已完成。C1′/C4′同模板训练，只用V_select_fit选模，4fold支持闭合。
- P2机制审计：已完成。F0—F9、q条件probe、M0、LOTO、3×3 head/mapping和2个satellite seed均有聚合artifact；F7按科学语义为不可用。
- P3 truth-blind gate：按计划未执行。A_FAIL时运行P3才是违反设计。
- P4路线：触发`STOP_CURRENT_PA_THETA_TRANSFER`。不进入连续challenge V3、gate扩展、OT、Soft-DTW、码本均衡或多seed确认。

用户计划中的14个聚合artifact已全部镜像到`docs/experiments/artifacts/PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C/`；未提交样本级q、theta、embedding、IQ、prediction stream或模型权重。

## 16.最终结论与下一路线

本轮回答了最关键的问题：PA sidecar theta确实包含TX相关信息，固定物理PA统计也比随机跨RX支持更有效；但当前theta跨receiver迁移仍比同receiver跨packet基准恶化13.6870%，且q–theta错误配对几乎不受惩罚。它不是足够稳定、足够选择性的条件系统辨识表示。

因此应停止当前PA theta迁移与truth-blind gate路线。若继续研究“条件系统辨识”，下一轮不应修补当前q或扩大seed，而应先取得真实payload/前导或采集事件级挑战键，建立可验证的跨receiver同激励配对，再做content、position和nuisance分解；若该数据条件暂时不可得，则转入多机制稳定比筛选，把PA只作为被审计的辅助机制，不作为Core90纠错operator。
