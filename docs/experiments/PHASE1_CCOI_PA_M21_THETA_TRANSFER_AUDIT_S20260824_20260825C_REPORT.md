# PA-M2.1独立theta迁移审计实验报告

## 1.当前状态

- run ID：`PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C`
- 当前最高状态：`LOCAL_VERIFIED`
- 性能结果：尚无；不得把本地测试、远端落地、PID、GPU占用或日志增长解释为性能结果
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

## 11.最终结果与问题

本节等待真实实验闭合后追加。当前没有性能结果。最终报告必须逐项记录阶段A/B状态、逐fold/seed/scene结果、覆盖、CI、q/M0/码本/LOTO诊断、gate rescue/harm、日志健康性、暴露问题和下一路线。

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
- 发布状态：`LANDED`，尚未启动，尚无性能结果。
