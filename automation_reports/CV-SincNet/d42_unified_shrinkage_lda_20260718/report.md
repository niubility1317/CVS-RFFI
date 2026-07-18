# D42统一自动收缩LDA实验报告

## 1.实验身份与状态

- 实验ID：`d42_unified_shrinkage_lda_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`LOCAL_IMPLEMENTATION_VERIFIED_PENDING_REAL_105_SUPPORT_ONLY`
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景；复用D18的`p2_min_v1/VALIDATED_ONCE`固定received-IQ enrollment capsule，query保持sealed。
- 目标：用同一个类对称共享判别几何同时提高Stage2-B old outer-held泛化、Stage2-C注册后old/new准确率和通用floor，避免D38–D41的两套score列失衡。

D42是D37–D39正式复盘后的第三轮探索。D42完成后必须先记录D40–D42技术复盘，才允许设计D43。设计、闭式解、单测、量化误差或资源通过均不等于性能成功。

## 2.直接证据与路线选择

同一D18 support-held cell的当前合法比较器为exact strong B3：before-old87.78%、after-old75.56%、seen-new72.67%、H74.08%、最低旧类60%、最低新类40%、最终argmax old→new33/180、new→old22/150、new→new19/150。D41报告中的31/150和25/150是pairwise true-class margin口径，D42必须同时保存最终argmax混淆与pairwise错序，不能混用。

设计前只读探针先验证了exact B3旧头编译：15/15折、180个held-old样本的FP32→D38两级residual-int8 argmax变化为0，最大score误差`1.1397e-4`，old state最大5217B。随后验证并淘汰D42-FONR：seen-new升至78%，但180/180旧样本被new列吞噬，after-old=0；int8/FP32同判，故失败来自几何标尺而非量化。

对三个统一类对称闭式候选的同一15折实测为：

|候选|before-old|after-old|seen-new|pooled-H诊断|最低旧类|最低新类|old→new|new→old|new→new|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|exact strong B3|87.78%|75.56%|72.67%|74.08%|60.00%|40.00%|33|22|19|比较器|
|B3-metric qKNN1|83.33%|75.00%|71.33%|73.12%|53.33%|50.00%|26|21|22|局部相似度有信号但联合不升|
|ridge classifier|85.56%|72.22%|66.00%|68.97%|56.67%|6.67%|34|29|22|淘汰|
|legacy-minibatch metric＋自动收缩LDA|91.67%|81.11%|82.00%|81.55%|50.00%|70.00%|27|10|17|机制发现探针；K20 step超限，不进入实现|
|D38 full-batch B20 metric＋自动收缩LDA|91.11%|81.11%|82.67%|81.88%|50.00%|73.33%|27|9|17|D42锁定主路线；floor风险待正式实现验证|

表中H是由全部15折汇总old/new准确率计算的只读探针`pooled-H`，不是Runner正式采用的15折同rowH均值；exact B3的正式同rowH均值仍为73.35%。实现审查发现legacy exact B3是20epoch但按32行minibatch，K20会达到80 optimizer step，超过正式50步硬门。改用D38逐公式锁定的full-batch B20 metric后，LDA正信号保持且K1/5/10/20均固定20step，因此正式D42采用full-batch版本。LDA相对B3同时改善四个汇总诊断，但最低旧类从60%降到50%，仍只是值得实现的protocol-valid正信号，不是可晋级结果。

## 3.锁定数学机制

基础288维表征不变；Stage2-B对角度量复用D38的B3式full-batch B20实现：

\[
x=\operatorname{norm}([\operatorname{norm}(z_{160});4\operatorname{norm}([FFT_{96};RF_{32}])]),\qquad
h=\operatorname{norm}(x\odot e^{\ell_B}).
\]

`\ell_B`只由old support按D38锁定的full-batch AdamW20得到：lr`0.01`、weight decay`0.002`、feature noise`0.01`、prototype anchor`0.05`、gradient clip`5`和相同block bounds；K变化不改变20个optimizer step。D42的单一主要差异是：B阶段和C阶段都不再给old/new使用不同来源的head；而是在当前全部注册类上，用相同公式拟合class-balanced自动收缩LDA。具体锁定sklearn1.7.2`lsqr/shrinkage=auto`数值路径：每类support分别StandardScaler标准化、Ledoit-Wolf估计并按feature scale重标，再按等先验加权为共享within-class covariance`\widetilde\Sigma_w`；不是对pooled residual执行一次Ledoit-Wolf。等先验判别函数为：

\[
s_c(h)=h^T\widetilde\Sigma_w^{-1}\mu_c-
\frac12\mu_c^T\widetilde\Sigma_w^{-1}\mu_c.
\]

- Stage2-B：使用无new参数的old-only B20 helper逐公式复现D38 full-batch轨迹，先拟合`\ell_B`并立即物化、冻结old-only D42 LDA before artifact；只有before snapshot完成后，D42 fit才首次解析new support并进入all-registry拟合。测试必须证明同seed同old support下helper的`log_diag`与D38 arm-A before逐bit一致，且改变new support不改变before artifact。
- Stage2-C：冻结`\ell_B`，用同一row的old＋new合法support重新闭式拟合全部注册类LDA；所有类使用相同公式、相同等先验和相同收缩规则。
- formal state将每类LDA coefficient按固定288维三block做两级residual-int8量化；intercept使用FP16，`\ell_B`使用FP32。matched FP32仅作量化ablation。
- Phase1 ground int8只读且不参与LDA covariance/mean；entry/exit必须真实重哈希相同。
- query逐样本读取full 288维view并同时计算全部注册类score；无query role、quota、排序、全局重排或跨query状态。

不扫描shrinkage、prior、temperature、bias、threshold、rank、step或类专属参数。K1每类只有一个support、within-class covariance不可估计时，预锁定对所有类统一使用单位协方差`I`，等价于等先验nearest-centroid判别；这不是sklearn auto的隐式输出，必须在state/audit中显式标为`unit_covariance_equal_prior_nearest_centroid`，且不得复制、扰动或伪造物理样本。

## 4.预期可观察结果与停止条件

D42 int8只有全部满足才可进入selected-only full-K10或N607：

1. before-old总体、每场景总体和最低旧类均严格高于exact B3；不得以91.67%总体掩盖旧类下尾。
2. after-old总体、每场景总体、每旧类和最低旧类均高于exact B3；forgetting不高于B3。
3. seen-new总体、每场景总体、每新类和最低新类均高于B3；最低新类必须>40%。
4. H、joint floor和下尾统计均高于B3；同row指标完整，不拼接跨fold最佳值。
5. 最终argmax old→new<33、new→old<22、new→new<19；pairwise三类margin错序也分别严格优于B3的独立口径。
6. D42 int8相对matched FP32的before/final outer-held argmax变化为0，pairwise margin符号翻转为0；保存最大score误差。
7. formal coefficient全部residual-int8、target intercept FP16、无FP32 target coefficient sidecar；ground entry/exit hash相同。
8. full-batch B3 metric20step、LDA闭式0 optimizer step、总epoch/step=20/20、K1/5/10/20步数不变、参数≤80k、state≤256KB、无dense query graph。

任一性能门失败即`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不打开query、不访问N607、不扩K/receiver/seed/new-count。普通负结果不结束active goal；D40–D42复盘后继续下一机制。

## 5.最小实现与验证矩阵

|候选|角色|
|---|---|
|identity-only single-qKNN|回退/遗忘基线|
|ProtoNet CDA|强制matched基线|
|exact strong B3 FP32|当前合法比较器|
|D40-HNBR int8|旧类主导负对照|
|D41-BEC int8|新类主导负对照|
|D42-USLDA int8|唯一可晋级路线|
|D42-USLDA FP32|matched量化ablation|

固定7×3场景×5fold=`105`行，每fold8-shot fit、2-shot physical-held。全部候选在同一Runner真实执行并共享held physical SHA；direct ADV3B02继续作为相同old-held的0-support锚，不占候选轴。outer全部门通过后才执行selected-only full-K10状态、时延和资源审计。

计划文件：`code/cvsrffi/stage2_d42_unified_shrinkage_lda.py`、`tests/test_stage2_d42_unified_shrinkage_lda.py`、`code/scripts/run_d25_support_only_concat.py`和`tests/test_run_d42_unified_shrinkage_lda_integration.py`。所有修改先在本地`ssr-gpu`验证、独立审查并提交；当前不进行N607 preflight或SCP。

## 6.当前版本与风险

- Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`，分支`codex/cvs-rffi-release-20260626`；D42开始前HEAD为`b7fd178209d57afd6f66ed0287ce1b188840a0ee`。
- 根目录`E:\type10-7`不是Git仓库；本报告同步镜像到根`automation_reports`，版本化权威在Git承载面。
- 工作树已有大量用户/其他任务改动；D42只stage明确列出的新文件和Runner精确改动，不覆盖无关内容。
- 最高风险是自动收缩LDA提高总体却继续损伤某个旧类floor；第二风险是coefficient/intercept量化改变边界；第三风险是K1 covariance退化。部署语义锁定`w_c=\widetilde\Sigma_w^{-1}\mu_c`为precision-weighted target prototype：全部target-old/new`w_c`必须residual-int8，`\mu_c/\widetilde\Sigma_w`不持久化、无FP32 coefficient sidecar；FP16 intercept只是标量校准，FP32`log_diag`是共享metric。三项风险必须由逐类outer-held、matched FP32和K闭包测试直接否证。

## 7.实现与本地验证

|文件|用途|
|---|---|
|`code/cvsrffi/stage2_d42_unified_shrinkage_lda.py`|old-only B20 helper、before/final USLDA、K1单位协方差、int8/FP32 state、pairwise和资源审计|
|`code/scripts/run_d25_support_only_concat.py`|`d42_v1`七候选105行、三类final/pairwise口径、strict selector、full-K10门和artifact闭包|
|`tests/test_stage2_d42_unified_shrinkage_lda.py`|B20逐bit匹配、poison-new时序、sklearn数值路径、K/new-count、量化、置换和逐样本测试|
|`tests/test_run_d42_unified_shrinkage_lda_integration.py`|105行physical闭包、12个独立selector反例、ground/source/state/resource/full-K10门测试|

本地`ssr-gpu`独立验证：

- 4个D42文件`python -m py_compile`通过。
- D42核心＋Runner：`44 passed`。
- D38/D40/D41/D42 Runner邻接回归：`68 passed`。
- 核心子任务另执行D42＋D38 core回归：`44 passed`。
- `git diff --check`通过；Windows pytest临时目录清理告警未改变退出码0。

实现明确区分：`formal_target_vectors_int8_no_fp32_sidecar=true`只描述全部target-old/new precision-weighted prototype为residual-int8；不会把含FP16 intercept和FP32共享`log_diag`的整个state误写为int8-only。所有类别数统一由锁定sklearn covariance/means显式求`W=lstsq(Σ,μ^T)^T`，二分类不使用中心化判别轴替代类原型；运行时和candidate lock精确要求scikit-learn`1.7.2`，版本漂移fail closed。int8/FP32 margin翻转按`margin<=0`布尔边界比较，final argmax new→old与new→wrong-new互斥且合计为新类最终错误数。

本阶段只证明实现、协议与证据闭包可执行，不构成性能晋级。下一步在隔离Git worktree运行真实105行development support-held矩阵；只有逐场景、逐类、floor、H、三类混淆、pairwise、量化和资源门全部通过才允许full-K10或N607。
