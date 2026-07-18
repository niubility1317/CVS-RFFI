# D42统一自动收缩LDA实验报告

## 1.实验身份与状态

- 实验ID：`d42_unified_shrinkage_lda_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
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

## 8.真实本地105行执行

- 隔离worktree：`E:\type10-7\code\snapshots\d42wt`，Git`55a76bc1f2a0de306febe613f822b607504c7d32`。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d42_unified_shrinkage_lda_20260718\local_support_screen_d42_v1`。
- 环境：本地`ssr-gpu`、`device=auto`、scikit-learn`1.7.2`；receipt wall time`33.823s`，外部命令wall time`41.5s`。
- 结果：105/105行完成，`selected_candidate_id=Z0_SUPPORT_ONLY`、`selected_positive_route=false`、`query_opened=false`、`formal_metric_claim_allowed=false`、`performance_claim_allowed=false`、`full_k10_fallback_reason=null`。
- D41已验证运行时通过只读包装器预加载并逐文件断言：`somph_predictor_bundle.py=49a05c6f…def48`、`somph_runtime_trust.py=4b1dee1d…1f9fc`、`stage2_predictor_bundle.py=bb27beaa…44aa9`；D42 core/Runner和candidate source closure均来自`55a76bc1`worktree。

执行参数与D41相同，只把Runner改为D42 worktree、输出改为本目录并锁定`--candidate-set d42_v1`。关键CLI为：

```powershell
python <只读runtime-preload-wrapper> `
  --before-root E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only `
  --before-seal E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --after-root E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only `
  --after-seal E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --component-dir E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output E:\type10-7\automation_reports\CV-SincNet\d42_unified_shrinkage_lda_20260718\local_support_screen_d42_v1 `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

第一次包装器尝试在support打开前因把摘要省略号SHA误补为错误中间串而fail closed，输出目录未创建。读取D41真实完整SHA后原配置重试成功；这属于本地启动包装错误，不是远端或实验失败。

## 9.完整同row结果

|候选|机制/角色|before-old|after-old|seen-new|同rowH均值|遗忘|joint floor|最低before旧类|最低after旧类|最低新类|final old→new/new→old/new→new|最终判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|`Z0_SUPPORT_ONLY`|identity回退|71.11%|48.33%|52.67%|48.97%|22.78pp|0|33.33%|13.33%|3.33%|N/A|matched基线|
|`D42-PROTOnet-CDA-ZID160`|ProtoNet matched基线|71.11%|48.33%|52.67%|48.97%|22.78pp|0|33.33%|13.33%|3.33%|N/A|matched基线|
|`B3_SINGLE_IQ_DIAG_FFTRF`|exact strong B3|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|当前合法比较器|
|`D42-D40-HNBR-INT8-NEGATIVE`|旧类主导负对照|85.56%|85.00%|15.33%|25.16%|0.56pp|0|66.67%|63.33%|0|2/N/A/N/A|诊断性负|
|`D42-D41-BEC-INT8-NEGATIVE`|新类主导负对照|86.11%|20.56%|78.67%|31.50%|65.56pp|0|76.67%|0|36.67%|142/0/32|诊断性负|
|`D42-USLDA-INT8`|D42正式路线|90.56%|81.67%|81.33%|80.63%|8.89pp|23.33%|76.67%|50.00%|70.00%|26/10/18|总体正信号，严格门失败|
|`D42-USLDA-FP32-MATCHED`|量化ablation|91.11%|81.11%|82.67%|80.97%|10.00pp|23.33%|80.00%|50.00%|73.33%|27/9/17|不可晋级；量化对照|

D42 int8相对B3总体改善before`+2.78pp`、after-old`+6.11pp`、seen-new`+8.67pp`、同rowH`+7.28pp`、遗忘`−3.33pp`，并把final三类错误从33/22/19降到26/10/18。这是D40–D42中第一个同时改善旧类与新类总体的机制正信号。但joint floor仅与B3持平23.33%，最低after旧类从60%降到50%，因此不能晋级。

## 10.场景与逐类结果

|场景|方法|before-old|after-old|seen-new|同rowH|遗忘|joint floor|
|---|---|---:|---:|---:|---:|---:|---:|
|clear|B3|88.33%|75.00%|82.00%|77.51%|13.33pp|30.00%|
|clear|D42 int8|98.33%|90.00%|94.00%|91.53%|8.33pp|40.00%|
|low-elev|B3|85.00%|75.00%|70.00%|71.71%|10.00pp|20.00%|
|low-elev|D42 int8|85.00%|76.67%|74.00%|73.73%|8.33pp|20.00%|
|rain|B3|90.00%|76.67%|66.00%|70.85%|13.33pp|20.00%|
|rain|D42 int8|88.33%|78.33%|76.00%|76.64%|10.00pp|10.00%|

|注册角色|匿名handle前缀|B3|D42 int8|差值|
|---|---|---:|---:|---:|
|before-old|`1f33`|96.67%|90.00%|−6.67pp|
|before-old|`33bb`|90.00%|93.33%|+3.33pp|
|before-old|`75aa`|90.00%|93.33%|+3.33pp|
|before-old|`8b02`|83.33%|76.67%|−6.67pp|
|before-old|`a53c`|86.67%|100.00%|+13.33pp|
|before-old|`f8df`|80.00%|90.00%|+10.00pp|
|after-old|`1f33`|93.33%|86.67%|−6.67pp|
|after-old|`33bb`|90.00%|93.33%|+3.33pp|
|after-old|`75aa`|73.33%|90.00%|+16.67pp|
|after-old|`8b02`|73.33%|50.00%|−23.33pp|
|after-old|`a53c`|60.00%|76.67%|+16.67pp|
|after-old|`f8df`|63.33%|93.33%|+30.00pp|
|seen-new|`09f8`|40.00%|70.00%|+30.00pp|
|seen-new|`1c2a`|86.67%|90.00%|+3.33pp|
|seen-new|`b8fb`|76.67%|70.00%|−6.67pp|
|seen-new|`d3af`|86.67%|86.67%|0pp|
|seen-new|`f608`|73.33%|90.00%|+16.67pp|

clear场景已接近目标，但low-elev/rain的下尾仍不稳。通用floor失败主要来自旧类`8b02`，after比B3低23.33pp；new侧`b8fb`低6.67pp且`d3af`只持平。历史handle只用于解释，下一轮不得为这些ID设置专属分支或阈值。

## 11.严格门、量化、完整日志与资源

|门|结果|直接证据|
|---|---|---|
|before总体/场景/逐类/floor严格优于B3|失败|聚合提高，但9/15折总体未严格提高、15/15折至少一旧类未严格提高；75/90逐类单元未严格提高|
|after-old总体/场景/逐类/floor严格优于B3|失败|聚合提高，但5/15折总体未严格提高、15/15折至少一旧类未严格提高；66/90逐类单元未严格提高；最低旧类50%<60%|
|seen-new总体/场景/逐类/floor严格优于B3|失败|聚合提高，但6/15折总体未严格提高、15/15折至少一新类未严格提高；56/75逐类单元未严格提高；`b8fb`退化、`d3af`持平|
|H/joint floor严格优于B3|失败|10/15折H或joint未严格提高；joint聚合23.33%只与B3持平|
|final三类混淆`<33/<22/<19`|通过|26/10/18|
|pairwise错序和最低margin严格优于B3|失败|错序31/20/19优于B3的42/31/25，但D42 score尺度下最低margin为−69.19/−139.95/−39.13，未通过预锁最低margin门|
|int8/FP32 0 argmax变化、0 margin翻转|失败|before argmax变化1、final变化3、margin符号翻转3；最大outer score误差1.0283|
|lifecycle/ground/source/state/resource|通过|old-only B20先物化before；ground逐bit；fit-held交集0；sklearn1.7.2；target vector int8；20step/2016参数/8583B|

D42 int8 pairwise的old→new/new→old/new→new margin统计分别为：最小值`−69.19/−139.95/−39.13`，P05`−11.51/−4.14/−9.61`，中位数`23.71/19.34/19.90`，错序`31/20/19`。B3对应最小值`−3.66/−4.71/−4.77`，P05`−1.77/−1.94/−1.43`，中位数`1.42/1.32/1.62`，错序`42/31/25`。D42减少错序但扩大原始score尺度，说明跨方法比较未标准化margin绝对值会混入尺度；本轮预注册门保持不变并据此失败，下一轮只能预先改用尺度不变的规范化margin。

完整105行均为finite。D42 int8的15条独立fit各有20条B阶段trace，共300条，全部完整：平均loss从1.031996降到0.102685，CE从1.031996降到0.102493，support accuracy从95.14%升到100%，末步prototype anchor loss0.003828、gradient norm0.13535。FP32 matched复用同一15×20轨迹；不得以support100%替代outer证据。

|资源|D42 int8真实outer值|硬门|
|---|---:|---:|
|trainable params|2016|≤80,000|
|epoch/optimizer step|20/20|≤30/50|
|persistent state|8583B|≤256KB|
|估算adaptation MAC|65,442,816|finite/报告|
|估算query MAC|6624|finite/报告|
|CUDA peak|22,886,912B，仅B20 CUDA scope|报告|
|host FP64 covariance peak|未实测，明确标记|正式Pareto前补齐|
|sklearn runtime|1.7.2，lock pass|精确匹配|

量化误差主要来自FP16 intercept而非int8 coefficient：final coefficient最大元素误差各fold为0.0210～0.0460，final intercept最大误差0.3431～0.9990，support score最大误差0.3544～1.0273。outer selector失败后未执行selected-only full-K10，因此没有D42真实batch1平均/P95 latency；不得补跑support refit绕过性能门。

## 12.Artifact闭包

|artifact|SHA256|大小|
|---|---|---:|
|`training_log.jsonl`|`4ee51dd3d21ae8751bfaa64eb82d2a5a5371728fc7c1502bdb3af221d349614a`|3,775,653B|
|`support_audit.json`|`89f4bca56fd35de36f8fd0a8adc541e572243785d1b53c680765eb6f0a1c37ad`|311,428B|
|`selection.json`|`599bf35d328b9742b4b2906d91c6890a80a77ccb632946232583e39840d55def`|2250B|
|`resource_audit.json`|`61e41c3bce7dacded91747305b8633e70c8157198b277d7baa208a56e1184fe1`|9039B|
|`geometry_audit.json`|`f4c935069d7e82be775150b1a62a345a70bb0b2e2836949b156c2b8be014474d`|4448B|
|`RECEIPT.json`|`ec0b58515f35bc6387ea2ca76f96a7146cb83747386aed0798d9dfd744119e28`|3065B|

candidate lock为`430f871d…e41538`；D42 core为`c2caacf7…e9f1b20`，Runner为`d66df9a4…bdd7eca`；source closure在support打开前后相同。真实ground NPZ为`3c08c823…0267d7`，entry/exit逐bit一致；query/source/clean/role/quota/global assignment全部关闭。

## 13.D40–D42强制技术复盘

|轮次|单一机制|before-old|after-old|seen-new|H|最低旧类/新类|主要失败模式|结论|
|---|---|---:|---:|---:|---:|---:|---|---|
|D40|HNBR old-null残差重编译|85.56%|85.00%|15.33%|25.16%|63.33%/0|旧类压倒新类，127/150 new→old|淘汰|
|D41|四view BEC联合梯度|86.11%|20.56%|78.67%|31.50%|0/36.67%|新类压倒旧类，142/180 old→new|淘汰|
|D42|full-batch B20＋统一auto-shrinkage LDA|90.56%|81.67%|81.33%|80.63%|50.00%/70.00%|总体联合改善但旧类floor、场景稳健性和量化边界失败|保留机制正信号，当前实现淘汰|

三轮都直接评估了同一row注册前old、注册后old、seen-new、H、逐类、遗忘和三类混淆；没有只优化一侧。D40/D41证明分离old/new列或联合梯度都容易发生标尺翻转；D42证明统一等先验共享判别几何是目前最强正信号，但full covariance下尾仍不稳，且大幅值LDA intercept使FP16量化跨越少数边界。

协议复核：三轮均只读单份LEO weak固定received IQ；old/new support和held physical token互斥；query sealed；无clean/source、query truth/role、batch class count、quota或global assignment；Phase1 ground int8 entry/exit逐bit相同。D42仍同时把旧类适应和新类注册放在同一统一LDA中，没有偏离active objective。

相对K10/new5正式目标，D42 int8 after-old仍差10.33pp，最低旧类差38pp，seen-new差10.67pp；当前只覆盖receiver`20-1`、seed`713101`的development support-held cell，不能外推到确认矩阵。下一轮最高价值实验锁定为D43结构化协方差：保持D42统一类对称LDA和old-only B20不变，只把full shared covariance替换为预定义`3-block(z/FFT/RF)`block-diagonal与纯diagonal两个高信息量结构，并把LDA系数/intercept编译为共享中心＋类残差的量化稳定等价score。D43不扫描阈值、rank、lr或类专属参数；先在同一15折support-held代理比较通用floor、场景稳健性和0量化翻转，再决定是否实现完整Runner。D42负结果不访问N607。
