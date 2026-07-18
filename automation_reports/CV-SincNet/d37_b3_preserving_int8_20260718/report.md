# D37 B3-preserving int8注册实验报告

## 1.实验身份与目标

- 实验ID：`d37_b3_preserving_int8_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
- 科学目标：在`p2_min_v1`下，以同一target receiver的固定单次`leo_*_weak`接收IQ形成K10旧类与新类support，仅用support内部物理LOSO证据，保留B3的Stage2-B目标域几何并完成target-old/new正式int8注册；逐样本面对全部注册类，不打开development query。
- 比较对象：`identity-only single-qKNN`、`D25-C0-DIM-CONCAT`、`D33-B3-FAST-FISHER-SPHERICAL-BALANCED`以及D37-A/B/C。

## 2.D34–D36三轮强制技术复盘

三轮均为同一合法K10 development cell的完整support-only屏，均未打开query，不能写成正式性能结果。D34与D35各解析105/105个structured rows且receipt五项哈希一致；D36同样完成105/105行并闭合artifact哈希。三轮直接推动旧类适应与新类注册的共同问题，没有新增数据权限、clean/source访问、query真值、角色Oracle、class quota、全局重分配或dense query graph。

|轮次|主要机制|注册前旧类|注册后旧类|seen-new代理|H|遗忘|关键完整日志诊断|结论|
|---|---|---:|---:|---:|---:|---:|---|---|
|D34-C|冻结FAST旧列＋winner-conditioned稀疏碰撞边＋int8新原型|82.22%|71.11%|57.33%|62.23%|11.11pp|180个outer held旧样本有20次旧→新侵入；75个new class-fold中68个不可达；最坏单折old/new/joint floor均为0|拒绝稀疏可见性门；冻结旧列不能阻止新增列越界|
|D35-C|冻结FAST旧列＋所有新类有限score＋fit旧support最大残差安全阈值|82.22%|55.00%|55.33%|53.17%|27.22pp|180个outer held旧样本有49次侵入；68个new class-fold不可达；winner校准cell过半缺证据并回退|拒绝硬winner分桶、fit最大残差阈值和继续调buffer/原型数|
|D36-A|joint compiled int8旧/新头＋无校准|81.11%|65.56%|53.33%|57.80%|15.56pp|注册前量化旧头已低于同折B3；outer held侵入28次、不可达51个class-fold|拒绝重建并联合编译旧头|
|D36-B|同上＋只读ground弱锚＋常数OOF校准|80.56%|62.22%|56.00%|57.91%|18.33pp|outer held侵入32次、不可达49个class-fold|ground弱锚与常数偏置未解除重叠|
|D36-C|同上＋fixed 6D OOF IRLS统一margin校准|80.56%|66.11%|52.00%|56.82%|14.44pp|outer held侵入25次、不可达53个class-fold；loss单调、无NaN/Inf、资源全通过|连续校准训练稳定，但错误的旧类基准几何使其无晋级价值|

### 2.1已拒绝路线

1. 以winner-conditioned edge或`nonedge≈winner_score-2`控制新类可见性。
2. 把旧score prefix逐bit不变或fit support不退化当作最终all-class argmax安全证明。
3. 在硬winner selector上继续增加top-k边、原型数、buffer、floor multiplier或epsilon。
4. 继续从support robust prototype重建、适配并联合编译旧类头；D36证明该步骤在注册前就破坏B3旧域结果。
5. 按历史难类ID设置白名单、专属阈值、权重或定向保护；逐类失败只用于诊断通用floor。
6. 在support-held硬门未过前打开query、进入K1/K5/K20或扩张确认矩阵。

### 2.2保留的证据与剩余假设

- B3的正值对角度量`log_diag`及最终单位类权重是当前最强合法Stage2-B support-only几何；D37直接量化B3最终旧类权重，不再重新估计另一套旧类原型。
- 新类在完全相同的B3变换空间内以同一类无关公式形成单位权重并量化；所有样本对全部新类始终取得有限score。
- 注册后旧类int8权重字节、scale和inverse norm必须append-only、逐bit不变；安全性仍以outer held最终all-class old→new侵入直接判断。
- D36的连续校准不再复用；D37改用可审计的support-OOF硬可行区间，区间为空即否证“单一公共offset足够”，不得用软loss掩盖冲突。

## 3.D37预注册机制

设B3在旧support上得到`d=exp(log_diag)>0`和单位旧类权重`w_i`。D37保留共享`log_diag`算子，并对每个权重按固定特征块`(160,96,32)`做两级残差int8量化：

```text
q1_ib=round(w_ib/scale1_ib)∈int8
e_ib=w_ib-scale1_ib*q1_ib
q2_ib=round(e_ib/scale2_ib)∈int8
u_i=concat_b(scale1_ib*q1_ib+scale2_ib*q2_ib)
z(x)=normalize(x⊙d)
g_i(x)=18<z(x),u_i>
```

每个新类用相同变换后的合法support形成类无关单位权重`v_j`，再使用完全相同的两级int8量化。注册后的基础score为`18<z(x),[u;v]>`。旧类`q1/q2/scale1/scale2`前缀从注册前状态原样append到注册后状态，不能因新类注册或校准而修改；state不保存FP32 target prototype。

对于inner rank-pair OOF基础分数，令公共new-group offset为`b`，预注册margin为`m`。全部旧support-held行给出安全上界：

```text
U=min_x_old(max_old(x)-max_new_raw(x)-m)
```

全部新support-held行给出可达下界：

```text
L=max_x_new(max(max_old(x),second_new_raw(x))-true_new_raw(x)+m)
```

只有`L<=U`时才取`b=(L+U)/2`，否则该候选fail closed。最终score为`[old_raw,new_raw+b]`。OOF可行只用于开发拟合，不能冒充outer held或query安全。

D37只保留三个高信息量臂，主要差异仅是固定margin：

|候选|旧/新int8几何|support-OOF校准|用途|
|---|---|---|---|
|D37-A|B3-preserving residual int8旧/新权重|硬可行区间，`m=0`|检验公共offset是否存在|
|D37-B|同A|硬可行区间，`m=0.05`|检验小正margin鲁棒性|
|D37-C|同A|硬可行区间，`m=0.10`|检验更严格的双侧margin|

所有校准只来自outer-train内部预登记rank-pair OOF行；不读取outer held标签以拟合，更不读取query。量化与区间求解均为闭式，0epoch、0 optimizer step。

声明边界：`B3-preserving`只表示旧权重直接来自B3、旧/新独立量化后旧字节append-only，以及fit-support决策门；任何有限精度量化都可能翻转未见过的近边界样本。因此晋级必须以每个matched scene×outer-fold×old-class对FP32 B3非劣为准，不能声称全输入域数学等价。OOF来源由runner实际构造的rank-pair排除路径和唯一physical ID审计保证，core中的source字符串本身不是安全证明。

## 4.可观察预期、硬门与停止条件

最小矩阵固定为7候选×3场景×5个outer physical folds=105行：`Z0`、`D25-C0`、`B3`、`D33-FAST`、D37-A/B/C。每行同时保存注册前old、注册后old、seen-new代理、H、forgetting、全部逐类、侵入和physical LOSO可达性。

D37候选只有全部满足以下门才可打开锁定development query：

1. 注册前量化旧类总体与每个旧类均不弱于matched FP32 B3。
2. 注册前→注册后旧类int8前缀字节、scale、inverse norm逐bit不变。
3. outer held最终all-class old→new侵入为0，而不是只看fit support。
4. 三个场景全部新类physical LOSO`margin_min>0`，不存在不可达class-fold。
5. 注册后旧类、新类、H、forgetting、逐类old/new floor同时不弱于matched B3/identity中更强者。
6. target-old/new实际预测组件均为int8生命周期，状态<256KB、参数<=50k、epoch<=20、optimizer steps<=20、无dense query graph和query-dependent batch optimization。

任一关键门失败即记`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不打开query、不扩展确认矩阵；若仅注册前旧类不弱于B3而Stage2-C仍呈侵入—不可达重叠，则下一轮必须更换分离机制，不能继续扫offset。

## 5.协议、数据与版本边界

- 协议：`protocol_schema=p2_min_v1`；复用匹配`VALIDATED_ONCE`的现有D18 development cell，不因method变化重验数据。
- 输入：每个physical_sample_id仅一份固定`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`接收IQ；场景之间及support/query物理ID不交。
- 权限：support-only fit/选择；query保持sealed；不访问clean/raw、source样本、query真值/角色/数量或类quota。
- 评分：每个样本独立面对全部已注册类，无跨query联合计算。
- 根目录`E:\type10-7`不是Git仓库；本报告另镜像到Git仓库`E:\type10-7\github_publish\CVS-RFFI-repo`。开始设计时该分支相对origin ahead 1603，存在大量与D37无关的既有修改/未跟踪文件，后续只暂存D37专属文件与对共享runner的最小差异。

## 6.实现、验证与运行记录

|项目|当前值|
|---|---|
|本地文件变更|新增`stage2_d37_b3_preserving_int8.py`、两份D37测试、traceability与本报告；最小修改共享`run_d25_support_only_concat.py`接入`d37_v1`|
|本地环境|`ssr-gpu`|
|窄验证命令|`python -m py_compile ...`通过；D37 core+integration 24 passed；D34–D37聚焦回归43 passed；`--help`包含`d37_v1`；`git diff --check`通过|
|Git commit|实现提交`fb2f39f01fb771e600c4776550b0c5ce6090cc69`，证据报告由后续独立文档提交承载|
|N607 sync目的地|未同步|
|服务器命令/环境/CWD|未启动；当前N607已有runtime组合不能同时满足模型与NumPy依赖，不修改远端环境|
|本地隔离worktree|`E:\type10-7\code\snapshots\d37wt`，起点`fb2f39f0`；为通过D18签名闭包，3个runtime文件按已验证D36闭包逐字节恢复|
|run/log目录|`E:\type10-7\automation_reports\CV-SincNet\d37_b3_preserving_int8_20260718\local_support_screen_d37_v1`；成功stdout为同级`local_support_screen_d37_v1.retry1.stdout.log`|
|PID/GPU|本地前置只读检查：RTX5070Ti，2226/16303MiB、3%；运行已结束，无常驻任务|
|实际artifact|`training_log.jsonl`、`selection.json`、`resource_audit.json`、`geometry_audit.json`、`support_audit.json`、`RECEIPT.json`、两份stdout|

隔离worktree运行时源码SHA256：D37 core=`e7e05ac2b5498c3a24dcb8099f43f7a93d3712f388a8b18a2826b7220adf4760`；runner=`b5a35be3cfd790796a80e9753a9dafd50dd37e90d3c4500eb3b706fd44dcd31d`。Windows checkout换行会改变原始字节，因此以receipt内隔离worktree哈希作为本次运行闭包，不把主worktree文本哈希混用为运行哈希。恢复的D18签名闭包文件SHA256为`somph_predictor_bundle.py=49a05c6f...def48`、`somph_runtime_trust.py=4b1dee1d...c1f9fc`、`stage2_predictor_bundle.py=bb27beaa...944aa9`，期望闭包`b0b7f2c2...9606f`；runner成功打开D18输入，说明闭包验签通过。

成功运行的可复现命令如下。所有路径均为本地只读D18 capsule或D22组件；唯一变化是方法候选集和输出目录。

```powershell
conda activate ssr-gpu
python E:\type10-7\code\snapshots\d37wt\code\scripts\run_d25_support_only_concat.py `
  --before-root E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only `
  --before-seal E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json `
  --before-formal-policy-authorization E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json `
  --before-signed-policy-authorization-envelope E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only `
  --after-seal E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json `
  --after-formal-policy-authorization E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json `
  --after-signed-policy-authorization-envelope E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output E:\type10-7\automation_reports\CV-SincNet\d37_b3_preserving_int8_20260718\local_support_screen_d37_v1 `
  --device auto --mode development_select_unverified_component --candidate-set d37_v1
```

首次命令误用了本地另一份D22 class-binding，其SHA为`4f701...`，在第一fold前被`ADV3B02 class binding contract drift`拒绝；未创建输出目录或receipt。该输入路径错误保存在`local_support_screen_d37_v1.stdout.log`，不属于D37候选数值结果。修正为上述D19锁定binding后一次完成，耗时18.5965s。

当前已知未闭合项：D37只预注册K10 development cell。K1每类仅一个物理support，无法构造排除自身的新类原型并完成physical OOF校准；在预注册、support-only且不借用K5/K10结果的统一K1规则出现前，D37不得进入目标K1确认行。

## 7.完整K10 support-only结果

|候选|机制|receiver/TX split|K|seed/场景|before-old|after-old|seen-new|H|forgetting|joint floor|侵入/不可达|最终判定|
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---|
|Z0|identity-only single-qKNN|20-1/6旧+5新|10|713101/3场景|71.11%|48.33%|52.67%|48.97%|22.78pp|0|—|诊断基线|
|D25-C0|DIM-CONCAT|同上|10|同上|71.67%|50.56%|54.00%|50.35%|21.11pp|0|—|receipt选择但`selected_positive_route=false`|
|B3_SINGLE_IQ_DIAG_FFTRF|强B3比较器|同上|10|同上|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|—|本cell最强比较器|
|D33-FAST|Fisher闭式旧头＋球面注册|同上|10|同上|82.22%|70.00%|59.33%|62.19%|12.22pp|3.33%|—|弱于强B3|
|D37-A|Fisher旧头两级residual-int8＋公共offset，`m=0`|同上|10|同上|82.22%|71.11%|58.67%|62.99%|11.11pp|0|33/40；0/15可行|负例，不可晋级|
|D37-B|同A，`m=0.05`|同上|10|同上|82.22%|71.11%|58.67%|62.99%|11.11pp|0|33/40；0/15可行|负例，不可晋级|
|D37-C|同A，`m=0.10`|同上|10|同上|82.22%|71.11%|58.67%|62.99%|11.11pp|0|33/40；0/15可行|负例，不可晋级|

表中D37的`侵入/不可达`依次为全部180个outer held旧样本中的旧→新次数、75个new class-fold中的不可达数。D37为空区间时不允许构造公开注册state，表内after/new/H仅为fail-closed诊断路径的原始分数，不是可部署候选性能。

### 7.1D37逐场景联合结果

|场景|before-old|after-old|seen-new|H|forgetting|旧→新侵入|不可达class-fold|OOF可行折|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|81.67%|71.67%|66.00%|67.56%|10.00pp|11|11|0/5|
|`leo_low_elev_weak`|76.67%|66.67%|62.00%|63.31%|10.00pp|12|12|0/5|
|`leo_rain_weak`|88.33%|75.00%|48.00%|58.10%|13.33pp|10|17|0/5|

三个margin臂逐场景结果完全一致，因为全部折都在应用offset前因同一硬门fail closed。所有15个OOF失败原因均为`empty OOF feasible interval: true new class does not strictly beat other new classes`；不是简单把公共offset调大就能解除的旧/新group偏置。

### 7.2逐类结果与TX边界

旧类TX集合为`{14-10,14-7,20-15,20-19,6-15,8-20}`；target-new的nested first5集合为`{1-16,1-18,1-8,10-10,11-19}`。artifact按匿名class handle报告，当前不使用未审计的handle→TX推断替代正式映射。

|匿名旧类handle前缀|TX|强B3 before|D37 before|D37 after|
|---|---|---:|---:|---:|
|`1f33`|20-15|96.67%|96.67%|96.67%|
|`33bb`|8-20|90.00%|90.00%|90.00%|
|`75aa`|14-10|73.33%|73.33%|63.33%|
|`8b02`|14-7|86.67%|86.67%|63.33%|
|`a53c`|6-15|83.33%|83.33%|56.67%|
|`f8df`|20-19|63.33%|63.33%|56.67%|

|匿名新类handle前缀|seen-new代理准确率|
|---|---:|
|`09f8`|13.33%|
|`1c2a`|90.00%|
|`b8fb`|76.67%|
|`d3af`|80.00%|
|`f608`|33.33%|

### 7.3量化、资源与证据闭环

- 每个D37臂的内部Fisher/B3源头决策违规为0；但runner实际传入的是`fit_b3_fisher_closed_form`产生的D33-FAST/Fisher旧头，不是表中87.78%的强`B3_SINGLE_IQ_DIAG_FFTRF`。因此D37保持的是82.22%的弱旧头，selection逐scene×fold×class对强B3比较后正确判为非劣门失败。
- full-K10三场景state均为7620bytes；0 trainable parameter、0epoch、0 optimizer step；post-backbone query MAC为6624；fit约96–104ms，head mean约0.067–0.086ms，p95约0.077–0.126ms；无dense query graph，资源门全部通过。
- 两级residual-int8量化均值误差约`0.91e-6–1.01e-6`、最大误差约`7.07e-6–7.60e-6`；相对单级int8残差改善`99.598%–99.608%`。量化内部fit-support决策等价均为true，说明量化本身不是本轮主因。
- `training_log.jsonl`为105/105唯一key，7候选各15行、3场景各35行、每折21行。所有structured numeric值均finite；全部行`query_opened=false`、`formal_metric_claim_allowed=false`、`performance_claim_allowed=false`。D37训练trace共165项：inner crossfit120、deploy refit30、OOF calibration15，全部finite。
- support审计确认每个physical sample仅一个固定LEO弱观测，三场景physical ID两两不交；无clean/source/role/quota/global assignment访问。receipt中的五个artifact哈希与文件逐项一致。

|artifact|SHA256|
|---|---|
|`training_log.jsonl`|`03d17e0b386790a5420ff0ce9dc73214279bd9b159aa5778788f98ca974f6c36`|
|`selection.json`|`c6283f586a4c61f110e5bd384e12c2ce7b558d80a8936b79fb0a2db6bd39c8b7`|
|`resource_audit.json`|`2910f8d34438866da8e6eee9ffc26bf63ff5d1b1887c66d196b62a7399028c56`|
|`geometry_audit.json`|`b23df49595aa35c9d1adafdfd4c3502e2d608c529c733829a351eb79227a8e78`|
|`support_audit.json`|`6c6da40c53ba3b597c8df01ed24c740b4c79b3aa3bcd15624d55fd821cd0360f`|
|`RECEIPT.json`|`ea5d79396037698d002ec97cc31968a36d856beae14faef6631bbe55f806f1f2`|

### 7.4最终解释与下一轮约束

D37技术执行完成，但性能门为负：`selected_positive_route=false`，三个D37臂均不满足强B3非劣、outer零侵入、全新类可达和joint floor门，状态为`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。不得打开query，不得扩展K1/K5/K20或确认矩阵，也不得把receipt的`DEVELOPMENT_SUPPORT_ONLY_COMPLETE`误写为性能成功。

根因有两层。第一，现有D37接线量化并保留了弱Fisher旧头，而不是当前最强B3比较器；下一轮必须先取得并锁定强B3最终旧头的合法预测身份。第二，即使忽略该接线差异，15/15折的新类真实类在应用公共offset前已经输给其他新类；共享new-group offset只能移动新类整体，无法修复new-new排序。下一机制必须提升类无关的新类判别几何或分类器，并同时保留强B3旧头；禁止继续扫offset/margin。

K1限制仍未闭合：每类仅一个物理support时，不能让同一物理样本同时作为prototype输入和自身OOF held行。未出现合法预锁定的统一K1规则前，任何D37/D38路线都不能进入K1确认行。完整goal仍为active，远未满足5 receivers×至少5 seeds×3场景×K×new-count正式确认矩阵。
