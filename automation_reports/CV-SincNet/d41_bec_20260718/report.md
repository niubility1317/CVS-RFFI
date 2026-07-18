# D41 block-erasure consistency实验报告

## 1.实验身份与状态

- 实验ID：`d41_bec_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景；复用同一D18固定received-IQ support，query保持sealed。
- 目标：用同一固定received IQ上的确定性block-erasure一致性训练，同时改善Stage2-B outer-held旧类泛化与Stage2-C old/new联合方向可比性；不引入第二个LEO观测、group bias、radius、HNBR或query gate。

D41是D37–D39三轮回顾后的第二个探索轮。设计、实现、单测、support-view一致性下降或资源通过都不是性能成功。

## 2.D40直接证据与路线选择

D40-HNBR int8在真实90行中得到before-old85.56%、after-old85.00%、seen-new15.33%、H25.16%、遗忘0.56pp、joint floor0、旧→新侵入2/180；150个新类held中127个由旧类取最高分，new-new错序33/150。exact strong B3为87.78%/75.56%/72.67%/73.35%/12.22pp/23.33%、侵入33/180、new-new错序25/150。D41对原始pairwise逐条复核后，exact B3实际新→旧为31/150；此前D40报告中的22/150是旧解析误记，已在D40报告中勘误。D41预注册`<22/150`硬门保持冻结。

D40不仅有跨阶段new→old翻转，也损伤Stage2-B：before-old比strong B3低2.22pp，90个scene×fold×old-class单元中6个退化；new-new错序也比strong B3多8条。因此只在Stage2-C把old/new联合HNBR重编译，无法修复已知before门，并可能把2/180旧→新侵入重新拉高。joint-HNBR协议上可行，但作为D41首选被拒绝，只保留为因果诊断备选。

D41只检验一个新机制：固定block-erasure consistency（BEC）能否让共享metric和target head不过度依赖某一表征块，从而同时改善old outer泛化、new注册、new-new下尾和old/new尺度。若view一致性下降而真实physical-held指标不升，BEC即被否决。

## 3.锁定表征、数学view与损失

### 3.1基础表征不变

完整view继续使用D40/D38的288维B3几何：`normalized z_id160 + 4×joint-normalized(FFT96,RF32)`再整体L2归一化。D41不改变基础block能量或query特征口径，避免把表示修改与BEC混成两个机制。

定义三个固定索引块：`z=[0,160)`、`fft=[160,256)`、`rf=[256,288)`。对同一个已封存received IQ的完整288维行`x`，构造：

\[
v_{full}=x,\quad
v_{-z}=\operatorname{norm}(x\odot m_{-z}),\quad
v_{-fft}=\operatorname{norm}(x\odot m_{-fft}),\quad
v_{-rf}=\operatorname{norm}(x\odot m_{-rf}).
\]

每个mask只把对应索引块置0，再对剩余向量L2归一化；norm必须finite且大于`1e-12`，否则fail closed。四个view都只读同一固定received IQ，不增加K、不产生额外physical sample或LEO overlay。query只计算`v_full`。

### 3.2共享metric、head与BEC目标

\[
h_\theta(v)=\operatorname{norm}\left(v\odot\exp(\operatorname{clamp}(\ell))\right),
\qquad
s_c(v)=18\,h_\theta(v)^\top\operatorname{norm}(w_c),
\]

其中`\ell`为288维`log_diag`，bounds完全继承D38；`w_c`为当前target注册类方向。令`p_v=softmax(s(v))`，对每个masked view定义：

\[
JS(p_{full},p_v)=\frac12KL(p_{full}\Vert m)+\frac12KL(p_v\Vert m),
\quad m=\frac12(p_{full}+p_v).
\]

实现使用`log_softmax`与`logaddexp-log(2)`计算，不添加可调epsilon。每个view的CE先按类内求均值，再对当前全部注册类等权平均，记为`CE_macro`。锁定总损失：

\[
L_{BEC}=\frac14\sum_{v\in\{full,-z,-fft,-rf\}}CE_{macro}(s(v),y)
+\frac13\sum_{v\in\{-z,-fft,-rf\}}JS(p_{full},p_v).
\]

两个主项系数均固定为1；不加入D38 feature noise、prototype anchor、worst-class surrogate、new anchor、margin、bias、radius、HNBR或mask概率，不扫描temperature、loss权重、step或其他超参数。类宏平均、JS和初始化都对标签置换同式，不是query quota。

## 4.锁定Stage2-B/C生命周期

### 4.1Stage2-B

- 只读取old support。
- `log_diag=0`；target-old权重以完整view类别centroid初始化。
- 使用D38锁定AdamW：learning rate`0.01`、weight decay`0.002`、gradient clip`5.0`、相同`log_diag` bounds，full-batch恰好20步。
- 唯一可训练状态为`log_diag＋全部target-old weights`；当前old6峰值参数`(1+6)×288=2016`。
- 第20步后生成独立不可变before artifact：target-old两级residual-int8＋FP32`log_diag`。它只用于注册前held评分，不得被Stage2-C覆盖或追写。

### 4.2Stage2-C

- 同一原子fit中继续使用Stage2-B最终FP32`log_diag`和old weights；禁止重置。若未来拆成跨调用实现，只能用相同support＋seed确定性重放B，不能保存隐藏FP32 deployment sidecar。
- 在Stage2-B metric下用完整view new support centroid初始化new weights。
- 使用old＋new全部合法support及四个view，对`log_diag＋全部target-old＋全部target-new weights`执行同一`L_BEC`。
- 使用D38锁定SGD：learning rate`0.05`、momentum`0`、gradient clip`5.0`，full-batch恰好10步；不得冻结old或只训练new。
- 第10步后将全部target registry一次性重新编译为两级residual-int8。formal final state不保存FP32 target方向、optimizer或回退副本；matched FP32只作同参考方向精度ablation。
- target-old在Stage2-C允许更新，但Phase1 sealed ground int8组件的code/scale/hash在entry/exit必须逐bit相同。必须在artifact中区分ground old和target-old，不能把target registry重编译误写成ground更新。

K1直接以每类唯一物理support初始化centroid并使用相同四个数学view；无物理LOO、伪样本或其他K统计。K1/K5/K20只在D41方法完全锁定后用于独立确认，不参与本轮选参。

## 5.资源预锁

|资源|current old6/new5|最大old20/new20|硬门|
|---|---:|---:|---:|
|Stage2-B峰值参数|2,016|6,048|≤80,000|
|Stage2-C峰值参数|3,456|11,808|≤80,000|
|epoch/optimizer step|30/30|30/30|≤30/50|
|Stage2-C step|10|10|=10|
|persistent target state|预计<8KB|预计<26KB|≤256KB|
|query view|1个full view|1个full view|无dense query graph|

适配MAC必须计入四个view的transform/classification及三个JS项，不能沿用single-view估算。query MAC只包含full view shared metric、全部注册类cosine和argmax。CUDA peak、完整30步trace、平均/P95 head latency只在真实获选候选的selected-only full-K10中实测；outer未获选时不得虚构。

## 6.固定候选与development矩阵

|候选|角色|
|---|---|
|identity-only single-qKNN|回退/遗忘基线|
|ProtoNet CDA|独立matched基线|
|exact strong B3 FP32|当前最强合法比较器|
|D40-HNBR int8|已证伪旧类主导结构负对照|
|D41-BEC int8|唯一可晋级路线|
|D41-BEC FP32|matched精度ablation，不可晋级|

固定6×3场景×5个outer physical folds=`90`行，每fold8-shot fit、2-shot held。direct ADV3B02只作相同old-held的0-support锚，不进入90行。全部候选共享held ranks、physical-token SHA和source closure。D40负对照必须在D41 Runner中从同fold真实执行，不从旧报告拼接。

## 7.严格晋级门

D41 int8只有全部满足才可进入selected-only full-K10或N607：

1. before-old每scene×fold总体及每旧类不弱于exact strong B3，15fold聚合严格提高。
2. after-old每scene×fold总体及每旧类不弱于D40；forgetting逐row不高于D40。
3. seen-new每scene×fold总体及每新类不弱于exact strong B3，15fold聚合严格提高；最低新类准确率严格高于strong B3最低40%。
4. old→new实际侵入`<33/180`、new→old实际最高分错误`<22/150`、new-new pairwise错序`<25/150`，三项均严格优于exact strong B3；最低new-new和new-old margin都严格提高。
5. 每个matched row的H和joint floor不弱于exact strong B3，15fold两项聚合均严格提高。
6. D41 int8/FP32的before/final outer-held argmax差异为0；formal target state为int8-only，Phase1 ground int8 entry/exit hash一致。
7. 30/30步、Stage2-C=10、当前峰值参数=3456、state≤256KB；四view/BEC MAC为finite且严格大于single-view下界；query/source/clean/role/quota/global assignment闭合。

任一结构、协议或资源门失败即fail closed；任一性能门失败即`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，回退identity，不调整mask、loss、optimizer、step或temperature，不打开query、不访问N607、不扩K/receiver/seed/new-count确认矩阵。

## 8.实现与本地验证

|面|锁定范围|
|---|---|
|D41 core|四view、macro-CE＋JS、B20/C10联合state更新、before/final int8与matched FP32、ground/pairwise/resource audit|
|Runner|`d41_v1`六候选、90行、actual old→new/new→old、new-new pairwise、strict selector、selected-only full-K10与五项artifact SHA|
|测试|view golden、JS golden、B/C参数更新范围、before不可变、ground逐bit、K1/5/10/20、new2/5/10/20、标签置换、row-local query、int8/FP32、90行physical closure与逐门反例|
|Git/N607|本地`ssr-gpu`验证并提交；只有真实90行全部晋级门通过才preflight/SCP/N607|

### 8.1本地变更

|文件|用途|
|---|---|
|`code/cvsrffi/stage2_d41_bec.py`|D41四view、BEC损失、B20/C10生命周期、int8/FP32编译、pairwise与资源审计|
|`code/scripts/run_d25_support_only_concat.py`|`d41_v1`六候选90行Runner、strict selector、真实ground重哈希、selected-only full-K10与artifact闭包|
|`tests/test_stage2_d41_bec.py`|核心golden、生命周期、量化、K/new-count、置换和逐样本评分测试|
|`tests/test_run_d41_bec_integration.py`|Runner接线、90行物理身份、12个selector独立反例、真实ground篡改和full-K10门测试|

### 8.2验证证据

- 在`ssr-gpu`中对上述4个文件执行`python -m py_compile`，通过。
- 主代理串行执行D36–D41全部核心与Runner集成测试，共`187 passed`。
- 独立Runner复核执行D38–D41集成测试，共`64 passed`；补齐D38 source closure后再次执行，仍为`64 passed`。
- D41核心独立验证`21 passed`；D41 Runner独立验证`17 passed`，包含12个strict selector逐门反例和真实临时ground NPZ入口/出口篡改反例。
- `git diff --check`通过；仅有Git的LF→CRLF未来转换提示。pytest退出时Windows临时目录清理报告`WinError 5`，测试退出码为0，不属于项目失败。

独立审计最初发现D41 candidate source closure遗漏直接依赖的D38量化核心哈希。现已修复为D38、D40或D41任一候选存在时都封存`d38_strong_b3_quantized_core_sha256`，并由D41集成测试显式断言；support打开后的closure复核因此覆盖D38、D40、D41与Runner全部直接代码依赖。其余已核路径未发现P0/P2/P3问题。

本阶段只证明实现与证据闭包可执行，不构成性能晋级。下一步必须在隔离Git worktree中运行真实90行development matrix；只有全部严格门通过才允许N607。

## 9.真实本地90行执行

- 隔离worktree：`E:\type10-7\code\snapshots\d41wt`，Git`7fb47ad486437132c757f5aaf78fd75ff7ae32dc`。
- 实现提交：`91894484 feat(stage2): implement D41 BEC screen`；本地验证报告提交：`7fb47ad4 docs(stage2): record D41 local verification`。
- 运行时恢复文件SHA256：`somph_predictor_bundle.py=49a05c6f…def48`、`somph_runtime_trust.py=4b1dee1d…c1f9fc`、`stage2_predictor_bundle.py=bb27beaa…944aa9`，与D40已验证运行面逐bit相同。
- 环境：本地`ssr-gpu`、`device=auto`；receipt wall time`38.379s`。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d41_bec_20260718\local_support_screen_d41_v1`。
- 结果：90/90行完成，`selected_candidate_id=Z0_SUPPORT_ONLY`、`selected_positive_route=false`、`query_opened=false`、`formal_metric_claim_allowed=false`、`performance_claim_allowed=false`。

执行命令与D40完全复用同一D18 capsule、D22 ground int8组件和class binding，仅替换隔离worktree、输出目录与`candidate-set`：

```powershell
python E:\type10-7\code\snapshots\d41wt\code\scripts\run_d25_support_only_concat.py `
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
  --output E:\type10-7\automation_reports\CV-SincNet\d41_bec_20260718\local_support_screen_d41_v1 `
  --device auto --mode development_select_unverified_component --candidate-set d41_v1
```

## 10.完整同row结果

统一cell为receiver`20-1`、seed`713101`、old6/new5、K10 support pool；每个outer row用8-shot fit、2-shot physical-held，3场景×5fold。unknown指标不适用于本support-only闭集development screen，记为N/A；coverage/rollback/defer无该Runner字段，回退由selector统一执行。

|候选|机制/类别|before-old|after-old|seen-new|H|遗忘|joint floor|old→new|new→old|new-new|old类均值最低|new类均值最低|loss/adapter摘要|最终判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|`Z0_SUPPORT_ONLY`|identity回退|71.11%|48.33%|52.67%|48.97%|22.78pp|0|64/180|N/A|N/A|13.33%|3.33%|0步closed-form|合法回退，不可晋级|
|`D41-PROTOnet-CDA-ZID160`|ProtoNet matched基线|71.11%|48.33%|52.67%|48.97%|22.78pp|0|64/180|N/A|N/A|13.33%|3.33%|与identity等价|matched基线|
|`B3_SINGLE_IQ_DIAG_FFTRF`|exact strong B3|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|33/180|31/150|25/150|60.00%|40.00%|20步single-view B3|最强合法比较器|
|`D41-D40-HNBR-INT8-NEGATIVE`|D40负对照|85.56%|85.00%|15.33%|25.16%|0.56pp|0|2/180|127/150|33/150|63.33%|0|B20＋HNBR closed-form|旧类主导负结果|
|`D41-BEC-INT8`|D41唯一可晋级路线|86.11%|20.56%|78.67%|31.50%|65.56pp|0|142/180|2/150|32/150|0|36.67%|B20＋C10四view BEC|负结果，回退identity|
|`D41-BEC-FP32-MATCHED`|matched精度ablation|86.11%|20.56%|78.67%|31.50%|65.56pp|0|142/180|2/150|32/150|0|36.67%|与int8同参考方向|不可晋级；证明非量化问题|

D41 int8分场景：

|场景|before-old|after-old|seen-new|H|遗忘|joint floor|old→new|new→old|new-new|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|88.33%|26.67%|78.00%|39.70%|61.67pp|0|44/60|0/50|11/50|
|`leo_low_elev_weak`|80.00%|16.67%|76.00%|26.36%|63.33pp|0|50/60|1/50|12/50|
|`leo_rain_weak`|90.00%|18.33%|82.00%|28.45%|71.67pp|0|48/60|1/50|9/50|

D41类均值下尾：

|注册角色|类handle前缀|before-old|after-old/seen-new|
|---|---|---:|---:|
|old|`cls_1f33441efa14…`|96.67%|40.00%|
|old|`cls_33bbd16556c6…`|90.00%|70.00%|
|old|`cls_75aa6d506081…`|83.33%|0|
|old|`cls_8b02d99905a8…`|83.33%|0|
|old|`cls_a53ca1280d8f…`|86.67%|0|
|old|`cls_f8dfc2edcccc…`|76.67%|13.33%|
|new|`cls_09f800392544…`|N/A|36.67%|
|new|`cls_1c2ad8827bdb…`|N/A|90.00%|
|new|`cls_b8fbace568ad…`|N/A|90.00%|
|new|`cls_d3afb5d16e93…`|N/A|93.33%|
|new|`cls_f608a348579f…`|N/A|83.33%|

## 11.晋级门逐项审计

|门|结果|完整证据|
|---|---|---|
|before-old vs B3|FAIL|聚合86.11%<87.78%；4/15fold总体退化，4/15fold存在旧类退化|
|after-old/forgetting vs D40|FAIL|20.56%<85.00%、65.56pp>0.56pp；15/15fold总体、逐类和遗忘均失败|
|seen-new vs B3|FAIL|聚合78.67%虽高6.00pp，但6/15fold总体、7/15fold逐类失败；最低新类36.67%<40%|
|三类混淆|FAIL|new→old2/150通过，old→new142/180远超`<33`，new-new32/150超过`<25`|
|两类最低margin|PASS|new-new最低`-3.2692`高于B3`-4.7748`；new-old最低`-0.1574`高于B3`-4.7121`|
|H/joint floor|FAIL|H15/15fold低于B3；joint floor7/15fold低于B3且聚合0<23.33%|
|int8/FP32|PASS|before和final outer-held argmax变化均为0；两候选prediction SHA与30步trace逐row一致|
|view/lifecycle|PASS|1个physical support→4个确定性数学view；query full-only；B20/C10连续生命周期闭合|
|ground/source|PASS|15/15fold真实ground NPZ entry/exit SHA=`3c08c823…67d7`；old/new fit与held交集0，query/source/clean访问0|
|resource|PASS|3456参数、30epoch、30 optimizer steps、C=10、state8647B、BEC MAC42303360、query MAC6624、CUDA peak24304640B|

通过门为`margin/precision/view/lifecycle/ground/source/resource`；失败门为`before/after-old/new/confusion/joint`。任一失败已经足以否决，本轮没有selected-only full-K10 refit，resource/geometry artifact对D41三场景均明确记录`not_globally_selected_by_outer_6x3x5_matrix`。

## 12.完整训练日志与机制诊断

已解析`training_log.jsonl`全部90行、D41 int8全部15×30=`450`个阶段step，以及matched FP32的450个逐step复制；FP32 trace与int8逐row相同。无active/incomplete pass，所有row与step均完整落盘。

|阶段|step|平均loss首→末|平均macro-CE首→末|平均JS首→末|full-view support acc首→末|
|---|---:|---:|---:|---:|---:|
|Stage2-B old-only|20|1.1169→0.2499|1.1095→0.2207|0.00737→0.02913|94.44%→100.00%|
|Stage2-C all-registry|10|4.0498→1.3726|4.0293→1.3296|0.02043→0.04299|41.29%→58.33%|

两阶段总loss下降，但JS反而上升；Stage2-C结束时连support full-view也只有58.33%。因此D41没有实现预期的block一致性，CE下降主要来自把大量旧类方向推入新类决策区。外部physical-held表现与该内部症状一致：D40是旧类压新类（127/150 new→old），D41则翻转为新类压旧类（142/180 old→new）。这不是“稍欠调参”，而是联合C阶段在10步资源锁下缺乏能同时约束旧类保持和新类注册的结构。

量化不是原因：int8/FP32全部outer argmax相同；ground/source/view/resource均闭合。基础B阶段也未解决：before-old比B3低1.67pp。D41-BEC因此同时未修复Stage2-B泛化，也在Stage2-C造成严重遗忘。

## 13.结论与下一轮约束

技术结论为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D41只完成了实现、协议闭包与真实development矩阵，不构成可部署性能成功，不进入N607，不扩K/receiver/seed/new-count确认矩阵，也不触碰query。

下一轮不得继续调BEC的mask、loss权重、optimizer或step。D40与D41共同把可行结构边界收紧为：必须保留B3 before artifact和Stage2-C旧类判别尺度，同时允许new注册提升；只靠old/new共同CE会在两个方向之间翻转。D42应设计显式的旧类函数保持与新类局部注册解耦结构，并在同一90行矩阵中继续同时优化Stage2-B和Stage2-C。D42完成后将达到本轮三次探索计数，必须先做D40–D42记录式回顾再启动D43。

## 14.Artifact SHA256

|artifact|SHA256|
|---|---|
|`training_log.jsonl`|`ebe1ba33ac9cc826fb3becbd92d5c3a0a07a54f1008ef34528895948031a2bf1`|
|`support_audit.json`|`51e881b3c5545ed9579101034d19027b078d748da7db353d4cfe424de51e78fb`|
|`selection.json`|`4906e90650ea16170463aad7b478d0c5b5629dc6d690ef36e13fad41dbf72f3d`|
|`resource_audit.json`|`844396dc370c9e3eab5e43e9b53087ba3ed32e07ab2d3f2611af19ec07d45d5a`|
|`geometry_audit.json`|`230c1f20f4983855a81681c0bc5dcd280bb57d1f290734a389d4982b70c9b8a6`|
|`RECEIPT.json`|`d6af129a13d1932df9ae61b59ffe6077aec569a11c1f74c2eac1d3325e0869e8`|

receipt还封存candidate lock`5a5fb4cd…1a9ff`、D38 core`1781cc83…0ec9`、D40 core`6ea54a70…a560`、D41 core`80ab3e32…d7b9`与Runner`c28ce67d…fd4`；`source_closure_unchanged_after_support=true`。

根目录`E:\type10-7`不是Git仓库；Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，根目录同名报告仅作非版本化运行镜像。当前goal保持active，D41 development screen不能替代完整确认矩阵。
