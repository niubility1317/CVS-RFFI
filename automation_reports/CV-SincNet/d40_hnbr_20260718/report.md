# D40 append-only HNBR实验报告

## 1.实验身份与状态

- 实验ID：`d40_hnbr_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景；复用D18固定received-IQ support，query保持sealed。
- 目标：在保留D38强Stage2-B共享metric和正式int8生命周期的同时，用0步、无可调系数的难负重心方向残差化同时改善old-old、new-new和old-new竞争。

本设计来自D37–D39三轮强制回顾。设计、实现、单测、资源通过或support screen启动都不是性能成功。

## 2.直接证据与单一假设

当前同row最强合法比较器是exact strong B3：before-old87.78%、after-old75.56%、seen-new72.67%、H73.35%、forgetting12.22pp、joint floor23.33%、旧→新侵入33/180；最低旧类60%、最低新类40%。

D38注册前old=87.22%，但D38-A/B在新方向加入后分别发生180/180和179/180旧→新侵入；D38-B虽把seen-new提高到78.67%，仍有32/150条new-new错序。D39只改变angular-radius标尺，侵入仍174/180且32/150错序完全不变。D38/D39的int8与FP32 outer-held argmax差异均为0。

D40只检验一个假设：类别方向与最相似竞争方向的正投影是当前共同混淆源；在每个注册阶段用同一无参数球面投影移除该分量，可提升Stage2-B旧类分离，并让Stage2-C新类方向同时避开冻结旧类和其他新类，而不引入group bias、radius或hard gate。

## 3.锁定数学机制

设D38共享`log_diag`变换后的单位基础方向为`b_c`，固定`T=18`继承D38 scorer。对当前类`c`的难负集合`ℕ_c`，定义：

\[
a_{cd}=\frac{\exp(Tb_c^\top b_d)}{\sum_{j\in\mathcal N_c}\exp(Tb_c^\top b_j)},\quad d\in\mathcal N_c,
\]

\[
n_c=\operatorname{normalize}\left(\sum_{d\in\mathcal N_c}a_{cd}b_d\right),\quad
\rho_c=\max(0,b_c^\top n_c),
\]

\[
w_c=\operatorname{normalize}(b_c-\rho_c n_c).
\]

softmax实现必须先减行最大值；所有norm必须finite且大于`1e-12`，否则fail closed。不增加投影强度、margin、temperature或shrinkage候选。

### 3.1Stage2-B

D40先执行与D38相同的20步full-batch old adaptation，得到old基础方向。随后所有old类同步以其余old基础方向为`ℕ_c`执行HNBR，量化为两级residual-int8 target-old状态。该状态用于注册前评分，也构成Stage2-C冻结old prefix。

### 3.2Stage2-C

每个new基础方向是在同一D38变换空间中对本类合法support求单位均值。所有new类同时计算HNBR；对new类`c`，难负集合由冻结target-old最终方向和其余new基础方向组成。不得把已残差化的某个new方向用于下一个new类，避免注册顺序依赖。

new HNBR方向独立量化后append。target-old code/scale/inverse norm、`log_diag`及密封ground int8组件逐bit不变。正式state不保存FP32 target方向、optimizer或回退副本。

### 3.3推理与置换边界

所有query使用统一`18<h(x),w_c>`面对全部注册类独立argmax；无role分支、batch统计、quota、global reassignment或dense query graph。机制对保持enrollment partition的任意类标签置换严格等变；old/new阶段差异来自合法state provenance，不来自query truth。

K1直接以每类唯一物理support形成new基础方向并执行同一闭式HNBR；0梯度、无伪LOO、不借用其他K统计。new2、只有2个old类、近零重心及近零残差均须单独测试。

## 4.固定候选与development矩阵

|候选|角色|
|---|---|
|identity-only single-qKNN|回退/遗忘基线|
|ProtoNet CDA|独立matched基线|
|exact strong B3 FP32|最强合法比较器|
|D38-B residual-int8|方向/尺度灾难负对照|
|D40-HNBR int8|唯一可晋级路线|
|D40-HNBR FP32|matched精度ablation，不可晋级|

固定6×3场景×5个outer physical folds=`90`行，每fold8-shot fit、2-shot held。direct ADV3B02只作相同old-held的0-support锚，不进入90行。全部候选必须共享相同held ranks、physical-token SHA和源数据闭包。

## 5.严格晋级门

D40 int8只有全部满足才可进入full-K10或N607：

1. before-old总体、每scene×fold×old-class不弱于exact strong B3，聚合严格提高。
2. after-old与每旧类floor不弱于strong B3；forgetting逐row不高于strong B3。
3. old→new侵入≤33/180且相对strong B3严格减少。
4. seen-new逐row不弱于strong B3；new-new错序<32/150；最低新类准确率和最低pairwise margin严格提高。
5. 每个matched row的H和joint floor不弱于strong B3，15fold聚合均严格提高。
6. int8/FP32 outer-held argmax差异为0；old prefix、ground int8和source closure闭合。
7. 0个Stage2-C optimizer step；总epoch/step恰好20/20，trainable parameters≤2016，state≤256KB，HNBR support MAC为finite且严格大于0，无dense query graph或query-dependent batch optimization。

任一关键门失败即`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`：回退identity，不扫描投影系数，不叠加bias/radius/gate，不打开query、不访问N607、不扩K或确认矩阵。

## 6.实现与本地验证

|面|锁定范围|
|---|---|
|D38公开接缝|只新增readonly feature transform、state weight decode、int8/FP32 compile和append接口；不让D40调用私有函数|
|D40 core|HNBR公式、同步old/new构造、append-only state、pairwise/geometry/resource audit|
|Runner|`d40_v1`六候选、90行、同physical匹配、selector、selected-only full-K10和五项artifact哈希|
|测试|公式golden、标签置换、同步性、K1/5/10/20、new2/5/10/20、近零fail-close、old prefix、int8/FP32、90行与selector/resource反例|
|Git/N607|本地`ssr-gpu`验证并提交；只有真实K10 outer-held全门通过才preflight/SCP/N607|

### 6.1实现文件

|文件|用途|
|---|---|
|`code/cvsrffi/stage2_d38_strong_b3_quantized.py`|新增readonly transform/decode/compile/append公开接缝|
|`code/cvsrffi/stage2_d40_hnbr.py`|D40-HNBR核心、int8/FP32状态、pairwise及资源/几何审计|
|`code/scripts/run_d25_support_only_concat.py`|`d40_v1`六候选90行Runner、strict selector、selected-only full-K10及artifact closure|
|`tests/test_stage2_d38_strong_b3_quantized.py`|D38公开接缝与append prefix回归|
|`tests/test_stage2_d40_hnbr.py`|公式、同步性、int8 decoded old negative、K/new-count、状态及协议测试|
|`tests/test_run_d40_hnbr_integration.py`|真实fold接线、exact strong B3 pairwise golden、90行/physical closure、selector与full-K10反例|

### 6.2验证证据

- Conda环境：`ssr-gpu`；本地CPU验证，无N607访问。
- `python -m py_compile`覆盖上述6个实现/测试文件：通过。
- `python -m pytest -q`覆盖D38/D39/D40 core及D36–D40 Runner integration：`124 passed`。
- `git diff --check`覆盖上述6个文件：通过；仅有Git的LF→CRLF提示，无whitespace error。
- 实现提交：`bc6c3539 feat(stage2): implement D40 HNBR screen`。
- D40 core不引用D38私有符号；new HNBR实际第二次调用的冻结old negative与`before_int8`解码方向逐元素相等，人为替换matched FP32 old ablation不改变new参考方向。
- 独立只读审查未发现blocker。审查要求的两项medium已修复：资源门改为固定20/20步且HNBR MAC>0；exact strong B3 pairwise补全函数新增独立held行、类别索引、physical token、margin与侵入golden测试。

当前只完成技术实现与本地测试闭环，尚未产生真实90行performance artifact，不能把`124 passed`解释为性能晋级。

根目录`E:\type10-7`不是Git仓库；Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，根目录同名报告仅作非版本化运行镜像。当前goal保持active，D40 development screen不能替代完整确认矩阵。

## 7.真实development screen执行闭环

### 7.1运行身份与命令

- 执行状态：6候选×3场景×5fold=`90/90`行完成；每fold为8-shot fit、2-shot outer-held。
- development cell：receiver`20-1`、seed`713101`、K10/new5、old6→new5密封类handle；unknown/query未打开。
- 执行工作树：`E:\type10-7\code\snapshots\d40wt`，detached commit`65be30dd873dfc7588124c30541efe831035d347`。
- 运行时恢复文件SHA256：`somph_predictor_bundle.py=49a05c6f…def48`、`somph_runtime_trust.py=4b1dee1d…c1f9fc`、`stage2_predictor_bundle.py=bb27beaa…944aa9`，与D39已验证运行面逐bit相同。
- 环境：本地`ssr-gpu`、`device=auto`；receipt记录wall time`16.726s`。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d40_hnbr_20260718\local_support_screen_d40_v1`。
- 选择结果：`selected_candidate_id=Z0_SUPPORT_ONLY`、`selected_positive_route=false`、`query_opened=false`、`formal_metric_claim_allowed=false`、`performance_claim_allowed=false`。
- 执行客户端在14秒返回窗口超时，stdout捕获未完成；只读landed probe随后确认无残留Python进程、6个artifact均存在且RECEIPT闭合。故不重试、不覆盖，也不伪造stdout文件。

```powershell
python E:\type10-7\code\snapshots\d40wt\code\scripts\run_d25_support_only_concat.py `
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
  --output E:\type10-7\automation_reports\CV-SincNet\d40_hnbr_20260718\local_support_screen_d40_v1 `
  --device auto --mode development_select_unverified_component --candidate-set d40_v1
```

## 8.完整结果

### 8.1六候选同row汇总

全部指标来自receiver`20-1`、seed`713101`、K10、old6→new5、3场景×5fold的同候选15行。unknown/query列均为sealed；未把不同候选的边际极值拼接为一行。

|候选|机制/类别|before-old|after-old|seen-new|H|遗忘|joint floor|旧→新侵入|new-new错序|实际新→旧|unknown/query|回退/晋级判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|`Z0_SUPPORT_ONLY`|identity回退|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|64/180|未记录|未记录|sealed|最终回退|
|`D40-PROTOnet-CDA-ZID160`|ProtoNet CDA|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|64/180|未记录|未记录|sealed|matched基线|
|`B3_SINGLE_IQ_DIAG_FFTRF`|exact strong B3|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|33/180|25/150|31/150|sealed|最强合法比较器；D41复核修正actual new→old统计|
|`D40-D38-B-RESIDUAL-INT8-NEGATIVE`|D38-B结构负对照|87.22%|0.56%|78.67%|0.99%|86.67pp|0.00%|179/180|32/150|0/150|sealed|诊断性负|
|`D40-HNBR-INT8`|D40正式候选|85.56%|85.00%|15.33%|25.16%|0.56pp|0.00%|2/180|33/150|127/150|sealed|不晋级|
|`D40-HNBR-FP32-MATCHED`|精度ablation|85.56%|85.00%|15.33%|25.16%|0.56pp|0.00%|2/180|33/150|127/150|sealed|与int8同预测，不可晋级|

D40确实修复了D38-B的旧类灾难：after-old从0.56%升至85.00%，旧→新侵入从179/180降至2/180，遗忘从86.67pp降至0.56pp。但它把错误方向完全翻转：150个新类held中127个由旧类取得最高分，seen-new仅15.33%，5个新类中2类为0%。因此它不是联合成功，而是从“new压倒old”切换成“old压倒new”。

> D41复核勘误：D40原始`training_log.jsonl`的150条exact B3`pairwise_support_diagnostics.new_old_margin`逐条重算为`31/150`，最低margin为`-4.7121`。此前表中的`22/150`是旧解析误记；D40-HNBR的`127/150`不变。D41预注册的`<22/150`硬门保持冻结，因而比修正后的B3比较器更严格。

### 8.2逐场景strong B3与D40同row结果

|候选|场景|before-old|after-old|seen-new|H|遗忘|joint floor|旧→新侵入|new-new错序|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|strong B3|`leo_clear_weak`|88.33%|75.00%|82.00%|77.51%|13.33pp|30.00%|12/60|6/50|
|D40 int8|`leo_clear_weak`|85.00%|85.00%|16.00%|26.53%|0.00pp|0.00%|0/60|12/50|
|strong B3|`leo_low_elev_weak`|85.00%|75.00%|70.00%|71.71%|10.00pp|20.00%|11/60|10/50|
|D40 int8|`leo_low_elev_weak`|81.67%|81.67%|18.00%|28.81%|0.00pp|0.00%|0/60|12/50|
|strong B3|`leo_rain_weak`|90.00%|76.67%|66.00%|70.85%|13.33pp|20.00%|10/60|9/50|
|D40 int8|`leo_rain_weak`|90.00%|88.33%|12.00%|20.13%|1.67pp|0.00%|2/60|9/50|

三个场景均出现相同方向：旧类保持与侵入显著改善，而seen-new分别下降66pp、52pp和54pp。该一致性排除“仅某一LEO场景异常”的解释。

### 8.3逐类结果

|类handle前缀|角色|strong B3|D40 int8|差值|
|---|---|---:|---:|---:|
|`1f33`|old before|96.67%|96.67%|0.00pp|
|`33bb`|old before|90.00%|90.00%|0.00pp|
|`75aa`|old before|90.00%|86.67%|−3.33pp|
|`8b02`|old before|83.33%|86.67%|+3.33pp|
|`a53c`|old before|86.67%|86.67%|0.00pp|
|`f8df`|old before|80.00%|66.67%|−13.33pp|
|`1f33`|old after|93.33%|96.67%|+3.33pp|
|`33bb`|old after|90.00%|90.00%|0.00pp|
|`75aa`|old after|73.33%|86.67%|+13.33pp|
|`8b02`|old after|73.33%|86.67%|+13.33pp|
|`a53c`|old after|60.00%|86.67%|+26.67pp|
|`f8df`|old after|63.33%|63.33%|0.00pp|
|`09f8`|seen-new|40.00%|0.00%|−40.00pp|
|`1c2a`|seen-new|86.67%|0.00%|−86.67pp|
|`b8fb`|seen-new|76.67%|13.33%|−63.33pp|
|`d3af`|seen-new|86.67%|50.00%|−36.67pp|
|`f608`|seen-new|73.33%|13.33%|−60.00pp|

聚合逐类old-after都不弱于strong B3，但strict门仍在90个scene×fold×old-class单元中出现3个退化；before-old有6/90个单元退化。新类则不是局部floor问题，5类全部下降，15/15个scene×fold的seen-new都弱于strong B3。

### 8.4pairwise与完整训练轨迹

|new类前缀|D40 new-new错序/30|D40实际新→旧/30|正确/30|new-new margin均值|最小值|true-new对top-old均值|最小值|
|---|---:|---:|---:|---:|---:|---:|---:|
|`09f8`|22|30|0|−1.6801|−6.4204|−5.4715|−9.4535|
|`1c2a`|2|30|0|2.6564|−0.5240|−3.4147|−5.5764|
|`b8fb`|2|26|4|4.0894|−3.5158|−1.3138|−6.9090|
|`d3af`|3|15|15|4.7014|−3.2194|−0.5772|−7.4092|
|`f608`|4|26|4|3.0425|−1.0625|−1.8222|−4.2722|

D40不是主要败在new-new排序：`1c2a`等类的new-new margin仍为正，但true-new对top-old margin在所有5类的均值均为负。D40共127/150条新held由旧类取最高分，且与FP32逐样本完全一致。这把失败定位到old/new跨阶段方向可比性，而不是新类内部排序或int8量化。

完整90行全部structured numeric值finite。D38-B记录450条轨迹；D40-int8和D40-FP32各记录300条轨迹，共1050条。D40每fold恰好20条`stage2b_fullbatch_old_adaptation`，平均loss从epoch1的1.031996降至epoch20的0.102685，support accuracy从95.14%升至100%；Stage2-C恰好0个optimizer step。int8/FP32的15/15 outer prediction、15/15 before prediction及15/15训练trace均匹配，outer argmax变化总数为0。

## 9.晋级门、协议与资源审计

### 9.1门控结果

|门|结果|证据|
|---|---|---|
|before-old逐类不弱且聚合严格提高|失败|85.56%对87.78%；6/90个逐类单元退化|
|after-old/forgetting逐row不弱|失败|聚合85.00%高于75.56%、15/15 forgetting不劣，但3/90个old-class单元退化|
|旧→新侵入≤33且严格减少|通过|2/180对33/180|
|seen-new/new-new/floor/margin|失败|15/15行seen-new退化；33/150不小于32；最低新类0%；最低margin−6.4204|
|H/joint floor逐row不弱且聚合严格提高|失败|15/15行H退化；7/15行joint floor退化；聚合H25.16%对73.35%，joint floor0对23.33%|
|D40 int8/FP32一致|通过|15/15 outer、before、trace匹配；argmax变化0|
|old prefix/lifecycle/source/query|通过|prefix逐bit不变；new只读decoded int8 old negatives；held fit=0、query row=0|
|资源|通过|固定20/20步、Stage2-C=0、2016参数、8611B state、HNBR MAC=84960|

任一核心门失败即否决；本轮多个独立性能门同时失败，故selected-only full-K10不执行，N607不访问，K/new-count/receiver/seed确认矩阵不扩展。

### 9.2协议与物理身份

6个候选的15个scene×fold键全部一致；每个键的held physical-token count/SHA完全matched。`support_audit.json`确认每个物理support仅有1个LEO弱观测、support view=1、derived support row=0、额外physical/overlay=0。query row/label均0；clean/source/cache/replay/sample-level feature、role Oracle、true batch class count、class quota和global assignment均不可达。`source_closure_unchanged_after_support=true`。

证据边界：component provenance仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，`formal_phase2_eligible=false`，RECEIPT明确禁止formal/performance claim。这与development support-only诊断模式一致；本报告不得把该负结果表述为正式Phase2性能结论。

### 9.3资源

|资源项|D40 int8|硬门|结果|
|---|---:|---:|---|
|trainable parameters|2,016|≤2,016|通过|
|adaptation epochs/optimizer steps|20/20|=20/20|通过|
|Stage2-C optimizer steps|0|=0|通过|
|D38 Stage2-B adaptation MAC|4,976,640|记录|通过|
|HNBR support MAC|84,960|>0并计入|通过|
|total adaptation MAC|5,061,600|记录|通过|
|MAC/query|6,624|记录|通过|
|persistent state|8,611B|≤262,144B|通过|
|CUDA peak|22,886,912B|记录|通过|

由于outer selector已经回退identity，`resource_audit.json`对D40三个场景均明确记录`full_k10_refit_performed=false`和`reason=not_globally_selected_by_outer_6x3x5_matrix`。不虚构full-K10延迟或状态结果。

## 10.artifact与版本闭环

|artifact|SHA256|核验|
|---|---|---|
|`training_log.jsonl`|`00ee05e25a5f02dc71ccd114deb9970940404aff5a18fd38e9722131ea8bb499`|与receipt一致；90行|
|`support_audit.json`|`223d05a2367ee70be39352baa19a0e6970fd7563a308d0b9f1334d57bed9475d`|与receipt一致|
|`selection.json`|`08cba8283a3f2715bc6094fb1e0bb00ed50fb1da43dc89ed91238bcb794543fb`|与receipt一致|
|`resource_audit.json`|`e0592ecf1b6e2aa02389e65c501b7333a3581b05bf0777349a35158817790c06`|与receipt一致|
|`geometry_audit.json`|`192146795b3508b26d194eb631c54983ca44a3dcd00a0acf5b23111d321715bc`|与receipt一致|
|`RECEIPT.json`|`dc48bc494b3c9602ac3ecdd64d48a3ccbc9185f33913c614132f15752b25cdbf`|自哈希|

candidate lock SHA为`978080d00a2c575e474478a78c2382edfb662c1a5b08d63d854148c5a23dfeb9`。receipt记录实际执行字节的D38 core、D40 core和Runner SHA分别为`1781cc83…ec9`、`6ea54a70…560`、`38bd12dc…717`，与isolated worktree文件实测一致。实现/本地验证提交为`bc6c3539`/`65be30dd`；本完成报告只stage D40报告与追踪文件，不纳入主工作树其他未归属改动。

## 11.最终解释与下一轮约束

D40完成了技术实现、真实90行support-held执行、全量日志解析和artifact闭环，但没有完成项目goal，也没有取得可晋级性能。它证明“append-only、分阶段HNBR”能保护旧类，却会制造相反的跨阶段偏置：Stage2-B old方向只相对其他old残差化，而Stage2-C new方向相对冻结old＋其他new残差化；两组方向经历不同竞争集合，却共用同一cosine temperature。结果是旧类对新held系统性占优。

因此D40记为`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不扫描投影系数，不叠加bias/radius/gate，不打开query、不访问N607、不扩展确认矩阵。下一轮如果继续检验HNBR，只能改变这一条已被证据指向的结构不对称：在Stage2-C用同一全注册类竞争集合同步重编译target-old/new，而不是保持old prefix；密封Phase1 ground int8知识仍须逐bit不变。该建议只是D41待预注册假设，不是D40结果的一部分。
