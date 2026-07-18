# D39全类angular-radius标准化实验报告

## 1.实验身份与当前状态

- 实验ID：`d39_angular_radius_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景；只使用已验证D18固定received-IQ support，query保持sealed。
- 目标：在不改变D38-B训练轨迹、int8原型身份和30step预算的前提下，用同一类无关公式校准全部old/new类的角距离尺度，同时改善旧→新侵入和new-new通用floor。

本报告预注册D39机制和停止门。设计、实现、单测、资源通过或support screen启动均不是性能成功。

## 2.D38证据与可证伪假设

D38真实screen完成90/90行和1200条finite trace，五项artifact SHA与receipt一致，query未打开。D38-B注册前old=87.22%，接近exact strong B3的87.78%；注册后old=0.56%、seen-new=78.67%、H=0.99%，179/180个held旧样本被新类击败。new-held的true-new对top-old margin全部为正，均值6.645、最小1.204；new-new仍有32/150条错序。int8与matched FP32在outer-held的argmax变化为0。

这些结果否定了“旧头或量化导致坍塌”的解释。D38的新类方向在多数新类上有效，但原始cosine score没有表示类内离散度，new权重对旧样本过度自信。D39检验如下单一假设：用全部注册类共享的收缩radius公式把cosine改写为angular Gaussian score，可同时恢复old/new尺度可比性并改变new-new排序。

共享new-group bias或temperature不进入矩阵。对任意正温度和共享偏置，new类内部argmax不变，因此它无法直接修复32/150条new-new错序；将bias与radius同时加入会破坏单机制归因。

## 3.锁定机制

### 3.1D38-B基座

D39完全复用D38-B：288D特征为`normalize([norm(z160);4*norm([FFT96;RF32])])`；Stage2-B用old support执行20步full-batch AdamW，量化旧头；Stage2-C冻结共享metric与decoded int8旧头，用全部old+new support执行10步new-weight-only class-balanced CE、worst-class surrogate和centroid anchor，最后独立量化新头并append。

D39不改变D38 loss、学习率、optimizer、epoch、step或权重轨迹。它只在量化权重后增加0步闭式radius状态和新的统一scorer。

### 3.2角距离与旧域prior

对量化并按存储inverse norm解码后的类权重`w_c`，以及D38共享metric变换后的单位特征`h(x_i)`，定义：

\[
\theta_{ic}=\arccos\{\operatorname{clip}[h(x_i)^\top w_c,-1,1]\}.
\]

Stage2-B结束后，只用old support及注册前int8旧头计算每个旧类的二阶角离散度：

\[
m_{2,c}=\frac{1}{K}\sum_{i:y_i=c}\theta_{ic}^{2},\qquad
r_0=\max\left(\sqrt{\frac{1}{|Y_{old}|}\sum_{c\in Y_{old}}m_{2,c}},0.05\right).
\]

`r0`在新类注册前量化为FP16并冻结。new support不得重估`r0`。

### 3.3统一收缩radius

全部类使用同一公式，固定`nu=4`：

\[
r_c^2=\frac{\nu r_0^2+(K-1)m_{2,c}}{\nu+K-1}.
\]

K1时`K-1=0`，所有类严格退化为`r_c=r_0`；不构造self-OOF，也不以单样本零残差制造虚假高置信度。K≥2时，类内离散度按相同公式进入radius，未使用handle、难类名单或类专属超参数。

旧类radius在Stage2-B后量化为FP16并冻结。Stage2-C结束后，只用final int8新权重和对应new support计算新类`m2/radius`并append；不得重算旧radius、`r0`、旧权重或`log_diag`。

### 3.4统一angular Gaussian score

推理只使用量化后FP16 radius，固定`epsilon=0.001`：

\[
S_c(x)=-\frac{1}{2}\left[\frac{\theta_c(x)}{r_c+\epsilon}\right]^2-\log(r_c+\epsilon).
\]

所有注册类进入同一argmax。每个query独立评分，不构造dense query graph，不读取query角色、真实batch类数、quota或其他query；`-log(r_c+epsilon)`惩罚宽类，避免通过放大radius无条件吞噬其他类。

## 4.状态生命周期与精度ablation

正式`D39AngularRadiusState`仅保存：

- D38正式int8 base state：FP32共享`log_diag`、old/new两级int8 code、FP16 block scale/inverse norm、类注册表；
- `radius_fp16[C]`；
- `r0_fp16`和D39 schema元数据。

注册后旧D38 int8 prefix、旧`radius_fp16`和`r0_fp16`必须逐bit不变。正式state不保存FP32 target prototype或FP32 radius回退。

D39 FP32 ablation复用与正式路线完全相同的FP16 radius和`r0`，只把D38 base weight identity替换为同轨迹FP32权重。这样outer-held argmax差异只反映prototype精度，不混入radius重估差异。FP32 ablation不可晋级。

## 5.最小development矩阵

固定6候选×3场景×5fold=`90`行，每折8-shot fit、2-shot held：

|候选|角色|
|---|---|
|identity-only single-qKNN|回退与遗忘基线|
|ProtoNet CDA|独立matched基线；保留equivalence audit|
|exact legacy strong B3 FP32|最强合法target-support-only比较器|
|D38-B residual-int8|179/180侵入的结构性负对照|
|D39 angular-radius int8|唯一promotable路线|
|D39 angular-radius FP32|matched精度ablation|

direct ADV3B02仍作为相同old-held行的0-support旁路锚，不面对未注册新类，不进入90行候选数。

## 6.严格晋级门

D39 int8只有全部满足以下条件才可锁定development query：

1. D39与D38-B的注册前预测逐row完全一致；每个scene×fold×old-class不弱于exact strong B3。
2. 旧→新侵入不高于strong B3的33/180，且严格少于D38-B的179/180。
3. after-old、forgetting和全部旧类结果逐row逐类不弱于strong B3。
4. seen-new总体不低于D38-B的78.67%；new-new错序严格少于32/150。
5. 最低新类准确率和最低new-new margin均改善；选择器只计算全部新类通用最小值，不指定handle。
6. H与joint floor逐matched row不弱于strong B3，15fold聚合值严格提高。
7. D39 int8相对D39 FP32的outer-held argmax变化为0。
8. 全部radius/`r0`有限、正值、FP16可表示；旧base prefix、旧radius prefix和`r0`逐bit不变。
9. `old_radius_new_support_row_count=0`、`held_radius_fit_row_count=0`、`query_rows_used=0`；所有类共享`nu=4`与`epsilon=0.001`。
10. 正式资源≤80k trainable params、≤30epoch、≤50step、≤256KB；需单独报告radius状态字节、radius拟合标量操作、每query的`acos/log`标量操作、平均/P95时延和峰值显存。

任一关键门失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，回退identity，不打开query、不进入K1/K5/K20、125 screen或确认矩阵，也不在本轮增加bias、offset或第二种radius公式。

## 7.实现与运行计划

|项目|计划|
|---|---|
|core|新增`code/cvsrffi/stage2_d39_angular_radius.py`；D38仅增加公开cosine/temperature接缝，不复制私有decode/transform|
|Runner|在`run_d25_support_only_concat.py`增加`d39_v1`、精确90行矩阵、pairwise诊断、selector、full-K10和artifact闭环|
|测试|公式golden、K1退化、old radius append-only、formal state无FP32、row-local、标签置换、int8/FP32、资源和90行integration|
|Git|本地`ssr-gpu`验证后只提交D39授权文件和共享Runner最小差异|
|N607|只有本地K10 support-held硬门通过后才preflight、最小SCP并运行；D39负结果不在N607重复|

当前goal保持active。D39 development screen已完成且为负，不是独立确认；完整目标仍要求5receivers×至少5seeds×3scenes×K1/5/10/20×new2/5/10/20及全部性能、floor和资源门。

## 8.本地实现、审查与版本前验证

### 8.1实现范围

本地实现没有增加第二个机制或超参数扫描。D39核心复用D38-B的20+10步轨迹，通过D38公开的`before_stage2c_hook`在20条Stage2-B trace闭合后、Stage2-C开始前实际物化old`m2/r0/radius`；新类radius仍在final int8新头生成后append。formal state强制`base_state.arm=B`，并保存D38 residual-int8 base、FP16 radius、FP16`r0`和schema元数据。

Runner增加`d39_v1`固定六候选、candidate lock v17、90行矩阵、真实radius来源token SHA、跨候选held physical-token SHA、显式D39-int8/FP32预测与radius/r0/trace匹配、严格selector、selected-only full-K10状态/资源审计和D39专属artifact schema。只有D39-int8可晋级；任一门失败回退identity。

### 8.2独立审查闭环

第一次独立只读审查发现4项问题：old radius实际物化晚于Stage2-C、selector未消费显式D39-FP32候选、formal state未强制B-arm、矩阵未验证matched held物理身份。四项均在实现层修正，没有修改预注册公式或降低晋级门。第二次独立只读复审结果为`阻断=0`、`中等问题=0`，并确认full-K10 gate未使用同support拟合性能替代outer-held资格。

### 8.3验证命令与结果

```powershell
python -m py_compile code\cvsrffi\stage2_d38_strong_b3_quantized.py code\cvsrffi\stage2_d39_angular_radius.py code\scripts\run_d25_support_only_concat.py
python -m pytest -q tests\test_stage2_d38_strong_b3_quantized.py tests\test_stage2_d39_angular_radius.py tests\test_run_d39_angular_radius_integration.py
python -m pytest -q tests\test_run_d38_strong_b3_quantized_integration.py tests\test_run_d37_b3_preserving_int8_integration.py
git diff --check -- code/cvsrffi/stage2_d38_strong_b3_quantized.py code/scripts/run_d25_support_only_concat.py tests/test_stage2_d38_strong_b3_quantized.py
```

结果为D39相关53/53通过，D38/D37共享Runner回归17/17通过，总计70/70；`py_compile`与`git diff --check`通过。测试覆盖K1/5/10/20、new2/5/10/20、公式golden、old lifecycle、B-arm拒绝、row-local推理、真实radius来源、90行候选身份、9类selector反例、3类full gate反例和selected-only full-K10。

|文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_d38_strong_b3_quantized.py`|`89ca681356e13de62414bde7681280c5e63b4267e027f6df3ee2a762775309bd`|
|`code/cvsrffi/stage2_d39_angular_radius.py`|`78018594de21ebdcb75822d4d14164ab4bbd4e41b231fdb71c0155854dbcd86c`|
|`code/scripts/run_d25_support_only_concat.py`|`51f08dc7e7ac95dcd3dd8813c4da54147a19ee2854dfcc2ae6758f523af68e22`|
|`tests/test_stage2_d38_strong_b3_quantized.py`|`de141ea3e899182904f5eee58cee8db81c78007f8e1d5f10cd11e8f38fe1d958`|
|`tests/test_stage2_d39_angular_radius.py`|`5b1dc9f98d4c5cd5a122b30d9f2d52a9a229845ebccdabf8f9075e9b3c6e1559`|
|`tests/test_run_d39_angular_radius_integration.py`|`4bc117736b84bf77cea25ecad28e53c2543054da411182f0b64009210b48244b`|

`E:\type10-7`根目录不是Git仓库；本报告的Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，根目录同名报告仅作非版本化运行镜像。真实K10/new5 support-only 90行已完成并否决D39；query保持sealed，N607未访问。

## 9.真实development screen执行闭环

### 9.1运行身份与命令

- 执行状态：90/90候选行完成，6候选×3场景×5fold；每fold以8-shot拟合、2-shot outer-held诊断。
- receiver/seed/K/new：`20-1`/`713101`/K10/new5。
- 执行工作树：`E:\type10-7\code\snapshots\d39wt`，detached commit`6098a3f0`。
- 环境：本地`ssr-gpu`、`device=auto`；wall time`19.629s`。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d39_angular_radius_20260718\local_support_screen_d39_v1`；stdout为同级`local_support_screen_d39_v1.stdout.log`。
- 选择结果：`selected_candidate_id=Z0_SUPPORT_ONLY`、`selected_positive_route=false`、`query_opened=false`、`formal_metric_claim_allowed=false`、`performance_claim_allowed=false`。

```powershell
python E:\type10-7\code\snapshots\d39wt\code\scripts\run_d25_support_only_concat.py `
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
  --output E:\type10-7\automation_reports\CV-SincNet\d39_angular_radius_20260718\local_support_screen_d39_v1 `
  --device auto --mode development_select_unverified_component --candidate-set d39_v1
```

### 9.2完整候选结果

下表每项指标都来自同一候选的15个matched outer-held行；没有把不同候选或不同fold的边际极值拼成一行。`joint floor`为每行`min(old-class floor,new-class floor)`再对15行求均值；旧→新侵入分母为180。

|候选|机制/角色|before-old|after-old|seen-new|H|遗忘|joint floor|旧→新侵入|new-new错序|最终判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`Z0_SUPPORT_ONLY`|identity回退|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|64/180|未记录|回退基线|
|`D39-PROTOnet-CDA-ZID160`|ProtoNet CDA|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|64/180|未记录|matched基线|
|`B3_SINGLE_IQ_DIAG_FFTRF`|exact strong B3|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|33/180|未记录|最强合法比较器|
|`D39-D38-B-RESIDUAL-INT8-NEGATIVE`|D38-B结构负对照|87.22%|0.56%|78.67%|0.99%|86.67pp|0.00%|179/180|32/150|诊断性负|
|`D39-ANGULAR-RADIUS-INT8`|D39正式候选|87.22%|2.78%|78.67%|4.94%|84.44pp|0.00%|174/180|32/150|不晋级|
|`D39-ANGULAR-RADIUS-FP32-MATCHED`|精度ablation|87.22%|2.78%|78.67%|4.94%|84.44pp|0.00%|174/180|32/150|与int8同预测，不可晋级|

D39相对D38-B只把after-old从0.56%提高到2.78%，救回5/180个旧样本；它仍比strong B3低72.78pp，旧→新侵入仍为174/180。seen-new和32/150条new-new错序完全不变，说明radius没有修复新类方向排序。正式D39与显式matched FP32在15个fold的outer prediction、radius、`r0`、训练trace均0 mismatch，内部int8/FP32 argmax变化也为0，因此负结果不是int8近似造成。

### 9.3逐场景同候选结果

|候选|场景|before-old|after-old|seen-new|H|遗忘|旧→新侵入|
|---|---|---:|---:|---:|---:|---:|---:|
|strong B3|`leo_clear_weak`|—|75.00%|82.00%|77.51%|—|12/60|
|D39 int8|`leo_clear_weak`|88.33%|5.00%|82.00%|8.76%|83.33pp|57/60|
|strong B3|`leo_low_elev_weak`|—|75.00%|70.00%|71.71%|—|11/60|
|D39 int8|`leo_low_elev_weak`|83.33%|0.00%|72.00%|0.00%|83.33pp|60/60|
|strong B3|`leo_rain_weak`|—|76.67%|66.00%|70.85%|—|10/60|
|D39 int8|`leo_rain_weak`|90.00%|3.33%|82.00%|6.06%|86.67pp|57/60|

低仰角场景最能否证机制：D39注册前old为83.33%，注册后降到0%，60/60旧样本全部被新类侵入。雨衰场景即使seen-new提高到82%，也以86.67pp遗忘为代价，不具部署意义。

### 9.4逐类结果

|类handle前缀|角色|strong B3|D39 int8|差值|
|---|---|---:|---:|---:|
|`1f33`|old after|93.33%|0.00%|-93.33pp|
|`33bb`|old after|90.00%|16.67%|-73.33pp|
|`75aa`|old after|73.33%|0.00%|-73.33pp|
|`8b02`|old after|73.33%|0.00%|-73.33pp|
|`a53c`|old after|60.00%|0.00%|-60.00pp|
|`f8df`|old after|63.33%|0.00%|-63.33pp|
|`09f8`|seen-new|40.00%|30.00%|-10.00pp|
|`1c2a`|seen-new|86.67%|93.33%|+6.67pp|
|`b8fb`|seen-new|76.67%|86.67%|+10.00pp|
|`d3af`|seen-new|86.67%|93.33%|+6.67pp|
|`f608`|seen-new|73.33%|90.00%|+16.67pp|

D39只保留`33bb`的少量旧类正确率，其余5个旧类几乎全灭；同时最弱新类`09f8`仍只有30%。这不是一个可由均值掩盖的局部退化，而是旧/新联合决策尺度整体失败。

### 9.5pairwise margin与训练轨迹

|new类前缀|错序/30|new-new margin均值|最小值|true-new对top-old均值|最小值|
|---|---:|---:|---:|---:|---:|
|`09f8`|21|−0.0466|−0.2490|0.3939|−0.0568|
|`1c2a`|2|0.1070|−0.0143|0.5496|0.4004|
|`b8fb`|4|0.1627|−0.2920|0.4811|0.1861|
|`d3af`|2|0.2555|−0.3684|0.5610|0.1514|
|`f608`|3|0.1627|−0.1347|0.3979|0.2704|

D39的new-new margin总体均值为0.1283、最小值−0.3684；D38-B对应均值1.4168、最小值−4.1373。radius只是压缩分数差，未改变32/150条错误排序；同时true-new对top-old margin从D38-B的全部为正且均值6.645，压缩到D39均值约0.477，并在`09f8`出现负值。

D38-B、D39-int8、D39-FP32各记录450条训练trace，共1350条，全部finite且轨迹完全一致：Stage2-B 300条、Stage2-C 150条；B阶段loss由1.031996降至0.102685、准确率由95.14%升至100%；C阶段loss由7.023763降至4.116024、准确率由41.21%升至42.80%。因此D39负结果可以归因于新scorer，而不是训练漂移。

## 10.晋级门、协议与资源审计

### 10.1门控结果

|门|结果|证据|
|---|---|---|
|注册前逐样本与D38-B完全同轨|失败|14/15 prediction SHA相同；`leo_low_elev_weak/fold0`哈希不同，但总体及逐类精度相同，训练trace相同|
|逐类before-old不弱于strong B3|失败|selector逐row逐类门未通过|
|侵入≤33/180且<179/180|失败|174/180，仅满足严格小于D38-B|
|after-old/forgetting逐类不弱于strong B3|失败|after-old 2.78%对75.56%，遗忘84.44pp对12.22pp|
|seen-new≥78.67%且new-new错序<32/150|失败|seen-new相等，但错序仍32/150|
|最低新类与最低margin改善|失败|`09f8`仅30%，最低margin−0.3684|
|H/joint floor逐row不弱且聚合严格提高|失败|H 4.94%对73.35%，joint floor 0对23.33%|
|D39 int8/FP32一致|通过|15/15 outer prediction、radius、`r0`、trace匹配；argmax变化0|
|old base/radius/`r0` append-only|通过|全部prefix逐bit不变，materialization发生在Stage2-B第20步后、Stage2-C前|
|来源、协议、资源|通过|held radius fit=0、old radius new-support=0、query row=0；资源低于硬上限|

注册前唯一SHA差异不能以相同准确率豁免：预注册门要求逐样本预测完全一致，故该门按失败记录。它不影响D39的最终否决，因为其余多个性能硬门以巨大幅度失败。

### 10.2协议审计

`support_audit.json`确认：`query_opened=false`、query row/label均0；clean/source/cache/replay/sample-level feature均不可达；role Oracle、true batch class count、class quota和global assignment均为false；每个物理support只有1个LEO观测、support view=1；held ranks和held physical-token SHA在6候选间matched；三场景support/query物理身份保持隔离。`source_closure_unchanged_after_support=true`。

### 10.3资源审计

|资源项|D39 int8实测/静态审计|上限|结果|
|---|---:|---:|---|
|trainable parameters|2,016|80,000|通过|
|optimizer steps/epochs|30/30|50/30|通过|
|adaptation MAC|13,340,160|—|记录|
|base MAC/query|6,624|—|记录|
|angular标量操作/query|11×`acos`+11×`log`=22|—|记录|
|persistent state|8,637B|262,144B|通过|
|radius state|24B|—|记录|
|wrapper metadata|91B|—|记录|
|CUDA peak|22,886,912B|—|记录|

`r0`跨15fold为1.07324–1.12207，均值1.10020；radius范围0.78369–1.14160，全部finite且为正。由于D39未通过outer-held门，selected-only full-K10 refit没有执行，故本轮不虚构full-K10平均/P95延迟；这项保持为后续获选候选的实测义务。

## 11.artifact与版本闭环

|artifact|SHA256|核验|
|---|---|---|
|`geometry_audit.json`|`2d894526b9002e8d9886b3232bbaa9526529235a0da6e7bdd6cd25b205ee8168`|与receipt一致|
|`resource_audit.json`|`c8dc1f4b4f08406596a49f24f7de4760edf8bd01b90da04520f74a1aa3a38277`|与receipt一致|
|`selection.json`|`e54d943767f07706d3e97d8dd1cae9347f349011a291877082d590aebcc6439b`|与receipt一致|
|`support_audit.json`|`7864dfa45ff43be35cc57b7fa071ea93f9d819b4d8a703927b3a40f85bc290aa`|与receipt一致|
|`training_log.jsonl`|`4d391938e2874ce301bb6bb96dbc29e702393e5e2a8e30a8e0254b8c9494595d`|与receipt一致；90行|
|`RECEIPT.json`|`e8b23069b777c53aee6c6b73f104b64c9997ba5cde51cbdd0ecd85e5f6fd2998`|自哈希|
|stdout|`a58e96dbad1c799546d95973e95984bfeee00b9d1e857d57019b0f9b329c0e6`|receipt JSON输出|

candidate lock SHA为`f35291ce2348f25275962cc8181915e0273d2fb2763feeaf6a1448c14846744f`。receipt记录的D38/D39/Runner源码SHA分别为`a5b222d7…e4cc`、`b9fb11a3…e71f`、`aa83dafd…edbc`；它们是commit`6098a3f0`在Windows isolated worktree中的CRLF字节哈希。Git blob/LF内容哈希仍为第8.3节的`89ca6813…09bd`、`78018594…d86c`、`51f08dc7…8e22`，内容版本相同，receipt对实际执行字节闭包。

实现提交为`6098a3f0 feat(stage2): implement D39 angular radius screen`。本完成报告和追踪表另行提交，只stage这两个授权文件；工作树中其余用户/其他任务改动不纳入。

## 12.最终解释与下一步

D39完成了技术实现、90行真实support-held筛选、全量日志解析和artifact闭环，但没有完成项目goal，也没有取得可晋级性能。统一angular Gaussian半径并不能让D38-B的old/new分数可比：它轻微降低旧→新侵入，却保留new-new错误顺序，并把新旧margin压缩到接近决策边界。D39因此记为`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，回退identity，不打开query、不访问N607、不扩展到K1/K5/K20或正式确认矩阵。

下一轮不得继续微调同一radius公式或叠加共享bias。D37–D39已构成三轮完整探索；在启动D40前必须先执行记录化回顾，重新核对active objective、`项目.md`、conversation index及D37/D38/D39完整报告和日志，再选择同时保护旧类与修复新类排序的单一新机制。

## 13.D37–D39三轮强制技术回顾与D40锁定

### 13.1回顾动作与证据面

本回顾在任何D40代码或实验启动前完成。已重新读取active objective和2026-07-18版`项目.md`，刷新`E:\type10-7\conversation_index`至1005条项目记录，并检索`D37/D38/D39/strong B3/int8/angular radius/旧类侵入/new-new`及`D36/ridge/low-rank`历史。随后复核D37、D38、D39三份权威报告、`selection.json`、`support_audit.json`、`RECEIPT.json`及完整`training_log.jsonl`。

全日志表面为D37 105/105行、D38 90/90行、D39 90/90行；全部structured numeric值finite，unique candidate×scenario×fold键完整。D38解析1200条optimizer trace，D39解析1350条同轨迹trace；D37的公共offset三臂均完整解析。三轮全部`query_opened=false`、`formal_metric_claim_allowed=false`、clean/source/role/quota/global assignment不可达，且`source_closure_unchanged_after_support=true`。

历史普查`analysis/stage2_method_goal_history_census_20260718.md`仍把D36写为“仅设计/core单测”，但live D36报告已补齐本地105行真实负结果：D36-A/B/C的H仅57.80%/57.91%/56.82%，共同弱于strong B3。故本回顾以当前D36报告和artifact为准，不从旧普查恢复“ridge仍未实测”的过时判断。

### 13.2三轮同row结果与共同根因

|轮次/候选|单一机制|before-old|after-old|seen-new|H|遗忘|joint floor|侵入/排序证据|结论|
|---|---|---:|---:|---:|---:|---:|---:|---|---|
|D37-A/B/C|弱Fisher旧头＋两级int8＋公共new-group offset/margin|82.22%|71.11%|58.67%|62.99%|11.11pp|0%|33/180旧→新；40/75 new class-fold不可达；15/15 OOF区间为空|公共offset不能修new-new顺序，三臂等价失败|
|D38-B|D38强Stage2-B几何＋all-support/new-weight-only CE10|87.22%|0.56%|78.67%|0.99%|86.67pp|0%|179/180旧→新；32/150 new-new错序|新方向增强但outer-held旧域被吞噬|
|D39 int8|D38-B同轨迹＋全类angular-radius score|87.22%|2.78%|78.67%|4.94%|84.44pp|0%|174/180旧→新；32/150错序不变|只救回5个旧样本，尺度修复失败|
|exact strong B3|合法FP32 matched比较器|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|33/180旧→新|当前最强同row，但远未达目标|

共同根因不是量化、资源或optimizer数值不稳定：D38/D39的int8与matched FP32在outer-held均0个argmax差异，量化误差约`1e-6`；全部loss有限并下降，状态和资源均通过。真正问题是support拟合目标与未见物理样本泛化错位：

1. 公共offset、temperature或new-group bias只能移动整组分数，不能改变`09f8`对`1c2a`等new-new排序。
2. D38的old support负证据足以让fit面收敛，却未约束outer-held旧样本；新类方向对未见旧物理样本系统性过强。
3. D39从同support估计radius，只压缩margin并奖励较窄方向，没有改变32/150条错误排序，也没有恢复old/new可比分数面。
4. 当前可保留的正信号是D38注册前87.22%的强Stage2-B几何、两级residual-int8近FP32身份、append-only生命周期和极轻资源；需要替换的是注册方向构造，而非继续调精度或标尺。

当前strong B3相对K10/new5目标仍有明确差距：after-old差16.44pp，最低旧类60.00%相对88%门差28.00pp，seen-new差19.33pp；最低新类仅40.00%，且当前只是receiver`20-1`、seed`713101`、K10/new5 development support-held屏，不是独立确认。

### 13.3拒绝继续重复的路线

下一轮明确拒绝：公共new-group offset/margin、共享bias、类专属bias/threshold/gate、D39 radius公式微调、D38式CE10的step/lr扫描、从弱Fisher或support prototype重建旧头、ground int8直接强融合、hard visibility/release/winner门、继续增加prototype/top-k/buffer，以及用FP32或更多int8层级补救。D36的连续ridge margin、D21-M6 support-fold低秩delta和更早support ridge均已有负证据；它们不作为D40首选。support-held硬门未过前不得打开query、访问N607或扩展K1/K5/K20。

### 13.4D40单一机制：append-only hard-negative barycentric residualization

D40锁定为`D40-HNBR`。它复用D38的20步Stage2-B共享`log_diag`和原始类方向，不使用D38 Stage2-C的CE10，不加bias、radius、gate或新超参数。设当前注册阶段的单位基础方向为`b_c`，固定temperature`T=18`直接继承D38 scorer。对类`c`的其他同时可见基础方向计算：

\[
a_{cd}=\frac{\exp(T b_c^\top b_d)}{\sum_{j\ne c}\exp(T b_c^\top b_j)},\quad d\ne c,
\]

\[
n_c=\operatorname{normalize}\left(\sum_{d\ne c}a_{cd}b_d\right),\quad
\rho_c=\max(0,b_c^\top n_c),
\]

\[
w_c=\operatorname{normalize}(b_c-\rho_c n_c).
\]

该操作把权重自动集中到与当前类最相似的难负方向，只移除正投影；没有class handle、难类名单、可调投影系数或类别专属分支。

生命周期按append-only收紧：

- Stage2-B在D38 old基础方向集合上同步计算HNBR，量化为target-old两级residual-int8状态；这是D40的注册前旧头，直接检验old-old混淆能否改善。
- Stage2-C冻结上述target-old int8字节和密封ground int8组件。每个new基础方向由同一D38变换空间的合法new support中心产生；所有new类同时以“冻结old最终方向＋其余new基础方向”为难负集合计算HNBR，再两级int8量化并append。不得按注册顺序串行更新，也不得重写old prefix。
- query只用统一`18<h(x),w_c>`对全部注册类逐样本argmax，无old/new role branch。机制对保持enrollment partition的任意类标签重命名严格等变；old/new阶段差异来自合法state provenance，不来自query真值。
- K1每类基础方向直接由唯一合法物理support形成，随后执行相同闭式HNBR；0梯度、无伪LOO、不借用K5/K10统计。new2和近零残差必须分别有分母与fail-closed测试。

该机制同时改变Stage2-B旧类方向和Stage2-C新类方向，但保留D38最有价值的共享metric、int8精度与轻量状态。它仍只是一个可证伪候选：若难负投影不能外推到outer-held，不允许在本轮扫描投影系数或叠加第二机制。

### 13.5D40最小矩阵与停止门

固定6候选×3场景×5fold=`90`行，每fold仍为8-shot fit、2-shot matched physical held：identity-only single-qKNN、ProtoNet CDA、exact strong B3 FP32、D38-B int8结构负对照、D40-HNBR int8、D40-HNBR FP32。direct ADV3B02继续只作相同old-held的0-support锚，不计候选行。D40 int8是唯一可晋级路线。

D40只有全部满足下列条件才可进入full-K10或N607：

1. before-old总体、每场景×fold×old-class均不弱于exact strong B3，15fold聚合严格提高。
2. after-old、全部旧类floor不弱于strong B3；forgetting逐matched row不高于strong B3。
3. old→new侵入不高于33/180且相对strong B3严格减少。
4. seen-new逐row不弱于strong B3；new-new错序严格少于32/150，最低新类准确率与最低pairwise margin严格提高。
5. 每个matched row的H和joint floor不弱于strong B3，15fold聚合均严格提高。
6. D40 int8/FP32 outer-held argmax差异为0；target-old/new预测均使用正式int8状态，ground及old prefix生命周期逐bit闭合。
7. 0个Stage2-C optimizer step，trainable parameter不超过D38 Stage2-B的2016，epoch/总step≤20/20，状态≤256KB；无dense query graph或query-dependent batch optimization。

任一关键门失败即`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`：回退identity，不调投影系数、不叠加bias/radius/gate，不打开query、不访问N607、不扩K或确认矩阵。若通过，本轮也只取得development正信号；完整goal仍必须完成5receivers×至少5seeds×3scenes×K1/5/10/20×new2/5/10/20确认矩阵及全部性能、floor和资源门。
