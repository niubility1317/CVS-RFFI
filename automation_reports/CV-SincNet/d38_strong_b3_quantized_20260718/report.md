# D38 full-batch B3-geometry residual-int8实验报告

## 1.实验身份与当前状态

- 实验ID：`d38_strong_b3_quantized_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
- 目标：修复D37的两个直接失败源——弱Fisher旧头和new-new排序错误——在同一合法K10 development cell上检验一个正式资源上限内、target-old/new均为两级residual-int8、逐样本面对全部注册类的轻型Stage2-B/C路线。
- 主要比较：identity-only single-qKNN、ProtoNet CDA、exact legacy strong B3 FP32、D38-A int8、D38-B int8、D38-B FP32 matched ablation；direct ADV3B02另作相同old-held样本的0-support锚。

本报告是D38开始编码前的预注册设计。任何实现完成、单测通过或support screen执行完成都不能自动改写为性能成功。

## 2.D37证据驱动的失败定位

D37真实support-only screen为105/105行，query始终sealed，五项artifact哈希与receipt一致。其两级int8量化均值误差约`0.91e-6–1.01e-6`，内部源旧头决策违规为0，因此量化不是主因。

|问题|D37证据|D38响应|
|---|---|---|
|旧头来源错误|D37保留的是82.22%的D33-FAST/Fisher旧头，而exact legacy strong B3为87.78%|重新实现B3几何中的无bias旧域适应，并把exact legacy strong B3保留为独立matched比较器|
|new-new排序错误|3个D37臂共45/45折以同一原因fail closed；每臂15/15折均有真实新类输给其他新类|取消公共offset主机制；让每个新类权重在统一loss下独立移动|
|旧→新重叠|每臂outer held旧→新侵入33/180；不是offset或旧头量化漂移|Stage2-C loss使用全部old+new support，但梯度只更新新权重|
|类内方差|`09f8`跨场景new margin正确4/30、mean=-0.245；`f608`为10/30、mean=-0.152；full-support自包含均升至20/30|首轮先用CE10做直接判因；若outer-held下尾不改善，再进入类无关whitening/radius，不在本轮混入多原型|

D38必须新增support-only pairwise诊断：`scenario/outer_fold/physical_rank/true_new_handle/top_competing_new_handle/true_new_score/top_competing_new_score/new_new_margin/top_old_score/new_old_margin`。这些字段只来自合法support-held行，不读取query或truth sidecar。

## 3.exact legacy strong B3与D38的声明边界

exact legacy strong B3使用：

- 288D固定received-IQ表征；
- Stage2-B 20epoch、batch size32、AdamW和feature noise；K10 outer train有48个旧support，因此是40 optimizer steps；
- Stage2-C最多20个new-only optimizer steps；
- target-old/new权重为FP32。

D38把Stage2-B改为20个full-batch optimizer steps，再给Stage2-C固定10个full-batch steps；训练动力学已改变。故D38只能称为`full-batch B3-geometry`或`B3-initialized`路线，不能把D38注册前结果直接命名为exact strong B3。exact legacy strong B3必须在完全相同scene、fold和held physical IDs上独立计算。

## 4.预注册数学机制

### 4.1固定288D表征

同一固定received IQ只前向一次得到`z_id160`，FFT/RF均是该received IQ的数学视图：

```text
phi(x)=normalize([normalize(z160); 4*normalize([FFT96; RF32])])
```

FFT96与RF32先拼接后共同归一化，不改成两个独立归一化块，避免把表征变化混入优化机制。

### 4.2Stage2-B：20步full-batch旧域适应

共享正值对角度量：

```text
d=exp(clip(a,lower,upper))
h(x)=normalize(d*phi(x))
s_c(x)=18<h(x),normalize(w_c)>
```

锁定范围：z160与RF32的`a_j∈[-1.5,1.5]`，FFT96的`a_j∈[-log(1.5),log(1.5)]`。旧类权重从各类support均值初始化，使用无class bias的full-batch AdamW：20step、lr`0.01`、weight decay`0.002`、gradient clip`5.0`、feature noise std`0.01`、prototype anchor`0.05`。每步将`a`投影回合法区间。

```text
L_B=mean_old CE + 0.05*mean_c ||normalize(w_c)-mu_c_init||^2
```

Stage2-B结束后，先对旧权重按固定块`(160,96,32)`独立做两级residual-int8编译。Stage2-C看到并冻结的是实际量化旧头的decode值，而不是随后会被替换的FP32旧头。

### 4.3Stage2-C：centroid对照与10步new-only判别训练

每个新类统一初始化：

```text
u_j0=normalize(mean_{y=j}(d*phi(x)))
```

D38-A直接量化`u_j0`，作为0步centroid注册对照。D38-B冻结`d`和已量化旧头，只训练全部新类`u_j`，但loss同时读取合法old与new support：

```text
L_c=mean_{i:y_i=c} CE(all_registered_scores_i,y_i)
L_wc=0.25*(logsumexp(L_c/0.25)-log(C))
L_C=mean_c L_c + 0.20*L_wc + 0.01*mean_j ||normalize(u_j)-u_j0||^2
```

D38-B固定full-batch SGD 10step、lr`0.05`、momentum`0`、gradient clip`5.0`。旧support产生的梯度只能推开新权重，不能更新旧权重或共享度量。最终新权重再独立做两级residual-int8量化并append；旧int8 code、FP16 scale和inverse norm前缀逐bit不变。

D38-B FP32使用完全相同训练轨迹，仅在最终部署权重精度上保留FP32，作为matched量化ablation，不是可晋级路线。正式D38 state只保存共享FP32`log_diag`、old/new int8 code、FP16 block scales/inverse norm和类注册表；不保存FP32 target prototype、optimizer state或FP32回退副本。

## 5.候选、矩阵与选择规则

|候选|机制|角色|
|---|---|---|
|identity-only single-qKNN|现有Z0 support centroid基线|遗忘/资源基线|
|ProtoNet CDA|ADV3B02 z_id160 support均值、最近原型|强制matched基线；若与Z0在本cell数学等价，仍保存equivalence audit|
|exact legacy strong B3 FP32|原20epoch mini-batch旧头＋原20step新注册|同fold性能上界比较器，不可冒充int8正式路线|
|D38-A residual-int8|20步full-batch旧头＋0步new centroid|判定新类判别训练是否必要|
|D38-B residual-int8|A＋10步all-support/new-weight-only训练|唯一promotable主路线|
|D38-B FP32|与B同轨迹、最终权重FP32|matched量化ablation，不可晋级|

最小开发矩阵固定为receiver`20-1`、seed`713101`、K10、new5、3个LEO弱场景、5个outer rank-pair folds，共`6×3×5=90`行。每折8-shot fit、2-shot held；held physical ID不得进入metric、weight、量化scale、checkpoint或candidate选择。A/B只在15fold聚合后全局选择一次，禁止按场景、fold、类或handle路由。

direct ADV3B02只在相同old-held行报告0-support准确率和逐类值，不面对尚未注册的新类，也不参与A/B超参数选择。full-K10 refit只生成锁定候选的部署/资源审计，不得反向更改臂或超参数。

K1/K5/K20固定执行K10锁定配置：A始终20step；B始终20+10=30step。K1不构造self-OOF、不重新选A/B、不early-stop或rollback。K20不用于开发选参。

## 6.资源预核算

当前旧类数6、特征维288。训练阶段FP32权重是瞬态；部署状态按两级int8＋FP16 scale/inverse norm计算。

|new类数|A/B峰值trainable params|B epoch/steps|部署state约值|逐query head MAC|
|---:|---:|---:|---:|---:|
|2|2016|30/30|5856B|4896|
|5|2016|30/30|7620B|6624|
|10|2880|30/30|10560B|9504|
|20|5760|30/30|16440B|15264|

A为20epoch/20step。所有规模均远低于80k参数、30epoch、50optimizer steps和256KB硬上限；无dense query graph或query-dependent batch optimization。实现后仍须现场测量平均/P95时延、峰值显存、适配MAC、backbone/FFT前向次数和实际状态字节，预核算不能替代resource audit。

## 7.可观察晋级门与停止条件

D38-B只有全部满足以下条件才可锁定development query：

1. 注册前D38量化旧头在每个scene×fold×old-class上不弱于exact legacy strong B3；若full-batch20不能保持强旧域结果，立即否证D38旧头路线。
2. B相对A同时改善seen-new总体、最低新类、new-new margin/混淆和H；若10step结束仍有系统性new-new错序，停止调step/lr，下一轮改为类无关whitening/radius。
3. B的after-old、seen-new、H、forgetting、joint floor及全部逐类结果不弱于同row exact legacy strong B3和identity/ProtoNet中更强者。
4. outer held旧→新侵入率不高于exact legacy strong B3；旧prefix逐bit不变只是必要条件，不能替代held安全。
5. D38-B int8相对matched FP32在outer-held的argmax变化数为0；同时报告量化误差，不能只以误差小推断决策不变。
6. target-old/new实际预测均使用int8生命周期；资源、协议、query sealed和逐样本all-registered-class审计全部通过。

任一关键门失败即记`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不打开query、不进入K1/K5/K20、125 screen或确认矩阵。不得继续盲扫epoch、lr、margin或offset。

## 8.协议、版本和运行计划

- 协议：`protocol_schema=p2_min_v1`；直接复用匹配`VALIDATED_ONCE`的D18 cell，不因D38 method变化重验数据。
- 数据：每个physical sample只有一个固定`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`received IQ；support/query及场景physical ID不交。
- 权限：support-only fit/选择；query sealed；无clean/raw、source样本/feature/logit/cache、role Oracle、class quota或global reassignment。
- 代码隔离：新增`stage2_d38_strong_b3_quantized.py`；不编辑或暂存当前有未归属修改的`stage2_diag_cosine_exploration.py`；D38 core不调用run_d19/run_d25私有函数。
- 根目录`E:\type10-7`不是Git仓库。本报告镜像到根目录，Git权威副本位于本文件；开始D38设计时分支ahead origin 1605，其他大量修改/未跟踪文件均不属于D38，后续只暂存D38专属文件和共享runner最小差异。
- N607：尚未触碰。先在`ssr-gpu`完成core/runner窄验证并提交Git；若需N607，按AGENTS.md先做direct preflight、占用审计、本地报告和最小SCP，短连接结束后核验无残留SSH/TCP22。

## 9.实施与实验记录

|项目|当前值|
|---|---|
|已新增core|`code/cvsrffi/stage2_d38_strong_b3_quantized.py`：20步full-batch Stage2-B、A0/B10 Stage2-C、old/new两级residual-int8、matched FP32 ablation、逐样本scorer与pairwise诊断|
|已接线runner|`code/scripts/run_d25_support_only_concat.py --candidate-set d38_v1`：预开封cell锁、精确90行矩阵、matched selector、full-K10审计与五项artifact哈希闭环|
|已新增测试|`tests/test_stage2_d38_strong_b3_quantized.py`、`tests/test_run_d38_strong_b3_quantized_integration.py`|
|本地验证|`ssr-gpu`下核心、D38 integration及D37共享Runner回归33/33通过；CUDA:0合成烟测完成30步，峰值分配显存`17124352B`|
|独立审查|发现并修复wrong receiver/seed/new-count未fail closed和registry状态字节漏算；复审无P0–P2|
|Git commit|实现提交`c3a55b8b`；本次报告回填提交待完成|
|N607 sync/command/PID/GPU|未启动|
|预期artifact|`training_log.jsonl`、`selection.json`、`resource_audit.json`、`geometry_audit.json`、`support_audit.json`、`RECEIPT.json`、完整stdout|

实际验证命令：

```powershell
(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate ssr-gpu
python -m pytest -q tests\test_stage2_d38_strong_b3_quantized.py tests\test_run_d38_strong_b3_quantized_integration.py tests\test_run_d37_b3_preserving_int8_integration.py
```

实现严格遵循D38预注册机制；它不是exact legacy strong B3的复现。后者仍以原20epoch mini-batch旧头和原20step新注册作为独立FP32比较器。当前尚无真实90行性能结果，不能判断D38是否promotable。

当前goal保持active。D38 development support screen不等于独立确认，更不等于完成5receivers×至少5seeds×3scenes×K×new-count正式矩阵。

## 10.真实K10 support-only screen

### 10.1执行与证据闭合

- 运行位置：本地隔离worktree`E:\type10-7\code\snapshots\d38wt`，detached HEAD`0580a0f9`；D38实现提交`c3a55b8b`。
- 输入：D18已验证receiver`20-1`、seed`713101`、K10/new5 cell；只读复用固定received-IQ support，未重验数据、未打开query。
- 输出：[local_support_screen_d38_v1](E:/type10-7/automation_reports/CV-SincNet/d38_strong_b3_quantized_20260718/local_support_screen_d38_v1)；stdout为同级`local_support_screen_d38_v1.stdout.log`。
- 本地GPU：GPU0 RTX5070Ti；启动前`2253/16303MiB`、利用率3%；运行已结束，无常驻训练任务。
- 用时：17.9992s；90/90行，6候选×3场景×5fold；共解析1200条optimizer trace，全部finite，无Traceback/OOM/Killed/NaN/Infinity。
- source closure：D38 core SHA256=`257c4ed21002eadb71fadd58db5b3a55994df85a42bff551398f48aa1f0925bb`；Runner SHA256=`e029b9fe5590b0995d5593dd880fc55c634622cedf3b15cc86ecef409be391cd`。
- 为保持D18签名闭包，隔离worktree逐字节恢复3个已验证runtime文件：`somph_predictor_bundle.py=49a05c6f...def48`、`somph_runtime_trust.py=4b1dee1d...c1f9fc`、`stage2_predictor_bundle.py=bb27beaa...944aa9`。

|artifact|SHA256|receipt匹配|
|---|---|---|
|`training_log.jsonl`|`40cd5dc05976dd3fafcceb898cd2fa3f45a630ed61353a0ff2dbbf2a53b3addb`|是|
|`support_audit.json`|`e1b4db924689f42f303d0414afe806f32dd483b05a00f4ae2f43176d158e98c6`|是|
|`selection.json`|`dc9e9f2992d1f4f6ce9b741520e38a56b6096116e556c0f59bb0aef995c090dc`|是|
|`resource_audit.json`|`c2a25a9d94377b8628dad187d38451312a64357137259b16392798fdb38d1350`|是|
|`geometry_audit.json`|`52e1e4c766760a466ec81e6f591f6a8ba4ca3ee3735e44b3fe3da0bf92365985`|是|
|`RECEIPT.json`|`41e77429035bd17d8fe92a258e6783fc4183a69248ad36e434666d1327e68084`|自身receipt|

协议审计为：`query_opened=false`、query row/label均0、clean/source access均false、role/true batch count/quota/global assignment均false、每个物理support仅1个LEO观测、support view=1、三场景physical ID/received-IQ hash/overlay token两两0重叠。

实际命令使用与D37相同的D18/D22锁定输入，仅替换已提交D38 Runner、输出目录与`--candidate-set d38_v1`：

```powershell
python E:\type10-7\code\snapshots\d38wt\code\scripts\run_d25_support_only_concat.py `
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
  --output E:\type10-7\automation_reports\CV-SincNet\d38_strong_b3_quantized_20260718\local_support_screen_d38_v1 `
  --device auto --mode development_select_unverified_component --candidate-set d38_v1
```

### 10.2完整候选结果

下表每一数值均为同一候选15个matched outer rows的均值；`joint floor`也是同row floor的均值，不拼接不同row极值。旧→新侵入分母为15fold×每fold12个held旧样本=`180`。direct ADV3B02旧类0-support旁路锚为65.56%，不构成候选row。

|候选|机制/角色|before-old|after-old|seen-new|H|遗忘|joint floor|旧→新侵入|最终判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|Z0|identity-only single-qKNN|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|64/180|回退基线|
|ProtoNet CDA|独立等价row|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|64/180|与Z0逐row等价|
|exact legacy strong B3 FP32|20epoch mini-batch＋20step注册|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|33/180|最强合法比较器，仍非成功|
|D38-A int8|full-batch20＋centroid|87.22%|0.00%|76.00%|0.00%|87.22pp|0.00%|180/180|结构性负结果|
|D38-B int8|A＋new-only CE10|87.22%|0.56%|78.67%|0.99%|86.67pp|0.00%|179/180|全部关键门失败|
|D38-B FP32|B同轨迹精度ablation|87.22%|0.56%|78.67%|0.99%|86.67pp|0.00%|179/180|与int8决策相同，不可晋级|

|场景|方法|before-old|after-old|seen-new|H|旧→新侵入|new-new混淆|
|---|---|---:|---:|---:|---:|---:|---:|
|clear|strong B3|88.33%|75.00%|82.00%|77.51%|12/60|未单独保存pairwise|
|clear|D38-B int8|88.33%|0.00%|82.00%|0.00%|60/60|9/50|
|low-elev|strong B3|85.00%|75.00%|70.00%|71.71%|11/60|未单独保存pairwise|
|low-elev|D38-B int8|83.33%|0.00%|72.00%|0.00%|60/60|14/50|
|rain|strong B3|90.00%|76.67%|66.00%|70.85%|10/60|未单独保存pairwise|
|rain|D38-B int8|90.00%|1.67%|82.00%|2.98%|59/60|9/50|

### 10.3逐类floor与pairwise诊断

下表是15fold边际逐类均值，仅用于展示通用floor失败位置；主结论仍基于上面的同row联合结果。handle仅显示SHA前缀，不映射TX身份。

|旧类handle|D38-B before|D38-B after|strong B3 after|
|---|---:|---:|---:|
|`1f33`|96.67%|0.00%|93.33%|
|`33bb`|90.00%|3.33%|90.00%|
|`75aa`|90.00%|0.00%|73.33%|
|`8b02`|80.00%|0.00%|73.33%|
|`a53c`|86.67%|0.00%|60.00%|
|`f8df`|80.00%|0.00%|63.33%|

|新类handle|D38-B seen-new|strong B3 seen-new|D38-B new-new错序/30|D38-B平均new-new margin|
|---|---:|---:|---:|---:|
|`09f8`|30.00%|40.00%|21|−0.3673|
|`1c2a`|93.33%|86.67%|2|0.9202|
|`b8fb`|86.67%|76.67%|4|1.8939|
|`d3af`|93.33%|86.67%|2|2.8465|
|`f608`|90.00%|73.33%|3|1.7906|

D38-A共有36/150条new-new错序；D38-B降至32/150，但`09f8`反而由19/30增至21/30，其中14次top competitor为`1c2a`。这说明CE10对多数新类有正信号，却没有修复通用最低类；不得为`09f8`定制权重、阈值或分支。

### 10.4训练、量化与资源

- Stage2-B：15个D38-B fold的平均loss从1.0320降至0.1027，最终support accuracy=100%；注册前old 87.22%，说明full-batch20旧头不是本次灾难性失败的首要来源。
- Stage2-C：平均loss从7.0238降至4.1160，最终all-support accuracy仅42.80%；10步结束仍未形成可靠old/new联合边界。
- int8 vs matched FP32：全部outer held样本argmax变化0；B-int8与B-FP32的after-old/new/H完全相同。平均量化误差约`1.22e-6`，最大约`5.27e-6`，量化不是根因。
- 正式D38-B outer state：2016峰值trainable params、30epoch/30step、8522B持久状态（含registry）、13,340,160估算适配MAC、6624逐query head MAC、峰值CUDA分配22,886,912B；均过资源硬门。
- 因outer矩阵已回退Z0，预注册规则禁止D38-B full-K10 refit，故没有生成其full-K10平均/P95 head latency；`resource_audit.json`明确记录`full_k10_refit_performed=false`。不得为补齐资源展示而绕过负结果停止门。

### 10.5根因与停止决策

D38的注册前旧头与strong B3只差0.56pp；灾难发生在新类后缀加入后。旧prefix逐bit不变，但D38-A的180/180、D38-B的179/180个held旧样本被任一新类击败，说明新类分数相对旧类严重过高。CE10只改变新类方向，未提供类无关的new-vs-old尺度/偏置校准；它提高多数新类和平均new-new margin，却不能阻止旧域被新头吞噬，也未修复最低新类。

因此D38全部晋级门失败：注册前逐类不弱strong B3门失败、B>A联合门失败、matched comparator门失败、旧→新侵入门失败；仅int8/FP32一致性、旧prefix和资源协议门通过。selection按预注册规则回退Z0，query保持sealed，K1/K5/K20、125 screen和正式确认矩阵均不启动。D38到此停止，不扫epoch、lr、margin或offset，也不把本地技术完成写成性能成功。

下一轮应改变一个可解释机制：优先用全部合法support按统一公式学习共享whitening/radius或单一共享new-vs-old校准量，使旧/新score尺度可比，同时继续保留new-weight判别训练；不得使用类handle定向规则。下一候选必须同时用outer-held旧→新侵入、new-new pairwise、全部逐类floor和matched strong B3约束，先通过K10 development support screen再考虑N607/query。当前goal继续active。
