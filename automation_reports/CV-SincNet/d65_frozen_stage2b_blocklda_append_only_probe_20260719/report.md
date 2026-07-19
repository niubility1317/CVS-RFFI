# D65冻结Stage2-B Block-LDA追加式注册探针报告

## 1.执行前登记

- ID：`d65_frozen_stage2b_blocklda_append_only_probe_20260719`；操作者：Codex`/root`；状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 目标：只在Stage2-B从旧类support学习一次目标域3-block covariance，Stage2-C冻结该几何并以完全相同的`Sigma_B^-1 mu_c`公式追加新类row，验证注册时不重写旧row能否同时保护after-old并保持new。
- D64教训：before92.78%但after74.44%、new77.33%、遗忘18.33pp；pair support100%却held混淆37/16/18，根因是6类到11类扩图重写旧类几何。D65没有pair图或pair权重。
- 当前聚合最强D62：before92.78%、after82.22%、new84.67%、H82.62%、forgetting10.56pp、joint26.67%、min-before80%、min-after53.33%、min-new73.33%、混淆23/8/15；仍不可晋升。
- cell：receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer fold、实际K8；复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 capsule，不重验数据。
- 版本：预注册`26f326ae`；实现`364a56c4`；worktree`E:\type10-7\code\snapshots\d65wt` detached clean；脚本SHA256=`bc0c6e14191e09f773e12e7e9f194e097204c7183d244b1e5d867a339f5e4acb`。
- 验证：py_compile通过；D43＋D65专项15/15通过，含真实D42 K5 FP32/INT8逐bit追加集成测试；D42–D65整链25文件299/299通过，用时85.8s；diff check通过。pytest退出后仅有Windows临时目录清理权限噪声，命令exit0。
- 本轮只在本地确认的`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`执行，不访问N607，无远端PID/GPU/sync。

## 2.机制、协议与资源预期

Stage2-B用旧类support估计equal-prior auto-shrinkage covariance并保留z160/FFT96/RF32三个对角块；去掉equal-prior公共常数后，每类row为`w_c=Sigma_B^-1 mu_c`、`b_c=-0.5 mu_c^T w_c`。Stage2-C不更新`Sigma_B`，旧类FP32 row、int8两段码、block scale和FP16 intercept必须逐bit不变，只用相同公式追加5个新类row。query仍是单次全注册类affine argmax，不读取注册时序或old/new角色。

每个target row只做1次Stage2-B covariance fit和5个新row求解；预计covariance MAC28,366,848、append MAC429,125，连同基础metric后总适配约33,772,613MAC。query6,624MAC、额外state/MAC0；参数、epoch、step、状态沿D42正式面。没有freeze强度、协方差混合、阈值、角色、场景、receiver或class ID分支。

必须相对D62完整比较总体、三场景、11类、15fold、混淆、量化、训练和资源。旧row任一逐bit不一致直接fail closed；任一主指标、场景、floor或混淆退化则停止D65，不扫描freeze系数或full/block混合。即使本cell通过，也不直接运行125。

## 3.精确执行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d65wt\code\scripts\probe_d65_frozen_stage2b_blocklda_append_only.py' `
  --d65-arm frozen_stage2b_blocklda_append_only `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d65wt' `
  --before-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only' `
  --before-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json' --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --before-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json' `
  --before-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json' --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only' `
  --after-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json' --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --after-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json' `
  --after-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json' --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding 'E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json' --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output 'E:\type10-7\automation_reports\CV-SincNet\d65_frozen_stage2b_blocklda_append_only_probe_20260719\frozen_stage2b_blocklda_append_only' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 4.执行闭包与总判定

- runner完成105/105行、7候选×3场景×5fold、33.2659s、exit0；60个fit audit、30组before/final冻结协方差、FP32/INT8旧row逐bit不变、query0全部闭合。
- D65 INT8总体：before92.22%、after86.11%、new59.33%、同rowH67.12%、forgetting6.11pp、joint16.67%、min-before80%、min-after70%、min-new46.67%、混淆16/28/33。
- 相对D62：after+3.89pp、forgetting−4.44pp、min-after+16.67pp、old→new−7，证明追加式冻结有效保护旧类；但new−25.33pp、H−15.50pp、joint−10pp、min-new−26.67pp，new→old+20、new→new+18。Stage2-C严重失败，不能用旧类改善掩盖。
- FP32 new60.00%、H67.90%、joint20%；INT8有4个outer argmax变化、4个margin sign flip，量化门也失败。
- 状态`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；D65不跑第二seed或125。D62仍是当前聚合最强开发点，但也未达到项目门槛。

## 5.七候选同row性能

|候选|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|B3|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|弱比较器|
|D40-HNBR|85.56%|85.00%|15.33%|25.16%|0.56pp|0%|66.67%|63.33%|0%|2/0/0|旧类稳定、新类塌缩|
|D41-BEC|86.11%|20.56%|78.67%|31.50%|65.56pp|0%|76.67%|0%|36.67%|142/0/32|旧类塌缩|
|ProtoNet/Z0|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|0/0/0|选择器回退基线|
|D65 FP32|92.22%|86.11%|60.00%|67.90%|6.11pp|20.00%|80.00%|66.67%|46.67%|16/27/33|matched参考|
|D65 INT8|92.22%|86.11%|59.33%|67.12%|6.11pp|16.67%|80.00%|70.00%|46.67%|16/28/33|诊断阴性|

D65比B3的after高10.56pp、遗忘少6.11pp、old→new少17，但new低13.33pp、H低6.23pp、joint低6.67pp。它是有效的旧类保护机制证据，不是有效的联合分类器。

## 6.分场景表现

|场景|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|相对D62|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|clear|98.33%|93.33%|86.00%|88.55%|5.00pp|50.00%|90.00%|80.00%|80.00%|3/4/3|after+1.67、forget−1.67，但new−12、H−5.89pp|
|low-elev|88.33%|81.67%|42.00%|51.40%|6.67pp|0%|80.00%|70.00%|10.00%|6/15/14|after+3.33、forget−6.67，但new−34、H−24.58、joint−20pp|
|rain|90.00%|83.33%|50.00%|61.41%|6.67pp|0%|60.00%|50.00%|20.00%|7/9/16|after+6.67、forget−5，但new−30、H−16.04、joint−10pp|

clear说明冻结几何在匹配较好时可同时获得93.33%after和86%new；但low-elev/rain的新类标尺完全失配。三个场景都没有达到“after/new/H不退化”的预注册门。

## 7.逐类性能

|类别|O0|O1|O2|O3|O4|O5|
|---|---:|---:|---:|---:|---:|---:|
|before|90.00%|96.67%|96.67%|80.00%|100.00%|90.00%|
|after|90.00%|93.33%|96.67%|70.00%|76.67%|90.00%|

|类别|N0|N1|N2|N3|N4|
|---|---:|---:|---:|---:|---:|
|seen-new|46.67%|66.67%|66.67%|70.00%|46.67%|

旧类最低从D62的53.33%升至70%，O3/O4仍是下尾但不再塌缩；所有新类都低于项目92%门，N0/N4仅46.67%。问题是全体新类尺度不足，而非单一难类。

## 8.十五fold同row性能

|场景-fold|before|after|new|H|forget|joint|floor(b/a/n)|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|clear-0|100.00%|100.00%|60.00%|75.00%|0pp|50%|100/100/50|0/3/1|
|clear-1|100.00%|91.67%|90.00%|90.83%|8.33pp|50%|100/50/50|1/0/1|
|clear-2|91.67%|83.33%|90.00%|86.54%|8.33pp|50%|50/50/50|1/1/0|
|clear-3|100.00%|100.00%|90.00%|94.74%|0pp|50%|100/100/50|0/0/1|
|clear-4|100.00%|91.67%|100.00%|95.65%|8.33pp|50%|100/50/100|1/0/0|
|low-0|91.67%|83.33%|60.00%|69.77%|8.33pp|0%|50/50/0|2/2/2|
|low-1|66.67%|58.33%|60.00%|59.15%|8.33pp|0%|50/50/0|2/1/3|
|low-2|91.67%|83.33%|50.00%|62.50%|8.33pp|0%|50/50/0|1/2/3|
|low-3|100.00%|100.00%|20.00%|33.33%|0pp|0%|100/100/0|0/5/3|
|low-4|91.67%|83.33%|20.00%|32.26%|8.33pp|0%|50/50/0|1/5/3|
|rain-0|83.33%|83.33%|50.00%|62.50%|0pp|0%|50/50/0|2/0/5|
|rain-1|100.00%|100.00%|40.00%|57.14%|0pp|0%|100/100/0|0/3/3|
|rain-2|91.67%|83.33%|40.00%|54.05%|8.33pp|0%|50/50/0|2/1/5|
|rain-3|91.67%|75.00%|60.00%|66.67%|16.67pp|0%|50/0/50|2/3/1|
|rain-4|83.33%|75.00%|60.00%|66.67%|8.33pp|0%|50/50/0|1/2/2|

相对D62有14/15个final prediction SHA变化。low-3/4虽然after为100%/83.33%，new都只有20%；rain五折joint全为0。新类失败是跨fold系统现象。

## 9.与既有版本同row比较

|版本|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|主要行为|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D46|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|80.00%|53.33%|73.33%|25/8/15|可靠基准|
|D61|90.00%|83.33%|76.00%|78.96%|6.67pp|26.67%|76.67%|60.00%|43.33%|18/16/20|旧类保护、新类弱|
|D62|92.78%|82.22%|84.67%|82.62%|10.56pp|26.67%|80.00%|53.33%|73.33%|23/8/15|当前聚合最强|
|D63|93.33%|82.78%|82.00%|81.65%|10.56pp|23.33%|80.00%|53.33%|63.33%|21/11/16|新类下尾退化|
|D64|92.78%|74.44%|77.33%|75.39%|18.33pp|43.33%|86.67%|60.00%|66.67%|37/16/18|注册重写旧row|
|D65|92.22%|86.11%|59.33%|67.12%|6.11pp|16.67%|80.00%|70.00%|46.67%|16/28/33|旧row冻结、新类塌缩|

D65取得目前最高after和最低遗忘之一，但以目前最低的可用路线new/H为代价。联合目标要求Stage2-B/C同等优化，因此D62仍是最强版本。

## 10.追加式生命周期与遗忘机制

- 30/30组before/final covariance SHA一致；60个fit audit闭合。所有FP32旧row、int8两段codes、三块FP16 scale、FP16 intercept和`log_diag`逐bit不变。
- before support准确率100%；final support准确率均值95.98%、最低92.05%。协方差condition为8.48e4–1.67e5，无unit fallback。
- 遗忘低的直接原因是注册不再重估旧类row：old→new从D62的23降到16，after由82.22%升至86.11%，min-after由53.33%升至70%。
- 但新增row完全沿用只由旧类target support估计的`Sigma_B`，新类没有参与尺度估计；弱场景新类大量被旧类吸收，new→old由8升至28，new→new由15升至33。

## 11.地面压缩旧类原型的实际使用边界

本轮没有把封存地面int8旧类原型作为D65拟合或预测输入。每条目标row均明确记录`ground_int8_component_input_count=0`、`ground_int8_update_access=false`。组件`int8_domain_class_prototypes.npz`只被Runner加载并做入口/出口哈希审计，SHA均为`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`，证明只读未变。

D42训练日志中的`prototype_anchor_loss`锚定的是当前target-old support生成的prototype，不是地面int8原型。故D65的低遗忘应归因于追加式冻结，不能归因于充分利用Phase1压缩知识。下一路线若使用地面原型，只能使用协议允许的共同封存int8多样本聚合，通过support-only学习对所有类别统一的变换；不能在query阶段按old/new角色分支。

## 12.量化、训练与资源

- 量化：before outer/support argmax变化0；final outer argmax变化4、support变化1、margin sign flip4，最大score误差1.000732。FP32 new60.00%→INT8 59.33%，joint20%→16.67%，new→old27→28。
- 原因：为了保证旧row逐bit不变，D65省略了随registry重新计算的类别公共仿射中心化，未中心化截距重新暴露FP16约1.0的误差。量化失败独立于新类均值塌缩，二者都需要解决。
- 训练：基础D42为20epoch/20step；epoch1 loss1.0320、support acc95.14%，epoch20 loss0.1027、support acc100%；所有epoch query rows总和0。D65自身0额外optimizer step。
- 资源：每row1次Stage2-B covariance fit；LDA/append合计28,795,973MAC，其中5个新row追加429,125MAC；连同metric总适配33,772,613MAC。query6,624MAC、参数2,016、持久态8,583B、registry941B、峰值CUDA22,886,912B。
- 参数/epoch/step/state/dense-query资源门均通过；性能与量化门失败，不能晋升。

## 13.协议、完整日志与artifact

- 完整解析7个JSON、1个JSONL及summary：105行、7候选、3场景、15fold、30条目标row、30组冻结covariance全部闭合，无JSON错误或非有限数。
- training trace query rows总和0，105行`query_opened=false`；无clean/source、query role、真实batch类数、quota、global assignment或query batch优化。
- 完整artifact文本无Traceback、RuntimeError、Exception、OOM、Killed或Infinity。

|artifact|字节|SHA256|
|---|---:|---|
|`training_log.jsonl`|6,953,705|`c5ba9fec6d736767c373e5a42e1193d34339c73c7ad1b0f1ccc8754f59c2943f`|
|`full_performance_summary.json`|97,242|`12c472f97f5a7bb1dbe8c27f1766766af871f4258ad30281825ece31651c8436`|
|`D65_PROBE_METADATA.json`|1,978|`28eb34b05f7bfae9807c5d830fc127f2cbefbade9b3945852118b7eefe09d35f`|
|`RECEIPT.json`|4,656|`fcbfa7ccdea4c087b9c4b811e9aff5a12fde250a7aa42b3b74db48b4281b4352`|
|`support_audit.json`|313,288|`090c51e3cfbfdf1713d559b66f51317b87c3fe1579fdd935706c5e472a543624`|
|`geometry_audit.json`|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|`resource_audit.json`|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|`selection.json`|2,993|`8a1572f2a189c7772f70622c4faadd672ebd0abff30d7cbd1218a0dd6c3b68a8`|

## 14.项目门槛差距与下一步

D65距K10门槛为after5.89pp、min-old18pp、new5 32.67pp；新类是决定性失败项，且量化不满足零翻转要求，不启动125。D65停止冻结旧covariance直接追加、freeze强度和full/block混合扫描。D66前先核对D36等ground-anchor历史路线；只有未被正式否证的“封存地面原型→统一类无关域变换→所有类别同一评分公式”才允许进入预注册。
