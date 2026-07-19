# D64全pair局部Block-LDA连续锦标赛探针报告

## 1.执行前登记

- ID：`d64_allpairs_blocklda_tournament_probe_20260719`；操作者：Codex`/root`；状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 目标：以全部匿名类别pair的局部3-block二类LDA连续margin替代D46/D62的全局共享协方差与行筛选，让O3、N0、N2等局部冲突边界分别获得support-only判别方向，同时保持单次全类affine query。
- 当前聚合最强D62：before92.78%、after82.22%、new84.67%、H82.62%、forgetting10.56pp、joint26.67%、min-before80%、min-after53.33%、min-new73.33%、混淆23/8/15；它仍有low遗忘恶化和rain-before下降，状态为不可晋升。
- cell：receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer fold、实际K8；复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 capsule，不重验数据。
- 版本：预注册`7ba296b9`；初始实现`4a598539`；状态契约修复`8c870efe`；worktree`E:\type10-7\code\snapshots\d64wt` detached`49a7c862` clean；当前脚本SHA256=`21b1b2bb00cb16985902e85b3f2fd4c0aac04ad61574e70cb7a80d7a202b5fb2`。
- 本地验证：py_compile通过；修复前后D43＋D64专项测试均14/14通过；修复后D42–D64整链24个文件291/291通过，用时96.2s；diff check通过。pytest退出后仅出现Windows临时目录清理`PermissionError`，命令exit0且全部测试已通过，判为包装清理噪声。
- 本轮只在本地确认的`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`执行，不访问N607，无远端PID/GPU/sync。

## 2.机制、协议与资源预期

对每个匿名类别pair `(c,d)`，D64仅使用该pair的support，在固定3-block centered特征上拟合equal-prior、auto-shrinkage二类LDA。将有向margin除以pair-support margin RMS，再对每个类别关联的全部pair连续margin取平均；最终中心化共同仿射项，并编译为一个`C×288` FP32 affine后进入既有target-old/new统一的残差int8系数＋FP16截距生命周期。

每个target row在before阶段有6类、15个pair，final阶段有11类、55个pair，共70次pair fit；30条目标row预期2100次pair fit。query不保留pair图、不做投票、阈值、图推理、batch优化或query-dependent适配，仍只执行一次全注册类affine评分，query额外MAC/state为0。方法没有角色、场景、receiver、class ID、query真值、真实batch类数、quota或global reassignment分支。

D64必须与D62/D46比较总体、三场景、11类、15fold、混淆、量化、训练和资源。只有before/after/new/H/forgetting、floor/joint以及混淆形成无交换改善时才继续第二开发seed；任一场景、new/H或floor显著受损则停止D64，不扫描pair阈值、投票权重或full/block变体。即使本cell通过，也不直接运行125。

## 3.精确执行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d64wt\code\scripts\probe_d64_allpairs_blocklda_tournament.py' `
  --d64-arm allpairs_blocklda_tournament `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d64wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d64_allpairs_blocklda_tournament_probe_20260719\allpairs_blocklda_tournament' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 4.执行闭包与总判定

- 真实runner完成105/105行、7候选×3场景×5fold、117.8342s、exit0；D64闭包60个fit audit、2100次pair fit、30条FP32/INT8目标row，query0。
- D64总体：before92.78%、after74.44%、seen-new77.33%、同rowH75.39%、forgetting18.33pp、joint43.33%、min-before86.67%、min-after60.00%、min-new66.67%、混淆37/16/18。
- 相对当前聚合最强D62：before同为92.78%，但after−7.78pp、new−7.33pp、H−7.23pp、forgetting+7.78pp；old→new+14、new→old+8、new→new+3。正信号只有aggregate class min-before/min-after分别+6.67pp、joint+16.67pp，不能覆盖主指标和三场景退化。
- receipt状态为`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，选择器回退`Z0_SUPPORT_ONLY`；D64不跑第二seed或125，D62仍是当前聚合最强开发点，但同样不满足项目门槛。

## 5.首次启动与最小修复记录

- 首次启动于35.3s、产生正式性能row前退出，异常为`D42UnifiedShrinkageLDAError: D42 state drift`；没有存活Python进程，output目录无结果文件，不能据此作任何性能判断。
- 根因：D64把pair局部结构名写入D42状态的`covariance_policy`字段，而该字段只接受受控求解器家族名。pair局部结构本已独立记录在`d43_covariance_structure`与D64审计中，因此这是状态命名空间闭包错误，不是方法公式、数据或协议失败。
- 修复只把状态字段恢复为`sklearn_lsqr_auto_shrinkage_equal_prior`并增加结构字段并存断言；没有改变pair公式、特征、训练、候选、数据、随机种子或判门。修复提交`8c870efe`已摘入干净执行工作树`49a7c862`，在291/291整链回归后允许按第3节原命令重跑。

## 6.七候选同row性能

|候选|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|B3|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|弱比较器|
|D40-HNBR|85.56%|85.00%|15.33%|25.16%|0.56pp|0%|66.67%|63.33%|0%|2/0/0|旧类稳定、新类塌缩|
|D41-BEC|86.11%|20.56%|78.67%|31.50%|65.56pp|0%|76.67%|0%|36.67%|142/0/32|旧类塌缩|
|ProtoNet/Z0|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|0/0/0|选择器回退基线|
|D64 FP32|92.78%|74.44%|77.33%|75.39%|18.33pp|43.33%|86.67%|60.00%|66.67%|37/16/18|matched参考|
|D64 INT8|92.78%|74.44%|77.33%|75.39%|18.33pp|43.33%|86.67%|60.00%|66.67%|37/16/18|诊断阴性|

D64比B3的before高5pp、new高4.67pp、H高2.04pp且joint高20pp，但after低1.11pp、遗忘多6.11pp、old→new多4。它不是全面弱于B3，而是注册后旧类保护明显不足；项目的Stage2-B/C同等优化要求因此失败。

## 7.分场景表现

|场景|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|相对D62|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|clear|96.67%|85.00%|88.00%|86.19%|11.67pp|60.00%|90.00%|70.00%|80.00%|7/4/2|before−1.67、after−6.67、new−10、H−8.25、forget+5pp|
|low-elev|88.33%|65.00%|70.00%|67.11%|23.33pp|30.00%|80.00%|50.00%|60.00%|15/8/7|before−3.33、after−13.33、new−6、H−8.87、forget+10pp|
|rain|93.33%|73.33%|74.00%|72.87%|20.00pp|40.00%|80.00%|50.00%|60.00%|15/4/9|before+5，但after−3.33、new−6、H−4.58、forget+8.33pp|

clear仍是最好场景且fold4达到100/100/100，但其均值和new下尾均下降；low-elev是主要失败面，after仅65%、遗忘23.33pp；rain的before改善没有延续到注册后。三个场景都没有形成无交换改善。

## 8.逐类性能

|类别|O0|O1|O2|O3|O4|O5|
|---|---:|---:|---:|---:|---:|---:|
|before|93.33%|93.33%|96.67%|86.67%|93.33%|93.33%|
|after|76.67%|73.33%|86.67%|63.33%|60.00%|86.67%|

|类别|N0|N1|N2|N3|N4|
|---|---:|---:|---:|---:|---:|
|seen-new|66.67%|90.00%|80.00%|83.33%|66.67%|

D64确实把D62的before旧类下尾从80%抬到86.67%，并把after全局旧类下尾从53.33%抬到60%；但O4仅60%、O3仅63.33%，N0/N4仅66.67%。也就是说，局部pair方向改善了部分最弱旧类的均衡性，却让更多旧类在新增类出现后共同失分，造成after均值和遗忘显著恶化。

## 9.十五fold同row性能

|场景-fold|before|after|new|H|forget|joint|floor(b/a/n)|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|clear-0|100.00%|83.33%|70.00%|76.09%|16.67pp|50%|100/50/50|1/2/1|
|clear-1|100.00%|83.33%|90.00%|86.54%|16.67pp|50%|100/50/50|2/0/1|
|clear-2|83.33%|75.00%|80.00%|77.42%|8.33pp|50%|50/50/50|2/2/0|
|clear-3|100.00%|83.33%|100.00%|90.91%|16.67pp|50%|100/50/100|2/0/0|
|clear-4|100.00%|100.00%|100.00%|100.00%|0pp|100%|100/100/100|0/0/0|
|low-0|100.00%|66.67%|80.00%|72.73%|33.33pp|50%|100/50/50|3/1/1|
|low-1|66.67%|58.33%|70.00%|63.64%|8.33pp|50%|50/50/50|4/0/3|
|low-2|83.33%|66.67%|60.00%|63.16%|16.67pp|0%|50/50/0|1/2/2|
|low-3|100.00%|75.00%|70.00%|72.41%|25.00pp|50%|100/50/50|3/2/1|
|low-4|91.67%|58.33%|70.00%|63.64%|33.33pp|0%|50/0/0|4/3/0|
|rain-0|100.00%|66.67%|60.00%|63.16%|33.33pp|0%|100/0/0|4/1/3|
|rain-1|100.00%|83.33%|90.00%|86.54%|16.67pp|50%|100/50/50|2/0/1|
|rain-2|91.67%|66.67%|70.00%|68.29%|25.00pp|50%|50/50/50|3/1/2|
|rain-3|83.33%|83.33%|60.00%|69.77%|0pp|50%|50/50/50|2/2/2|
|rain-4|91.67%|66.67%|90.00%|76.60%|25.00pp|50%|50/50/50|4/0/1|

15/15个final prediction SHA相对D62变化。只有clear-4全指标100%；low-4和rain-0同时出现33.33pp遗忘与零joint，low-2也有零new floor。损害不是单一异常fold，而是弱场景中的系统性注册漂移。

## 10.与既有版本同row比较

|版本|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D46|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|80.00%|53.33%|73.33%|25/8/15|可靠基准|
|D61|90.00%|83.33%|76.00%|78.96%|6.67pp|26.67%|76.67%|60.00%|43.33%|18/16/20|旧类保护、新类弱|
|D62|92.78%|82.22%|84.67%|82.62%|10.56pp|26.67%|80.00%|53.33%|73.33%|23/8/15|当前聚合最强|
|D63|93.33%|82.78%|82.00%|81.65%|10.56pp|23.33%|80.00%|53.33%|63.33%|21/11/16|新类下尾退化|
|D64|92.78%|74.44%|77.33%|75.39%|18.33pp|43.33%|86.67%|60.00%|66.67%|37/16/18|注册漂移，停止|

D64相对D62的优势集中在before旧类均衡和joint离散floor，主指标却全面下降。相对D63也有after−8.33pp、new−4.67pp、H−6.26pp、forget+7.78pp。相对D61虽然new+1.33pp、min-new+23.33pp，但after−8.89pp、forget+11.67pp。没有任何合理联合排序会把D64列为最强版本。

## 11.Pair机制诊断

- INT8目标row的before为225次pair fit、final为825次，共1050次；matched FP32再执行同量，metadata闭合2100次。pair audit计数与组合数完全一致。
- before/final每个二类pair的support准确率均为100%；编译后的before support准确率100%，final均值99.02%、最低95.45%。但held after只有74.44%、new77.33%，形成强烈的support内插—held泛化缺口。
- pair margin RMS：before最小55.50、均值174.79、最大1720.63；final最小37.59、均值195.60、最大2581.28。协方差条件数before为8.03e4–8.82e5，final为5.76e4–1.11e6；没有unit-covariance fallback。
- 注册从6类15个pair扩为11类55个pair后，每个旧类row都加入与5个新类的局部方向并重新平均。虽然公式置换等变，但不具备registry-size下的旧row不变性；结果是old→new由D62的23升至37，同时new→old由8升至16。
- 结论：失败源不是pair缺失、数值fallback或support欠拟合，而是小样本局部协方差把support完全分开后，在新增类扩图时重写所有类别几何。停止全pair局部协方差、pair RMS权重和投票/阈值变体。

## 12.量化、训练与资源

- 量化：before/final outer argmax变化0/0，support argmax变化0/0，margin sign flip0；最大score误差0.000508。FP32与INT8总体、场景、类别、fold和混淆完全相同，性能退化不是量化造成。
- margin下尾：old→new最小margin跨fold均值−0.3687，new→old−0.2571，new→new−0.1989；三类margin都出现负值，与37/16/18混淆一致。
- 训练：基础D42仍20epoch/20step；epoch1 loss1.0320、support acc95.14%、gradient norm1.0838，epoch20 loss0.1027、support acc100%、gradient norm0.1354。全部epoch的query rows总和0；D64自身0额外optimizer step。
- 资源：每row70次pair LDA fit，LDA MAC1,776,660,480；margin归一化648,480MAC，affine编译40,460MAC，连同基础metric后总适配1,782,326,060MAC。query6,624MAC、参数2,016、持久态8,583B、registry941B、峰值CUDA22,886,912B。
- 正式资源门全部通过：参数≤80k、epoch20≤30、optimizer step20≤50、持久态8,583B≤256KB、dense query graph0、query额外state/MAC0。技术可部署不等于性能可晋升。

## 13.协议与完整日志审计

- 完整解析7个JSON、1个JSONL和新增summary；105行、7候选、3场景、15fold、30条目标row、2100次pair fit均闭合，无JSON错误或非有限数。
- 全部目标training trace的query rows总和0，105行`query_opened`全为false；不访问query特征/标签/角色、真实batch类数、quota或global assignment。
- `source_sample_access=false`、`clean_sample_access=false`、单物理support单LEO观测、support/query不相交；target-old/new走同一int8/FP16 affine state。
- 完整artifact文本没有Traceback、RuntimeError、Exception、OOM或Killed。字符串搜索命中的`NaN`仅来自`provenance`单词片段，递归数值检查确认非有限数为0。
- 本轮是`development_select_unverified_component`诊断证据，formal/performance claim均不允许；不能把单receiver、单seed结果表述为正式性能。

## 14.Artifact与门槛差距

|artifact|字节|SHA256|
|---|---:|---|
|`training_log.jsonl`|10,389,362|`f32334be571bf510607caefe62eed8423ba964f9e1582a61d1288d308f0b7c49`|
|`full_performance_summary.json`|84,062|`cf58b9d7efe87fced55789bf745d5d7fa35007557c5cd1b3c5607d2b690d6e36`|
|`D64_PROBE_METADATA.json`|1,798|`9191808564d42021e3b70105df7351dcb073986681a26db3d3a557a56e334caf`|
|`RECEIPT.json`|4,656|`faba60b7681ba02c99e2c81f142537711b5341c747f86f04dbab3e1e4062ef9e`|
|`support_audit.json`|313,279|`91503438166663475d3294ed5506ba7cf15d3830a75324bf4ea4a4c95aa76408`|
|`geometry_audit.json`|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|`resource_audit.json`|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|`selection.json`|2,992|`ad57d006e0fe64bdca53d3052dffbeda1f3d18dd25f217aa14710bd5e22b29a4`|

D64距K10门槛仍差after17.56pp、min-old28pp、new5 14.67pp，且遗忘18.33pp；因此不允许启动125。当前最强仍为D62，但D62距门槛也差after9.78pp、min-old34.67pp、new5 7.33pp。下一轮必须换成registry-consistent、每类局部状态独立且不因新增类重写旧row的机制，同时保持类别公式相同；不能继续微调D64的pair权重或门限。
