# D56 LOO混淆流平衡报告

## 1.状态、目标与单一差异

- 状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；operator Codex；105/105行完成，exit0，Runner elapsed111.533s；本轮不运行125。
- 固定development cell：receiver20-1、seed713101、K10/new5、3个`leo_*_weak`场景×5fold；复用`VALIDATED_ONCE p2_min_v1`。
- 当前最强D46为before92.22%、after81.67%、new84.67%、H82.33%、forget10.56pp、min-after53.33%、min-new73.33%，仍未达到项目门槛。
- D55证明raw LOO-CE不能直接作为logit截距。D56仅把D46的support内部held预测变成离散有向混淆流，不使用CE幅值、class ID、old/new角色、scene、receiver、outer-held或query。

## 2.预注册公式

对D46 full/block两个head的每个inner-held样本，以D46已锁定的类级权重和RMS尺度形成held分数并独立argmax。若真实support类为`y`、held预测为`p!=y`，在有向图中记录边`y→p`。对每个匿名注册类`c`：

`out_c=sum_j!=c count(c→j)`

`in_c=sum_i!=c count(i→c)`

`Delta b_c=(out_c-in_c)/(K*C)`

`W_D56=W_D46`，`b_D56=b_D46+Delta b`，最后只删除类公共截距常数。因为图中每条错误边同时贡献一个out和一个in，`sum_c Delta b_c=0`。分母固定为全部held support数`K*C`，不是可调尺度；只执行一次，不回流重算图。K1/K2精确D46 fallback。

## 3.协议与禁止项

- support label只用于合法inner-held真实类和混淆边；query rows/features/labels/role/quota/true-count/global assignment均不可达。
- clean/source访问false；不恢复clean，不生成第二LEO观测，不改变capsule/split/schema。
- 所有类别使用同一公式；类标签置换时图、修正和输出同步置换；无具体TX名单。
- 禁止alpha、temperature、clip、threshold、第二arm、场景门、旧新类门、development结果后缩放及第二seed调参。
- 最终仍是单affine int8系数＋FP16截距逐query独立argmax；dense query graph为0。混淆图只在adaptation时由support构造，不进入query路径。

## 4.成功门、停止门与可观测结果

D56必须至少保持D46总体after81.67%、new84.67%、H82.33%、min-after53.33%、min-new73.33%、joint23.33%，forget不得高于10.56pp；clear/low/rain不得出现以一侧换另一侧的场景伤害；相对D46至少改变1个final prediction；INT8/FP32 before/final/margin翻转必须为0/0/0。若任一门失败，标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不跑第二seed、formal或125。

重点观测：三类总体与场景混淆、逐类old before→after、逐类new、15个outer rows、D46同折correct/wrong变化、图的out/in/net-flow分布、修正L1/L2/max、20epoch训练、量化误差、额外inner-fit资源和全部artifact SHA。报告必须保留7候选同排性能，不能只写缺陷。

## 5.实现与执行计划

1. 在D46之外重建一次相同的support inner-held full/block head，仅收集分数，不改变B20或外层fit；D46最终权重与RMS保持锁定。
2. 为混淆流、零和、类置换、rank置换、K1/K2回退、单次应用、资源闭合和tamper fail-close添加定向测试。
3. 在`ssr-gpu`下执行`py_compile`和D46＋D56窄回归；进入Git提交后，从clean detached worktree运行同一105行development矩阵。
4. 输出和本报告完成前不启动D57；D56若失败，下一轮不得扫描流强度。

本地实现已落在`code/scripts/probe_d56_loo_confusion_flow_intercept.py`，定向测试为`tests/test_probe_d56_loo_confusion_flow_intercept.py`。`py_compile`通过，D56＋D46定向回归23/23通过；覆盖混淆边流守恒、类置换、无效held score fail-close、K1/K2、固定分母、额外32次inner LDA fit及MAC/比较计数。尚未读取本轮outer结果。

## 6.版本与远端边界

Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，分支`codex/cvs-rffi-release-20260626`；根目录不是Git仓库，完成后镜像本报告。当前尚未访问N607；任何远端同步或执行必须先完成本地实现、测试、提交和N607只读preflight。

## 7.执行锁

- 实现提交：`8e6264470b2ae1b905278581e599cfb2db4d56e2`；clean detached worktree：`E:\type10-7\code\snapshots\d56wt`，状态仅`HEAD (no branch)`。
- clean探针SHA256：`95b38dd5cc7fafabfc3a06584a1571145c8c543f63da295a40caa2b523612252`；clean环境下`py_compile`和D56＋D46测试23/23通过。
- runtime只读复用`E:\type10-7\code\snapshots\d41wt`。before/after seal、envelope、component manifest和class binding继续锁定D46—D55同一组SHA：`53ace286…d9f75`、`c70aedf3…b50ff`、`31a2ad99…ceb0e`、`a2483d6e…be76`、`15b5e144…629c`、`bb89a1db…c901f`。
- 输出`E:\type10-7\automation_reports\CV-SincNet\d56_loo_confusion_flow_probe_20260719\loo_confusion_flow_intercept`启动前必须不存在；本地串行`device=auto`，不访问N607、不生成125。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d56wt\code\scripts\probe_d56_loo_confusion_flow_intercept.py' `
  --d56-arm loo_confusion_flow_intercept `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d56wt' `
  --before-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only' `
  --before-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json' `
  --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --before-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json' `
  --before-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json' `
  --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only' `
  --after-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json' `
  --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --after-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json' `
  --after-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json' `
  --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding 'E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json' `
  --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output 'E:\type10-7\automation_reports\CV-SincNet\d56_loo_confusion_flow_probe_20260719\loo_confusion_flow_intercept' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 8.结论先行

D56产生了清晰但不可晋级的old/new交换：相对D46，after-old从81.67%升至83.33%，forgetting从10.56pp降至8.33pp，min-after从53.33%升至56.67%，old→new减少4次；seen-new却从84.67%降至80.67%，H从82.33%降至80.95%，min-new从73.33%降至60.00%，new→old增加4次。15折中5折预测发生变化，clear5折完全不变；负交换集中在low-elev和rain。D56不晋级、不跑第二seed、不formalize、不运行125。

这不是量化或优化失败。INT8与matched FP32的before/final outer argmax、support argmax和margin sign flip均为0，最大score误差0.001915；20epoch support训练正常收敛。失败机制是类对称的混淆流平衡会降低support中“过度吸收”其他类的类别分数，它确实减少old→new，但同一动作增加new→old并压低新类floor，无法同时满足Stage2-B与Stage2-C。

## 9.七候选完整同排性能

unknown/coverage/rollback/defer不属于本闭集support-only Runner，均为N/A。表中每行指标来自同一候选的15个outer rows，不拼接边际极值。

|候选|机制|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆old→new/new→old/new→new|判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|B3_SINGLE_IQ_DIAG_FFTRF|B3单IQ对角FFTRF|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|低于D46|
|D42-D40-HNBR-INT8-NEGATIVE|HNBR旧负路线|85.56%|85.00%|15.33%|25.16%|0.56pp|0.00%|66.67%|63.33%|0.00%|2/0/0|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|BEC旧负路线|86.11%|20.56%|78.67%|31.50%|65.56pp|0.00%|76.67%|0.00%|36.67%|142/0/32|旧类崩塌|
|D42-PROTOnet-CDA-ZID160|ProtoNet-CDA|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|弱基线|
|D42-USLDA-FP32-MATCHED|D56 matched FP32|91.67%|83.33%|80.67%|80.95%|8.33pp|23.33%|80.00%|56.67%|60.00%|21/12/17|与INT8一致，负结果|
|D42-USLDA-INT8|D56混淆流平衡|91.67%|83.33%|80.67%|80.95%|8.33pp|23.33%|80.00%|56.67%|60.00%|21/12/17|主候选，old改善但new退化|
|Z0_SUPPORT_ONLY|support-only原型|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|弱基线|

## 10.三场景性能与行为

|场景|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|相对D46表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|leo_clear_weak|98.33%|90.00%|98.00%|93.57%|8.33pp|40.00%|90.00%|70.00%|90.00%|4/1/0|全部指标与预测完全不变|
|leo_low_elev_weak|88.33%|78.33%|74.00%|75.10%|10.00pp|20.00%|80.00%|60.00%|40.00%|8/5/8|after不变，new−2.00pp、H−0.88pp，min-new−10pp|
|leo_rain_weak|88.33%|81.67%|70.00%|74.17%|6.67pp|10.00%|60.00%|40.00%|50.00%|9/6/9|after+5.00pp、forget−6.67pp，但new−10.00pp、H−3.28pp、min-new−20pp|

rain证明混淆流包含修复旧类遗忘的有效信号；但该信号通过压低新类竞争力实现，违反“旧域适应和新类注册同等重要”的联合门，不能作为成功版本或按角色拆分使用。

## 11.逐类别性能

|旧类|哈希前缀|before→after|变化|
|---|---|---:|---:|
|O0|cls_1f33|90.00→90.00%|0.00pp|
|O1|cls_33bb|96.67→90.00%|-6.67pp|
|O2|cls_75aa|93.33→90.00%|-3.33pp|
|O3|cls_8b02|80.00→56.67%|-23.33pp，旧类floor|
|O4|cls_a53c|100.00→80.00%|-20.00pp|
|O5|cls_f8df|90.00→93.33%|+3.33pp|

|新类|哈希前缀|seen-new|表现|
|---|---|---:|---|
|N0|cls_09f8|70.00%|困难类|
|N1|cls_1c2a|93.33%|最佳|
|N2|cls_b8fb|60.00%|全局新类floor|
|N3|cls_d3af|90.00%|稳定|
|N4|cls_f608|90.00%|稳定|

逐场景瓶颈是low-elev N2=40%、N0=50%，rain N2=50%、N0=60%；D56没有把错误平均化为可接受的统一floor。

## 12.十五折完整表现

|场景|fold|before|after|new|H|forget|joint|before/after/new floor|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|0|100.00%|100.00%|90.00%|94.74%|0.00pp|50.00%|100/100/50%|0/1/0|
|clear|1|100.00%|83.33%|100.00%|90.91%|16.67pp|0.00%|100/0/100%|1/0/0|
|clear|2|91.67%|83.33%|100.00%|90.91%|8.33pp|50.00%|50/50/100%|1/0/0|
|clear|3|100.00%|91.67%|100.00%|95.65%|8.33pp|50.00%|100/50/100%|1/0/0|
|clear|4|100.00%|91.67%|100.00%|95.65%|8.33pp|50.00%|100/50/100%|1/0/0|
|low|0|91.67%|66.67%|80.00%|72.73%|25.00pp|50.00%|50/50/50%|4/1/1|
|low|1|66.67%|58.33%|70.00%|63.64%|8.33pp|0.00%|50/50/0%|1/0/3|
|low|2|91.67%|91.67%|70.00%|79.38%|0.00pp|0.00%|50/50/0%|0/2/1|
|low|3|100.00%|100.00%|70.00%|82.35%|0.00pp|0.00%|100/100/0%|0/1/2|
|low|4|91.67%|75.00%|80.00%|77.42%|16.67pp|50.00%|50/50/50%|3/1/1|
|rain|0|83.33%|83.33%|40.00%|54.05%|0.00pp|0.00%|50/50/0%|2/2/4|
|rain|1|91.67%|83.33%|70.00%|76.09%|8.33pp|0.00%|50/50/0%|2/3/0|
|rain|2|91.67%|83.33%|80.00%|81.63%|8.33pp|50.00%|50/50/50%|1/0/2|
|rain|3|91.67%|83.33%|80.00%|81.63%|8.33pp|0.00%|50/0/50%|2/0/2|
|rain|4|83.33%|75.00%|80.00%|77.42%|8.33pp|0.00%|50/50/0%|2/1/1|

## 13.与D46当前最强版本比较

|指标|D46|D56|差值|
|---|---:|---:|---:|
|before|92.22%|91.67%|-0.56pp|
|after|81.67%|83.33%|+1.67pp|
|seen-new|84.67%|80.67%|-4.00pp|
|H|82.33%|80.95%|-1.39pp|
|forgetting|10.56pp|8.33pp|-2.22pp，改善|
|joint|23.33%|23.33%|0.00pp|
|min-before|min80.00%|80.00%|0.00pp|
|min-after|53.33%|56.67%|+3.33pp|
|min-new|73.33%|60.00%|-13.33pp|
|混淆|25/8/15|21/12/17|-4/+4/+2|

D56相对D46改变5/15个预测SHA：low fold4丢1个new正确；rain fold0丢2个new正确；rain fold1以before−8.33pp、new−20pp换after+16.67pp；rain fold3以new−10pp换after+8.33pp；rain fold4仅改变预测分布而汇总指标不变。不能用after和forget的改善覆盖这些同折损失。

## 14.混淆流机制审计

|阶段|每fit错误边min/mean/max|out/in degree mean|max净流|补偿L2 mean/max|单类补偿abs mean/max|
|---|---|---:|---:|---:|---:|
|before|1/5.07/9|0.844/0.844|5|0.0682/0.1141|0.0514/0.1042|
|final|6/18.13/30|1.648/1.648|9|0.0747/0.1315|0.0523/0.1023|

补偿和最大绝对误差before/final为6.94e-18/1.73e-17，图流守恒闭合。clear final平均错误边8、补偿L2 0.0417；low为22.8/0.0800；rain为23.6/0.1023。修正只在困难场景跨过边界，和实际5个changed rows一致。系数变化L2仅约2.6e-7，来自公共仿射重中心化舍入；机制作用来自截距流而非系数或量化。

## 15.训练、量化、资源与协议

- 训练：epoch1/10/20的loss mean为1.0320/0.2161/0.1027，support accuracy为95.14%/99.03%/100%，gradient norm为1.0838/0.2359/0.1354；全部20epoch的query rows合计均为0，完整逐epoch值在summary中。
- 量化：INT8与matched FP32 before/final outer argmax0/0、support argmax0/0、margin sign flip0；score绝对误差min/mean/max为0.000530/0.001001/0.001915。
- 资源：68次LDA fit、2,010,728,448 LDA MAC；相对D46新增32次fit、944,898,048 LDA MAC、8,080数值MAC-equivalent和1,120比较；总适配2,022,234,098 MAC。query仍为6,624 MAC，参数2,016，state8,583B，registry941B，CUDA峰值22,886,912B，20epoch/20step。
- 协议：coefficient int8、intercept float16；query rows/features/labels/role/quota/count/global/dependent optimization均0/false；clean/source false；dense query graph0B。support混淆图不持久化到query state。

额外32次inner fit使D56适配MAC接近D46的1.88倍，虽然仍满足参数、epoch、step和状态硬上限，但在性能未联合改善时没有Pareto价值。

## 16.Artifact闭包

输出目录：`E:\type10-7\automation_reports\CV-SincNet\d56_loo_confusion_flow_probe_20260719\loo_confusion_flow_intercept`。

|artifact|bytes|SHA256|
|---|---:|---|
|`training_log.jsonl`|12,872,957|`4d2ed4dca07caba92f20fe1aa3eeb22391a9064d8f15fa8ae27e2b575154cb6e`|
|`support_audit.json`|313,579|`6fa6486469cd6f1febaa4795ad18a3dbf49d784238f05bc4553a8494ca356539`|
|`resource_audit.json`|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|`geometry_audit.json`|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|`selection.json`|2,991|`b30d5debe462c6e966c99456842f616ea4713d44d42f95e03e9c939ed8b1675f`|
|`RECEIPT.json`|4,941|`6ee8577a916cbd2d8fb54847ecc6460907216d0d1730fc5a2f98ca789e6c514a`|
|`D56_PROBE_METADATA.json`|1,868|`4479f3acaa382d4c1fbf7f65ac6e1ab589c257f6320024ec5430182fc0c0b2c7`|
|`full_performance_summary.json`|116,006|`b5e99c7615117bebffb1d5288ae849677671a3daa4611afd9490ec344fecf815`|

## 17.门槛、缺陷与停止动作

|门槛|要求|D56|判定|
|---|---:|---:|---|
|K10 after|≥92%|83.33%|失败，差8.67pp|
|K10 min-old|≥88%|56.67%|失败，差31.33pp|
|K10 new5|≥92%|80.67%|失败，差11.33pp|
|保持D46 new/H/min-new|≥84.67/82.33/73.33%|80.67/80.95/60.00%|全部失败|
|forgetting不增|≤10.56pp|8.33pp|通过|
|协议与量化|闭合且0翻转|全部闭合|通过|

D56的具体缺陷是“无角色混淆质量守恒”：减少old→new的4次错误，恰好伴随new→old增加4次，另增2次new→new。该结果保留一个可复用事实——support混淆入流能识别rain旧类遗忘——但禁止把这一事实转成old/new角色门、场景门或流强度扫描。D56停止；当前最强仍是D46，不运行125。

下一候选必须在统一类对称公式中直接保护每类的正样本margin和负样本吸收上界，而不是只平衡预测质量或继续加截距。D57开始前先记录D55—D57三轮节奏中的第二轮状态；完成D57后执行正式三轮回顾。
