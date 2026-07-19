# D70交叉拟合原子生命周期行替换探针

## 1.执行前登记

- 实验ID：`d70_crossfitted_atomic_lifecycle_row_replacement_probe_20260719`；operator：Codex；最终状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 当前联合最强D62：B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67，min-B/A/N=80.00/53.33/73.33，混淆23/8/15。
- D69完整结果为92.78/81.67/74.67/77.39/11.11/30.00，min-N53.33%、混淆27/23/15；全旧行冻结跨坐标系交换已否决。D67–D69正式复盘见D69报告第15节，提交`6c5f924e`。
- 根目录`E:\type10-7`非Git；本报告镜像、代码、测试和追踪进入`E:\type10-7\github_publish\CVS-RFFI-repo`。其他工作树改动与D70无关，只暂存D70拥有路径。

## 2.方法锁

K>=2时使用两个按physical rank预定的互斥support-held fold。每折在train部分分别拟合D62 before-old和D62 final-joint；held全部类上，以final-joint为base，逐个旧行测试before行替换。单行要求本类TP不降、FP不增且至少一项严格改善；全部初选行联合替换后，必须对11类逐类TP不降且FP不增，否则mask全清零。full support只按mask在D62 final head中替换旧行，新行始终为final joint行。K1精确D62 fallback。

没有连续权重、center/scale、符号、温度、offset、class名单、scene/receiver或query角色分支。所有候选旧行使用同一计数公式；最终是一个全注册类affine head。

## 3.目标、停止条件与完整报告

- before、空mask fallback必须精确D62；两折partition exact-once，gate联合TP/FP原子安全，旧/新类评价同等。
- 相对D62必须无A/N/H/J/min-A/min-N/场景floor交换，并至少改善A/F/J/floor之一；否则`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- INT8相对FP32的argmax变化和margin sign flip为0；资源、query独立性和状态上限通过。
- 真实105行后完整报告7候选、3场景、11类、15fold、mask/TP/FP、训练、量化、资源、artifact及D62/D65/D66/D67/D68/D69对照。失败不跑第二seed/125。

## 4.数据与协议

固定receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8；复用D18`VALIDATED_ONCE/p2_min_v1`enrollment-only capsule，不重验数据。query只测试，no clean/source、query truth/role/quota/global assignment、class-ID规则或dense query graph。ground输入锁0：D22仍`formal_phase2_eligible=false`，D66的84-cell接入为负交换。

## 5.实施计划

新增独立D70 partition/gate/lifecycle core、probe和专项测试，不修改D62/D69历史实现。先做合成partition、原子gate、置换、空mask精确fallback、K1、旧行选择、新行恒定、compiled state、禁止访问和资源闭包测试；再跑D42–D70完整链、提交、干净worktree复跑，最后才登记真实105行命令。

## 6.实现与本地验证

- `code/cvsrffi/stage2_d70_atomic_lifecycle.py`：两折rank partition、TP/FP计数、coordinate gate、all-class atomic gate和Stage2-B/Stage2-C配对生命周期。
- `code/scripts/probe_d70_crossfitted_atomic_lifecycle_row_replacement.py`：复用锁定D62与D42 runner，记录60次top-level fit、30对生命周期、120次inner D62和2280条component fit；单独计入inner LDA/Fisher/held-score/gate MAC。
- 两个测试文件共10项，覆盖partition exact-once、原子安全、置换等变、K1精确D62、选择性旧行、新行joint不变、support漂移拒绝、source closure和禁止分支。
- 专项10/10通过；D42–D70完整链345/345通过，用时81.5s，34个测试文件，包含D42 integration20项。
- 主工作树source SHA：core`f2e67c142ba8fbe797a019e724435a86b67db8446efb9ba49c96abb593b47459`；probe`ff74748be440648ade9c45c60d12c53ea71e149d74180b30a4c1570a257072c2`。

当前只有代码/合成验证，不能声明性能。下一步提交精确文件，建立干净worktree复跑345项；干净链通过后才登记真实105行命令。

## 7.干净版本与真实运行锁

- 实现提交`10536c01`；干净worktree`E:\type10-7\code\snapshots\d70wt`为detached HEAD且`git status -sb`仅`## HEAD (no branch)`。
- 干净D42–D70完整链345/345通过，用时82.8s；运行目录`E:\type10-7\local_artifacts\d70_clean_full_chain_345`。
- 实际checkout SHA：probe`1024ce5bcc4abed430a19acda811bb0fedca422ba76206acb60984e819d87ecc`、core`94d50258db904a5e289fefe8300966435895c171961892cbb07490c4e2027003`、D62 helper`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。
- 本地运行，不用N607。输出目录登记时不存在，禁止覆盖或原目录重跑。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d70wt\code\scripts\probe_d70_crossfitted_atomic_lifecycle_row_replacement.py' `
  --d70-arm crossfitted_atomic_lifecycle_row_replacement `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d70wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d70_crossfitted_atomic_lifecycle_row_replacement_probe_20260719\crossfitted_atomic_lifecycle_row_replacement' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期闭包：105行、30个目标row、60个D70 fit audit、30对生命周期、120个inner D62 fit、2280条component fit。每个final audit两折覆盖88个support row exact-once；active mask必须all-class atomic safe，空mask精确D62；ground/query/clean/source/role/quota访问0。

## 8.真实运行完成状态

- 本地进程正常退出，105/105行完整，runner用时211.610s，含shell总用时219.817s；未使用N607。
- 结果目录：`E:\type10-7\automation_reports\CV-SincNet\d70_crossfitted_atomic_lifecycle_row_replacement_probe_20260719\crossfitted_atomic_lifecycle_row_replacement`。
- selection最终仍为`Z0_SUPPORT_ONLY`，receipt为diagnostic negative；这只表示D70未达到promotion gate，不改变目标候选的完整诊断有效性。
- 验证闭包：15个INT8 before、15个INT8 final、15个matched FP32对应行；query/clean/source/ground输入均为0，单一全注册类affine state成立。

## 9.七候选完整同一行性能

所有值为15个同一候选outer row的均值；B/A/N/H/F/J与min-B/A/N均为百分数，混淆依次为旧→新/新→旧/新→新。

|候选|B|A|N|H|F|J|min-B|min-A|min-N|混淆|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D42-USLDA-INT8，即D70|92.78|82.22|84.67|82.62|10.56|26.67|80.00|53.33|73.33|25/8/15|目标候选；未优于D62|
|D42-USLDA-FP32-MATCHED|92.78|82.22|84.67|82.62|10.56|26.67|80.00|53.33|73.33|25/8/15|与INT8 argmax完全一致|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78|75.56|72.67|73.35|12.22|23.33|80.00|60.00|40.00|33/22/19|诊断基线|
|D42-D40-HNBR-INT8-NEGATIVE|85.56|85.00|15.33|25.16|0.56|0.00|66.67|63.33|0.00|2/0/0|低遗忘来自新类失效|
|D42-D41-BEC-INT8-NEGATIVE|86.11|20.56|78.67|31.50|65.56|0.00|76.67|0.00|36.67|142/0/32|旧类崩溃|
|D42-PROTOnet-CDA-ZID160|71.11|48.33|52.67|48.97|22.78|0.00|33.33|13.33|3.33|0/0/0|诊断基线|
|Z0_SUPPORT_ONLY|71.11|48.33|52.67|48.97|22.78|0.00|33.33|13.33|3.33|0/0/0|selection fallback|

## 10.场景与类别表现

|场景|B|A|N|H|F|J|min-B|min-A|min-N|row-floor B/A/N|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|98.33|91.67|98.00|94.44|6.67|50.00|90.00|70.00|90.00|90.00/60.00/90.00|4/1/0|
|leo_low_elev_weak|91.67|78.33|76.00|75.98|13.33|20.00|80.00|60.00|50.00|70.00/60.00/20.00|8/5/7|
|leo_rain_weak|88.33|76.67|80.00|77.45|11.67|10.00|60.00|30.00|70.00|60.00/30.00/30.00|13/2/8|

类映射按注册顺序固定：old1=`cls_1f33441efa14970113b27483344b7df852a9041984b38d34ce570fafbab6689c`，old2=`cls_33bbd16556c6e6305d1b7162f5ea71393afba910a922f9abca5999d5921a2d9d`，old3=`cls_75aa6d506081240f50cf3b79a0bd91714fa0084a635a472ca63194e57ec1dca2`，old4=`cls_8b02d99905a8fe579368ac8e37eff51c505aaa89a646eba8892d5d800aa08416`，old5=`cls_a53ca1280d8fca58e3f4d6d1e9ddabfdab6027a941ee8c3f8c01d9d8ec945725`，old6=`cls_f8dfc2edcccc5344f8e2535a959f13b53a1cddfd6fb22aed6e714de382b58d24`；new1=`cls_09f8003925445192b3169f4df2344c403ffdefd70df7104f8d65cadec76dfa30`，new2=`cls_1c2ad8827bdb06130adbedbf210f11e376a8e1da374ac19f2658d378109379e5`，new3=`cls_b8fbace568adba605b88a9b564536a6225d683b30ad152e802e7263331fe57fa`，new4=`cls_d3afb5d16e93d949709e63ffbc3589b70067a36f2d4250f07f8d1f8526d4697f`，new5=`cls_f608a348579f723bab7edda1bc314a85a47c1775afd7f97a1210c5ada51ead0d`。

|类别|old-before|old-after|new-after|主要观察|
|---|---:|---:|---:|---|
|old1|96.67|90.00|—|轻度遗忘|
|old2|96.67|90.00|—|轻度遗忘|
|old3|96.67|93.33|—|最稳定旧类|
|old4|80.00|53.33|—|全局min-A瓶颈|
|old5|93.33|73.33|—|第二旧类瓶颈|
|old6|93.33|93.33|—|无均值遗忘|
|new1|—|—|73.33|全局min-N瓶颈|
|new2|—|—|93.33|最佳新类|
|new3|—|—|76.67|低仰角波动明显|
|new4|—|—|90.00|较稳定|
|new5|—|—|90.00|较稳定|

## 11.十五fold逐行表现

|场景/fold|B|A|N|H|F|J|row-floor B/A/N|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clear/0|100.00|100.00|90.00|94.74|0.00|50.00|100/100/50|0/1/0|
|clear/1|100.00|83.33|100.00|90.91|16.67|0.00|100/0/100|2/0/0|
|clear/2|91.67|83.33|100.00|90.91|8.33|50.00|50/50/100|1/0/0|
|clear/3|100.00|100.00|100.00|100.00|0.00|100.00|100/100/100|0/0/0|
|clear/4|100.00|91.67|100.00|95.65|8.33|50.00|100/50/100|1/0/0|
|low/0|100.00|66.67|80.00|72.73|33.33|50.00|100/50/50|4/1/1|
|low/1|83.33|58.33|70.00|63.64|25.00|0.00|50/50/0|1/0/3|
|low/2|83.33|91.67|70.00|79.38|-8.33|0.00|50/50/0|0/2/1|
|low/3|100.00|100.00|70.00|82.35|0.00|0.00|100/100/0|0/1/2|
|low/4|91.67|75.00|90.00|81.82|16.67|50.00|50/50/50|3/1/0|
|rain/0|83.33|83.33|60.00|69.77|0.00|0.00|50/50/0|2/0/4|
|rain/1|100.00|66.67|90.00|76.60|33.33|0.00|100/0/50|4/1/0|
|rain/2|91.67|83.33|80.00|81.63|8.33|50.00|50/50/50|1/0/2|
|rain/3|83.33|75.00|90.00|81.82|8.33|0.00|50/0/50|3/0/1|
|rain/4|83.33|75.00|80.00|77.42|8.33|0.00|50/50/0|3/1/1|

负F只表示low/2注册后旧类均值偶然上升8.33pp，不代表无遗忘；正式结论仍依据15折同一行均值与逐类floor。

## 12.原子门、训练、量化与资源

- final gate状态：9折`joint_atomic_failure_exact_d62_fallback`，5折`no_row_accepted_exact_d62_fallback`，仅clear/fold1为active；14折mask为空，1折mask=`[true,false,false,false,false,false]`，总共只接受1个旧行。
- 唯一active折的held计数：base TP=`[7,5,7,7,7,6,7,8,3,8,6]`、FP=`[2,0,0,4,1,0,1,4,0,4,1]`；joint TP=`[8,5,7,7,7,6,7,8,3,8,6]`、FP=`[2,0,0,4,1,0,1,4,0,4,0]`。support-held上old1 TP+1且new5 FP-1，但outer准确率汇总不增。
- 30个partition全部exact-once；compiled support accuracy min/mean/max=98.86/99.85/100%；新类行始终等于D62 final joint；outer-held/query未参与门控。
- 20epoch support优化：epoch1 loss mean/min/max=1.0320/0.9732/1.1174，support acc=95.14%，gradient norm=1.0838；epoch20为0.1027/0.0756/0.1274、100%、0.1354。query rows每epoch均为0。
- INT8对matched FP32：before/final outer argmax变化0，support argmax变化0，margin sign flip0；最大score绝对误差min/mean/max=0.000377/0.000882/0.001915。
- 每target row总adaptation MAC=41,385,572,690，其中D70额外16,494,348,720；query MAC=6,624；trainable参数2,016；persistent state=8,583B，registry=941B，peak CUDA=22,886,912B；20epoch/20 optimizer steps。D70没有额外query MAC、持久状态或optimizer step，却消耗大量额外support闭式拟合计算。
- 部署为INT8 coefficient+FP16 intercept；query/clean/source/ground、role Oracle、quota、global assignment和dense query graph访问均为0，persistent state cap通过。

## 13.与D62及近期路线的同一行比较

|版本|B|A|N|H|F|J|min-A|min-N|混淆|D70相对结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D62|92.78|82.22|84.67|82.62|10.56|26.67|53.33|73.33|23/8/15|所有汇总与floor差0；D70旧→新+2且计算更大|
|D65|92.22|86.11|59.33|67.12|6.11|16.67|70.00|46.67|16/28/33|D65低F伴随N崩塌；D70 N+25.33pp但A-3.89pp|
|D66|93.33|83.33|83.33|82.59|10.00|23.33|53.33|66.67|20/9/16|D70 N+1.33pp、H+0.03pp、J+3.33pp，但B-0.56pp、A-1.11pp、F+0.56pp|
|D67|92.78|82.78|83.33|82.16|10.00|26.67|53.33|73.33|22/11/14|D70 N+1.33pp、H+0.47pp，但A-0.56pp、F+0.56pp|
|D68|58.89|51.67|14.00|18.66|7.22|0.00|43.33|0.00|20/118/11|D68低F来自before/after同时塌缩；不可解释为保旧成功|
|D69|92.78|81.67|74.67|77.39|11.11|30.00|53.33|53.33|27/23/15|D70 A+0.56pp、N+10pp、H+5.24pp、F-0.56pp，但J-3.33pp|

D70相对D62只有1/15fold预测hash变化，aggregate、三类floor和所有主指标差值均为0，旧→新混淆从23增至25。故D70严格受D62支配，不是新最强版本。

## 14.目标差距、ground事实与缺陷

- K10/new5门槛要求A>=92%、min-A>=88%、N>=92%；D70分别为82.22%、53.33%、84.67%，差9.78pp、34.67pp、7.33pp。当前development cell尚未过门，不运行第二seed，更不运行125矩阵。
- D70的ground component input count严格为0；它没有利用地面压缩旧类原型。当前唯一真实使用D22 int8地面聚合原型的是D66：84个domain-class cell、每类14个，但manifest仍`formal_phase2_eligible=false`且结果是负交换，所以不能把D66当正式ground成功证据。
- 主要缺陷不是原子门失守，而是support-held TP/FP改善对真实outer不具迁移性；唯一active替换没有提高任何汇总或floor，并增加2个旧→新错误。旧类row replacement路线至此停止，不再扫描fold、阈值、权重或温度。

## 15.artifact封存

|artifact|bytes|SHA256|
|---|---:|---|
|training_log.jsonl|21,260,641|`776f7b7c2b4088fe1723dbc8ba0f9500c2cfb33a4e0e9e8e14af192ed8e4fcfc`|
|support_audit.json|313,691|`e1d32338cd8fb52608d8b3f20d3b73da0cbcfb4ba17b4d8fb6957bcc94d85ce1`|
|resource_audit.json|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|geometry_audit.json|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|selection.json|2,990|`4fc4677bb7f6a016afa5fcc97085bfc2750b301efa0ddfce0c4f9523666a94fe`|
|RECEIPT.json|5,029|`74d763329f5877a89e3147677443f5ffa820ac1000e71afa50547a232402c4ee`|
|D70_PROBE_METADATA.json|2,329|`b0a69781597c01f1e51aabb563b7689b860584c59e0b4a6bc57a4d83ec7b2d7b`|
|d70_full_performance_summary.json|104,965|`2f6abd70ec456576849bc2a9c93e69004f8c4fbd84bd8defba9405df1affadea`|

## 16.最终判定与下一步

状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D70证明原子support gate能把大多数不安全生命周期替换回退到D62，但没有产生可推广性能，且唯一active折造成额外旧→新混淆。D62继续保持本对话当前联合最强：92.78/82.22/84.67/82.62/10.56/26.67，min-B/A/N=80.00/53.33/73.33，混淆23/8/15。下一版本必须离开旧类生命周期行交换，继续以D62全类joint head为不可破坏基座，同时直接针对low/rain的old4、old5、new1、new3共同floor问题设计类无关机制。
