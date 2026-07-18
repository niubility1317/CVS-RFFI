# D51 resultant缩放中位数centroid残差探针

## 1.状态、目标与数据单元

- 实验ID：`d51_resultant_median_centroid_residual_probe_20260719`；状态`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景×5 outer折，实际fit K8；复用同一`VALIDATED_ONCE`、`p2_min_v1`固定received-IQ capsule/split。
- 目标：保留D45全局LOO full/block融合的稳定底座，使用support内稳健centroid方向改变困难类几何；避免D46/D47/D50只重加权相同head而几乎不跨决策边界、D48截距残差过强、D49全局cosine权重失配。
- 本轮只跑本地development 105行；不访问N607、不跑第二seed、不运行125。

## 2.唯一锁定方法

D45完整输出为`(W_D,b_D)`。D51读取D42已固定的全局单位球support特征`x_{c,i}∈R^288`，对每个匿名类完全同式计算：

`a_c=mean_i(x_{c,i})`，`rho_c=||a_c||_2`，`p_c=a_c/rho_c`；

`q_c=normalize(coordinate_median_i(x_{c,i}))`；

`u_c=q_c-p_c`，`gamma_c=1-rho_c`。

先用未缩放`u_c`在全部support上形成logit，按D44相同的class-centered RMS得到单一`scale_u`。最终：

`DeltaW_c=gamma_c×u_c/scale_u`，`W'=canonical_center(W_D+DeltaW)`，`b'=canonical_center(b_D)`。

因此D51只加入无intercept的稳健方向残差；`gamma_c`连续反映本类support离散程度，范围必须在`[0,1)`，不含阈值。K1和K2的坐标中位数等于算术均值，必须逐位回退D45；K≥3若所有`u_c`为0也精确回退，否则`scale_u`退化即fail-close。coordinate median使用偶数K两个中间次序统计量的均值。

该方法只称`resultant-scaled coordinate-median centroid direction residual`。固定B20坐标有语义，但不宣称该残差对任意特征旋转等变、是posterior或有query泛化保证。

## 3.协议与禁止项

- 只读合法support特征/标签；before必须在读取new support前物化且不可变。old/new按同一公式，不读class ID、角色、receiver、scene、handle、outer-held、query、clean/source。
- query仍为一个`C×288+C`affine state，对全注册类逐样本独立argmax；不允许truth、role Oracle、batch class count、quota、global reassignment、query-dependent optimization或dense query graph。
- 不增加残差系数、temperature、clip、阈值、sign gate、trim比例、坐标块、第二arm或扫描；不得根据outer结果切换median/medoid/geometric median。
- support输入/targets、D45 base state、mean/median/resultant/RMS、量化前实际state和int8编译必须在artifact中逐项绑定并由末端verifier重算。

## 4.资源与晋级门

D51复用D45的36次LDA、20 epoch/20 step和一个query state；新增mean/median/norm、support residual scoring及一次FP32 coefficient加法，预期额外适配低于1M MAC-equivalent，query MAC、参数和state仍按实际`C×288+C`artifact报告。host FP64 peak未测不得冒充实测。

晋级必须：相对D45至少改变1条final预测；总体及各场景after/new/H/joint/min-old/min-new不退化且forgetting不增；new/min-new≥D46的84.67/73.33%；rain after/forget≥D42的78.33/≤10pp；总体forget≤8.89pp；混淆不超过D42 26/10/18；量化0/0/0和全部协议/资源/artifact门通过。任一失败即详细记录并停止本路线，不增加变体、第二seed或125。

## 5.详细性能交付要求

完成后必须写入7候选总体、3场景、11类、15折、相对D42/D45/D46、mean/median/resultant/gamma/RMS/残差幅度、20步训练、混淆、量化、资源及全部artifact SHA/大小，并解释表现与缺陷。D51结束后D49–D51满三轮，启动任何D52前必须完成强制复盘。

## 6.文件与执行占位

- 计划实现：`code/scripts/probe_d51_resultant_median_centroid_residual.py`；测试：`tests/test_probe_d51_resultant_median_centroid_residual.py`；追踪：`analysis/d51_resultant_median_centroid_residual_traceability_20260719.md`。
- Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`；根目录不是可用Git仓库，根报告仅作运行镜像。
- 代码提交、clean worktree、SHA、exact command和输出在首次运行前补锁。

## 7.实现与预运行验证

D51以wrapper形式先完成D45 fit，再从传入的正式support transformed rows/targets重算mean、coordinate median、resultant、RMS和coefficient correction；audit持久化support、base state、全部中间几何和实际FP32 state。末端verifier从持久化support独立复算，并临时还原D45 audit调用既有D45 verifier，形成新增几何＋继承分区/权重/资源双层闭包。

本地验证：D51定向`9 passed`；D45＋D51联合`20 passed`；D42–D51全链`161 passed`；`py_compile`通过且退出码均0。代码复核P0=0、P1=0：K1/K2 correction严格为0；rank置换不变、class置换等变；非unit-sphere、unequal K、非有限/退化norm/RMS均fail-close；K8额外数值上界`831,296`MAC-equivalent、coordinate-median比较上界`117,504`，不新增fit/step/state。outer尚未运行。

## 8.执行锁与exact command

- 实现提交：`a0bbb75cbe0f6132e808fc7600816c8b4d0ff75b`；clean detached worktree`E:\type10-7\code\snapshots\d51wt`，状态仅`## HEAD (no branch)`；探针SHA256`bd35a3fa4f2614b47c4afda08fe8b90e8dcb1a768d1858bd782021a9c0d2de80`。
- 历史runtime`E:\type10-7\code\snapshots\d41wt`只读bootstrap source closure通过；六个输入seal/envelope/manifest/binding继续使用已核验SHA`53ace286…d9f75`、`c70aedf3…b50ff`、`31a2ad99…ceb0e`、`a2483d6e…be76`、`15b5e144…629c`、`bb89a1db…c901f`。
- 输出`E:\type10-7\automation_reports\CV-SincNet\d51_resultant_median_centroid_residual_probe_20260719\resultant_median_centroid_residual`启动前不存在。本地串行`device=auto`，不访问N607、不运行125。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d51wt\code\scripts\probe_d51_resultant_median_centroid_residual.py' `
  --d51-arm resultant_median_centroid_residual `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d51wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d51_resultant_median_centroid_residual_probe_20260719\resultant_median_centroid_residual' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 9.attempt0资源字段失败与修复边界

首次执行在第一个fold的D45 top-level fit完成后、写入任何性能行前，由D51资源wrapper读取不存在的`resource["coefficient_dimension"]`而exit1，wall`34.958s`。输出目录已创建但无成功artifact，原样保留为`resultant_median_centroid_residual`；它不是性能失败、不能计作105行结果。

修复只把feature dimension来源改为实际formal state的`len(state.log_diag_fp32)`并要求一维正长度；正式D42 state固定得到288。该修复不改变support、几何公式、系数、量化、候选、fold或任何性能路径。新增回归直接验证实际state取维和错误shape fail-close；成功复跑只允许写新目录`resultant_median_centroid_residual_retry1`。

修复后D51＋D45联合`21 passed`，D42–D51全链`162 passed`，`py_compile`和`git diff --check`通过；代码复核确认修复不进入任何分数计算。

## 10.retry1执行锁

- 修复提交`f82cb192`；clean worktree`E:\type10-7\code\snapshots\d51retry1wt`，探针SHA256`12bce4b5a6380e78b4c8807a09db8329fe2e0efa30a6f309d9319d5b79fb7b34`。
- retry1输出启动前不存在。第8节命令仅把`--probe-root`和探针路径切换到`d51retry1wt`，把`--output`切换到`resultant_median_centroid_residual_retry1`；其他输入、SHA、runtime、mode、candidate-set与device完全不变。

## 11.retry1完成状态

- retry1完成105/105行、exit0；wall`85.477s`，receipt elapsed`73.814s`；7候选各15行。query0、source closure、support/query disjoint、ground int8、末端几何/D45双层verifier和artifact SHA全部通过。
- 未访问N607，未运行125。attempt0失败目录和retry1成功目录均保留。
- runner状态`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；按预注册门复核后的最终状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

## 12.七候选总体性能

H为15个matched row内H的均值；`min-*`为逐类跨15行均值的最小值；混淆为`old→new/new→old/new→new`。

|Candidate|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|Z0_SUPPORT_ONLY|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|identity fallback|
|D42-PROTOnet-CDA-ZID160|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|与Z0同指标|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|诊断比较器|
|D42-D40-HNBR-INT8-NEGATIVE|85.56%|85.00%|15.33%|25.16%|0.56pp|0.00%|66.67%|63.33%|0.00%|2/0/0|保旧、新类崩溃|
|D42-D41-BEC-INT8-NEGATIVE|86.11%|20.56%|78.67%|31.50%|65.56pp|0.00%|76.67%|0.00%|36.67%|142/0/32|旧类灾难遗忘|
|D51-INT8|92.22%|82.22%|82.00%|81.16%|10.00pp|26.67%|83.33%|46.67%|70.00%|23/12/15|rain改善但总体new/floor恶化|
|D51-FP32-MATCHED|92.22%|82.22%|82.00%|81.16%|10.00pp|26.67%|83.33%|46.67%|70.00%|23/12/15|与int8完全一致|

## 13.分场景性能

|场景|before|after|new|H|forget|joint|min-after|min-new|混淆|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|leo_clear_weak|95.00%|88.33%|98.00%|92.53%|6.67pp|40.00%|60.00%|90.00%|5/1/0|new高，但before/after与old floor低于D45|
|leo_low_elev_weak|90.00%|78.33%|70.00%|72.51%|11.67pp|30.00%|50.00%|50.00%|8/6/9|old/new/H/遗忘均未通过|
|leo_rain_weak|91.67%|80.00%|78.00%|78.43%|11.67pp|10.00%|30.00%|70.00%|10/5/6|after较D45+3.33pp、forget-1.67pp，但new-2pp且仍未达门|
|总体|92.22%|82.22%|82.00%|81.16%|10.00pp|26.67%|46.67%|70.00%|23/12/15|场景交换伤害，不可晋级|

## 14.逐类性能

总体old为before→after；场景old为after；new为注册后准确率。O/N标签仅报告匿名顺序。

|角色|类|总体|clear|low-elev|rain|表现|
|---|---|---:|---:|---:|---:|---|
|old|O0/`1f33441e`|86.67→90.00%|100%|80%|90%|注册后恢复|
|old|O1/`33bbd165`|96.67→93.33%|90%|90%|100%|稳健|
|old|O2/`75aa6d50`|96.67→93.33%|90%|90%|100%|rain改善|
|old|O3/`8b02d999`|83.33→46.67%|60%|50%|30%|主要old floor失败类|
|old|O4/`a53ca128`|100.00→76.67%|90%|70%|70%|rain较D45改善10pp|
|old|O5/`f8dfc2ed`|90.00→93.33%|100%|90%|90%|稳定|
|new|N0/`09f80039`|70.00%|90%|50%|70%|floor类|
|new|N1/`1c2ad882`|86.67%|100%|80%|80%|低于D45|
|new|N2/`b8fbace5`|83.33%|100%|50%|100%|场景分裂|
|new|N3/`d3afb5d1`|86.67%|100%|90%|70%|rain下降|
|new|N4/`f608a348`|83.33%|100%|80%|70%|rain下降|

## 15.十五个outer行

floor为`before/after/new`，混淆为`old→new/new→old/new→new`。

|场景|fold|before|after|new|H|forget|joint|floor|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|0|91.67%|100.00%|90.00%|94.74%|-8.33pp|50%|50/100/50%|0/1/0|
|clear|1|91.67%|83.33%|100.00%|90.91%|8.33pp|0%|50/0/100%|1/0/0|
|clear|2|91.67%|75.00%|100.00%|85.71%|16.67pp|50%|50/50/100%|2/0/0|
|clear|3|100.00%|91.67%|100.00%|95.65%|8.33pp|50%|100/50/100%|1/0/0|
|clear|4|100.00%|91.67%|100.00%|95.65%|8.33pp|50%|100/50/100%|1/0/0|
|low-elev|0|91.67%|75.00%|80.00%|77.42%|16.67pp|50%|50/50/50%|3/1/1|
|low-elev|1|75.00%|58.33%|70.00%|63.64%|16.67pp|0%|50/50/0%|1/0/3|
|low-elev|2|91.67%|91.67%|50.00%|64.71%|0.00pp|0%|50/50/0%|0/2/3|
|low-elev|3|100.00%|91.67%|70.00%|79.38%|8.33pp|50%|100/50/50%|1/2/1|
|low-elev|4|91.67%|75.00%|80.00%|77.42%|16.67pp|50%|50/50/50%|3/1/1|
|rain|0|83.33%|83.33%|60.00%|69.77%|0.00pp|0%|50/50/0%|1/0/4|
|rain|1|100.00%|75.00%|90.00%|81.82%|25.00pp|0%|100/0/50%|3/1/0|
|rain|2|91.67%|83.33%|90.00%|86.54%|8.33pp|50%|50/50/50%|1/0/1|
|rain|3|91.67%|83.33%|80.00%|81.63%|8.33pp|0%|50/0/50%|2/2/0|
|rain|4|91.67%|75.00%|70.00%|72.41%|16.67pp|0%|50/50/0%|3/2/1|

## 16.相对D42/D45/D46

|版本|before|after|new|H|forget|joint|min-after|min-new|混淆|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D42 original|90.56%|81.67%|81.33%|80.63%|8.89pp|23.33%|50.00%|70.00%|26/10/18|forget较好|
|D45 global LOO|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|直接matched基线|
|D46 classwise LOO|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|53.33%|73.33%|25/8/15|当前最强合法开发点|
|D51|92.22%|82.22%|82.00%|81.16%|10.00pp|26.67%|46.67%|70.00%|23/12/15|rain修复伴随new/low-elev伤害|

D51相对D45改变11/15个outer SHA：总体after与forget不变，joint`+3.33pp`、before类floor`+3.33pp`，但new`-2.00pp`、H`-1.00pp`、min-after`-6.67pp`、new→old`+4`。相对D46改变12/15行，after`+0.56pp`、forget`-0.56pp`，但new`-2.67pp`、H`-1.17pp`、min-new`-3.33pp`。这是明确的交换伤害，不能按rain单项提升晋级。

## 17.几何残差行为

|阶段|量|min|mean|max|
|---|---|---:|---:|---:|
|before|resultant rho|0.8895|0.9386|0.9731|
|before|gamma=1-rho|0.0269|0.0614|0.1105|
|before|residual RMS|0.00835|0.00981|0.01118|
|before|direction L2|0.0628|0.1241|0.1916|
|before|correction L2|0.1682|0.8109|2.1593|
|final|resultant rho|0.8793|0.9399|0.9813|
|final|gamma=1-rho|0.0187|0.0601|0.1207|
|final|residual RMS|0.00928|0.01117|0.01391|
|final|direction L2|0.0442|0.1213|0.2651|
|final|correction L2|0.0790|0.7357|2.5084|

final correction L2均值在clear/low-elev/rain为`0.6435/0.8171/0.7466`。虽然gamma仅约0.06，但除以很小的residual RMS后，部分class correction L2达到2.51；这足以产生11/15行变化并修复rain部分旧类，也会放大low-elev/new方向误差。下一步不能扫描残差系数；应在复盘后研究不依赖全局小RMS放大的有界、协议内几何机制。

## 18.B20训练、量化与资源

B20与D45同一冻结训练轨迹：epoch1/5/10/15/20的support acc为`95.14/97.78/99.03/99.72/100.00%`，loss为`1.031996/0.415989/0.216143/0.142408/0.102685`；20个epoch完整、query rows始终0。完整逐epoch数据在summary中。

|项目|结果|判定|
|---|---:|---|
|FP32/int8 before/final argmax变化|0/0|通过|
|margin符号翻转|0|通过|
|support argmax变化|0/0|通过|
|int8最大score误差|min`3.929e-4`、mean`9.313e-4`、max`1.582e-3`|未改变决策|
|LDA fit/MAC|36/1,065,830,400|D45闭合|
|D51额外适配|831,296 MAC-equivalent|闭合|
|coordinate median比较上界|117,504|单列，不伪装MAC|
|总适配/query MAC|1,071,638,336/6,624|闭合|
|参数/state|2,016/8,583B|通过|
|epoch/step|20/20|通过|
|CUDA peak|22,886,912B|实测|
|query/role/quota/count/global assignment|全部0/false|通过|
|clean/source/dense graph|false/false/0B|通过|

## 19.Artifact清单

|文件|大小/B|SHA256|
|---|---:|---|
|D51_PROBE_METADATA.json|1,827|`058f72481198ea04c45825c1c27e0b137ac66113c755fc6f013116d953c6228f`|
|RECEIPT.json|4,844|`f7a0a7f2f78c046644c852a8fb01ead79eb98daf2b8d9cbacae82b47350f03b3`|
|geometry_audit.json|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|resource_audit.json|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|selection.json|2,990|`faa323767e8b6e65819210f5ea1a1944ffc73d4082dff9828e898010aff037f5`|
|support_audit.json|313,482|`a7168d797b38834f2cbdf7d5606faf68936f49ec54bb96acbffd91a567e32954`|
|training_log.jsonl|39,648,180|`69f032401f073c998ba38ce4a8f8f2dfa09aef0c9f670ee1cdae5734ca5fcdc6`|
|full_performance_summary.json|62,560|`1cd9d2473618d139191c4ad4edc12f7931e0902e24a8601f8b31e088143188ea`|

summary完整读取D51/D45/D46各105行，生成器为`code/scripts/summarize_d51_performance.py`。

## 20.晋级门、缺陷与停止动作

|门|结果|判定|
|---|---|---|
|相对D45改变≥1预测|11/15行|通过|
|总体/各场景after/new/H/joint/floor不退化|多项退化|失败|
|new/min-new≥84.67/73.33%|82.00/70.00%|失败|
|rain after≥78.33%、forget≤10pp|80.00%、11.67pp|forget失败|
|总体forget≤8.89pp|10.00pp|失败|
|混淆≤26/10/18|23/12/15|new→old失败|
|量化0/0/0|0/0/0|通过|

D51证明稳健centroid方向残差能真正跨越决策边界并改善rain old，但全局RMS归一将小方向差放大，造成low-elev和new交换伤害。停止本路线：不扫描残差系数、不clip、不加门控、不跑第二seed、不formalize、不运行125。当前最强仍为D46，但仍不promotable。

## 21.D49–D51三轮回顾与D52预注册

### 21.1重新核对目标和数据协议

本轮回顾已重新读取活动目标、`项目.md`，并以`ssr-gpu`重建项目对话索引（`indexed=1008`）。D49–D51均使用同一`VALIDATED_ONCE`的`p2_min_v1`数据胶囊、receiver20-1、seed713101、K10/new5、3场景×5 outer folds；方法改变没有触发数据重验证。三个版本都在同一run中同时报告注册前old、注册后old、seen-new、H、遗忘、逐类floor和混淆，没有把domain adaptation与new-class registration拆开择优。

协议审计再次确认：只读固定`leo_*_weak`接收IQ及其允许视图；support-only适配；query rows/features/labels、query truth/role、真实batch类数、class quota、global reassignment与query-dependent optimization均为0/false；clean/source访问为false；dense query graph为0B。D49–D51均为开发单元证据，不得外推为125确认结论。

### 21.2三轮联合性能账本

|版本|机制|before|after|new|H|forget|joint|min-after|min-new|混淆|相对结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D46|classwise LOO可靠度融合|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|53.33%|73.33%|25/8/15|当前最强合法开发点，但不promotable|
|D49|strict-nested全局cosine融合|91.11%|76.67%|72.67%|73.90%|14.44pp|20.00%|63.33%|40.00%|29/26/15|全局support CE在low/rain给出过大cosine权重，全面退化|
|D50|centered median classwise evidence|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|权重非平凡，但0/15预测变化，未产生新决策机制|
|D51|resultant-median centroid residual|92.22%|82.22%|82.00%|81.16%|10.00pp|26.67%|46.67%|70.00%|23/12/15|11/15行改变；rain old改善，但low-elev/new与old floor受损|

三轮没有任何版本同时提高注册后old与seen-new。D49显示支持集全局可靠度不是可靠的query泛化代理；D50显示在同一D45 head内继续变换权重难以跨越决策边界；D51首次证明稳健几何方向能改变预测并改善rain old，但以全局极小residual RMS作除数会把平均gamma约0.06放大为最高2.51的系数修正，导致交换伤害。

### 21.3成功经验、淘汰路线与剩余假设

保留的成功经验：

- D25的288维block-normalized表示和D42的full+block shrinkage LDA仍是稳定底座，不重建特征路线；
- D45/D46说明小规模、classwise、support-only证据可以保持old/new平衡，其中D46的新类与new floor仍是当前最佳；
- D51说明coordinate-median相对mean的方向包含rain场景可用信号，下一轮只保留该方向，不保留其全局RMS尺度。

明确淘汰：

- 不再使用全局support CE/cosine权重，不把D49的scene-level可靠度当query代理；
- 不再继续D50式同head权重重排；
- 不扫描D51残差系数、不做事后clip/场景门控、不使用query选择尺度；
- 不重跑相同失败式的第二seed，不进入formalization或125。

剩余可证伪假设：D51的稳健方向本身有用，失败来自尺度而非方向。D52仅检验一个预注册的、内禀有界的base-relative修正，不做超参数网格：

```text
u_c = coordinate_median_r(x_rc) - mean_r(x_rc)
v_c = u_c / max(||u_c||_2, eps)
gamma_c = 1 - ||mean_r(x_rc / ||x_rc||_2)||_2
s_c = ||W_D45,c - mean_j(W_D45,j)||_2
DeltaW_c = gamma_c * s_c * v_c
W_D52,c = W_D45,c + DeltaW_c
b_D52,c = b_D45,c
```

该式把每类修正范数严格限制为`gamma_c * s_c`，不再除以全局小RMS；K1/K2保持D45精确fallback；before/final使用同一预注册公式；最终仍量化为int8系数+FP16截距。D52只在相同开发单元运行一次，并完整报告总体、场景、逐类、15 folds、训练、混淆、量化、资源和相对D45/D46/D51的表现。只有同时保持或改善old/new联合指标与逐类floor，才允许讨论下一阶段。
