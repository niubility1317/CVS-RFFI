# D47正部锚定可靠度收缩探针报告

## 1.身份与目标

- 实验ID：`d47_anchored_reliability_shrinkage_probe_20260718`。
- 操作者：Codex`/root`。
- 当前状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景、5个outer physical-rank held折。
- query sealed；不访问confirmation seeds，不生成125结果；当前不访问N607。

D46证明类级support inner-LOO可靠度可以改变真实决策并提高seen-new与最低new，但类间估计噪声同时放大旧类遗忘。D47保持B20、full/3-block LDA、canonical gauge、support RMS、int8生命周期、数据capsule、outer folds和比较门不变，只把D46逐类log-odds向D45全局锚点作无可调超参的正部矩收缩。目标是在保留D46新类收益的同时恢复rain与总体旧类稳定性。

## 2.机制与统计含义边界

对匿名类`c`和inner fold`r`，从合法support-held交叉熵构造：

`d_c,r=CE_block,c,r-CE_full,c,r`，`dbar_c=mean_r(d_c,r)`，`s_c²=Var_r(d_c,r)`。

D46类观察log-odds为`z_c=K×dbar_c`，其within-class log-odds方差代理为`u_c=K²×(s_c²/K)=K×s_c²`。令`mu=mean_c(dbar_c)`、`zbar=mean_c(z_c)=K×mu`。D45使用所有`C×K`个inner held样本，因此其全局锚点必须是`z0=C×mu`；`z0`与`zbar`在`C!=K`时不同，必须分别持久化，禁止把`K×mu`误当D45。

类间异质性采用固定正部矩估计：

`tau²=max(0,Var_c(z_c)-mean_c(u_c))`。

若`tau²=0`，固定`a_c=0`；若`tau²>0,u_c=0`，固定`a_c=1`；其余`a_c=tau²/(tau²+u_c)`。最终：

`zpost_c=(1-a_c)×z0+a_c×z_c`，`w_full,c=sigmoid(zpost_c)`，`w_block,c=1-w_full,c`。

`tau²=0`时精确退回D45权重公式；`a_c=1`时该类精确到D46权重公式。这里是公式级端点，不预先宣称candidate state字节等价，真实state关系必须由同一运行实测。该构造只称`positive-part anchored reliability shrinkage`，审计声明固定为`eb_inspired_deterministic_shrinkage_not_calibrated_posterior`。由于类间样本少且inner LOO折重叠，本探针不得把它描述成校准后的经验贝叶斯posterior或不确定性区间。

## 3.协议边界与特殊K

公式不读取class ID、TX、old/new角色、receiver、handle、场景、outer-held或query；无temperature、clip、阈值或权重扫描。support label仅用于合法的support监督拟合和inner可靠度计算。每个query仍独立对全部注册类argmax，无truth、role Oracle、class quota或global reassignment。

K1固定1:1等价回退。K2只有full/block逐fold逐类CE在数值容差内完全相等时才允许1:1，否则fail closed。sigmoid若因极端log-odds在FP64舍入到0或1也fail closed，不以事后clip掩盖。before state必须在首次new support读取前物化且不可变。

## 4.资源口径

D47复用D46主体计算：B20为2016个trainable parameters、20 epoch、20 optimizer steps；LDA inventory在K>1时为`4K+4`，可靠度评分与类级仿射融合MAC沿用D46精确公式。D47对已经持久化的`C×K`标量证据计算矩和权重，不能把新增计算记为0。每个before/final state的保守MAC-equivalent上界拆为：`6KC`覆盖fold evidence和一/二阶矩，`16C+8`覆盖跨类矩及正部收缩，`8C+8`覆盖post-logit、sigmoid和端点检查；两state合计`0 if K1 else 6K(C_old+C_all)+24(C_old+C_all)+32`并计入`estimated_adaptation_macs`。该上界对任意整数`K>=2`锁定，K1不执行矩代数且为0。新增LDA fit=0、新增optimizer step=0、新增query state=0、query sidecar=0。最终仍只持久化一个int8/FP16 query state。host FP64 covariance peak继续标记未实测，不能以CUDA峰值替代。

## 5.预注册晋级门

先继承D42全部协议、lifecycle、source、ground、state、resource、artifact、聚合、floor、逐场景、forgetting、joint、量化和混淆门。D47还必须同时满足：

- 聚合seen-new和最低new不低于D46的`84.67%/73.33%`；
- rain after-old不低于D42的`78.33%`，rain forgetting不高于D42的`10.00pp`；
- 相对D46至少改变1个final held预测；
- before/final int8-FP32 argmax变化与margin翻转均为0。

若`tau²=0`导致D47退回D45、全部final预测与D46相同、任一通用门失败或量化翻转，D47直接记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；不得事后添加第二arm、temperature、clip或shrinkage扫描。即使所有门通过，本探针仍是强制identity、禁止full-K10 refit的开发探针，只能进入另行正式候选实现与封闭开发验证，不能直接生成125或宣称正式性能。

## 6.文件、版本与计划命令

- 探针：`code/scripts/probe_d47_anchored_reliability_shrinkage.py`。
- 共享helper最小扩展：`code/scripts/probe_d46_classwise_loo_reliability_fusion.py`仅增加可选策略回调，默认D46路径不变。
- 单测：`tests/test_probe_d47_anchored_reliability_shrinkage.py`及D42–D46回归。
- 追溯：`analysis/d47_anchored_reliability_shrinkage_traceability_20260718.md`。
- 预期输出：`E:\type10-7\automation_reports\CV-SincNet\d47_anchored_reliability_shrinkage_probe_20260718\anchored_reliability_shrinkage`。
- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，本地串行，device=`auto`。
- runtime：`E:\type10-7\code\snapshots\d41wt`；探针在本轮预注册提交的detached clean worktree运行。
- 输入：D18 receiver`20-1`/seed`713101`/K10-new5密封capsule及D22 component manifest、D19 class binding、D18 before/final enrollment seals。

根目录`E:\type10-7`不是Git仓库；代码、测试、追溯和正式报告进入`github_publish/CVS-RFFI-repo`，只暂存本轮精确文件；根目录只保留报告镜像。真实命令及所有输入hash将在预注册提交后写入本报告，输出目录必须预先不存在。

## 7.本地验证

- D47+D46定向测试：首轮`23 passed`；修复独立复核发现的2项P1、2项P2和1项P3后最终为`37 passed`，exit0。
- D42–D47继承链：修复前`90 passed`；最终`104 passed`，exit0。
- py_compile：通过。
- `C!=K`、complete-pooling D45公式权重端点、no-shrinkage D46公式权重端点、零异质性、手算部分收缩矩、标签置换、K1/K2完整链、稳定sigmoid、非零标量资源重算、K1/2/5/8/10/20上界常数、integrated fit/verifier和核心字段tamper拒绝均有测试。
- pytest结束后本机`pytest-current`出现既知`WinError 5`临时目录清理噪声，不影响测试退出码和结论。

## 8.执行与闭包

- 预注册提交：`07b6baecf77bb48e48d008b21dd9f4d683354ea2`。
- detached clean worktree：`E:\type10-7\code\snapshots\d47wt`，运行时HEAD同上；runtime为`E:\type10-7\code\snapshots\d41wt`。
- Python：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`；device=`auto`；本地串行。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d47_anchored_reliability_shrinkage_probe_20260718\anchored_reliability_shrinkage`。
- 完成：105/105行；receipt elapsed`73.7794s`，外部命令wall time`80.7s`；query0；formal/performance claim均为false；N607未访问。
- D47 metadata通过30条int8/FP32 fit row、source helper hash、D43外层probe和105行总闭包。

启动前只读检查发现clean worktree内历史D19 binding的SHA为`39cb…`，不等于锁定`bb89…`；因此没有启动或创建输出，改用主Git承载面中D42–D46已实际验证的`analysis/d19_adv3b02_class_binding_20260717.json`，其SHA精确为`bb89a1db…c901f`。其余before/after seal、authorization envelope和component manifest分别精确匹配`53ace…`、`31a2…`、`c70aed…`、`a2483…`和`15b5e…`。

执行入口为：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe `
  E:\type10-7\code\snapshots\d47wt\code\scripts\probe_d47_anchored_reliability_shrinkage.py `
  --d47-arm anchored_reliability_shrinkage `
  --runtime-root E:\type10-7\code\snapshots\d41wt `
  --probe-root E:\type10-7\code\snapshots\d47wt `
  <D18锁定before/after root、seal、policy、authorization、envelope参数> `
  --component-dir E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component `
  --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json `
  --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output E:\type10-7\automation_reports\CV-SincNet\d47_anchored_reliability_shrinkage_probe_20260718\anchored_reliability_shrinkage `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

完整D18参数与第6节列出的输入一起进入本地命令记录；没有省略或改变policy/authorization/envelope。

## 9.同row候选结果

|Candidate|机制/精度|before-old|after-old|seen-new|H|forgetting|joint|min before|min after|min new|old→new/new→old/new-new|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D42-USLDA-INT8|D47正部锚定可靠度收缩/int8|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|80.00%|53.33%|70.00%|24/8/16|负面，不晋级|
|D42-USLDA-FP32-MATCHED|同一D47解/FP32|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|80.00%|53.33%|70.00%|24/8/16|matched ablation|
|D42-D40-HNBR-INT8-NEGATIVE|old-heavy HNBR/int8|85.56%|85.00%|15.33%|25.16%|0.56pp|0%|66.67%|63.33%|0%|2/N/A/N/A|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|new-heavy BEC/int8|86.11%|20.56%|78.67%|31.50%|65.56pp|0%|76.67%|0%|36.67%|142/0/32|旧类崩溃|
|B3_SINGLE_IQ_DIAG_FFTRF|单IQ B3比较器|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|弱比较器|
|D42-PROTOnet-CDA-ZID160|ProtoNet CDA|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|N/A|负面|
|Z0_SUPPORT_ONLY|identity/support-only control|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|N/A|control|

固定TX切分为6 old＋5 new，receiver`20-1`、seed`713101`、K10 capsule、3场景、5折；每个outer fit实际K8。表中H为15个matched row内`H_old_new`的算术均值，不是pooled-H。unknown、coverage、rollback和defer不属于本support-only闭集development Runner，记为N/A。

## 10.基准、场景与预注册门

|版本|before-old|after-old|seen-new|H|forgetting|joint|min after|min new|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|D42 original|90.56%|81.67%|81.33%|80.63%|8.89pp|23.33%|50.00%|70.00%|26/10/18|
|D45 global LOO|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|
|D46 classwise LOO|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|53.33%|73.33%|25/8/15|
|D47 anchored shrinkage|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|

|场景|before-old|after-old|seen-new|H|forgetting|joint|min after|min new|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|98.33%|90.00%|98.00%|93.57%|8.33pp|40.00%|70.00%|90.00%|4/1/0|
|low-elev|88.33%|80.00%|74.00%|75.45%|8.33pp|20.00%|60.00%|40.00%|7/5/8|
|rain|90.00%|76.67%|80.00%|77.45%|13.33pp|10.00%|30.00%|70.00%|13/2/8|

D47相对D42的before/after/new/H、joint和聚合floors均不差，聚合混淆也更少；但aggregate forgetting`10.00pp>8.89pp`，low-elev最低new`40.00%<50.00%`且new→old`5>4`，rain after-old`76.67%<78.33%`、forgetting`13.33pp>10.00pp`且old→new`13>12`。D47附加门也失败：seen-new`84.00%<84.67%`、最低new`70.00%<73.33%`，没有保住D46的新类收益。

D47相对D46有2/15个outer prediction SHA和2/330个final argmax变化，均在low-elev；相对D45则15/15个outer SHA和330/330个final argmax完全相同，before prediction SHA也15/15相同。D47不是数值上没有运行：before的12/15个fit、final的9/15个fit有正`tau²`；只是收缩后的边界决策和全部同row指标精确回到D45。

## 11.全部匿名类×场景结果

类名按opaque class handle排序后匿名化为O0–O5和N0–N4，仅用于完整报告，不参与方法或调参。

|场景|类|before|after/new|
|---|---|---:|---:|
|clear|O0/O1/O2/O3/O4/O5|100/90/100/100/100/100%|100/90/90/70/90/100%|
|clear|N0/N1/N2/N3/N4|—|100/100/90/100/100%|
|low-elev|O0/O1/O2/O3/O4/O5|80/100/90/80/100/80%|80/90/90/60/70/90%|
|low-elev|N0/N1/N2/N3/N4|—|40/100/50/90/90%|
|rain|O0/O1/O2/O3/O4/O5|90/100/100/60/100/90%|90/100/90/30/60/90%|
|rain|N0/N1/N2/N3/N4|—|70/80/90/80/80%|

聚合O0–O5 before为`90.00/96.67/96.67/80.00/100.00/90.00%`，after为`90.00/93.33/90.00/53.33/73.33/93.33%`；N0–N4为`70.00/93.33/76.67/90.00/90.00%`。最主要的旧类失效仍是rain O3从60%降到30%，最弱新类是low-elev N0的40%。

## 12.收缩、量化与资源

- before：12/15个fit为正`tau²`、3/15 complete pooling；`tau²`范围/均值为`0–0.312341/0.059228`，`a_c`范围/均值为`0–0.977662/0.457696`，`w_full`范围/均值为`0.338733–0.650126/0.437130`。
- final：9/15个fit为正`tau²`、6/15 complete pooling；`tau²`范围/均值为`0–0.085899/0.023342`，`a_c`范围/均值为`0–0.945993/0.290548`，`w_full`范围/均值为`0.432920–0.581348/0.509008`。
- `max|z0-zbar|`在before/final分别为`0.099701/0.087853`，证明`C×mu`与`K×mu`被正确分离。
- before/final/margin量化变化为`0/0/0`，max score error`0.0016140938`；int8与matched FP32同row指标完全一致。
- trainable parameters`2016`；20 epoch/20 optimizer steps；persistent state`8583B`；query MAC`6624`；CUDA peak`22,886,912B`。
- LDA fit`36`次；LDA MAC`1,065,830,400`；metric MAC`4,976,640`；D46可靠度评分MAC`6,511,104`；类级融合MAC`9,826`；D47标量保守MAC-equivalent`1,256`；总adaptation`1,077,329,226`。host FP64 covariance peak未实测。
- 300条int8 B20 trace全部finite，epoch/step完整覆盖1–20，trace query rows为0。

## 13.artifact闭包

|Artifact|Bytes|SHA256|
|---|---:|---|
|training_log.jsonl|4,965,477|`bc27d35b1655e3d2af9378e93a1d13b425b2a41a6fe0cb8aed0d9b5dff3fdbff`|
|support_audit.json|313,579|`fee9026be0ba0c166babd8ef4845d3fbeb8b2ab4de84712800c4881bbfcbe4be`|
|selection.json|2,990|`b2e20b59e3767e117d5e7854bfaf27d68794da181eb398709e82575b49d638a8`|
|RECEIPT.json|4,940|`1044e7fe176a57529c8c2d9e6f09f7553f30b658f019750b160cd6b7a43ef044`|
|D47_PROBE_METADATA.json|2,454|`93bd19dc621d3d96c7f384a34b04d28090a98b382f975469d257809b0c823653`|
|geometry_audit.json|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|resource_audit.json|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|

## 14.判定与下一轮

D47为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。它证明无超参的正部锚定收缩能把D46的两条不稳定边界恢复到D45，但恢复方式同时消除了D46的seen-new/min-new增益，且rain旧类失效不受影响。D47不正式化、不生成125、不访问N607，也不添加第二收缩arm。

下一轮D48应避免继续对full/block权重作全局或类级统计平滑；D42–D47已经表明该轴只能在D45旧类与D46新类之间移动，无法触及rain O3这一共同失败。D48转向support-only、类置换等变的决策几何修复，并在完成后按三轮节奏执行D46–D48强制回顾。
