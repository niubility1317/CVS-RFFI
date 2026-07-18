# D48一次性OOF-head margin残差探针报告

## 1.身份与目标

- 实验ID：`d48_one_shot_oof_margin_residual_probe_20260718`。
- 操作者：Codex`/root`。
- 当前状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景、5个outer physical-rank held折；每个outer fit实际K8。
- query sealed；不访问confirmation seeds，不生成125结果；当前不访问N607。

D45与D47最终决策完全相同，D46只在low-elev改变2条final argmax；三者共同保留rain O3旧类失效，说明继续平滑full/block权重无法触及主瓶颈。D48回到D45稳定的global LOO融合state，只增加一组由合法support inner-held margin统一生成的一次性class intercept residual，目标是同时修复低margin旧类和新类，而不使用类ID或old/new角色。

## 2.锁定算法顺序

完整继承D45 frozen-outer-B20 head-only LOO：full与3-block组件先canonical，各inner-train fold分别计算RMS；D45 global weight只由组件inner-LOO class-balanced CE计算。对inner fold`r`和匿名true类`c`：

`q_c,r,j=w_full×delta_full,c,r,j/s_full,r+w_block×delta_block,c,r,j/s_block,r`。

这里必须使用inner-train组件RMS，不能混用完整support RMS。随后：

`margin_c,r=q_c,r,c-max_{j!=c}(q_c,r,j)`；

`m_c=mean_r(margin_c,r)`，`mbar=mean_c(m_c)`；

`beta_raw,c=mbar-m_c`，`beta_c=beta_raw,c-mean_c(beta_raw)`。

完整support D45 affine state仍以完整support组件RMS和同一个global weight合成。只把`beta_c`一次性加入其intercept，再做canonical class-centering并进入既有residual-int8 coefficient/FP16 intercept编译。coefficient必须逐bit不变。禁止beta回流RMS、weight、margin、LDA或B20，禁止第二次beta或迭代到收敛。

## 3.协议与声明边界

support label用于选择true logit并排除true类后的`max_other`，属于合法support监督。每类使用同一mean/zero-sum公式，标签或support rank置换时beta和prediction列同步置换。无class ID表、old/new角色、receiver、scene、handle、outer-held、query、temperature、clip、threshold或扫描。max-other并列只使用相同最大值，不按class ID改变算法，tie仅审计计数。

本方法称`support-supervised one-shot OOF-head margin residual`。global weight与beta复用同一OOF support标签，且outer B20看过完整outer-fit support；这在协议内合法，但不是独立校准集，也没有无泄漏或泛化保证。support margin改善只是训练证据，不能替代outer-held性能门。

K1无inner fold，beta严格为0并逐bit回退D45 unit fallback。K2仍使用同一mean公式，不添加shrink或median；D45 unit components和global 1:1必须在`1e-12`内闭合，否则fail closed。C<2、非有限score/margin/beta、RMS或weight漂移、partition非exact-once、FP16溢出均fail closed。

## 4.资源口径

D48复用D45的B20、`4K+4` LDA inventory和一个query state，不新增fit、optimizer step、query state或sidecar，persistent state与query MAC不变。新增资源包括：

- full/block inner held component scoring：与D46同一精确公式；
- 完整support affine fusion：`2(D+1)(C_old+C_all)`；
- OOF margin residual保守MAC-equivalent上界：K1为0，K>1为`4K(C_old²+C_all²)+8K(C_old+C_all)+16(C_old+C_all)+32`。

当前实际K8/old6/all11的新增margin上界为6416；预计总adaptation为`1,077,334,386`。full/block/fused held logits与margin派生量的adaptation-time peak numeric evidence为26376B；持久化fit-audit改为对实际before/final证据和形式化量化数组执行canonical compact JSON UTF-8序列化后精确计数，真实字节数将在实验结果中报告，不再使用错误的numeric payload上界。它们只属于support训练审计，不是query sidecar，也不计入formal state。最终state仍为2016 trainable parameters、20 epoch/20 optimizer step、8583B、6624 query MAC；host FP64 covariance peak必须保留未实测状态。

## 5.预注册性能门

必须同时满足D42全部协议/lifecycle/source/ground/state/resource/artifact、聚合、三场景、最低before/after/new、joint、forgetting、混淆`26/10/18`和量化`0/0/0`门；D45 seen-new`84.00%`与matched-row H`82.16%`不得退化；全局min-new不得低于D46的`73.33%`；rain after-old/forgetting至少达到D42的`78.33%/10.00pp`；low-elev min-new至少达到D42的50%。

此外，至少一个真实fit的beta必须非全0，且final outer prediction相对D45至少改变1条。若全部预测不变、beta全0、support margin改善但outer门失败，均记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。不允许事后换median、缩放beta、clip、迭代或加第二arm。即使全部通过，本探针也必须另行正式化和封闭开发验证，不能直接生成125。

## 6.文件、版本与验证

- 探针：`code/scripts/probe_d48_one_shot_oof_margin_residual.py`。
- D45 helper最小扩展：可选private held-score collector和post-fusion calibration callback；默认关闭，历史D45路径不变。
- 单测：`tests/test_probe_d48_one_shot_oof_margin_residual.py`及D42–D47回归。
- 追溯：`analysis/d48_one_shot_oof_margin_residual_traceability_20260718.md`。
- 预期输出：`E:\type10-7\automation_reports\CV-SincNet\d48_one_shot_oof_margin_residual_probe_20260718\one_shot_oof_head_margin_residual`。
- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，本地串行，device=`auto`；runtime=`E:\type10-7\code\snapshots\d41wt`。

当前本地验证：首轮D48＋D45定向`31 passed`、D42–D48全链`124 passed`；修复独立代码初审发现的2项P1和3项P2后为定向`35 passed`、全链`128 passed`；进一步闭合formal int8 coefficient实际数组/fit FP32重编译绑定和真实JSON UTF-8计数后，定向`37 passed`、全链`130 passed`，py_compile通过。第一次`conda activate`落到base Python并因无pytest退出，随后用确认的`ssr-gpu`解释器串行重跑成功，未把包装噪声计为项目失败。设计审计确认协议P0=0并锁定同score单位、one-shot无回流、mean非median和声明边界；最终独立代码复审确认P0=0、P1=0、P2=0，且HEAD与当前D45默认路径在同一K1/K5输入上的coef、intercept与完整audit canonical JSON完全一致。

根目录`E:\type10-7`不是Git仓库；代码、测试、追溯和正式报告进入`github_publish/CVS-RFFI-repo`，只暂存本轮精确文件；根目录保留报告镜像。预注册提交、detached clean worktree、真实105行和完整日志判定待完成。

## 7.执行预检与锁定命令

- 预注册提交：`5bb494d747797ef6557531287b2236a10d1f2798`。
- detached clean worktree：`E:\type10-7\code\snapshots\d48wt`，HEAD与预注册提交一致，`git status --porcelain`为空。
- runtime：`E:\type10-7\code\snapshots\d41wt`；Python：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`；本地串行，device=`auto`；N607不访问。
- 输出目录在启动前不存在：`E:\type10-7\automation_reports\CV-SincNet\d48_one_shot_oof_margin_residual_probe_20260718\one_shot_oof_head_margin_residual`。
- 输入SHA：before seal`53ace286…d9f75`、after seal`c70aedf3…b50ff`、before envelope`31a2ad99…ceb0e`、after envelope`a2483d6e…be76`、component manifest`15b5e144…629c`、主Git承载面D19 class binding`bb89a1db…c901f`；formal policy、before authorization、after authorization分别为`1f347d7c…fc2be`、`e7880cf8…549ed`、`03f9396c…1a70`。

锁定执行命令为：

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d48wt\code\scripts\probe_d48_one_shot_oof_margin_residual.py' `
  --d48-arm one_shot_oof_head_margin_residual `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d48wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d48_one_shot_oof_margin_residual_probe_20260718\one_shot_oof_head_margin_residual' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 8.首次执行失败与直接修复

首次命令wall time`84.1275s`，底层runner完成并密封105/105行，receipt elapsed`74.2176s`、query0，但D48后置verifier在`D48 coefficient/intercept residual drift`处fail closed，进程exit1，因此本次不构成完成的D48性能实验，也不据此晋级或淘汰方法。失败输出保留且不覆盖：`one_shot_oof_head_margin_residual`；其中training log为11910002B，未生成D48 metadata。

只读全日志定位显示：30条D48 fit audit中仅6条low-elev记录触发，所有coefficient/intercept SHA、shape、finite、coefficient bitwise unchanged和`delta_fp32==final_fp32-base_fp32`均通过；唯一失败是逻辑beta与两个独立FP32截距舍入后差值的固定`atol=2e-7`，最大误差`2.3576668e-7`。直接修复仅把该编译闭合改为逐元素FP32 ULP舍入包络：`0.5×(ulp(base)+ulp(final)+ulp(delta))+centering residual+64eps×magnitude`；逻辑beta仍由完整OOF证据逐元素重算，delta仍须与formal FP32截距差逐bit一致。算法、support、权重、beta、预测、资源和性能门均不变。修复后必须新增舍入边界反例、重跑定向与全链测试、提交新代码，并使用新输出目录`one_shot_oof_head_margin_residual_retry1`，不得覆盖失败artifact。

修复后py_compile通过，D48＋D45定向`38 passed`，D42–D48全链`131 passed`；使用修复版fit verifier只读复算失败artifact的完整105行，30/30条fit audit通过。该只读复算只证明直接数值根因已修复，不补写metadata，也不把首次失败转为完成实验；retry1仍必须从提交后的新clean worktree完整重跑。独立ULP复核确认P0=0、P1=0、P2=0，并独立复跑D48`27 passed`；上界只覆盖三个0.5 ULP、centering残差和64eps项，delta逐bit与SHA绑定不变。

## 9.retry1执行锁

- 修复提交：`c6db747cad30160813e7f9b8c98f30cd98a103fa`。
- 新detached clean worktree：`E:\type10-7\code\snapshots\d48retry1wt`，HEAD与修复提交一致，工作树干净。
- 新输出：`E:\type10-7\automation_reports\CV-SincNet\d48_one_shot_oof_margin_residual_probe_20260718\one_shot_oof_head_margin_residual_retry1`，启动前不存在。
- 第7节锁定命令只允许两处路径替换：探针脚本和`--probe-root`由`d48wt`改为`d48retry1wt`，`--output`增加`_retry1`；所有D18/D19/D22输入、SHA、runtime、arm、device、mode和candidate-set逐项不变。

## 10.retry1执行闭包

- 完成：105/105行；receipt elapsed`73.3134s`，外部wall time`81.6456s`，exit0。
- receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer folds；每个outer fit实际K8。
- query0；formal/performance claim均为false；30/30条D48 fit audit、30/30条继承D43 audit、source closure、ground int8 entry/exit和artifact SHA全部通过。
- 运行设备`cuda:0`；N607未访问；未生成125；failed attempt与retry1两个目录均保留。
- 最终判定：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D48满足协议、实现作用、量化和资源边界，但显著破坏outer-held性能，不能进入125或正式确认。

## 11.全部同row候选性能

|Candidate|机制/精度|before-old|after-old|seen-new|matched-row H|forgetting|joint|min before|min after|min new|old→new/new→old/new→new|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D42-USLDA-INT8|D48一次性OOF margin residual/int8|74.44%|57.78%|56.67%|56.23%|16.67pp|13.33%|66.67%|40.00%|30.00%|39/31/34|显著负面，不晋级|
|D42-USLDA-FP32-MATCHED|同一D48解/FP32|74.44%|57.78%|56.67%|56.23%|16.67pp|13.33%|66.67%|40.00%|30.00%|39/31/34|matched ablation|
|D42-D40-HNBR-INT8-NEGATIVE|old-heavy HNBR/int8|85.56%|85.00%|15.33%|25.16%|0.56pp|0%|66.67%|63.33%|0%|2/0/0|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|new-heavy BEC/int8|86.11%|20.56%|78.67%|31.50%|65.56pp|0%|76.67%|0%|36.67%|142/0/32|旧类崩溃|
|B3_SINGLE_IQ_DIAG_FFTRF|单IQ B3比较器|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|弱比较器|
|D42-PROTOnet-CDA-ZID160|ProtoNet CDA|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|N/A|负面|
|Z0_SUPPORT_ONLY|identity/support-only control|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|N/A|control|

表中H是15个matched row的`H_old_new`算术均值，不是pooled-H；所有min是同一候选的逐类跨15行均值下限。D48相对direct ADV3B02的before-old平均适应增益仍为`+8.89pp`，但只在10/15 outer rows非负，最差row为`-41.67pp`；clear/low/rain增益分别为`+18.33/+21.67/-13.33pp`，因此不能用总体正均值掩盖rain反向适应。

## 12.D48三场景表现

|场景|before-old|after-old|seen-new|H|forgetting|joint|min before|min after|min new|old→new/new→old/new→new|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|85.00%|78.33%|80.00%|78.14%|6.67pp|40.00%|70.00%|60.00%|70.00%|4/8/2|
|low-elev|78.33%|45.00%|40.00%|41.16%|33.33pp|0%|50.00%|20.00%|0%|29/4/26|
|rain|60.00%|50.00%|50.00%|49.40%|10.00pp|0%|40.00%|30.00%|20.00%|6/19/6|

相对D45同场景，D48在clear的before/after/new/H分别下降`13.33/11.67/18.00/15.44pp`；low-elev下降`10.00/35.00/34.00/34.29pp`且forgetting增加25pp；rain下降`30.00/26.67/30.00/28.05pp`。clear的forgetting改善1.67pp、rain改善3.33pp只是因为before state本身被beta先破坏，不能解释成稳定性提升。low-elev出现29个old→new和26个new→new混淆；rain出现19个new→old，说明同一个截距残差同时放大两侧错误，而非只偏向旧类或新类。

## 13.D48全部15个outer row

|场景|fold|before|after|new|H|forget|joint|min-after|min-new|old→new/new→old/new→new|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|0|75.00%|91.67%|80.00%|85.44%|-16.67pp|50.00%|50.00%|50.00%|0/2/0|
|clear|1|91.67%|66.67%|90.00%|76.60%|25.00pp|50.00%|50.00%|50.00%|2/0/1|
|clear|2|75.00%|66.67%|60.00%|63.16%|8.33pp|0%|0%|0%|1/4/0|
|clear|3|83.33%|66.67%|90.00%|76.60%|16.67pp|50.00%|50.00%|50.00%|1/1/0|
|clear|4|100.00%|100.00%|80.00%|88.89%|0pp|50.00%|100.00%|50.00%|0/1/1|
|low-elev|0|83.33%|50.00%|50.00%|50.00%|33.33pp|0%|50.00%|0%|5/0/5|
|low-elev|1|66.67%|33.33%|20.00%|25.00%|33.33pp|0%|0%|0%|8/0/8|
|low-elev|2|91.67%|66.67%|40.00%|50.00%|25.00pp|0%|50.00%|0%|3/2/4|
|low-elev|3|83.33%|41.67%|40.00%|40.82%|41.67pp|0%|0%|0%|7/1/5|
|low-elev|4|66.67%|33.33%|50.00%|40.00%|33.33pp|0%|0%|0%|6/1/4|
|rain|0|66.67%|50.00%|40.00%|44.44%|16.67pp|0%|0%|0%|0/5/1|
|rain|1|58.33%|58.33%|50.00%|53.85%|0pp|0%|0%|0%|1/3/2|
|rain|2|58.33%|41.67%|60.00%|49.18%|16.67pp|0%|0%|50.00%|4/2/2|
|rain|3|75.00%|66.67%|60.00%|63.16%|8.33pp|0%|0%|50.00%|1/3/1|
|rain|4|41.67%|33.33%|40.00%|36.36%|8.33pp|0%|0%|0%|0/6/0|

只有clear fold0达到after-old>90%，只有clear folds1/3达到new=90%；没有任何row同时接近项目总体门槛，10/15 rows的joint floor为0。

## 14.逐类性能与下尾

旧类handle使用完整匿名SHA的前8位；完整值保存在training log和class binding中。

|old handle|D48 before|D45 after|D48 after|Δvs D45|clear after|low after|rain after|
|---|---:|---:|---:|---:|---:|---:|---:|
|1f33441e|66.67%|90.00%|60.00%|-30.00pp|70.00%|60.00%|50.00%|
|33bbd165|73.33%|93.33%|60.00%|-33.33pp|80.00%|50.00%|50.00%|
|75aa6d50|66.67%|90.00%|60.00%|-30.00pp|90.00%|50.00%|40.00%|
|8b02d999|93.33%|53.33%|76.67%|+23.33pp|90.00%|50.00%|90.00%|
|a53ca128|70.00%|73.33%|50.00%|-23.33pp|80.00%|40.00%|30.00%|
|f8dfc2ed|76.67%|93.33%|40.00%|-53.33pp|60.00%|20.00%|40.00%|

|new handle|D45 new|D48 new|Δvs D45|clear|low|rain|
|---|---:|---:|---:|---:|---:|---:|
|09f80039|70.00%|80.00%|+10.00pp|80.00%|90.00%|70.00%|
|1c2ad882|93.33%|50.00%|-43.33pp|90.00%|10.00%|50.00%|
|b8fbace5|76.67%|30.00%|-46.67pp|70.00%|0%|20.00%|
|d3afb5d1|90.00%|60.00%|-30.00pp|70.00%|50.00%|60.00%|
|f608a348|90.00%|63.33%|-26.67pp|90.00%|50.00%|50.00%|

D48只改善1个旧类和1个新类，却使其余9类全部下降；最低旧类由D45的53.33%降至40%，最低新类由70%降至30%。这证明残差不是通用floor修复，而是把支持集难度重新分配成更尖锐的类间截距偏置。

## 15.与D42–D47同协议路线比较

|版本|before|after|new|H|forget|joint|min-after|min-new|混淆总计|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|D42|90.56%|81.67%|81.33%|80.63%|8.89pp|23.33%|50.00%|70.00%|26/10/18|
|D45|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|
|D46|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|53.33%|73.33%|25/8/15|
|D47|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|
|D48|74.44%|57.78%|56.67%|56.23%|16.67pp|13.33%|40.00%|30.00%|39/31/34|

相对D45，D48的before/after/new/H分别下降`17.78/24.44/27.33/25.92pp`，forgetting恶化6.67pp，joint下降10pp，三类混淆从`24/8/16`增至`39/31/34`。因此D46仍是当前最强合法development点：new、H和min-new略高于D45/D47，但其after-old81.67%、min-old53.33%仍远低于项目要求，不能称为 promotable。

## 16.机制作用与失败原因

D48确实执行而非退化回D45：before＋final共30个fit的beta全部非零，值域`[-1.5916,+1.0564]`，平均绝对值`0.3483`，单fit最大mean-abs`0.4893`；D45 full weight只在`[0.4258,0.5798]`变化。相对D45，15/15 outer prediction SHA均变化，330个held样本中135个final argmax改变：98个正确→错误、13个错误→正确、24个错误→另一错误，净损失85个正确预测。变化分布为clear23、low-elev58、rain54；其中old75、new60，说明失败同时覆盖Stage2-B和Stage2-C。

按场景，clear为18个正确→错误、2个错误→正确；low-elev为43/5；rain为37/6。根因是`beta=mbar-m_c`把support OOF低margin直接当作完整幅度的可迁移截距误差，没有任何统一收缩或相对外层margin尺度约束。其幅度与outer决策margin同量级甚至更大，导致难类获得大正偏置、易类获得大负偏置；support OOF难度在不同outer物理样本和LEO场景中不稳定，因而只修复2类，却破坏9类。该解释来自support-held与outer-held同row证据，不使用query、类ID或角色Oracle。

## 17.完整训练轨迹、量化与资源

15个D48 int8 outer rows均有20/20 B20 trace。epoch1平均loss`1.0320`、support accuracy`95.14%`；epoch8为`0.2610/98.89%`；epoch14为`0.1527/99.86%`；epoch20为`0.1027/100%`，最终prototype-anchor loss均值`0.003828`、gradient norm均值`0.1354`。完整support拟合持续改善而outer性能大幅下降，再次表明support拟合不能作为held泛化替代。

|资源项|D48实际值|判定|
|---|---:|---|
|trainable parameters|2016|通过80k上限|
|adaptation epochs/optimizer steps|20/20|通过30/50上限|
|persistent formal state|8583B|通过256KB上限|
|query MAC|6624|单样本、全注册类|
|metric/B20 MAC|4,976,640|已计入|
|36次LDA fit MAC|1,065,830,400|D45继承|
|component scoring MAC|6,511,104|D48新增|
|affine fusion/margin上界|9826/6416|D48新增|
|总adaptation MAC-equivalent|1,077,334,386|完整计数|
|CUDA peak|22,886,912B|仅metric-fit allocation scope|
|support审计peak numeric|26,376B|非formal state|
|持久化fit-audit JSON|min227,942B/mean228,424B/max229,009B|非query sidecar|
|host FP64 covariance peak|未实测|不得补写|

int8相对matched FP32的before/final outer argmax变化、margin sign flip均为0；最大score绝对误差`0.0010438`，所以性能崩溃来自D48逻辑beta，而非量化。ground int8 entry/exit均为`3c08c823…267d7`，bitwise unchanged；query rows/labels/role/quota/global assignment/sidecar均为0或false。

## 18.artifact清单

|Artifact|Bytes|SHA256|
|---|---:|---|
|`training_log.jsonl`|11,910,001|`2ca433c8fa70c3def982dfc24eeabdd03191bd760c28dd97cd8c30afc5afb8b6`|
|`support_audit.json`|313,484|`dc2d108a039499d63abc62b80f0542baa48c389a0dc4620b9378238992683b72`|
|`selection.json`|2,992|`c60b369b6151666ba79384dae80e3f1b33d869bdb5c9fb27f4681f4428d8ab95`|
|`RECEIPT.json`|4,845|`dd45e78e200fe5364e5d04b4d002706ed400ebf9dedebbc2b4979624c1048a76`|
|`D48_PROBE_METADATA.json`|2,343|`59d211edf53f41755f3bae14f6558804263bc7163ca746d1e52cfdd664cb69fd`|
|`geometry_audit.json`|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|`resource_audit.json`|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|

## 19.预注册门与下一步

|门|结果|
|---|---|
|继承D42 aggregate/scene/floor/joint/混淆门|失败；after/new/H/floor与混淆全面退化|
|D45 seen-new84.00%与H82.16%不退化|失败；56.67%/56.23%|
|min-new≥D46 73.33%|失败；30.00%|
|rain after≥78.33%、forget≤10pp|after失败50.00%；forget恰好10pp|
|low-elev min-new≥50%|失败；0%|
|至少一个beta非零、至少一个prediction相对D45变化|通过；30/30 fit非零、135/330变化|
|量化0/0/0与资源边界|通过|

D48不进入125，不进行事后beta缩放、clip、median或参数扫描。下一轮必须先完成D46–D48正式复盘，再设计单一新机制；保留D46的类级异质性正信号，但必须从“直接改写完整截距”转向有解析幅度上界、类置换等变、support-only且同时保护old/new floor的受约束机制。

## 20.D46–D48三轮强制复盘与D49决策

复盘时重新核对活动目标和`项目.md`，并用确认的`ssr-gpu`解释器刷新conversation index：共1008条`E:\type10-7`记录。关键词检索没有命中本轮尚未压缩入历史索引的D46–D48当前对话，因此不以低相关旧索引结果替代证据；本节以D46/D47/D48三个正式报告、各自完整105行日志及artifact为权威。

|轮次|唯一机制变化|合法正信号|决定性缺陷|处置|
|---|---|---|---|---|
|D46|D45 global full/block weight改为同式classwise LOO weight|new+0.67pp、min-new+3.33pp、new-new混淆-1；2/330真实变化|after-old-0.56pp、forget+0.56pp；rain与low floor未解|保留“类异质性存在”结论，不保留该版本晋级|
|D47|对D46逐类log-evidence差做无超参层次收缩|恢复D45旧类/forgetting边界|330/330决策回到D45，D46新类收益全部消失；rain不动|淘汰继续平滑full/block权重轴|
|D48|D45后加入一次性OOF margin intercept residual|30/30 beta非零，作用充分；资源/量化闭合|135/330变化中98正确→错误；after/new/H跌至57.78/56.67/56.23%|淘汰直接完整幅度类截距改写及事后缩放/clip/median|

当前最强合法development点仍是D46，但其after-old81.67%、min-old53.33%、rain after76.67%与项目92%/88%门槛存在大差距。共同剩余瓶颈是rain旧类下尾与low-elev新类下尾；D45–D47的full/block统计权重轴太弱，D48的类截距轴又过强，二者均不能同时改善Stage2-B和Stage2-C。

D49选择新的单一机制轴：`strict nested support-LOO global cosine prototype fusion`。D42已经把`exp(logdiag)⊙f`变成288d全局单位球特征，因此D49不再增加三块query归一化，而是在该固定空间为每类生成`p_c=normalize(mean_{i∈S_c}x_i)`，使`x^T p_c`可精确编译为现有单一affine state。顶层每个held rank都以其余`K−1`个rank完整重拟合D45，包括D45自身的inner-LOO权重；D45与cosine各自只用inner-train RMS归一，按全类class-balanced CE经`softmax(-C×CE)`得到一个global权重。完整support重新拟合两head、重新计算各自RMS，只在FP32合成一次，canonical center后进入既有int8/FP16生命周期。target-old和target-new各自只由本类合法support按同式生成prototype；无class ID、角色、场景、receiver、query、clean/source、温度、扫描、clip或逐类截距。

D49预期可观察结果是：相对D45至少真实改变outer决策，同时不再出现D48的大规模正确→错误；优先提高rain旧类尾部或rain forgetting，并要求aggregate和三场景的after-old、seen-new、H、joint、min-old/min-new均不退化。K1必须逐位回退D45；K8的before+final严格nested适配必须审计为292次LDA。若权重数值落到0/1、FP32或int8出现精确top tie、预测完全不变、量化产生before/final/margin翻转，或任一性能门失败，则记为新负结果，不扫描温度、block-cosine、固定权重或第二development seed。该候选仍只跑同一development 105行；通过全部门后才讨论正式化，不能直接进入125。
