# D48一次性OOF-head margin残差探针报告

## 1.身份与目标

- 实验ID：`d48_one_shot_oof_margin_residual_probe_20260718`。
- 操作者：Codex`/root`。
- 当前状态：`PRE_REGISTERED_NOT_RUN`。
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
