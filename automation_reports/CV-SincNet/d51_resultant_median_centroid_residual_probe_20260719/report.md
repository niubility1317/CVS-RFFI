# D51 resultant缩放中位数centroid残差探针

## 1.状态、目标与数据单元

- 实验ID：`d51_resultant_median_centroid_residual_probe_20260719`；状态`IMPLEMENTED_VERIFIED_NOT_RUN`。
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
