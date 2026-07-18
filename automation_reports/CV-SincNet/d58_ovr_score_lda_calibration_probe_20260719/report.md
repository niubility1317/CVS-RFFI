# D58逐类one-vs-rest分数LDA校准报告

## 1.状态与目标

- 状态：`IMPLEMENTED_LOCAL_VALIDATED_PENDING_CLEAN_LOCK`；operator Codex；当前不运行125。
- 固定development cell：receiver20-1、seed713101、K10/new5、3个LEO弱场景×5fold；复用`VALIDATED_ONCE p2_min_v1`，实际outer-fit K8。
- 当前最强合法点仍为D46：before92.22%、after81.67%、new84.67%、H82.33%、forget10.56pp、min-after53.33%、min-new73.33%，不promotable。
- D55—D57复盘已停止全部CE/离散混淆流截距修正。D58检验与这些路线正交的机制：用D46 support inner-held连续分数学习每个匿名类的一维one-vs-rest LDA仿射校准，同时改变该类系数尺度与截距。

## 2.唯一公式

复用D56已经闭合的D46 full/block inner-held分数库存，但不应用D56混淆流。对每个held physical rank`r`、真实匿名类`t`和候选匿名类`c`：

`s_rtc=w_full,c*f_rtc/RMS_full+w_block,c*g_rtc/RMS_block`。

对每类`c`，正集合是`t=c`的`K`个`s_rcc`，负集合是`t!=c`的`K(C−1)`个`s_rtc`。两侧各占相同总权重，计算：

`mu+_c=mean(s|t=c)`，`mu-_c=mean(s|t!=c)`；

`v_c=(mean((s+−mu+)²)+mean((s−−mu−)²))/2`；

`a_raw,c=(mu+_c−mu-_c)/v_c`，`d_raw,c=−0.5*a_raw,c*(mu+_c+mu-_c)`。

所有`a_raw,c`必须有限且严格为正、`v_c>EPSILON`；否则该fit精确回退D46。以类置换不变的公共正尺度`abar=mean_c(a_raw,c)`消除纯数值尺度：`a_c=a_raw,c/abar`，`d_c=d_raw,c/abar`。最终：

`W_D58,c=a_c*W_D46,c`，`b_D58,c=a_c*b_D46,c+d_c`，再删除类公共仿射分量并进入既有int8 coefficient＋FP16 intercept编译。

公共`abar`不会改变argmax，只使审计数值稳定；不是超参数。K1/K2精确D46 fallback。before/final分别只用各自合法support按同一公式独立拟合。

## 3.统计声明与协议

D58称`support-supervised inner-held one-vs-rest score-space LDA calibration`，不是独立校准集、posterior或泛化保证。support标签只用于合法的正/负分组；outer-held/query不参与校准、选择、早停或回滚。

公式不读取class ID、TX、old/new角色、receiver、scene、handle、query truth/role/count/quota/order，也不做全局重分配或dense query graph。类标签和support rank置换时全部统计、仿射行与输出同步置换。clean/source不可达，不生成第二LEO观测，不改变capsule/split/schema。

无alpha、temperature、clip、threshold、ridge、分位点、场景门、角色门、第二arm或扫描。D58与D48的差异是：D48只把true-vs-max-other平均margin变为中心化截距；D58在每类完整正/负分数分布上闭式估计正斜率和midpoint，同时校准系数与截距。D58与D47/D50的差异是：不改变full/block可靠性权重。

## 4.成功门与停止门

D58必须至少保持D46的before92.22%、after81.67%、new84.67%、H82.33%、forget≤10.56pp、joint23.33%、min-before80%、min-after53.33%、min-new73.33%；相对D46至少改变1个final预测，并严格改善after、forget、joint或任一floor，且三场景不得以旧换新或以新换旧。INT8/FP32 before/final/margin翻转必须0/0/0。

若非正分离导致全部回退、预测不变、任一联合指标/场景退化或协议/量化闭包失败，则停止该公式，不扫描variance floor、斜率、ridge、clip、temperature，不跑第二seed、formal或125。即使开发门全部通过，也只能另行formalize后再考虑125。

完整报告必须包含7候选、3场景、11类、15fold、D46/D56/D57同折变化、每类mu+/mu−/variance/slope/intercept、inner-held前后表现、20epoch、量化、资源、协议和全部artifact SHA；不能只写缺陷。

## 5.资源预注册

D58复用D56的68次LDA fit和held score库存，不新增LDA fit、optimizer step、query state或sidecar。新增只包括每类正负矩、斜率、midpoint、完整affine行缩放与审计比较；将给出保守整数MAC-equivalent闭式并通过K1/2/5/8/10/20测试。最终参数、state、query MAC、epoch/step上限保持D46/D57口径。

## 6.实施计划

1. 在D56/D46 helper之上实现纯函数、integrated fit、资源账本和tamper verifier。
2. 覆盖手算二类/三类、正负平移、公共正尺度、类/支持rank置换、非正分离、零方差、K1/K2、int8与D56/D46回归。
3. `ssr-gpu`窄验证、精确Git提交、clean detached worktree锁定后，只运行一次105行development矩阵。
4. 当前不访问N607；任何后续远端动作必须先执行规定preflight。

## 7.本地实现与验证

- 方法脚本：`code/scripts/probe_d58_ovr_score_lda_calibration.py`，工作树SHA256=`f971ab5acf48919d3d1d371ae0935cb1a54f0e012589f72b35bdd1bbb6c24240`。
- 测试脚本：`tests/test_probe_d58_ovr_score_lda_calibration.py`，工作树SHA256=`7cf3762379a4f2473791fe936132a41e1af694414bc74b344e748119b4953336`。
- `py_compile`通过；D58＋D56＋D46定向链33/33通过。
- 已验证：闭式正负矩与正斜率、每类分数平移吸收、公共正尺度预测不变、类标签和support rank置换、非正分离整fit回退、K1/K2、坏score/weight闭锁、资源公式及D56/D46回归。
- verifier从training log逐fit重算全部mu、variance、slope、intercept、held预测、actual W/b和资源，再把D58附加账本剥离后调用D56完整闭包；D58新增LDA fit和query state均为0。

## 8.执行锁

- 实现提交：`461f7387`；clean detached worktree：`E:\type10-7\code\snapshots\d58wt`，状态`HEAD (no branch)`；clean脚本SHA256=`961e1e8ca1fb849f997eb99d010c9c2da810000cd84f7af53b681874b851bb13`。
- clean环境D58＋D56＋D46测试33/33通过。runtime只读复用`E:\type10-7\code\snapshots\d41wt`。
- 主Git承载面的class-binding SHA256已验证为`bb89a1db…c901f`；不用clean checkout中因CRLF变化而不匹配的副本。
- 本地前台串行、Conda`ssr-gpu`、`--device auto`，不访问N607。输出目录启动前不存在；只允许以下105行development命令执行一次。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d58wt\code\scripts\probe_d58_ovr_score_lda_calibration.py' `
  --d58-arm ovr_score_lda_calibration `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d58wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d58_ovr_score_lda_calibration_probe_20260719\ovr_score_lda_calibration' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```
