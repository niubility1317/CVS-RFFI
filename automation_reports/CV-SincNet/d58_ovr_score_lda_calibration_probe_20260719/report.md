# D58逐类one-vs-rest分数LDA校准报告

## 1.状态与目标

- 状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；operator Codex；105/105行完成，exit0，Runner elapsed112.013s；不运行125。
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

## 9.结论先行

D58在before/final各15个fit全部激活，不是回退或实现失败。final inner-held正确数平均从69.87/88提高到71.07/88（+1.20，范围−2到+6），但独立outer性能全面退化：before80.00%、after74.44%、seen-new69.33%、同排H70.92%、forgetting5.56pp、joint3.33%、min-before20.00%、min-after26.67%、min-new33.33%，混淆16/26/20。

相对D46，before−12.22pp、after−7.22pp、new−15.33pp、H−11.42pp、joint−20pp、min-before−60pp、min-after−26.67pp、min-new−40pp；15/15个prediction SHA全部变化。forgetting改善5pp不能单独视为成功，因为before与after同步崩塌，且new→old增加18次。D58不晋级、不跑第二seed、不formalize、不运行125；当前最强仍是D46。

## 10.七候选完整同排性能

|候选|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆old→new/new→old/new→new|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|低于D46|
|D42-D40-HNBR-INT8-NEGATIVE|85.56%|85.00%|15.33%|25.16%|0.56pp|0.00%|66.67%|63.33%|0.00%|2/0/0|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|86.11%|20.56%|78.67%|31.50%|65.56pp|0.00%|76.67%|0.00%|36.67%|142/0/32|旧类崩塌|
|D42-PROTOnet-CDA-ZID160|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|弱基线|
|D42-USLDA-FP32-MATCHED|80.00%|74.44%|69.33%|70.92%|5.56pp|3.33%|20.00%|26.67%|33.33%|16/26/20|与INT8一致，负结果|
|D42-USLDA-INT8|80.00%|74.44%|69.33%|70.92%|5.56pp|3.33%|20.00%|26.67%|33.33%|16/26/20|D58主候选，全面退化|
|Z0_SUPPORT_ONLY|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|弱基线|

unknown、coverage、rollback、defer均为N/A；每行指标来自同一候选15个outer rows。

## 11.三场景性能与表现

|场景|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|相对D46表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|clear|80.00%|76.67%|86.00%|80.32%|3.33pp|10.00%|10.00%|30.00%|50.00%|5/3/4|before−18.33、after−13.33、new−12、H−13.25pp|
|low-elev|78.33%|70.00%|64.00%|66.73%|8.33pp|0.00%|30.00%|40.00%|10.00%|8/9/9|after−8.33、new−12、min-new−40pp|
|rain|81.67%|76.67%|58.00%|65.70%|5.00pp|0.00%|20.00%|10.00%|20.00%|3/14/7|after持平但new−22、min-new−50pp；new→old集中爆发|

rain after持平并不构成联合收益：它用新类下降22pp和new→old14次换取forgetting下降8.33pp。clear原本最稳定，却因类斜率差异出现before floor10%，说明校准损害并非只发生在困难场景。

## 12.逐类别性能

|旧类|before→after|变化|
|---|---:|---:|
|O0 cls_1f33|83.33→66.67%|−16.67pp|
|O1 cls_33bb|86.67→83.33%|−3.33pp|
|O2 cls_75aa|96.67→96.67%|0.00pp|
|O3 cls_8b02|20.00→26.67%|+6.67pp，但仍为旧类floor|
|O4 cls_a53c|100.00→76.67%|−23.33pp|
|O5 cls_f8df|93.33→96.67%|+3.33pp|

|新类|seen-new|表现|
|---|---:|---|
|N0 cls_09f8|33.33%|全局新类floor|
|N1 cls_1c2a|93.33%|唯一稳定类|
|N2 cls_b8fb|63.33%|显著不足|
|N3 cls_d3af|86.67%|中等|
|N4 cls_f608|70.00%|显著不足|

D58把弱分离类的normalized slope压低到0.018—0.058量级，使其正样本与负吸收都被压平；这解释了O3 before跌至20%和N0跌至33.33%。

## 13.十五折完整表现

|场景|fold|before|after|new|H|forget|joint|before/after/new floor|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|0|83.33%|91.67%|80.00%|85.44%|−8.33pp|50.00%|0/50/50%|0/1/1|
|clear|1|83.33%|75.00%|90.00%|81.82%|8.33pp|0.00%|0/0/50%|1/1/0|
|clear|2|75.00%|66.67%|80.00%|72.73%|8.33pp|0.00%|0/0/0%|2/0/2|
|clear|3|75.00%|66.67%|100.00%|80.00%|8.33pp|0.00%|0/0/100%|1/0/0|
|clear|4|83.33%|83.33%|80.00%|81.63%|0.00pp|0.00%|0/0/50%|1/1/1|
|low|0|83.33%|66.67%|70.00%|68.29%|16.67pp|0.00%|50/0/0%|2/1/2|
|low|1|75.00%|58.33%|50.00%|53.85%|16.67pp|0.00%|50/0/0%|2/3/2|
|low|2|83.33%|75.00%|70.00%|72.41%|8.33pp|0.00%|50/0/0%|0/2/1|
|low|3|75.00%|75.00%|60.00%|66.67%|0.00pp|0.00%|0/50/0%|2/3/1|
|low|4|75.00%|75.00%|70.00%|72.41%|0.00pp|0.00%|0/50/0%|2/0/3|
|rain|0|75.00%|75.00%|50.00%|60.00%|0.00pp|0.00%|0/0/0%|0/2/3|
|rain|1|75.00%|83.33%|50.00%|62.50%|−8.33pp|0.00%|0/0/0%|1/4/1|
|rain|2|91.67%|83.33%|70.00%|76.09%|8.33pp|0.00%|50/50/0%|1/2/1|
|rain|3|83.33%|83.33%|70.00%|76.09%|0.00pp|0.00%|0/0/50%|1/2/1|
|rain|4|83.33%|58.33%|50.00%|53.85%|25.00pp|0.00%|50/0/0%|0/4/1|

## 14.机制、训练、量化与资源

- 30/30fit均为`support_inner_held_ovr_score_lda_calibration_active`。before separation mean/min/max为0.03121/0.00322/0.08027，final为0.04543/0.00135/0.15448；信号均正，但弱类分离接近0。
- normalized slope before min/mean/max为0.0581/1/1.9513，final为0.0183/1/1.9877；final coefficient变化L2 mean23.45、intercept变化L2 mean1.864，远大于D53/D54的安全0.1量级修正。
- final inner-held正确数平均+1.20，但范围−2到+6；clear/low/rain平均+1.4/+0.2/+2.0。support-held改进与outer退化直接证明score-space二次拟合过度校准。
- 20epoch训练正常：loss mean从1.0320降至0.1027，support accuracy从95.14%升至100%，query rows始终0；失败不是优化未收敛。
- INT8与FP32 outer/support argmax及margin sign flip均为0；最大score绝对误差min/mean/max0.000579/0.000983/0.001630。失败不是量化。
- 资源：68次LDA fit、2,010,728,448 LDA MAC；D58新增fit0、MAC-equivalent23,342、比较323，总适配2,022,257,440 MAC；query6,624 MAC、参数2,016、state8,583B、registry941B、CUDA峰值22,886,912B、20epoch/20step。
- query/source/clean/Oracle/quota/count/global/dependent optimization均0/false；dense query graph0B。平均/P95单query wall-clock与backbone/FFT次数未由本Runner单独记录，均为N/A。

## 15.相对D46/D56/D57与判定

|指标|D46/D57|D56|D58|D58−D46|
|---|---:|---:|---:|---:|
|before|92.22%|91.67%|80.00%|−12.22pp|
|after|81.67%|83.33%|74.44%|−7.22pp|
|new|84.67%|80.67%|69.33%|−15.33pp|
|H|82.33%|80.95%|70.92%|−11.42pp|
|forget|10.56pp|8.33pp|5.56pp|−5.00pp，表面改善|
|joint|23.33%|23.33%|3.33%|−20.00pp|
|min-before|80.00%|80.00%|20.00%|−60.00pp|
|min-after|53.33%|56.67%|26.67%|−26.67pp|
|min-new|73.33%|60.00%|33.33%|−40.00pp|
|混淆|25/8/15|21/12/17|16/26/20|−9/+18/+5|

成功门中只有量化、协议和forget数值通过；保持D46、场景无交换伤害、floor、H、预测联合改善和项目K10门全部失败。最终判定`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。停止D58公式，不扫描variance floor、斜率、ridge、clip或temperature。

## 16.产物与完整性

|artifact|bytes|SHA256|
|---|---:|---|
|D58_PROBE_METADATA.json|1,987|`af57042beba6dfd9a24c99e7769fd7b69ce272f5135e2933c2a2430dddb0bcc2`|
|full_performance_summary.json|144,427|`6d31589352df98cee860b0599f661d2091233182612496ac00fff1a6ea41fdeb`|
|geometry_audit.json|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|RECEIPT.json|5,035|`28fe175f13863e828ef5e2cc5b8c3af7172d613b5f06d5156496378c105bdc93`|
|resource_audit.json|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|selection.json|2,992|`da9b8f7b148b88454de17034722c847ef00219c9fc91690fc656c1d7db2142dc`|
|support_audit.json|313,674|`caf1894ee19a31f8ec9c79023dc7cc40ad375f987f6c2137ca6ac5557998b331`|
|training_log.jsonl|19,592,713|`1c81e1c13af913b72f8995c93b487253edbc01de515921bc0819be4b8f4b49e3`|

输出：`E:\type10-7\automation_reports\CV-SincNet\d58_ovr_score_lda_calibration_probe_20260719\ovr_score_lda_calibration`；parser：`code/scripts/summarize_d58_performance.py`。摘要包含7候选、3场景、11类、15fold、全部历史matched差值、机制、训练、量化与资源。

## 17.缺陷与研发结论

D58的具体缺陷是“one-vs-rest slope由小方差放大相对分离差”。虽然公共平均斜率被归一化为1，但类间斜率仍跨0.018—1.988，弱类被近乎关闭，强类主导argmax；inner-held平均+1.2掩盖了outer15/15折全面改变和新类吸收恶化。这个结果与D48的截距过强、D51/D52的几何尺度过强形成一致教训：support内部二次校准若允许类间大尺度自由度，会牺牲独立query下尾。

保留的正证据只有：连续正/负吸收分布比离散流更细，且rain inner-held可改善；但该证据不能继续通过类斜率或截距直接作用于部署logit。下一轮必须回到D46的原始尺度，研究不改变类间logit幅度的边界局部机制，或先对D46错误做只读可分解诊断；不得把D58改成ridge/clip/温度扫描。
