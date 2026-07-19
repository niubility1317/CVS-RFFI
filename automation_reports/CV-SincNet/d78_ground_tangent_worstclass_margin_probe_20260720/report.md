# D78地面域切向最差类边界实验报告

## 1.实验身份

|字段|值|
|---|---|
|实验ID|`d78_ground_tangent_worstclass_margin_probe_20260720`|
|候选|`ground_tangent_worstclass_top2_margin`|
|operator|Codex `/root`|
|状态|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|目标|用地面int8域×类压缩中心形成低秩域切向基，在target support内直接改善最差类top-2边界|
|matched baseline|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.假设与单一主要差异

D77对角预条件只降低连续CE，outer prediction变化为0/15。D78保留地面跨坐标域残差的最多13维联合方向，并把D62 final rows的修正限制在该子空间；优化目标改为class-symmetric smooth worst-class top-2 logistic margin。相对D62仅增加一个直接编译的低秩边界残差，不改数据、基线组件、候选集合或评测协议。

## 3.协议与资格边界

- 复用D18 `VALIDATED_ONCE/p2_min_v1`，receiver`20-1`、seed`713101`、K10/new5、3场景×5fold、actual K8。
- 单LEO_weak observation、support-only、query独立全类argmax；clean/source/query truth/role/quota/global reassignment访问0。
- 地面组件84个cell、逻辑状态25,428B，当前manifest为`formal_phase2_eligible=false`和`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，因此D78只做development diagnostic。

## 4.预注册性能门

|candidate|机制|receiver/TX|K/seed|B|A|N|H|F|J|min-B/A/N|状态/MAC|判定|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D78 INT8|D62＋ground tangent smooth-worst top-2 residual|20-1/new5|K10(actual K8)/713101|92.78|84.44|82.00|82.14|8.33|30.00|80.00/63.33/63.33|40step；34,011B|负结果，不晋级|

相对D62要求`A/N/H/min-A/min-N`均不退化、`F`不升且至少一项严格改善，三类混淆不得交换伤害。D78虽然改善`A/F/min-A/J`，但`N/H/min-N`退化且新→旧混淆增加4次，因此开发门失败。关闭D78，不扫参数、不运行第二seed、125或N607。

## 5.计划实现、验证与运行

|文件|作用|
|---|---|
|`code/cvsrffi/stage2_d78_ground_tangent_worstclass_margin.py`|地面域切向SVD、8折OOF top-2数据、smooth-worst目标、20步低秩优化|
|`code/scripts/probe_d78_ground_tangent_worstclass_margin.py`|D62 final-row集成、INT8/FP32编译、协议/资源/105行闭包|
|`tests/test_stage2_d78_ground_tangent_worstclass_margin.py`|置换等变、目标单调、top-2 margin、K1回退与确定性|
|`tests/test_probe_d78_ground_tangent_worstclass_margin.py`|公式锁、资源上限、ground只读和协议字段|

`E:\type10-7`根不是Git仓库；上述代码、追溯和本报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`的Git工作流，根报告同步镜像。实现、测试、clean worktree、命令、PID、完整性能与artifact SHA将在运行前后补录。

## 6.本地实现与验证

- core SHA256=`0139e315e0fda570c2f96a572c61de4be68f899074eba197e18d9a856baac49f`；probe SHA256=`2c656afa386495a374103162d330b452b17f4a3748dc7ef71168315e22561669`。
- `ssr-gpu`下core/probe/test `py_compile`通过；专项9/9通过。
- D42-D78邻接47文件390项全部通过，用时83.4秒。pytest退出码为0；结束后的Windows临时目录`pytest-current`清理出现一次`PermissionError`，属于atexit清理噪声，不是测试失败。
- 真实ground组件烟测：26个registry domain中14个完整有效域、84个cell；切向rank13，保留残差能量77.7513%，basis只读；组件formal资格仍为false。

## 7.运行锁

- clean detached worktree：`E:\type10-7\code\snapshots\d78wt`；本地`cuda:0`运行，不同步或启动N607。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d78_ground_tangent_worstclass_margin_probe_20260720\ground_tangent_worstclass_top2_margin`；stdout/stderr独立保存在报告根。
- 预期：105行、30个target fit、1,080个D62 component execution；每target row8个OOF LDA、88个held行、rank13、20个接受步；query0。
- 精确命令如下；进程参数固定，禁止覆盖已有输出：

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d78wt\code\scripts\probe_d78_ground_tangent_worstclass_margin.py' `
  --d78-arm ground_tangent_worstclass_top2_margin `
  --ground-component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d78wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d78_ground_tangent_worstclass_margin_probe_20260720\ground_tangent_worstclass_top2_margin' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

- 2026-07-20T03:23:32+08:00以隐藏本地进程启动，PID`14308`；stdout=`launcher.stdout.log`，stderr=`launcher.stderr.log`。启动后只读检查确认完整命令行与锁定参数一致。

## 8.完成状态与证据闭包

- PID`14308`完整退出；105/105条training row、30/30个target fit、1,080/1,080个D62 component fit均已解析，stderr为0B。
- `training_log.jsonl`为15,597,883B，SHA256=`f398089c06e16e0c45fb00a3689ef49591adab3aebe73cc5a48d2769d75edaab`；RECEIPT SHA256=`63c9d567d0fb5a97f6dfc10af7f3a4b9f2aaa28ca9d8bf2ef69664482d5e008e`。
- `d78_full_performance_summary.json`完整读取D78/D77/D75/D66/D62各105行及全部stdout/stderr，SHA256=`417e54102b8b813be00978b6ed1dcfd655c11939f435fbfc2d29ea857ea3234e`；错误marker均为0。
- ground NPZ入口/出口SHA均为`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`，manifest入口/出口SHA均为`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`；84个cell只读，query未打开。
- 当前组件仍为`formal_phase2_eligible=false`和`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，probe强制`formal_candidate=false`，本轮数值只能作为development diagnostic。

## 9.完整候选性能

所有百分比来自同一候选15个outer row的联合统计；`B/A/N`为增量前旧类、增量后旧类、已见新类准确率，`H`为同row调和均值，`F=B-A`，`J`为同row联合floor。

|candidate|机制|B|A|N|H|F|J|min-class B/A/N|mean row floor B/A/N|old→new/new→old/new→wrong-new|判定|
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D78 INT8|ground tangent smooth-worst top-2|92.78|84.44|82.00|82.14|8.33|30.00|80.00/63.33/63.33|73.33/56.67/43.33|19/12/15|主候选，负结果|
|D78 FP32 matched|同一连续头|92.78|84.44|82.00|82.14|8.33|30.00|80.00/63.33/63.33|73.33/56.67/43.33|19/12/15|与INT8完全一致|
|D62/D77 INT8|当前最强合法开发基线|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|保持最强|
|B3|single-IQ diag FFT/RF|87.78|75.56|72.67|73.35|12.22|23.33|80.00/60.00/40.00|53.33/33.33/36.67|33/22/19|matched弱基线|
|D42-D40|HNBR INT8|85.56|85.00|15.33|25.16|0.56|0.00|66.67/63.33/0.00|40.00/40.00/0.00|2/0/0|新类坍塌|
|D42-D41|BEC INT8|86.11|20.56|78.67|31.50|65.56|0.00|76.67/0.00/36.67|46.67/0.00/26.67|142/0/32|旧类坍塌|
|D42 ProtoNet-CDA/Z0|support-only原型|71.11|48.33|52.67|48.97|22.78|0.00|33.33/13.33/3.33|13.33/0.00/0.00|0/0/0|整体弱|

### D78与地面路线基线的同row差值

|比较|ΔB|ΔA|ΔN|ΔH|ΔF|ΔJ|Δmin-class B/A/N|Δold→new/new→old/new→wrong-new|
|---|---:|---:|---:|---:|---:|---:|---|---|
|D78−D62/D77|0.00|+2.22|−2.67|−0.48|−2.22|+3.33|0.00/+10.00/−10.00|−4/+4/0|
|D78−D66|−0.56|+1.11|−1.33|−0.45|−1.67|+6.67|0.00/+10.00/−3.33|−1/+3/−1|

D78首次证明ground域切向能实质改变部署边界并保护旧类，但改善不是联合Pareto：减少的4次old→new全部交换成4次new→old，导致新类与H退化。

## 10.逐场景性能

|场景|rows|B|A|N|H|F|J|min-class B/A/N|mean row floor B/A/N|old→new/new→old/new→wrong-new|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
|`leo_clear_weak`|5|98.33|91.67|98.00|94.44|6.67|50.00|90.00/70.00/90.00|90.00/60.00/90.00|2/1/0|
|`leo_low_elev_weak`|5|91.67|78.33|76.00|75.98|13.33|20.00|80.00/60.00/50.00|70.00/60.00/20.00|8/5/7|
|`leo_rain_weak`|5|88.33|83.33|72.00|76.00|5.00|20.00|60.00/60.00/50.00|60.00/50.00/20.00|9/6/8|

相对D62，clear与low-elev所有outer指标完全不变；变化集中在rain：`A+6.67pp`、`F−6.67pp`、min-A`+30pp`、J`+10pp`，但`N−8pp`、min-N`−20pp`、H`−1.44pp`。15个fold中只有3个rain fold改变prediction hash。

## 11.逐类性能

|角色|类/真实TX|B|A或N|遗忘B−A|
|---|---|---:|---:|---:|
|old|`cls_75aa…`/14-10|96.67|93.33|3.33|
|old|`cls_8b02…`/14-7|80.00|63.33|16.67|
|old|`cls_1f33…`/20-15|96.67|90.00|6.67|
|old|`cls_f8df…`/20-19|93.33|93.33|0.00|
|old|`cls_a53c…`/6-15|93.33|76.67|16.67|
|old|`cls_33bb…`/8-20|96.67|90.00|6.67|
|new|`cls_09f8…`|—|73.33|—|
|new|`cls_1c2a…`|—|93.33|—|
|new|`cls_b8fb…`|—|63.33|—|
|new|`cls_d3af…`|—|90.00|—|
|new|`cls_f608…`|—|90.00|—|

D78把最差旧类`14-7`从53.33%提升到63.33%，也把`6-15`从73.33%提升到76.67%；代价几乎全部落在新类`cls_b8fb…`，从76.67%降到63.33%。这进一步支持“切向残差产生类吸引偏置”而非全局噪声的判断。

## 12.机制表现与缺陷

|证据|结果|解释|
|---|---:|---|
|ground registry/effective domain/cell|26/14/84|只读真实组件完整使用|
|numerical/tangent rank|78/13|固定域对比上限，不扫rank|
|切向保留能量|77.7513%|保留主要跨坐标域形变|
|有效更新/fallback|15/0|每个INT8 row均产生非零残差|
|残差Frobenius均值|1.1200|全部恰好触及类无关trust radius|
|smooth-worst目标变化均值|−0.05574|20步、300条trace全部单调非增|
|OOF CE变化均值|−0.00581|比D77的−0.000251更强|
|OOF非正margin数变化|0|18.87→18.87，未纠正任何crossfit argmax|
|full-support预测变化|0|15/15个INT8 fit均未改变训练support argmax|
|outer prediction变化|3/15|只在rain改变，说明边界转动已到达部署近边界样本|

D78解决了D77“更新太小/大量identity”的问题，但暴露第二层根因：低秩系数每行都饱和到trust ball，OOF目标下降仍没有减少任何误分类；由于切向特征没有以target support中心化，`AU^T x`在support/query均值处形成类相关常量，相当于隐式改变类别先验。地面组件虽不直接输出旧类分数，但其Phase1域残差子空间来自旧类聚合，最终在rain把新类样本推向旧类。

## 13.量化、资源与效率

|项目|结果|上限/结论|
|---|---:|---|
|INT8/FP32 outer argmax差异|0|量化无性能损失|
|INT8/FP32 margin sign flip|0|量化边界稳定|
|max score abs error|min/mean/max=0.000377/0.000882/0.001915|很小|
|trainable/peak parameters|2,159/2,159|含143切向系数，≤80k|
|epoch/total optimizer steps|20/40|≤30/≤50|
|持久状态|34,011B|8,583B affine＋25,428B ground，≤256KB|
|D78额外适配MAC|351,229,416|总适配25,242,453,386，约比D62增加1.41%|
|ground SVD保守MAC上界|97,944,320|一次性适配开销|
|query MAC|6,624|D78相对D62额外query MAC/state为0|
|CUDA峰值|22,886,912B|本地实测|
|dense query graph/query fit rows|0B/0|通过|

## 14.最终判定与下一步

最终状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D62继续是当前最强合法开发版本；D78不进入第二seed、125或N607。

下一候选D79只修复D78识别出的类先验漂移：切向优化改用`(x−mu_support)U`，并把`−DeltaW mu_support`编译进bias，使每类残差在target support均值处严格为0。该“中心化切向旋转”保留D78的ground域方向与旧类保护潜力，同时消除无角色条件下的全局类吸引项；rank、目标、20步、trust ball和全部协议边界保持不变。若仍交换old/new混淆，则关闭该路线。
