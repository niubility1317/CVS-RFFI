# D77地面预条件全类共同下降实验报告

## 1.实验身份

|字段|值|
|---|---|
|实验ID|`d77_ground_preconditioned_allclass_common_descent_probe_20260720`|
|候选|`ground_preconditioned_allclass_common_descent`|
|operator|Codex `/root`|
|状态|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|目标|高效利用地面int8域×类原型定义优化几何，以全注册类target-support OOF共同下降直接修正D62最终边界|
|matched baseline|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.假设与单一主要差异

D66证明静态地面可靠性缩放能略微保护旧类，却压低新类与new floor。D77不把地面统计应用到特征后重新拟合，而把它作为target OOF多类梯度的正定预条件器；地面决定坐标可信度，11类target support共同决定方向。相对D62只增加一个直接编译到final rows的地面预条件共同下降residual。

## 3.协议与资格边界

- 复用D18 `VALIDATED_ONCE/p2_min_v1`，receiver`20-1`、seed`713101`、K10/new5、3场景×5fold、actual K8。
- 单LEO_weak observation、support-only、query独立全类argmax；clean/source/query truth/role/quota/global reassignment访问0。
- 当前D19历史地面组件SHA为`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`，84个有效cell、逻辑状态25,428B，但manifest仍为`formal_phase2_eligible=false`和`UNVERIFIED_UNDER_CURRENT_PROTOCOL`。因此本轮只能是development diagnostic，不产生formal性能声明。

## 4.开发门与最终结果

|candidate|机制|receiver/TX|K/seed|B|A|N|H|F|J|min-B/A/N|状态/MAC|判定|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D77 INT8|D62＋ground-M预条件11类OOF共同下降|20-1/new5|K10(actual K8)/713101|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|15/15覆盖；40step；34,011B|负结果，不晋级|

相对D62要求`A/N/H/min-A/min-N`均不退化、`F`不升且至少一项严格改善，三类混淆不得交换伤害。D77全部指标与D62相同，严格改善项为0，15/15个outer prediction hash均未变化，因此开发门失败。按预注册规则关闭D77，不扫参数、不运行第二seed、125或N607矩阵。

## 5.版本与运行记录

`E:\type10-7`根目录不是Git仓库；代码、追溯、summarizer和本报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`，根报告同步镜像。实验在clean detached worktree`E:\type10-7\code\snapshots\d77wt`完成，本地CUDA设备为`cuda:0`，未接触N607。

## 6.本地实现与验证

|文件|作用|
|---|---|
|`code/cvsrffi/stage2_d77_ground_preconditioned_common_descent.py`|地面正定预条件器、8折OOF类梯度、20步M-Frank-Wolfe、解析步长与trust cap|
|`code/scripts/probe_d77_ground_preconditioned_allclass_common_descent.py`|D62 final-row集成、INT8/FP32编译、协议/资源/105行闭包|
|`tests/test_stage2_d77_ground_preconditioned_common_descent.py`|确定性、逐类CE安全、类置换等变、K1回退|
|`tests/test_probe_d77_ground_preconditioned_allclass_common_descent.py`|公式、固定20步、MAC加总、34,011B状态和协议字段|

- `ssr-gpu`下core/probe `py_compile`通过；专项9/9通过。
- D42–D77相邻47文件、424项全部通过，用时85.4秒。
- D25旧测试的2个源码字符串断言在D76未修改干净worktree同样失败；D77未改D25 runner，属于既有基线漂移。

## 7.运行锁

- clean worktree：`E:\type10-7\code\snapshots\d77wt`，commit`0831101802b5590a848fc62ca3b569629272698d`。
- core SHA256：`cb771f843d83b6fb11c1d373183421cd400d33a1636d4fc05d5be4fec69f603e`；probe SHA256：`198ecbd65bb91a83571bea123d1a5d28377ad5c779deaba050a56cd3ce7a51a3`。
- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，`--device auto`；本地运行，不同步或启动N607。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d77_ground_preconditioned_allclass_common_descent_probe_20260720\ground_preconditioned_allclass_common_descent`；stdout/stderr位于实验报告根。
- 预期：105行、30个target row、30次top fit、1,080次D62 component execution；每target row8个OOF LDA、88个held行、11个类梯度、20步Frank-Wolfe；query0。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d77wt\code\scripts\probe_d77_ground_preconditioned_allclass_common_descent.py' `
  --d77-arm ground_preconditioned_allclass_common_descent `
  --ground-component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d77wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d77_ground_preconditioned_allclass_common_descent_probe_20260720\ground_preconditioned_allclass_common_descent' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 8.首次运行失败与Retry1修复

- 02:44:22启动PID`21560`，02:44:58退出；原输出目录已创建但无artifact，stdout为0B，stderr为1,239B，SHA256=`c3fd5983191daa20be7297100640ab460be03f1b5318fb713a60221dbc709d83`。
- 失败发生在首个D77 target fit的资源trace拼接：D42 fit结果的20步训练记录位于`result.training_trace`，而`complete_loss_trace`只在runner评估层生成。错误为`KeyError: 'complete_loss_trace'`；尚无training row、性能或query访问。
- Retry1只把D77的20步FW trace附加到不可变dataclass的`training_trace`，runner随后自然物化40步`complete_loss_trace`。公式、数据、组件、量化、资源数值和性能门不变。
- 修复probe SHA256=`4df65ae22b04f3df99c6d2790446a062a1eaf6b40505e32703d1b05ff7582c14`；`py_compile`和专项9/9通过。
- Retry1使用新输出`ground_preconditioned_allclass_common_descent_retry1`和独立`launcher_retry1.stdout.log`/`launcher_retry1.stderr.log`，不覆盖首次失败证据。

### Retry1结果与Retry2修复

- Retry1 PID`17048`于02:47:24启动、02:47:59退出；仍无training row、性能或query访问。stderr 1,231B，SHA256=`896d06462433a76507a207c591d1bc374ad92a83350d424a547457f270ac5c87`。
- 根因：`total_optimizer_steps`和`peak_trainable_parameters`同样由runner从fit层的`optimizer_steps`与`trainable_parameters`派生，不能在fit层提前读取。
- Retry2只在fit层更新实际存在的`optimizer_steps`、`stage2c_optimizer_steps`和`trainable_parameters`；runner继续生成`total_optimizer_steps`与`peak_trainable_parameters`。公式、数据和候选均不变。
- Retry2 probe SHA256=`5c36b7536a7fb8d702995fba54bc258bb128e9df681d348148bc785b5d9d1d5b`；`py_compile`和专项9/9通过。使用新输出`ground_preconditioned_allclass_common_descent_retry2`。

## 9.Retry2完成状态与证据闭包

- Retry2 PID`6588`完整结束，stderr为0B；105/105条training row、30/30个target fit、1,080/1,080个D62 component fit均已解析。
- 完整`training_log.jsonl`为15,450,342B，SHA256=`dbf912a8c126f97f7f55043f4be2125a341fbc72d5a18ce5ca031ef08c130512`；RECEIPT SHA256=`1ef5b4ace989ddb153639fa6dd5e2aad528af17f79ac25188b789789e452166c`。
- 汇总器完整读取105行以及全部stdout/stderr，错误marker计数均为0。汇总文件`d77_full_performance_summary.json` SHA256=`7820ae8c2863d6910012f372fd00249f7108243498022972305683bf4ae0a213`。
- 地面组件入口/出口NPZ SHA均为`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`，manifest SHA均为`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`；84个cell全程只读，query未打开。
- 当前地面组件仍为`formal_phase2_eligible=false`、`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，且probe强制`formal_candidate=false`，所以本轮数值只能作为development diagnostic。

## 10.完整候选性能

所有百分比都来自同一候选的15个outer row均值；`B/A/N`分别为增量前旧类、增量后旧类和已见新类准确率，`H`为同row old/new调和均值，`F=B-A`，`J`为同row联合floor。没有拼接不同row的边际最大值。

|candidate|机制|B|A|N|H|F|J|min-class B/A/N|mean row floor B/A/N|old→new/new→old/new→wrong-new|判定|
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D77 INT8|地面M预条件全类共同下降|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|主候选，负结果|
|D77 FP32 matched|同一连续头|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|与INT8完全一致|
|D62 INT8|当前最强合法开发基线|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|保持最强|
|B3|single-IQ diag FFT/RF|87.78|75.56|72.67|73.35|12.22|23.33|80.00/60.00/40.00|53.33/33.33/36.67|33/22/19|matched弱基线|
|D42-D40|HNBR INT8|85.56|85.00|15.33|25.16|0.56|0.00|66.67/63.33/0.00|40.00/40.00/0.00|2/0/0|新类坍塌|
|D42-D41|BEC INT8|86.11|20.56|78.67|31.50|65.56|0.00|76.67/0.00/36.67|46.67/0.00/26.67|142/0/32|旧类坍塌|
|D42 ProtoNet-CDA/Z0|support-only原型基线|71.11|48.33|52.67|48.97|22.78|0.00|33.33/13.33/3.33|13.33/0.00/0.00|0/0/0|整体弱|

### 与D66地面可靠性缩放的同row比较

|版本|B|A|N|H|F|J|min-class B/A/N|old→new/new→old/new→wrong-new|
|---|---:|---:|---:|---:|---:|---:|---|---|
|D66|93.33|83.33|83.33|82.59|10.00|23.33|80.00/53.33/66.67|20/9/16|
|D77|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|23/8/15|
|D77−D66|−0.56|−1.11|+1.33|+0.03|+0.56|+3.33|0.00/0.00/+6.67|+3/−1/−1|

D77相对D66恢复了新类和new floor，但这只是回到D62边界；它没有同时获得D66的旧类保护。相对D62，全部聚合、逐场景、class-floor、row-floor、混淆指标的差值均为0，变化prediction hash为0/15。

## 11.逐场景性能

|场景|rows|B|A|N|H|F|J|min-class B/A/N|mean row floor B/A/N|old→new/new→old/new→wrong-new|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
|`leo_clear_weak`|5|98.33|91.67|98.00|94.44|6.67|50.00|90.00/70.00/90.00|90.00/60.00/90.00|2/1/0|
|`leo_low_elev_weak`|5|91.67|78.33|76.00|75.98|13.33|20.00|80.00/60.00/50.00|70.00/60.00/20.00|8/5/7|
|`leo_rain_weak`|5|88.33|76.67|80.00|77.45|11.67|10.00|60.00/30.00/70.00|60.00/30.00/30.00|13/2/8|

主要缺陷集中在低仰角和雨衰：rain的旧类最差类只有30%，low-elev的新类最差类只有50%。clear虽均值较高，但D77在5/5个clear row都退化为identity，没有利用地面几何产生有效更新。

## 12.逐类性能

|角色|类/真实TX|B|A或N|遗忘B−A|
|---|---|---:|---:|---:|
|old|`cls_75aa…`/14-10|96.67|93.33|3.33|
|old|`cls_8b02…`/14-7|80.00|53.33|26.67|
|old|`cls_1f33…`/20-15|96.67|90.00|6.67|
|old|`cls_f8df…`/20-19|93.33|93.33|0.00|
|old|`cls_a53c…`/6-15|93.33|73.33|20.00|
|old|`cls_33bb…`/8-20|96.67|90.00|6.67|
|new|`cls_09f8…`|—|73.33|—|
|new|`cls_1c2a…`|—|93.33|—|
|new|`cls_b8fb…`|—|76.67|—|
|new|`cls_d3af…`|—|90.00|—|
|new|`cls_f608…`|—|90.00|—|

旧类遗忘不是整体均匀下降，而是`14-7`和`6-15`两个旧类分别损失26.67pp和20.00pp；新类瓶颈是`cls_09f8…`与`cls_b8fb…`。因此下一版必须直接改善困难类边界，而不是继续优化平均CE。

## 13.地面原型实际使用与失败机理

|证据|结果|解释|
|---|---:|---|
|INT8 target rows|15|每行都读取84个地面cell并生成同一只读预条件器|
|有效更新/identity fallback|4/11|clear 0/5、low-elev 3/5、rain 1/5|
|地面可靠性min/mean/max|0.0243/0.7699/0.9999|地面域间稳定性被真实编码|
|预条件器min/mean/max|0.1947/1.0321/1.2495|条件数6.418，z160几何均值严格为1|
|FW目标initial→final均值|0.005925→0.000259|20步固定优化正常收敛|
|OOF CE变化均值|−0.000251|11类均为非正变化，数学安全条件成立|
|OOF正确数变化|0|连续损失下降未改变任一held样本argmax|
|support预测变化|0|更新过小或方向与离散决策边界近似正交|
|residual Frobenius均值/最大|0.00503/0.04493|4个有效更新的总体幅度仍不足|
|outer prediction hash变化|0/15|最终部署预测与D62逐row完全相同|

结论：D77不是没有利用地面原型。它把地面域×类压缩中心用于288维优化度量，并在4个INT8 row产生非零更新；但是Frank-Wolfe寻找的是“11类平均CE都不升”的共同下降锥。困难场景中各类梯度相互冲突，11/15个row的最小范数解退化为0；其余4个row也只降低极小的连续CE，没有推动top-2 margin跨过边界。创新点与协议闭包成立，但优化目标选错了：它保护平均类损失，没有直接对准旧→新、新→旧与新→错新三种部署错误以及最差类floor。

## 14.量化、资源与效率

|项目|结果|上限/结论|
|---|---:|---|
|INT8/FP32 outer argmax差异|0|量化无性能损失|
|INT8/FP32 margin sign flip|0|量化边界稳定|
|max score abs error|min/mean/max=0.000371/0.000874/0.001915|很小|
|trainable/peak parameters|2,027/2,027|≤80k，通过|
|epoch/total optimizer steps|20/40|≤30/≤50，通过|
|持久状态|34,011B|8,583B affine＋25,428B ground，≤256KB|
|query MAC|6,624|D77相对D62额外query MAC为0|
|D77额外适配MAC|255,250,056|总适配25,146,474,026，约增加1.03%|
|地面统计MAC|58,880|一次性适配开销，不逐query检索|
|CUDA峰值|22,886,912B|本地实测|
|dense query graph/query fit rows|0B/0|通过|

D77的效率目标达成：只增加约1.03%适配MAC，部署仍是单一INT8仿射头，query零额外状态和零额外计算。但“便宜”不能替代“有效”，零预测改变使其不具备晋级价值。

## 15.最终判定与下一步

最终状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D62继续是当前最强合法开发版本。D77不进入第二seed、125实验或N607；这不是实验未完成，而是严格执行预注册开发门，避免对零效应机制浪费矩阵算力。

下一候选不再使用坐标级对角预条件或平均CE共同下降，而应把地面压缩原型转化为低秩“域切向基”，再由target support的class-symmetric top-2 boundary margin决定基内更新：地面知识提供可迁移的形变方向，target support决定符号与幅度，最终仍直接编译为一个INT8 affine state。设计必须固定rank与步数、不按old/new角色分支、不读取query，并优先验证是否真正改变困难类support/outer margin；若仍无离散预测变化，立即关闭。
