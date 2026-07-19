# D80地面跨域质心漂移协方差实验报告

## 1.实验身份

|字段|值|
|---|---|
|实验ID|`d80_ground_commonmode_covariance_denoiser_probe_20260720`|
|候选|`ground_commonmode_covariance_denoiser`|
|operator|Codex `/root`|
|状态|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|目标|把地面压缩原型仅作为所有注册类共享的域噪声协方差先验，联合改善旧类域适应与新类注册|
|开发单元|receiver`20-1`、new5、seed`713101`、K10(actual K8)、3场景×5fold|
|matched baseline|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.预运行审查后的方法锁

独立数学和代码审查在任何D80性能运行前纠正了初始设计：当前D22 v1 bundle只有int8域×类质心、FP16 scale、mask和registry，没有sample radius、count或域内散度。因此D80不能声称使用“地面类内样本协方差”，也不能在D62最终row后做未经过OOF审查的post-hoc投影。

最终锁定方法如下。对解量化地面质心`g_dc=s_dc q_dc`先逐类去中心：

`r_dc=g_dc−mean_d(g_dc)`，

再用全部84个cell形成共享的“同类跨域质心漂移协方差”：

`G=sum_dc(r_dc r_dc^T)/[C_g(D−1)] + mean(s_dc^2/12)I`。

类中心在残差化后立即丢弃，ground不产生anchor、类别分数或class-row residual。量化噪声底固定为均匀舍入模型`mean(scale²/12)`，不扫描ridge。

每个D62 full/block、outer/physical-rank-held LDA fit均在自己的合法train support内估计target shrinkage covariance`T`，把`G`按target z160块trace匹配后，以固定自由度权重

`lambda=(D_eff−1)/[(D_eff−1)+C(K−1)]`

构造PSD后验协方差。当前`D_eff=14`，所以before`C=6,K=8`时`lambda=13/55=0.23636`，after`C=11,K=8`时`lambda=13/90=0.14444`。FFT96/RF32使用target block covariance；ground只进入z160。最终求解equal-prior Mahalanobis`W=Sigma_post^−1 mu`并进入锁定D62 row splice，部署仍为单个INT8 affine head，query额外MAC/state为0。

## 3.协议与资格边界

- 复用D18 `VALIDATED_ONCE/p2_min_v1`；不改变received-IQ、physical ID、receiver/TX、场景、K或support/query划分。
- single-LEO_weak、support-only、query独立全类argmax；clean/source/query truth/role/quota/global assignment访问0。
- ground组件26个registry domain、14个完整有效域、6个ground类、84 cell、逻辑状态25,428B；只读。
- 当前组件`formal_phase2_eligible=false`、`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，所以本轮强制development diagnostic；即使性能为正也不能直接进入125。
- 不扫描`lambda`、rank、量化ridge、类权重、场景权重或旧/新门。

## 4.实现、测试与真实ground烟测

|文件|作用|
|---|---|
|`code/cvsrffi/stage2_d80_ground_commonmode_denoiser.py`|class-centered ground covariance、量化噪声底、trace match、固定自由度EB full/block LDA|
|`code/scripts/probe_d80_ground_commonmode_covariance_denoiser.py`|D66严格v1 loader、D62全部closure注入、协议/资源/hash/105行闭包|
|`tests/test_stage2_d80_ground_commonmode_denoiser.py`|置换不变、PSD、量化底、K1、类等变、full/block闭包|
|`tests/test_probe_d80_ground_commonmode_covariance_denoiser.py`|factory注入顺序、source lock、协议/资源/hash和无radius/count声明|

- core SHA256=`e6edea077beeb02f69f898cec4d3ee89c23bfe4f1b7e5044fba68533a20eb5b2`；probe SHA256=`d37d629085e5f2dd1d7c1e02993964a295996945af6bef2b4c79185bd9a73183`。
- `ssr-gpu`下py_compile通过；D80专项10/10通过；D62/D78/D79/D80相邻专项34/34通过。
- synthetic D62 full-stack烟测：11类×K8、输出`W[11,288]/b[11]`有限，D62内部18个component fit全部执行，after权重精确`0.14444444444444443`，full/block均注入。
- 真实只读ground烟测：有效域14、类6、cell84、残差rank78、participation effective rank13.6446、量化噪声底`5.2414323e−7`、后验前ground协方差特征值`5.2414323e−7`至`2.9496292e−4`。这证明当前v1数据能提供域质心漂移形状，但仍不含sample radius/count。
- `E:\type10-7`根不是Git仓库；Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，clean detached worktree为`E:\type10-7\code\snapshots\d80wt`。

## 5.性能门与停止条件

相对D62要求总体及每个场景的`A/N/H/J/min-A/min-N`不退化、`F`不升，三项mean row floor不退化，且至少一项严格改善；三类混淆`old→new/new→old/new→wrong-new`均不得增加。INT8相对FP32要求outer argmax变化和margin sign flip均为0。完整报告必须给出同row`B/A/N/H/F/J`、逐场景、全部逐类旧类遗忘和新类准确率、15fold、混淆、量化、协方差机制与资源。

若与D62完全相同，说明ground prior被target shrinkage/D62吸收；若`A`升而`N/min-N`降，说明old-only ground残差仍把身份方向误作噪声；若support-held改善而outer退化，说明proxy mismatch。任一情况都关闭本路线，不扫参数、不启第二seed、125或N607。

## 6.运行锁

运行固定复用D79/D78的D18 before/after capsule、seal、authorization、D22 component、class binding、`--device auto --mode development_select_unverified_component --candidate-set d42_v1`；仅替换为：

- 入口`probe_d80_ground_commonmode_covariance_denoiser.py`；
- `--d80-arm ground_commonmode_covariance_denoiser`；
- `--ground-component-dir`及锁定manifest SHA；
- probe root=`E:\type10-7\code\snapshots\d80wt`；
- 独立output=`E:\type10-7\automation_reports\CV-SincNet\d80_ground_commonmode_covariance_denoiser_probe_20260720\ground_commonmode_covariance_denoiser`。

预期105行、30个target fit、1,080个D62 component fit；每个held fit在排除对应physical rank后独立重算target covariance，query0。detached实现提交=`7f08fcba`，主分支实现提交=`b6b8a2ce`。

精确运行命令如下：

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d80wt\code\scripts\probe_d80_ground_commonmode_covariance_denoiser.py' `
  --d80-arm ground_commonmode_covariance_denoiser `
  --ground-component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d80wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d80_ground_commonmode_covariance_denoiser_probe_20260720\ground_commonmode_covariance_denoiser' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

2026-07-20T04:14:55+08:00以隐藏本地进程启动，PID`18884`；完整cmdline只读核对与上述锁一致。进程运行114.81秒后正常退出。

## 7.完成状态与证据闭包

- 105/105条training row、30/30个target fit、1,080/1,080个D62 component fit全部解析；stdout为5,026B，stderr为0B，错误marker均为0。
- `training_log.jsonl`为16,967,923B，SHA256=`070921da4ffe87b751875f4897887bbdd2a7c7501e9447ee5653982966b365b0`；RECEIPT SHA256=`3dc8d1f07416252489aa8a457e093bfb26fddbe16feabd96d2bf3bc47b29cfda`；metadata SHA256=`ef50aa774825633cb0e521c6c6f8679021f22dfe4037f2533b3a15a41d7af13f`。
- `d80_full_performance_summary.json`完整读取D80/D79/D78/D77/D66/D62各105行以及全部stdout/stderr，SHA256=`1b0211812801fd840e449c3c9e324f56ecb00a8e83ea76dcb1787c458fb3c694`。
- ground NPZ入口/出口SHA均为`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`；manifest入口/出口SHA均为`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`；组件bitwise unchanged。
- `query_opened=false`、formal candidate=false；当前组件资格仍为false/UNVERIFIED，结果仅为development diagnostic。

## 8.完整候选性能

所有百分比来自同一候选15个outer row；`B/A/N`为注册前旧类、注册后旧类、已见新类准确率，`H`为同row调和均值，`F=B−A`，`J`为同row联合floor。

|candidate|机制|B|A|N|H|F|J|min-class B/A/N|mean row floor B/A/N|old→new/new→old/new→wrong-new|判定|
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D80 INT8|ground covariance EB＋D62|93.89|82.78|84.00|82.45|11.11|26.67|80.00/53.33/70.00|73.33/50.00/46.67|22/8/16|主候选，负结果|
|D80 FP32 matched|同一Mahalanobis头|93.89|82.78|84.00|82.45|11.11|26.67|80.00/53.33/70.00|73.33/50.00/46.67|22/8/16|与INT8完全一致|
|D62/D77 INT8|当前最强合法开发基线|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|仍保持最强|
|B3|single-IQ diag FFT/RF|87.78|75.56|72.67|73.35|12.22|23.33|80.00/60.00/40.00|53.33/33.33/36.67|33/22/19|matched弱基线|
|D42-D40|HNBR INT8|85.56|85.00|15.33|25.16|0.56|0.00|66.67/63.33/0.00|40.00/40.00/0.00|2/0/0|新类坍塌|
|D42-D41|BEC INT8|86.11|20.56|78.67|31.50|65.56|0.00|76.67/0.00/36.67|46.67/0.00/26.67|142/0/32|旧类坍塌|
|D42 ProtoNet-CDA/Z0|support-only原型|71.11|48.33|52.67|48.97|22.78|0.00|33.33/13.33/3.33|13.33/0.00/0.00|0/0/0|整体弱|

### 同row差值

|比较|ΔB|ΔA|ΔN|ΔH|ΔF|ΔJ|Δmin-class B/A/N|Δold→new/new→old/new→wrong-new|outer hash变化|
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
|D80−D62/D77|+1.11|+0.56|−0.67|−0.18|+0.56|0.00|0.00/0.00/−3.33|−1/0/+1|3/15|
|D80−D79|+1.11|−1.67|+1.33|−0.27|+2.78|−3.33|0.00/−6.67/0.00|+3/−3/+1|6/15|
|D80−D78|+1.11|−1.67|+2.00|+0.31|+2.78|−3.33|0.00/−10.00/+6.67|+3/−4/+1|—|
|D80−D66|+0.56|−0.56|+0.67|−0.14|+1.11|+3.33|0.00/0.00/+3.33|+2/−1/0|—|

D80提高了注册前旧类和少量注册后旧类，但`N/H/min-N`退化、遗忘增加，并把减少的1次old→new换成1次new→wrong-new。它不满足总体门，也不满足逐场景门，因此不能用`B`或`A`的边际收益晋级。

## 9.逐场景性能

|场景|rows|B|A|N|H|F|J|min-class B/A/N|mean row floor B/A/N|old→new/new→old/new→wrong-new|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
|`leo_clear_weak`|5|98.33|91.67|98.00|94.44|6.67|50.00|90.00/70.00/90.00|90.00/60.00/90.00|2/1/0|
|`leo_low_elev_weak`|5|93.33|80.00|74.00|75.45|13.33|20.00|80.00/60.00/40.00|70.00/60.00/20.00|7/5/8|
|`leo_rain_weak`|5|90.00|76.67|80.00|77.45|13.33|10.00|60.00/30.00/70.00|60.00/30.00/30.00|13/2/8|

相对D62，clear完全相同；low-elev为`B/A+1.67pp`，却`N−2pp`、H`−0.53pp`、min-N`−10pp`；rain仅`B+1.67pp`，注册后`A/N/H`完全不变，因此遗忘反而`+1.67pp`。ground prior的正信号主要停留在注册前，未形成Stage2-C联合增益。

## 10.逐类性能与遗忘

|角色|类/真实TX|B|A或N|遗忘B−A|
|---|---|---:|---:|---:|
|old|`cls_75aa…`/14-10|96.67|93.33|3.33|
|old|`cls_8b02…`/14-7|80.00|53.33|26.67|
|old|`cls_1f33…`/20-15|96.67|90.00|6.67|
|old|`cls_f8df…`/20-19|93.33|93.33|0.00|
|old|`cls_a53c…`/6-15|100.00|73.33|26.67|
|old|`cls_33bb…`/8-20|96.67|93.33|3.33|
|new|`cls_09f8…`|—|70.00|—|
|new|`cls_1c2a…`|—|93.33|—|
|new|`cls_b8fb…`|—|76.67|—|
|new|`cls_d3af…`|—|90.00|—|
|new|`cls_f608…`|—|90.00|—|

最差旧类仍为`14-7=53.33%`，`6-15`遗忘26.67pp；最差新类`cls_09f8…=70%`，比D62最低73.33%更差。均值提升没有修复任何通用floor。

## 11.机制表现与缺陷

|证据|结果|解释|
|---|---:|---|
|ground registry/effective domain/class/cell|26/14/6/84|真实只读组件完整使用|
|ground残差rank/effective rank|78/13.6446|保留D78丢弃的全78维谱形状|
|ground独立domain自由度|13|没有把84 cell伪当84个独立域|
|量化噪声底|`5.2414e−7`|固定`mean(scale²/12)`，无扫描|
|ground协方差condition|562.75|加入量化底后SPD|
|before/final target自由度|42/77|来自`C(K−1)`|
|before/final ground权重|0.23636/0.14444|固定13/55与13/90|
|before/final posterior condition均值|624,536/322,615|trace matching后仍高度病态|
|D62 before/final active fold|5/15、1/15|注册后row-splice大多回退，ground信号难以穿过support-held安全门|
|before/final平均接受row|0.867/0.200|Stage2-C新增类后ground covariance显著压低可接受残差行|
|outer prediction变化|3/15|不是identity，但变化不足且方向错误|

核心缺陷是：当前ground只有old6类的跨域质心漂移。即使逐类去中心并作为共享协方差，它仍把一部分对新类有用的身份方向当成域噪声；low-elev的新类损失就是直接证据。另一方面，加入11类后固定权重下降到0.14444，D62 final row-splice又只在1/15 fold激活，导致注册前旧类收益没有延续到注册后。继续调`lambda`、rank或ridge属于开发query驱动扫描，不能解决这个结构性不对称。

## 12.量化、训练与资源

|项目|结果|结论|
|---|---:|---|
|INT8/FP32 outer argmax差异|0|量化稳定|
|INT8/FP32 margin sign flip|0|通过量化门|
|before/final support argmax差异|0/0|量化未改变support决策|
|max score abs error|min/mean/max=`0.000362/0.000901/0.001999`|数值误差小|
|基础training trace|20条/row|完整D42 Stage2-B trace|
|D80新增optimizer step/parameter|0/0|闭式协方差，不训练额外参数|
|trainable/peak parameters|2,016/2,016|≤80k|
|epoch/total optimizer steps|20/20|≤30/≤50|
|持久状态|34,011B|8,583B affine＋25,428B ground，≤256KB|
|D80新增适配MAC|141,099,008|约为D78新增351,229,416的40.2%|
|总适配MAC|25,032,322,978|基于D62完整闭式链|
|query MAC/额外MAC|6,624/0|单affine部署|
|ground FP64 covariance瞬时内存|204,800B|不持久化|
|CUDA峰值|22,886,912B|本地实测|
|dense query graph/query fit rows|0B/0|通过|

D80相对D78更高效且完全消除了INT8翻转，但性能门失败；效率和量化稳定不能替代联合性能。

## 13.最终判定与下一步

D80最终状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。不运行第二seed、125或N607；D62继续是当前最强合法开发版本。

关闭“ground covariance直接进入query判别度量”的路线。下一候选不再让old-only ground几何作用于query分数，而只用它在每个target类内部评估support样本的域扰动影响，生成类对称、K保持不变的鲁棒target中心；最终query度量仍由target support本身决定。该机制若无法同时改善旧类与新类floor，也应关闭当前v1 ground组件的Phase2决策用途，把正式改进转移到Phase1重新封存包含合法聚合radius/dispersion的bundle。
