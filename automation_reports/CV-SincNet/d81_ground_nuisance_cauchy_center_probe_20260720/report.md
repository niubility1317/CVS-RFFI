# D81地面扰动谱稳健target原型实验报告

## 1.实验登记

|字段|值|
|---|---|
|实验ID|`d81_ground_nuisance_cauchy_center_probe_20260720`|
|候选|`ground_nuisance_cauchy_center`|
|operator|Codex`/root`|
|状态|`COMPLETED_DEVELOPMENT_GATE_PASS_CONFIRMATION_PENDING_UNVERIFIED_COMPONENT`|
|目标|高效利用全部地面压缩原型估计support样本的跨域扰动可靠性，同时让query判别几何完全由target support决定|
|开发单元|receiver`20-1`、new5、seed`713101`、K10(actual K8)、3场景×5fold|
|matched baseline|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|
|formal状态|当前ground组件资格false/UNVERIFIED，D81仅development diagnostic|

## 2.假设与创新点

D77-D80已经排除了把ground质心、低秩投影或ground协方差直接放进query距离/协方差的路线：这些方法能保护部分旧类，却把新类身份方向误当域噪声。D81把ground的作用前移到注册阶段，只回答“同一target类中哪个support样本更像受到已知跨域扰动”，再以target support自己形成稳健类中心。

该设计有三个隔离性质：

1. 地面old6类不提供任何类别锚点或query score，只提供类无关扰动方向；
2. 每类共同平移保持类内残差和target协方差不变，因此不会重写D62的target度量；
3. 权重在每个OOF fit内重算，held support和query均不可见。

## 3.锁定公式

从84个地面domain-class类中心构造`r_dc=g_dc−mean_d(g_dc)`与协方差`G`。对正特征值`lambda_j`计算：

`r_eff=(sum_j lambda_j)^2/sum_j lambda_j^2`，`r=ceil(r_eff)`。

固定保留前`r`个方向，并令`pi_j=lambda_j/sum_{l<=r}lambda_l`。对当前fit可见的target类`c`：

`e_ci=sum_{j<=r} pi_j [u_j^T(z_ci−mean_i z_ci)]^2`

`raw_w_ci=1/(1+e_ci/mean_i e_ci)`，`w_ci=raw_w_ci/sum_i raw_w_ci`

`mu_robust_c=sum_i w_ci z_ci`

`z'_ci=z_ci+(mu_robust_c−mean_i z_ci)`。

若能量为0则等权；K1显式identity，K2因两个中心残差互为相反数而严格等权identity。只变换z160，FFT96/RF32保持bitwise不变。禁止rank、尺度、温度、平移系数或场景/类别权重扫描。

## 4.协议与资源边界

- 数据状态沿用D18`VALIDATED_ONCE`；方法变更不触发重建/重验。
- 单一固定`LEO_weak`观测；support-only；query独立一次评分；无clean/source/query truth/role/quota/global assignment。
- target-old/new完全相同公式；不访问类ID语义、old/new角色、receiver handle或scene handle。
- 使用全部84个ground cell估计谱；当前组件无sample radius/count，不伪造这些统计。
- 预计新增适配复杂度为每次fit`O(N*r*160)`，其中`r`由ground effective rank自动确定；新增参数/optimizer step/query MAC均为0。
- 持久状态仍为D62单affine＋25,428B ground组件，≤256KB。

## 5.联合晋级门

相对同row D62：总体`A/N/H/J/min-class B/A/N`不得下降、`F`不得上升；每个场景`A/N/H`不得下降、`F`不得上升；`old→new`、`new→old`、`new→wrong-new`均不得增加；且至少一个联合指标严格改善。INT8/FP32不得发生outer argmax或margin-sign翻转。任一失败即停止，不启第二seed、125或N607。

## 6.版本状态

根目录`E:\type10-7`非Git仓库。实现、trace和本报告先进入独立Git worktree`E:\type10-7\code\snapshots\d81wt`，基于主发布分支提交`4dcf066b`；完成本地验证后再以精确commit闭环回主发布分支。服务器暂不使用。

## 7.实现与验证

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d81_ground_nuisance_cauchy_center.py`|地面扰动谱、固定rank、support稳健中心平移|`44111f8d7ecd0ffcfbd887c09468a167e4e1134bad3c2798bd7f0f5f89c3dc7a`|
|`code/scripts/probe_d81_ground_nuisance_cauchy_center.py`|D62全部full/block、outer/held闭包注入、资源和hash审计|`85baac449d2cd1c5b21bff63ba9b01fe95bb2025fcdfa8ee3127ae41a5e99e82`|

- D81专项与合成D62全栈：11/11通过。
- D62/D80/D81相邻链：30/30通过。
- 真实ground smoke：84 cells，effective rank=`13.6445898983`，retained rank=`14`，保留信号trace比例=`0.7975861768`，basis SHA=`f55174f1e1479eed4bd62b927ef7b4e952f14fa03cadc0e70b315e183426ed7f`，radius/count均false。
- 合成D62链确认每次fit的full、block及其inner-LOO都经过独立center transform；K1/K2 bitwise identity，query extra MAC=0。

## 8.锁定运行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d81wt\code\scripts\probe_d81_ground_nuisance_cauchy_center.py' `
  --d81-arm ground_nuisance_cauchy_center `
  --ground-component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d81wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d81_ground_nuisance_cauchy_center_probe_20260720\ground_nuisance_cauchy_center' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期105行、30个target row、1,080个D62 component fit、2,160个support-center transform。先本地运行；不使用N607。

## 9.运行前锁定

- detached实现提交：`2f6a26d3c02fa7b33ee2efc1183748f55a396fdf`；主发布分支对应实现提交：`db4013dd`。
- worktree在锁定前为0项未提交改动；输出目录不存在，不会覆盖历史结果。
- 本地GPU0为RTX5070Ti，检查时1,083/16,303MiB、利用率0%；本实验锁定`--device auto`，由runner记录实际runtime device与CUDA峰值。
- 数据复用D18 matching capsule/seal/policy/authorization；方法变化不触发数据重验。
- ground NPZ/manifest在入口和出口分别复核SHA；任何hash变化、105-row不完整、1,080 component或2,160 transform计数不匹配均判运行失败。

## 10.完成状态与证据闭包

- runner耗时123.53秒，外层wall time约132.2秒，正常退出；105/105条training row、30/30个target row、1,080/1,080个D62 component fit、2,160/2,160个support-center transform全部验证。
- `training_log.jsonl` SHA256=`6a362af9bb935fcd5592d3f91b8695ee408540f7ce5de75a2a8bf03341e0c9dc`；RECEIPT SHA256=`7770657bd0aa860be4c81a3a7709ea67e6ab094297146261ee47bea61d89a6c8`；metadata SHA256=`cde3352826caefba95a7c89d0654d54b72e736f0cd1e84d536f557d436538bc2`。
- `d81_full_performance_summary.json`完整读取D81/D80/D79/D62各105行以及全部stdout/stderr，SHA256=`1451127d56cca6a8716d878540c14b3f22353e379e8f3b90a2029e9292352274`。
- stdout 5,119B，SHA256=`b4336bf22da5237d69a001042c2a2bf83357a3cb9426f97a0ee6005ba7fe81ae`；stderr 0B，错误marker全部为0。
- ground NPZ入口/出口SHA均为`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`；manifest入口/出口SHA均为`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`；bitwise unchanged。
- `query_opened=false`；formal candidate=false。当前组件资格仍为false/UNVERIFIED，以下结果只支持开发选择。

## 11.完整候选性能

所有百分比来自同一候选15个outer row；`B/A/N`为注册前旧类、注册后旧类、已见新类准确率，`H`为同row调和均值，`F=B−A`，`J`为同row联合floor。

|candidate|机制|B|A|N|H|F|J|min-class B/A/N|mean row floor B/A/N|old→new/new→old/new→wrong-new|判定|
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D81 INT8|ground谱support稳健中心＋D62|92.78|82.78|84.67|82.94|10.00|26.67|80.00/53.33/73.33|73.33/50.00/46.67|22/8/15|开发联合门通过|
|D81 FP32 matched|同一target-support D62头|92.78|82.78|84.67|82.94|10.00|26.67|80.00/53.33/73.33|73.33/50.00/46.67|22/8/15|与INT8完全一致|
|D62|当前matched基线|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|被D81严格支配|
|B3|single-IQ diag FFT/RF|87.78|75.56|72.67|73.35|12.22|23.33|80.00/60.00/40.00|53.33/33.33/36.67|33/22/19|matched弱基线|
|D42-D40|HNBR INT8|85.56|85.00|15.33|25.16|0.56|0.00|66.67/63.33/0.00|40.00/40.00/0.00|2/0/0|新类坍塌|
|D42-D41|BEC INT8|86.11|20.56|78.67|31.50|65.56|0.00|76.67/0.00/36.67|46.67/0.00/26.67|142/0/32|旧类坍塌|
|ProtoNet-CDA/Z0|support-only原型|71.11|48.33|52.67|48.97|22.78|0.00|33.33/13.33/3.33|13.33/0.00/0.00|0/0/0|整体弱|

### 同row差值

|比较|ΔB|ΔA|ΔN|ΔH|ΔF|ΔJ|Δmin-class B/A/N|Δold→new/new→old/new→wrong-new|outer hash变化|
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
|D81−D62|0.00|+0.56|0.00|+0.31|−0.56|0.00|0.00/0.00/0.00|−1/0/0|1/15|
|D81−D80|−1.11|0.00|+0.67|+0.49|−1.11|0.00|0.00/0.00/+3.33|0/0/−1|2/15|
|D81−D79|0.00|−1.67|+2.00|+0.23|+1.67|−3.33|0.00/−6.67/+3.33|+3/−3/0|—|

D81是第一个相对D62满足预注册严格联合门的ground候选：没有牺牲新类、任何总体/场景/class floor或混淆项，并严格提高`A/H`、降低`F`和old→new。D79的`A/J`边际值更高，但以`N/min-A`下降为代价，不满足联合门；因此D81在联合准则下成为当前最强开发版本。

## 12.逐场景性能

|场景|rows|B|A|N|H|F|J|min-class B/A/N|mean row floor B/A/N|old→new/new→old/new→wrong-new|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
|`leo_clear_weak`|5|98.33|91.67|98.00|94.44|6.67|50.00|90.00/70.00/90.00|90.00/60.00/90.00|2/1/0|
|`leo_low_elev_weak`|5|91.67|80.00|76.00|76.92|11.67|20.00|80.00/60.00/50.00|70.00/60.00/20.00|7/5/7|
|`leo_rain_weak`|5|88.33|76.67|80.00|77.45|11.67|10.00|60.00/30.00/70.00|60.00/30.00/30.00|13/2/8|

相对D62，clear和rain所有指标完全相同；low-elev为`A+1.67pp`、H`+0.94pp`、F`−1.67pp`，`B/N/J`及全部floor不变。增益没有跨场景扩散，但也没有D78-D80常见的新类负收益。

## 13.逐类性能与遗忘

|角色|类/真实TX|B|A或N|遗忘B−A|相对D62|
|---|---|---:|---:|---:|---|
|old|`cls_75aa…`/14-10|96.67|93.33|3.33|不变|
|old|`cls_8b02…`/14-7|80.00|53.33|26.67|不变|
|old|`cls_1f33…`/20-15|96.67|90.00|6.67|不变|
|old|`cls_f8df…`/20-19|93.33|93.33|0.00|不变|
|old|`cls_a53c…`/6-15|93.33|73.33|20.00|不变|
|old|`cls_33bb…`/8-20|96.67|93.33|3.33|A`+3.33pp`|
|new|`cls_09f8…`|—|73.33|—|不变|
|new|`cls_1c2a…`|—|93.33|—|不变|
|new|`cls_b8fb…`|—|76.67|—|不变|
|new|`cls_d3af…`|—|90.00|—|不变|
|new|`cls_f608…`|—|90.00|—|不变|

唯一性能变化是low-elev fold0中旧类`8-20`少1次old→new，聚合A提高3.33pp。最差旧类仍为`14-7=53.33%`，最差新类仍为`cls_09f8…=73.33%`；因此D81解决了一个遗忘错误，但尚未修复通用floor。

## 14.机制表现

|证据|before|final|解释|
|---|---:|---:|---|
|outer最大中心平移L2均值|0.05381|0.05726|新增类后仍为小幅中心修正|
|所有类中心平移L2均值|0.03312|0.03157|不是只修旧类|
|最小归一化support权重均值|0.03143|0.02601|会抑制明显ground-aligned离群样本|
|最大归一化support权重均值|0.18709|0.19484|未形成单样本支配|
|effective support size均值|7.377/8|7.333/8|保持大部分K-shot信息|
|类内残差最大误差|max`2.78e−17`|max`2.78e−17`|target协方差数值上不变|
|FFT96/RF32误差|0|0|其他视图bitwise不变|
|D62 active fit|7/15|3/15|较D62历史5/15、1/15增加|
|D62 accepted rows|20|6|稳健中心让更多Fisher residual row通过support OOF门|

ground谱使用全部84 cells、14个有效domain、6个ground类；原始残差rank78、effective rank13.6446，固定保留14维和79.76%信号trace。真实全闭包中最大中心平移0.06401，最小单样本权重0.02009，但最小effective sample size仍为6.965/8。结果支持“地面原型更适合评估support可靠性，而不是直接定义query距离”的假设。

## 15.量化、训练与资源

|项目|结果|结论|
|---|---:|---|
|INT8/FP32 outer argmax差异|0|通过|
|INT8/FP32 margin sign flip|0|通过|
|before/final support argmax差异|0/0|通过|
|max score abs error|min/mean/max=`0.000425/0.000810/0.001773`|量化误差小|
|完整training trace|20条/row，300条|无截断|
|loss min/mean/max|0.07560/0.30719/1.11738|沿用D42 Stage2-B训练|
|support acc min/mean/max|89.58/98.77/100%|训练正常|
|新增optimizer step/parameter|0/0|闭式注册|
|trainable/peak parameters|2,016/2,016|≤80k|
|epoch/total optimizer steps|20/20|≤30/≤50|
|持久状态|34,011B|8,583B affine＋25,428B ground，≤256KB|
|ground谱统计MAC|90,521,600|一次性适配|
|support中心平移MAC上界|21,890,560|低秩14维投影|
|D81总新增适配MAC|112,412,160|约为D80新增141,099,008的79.7%|
|总适配MAC|25,003,636,130|D62闭式链占主导|
|query MAC/新增MAC|6,624/0|部署仍为单affine|
|ground basis瞬时FP64|18,032B|不持久化|
|CUDA峰值|22,886,912B|本地cuda:0实测|
|dense query graph/query fit rows|0B/0|协议通过|

## 16.缺陷、判定与下一步

D81通过开发联合门，但增益仅来自15个outer row中的1个预测、1个low-elev fold、1个旧类；绝对性能仍远低于正式目标：`A=82.78%<92%`、`N(new5)=84.67%<92%`、`min-A=53.33%<88%`。当前ground组件还未联合封存和外部authority签名，不能成为formal candidate。

因此不运行125，也不作正式晋级。按预注册进入第二独立seed、同receiver/new5/K10/3场景×5fold复核；要求再次相对该seed matched D62满足同一严格联合门。若第二seed不复现，D81降为单seed偶然改进；若复现，再考虑receiver扩展和正式ground封存，而不是扫描Cauchy系数。
