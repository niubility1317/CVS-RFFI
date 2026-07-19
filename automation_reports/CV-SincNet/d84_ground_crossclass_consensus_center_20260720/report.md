# D84地面跨类一致域模板稳健中心

状态：`COMPLETED_DIAGNOSTIC_PERFORMANCE_NEUTRAL_EFFICIENCY_POSITIVE_NOT_PROMOTABLE`。实验ID：`d84_ground_crossclass_consensus_center_20260720`；时间：2026-07-20 06:35 HKT；操作者：Codex。

## 目标与历史边界

D83的统一协方差loading相对D81在15/15 outer rows上零预测变化，却增加114.26M adaptation MAC；D78/D79直接地面切向logit residual曾用旧类收益交换新类损失；D80/D82直接修改协方差或残差均已判负。D84不重启这些路线，只沿D81已在两个seed复现的center-only正收益机制继续研发。

## 单一机制差异

地面bundle包含14 domains×6 old classes的84个int8中心。D84先对每个ground class跨domain去中心，随后在每个domain内对6类残差求共同漂移`g_d`；以

```text
rho_d = ||g_d||^2 / (||g_d||^2 + mean_c ||r_dc - g_d||^2 + eps)
pi_d = rho_d / sum_d rho_d
energy_i = sum_d pi_d * <x_i - mean_y, normalize(g_d)>^2
```

替换D81的全84-cell协方差特征谱。ground类中心和类特有漂移在生成模板后丢弃，不做target identity映射、不产生旧类分数。target每类仍使用一步Cauchy权重，只平移z160类共同中心，类内残差和FFT96/RF32逐位保持；最终仍由target support D62产生单个INT8 affine head。

## 创新性与效率假设

- 跨地面类一致性只保留能够跨6个旧类复现的domain drift，过滤Phase1旧类身份交互，预期比D81的混合协方差对新类更公平。
- 14个domain模板直接闭式构造，无160×160协方差和特征分解、无rank/weight/强度扫描；target translation维度不超过14。预期地面统计从D83的90.52M MAC降到低于0.2M，且不再有1.84M covariance loading。
- K≤2精确identity；新旧类同式；query/clean/source/role/quota/global assignment访问0；ground只读且不写回。

## 预注册开发门

固定复用D18 `VALIDATED_ONCE` cell：receiver=`20-1`、seed=`713101`、K10（actual K8）、new5、3场景×5fold。相对D81要求总体及每场景B/A/N/H/J、全部class floors和mean-row floors不退化、F不升、三类混淆不增加，且A/H/F、rain、最差旧类或新类中至少一项严格改善。未通过即停止第二seed和125；ground组件仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，任何结果仅为development diagnostic。

## 版本、验证与待办

`E:\type10-7`根不是Git仓库；实现位于隔离Git worktree`E:\type10-7\code\snapshots\d81wt`，将精确提交并cherry-pick到`E:\type10-7\github_publish\CVS-RFFI-repo`。核心SHA256=`5bebd29643767ee349059dbf80e0208a349a5e8870ea599c3d5e96cd63e17dff`；probe SHA256=`0642252d34d9c50c90aecb1478163f66304f355415ed3a4c8e86cedb4305db30`。D84专项12/12、D42/D62/D81-D84相邻链87/87 PASS，`py_compile`与`git diff --check`PASS。

真实ground只读加载确认：26个registry domain槽位中14个有效domain、84个有效cell，保留14个一致域模板；跨类一致性`rho` min/mean/max=`0.14591/0.17696/0.21422`，模板权重`0.05889..0.08647`；template SHA256=`9a36dd9b85841282987fb8093dd3fe5daf8003c945995dfd5b0f651f8caa6eb4`，weight SHA256=`adcbb4724213c5d80d825756b4b17c3605c18ea8c35d0ccf4a38f9be264aafea`。地面统计上界179,200 MAC，较D83的90.52M减少99.80%；加上预计21.89M center translation，D84新增适配约22.07M MAC，较D83的114.26M减少约80.7%。

下一步在独立输出`ground_crossclass_consensus_center/`执行真实105-row实验；未完成前无性能结果。

## 完成结果

真实运行完整105/105 rows，runner耗时123.05秒、外层131.1秒；stderr 0B，stdout 5,119B，完整日志无Traceback/OOM/NaN/Inf。D84与D81的15/15 outer prediction hash完全相同，未满足至少一项性能严格改善的预注册门，不运行独立seed或125；但以相同性能把ground相关适配MAC降低80.4%，形成独立的效率正结果。

### 总体同row性能

|版本|B旧类注册前|A旧类注册后|N新类|H_old_new|F遗忘|J|min class B/A/N|mean-row floor B/A/N|old→new/new→old/new→new|
|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D62|92.78%|82.22%|84.67%|82.6238%|10.56%|26.67%|80.00/53.33/73.33%|73.33/50.00/46.67%|23/8/15|
|D81/D83|92.78%|82.78%|84.67%|82.9366%|10.00%|26.67%|80.00/53.33/73.33%|73.33/50.00/46.67%|22/8/15|
|D84|92.78%|82.78%|84.67%|82.9366%|10.00%|26.67%|80.00/53.33/73.33%|73.33/50.00/46.67%|22/8/15|
|D84−D81|0|0|0|0|0|0|0/0/0|0/0/0|0/0/0|

D84相对K10/new5目标的缺口仍为`A −9.22pp`、`minA −34.67pp`、`new5 −7.33pp`。相对D62仅low-elev fold0减少1次old→new，与D81相同；没有把效率提升误报为性能提升。

### 逐场景性能

|场景|B|A|N|H|F|J|min class B/A/N|mean-row floor B/A/N|混淆old→new/new→old/new→new|
|---|---:|---:|---:|---:|---:|---:|---|---|---|
|clear|98.33%|91.67%|98.00%|94.441%|6.67%|50.00%|90/70/90%|90/60/90%|2/1/0|
|low-elev|91.67%|80.00%|76.00%|76.922%|11.67%|20.00%|80/60/50%|70/60/20%|7/5/7|
|rain|88.33%|76.67%|80.00%|77.447%|11.67%|10.00%|60/30/70%|60/30/30%|13/2/8|

上述每项均与D81/D83相同；主要性能缺陷仍是rain旧类after最低30%、low-elev新类最低50%，以及总体最弱旧类`14-7`仅53.33%。

### 逐类性能

|TX|角色|B|A/N|D84−D81|
|---|---|---:|---:|---:|
|14-10|旧类|96.67%|93.33%|0|
|14-7|旧类|80.00%|53.33%|0|
|20-15|旧类|96.67%|90.00%|0|
|20-19|旧类|93.33%|93.33%|0|
|6-15|旧类|93.33%|73.33%|0|
|8-20|旧类|96.67%|93.33%|0|
|1-16|新类|—|93.33%|0|
|1-18|新类|—|73.33%|0|
|18-10|新类|—|90.00%|0|
|14-11|新类|—|76.67%|0|
|8-3|新类|—|90.00%|0|

### 七候选、训练与量化

|candidate|B|A|N|H|F|min A|min N|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|D42-USLDA-INT8|92.78%|82.78%|84.67%|82.94%|10.00%|53.33%|73.33%|D84目标，性能中性/效率正|
|D42-USLDA-FP32-MATCHED|92.78%|82.78%|84.67%|82.94%|10.00%|53.33%|73.33%|与INT8 outer argmax一致|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78%|75.56%|72.67%|73.35%|12.22%|60.00%|40.00%|诊断baseline|
|D42-D40-HNBR-INT8-NEGATIVE|85.56%|85.00%|15.33%|25.16%|0.56%|63.33%|0.00%|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|86.11%|20.56%|78.67%|31.50%|65.56%|0.00%|36.67%|旧类崩溃|
|ProtoNet-CDA/Z0|71.11%|48.33%|52.67%|48.97%|22.78%|13.33%|3.33%|弱baseline|

- 20步trace×15 rows，共300条；loss min/mean/max=`0.07560/0.30719/1.11738`，support accuracy=`89.58/98.77/100%`。
- FP32/INT8 before、support和outer argmax变化均为0，margin sign flip=0；score绝对误差min/mean/max=`0.000433/0.000924/0.001769`。零性能差异不是量化噪声抵消。

### 地面机制与连续影响

- 26个registry domain槽位中14个有效domain、84个有效cell，保留14个跨地面类一致域模板；跨类一致性`rho=0.14591..0.21422`，归一化模板权重`0.05889..0.08647`。ground类中心与类特有残差不进入target类别分支。
- 实际执行1,080个D62 full/block component fits和2,160次center transforms。中心平移L2总体`0.00145..0.06234`，有效样本数`7.0038..7.8150`；类内残差误差≤`2.78e-17`，FFT96/RF32误差0。
- D84不是恒等实现：相对D81，before连续系数最大绝对差的15-row min/mean/max=`0.000466/0.04753/0.18497`，final为`0.001166/0.003631/0.006736`；但D62 before/final接纳mask均0/15变化，最终argmax也0/15变化。跨类共识模板成功过滤了旧类特异方向，却没有提供新的纠错方向。

### 资源与效率

|项目|D81|D83|D84|D84结论|
|---|---:|---:|---:|---|
|ground统计MAC|90,521,600|90,521,600|179,200|较D81/D83减少99.80%|
|center/额外fit MAC|21,890,560|23,733,760|21,890,560|无协方差loading|
|总新增适配MAC|112,412,160|114,255,360|22,069,760|较D81减少80.37%，较D83减少80.69%|
|总适配MAC|25,003,636,130|25,005,479,330|24,913,293,730|三者最低|
|query MAC/额外query MAC|6,624/0|6,624/0|6,624/0|部署不增负担|
|params/epochs/steps|2,016/20/20|2,016/20/20|2,016/20/20|均过门|
|state/peak CUDA|34,011B/22,886,912B|相同|相同|均过门|

D84以约D81五分之一的ground相关适配成本复现其全部离散性能，是当前最有效率的地面原型使用版本；但项目主目标是性能与floor，故只能记为效率Pareto改进，不能作为性能晋级或125依据。

### 协议、缺陷与结论

ground NPZ/manifest进出hash逐位不变；新旧类同式、K1/K2恒等、query/clean/source/role/quota/global assignment访问均为0，最终仍是单一INT8 affine head。ground组件依旧`UNVERIFIED_UNDER_CURRENT_PROTOCOL`且`formal_phase2_eligible=false`，所有结论仅为development diagnostic。

核心缺陷：v1 ground质心能提供一个便宜、稳定的异常样本加权几何，但无radius/count/类内散度，也没有与target新类身份对齐的信息。D81与D84连续中心不同却落入同一离散决策区，说明继续更换全局ground能量汇聚方式收益已饱和。D84不进入第二seed或125；按性能D81/D84并列，按效率D84占优，但两者均远未满足项目目标。

下一轮不应扫描D84权重或放大平移。更合理的研发分叉是：一是回到Phase1合法重新封存带聚合radius/dispersion的int8 bundle，为Stage2提供当前缺失的不确定度；二是在现有v1限制下，把D84仅作为低成本D81替代基线，研发完全由target support交叉拟合驱动的困难类保护机制，ground不再承担纠错方向。

### 证据与版本

- training log：18,388,558B，SHA256=`679758e1378d7b1ef7046c0cf4d5824b389282448a58b40b0a881545de788fb3`；receipt 5,220B，SHA256=`fc6655d86ea13eed10998ecb1151b95a4cfc115d847652a8e34e2e8b7fa8057e`。
- metadata 6,106B，SHA256=`0be48f6dc8128d0fe01e8acaed2ceaee310af18b41dfaae152a5aaceedf70a88`；完整汇总82,531B，SHA256=`19e64b23a774816d7f188a0dd037195bcac65460f6bacabf90b6360af2d3cfcb`；汇总器SHA256=`f34f5893c6e6a01b9e5822a45ee2c00875c4367ac311bfc81221f3782c46a3d6`。
- 实现提交：worktree`20f10446`，发布仓`aacfe5da`；完成报告和汇总器将另行精确提交。
