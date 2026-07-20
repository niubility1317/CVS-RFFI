# D87地面半径sigma-point margin head研发报告

## 实验身份与目标

|字段|值|
|---|---|
|实验ID|`D87_ADV3B02_GROUND_RADIUS_SIGMA_MARGIN_HEAD`|
|时间/执行者|2026-07-20 CST/Codex|
|目标|把D85压缩地面原型的14个类无关域方向和真实p90半径直接用于target support分类边界，而不是继续做共享prototype平移|
|比较目标|D78、D79、D81、D85、D86同receiver20-1、seed713101、K10实际K8、new5矩阵|
|状态|`COMPLETED_DIAGNOSTIC_OLD_GAIN_NEW_REGRESSION_NOT_PROMOTABLE`|

## 锁定机制与创新性

从D85 v2组件重构14×6中心和半径，按D84/D86形成14个跨ground类共识单位方向`u_d`，随后丢弃ground类身份。每个domain固定`r_d=median_c(r_d,c)`、`a_d=sqrt(2r_d)`，对同一support特征形成不增加K的数学view：`z`、`z+a_d u_d`、`z-a_d u_d`。

每个physical rank整体进入一个leave-one-rank OOF fold；正负view不能跨fold。对全注册类交叉熵使用唯一sigma期望：

`R=mean_d[0.5*CE(z)+0.25*CE(z+a_d*u_d)+0.25*CE(z-a_d*u_d)]`。

权重不是超参数：三点分布均值严格为`z`，协方差严格为`r_d u_d u_d^T`。14个domain等权；不再乘D85 reliability，避免重复使用半径。对逐类sigma风险用初始均值温度的smooth-worst聚合，在有效rank-13 span内固定20步确定性回溯；trust ball沿用D78。残差逐列class-mean-zero，按D79中心化编译`delta_b=-delta_W*support_mean`，最终仍是单一INT8 affine head。

D87与D83二次covariance loading不同：它保留softmax/CE曲率和每个support当前全类margin；与D84-D86不同：它直接改变决策边界；与D78/D79不同：方向幅度由真实v2 p90半径锁定，并用对称sigma风险而非单向top-2推力。没有radius/rank/loss/step/trust扫描。

## 协议和正式性边界

- `p2_min_v1`；复用匹配`VALIDATED_ONCE`D18 capsule，不重验数据。
- 单物理样本单`LEO_weak`接收观测；counterfactual仅是固定received IQ特征数学view，不增加K。
- support-only、query0；无clean/source、query truth、role Oracle、class quota、全局重分配。
- target-old/new完全同式，类标签置换等变；无ground→target身份映射。
- v2组件状态仍为`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`，本轮强制development diagnostic、不可推广。
- checkpoint/组件manifest/pre-sign root分别锁定为`2699eedc...d59c98`、`6990ce8c...95f112`、`098badd1...d0055`。

## 实现、验证与Git

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d87_ground_radius_sigma_margin.py`|v2 sigma几何、分组OOF非二次margin优化、中心化affine残差|`3c1a93d3fd4e02ae8772fe112ab3b13ddcb5abfc960fc0f607a5fa8762a6d12b`|
|`code/scripts/probe_d87_ground_radius_sigma_margin.py`|D79/D62 runner接入、v2 loader、资源/量化/协议闭包|`57bf950ce3b4230c36ec2f3e4fd063aa9ece4b338a4ffcb253e13f45c49a1210`|
|`tests/test_stage2_d87_ground_radius_sigma_margin.py`|sigma协方差、目标单调性、中心化、类置换、K1回退|`46d7ec5b7646e6f1d66a2a9e265a8eb8d2c0ea4d513c5ccb18812b4893680395`|
|`tests/test_probe_d87_ground_radius_sigma_margin.py`|锁定公式、资源上界、v2/query-free闭包|`450f0b153712afb81dd9e94fdf7cfe27ae62a147b97534d1058d7dc3dd0dcfd6`|

`ssr-gpu`下py_compile通过；D87+D79+D86相邻测试20项全部通过。真实v2组件的sigma span已只读验证为`[288,13]`，不是合成占位。根目录非Git；本报告与实现先进入干净`d81wt`提交，再精确cherry-pick到主发布仓库，既有无关脏改动不动。

## 预登记运行与停止门

仅运行receiver`20-1`、seed`713101`、K10实际K8、new5、3场景×5fold、7候选，共105行。本地`ssr-gpu`、runtime-root=`d41wt`、probe-root=`d81wt`，输出根为`E:\type10-7\automation_reports\CV-SincNet\d87_ground_radius_sigma_margin_20260720`；本轮不访问N607。

相对D81/D85基线`B/A/N/H/F/J=92.78/82.78/84.67/82.9366/10.00/26.67%`，seed2必须同时满足：总体和三场景`B/A/N/H/J/min-class/row-floor`均不下降、`F≤10%`、三类混淆不增加；至少一项A/H/minA/J/rain-A/old→new严格改善；至少1/15 outer预测变化且old/new paired transition无净负；INT8/FP32 outer argmax和margin sign flip均为0；ground预处理<0.5M MAC、D87额外适配<5%D62、状态≤256KB、query额外MAC/state=0/0。

任一门失败即停止seed2/125，不扫描radius倍率、sigma权重、rank、step、trust或loss权重。完成后报告七候选、三场景、15 row、逐类遗忘、混淆、OOF clean/sigma风险、量化和资源。

## 运行结果

attempt0完成105行后仅因证据层把未改变的D62 covariance结构误标为D87而失败，原始证据保留；修复提交`43ce0d04`只恢复正确结构。retry1耗时131.84秒完成105/105行，stderr=0B，证据闭包通过。

### 七候选联合指标

|candidate|B/A/N/H/F/J|min B/A/N|row floor B/A/N|o→n/n→o/n→wrong-n|
|---|---|---|---|---|
|D42-USLDA-INT8|92.78/85.00/83.33/83.58/7.78/30.00|80/60/73.33|73.33/53.33/43.33|18/10/15|
|D42-USLDA-FP32-MATCHED|92.78/85.00/83.33/83.58/7.78/30.00|80/60/73.33|73.33/53.33/43.33|18/10/15|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78/75.56/72.67/73.35/12.22/23.33|80/60/40|53.33/33.33/36.67|33/22/19|
|D42-D40-HNBR-INT8-NEGATIVE|85.56/85.00/15.33/25.16/0.56/0|66.67/63.33/0|40/40/0|2/0/0|
|D42-D41-BEC-INT8-NEGATIVE|86.11/20.56/78.67/31.50/65.56/0|76.67/0/36.67|46.67/0/26.67|142/0/32|
|D42-PROTOnet-CDA-ZID160|71.11/48.33/52.67/48.97/22.78/0|33.33/13.33/3.33|13.33/0/0|0/0/0|
|Z0_SUPPORT_ONLY|71.11/48.33/52.67/48.97/22.78/0|33.33/13.33/3.33|13.33/0/0|0/0/0|

### 场景和15 row

|场景|B/A/N/H/F/J|min B/A/N|row floor B/A/N|混淆|
|---|---|---|---|---|
|clear|98.33/91.67/96.00/93.567/6.67/50|90/70/90|90/60/80|2/2/0|
|low|91.67/81.67/76.00/77.764/10/20|80/60/50|70/60/20|6/5/7|
|rain|88.33/81.67/78.00/79.398/6.67/20|60/50/70|60/40/30|10/3/8|

|row|B/A/N/H/F/J|floor B/A/N|混淆|
|---|---|---|---|
|clear0|100/100/90/94.74/0/50|100/100/50|0/1/0|
|clear1|100/83.33/100/90.91/16.67/0|100/0/100|0/0/0|
|clear2|91.67/83.33/90/86.54/8.33/50|50/50/50|1/1/0|
|clear3|100/100/100/100/0/100|100/100/100|0/0/0|
|clear4|100/91.67/100/95.65/8.33/50|100/50/100|1/0/0|
|low0|100/83.33/80/81.63/16.67/50|100/50/50|2/1/1|
|low1|83.33/58.33/70/63.64/25/0|50/50/0|1/0/3|
|low2|83.33/91.67/70/79.38/-8.33/0|50/50/0|0/2/1|
|low3|100/100/70/82.35/0/0|100/100/0|0/1/2|
|low4|91.67/75/90/81.82/16.67/50|50/50/50|3/1/0|
|rain0|83.33/83.33/60/69.77/0/0|50/50/0|2/0/4|
|rain1|100/83.33/80/81.63/16.67/50|100/50/50|2/2/0|
|rain2|91.67/83.33/80/81.63/8.33/50|50/50/50|1/0/2|
|rain3|83.33/83.33/90/86.54/0/0|50/0/50|2/0/1|
|rain4|83.33/75/80/77.42/8.33/0|50/50/0|3/1/1|

逐类B→A：14-10`96.67→93.33`、14-7`80→60`、20-15`96.67→93.33`、20-19`93.33→93.33`、6-15`93.33→76.67`、8-20`96.67→93.33`；新类1-16/1-18/18-10/14-11/8-3=`93.33/73.33/86.67/73.33/90%`。

相对D85，D87改变4/15 row：A`+2.22pp`、N`-1.33pp`、H`+0.64pp`、F`-2.22pp`、J`+3.33pp`、minA`+6.67pp`、new row floor`-3.33pp`；old→new减少4但new→old增加2。clear N下降2pp，rain N下降2pp，属于旧类收益换新类损失。

15/15 fit激活，sigma objective单调下降`0.02385..0.09027`；但逐行最差类clean CE增加`0.0000006..0.01655`。residual Frobenius=`1.0307..1.2566`并触及trust边界；support预测0/15变化，outer却4/15变化。INT8/FP32 outer/support均0 flip，量化门通过。

资源：Stage2-C新增285,487,624 MAC，总适配25,176,711,594 MAC，额外占1.134%；query额外MAC/state=0/0；状态14,399B；参数2,159；总step40、Stage2-C step20；dense query graph0。资源门通过。

性能门因N、clear/rain N、new floor和new→old退化而失败，停止seed2/125，不扫描任何权重/半径/trust。结论为`COMPLETED_DIAGNOSTIC_OLD_GAIN_NEW_REGRESSION_NOT_PROMOTABLE`。下一同族候选只能加入类对称逐类clean-CE非增或margin保护，不能按old/new角色加权。

证据：receipt`29cbfb1f...9a39f21`；log`a5b7dbb0...d8b9e`；metadata`474afe84...c50a6f`；summary`110af923...9ea7d`。完整原始证据位于根报告目录。
