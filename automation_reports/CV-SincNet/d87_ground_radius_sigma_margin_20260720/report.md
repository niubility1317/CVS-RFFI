# D87地面半径sigma-point margin head研发报告

## 实验身份与目标

|字段|值|
|---|---|
|实验ID|`D87_ADV3B02_GROUND_RADIUS_SIGMA_MARGIN_HEAD`|
|时间/执行者|2026-07-20 CST/Codex|
|目标|把D85压缩地面原型的14个类无关域方向和真实p90半径直接用于target support分类边界，而不是继续做共享prototype平移|
|比较目标|D78、D79、D81、D85、D86同receiver20-1、seed713101、K10实际K8、new5矩阵|
|状态|`IMPLEMENTED_LOCAL_VALIDATED_PENDING_105_ROW_RUN`|

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

待105行完成后更新。
