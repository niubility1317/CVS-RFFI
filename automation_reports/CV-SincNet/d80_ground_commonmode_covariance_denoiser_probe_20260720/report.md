# D80地面跨域质心漂移协方差实验报告

## 1.实验身份

|字段|值|
|---|---|
|实验ID|`d80_ground_commonmode_covariance_denoiser_probe_20260720`|
|候选|`ground_commonmode_covariance_denoiser`|
|operator|Codex `/root`|
|状态|`IMPLEMENTED_TESTED_NOT_RUN`|
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

预期105行、30个target fit、1,080个D62 component fit；每个held fit在排除对应physical rank后独立重算target covariance，query0。实现提交和完整精确命令在启动前补录。
