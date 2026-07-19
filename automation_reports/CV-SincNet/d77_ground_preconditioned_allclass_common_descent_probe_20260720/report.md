# D77地面预条件全类共同下降实验报告

## 1.实验身份

|字段|值|
|---|---|
|实验ID|`d77_ground_preconditioned_allclass_common_descent_probe_20260720`|
|候选|`ground_preconditioned_allclass_common_descent`|
|operator|Codex `/root`|
|状态|`PREREGISTERED_NOT_RUN`|
|目标|高效利用地面int8域×类原型定义优化几何，以全注册类target-support OOF共同下降直接修正D62最终边界|
|matched baseline|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.假设与单一主要差异

D66证明静态地面可靠性缩放能略微保护旧类，却压低新类与new floor。D77不把地面统计应用到特征后重新拟合，而把它作为target OOF多类梯度的正定预条件器；地面决定坐标可信度，11类target support共同决定方向。相对D62只增加一个直接编译到final rows的地面预条件共同下降residual。

## 3.协议与资格边界

- 复用D18 `VALIDATED_ONCE/p2_min_v1`，receiver`20-1`、seed`713101`、K10/new5、3场景×5fold、actual K8。
- 单LEO_weak observation、support-only、query独立全类argmax；clean/source/query truth/role/quota/global reassignment访问0。
- 当前D19历史地面组件SHA为`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`，84个有效cell、逻辑状态25,428B，但manifest仍为`formal_phase2_eligible=false`和`UNVERIFIED_UNDER_CURRENT_PROTOCOL`。因此本轮只能是development diagnostic，不产生formal性能声明。

## 4.开发门与结果占位

|candidate|机制|receiver/TX|K/seed|B|A|N|H|F|J|min-B/A/N|状态/MAC|判定|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D77|D62＋ground-M预条件11类OOF共同下降|20-1/new5|K10/713101|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待审计|待跑|

相对D62要求`A/N/H/min-A/min-N`均不退化、`F`不升且至少一项严格改善，三类混淆不得交换伤害。失败即关闭D77，不扫参数、不运行第二seed或125。

## 5.版本与运行占位

`E:\type10-7`根目录不是Git仓库；代码、追溯和Git版报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`，根报告镜像到本目录。实现后补录commit、clean worktree、测试、运行命令、PID/GPU、完整105行、逐场景/逐类/15fold/混淆/量化/资源表和最终判定。
