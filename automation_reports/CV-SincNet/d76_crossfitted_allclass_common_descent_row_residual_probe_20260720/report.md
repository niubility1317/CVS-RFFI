# D76全类共同下降row residual实验报告

## 1.实验身份与状态

|字段|值|
|---|---|
|实验ID|`d76_crossfitted_allclass_common_descent_row_residual_probe_20260720`|
|候选|`crossfitted_allclass_common_descent_row_residual`|
|operator|Codex `/root`|
|状态|`DEFERRED_BY_USER_DIRECTION_BEFORE_IMPLEMENTATION`|
|目标|将全注册类OOF CE共同下降方向直接编译到D62最终类行，同时改善旧类适应、新类注册与通用floor|
|比较目标|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.协议、机制与开发门

- 复用D18 `VALIDATED_ONCE/p2_min_v1`，receiver`20-1`、seed`713101`、K10/new5、3场景×5fold、actual K8；不重验数据。
- 每个物理rank用K−1 support拟合统一LDA，在88个held support上形成11个类CE梯度；用20次固定Frank-Wolfe求minimum-norm共同下降方向，解析Lipschitz步长和类无关trust cap后更新D62 final rows。
- 公式对类置换等变，不知道old/new角色，不读取query、clean/source或ground组件；query MAC/state增量0。
- 相对D62的`A/N/H/min-A/min-N`均不退化、`F`不升且至少一项严格改善才晋级；否则关闭，不扫参数、不跑125。

|candidate|机制|receiver/TX|K/seed|B|A|N|H|F|J|min-B/A/N|资源|判定|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D76|D62固定头＋全类OOF共同下降row residual|20-1/new5|K10/713101|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|

## 3.版本、验证与运行占位

`E:\type10-7`不是Git仓库；设计、代码、测试和完整报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`。实现后补录commit、clean worktree、source SHA、完整命令、PID/GPU/log/output、105行闭包、共同下降审计、详细性能、资源、artifact和最终判定。

## 4.路线变更

2026-07-20用户要求优先研发更高效、创新的地面压缩原型域适应。D76尚未实现、未运行、无性能结果；保留为D77的target-only matched control，不得写成完成实验。
