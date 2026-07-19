# D78地面域切向最差类边界实验报告

## 1.实验身份

|字段|值|
|---|---|
|实验ID|`d78_ground_tangent_worstclass_margin_probe_20260720`|
|候选|`ground_tangent_worstclass_top2_margin`|
|operator|Codex `/root`|
|状态|`PREREGISTERED_NOT_RUN`|
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
|D78|D62＋ground tangent smooth-worst top-2 residual|20-1/new5|K10/713101|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待审计|待跑|

相对D62要求`A/N/H/min-A/min-N`均不退化、`F`不升且至少一项严格改善，三类混淆不得交换伤害。失败即关闭D78，不扫参数、不运行第二seed、125或N607。

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
- 精确运行命令沿用D77已锁定的D18 before/after capsule、seal、policy、authorization、class binding和D22 component参数，只把入口换为`probe_d78_ground_tangent_worstclass_margin.py`，增加`--d78-arm ground_tangent_worstclass_top2_margin`，并把`--output`换为上述D78独立目录。完整PowerShell进程参数与PID在启动时追加，禁止覆盖已有输出。
