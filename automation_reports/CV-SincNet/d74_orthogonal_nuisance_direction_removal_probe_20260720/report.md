# D74类中心正交nuisance方向删除实验报告

## 1.实验身份与状态

|字段|值|
|---|---|
|实验ID|`d74_orthogonal_nuisance_direction_removal_probe_20260720`|
|候选|`orthogonal_nuisance_direction_removal`|
|operator|Codex `/root`|
|状态|`PREREGISTERED_NOT_RUN`|
|目标|删除一个不承载类中心差异、但具有最大类内残差能量的非可逆方向，检验能否突破D62/D73等价边界|
|比较目标|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.协议与机制锁

- `p2_min_v1`、D18匹配`VALIDATED_ONCE` capsule、receiver`20-1`、seed`713101`、K10/new5、3场景×5fold，outer-fit实际K8。
- 单LEO_weak观测、support-only、query逐样本全注册类argmax；无clean/source/query truth/role/quota/global assignment。
- before精确D62；final在D42特征中删除一个与中心化类均值span正交的最大类内残差方向，D62 refit后把投影编译进单一int8头。
- rank固定1，无阈值、强度、场景、类、角色或结果扫描；地面组件输入0。

## 3.开发门

相对D62要求`A/N/H/min-A/min-N`不退化且至少一项严格提高，同时`B/F`、场景和混淆无交换伤害。失败即负向关闭，不开第二seed或125矩阵。

## 4.版本、验证、运行和结果占位

`E:\type10-7`不是Git仓库；所有代码、测试、追溯和完整报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`，根目录只保留同步镜像。实现后补录commit、clean worktree、source SHA、完整命令、PID/GPU/log/output、105行闭包、7候选、3场景、11类、15fold、机制、训练、量化、资源、artifact、缺陷和最终判定。

|candidate|机制|receiver/TX|K/seed|B|A|N|H|F|J|min-B/A/N|混淆|资源|判定|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D74|rank-1非可逆nuisance删除＋D62 refit|20-1/new5|K10/713101|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|
