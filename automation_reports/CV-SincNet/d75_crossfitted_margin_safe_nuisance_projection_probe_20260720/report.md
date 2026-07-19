# D75交叉拟合margin安全nuisance投影实验报告

## 1.实验身份与状态

|字段|值|
|---|---|
|实验ID|`d75_crossfitted_margin_safe_nuisance_projection_probe_20260720`|
|候选|`crossfitted_margin_safe_nuisance_projection`|
|operator|Codex `/root`|
|状态|`PREREGISTERED_NOT_RUN`|
|目标|以全注册类nested support-held margin安全门过滤D74非可逆方向，同时保护旧类适应、新类注册和通用floor|
|比较目标|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.协议与数据复用

- `p2_min_v1`、D18匹配`VALIDATED_ONCE` capsule、receiver`20-1`、seed`713101`、K10/new5、3场景×5fold，outer-fit实际K8。
- 单LEO_weak观测、support-only、query逐样本全注册类argmax；无clean/source/query truth/role/quota/global assignment。
- 数据字节、物理ID、receiver/TX、场景、K、support/query split和schema均未变化，不触发重复数据验证。
- 地面int8组件输入0；D22未获正式资格，不能用于D75候选选择或状态更新。

## 3.机制锁

每个类内物理rank用其余K−1样本同时拟合equal-prior shrinkage LDA和D74方向，并在每类一个held样本上比较固定头投影前后的true-vs-best-other margin。只有全部类平均margin、全体平均margin和held正确数均不退化才接受full-support rank-1投影，否则精确回退D62。容差仅为机器舍入界；无可调阈值、rank、强度、角色或场景分支。

## 4.开发门与结果占位

要求相对D62的`A/N/H/min-A/min-N`均不退化、`F`不升且至少一项严格提高；失败即负向关闭，不开第二seed或125矩阵。

|candidate|机制|receiver/TX|K/seed|B|A|N|H|F|J|min-B/A/N|安全门|资源|判定|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D75|D62固定头＋nested margin安全rank-1投影|20-1/new5|K10/713101|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|

## 5.版本、验证与运行占位

`E:\type10-7`不是Git仓库；设计、代码、测试和完整报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`。实现后补录commit、clean worktree、source SHA、完整命令、PID/GPU/log/output、105行闭包、机制门、训练、量化、资源、artifact、完整性能和最终判定。

## 6.实现与主工作树验证

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d75_crossfitted_margin_safe_projection.py`|物理rank留一、LDA margin、全类floor安全门与identity回退|`8b4a59ca9b7ded3f144f592dfe710570e595d1e3864814dfc403733b6e60fc46`|
|`code/scripts/probe_d75_crossfitted_margin_safe_nuisance_projection.py`|D62/D74包装、资源核算、Runner闭包和metadata|`0e9b08b410305879153d5d5e936cdc5e6d5ead1da5f4d644b1df4ea0c610d0cc`|
|`tests/test_stage2_d75_crossfitted_margin_safe_projection.py`|held margin拒绝/接受、rank交错、对称support fail-closed|`ae4e8fdddef37d7b4b47c2ad5b18064f63ad28457010f470f722501b52b3d7f5`|
|`tests/test_probe_d75_crossfitted_margin_safe_nuisance_projection.py`|公式、继承结构和query/state/ground闭包|`4aa7a9bb347fbd79cb675ef2706e6faadf21059d4326edc4f915561d93e07751`|

- `ssr-gpu`专项7/7通过，core/probe `py_compile`通过。
- D42–D75相邻完整链43文件、392项全部通过，用时82.2秒；显式仓内basetemp，无数据重验。
- 实现不扫描margin阈值、rank或强度；门限仅为机器舍入界。每个target row预期8次LOO LDA、8次LOO方向、88个held support margin，optimizer/epoch仍为20/20，query额外MAC/state0。
