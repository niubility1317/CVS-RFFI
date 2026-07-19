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

## 7.clean验证与运行锁

- 实现commit=`e2fd8cf8580f3072529460295fb187b7b7a3d0dc`；clean worktree=`E:\type10-7\code\snapshots\d75wt`，detached HEAD且clean。
- clean D42–D75相邻完整链43文件、392项全部通过，用时82.5秒；core/probe `py_compile`通过。
- clean执行SHA：D75 core=`a41456c85437125203a54d069d90dcbebc6462df4519e77b5f4cbbed6fdbc99a`、D75 probe=`6e14688f1049b67c3da57b80d5a9636ca8e27263bc9cc508b43a69dc3147af51`、D74 helper=`427be77328700c524173689567423b861bd18dd57fb8d96d7a4fcd5c6d4e363d`、D62 helper=`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。
- 启动前输出目录不存在，无D73/D74/D75 Python任务；GPU0 RTX5070Ti显存`1097/16303MiB`、利用率0%。本轮本地执行，不访问N607。
- 预期闭包：105行、30个target row、30次top fit、1080次D62 component execution、每个target row 8次LOO LDA和8次LOO方向；ground/query-fit/clean/source/role/quota访问0。
- 每个target row预期新增：LOO LDA MAC`249,495,552`、LOO方向MAC上界`111,817,728`、full方向＋编译`18,190,656`；相对D62总新增`379,503,936`，总适配MAC`25,270,727,906`，query/state增量0。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d75wt\code\scripts\probe_d75_crossfitted_margin_safe_nuisance_projection.py' `
  --d75-arm crossfitted_margin_safe_nuisance_projection `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d75wt' `
  --before-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only' `
  --before-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json' --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --before-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json' `
  --before-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json' --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only' `
  --after-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json' --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --after-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json' `
  --after-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json' --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding 'E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json' --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output 'E:\type10-7\automation_reports\CV-SincNet\d75_crossfitted_margin_safe_nuisance_projection_probe_20260720\crossfitted_margin_safe_nuisance_projection' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```
