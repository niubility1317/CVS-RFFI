# NEXT-R5 FA-RDCE3→qKNN Target125实验报告

## 1.身份与状态

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r1`
- 日期：2026-08-05
- 当前状态：`LOCAL_VERIFIED / P0=0 / P1=0 / NOT_LANDED / NOT_LAUNCHED`
- 候选：`NEXT-R5-FA-RDCE3-Q-TARGET125`
- 主agent：`gpt-5.6-sol/high`
- 科学实现与独立审查：不同`gpt-5.6-terra/max`agent
- 后续唯一N607 runner：方法、矩阵、命令和路径冻结后使用`Luna/max`
- Git worktree：`E:\fa125wt`；branch=`codex/next-r5-fa-q-target125-20260805`
- 起点commit：`70d775bc`

## 2.用户覆盖与实验目标

用户在2026-08-05明确要求“跑FA-RDCE3+qKNN的125实验”，因此覆盖此前只允许Target5的工作目标，但不改变`p2_min_v1`、received-IQ、support/query物理ID互斥、query零fit/update/selection、全注册类逐query竞争和truth-after-prediction-seal规则。

本实验不调FA的rank、`rho`、Wiener系数、量化或qKNN参数，不运行CER、RPPF或D92-Lite新头。历史formal D92、D62、D91、SVRN只在完全同键时作结果侧比较，不参与FA拟合或选择。

## 3.冻结矩阵草案

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed={713102,713103,713104,713105,713106}
(K,new)={(10,5),(10,10),(10,20),(5,20),(1,20)}
5×5×5=125 outer jobs
每outer覆盖3个leo_*_weak场景×4状态
=375 scene rows / 1500 state prediction surfaces
```

四状态统一为：

|状态码|中文主名称|主指标|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|old BA、old-floor、总正确数；seen-new/H=`N/A`|
|`DA1_REG0`|域适应后/新类注册前|同上|
|`DA0_REG1`|域适应前/新类注册后|old BA、seen-new、H、all-floor、总正确数|
|`DA1_REG1`|域适应后/新类注册后|同上|

K1已由完整Proxy24证明FA负收益，固定严格旁路并以alias receipt闭合；K5/K10使用同一闭式FA，唯一新锁为`FA_FIT_K={5,10}`。K10不改rank、`rho`、Wiener系数或量化；公式闭合不代表收益。K5/K10的FA只由REG0 6个old类support拟合一次并在REG1逐bit复用。

## 4.因果比较与裁决

- 域适应前后：固定注册状态比较`DA1_REG0−DA0_REG0`和`DA1_REG1−DA0_REG1`。
- 注册前后：固定DA状态比较`REG1−REG0`，但REG0的seen-new/H保持`N/A`。
- 主结果按完整125 outer、同row、同query根报告；同时完整列出receiver、seed、K/new、scene和逐class old floor。
- 不按中间accuracy、H、floor停止、重跑或选择slice；只有协议/安全或重复确定性prediction前异常可以技术停止。

## 5.本地实现与验证状态

|项目|当前状态|
|---|---|
|Target125既有input/materializer/scorer复用边界|已实现并通过独立审查|
|FA K10/K1方法锁|已冻结：K5/K10 fit，K1严格旁路|
|四状态125 matrix/runner/CLI|已完成：prepare、smoke、predict-shard、merge、truth-open、score|
|query零fit/update/selection聚焦负测|已通过|
|真实checkpoint无truth smoke|待N607落地后作为第一个运行动作执行|
|独立代码复核|Terra/max：`P0=0，P1=0`|
|本地验证|六个Python入口编译通过；四份聚焦测试`14 passed`；`git diff --check`通过|
|Git提交与release archive|待本报告更新后立即冻结|

实际实现文件为：`stage2_next_r5_fa_target125_matrix.py`、`stage2_next_r5_fa_target125_core.py`、`stage2_next_r5_fa_target125_runtime.py`、`stage2_next_r5_fa_target125.py`、`build_next_r5_fa_target125_asset.py`、`run_next_r5_fa_target125.py`和对应四份聚焦测试。method lock为`configs/next_r5_fa_rdce3_q_target125_20260805.json`，SHA256=`0934b59c81ed5f422de503528d0ef48400210c817fae8c301f55b5e2d2775e34`。

独立审查实际推动关闭四项直接正确性缺陷：truth catalog逐surface物理ID顺序绑定、DA0/DA1同REG query严格一致、REG0 query等于REG1的有序`target_old`子序列、FA资产同时绑定checkpoint与method lock。四状态差分表已覆盖old BA、old/all floor和total correct；REG0的seen-new/H仍严格为`N/A`。

## 6.发布前必要字段

当前实现已通过本地与独立审查；补齐Git commit、release archive、N607直连预检、远端输入存在性和不可覆盖run root后即授权唯一runner启动，不增加其他gate。

已核对的只读复用输入：

|输入|路径|SHA256|
|---|---|---|
|Phase1 588-row strict tap|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|strict tap receipt|同目录`d106_ls_strict_tap.receipt.json`|`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|D92 Target125 matrix|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/matrix_manifest.json`|`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|D92 sealed runtime|由每个sealed package descriptor绑定|`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`|

只读复用D138-r6中已验证的D108 prepared plan/context：plan SHA=`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`，context SHA=`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`。新run只在自己的不可覆盖`prepared/`中写入NEXT-R5绑定，不修改历史文件；这不改变数据，因此不重复执行VALIDATED_ONCE数据验证。

## 7.N607冻结发布拓扑

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r1`，必须不存在且首次创建。
- source CWD：`RUN_ROOT/source`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- D108 prepared候选来源：`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/prepared/target125_plan.json`和同目录`target125_context.json`；runner在启动前核对存在性与上述SHA，若路径已迁移只允许定位同SHA文件，不重建数据。
- FA资产：用本报告列出的D106 strict tap、checkpoint SHA和method lock SHA直接构建到`RUN_ROOT/input/fa_asset`。
- NEXT-R5 prepare：写`RUN_ROOT/prepared`；真实row0/`leo_clear_weak`无truth smoke写`RUN_ROOT/smoke`。
- 8个固定分片：物理GPU i设置`CUDA_VISIBLE_DEVICES=i`，CLI统一传`--device cuda:0 --shard-index i`，输出`RUN_ROOT/shards/shard_i`，i=0至7；每张卡不得使总训练/预测任务超过项目允许上限。
- merge写`RUN_ROOT/predictions`；只有1500/1500逻辑surface封存完成后才执行`truth-open`写`RUN_ROOT/truth_catalog.json`，随后`score`写`RUN_ROOT/score`。
- 健康停止只允许P0协议/安全/hash/覆盖错误，或至少两个不同outer row在prediction前出现相同确定性异常指纹；禁止按accuracy、H、BA、floor或局部结果停止、重跑、调参或选择性补片。
- 预期闭合：125/125 outer、375/375 scene、1500/1500逻辑surface、1350 unique prediction、150 K1 alias、8/8 shard、truth-open与score全部成功。

预期artifact至少包括：`target125_plan.json`、`target125_context.json`、真实smoke receipt/prediction、8个prediction shard manifest、合并prediction manifest、FA resource/state reuse receipt、truth-open event、truth catalog、score manifest、coverage和completion receipt。

## 8.证据边界

本地设计、代码、测试、landed或RUNNING均不是性能结果。只有125/125 outer、375/375 scene和1500/1500四状态prediction完整封存并由独立truth-side scorer闭合后，才能报告性能；partial artifact只作技术诊断，不进入性能比较。
