# C-DOM-SCXMAP-D92-GLF/r1 Phase1留出证伪报告

## 基本信息

|字段|值|
|---|---|
|实验ID|`scxmap_p1_held_falsifier_r1_20260724`|
|日期|2026-07-24|
|操作者|Codex主代理；N607唯一launch owner待交接|
|当前状态|`LOCAL_VERIFIED / REVIEW_APPROVED / PREREGISTERED / NOT_LANDED`|
|目标|在打开任何目标25矩阵前，用Phase1未见接收机留出代理证伪SCXMAP是否产生稳定净纠错|
|候选|`C-DOM-SCXMAP-D92-GLF/r1`|
|claim范围|`PHASE1_HELD_PROXY_NON_PROMOTABLE`；不是D92完整288维四臂证据，不是目标性能结果|

## 假设与固定比较

SCXMAP从类内中心化Phase1样本学习rank4的`z_dom→z_id`接收域残差映射；Stage2只用target-old support相对封存ground anchor的残差拟合一个非负连续收缩标量。query只用自身`z_id/z_dom`逐样本变换，不更新任何状态。

固定两臂：

|臂|特征|分类头|
|---|---|---|
|`M0`|原始`z_id160`|同一INT8 Student-t qKNN|
|`M_DA`|SCXMAP变换后的`z_id160`|同一INT8 Student-t qKNN|

本代理不包含D92的288维全局头、qKNN融合头或弱类零和修正，因此只能判断SCXMAP局部邻域是否值得进入后续目标实现。

## 冻结矩阵与门禁

|维度|冻结值|
|---|---|
|held receiver|由coverage SHA确定的1个未见接收机|
|pseudo-new|6个Phase1类逐一轮换|
|场景|`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|
|K|1、5、10|
|总row|6×3×3=54|
|状态拟合|只读每个ground-old类相同合法K的support；new support不进入DA标量拟合|
|query|test-only；query-fit=0；逐样本全注册类竞争|

代理通过要求每个K聚合、每个K×场景和每个K×pseudo-new分层同时满足：

- argmax变化数>0；
- wrong→correct严格多于correct→wrong；
- old-after与seen-new均不下降；
- `H_old_new`严格提高。

即使全部通过，packet、build receipt和score仍固定`target25_release_authorized=false`。目标25必须另行实现完整D92/288维四臂、重新预注册和独立复核。

## 本地实现、验证与审查

|文件|作用|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_scxmap_transform.py`|INT8 Phase1 lock、support标量拟合、逐样本变换、wire与资源账本|`8298ed9f879715e77805e48d2272a7fa640a758554dec18f6f3e189187626944`|
|`code/cvsrffi/scxmap_phase1_held_falsifier.py`|54-row构建、双向承诺、外部build receipt、predict、独立score、真实support-only smoke|`ce4edb7badaa1fe39efb324e8ec3f3d7f191f54051918f6028381f529a5df976`|
|`docs/C_DOM_SCXMAP_D92_GLF_R1_DESIGN_FROZEN.md`|冻结机制、门禁与证据边界|`a03e48d6224826f3241d60eaf432e906773ebf9ffeb05f4c153b8cf2fab85962`|
|`tests/test_stage2_scxmap_transform.py`|支持拟合、恒等、置换、资源与协议负测|`8b681752aaa124f15c021e8629386179369b88b07bb9d21ba30c07400116194f`|
|`tests/test_scxmap_phase1_held_falsifier.py`|artifact闭环、篡改、重签、类别重命名全输出等变|`9715340b594714dcde22fef4dc7346cf4b03acf9dd48f9c8fd7e50037f05ed0b`|

本地`ssr-gpu`验证：

- 两个模块`py_compile`通过；
- SCXMAP定向测试加R2A回归共`16 passed`；
- `git diff --check`通过；
- 真实checkpoint SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`以`weights_only`安全加载；
- 真实8400行双特征archive逐数组SHA通过；support-only烟雾`PASS`，`support_rows=5`、`query_rows_read=0`、`truth_access=false`、`beta=0.01806582697`、状态wire=4907B、每query额外1296MAC。

独立复核终态：`P0=0、P1=0、P2=2`。query字节替换、truth重签、argmax重签均fail-closed；全prediction类别重命名等变。P2仅为claim范围和重复本地smoke回执清理，不阻断本代理发布。

## Git与发布输入

本地Git工作树：`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt`，分支`codex/cdom-scxmap-d92-glf-r1`。实现与预注册精选commit为`f4dcb950`；release-surface修复commit、source archive与外部release receipt将在二次复核通过后冻结。禁止`git add -A`和GitHub上传。

N607既有只读输入：

|输入|远端路径|SHA256|
|---|---|---|
|双特征archive|`runs/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8/output/archive/phase1_singleobs_dual_feature_archive.npz`|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`|
|archive manifest|同目录`phase1_singleobs_dual_feature_archive.manifest.json`|`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|
|coverage|同run`output/coverage_receipt.json`|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|
|checkpoint|`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|

## N607冻结发布信息

|字段|冻结值|
|---|---|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/scxmap_p1_held_falsifier_r1_20260724`|
|source|`<run>/source`，由冻结Git commit归档安全解包；所有文件去除写权限，pipeline启动时fail-close复核|
|output|`<run>/output`，启动前必须不存在|
|log|`/home/szu2070436088/2510044040/CV-SincNet/logs/scxmap_p1_held_falsifier_r1_20260724`|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|GPU|由独立runner选择1张不超过占用上限的GPU，并映射为进程内`cuda:0`|
|入口|`automation_reports/CV-SincNet/scxmap_p1_held_falsifier_r1_20260724/run_pipeline.sh`|
|预期产物|real-support smoke、packet、truth、query、build receipt、prediction、score、SHA清单、完成marker和exit|

冻结child命令：

`CUDA_VISIBLE_DEVICES=<single-0-to-7> RELEASE_RECEIPT_SHA256=<frozen-sha256> bash /home/szu2070436088/2510044040/CV-SincNet/runs/scxmap_p1_held_falsifier_r1_20260724/run_pipeline.sh`

## 健康控制、风险与完结标准

- P0协议/安全错误立即停止。
- 这是单一进程内的54-row原子pipeline，不按row独立dispatch；任一build、predict、score或artifact技术阶段首次非零即由`set -e`停止，保留partial artifact并标记`NO_PERFORMANCE_RESULT`。不得因accuracy、H或其他性能数值停止。
- 只有54/54 row、prediction与COMMIT、独立score、build receipt双向绑定和全部SHA闭合才进入`ARTIFACTS_COMPLETE`。
- 不覆盖既有run或输出；失败run保留全部partial artifact，修复必须使用新run ID。
- 主要风险是Phase1代理对目标D92空间外推失败、DA产生非零参数但不改argmax，以及aggregate掩盖特定场景/伪新类负迁移；分层门用于直接证伪这些情况。
- 完成后在本报告追加逐K、K×scene、K×pseudo-new、active-beta、tie rows、wrong→correct/correct→wrong、资源和最终裁决。不同证据范围不得与D62/D92/SVRN完整125绝对值直接混排。
