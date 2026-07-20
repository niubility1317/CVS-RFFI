# D89 v2半径可靠度Cauchy support中心报告

## 预注册

- 状态：`PLANNED_LOCAL_DEVELOPMENT_DIAGNOSTIC`。
- 要修复的失败：D87用地面sigma边界换取旧类收益但损伤新类；D88逐类CE硬保护使9/15行零更新并相对D85退化。D89恢复D81已验证的support可靠性路线，同时使用D85的高效v2组件和真实p90半径。
- 数学机制：先从每个ground cell相对其类内跨域均值的偏移得到有效信号`s_dc=||g_dc-mean_d(g_dc)||²`，再把p90余弦半径转为弦长方差`v_dc=2r_dc`，固定可靠度`rho_dc=s_dc/(s_dc+v_dc)`；在每个ground类内部归一化为`q_dc`，以确保6个ground类等权，再汇总84个cell的加权协方差。固定减去manifest中的`reconstruction_rmse²`噪声底，ground类身份随后丢弃，固定保留`ceil(participation ratio)`谱。每个实际target类继续使用D81同一one-step Cauchy权重和中心平移。
- 单一主要差异：相对D81仅替换地面谱来源；target support中心公式、D62 full/block OOF、INT8 head和query路径不变。相对D85不做共享中心，而把radius用于support样本可靠性。
- 协议：v2组件只读；不访问clean/source/query truth/role/quota，不建立ground→target映射；同一received IQ不生成新物理样本；K1/K2恒等；query独立全类单次仿射评分。
- 停止门：相对D81/D85总体和每场景`A/N/H/J/min-class/row-floor`不退化、`F`不升、三类混淆不增加，且至少一项A/H/F/floor/混淆严格改善；INT8/FP32无outer flip和margin sign flip。失败则不进seed2/125且不扫描radius公式。
- 最小矩阵：development seed713101、receiver20-1、K10/new5、3场景×5fold×7候选，共105行。
- 组件限制：`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`，强制nonpromotable diagnostic。

## 文件与环境

- worktree：`E:\type10-7\code\snapshots\d81wt`。
- core：`code/cvsrffi/stage2_d89_v2_radius_cauchy_center.py`。
- probe：`code/scripts/probe_d89_v2_radius_cauchy_center.py`。
- 环境：`ssr-gpu`；本地开发cell，不使用N607/SSH。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d89_v2_radius_cauchy_center_20260720\v2_radius_reliability_cauchy_center`。

## 版本、验证与固定谱

- 实现提交：worktree`5346d1cda0d33cbc7de861b9effc7db255d84cd7`；Git承载面cherry-pick提交`d4a0b353`。根目录`E:\type10-7`不是Git仓库，实验报告同时保存在根报告面与Git承载面。
- 变更：新增D89 core、probe及两个聚焦测试；只给D81 probe增加可选`EXTRA_SOURCE_CLOSURE`合并钩子，默认空字典，不改变D81既有公式。
- SHA256：D89 core=`b1b4cd983fa85c7b89d7bd33b2fe83267aae1a1a0f165d02fff028c59cd6ebe2`；D89 probe=`24d3454147d6dbe51677609dad6c7aed36a13107714b90a8458a9efbd46f7752`；D81 scaffold=`7feb50a4210be8f87bc23b1ae9d084436ad09996f2517cbc17bfae13ca14dea4`。
- 2026-07-20 09:09 CST在`ssr-gpu`中执行`py_compile`及D89、D81、D85相邻测试，共19项通过。
- 真实只读v2组件代入：`rho min/mean/max=0.002458/0.189878/0.656824`；每ground类`sum_d(rho)=2.065887..3.012965`；`trace(G89)=0.0011869368`；`reconstruction_rmse²=2.2686737e-6`；正谱rank18、有效rank10.246800、固定保留rank11、保留能量92.0768%。每类domain权重和最大误差`2.22e-16`，加权残差中心最大误差`1.31e-16`。
- 资源预锁：v2持久状态5,816B、目标INT8 affine head 8,583B、总持久状态14,399B；D89额外可训练参数/optimizer step/query MAC/query state均为0；地面重构与谱统计MAC上界不超过0.5M。

## 固定执行命令

本轮为本地development cell，不使用N607、GPU分配、远端PID或远端日志。固定在`E:\type10-7\code\snapshots\d81wt`激活`ssr-gpu`后执行：

```powershell
python code/scripts/probe_d89_v2_radius_cauchy_center.py --d89-arm v2_radius_reliability_cauchy_center --ground-v2-component-dir E:\type10-7\code\snapshots\d81wt\automation_reports\CV-SincNet\d85_ground_radius_v2_20260720\artifacts\component --ground-v2-manifest-sha256 6990ce8c274d6f46c9a86f7272a76960b4a0055e06ae75e14f83e0b70595f112 --expected-checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --expected-class-handle-binding-sha256 76735ae6d9b2d7e58f683635ca2644e00fbd27a515246aab9d47488c1ab5111f --expected-pre-sign-content-root-sha256 098badd1e82c05c1029cb02c024fe7d3c433488e8ab22e5c6e2ba0516b8d0055 --runtime-root E:\type10-7\code\snapshots\d41wt --probe-root E:\type10-7\code\snapshots\d81wt --before-root E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only --before-seal E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 --before-formal-policy E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json --before-formal-policy-authorization E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json --before-signed-policy-authorization-envelope E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e --after-root E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only --after-seal E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff --after-formal-policy E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json --after-formal-policy-authorization E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json --after-signed-policy-authorization-envelope E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 --component-dir E:\type10-7\code\snapshots\d81wt\automation_reports\CV-SincNet\d85_ground_radius_v2_20260720\artifacts\component --component-manifest-sha256 6990ce8c274d6f46c9a86f7272a76960b4a0055e06ae75e14f83e0b70595f112 --class-binding E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f --output E:\type10-7\automation_reports\CV-SincNet\d89_v2_radius_cauchy_center_20260720\v2_radius_reliability_cauchy_center --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期输出包括`training_log.jsonl`、`support_audit.json`、`resource_audit.json`、`geometry_audit.json`、`selection.json`、`RECEIPT.json`、`D81_PROBE_METADATA.json`与`D89_PROBE_METADATA.json`。成功执行要求105/105行、目标INT8/FP32各15行、source closure不变、query未打开；性能晋级另按预注册停止门判断。

## 启动记录

- attempt0于2026-07-20 09:16 CST在进入runner、打开数据或创建输出目录前失败：`UnboundLocalError: resource referenced before assignment`。根因是probe读取manifest的`reconstruction_rmse`时，`resource_audit()`赋值位于其后；这是本地集成顺序错误，不构成实验性能证据，输入与公式锁不变。
