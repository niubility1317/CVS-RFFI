# D90 v2逐方向Cauchy support中心报告

## 预注册

- 状态：`PLANNED_LOCAL_DEVELOPMENT_DIAGNOSTIC`。
- 要修复的D89缺陷：D89相对D81/D85内部support权重明显变化，但K8有效样本数仍约7.3/8，15/15 outer预测不变；单个ground谱总能量只产生每样本一个标量权重，某一坏方向会连带降低该样本全部160维信息。
- 方法：完全复用D89固定v2 SNR半径谱。在每个target类中先得到D81径向中心；保留其ground子空间正交分量，但在保留的11个ground方向内分别计算无扫描Cauchy中心，以逐方向中心替换D81径向子空间中心。最后仍只对z160做一次类公共平移。
- 创新点：压缩ground原型不再仅产生样本级可靠度，而成为“方向级异常解释器”；一个方向的异常不会丢弃同一样本在其他方向的信息。
- 协议：公式对全部注册类相同；K1/K2逐位恒等；类内残差、FFT96、RF32不变；不读取clean/source/query truth/role/quota，不做ground→target类映射，不更新ground组件，不增加query计算。
- 资源预期：仍为5,816B ground＋8,583B affine＝14,399B；额外参数、optimizer step、query MAC/state均为0。逐方向权重只增加`O(CKr)`support标量运算，远低于LDA闭包。
- 停止门：相对D81/D85/D89总体、场景、逐类和15-row floor不退化、混淆不增加，且至少2/15个row变化、净纠错≥2、旧类和新类各自无correct→wrong；否则不进seed2/125，不扫描方向权重或混合系数。
- 固定cell：receiver20-1、seed713101、K10/new5、3场景×5fold×7候选，共105行；v2组件仍pending joint seal，强制nonpromotable。

## 文件与环境

- worktree：`E:\type10-7\code\snapshots\d81wt`。
- core：`code/cvsrffi/stage2_d90_v2_directionwise_cauchy_center.py`。
- probe：`code/scripts/probe_d90_v2_directionwise_cauchy_center.py`。
- 复用：D89谱、D81/D62 full/block OOF和单一INT8 affine query head。
- 环境：`ssr-gpu`；本地开发cell，不使用N607/SSH。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d90_v2_directionwise_cauchy_center_20260720\v2_directionwise_cauchy_center`。

## 版本与验证

- 实现提交：worktree`f33c3d8f27f24cdfc16094fc22afb0159bbd196c`，Git承载面`59a2c3b4`。
- SHA256：D90 core=`11138760634ad16e9b4b8b2fad8371f624866d342692cc3e8b4a6fd491fae1eb`；D90 probe=`cddcbae1f051afc547b2dfa32714bc996c0e9d387bb18bd8c62297b1389ceef1`；D89 scaffold=`b1b8e7c0191ddf930d1e796997dfa6ae683a6e073b147be3ac563cd73637db8e`；D81 scaffold=`68d4cd676094924112ee30184b9c544f865dba9dc3147a253b9bc467b6da64ca`。
- `ssr-gpu`中D90/D89/D81聚焦与回归测试20项通过，`git diff --check`通过。
- source closure包含D90 probe/core、D89 probe/core、v2 codec、D85/D81 scaffold及D62以下既有闭包。D81/D89的新增hook默认值为0或空字典，不改变其既有公式。

## 固定执行命令

```powershell
python code/scripts/probe_d90_v2_directionwise_cauchy_center.py --d90-arm v2_directionwise_cauchy_center --ground-v2-component-dir E:\type10-7\code\snapshots\d81wt\automation_reports\CV-SincNet\d85_ground_radius_v2_20260720\artifacts\component --ground-v2-manifest-sha256 6990ce8c274d6f46c9a86f7272a76960b4a0055e06ae75e14f83e0b70595f112 --expected-checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --expected-class-handle-binding-sha256 76735ae6d9b2d7e58f683635ca2644e00fbd27a515246aab9d47488c1ab5111f --expected-pre-sign-content-root-sha256 098badd1e82c05c1029cb02c024fe7d3c433488e8ab22e5c6e2ba0516b8d0055 --runtime-root E:\type10-7\code\snapshots\d41wt --probe-root E:\type10-7\code\snapshots\d81wt --before-root E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only --before-seal E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 --before-formal-policy E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json --before-formal-policy-authorization E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json --before-signed-policy-authorization-envelope E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e --after-root E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only --after-seal E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff --after-formal-policy E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json --after-formal-policy-authorization E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json --after-signed-policy-authorization-envelope E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 --component-dir E:\type10-7\code\snapshots\d81wt\automation_reports\CV-SincNet\d85_ground_radius_v2_20260720\artifacts\component --component-manifest-sha256 6990ce8c274d6f46c9a86f7272a76960b4a0055e06ae75e14f83e0b70595f112 --class-binding E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f --output E:\type10-7\automation_reports\CV-SincNet\d90_v2_directionwise_cauchy_center_20260720\v2_directionwise_cauchy_center --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

成功执行要求105/105行、`D90_PROBE_METADATA.json`闭合、query未打开、组件hash不变。性能晋级另按预注册停止门判断。
