# D81地面扰动谱稳健target原型实验报告

## 1.实验登记

|字段|值|
|---|---|
|实验ID|`d81_ground_nuisance_cauchy_center_probe_20260720`|
|候选|`ground_nuisance_cauchy_center`|
|operator|Codex`/root`|
|状态|`IMPLEMENTED_TESTED_NOT_RUN`|
|目标|高效利用全部地面压缩原型估计support样本的跨域扰动可靠性，同时让query判别几何完全由target support决定|
|开发单元|receiver`20-1`、new5、seed`713101`、K10(actual K8)、3场景×5fold|
|matched baseline|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|
|formal状态|当前ground组件资格false/UNVERIFIED，D81仅development diagnostic|

## 2.假设与创新点

D77-D80已经排除了把ground质心、低秩投影或ground协方差直接放进query距离/协方差的路线：这些方法能保护部分旧类，却把新类身份方向误当域噪声。D81把ground的作用前移到注册阶段，只回答“同一target类中哪个support样本更像受到已知跨域扰动”，再以target support自己形成稳健类中心。

该设计有三个隔离性质：

1. 地面old6类不提供任何类别锚点或query score，只提供类无关扰动方向；
2. 每类共同平移保持类内残差和target协方差不变，因此不会重写D62的target度量；
3. 权重在每个OOF fit内重算，held support和query均不可见。

## 3.锁定公式

从84个地面domain-class类中心构造`r_dc=g_dc−mean_d(g_dc)`与协方差`G`。对正特征值`lambda_j`计算：

`r_eff=(sum_j lambda_j)^2/sum_j lambda_j^2`，`r=ceil(r_eff)`。

固定保留前`r`个方向，并令`pi_j=lambda_j/sum_{l<=r}lambda_l`。对当前fit可见的target类`c`：

`e_ci=sum_{j<=r} pi_j [u_j^T(z_ci−mean_i z_ci)]^2`

`raw_w_ci=1/(1+e_ci/mean_i e_ci)`，`w_ci=raw_w_ci/sum_i raw_w_ci`

`mu_robust_c=sum_i w_ci z_ci`

`z'_ci=z_ci+(mu_robust_c−mean_i z_ci)`。

若能量为0则等权；K1显式identity，K2因两个中心残差互为相反数而严格等权identity。只变换z160，FFT96/RF32保持bitwise不变。禁止rank、尺度、温度、平移系数或场景/类别权重扫描。

## 4.协议与资源边界

- 数据状态沿用D18`VALIDATED_ONCE`；方法变更不触发重建/重验。
- 单一固定`LEO_weak`观测；support-only；query独立一次评分；无clean/source/query truth/role/quota/global assignment。
- target-old/new完全相同公式；不访问类ID语义、old/new角色、receiver handle或scene handle。
- 使用全部84个ground cell估计谱；当前组件无sample radius/count，不伪造这些统计。
- 预计新增适配复杂度为每次fit`O(N*r*160)`，其中`r`由ground effective rank自动确定；新增参数/optimizer step/query MAC均为0。
- 持久状态仍为D62单affine＋25,428B ground组件，≤256KB。

## 5.联合晋级门

相对同row D62：总体`A/N/H/J/min-class B/A/N`不得下降、`F`不得上升；每个场景`A/N/H`不得下降、`F`不得上升；`old→new`、`new→old`、`new→wrong-new`均不得增加；且至少一个联合指标严格改善。INT8/FP32不得发生outer argmax或margin-sign翻转。任一失败即停止，不启第二seed、125或N607。

## 6.版本状态

根目录`E:\type10-7`非Git仓库。实现、trace和本报告先进入独立Git worktree`E:\type10-7\code\snapshots\d81wt`，基于主发布分支提交`4dcf066b`；完成本地验证后再以精确commit闭环回主发布分支。服务器暂不使用。

## 7.实现与验证

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d81_ground_nuisance_cauchy_center.py`|地面扰动谱、固定rank、support稳健中心平移|`44111f8d7ecd0ffcfbd887c09468a167e4e1134bad3c2798bd7f0f5f89c3dc7a`|
|`code/scripts/probe_d81_ground_nuisance_cauchy_center.py`|D62全部full/block、outer/held闭包注入、资源和hash审计|`85baac449d2cd1c5b21bff63ba9b01fe95bb2025fcdfa8ee3127ae41a5e99e82`|

- D81专项与合成D62全栈：11/11通过。
- D62/D80/D81相邻链：30/30通过。
- 真实ground smoke：84 cells，effective rank=`13.6445898983`，retained rank=`14`，保留信号trace比例=`0.7975861768`，basis SHA=`f55174f1e1479eed4bd62b927ef7b4e952f14fa03cadc0e70b315e183426ed7f`，radius/count均false。
- 合成D62链确认每次fit的full、block及其inner-LOO都经过独立center transform；K1/K2 bitwise identity，query extra MAC=0。

## 8.锁定运行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d81wt\code\scripts\probe_d81_ground_nuisance_cauchy_center.py' `
  --d81-arm ground_nuisance_cauchy_center `
  --ground-component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d81wt' `
  --before-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only' `
  --before-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json' `
  --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --before-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json' `
  --before-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json' `
  --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only' `
  --after-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json' `
  --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --after-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json' `
  --after-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json' `
  --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding 'E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json' `
  --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output 'E:\type10-7\automation_reports\CV-SincNet\d81_ground_nuisance_cauchy_center_probe_20260720\ground_nuisance_cauchy_center' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期105行、30个target row、1,080个D62 component fit、2,160个support-center transform。先本地运行；不使用N607。

## 9.运行前锁定

- detached实现提交：`2f6a26d3c02fa7b33ee2efc1183748f55a396fdf`；主发布分支对应实现提交：`db4013dd`。
- worktree在锁定前为0项未提交改动；输出目录不存在，不会覆盖历史结果。
- 本地GPU0为RTX5070Ti，检查时1,083/16,303MiB、利用率0%；本实验锁定`--device auto`，由runner记录实际runtime device与CUDA峰值。
- 数据复用D18 matching capsule/seal/policy/authorization；方法变化不触发数据重验。
- ground NPZ/manifest在入口和出口分别复核SHA；任何hash变化、105-row不完整、1,080 component或2,160 transform计数不匹配均判运行失败。
