# D66地面域可靠性残差开发探针

## 1.执行前登记

- 实验ID：`d66_ground_domain_reliability_residual_probe_20260719`。
- 时间：2026-07-19；operator：Codex。
- 目标：真正使用不可变Phase1地面int8域×类聚合知识，同时避免旧类专属anchor导致的新类塌缩；检验共享地面域可靠性变换能否在D62基础上同时改善旧类域适应与新类注册。
- 比较目标D62：before92.78%、after82.22%、new84.67%、H82.62%、forget10.56pp、joint26.67%、min-before80%、min-after53.33%、min-new73.33%、混淆23/8/15。
- cell：receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer fold，实际K8；复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 enrollment-only support，不重验数据。
- 根目录`E:\type10-7`不是Git仓库；版本化实现和本报告镜像位于`E:\type10-7\github_publish\CVS-RFFI-repo`。执行前Git HEAD为`51e375ada1ffcd56516b01dce88dd0b5b359d937`；工作树存在大量不属于本轮的既有修改，本轮只暂存D66精确路径。

## 2.机制与历史边界

D66从84个有效的地面域×旧类int8聚合单元计算每个z160坐标的类间身份方差`B`和同类跨域漂移`W`，固定`r=(B+eps)/(B+W+2eps)`、`s=sqrt(1+r)`。z160使用共享尺度`s`，FFT96/RF32恒等；D62全部支持拟合在共享坐标执行，再把系数编译回原坐标。对旧类、新类和未来query没有不同公式，query零额外MAC/state。

历史停止项：D19/D25/D36旧类专属anchor中心融合、独立半径似然、角色offset/IRLS、D30 old-old DALI、旧anchor Procrustes/transport及query batch统计。D66不复用这些机制，不持久化反量化ground bank，不读取clean/source样本或query。

当前组件manifest标记`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，所以本轮严格限定为开发support内部held-rank诊断，formal/query/performance claim和125权限均为false。组件必须只读，入口/出口SHA均应为`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`。

## 3.预注册门

- 完成七候选×三场景×五折=105行，query/clean/source/role/quota/global assignment访问均为0。
- 相对D62总体、三场景、逐类floor、遗忘、混淆和量化不得交换伤害，并至少严格改善after、forgetting、joint或任一floor。
- 必须报告七候选、三场景、11类、15fold、地面尺度统计、FP32/int8量化、训练/适配MAC、状态、延迟和完整artifact。
- 失败即状态`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`并停止本路线；成功也先完成D64–D66三轮回顾，不直接启动125。

## 4.待完成实现与运行信息

初版专项4/4、D42–D66完整回归303/303通过。额外随机合成的真实D42＋D62烟测在任何项目support/query打开前触发`D42 sklearn coefficient deployment prediction drift`；移除D66后，未改动D62在同一合成数据上复现完全相同错误，故不能归因于共享尺度。未放宽D42闭包断言，保留预注册的共享坐标拟合公式；真正集成判据为锁定项目enrollment-only support上的105行fail-closed运行。

待补：本地变更、验证命令、Git提交、干净worktree、精确运行命令、环境、输出路径、运行时、完整结果与下一实验建议。

## 5.实现、验证与版本状态

- 新增`code/scripts/probe_d66_ground_domain_reliability_residual.py`：组件策略/allowlist/SHA闭包、规范registry排序、84-cell反量化、共享可靠性尺度、D62坐标注入、系数编译、资源和输出验证。
- 新增`tests/test_probe_d66_ground_domain_reliability_residual.py`：组件只读与策略fail-closed、类置换逐bit尺度不变、尺度边界、全类统一编译等价和无角色/场景/可调分支。
- 预注册提交`fc7c0977`；实现提交`684e110edddf5adaafe22200cb044ddd56059bcd`；实现脚本SHA将在运行artifact中自动锁定。
- 主工作树专项4/4、D42–D66完整26文件303/303通过；干净worktree`E:\type10-7\code\snapshots\d66wt`在同一提交再次303/303通过，用时85.2s；`py_compile`和`git diff --check`通过。
- 真实组件：26域×6类、84个有效cell，每类14个；可靠性0.0242749–0.9999186，尺度1.0120647–1.4141848，条件数1.3973265，尺度SHA256=`70a8e94327e7100695f691d6ae49e246305036cefd92579e977e3d536c37df6c`；组件逻辑状态25,428B，瞬时反量化53,760B，统计58,880MAC。
- 本轮完全本地，不需要SSH/SCP，不占用或干预N607。Conda/Python环境为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`；运行设备`auto`，由锁定runner记录实际GPU/CPU和峰值显存。

## 6.精确运行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d66wt\code\scripts\probe_d66_ground_domain_reliability_residual.py' `
  --d66-arm ground_domain_reliability_residual `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d66wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d66_ground_domain_reliability_residual_probe_20260719\ground_domain_reliability_residual' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期输出为105行training log、support/query/selection/receipt、geometry/resource和D66 metadata。任何组件、策略、support、D42/D62、编译、资源或输出闭包失败均停止，不覆盖输出、不重跑同目录。

## 7.首次运行与Resource-R1预注册

- 首次运行完成105/105行、query0、Runner129.0378s、外层137.7s，组件入口/出口SHA、source closure和D66 metadata均通过。
- 完整日志解析发现资源主字段漏加地面组件：`d66_ground_component_logical_state_bytes=25,428`和`ground_int8_component_input_count=84`正确，但runner后置逻辑把`persistent_state_bytes`覆盖为仅仿射头8,583B。正确组件含总状态为34,011B，仍低于256KB。
- 该缺陷不影响预测或性能，但首次artifact不能封为最终资源证据，先不发布最终D66性能判定。首次目录原样保留。
- Resource-R1只修资源后置加总与硬断言，不改公式、support、候选、训练、量化或预测；新输出为`ground_domain_reliability_residual_resource_r1`，不得覆盖首次目录。
