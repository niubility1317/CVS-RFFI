# D20地面int8安全身份重排与单观测FFT/RF路线

- experiment ID：`d20_int8_maxold_fftrf_20260717`
- timestamp：2026-07-17
- operator：Codex
- 当前状态：本地设计与实现验证完成；N607尚未启动
- objective：在正式`项目.md`Stage2-B/Stage2-C约束下，将Phase1域×类int8原型与合法单一LEO_weak received IQ上的轻量表征适配、新类注册有机结合，优先修复旧类floor并保证int8分支不压制seen-new。
- comparison：B0纯target centroid、B1 int8 max-old、B2 int8+同IQ logits max-old、B3单IQ FFT/RF轻头、B4轻头+max-old；最终需相对identity-only单qKNN报告MAC/延迟/显存/状态Pareto。

## 协议与设计结论

- support/query只能是唯一LEO_weak观测；FFT96、RF32和z_id160均来自同一密封received IQ，不生成第二LEO状态、不增加K。
- Phase2不读取clean/source样本、样本级feature或full-precision prototype；历史int8组件仅用于一次`PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`，不得打开query或作正式性能声明。
- 84个有效域×类原型为13,608B有效payload；当前稠密26域逻辑状态25,428B。跨域同类余弦均值0.9972–0.9989，说明它适合旧类身份方向，不适合强domain selector。
- D19强anchor提高旧类但把seen-new压至6.67%–12%；D19b弱anchor退化为Z0。因此停止扫描CIAF强弱融合。
- D20采用固定component-only max-min medoid。int8与同IQ direct logits只在旧类内部重排；重排后严格恢复`max_old`，并保持全部新类score逐位不变。
- B3/B4的Stage2-B拟合单IQ FFT/RF对角头；Stage2-C冻结旧scale与旧head，只用全部注册support对新类权重执行类平衡、worst-class-aware训练。K1退化为0-epoch centroid注册。
- support-only晋升必须在全部15个场景×L2O fold逐类非劣，并严格改善最坏after-old floor；B4还必须相对B3严格改善floor且seen-new逐类完全相同。

## int8几何与压缩口径

|项目|结果|
|---|---:|
|逻辑张量|`int8[26,6,160]`|
|有效域×类cell|84|
|有效int8质心|13,440B|
|有效FP16 scale|168B|
|有效payload|13,608B|
|当前稠密逻辑组件|25,428B|
|NPZ压缩文件|5,363B|
|固定max-min medoid|domain index 9|
|六类最大非对角共识余弦|0.0169|

## 本地变更

- `analysis/d20_ground_int8_prototype_organic_integration_design_20260717.md`：完整机制、数学保证、负路线和分阶段验证设计；提交`fde7e95a`。
- `code/cvsrffi/stage2_dali.py`：固定medoid、K-shot收缩、support锁定尺度、`max_old`保持、new-score bitwise不变、不可变状态和资源审计。
- `code/scripts/run_d19_support_only_ciaf.py`：历史int8特许support-only runner升级为D20 B0–B4；同一次前向提取z_id/logits；同IQ FFT/RF；显式view lineage；严格floor门；无query接口。
- `analysis/d19_adv3b02_class_binding_20260717.json`：显式direct-logit列、封存runtime SHA、head tensor SHA和逐行SHA；runner在support materialization前验证。
- `tests/test_stage2_dali.py`、`tests/test_run_d19_support_only_ciaf.py`：数学不变量、K1、重复注册、资源、runtime绑定、防篡改、B0纯基线、view lineage和严格floor选择测试。

## 本地验证

- 环境：`ssr-gpu`。
- 命令：`conda run -n ssr-gpu python -m pytest -q tests/test_stage2_dali.py tests/test_run_d19_support_only_ciaf.py tests/test_stage2_diag_cosine_exploration.py`。
- 结果：37项PASS，退出码0。其中新增K>1行为测试实际执行Stage2-C优化路径，发现并修复遗漏的`math`导入；另有2项DLPack桥接与禁止`torch.from_numpy`回归测试。
- `py_compile`：DALI模块、runner及两组测试通过。
- `git diff --check`：通过；仅有Git的LF/CRLF提示。
- pytest退出后出现Windows临时目录`PermissionError`清理噪声，但测试进程退出码为0，不属于项目失败。
- 独立复审曾发现Stage2-C enrollment MAC未并入总适配MAC、K1零epoch仍误报可训练参数；现已新增`stage2b_estimated_adaptation_macs`、`stage2c_estimated_adaptation_macs`及合计字段，总可训练参数也严格按Stage2-B与Stage2-C实际训练参数求和。K1的Stage2-C可训练参数与MAC均为0，K>1按`epoch×support rows×registered classes×288×3`审计。
- 当前class binding SHA256：`bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f`。
- 最终独立复审：`Approve`；未发现协议泄漏、class order错误、fold/cache污染或剩余资源口径不一致。

## 资源预期

- 20个新类、26个注册类时，FFT/RF轻头持久状态约为共享288维scale+26×288权重+registry；可训练参数远低于50k。
- B4额外包含25,428B当前历史int8逻辑组件和极小DALI状态；总状态仍应低于256KB，runner按实际数组逐项计数。
- Stage2-C只训练新类权重，旧scale与旧head逐位冻结；资源报告分别列出Stage2-B与Stage2-C参数、optimizer step和适配MAC，避免把新增注册训练隐藏在Stage2-B口径中。
- 头部基础MAC约`288+26×288`，max-old额外只计算6×160固定medoid方向及标量操作；不重复计算DALI的z_id prototype base。
- identity-only单qKNN在K10、26类时仅点积约41,600MAC/query，K20约83,200MAC/query；D20状态/MAC不随K线性增长。正式延迟、P95和峰值显存必须实测。

## N607计划

- 启动前执行`tools\n607_ssh_preflight.ps1`，检查实时GPU/进程并记录。
- 本地Git提交后只同步DALI模块、runner、class binding和必要依赖；不上传原始样本、full-precision prototype或source cache。
- 输入复用D19已验证的receiver 20-1、seed 713101、K10/new5、三个物理样本集合互斥的LEO_weak enrollment-only包；不得打开query。
- candidate：B0–B4，共5候选×3场景×5个leave-two-out fold=75个support-only原子结果；B3/B4共享同fold基础头缓存。
- 输出：`training_log.jsonl`、`support_audit.json`、`selection.json`、`resource_audit.json`、`RECEIPT.json`。
- 正路线出现后才重建共同封存checkpoint+int8 bundle并进入开发query；否则直接回退/淘汰，不启动125矩阵。
+

## 2026-07-17 12:27 N607启动前记录

- 本地Git承载：`E:\type10-7\github_publish\CVS-RFFI-repo`；实现提交：`9f4e51692940947138ddf8afe7fa333229f90af4`；三项待同步文件均与提交一致，相关路径工作区无未提交修改。
- 根目录`E:\type10-7`不是Git仓库；本报告同时镜像到上述Git承载面。
- 本地验证：`ssr-gpu`环境35项相关测试PASS、`py_compile`PASS、`git diff --check`PASS、独立复审`Approve`。
- direct N607 preflight：2026-07-17 12:27 CST PASS；8张RTX 3090均0%利用率/10MiB，无用户训练进程；项目盘剩余7.6TB。
- GPU：0；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；cwd：`/home/szu2070436088/2510044040/CV-SincNet`。
- 同步映射：
  - `code/cvsrffi/stage2_dali.py`→`/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/stage2_dali.py`，SHA256=`c51e1c028a7b6994243001dd6fd8c47de5168822241d6c4e0fd1e9085455003e`；
  - `code/scripts/run_d19_support_only_ciaf.py`→`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/run_d19_support_only_ciaf.py`，SHA256=`f226daf8a1a0fb8160dcca030aa5df4707e1ef60af4c90c35b0785dad1c9f934`；
  - `analysis/d19_adv3b02_class_binding_20260717.json`→`/home/szu2070436088/2510044040/CV-SincNet/runs/d20_int8_maxold_fftrf_20260717/input/class_binding.json`，SHA256=`bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f`。
- 不上传任何原始IQ、clean/source样本、样本级feature、full-precision prototype或query；int8组件复用远端既有密封研发组件，manifest SHA256=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`。
- 输入：receiver=`20-1`、seed=`713101`、K=`10`、seen-new=`5`；before/after均只指向`new_5_retry3/predictor/*/enrollment_only`，runner无query参数和scorer入口。
- 远端输出：`runs/d20_int8_maxold_fftrf_20260717/output/k10_new5_rx20_1_seed713101`；日志：`logs/d20_int8_maxold_fftrf_20260717/k10_new5_rx20_1_seed713101.log`。
- 精确运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_d19_support_only_ciaf.py --before-root runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/phase2_capsules/rx_20_1/seed_713101/k_10/new_5_retry3/predictor/before/enrollment_only --before-seal runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/phase2_capsules/rx_20_1/seed_713101/k_10/new_5_retry3/seals/before_enrollment.seal.json --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 --before-formal-policy runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/runtime_authorization_k10_new5/formal_execution_policy.json --before-formal-policy-authorization runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/runtime_authorization_k10_new5/before_formal_policy_authorization.v2.json --before-signed-policy-authorization-envelope runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/runtime_authorization_k10_new5/before_signed_policy_authorization_envelope.v2.json --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e --after-root runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/phase2_capsules/rx_20_1/seed_713101/k_10/new_5_retry3/predictor/after/enrollment_only --after-seal runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/phase2_capsules/rx_20_1/seed_713101/k_10/new_5_retry3/seals/after_enrollment.seal.json --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff --after-formal-policy runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/runtime_authorization_k10_new5/formal_execution_policy.json --after-formal-policy-authorization runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/runtime_authorization_k10_new5/after_formal_policy_authorization.v2.json --after-signed-policy-authorization-envelope runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/runtime_authorization_k10_new5/after_signed_policy_authorization_envelope.v2.json --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 --component-dir runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c --class-binding runs/d20_int8_maxold_fftrf_20260717/input/class_binding.json --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f --output runs/d20_int8_maxold_fftrf_20260717/output/k10_new5_rx20_1_seed713101 --device auto --mode development_select_unverified_component
```

- 成功条件：生成75个support-only fold结果；B1–B4必须15/15逐类非劣于B0并严格改善最坏旧类floor，B4还需相对B3严格改善floor且seen-new逐类结果完全相同。失败/回退时不启动query和125矩阵。
+

## 2026-07-17 12:35首次启动阻断与兼容闭包修复

- 首次运行退出码1，在`materialize_somph_enrollment_with_signed_authority`的manifest预检阶段被`SOMP-H bundle manifest exact schema mismatch`阻断；output目录尚未创建，0个fold、0条support物化、0个query访问。
- 根因：远端当前全局`code/cvsrffi/stage2_predictor_bundle.py`SHA256=`8bf20101130acbfc8063b8e42d47fbc811c4153ab85767341b3923bd8a9dbc05`，而D19成功打开同一密封包时封存源码版本SHA256=`bb27beaa94c4245b2135b5493e1be305985e05ff9f88c01bc0b9f60955944aa9`；属于加载器schema版本漂移，不是数据重建或D20模型失败。
- 修复边界：不放宽manifest校验、不修改密封输入、不触碰query；将D19成功运行的完整只读源码闭包复制到新的`runs/d20_int8_maxold_fftrf_20260717/source`，再覆盖D20已提交的runner与DALI模块。历史D19源码快照保持不变，全局loader不再改动。
- 首次失败日志保留为`logs/d20_int8_maxold_fftrf_20260717/k10_new5_rx20_1_seed713101.log`；重试写入`.../k10_new5_rx20_1_seed713101_attempt2.log`。

## 2026-07-17 12:40第二次启动环境兼容阻断

- attempt2通过密封包校验并进入首个合法support的特征提取，但在首个`torch.from_numpy`调用处退出：远端`CVS-RFFI`环境为Torch2.1.0+NumPy2.2.5，该二进制组合的NumPy C-API桥接最小复现同样失败。尚未生成任何候选fold或性能结果，query仍为0访问。
- 已确认另一个`SDG-SEI`环境为Torch1.11+NumPy1.24，虽然NumPy桥接正常，但无法加载当前TorchScript runtime，因此不更换模型环境。
- 本地修复：D20 runner的NumPy→Torch改为DLPack、Torch→NumPy改为`tolist`后显式float32重建；导入D1 fit中唯一NumPy `as_tensor`路径仅在单线程context内临时路由到DLPack并通过`finally`恢复。该改动只修复实验运行时数据桥接，张量值、样本、LEO信道、候选、损失和超参数均不变。
- 修复验证：相关全套37项PASS；新增测试强制禁用`torch.from_numpy`并验证DLPack的float32/int64值一致性和`torch.as_tensor`恢复。
- 修复提交：`44c238c7 fix: bridge D20 enrollment tensors with DLPack`；runner SHA256=`7e46db1e99ac40f4e9d7679dcb7f668553d928a0672a7bcf07022383949c8553`。独立复审再次`Approve`；仅指出未来若改为同进程多线程拟合需给临时`torch.as_tensor`兼容context增加锁，当前串行CLI无竞态。
- attempt3将从隔离闭包`runs/d20_int8_maxold_fftrf_20260717/source/code/scripts/run_d19_support_only_ciaf.py`启动，日志写入`logs/d20_int8_maxold_fftrf_20260717/k10_new5_rx20_1_seed713101_attempt3.log`；输入、候选、输出和全部hash参数不变。
