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
- 结果：35项PASS，退出码0。其中新增K>1行为测试实际执行Stage2-C优化路径，发现并修复遗漏的`math`导入；不再只靠K1零epoch路径验证。
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
