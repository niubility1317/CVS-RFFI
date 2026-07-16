# D15 CL-CDR最终独立红队复核

复核对象：

```text
code/cvsrffi/stage2_cl_cdr_envelope.py
tests/test_stage2_cl_cdr_envelope.py
analysis/d15_cl_cdr_envelope_traceability_20260717.md
```

最终结论分为两个相互独立的边界：

```text
MODULE_SAFETY_GO
TRUE_PERFORMANCE_ROUTE_NO_GO
```

## 已关闭问题

|项目|结论|独立证据|
|---|---|---|
|二阶self-exclusion|GO|对6个inner-held样本分别实施own/rest极端变异；其专属nested `q_pos/q_neg/gap/mid/half_gap/stability/nested_dims`逐位不变。每个校准record `j`均由`rows\{i,j}`重新拟合|
|nested stability|GO|评价`i`时只使用`rows\{i,j}`模型的维度集合计算pairwise consensus；含`i`模型不进入其安全门|
|K1|GO|canonical `rank=0/gamma=0/force_zero=true`，空dims/stat；Before/After状态数组分别436B/654B，不借用其他K|
|旧类score锁|GO|11旧类+2新类、K10、13个enabled envelope、4096个随机probe；After旧11列与Before逐位相同，最大差值0|
|state外部钉住|GO|NPZ/JSON双外部SHA、`allow_pickle=false`、NPZ member allowlist、metadata顶层和hyperparameter exact key allowlist|
|state篡改与覆盖|GO|缺/多NPZ member、额外JSON key、缺文件均统一拒绝；二次save拒绝且原文件hash不变|
|query权限|GO|formal API恰好单query、all registered classes；无query role、truth batch class count、class quota或global assignment入口|

验证命令使用`ssr-gpu`环境：

```text
python -m py_compile code/cvsrffi/stage2_cl_cdr_envelope.py tests/test_stage2_cl_cdr_envelope.py
python -m pytest -q tests/test_stage2_cl_cdr_envelope.py
```

结果为17/17 PASS。与threshold control联合回归为27/27 PASS。

## 残余工程边界

module级save拒绝已存在路径，但NPZ与JSON仍是两个独立写入，尚未形成runner级双文件原子发布事务；exists-check也不是并发原子no-clobber。正式runner必须使用同目录临时文件、fsync、独占创建或原子rename，并以最终COMMIT绑定两个外部SHA。该项为runner集成P2，不影响当前support-only module诊断，但阻断formal deployment loader声明。

## 性能与协议边界

最终nested版本尚需真实最小复跑。此前共享`gamma`真实D8b support诊断中的所有正候选都未同时通过三场景Before-old、After-old和seen-new逐类非退化门；因此当前性能结论保持：

```text
PERFORMANCE_SUPPORT_GATE_NOT_PASSED
SUPPORT_ONLY_NO_GO_TRUE_Z0
QUERY_NOT_OPENED
125_MATRIX_NOT_OPENED
```

此外，真实D8b package authority仍为`LOCAL_PROTOCOL_REPAIR_REQUIRED`。即使module安全检查通过，也不得把当前结果登记为formal launch、confirmation、deployment evidence或论文性能结论。
