# v5复用641个COMPLETE的控制证据

## 目的和边界

本目录保存v4冻结矩阵的3个小型控制文件，用于构造“复用641个v4 COMPLETE physical，仅补跑其余709个physical”的新v5计划。未回收641个prediction大文件，未复制数据集或query truth，未重验数据、未修改N607，也未创建或启动v5。

v5必须使用新的不可覆盖run ID。641个COMPLETE只可在v5冻结计划显式绑定原physical ID、原row receipt和原registry关系并重新验证后复用；不得向v4补写或覆盖任何状态。

## 原始路径与SHA256

|文件|原始N607路径|字节|SHA256|
|---|---|---:|---|
|`sealed_plan.json`|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58/sealed_plan.json`|4697058|`cf1f98e1a17c8df52ee94c3f17b28df5e725be8cbee96cdfd5f97959dbc258cd`|
|`binding_registry.json`|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58/binding_registry.json`|2174832|`a413e160bcc9bfa0a6c40864b5f716c0dc5805192a3ff3ba73d708dbd2c430a4`|
|`cache_binding_index.json`|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58/cache_binding_index.json`|99809|`85fe57e8b6234a660e687165ac900439a2a0089a276fda20ec8e78fe1a7b190b`|

## 只读闭合检查

- `sealed_plan.json`包含1350个唯一physical、1425个logical、75个alias，冻结8张GPU、每卡2槽。
- `binding_registry.json`包含1425个binding；logical row集合与sealed plan完全一致。
- `cache_binding_index.json`包含75个cache entry；三个控制文件的candidate lock一致。
- 本地`terminal_evidence/runner_summary.json`包含1350个terminal status，其physical ID集合与sealed plan完全一致：641个`COMPLETE`，18个`FAILED`，691个`NOT_LAUNCHED_SYSTEMIC_STOP`。
- 709个v5补跑physical由`sealed_plan.physical_rows`减去runner summary中641个`COMPLETE` physical得到；不需要读取性能值。
- N607上641个`COMPLETE`的launch artifact均可解析，并能导出641个`row_execution_receipt.json`路径；641个receipt全部存在且可解析，missing=0、unreadable=0、physical binding mismatch=0。

本目录不包含prediction或receipt副本；v5 seal应使用原路径和哈希绑定，而不是复制大产物。
