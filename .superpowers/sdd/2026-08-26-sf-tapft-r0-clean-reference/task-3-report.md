# Task3：阶段指标与统一refit schedule

## 可追溯清单

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|T3-1|Required interfaces|不可变`StageValidationMetrics`与`StageValidationRow`|`code/cvsrffi/target_only_progressive_adapt.py`、`tests/test_target_only_progressive_adapt.py`|verified|20项聚焦测试通过|覆盖完整指标与阶段best/end遥测|
|T3-2|Metric definitions|按注册类计算BA、macro-F1、floor、NLL、recall、margin、flips及仅许可参数距离|同上|verified|手算两类literal测试通过|许可距离只遍历A/B/C模型参数并排除head、buffer和完整state|
|T3-3|Selection ordering|阶段内按BA→floor→-NLL→macro-F1→mean margin→-distance字典序选best|同上|verified|equal-BA且floor与NLL反向的fixture通过|floor先于NLL破平|
|T3-4|Fold telemetry|每个非零A/B/C阶段记录best step/global step、best/end metrics并暴露到fold row|同上|verified|3-fold A/B/C行及零步阶段测试通过|`best_step_in_phase`和`best_global_step`均为1-based|
|T3-5|Unified schedule|各阶段取fold-best长度的保守lower median并写入selection result|同上|verified|指定fixture得到`(450,1100,400)`|无row阶段长度为0|
|T3-6|Protocol boundary|只使用当前fold不相交target-inner validation及support标签，不引入source/query/truth-sidecar/target-eval|同上|verified|group-disjoint、`query_opened=False`及既有审计测试通过|保留Tasks1–2 exact-state audit|
|T3-7|Non-goals|不全support refit、不写V2 bundle、不做query prediction、不实现R1、不改容量/loss/optimizer/top-k最终行为|同上|verified|diff反向审查|现有snapshot score与首fold候选返回逻辑未改|

## RED证据

- 测试命令：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest tests\test_target_only_progressive_adapt.py -q`
- 结果：`15 passed,5 failed`。
- 失败均为预期缺口：`stage_validation_rows`、`_stage_validation_metrics`、`StageValidationMetrics`、`_lower_median`和`selected_phase_steps`尚不存在；没有语法、fixture或环境错误。

## GREEN证据

- 同一测试命令结果：`20 passed`，退出码0。
- Task2精确状态回归`test_top_checkpoint_average_restores_nonpermitted_sinc_buffer_to_post_adapter_anchor`保持通过，`nonpermitted_changed_names==()`。
- `git diff --check`退出码0；仅报告Git的LF→CRLF工作区提示，没有空白错误。
- 测试仍显示既有Torch2.1兼容分支`torch.cuda.amp.GradScaler`的1条`FutureWarning`，不影响通过计数，本任务未扩张到兼容层清理。

## 实现与选择决策

- 新增冻结数据结构`StageValidationMetrics`和`StageValidationRow`；每个有inner validation的optimizer step计算BA、macro-F1、class floor、NLL、逐类recall、逐类true-class margin、正负flip和A/B/C许可模型参数距离。
- frozen validation logits在optimizer loop前只计算一次；计算后恢复student/head状态和CPU/CUDA RNG，避免改变Task2锚点及训练随机序列。
- 各阶段独立按`BA→floor→-NLL→macro-F1→mean margin→-distance`保留best，并另存该阶段最后一步的`end_metrics`。零步阶段不产生遥测行。
- 每个`SFTAPFTFoldRow`承载对应fold的阶段行；`SFTAPFTSelectionResult.selected_phase_steps`逐阶段对fold-best长度取lower median。指定四fold fixture严格得到`(450,1100,400)`。
- 现有top-k snapshot score、loss、optimizer、参数scope、最终checkpoint averaging和首fold适配候选行为未改变；未执行全support refit，也未加入query、V2 bundle、R1或新模型能力。
- 交付提交：本报告与代码、测试由同一Task3提交固定；最终OID以任务返回值为准。

## 自审与关注项

- 反向可追溯审查：7项verified，0项deferred，0项rejected，0项blocked；属于Task3严格设计同构，不是近似实现。
- 最高风险项是阶段遥测额外增加每个optimizer step的inner-validation指标与许可参数距离计算成本；这是简报明确要求的R0选择遥测，未改变科学输入边界或最终模型选择规则。
- 没有触碰`conversation_index/`、Tasks1–2所有权外文件、N607、query、source或远端状态。
