# Task9：最终P1定点修复

## 结论

本轮只关闭最终审查指定的两个P1，状态均为`PASS`：

- P1-A：R6/R8现在只在共同685D特征空间执行OT。Phase1 bundle保存与`bank.task_keys`逐行严格绑定的int8多物理样本聚合descriptor；Phase2只解量化该字段，不再拼接`task_coefficients`。
- P1-B：promotion新增全场景全arm的`zero_id_count==0`门与逐场景support-CV/query P3 BA方向一致门。证据在truth打开前完成重算与跨artifact绑定；科学门失败保持`ANALYZED/NO_PROMOTION_TO_TARGET25`，不会被误报为技术失败。

## P1-A闭合

- 新增`marc_ot.task_domain_bank.int8.v1`，字段固定为task-key顺序、int8`[T,685]`值、fp16有限正scale、对称int16 zero-point、int64 aggregation count、support feature schema/config/dim及固定聚合方法。
- strict loader逐项核对bundle成员、task-key顺序与唯一性、T、685D、dtype、scale、zero-point、`aggregation_count>=2`及feature ABI；旧bundle、缺字段、错序、错几何、错dtype、非有限scale、非零zero-point、单样本count、重复key和sample-level成员全部fail closed。
- Phase1规范入口从每个episode的`query_adapt`显式`rx_i/day_i/view`与`episode.k_shot`构造`DeltaTaskKey`。不读取opaque token，不把capture拼入自由字符串；capture仍只参与episode语义与物理隔离。
- 默认descriptor partition为`query_adapt`。扩展selector只能显式选择`support`或`query_adapt`，`query_guard`被禁止。同partition facts不一致、key与facts不一致、物理ID重复、semantic覆盖漂移、bank key缺失或多余均在保存前拒绝。
- 每episode先用生产685D builder得到合法row mean，再按同一task key以物理样本数加权聚合；bundle不保存source IQ或sample-level source embedding。
- CLI`_bank_task_features()`只调用task-domain int8解量化并返回detached float32`[T,685]`。R6/R8真实`_default_stage_update()`测试已从Phase1 builder→strict save/load→解量化贯通，未发生维度失败。

## P1-B闭合

- runner audit新增`support_cv_evidence`。K≥2在最终选择后对每个真实held-out fold从原始state独立replay选中初始化与stage alpha，只用fit fold训练，并用固定nearest-prototype evaluator计算同row baseline/selected BA、class floor与差值；不信任自定义selection evaluator的自报数值。K1只记录`UNAVAILABLE_K1`。
- 正式config新增固定有限`zero_id_norm_threshold=1e-12`及`support_query_direction_tolerance_pp=1e-9`，两者均严格校验且被pilot结果冻结。
- prediction必须包含`query_z_id`。纯prediction preflight按固定阈值重算每artifact的zero-id row数，并要求与prediction receipt及pilot summary完全一致；缺失、错几何、非有限或count篡改均在truth打开前拒绝。
- support CV evidence同时写入training audit、support-state receipt、prediction receipt、adaptation summary与prediction summary；scorer在truth打开前校验字段全集、来源、fold count、有限范围、差值算术和五处完全一致。
- truth连接后，paired row保留control/candidate的zero-id与support CV。每个arm的promotion decision扫描全部paired rows，任何场景任何arm的zero-id非零都会使所有arm不晋级。
- 方向门逐场景比较candidate support CV BA delta与query P3 BA delta。support正向时query不得超容差负向；support负向时query不得超容差正向；support近零时query也必须在容差内。返回逐场景方向诊断与zero-id诊断。
- 六个原数值门全部通过但zero-id非零，以及support正向而query负向的完整score测试，结果均为`status=ANALYZED`、`next_state=NO_PROMOTION_TO_TARGET25`、`best_promotable_arm=null`。

## TDD与负测

- A项RED首先稳定复现缺少`TaskDomainDescriptor`、bundle字段、规范key构造、精确coverage及CLI仍输出低秩系数等失败；最小实现后转绿。
- B项RED分别复现runner无真实support CV summary、scorer不重算zero-id、不校验support差值、旧promotion错误通过zero-id/方向冲突；实现后逐项转绿。
- 篡改覆盖：task-domain缺字段/错序/几何/dtype/scale/zero-point/count/重复key/禁止sample成员；zero count summary、support CV adaptation summary、prediction receipt、support receipt held-out状态及K绑定。
- K1、未执行pilot或缺held-out选择证据继续作为`ANALYSIS_ONLY`；它们不伪造cross-fit evidence，也不被当成技术失败。

## 最终验证

- P1聚焦回归：112项通过，退出码0。
- Task1-9及相邻Stage2完整回归：403项通过，退出码0；仅有既有TorchScript弃用警告。
- CLI help：退出码0，显示唯一三个子命令`smoke/pilot/score`。
- 13个相关生产入口`py_compile`：退出码0。
- 正式JSON经`python -m json.tool`及CLI严格`_validate_config_payload()`验证通过；回读阈值为`1e-12`与`1e-09`。
- `git diff --check`：退出码0，无whitespace error；仅有仓库既有LF/CRLF提示。
- 未访问N607、未运行正式K10 pilot、未重验`VALIDATED_ONCE`数据、未读取query truth/role用于训练或选择、未修改正式`analysis/`或`docs/`。
- 精确stage只纳入本Task必要代码、config、测试与本报告；既有`conversation_index/`和`local_artifacts/`不纳入提交。
- 提交主题固定为`fix: close MARC-OT final P1s`；本报告所在提交无法自引用最终OID，local/remote OID一致性由提交后交付状态记录。

## 文件范围

- 生产代码：`marc_ot_phase1.py`、`meta_weight_bank_checkpoint.py`、`stage2_marc_ot_pilot.py`、`stage2_marc_ot_runner.py`、`stage2_marc_ot_scoring.py`、`run_stage2_marc_ot_pilot.py`。
- 配置：`marc_ot_k10_pilot_20260901.json`。
- 测试：`test_marc_ot_task8.py`、`test_meta_weight_bank_checkpoint.py`、`test_run_stage2_marc_ot_pilot.py`、`test_stage2_marc_ot_pilot.py`、`test_stage2_marc_ot_runner.py`、`test_stage2_marc_ot_scoring.py`。
- 报告：本文件。
