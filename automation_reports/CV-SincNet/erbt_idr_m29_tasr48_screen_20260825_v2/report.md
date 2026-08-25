# ERBT-IDR M2.9 TASR48 Phase2实验v2预登记

- run ID：`erbt_idr_m29_tasr48_screen_20260825_v2`
- 当前状态：`LOCAL_VERIFIED`
- protocol：`p2_min_v1`，只复用原`VALIDATED_ONCE`数据及匹配的`capsule_id/split_id`
- 方法与矩阵：完全复用v1冻结的TASR48实现、5个arm、2个receiver、3个K/new条件和seed `7282101`；不改变任何科学参数
- 新run原因：v1在prediction写出前因确定性`KeyError: 'quantization'`技术失败，0个合法prediction，必须保留现场并使用不可覆盖新输出
- 代码分支：`codex/m29-tasr48-20260825`
- 修复commit：`c80f0f577ce3fd53928c7ef16724cb15ac384eff`

## 一、定点修复

1. 按真实接口读取`state.audit["compiler"]`中的量化审计，不再错误访问不存在的下一层`["quantization"]`；
2. 并行执行器只维持最多`max-workers`个已派发row；任一row失败后停止继续派发，取消尚未运行的future并保留已产生现场；
3. 不改TASR48、FFT96、identity160、D92、数据、评分器、矩阵或晋级阈值。

v1状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；v2只有在30个prediction receipt与`matrix_index.json`齐全后才启动truth-last scorer。

## 二、冻结输入与输出

- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m29_tasr48_screen_20260825_v2_r3`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v2`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m29_tasr48_screen_20260825_v2`
- 复用只读Phase1 bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v1/control/m29_tasr_bundle.npz`
- checkpoint SHA-256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`
- supplemental feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`
- scoring root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`
- supplemental scoring root：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3`
- CWD：release内`code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 设备：CPU，`max-workers=2`

## 三、正式命令

prediction：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/run_m29_tasr48_matrix.py \
  --run-id erbt_idr_m29_tasr48_screen_20260825_v2 \
  --feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features \
  --supplemental-feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features \
  --tasr-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v1/control/m29_tasr_bundle.npz \
  --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 \
  --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v2/predictions \
  --device cpu --max-workers 2
```

truth-last评分：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/score_m29_tasr48_matrix.py \
  --matrix-index /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v2/predictions/matrix_index.json \
  --scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars \
  --supplemental-scoring-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3 \
  --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v2/scores \
  --bootstrap-repeats 2000
```

## 四、停止规则、artifact与晋级门槛

仅在协议/query泄漏、错误receiver/seed/K/new/scene/split、错误checkout、输出碰撞、prediction不闭合、scorer连错truth，或同一确定性pre-prediction异常至少出现两行时技术停止。不得因性能低停止。

预期30个`predictions.cvspred`、30个`row_execution_receipt.json`、`matrix_index.json`、`scored_matrix_index.json`、30个same-row/four-state分数、24个paired比较及`results_summary.json`。

TASR48相对评分后最优FFT96对照必须同时满足：`Delta H>=0.002`、`N_help>N_harm`、old/new floor下降均不超过0.005、deployment state bytes更低，才允许进入多seed或完整125；否则以单seed负结果闭环。

## 五、本地验证

- 量化审计真实行回归先复现`KeyError: 'quantization'`，修复后通过；
- fail-fast回归先因缺少按需派发实现失败，修复后确认首行失败时只执行首行；
- M29及M2.4/M2.5/M2.8相关完整回归共40项通过；
- 6个M29模块/脚本Python编译通过；
- 本次修复不改变数据、方法、矩阵和评分合同。

## 六、v2 prediction与首次scorer结果

- r3 release HEAD：`29fb336d061f65d8f15335053e068140e6a842cd`；归档本地/远端SHA-256均为`16f44de512b4001f2d5a2ccbffd873186cea1e5210947513ebf64363c24df694`；远端编译通过。
- prediction主PID `3012439`及2个CPU worker的CWD、cmdline、父子关系和日志增长均通过启动后检查。
- prediction最终闭合：30个`predictions.cvspred`、30个`row_execution_receipt.json`和`matrix_index.json`齐全；独立读回为`row_count=30`、30个receipt均为`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`、`fit_query_rows_used=0`、`query_truth_opened=false`。
- 首次truth-last scorer仅创建空`scores`根，0个分数文件，随后因M29行为receipt中的`full_block_weights={full:1,block3:1}`不满足旧评分合同“和为1”而停止；prediction已经先完整闭合，scorer结果没有反馈预测器。
- 根因是行为receipt兼容字段的固定值笔误：M29使用全空间D92，合法旧合同表示应为`{full:1,block3:0}`。这不影响已冻结预测值、量化结果、资源值或数据协议。
- 定点修复同时覆盖两条路径：新prediction写合法权重；scorer只对已闭合M29旧receipt的唯一已知`1/1`指纹映射为`1/0`，已合法`1/0`原样保留，其他权重fail-closed。原prediction和空`scores`现场不修改，新评分使用不可覆盖`scores_v2`。
- 红→绿回归已覆盖新receipt与旧receipt评分适配；完整相关回归41项通过。下一release使用不可覆盖`erbt_idr_m29_tasr48_screen_20260825_v2_r4`，不重跑prediction。
