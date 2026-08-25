# ERBT-IDR M2.9 TASR48 Phase2实验v2预登记

- run ID：`erbt_idr_m29_tasr48_screen_20260825_v2`
- 当前状态：`LOCAL_FIX_PENDING_VERIFICATION`
- protocol：`p2_min_v1`，只复用原`VALIDATED_ONCE`数据及匹配的`capsule_id/split_id`
- 方法与矩阵：完全复用v1冻结的TASR48实现、5个arm、2个receiver、3个K/new条件和seed `7282101`；不改变任何科学参数
- 新run原因：v1在prediction写出前因确定性`KeyError: 'quantization'`技术失败，0个合法prediction，必须保留现场并使用不可覆盖新输出
- 代码分支：`codex/m29-tasr48-20260825`
- 修复commit：待本次定点修复提交绑定

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
