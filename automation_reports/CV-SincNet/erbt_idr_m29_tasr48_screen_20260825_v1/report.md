# ERBT-IDR M2.9 TASR48 Phase2方法优化与实验预登记

- run ID：`erbt_idr_m29_tasr48_screen_20260825_v1`
- 当前状态：`LOCAL_VERIFIED`
- protocol：`p2_min_v1`，只复用`VALIDATED_ONCE`且匹配原`capsule_id/split_id`
- 设计来源：用户提供的FFT96替代设计报告
- 代码分支：`codex/m29-tasr48-20260825`
- 实现与冻结配置Git commit：`6127d67ee3aa14cec2691083e0eb8fda036a39bb`

## 一、落实结论

本轮不删除频谱分支，也不把FFT96随意替换为另一组96维统计量。新增`identity160+TASR48`，并与冻结的`identity160+FFT96`权重4、1、0.5及`identity160-only`做同输入、同D92、同评分比较。

TASR48严格按报告实现：

1. Phase1仅从`dataset_role=source`的FFT96构建receiver×class中心；
2. 对每类减去跨receiver类中心，计算receiver扰动协方差，固定取前8个方向；
3. 只发布int8全局均值、int8扰动基、一个真正变化的int8特征值向量及单个共享FP16量化尺度、int8 TASR48冻结缩放量及量化尺度元数据，不发布样本或样本数；`tau`由解码后的int8特征值确定性取中位数，不单独持久化；
4. Phase2按类等权计算support均值，用`lambda/(lambda+tau)`收缩后得到一个冻结96维校准量；
5. query只减去该冻结校准量，不更新任何状态；
6. 固定48维描述为9点对称平滑残差后的16维有符号均值、16维RMS、8维一阶差分RMS、8维二阶差分RMS；
7. 使用Phase1冻结median/MAD缩放并L2归一化；
8. D92直接在160、208或256真实维度上执行全空间/分块support-only LOO，不补零到288维。

DSQ、Gabor/PCEN、逐类频段选择和频谱能量加权均标记为`NONBLOCKING`且本轮不实现。

## 二、冻结矩阵与晋级门槛

- receiver：`3-19`、`8-8`
- method seed：`7282101`
- 条件：`K1/new20`、`K5/new20`、`K10/new5`
- arms：
  - `M29-FFT96-A4`
  - `M29-FFT96-A1`
  - `M29-FFT96-A05`
  - `M29-IDENTITY160`
  - `M29-TASR48-A1`
- 共6个paired input identity、30个方法row、90个scene unit。

TASR48相对评分后最好的冻结FFT96权重满足以下全部条件才允许进入多seed或完整125：

- 同row汇总`Delta H>=0.002`；
- `N_help>N_harm`；
- old-class floor下降不超过0.005；
- new-class floor下降不超过0.005；
- deployment state bytes低于对应FFT96对照。

未达到时以单seed负结果闭环，不启动完整125。

## 三、正式输入与输出

- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- checkpoint SHA-256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- Phase1 source NPZ：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_publication_adv3b02_fft96_singleview_20260714/leo_clear_weak.npz`
- feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`
- supplemental feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`
- scoring root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`
- supplemental scoring root：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m29_tasr48_screen_20260825_v1`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v1`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m29_tasr48_screen_20260825_v1`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：release内`code`
- 设备：CPU，`max-workers=2`，不占用训练GPU。

## 四、正式命令

Phase1聚合束：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/build_m29_phase1_tasr_bundle.py \
  --source-npz /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_publication_adv3b02_fft96_singleview_20260714/leo_clear_weak.npz \
  --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 \
  --class-binding-json ../analysis/d19_adv3b02_class_binding_20260717.json \
  --rank 8 \
  --output-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v1/control/m29_tasr_bundle.npz \
  --audit-json /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v1/control/m29_tasr_bundle_audit.json
```

prediction：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/run_m29_tasr48_matrix.py \
  --run-id erbt_idr_m29_tasr48_screen_20260825_v1 \
  --feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features \
  --supplemental-feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features \
  --tasr-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v1/control/m29_tasr_bundle.npz \
  --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 \
  --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v1/predictions \
  --device cpu --max-workers 2
```

truth-last评分：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/score_m29_tasr48_matrix.py \
  --matrix-index /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v1/predictions/matrix_index.json \
  --scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars \
  --supplemental-scoring-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3 \
  --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v1/scores \
  --bootstrap-repeats 2000
```

## 五、直接技术停止规则与预期artifact

仅在以下情况停止：协议/query泄漏、错误receiver/seed/K/new/scene/split、错误checkout、输出路径已存在、相同确定性pre-prediction异常至少出现两行、prediction不完整或scorer连接truth错误。不得因中间或最终性能低而停止。

预期artifact：

- `control/m29_tasr_bundle.npz`
- `control/m29_tasr_bundle_audit.json`
- `predictions/matrix_index.json`
- 30个不可覆盖`predictions/<row-id>/predictions.cvspred`
- 30个`row_execution_receipt.json`
- `scores/scored_matrix_index.json`
- 30个same-row与four-state分数
- 24个TASR48相对四个对照的paired比较
- `results_summary.json`
- prediction完成前truth不打开，评分结果不得反馈预测器。

## 六、本地验证

- TASR48核心契约：5项通过；
- 真维度D92与表示契约：4项通过，其中包含一次160维真实D92拟合、编译和query评分；
- M2.4/M2.5/M2.8相关回归：25项通过；
- 合计：37项通过；
- Python编译检查：新增7个模块/脚本通过；
- 独立P0/P1审查：首轮发现3个P1、无P0；一次定点复审关闭2项并指出特征值仍为伪int8量化，现已改为单个共享尺度+真实变化int8向量，并由10项M29聚焦测试和NPZ成员读回闭合；按一次复审上限不再新增审查轮；
- 真实checkpoint无query smoke：待release后完成。
