# ERBT-IDR M2.6 TD-SRC256多表征screen正式预登记报告

日期：2026-08-23
run ID：`erbt_idr_m26_td_src256_repr_screen_20260823_v1`
当前状态：`LOCAL_VERIFIED`
实现提交：`18f630dfc06c83591dd7810a2e60d4562b147150`
分支：`codex/m26-td-src256-20260823`

## 一、目标与证据边界

本实验以去RF32的D92 E0为唯一主基线，验证一个真正读取目标旧类support域偏移、但不允许query参与拟合的有界残差。除既有FFT96的正交包络/纹理分解外，本轮新增MGD96频谱幅度几何表征：低阶DCT去趋势残差、局部斜率和精确镜像不对称各32维。所有表征都由同一固定接收IQ的确定性FFT96派生，不增加物理K。

当前报告只完成实验前预登记，不含性能结论。性能只能在24行prediction完整、truth保持未打开并由独立scorer连接后报告。

## 二、冻结方法与矩阵

|arm|机制|
|---|---|
|`M24-D1-COMPILE-PARITY`|B0：去RF32 D92 E0|
|`M26-T1-G0-IDENTITY-DOMAIN-RESIDUAL`|T1：identity160目标域传输|
|`M26-T2-G0-SPECTRAL-DECOMP-RESIDUAL`|T2：Envelope32＋Ripple64稳健support残差|
|`M26-T3-G0-JOINT-TARGET-SHIFT-RESIDUAL`|T3：identity传输＋Envelope/Ripple|
|`M26-T4-G0-MAGNITUDE-GEOMETRY-RESIDUAL`|T4：MGD96稳健support残差|
|`M26-T5-G0-JOINT-MAGNITUDE-GEOMETRY-TARGET-SHIFT-RESIDUAL`|T5：identity传输＋MGD96传输|

screen固定为：

- receiver：`3-19`、`8-8`；
- method seed：`7282101`；
- 条件：K5/new20、K10/new5；
- 4个配对输入身份×6个arm＝24个方法行；
- 3个LEO场景/行，共72个场景单元；
- 强度网格：`{0,0.01,0.02,0.04}`；
- K1不在screen；未来full125中的K1必须逐位回退B0；
- query逐条在全部注册类上独立argmax。

完整125触发条件：T3或T5至少一者相对B0达到`ΔH>=0.002`、`N_help>N_harm`，且`min-old`和`min-new`下降均不超过0.005。达到后使用新run ID运行5 receiver×5 seed×5 K/new×6 arm＝750行完整矩阵；未达到则以screen负结果闭环，不启动局部或伪full125。

## 三、协议与Phase1锚点

- `protocol_schema=p2_min_v1`；
- 复用`phase2_data_status=VALIDATED_ONCE`且匹配原`capsule_id/split_id`；
- target域状态只由六个目标旧类support与Phase1六类聚合锚点估计；
- 新类support、query、query truth均不进入域状态；
- Phase1锚点只保存6×identity160和6×FFT96的int8中心及FP16尺度；
- 锚点绑定真实checkpoint、D19类映射和量化内容component ID；该component ID进入候选锁；
- 混合Phase1导出只筛选`dataset_role=source`，target行使用数为0。

正式输入：

- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- checkpoint SHA-256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- source导出：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_publication_adv3b02_fft96_singleview_20260714/leo_clear_weak.npz`
- D19类绑定：release内`analysis/d19_adv3b02_class_binding_20260717.json`
- feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`
- supplemental feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`

## 四、环境与不可覆盖路径

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m26_td_src256_repr_screen_20260823_v1`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m26_td_src256_repr_screen_20260823_v1`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：release内`code`
- prediction：CPU，`max-workers=2`；不占用训练GPU。

## 五、预登记命令

锚点构建：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/build_m26_phase1_spectral_anchor.py \
  --source-npz /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_publication_adv3b02_fft96_singleview_20260714/leo_clear_weak.npz \
  --output /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/control/m26_source_anchor.npz \
  --audit-output /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/control/m26_source_anchor_audit.json \
  --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 \
  --class-binding-json ../analysis/d19_adv3b02_class_binding_20260717.json
```

prediction：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/run_m26_td_src256_matrix.py \
  --run-id erbt_idr_m26_td_src256_repr_screen_20260823_v1 --matrix-kind screen \
  --feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features \
  --supplemental-feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features \
  --source-anchor /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/control/m26_source_anchor.npz \
  --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 \
  --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/predictions \
  --device cpu --max-workers 2
```

truth-last评分：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/score_m26_td_src256_matrix.py \
  --matrix-index /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/predictions/matrix_index.json \
  --scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars \
  --supplemental-scoring-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3 \
  --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/scores \
  --bootstrap-repeats 2000
```

汇总：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/summarize_m26_td_src256_matrix.py \
  --prediction-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/predictions \
  --score-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/scores \
  --output /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/results_summary.json
```

## 六、实验前证据与停止规则

- M2.6聚焦测试：18项通过；
- M2.4/M2.5/M2.6相邻回归：25项通过；
- `py_compile`和`git diff --check`通过；
- 独立审查首次P0=0、P1=2；最小修复后定点复审P0=0、P1=0、READY；
- 实现提交已推送且远端OID等于本地HEAD。

只允许因协议/query泄漏、错误矩阵/checkout、输出碰撞、进程归属不清、确定性重复异常、prediction不闭合或scorer连接错误停止；低性能不得作为技术停止理由。

预期artifact：24份`row_execution_receipt.json`、`matrix_index.json`且状态为`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`、B0及T1–T5各4行、24份same-row/four-state score、20份paired-vs-B0结果、`scored_matrix_index.json`和`results_summary.json`。最终报告必须覆盖总体、K/new、receiver、seed、scene、四状态、old/new、class、margin、中心角距、help/harm、F_within/F_std、域偏移、LOO可靠度、强度、门控、回退和资源。
