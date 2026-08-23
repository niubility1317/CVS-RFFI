# ERBT-IDR M2.7 B3条件频域共识否决screen正式实验报告

日期：2026-08-23
run ID：erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1
当前状态：LOCAL_VERIFIED / PRELAUNCH
实现提交：f231080d2e33223f99e64faf2b8414907562f826
分支：codex/m27-b3-spectral-veto-20260823

## 一、实验目标与证据边界

本轮继续以去RF32的D92 E0为唯一主基线B0，并保留已获得完整125正证据的M2.5 B3作为唯一性能分支。M2.7不新增独立类别logit，而是只判断B3相对B0的类别翻转是否得到第二种support-only频域表征的共识。

V1使用MGD96幅度几何；V2从同一固定received IQ构造32维相位增量、相位曲率、镜像相位相干和归一化倒谱差分。两种表征均不增加物理K，不读取RF32，不读取query truth/role，不更新query状态。目标域共享中心只由当前row合法旧类target support的类平衡稳健中心估计。

本报告在prediction、truth-last评分和screen裁决完成前不包含性能结论。首轮仅是4个paired identity的机制筛选，不能与历史full125绝对指标直接作最终排序。

## 二、冻结方法与矩阵

|arm|角色|
|---|---|
|M24-D1-COMPILE-PARITY|B0：去RF32 D92 E0主基线|
|M25-B3-G0-STABLE-DUAL-PROTOTYPE-RESIDUAL|B3：完整125已验证性能分支|
|M27-V1-B3-MGD-CONSENSUS-VETO|V1：MGD96只接受或否决B3完整分数行|
|M27-V2-B3-PHASE32-CONSENSUS-VETO|V2：Phase/Cepstral32只接受或否决B3完整分数行|

冻结screen：

- receiver：3-19、8-8；
- method seed：7282101；
- 条件：K5/new20、K10/new5；
- paired input identity：4；
- method row：16；
- 每行3个LEO弱场景，共48个场景单元；
- prediction固定CPU，max-workers=2。

未来full125只有在screen门槛通过后才允许使用新run ID启动。完整125固定为5 receiver×5 seed×5 K/new×4 arm＝500行；K1因无法做support留一可靠性校准，V1/V2必须逐行精确回退B0。

## 三、协议和正式输入

- protocol_schema=p2_min_v1；
- phase2_data_status=VALIDATED_ONCE；
- 复用原capsule_id、split_id、support/query物理身份和三场景；
- 每个query独立在全部已注册类上argmax；
- query及其数学视图仅参与逐样本推理；
- truth只允许在16行prediction全部闭合后由独立scorer连接。

正式输入：

- feature root：/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features
- supplemental feature root：/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features
- scoring root：/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars
- supplemental scoring root：/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3
- Stage2-C binding index：/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v3_47212437/cache_binding_index.json
- checkpoint：/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth
- checkpoint SHA-256：2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98

4个输入身份均已只读确认VALIDATED_ONCE，且package root/seal、feature payload/manifest存在并一一匹配。

## 四、环境与不可覆盖路径

- 远端项目根：/home/szu2070436088/2510044040/CV-SincNet
- release root：/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1
- run root：/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1
- log root：/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1
- Phase32 root：/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1/control/phase_caches
- Python：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
- CWD：release内code目录
- prediction设备：CPU

2026-08-23T21:55:51+08:00直连N607预检确认上述release/run/log目标均不存在；正式输入和Python存在。GPU0–2有其他任务，GPU3–7空闲，本轮不占用GPU。

## 五、本地验证与独立审查

- M2.7聚焦测试：17项通过；
- M2.5–M2.7相邻回归：50项通过；
- M2.7模块、builder、runner、scorer、summarizer和row executor的py_compile通过；
- git diff --check通过；
- K1状态级和退化完整row测试均证明V1/V2精确回退B0；
- 独立审查首次P0=0、P1=1，发现scorer接受自声明缩小矩阵；
- 修复后唯一一次定点复审为P0=0、P1=0、READY；
- 实现提交已推送，远端OID=f231080d2e33223f99e64faf2b8414907562f826，与本地HEAD一致。

## 六、预登记执行命令

### 6.1 Phase32缓存

以下命令分别执行4次，参数由已冻结的Stage2-C binding直接展开；输出目录均采用不存在路径。

第1行：rx3-19、K5/new20

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/build_m27_phase_side_cache.py --base-feature-cache-payload /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_3_19/method_7282101/new20/k_5/stage2c/features.npz --base-feature-cache-manifest /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_3_19/method_7282101/new20/k_5/stage2c/features.manifest.json --base-feature-cache-payload-sha256 7672346cfdff969e265cb133f1d30b2fe2033455821352a2c070bfaccc76bcdc --base-feature-cache-manifest-sha256 63d2c1bdb9ddafb3a649debc1cf2da1365a850de10fccc76a25b80976d2c102c --after-package-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e/artifacts/packages/rx_3_19/method_7282101/new20/predictor --after-seal-path /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e/artifacts/packages/rx_3_19/method_7282101/new20/predictor.seal.json --after-seal-sha256 4292a61e370e691f1adc2d8f0d946a1a0efd9cce4a911c5e336ec4e616c1b7a5 --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1/control/phase_caches/rx_3_19/method_7282101/new20/k_5/stage2c

第2行：rx3-19、K10/new5

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/build_m27_phase_side_cache.py --base-feature-cache-payload /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_3_19/method_7282101/new5/k_10/stage2c/features.npz --base-feature-cache-manifest /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_3_19/method_7282101/new5/k_10/stage2c/features.manifest.json --base-feature-cache-payload-sha256 5ad18eea6cac4c7395cc8eddbaac82a61f4c42d11e7f900ab41859687b61bd3c --base-feature-cache-manifest-sha256 187d738385079b60919a90eecf01457ffe7d1a3fbfbb957e69b72d908aff623f --after-package-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e/artifacts/packages/rx_3_19/method_7282101/new5/predictor --after-seal-path /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e/artifacts/packages/rx_3_19/method_7282101/new5/predictor.seal.json --after-seal-sha256 0c08a826394d3729d774850d3b1007d42b62345600963c562273a5b14539b9c4 --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1/control/phase_caches/rx_3_19/method_7282101/new5/k_10/stage2c

第3行：rx8-8、K5/new20

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/build_m27_phase_side_cache.py --base-feature-cache-payload /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_8_8/method_7282101/new20/k_5/stage2c/features.npz --base-feature-cache-manifest /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_8_8/method_7282101/new20/k_5/stage2c/features.manifest.json --base-feature-cache-payload-sha256 373fcea50ec4f1119f1c961845eb9d9030772a210e4c6b2c96e9ca573f97f949 --base-feature-cache-manifest-sha256 038a211696c8116bf6be6249acb3102b5009cec777d62e7978242272b2fba149 --after-package-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e/artifacts/packages/rx_8_8/method_7282101/new20/predictor --after-seal-path /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e/artifacts/packages/rx_8_8/method_7282101/new20/predictor.seal.json --after-seal-sha256 bb0bd19621a36f64b5dec2586a916780366066c889b4c374b7828a91d933c422 --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1/control/phase_caches/rx_8_8/method_7282101/new20/k_5/stage2c

第4行：rx8-8、K10/new5

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/build_m27_phase_side_cache.py --base-feature-cache-payload /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_8_8/method_7282101/new5/k_10/stage2c/features.npz --base-feature-cache-manifest /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_8_8/method_7282101/new5/k_10/stage2c/features.manifest.json --base-feature-cache-payload-sha256 ac3bf73efd000f2a86ad572e5a4bc407a1360b5a734634c736dd0c04b6fef160 --base-feature-cache-manifest-sha256 c3b9b4e3e8bf60022539a72a8b0860b3577f3f2f5e3a9aa10d104d8e4c941ba5 --after-package-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e/artifacts/packages/rx_8_8/method_7282101/new5/predictor --after-seal-path /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e/artifacts/packages/rx_8_8/method_7282101/new5/predictor.seal.json --after-seal-sha256 e4cac9f3c92dc17e948e633981fa7ed7aa0fdd1669e5387d30bd2850b080b779 --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1/control/phase_caches/rx_8_8/method_7282101/new5/k_10/stage2c

### 6.2 prediction

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/run_m27_spectral_veto_matrix.py --run-id erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1 --matrix-kind screen --feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features --supplemental-feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features --phase-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1/control/phase_caches --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1/predictions --device cpu --max-workers 2

### 6.3 truth-last scorer

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/score_m27_spectral_veto_matrix.py --matrix-index /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1/predictions/matrix_index.json --scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars --supplemental-scoring-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3 --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1/scores --bootstrap-repeats 2000

### 6.4 汇总

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/summarize_m27_spectral_veto_matrix.py --prediction-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1/predictions --score-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1/scores --output /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1/results_summary.json

## 七、启动前和运行停止规则

启动前还必须完成：

1. 唯一release归档的本地/远端SHA-256一致；
2. release远端一次py_compile；
3. 真实checkpoint无query smoke，严格加载missing=0、unexpected=0，输出有限且query_input_count=0；
4. Phase32四个cache均闭合且query truth/role字段为false；
5. 启动后一次PID/CWD/cmdline/run root和日志增长核对。

只允许因协议/query泄漏、错误矩阵/checkout、输出碰撞、进程归属不清、确定性重复异常、prediction不闭合或scorer连接错误停止。低性能不得作为技术停止理由。

prediction结束后必须先确认：

- matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED；
- row_count=16；
- paired_input_identity_count=4；
- B0、B3、V1、V2各4行；
- scenario_unit_count=48；
- query_truth_opened=false；
- B0注册前后与历史头prediction disagreement均为0。

只有以上全部通过后才运行truth-last scorer。

## 八、screen晋级门槛

V1或V2只有同时满足以下条件才允许用新run ID进入完整125：

- 相对B0：DeltaH>=0.002；
- 相对B3：DeltaH>=0.0002；
- 相对B0：N_help>N_harm；
- 相对B0：min-old下降不超过0.005；
- 相对B0：min-new下降不超过0.005。

若无候选通过，正式裁决为SCREEN_NEGATIVE_NO_FULL125，不启动局部矩阵或伪full125。最终分析必须覆盖总体、K/new、receiver、seed、scene、四状态、old/new、class、margin、中心角距、help/harm、F_within/F_std、目标中心、可靠度、共识接受/否决和资源。
