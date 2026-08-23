# ERBT-IDR M2.7 B3条件频域共识否决screen正式实验报告

日期：2026-08-23
run ID：erbt_idr_m27_b3_spectral_veto_repr_screen_20260823_v1
当前状态：ANALYZED / SCREEN_NEGATIVE_NO_FULL125
实现提交：f231080d2e33223f99e64faf2b8414907562f826
分支：codex/m27-b3-spectral-veto-20260823

## 一、实验目标与证据边界

本轮继续以去RF32的D92 E0为唯一主基线B0，并保留已获得完整125正证据的M2.5 B3作为唯一性能分支。M2.7不新增独立类别logit，而是只判断B3相对B0的类别翻转是否得到第二种support-only频域表征的共识。

V1使用MGD96幅度几何；V2从同一固定received IQ构造32维相位增量、相位曲率、镜像相位相干和归一化倒谱差分。两种表征均不增加物理K，不读取RF32，不读取query truth/role，不更新query状态。目标域共享中心只由当前row合法旧类target support的类平衡稳健中心估计。

prediction、truth-last评分和screen裁决均已完成。首轮是4个paired identity的机制筛选，只能使用同row差值判断候选，不能把其绝对指标与历史full125直接作最终排序。

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

## 八、执行闭合

本轮从实现提交`f231080d2e33223f99e64faf2b8414907562f826`生成唯一release归档。归档本地与远端SHA-256均为`b5b7a27d3c4d59e581a1e883f452b62838a7651f4c6dffd22bb18291bc607c4d`，远端一次性py_compile通过。真实checkpoint无query smoke严格加载195个状态张量，missing=0、unexpected=0、skipped mismatch=0，75个输出张量全部有限，`query_input_count=0`。

4个Phase32缓存全部闭合。每个缓存均声明`immutable=true`、`query_truth_opened=false`、`query_role_opened=false`、`raw_dataset_opened=false`、`source_or_clean_sample_opened=false`和`phase2_data_revalidated=false`。K5/new20每个receiver读取390个support received-IQ视图和1560个query received-IQ视图；K10/new5分别为330和660。所有视图来自同一固定接收IQ，不增加物理K。

prediction父PID为`1764409`，CWD和cmdline均绑定到本run release及唯一输出根。最终生成16/16个`row_execution_receipt.json`，四臂各4行，paired input identity=4，scenario unit=48，异常文件=0。`matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED`，`query_truth_opened=false`。4个B0行在注册前和注册后均与历史D92 E0头0分歧；24个M2.7场景审计均确认完整分数行只来自B0或B3、fit使用0个query、query不更新状态。

上述闭合后才启动truth-last scorer。scorer完成16行、65个评分JSON且错误0；汇总器生成`status=ANALYZED`。本轮不存在协议、执行、输出碰撞、错误checkout、prediction闭合或scorer连接失败。

## 九、总体结果

下表均为本screen内部同row结果，聚合采用query加权口径。F越低越好。

|arm|A_o_pre|A_o_post|A_n|H|F|min-old|min-new|H相对B0|H相对B3|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|B0：去RF32 D92 E0|0.742005|0.649362|0.588896|0.610486|0.092643|0.268919|0.251577|0|−0.005385|
|B3：M2.5稳定双原型残差|0.744482|0.655856|0.592838|0.615871|0.088626|0.281306|0.256532|+0.005385|0|
|V1：B3＋MGD96共识否决|0.742005|0.650338|0.590360|0.611788|0.091667|0.268919|0.251577|+0.001302|−0.004083|
|V2：B3＋Phase32共识否决|0.742005|0.649362|0.588896|0.610486|0.092643|0.268919|0.251577|0|−0.005385|

V1相对B0在A_o_post、A_n和H上分别提高0.000976、0.001464和0.001302，F降低0.000976，且两类floor不下降；但V1只恢复了B3小部分增益，相对B3的H下降0.004083。V2在全部总体指标上与B0完全一致，未形成有效决策变化。

## 十、预登记screen门槛裁决

|候选|DeltaH vs B0|DeltaH vs B3|help/harm vs B0|min-old变化|min-new变化|通过|
|---|---:|---:|---:|---:|---:|---|
|V1|+0.001302|−0.004083|6/0|0|0|否|
|V2|0|−0.005385|0/0|0|0|否|

V1满足help>harm和两类floor保护，但未达到`DeltaH vs B0>=0.002`，且明显未达到`DeltaH vs B3>=0.0002`。V2没有任何同row性能变化。两候选均未通过，因此正式裁决为：

`SCREEN_NEGATIVE_NO_FULL125`

依照预登记，不启动500行完整125、不使用局部矩阵补评分，也不把本screen包装为full125证据。

## 十一、K/new、receiver、seed与scene

### 11.1 K/new

|条件|B0 H|B3 H|V1 H|V2 H|V1−B0|B3−B0|
|---|---:|---:|---:|---:|---:|---:|
|K10/new5|0.719264|0.731394|0.719264|0.719264|0|+0.012130|
|K5/new20|0.564465|0.566995|0.566317|0.564465|+0.001852|+0.002531|

V1的全部收益来自K5/new20；在K10/new5上V1虽然部分场景的MGD96可靠度通过，但所有B3翻转均被否决，最终精确回退B0。V2两个条件均等价于B0。

### 11.2 receiver

|receiver|B0 H|B3 H|V1 H|V2 H|V1−B0|B3−B0|
|---|---:|---:|---:|---:|---:|---:|
|3-19|0.526178|0.527226|0.526178|0.526178|0|+0.001048|
|8-8|0.694794|0.704515|0.697398|0.694794|+0.002603|+0.009721|

MGD96的有效性高度集中在receiver 8-8：receiver 3-19的6个场景可靠度全部失败，receiver 8-8有5/6个场景通过。该不对称性说明当前目标域中心和竞争可靠度仍未获得跨receiver稳定性。

### 11.3 scene

|scene|B0 H|B3 H|V1 H|V2 H|V1−B0|B3−B0|
|---|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|0.670156|0.675919|0.672052|0.670156|+0.001896|+0.005762|
|leo_low_elev_weak|0.581155|0.588843|0.583164|0.581155|+0.002009|+0.007689|
|leo_rain_weak|0.580147|0.582850|0.580147|0.580147|0|+0.002703|

V1在clear和low-elevation场景有小幅正增益，在rain场景完全回退B0。V2三个场景均完全回退B0。

本screen只有method seed 7282101，因此seed切片与总体相同，不能据此提出跨seed稳定性结论。跨seed结论仍以M2.5 B3完整125证据为准。

## 十二、help/harm、old/new与类别

相对B0的注册后逐query变化为：B3 help/harm=28/5，净增23条正确预测；V1=6/0，净增6条；V2=0/0。V1的6条纠正中，新类5条、旧类1条；对应准确率分别提高0.001667和0.000694。V1的收益全部位于receiver 8-8和K5/new20，其中clear场景4条、low-elevation场景2条、rain场景0条。

相对B3，V1 help/harm=5/22，准确率下降0.003829；V2为5/28，下降0.005180。这个配对关系揭示了安全否决的真实代价：V1确实通过回退B0消除了B3的5条有害翻转，但同时丢失B3的22条有效纠正。

按26个类别汇总：B3有10类提高、1类下降、15类持平；V1有4类提高、0类下降、22类持平；V2为0类提高、0类下降、26类持平。V1提升最大的类别为`1-18`和`19-13`，均提高0.016667，其次为`19-6`提高0.008333、`6-15`提高0.004167。V1没有类别准确率下降，但这是以大量回退和低召回为代价的保护性结果。

## 十三、四状态与遗忘

|arm|DA0_REG0 old|DA0_REG1 old/new/H|DA1_REG0 old|DA1_REG1 old/new/H|
|---|---:|---:|---:|---:|
|B0|0.709234|0.548198/0.430563/0.478404|0.742005|0.649362/0.588896/0.610486|
|B3|0.709234|0.548198/0.430563/0.478404|0.744482|0.655856/0.592838/0.615871|
|V1|0.709234|0.548198/0.430563/0.478404|0.742005|0.650338/0.590360/0.611788|
|V2|0.709234|0.548198/0.430563/0.478404|0.742005|0.649362/0.588896/0.610486|

四臂的两个DA0状态完全一致，说明候选没有污染无域适应路径。旧类准确率的`DA1_REG0−DA0_REG0`分别为B0 +0.032770、B3 +0.035248、V1/V2 +0.032770；`DA1_REG1−DA0_REG1`分别为+0.101164、+0.107658、+0.102140和+0.101164。注册效应`DA1_REG1−DA1_REG0`分别为−0.092643、−0.088626、−0.091667和−0.092643；对应交互项分别为+0.068393、+0.072410、+0.069369和+0.068393。

标准化遗忘结果为：B3 `F_within=0.088626`、`F_std=0.086149`；V1均为0.091667；V2均为0.092643。V1只保留了B3的一部分遗忘改善，V2没有改善。

## 十四、margin、中心角距与资源

|arm|margin均值|margin p05|margin中位数|中心角距均值|中心角距p95|
|---|---:|---:|---:|---:|---:|
|B0|0.566556|0.018540|0.295723|24.968851°|34.432658°|
|B3|0.567031|0.018894|0.295723|30.596270°|41.015567°|
|V1|0.566980|0.018587|0.295723|49.526676°|104.122224°|
|V2|0.566556|0.018540|0.295723|40.847773°|55.266444°|

V1/V2中心角距来自各自的MGD96/Phase32竞争空间，与B0/B3分类头中心不在同一表征空间，不能直接把更大角度解释为更好的最终分类几何。最终决策margin显示V1只产生极小变化，V2与B0完全一致。

|arm|batch head latency ms/row|registration ms|state bytes|query head MAC|MAC upper bound|
|---|---:|---:|---:|---:|---:|
|B0|2.906|9.171|13921|5514|7848960|
|B3|34.641|10551.281|38032|9071|12300869|
|V1|48.790|11012.763|48992|11139|15244229|
|V2|29.279|10065.971|41685|9760|13281989|

V1相对B3增加约28.8%的状态、22.8%的query-head MAC和40.8%的当前batch head耗时，却未达到性能门槛。V2维度较低，但仍比B3增加约9.6%的状态和7.6%的query-head MAC，同时完全没有决策收益。资源证据进一步排除两者作为部署候选。

## 十五、机制定位

B3在4440条注册后query上产生49次相对B0的argmax翻转，其中28次纠正错误、5次破坏正确预测、16次只是不同错误类别之间切换。V1的MGD96竞争状态在12个scene级fit中仅5个通过可靠度，全部位于receiver 8-8；其余7个scene整行回退。V1最终接受6/49次翻转并否决43次。被接受的6次全部是有效纠正，接受精度为100%，但对B3有效纠正的召回仅为6/28=21.43%。被否决的43次包括22次有效纠正、5次有害翻转和16次中性错误切换。

因此V1不是方向错误，而是“高精度、低召回”的过保守筛选器。它成功证明第二表征可以识别一小组极可信B3翻转，却不能证明当前row级可靠度和固定margin阈值能覆盖足够多的B3收益。

V2的Phase32在12个scene级fit中可靠度通过数为0，平均LOO竞争准确率仅0.2691，旧类/新类分别为0.2875/0.2575；MGD96对应0.4523/0.5153/0.4158。V2否决全部49次B3翻转并精确退化为B0。当前相位增量、曲率、镜像相位相干和倒谱差分虽满足同IQ、全局相位和增益不变约束，但在少样本receiver域内不具备足够稳定的类别竞争信息。

目标域共享中心的类平衡稳健估计、旧类中心去共享偏移零均值和query只读边界均在工程与协议上成立；失败点不在“是否包含目标域状态”，而在该状态被压缩成row级二元可靠度后过度丢弃局部有效翻转。目标域偏移应继续用于风险校准和局部置信度，不应再作为独立类别头或整row开关。

## 十六、与去RF32 D92 E0和M2.5完整125的关系

|证据|矩阵|方法|H|同row DeltaH|help/harm|
|---|---|---|---:|---:|---:|
|D92 E0/M2.5历史主基线|完整125|去RF32 B0|0.537558|0|—|
|M2.5完整125|完整125|B3稳定双原型残差|0.539228|+0.001669|352/98|
|本轮screen|4 identity|B0|0.610486|0|—|
|本轮screen|4 identity|B3|0.615871|+0.005385|28/5|
|本轮screen|4 identity|V1|0.611788|+0.001302|6/0|
|本轮screen|4 identity|V2|0.610486|0|0/0|

本screen的绝对H较高是因为只包含两个receiver、一个seed和两个条件，不能与完整125绝对值直接排名；有效信息是同row差值。完整125已证明B3相对去RF32 D92 E0在5 receiver、5 seed和3 scene方向总体稳定；本screen进一步显示B3在所选困难/高收益identity上仍明显优于B0。M2.7 V1/V2均没有超过B3，因此不改变既有定位：去RF32 D92 E0仍是部署默认，M2.5 B3仍是模块二当前最佳完整125科学分支。

## 十七、下一轮优化建议

1.保留B0/B3完整分数和RF32删除状态，不再增加独立频域分类头，也不改变B3残差方向或幅度。
2.把row级“可靠/不可靠”开关改为逐query、逐候选类别对的support-only风险校准。输入只使用B3残差margin、MGD局部共识margin、目标中心距离、双原型jackknife稳定度和support类别对混淆；输出仍只能选择B0或B3完整行。
3.用support LOO标签拟合带Beta-Binomial收缩的局部翻转正确率，并设置最小样本回退。目标是保留V1当前100%接受精度，同时把对B3有效翻转的召回从21.43%提高到至少60%；没有足够support证据时仍精确回退B0。
4.目标域偏移继续由旧类target support类平衡估计，但应增加类内残差的稳健nuisance子空间或对角尺度，使receiver共享漂移与类别差异分离；该状态只调节风险阈值，不直接生成logit。
5.Phase32不再单独承担类别竞争。若继续保留，只作为MGD风险模型的低权重辅助量，并优先改为圆统计group-delay、跨bin相位相干和接收器内标准化的局部复谱比；任何新相位视图仍来自同一received IQ且不增加K。
6.下一候选建议命名为`M2.8 B3-LOCAL-CONFORMAL-FLIP-RISK`，仍先使用相同4 identity精确screen。只有同时达到本轮同一双基线门槛，才启动新的完整125 run。

## 十八、证据与发布路径

- 机器可读汇总：`results_summary.json`；
- prediction闭合索引：`evidence/matrix_index.json`；
- 16行执行收据：`evidence/receipts/`；
- truth-last评分：`evidence/scores/`；
- 真实checkpoint smoke与闭合核对：`evidence/control/`；
- Phase32缓存清单：`evidence/phase_caches/`；
- 运行日志：`evidence/logs/`；
- truth-blind诊断：`evidence/truth_blind_diagnostics/`；
- 实现追踪：`docs/ERBT_IDR_M27_B3_SPECTRAL_VETO_TRACE_20260823.md`；
- D92 E0综合对比：`docs/D92_E0_ALL_ABLATION_EXPERIMENTS_REPORT_20260819.md`。

完整prediction payload继续保存在N607唯一run root，不复制进Git；Git证据包含全部执行收据、评分JSON、truth-blind诊断、控制证据和日志，可从报告定位原始远端路径。
