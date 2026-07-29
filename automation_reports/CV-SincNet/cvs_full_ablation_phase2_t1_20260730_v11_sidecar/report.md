# CVS-RFFI Phase2 T1 fresh v11 sidecar运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2_t1_20260730_v11_sidecar`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|使用已验证的v2→v3 scorer-side兼容路径发布25个canonical Stage2-A sidecar；仅在25/25成功后生成states registry与seal|
|Git实现|待主代理提交兼容修复后回填|
|状态|`PREREGISTERED / SIDECAR_PENDING / NO_PERFORMANCE_RESULT`|

## 复用与路径

|字段|值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260730_v11_sidecar`|
|package root|v9 package根`.../stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v7_d1f5e45c`中的48份，加v5 reuse root中的2份|
|feature root|v10的297份scope cache，加v8 smoke复用3份；300/300 loader已PASS|
|sidecar output root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v9_d1f5e45c`，启动前必须完全不存在|
|driver log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2_t1_20260730_v11_sidecar`，fresh且`noclobber`|
|controller|`<release>/artifacts/v11_sidecar_completion/sidecar_completion_controller.py`|

## 冻结矩阵与门

只发布5 receiver×5 method seed=25个K=10 canonical Stage2-A scoring sidecar。`rx20-1/method7283101`的before scorer来自v5 reuse，其余24个来自v9 package root。每个sidecar发布`truth_sidecar.json`与`scoring_manifest.json`，合计50文件。

sidecar使用最多8个CPU worker分波；每波结束统计标准化异常指纹，同一非空确定性指纹出现≥2次就停止后续dispatch。controller在创建任何v11 output/log目录前，将`<release>/code`置于`sys.path`首位，并使用`cvsrffi.stage2_scoring_sidecar.load_verified_scoring_sidecar`预检全部25份source：manifest/truth必须是普通非符号链接文件，manifest字段与schema精确、truth hash匹配，truth只含5个顶层字段、schema为v2或v3、stage为`stage2b`、receiver/seed匹配且rows非空；再由`stage2_metric_scorer._validate_truth_rows`逐份验证完整truth row合同，来源计数必须24+1。

每个进程exit0后仍不视为成功产物。controller从日志读取唯一JSON receipt，核对receipt双路径恰在任务output目录、source/published schema迁移正确且两个文件存在；再用`cvsrffi.stage2_metric_scorer.load_verified_scoring_sidecar`按receipt manifest hash正式验收，核对Stage2-A v3 truth、receiver/seed、predictor root/seal与source一致、truth rows逐行相等。每行记录`artifact_validated`；summary记录`process_succeeded`，并令`succeeded=validated`、`failed=effective returncode!=0`，保证`succeeded+failed=completed`；另记录`validated/validation_failed/not_launched/exception_fingerprints/systemic_stop/published_sidecar_files/registry_and_seal_authorized`。`published_sidecar_files`只统计已验证文件；仅`validated=25 AND failed=0`时授权registry/seal。合成exit0但缺文件或坏hash必须保持未授权。

## 本地启动前验证

|检查|结果|
|---|---|
|controller编译与`--help`|PASS|
|source逐行预检顺序|PASS；`_validate_truth_rows(truth)`先于任何v11 output/log目录创建|
|来源分布|PASS；v9 package root=24，v5 reuse root=1|
|summary不变量|PASS；`succeeded=validated`、`failed=effective returncode!=0`且`succeeded+failed=completed`|
|合成制品故障|PASS；exit0但文件缺失或manifest hash错误均fail-closed，registry/seal不授权|
|重复故障停发|PASS；首波8个相同验证故障后停止，未派发后续17个|
|相关测试|`conda run -n ssr-gpu python -m pytest -q tests/test_stage2_ablation_scoring_sidecars.py tests/test_stage2_metric_scorer.py tests/test_build_full_ablation_stage2_binding_registry.py`：29 passed|

## 完成后回填

回填Git commit、remote hashes/compile、fresh-root检查、first-wave与最终25行状态、sidecar文件数、registry/seal路径和逻辑/物理行计数。完整states artifact前不作性能结论。
