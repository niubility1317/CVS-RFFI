# CVS-RFFI Phase2 T1 fresh v12 Stage2-B sidecar运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2_t1_20260730_v12_stage2b_sidecar`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|把25份已验证Stage2-B scorer truth从v2规范化发布为v3，stage与rows保持不变，为100个Stage2-B物理binding提供formal scorer输入|
|Git实现|待主代理提交Stage2-B v3 publisher与v12 controller后回填|
|状态|`PREREGISTERED / STAGE2B_SIDECAR_PENDING / NO_PERFORMANCE_RESULT`|

## 复用与路径

|字段|值|
|---|---|
|publisher release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260730_v12_stage2b_sidecar`|
|formal execution release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_states_t1_20260730_v12_d1f5e45c`；已fresh checkout到`d1f5e45c72f20e6d81ea5d6fef5e05fcd5f56f0e`且tracked/untracked clean|
|source package root|v9 package根中的24份before scorer，加v5 reuse root中的1份|
|Stage2-A sidecar root|v11已25/25 formal loader通过；只供Stage2-A binding，不冒用为Stage2-B|
|Stage2-B output root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v10_d1f5e45c`，启动前必须完全不存在|
|driver log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2_t1_20260730_v12_stage2b_sidecar`，fresh且`noclobber`|
|controller|`<publisher release>/artifacts/v12_stage2b_sidecar_completion/sidecar_completion_controller.py`|

## 冻结发布与门

只发布5 receiver×5 method seed=25个canonical Stage2-B v3 scoring sidecar。`rx20-1/method7283101`来源为v5 reuse，其余24个来源为v9 package root。每个source必须先由source loader验证manifest/truth、detached truth hash、receiver、seed、stage=`stage2b`、v2/v3 schema、非空rows，并在创建任何v12 output/log目录前调用`stage2_metric_scorer._validate_truth_rows`逐行验收；来源计数必须24+1。

publisher只允许把顶层schema升级为`cvs.phase2.query_truth_sidecar.v3`；stage保持`stage2b`，rows逐行相等，predictor package root/seal绑定不变。每个进程exit0后，controller仍须读取唯一JSON receipt、核对输出路径与schema迁移，再由formal `stage2_metric_scorer.load_verified_scoring_sidecar`逐项验收。summary区分`process_succeeded`与`validated`，令`succeeded=validated`、`failed=effective returncode!=0`并保证`succeeded+failed=completed`。

最多8个CPU worker分波。同一非空确定性异常指纹出现至少2次时停止后续dispatch。只有25/25均`artifact_validated=true`、`failed=0`且实际发布25份truth+25份manifest时，才允许构建125物理binding registry与325逻辑行sealed plan。否则保留原根并封口为`NO_PERFORMANCE_RESULT`。

## 完成后回填

回填Git commit、独立复审、远端hash/compile/import、fresh-root检查、首波和最终25项状态、formal loader计数、registry/seal计数及正式states runner PID。不得重做D18数据校验，不读取性能值。
