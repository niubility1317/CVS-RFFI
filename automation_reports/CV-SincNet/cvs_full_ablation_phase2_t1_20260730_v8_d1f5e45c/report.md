# CVS-RFFI Phase2 T1 fresh v8运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2_t1_20260730_v8_d1f5e45c`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|先闭合真实formal CPU预检与D18 feature smoke，再补齐并启动设计报告中的Stage2 states 325行和Stage2-C 1425行|
|假设|绕开Torch2.1/NumPy2的Tensor`.numpy()`ABI后，既有正式runtime、prototype、v2谱和D18 package可直接复用并完整发布feature cache|
|比较目标|技术闭合对照为v7 CPU SIGSEGV；方法对照由冻结计划中的same-row arms定义|
|Git实现|`d1f5e45c72f20e6d81ea5d6fef5e05fcd5f56f0e`|
|状态|`LOCAL_VERIFIED / CPU_PREFLIGHT_PENDING / NO_PERFORMANCE_RESULT`|

## 本地变更与验证

|文件|目的|
|---|---|
|`code/cvsrffi/stage2_ablation_feature_builder.py`|将6×160 prototype的Torch→NumPy转换改为显式`cpu().tolist()`复制|
|`tests/test_stage2_ablation_feature_builder.py`|保留数值闭合并禁止正式路径恢复`.numpy()`ABI bridge|
|`analysis/full_ablation_phase2_traceability_20260729.md`|登记P2-TR-31|

`ssr-gpu`本地九文件117项通过；独立复审九文件100项及多dtype/非连续tensor等价检查通过，P0=0、P1=0；计划/release 12项通过；`git diff --check`通过。

## 复用输入与发布位置

不重验或重建D18数据，不要求跨批次数据一致。复用v5完整before/after predictor package、正式Phase1 deployment bundle、prototype和外层封存v2 component。

|字段|值|
|---|---|
|N607 Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260730_v8_d1f5e45c`|
|input|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v6_d1f5e45c`|
|states run ID|`cvs_full_ablation_phase2_states_t1_20260730_v6_d1f5e45c`|
|Stage2-C run ID|`cvs_full_ablation_phase2c_t1_20260730_v6_d1f5e45c`|
|states plan|`stage2_states_plan_d1f5e45c.json`，SHA256=`f8d4687b05ffeada700d9e1b76da30b5fbc4e8657e50e7e2b08cf97a767d77f2`|
|Stage2-C plan|`stage2c_screening_plan_d1f5e45c.json`，SHA256=`546fe838730f6c22d1bb3197213e9ae246c85874e63dbda810421898a186dcdf`|
|初始GPU|物理GPU2；若启动前占用变化，先更新本报告且确保每GPU总进程≤2|
|并发目标|8张GPU，每张最多2个训练/adapter进程，共最多16 slots；既有进程计入上限|

## CPU formal预检

先在不打开D18 package/query、不占GPU、不写远端项目输出的条件下运行：

1. `_load_formal_runtime`：checkpoint lineage=`1eb6d07b9d6339400892c5553f33261f40513922d4b08c907446e44e993307d7`，6 handles，signature/runtime parity PASS。
2. `_verified_deployment_prototypes`：返回`[6,288]`有限单位范数prototype，source checkpoint同上。
3. `_ground_spectrum_from_formal_v2_component`：读取`phase1_unsigned/package/component`，manifest SHA=`03b5761d9cfd0f09a6b64710f5ebe7c270314bf5d73215206e5e8cf84606448a`，返回`[160,r]`正归一weights、input count=84、outer seal=true。

任一失败时不启动GPU、不创建input/request/run/log根，保留证据并以新run ID返回本地修复。

## feature smoke精确命令

工作目录为release根，`PYTHONPATH=<release>/code`：

```text
CUDA_VISIBLE_DEVICES=2 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/build_full_ablation_stage2_feature_cache.py --before-package-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v3_5097a33d/smoke/rx_20_1_method_7283101/before/predictor --before-seal-path /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v3_5097a33d/smoke/rx_20_1_method_7283101/before/predictor.seal.json --before-seal-sha256 6089f40b0e3a9412609661ab1d159019ce7e8fbc3ff2e9d9ace2548895709274 --after-package-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v3_5097a33d/smoke/rx_20_1_method_7283101/new20/predictor --after-seal-path /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v3_5097a33d/smoke/rx_20_1_method_7283101/new20/predictor.seal.json --after-seal-sha256 220231dea15378e7fdbf000b9188cc0f293b30d142273818512507dd7882689d --phase1-deployment-binding-path /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_final/deployment_binding.json --ground-component-dir /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_unsigned/package/component --ground-manifest-sha256 03b5761d9cfd0f09a6b64710f5ebe7c270314bf5d73215206e5e8cf84606448a --phase1-prototype-path /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_normalized/deployment_prototype/phase2_zid_prototypes.pt --phase1-prototype-manifest-path /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260729_v3_6fd77c22/artifacts/phase1_normalized/deployment_prototype/phase2_zid_prototypes.json --expected-phase1-prototype-sha256 e0e10b671dec5088bcb6e59b475dc3a99060b0ccbc03581e345a5e953b6088f0 --expected-phase1-prototype-manifest-sha256 89c1f21a5476e8d6b6a27264af6505d9ac4ab6eb66b32ea1e54c1d21405fc527 --expected-phase1-bundle-sha256 1eb6d07b9d6339400892c5553f33261f40513922d4b08c907446e44e993307d7 --cache-output-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v6_d1f5e45c/smoke/rx_20_1_method_7283101/feature_k10 --phase2-data-status VALIDATED_ONCE --capsule-id d18-rx20-1-cache713101-m7283101-k10-new20 --split-id p2_min_v1-rx20-1-m7283101-s7283201-q7283301-d7282401-k10 --k-shot 10 --method-seed 7283101 --support-seed 7283201 --query-seed 7283301 --new-class-draw-seed 7282401 --device cuda:0
```

## 成功、停止与产物标准

- smoke必须exit0并产生Stage2-A/B/C各`features.npz`和`features.manifest.json`，共6文件；当前loader逐份重载PASS。
- audit必须记录basis`[160,r]`、正归一weights、input count=84、outer seal=true、`query_truth_opened=false`、`raw_dataset_opened=false`。
- smoke成功后只补剩余缺失package/cache/sidecar/registry/seal；所有输出不可覆盖。
- 输入全集闭合后启动states；states达到`ARTIFACTS_COMPLETE`后启动Stage2-C。
- P0或两个不同row在prediction前出现同一确定性异常指纹时，停止精确run-owned进程树；不按准确率停止。
- 正式矩阵第一行与first wave记录launched/completed/succeeded/failed、prediction/score counts、PIDs、GPU映射和异常指纹。
- 预期输出为每row prediction、behavior、quantization、resource、score和terminal artifact，以及完整registry/seal/summary；完成前不作性能结论。
