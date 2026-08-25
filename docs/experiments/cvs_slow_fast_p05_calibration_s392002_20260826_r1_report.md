# Slow-Fast P0.5地面校准与独立目标审计报告

- run ID：`cvs_slow_fast_p05_calibration_s392002_20260826_r1`
- 当前状态：`LOCAL_VERIFIED`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- Git提交：`ec51ac8c0ffacd78471a1304dbfedae2dace829f`
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`

## 候选与最小矩阵

- 主候选仅为`FAST_FILM_R8`；LOWRANK和COMMON不进入P0.5地面校准。
- source receiver-held-out episode固定`K=10`，覆盖`clean`、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`；support/query物理ID互斥，样本不足的receiver/scene明确跳过且不降低K。
- 比较四个预登记门控：`P05_Q90_HARD`、`P05_RELATIVE_K12`、`P05_RELATIVE_K08`和`P05_RELATIVE_K08_FOLD_LCB`。
- 本run只冻结纯deployment参数。完整source校准JSON不得被Phase2 runner打开；正式Phase2 config只抄入`p05_*`数值/布尔参数。

## 本地验证与独立审查

- Slow-Fast聚焦回归：`51 passed`。
- 语法编译：selection、scorer、diagnostics、calibration、runner和校准CLI全部通过。
- `git diff --check`通过。
- 唯一独立P0/P1审查最初发现Phase2打开完整source校准JSON；已删除`calibration_path`输入并改成纯row config参数。原问题定点复审结论：`FIXED`。

## N607预登记

- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/slow_fast_p05_ec51ac8c/checkout`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，沿用已验证的r3运行环境；N607不存在独立`ssr-gpu`可执行路径。
- GPU：`0`。预检时8张RTX3090均为0%利用率、1MiB显存占用，当前用户无活跃训练进程。
- 输入cache：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/ground_feature_cache.pt`
- 输入FILM bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/FAST_FILM_R8.pt`
- 输出root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_slow_fast_p05_calibration_s392002_20260826_r1`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_slow_fast_p05_calibration_s392002_20260826_r1.out`
- 预期artifact：`calibration.json`和完整stdout日志。
- release归档：本地`local_artifacts/slow_fast_p05_ec51ac8c/release.tar`→远端`releases/slow_fast_p05_ec51ac8c/release.tar`；本地SHA256=`7049af4c069b24405bce6575472ce4ab622068bc43a9b4ea85c7a830f9af74fe`。只进行这一次本地到远端归档SHA比较。

## 精确命令

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/calibrate_slow_fast_p05.py --ground-cache /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/ground_feature_cache.pt --film-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/FAST_FILM_R8.pt --output /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_slow_fast_p05_calibration_s392002_20260826_r1/calibration.json --k-shot 10 --seed 392002
```

## 技术停止规则

- 仅因协议/source/query越权、错误checkout或输入、输出覆盖、进程归属不清、无法启动、确定性重复异常、无校准artifact闭合而停止。
- 不因低性能或门控全部回退停止；低性能只进入分析。
- target验证只允许新的receiver／seed capsule。旧receiver20-1、旧方法seed392002对应的truth仅可用于回溯诊断，不得用于调参后重跑并声称独立收益。

## Capsule审计

- 现有V2使用capsule=`d18-reuse-validated-once-rx20-1-seed713101-m7282101-k10-new10`，方法seed=`392002`，不满足独立目标确认要求。
- 新receiver／新seed且未消费旧truth的`p2_min_v1`、`VALIDATED_ONCE` capsule状态：`PENDING_READ_ONLY_AUDIT`。
