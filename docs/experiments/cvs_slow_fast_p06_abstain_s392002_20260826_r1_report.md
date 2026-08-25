# Slow-Fast P0.6空策略竞争重校准报告

- run ID：`cvs_slow_fast_p06_abstain_s392002_20260826_r1`
- 当前状态：`LOCAL_VERIFIED`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 实验代码提交：`8c15bb74fc1dacf4dd3e5882776d102e35792c08`
- 候选：`P05_ALWAYS_DA0`与四个原P0.5门控；主Adapter仍为`FAST_FILM_R8`
- 矩阵：source receiver-held-out、`K=10`、seed=`392002`，覆盖`clean`、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`，不读取任何Phase2目标样本。

## 本轮变更

- 空策略以mean/floor/intrusion均为0参与同一排序；非零策略不严格优于它时输出`CALIBRATED_TO_ABSTAIN`。
- 每个cross-fit方向只用train half计算强度normalizer。
- 每对互补fold先取平均，稳定性按repeat gain和repeat LCB判断。
- trust增加有方向margin约束：基线正确样本至少保留50% margin，基线错误样本必须严格改善margin。
- 计算量拆分为cross-fit、full-support fit、committed、total selection和query inference updates。

## 本地验证与审查

- 36项Slow-Fast聚焦回归通过。
- selection、calibration和runner语法编译通过；`git diff --check`通过。
- 唯一独立P0/P1定点审查：`NO_P0_P1`。

## N607预登记

- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/slow_fast_p06_8c15bb74/checkout`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 计算设备：CPU；校准只处理已缓存160维地面特征，不占用当前GPU训练槽位。
- 输入cache：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/ground_feature_cache.pt`
- 输入FILM bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/FAST_FILM_R8.pt`
- 输出root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_slow_fast_p06_abstain_s392002_20260826_r1`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_slow_fast_p06_abstain_s392002_20260826_r1.out`
- 预期artifact：`calibration.json`和stdout日志。
- release：本地`E:\type10-7\local_artifacts\slow_fast_p06_8c15bb74\release.tar`→远端`releases/slow_fast_p06_8c15bb74/release.tar`；本地SHA256=`1d9be359b65ea7906433eae99619ddf396bcd4ab44a28d8bc9c4ac6c5986607c`。

## 精确命令

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/calibrate_slow_fast_p05.py --ground-cache /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/ground_feature_cache.pt --film-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/FAST_FILM_R8.pt --output /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_slow_fast_p06_abstain_s392002_20260826_r1/calibration.json --k-shot 10 --seed 392002
```

## 技术停止规则

仅因source/query权限越界、错误checkout或输入、输出覆盖、不能启动、确定性异常或缺少`calibration.json`而停止。不因结果选择DA0或低性能停止。若空策略胜出，记录为科学结论并进入P1因子化慢基设计，不复用旧目标query重跑。
