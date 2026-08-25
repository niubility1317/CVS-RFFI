# Slow-Fast P0.6空策略竞争重校准报告

- run ID：`cvs_slow_fast_p06_abstain_s392002_20260826_r1`
- 当前状态：`ANALYZED`
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

## N607执行闭合

- release归档远端SHA256与本地一致：`1d9be359b65ea7906433eae99619ddf396bcd4ab44a28d8bc9c4ac6c5986607c`；远端编译通过。
- 真实FILM bundle source-only smoke输出`CALIBRATED_TO_ABSTAIN/P05_ALWAYS_DA0`，`target_support_used=false`、`target_query_used=false`。
- 正式wrapper PID=`3226880`、Python PID=`3226882`，CWD和cmdline与本run绑定；进程自然退出。
- 日志39742字节，无`Traceback/ERROR/Exception`指纹。
- `calibration.json`独立回读：schema=`cvs.slow_fast.p05.calibration.v1`，28个episode、7个receiver、全部数值有限，无目标域访问。

## 实验结果

|策略|非零提交episode|全episode平均mean变化(pp)|全episode平均floor变化(pp)|最差receiver平均变化(pp)|最差episode floor变化(pp)|最大置信侵入代理|
|---|---:|---:|---:|---:|---:|---:|
|`P05_Q90_HARD`|12/28|+0.054945|+0.302198|0|0|0.00027455|
|`P05_RELATIVE_K12`|12/28|+0.054945|+0.302198|0|0|0.00027455|
|`P05_RELATIVE_K08`|12/28|+0.054945|+0.302198|0|0|0.00027455|
|`P05_RELATIVE_K08_FOLD_LCB`|12/28|+0.050366|+0.274725|0|0|0.00027455|
|`P05_ALWAYS_DA0`|0/28|0|0|0|0|0|

四个非零门控都消除了上一版的最差receiver负变化和最差episode floor负变化，说明fold-local normalizer、repeat稳定性和有方向margin约束确实改善了安全性。但它们没有在任何receiver上形成严格为正的最差平均收益，且置信侵入代理仍为正；`P05_ALWAYS_DA0`以相同的worst mean/floor、零侵入和零选择计算胜出。最终状态为：

```text
status=CALIBRATED_TO_ABSTAIN
selected_config=P05_ALWAYS_DA0
```

## 计算量解释

- 每个FAST自适应策略仍执行6个cross-fit fit×3步+1个full-support fit×3步=`21`次selection updates。
- `mean_gradient_updates=1.285714`只表示最终提交的平均更新数，即12/28个episode各提交3步；它不代表选择计算从21次下降到1.285714次。
- query只做冻结推理，`query_inference_updates=0`。
- `P05_ALWAYS_DA0`不执行cross-fit、full-support fit或committed update。

## 结论与下一阶段

1. 本轮证明P0.5旧结论确实存在空模型缺失偏差；强制从四个非零门控中选一个不合理。修复后系统正确选择拒绝适配。
2. +0.05pp左右的source平均信号远低于预登记的目标确认门槛，且不是独立Phase2目标性能；不得表述为快速域适应已经有效。
3. 不再继续调`trust_radius`、LCB或lambda网格。下一主候选进入`CVS_FACTORED_SLOW_FAST_ADAPTER_V2`：地面分离receiver rank4与LEO rank4慢基，星上只更新8个域上下文参数。
4. P1必须采用嵌套receiver留出：外层receiver完全不参与慢基拟合，内层source receiver选择策略，episode内部仅用support cross-fit。决策几何中心与冻结分类原型分开使用。
5. 在新的合法独立目标capsule出现前，不复用旧`rx20-1` query调参、重跑或声称确认收益。

最高交付状态：P0.6 source-only实现与校准为`ANALYZED/CALIBRATED_TO_ABSTAIN`；Phase2独立目标性能仍为`UNKNOWN/MISSING_INDEPENDENT_TARGET_CAPSULE`。
