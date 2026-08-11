# D92-E0OCF Hard12-v3 v3一次性发布报告

|字段|内容|
|---|---|
|run ID|`d92_e0ocf_5arm_hard12v3_20260811_v3`|
|状态|`ANALYZED / NO_E0_OCF25_PROMOTION`|
|代码commit|`7f2255ea0bb0f112ac17b83a78491f2f86b93549`|
|协议/范围|`p2_min_v1`；`DEVELOPMENT_ONLY_PSEUDO_BLIND_DISJOINT_STRESS_SCREEN`|
|矩阵|冻结12outer×5arm=60job，180scene-arm；selection SHA=`20edc97b914c1031d9f917c63dee8e8ddb94223b31750151a05aabf0375d65f1`|
|候选|`E0_OCF25`唯一primary；`E0_OCF50`diagnostic-only|
|本地证据|153/153主回归通过；核心、Task2、全分支独立复审均`P0=0，P1=0`|

v2在任何prepare/smoke/job之前因归档错误多一层`code/code`停止，无性能结果且不重试。v3只修正Git归档布局；方法、代码、配置、矩阵、seed、K、receiver、场景、阈值与停止规则全部不变。新归档已本地核对包含`code/cvsrffi/__init__.py`和运行入口。

## 交付物

|本地文件|bytes|SHA256|
|---|---:|---|
|`E:\type10-7\code\snapshots\d92_e0ocf_runtime_closure_e5e498fb.tar.gz`|4987604|`e5e498fb6023415f778d43e176a9e41e998f9aa20688c596f3a26967fc74841a`|
|`configs/stage2_d92_e0ocf_5arm_hard12v3_v1.json`|3045|`f16b21e03969b7a1710594a76aac03fe40d1f6b1b4fd361b15e4e45153f0a3cc`|
|本目录`launch.sh`|3603|`c56129e9b386a2a9b9a35d1a278ffc3d35da47103e322da011efa8229ca3c4c2`|

远端source root=`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0ocf_source_snapshot_20260811_v3`；output=`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0ocf_5arm_hard12v3_20260811_v3`；logs=`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0ocf_5arm_hard12v3_20260811_v3`；smoke=`$output/smoke`。

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0ocf_source_snapshot_20260811_v3 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

GPU0–7各一个shard。先真实checkpoint truth-free smoke，再执行完整60job。仅在P0、覆盖风险、launcher故障或两个不同outer同一prediction前异常指纹时停止；禁止按性能早停，fresh retry=false。健康完成预期为60 receipt、120 prediction/COMMIT、60 score、120 fit-audit文件/360scene rows、120 resource-audit、8/8 PASS summary。结果返回后追加同排性能/资源和晋级裁决。

## 运行闭合

真实checkpoint smoke状态为`D92_E0OCF_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS`，query truth/fit/update/selection/role/quota/global全部为`false`。完整运行60/60job、8/8shard PASS、failed=0；prediction/COMMIT/fit/resource分别为120/120/120/120，score与job receipt均为60，fit audit共360个scene rows；所有stderr与异常指纹为空，结束后进程、GPU和SSH均清理。完整artifact取回至`E:\type10-7\local_artifacts\d92_e0ocf_5arm_hard12v3_20260811_v3`。

## 冻结分析结果

analyzer状态=`ANALYZED`，verdict=`NO_E0_OCF25_PROMOTION`，`all_gates_pass=false`。下表均为10个performance outer的同排聚合；准确率、H、floor和forgetting单位为百分数，wall为注册中位耗时。

|arm|H|old BA|old floor|seen-new|forgetting|wall(ms)|peak bytes|query MAC|state bytes|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`D92_FULL`|64.2910|67.1389|37.5000|62.1500|15.3056|3008.440|2426880|6048|15193|
|`E0_FULL_ONLY`|64.5349|67.6111|38.5000|62.2833|14.8333|99.711|1527808|6048|15193|
|`E0_FIXED50`|64.1237|66.9444|37.1667|62.0167|15.5000|187.560|1652736|6048|15193|
|`E0_OCF25`|64.4727|67.4444|38.6667|62.2917|15.0000|186.106|1654784|6048|15193|
|`E0_OCF50`|64.3124|67.1389|37.6667|62.1917|15.3056|184.921|1693696|6048|15193|

### OCF25冻结门

|比较|ΔH(pp)|Δold BA(pp)|Δfloor(pp)|Δseen-new(pp)|Δforgetting(pp)|paired median wall|paired median peak|H非负行|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`OCF25−D92_FULL`|+0.1817|+0.3056|+1.1667|+0.1417|−0.3056|−95.10%|−706560B|6/10|未达到ΔH≥0.5pp与8/10|
|`OCF25−FULL_ONLY`|−0.0622|−0.1667|+0.1667|+0.0083|+0.1667|+96.30%|+10240B|4/10|H、old BA、forgetting失败|

OCF25的floor恢复信号存在：相对FULL_ONLY平均+0.1667pp且8/10行不降；但收益太小，并以H、old BA、forgetting和约一倍注册耗时为代价。因此固定旧类contrast融合不晋级，OCF50也只保留为负向诊断。

### 当前最优瘦身结论

`E0_FULL_ONLY`相对原`D92_FULL`：H`+0.2439pp`、old BA`+0.4722pp`、old floor`+1.0000pp`、seen-new`+0.1333pp`、forgetting`−0.4722pp`；paired median wall下降`97.63%`，paired median peak下降`645120B`，query MAC和永久state完全不变。其two-state fit在K5/K10均为2，相对D92的48/88分别减少`95.83%/97.73%`。

因此本轮数据支持：保留288维A、B、task-balanced C与F0 query head；关闭E Fisher/Pareto，并把D从full/block K折LOO-soft fusion缩为单次after full geometry。B0历史上没有明确算力收益且组合结果较弱，不作为当前主线。下一步应冻结`E0_FULL_ONLY`做完整Target125确认；Hard12-v3本身仍是development-only stress screen，不能替代正式125结论。
