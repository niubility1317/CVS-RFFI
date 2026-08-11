# D92-E0OCF Hard12-v3 v3一次性发布报告

|字段|内容|
|---|---|
|run ID|`d92_e0ocf_5arm_hard12v3_20260811_v3`|
|状态|`LOCAL_VERIFIED / RELEASE_READY / NOT_LAUNCHED`|
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
