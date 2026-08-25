# CVS Cached Slow-Fast Domain Adapter诊断实验r3报告

- run ID：`cvs_cached_slow_fast_diag9_s392002_20260825_r3`
- 状态：`LOCAL_VERIFIED / PRELAUNCH`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 代码/config提交：`40fc52441cd5ad3fc0c92883d0ef034ad974647f`；push后远端OID独立回读一致。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`

## 方法、矩阵与协议

- 三候选：`COMMON_SHIFT_R4`、`FAST_FILM_R8`、`FAST_LOWRANK_R8`；固定receiver=`20-1`、seed=`392002`、`K10/new10`和三种LEO weak场景，共9行。
- 沿用`p2_min_v1`、`VALIDATED_ONCE`、capsule=`d18-reuse-validated-once-rx20-1-seed713101-m7282101-k10-new10`、split=`p2_min_v1-rx20-1-m7282101-s7282201-q7282301-d7282401-k10-new10`。
- Phase1.5只读取source `L_s` day0～1；部署bundle不含cache。Phase2只更新4/16/24个快参数，query只读，并输出同row`DA0_REG0/DA1_REG0`。

## 重复失败恢复

- r1、r2均在Phase1.5首个FAST objective前因device mismatch技术停止；均无smoke、无prediction、无truth、无性能结果，输出原地保留。
- 第二次匹配指纹后停止同层重试，枚举Phase1.5全部张量构造点。r3统一将FAST参数和clean/LEO pair索引显式绑定`cache.features.device`。
- 新增“进程默认设备与cache设备不同”的回归测试，先RED后GREEN；本方法及邻近scorer共43项通过。正式启动前还必须在N607 GPU执行一次只验证pair索引设备的定点smoke。

## N607路径与资源

- GPU1；r3全部目标路径只读确认不存在，GPU1空闲。GPU0存在无关任务，本run不触碰。
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_cached_slow_fast_diag9_s392002_20260825_r3/checkout`
- Phase1.5：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3`
- smoke：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_diag9_s392002_20260825_r3_smoke.json`
- prediction：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_diag9_s392002_20260825_r3`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_cached_slow_fast_phase15_s392002_20260825_r3.out`和`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_cached_slow_fast_diag9_s392002_20260825_r3.out`
- release归档本地/远端唯一一次SHA256均为`1d908e09aa1a1ed5d90f9346d21b8a466f43def2826e5552e6aefd92a8083d39`；远端编译通过。
- N607 GPU1定点device smoke：`cache=cuda:0 clean_index=cuda:0 leo_index=cuda:0 status=PASS`。重复失败恢复`READBACK`闭合，允许r3启动。

## 命令

```text
CUDA_VISIBLE_DEVICES=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/train_slow_fast_phase15.py --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --base-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --prototype-path /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1/frozen_prototypes.npz --source-config configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3 --base-checkpoint-id ADV3B02_CORE90_SOFT_E200 --device cuda:0 --batch-size 128 --steps 200 --meta-steps 200 --learning-rate 0.01 --seed 392002
```

```text
CUDA_VISIBLE_DEVICES=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/smoke_slow_fast_no_query.py --config configs/stage2_slow_fast_smoke_common_s392002_20260825.json --output /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_diag9_s392002_20260825_r3_smoke.json --device cuda:0
```

```text
CUDA_VISIBLE_DEVICES=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_slow_fast_matrix.py --config configs/stage2_slow_fast_diag9_s392002_20260825.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_diag9_s392002_20260825_r3 --device cuda:0
```

## 停止与晋级

- 仅因协议/query越权、错误输入/checkout、输出覆盖、进程归属不清、无法启动、prediction不完整或重复确定性pre-prediction异常停止；不得因低性能停止。
- prediction闭合后才连接truth。mean≥+1.0pp、floor≥+0.5pp且任一旧类退化不超过5pp才进入Target25，否则记`SCIENTIFIC_FAILURE_NO_PROMOTION`。
