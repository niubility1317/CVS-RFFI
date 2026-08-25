# CVS Cached Slow-Fast Domain Adapter诊断实验r2报告

- run ID：`cvs_cached_slow_fast_diag9_s392002_20260825_r2`
- 当前状态：`LOCAL_VERIFIED / PRELAUNCH`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 代码/config提交：`097773c513d184aeea23e9330ee1863bb1fdd673`；自动push后远端OID独立回读一致。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`

## 候选与矩阵

- 候选：`COMMON_SHIFT_R4`、`FAST_FILM_R8`、`FAST_LOWRANK_R8`。
- 固定receiver=`20-1`、seed=`392002`、`K10/new10`，三候选×三种LEO weak场景，共9行。
- 沿用`p2_min_v1`、`VALIDATED_ONCE`、capsule=`d18-reuse-validated-once-rx20-1-seed713101-m7282101-k10-new10`、split=`p2_min_v1-rx20-1-m7282101-s7282201-q7282301-d7282401-k10-new10`，不因方法变更重验数据。
- `DA0_REG0/DA1_REG0`同row共用checkpoint、原型、support、query和类别映射；query只读。K=1虽不在本矩阵，代码固定回退DA0。

## r1失败与r2修复

- r1在Phase1.5首个FAST objective因`gamma/beta`位于CPU、cache/慢基位于GPU而技术停止；没有prediction、truth或性能结果，部分artifact原地保留。
- r2定点修复FAST的`gamma/beta/direction_gate`按cache设备初始化；新增状态设备断言。修复后24项本方法测试和18项邻近scorer回归，共42项通过。
- 唯一一次P0/P1审查及定点复审已在r1前完成；本次只修复真实启动暴露的同一device问题，不追加全量审查。

## N607预登记

- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_cached_slow_fast_diag9_s392002_20260825_r2/checkout`
- Phase1.5输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r2`
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_diag9_s392002_20260825_r2_smoke.json`
- prediction输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_diag9_s392002_20260825_r2`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_cached_slow_fast_phase15_s392002_20260825_r2.out`和`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_cached_slow_fast_diag9_s392002_20260825_r2.out`
- GPU1；2026-08-25只读复核确认r2全部目标路径不存在，GPU1空闲。GPU0有不属于本run的PID`2991016`，本run不触碰。

## 精确命令

```text
CUDA_VISIBLE_DEVICES=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/train_slow_fast_phase15.py --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --base-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --prototype-path /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1/frozen_prototypes.npz --source-config configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r2 --base-checkpoint-id ADV3B02_CORE90_SOFT_E200 --device cuda:0 --batch-size 128 --steps 200 --meta-steps 200 --learning-rate 0.01 --seed 392002
```

```text
CUDA_VISIBLE_DEVICES=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/smoke_slow_fast_no_query.py --config configs/stage2_slow_fast_smoke_common_s392002_20260825.json --output /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_diag9_s392002_20260825_r2_smoke.json --device cuda:0
```

```text
CUDA_VISIBLE_DEVICES=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_slow_fast_matrix.py --config configs/stage2_slow_fast_diag9_s392002_20260825.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_diag9_s392002_20260825_r2 --device cuda:0
```

## 停止规则、artifact与晋级

- 仅在协议/query越权、错误checkpoint/capsule/split/receiver/K/scene、错误checkout、输出覆盖、进程归属不明、无法启动、prediction不完整或同一确定性pre-prediction异常至少出现两次时停止。不得因低性能停止。
- 预期Phase1.5 cache、三个bundle、summary；无query smoke receipt；9行两状态prediction/receipt和matrix receipt；prediction闭合后才生成truth-last score。
- 候选只有聚合mean≥+1.0pp、floor≥+0.5pp且任一旧类退化不超过5pp才进入Target25，否则记`SCIENTIFIC_FAILURE_NO_PROMOTION`。
