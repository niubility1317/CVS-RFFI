# CVS Cached Slow-Fast Domain Adapter诊断实验报告

- run ID：`cvs_cached_slow_fast_diag9_s392002_20260825_r1`
- 当前状态：`LOCAL_VERIFIED / PRELAUNCH`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 实现与9行矩阵配置提交：`f41f515fa86bb265a4594e9d8b506dd247ce10da`；push后远端OID独立回读一致。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`

## 候选与矩阵

- `COMMON_SHIFT_R4`：地面类中心化SVD方向，目标support闭式估计4个公共偏移系数。
- `FAST_FILM_R8`：地面学习`U/V/rho/eta0/统一步长`，目标support只更新16个`gamma/beta`快参数。
- `FAST_LOWRANK_R8`：在FAST-FILM基础上增加8维方向gate，目标support只更新24个快参数。
- 诊断矩阵固定receiver=`20-1`、seed=`392002`、`K10/new10`，三候选×`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`，共9行。
- Phase2沿用`p2_min_v1`、`VALIDATED_ONCE`、capsule=`d18-reuse-validated-once-rx20-1-seed713101-m7282101-k10-new10`、split=`p2_min_v1-rx20-1-m7282101-s7282201-q7282301-d7282401-k10-new10`，不因候选变化重验数据。

## 本地实现与验证

- 新增Phase1.5地面缓存、两阶段慢参数/episodic快更新训练、严格部署bundle、support-only LOO门控、K=1 DA0回退、真实checkpoint无query smoke、同row runner、diag9矩阵和truth-last scorer。
- Phase1.5只读取既有source划分中的`L_s`。附件建议缓存day0～3，但CVS正式协议把day2～3作为clean test，因此本次优化为只使用source day0～1，避免把测试域混入地面训练。
- 正式特征宽度从冻结原型和bundle运行时读取，当前预期160，不沿用附件示例中的256。
- 24项本方法测试和18项邻近truth-last scorer回归通过，共42项；生产入口编译通过。
- 唯一一次独立P0/P1审查：P0=0，P1=4；已定点修复episodic外循环、support拟合`rho`一致性、`lambda=0.75`和diag9同输入约束，定点复审未新增检查项。

## N607预检与路径

- 普通`N607`账户；环境`/home/szu2070436088/.conda/envs/CVS-RFFI`；GPU0。
- 2026-08-25 17:09 CST只读预检：项目根可见，GPU0～7利用率0、显存1MiB，无本run进程。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_cached_slow_fast_diag9_s392002_20260825_r1/checkout`
- Phase1.5输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r1`
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_diag9_s392002_20260825_r1_smoke.json`
- prediction输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_diag9_s392002_20260825_r1`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_cached_slow_fast_phase15_s392002_20260825_r1.out`和`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_cached_slow_fast_diag9_s392002_20260825_r1.out`
- 输入文件只读回读：ManySig=2359341461字节，ADV3B02=8582116字节，冻结原型=4412字节；三场景support各140488字节、query各742926字节。
- release归档：本地`E:\type10-7\local_artifacts\cvs_cached_slow_fast_diag9_s392002_20260825_r1_release.zip`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_cached_slow_fast_diag9_s392002_20260825_r1/release.zip`；唯一一次本地/远端SHA256比较均为`29336fda0ae055c7785fbf3eb64999371f4c5d84b45635267c75b3081f2a36a6`。
- release checkout远端编译通过；独立回读确认核心源码、对应Python3.10字节码和diag9配置均存在。发布状态：`VERIFIED`。

## 精确命令

Phase1.5：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/train_slow_fast_phase15.py --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --base-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --prototype-path /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1/frozen_prototypes.npz --source-config configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r1 --base-checkpoint-id ADV3B02_CORE90_SOFT_E200 --device cuda:0 --batch-size 128 --steps 200 --meta-steps 200 --learning-rate 0.01 --seed 392002
```

无query smoke：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/smoke_slow_fast_no_query.py --config configs/stage2_slow_fast_smoke_common_s392002_20260825.json --output /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_diag9_s392002_20260825_r1_smoke.json --device cuda:0
```

diag9 prediction：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_slow_fast_matrix.py --config configs/stage2_slow_fast_diag9_s392002_20260825.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_diag9_s392002_20260825_r1 --device cuda:0
```

## 停止规则与预期artifact

- 仅在数据/query越权、错误checkpoint/capsule/split/receiver/K/scene、错误checkout、输出覆盖、进程归属不明、无法启动、prediction不完整或同一确定性pre-prediction异常至少出现两次时停止。不得因低性能停止。
- Phase1.5预期生成`ground_feature_cache.pt`、三个候选bundle和`phase15_summary.json`；缓存不得进入任何bundle。
- smoke预期`SMOKE_PASS`、`query_input_capability=false`、`query_opened=false`。
- diag9预期9行各自的`DA0_REG0/DA1_REG0`prediction与receipt，以及`matrix_receipt.json`。
- prediction全部闭合后才由独立scorer连接truth。候选只有同时满足聚合mean至少+1.0pp、floor至少+0.5pp且任一旧类退化不超过5pp，才建议进入Target25；否则记录`SCIENTIFIC_FAILURE_NO_PROMOTION`。
