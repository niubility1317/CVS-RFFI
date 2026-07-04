# phase2_adv3b02_arc_ci_20260704

## 基本信息

- 时间：2026-07-04
- 操作方：Codex
- 目标：实现并测试`OPC-ARC-CI`候选集空集拒识算法，优先改善unknown拒识，同时要求旧类准确率不低于OPU基线。
- 场景：Stage2-C target receiver domain，`R_t={20-1,3-19,7-14,7-7,8-8}`，`Y_old={14-10,14-7,20-15,20-19,6-15,8-20}`，`Y_new={19-3,3-8}`，`Y_unknown={10-1,10-10}`。
- 星地视图：`target_channel_view=leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。

## 算法

`OPC-ARC-CI`复用OPU-CI双路线证据，新增candidate-set空集拒识层：

1. 每个接收机只使用top-1/top-2标签、score、margin、support density、receiver-class reliability和support/proxy-derived risk。
2. 若top-1旧类满足强旧类条件，则将`unknown_risk`硬封顶，保护旧类接受。
3. 若known证据弱且tail/safety risk高，则提高empty-candidate unknown risk。
4. 评估器继续使用`support_quality_prior`逐步选择接收机，协同数量覆盖`1..|R_t|`。
5. `target_unknown_training_count=0`，`target_unknown_selection_count=0`，summary按预注册profile/policy/collab_count顺序输出，不用unknown评估指标选择profile。

## 本地变更

| 文件 | 目的 |
|---|---|
| `code/scripts/phase2_old_protected_arc_ci_eval.py` | 新增OPC-ARC-CI评估入口，构造强旧类保护和candidate-set空集拒识证据。 |
| `code/tests/test_phase2_old_protected_arc_ci_eval.py` | 覆盖强旧类风险封顶、弱候选空集风险提升、默认profile解析。 |

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_old_protected_arc_ci_eval.py code\tests\test_phase2_old_protected_arc_ci_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_old_protected_arc_ci_eval.py -q` | PASS，3 passed；`.pytest_cache`权限警告不影响测试 |

本地评估命令：

```bash
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_old_protected_arc_ci_eval.py \
  --feature_npz remote_artifacts\phase2_adv3b02_features\features.npz \
  --output_dir local_artifacts\phase2_adv3b02_arc_ci_20260704\local_all \
  --profiles all \
  --policies opu_old_preserve,opu_old_guarded \
  --max_event_latency_ms 20.0 \
  --force
```

本地结果：`local_artifacts\phase2_adv3b02_arc_ci_20260704\local_all`，30行，覆盖`collab_count=1..5`。

| profile | policy | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes_per_event | latency_ms_p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arc_old_floor | opu_old_preserve | 4 | 0.796791 | 0.300000 | 0.766667 | 0.725000 | 0.200000 | 0.716667 | 496.339 | 0.171549 |
| arc_balanced | opu_old_preserve | 4 | 0.673797 | 0.300000 | 0.700000 | 0.650000 | 0.466667 | 0.450000 | 496.339 | 0.171549 |
| arc_unknown_safe | opu_old_preserve | 4 | 0.459893 | 0.200000 | 0.433333 | 0.375000 | 0.766667 | 0.166667 | 496.339 | 0.171549 |

对OPU基线`opu_old_preserve,k=4`：`old_acc=0.802139`，`unknown_reject=0.183333`。本地未找到`old_acc>=0.802139`且`unknown_reject>0.183333`的ARC候选。

## N607执行

远端输入：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_orbit_c3r_guard_20260704/input/features.npz
```

该feature SHA256为`db559d78db305894307851750ef7d698db387f0984ff13c980fea99db85b8532`，与本地输入一致。

远端验证：

| 项目 | 结果 |
|---|---|
| N607预检 | PASS，直连`N607`，项目根可见，8张RTX3090均约10MiB显存占用。 |
| 远端Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python3.10.19。 |
| 同步后SHA | 脚本`e26513e5d0c2bd28554078e0988bb85b3a492723d94c07aa98a21468110fe8bb`；测试`73bd4eabd26fbba57a4e5bb6f44fa19e87726766830a42bdbac24cdb2a0b456e`。 |
| 远端编译/测试 | `py_compile`PASS；`python code/tests/test_phase2_old_protected_arc_ci_eval.py`，3 tests OK。 |
| SSH清理 | 每次SSH/SCP后检查，本地无`ssh.exe`残留，无到`172.31.111.215:22`的ESTABLISHED连接。 |

远端命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_old_protected_arc_ci_eval.py \
  --feature_npz runs/phase2_adv3b02_orbit_c3r_guard_20260704/input/features.npz \
  --output_dir runs/phase2_adv3b02_arc_ci_20260704 \
  --profiles all \
  --policies opu_old_preserve,opu_old_guarded \
  --max_event_latency_ms 2.0 \
  --force
```

GPU：选择GPU0；运行前后GPU0显存均为10MiB，未出现额外显存占用。

远端输出：

| 文件 | SHA256 |
|---|---|
| `runs/phase2_adv3b02_arc_ci_20260704/arc_ci_summary.csv` | `9427da1147e7a28a21ad3e4279038877e2863d46e49c88f47fb107435ca27075` |
| `runs/phase2_adv3b02_arc_ci_20260704/arc_ci_summary.json` | `46291dfe3c7acedddfccb9b086f464c71ae1ecb020a8d32ef73446d7ef3f196d` |
| `logs/phase2_adv3b02_arc_ci_20260704/arc_ci.log` | `b0753d1d84a7a6cfb4eedc973dc216e0de1b27602af005064272be0b7a9450eb` |

本地拉回：`remote_artifacts\phase2_adv3b02_arc_ci_20260704\`。

远端全量结果覆盖`collab_count=1..5`。`policy=opu_old_preserve`、`collab_count=4`同row结果：

| profile | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes_per_event | latency_ms_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arc_old_floor | 0.796791 | 0.300000 | 0.766667 | 0.725000 | 0.200000 | 0.716667 | 496.339 | 0.184152 |
| arc_balanced | 0.673797 | 0.300000 | 0.700000 | 0.650000 | 0.466667 | 0.450000 | 496.339 | 0.184152 |
| arc_unknown_safe | 0.459893 | 0.200000 | 0.433333 | 0.375000 | 0.766667 | 0.166667 | 496.339 | 0.184152 |

远端未找到`old_acc>=0.802139`且`unknown_reject>0.183333`的ARC候选。summary字段确认：`summary_order=pre_registered_profile_policy_collab_count`，`profile_selection_uses_target_unknown=False`，`target_unknown_training_count=0`，`target_unknown_selection_count=0`。

## 当前解释

ARC验证了候选集空集拒识方向，但在当前特征上仍未满足“旧类不下降”。说明后处理层已经触及上限，下一步需要进入表征/原型层：source/target-old shrinkage原型、class-wise EVT/Mahalanobis、弱星地多视图一致性和conformal选择性拒识的联合校准。TENT/SHOT/ODIN等依赖闭集测试批或OOD验证调参的在线适应路线不进入当前主线。
