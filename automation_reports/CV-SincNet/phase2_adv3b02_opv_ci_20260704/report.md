# phase2_adv3b02_opv_ci_20260704

## 基本信息

- 时间：2026-07-04
- 操作方：Codex
- 目标：在ADV3B02_CORE90_SOFT_E200高known特征和qknn8在轨少样本协议上，测试old-protected PCET veto协同拒识算法，尝试在旧类不下降的前提下改善未知类拒识。
- 场景：Stage2-C target receiver domain，`R_t={20-1,3-19,7-14,7-7,8-8}`，`R_s`不相交；`Y_old={14-10,14-7,20-15,20-19,6-15,8-20}`，`Y_new={19-3,3-8}`，`Y_unknown={10-1,10-10}`。
- 星地视图：`target_channel_view=leo_clear_weak,leo_low_elev_weak,leo_rain_weak`；feature内含target-old、target-new、target-unknown，target unknown只作eval-only query。
- 资源约束文档：未在`E:\type10-7`找到`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`或包含`资源约束`的同名文件。本轮按现有评估字段记录`collab_count`、`bytes_per_event`、`latency_ms_p95`、GPU显存占用和N607进程状态。

## 算法

新增`OPV-CI`：old-protected PCET veto on OPU-CI。

1. 保留OPU-CI dual-route已验证的known route和support-only safety route。
2. 在证据层追加query-free风险修正：
   - `opv_safety_unknown_risk`来自support-only safety route；
   - `opv_tail_risk`来自class-negative、class-shell、EVT、Mahalanobis等support/proxy派生风险；
   - `opv_proto_instability`由known score、margin、support density、receiver class reliability构造。
3. 若事件有强known证据，则封顶`unknown_risk`，保护旧类和seen-new接受。
4. 若known证据弱且tail/safety风险高，则提高`unknown_risk`，促使协同层拒识或request-more。
5. `target_unknown_training_count=0`，`target_unknown_selection_count=0`；target unknown只用于最终评估。

## 本地变更

| 文件 | 目的 |
|---|---|
| `code/scripts/phase2_old_protected_pcet_veto_ci_eval.py` | 新增OPV-CI评估入口，基于OPU dual-route证据做old-protected风险修正，输出逐profile、policy、collab_count结果。 |
| `code/tests/test_phase2_old_protected_pcet_veto_ci_eval.py` | 覆盖强known风险封顶、弱known高tail风险提升、默认profile解析、summary不按unknown指标排序。 |

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_old_protected_pcet_veto_ci_eval.py code\tests\test_phase2_old_protected_pcet_veto_ci_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_old_protected_pcet_veto_ci_eval.py -q` | PASS，4 passed；`.pytest_cache`权限警告不影响测试 |

本地Git镜像提交：

| commit | 内容 |
|---|---|
| `e0c1c34` | 新增OPV-CI脚本、测试和初始报告。 |
| `faa91ac` | 将summary顺序改为预注册profile/policy/collab_count顺序，避免用unknown评估指标隐式选择profile。 |

## 本地对比结果

输入feature：`remote_artifacts\phase2_adv3b02_features\features.npz`。

OPU-CI高known基线：

| policy | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_coverage | latency_ms_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opu_old_preserve | 4 | 0.802139 | 0.300000 | 0.783333 | 0.725000 | 0.183333 | 0.750000 | 0.959514 | 0.158417 |
| opu_old_guarded | 4 | 0.770053 | 0.300000 | 0.666667 | 0.650000 | 0.250000 | 0.550000 | 0.805668 | 0.158417 |

PCET-CI在同feature上的观察：

| profile | collab_count | old_acc | min_old | seen_new_acc | unknown_reject | unknown_FAR |
|---|---:|---:|---:|---:|---:|---:|
| pcet_known_preserving | 1 | 0.682540 | 0.000000 | 0.150000 | 0.433333 | 0.566667 |
| pcet_balanced | 4 | 0.132275 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |

结论：PCET能提升unknown，但会严重牺牲old/seen-new，不能直接替换OPU。

OPV-CI本地干净全profile结果：`local_artifacts\phase2_adv3b02_opv_ci_20260704\local_all_clean`。summary主顺序为`pre_registered_profile_policy_collab_count`，`joint_score_scope=evaluation_ranking_only_uses_target_unknown_metrics`，`profile_selection_uses_target_unknown=False`。

OPV-CI本地全profile结果中`collab_count=4`、`policy=opu_old_preserve`：

| profile | policy | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| opv_ultra_preserve | opu_old_preserve | 4 | 0.796791 | 0.300000 | 0.766667 | 0.700000 | 0.216667 | 0.700000 | 0.939271 |
| opv_preserve | opu_old_preserve | 4 | 0.770053 | 0.300000 | 0.716667 | 0.675000 | 0.366667 | 0.550000 | 0.898785 |
| opv_balanced | opu_old_preserve | 4 | 0.737968 | 0.300000 | 0.666667 | 0.625000 | 0.650000 | 0.350000 | 0.850202 |
| opv_unknown_push | opu_old_preserve | 4 | 0.614973 | 0.300000 | 0.533333 | 0.475000 | 0.766667 | 0.200000 | 0.704453 |

资源字段：

| profile | policy | collab_count | bytes_per_event | latency_ms_p95 | request_more_rate | defer_rate |
|---|---|---:|---:|---:|---:|---:|
| opv_ultra_preserve | opu_old_preserve | 1 | 168.000 | 0.246646 | 0.019544 | 0.000000 |
| opv_ultra_preserve | opu_old_preserve | 2 | 304.808 | 0.246646 | 0.071661 | 0.035831 |
| opv_ultra_preserve | opu_old_preserve | 3 | 414.254 | 0.246646 | 0.100977 | 0.055375 |
| opv_ultra_preserve | opu_old_preserve | 4 | 496.339 | 0.246646 | 0.000000 | 0.055375 |
| opv_ultra_preserve | opu_old_preserve | 5 | 547.231 | 0.246646 | 0.000000 | 0.078176 |

局部参数网格：

- 基线：`old_acc=0.802139`，`unknown_reject=0.183333`。
- 只要非零veto权重带来unknown提升，old_acc最低降到`0.796791`。
- 在本地小网格中没有找到`old_acc>=0.802139`且`unknown_reject>0.183333`的候选。

## 解释

OPV-CI证明了一个重要边界：在当前高known feature上，unknown与旧类弱证据区域仍高度重叠。证据层reject veto可以改善unknown，但即使极轻量也会误伤一部分旧类，因此不能满足“旧类不下降”。这说明下一步需要在feature/prototype层增强类间分离，而不是继续只调协同融合阈值。

## N607执行

同步文件：

| local | remote |
|---|---|
| `E:\type10-7\code\scripts\phase2_old_protected_pcet_veto_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_old_protected_pcet_veto_ci_eval.py` |
| `E:\type10-7\code\tests\test_phase2_old_protected_pcet_veto_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_old_protected_pcet_veto_ci_eval.py` |

远端输入：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_orbit_c3r_guard_20260704/input/features.npz
```

该feature的SHA256为`db559d78db305894307851750ef7d698db387f0984ff13c980fea99db85b8532`，与本地`remote_artifacts\phase2_adv3b02_features\features.npz`一致。原计划路径`runs/phase2_adv3b02_features/features.npz`在N607不存在。

远端验证：

| 项目 | 结果 |
|---|---|
| N607预检 | PASS，直连`N607`，项目根可见，8张RTX3090均约10MiB显存占用。 |
| 远端Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python3.10.19。 |
| 同步后SHA | 脚本`1dd73592e1be977cda193871b6492e98333de5ba218dd159885b6e3e18580fdd`；测试`353bc037e2c4456ac60a94425665d525424e12d5ebfe08b6b1d2c2b35e3bcf36`。 |
| 远端编译/测试 | `py_compile`PASS；`python code/tests/test_phase2_old_protected_pcet_veto_ci_eval.py`，4 tests OK。 |
| SSH清理 | 每次SSH/SCP后检查，本地无`ssh.exe`残留，无到`172.31.111.215:22`的ESTABLISHED连接。 |

远端命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_old_protected_pcet_veto_ci_eval.py \
  --feature_npz runs/phase2_adv3b02_orbit_c3r_guard_20260704/input/features.npz \
  --output_dir runs/phase2_adv3b02_opv_ci_20260704 \
  --profiles all \
  --policies opu_old_preserve,opu_old_guarded \
  --max_event_latency_ms 2.0 \
  --force
```

GPU：选择GPU0；运行前后GPU0显存均为10MiB，未出现额外显存占用。

远端输出：

| 文件 | SHA256 |
|---|---|
| `runs/phase2_adv3b02_opv_ci_20260704/opv_ci_summary.csv` | `8ee1699b3e94b76a29e4065248c0411a0903385028ed1a1b34ea1fbc0edc5aa5` |
| `runs/phase2_adv3b02_opv_ci_20260704/opv_ci_summary.json` | `60d0cfcadb998400f7129bb1a1c28a75ec60faf11b6d9af07c0599026913ac64` |
| `logs/phase2_adv3b02_opv_ci_20260704/opv_ci.log` | `c8c8eff3f281fd91cd33155981a43760278dab5ae91966c34c29faf006d37dd4` |

本地拉回：`remote_artifacts\phase2_adv3b02_opv_ci_20260704\`。

远端全量结果覆盖`collab_count=1..5`。`policy=opu_old_preserve`、`collab_count=4`同row结果：

| profile | old_acc | seen_new_acc | unknown_reject | unknown_FAR | known_coverage | bytes_per_event | latency_ms_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| opv_ultra_preserve | 0.796791 | 0.766667 | 0.216667 | 0.700000 | 0.939271 | 496.339 | 0.246646 |
| opv_preserve | 0.770053 | 0.716667 | 0.366667 | 0.550000 | 0.898785 | 496.339 | 0.246646 |
| opv_balanced | 0.737968 | 0.666667 | 0.650000 | 0.350000 | 0.850202 | 496.339 | 0.246646 |
| opv_unknown_push | 0.614973 | 0.533333 | 0.766667 | 0.200000 | 0.704453 | 496.339 | 0.246646 |

## 监督与review结论

子agent监督结论：本轮已完成本地验证、Git镜像提交、N607同步、`CVS-RFFI`远端测试、低显存GPU选择、结果拉回和报告更新；核心算法目标仍为NEGATIVE，因为unknown提升不能在旧类不下降约束下成立。

子agent review指出的协议风险已处理：

| 风险 | 处理 |
|---|---|
| summary可能按unknown评估指标排序，被误读成profile selection | 已改为`pre_registered_profile_policy_collab_count`顺序；`joint_score_scope`标为`evaluation_ranking_only_uses_target_unknown_metrics`。 |
| 本地报告混合`local_highknown`和`local_ultra`产物 | 已用`local_all_clean`一次性重跑`--profiles all`并改写报告口径。 |
| 资源字段不足 | 已补`bytes_per_event`、`latency_ms_p95`、`request_more_rate`、`defer_rate`。 |
| TX集合与unknown eval-only可追溯性不足 | 已补`Y_old/Y_new/Y_unknown`、`target_unknown_training_count=0`、`target_unknown_selection_count=0`。 |

## 下一步算法建议

当前OPV-CI负结果说明：未知类拒识与旧类弱证据区域重叠，证据层veto无法在不伤旧类的情况下继续压FAR。下一版应改为`OPC-ARC-CI`：Old-Protected Candidate-set Adaptive Receiver Consensus。

核心设计：

1. 强旧类保护先行：旧类候选必须通过support density、conformal p-value、margin和低risk四个条件；通过后直接accept或对unknown risk硬封顶。
2. candidate-set空集拒识：只有旧类/seen-new候选集为空或全为弱证据时，才触发unknown reject；若接收机分歧大则`request_more`或`defer`。
3. 分层协同：按receiver-class reliability逐步请求第`1..|R_t|`个接收机，每个接收机只上传top-2标签、score、margin、support_density、conformal p-value和risk分量。
4. 阈值来源：只用source-ID、target-old support和seen-new support留出或support-derived virtual negative；`Y_unknown`继续eval-only。
5. 排序口径：先满足`old_acc>=OPU baseline`和`min_old`不下降，再比较unknown FAR/拒识；不能用unknown query选择profile。

可引入的轻量机制：Energy/margin双门、Mahalanobis/EVT类条件尾部、SnaTCHer式弱扰动一致性、conformal选择性拒识。TENT/SHOT/ODIN若依赖test unknown或闭集熵最小化，不适合当前星上实时协议。
