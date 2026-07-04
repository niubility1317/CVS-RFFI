# phase2_adv3b02_opv_ci_20260704

## 基本信息

- 时间：2026-07-04
- 操作方：Codex
- 目标：在ADV3B02_CORE90_SOFT_E200高known特征和qknn8在轨少样本协议上，测试old-protected PCET veto协同拒识算法，尝试在旧类不下降的前提下改善未知类拒识。
- 场景：Stage2-C target receiver domain，`R_t={20-1,3-19,7-14,7-7,8-8}`，`R_s`不相交；`Y_old/Y_new/Y_unknown`沿feature metadata。
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
| `code/tests/test_phase2_old_protected_pcet_veto_ci_eval.py` | 覆盖强known风险封顶、弱known高tail风险提升、默认profile解析。 |

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_old_protected_pcet_veto_ci_eval.py code\tests\test_phase2_old_protected_pcet_veto_ci_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_old_protected_pcet_veto_ci_eval.py -q` | PASS，3 passed；`.pytest_cache`权限警告不影响测试 |

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

OPV-CI本地全profile结果中old最高行：

| profile | policy | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| opv_ultra_preserve | opu_old_preserve | 4 | 0.796791 | 0.300000 | 0.766667 | 0.700000 | 0.216667 | 0.700000 | 0.939271 |
| opv_preserve | opu_old_preserve | 4 | 0.770053 | 0.300000 | 0.716667 | 0.675000 | 0.366667 | 0.550000 | 0.898785 |
| opv_balanced | opu_old_preserve | 4 | 0.737968 | 0.300000 | 0.666667 | 0.625000 | 0.650000 | 0.350000 | 0.850202 |
| opv_unknown_push | opu_old_preserve | 4 | 0.614973 | 0.300000 | 0.533333 | 0.475000 | 0.766667 | 0.200000 | 0.704453 |

局部参数网格：

- 基线：`old_acc=0.802139`，`unknown_reject=0.183333`。
- 只要非零veto权重带来unknown提升，old_acc最低降到`0.796791`。
- 在本地小网格中没有找到`old_acc>=0.802139`且`unknown_reject>0.183333`的候选。

## 解释

OPV-CI证明了一个重要边界：在当前高known feature上，unknown与旧类弱证据区域仍高度重叠。证据层reject veto可以改善unknown，但即使极轻量也会误伤一部分旧类，因此不能满足“旧类不下降”。这说明下一步需要在feature/prototype层增强类间分离，而不是继续只调协同融合阈值。

## N607计划

同步文件：

| local | remote |
|---|---|
| `E:\type10-7\code\scripts\phase2_old_protected_pcet_veto_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_old_protected_pcet_veto_ci_eval.py` |
| `E:\type10-7\code\tests\test_phase2_old_protected_pcet_veto_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_old_protected_pcet_veto_ci_eval.py` |

远端输入优先查找：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_features/features.npz
```

若该路径不存在，则使用已同步/可见的等价feature路径，必须在完成段记录。

远端命令计划：

```bash
CUDA_VISIBLE_DEVICES=<low-vram-gpu> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_old_protected_pcet_veto_ci_eval.py \
  --feature_npz <feature-path> \
  --output_dir runs/phase2_adv3b02_opv_ci_20260704 \
  --profiles all \
  --policies opu_old_preserve,opu_old_guarded \
  --max_event_latency_ms 2.0
```

预期输出：`opv_ci_summary.csv`、`opv_ci_summary.json`、各profile证据CSV；协同数量覆盖`1..5`。
