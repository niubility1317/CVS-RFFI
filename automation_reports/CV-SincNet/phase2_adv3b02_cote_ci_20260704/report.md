# phase2_adv3b02_cote_ci_20260704

## 基本信息

- 实验ID：phase2_adv3b02_cote_ci_20260704
- 时间：2026-07-04
- 操作：Codex
- 目标：在ADV3B02_CORE90_SOFT_E200/qknn8基础上验证COTE-CI，即candidate-over-topM evidence卫星群协同推理。该方法复用PCET类级EVT、Mahalanobis、class conformal和receiver reliability证据，不训练、不使用target_unknown做阈值或参数选择。
- 协议文件：已读取`AGENTS.md`和`项目.md`。

## 资源约束文件状态

用户指定需要查看`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`。本轮在`E:\type10-7`内按`*卫星协同*`、`*资源约束*`、`*RFFI*系统资源*`递归查找，未找到该文件。当前资源约束只能按已有可观测字段执行：`bytes_per_event`、`latency_ms`、receiver数、feature/prototype状态字节、GPU显存快照。若后续提供该文件，应把报告中的资源门槛与该文件逐项对齐。

## 方法

COTE-CI流程：

```text
ADV3B02 frozen feature -> qknn8/PCET evidence
each receiver uploads top-M class evidence + risk + reliability
candidate-over-topM aggregation
known old/seen-new shield if support quality is sufficient
unknown rejection only if no known shield and cross-receiver risk is high
```

关键边界：

| 项目 | 设置 |
|---|---|
| backbone | ADV3B02_CORE90_SOFT_E200，冻结 |
| in-orbit method | qknn8 |
| adapter | 无 |
| unknown query | evaluation-only |
| target_unknown_training_count | 0 |
| 协同对象 | target receiver domain `R_t` |
| 协同数量 | `collab_counts=all`，覆盖1..5 |
| 每接收机证据包 | 128B |
| 资源门槛 | `max_event_bytes=1152`，`max_event_latency_ms=20` |

## 本地变更

| 文件 | 目的 |
|---|---|
| `code/scripts/phase2_cote_ci_eval.py` | 新增COTE-CI证据融合评估脚本 |
| `code/tests/test_phase2_cote_ci_eval.py` | 新增known shield和unknown confirm单测 |
| `automation_reports/CV-SincNet/phase2_adv3b02_cote_ci_20260704/report.md` | 本报告 |

## 本地验证

```text
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_cote_ci_eval.py code\tests\test_phase2_cote_ci_eval.py
```

结果：通过。

```text
conda run -n ssr-gpu python code\tests\test_phase2_cote_ci_eval.py
```

结果：2项测试通过。

## 本地数据与协议证据

- feature NPZ：`local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz`
- sha256：`7f5c2956ce78f0a2b44c6f41fee453613eede5cf916be0ff6899365fac7a3297`
- target receivers：20-1，3-19，7-14，7-7，8-8
- 角色计数：source=2400，proxy_unknown=1600，target_old=2400，target_new=800，target_unknown=800
- 星地信道：沿用feature中的`channel_views`和`sat_scenarios`，包括LEO/satellite stress特征字段。

## 本地结果

### k_shot=5，same_max_budget

```text
conda run -n ssr-gpu python code\scripts\phase2_cote_ci_eval.py --feature_npz local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz --output_json local_artifacts\phase2_adv3b02_cote_ci_20260704\local\cote_summary.json --output_summary_csv local_artifacts\phase2_adv3b02_cote_ci_20260704\local\cote_summary.csv --output_evidence_csv local_artifacts\phase2_adv3b02_cote_ci_20260704\local\cote_evidence.csv --profiles all --collab_counts all --collab_group_policy same_max_budget --k_shot 5 --query_per_class 20 --qknn_k 8 --top_m 3 --max_event_bytes 1152 --max_event_latency_ms 20
```

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes_per_event | latency_ms | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cote_known_anchor | 4 | 0.9674 | 0.0000 | 0.2432 | 0.2000 | 0.2647 | 0.7353 | 512.0 | 0.9811 | false |
| cote_known_anchor | 5 | 1.0000 | 0.0000 | 0.2500 | 0.0000 | 0.2500 | 0.7500 | 640.0 | 0.9811 | false |

说明：`same_max_budget`在高协同数下筛掉缺少全接收机对齐的事件，部分类无样本，`min_old=0`是保守惩罚，不能作为达标证据。

### k_shot=5，available_up_to_k

```text
conda run -n ssr-gpu python code\scripts\phase2_cote_ci_eval.py --feature_npz local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz --output_json local_artifacts\phase2_adv3b02_cote_ci_20260704\local_available\cote_summary.json --output_summary_csv local_artifacts\phase2_adv3b02_cote_ci_20260704\local_available\cote_summary.csv --output_evidence_csv local_artifacts\phase2_adv3b02_cote_ci_20260704\local_available\cote_evidence.csv --profiles all --collab_counts all --collab_group_policy available_up_to_k --k_shot 5 --query_per_class 20 --qknn_k 8 --top_m 3 --max_event_bytes 1152 --max_event_latency_ms 20
```

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes_per_event | latency_ms | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cote_known_anchor | 5 | 0.8842 | 0.7000 | 0.2000 | 0.1750 | 0.1833 | 0.8167 | 412.9 | 1.0609 | false |

### k_shot=8，available_up_to_k

```text
conda run -n ssr-gpu python code\scripts\phase2_cote_ci_eval.py --feature_npz local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz --output_json local_artifacts\phase2_adv3b02_cote_ci_20260704\local_k8\cote_summary.json --output_summary_csv local_artifacts\phase2_adv3b02_cote_ci_20260704\local_k8\cote_summary.csv --output_evidence_csv local_artifacts\phase2_adv3b02_cote_ci_20260704\local_k8\cote_evidence.csv --profiles all --collab_counts all --collab_group_policy available_up_to_k --k_shot 8 --query_per_class 20 --qknn_k 8 --top_m 3 --max_event_bytes 1152 --max_event_latency_ms 20
```

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes_per_event | latency_ms | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cote_known_anchor | 5 | 0.8824 | 0.8000 | 0.2833 | 0.2000 | 0.1833 | 0.8167 | 416.9 | 1.0808 | false |

## 本地结论

COTE-CI改善了旧类总体识别，尤其在`same_max_budget`高协同时old_acc接近或达到1.0，但完整类覆盖口径下仍只有old_acc≈0.88、min_old≈0.80、seen_new≈0.28、unknown_reject≈0.18。该路线未达成目标。强unknown配置会把旧类和新类几乎全部拒识，因此只能作为diagnostic，不是可部署结果。

## N607计划

- 先运行`tools\n607_ssh_preflight.ps1`。
- 同步：
  - `code/scripts/phase2_cote_ci_eval.py` -> `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_cote_ci_eval.py`
  - `code/tests/test_phase2_cote_ci_eval.py` -> `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_cote_ci_eval.py`
- 远程环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 远程输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_cote_ci_20260704`
- GPU：启动前按`nvidia-smi`选择显存占用最低GPU；若均相同，选择GPU0。
- SSH/SCP后检查本地`ssh.exe`、`172.31.111.215:22`和`172.31.105.18:22`，确认无残留连接。

## N607执行结果

### 远程验证

- preflight：通过。服务器时间为2026-07-04 12:52:49 CST，项目根目录可见。
- 启动前GPU快照：GPU0-7均为NVIDIA GeForce RTX 3090，显存10/24576MiB，utilization 0%。
- 选择GPU：GPU0。理由：所有GPU显存占用相同且最低。
- 远程Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 远程编译和单测：

```text
cd /home/szu2070436088/2510044040/CV-SincNet && /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_cote_ci_eval.py code/tests/test_phase2_cote_ci_eval.py && /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_cote_ci_eval.py
```

结果：2项测试通过。

同步映射：

| 本地 | N607 |
|---|---|
| `code/scripts/phase2_cote_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_cote_ci_eval.py` |
| `code/tests/test_phase2_cote_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_cote_ci_eval.py` |

### 远程命令

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_cote_ci_eval.py --feature_npz /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz --output_json /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_cote_ci_20260704/k8_available/cote_summary.json --output_summary_csv /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_cote_ci_20260704/k8_available/cote_summary.csv --output_evidence_csv /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_cote_ci_20260704/k8_available/cote_evidence.csv --profiles all --collab_counts all --collab_group_policy available_up_to_k --k_shot 8 --query_per_class 20 --qknn_k 8 --top_m 3 --max_event_bytes 1152 --max_event_latency_ms 20
```

日志：

```text
/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_cote_ci_20260704/k8_available.log
```

输出目录：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_cote_ci_20260704/k8_available
```

### 远程结果

远程结果已拉回：

```text
local_artifacts\phase2_adv3b02_cote_ci_20260704\remote\k8_available
local_artifacts\phase2_adv3b02_cote_ci_20260704\remote\logs
```

最佳同row：

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | bytes_per_event | latency_ms | target_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cote_known_anchor | 5 | 0.8824 | 0.8000 | 0.2833 | 0.2000 | 0.1833 | 0.8167 | 0.0040 | 0.0167 | 416.94 | 1.8995 | false |

远程输出hash：

| 文件 | sha256 |
|---|---|
| `cote_summary.json` | `4b514086f82816909c4bc6055f2bb8a68a14a1ef4905aa48707431cd739a82ef` |
| `cote_summary.csv` | `1b545b41ae827b989937faf8bf897c560365f3738aaf81e68a622cab1888435c` |
| `cote_evidence.csv` | `2ca2fd427297381f5a92032124cda9dde0e345d990458ce33b5ebe982a8a249f` |

收尾GPU快照：GPU0-7均为10/24576MiB，utilization 0%，没有残留显存占用。

SSH/SCP断连检查：每次SSH/SCP后检查本地`ssh.exe`、`172.31.111.215:22`和`172.31.105.18:22`，均无残留连接。

## 总结

COTE-CI没有达成目标。它说明top-M候选协同可以提升旧类总体识别，但仍无法同时保持每类旧类、新类和未知拒识。与OPR-CI、ENPC、SLEV、proxy_unknown诊断一致，当前瓶颈已经不是单个部署侧融合规则，而是ADV3B02/qknn8特征空间在LEO星地信道下对unknown与known的可分性不足。下一步应转向地面训练阶段的open-set representation约束：source/proxy outlier exposure、class-wise Gaussian/EVT margin、receiver-invariant identity subspace和旧类蒸馏联合训练。
