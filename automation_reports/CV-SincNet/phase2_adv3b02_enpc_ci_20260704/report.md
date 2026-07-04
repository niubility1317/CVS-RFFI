# phase2_adv3b02_enpc_ci_20260704

## 基本信息

- 时间：2026-07-04
- 操作员/agent：Codex
- 目标：在ADV3B02_CORE90_SOFT_E200特征与qknn8部署证据上，实现ENPC-CI（Episode-Negative Prototype Conservative Collaborative Inference）候选算法，测试是否能在保留旧类识别的同时提升unknown拒识。
- 场景：Stage2-C；`R_t`与`R_s`不相交；`Y_old/Y_new/Y_unknown`互斥；unknown query仅用于最终评估，不参与阈值拟合、profile选择或校准。
- 输入特征：`E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz`。
- 对照：PCET-CI、SOVC-CI。

## 方法依据

PCET/SOVC负结果显示，后验强拒识可以提高unknown reject，但会同步拒掉old/seen-new。ENPC-CI改为先构造support-only的known保护，再做episode negative pressure拒识。

相关方法依据：

- KNN-OOD说明近邻距离可作为OOD信号，适合非参数、低资源部署。
- ProtoNet说明少样本类可由support prototype表示，适合seen-new注册。
- OpenMax/EVT适合尾部风险建模，但K=8下不应作为唯一裁判。
- Energy-based OOD可转化为distance-energy风险分量，但当前qknn特征栈不应强依赖logit energy。
- TANE类few-shot open-set方法启发“任务内negative prototype/negative shell”，但CVS中必须由source/support几何构造，不能使用`Y_unknown`query。

## 算法设计

每个receiver上传固定小证据包：top-1 label、qknn score、margin、support/verifier pvalue、receiver-class reliability、verifier score、negative risk、bytes和latency。

ENPC派生两个标量：

```text
support_confidence =
  0.30 * known_score
+ 0.25 * verifier_pvalue
+ 0.20 * receiver_class_reliability
+ 0.15 * normalized_margin
+ 0.10 * verifier_score
```

```text
episode_negative_pressure =
  0.28 * base_unknown_risk
+ 0.18 * qknn_ambiguity
+ 0.16 * verifier_ambiguity
+ 0.14 * verifier_unknown_risk
+ 0.10 * (1 - verifier_pvalue)
+ 0.06 * (1 - receiver_class_reliability)
+ 0.04 * class_shell_risk
+ 0.03 * verifier_changed
+ 0.01 * class_negative_risk
```

协同融合：

```text
label_score_c = sum_m support_confidence_m * (1 - 0.45 * pressure_m)

accept if:
  confidence >= tau_accept
  margin >= tau_margin
  and (mean_pressure <= tau_accept_pressure or mean_support_confidence >= tau_support)

reject unknown if:
  mean_pressure >= tau_reject
  or high_pressure_fraction >= tau_high_fraction
  or disagreement high under moderate pressure

otherwise defer
```

## 本地变更

| 文件 | 作用 |
|---|---|
| `E:\type10-7\code\scripts\phase2_orbit_enpc_ci_eval.py` | 新增ENPC-CI候选算法、独立协同决策器、summary/evidence/json输出。 |
| `E:\type10-7\code\tests\test_phase2_orbit_enpc_ci_eval.py` | 覆盖support-only增强、协同accept/reject、per-class审计字段。 |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_enpc_ci_20260704\report.md` | 本报告。 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_enpc_ci_20260704\` | 本地同步前快照。 |

根目录不是Git仓库，Git-backed镜像为`E:\type10-7\github_publish\CVS-RFFI-repo`。

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_orbit_enpc_ci_eval.py code\tests\test_phase2_orbit_enpc_ci_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_orbit_enpc_ci_eval.py -q` | PASS，3 passed；`.pytest_cache`warning不影响结论。 |
| `phase2_orbit_enpc_ci_eval.py --collab_counts all --collab_group_policy available_up_to_k --k_shot 8 --qknn_k 8` | PASS |
| `phase2_orbit_c3r_failure_audit.py --input_json ...\enpc_ci.json` | PASS，`floor_failure_count=8`，`no_event_count=0`，无schema错误。 |

## 本地结果摘要

目标门槛：`old_acc>=0.99`，`min_old>=0.95`，`seen_new_acc>=0.97`，`min_seen>=0.93`，`unknown_reject>=0.99`。

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | bytes/event | latency_ms | resource_pass | target_pass | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| enpc_known_anchor | 5 | 0.8307 | 0.5500 | 0.5500 | 0.3500 | 0.0000 | 1.0000 | 0.0000 | 414.2 | 0.159 | true | false | 旧类达到OLD80，但完全不拒unknown。 |
| enpc_balanced | 4 | 0.8254 | 0.5500 | 0.5000 | 0.3500 | 0.0500 | 0.9000 | 0.0201 | 376.5 | 0.159 | true | false | 旧类保持，unknown很弱。 |
| enpc_old80_unknown_probe | 5 | 0.8201 | 0.5500 | 0.4667 | 0.3250 | 0.4500 | 0.5500 | 0.0080 | 414.2 | 0.159 | true | false | 当前最有价值Pareto点：保住OLD80并提升unknown，但仍远低于目标。 |
| enpc_unknown_strict | 5 | 0.6085 | 0.2778 | 0.4167 | 0.2500 | 0.4500 | 0.3833 | 0.2048 | 414.2 | 0.159 | true | false | unknown略高但old/seen-new下降，不可作为主线。 |

## 监督审查

合理性：

- ENPC没有使用unknown query校准阈值，符合Stage2-C边界。
- `collab_count=1..5`均已输出，资源字段包含`bytes_per_event`和`latency_ms`。
- 相比SOVC，ENPC首次在当前特征上达到`old_acc>0.80`，说明“先证明known，再拒unknown”的方向比“后验强拒识”更合理。

问题：

- unknown拒识仍远低于99%，最佳OLD80保留点只有`unknown_reject=0.45`。
- `min_old=0.55`、`min_seen=0.325`远低于最终floor目标。
- 当前仍是feature/evidence级时延，不是完整模型前向+qknn搜索+融合链路端到端实测。
- 若要达到最终目标，需要训练期source-side open-set verifier或轻量Siamese/Proto-EVT head，而不仅是部署后证据融合。

## N607计划

- 远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`
- 远端环境：`CVS-RFFI`
- 远端输出：`runs/phase2_adv3b02_enpc_ci_20260704/`
- 同步文件：
  - `code/scripts/phase2_orbit_enpc_ci_eval.py`
  - `code/tests/test_phase2_orbit_enpc_ci_eval.py`
- GPU策略：选择显存占用最低GPU；用户允许显卡有其他进程但显存占用低时继续开启实验。

## N607执行结果

远端preflight通过：直连`N607`可用，项目根目录存在，8张RTX 3090均约10MiB显存占用。实际使用GPU0，执行后`nvidia-smi --query-gpu=index,memory.used`显示GPU0到GPU7均为10MiB。

| 远端步骤 | 结果 |
|---|---|
| `py_compile code/scripts/phase2_orbit_enpc_ci_eval.py code/tests/test_phase2_orbit_enpc_ci_eval.py` | PASS |
| `PYTHONPATH=code:code/scripts CUDA_VISIBLE_DEVICES=0 ... test_phase2_orbit_enpc_ci_eval.py` | PASS，3 tests OK |
| `CUDA_VISIBLE_DEVICES=0 ... phase2_orbit_enpc_ci_eval.py --collab_counts all --collab_group_policy available_up_to_k` | PASS |
| `CUDA_VISIBLE_DEVICES=0 ... phase2_orbit_c3r_failure_audit.py --input_json .../enpc_ci.json` | PASS，`floor_failure_count=8`，`no_event_count=0`，无schema错误 |

远端结果归档：

| 远端文件 | 本地归档 |
|---|---|
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_enpc_ci_20260704/enpc_ci.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_enpc_ci_20260704\remote\enpc_ci.json` |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_enpc_ci_20260704/enpc_ci_summary.csv` | `E:\type10-7\local_artifacts\phase2_adv3b02_enpc_ci_20260704\remote\enpc_ci_summary.csv` |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_enpc_ci_20260704/enpc_ci_failure_audit.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_enpc_ci_20260704\remote\enpc_ci_failure_audit.json` |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_enpc_ci_20260704/enpc_ci_class_floor.csv` | `E:\type10-7\local_artifacts\phase2_adv3b02_enpc_ci_20260704\remote\enpc_ci_class_floor.csv` |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_enpc_ci_20260704/enpc_ci_confusion.csv` | `E:\type10-7\local_artifacts\phase2_adv3b02_enpc_ci_20260704\remote\enpc_ci_confusion.csv` |

SSH/SCP后检查本地`ssh.exe`进程和到`172.31.111.215:22`的ESTABLISHED连接，未发现残留连接。

## 当前结论

ENPC-CI不是最终成功，但它比SOVC/PCET提供了更好的下一步主线：当前最佳same-row结果为`old_acc=0.8201`、`min_old=0.55`、`seen_new_acc=0.4667`、`min_seen=0.325`、`unknown_reject=0.45`。这说明OLD80阶段可以保住，但unknown边界仍需要训练式open-set verifier增强。
