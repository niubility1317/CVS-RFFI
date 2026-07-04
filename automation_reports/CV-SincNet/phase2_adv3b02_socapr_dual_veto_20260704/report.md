# phase2_adv3b02_socapr_dual_veto_20260704

## 基本信息

| 字段 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_socapr_dual_veto_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 在ADV3B02_CORE90_SOFT_E200+qknn8上实现SO-CAPR双路仲裁：known route输出old/seen-new候选，safety route只做折扣unknown veto |
| 特征 | `remote_artifacts\phase2_adv3b02_features\features.npz`，SHA256=`DB559D78DB305894307851750EF7D698DB387F0984FF13C980FEA99DB85B8532` |
| 状态 | NON_DEPLOYMENT_DIAGNOSTIC |

## 算法

双路仲裁不使用unknown query拟合阈值。known route使用qknn8保留old/seen-new候选；safety route使用support envelope、class conformal、virtual unknown、class negative、class shell生成安全风险。最终每个接收机事件的风险为：

```text
score_discount = max(0, 1 - known_score / score_anchor)
margin_discount = max(0, 1 - known_margin / margin_anchor)
discount = score_discount * margin_discount
dual_unknown_risk = max(known_route_unknown_risk, safety_weight * safety_route_unknown_risk * discount)
```

默认参数：

| 参数 | 值 |
|---|---:|
| `qknn_k` | 8 |
| `k_shot` | 8 |
| `score_anchor` | 0.70 |
| `margin_anchor` | 0.40 |
| `safety_weight` | 0.20 |
| `discount_mode` | `prod` |
| `thresholds` | `0.4,0.6,0.8` |
| `evidence_bytes_per_receiver_event` | 168 |

## 本地变更

| 文件 | 作用 |
|---|---|
| `E:\type10-7\code\scripts\phase2_socapr_dual_route_veto_eval.py` | 新增SO-CAPR双路veto评估脚本 |
| `E:\type10-7\code\tests\test_phase2_socapr_dual_route_veto_eval.py` | 新增risk折扣和资源合并单测 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_socapr_dual_veto_20260704\phase2_socapr_dual_route_veto_eval.py` | 非Git根代码快照 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_socapr_dual_veto_20260704\test_phase2_socapr_dual_route_veto_eval.py` | 非Git根测试快照 |

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_socapr_dual_route_veto_eval.py code\tests\test_phase2_socapr_dual_route_veto_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_socapr_dual_route_veto_eval.py -q` | PASS，3 passed；`.pytest_cache`权限警告不影响测试 |

本地运行命令：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_socapr_dual_route_veto_eval.py --feature_npz remote_artifacts\phase2_adv3b02_features\features.npz --output_dir local_artifacts\phase2_adv3b02_socapr_dual_veto_20260704 --force --thresholds 0.4,0.6,0.8 --score_anchor 0.7 --margin_anchor 0.4 --safety_weight 0.2 --discount_mode prod
```

本地输出：

| 文件 | 内容 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_dual_veto_20260704\dual_route_veto_summary.csv` | 1..5协同摘要 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_dual_veto_20260704\dual_route_veto_summary.json` | 完整结果JSON |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_dual_veto_20260704\dual_route_veto_evidence.csv` | 合并evidence |

## 本地结果

| threshold | k | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_coverage | defer_rate | bytes/event | latency_ms_p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 1 | 0.8021 | 0.1500 | 0.8667 | 0.8250 | 0.0667 | 0.7667 | 0.9636 | 0.0619 | 168.0 | 0.1534 |
| 0.4 | 2 | 0.7433 | 0.2000 | 0.7667 | 0.7000 | 0.0333 | 0.4667 | 0.8866 | 0.1857 | 304.8 | 0.1534 |
| 0.4 | 3 | 0.7005 | 0.2000 | 0.7000 | 0.6500 | 0.0333 | 0.3333 | 0.8259 | 0.2573 | 414.3 | 0.1534 |
| 0.4 | 4 | 0.7273 | 0.3500 | 0.7333 | 0.6500 | 0.0667 | 0.3667 | 0.8421 | 0.2345 | 496.3 | 0.1534 |
| 0.4 | 5 | 0.7540 | 0.5500 | 0.7167 | 0.6500 | 0.0500 | 0.3000 | 0.8381 | 0.2541 | 547.2 | 0.1534 |
| 0.6 | 5 | 0.7968 | 0.6000 | 0.7500 | 0.6750 | 0.0500 | 0.5333 | 0.8907 | 0.1661 | 547.2 | 0.1534 |
| 0.8 | 5 | 0.8235 | 0.6000 | 0.8000 | 0.7250 | 0.0500 | 0.7000 | 0.9352 | 0.0977 | 547.2 | 0.1534 |

## 解释

双路veto比上一轮safety route保留了known性能：k=5最高old从0.0963提升到0.8235，seen-new从0提升到0.8000。但unknown拒识仍只有0.05，FAR仍高达0.70；当阈值压到0.4时FAR下降到0.30，但old/seen-new同步下降且defer升高。该结果仍远低于目标`old 99%/floor 95%`、`seen-new 97%/floor 93%`和`unknown reject 99%`。

直接结论：折扣veto只能缓解“安全门控打穿known”的问题，不能解决高置信unknown落入known候选的问题。下一步必须引入更强的事件级跨接收机一致性异常检测或按类源原型边界，而不是继续线性折扣安全风险。

## N607验证

N607预检：

| 字段 | 内容 |
|---|---|
| 预检时间 | 2026-07-04 08:45:46 CST |
| SSH目标 | direct `N607`，配置`E:\type10-7\tools\n607_ssh_config` |
| 项目根目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python 3.10.19 |
| GPU选择 | GPU0；运行前8张RTX 3090均为10MiB显存占用 |
| SSH清理 | 远端验证和SCP拉回后，本地`ssh.exe`与到`172.31.111.215:22`/`172.31.105.18:22`的连接均为空 |

同步目标：

| 本地文件 | N607目标 |
|---|---|
| `E:\type10-7\code\scripts\phase2_socapr_dual_route_veto_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_socapr_dual_route_veto_eval.py` |
| `E:\type10-7\code\tests\test_phase2_socapr_dual_route_veto_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_socapr_dual_route_veto_eval.py` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_socapr_dual_veto_20260704\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_socapr_dual_veto_20260704/report.md` |

远端验证命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && set -e
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -m py_compile code/scripts/phase2_socapr_qknn8_pareto_eval.py code/scripts/phase2_socapr_dual_route_veto_eval.py code/tests/test_phase2_socapr_dual_route_veto_eval.py
CUDA_VISIBLE_DEVICES=0 $PY code/tests/test_phase2_socapr_dual_route_veto_eval.py
CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_socapr_dual_route_veto_eval.py --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz --output_dir runs/phase2_adv3b02_socapr_dual_veto_20260704 --force --thresholds 0.4,0.6,0.8 --score_anchor 0.7 --margin_anchor 0.4 --safety_weight 0.2 --discount_mode prod
```

远端结果：

| threshold | k | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_coverage | defer_rate | bytes/event | latency_ms_p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 1 | 0.8021 | 0.1500 | 0.8667 | 0.8250 | 0.0667 | 0.7667 | 0.9636 | 0.0619 | 168.0 | 0.1621 |
| 0.4 | 2 | 0.7433 | 0.2000 | 0.7667 | 0.7000 | 0.0333 | 0.4667 | 0.8866 | 0.1857 | 304.8 | 0.1621 |
| 0.4 | 3 | 0.7005 | 0.2000 | 0.7000 | 0.6500 | 0.0333 | 0.3333 | 0.8259 | 0.2573 | 414.3 | 0.1621 |
| 0.4 | 4 | 0.7273 | 0.3500 | 0.7333 | 0.6500 | 0.0667 | 0.3667 | 0.8421 | 0.2345 | 496.3 | 0.1621 |
| 0.4 | 5 | 0.7540 | 0.5500 | 0.7167 | 0.6500 | 0.0500 | 0.3000 | 0.8381 | 0.2541 | 547.2 | 0.1621 |
| 0.6 | 5 | 0.7968 | 0.6000 | 0.7500 | 0.6750 | 0.0500 | 0.5333 | 0.8907 | 0.1661 | 547.2 | 0.1621 |
| 0.8 | 5 | 0.8235 | 0.6000 | 0.8000 | 0.7250 | 0.0500 | 0.7000 | 0.9352 | 0.0977 | 547.2 | 0.1621 |

远端输出已拉回：

| 文件 | 内容 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_dual_veto_20260704\remote_dual_route_veto_summary.csv` | N607 1..5协同摘要 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_dual_veto_20260704\remote_dual_route_veto_summary.json` | N607完整摘要JSON |

## 远端结论

N607复测与本地结论一致。dual-veto可保护known route，不会像safety route那样把old打穿，但仍无法解决高置信unknown被接收为known的问题：在old_acc最高的k=5、threshold=0.8行，unknown_FAR仍为0.70；在unknown_FAR最低的k=5、threshold=0.4行，old_acc降至0.7540、seen_new_acc降至0.7167。该路线不满足“优先解决未知拒识且旧类准确性不能下降”的下一阶段要求，只能作为负诊断基线。

下一步算法应采用旧类保护优先的多证据拒识：先用known route产生old/seen-new候选并检查旧类保护边界，再由class radius/EVT或KNN/Mahalanobis双证据给出unknown风险；证据不足时输出`defer`，不得为了降低FAR直接牺牲old accepted accuracy。
