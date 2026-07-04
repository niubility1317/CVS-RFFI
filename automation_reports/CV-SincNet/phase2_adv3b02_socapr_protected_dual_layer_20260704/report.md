# phase2_adv3b02_socapr_protected_dual_layer_20260704

## 基本信息

| 字段 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_socapr_protected_dual_layer_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 底座模型 | ADV3B02_CORE90_SOFT_E200 |
| 目标 | 在qknn8协同推理上评估OLD80保护的SO-CAPR双层拒识：强known候选回退known route，只有弱known候选且多安全证据一致时才veto |
| 特征 | `remote_artifacts\phase2_adv3b02_features\features.npz`，SHA256=`DB559D78DB305894307851750EF7D698DB387F0984FF13C980FEA99DB85B8532` |
| 状态 | NON_DEPLOYMENT_DIAGNOSTIC |
| 资源约束说明 | 未在`E:\type10-7`中找到`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`；本报告临时记录bytes/event、latency、prototype/storage代理字段 |

## 协议边界

该诊断不使用unknown query拟合阈值。合法校验字段保持为`threshold_selection_label_scope=support_known_only`，并额外记录`protected_threshold_selection_detail=source_old_and_allowed_support_only_unknown_query_eval`。当前多接收机结果仍按`receiver_domain_ranked`组织，属于target receiver domain/deployment proxy诊断，不声明严格同事件卫星群协同。

## 算法

Protected Dual-Layer SO-CAPR采用两层决策：

```text
known_route: qknn8 -> old/seen-new候选与known_score/known_margin
safety_route: support envelope + virtual unknown + class negative + shell/EVT/Mahalanobis风险

if strong_known_guard:
  unknown_risk = min(known_unknown_risk, protected_risk_cap)
elif weak_known_candidate and safety_signal_count >= min_safety_signals:
  unknown_risk = max(known_unknown_risk, safety_unknown_risk)
else:
  unknown_risk = known_unknown_risk
```

默认本地诊断参数：

| 参数 | 值 |
|---|---:|
| `qknn_k` | 8 |
| `k_shot` | 8 |
| `thresholds` | `0.8,0.9,0.95,0.99` |
| `old_min_score` | 0.45 |
| `old_min_margin` | 0.05 |
| `seen_new_min_score` | 0.45 |
| `seen_new_min_margin` | 0.05 |
| `min_support_density` | 0.0 |
| `min_conformal_pvalue` | 0.0 |
| `safety_signal_threshold` | 0.95 |
| `min_safety_signals` | 5 |
| `veto_max_score` | 0.60 |
| `veto_max_margin` | 0.15 |
| `protected_risk_cap` | 0.02 |
| `evidence_bytes_per_receiver_event` | 168 |

## 本地变更

| 文件 | 作用 |
|---|---|
| `E:\type10-7\code\scripts\phase2_socapr_protected_dual_layer_eval.py` | 新增OLD80保护的双层SO-CAPR评估脚本 |
| `E:\type10-7\code\tests\test_phase2_socapr_protected_dual_layer_eval.py` | 新增强known保护、弱候选veto、fallback和summary delta单测 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_socapr_protected_dual_layer_20260704\phase2_socapr_protected_dual_layer_eval.py` | 非Git根代码快照 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_socapr_protected_dual_layer_20260704\test_phase2_socapr_protected_dual_layer_eval.py` | 非Git根测试快照 |

根目录`E:\type10-7`不是Git仓库；后续同步到Git镜像`E:\type10-7\github_publish\CVS-RFFI-repo`提交。

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_socapr_protected_dual_layer_eval.py -q` | PASS，4 passed；`.pytest_cache`权限警告不影响测试 |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_socapr_protected_dual_layer_eval.py code\tests\test_phase2_socapr_protected_dual_layer_eval.py` | PASS |

本地运行命令：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_socapr_protected_dual_layer_eval.py --feature_npz remote_artifacts\phase2_adv3b02_features\features.npz --output_dir local_artifacts\phase2_adv3b02_socapr_protected_dual_layer_20260704_final --force --thresholds 0.8,0.9,0.95,0.99 --old_min_score 0.45 --old_min_margin 0.05 --seen_new_min_score 0.45 --seen_new_min_margin 0.05 --min_support_density 0.0 --min_conformal_pvalue 0.0 --safety_signal_threshold 0.95 --min_safety_signals 5 --veto_max_score 0.60 --veto_max_margin 0.15 --protected_risk_cap 0.02
```

本地输出：

| 文件 | 内容 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_protected_dual_layer_20260704_final\protected_dual_layer_summary.csv` | 1..5协同摘要 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_protected_dual_layer_20260704_final\protected_dual_layer_summary.json` | 完整结果JSON |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_protected_dual_layer_20260704_final\protected_dual_layer_evidence.csv` | 合并evidence |

## 本地结果

严格基线约束`old_acc_delta_vs_known_route>=0`且`known_coverage_delta_vs_known_route>=0`下，20个工作点中`baseline_constraint_pass=0`。主要同row结果如下：

| threshold | k | old_acc | baseline_old | delta_old | seen_new_acc | unknown_reject | unknown_FAR | known_coverage | baseline_cov | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.95 | 3 | 0.7594 | 0.7807 | -0.0214 | 0.7667 | 0.0000 | 0.7000 | 0.9109 | 0.9798 | FAIL_OLD_COVERAGE |
| 0.95 | 2 | 0.7807 | 0.6524 | +0.1283 | 0.8000 | 0.0000 | 0.8333 | 0.9474 | 1.0000 | FAIL_FAR_COVERAGE |
| 0.99 | 1 | 0.8075 | 0.6578 | +0.1497 | 0.8667 | 0.0000 | 0.9667 | 0.9838 | 1.0000 | FAIL_FAR |
| 0.95 | 5 | 0.8235 | 0.8663 | -0.0428 | 0.8000 | 0.0000 | 0.7167 | 0.9433 | 1.0000 | FAIL_OLD_FAR |
| 0.80 | 5 | 0.7968 | 0.8663 | -0.0695 | 0.7667 | 0.0500 | 0.5333 | 0.9028 | 1.0000 | FAIL_OLD_FAR |

## 解释

保护层修复了dual-veto中“非veto样本也被safety风险抬高”的问题，但当前ADV3B02特征几何下，unknown与known候选仍高度重叠。只要保持旧类不下降，unknown_FAR仍高；只要压低FAR，old_acc和known coverage会下降。该结果说明下一阶段不能只依赖后处理拒识层，需要回到特征几何、事件级同观测协同或更强的query-free背景/负锚建模。

## N607计划

同步新脚本、测试和报告到N607，使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`在低显存占用GPU上复测。远端输出到：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_socapr_protected_dual_layer_20260704
```

## N607验证

N607预检：

| 字段 | 内容 |
|---|---|
| 预检时间 | 2026-07-04 08:58:53 CST |
| SSH目标 | direct `N607`，配置`E:\type10-7\tools\n607_ssh_config` |
| 项目根目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python 3.10.19 |
| GPU选择 | GPU0；运行前后8张RTX 3090均为10MiB显存占用 |
| SSH清理 | 远端验证和SCP拉回后，本地`ssh.exe`与到`172.31.111.215:22`/`172.31.105.18:22`的连接均为空 |

同步目标：

| 本地文件 | N607目标 |
|---|---|
| `E:\type10-7\code\scripts\phase2_socapr_protected_dual_layer_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_socapr_protected_dual_layer_eval.py` |
| `E:\type10-7\code\tests\test_phase2_socapr_protected_dual_layer_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_socapr_protected_dual_layer_eval.py` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_socapr_protected_dual_layer_20260704\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_socapr_protected_dual_layer_20260704/report.md` |

远端验证命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && set -e
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -m py_compile code/scripts/phase2_socapr_protected_dual_layer_eval.py code/tests/test_phase2_socapr_protected_dual_layer_eval.py
CUDA_VISIBLE_DEVICES=0 $PY code/tests/test_phase2_socapr_protected_dual_layer_eval.py
CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_socapr_protected_dual_layer_eval.py --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz --output_dir runs/phase2_adv3b02_socapr_protected_dual_layer_20260704 --force --thresholds 0.8,0.9,0.95,0.99 --old_min_score 0.45 --old_min_margin 0.05 --seen_new_min_score 0.45 --seen_new_min_margin 0.05 --min_support_density 0.0 --min_conformal_pvalue 0.0 --safety_signal_threshold 0.95 --min_safety_signals 5 --veto_max_score 0.60 --veto_max_margin 0.15 --protected_risk_cap 0.02
```

远端测试结果：`py_compile`通过，`unittest`通过，4 tests OK。

远端关键结果：

| threshold | k | old_acc | baseline_old | delta_old | seen_new_acc | unknown_reject | unknown_FAR | known_coverage | baseline_cov | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 1 | 0.8075 | 0.6578 | +0.1497 | 0.8667 | 0.0500 | 0.9167 | 0.9838 | 1.0000 | FAIL_FAR |
| 0.80 | 5 | 0.7968 | 0.8663 | -0.0695 | 0.7667 | 0.0500 | 0.5333 | 0.9028 | 1.0000 | FAIL_OLD_FAR |
| 0.95 | 2 | 0.7807 | 0.6524 | +0.1283 | 0.8000 | 0.0000 | 0.8333 | 0.9474 | 1.0000 | FAIL_FAR_COVERAGE |
| 0.95 | 5 | 0.8235 | 0.8663 | -0.0428 | 0.8000 | 0.0000 | 0.7167 | 0.9433 | 1.0000 | FAIL_OLD_FAR |
| 0.99 | 3 | 0.7807 | 0.7807 | 0.0000 | 0.7667 | 0.0000 | 0.7833 | 0.9312 | 0.9798 | FAIL_FAR_COVERAGE |

远端输出已拉回：

| 文件 | 内容 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_protected_dual_layer_20260704_final\remote_protected_dual_layer_summary.csv` | N607 1..5协同摘要 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_protected_dual_layer_20260704_final\remote_protected_dual_layer_summary.json` | N607完整摘要JSON |

## 远端结论

N607复测确认该路线仍未解决核心矛盾。保护层减少了非veto样本的误拒，但在不牺牲old_acc和known coverage的条件下，unknown_FAR仍高；压低FAR的工作点仍伴随old或coverage下降。该诊断应作为“后处理拒识层不足”的证据，不能作为Stage2-C成功、部署成功或论文主结论。

下一步建议转向两个方向之一：

| 方向 | 理由 |
|---|---|
| 严格同事件协同 | 当前`receiver_domain_ranked`只是deployment proxy，缺少同一发射事件多星观测的互补几何，难以用一致性拒识 |
| 表征/负锚重建 | unknown与known在ADV3B02 qknn8空间内重叠，后处理阈值难以同时满足old保护和低FAR |
