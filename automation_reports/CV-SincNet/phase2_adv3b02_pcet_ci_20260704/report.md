# phase2_adv3b02_pcet_ci_20260704

## 基本信息

| 项目 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_pcet_ci_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 在`ADV3B02_CORE90_SOFT_E200`和在轨`qknn8`基础上，追加PCET-CI原型一致性+EVT/尾部风险协同拒识，验证是否比ORBIT-C3R更好地兼顾unknown拒识与旧类/新类保留 |
| 协议边界 | Stage2-C；`Y_unknown`只用于query评估，不参与support、阈值拟合或receiver选择监督 |
| 输入feature | `E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz` |
| 本地输出 | `E:\type10-7\local_artifacts\phase2_adv3b02_pcet_ci_20260704\default\` |
| 协同范围 | `collab_count=1..5`，`collab_group_policy=available_up_to_k` |
| 资源预算 | 128B/receiver/event，最大1152B/event，最大20ms/event |

## 算法假设

PCET-CI是ORBIT-C3R之后的轻量补充路线。它不重训主干，不共享原始IQ或full precision embedding，只在每接收机qknn8 evidence上增加两个支持集可得的风险：

| 组件 | 机制 | 协议边界 |
|---|---|---|
| prototype-consistency risk | 根据top1/top2原型分数间隔、conformal p-value、receiver-class reliability和support count估计“替换/扰动原型后是否不稳定” | 只使用old/seen-new support统计和当前query evidence，不用unknown query拟合 |
| EVT/tail risk | 汇总已有`evt_risk`、`mahalanobis_risk`、`class_shell_risk`和半径风险 | 阈值来自support/leave-one-out/proxy机制 |
| safe-known cap | 当p-value、reliability、margin、support count和tail risk同时满足安全条件时，对PCET unknown risk上限截断 | 不读取真实query role，仅使用预测侧证据 |

默认PCET参数：

```text
pcet_base_weight=0.70
pcet_proto_weight=0.15
pcet_tail_weight=0.15
pcet_safe_known_risk_cap=0.45
seed=4070404
```

## 本地改动

| 文件 | 目的 |
|---|---|
| `E:\type10-7\code\scripts\phase2_orbit_pcet_ci_eval.py` | 新增PCET-CI包装器：构建qknn8 evidence，追加原型一致性/尾部风险，按profile输出1..N协同指标和资源字段 |
| `E:\type10-7\code\tests\test_phase2_orbit_pcet_ci_eval.py` | 验证PCET风险增强、safe-known保护，以及增强evidence可通过Stage2协议评估 |

版本状态：`E:\type10-7`不是Git仓库；改动将快照并同步到Git-backed镜像`E:\type10-7\github_publish\CVS-RFFI-repo`。

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_orbit_pcet_ci_eval.py code\tests\test_phase2_orbit_pcet_ci_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_orbit_pcet_ci_eval.py -q` | PASS，3 passed；根目录pytest cache写入被Windows拒绝，不影响测试结果 |

## 本地结果

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes/event | target_pass | resource_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| pcet_known_preserving | 1 | 0.6825 | 0.0000 | 0.1500 | 0.0000 | 0.4333 | 0.5667 | 128.0 | false | true |
| pcet_known_preserving | 2 | 0.6402 | 0.3500 | 0.4000 | 0.3000 | 0.2667 | 0.4333 | 231.6 | false | true |
| pcet_known_preserving | 3 | 0.6138 | 0.3000 | 0.4333 | 0.3750 | 0.5000 | 0.4000 | 314.4 | false | true |
| pcet_known_preserving | 4 | 0.6296 | 0.4000 | 0.4833 | 0.3750 | 0.5333 | 0.4333 | 376.5 | false | true |
| pcet_known_preserving | 5 | 0.5714 | 0.3000 | 0.4333 | 0.3750 | 0.5833 | 0.3667 | 414.2 | false | true |
| pcet_balanced | 5 | 0.1111 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 414.2 | false | true |
| pcet_unknown_strict | 5 | 0.1111 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 414.2 | false | true |

类地板审计：`floor_failure_count=64`，`no_event_count=0`，`schema_errors=[]`，`schema_warnings=0`。

## 初步解释

与ORBIT-C3R的`available_up_to_k`相比，PCET-CI在宽松`known_preserving`档有边际正向信号：`collab_count=5`下old_acc从0.5185提升到0.5714，`unknown_reject`保持0.58左右，资源仍通过。但这不是目标达成：`min_old=0.3000`、`seen_new_acc=0.4333`、`unknown_FAR=0.3667`均远低于门槛。严格拒识档可达到`unknown_reject=1.0000`和`unknown_FAR=0.0000`，但旧类和新类几乎被拒掉，不能作为Stage2-C成功。

当前结论：PCET-CI比单纯强门控更适合作为后续优化方向，但仅靠feature-level原型一致性/尾部风险仍不足以解决unknown拒识与旧/新类保留冲突。下一步若继续，应把PCET风险从硬阈值改成可学习/可校准的source-side episode open-set scorer，或者引入轻量Siamese/verifier头。

## N607计划

远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
远端输出：`runs/phase2_adv3b02_pcet_ci_20260704/`
计划GPU：选择preflight中显存占用最低的GPU，当前预期为GPU0。

同步文件：

| 本地 | 远端 |
|---|---|
| `code\scripts\phase2_orbit_pcet_ci_eval.py` | `code/scripts/phase2_orbit_pcet_ci_eval.py` |
| `code\tests\test_phase2_orbit_pcet_ci_eval.py` | `code/tests/test_phase2_orbit_pcet_ci_eval.py` |

远端命令摘要：

```text
cd /home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
OUT=runs/phase2_adv3b02_pcet_ci_20260704
CUDA_VISIBLE_DEVICES=0 $PY -m py_compile \
  code/scripts/phase2_orbit_pcet_ci_eval.py \
  code/tests/test_phase2_orbit_pcet_ci_eval.py
PYTHONPATH=code:code/scripts CUDA_VISIBLE_DEVICES=0 \
  $PY code/tests/test_phase2_orbit_pcet_ci_eval.py
CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_orbit_pcet_ci_eval.py \
  --feature_npz $OUT/features.npz \
  --output_json $OUT/pcet_ci.json \
  --output_summary_csv $OUT/pcet_ci_summary.csv \
  --output_evidence_csv $OUT/pcet_ci_evidence.csv \
  --profiles all --collab_counts all \
  --collab_group_policy available_up_to_k \
  --partial_collab_min_receivers 1 \
  --k_shot 8 --qknn_k 8 --query_per_class 20 \
  --max_event_bytes 1152 --max_event_latency_ms 20
CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_orbit_c3r_failure_audit.py \
  --input_json $OUT/pcet_ci.json \
  --output_json $OUT/pcet_ci_failure_audit.json \
  --output_class_csv $OUT/pcet_ci_class_floor.csv \
  --output_confusion_csv $OUT/pcet_ci_confusion.csv
```

## N607结果

N607只读preflight通过：直接`N607`目标、项目根目录和8张RTX3090均可见。运行前GPU显存均为`10MiB`，选择GPU0执行；运行结束后8张GPU显存仍为`10MiB`。

远端验证：

| 命令 | 结果 |
|---|---|
| `py_compile` PCET脚本和测试 | PASS |
| `PYTHONPATH=code:code/scripts ... code/tests/test_phase2_orbit_pcet_ci_eval.py` | PASS，3 tests OK |
| PCET-CI全profile、全`collab_count=1..5`、`available_up_to_k`评估 | PASS |
| PCET-CI类地板审计 | PASS，`floor_failure_count=64`，`no_event_count=0`，`schema_errors=[]`，`schema_warnings=0` |

拉回产物：

| 远端 | 本地 |
|---|---|
| `runs/phase2_adv3b02_pcet_ci_20260704/pcet_ci.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_pcet_ci_20260704\remote\pcet_ci.json` |
| `runs/phase2_adv3b02_pcet_ci_20260704/pcet_ci_summary.csv` | `E:\type10-7\local_artifacts\phase2_adv3b02_pcet_ci_20260704\remote\pcet_ci_summary.csv` |
| `runs/phase2_adv3b02_pcet_ci_20260704/pcet_ci_failure_audit.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_pcet_ci_20260704\remote\pcet_ci_failure_audit.json` |
| `runs/phase2_adv3b02_pcet_ci_20260704/pcet_ci_class_floor.csv` | `E:\type10-7\local_artifacts\phase2_adv3b02_pcet_ci_20260704\remote\pcet_ci_class_floor.csv` |
| `runs/phase2_adv3b02_pcet_ci_20260704/pcet_ci_confusion.csv` | `E:\type10-7\local_artifacts\phase2_adv3b02_pcet_ci_20260704\remote\pcet_ci_confusion.csv` |

远端summary与本地一致：

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes/event | target_pass | resource_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| pcet_known_preserving | 1 | 0.6825 | 0.0000 | 0.1500 | 0.0000 | 0.4333 | 0.5667 | 128.0 | false | true |
| pcet_known_preserving | 2 | 0.6402 | 0.3500 | 0.4000 | 0.3000 | 0.2667 | 0.4333 | 231.6 | false | true |
| pcet_known_preserving | 3 | 0.6138 | 0.3000 | 0.4333 | 0.3750 | 0.5000 | 0.4000 | 314.4 | false | true |
| pcet_known_preserving | 4 | 0.6296 | 0.4000 | 0.4833 | 0.3750 | 0.5333 | 0.4333 | 376.5 | false | true |
| pcet_known_preserving | 5 | 0.5714 | 0.3000 | 0.4333 | 0.3750 | 0.5833 | 0.3667 | 414.2 | false | true |
| pcet_balanced | 5 | 0.1111 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 414.2 | false | true |
| pcet_unknown_strict | 5 | 0.1111 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 414.2 | false | true |

SSH/SCP清理：preflight、同步、远端运行和结果拉回后均检查本地`ssh.exe`与到`172.31.111.215:22`/`172.31.105.18:22`的`ESTABLISHED`连接；结果均为`none`。

## 最终判定

PCET-CI是对ORBIT-C3R的有效诊断推进，但不是成功候选。它证明了“低权重原型一致性+尾部风险”可以在宽松profile下略微提升旧类保留，但未知拒识仍未接近目标；严格拒识仍会把旧类和seen-new一起拒掉。下一步应进入可学习的source-side open-set episode scorer或轻量Siamese/verifier，而不是继续手调硬阈值。
