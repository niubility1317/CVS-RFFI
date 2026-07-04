# phase2_adv3b02_strict_event_alignment_20260704

## 基本信息

| 字段 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_strict_event_alignment_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 底座模型 | ADV3B02_CORE90_SOFT_E200 |
| 目标 | 将协同推理从`receiver_domain_ranked`代理诊断推进到`strict_event_key`同事件候选评估，核查真实同事件卫星群协同是否改善unknown拒识 |
| 特征 | `remote_artifacts\phase2_adv3b02_features\features.npz`，checkpoint为`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| 状态 | NON_DEPLOYMENT_DIAGNOSTIC |

## 协议与资源边界

`项目.md`允许多接收机target receiver domain，但报告必须区分deployment proxy与严格同事件协同。本轮新增审计确认：在按qknn8的`k_shot=8,query_per_class=20,seed=4070303,stable_first`划分后，ADV3B02特征包存在部分同事件query候选，但没有覆盖全部5个target receivers的query组。

资源约束说明文件`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`仍未在`E:\type10-7`中找到；本轮继续记录代理资源字段：参与receiver数、bytes/event、latency_ms_p95、prototype storage。

## 本地变更

| 文件 | 作用 |
|---|---|
| `E:\type10-7\code\scripts\phase2_event_alignment_audit.py` | 新增同事件候选审计脚本，按主评估器support/query split统计strict key覆盖 |
| `E:\type10-7\code\tests\test_phase2_event_alignment_audit.py` | 新增strict key覆盖和query-only审计单测 |
| `E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py` | 新增`strict_event_min_receivers`，允许strict同事件部分receiver组进入证据构建 |
| `E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py` | 新增strict同事件部分receiver组回归测试 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_strict_event_alignment_20260704\*` | 非Git根代码快照 |

根目录`E:\type10-7`不是Git仓库；最终变更将同步到Git镜像`E:\type10-7\github_publish\CVS-RFFI-repo`。

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_event_alignment_audit.py -q` | PASS，3 passed；`.pytest_cache`权限警告不影响测试 |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_event_alignment_audit.py code\tests\test_phase2_event_alignment_audit.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_collaborative_open_set_qknn_eval.py -q -k "strict_event_key"` | PASS，2 passed |

## 同事件候选审计

审计命令：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_event_alignment_audit.py --feature_npz remote_artifacts\phase2_adv3b02_features\features.npz --output_json local_artifacts\phase2_adv3b02_event_alignment_audit_20260704\event_alignment_query_audit.json --output_groups_csv local_artifacts\phase2_adv3b02_event_alignment_audit_20260704\event_alignment_query_groups.csv --k_shot 8 --query_per_class 20 --seed 4070303
```

审计结果：

| 字段 | 值 |
|---|---:|
| query row count | 1320 |
| strict query group count | 925 |
| groups with >=2 receivers | 65 |
| groups with >=3 receivers | 10 |
| groups with >=4 receivers | 0 |
| groups with 5 receivers | 0 |
| max receivers per strict key | 3 |

结论：严格同事件协同可做`2..3`接收机候选诊断，但当前划分不能支持5接收机全体同事件协同。报告中不得把`k=4/5`解释为真实5星同事件协同；对于strict_event_key，`k=4/5`只能表示预算上限，实际最多仍为3个同事件receiver。

## 本地strict known-route结果

运行命令核心参数：

```text
event_alignment_policy=strict_event_key
strict_event_min_receivers=2
unknown_gate_mode=score
fusion_policy=risk_margin
evidence_packet_bytes=40
```

| k预算 | actual receiver histogram | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_coverage | bytes/event | latency_ms_p95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `{'1':65}` | 0.3333 | 0.0000 | 0.6897 | 0.6190 | 0.1818 | 0.8182 | 0.9688 | 40.0 | 0.0812 |
| 2 | `{'2':65}` | 0.6667 | 0.0000 | 0.7241 | 0.6190 | 0.0000 | 0.8182 | 0.9375 | 80.0 | 0.0812 |
| 3 | `{'2':55,'3':10}` | 0.6667 | 0.0000 | 0.7241 | 0.6190 | 0.0000 | 0.8182 | 0.9375 | 86.2 | 0.0812 |
| 4 | `{'2':55,'3':10}` | 0.6667 | 0.0000 | 0.7241 | 0.6190 | 0.0000 | 0.8182 | 0.9375 | 86.2 | 0.0812 |
| 5 | `{'2':55,'3':10}` | 0.6667 | 0.0000 | 0.7241 | 0.6190 | 0.0000 | 0.8182 | 0.9375 | 86.2 | 0.0812 |

## 本地strict safety-route结果

运行命令核心参数：

```text
event_alignment_policy=strict_event_key
strict_event_min_receivers=2
unknown_gate_mode=support_envelope_consensus
fusion_policy=candidate_set_cvs
evidence_packet_bytes=128
```

| k预算 | actual receiver histogram | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_coverage | bytes/event | latency_ms_p95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `{'1':65}` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8788 | 0.0000 | 0.0000 | 128.0 | 0.1337 |
| 2 | `{'2':65}` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 256.0 | 0.1337 |
| 3 | `{'2':55,'3':10}` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 275.7 | 0.1337 |
| 4 | `{'2':55,'3':10}` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 275.7 | 0.1337 |
| 5 | `{'2':55,'3':10}` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 275.7 | 0.1337 |

## 解释

strict same-event协同没有解决当前核心矛盾。known-route在同事件约束下样本覆盖下降，old_acc最高只有0.6667且unknown_FAR仍为0.8182；safety-route可使unknown_FAR为0，但old/new全被拒识，属于过拒识负诊断。严格同事件键本身是必要工程修复，但在当前ADV3B02+qknn8特征几何上，单靠同事件融合仍不足以达到目标。

下一步应基于strict same-event修复继续做两类工作：

| 方向 | 具体要求 |
|---|---|
| 数据/事件构造 | 增加query同事件覆盖，保证每个role/tx/scenario下有更多`>=4/5 receiver`同事件组，否则无法验证全体卫星协同 |
| 表征/负锚 | 在冻结主干下构造query-free背景原型或源端/target support派生负锚，解决unknown与known特征重叠，而不是继续只调后处理阈值 |

## N607计划

同步脚本、测试和报告到N607，使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`复测event audit、strict known-route和strict safety-route。远端输出目标：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_strict_event_alignment_20260704
```

## N607验证

| 字段 | 内容 |
|---|---|
| 预检时间 | 2026-07-04 09:10:57 CST |
| SSH目标 | direct `N607`，配置`E:\type10-7\tools\n607_ssh_config` |
| 远端Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python 3.10.19 |
| GPU选择 | GPU0；运行前后8张RTX 3090均为10MiB显存占用 |
| 远端测试 | `py_compile`通过；`test_phase2_event_alignment_audit.py`通过3 tests；`test_phase2_collaborative_open_set_qknn_eval.py`通过51 tests |
| 远端pytest状态 | CVS-RFFI环境无`pytest`模块，因此远端主测试使用`unittest`直接执行 |
| SSH清理 | 远端运行和SCP拉回后，本地`ssh.exe`与到`172.31.111.215:22`/`172.31.105.18:22`的连接均为空 |

远端命令执行了：

```bash
CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_event_alignment_audit.py --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz --output_json runs/phase2_adv3b02_strict_event_alignment_20260704/event_alignment_query_audit.json --output_groups_csv runs/phase2_adv3b02_strict_event_alignment_20260704/event_alignment_query_groups.csv --k_shot 8 --query_per_class 20 --seed 4070303

CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_collaborative_open_set_qknn_eval.py ... --event_alignment_policy strict_event_key --strict_event_min_receivers 2 ... --output_json runs/phase2_adv3b02_strict_event_alignment_20260704/strict_event_known_min2.json

CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_collaborative_open_set_qknn_eval.py ... --event_alignment_policy strict_event_key --strict_event_min_receivers 2 ... --output_json runs/phase2_adv3b02_strict_event_alignment_20260704/strict_event_safety_min2.json
```

远端输出已拉回：

| 文件 | 内容 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_event_alignment_audit_20260704\remote_event_alignment_query_audit.json` | N607 query-only strict event候选审计 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_strict_event_known_20260704\remote_strict_event_known_min2.json` | N607 strict known-route结果 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_strict_event_known_20260704\remote_strict_event_safety_min2.json` | N607 strict safety-route结果 |

N607结论与本地一致：query-only strict同事件候选只支持`>=2 receiver`的65组和`>=3 receiver`的10组，不支持4/5 receiver同事件query组；known-route在strict条件下unknown_FAR仍为0.8182；safety-route的unknown_FAR为0但old/new全崩。
