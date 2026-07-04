# phase2_adv3b02_orbit_c3r_guard_20260704

## 基本信息

| 项目 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_orbit_c3r_guard_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 在`ADV3B02_CORE90_SOFT_E200`特征和在轨`qknn8`少样本old/new support基础上，实现可部署的卫星群协同开集拒识ORBIT-C3R Guard，并报告`collab_count=1..N`、时延、通信和目标差距 |
| 协议边界 | Stage2-C；`Y_unknown`只用于query评估，不参与support、阈值拟合或receiver选择监督 |
| 输入feature | `E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz` |
| 本地输出 | `E:\type10-7\local_artifacts\phase2_adv3b02_orbit_c3r_guard_20260704\` |
| 协同范围 | `collab_count=1..5`；区别于`K-shot=8` |
| 资源预算 | 128B/receiver/event，默认最大1152B/event，最大20ms/event |

## 本地改动

| 文件 | 目的 |
|---|---|
| `E:\type10-7\code\scripts\phase2_orbit_c3r_guard_eval.py` | 新增ORBIT-C3R部署式协同评估封装：生成qknn8证据，运行old-preserving、old-guarded、balanced、unknown-strict四档profile，输出目标差距和资源字段 |
| `E:\type10-7\code\tests\test_phase2_orbit_c3r_guard_eval.py` | 验证profile输出、`collab_count=1..N`、资源字段、`unknown_query_eval_only=True`和非正shot拒绝 |
| `E:\type10-7\code\scripts\phase2_orbit_c3r_failure_audit.py` | 新增ORBIT-C3R类地板和open-set confusion审计；区分`class_total==0`覆盖伪影与`class_total>0`真实类地板失败；增加schema错误、有限数校验和稳定CSV表头 |
| `E:\type10-7\code\tests\test_phase2_orbit_c3r_failure_audit.py` | 覆盖真实floor failure、no-event分层、缺失schema非零退出、`nan/inf`无效acc和`"1.0"`计数解析 |

版本状态：`E:\type10-7`不是Git仓库；将创建快照并同步到Git-backed镜像`E:\type10-7\github_publish\CVS-RFFI-repo`。

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_orbit_c3r_guard_eval.py code\tests\test_phase2_orbit_c3r_guard_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_orbit_c3r_guard_eval.py -q` | PASS，2 passed；根目录pytest cache写入被Windows拒绝，不影响测试结果 |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_orbit_c3r_failure_audit.py code\tests\test_phase2_orbit_c3r_failure_audit.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_orbit_c3r_failure_audit.py -q` | PASS，3 passed；根目录pytest cache写入被Windows拒绝，不影响测试结果 |

## 算法配置

ORBIT-C3R Guard当前实现为部署层封装，不改`项目.md`协议：

```text
base feature: ADV3B02 z_id
in-orbit support: target old + seen-new K-shot support
local learner: qknn8
unknown calibration: support-generated virtual/class-negative/class-shell risk only
unknown query: evaluation-only
fusion: SCG qknn evidence guard with support-quality receiver selection
```

四档profile：

| profile | 用途 |
|---|---|
| old_preserving | 宽松support-confirmed known基线，先观察旧类/新类保护能力 |
| old_guarded | 旧类保护+强多源unknown风险拒识 |
| balanced | 折中旧类/新类接受与unknown共识拒识 |
| unknown_strict | 严格unknown安全档，用于暴露旧类/新类损伤 |

## 本地结果

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | request_more | bytes/event | latency_ms | target_pass | resource_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| old_preserving | 1 | 0.6720 | 0.0000 | 0.1500 | 0.0000 | 0.4333 | 0.5667 | 0.0000 | 0.0000 | 128.0 | 0.0 | false | true |
| old_preserving | 2 | 0.6078 | 0.3500 | 0.4898 | 0.4138 | 0.1250 | 0.4792 | 0.0000 | 0.3750 | 256.0 | 0.0 | false | true |
| old_preserving | 3 | 0.5417 | 0.0000 | 0.5250 | 0.5000 | 0.4500 | 0.3000 | 0.0000 | 0.2500 | 384.0 | 0.0 | false | true |
| old_preserving | 4 | 0.6092 | 0.0000 | 0.6129 | 0.4545 | 0.5938 | 0.3438 | 0.0000 | 0.0625 | 512.0 | 0.0 | false | true |
| old_preserving | 5 | 0.5294 | 0.0000 | 0.5500 | 0.0000 | 0.8000 | 0.0500 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |
| old_guarded | 1 | 0.3175 | 0.0000 | 0.0500 | 0.0000 | 0.7500 | 0.2333 | 0.0000 | 0.0167 | 128.0 | 0.0 | false | true |
| old_guarded | 2 | 0.1307 | 0.0000 | 0.0000 | 0.0000 | 0.7083 | 0.0417 | 0.0000 | 0.2500 | 256.0 | 0.0 | false | true |
| old_guarded | 3 | 0.1583 | 0.0000 | 0.0000 | 0.0000 | 0.8750 | 0.0000 | 0.0000 | 0.1250 | 384.0 | 0.0 | false | true |
| old_guarded | 4 | 0.0460 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 512.0 | 0.0 | false | true |
| old_guarded | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |
| balanced | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |
| unknown_strict | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |

## 解释

本地结果表明ORBIT-C3R部署层封装满足通信资源预算，但不能解决目标问题。`old_preserving`在`collab_count=5`可把`unknown_FAR`降到0.05，但`old_acc=0.5294`、`min_old=0.0000`，旧类性能严重不合格。更严格profile可达到`unknown_reject=1.0000`，但旧类和新类识别被全部拒掉或错误路由，不能作为候选路线。

该结果与oracle unknown holdout负证据一致：当前ADV3B02`z_id`下，部署层拒识门控会在unknown与old/seen-new之间产生强冲突。下一步不能继续只调门控，应回到地面表示学习，加入source/proxy outlier exposure、energy/open-space margin、receiver-invariant identity约束或旧类类地板约束，再回到ORBIT-C3R做部署层验证。

## N607执行

远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
远端输出：`runs/phase2_adv3b02_orbit_c3r_guard_20260704/`
拉回目录：`E:\type10-7\local_artifacts\phase2_adv3b02_orbit_c3r_guard_20260704\remote\`

N607只读preflight通过：直接`N607`目标、项目根目录和8张RTX 3090均可见。运行前GPU占用均为`10MiB`，未发现本用户训练进程；选择低占用GPU0执行诊断。运行结束后`nvidia-smi`显示8张GPU仍为`10MiB`。

同步文件：

| 本地 | 远端 |
|---|---|
| `code\scripts\phase2_orbit_c3r_guard_eval.py` | `code/scripts/phase2_orbit_c3r_guard_eval.py` |
| `code\tests\test_phase2_orbit_c3r_guard_eval.py` | `code/tests/test_phase2_orbit_c3r_guard_eval.py` |
| `remote_artifacts\phase2_adv3b02_features\features.npz` | `runs/phase2_adv3b02_orbit_c3r_guard_20260704/input/features.npz` |

远端验证：

| 命令 | 结果 |
|---|---|
| `py_compile` ORBIT脚本和测试 | PASS |
| `PYTHONPATH=code:code/scripts ... code/tests/test_phase2_orbit_c3r_guard_eval.py` | PASS，2 tests OK；负值shot测试产生预期argparse错误文本 |
| ORBIT-C3R全profile全`collab_count=1..5`评估 | PASS，输出JSON/CSV |

远端结果与本地一致：

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | request_more | bytes/event | latency_ms | target_pass | resource_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| old_preserving | 1 | 0.6720 | 0.0000 | 0.1500 | 0.0000 | 0.4333 | 0.5667 | 0.0000 | 0.0000 | 128.0 | 0.0 | false | true |
| old_preserving | 2 | 0.6078 | 0.3500 | 0.4898 | 0.4138 | 0.1250 | 0.4792 | 0.0000 | 0.3750 | 256.0 | 0.0 | false | true |
| old_preserving | 3 | 0.5417 | 0.0000 | 0.5250 | 0.5000 | 0.4500 | 0.3000 | 0.0000 | 0.2500 | 384.0 | 0.0 | false | true |
| old_preserving | 4 | 0.6092 | 0.0000 | 0.6129 | 0.4545 | 0.5938 | 0.3438 | 0.0000 | 0.0625 | 512.0 | 0.0 | false | true |
| old_preserving | 5 | 0.5294 | 0.0000 | 0.5500 | 0.0000 | 0.8000 | 0.0500 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |
| old_guarded | 3 | 0.1583 | 0.0000 | 0.0000 | 0.0000 | 0.8750 | 0.0000 | 0.0000 | 0.1250 | 384.0 | 0.0 | false | true |
| old_guarded | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |
| balanced | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |
| unknown_strict | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 640.0 | 0.0 | false | true |

SSH/SCP清理：preflight、进程检查、同步、运行和结果拉回之后均检查本地`ssh.exe`与到`172.31.111.215:22`/`172.31.105.18:22`的`ESTABLISHED`连接；结果均为`none`。

最终判定：本轮ORBIT-C3R实现是可部署资源约束下的诊断/候选评估工具，但当前ADV3B02+qknn8表示无法达到目标。所有profile的`target_pass=false`，因此不能登记为Stage2-C成功或部署证据。

## available_up_to_k追加审计

追加目的：`exact_k`要求每个事件恰好凑齐`k`个接收机证据，高协同数量下会出现部分类别`class_total==0`的覆盖伪影。为回应“协同推理数量从1到接收机数量可选”的部署语义，追加`collab_group_policy=available_up_to_k`。该设置把`collab_count=k`解释为“每个事件最多使用k个可用接收机证据”，底层实际使用`selected_k=min(k,event_receiver_count)`；因此`collab_count=5`不是每个事件必然使用5个接收机。

本地输出：

| 文件 | 说明 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_orbit_c3r_guard_20260704\available_up_to_k\orbit_c3r_guard_available.json` | 本地available-up-to-k全profile结果 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_orbit_c3r_guard_20260704\available_up_to_k\orbit_c3r_guard_available_summary.csv` | 本地summary |
| `E:\type10-7\local_artifacts\phase2_adv3b02_orbit_c3r_guard_20260704\available_up_to_k\orbit_c3r_available_failure_audit.json` | 本地修正口径类地板审计 |

N607输出拉回：

| 文件 | 说明 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_orbit_c3r_guard_20260704\remote_available\orbit_c3r_guard_available.json` | N607的available-up-to-k全profile结果 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_orbit_c3r_guard_20260704\remote_available\orbit_c3r_guard_available_summary.csv` | N607summary |
| `E:\type10-7\local_artifacts\phase2_adv3b02_orbit_c3r_guard_20260704\remote_available\orbit_c3r_available_failure_audit.json` | N607修正口径类地板审计 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_orbit_c3r_guard_20260704\remote_available\orbit_c3r_available_class_floor.csv` | N607逐类accuracy、decision和output计数 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_orbit_c3r_guard_20260704\remote_available\orbit_c3r_available_confusion.csv` | N607open-set confusion计数 |

N607执行命令摘要：

```text
cd /home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_orbit_c3r_guard_eval.py \
  --feature_npz runs/phase2_adv3b02_orbit_c3r_guard_20260704/input/features.npz \
  --output_json runs/phase2_adv3b02_orbit_c3r_guard_20260704/orbit_c3r_guard_available.json \
  --output_summary_csv runs/phase2_adv3b02_orbit_c3r_guard_20260704/orbit_c3r_guard_available_summary.csv \
  --output_evidence_csv runs/phase2_adv3b02_orbit_c3r_guard_20260704/orbit_c3r_guard_available_evidence.csv \
  --profiles all --collab_counts all --collab_group_policy available_up_to_k \
  --partial_collab_min_receivers 1 --k_shot 8 --qknn_k 8 --query_per_class 20 \
  --max_event_bytes 1152 --max_event_latency_ms 20
CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_orbit_c3r_failure_audit.py \
  --input_json runs/phase2_adv3b02_orbit_c3r_guard_20260704/orbit_c3r_guard_available.json \
  --output_json runs/phase2_adv3b02_orbit_c3r_guard_20260704/orbit_c3r_available_failure_audit.json \
  --output_class_csv runs/phase2_adv3b02_orbit_c3r_guard_20260704/orbit_c3r_available_class_floor.csv \
  --output_confusion_csv runs/phase2_adv3b02_orbit_c3r_guard_20260704/orbit_c3r_available_confusion.csv
```

N607验证：`py_compile`通过；`PYTHONPATH=code:code/scripts ... test_phase2_orbit_c3r_failure_audit.py`通过，3 tests OK；审计输出`floor_failure_count=95`、`no_event_count=0`、`schema_errors=[]`、`schema_warnings=0`。运行前后GPU0和其他GPU显存均为`10MiB`。每次preflight、SCP、SSH运行和拉回后均检查本地`ssh.exe`与到`172.31.111.215:22`/`172.31.105.18:22`的`ESTABLISHED`连接，结果均为`none`。

### 实际协同覆盖

以下表格来自N607`old_preserving`结果。`actual_receiver_hist`为每个预算下事件实际使用接收机数量分布；`partial_groups`为实际接收机数量小于预算的事件数。

| receiver_budget | actual_receiver_hist | events | excluded_incomplete | partial_groups | exact_budget_groups | avg_receivers_used | max_receivers_used | bytes/event |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `{"1": 309}` | 309 | 0 | 0 | 309 | 1.0000 | 1 | 128.0 |
| 2 | `{"1": 59, "2": 250}` | 309 | 0 | 59 | 250 | 1.8091 | 2 | 231.6 |
| 3 | `{"1": 59, "2": 50, "3": 200}` | 309 | 0 | 109 | 200 | 2.4563 | 3 | 314.4 |
| 4 | `{"1": 59, "2": 50, "3": 50, "4": 150}` | 309 | 0 | 159 | 150 | 2.9417 | 4 | 376.5 |
| 5 | `{"1": 59, "2": 50, "3": 50, "4": 59, "5": 91}` | 309 | 0 | 218 | 91 | 3.2362 | 5 | 414.2 |

### N607结果

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes/event | target_pass | resource_pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| old_preserving | 1 | 0.6720 | 0.0000 | 0.1500 | 0.0000 | 0.4333 | 0.5667 | 128.0 | false | true |
| old_preserving | 2 | 0.6402 | 0.3500 | 0.4000 | 0.3000 | 0.2500 | 0.4333 | 231.6 | false | true |
| old_preserving | 3 | 0.6085 | 0.3000 | 0.4333 | 0.3750 | 0.4500 | 0.3667 | 314.4 | false | true |
| old_preserving | 4 | 0.5767 | 0.4000 | 0.4833 | 0.3750 | 0.5500 | 0.4000 | 376.5 | false | true |
| old_preserving | 5 | 0.5185 | 0.3000 | 0.4333 | 0.3750 | 0.5667 | 0.3667 | 414.2 | false | true |
| old_guarded | 5 | 0.0635 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 414.2 | false | true |
| balanced | 5 | 0.0635 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 414.2 | false | true |
| unknown_strict | 5 | 0.0582 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 414.2 | false | true |

### 类地板审计

修正后的审计口径：

| 口径 | 说明 | 计数 |
|---|---|---:|
| exact-k real floor failure | `class_total>0`且`class_acc<=0` | 88 |
| exact-k no-event coverage artifact | `class_total==0`，不计入真实类地板失败 | 16 |
| available-up-to-k real floor failure | `class_total>0`且`class_acc<=0` | 95 |
| available-up-to-k no-event coverage artifact | `class_total==0` | 0 |

`available_up_to_k`消除了no-event伪影，但真实逐类地板仍明显不达标。`old_preserving`的代表性低分行如下：

| profile | collab_count | role | label | class_acc | class_total | accept | defer | request_more | unknown_reject |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| old_preserving | 2 | seen_new | 3-8 | 0.3000 | 40 | 24 |  | 4 | 12 |
| old_preserving | 5 | old | 14-10 | 0.3000 | 20 | 8 |  |  | 12 |
| old_preserving | 2 | old | 14-10 | 0.3500 | 20 | 11 |  | 4 | 5 |
| old_preserving | 5 | old | 8-20 | 0.3636 | 33 | 12 | 20 |  | 1 |
| old_preserving | 5 | seen_new | 3-8 | 0.3750 | 40 | 27 | 1 |  | 12 |
| old_preserving | 5 | old | 20-15 | 0.4000 | 40 | 17 | 19 |  | 4 |
| old_preserving | 2 | old | 20-15 | 0.4500 | 40 | 18 | 3 | 18 | 1 |
| old_preserving | 2 | old | 14-7 | 0.4750 | 40 | 27 | 1 | 9 | 3 |
| old_preserving | 5 | old | 14-7 | 0.4750 | 40 | 27 | 5 |  | 8 |
| old_preserving | 5 | seen_new | 19-3 | 0.5500 | 20 | 12 |  |  | 8 |

## 结论边界与下一步算法

结论边界：当前结论只针对`ADV3B02+qknn8+ORBIT-C3R Guard`这一组证据，不外推为所有协同推理算法无效。`available_up_to_k`已经更贴近“最多使用可用接收机”的卫星群部署语义，但仍不能满足目标：宽松`old_preserving`无法把unknown拒识提升到目标，同时旧类/seen-new类地板远低于目标；严格profile可把`unknown_FAR`压到0，但以拒掉几乎全部旧类和新类为代价。因此本轮不是Stage2-C成功，不是部署证据，只能作为负向诊断。

下一步优先级不应继续单纯调门控，而应引入可在星上轻量部署的表示/拒识联合机制：

| 优先级 | 方法 | 作用 | 可落地实验 |
|---:|---|---|---|
| 1 | SnaTCHer式原型一致性 | 未知样本即使有近邻，也会破坏old/new原型几何 | 冻结ADV3B02，计算query替换best prototype前后的原型一致性差值，作为每接收机`unknown_risk` |
| 2 | ProtoNet-EVT/OpenMax尾部半径 | 为每个old/seen-new类学习距离尾部分布和拒识半径 | 只用support、source proxy和leave-out TX拟合Weibull/EVT，unknown query只评估 |
| 3 | Energy/OE联合risk | 用source-side伪异常改善FAR与覆盖冲突 | 从ManyTx非目标评估TX构造outlier exposure，输出`energy+radius+margin`三路risk |
| 4 | qknn后置Siamese/verifier | top1近邻不等于同一发射机 | 对qknn top1/top2与support/prototype做轻量同源验证，失败则defer/reject |
| 5 | FedProto/COIN-LEO式top-M调度 | 降低星间通信并避免坏接收机污染融合 | 比较score-only包与score+int8 sketch包，按support density、unknown冲突、链路时延选择卫星 |

这些方法仍需保持`Y_unknown`只用于query评估，不参与阈值拟合、support选择或训练监督；若使用source-side proxy outlier，必须显式记录其来源并与最终`Y_unknown`互斥。
