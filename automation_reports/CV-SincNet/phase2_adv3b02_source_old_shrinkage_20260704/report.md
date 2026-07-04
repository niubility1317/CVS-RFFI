# phase2_adv3b02_source_old_shrinkage_20260704

## 基本信息

| 项目 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_source_old_shrinkage_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 在OPUC-qKNN负结果基础上，引入source-old prototype shrinkage，验证是否能在提升unknown拒识时保护旧类准确率 |
| 底座模型 | `ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| 本地特征 | `E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz` |
| 协同范围 | `receiver_domain_ranked`覆盖1..5；strict同事件仍受限于最多3接收机 |
| 状态 | NON_DEPLOYMENT_DIAGNOSTIC |

## 算法改动

新增source-old prototype shrinkage：

```text
c_old' = normalize((1-alpha) * c_target_old + alpha * c_source_old)
```

其中`c_target_old`只由目标接收机`R_t`的target-old support构建，`c_source_old`只由`source`角色中同一旧类TX构建。该机制只作用于`Y_old`原型，不改变seen-new注册，不使用unknown query，不改变`R_s/R_t`和`Y_old/Y_new/Y_unknown`协议。

新增CLI参数：

```text
--source_old_prototype_shrinkage_alpha
```

默认`0.0`保持既有行为。

## 本地变更

| 文件 | 目的 |
|---|---|
| `E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py` | 在qknn memory中加入source-old prototype shrinkage、metadata和evidence字段 |
| `E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py` | 新增旧类原型shrinkage只作用于old centroid的回归测试 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_source_old_shrinkage_20260704\*` | 非Git根代码快照，N607同步前保存 |

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_collaborative_open_set_qknn_eval.py -q -k "source_old_prototype_shrinkage or support_selection_policy or strict_event_query_preserve"` | PASS，2 passed |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase2_event_alignment_audit.py -q` | PASS，56 passed |

pytest存在`.pytest_cache`写入权限warning，不影响测试结论。

## 本地矩阵结果

配置基线继承OPUC-qKNN：`support_envelope_evt`、`old_protected_unknown_confirm_cvs`、`strict_event_query_preserve`、`class_conformal_enabled`、`class_shell_unknown_risk_enabled`、`virtual_unknown_risk_enabled`、`evidence_packet_bytes=96`。

### source-old shrinkage alpha扫描

| 输出 | k | alpha | old_acc | seen_new_acc | unknown_reject_rate | known_coverage | defer_rate | bytes/event | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `receiver_domain_ranked_alpha02.json` | 5 | 0.2 | 0.5283 | 0.5500 | 0.8500 | 0.5616 | 0.0753 | 480 | 未改善old |
| `receiver_domain_ranked_alpha05.json` | 5 | 0.5 | 0.5283 | 0.5500 | 0.8500 | 0.5616 | 0.0753 | 480 | 未改善old |
| `receiver_domain_ranked_alpha08.json` | 5 | 0.8 | 0.5283 | 0.5500 | 0.8500 | 0.5616 | 0.0753 | 480 | 未改善old |

alpha扫描说明：单纯把旧类原型向source-old锚定不能突破OPUC的old/unknown冲突，k=5主结果与无shrinkage基本一致。

### 宽门控诊断

| 输出 | k | old_acc | seen_new_acc | unknown_reject_rate | known_coverage | bytes/event | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `receiver_domain_ranked_loose_gate.json` | 5 | 0.8679 | 0.7500 | 0.1500 | 1.0000 | 480 | old恢复但unknown崩 |
| `receiver_domain_ranked_loose_gate_shell05.json` | 5 | 0.7925 | 0.7500 | 0.4500 | 0.9315 | 480 | unknown改善有限且old低于OLD80/known基线 |
| `receiver_domain_ranked_loose_gate_score.json` | 5 | 0.8679 | 0.7500 | 0.1500 | 1.0000 | 480 | 与宽门控一致 |

宽门控说明：如果放宽old accept gate，旧类可以恢复到接近known route水平，但unknown拒识退化到0.15；如果收紧shell拒识，unknown最多到0.45，old又低于0.80。这证明当前ADV3B02特征空间中old与unknown仍严重重叠，融合层无法同时满足目标。

## 资源约束

用户指定的`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`按精确文件名和关键词在当前工作区未找到。当前只能记录代码可验证代理指标：

| 协同k | bytes/event |
|---:|---:|
| 1 | 96 |
| 2 | 192 |
| 3 | 288 |
| 4 | 384 |
| 5 | 480 |

后续若提供资源约束原文，需要把`max_event_bytes`、`max_event_latency_ms`、state size、prototype storage和星间链路预算重新对齐。

## 判定

source-old prototype shrinkage模块实现有效，但在ADV3B02+qknn8当前特征上未达成目标。当前最重要结论是：unknown拒识困难不是单纯旧类原型不稳，而是星地信道下unknown样本会落入old/seen-new高置信邻域。下一步应进入特征层/训练层：per-class EVT/Mahalanobis tail强化、virtual future class空域约束、cache adapter蒸馏或target-old/unknown proxy对比约束，而不是继续单纯调融合门限。

## N607计划

远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`  
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`  
远端输出：`runs/phase2_adv3b02_source_old_shrinkage_20260704/`

待同步文件：

| 本地 | 远端 |
|---|---|
| `code\scripts\phase2_collaborative_open_set_qknn_eval.py` | `code/scripts/phase2_collaborative_open_set_qknn_eval.py` |
| `code\tests\test_phase2_collaborative_open_set_qknn_eval.py` | `code/tests/test_phase2_collaborative_open_set_qknn_eval.py` |

待运行远端命令：`py_compile`、目标单元测试、alpha=0.2 OPUC主矩阵、loose gate诊断矩阵。运行前执行N607 preflight，运行后拉回结果并检查SSH清理。

## N607验证结果

| 项目 | 结果 |
|---|---|
| preflight | PASS，直连`N607`，项目根可见，8张RTX3090均为10MiB |
| 远端Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 远端目录 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_source_old_shrinkage_20260704` |
| 远端测试 | `py_compile` PASS；`test_phase2_collaborative_open_set_qknn_eval.py` 53 tests OK |
| 远端GPU | 运行结束后8张GPU均为10MiB |
| SSH清理 | preflight、scp、远端运行和拉回后，本地`ssh.exe`与到`172.31.111.215:22`/`172.31.105.18:22`的ESTABLISHED连接均为空 |

远端输出已拉回：

| 远端结果 | 本地副本 |
|---|---|
| `runs/phase2_adv3b02_source_old_shrinkage_20260704/receiver_domain_ranked_alpha02.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_source_old_shrinkage_20260704\remote\receiver_domain_ranked_alpha02.json` |
| `runs/phase2_adv3b02_source_old_shrinkage_20260704/receiver_domain_ranked_loose_gate.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_source_old_shrinkage_20260704\remote\receiver_domain_ranked_loose_gate.json` |

### N607 alpha=0.2主矩阵

| k | old_acc | seen_new_acc | unknown_reject_rate | known_coverage | bytes/event | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1979 | 0.0000 | 0.9667 | 0.1700 | 96 | old严重下降 |
| 2 | 0.2632 | 0.4314 | 0.5870 | 0.4828 | 192 | old严重下降 |
| 3 | 0.3500 | 0.3000 | 0.6500 | 0.4562 | 288 | old严重下降 |
| 4 | 0.4545 | 0.4483 | 0.6176 | 0.5812 | 384 | old严重下降 |
| 5 | 0.5283 | 0.5500 | 0.8500 | 0.5616 | 480 | unknown改善但old未保护 |

### N607宽门控诊断

| k | old_acc | seen_new_acc | unknown_reject_rate | known_coverage | bytes/event | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1979 | 0.0000 | 0.9667 | 0.1700 | 96 | old严重下降 |
| 2 | 0.5263 | 0.4510 | 0.1739 | 0.8621 | 192 | unknown崩 |
| 3 | 0.7667 | 0.4000 | 0.1500 | 0.9375 | 288 | unknown崩 |
| 4 | 0.7386 | 0.5172 | 0.1765 | 0.9487 | 384 | unknown崩 |
| 5 | 0.8679 | 0.7500 | 0.1500 | 1.0000 | 480 | old恢复但unknown崩 |

N607复核结论与本地一致：source-old prototype shrinkage没有解决unknown/old冲突。宽门控能恢复old，但unknown拒识显著退化；收紧unknown门控能提高拒识，但old和seen-new下降。该模块只能作为诊断组件保留。
