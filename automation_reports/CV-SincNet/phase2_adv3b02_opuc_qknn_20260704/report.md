# phase2_adv3b02_opuc_qknn_20260704

## 基本信息

| 项目 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_opuc_qknn_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 设计并验证天基射频指纹识别卫星群协同推理算法，优先降低unknown误接受，同时不降低old准确率 |
| 权重/特征 | `ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`，本地特征`remote_artifacts\phase2_adv3b02_features\features.npz` |
| Conda环境 | 本地`ssr-gpu`，远端按用户要求使用`CVS-RFFI` |
| 结论状态 | NON_DEPLOYMENT_DIAGNOSTIC |

## 算法设计

新增`old_protected_unknown_confirm_cvs`融合策略，内部复用已验证的`scg_qknn_cvs`证据计算路径，但用更明确的部署语义记录：

1. 旧类/seen-new先走support-confirmed known保护：候选类需要多接收机候选支持、top1支持、conformal pvalue、receiver reliability、support density、margin和pair disagreement约束。
2. unknown拒识必须由多源证据确认：event unknown risk、label unknown risk、class shell risk、risk component agreement、receiver unknown majority等至少形成复合证据；单源unknown证据不应直接覆盖强known。
3. 边界样本优先`request_more/defer`，不将不充分证据写成unknown成功。
4. 支持`strict_event_query_preserve`support选择策略：优先把单接收机事件用作support，把跨接收机共享同事件样本保留给query。

文献/方法依据包括OpenMax/EVT、KNN-OOD、ProtoNet/SimpleShot、Mahalanobis/MLGPN、Tip-Adapter式cache、LoRA/小adapter和多星early-exit协同。对CVS-RFFI当前可落地的路线是冻结`z_id`特征，星上只更新qknn8/cache/prototype和小adapter，避免整网在线微调导致old遗忘。

## 本地变更

| 文件 | 目的 |
|---|---|
| `E:\type10-7\code\evaluation\collaborative_open_set_qknn_eval.py` | 新增`old_protected_unknown_confirm_cvs`策略入口，保留请求策略名并内部映射到`scg_qknn_cvs` |
| `E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py` | CLI支持新融合策略和`strict_event_query_preserve`support策略 |
| `E:\type10-7\code\scripts\phase2_event_alignment_audit.py` | 同事件审计支持指定support选择策略并记录到JSON |
| `E:\type10-7\code\tests\test_collaborative_open_set_qknn_eval.py` | 新增OPUC策略别名和决策回归测试 |
| `E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py` | 新增strict-event query preserve支持选择测试 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_opuc_qknn_20260704\*` | 非Git根代码快照 |

根目录`E:\type10-7`不是Git仓库；变更已复制到Git镜像`E:\type10-7\github_publish\CVS-RFFI-repo`，待N607验证后提交。

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\scripts\phase2_event_alignment_audit.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_collaborative_open_set_qknn_eval.py -q -k "scg_qknn or old_protected_unknown_confirm"` | PASS，3 passed |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase2_event_alignment_audit.py -q` | PASS，55 passed |

pytest仅有`.pytest_cache`写入权限warning，不影响测试结论。

## 同事件覆盖审计

使用`strict_event_query_preserve`后，query-only strict同事件覆盖为：

| 指标 | 结果 |
|---|---|
| group_count | 917 |
| coverage>=1 | 917 |
| coverage>=2 | 75 |
| coverage>=3 | 8 |
| coverage>=4 | 0 |
| coverage>=5 | 0 |
| max_receivers_per_strict_key | 3 |

解释：当前ADV3B02特征切分不支持4/5接收机严格同事件协同。`receiver_domain_ranked`可以覆盖`1..5`作为接收机域代理协同诊断，但不能声明为真实同事件多星协同。

## 本地OPUC结果

### receiver_domain_ranked，覆盖1..5接收机

配置：`support_envelope_evt`，`old_protected_unknown_confirm_cvs`，`strict_event_query_preserve`，`class_conformal_enabled`，`class_shell_unknown_risk_enabled`，`virtual_unknown_risk_enabled`，`evidence_packet_bytes=96`。

| 输出 | k | old_acc | seen_new_acc | unknown_reject_rate | known_coverage | defer_rate | bytes/event | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `receiver_domain_ranked_all.json` | 1 | 0.1551 | 0.0000 | 0.9667 | 0.1255 | 0.0586 | 96 | unknown好但old崩 |
| `receiver_domain_ranked_all.json` | 2 | 0.0855 | 0.0588 | 0.8043 | 0.1182 | 0.1446 | 192 | old崩 |
| `receiver_domain_ranked_all.json` | 3 | 0.1750 | 0.0750 | 0.7750 | 0.1812 | 0.1250 | 288 | old崩 |
| `receiver_domain_ranked_all.json` | 4 | 0.2045 | 0.1724 | 0.7647 | 0.2308 | 0.0596 | 384 | old崩 |
| `receiver_domain_ranked_all.json` | 5 | 0.1887 | 0.2500 | 0.9500 | 0.2055 | 0.0645 | 480 | old崩 |
| `receiver_domain_ranked_all_pv02_top1.json` | 1 | 0.2246 | 0.0000 | 0.9667 | 0.1903 | 0.0033 | 96 | old仍不足 |
| `receiver_domain_ranked_all_pv02_top1.json` | 2 | 0.2566 | 0.4314 | 0.6304 | 0.4778 | 0.1365 | 192 | old不足 |
| `receiver_domain_ranked_all_pv02_top1.json` | 3 | 0.3500 | 0.3000 | 0.6250 | 0.4625 | 0.1250 | 288 | old不足 |
| `receiver_domain_ranked_all_pv02_top1.json` | 4 | 0.4659 | 0.4483 | 0.6176 | 0.5897 | 0.0596 | 384 | old不足 |
| `receiver_domain_ranked_all_pv02_top1.json` | 5 | 0.5283 | 0.5500 | 0.8500 | 0.5616 | 0.0645 | 480 | old不足 |

### strict_event_key，实际只支持1..3接收机

| 输出 | k | old_acc | seen_new_acc | unknown_reject_rate | known_coverage | bytes/event | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `strict_event_min2_123.json` | 1 | 0.1111 | 0.0000 | 0.8485 | 0.0476 | 96 | old崩 |
| `strict_event_min2_123.json` | 2 | 0.2222 | 0.0909 | 0.8788 | 0.1429 | 192 | old崩 |
| `strict_event_min2_123.json` | 3 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 288 | old/new全崩 |

## 当前解释

OPUC-qKNN验证了“多源unknown确认”可以显著提高unknown拒识，但当前ADV3B02特征空间中unknown与old邻域重叠严重；一旦强行拒识unknown，就会大量误杀old/seen-new。该结果不能满足“unknown优先且old准确性不能下降”，只能作为下一步算法约束证据。

当前推荐路线：

1. 保留OPUC作为诊断融合层，但不作为部署路线。
2. 下一步需要在特征层而非阈值层处理unknown：source-old prototype shrinkage、Mahalanobis/EVT per-class tail、virtual future class空域约束、few-shot cache adapter蒸馏。
3. 实验矩阵必须分开报告`receiver_domain_ranked`和`strict_event_key`，并把strict事件的4/5接收机缺失作为数据切分限制。
4. 任何候选必须同一行同时满足old不下降、known_coverage不下降、unknown FAR下降；否则标诊断。

## N607计划

远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`  
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`  
远端输出：`runs/phase2_adv3b02_opuc_qknn_20260704/`

待同步文件：

| 本地 | 远端 |
|---|---|
| `code\evaluation\collaborative_open_set_qknn_eval.py` | `code/evaluation/collaborative_open_set_qknn_eval.py` |
| `code\scripts\phase2_collaborative_open_set_qknn_eval.py` | `code/scripts/phase2_collaborative_open_set_qknn_eval.py` |
| `code\scripts\phase2_event_alignment_audit.py` | `code/scripts/phase2_event_alignment_audit.py` |
| `code\tests\test_collaborative_open_set_qknn_eval.py` | `code/tests/test_collaborative_open_set_qknn_eval.py` |
| `code\tests\test_phase2_collaborative_open_set_qknn_eval.py` | `code/tests/test_phase2_collaborative_open_set_qknn_eval.py` |

待运行远端命令包括`py_compile`、目标单元测试、event audit、`receiver_domain_ranked`全量1..5和`strict_event_key`实际1..3诊断。运行前必须执行N607 preflight，运行后检查SSH/SCP无残留。

## N607验证结果

| 项目 | 结果 |
|---|---|
| preflight | PASS，直连`N607`，项目根可见，8张RTX3090均为10MiB |
| 远端Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python3.10.19 |
| 远端目录 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_opuc_qknn_20260704` |
| 远端测试 | `py_compile` PASS；`test_collaborative_open_set_qknn_eval.py` 67 tests OK；`test_phase2_collaborative_open_set_qknn_eval.py` 52 tests OK |
| 远端GPU | 运行结束后8张GPU均为10MiB |
| SSH清理 | preflight、scp、远端运行和拉回后，本地`ssh.exe`与到`172.31.111.215:22`/`172.31.105.18:22`的ESTABLISHED连接均为空 |

远端输出已拉回：

| 远端结果 | 本地副本 |
|---|---|
| `runs/phase2_adv3b02_opuc_qknn_20260704/event_alignment_query_audit.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_opuc_qknn_20260704\remote\event_alignment_query_audit.json` |
| `runs/phase2_adv3b02_opuc_qknn_20260704/receiver_domain_ranked_all_pv02_top1.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_opuc_qknn_20260704\remote\receiver_domain_ranked_all_pv02_top1.json` |
| `runs/phase2_adv3b02_opuc_qknn_20260704/strict_event_min2_123.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_opuc_qknn_20260704\remote\strict_event_min2_123.json` |

### N607 receiver_domain_ranked全量1..5

| k | old_acc | seen_new_acc | unknown_reject_rate | known_coverage | bytes/event | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.2246 | 0.0000 | 0.9667 | 0.1903 | 96 | old严重下降 |
| 2 | 0.2566 | 0.4314 | 0.6304 | 0.4778 | 192 | old严重下降 |
| 3 | 0.3500 | 0.3000 | 0.6250 | 0.4625 | 288 | old严重下降 |
| 4 | 0.4659 | 0.4483 | 0.6176 | 0.5897 | 384 | old严重下降 |
| 5 | 0.5283 | 0.5500 | 0.8500 | 0.5616 | 480 | unknown改善但old未保护 |

### N607 strict_event_key实际1..3

| k | old_acc | seen_new_acc | unknown_reject_rate | known_coverage | bytes/event | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1111 | 0.0000 | 0.8485 | 0.0952 | 96 | old崩 |
| 2 | 0.5556 | 0.3030 | 0.6364 | 0.5476 | 192 | old不足 |
| 3 | 0.0000 | 0.3333 | 1.0000 | 0.3333 | 288 | old崩 |

远端同事件审计仍为`coverage_by_min_receivers={"1":917,"2":75,"3":8,"4":0,"5":0}`，证明当前特征切分不支持4/5接收机严格同事件协同。

## 最终判定

`old_protected_unknown_confirm_cvs`作为模块和诊断工具已经实现并通过本地/N607测试，但当前ADV3B02特征上没有达成用户目标。unknown拒识率最高可到0.9667/0.85，但old准确率从已知known route基线约0.8663降到0.2246/0.5283，违反“旧类准确性不能下降”。该路线不得作为部署成功或论文成功结论。

下一步应从特征和校准机制解决，而不是继续调融合阈值：引入source-old prototype shrinkage、per-class Mahalanobis/EVT尾部、virtual future class空域约束和cache adapter蒸馏；再用support/proxy冻结参数后，对unknown query一次性盲评。
