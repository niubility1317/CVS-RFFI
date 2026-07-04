# phase2_adv3b02_scorpion_cvs_minimal_20260704

## 基本信息

| 项目 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_scorpion_cvs_minimal_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 将SCORPION-CVS从设计推进到最小可运行evidence层诊断，检查unknown-first门控、多接收机证据融合和old retention shield能否改善ADV3B02+qknn8 |
| 底座模型 | `ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| 输入evidence | `E:\type10-7\remote_artifacts\phase2_adv3b02_collab_open_set_qknn_full_20260703\collab_open_set_qknn_candidate_set_cvs_support_calibrated_event098_adv3b02_evidence.csv` |
| 输出目录 | `E:\type10-7\local_artifacts\phase2_adv3b02_scorpion_cvs_minimal_20260704\` |
| 协同范围 | `k=1..5` receiver-domain ranked ensemble；strict同事件证据不足时不得写成全量同事件卫星群 |
| 结论边界 | DIAGNOSTIC_ONLY；该最小版不训练adapter，不使用unknown query拟合阈值 |

## 本地改动

| 文件 | 目的 |
|---|---|
| `E:\type10-7\code\scripts\phase2_scorpion_cvs_eval.py` | 新增SCORPION-CVS最小evidence层评估：风险融合、接收机可靠度加权、unknown-first决策、old shield、资源统计 |
| `E:\type10-7\code\tests\test_phase2_scorpion_cvs_eval.py` | 覆盖unknown拒识、old retention shield、CLI JSON/CSV输出 |

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_scorpion_cvs_eval.py code\tests\test_phase2_scorpion_cvs_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_scorpion_cvs_eval.py -q` | PASS，3 passed |

## 算法定义

该最小版只在已有qknn8 evidence上运行。每个接收机事件计算：

```text
r_r(x)=Σ_i w_i risk_i(x)/Σ_i w_i
q_r=(reliability)(receiver_class_reliability)(receiver_deployment_prior)(support_density factor)(1-0.5r_r)
```

事件级unknown分数：

```text
U(x)=0.72 mean_q(r_r)+0.18 high_risk_fraction+0.10 disagreement
```

决策规则：

```text
if U(x)<theta_U and majority local evidence passes: accept argmax_c L_c
elif predicted c in Y_old and U(x)<theta_old and old shield passes: accept c
else reject/defer
```

其中`Y_old`是协议已知旧类集合；`Y_unknown`只用于query评估，不参与阈值拟合。

## 本地结果

默认门控`unknown_gate=0.52, old_shield_gate=0.68`：

| k | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | bytes/event |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8571 | 0.6500 | 0.6500 | 0.6500 | 0.0333 | 0.9667 | 0.0289 | 96.0 |
| 2 | 0.8297 | 0.3000 | 0.7500 | 0.6750 | 0.0333 | 0.9667 | 0.0207 | 176.1 |
| 3 | 0.8407 | 0.3500 | 0.7500 | 0.6750 | 0.1000 | 0.9000 | 0.0248 | 239.7 |
| 4 | 0.8352 | 0.3000 | 0.7500 | 0.6750 | 0.0833 | 0.9167 | 0.0248 | 286.7 |
| 5 | 0.8407 | 0.3500 | 0.7500 | 0.6750 | 0.1333 | 0.8667 | 0.0289 | 317.9 |

门控曲线，以下只列`k=5`：

| unknown_gate | old_shield_gate | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.35 | 0.6813 | 0.2500 | 0.5000 | 0.4500 | 0.7667 | 0.2333 | 0.2562 |
| 0.30 | 0.45 | 0.7527 | 0.3000 | 0.6833 | 0.6000 | 0.7000 | 0.3000 | 0.1322 |
| 0.40 | 0.55 | 0.7967 | 0.3000 | 0.7333 | 0.6500 | 0.4667 | 0.5333 | 0.0785 |
| 0.52 | 0.68 | 0.8407 | 0.3500 | 0.7500 | 0.6750 | 0.1333 | 0.8667 | 0.0289 |
| 0.65 | 0.75 | 0.8462 | 0.3500 | 0.7500 | 0.6750 | 0.0667 | 0.9333 | 0.0248 |

## 解释

SCORPION-CVS最小evidence层没有达成目标。严格unknown门控可把`unknown_reject`提高到0.7667，但`old_acc`降到0.6813、`seen_new_acc`降到0.5000；宽门控保留old约0.84，但unknown几乎全部误接收。这个结果与上一轮goal feasibility诊断一致：当前ADV3B02+qknn8 evidence层没有足够的known/unknown几何间隔。

下一步不能继续只调事件门控。应实现SCORPION-CVS的训练/适配版：在feature或adapter层加入source/proxy unknown energy margin、EVT尾部校准、旧类收缩原型和old retention rollback，然后再用本脚本作为row-level审计入口。

## N607计划

远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`  
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`  
远端输出：`runs/phase2_adv3b02_scorpion_cvs_minimal_20260704/`

待同步文件：

| 本地 | 远端 |
|---|---|
| `code\scripts\phase2_scorpion_cvs_eval.py` | `code/scripts/phase2_scorpion_cvs_eval.py` |
| `code\tests\test_phase2_scorpion_cvs_eval.py` | `code/tests/test_phase2_scorpion_cvs_eval.py` |

待运行远端命令：`py_compile`、目标pytest、默认门控评估和门控曲线评估。运行前执行N607 preflight，运行后拉回JSON/CSV并检查SSH清理。

## N607验证结果

| 项目 | 结果 |
|---|---|
| preflight | PASS；直连`N607`，项目根目录可见 |
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 远端工作目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端输出 | `runs/phase2_adv3b02_scorpion_cvs_minimal_20260704/` |
| 拉回目录 | `E:\type10-7\local_artifacts\phase2_adv3b02_scorpion_cvs_minimal_20260704\remote\` |
| GPU选择 | GPU0；运行后8张GPU均为10MiB |
| 远端测试 | `py_compile` PASS；`pytest`模块缺失，改用标准库`unittest`入口，3 tests OK |
| SSH清理 | SCP和SSH后本地`ssh.exe`为空；到`172.31.111.215:22`和`172.31.105.18:22`的ESTABLISHED连接为空 |

远端执行命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
OUT=runs/phase2_adv3b02_scorpion_cvs_minimal_20260704
EVID=runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_support_calibrated_event098_adv3b02_evidence.csv
mkdir -p $OUT
CUDA_VISIBLE_DEVICES=0 $PY -m py_compile code/scripts/phase2_scorpion_cvs_eval.py code/tests/test_phase2_scorpion_cvs_eval.py
PYTHONPATH=code:code/scripts CUDA_VISIBLE_DEVICES=0 $PY code/tests/test_phase2_scorpion_cvs_eval.py
CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_scorpion_cvs_eval.py --evidence_csv $EVID --output_json $OUT/scorpion_cvs_event098.json --output_rows_csv $OUT/scorpion_cvs_event098_rows.csv --collab_counts all --evidence_packet_bytes 96
```

远端门控曲线与本地一致，以下列`k=5`：

| unknown_gate | old_shield_gate | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.35 | 0.6813 | 0.2500 | 0.5000 | 0.4500 | 0.7667 | 0.2333 | 0.2562 |
| 0.30 | 0.45 | 0.7527 | 0.3000 | 0.6833 | 0.6000 | 0.7000 | 0.3000 | 0.1322 |
| 0.40 | 0.55 | 0.7967 | 0.3000 | 0.7333 | 0.6500 | 0.4667 | 0.5333 | 0.0785 |
| 0.52 | 0.68 | 0.8407 | 0.3500 | 0.7500 | 0.6750 | 0.1333 | 0.8667 | 0.0289 |
| 0.65 | 0.75 | 0.8462 | 0.3500 | 0.7500 | 0.6750 | 0.0667 | 0.9333 | 0.0248 |

远端结论：最小SCORPION-CVS evidence层仍未解决目标。该结果把下一步实验范围缩小到feature/adapter层训练约束，而不是继续在事件融合层搜索。
