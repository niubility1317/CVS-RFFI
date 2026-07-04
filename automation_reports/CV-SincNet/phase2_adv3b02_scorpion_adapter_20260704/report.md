# phase2_adv3b02_scorpion_adapter_20260704

## 基本信息

| 项目 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_scorpion_adapter_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 将SCORPION-CVS推进到feature/adapter层：冻结ADV3B02`z_id`，只用target old/seen-new support拟合虚拟负样本边界，再做协同unknown-first审计 |
| 底座模型 | `ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| 输入feature | `E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz` |
| 输入规模 | 6400条`z_id`特征，维度160 |
| 输出目录 | `E:\type10-7\local_artifacts\phase2_adv3b02_scorpion_adapter_20260704\` |
| 协同范围 | `k=1..5` receiver-domain ranked ensemble |
| 结论边界 | DIAGNOSTIC_ONLY；`Y_unknown`仅参与query评估，不参与adapter或阈值拟合 |

## 本地改动

| 文件 | 目的 |
|---|---|
| `E:\type10-7\code\scripts\phase2_scorpion_adapter_eval.py` | 新增SCORPION-CVS support-only adapter入口：调用虚拟负样本ridge boundary，再用SCORPION事件门控输出`k=1..N`指标 |
| `E:\type10-7\code\tests\test_phase2_scorpion_adapter_eval.py` | 覆盖unknown query不参与拟合、输出JSON/CSV、协同数量`1..N` |

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_scorpion_adapter_eval.py code\tests\test_phase2_scorpion_adapter_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_scorpion_adapter_eval.py -q` | PASS，2 passed |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_virtual_negative_adapter_eval.py code\tests\test_phase2_virtual_negative_adapter_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_virtual_negative_adapter_eval.py -q` | PASS，3 passed |

## 算法配置

该入口保持ADV3B02主干冻结。每个target receiver只用`Y_old`和`Y_new`的K-shot support拟合：

```text
class head: multiclass ridge on target old + seen-new support
known boundary: binary ridge on support positives + generated virtual negatives
virtual negatives: shell/midpoint/mix from support geometry
event decision: SCORPION-CVS unknown-first gate + old retention shield
```

默认配置：

| 参数 | 值 |
|---|---|
| `k_shot` | 8 |
| `query_per_class` | 20 |
| `virtual_negative_policy` | `shell_mix` |
| `support_threshold_quantile` | 0.05 |
| `risk_components` | `virtual_unknown_risk:0.45,class_negative_risk:0.25,score_risk:0.20,margin_risk:0.10` |
| `unknown_gate` | 0.52 |
| `old_shield_gate` | 0.68 |
| evidence packet | 112B/receiver-event |
| adapter state | 58130B |

## 本地结果

默认adapter+SCORPION结果：

| k | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | bytes/event | latency_ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6257 | 0.4500 | 0.5167 | 0.4500 | 0.0000 | 1.0000 | 0.0000 | 112.0 | 0.0450 |
| 2 | 0.6578 | 0.4250 | 0.6000 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 203.2 | 0.0450 |
| 3 | 0.7219 | 0.5250 | 0.6333 | 0.5500 | 0.0000 | 1.0000 | 0.0000 | 276.2 | 0.0450 |
| 4 | 0.7754 | 0.5250 | 0.6833 | 0.6000 | 0.0000 | 1.0000 | 0.0000 | 330.9 | 0.0450 |
| 5 | 0.7914 | 0.5250 | 0.7167 | 0.6000 | 0.0000 | 1.0000 | 0.0000 | 364.8 | 0.0450 |

虚拟负样本网格显示，`shell`、`midpoint`、`mix`、`shell_mix`在`support_threshold_quantile={0.05,0.15,0.30,0.50}`和`shell_scale={1.5,2.5,4.0}`下均未产生真实unknown拒识。最佳known点为`midpoint,q=0.50,k=5`，`old_acc=0.83,min_old=0.62,seen_new_acc=0.65,min_seen=0.50,unknown_reject=0.00`。

## 分布审计

`midpoint,q=0.50`的evidence分布说明失败原因不是门控太宽，而是虚拟负样本边界没有覆盖真实unknown方向。

| role | field | min | p10 | median | p90 | max |
|---|---|---:|---:|---:|---:|---:|
| old | known_score | 0.6589 | 0.8656 | 0.9873 | 0.9925 | 0.9985 |
| seen_new | known_score | 0.6653 | 0.8666 | 0.9829 | 0.9921 | 0.9988 |
| unknown | known_score | 0.7158 | 0.8146 | 0.9845 | 0.9943 | 0.9992 |
| old | boundary_known_probability | 0.8933 | 0.9685 | 0.9775 | 0.9877 | 0.9984 |
| seen_new | boundary_known_probability | 0.9133 | 0.9652 | 0.9771 | 0.9881 | 0.9976 |
| unknown | boundary_known_probability | 0.8764 | 0.9686 | 0.9809 | 0.9938 | 0.9996 |
| old | unknown_risk | 0.0016 | 0.0133 | 0.0242 | 0.1225 | 0.3336 |
| seen_new | unknown_risk | 0.0024 | 0.0127 | 0.0245 | 0.1220 | 0.3261 |
| unknown | unknown_risk | 0.0004 | 0.0083 | 0.0240 | 0.1749 | 0.2749 |

真实unknown在adapter边界上比old/seen-new更像known：`unknown`的median known_score为0.9845，median boundary known probability为0.9809。该结果否定了“只用support几何生成虚拟负样本即可解决unknown拒识”的假设。

## 解释

SCORPION-CVS的feature/adapter最小版没有达成目标。它把协同数量、资源和延迟统计推进到可复现入口，但真实unknown仍被高置信吸入已知空间。下一步需要引入协议允许的source/proxy outlier或source阶段open-set训练约束，而不是继续扩大support-generated negatives。若没有合法proxy unknown，下一步应先做oracle holdout诊断：用一小部分`Y_unknown`只作NON_DEPLOYMENT_DIAGNOSTIC，测量当前`z_id`是否存在可线性分开的unknown方向；若oracle也失败，必须回到ADV3B02地面训练阶段加入open-set/energy margin。

## N607计划

远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`  
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`  
远端输出：`runs/phase2_adv3b02_scorpion_adapter_20260704/`

待同步文件：

| 本地 | 远端 |
|---|---|
| `code\scripts\phase2_scorpion_adapter_eval.py` | `code/scripts/phase2_scorpion_adapter_eval.py` |
| `code\tests\test_phase2_scorpion_adapter_eval.py` | `code/tests/test_phase2_scorpion_adapter_eval.py` |

待运行远端命令：`py_compile`、标准库`unittest`、默认adapter+SCORPION评估。运行前执行N607 preflight，运行后拉回JSON/CSV并检查SSH清理。

## N607验证结果

| 项目 | 结果 |
|---|---|
| preflight | PASS；直连`N607`，项目根目录可见 |
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 远端工作目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端输入feature | `runs/phase2_adv3b02_scorpion_adapter_20260704/input/features.npz` |
| 远端输出 | `runs/phase2_adv3b02_scorpion_adapter_20260704/scorpion_adapter_default.json` |
| 拉回目录 | `E:\type10-7\local_artifacts\phase2_adv3b02_scorpion_adapter_20260704\remote\` |
| GPU选择 | GPU0；运行后8张GPU均为10MiB |
| 远端测试 | `py_compile` PASS；标准库`unittest`2 tests OK |
| SSH清理 | SCP和SSH后本地`ssh.exe`为空；到`172.31.111.215:22`和`172.31.105.18:22`的ESTABLISHED连接为空 |

远端第一次运行时`runs/phase2_adv3b02_features/features.npz`不存在；本轮没有重跑特征导出，而是把本地已拉回的5.9MB feature工件同步到本次run目录`input/features.npz`。该文件是ADV3B02`z_id`派生特征，不是原始数据集或checkpoint。

远端执行命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
OUT=runs/phase2_adv3b02_scorpion_adapter_20260704
FEAT=$OUT/input/features.npz
CUDA_VISIBLE_DEVICES=0 $PY -m py_compile code/scripts/phase2_scorpion_adapter_eval.py code/tests/test_phase2_scorpion_adapter_eval.py
PYTHONPATH=code:code/scripts CUDA_VISIBLE_DEVICES=0 $PY code/tests/test_phase2_scorpion_adapter_eval.py
CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_scorpion_adapter_eval.py --feature_npz $FEAT --output_json $OUT/scorpion_adapter_default.json --output_rows_csv $OUT/scorpion_adapter_default_rows.csv --output_evidence_csv $OUT/scorpion_adapter_default_evidence.csv --collab_counts all --k_shot 8 --query_per_class 20
```

远端结果与本地一致：

| k | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | bytes/event | latency_ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6257 | 0.4500 | 0.5167 | 0.4500 | 0.0000 | 1.0000 | 0.0000 | 112.0 | 0.0621 |
| 2 | 0.6578 | 0.4250 | 0.6000 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 203.2 | 0.0621 |
| 3 | 0.7219 | 0.5250 | 0.6333 | 0.5500 | 0.0000 | 1.0000 | 0.0000 | 276.2 | 0.0621 |
| 4 | 0.7754 | 0.5250 | 0.6833 | 0.6000 | 0.0000 | 1.0000 | 0.0000 | 330.9 | 0.0621 |
| 5 | 0.7914 | 0.5250 | 0.7167 | 0.6000 | 0.0000 | 1.0000 | 0.0000 | 364.8 | 0.0621 |

远端结论：support-only虚拟负样本adapter没有解决unknown拒识。下一步应先补合法proxy unknown或做明确标注为`NON_DEPLOYMENT_DIAGNOSTIC`的oracle unknown holdout上限诊断，再决定是否回到ADV3B02地面训练阶段加入open-set/energy margin。
