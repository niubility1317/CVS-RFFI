# phase2_adv3b02_oracle_unknown_holdout_20260704

## 基本信息

| 项目 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_oracle_unknown_holdout_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 诊断ADV3B02`z_id`是否存在可学习的known-vs-unknown方向：用少量`Y_unknown`作为oracle负样本训练边界，再在held-out unknown query上评估 |
| 底座模型 | `ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| 输入feature | `E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz` |
| 输出目录 | `E:\type10-7\local_artifacts\phase2_adv3b02_oracle_unknown_holdout_20260704\` |
| 协同范围 | `collab_count=1..5` receiver-domain ranked ensemble；区别于`K-shot=8` |
| 结论边界 | `NON_DEPLOYMENT_DIAGNOSTIC`；使用了带标签target unknown support，不可作为部署门控或Stage2-C成功 |

## 本地改动

| 文件 | 目的 |
|---|---|
| `E:\type10-7\code\scripts\phase2_oracle_unknown_holdout_eval.py` | 新增oracle unknown holdout上限诊断：known support + oracle unknown support拟合边界，held-out unknown query评估 |
| `E:\type10-7\code\tests\test_phase2_oracle_unknown_holdout_eval.py` | 验证输出明确标记`non_deployment_diagnostic`、`labeled_unknown_support_used_for_boundary_fit`、`oracle_unknown_holdout`和不可部署原因，并拒绝非正shot参数 |

版本状态：`E:\type10-7`不是Git仓库；代码快照写入`E:\type10-7\code\snapshots\phase2_adv3b02_oracle_unknown_holdout_20260704\`。Git-backed镜像为`E:\type10-7\github_publish\CVS-RFFI-repo`，同步文件已在镜像路径通过验证并准备提交。

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_oracle_unknown_holdout_eval.py code\tests\test_phase2_oracle_unknown_holdout_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_oracle_unknown_holdout_eval.py -q` | PASS，2 passed；根目录pytest cache写入被Windows拒绝，不影响测试结果 |

## 算法配置

每个target receiver使用：

```text
known positives: target old support + seen-new support
oracle negatives: oracle_unknown_shot labeled Y_unknown samples
evaluation unknown: held-out Y_unknown query rows
```

默认配置：

| 参数 | 值 |
|---|---|
| `k_shot` | 8 |
| `oracle_unknown_shot` | 4 |
| `query_per_class` | 20 |
| `support_threshold_quantile` | 0.05 |
| `risk_components` | `virtual_unknown_risk:0.55,class_negative_risk:0.25,score_risk:0.10,margin_risk:0.10` |
| `unknown_gate` | 0.52 |
| `old_shield_gate` | 0.68 |
| evidence packet | 128B/receiver-event |
| adapter state | 57960B |

## 本地结果

默认oracle holdout+SCORPION结果：

| collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | bytes/event | latency_ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7407 | 0.5556 | 0.4000 | 0.3500 | 0.0000 | 1.0000 | 0.0000 | 128.0 | 0.0381 |
| 2 | 0.7831 | 0.5833 | 0.3833 | 0.3500 | 0.0000 | 1.0000 | 0.0000 | 231.1 | 0.0381 |
| 3 | 0.8042 | 0.6389 | 0.4167 | 0.3000 | 0.0000 | 1.0000 | 0.0000 | 314.0 | 0.0381 |
| 4 | 0.8360 | 0.6944 | 0.4833 | 0.3500 | 0.0000 | 1.0000 | 0.0000 | 376.5 | 0.0381 |
| 5 | 0.8466 | 0.7222 | 0.5167 | 0.3500 | 0.0000 | 1.0000 | 0.0000 | 414.2 | 0.0381 |

更严格门控曲线，以下列`collab_count=5`且`old_shield_gate=0.10`：

| unknown_gate | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.7249 | 0.3000 | 0.0500 | 0.0500 | 0.6333 | 0.3667 | 0.2771 |
| 0.05 | 0.7249 | 0.3000 | 0.2000 | 0.1750 | 0.6333 | 0.3667 | 0.2410 |
| 0.10 | 0.7725 | 0.5833 | 0.5167 | 0.3500 | 0.5833 | 0.4167 | 0.0803 |
| 0.20 | 0.8095 | 0.5833 | 0.5167 | 0.3500 | 0.1333 | 0.8667 | 0.0241 |
| 0.30 | 0.8095 | 0.5833 | 0.5167 | 0.3500 | 0.1333 | 0.8667 | 0.0241 |
| 0.52 | 0.8095 | 0.5833 | 0.5167 | 0.3500 | 0.1333 | 0.8667 | 0.0241 |

## 分布审计

oracle unknown支持让真实unknown的风险分布略有抬升，但没有形成可部署间隔。

| role | field | min | p10 | median | p90 | max |
|---|---|---:|---:|---:|---:|---:|
| old | known_score | 0.6118 | 0.8650 | 0.9843 | 0.9915 | 0.9984 |
| seen_new | known_score | 0.5659 | 0.8723 | 0.9856 | 0.9924 | 0.9986 |
| unknown | known_score | 0.5673 | 0.7485 | 0.9617 | 0.9899 | 0.9965 |
| old | boundary_known_probability | 0.7088 | 0.9480 | 0.9771 | 0.9847 | 0.9967 |
| seen_new | boundary_known_probability | 0.7697 | 0.9587 | 0.9812 | 0.9865 | 0.9973 |
| unknown | boundary_known_probability | 0.3598 | 0.8227 | 0.9571 | 0.9827 | 0.9936 |
| old | unknown_risk | 0.0033 | 0.0158 | 0.0250 | 0.0857 | 0.2988 |
| seen_new | unknown_risk | 0.0027 | 0.0138 | 0.0193 | 0.0604 | 0.3767 |
| unknown | unknown_risk | 0.0064 | 0.0182 | 0.0496 | 0.2560 | 0.6402 |

## 解释

该oracle上限诊断仍然远低于目标。即使允许每个target receiver使用4个带标签unknown样本训练二分类边界，`collab_count=5`默认门控下`unknown_reject=0.0000`；把门控压到`unknown_gate=0.02`只能得到`unknown_reject=0.6333`，同时`old_acc=0.7249`、`min_old=0.3000`、`seen_new_acc=0.0500`、`known_defer=0.2771`。该严格门控违反OLD80_FIRST旧类优先边界，不能作为下一阶段候选。这说明当前ADV3B02`z_id`中的真实unknown与old/seen-new仍深度重叠。

下一步应回到地面或feature训练阶段，而不是继续调少样本部署门控。最小可执行路线是：在ADV3B02后续训练中加入source/proxy outlier exposure、energy margin或reciprocal-point/open-space约束，并用本报告的oracle holdout结果作为“现有表示不可分”的负证据。

## N607执行

远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
远端输出：`runs/phase2_adv3b02_oracle_unknown_holdout_20260704/`
拉回目录：`E:\type10-7\local_artifacts\phase2_adv3b02_oracle_unknown_holdout_20260704\remote\`

N607只读preflight通过：直接`N607`目标、项目根目录和8张RTX 3090均可见。运行前GPU占用均为`10MiB`，未发现本用户训练进程；选择低占用GPU0执行诊断。运行结束后`nvidia-smi`显示8张GPU仍为`10MiB`。

同步文件：

| 本地 | 远端 |
|---|---|
| `code\scripts\phase2_oracle_unknown_holdout_eval.py` | `code/scripts/phase2_oracle_unknown_holdout_eval.py` |
| `code\tests\test_phase2_oracle_unknown_holdout_eval.py` | `code/tests/test_phase2_oracle_unknown_holdout_eval.py` |
| `code\scripts\phase2_scorpion_cvs_eval.py` | `code/scripts/phase2_scorpion_cvs_eval.py` |
| `remote_artifacts\phase2_adv3b02_features\features.npz` | `runs/phase2_adv3b02_oracle_unknown_holdout_20260704/input/features.npz` |

远端验证：

| 命令 | 结果 |
|---|---|
| `py_compile` oracle脚本和测试 | PASS |
| `PYTHONPATH=code:code/scripts ... code/tests/test_phase2_oracle_unknown_holdout_eval.py` | PASS，2 tests OK；负值shot测试产生预期argparse错误文本 |
| oracle holdout默认评估 | PASS，输出JSON/CSV |
| `unknown_gate=0.02,0.05,0.10,0.20,0.30,0.52`门控曲线 | PASS，输出JSON/CSV |

远端默认oracle holdout+SCORPION结果：

| collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | bytes/event | latency_ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7407 | 0.5556 | 0.4000 | 0.3500 | 0.0000 | 1.0000 | 0.0000 | 128.0 | 0.0672 |
| 2 | 0.7831 | 0.5833 | 0.3833 | 0.3500 | 0.0000 | 1.0000 | 0.0000 | 231.1 | 0.0672 |
| 3 | 0.8042 | 0.6389 | 0.4167 | 0.3000 | 0.0000 | 1.0000 | 0.0000 | 314.0 | 0.0672 |
| 4 | 0.8360 | 0.6944 | 0.4833 | 0.3500 | 0.0000 | 1.0000 | 0.0000 | 376.5 | 0.0672 |
| 5 | 0.8466 | 0.7222 | 0.5167 | 0.3500 | 0.0000 | 1.0000 | 0.0000 | 414.2 | 0.0672 |

远端严格门控曲线，以下列`collab_count=5`且`old_shield_gate=0.10`：

| unknown_gate | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | latency_ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.7249 | 0.3000 | 0.0500 | 0.0500 | 0.6333 | 0.3667 | 0.2771 | 0.0672 |
| 0.05 | 0.7249 | 0.3000 | 0.2000 | 0.1750 | 0.6333 | 0.3667 | 0.2410 | 0.0672 |
| 0.10 | 0.7725 | 0.5833 | 0.5167 | 0.3500 | 0.5833 | 0.4167 | 0.0803 | 0.0672 |
| 0.20 | 0.8095 | 0.5833 | 0.5167 | 0.3500 | 0.1333 | 0.8667 | 0.0241 | 0.0672 |
| 0.30 | 0.8095 | 0.5833 | 0.5167 | 0.3500 | 0.1333 | 0.8667 | 0.0241 | 0.0672 |
| 0.52 | 0.8095 | 0.5833 | 0.5167 | 0.3500 | 0.1333 | 0.8667 | 0.0241 | 0.0672 |

远端JSON字段已修正为`labeled_unknown_support_used_for_boundary_fit=True`、`unknown_query_used_for_threshold_fit=False`、`oracle_metadata.unknown_query_eval_only=True`。该oracle诊断使用带标签`Y_unknown` support拟合known-vs-unknown边界；held-out unknown query只用于评估，但该诊断仍不满足部署权限。不得将本结果作为Stage2-C部署门控、论文成功声明或卫星群上线证据。

SSH/SCP清理：preflight、进程检查、同步、运行和结果拉回之后均检查本地`ssh.exe`与到`172.31.111.215:22`/`172.31.105.18:22`的`ESTABLISHED`连接；结果均为`none`。
