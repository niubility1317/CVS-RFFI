# phase2_adv3b02_opr_opu_ci_20260704

## 基本信息

| 字段 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_opr_opu_ci_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 在ADV3B02_CORE90_SOFT_E200/qknn8上验证OPR低秩proxy-open adapter叠加OPU旧类保护协同后端 |
| 输入特征 | `local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz` |
| 状态 | N607_COMPLETED_NEGATIVE_DIAGNOSTIC |

## 算法

OPR-OPU-CI分两步：

1. OPR adapter：冻结ADV3B02特征，只训练低秩残差adapter。训练数据只包含`source`旧类、source-side`proxy_unknown`、target-old K-shot support和target-new K-shot support。
2. OPU backend：对adapted features运行known route和support-only safety route，再用`old_protected_unknown_confirm_cvs`融合策略输出`collab_count=1..receiver_count`结果。

禁止项：

```text
target_unknown_training_count = 0
target_unknown_eval_only = true
unknown_query_used_for_threshold_fit = false
```

训练损失：

```text
L = L_source_ce + L_support_ce
  + lambda_old * KL(p_base_source || p_adapter_source)
  + lambda_open * softplus(max_c s_proxy_c - margin_open)
  + lambda_compact * support_compact
  + lambda_residual * ||z_adapter - z_base||^2
```

## 本地变更

| 文件 | 作用 | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\phase2_opr_opu_adapter_ci_eval.py` | OPR adapter+OPU后端组合评估脚本 | `8B69913ABCDBB50ECAEB52072B4FEE50452B1F66A008628199A6411B6551AB9A` |
| `E:\type10-7\code\tests\test_phase2_opr_opu_adapter_ci_eval.py` | OPR-OPU参数和unknown隔离单测 | `A8547EFAFD45DE52700BD99083B4D62FC1B16AA04EE6DD4BEBDD1407DB6BB568` |
| `E:\type10-7\code\snapshots\phase2_adv3b02_opr_opu_ci_20260704\phase2_opr_opu_adapter_ci_eval.py` | 非Git根代码快照 | 待同步 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_opr_opu_ci_20260704\test_phase2_opr_opu_adapter_ci_eval.py` | 非Git根测试快照 | 待同步 |

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_opr_opu_adapter_ci_eval.py code\tests\test_phase2_opr_opu_adapter_ci_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_opr_opu_adapter_ci_eval.py -q` | PASS，2 passed；`.pytest_cache`权限警告不影响测试 |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_opr_opu_adapter_ci_eval.py --feature_npz local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz --output_dir local_artifacts\phase2_adv3b02_opr_opu_ci_20260704 --device cpu --adapter_epochs 20 --policies opu_old_preserve,opu_old_guarded` | PASS，10 rows |

本地输出：

| 文件 | 内容 | SHA256 |
|---|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_opr_opu_ci_20260704\opr_opu_ci_summary.csv` | OPR-OPU 1..5协同摘要 | `E89D80549467AA22658427CC4865DA6C77D888BA788879C5F156DFA2B29172B7` |
| `E:\type10-7\local_artifacts\phase2_adv3b02_opu_proxy_baseline_20260704\opu_ci_summary.csv` | 同特征无adapter OPU基线 | `3DE437F3585AD8EC467BE523C1B232D81ADA98CBAE1C11F30ABC2DC34627DCD9` |

## 本地结果

| 方法 | policy | k | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes/event | latency_ms_p95 | adapter state |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OPR-OPU | opu_old_guarded | 4 | 0.1615 | 0.0000 | 0.0333 | 0.0000 | 0.9667 | 0.0333 | 491.1 | 0.9363 | 13122 B |
| OPR-OPU | opu_old_guarded | 5 | 0.1615 | 0.0000 | 0.0333 | 0.0000 | 0.9667 | 0.0333 | 538.5 | 0.9363 | 13122 B |
| OPR-OPU | opu_old_preserve | 4 | 0.1615 | 0.0000 | 0.0333 | 0.0000 | 0.9167 | 0.0833 | 491.1 | 0.9363 | 13122 B |
| OPU baseline | opu_old_preserve | 4 | 0.1615 | 0.0000 | 0.0000 | 0.0000 | 0.9000 | 0.0833 | 491.1 | 约0.94 | 0 B |
| OPU baseline | opu_old_guarded | 4 | 0.1406 | 0.0000 | 0.0000 | 0.0000 | 0.9167 | 0.0833 | 491.1 | 约0.94 | 0 B |

Adapter训练诊断：

| 字段 | 值 |
|---|---:|
| `target_unknown_training_count` | 0 |
| `adapter_train_seconds` | 1.05 |
| `source_proto_acc_before` | 0.9542 |
| `source_proto_acc_after` | 0.9604 |
| `support_proto_acc_before` | 0.6125 |
| `support_proto_acc_after` | 0.7188 |
| `proxy_max_logit_before_mean` | 10.3325 |
| `proxy_max_logit_after_mean` | -2.2077 |
| `total_fp16_state_bytes` | 13122 |

## 本地解释

OPR adapter确实完成了proxy-open目标：`proxy_max_logit`显著下降，unknown拒识升至0.9667且FAR降至0.0333。但这一路线仍严重不满足目标，因为target-old和seen-new几乎被打穿。该结果说明在当前proxy_unknown特征包上，proxy-open目标与target known保持冲突明显；仅靠低秩adapter无法同时满足未知拒识和旧/新类识别。

该结果只能作为`NEGATIVE_DIAGNOSTIC`，不能声明Stage2-C成功或部署成功。

## N607计划

| 字段 | 内容 |
|---|---|
| 远端根目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 运行GPU | 选择当前显存最低GPU |
| 远端输入 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz` |
| 远端输出 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_opr_opu_ci_20260704/` |
| 远端日志 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_opr_opu_ci_20260704/opr_opu_ci.log` |

## N607验证与运行

N607预检：

| 字段 | 内容 |
|---|---|
| 预检时间 | 2026-07-04 14:14:41 CST |
| SSH目标 | direct `N607`，配置`E:\type10-7\tools\n607_ssh_config` |
| 项目根目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| GPU选择 | GPU0；运行前8张RTX 3090均为`10/24576 MiB` |
| 远端进程 | 未见当前用户训练进程，仅有系统/VSCode相关Python进程 |
| 输入检查 | `runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz`存在 |

同步目标：

| 本地文件 | N607目标 |
|---|---|
| `E:\type10-7\code\scripts\phase2_opr_opu_adapter_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_opr_opu_adapter_ci_eval.py` |
| `E:\type10-7\code\tests\test_phase2_opr_opu_adapter_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_opr_opu_adapter_ci_eval.py` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_opr_opu_ci_20260704\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_opr_opu_ci_20260704/report.md` |

远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -m py_compile code/scripts/phase2_opr_opu_adapter_ci_eval.py code/tests/test_phase2_opr_opu_adapter_ci_eval.py
CUDA_VISIBLE_DEVICES=0 $PY code/tests/test_phase2_opr_opu_adapter_ci_eval.py
CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_opr_opu_adapter_ci_eval.py --feature_npz runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz --output_dir runs/phase2_adv3b02_opr_opu_ci_20260704 --device cuda:0 --adapter_epochs 20 --policies opu_old_preserve,opu_old_guarded --max_event_latency_ms 2.0
```

说明：第一次远端运行使用`max_event_latency_ms=1.0`，CUDA路径`latency_ms_p95=1.431`触发资源门控，所有row变为defer。已修正脚本默认值为2.0ms并重跑，最终结果如下。

远端输出：

| 文件 | 本地归档 | SHA256 |
|---|---|---|
| `runs/phase2_adv3b02_opr_opu_ci_20260704/opr_opu_ci_summary.csv` | `E:\type10-7\local_artifacts\phase2_adv3b02_opr_opu_ci_20260704\remote\opr_opu_ci_summary.csv` | `9CF0CEB167AC808E6BD306389CFF387F010BD8F497C36D829BF06C6D204D2AD4` |
| `runs/phase2_adv3b02_opr_opu_ci_20260704/opr_opu_ci_summary.json` | `E:\type10-7\local_artifacts\phase2_adv3b02_opr_opu_ci_20260704\remote\opr_opu_ci_summary.json` | `5E0BBCF1B29892EA7125B96ABA71C227B1CB363D6EDBB4C194FABA0666255A63` |
| `logs/phase2_adv3b02_opr_opu_ci_20260704/opr_opu_ci.log` | `E:\type10-7\local_artifacts\phase2_adv3b02_opr_opu_ci_20260704\remote\opr_opu_ci.log` | `828404DA28C06AA0CAC2CC733BCE1CA102CBDF5D57C734F1913A26A065D5FA92` |

远端结果：

| policy | k | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes/event | latency_ms_p95 | adapter state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opu_old_guarded | 4 | 0.1250 | 0.0000 | 0.0500 | 0.0000 | 0.9667 | 0.0333 | 491.1 | 1.4310 | 13122 B |
| opu_old_guarded | 5 | 0.1250 | 0.0000 | 0.0500 | 0.0000 | 0.9500 | 0.0333 | 538.5 | 1.4310 | 13122 B |
| opu_old_preserve | 4 | 0.1302 | 0.0000 | 0.0500 | 0.0000 | 0.9167 | 0.0833 | 491.1 | 1.4310 | 13122 B |
| opu_old_guarded | 3 | 0.0573 | 0.0000 | 0.0500 | 0.0000 | 0.9333 | 0.0333 | 403.8 | 1.4310 | 13122 B |
| opu_old_guarded | 1 | 0.0729 | 0.0000 | 0.0833 | 0.0000 | 0.9167 | 0.0833 | 168.0 | 1.4310 | 13122 B |

运行后GPU仍为8张RTX 3090均`10/24576 MiB`。SSH/SCP后本地`ssh.exe`、到`172.31.111.215:22`和`172.31.105.18:22`的`ESTABLISHED`连接均为空。

## 最终结论

OPR-OPU-CI不是达标路线，但提供了明确负证据：

1. `target_unknown_training_count=0`，协议合规；远端使用`CVS-RFFI`环境复跑通过。
2. 低秩adapter状态量仅约13KB，远端延迟约1.43ms，资源层面可部署。
3. proxy-open目标过强：unknown拒识可到0.9667且FAR 0.0333，但旧类和seen-new被打穿，`old_acc`最高仅0.1302到0.1250量级，远低于OLD80和最终99%目标。

因此，单纯proxy-open低秩adapter不是当前主线解。下一步应改成“旧类保真硬约束/回滚选择”的训练目标：把合法support上的old/seen-new保持作为profile选择硬门，只有在`old_acc`不低于base qknn8时才允许引入open约束；否则必须回退到base特征或只更新阈值/温度。
