# phase2_adv3b02_proxy_adapter_ci_20260704

## 基本信息

- 实验ID：phase2_adv3b02_proxy_adapter_ci_20260704
- 时间：2026-07-04
- 操作：Codex
- 目标：实现并验证OPR-CI，面向天基射频指纹识别卫星群协同推理，在冻结ADV3B02_CORE90_SOFT_E200/qknn8特征证据路径的前提下，用source旧类、source-side proxy_unknown和target old/seen-new K-shot support训练轻量adapter，优先提升未知类拒识，同时不牺牲旧类。
- 协议文件：已读取`AGENTS.md`和`项目.md`。

## 方法

OPR-CI采用低秩残差feature adapter：

```text
A_theta(z)=normalize(z+alpha*W2(GELU(W1(LN(z)))))
```

训练角色：

| 角色 | 用途 |
|---|---|
| source | 旧类保真和base logits蒸馏 |
| proxy_unknown | source-side outlier exposure |
| target_old support | 在轨旧类少样本校准 |
| target_new support | 在轨新类少样本注册 |
| target_unknown | 仅最终评估，不进入训练、阈值、profile选择、早停或标准化统计 |

损失：

```text
L=source_cls+support_cls+old_preserve+proxy_open+support_compact+residual
```

adapter后生成adapted feature NPZ，再复用`phase2_orbit_enpc_ci_eval.py`和`phase2_orbit_slev_ci_eval.py`进行完整qknn8协同评估。

## 本地变更

| 文件 | 目的 |
|---|---|
| `code/scripts/phase2_proxy_adapter_ci_eval.py` | 新增OPR-CI训练、adapted NPZ生成、ENPC/SLEV后端评估 |
| `code/tests/test_phase2_proxy_adapter_ci_eval.py` | 新增target_unknown不泄漏与manifest标记测试 |
| `automation_reports/CV-SincNet/phase2_adv3b02_proxy_adapter_ci_20260704/report.md` | 本报告 |

## 本地验证

```text
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_proxy_adapter_ci_eval.py code\tests\test_phase2_proxy_adapter_ci_eval.py
```

结果：通过。

```text
conda run -n ssr-gpu python code\tests\test_phase2_proxy_adapter_ci_eval.py
```

结果：2项测试通过。

首次并行运行`conda run`时触发Windows临时文件锁，随后改为串行执行；该锁不是代码或实验失败。

## 数据与协议证据

- feature NPZ：`local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz`
- sha256：`7f5c2956ce78f0a2b44c6f41fee453613eede5cf916be0ff6899365fac7a3297`
- 来源N607路径：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz`
- 角色计数：source=2400，proxy_unknown=1600，target_old=2400，target_new=800，target_unknown=800
- target receivers：20-1，3-19，7-14，7-7，8-8
- collab_counts：`all`，即当前target receiver domain的1..5。当前脚本评估的是卫星群目标接收机协同，不是source receiver协同。
- 星地信道：沿用proxy feature manifest中的LEO/satellite stress特征；adapted NPZ保留原始`channel_views`和`sat_scenarios`字段。
- 泄漏防线：`target_unknown_training_count=0`，`target_unknown_eval_only=true`。

## 本地结果

### OPR-CI默认配置

命令：

```text
conda run -n ssr-gpu python code\scripts\phase2_proxy_adapter_ci_eval.py --feature_npz local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz --output_dir local_artifacts\phase2_adv3b02_opr_ci_20260704\local --backend both --collab_counts all --collab_group_policy same_max_budget --k_shot 5 --query_per_class 20 --qknn_k 8 --adapter_epochs 40 --adapter_rank 16 --adapter_alpha 0.20 --device cuda:0 --max_event_bytes 1152 --max_event_latency_ms 20
```

训练计数：

| source_old | proxy_unknown | target_support | target_unknown_eval_only | target_unknown_training_count | state_bytes |
|---:|---:|---:|---:|---:|---:|
| 2400 | 1600 | 200 | 800 | 0 | 13122 |

主要同row结果：

| backend | profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes_per_event | latency_ms | target_pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ENPC | enpc_balanced | 5 | 0.8627 | 0.0000 | 0.4500 | 0.0000 | 0.3500 | 0.6500 | 640.0 | 1.0954 | false |
| SLEV | slev_balanced | 5 | 0.8235 | 0.0000 | 0.3000 | 0.0000 | 0.5000 | 0.5000 | 640.0 | 1.0541 | false |
| SLEV | slev_energy_strict | 4 | 0.2308 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 512.0 | 1.0541 | false |

结论：默认OPR-CI相比proxy ENPC的低unknown旧类保护行提高未知拒识，但仍无法满足旧类/seen-new/min类指标；强拒识行仍通过牺牲旧类取得，标为diagnostic-only。

### OPR-CI保守旧类保护配置

命令：

```text
conda run -n ssr-gpu python code\scripts\phase2_proxy_adapter_ci_eval.py --feature_npz local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz --output_dir local_artifacts\phase2_adv3b02_opr_ci_20260704\local_guarded --backend both --collab_counts all --collab_group_policy same_max_budget --k_shot 5 --query_per_class 20 --qknn_k 8 --adapter_epochs 40 --adapter_rank 8 --adapter_alpha 0.10 --old_preserve_weight 8.0 --residual_weight 0.50 --proxy_open_weight 0.60 --support_compact_weight 0.25 --device cuda:0 --max_event_bytes 1152 --max_event_latency_ms 20
```

训练计数：

| source_old | proxy_unknown | target_support | target_unknown_eval_only | target_unknown_training_count | state_bytes |
|---:|---:|---:|---:|---:|---:|
| 2400 | 1600 | 200 | 800 | 0 | 8002 |

主要同row结果：

| backend | profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes_per_event | latency_ms | target_pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ENPC | enpc_balanced | 5 | 0.8235 | 0.0000 | 0.6000 | 0.0000 | 0.1000 | 0.9000 | 640.0 | 1.0967 | false |
| SLEV | slev_known_anchor | 5 | 0.8431 | 0.0000 | 0.7000 | 0.0000 | 0.0000 | 1.0000 | 640.0 | 1.0543 | false |
| ENPC | enpc_unknown_strict | 2 | 0.5168 | 0.2000 | 0.0213 | 0.0000 | 0.8222 | 0.1778 | 256.0 | 1.0967 | false |

结论：保守配置改善seen-new但未知拒识回落，仍未达成目标。

## 当前判断

OPR-CI作为部署侧轻量adapter可以提高部分unknown拒识，但现有ADV3B02/qknn8特征空间中真实unknown仍与known高度重叠。只在部署后处理或轻量adapter层修复不足以达到`old_acc≥99%`、`min_old≥95%`、`seen_new_acc≥97%`、`min_seen≥93%`、`unknown_reject≥99%`。若N607复测复现该结论，下一步应回到ADV3B02训练阶段加入source/proxy open-set representation loss、class-wise EVT/GPD或Mahalanobis shrinkage原型、receiver reliability加权，而不是继续只调阈值。

## N607计划

- 先运行`tools\n607_ssh_preflight.ps1`。
- 记录`nvidia-smi`显存快照，选择显存占用最少GPU。当前preflight显示GPU0-7均约10MiB，默认选择GPU0，若复测前状态变化则选择最低显存GPU。
- 同步本地验证后的脚本和测试到N607：
  - `code/scripts/phase2_proxy_adapter_ci_eval.py` -> `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_proxy_adapter_ci_eval.py`
  - `code/tests/test_phase2_proxy_adapter_ci_eval.py` -> `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_proxy_adapter_ci_eval.py`
- 远程环境：`conda run -n CVS-RFFI ...`
- 远程输出目录：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_proxy_adapter_ci_20260704`
- 每次SSH/SCP后检查本地`ssh.exe`和到`172.31.111.215:22`、`172.31.105.18:22`的`ESTABLISHED`连接。

## N607执行结果

### SSH与同步

- preflight：`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`通过。
- GPU启动前快照：GPU0-7均为NVIDIA GeForce RTX 3090，显存10/24576MiB，utilization 0%；无compute进程。
- 选择GPU：GPU0。理由：所有GPU显存占用相同且最低。
- 同步命令：

```text
scp -F E:\type10-7\tools\n607_ssh_config code\scripts\phase2_proxy_adapter_ci_eval.py N607:/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_proxy_adapter_ci_eval.py
scp -F E:\type10-7\tools\n607_ssh_config code\tests\test_phase2_proxy_adapter_ci_eval.py N607:/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_proxy_adapter_ci_eval.py
```

- 远程验证环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 远程验证命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet && /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_proxy_adapter_ci_eval.py code/tests/test_phase2_proxy_adapter_ci_eval.py && /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_proxy_adapter_ci_eval.py
```

结果：2项测试通过。非交互shell中`conda`和`python`不在PATH，因此最终使用CVS-RFFI环境的绝对Python路径。

- SSH/SCP断连检查：每次SSH/SCP后检查本地`ssh.exe`、`172.31.111.215:22`和`172.31.105.18:22`，均无残留连接。

### 远程命令

默认配置：

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_proxy_adapter_ci_eval.py --feature_npz /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz --output_dir /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_proxy_adapter_ci_20260704/default --backend both --collab_counts all --collab_group_policy same_max_budget --k_shot 5 --query_per_class 20 --qknn_k 8 --adapter_epochs 40 --adapter_rank 16 --adapter_alpha 0.20 --device cuda:0 --max_event_bytes 1152 --max_event_latency_ms 20
```

保守旧类保护配置：

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_proxy_adapter_ci_eval.py --feature_npz /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz --output_dir /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_proxy_adapter_ci_20260704/guarded --backend both --collab_counts all --collab_group_policy same_max_budget --k_shot 5 --query_per_class 20 --qknn_k 8 --adapter_epochs 40 --adapter_rank 8 --adapter_alpha 0.10 --old_preserve_weight 8.0 --residual_weight 0.50 --proxy_open_weight 0.60 --support_compact_weight 0.25 --device cuda:0 --max_event_bytes 1152 --max_event_latency_ms 20
```

日志：

| 配置 | 日志路径 |
|---|---|
| default | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_proxy_adapter_ci_20260704/default.log` |
| guarded | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_proxy_adapter_ci_20260704/guarded.log` |

收尾GPU快照：GPU0-7均为10/24576MiB，无额外残留显存占用。

### 远程结果

远程结果已拉回：

```text
local_artifacts\phase2_adv3b02_opr_ci_20260704\remote\default
local_artifacts\phase2_adv3b02_opr_ci_20260704\remote\guarded
local_artifacts\phase2_adv3b02_opr_ci_20260704\remote\logs
```

默认配置：

| backend | profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes_per_event | latency_ms | target_pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ENPC | enpc_balanced | 5 | 0.8235 | 0.0000 | 0.6000 | 0.0000 | 0.2000 | 0.8000 | 640.0 | 1.3399 | false |
| SLEV | slev_balanced | 5 | 0.8039 | 0.0000 | 0.5000 | 0.0000 | 0.6000 | 0.4000 | 640.0 | 1.2211 | false |

保守旧类保护配置：

| backend | profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | bytes_per_event | latency_ms | target_pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ENPC | enpc_known_anchor | 5 | 0.8431 | 0.0000 | 0.8000 | 0.0000 | 0.0000 | 1.0000 | 640.0 | 1.6773 | false |
| SLEV | slev_known_anchor | 5 | 0.8431 | 0.0000 | 0.8000 | 0.0000 | 0.0000 | 1.0000 | 640.0 | 1.2394 | false |

远程输出hash：

| 配置 | 文件 | sha256 |
|---|---|---|
| default | `opr_ci_summary.json` | `2be42032c06078c4c2c063ba7cb403dee7e6a672c2d0cebb3a46e625a96c0ac0` |
| default | `opr_ci_enpc_summary.csv` | `6d33e6eeb085c8ab9b5703338e13df0bb60dcf97ce093a0818a45d9d6e74689e` |
| default | `opr_ci_slev_summary.csv` | `663263777856d3b21bd97ff1ae77c10834e0e6ebe72cd1e2a9aee274dc6daec7` |
| default | `opr_ci_adapted_features.npz` | `a201e9091b49e884fba2586976e9941d25daad8ad9ac8c390494b461af0c47ba` |
| guarded | `opr_ci_summary.json` | `58cef8a7e212c204173369b1c111202cb86abf3fd306b789d73c13b450ffce98` |
| guarded | `opr_ci_enpc_summary.csv` | `cf8896cfecb93344b5b0794c4e1fd79c11524f97e9c06d2c4358dd69be53416a` |
| guarded | `opr_ci_slev_summary.csv` | `9c59d4c2f2650554714c1d1ba7595b92705a49545c189b4a148d42f76b03aceb` |
| guarded | `opr_ci_adapted_features.npz` | `5af6d0bfc951f1ca2ca5434c38f022d60e2aeb35a4ecb9d287f9cb11186136c9` |

### 远程结论

N607复测未达标。默认配置最高同row未知拒识为0.6000，但old_acc仅0.8039且min_old=0；保守配置提升old/seen-new到0.8431/0.8000，但unknown_reject=0。该结果支持本地判断：仅靠部署侧adapter和协同后处理不足以同时满足未知拒识与旧类保真，需要回到ADV3B02训练阶段加入开放集表征约束。
