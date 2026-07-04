# phase2_adv3b02_socapr_pareto_20260704

## 基本信息

| 字段 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_socapr_pareto_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 在ADV3B02_CORE90_SOFT_E200特征上固化SO-CAPR qknn8双路线Pareto诊断，确认当前代码下known性能和unknown拒识的真实边界 |
| 底座 | ADV3B02_CORE90_SOFT_E200；特征SHA256=`DB559D78DB305894307851750EF7D698DB387F0984FF13C980FEA99DB85B8532` |
| 在轨少样本方法 | qknn8，K=8 |
| 协同范围 | `collab_counts=all`，即1到5个target receivers |
| 状态 | NON_DEPLOYMENT_DIAGNOSTIC |

## 资源约束说明

用户指定的`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`未在工作区按文件名检出。本轮使用仓库已有设计文档`analysis/adv3b02_qknn8_satellite_collab_open_set_design_20260703.md`和`code/analysis/aware_ci_satellite_collaborative_inference_design.md`中的资源字段：参与接收机数、`bytes_per_event`、`latency_ms_p95`、prototype/qknn状态规模。指定文档缺失不改变实验协议，但报告中不把资源结论写成已按该文档逐条验证。

## 本轮新增文件

| 文件 | 作用 |
|---|---|
| `E:\type10-7\code\scripts\phase2_socapr_qknn8_pareto_eval.py` | 固化SO-CAPR qknn8的known route和safety route，并对融合策略/unknown阈值做Pareto重评估 |
| `E:\type10-7\code\tests\test_phase2_socapr_qknn8_pareto_eval.py` | 验证qknn8、资源packet字段和joint score排序 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_socapr_pareto_20260704\phase2_socapr_qknn8_pareto_eval.py` | 非Git根目录快照 |
| `E:\type10-7\code\snapshots\phase2_adv3b02_socapr_pareto_20260704\test_phase2_socapr_qknn8_pareto_eval.py` | 非Git根目录快照 |

## 算法定义

SO-CAPR在本轮固化为两条可复现实验路线：

| 路线 | 目的 | 核心配置 |
|---|---|---|
| known route | 保留旧类/seen-new识别能力，测known上界 | `qknn_k=8`，prototype blend=2.0，Mahalanobis blend=1.0，`unknown_gate_mode=score`，`evidence_packet_bytes=40` |
| safety route | 强化unknown拒识，测安全上界 | `qknn_k=8`，support envelope consensus，class conformal，virtual unknown，class negative，class shell，`evidence_packet_bytes=128` |

重评估不重新拟合特征或阈值，只在已有evidence上扫描：

```text
fusion_policy in {risk_margin, scg_qknn_cvs, support_router_cvs, candidate_set_cvs}
unknown_risk_threshold in {0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90}
collab_count in {1,2,3,4,5}
```

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_socapr_qknn8_pareto_eval.py code\tests\test_phase2_socapr_qknn8_pareto_eval.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_socapr_qknn8_pareto_eval.py -q` | PASS，2 passed；`.pytest_cache`权限警告不影响测试 |

本地运行命令：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_socapr_qknn8_pareto_eval.py --feature_npz remote_artifacts\phase2_adv3b02_features\features.npz --output_dir local_artifacts\phase2_adv3b02_socapr_pareto_20260704 --force
```

本地输出：

| 文件 | 内容 |
|---|---|
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_pareto_20260704\known_route.json` | known route原始评估 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_pareto_20260704\known_route_evidence.csv` | known route evidence |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_pareto_20260704\safety_route.json` | safety route原始评估 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_pareto_20260704\safety_route_evidence.csv` | safety route evidence |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_pareto_20260704\socapr_pareto_summary.csv` | 320行Pareto汇总 |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_pareto_20260704\socapr_pareto_summary.json` | Pareto汇总JSON |

## 当前权威边界

### known route原始k=5

| collab_count | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | known_coverage | defer_rate | bytes/event | latency_ms_p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.8663 | 0.6000 | 0.8500 | 0.8000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 130.3 | 0.0780 |

解释：qknn8+协同能明显提高旧类和seen-new，但unknown完全被误接收。这是下一步拒识算法必须保留的known上界参考。

### safety route原始k=5

| collab_count | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | known_coverage | defer_rate | bytes/event | latency_ms_p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.0267 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0202 | 0.0000 | 416.9 | 0.1348 |

解释：强安全门控能拒识unknown，但几乎拒掉所有known，不能作为可用协同推理。

### Pareto代表点

| 视角 | route | fusion | threshold | k | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_coverage | defer_rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| highest joint score | safety_route | scg_qknn_cvs | 0.20 | 2 | 0.4813 | 0.0500 | 0.3167 | 0.1500 | 0.6833 | 0.3167 | 0.5506 | 0.0000 |
| best k=5 known Pareto | known_route | risk_margin | 0.90 | 5 | 0.7968 | 0.2500 | 0.7833 | 0.7250 | 0.0167 | 0.7667 | 0.9595 | 0.0717 |
| only OLD80 Pareto region | known_route | risk_margin | 0.30-0.90 | 1 | 0.8021-0.8128 | 0.1500 | 0.8500-0.8667 | 0.8000-0.8250 | 0.0500-0.0833 | 0.7500-0.9500 | 0.9555-1.0000 | 0.0000-0.0684 |
| low-FAR safety | safety_route | support_router_cvs | 0.70 | 5 | 0.0963 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0729 | 0.0261 |

## 结论

当前代码和当前ADV3B02特征下，没有任何k=5 Pareto工作点达到`old_acc>=0.80`；只有k=1的known route局部达到OLD80，但unknown_FAR仍为0.75-0.95。没有任何工作点同时满足：

1. `old_acc>=0.80`；
2. per-old floor接近0.95；
3. seen-new接近0.97；
4. unknown拒识接近0.99且FAR低；
5. known覆盖可接受。

## N607复测

N607使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`和`CUDA_VISIBLE_DEVICES=0`复测。预检时GPU0显存占用10MiB，满足低显存占用测试要求。

远端验证：

| 命令 | 结果 |
|---|---|
| `python -m py_compile code/scripts/phase2_socapr_qknn8_pareto_eval.py code/tests/test_phase2_socapr_qknn8_pareto_eval.py` | PASS |
| `python code/tests/test_phase2_socapr_qknn8_pareto_eval.py` | PASS，2 tests OK |
| `python code/scripts/phase2_socapr_qknn8_pareto_eval.py --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz --output_dir runs/phase2_adv3b02_socapr_pareto_20260704 --force` | PASS，生成320行Pareto汇总 |

远端输出：

| 文件 | 内容 |
|---|---|
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_socapr_pareto_20260704/socapr_pareto_summary.csv` | 远端Pareto汇总 |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_socapr_pareto_20260704/socapr_pareto_summary.json` | 远端Pareto汇总JSON |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_pareto_20260704\remote_socapr_pareto_summary.csv` | 拉回的远端CSV |
| `E:\type10-7\local_artifacts\phase2_adv3b02_socapr_pareto_20260704\remote_socapr_pareto_summary.json` | 拉回的远端JSON |

这不是阈值单点选择问题，而是证据空间问题：known route能分类但没有unknown风险，safety route有unknown风险但无法保留known。下一步应实现真正的双路仲裁：先用known route给old/seen-new候选，再用独立的source/target support外部空间模型只对高风险事件触发reject/defer，不能让安全门控直接覆盖所有known。

## 下一步

1. 在`phase2_collaborative_open_set_qknn_eval.py`中新增SO-CAPR双路融合策略或单独脚本：known route负责候选标签，safety route只作为unknown veto，不直接决定known分类。
2. 对每个候选标签记录per-class support geometry，并按类而不是全局阈值做veto。
3. 引入source old prototype shrinkage，提高`14-7`、`20-19`等低floor旧类。
4. 保留本轮`socapr_pareto_summary.csv`作为回归基线；后续任何改动必须同时提升OLD80和unknown，不接受单边改善。
