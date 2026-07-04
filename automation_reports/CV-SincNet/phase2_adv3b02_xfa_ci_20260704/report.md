# phase2_adv3b02_xfa_ci_20260704

## 基本信息

| 字段 | 值 |
|---|---|
| 实验ID | phase2_adv3b02_xfa_ci_20260704 |
| 时间 | 2026-07-04 |
| 操作者 | Codex |
| 目标 | 面向卫星群多接收机协同推理，验证cross-feature auxiliary unknown-risk XFA-CI是否能优先改善未知类拒识，并保持旧类准确率不下降 |
| 基础模型/特征 | ADV3B02_CORE90_SOFT_E200 base qKNN evidence |
| 场景 | CVS Stage2-C；target receiver domain；satellite/LEO channel view；`Y_old/Y_new/Y_unknown`互斥 |

## 算法设计

XFA-CI保持base qKNN为唯一标签来源，辅助适配feature只允许输出`unknown_risk/defer_risk`。融合规则为：

| 组件 | 规则 |
|---|---|
| base label authority | `predicted_label`、top-k label和known-route分数全部来自base qKNN，不允许aux改写TX标签 |
| paired evaluation | base-only和base+aux使用完全相同的`event_id,receiver_id,role,true_label`交集 |
| weak-known gate | 当base score、margin、support density、receiver reliability不足时，aux风险按弱度比例提高`unknown_risk` |
| strong-known cap | 当base为强已知样本时，aux风险贡献被`strong_aux_cap`限幅，且不能降低base已有风险 |
| 通信预算 | 默认每接收机额外上传16 bytes量化aux风险摘要，保留`bytes_per_event`和`latency_ms_p95` |
| 阈值约束 | `threshold_fit_scope`沿用support/source安全范围；`unknown_query_used_for_threshold=false` |

判定规则：若`old_acc < 0.80`、`delta_old_acc < 0`、未知拒识没有提升，或unknown FAR恶化，均标为`diagnostic_only`，不能写成成功候选。

## 本地改动

| 文件 | 目的 |
|---|---|
| `github_publish/CVS-RFFI-repo/code/scripts/phase2_cross_feature_aux_ci_eval.py` | 新增XFA-CI paired cross-feature辅助拒识评估脚本 |
| `github_publish/CVS-RFFI-repo/code/tests/test_phase2_cross_feature_aux_ci_eval.py` | 覆盖paired子集、base label authority、strong-known aux cap |

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_cross_feature_aux_ci_eval.py -q` | PASS，3 passed |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_cross_feature_aux_ci_eval.py` | PASS |
| XFA-CI local smoke，aux=`MOE local_k8`，`aux_weight=0.25`，`strong_aux_cap=0.05`，policies=`opu_old_preserve,opu_old_guarded` | PASS，产出`local_artifacts/phase2_adv3b02_xfa_ci_20260704/local_smoke_moe/` |

## 本地烟测结果

| policy | collab_count | old_acc | seen_new_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_unknown_reject_rate | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| opu_old_preserve | 4 | 0.8074866310 | 0.8166666667 | 0.1833333333 | 0.7500000000 | 0.0000000000 | 0.0000000000 | diagnostic_only |
| opu_old_guarded | 4 | 0.7700534759 | 0.7166666667 | 0.2500000000 | 0.5500000000 | 0.0000000000 | 0.0000000000 | diagnostic_only |
| opu_old_preserve | 5 | 0.6256684492 | 0.5166666667 | 0.0333333333 | 0.5666666667 | -0.1764705882 | -0.0500000000 | diagnostic_only |
| opu_old_guarded | 5 | 0.5935828877 | 0.4833333333 | 0.1000000000 | 0.5000000000 | -0.1711229947 | -0.1000000000 | diagnostic_only |

结论：MOE辅助特征烟测没有形成有效拒识收益；`collab_count=5`反而伤害旧类和seen-new，必须作为负面诊断保留。

## N607计划

| 项 | 值 |
|---|---|
| Conda环境 | `CVS-RFFI` |
| 工作目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端脚本 | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_cross_feature_aux_ci_eval.py` |
| 远端测试 | `python -m pytest code/tests/test_phase2_cross_feature_aux_ci_eval.py -q`和XFA-CI全collab_count运行 |
| GPU策略 | 先用`nvidia-smi`选择显存占用较低GPU；XFA-CI本身为证据表CPU评估，GPU仅记录环境和占用 |
| 日志路径 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_xfa_ci_20260704/` |
| 输出路径 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_xfa_ci_20260704/` |

## 风险和待检

| 风险 | 处理 |
|---|---|
| aux feature可能只是domain/channel detector | 必须分receiver/TX/view报告，并保留unknown->old混淆和old_to_reject |
| paired子集缺失造成虚假收益 | `same_subset=true`、`sample_count_matched`、`missing_aux_row_count`必填 |
| 多配置挑最好形成选择偏差 | 全矩阵保留`xfa_ci_summary.csv`；未通过旧类保底的row标`diagnostic_only` |
| 本地全aux网格慢 | 已修脚本默认不写大型证据CSV并缓存base-only评估；远端做全量/代表性扩展 |
| N607项目根目录`evaluation/`遮蔽`code/evaluation` | XFA-CI脚本强制将`CODE_ROOT`放到`sys.path[0]`，避免远端布局导致`collaborative_open_set_qknn_eval`导入失败 |
