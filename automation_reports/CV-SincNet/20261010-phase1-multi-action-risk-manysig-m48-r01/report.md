# 下一版多解耦：真实动作提案与R判别分布风险

本轮纯CE底座，无半监督、EMA教师和星地增强。身份网络结构与cosine日程保留；作用模型使用本run周期冻结学生快照。所有身份模型从零训练E200/44400步。

先完成三seed源机制诊断，随后自动运行以下16种配置，每种2026092701/02/03，共48个身份模型。源诊断使用历史固定native E200合规权重，仅用于诊断，不初始化新模型。

| 配置 | 实际计划 |
|---|---|
| native | `{"paths": [], "action_mode": "proposal", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": [], "conditional_edges": true}` |
| random_ce | `{"paths": [], "action_mode": "random", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": ["linear", "temporal", "joint"], "conditional_edges": true}` |
| random_cons | `{"paths": [], "action_mode": "random", "shared": false, "r_target": "distribution", "consistency_weight": 0.01, "action_label_free": false, "source_u_fit": false, "replacements": ["linear", "temporal", "joint"], "conditional_edges": true}` |
| exact_g | `{"paths": [], "action_mode": "exact_g", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": ["linear", "temporal", "joint"], "conditional_edges": true}` |
| L | `{"paths": ["linear"], "action_mode": "proposal", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": [], "conditional_edges": true}` |
| LT | `{"paths": ["linear", "temporal"], "action_mode": "proposal", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": [], "conditional_edges": true}` |
| LR | `{"paths": ["linear", "receiver"], "action_mode": "proposal", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": [], "conditional_edges": true}` |
| LTR | `{"paths": ["linear", "temporal", "receiver"], "action_mode": "proposal", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": [], "conditional_edges": true}` |
| LTR_mean | `{"paths": ["linear", "temporal", "receiver"], "action_mode": "proposal", "shared": false, "r_target": "mean", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": [], "conditional_edges": true}` |
| shared_LTR | `{"paths": ["linear", "temporal", "receiver"], "action_mode": "proposal", "shared": true, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": [], "conditional_edges": true}` |
| L_budget | `{"paths": ["linear"], "action_mode": "proposal", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": ["temporal", "receiver", "joint"], "conditional_edges": true}` |
| LT_budget | `{"paths": ["linear", "temporal"], "action_mode": "proposal", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": ["receiver"], "conditional_edges": true}` |
| LR_budget | `{"paths": ["linear", "receiver"], "action_mode": "proposal", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": ["temporal", "joint"], "conditional_edges": true}` |
| LT_no_edges | `{"paths": ["linear", "temporal"], "action_mode": "proposal", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": false, "source_u_fit": false, "replacements": [], "conditional_edges": false}` |
| LT_label_free | `{"paths": ["linear", "temporal"], "action_mode": "proposal", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": true, "source_u_fit": false, "replacements": [], "conditional_edges": true}` |
| LT_label_free_U | `{"paths": ["linear", "temporal"], "action_mode": "proposal", "shared": false, "r_target": "distribution", "consistency_weight": 0.0, "action_label_free": true, "source_u_fit": true, "replacements": [], "conditional_edges": true}` |

L/T提案经过真实IQ核验，身份E/G/C沿真实端点回传；R拟合5维竞争margin分布，身份目标为CE与CVaR，仅更新G/C。保持固定分支系数，不因缺分支重分配。

每个row完成源V压力验证并冻结后，自动测试clean与六种practical；本row预测完整固定后独立评分。完整矩阵汇总实际seed数、配对差、RX/TX/day分层。源排名只读取source_stress，不读取目标成绩。

旧结果完整恢复316份结构化源文件，位于`E:\type10-7\code\worktrees\receiver_residual_dg_20261008\local_artifacts\multi_action_risk_historical_r01`；压缩原始证据约143MB，保留本地产物，不入Git大文件。完整文件清单见[恢复记录](historical_recovery.json)。历史稀疏梯度epoch归一化错误保留说明，不把旧值解释为实际余弦。

逐项要求见[设计追溯](traceability.md)，原文见[设计报告](design_report.md)。第四个交互/随机作用网络遵照条件触发条款，本轮不添加。当前状态以experiment.json与远端读回为准。

本地验证与独立P0/P1审查通过，已修复R单包协方差准入缺陷。每辅助调用32个新L包与R最多32个缓存审计包重编码分开记账，不把两者合计说成32包。源压力固定8视图（含未训练LTL和人工曲率）；不是新增目标测试场景。详细验证见evidence。

## N607发布核验

状态：**VERIFIED / RUNNING**。当前处于三seed源机制诊断；48行身份训练已经预登记并由同一控制器自动衔接，尚不能表述为48行已经开始训练。

代码提交：`33773918f509e78c1c0aa9f6fa36d85db099cbbe`。控制器PID：`3214285`。远端预检PASS；二次独立读回确认工作目录、命令、进程及日志增长，无失败记录。

| 源诊断seed | GPU | PID | 日志字节 | 两次读回增长 |
|---|---:|---:|---:|---:|
| diagnostic-s2026092701 | 3 | 3214304 | 6826 | 6559 |
| diagnostic-s2026092702 | 4 | 3214309 | 6832 | 6565 |
| diagnostic-s2026092703 | 5 | 3214314 | 6821 | 6821 |

远端证据：[发布参数](evidence/package.json)、[提交回执](evidence/submit.json)、[首次读回](evidence/readback_initial.json)、[再次读回](evidence/readback.json)。

设计追溯：25项verified、1项按报告条件deferred、0项rejected、0项blocked。严格遵照设计实现；报告旧训练配方由用户明确选择的纯CE、无半监督、无星地增强覆盖。第四网络遵照源证据触发条件延后。

最高剩余不确定性是实际性能收益和正式训练中的动作/R准入率，不能由本地回归或启动成功推断。正式训练和测试尚未完成。
