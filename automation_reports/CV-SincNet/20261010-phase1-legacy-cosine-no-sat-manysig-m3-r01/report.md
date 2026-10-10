# 旧门控＋cos、关闭星地增强：三seed基线

状态：**RUNNING / VERIFIED**。N607远端预检通过；独立读回确认3个训练进程存活、实际参数匹配、日志持续增长。当前没有测试结果。

保留reference_response身份网络、EMA教师0.999、旧batch_neighbor门控、130+70半监督、label_smoothing0.01及cosine2e-4至1e-6。只关闭星地训练增强及其损失；无额外解耦网络。从零训练E200/44400更新。

| Seed | GPU | PID |
|---|---|---|
| 2026092701 | 0 | 3197269 |
| 2026092702 | 1 | 3197274 |
| 2026092703 | 2 | 3197280 |

每个seed训练完成后冻结ownE200，立即进入clean＋六种practical测试；本row七视图预测固定后，由独立scorer连接truth。其余seed训练不读取评分。三seed全部完成后自动汇总均值、样本标准差、接收机与日期分层结果。

既有接收视图按VALIDATED_ONCE复用。属于已暴露代理基准；不宣称新盲测或物理卫星证据。Phase2适应、K与新类指标为N/A。

CPU/CUDA原生冒烟覆盖E1/80/131/132/200并通过；独立P0/P1审查及登记检查通过。每GPU最多2个实验进程，本轮最多3个并行row。

执行commit：`0c79b8daf3bcf1f1e885a43940348c1da5d6c371`。证据：[远端读回](evidence/readback.json)、[首次读回](evidence/readback_initial.json)、[实验登记](experiment.json)。
